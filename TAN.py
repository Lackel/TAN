import os
from init_parameter import init_model
from data import Data
from model import BertForModel, PretrainModelManager
import numpy as np
import torch.nn.functional as F
from util import clustering_score
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset, RandomSampler
import torch
from sklearn.cluster import KMeans
from transformers import logging, AutoTokenizer, WEIGHTS_NAME
import random
from pytorch_pretrained_bert.optimization import BertAdam
import math
import warnings
from scipy.special import softmax

class ModelManager:
    def __init__(self, args, data, pretrained_model=None):
        self.set_seed(args.seed)
        if pretrained_model is None:
            pretrained_model = BertForModel(args, data.num_known)
            if os.path.exists(args.pretrain_dir):
                pretrained_model = self.restore_model(args, pretrained_model)
        self.pretrained_model = pretrained_model
        self.labelMap = None
        
        self.seed = args.seed         
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_known = data.num_known
        if args.cluster_num_factor > 1:
            self.num_labels = self.predict_k(args, data) 
        else:
            self.num_labels = data.num_labels
        self.model = BertForModel(args, self.num_labels)
        
        self.load_pretrained_model()

        if args.freeze_bert_parameters:
            self.freeze_parameters(self.model)
            
        self.model.to(self.device)
        
        self.data = data
        num_train_examples = 2 * len(data.train_labeled_examples) + 2 * len(data.train_unlabeled_examples)
        self.num_training_steps = math.ceil(num_train_examples / args.train_batch_size) * 100
        self.num_warmup_steps= int(args.warmup_proportion * self.num_training_steps) 
        self.optimizer = self.get_optimizer(args)
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        
    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def get_features_labels(self, dataloader, model, args):      
        model.eval()
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        for _, batch in enumerate(dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                feature = model(input_ids, segment_ids, input_mask, mode='feature_extract')
            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))
        return total_features, total_labels
    
    def get_optimizer(self, args):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                         lr = args.lr,
                         warmup = args.warmup_proportion,
                         t_total = self.num_training_steps)   
        return optimizer

    def evaluation(self, args, data):
        self.model.eval()
        feats, labels = self.get_features_labels(data.test_dataloader, self.model, args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters = self.num_labels, n_init=20, random_state=args.seed).fit(feats)
        y_true = labels.cpu().numpy()
        results = clustering_score(y_true, km.labels_, data.known_lab)
        print(results)
        
    def update_cluster_ids(self, pseudo_labels, args, data):
        train_data = TensorDataset(data.semi_input_ids, data.semi_input_mask, data.semi_segment_ids, pseudo_labels)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = args.train_batch_size)
        return train_dataloader

    def train(self, args, data):
        labelediter = iter(self.data.train_labeled_dataloader)
        train_dataloader, proto_label, proto_calibration, maps = self.calibration(data, args)
    
        for epoch in range(1, int(args.num_train_epochs)+1, 1):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            self.model.train()   

            for _, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                
                # generate augmented instance.
                input_ids1 = self.random_token_replace(input_ids.cpu(), self.tokenizer).to(self.device)
                pooled1 = self.model(input_ids1, segment_ids, input_mask, label_ids, mode='feature_extract')
                
                # cross entropy loss on unlabeled data with cluster ids as pseudo labels
                loss_u, pooled = self.model(input_ids, segment_ids, input_mask, label_ids, mode='train')
                
                # instance-to-instance alignment loss.
                pooled_cont = torch.cat([F.normalize(pooled, dim=1), F.normalize(pooled1, dim=1)], dim=0)
                loss_i2i = self.model.simcse_loss(pooled_cont)

                # instance-to-prototype alignment loss.
                cost_mat = self.EuclideanDistances(pooled, proto_calibration)
                mask = torch.zeros_like(cost_mat)
                for i in range(cost_mat.shape[0]):
                    mask[i][label_ids[i]] = 1
                loss_i2p = (cost_mat * mask).sum(1).mean()

                # prototype-to-instance transfer loss.
                cost_mat = self.EuclideanDistances(pooled, proto_label)
                mask = torch.zeros_like(cost_mat)
                list_known = torch.tensor(list(maps.keys())).to(self.device)
                for i in range(cost_mat.shape[0]):
                    if label_ids[i] in list_known:
                        mask[i][maps[label_ids[i].item()]] = 1
                loss_p2i = (cost_mat * mask).sum(1).mean()

                loss_pro = loss_i2p + loss_i2i + loss_p2i + loss_u

                try:
                    batch = labelediter.next()
                except StopIteration:
                    labelediter = iter(data.train_labeled_dataloader)
                    batch = labelediter.next()
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss_ce, pooled = self.model(input_ids, segment_ids, input_mask, label_ids, mode='train') 
                
                loss = loss_pro + args.gamma * loss_ce
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                self.optimizer.step()
                self.optimizer.zero_grad()

            # update labeled prototypes every epoch.
            feats_label, labels = self.get_features_labels(data.train_labeled_dataloader, self.model, args)
            for i in range(self.num_known):
                t_datas = feats_label[labels == i, :]
                proto_label[i] =  torch.mean(t_datas, axis=0)
            print('Epoch ' + str(epoch) + ' loss:' + str(tr_loss/nb_tr_steps))
              
    def load_pretrained_model(self):
        pretrained_dict = self.pretrained_model.state_dict()
        classifier_params = ['classifier.weight','classifier.bias']
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)
        
    def restore_model(self, args, model):
        output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))
        return model
    
    def freeze_parameters(self, model):
        for name, param in model.bert.named_parameters():
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name or "encoder.layer.10" in name or "encoder.layer.9" in name or "encoder.layer.8" in name:
                param.requires_grad = True

    def EuclideanDistances(self, a, b):
        sq_a = a**2
        sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
        sq_b = b**2
        sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
        bt = b.t()
        return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))
    
    def calibration(self, data, args):
        feats_label, labels = self.get_features_labels(data.train_labeled_dataloader, self.model, args)
        feats_label = feats_label.cpu().numpy()
        labels = labels.cpu().numpy()
        proto_label = np.zeros((self.num_known, feats_label.shape[-1]))
        for i in range(self.num_known):
            t_datas = feats_label[labels == i, :]
            proto_label[i] =  np.mean(t_datas, axis=0)

        feats_unlabel, labels_unlabel = self.get_features_labels(data.train_semi_dataloader, self.model, args)
        print("Performing k-means...")
        feats_unlabel = feats_unlabel.cpu().numpy()
        labels_unlabel = labels_unlabel.cpu().numpy()
        km = KMeans(n_clusters = self.num_labels, n_init=20, random_state=args.seed).fit(feats_unlabel)
        print("K-means finished")

        train_dataloader = self.update_cluster_ids(torch.tensor(km.labels_, dtype=torch.long).to(self.device), args, data)

        means_unlabel = np.zeros((self.num_labels, feats_label.shape[-1]))
        for i in range(self.num_labels):
            t_datas = feats_unlabel[km.labels_ == i, :]
            means_unlabel[i] = np.mean(t_datas, axis=0)

        dist_matrix = np.zeros((self.num_labels, self.num_known))
        for i in range(self.num_labels):
            for j in range(self.num_known):
                dist_matrix[i, j] = -(np.linalg.norm(means_unlabel[i] - proto_label[j])) / (feats_unlabel.shape[-1] ** (1/2))
            index = dist_matrix[i].argsort()[0:self.num_known-args.topk]
            dist_matrix[i][index] = -1e9
            dist_matrix[i] = softmax(dist_matrix[i])
        proto_calibration = args.alpha * means_unlabel + (1 - args.alpha) * dist_matrix.dot(proto_label)

        proto_label = torch.tensor(proto_label).float().to(self.device)
        proto_calibration = torch.tensor(proto_calibration).float().to(self.device)

        # index map between labeled and unlabeled prototypes
        temp = dist_matrix.T
        index = np.argmax(temp, axis=-1)
        # getting cluster ids corresponding to known categories
        maps = {index[i]:i for i in range(self.num_known)}
        
        return train_dataloader, proto_label, proto_calibration, maps

    def random_token_replace(self, ids, tokenizer):
        mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        ids, _ = self.mask_tokens(ids, tokenizer, mlm_probability=0.25)
        random_words = torch.randint(len(tokenizer), ids.shape, dtype=torch.long)
        indices_replaced = torch.where(ids == mask_id)
        ids[indices_replaced] = random_words[indices_replaced]
        return ids
    
    def mask_tokens(self, inputs, tokenizer, special_tokens_mask=None, mlm_probability=0.15):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix[torch.where(inputs==0)] = 0.0
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    def predict_k(self, args, data):
        feats, _ = self.get_features_labels(data.train_semi_dataloader, self.pretrained_model.cuda(), args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters = data.num_labels).fit(feats)
        y_pred = km.labels_

        pred_label_list = np.unique(y_pred)
        drop_out = len(feats) / data.num_labels * 0.9
        print('drop',drop_out)

        cnt = 0
        for label in pred_label_list:
            num = len(y_pred[y_pred == label]) 
            if num < drop_out:
                cnt += 1

        num_labels = len(pred_label_list) - cnt
        print('pred_num',num_labels)

        return num_labels

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print('Data and Parameters Initialization...')
    parser = init_model()
    args = parser.parse_args()
    data = Data(args)
    
    if args.pretrain:
        print('Pre-training begin...')
        manager_p = PretrainModelManager(args, data)
        manager_p.train()
        print('Pre-training finished!')
        manager = ModelManager(args, data, manager_p.model)
    else:
        manager = ModelManager(args, data, None)
    
    print('Training begin...')
    manager.evaluation(args, data)
    manager.train(args, data)
    print('Training finished!')

    
    print('Evaluation begin...')
    manager.evaluation(args, data)
    # You can run evaluation multiple times to eliminate the effects of clustering randomness.
    # manager.evaluation(args, data)
    # manager.evaluation(args, data)
    print('Evaluation finished!')

