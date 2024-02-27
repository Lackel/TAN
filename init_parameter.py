from argparse import ArgumentParser

def init_model():
    parser = ArgumentParser()
    
    parser.add_argument("--data_dir", default='data', type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    
    parser.add_argument("--pretrain_dir", default='./premodel_stackoverflow_0', type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.") 
    
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, help="The path for the pre-trained bert model.")
    
    parser.add_argument("--max_seq_length", default=None, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--feat_dim", default=768, type=int, help="The feature dimension.")
    
    parser.add_argument("--warmup_proportion", default=0.1, type=float)

    parser.add_argument("--freeze_bert_parameters", action="store_true", help="Freeze the last parameters of BERT.")

    parser.add_argument("--save_model", action="store_true", help="Save pretrained model.")

    parser.add_argument("--pretrain", action="store_true", help="Pre-train the model with labeled data.")
    
    parser.add_argument("--dataset", default=None, type=str, required=True, 
                        help="The name of the dataset to train selected.")
    
    parser.add_argument("--gamma", default=100, type=float, help="The weighting factor for the cross entropy loss.")

    parser.add_argument("--topk", default=5, type=float, help="The number of the selected similar labeled prototypes.")

    parser.add_argument("--alpha", default=0.8, type=float, help="The weighting factor for the original unlabeled prototypes.")

    parser.add_argument("--temperature", default=0.07, type=float,
                        help="The temperature for dot product.")

    parser.add_argument("--known_cls_ratio", default=0.75, type=float, required=True, help="The number of known classes.")

    
    parser.add_argument("--cluster_num_factor", default=1.0, type=float, required=True, help="The factor (magnification) of the number of clusters K.")

    parser.add_argument('--seed', type=int, default=0, help="Random seed for initialization.")


    parser.add_argument("--labeled_ratio", default=0.1, type=float, help="The ratio of labeled samples in the training set.")
    
    parser.add_argument("--gpu_id", type=str, default='0', help="Select the GPU id.")

    parser.add_argument("--train_batch_size", default=256, type=int,
                        help="Batch size for training.")
    
    parser.add_argument("--pretrain_batch_size", default=32, type=int,
                        help="Batch size for training.")
    
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size for evaluation.")

    parser.add_argument("--wait_patient", default=20, type=int,
                        help="Patient steps for Early Stop.") 

    parser.add_argument("--num_pretrain_epochs", default=100, type=float,
                        help="The pre-training epochs.")

    parser.add_argument("--num_train_epochs", default=10, type=float,
                        help="The training epochs.")

    parser.add_argument("--lr_pre", default=5e-5, type=float,
                        help="The learning rate for pre-training.")
    
    parser.add_argument("--lr", default=1e-5, type=float,
                        help="The learning rate for training.")  
    
    
    return parser
