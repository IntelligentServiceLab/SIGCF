import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int,default=1024,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=320,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=1,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.002,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=0.001,
                        help="the weight decay for l2 normaliztion")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold to drop out")
    parser.add_argument('--testbatch', type=int,default=4096,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='pw',
                        help="available datasets: [pw]")
    parser.add_argument('--epochs', type=int,default=200,
                        help="the number of training epochs for SIGCF")
    parser.add_argument('--phase1_epochs', type=int, default=100,
                        help='LightGCN warm-up epochs before intent construction')
    parser.add_argument('--adapter_epochs', type=int, default=10,
                        help='adapter pretraining epochs')
    parser.add_argument('--full_epochs', type=int, default=150,
                        help='full SIGCF training epochs')
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='evaluate every N epochs')
    parser.add_argument('--trial_seeds', type=str, default='2020,2021,2022',
                        help='comma-separated seeds for repeated full-training trials')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/pretrain',
                        help='directory to save logs and checkpoints')
    parser.add_argument('--data_path', type=str, default=None,
                        help='dataset directory containing train.txt and the evaluation file')
    parser.add_argument('--train_file', type=str, default='train.txt',
                        help='training interaction file relative to data_path')
    parser.add_argument('--test_file', type=str, default='test.txt',
                        help='final test file relative to data_path')
    parser.add_argument('--eval_file', type=str, default='test.txt',
                        help='evaluation file name in data_path, e.g. val.txt for tuning or test.txt for final testing')
    parser.add_argument('--run_mode', type=str, default='tune', choices=['tune', 'final'],
                        help='tune uses eval_file for early stopping; final trains once and evaluates test only once at the end')
    parser.add_argument('--early_stop_patience', type=int, default=20,
                        help='number of epochs without validation improvement before early stopping')
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--csv_dir', type=str, default="./log/",
                        help="csv_dir")
    parser.add_argument('--tensorboard', type=int, default=0,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str, default="lgn")
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--multicore', type=int, default=0, 
                        help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, 
                        help='whether we use pretrained weight or not')
    parser.add_argument('--topk', type=int, default=20,
                        help='topk for evaluation')
    parser.add_argument('--lambda_nca', type=float, default=0.01,
                        help='NCA contrastive loss weight')
    parser.add_argument('--topm_mashup', type=int, default=19,
                        help='Gumbel Top-M semantic fallback for Mashups whose tags match no API semantic intent')
    parser.add_argument('--warmup_epochs', type=int, default=50,
                        help="number of warmup epochs before introducing intent nodes")
    # Missing args that world.py relies on
    parser.add_argument('--seed', type=int, default=2020,
                        help='random seed')
    parser.add_argument('--model', type=str, default='lgn_intent',
                        help='model name')
    parser.add_argument('--topks', type=str, default='[5,10,20]',
                        help='top-k list for evaluation')
    parser.add_argument('--use_node_semantic', type=int, default=1,
                        help='enable node-level SBERT semantic gated fusion')
    parser.add_argument('--semantic_lr', type=float, default=0.00005,
                        help='learning rate used only by node-text semantic modules')
    parser.add_argument('--semantic_gate_init_bias', type=float, default=0.0,
                        help='initial bias of semantic gate; positive values favor intent')
    parser.add_argument('--intent_lr', type=float, default=0.001,
                        help='learning rate for intent module')
    parser.add_argument('--intent_weight_fixed', type=float, default=1.5,
                        help='fixed strength of semantic-intent residual injection')
    parser.add_argument('--nca_weight_ui', type=float, default=2.0,
                        help='NCA user-item weight')
    parser.add_argument('--nca_weight_ii', type=float, default=1.0,
                        help='NCA item-item weight')
    parser.add_argument('--nca_temperature', type=float, default=0.10,
                        help='NCA temperature')
    parser.add_argument('--nca_delay', type=int, default=5,
                        help='NCA delay epochs')
    parser.add_argument('--sbert_model_name', type=str, default='all-MiniLM-L6-v2',
                        help='SBERT model name or local path for API category+description encoding')
    return parser.parse_args()
