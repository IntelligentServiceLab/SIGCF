import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = os.path.dirname(__file__)
CODE_PATH = ROOT_PATH
DATA_PATH = args.data_path if args.data_path else join(os.path.dirname(CODE_PATH), 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys
sys.path.append(join(CODE_PATH, 'sources'))


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


config = {}
all_dataset = ['pw']
all_models  = ['lgn_intent']

dataset = args.dataset
model_name = args.model
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")


config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['dropout'] = args.dropout
config['keep_prob']  = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False

config['tag_data_path'] = DATA_PATH
config['use_node_semantic'] = bool(args.use_node_semantic)
config['semantic_lr'] = args.semantic_lr
config['semantic_gate_init_bias'] = args.semantic_gate_init_bias

config['topm_mashup'] = args.topm_mashup
config['warmup_epochs'] = args.warmup_epochs
config['intent_lr'] = args.intent_lr

config['intent_schedule'] = 'fixed'
config['intent_weight_fixed'] = args.intent_weight_fixed
config['intent_weight_max'] = 0.2
config['intent_sigmoid_scale'] = 20.0

config['normalize_node_emb'] = False
config['intent_freeze_epochs'] = 3

config['lambda_nca'] = args.lambda_nca
config['nca_weight_ui'] = args.nca_weight_ui
config['nca_weight_ii'] = args.nca_weight_ii
config['nca_temperature'] = args.nca_temperature
config['nca_delay'] = args.nca_delay
config['sbert_model_name'] = args.sbert_model_name

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
config['device'] = device
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

config['csv_dir'] = args.csv_dir
config['train_file'] = args.train_file
config['eval_file'] = args.eval_file
config['test_file'] = args.test_file





TRAIN_epochs = args.epochs
PHASE1_epochs = args.phase1_epochs
ADAPTER_epochs = args.adapter_epochs
FULL_epochs = args.full_epochs
EVAL_interval = args.eval_interval
TRIAL_seeds = [int(s.strip()) for s in args.trial_seeds.split(',') if s.strip()]
CKPT_DIR = args.ckpt_dir
RUN_MODE = args.run_mode
EARLY_STOP_PATIENCE = args.early_stop_patience
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)




logo = ''

