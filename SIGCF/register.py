import world
import dataloader

from models.p2_intent import LightGCN_Intent 

if world.dataset in ['pw']:
    dataset_path = world.DATA_PATH
    dataset = dataloader.Loader(path=dataset_path)
else:
    raise NotImplementedError

from models.model import LightGCN
RECMODELS = {
    'lgn_intent': LightGCN_Intent      
}

print('===========config================')
print("dataset:", world.dataset)
print("device:", world.device)
print("model_name:", world.model_name)
print("topks:", world.topks)
print('===========end===================')
