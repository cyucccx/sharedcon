tuning_param  = ["dataset", "load_dir", "model_filename"]
dataset = ["xlmr_ihc_pure_c10"] # dataset for evaluation

# saved model location
load_dir = ['./save/xlmr/xlmr_ihc_pure_c10/0']
model_filename = [""]


train_batch_size = 8
eval_batch_size = 8
hidden_size = 768
model_type = "xlm-roberta-base"
SEED = 0
test_only = False

param = {"dataset":dataset,"train_batch_size":train_batch_size,"eval_batch_size":eval_batch_size,"hidden_size":hidden_size,"dataset":dataset,"SEED":SEED,"model_type":model_type, "load_dir":load_dir, "model_filename":model_filename, "test_only":test_only}
