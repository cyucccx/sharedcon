tuning_param  = ["dataset", "load_dir"]
dataset = ["cold"]  # dataset for evaluation

# saved model location (folder containing model.pt from your training run)
load_dir = ["./save/simcse/sbert-multi_ihc_pure_c10/0"]

train_batch_size = 8
eval_batch_size = 8
hidden_size = 768
model_type = "bert-base-multilingual-cased"
SEED = 0

param = {"dataset":dataset,"train_batch_size":train_batch_size,"eval_batch_size":eval_batch_size,"hidden_size":hidden_size,"dataset":dataset,"SEED":SEED,"model_type":model_type, "load_dir":load_dir}

