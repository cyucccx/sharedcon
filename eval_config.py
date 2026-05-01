tuning_param  = ["dataset", "load_dir", "model_filename"]
dataset = ["sbert-multi_ihc_pure_c10", "toxicn"]  # datasets for evaluation

# saved model location (folder containing model.pt from your training run)
load_dir = ["./save/sbert-multi/sbert-multi_ihc_pure_c10/0"]
model_filename = [""]

train_batch_size = 8
eval_batch_size = 8
hidden_size = 768
model_type = "bert-base-multilingual-cased"
SEED = 0
train_only = False
test_only = False
save_train_predictions = False

param = {
    "dataset":dataset,
    "train_batch_size":train_batch_size,
    "eval_batch_size":eval_batch_size,
    "hidden_size":hidden_size,
    "dataset":dataset,
    "SEED":SEED,
    "model_type":model_type,
    "load_dir":load_dir,
    "model_filename":model_filename,
    "train_only":train_only,
    "test_only":test_only,
    "save_train_predictions":save_train_predictions,
}
