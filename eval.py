import numpy as np
import json
import random
import os
from easydict import EasyDict as edict
import time

import torch
import torch.utils.data
from torch import nn

import eval_config as train_config
from dataset_loader import get_dataloader
from util import (
    build_versioned_output_path,
    extract_run_version,
    get_run_tag,
    iter_product,
    resolve_checkpoint_path,
)
from sklearn.metrics import f1_score

from model import primary_encoder_v2_no_pooler_for_con

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def apply_eval_env_overrides(param):
    overridden = dict(param)

    env_datasets = os.environ.get("EVAL_DATASETS")
    if env_datasets:
        overridden["dataset"] = [item.strip() for item in env_datasets.split(",") if item.strip()]

    env_load_dir = os.environ.get("EVAL_LOAD_DIR")
    if env_load_dir:
        overridden["load_dir"] = [env_load_dir]

    env_model_filename = os.environ.get("EVAL_MODEL_FILENAME")
    if env_model_filename is not None:
        overridden["model_filename"] = [env_model_filename]

    env_test_only = os.environ.get("EVAL_TEST_ONLY")
    if env_test_only is not None:
        overridden["test_only"] = env_test_only.lower() in {"1", "true", "yes"}

    return overridden

# Credits https://github.com/varsha33/LCL_loss
def test(test_loader,model_main,log):
    model_main.eval()
    
    total_pred_1,total_true,total_pred_prob_1 = [],[],[]
    save_pred = {"true":[],"pred_1":[],"pred_prob_1":[],"feature":[]}

    total_feature = []
    total_num_corrects = 0
    total_num = 0
    print(len(test_loader))
    with torch.no_grad():
        for idx,batch in enumerate(test_loader):
            if "ihc" in log.param.dataset:
                text_name = "post"
                label_name = "label"
            elif "dynahate" in log.param.dataset:
                text_name = "post"
                label_name = "label"
            elif "sbic" in log.param.dataset or "toxicn" in log.param.dataset:
                text_name = "post"
                label_name = "label"
            else:
                text_name = "cause"
                label_name = "emotion"
                raise NotImplementedError

            text = batch[text_name]
            attn = batch[text_name+"_attn_mask"]

            label = batch[label_name]
            label = torch.tensor(label)
            label = torch.autograd.Variable(label).long()

            text = text.to(device)
            attn = attn.to(device)
            label = label.to(device)

            last_layer_hidden_states, supcon_feature_1 = model_main.get_cls_features_ptrnsp(text,attn) # #v2
            pred_1 = model_main(last_layer_hidden_states)

            num_corrects_1 = (torch.max(pred_1, 1)[1].view(label.size()).data == label.data).float().sum()

            pred_list_1 = torch.max(pred_1, 1)[1].view(label.size()).data.detach().cpu().tolist()
            true_list = label.data.detach().cpu().tolist()

            total_num_corrects += num_corrects_1.item()
            total_num += text.shape[0]

            total_pred_1.extend(pred_list_1)
            total_true.extend(true_list)
            total_feature.extend(supcon_feature_1.data.detach().cpu().tolist())
            total_pred_prob_1.extend(pred_1.data.detach().cpu().tolist())

    f1_score_1 = f1_score(total_true,total_pred_1, average="macro")
    f1_score_1_w = f1_score(total_true,total_pred_1, average="weighted")
    f1_score_1 = {"macro":f1_score_1,"weighted":f1_score_1_w}

    total_acc = 100 * total_num_corrects / total_num

    save_pred["true"] = total_true
    save_pred["pred_1"] = total_pred_1

    save_pred["feature"] = total_feature
    save_pred["pred_prob_1"] = total_pred_prob_1

    return total_acc,f1_score_1,save_pred

##################################################################################################
def cl_test(log):

    np.random.seed(log.param.SEED)
    random.seed(log.param.SEED)
    torch.manual_seed(log.param.SEED)
    torch.cuda.manual_seed(log.param.SEED)
    torch.cuda.manual_seed_all(log.param.SEED)

    torch.backends.cudnn.deterministic = True #
    torch.backends.cudnn.benchmark = False #

    print("#######################start run#######################")
    print("log:", log)
    run_tag = get_run_tag()

    _,valid_data,test_data = get_dataloader(log.param.train_batch_size,log.param.eval_batch_size,log.param.dataset,w_aug=False,w_double=False,label_list=None)


    model_main = primary_encoder_v2_no_pooler_for_con(log.param.hidden_size,log.param.label_size,log.param.model_type) # v2
    
    #################################################################
    # load model
    configured_model_filename = getattr(log.param, "model_filename", "")
    model_path = resolve_checkpoint_path(
        log.param.load_dir,
        preferred_filename=configured_model_filename if configured_model_filename else None,
    )
    run_version = extract_run_version(os.path.basename(model_path))
    if run_version is None:
        run_version = 0
    model_main.load_state_dict(torch.load(model_path))
    print(f"model is loaded from {model_path}")
    
    model_main.eval()
    model_main.to(device)
    ###################################################################
    
    test_only = getattr(log.param, "test_only", False)
    if not test_only:
        val_acc_1,val_f1_1,val_save_pred = test(valid_data,model_main,log)
    else:
        val_acc_1 = None
        val_f1_1 = None
    test_acc_1,test_f1_1,test_save_pred = test(test_data,model_main,log)
    

    print("Model 1")
    if not test_only:
        print(f'Valid Accuracy: {val_acc_1:.2f} Valid F1: {val_f1_1["macro"]:.2f}')
    print(f'Test Accuracy: {test_acc_1:.2f} Test F1: {test_f1_1["macro"]:.2f}')

    log.valid_f1_score_1 = val_f1_1
    log.test_f1_score_1 = test_f1_1
    log.valid_accuracy_1 = val_acc_1
    log.test_accuracy_1 = test_acc_1

    if log.param.dataset == "dynahate":
        log_base_name = "dynahate_test_log.json"
    elif log.param.dataset == "sbic":
        log_base_name = "sbic_test_log.json"
    elif "ihc" in log.param.dataset:
        log_base_name = "ihc_test_log.json"
    elif log.param.dataset == "toxicn":
        log_base_name = "toxicn_test_log.json"
    elif log.param.dataset == "sbic_hate":
        log_base_name = "sbic_hate_test_log.json"
    else:
        raise NotImplementedError
    log_path = build_versioned_output_path(log.param.load_dir, log_base_name, run_tag, run_version=run_version)
    with open(log_path, 'w') as fp:
        json.dump(dict(log), fp,indent=4)
    print(f"evaluation log is saved at {log_path}")


if __name__ == '__main__':

    tuning_param = train_config.tuning_param
    effective_param = apply_eval_env_overrides(train_config.param)

    param_list = [effective_param[i] for i in tuning_param]
    param_list = [tuple(tuning_param)] + list(iter_product(*param_list)) ## [(param_name),(param combinations)]

    for param_com in param_list[1:]: # as first element is just name

        log = edict()
        log.param = effective_param

        for num,val in enumerate(param_com):
            log.param[param_list[0][num]] = val

        log.param.label_size = 2
        
        assert log.param.load_dir is not None, "to load a model, log.param.load_dir should be given!!"
        cl_test(log)
