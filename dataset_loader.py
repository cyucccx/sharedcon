import os
import pickle

import torch
import torch.utils.data
from torch.utils.data import Dataset

from util import build_versioned_output_path, get_run_tag
from collate_fns_sharedcon import (
    collate_fn_ihc,
    collate_fn_w_aug_ihc_imp_con,
    collate_fn_dynahate,
    collate_fn_sbic,
    collate_fn_w_aug_sbic_imp_con,
    collate_fn_w_aug_dynahate_imp_con,
)


def resolve_preprocessed_path(dataset):
    env_override = os.environ.get("PREPROCESSED_DATA_PATH")
    candidate_paths = []
    if env_override:
        candidate_paths.append(env_override)

    candidate_paths.extend(
        [
            os.path.join("preprocessed_data", f"preprocessed_{dataset}.pkl"),
            os.path.join("preprocessed_data", f"preprocessed_sbert-multi_{dataset}.pkl"),
            os.path.join("preprocessed_data", f"preprocessed_simcse_{dataset}.pkl"),
        ]
    )

    for candidate_path in candidate_paths:
        if os.path.isfile(candidate_path):
            return candidate_path

    raise FileNotFoundError(
        f"No preprocessed dataset found for {dataset}. Tried: {candidate_paths}"
    )


# Credits https://github.com/varsha33/LCL_loss
class ihc_dataset(Dataset):

    def __init__(self,data,training=True,w_aug=False):

        self.data = data
        self.training = training
        self.w_aug = w_aug


    def __getitem__(self, index):

        item = {}

        if self.training and self.w_aug:
            item["post"] = self.data["tokenized_post"][index]
            item["cluster_label"] = self.data["cluster_label"][index]

        else:
            item["post"] = torch.LongTensor(self.data["tokenized_post"][index])
        
        item["label"] = self.data["label"][index]
        
        return item

    def __len__(self):

        try:
            return len(self.data["label"])
        except KeyError:
            print(self.data)
            error_path = build_versioned_output_path(".", "key_error.pickle", get_run_tag())
            with open(error_path, 'wb') as f:
                pickle.dump(self.data, f)
            print(f"Saved debug payload to {error_path}")
            

        

class dynahate_dataset(Dataset):

    def __init__(self,data,training=True,w_aug=False):

        self.data = data
        self.training = training
        self.w_aug = w_aug


    def __getitem__(self, index):

        item = {}

        if self.training and self.w_aug:
            item["post"] = self.data["tokenized_post"][index]
            item["cluster_label"] = self.data["cluster_label"][index]
        else:
            item["post"] = torch.LongTensor(self.data["tokenized_post"][index])

        item["label"] = self.data["label"][index]

        return item

    def __len__(self):
        return len(self.data["label"])

class sbic_dataset(Dataset):

    def __init__(self,data,training=True,w_aug=False):

        self.data = data
        self.training = training
        self.w_aug = w_aug


    def __getitem__(self, index):

        item = {}

        if self.training and self.w_aug:
            item["post"] = self.data["tokenized_post"][index]
            item["cluster_label"] = self.data["cluster_label"][index]
        else:
            item["post"] = torch.LongTensor(self.data["tokenized_post"][index])

        item["label"] = self.data["label"][index]

        return item

    def __len__(self):
        return len(self.data["label"])


def get_dataloader(train_batch_size,eval_batch_size,dataset,seed=None,w_aug=True,w_double=False,label_list=None):

    preprocessed_path = resolve_preprocessed_path(dataset)
    with open(preprocessed_path, "rb") as f:

        data = pickle.load(f)

    if "ihc" in dataset:
        train_dataset = ihc_dataset(data["train"],training=True,w_aug=w_aug)
        valid_dataset = ihc_dataset(data["valid"],training=False,w_aug=w_aug)
        test_dataset = ihc_dataset(data["test"],training=False,w_aug=w_aug)

    elif "dynahate" in dataset:
        train_dataset = dynahate_dataset(data["train"],training=True,w_aug=w_aug)
        valid_dataset = dynahate_dataset(data["valid"],training=False,w_aug=w_aug)
        test_dataset = dynahate_dataset(data["test"],training=False,w_aug=w_aug)
    elif "sbic" in dataset:
        train_dataset = sbic_dataset(data["train"],training=True,w_aug=w_aug)
        valid_dataset = sbic_dataset(data["valid"],training=False,w_aug=w_aug)
        test_dataset = sbic_dataset(data["test"],training=False,w_aug=w_aug)
    elif "cold" in dataset:
        train_dataset = sbic_dataset(data["train"],training=True,w_aug=w_aug)
        valid_dataset = sbic_dataset(data["valid"],training=False,w_aug=w_aug)
        test_dataset = sbic_dataset(data["test"],training=False,w_aug=w_aug)
    else:
        raise NotImplementedError

    if "ihc" in dataset:
        collate_fn = collate_fn_ihc
        if w_double:
            raise NotImplementedError("w_double=True is not supported for ihc; double-augmentation collate_fn is missing.")
        else:
            collate_fn_w_aug = collate_fn_w_aug_ihc_imp_con # original1, original2, .... aug1, aug2 
    elif "dynahate" in dataset:
        # assert not w_aug, "for cross dataset evaluation, we do not consider w_aug"
        collate_fn = collate_fn_dynahate
        collate_fn_w_aug = collate_fn_w_aug_dynahate_imp_con # original1, original2, .... aug1, aug2 
    elif "sbic" in dataset:
        # assert not w_aug, "for cross dataset evaluation, we do not consider w_aug"
        # EXCEPT FOR SBIC, WHICH IS USED FOR TRAIN AS WELL
        collate_fn = collate_fn_sbic
        if w_double:
            raise NotImplementedError("w_double=True is not supported for sbic; double-augmentation collate_fn is missing.")
        collate_fn_w_aug = collate_fn_w_aug_sbic_imp_con # original1, original2, .... aug1, aug2
    elif "cold" in dataset:
        collate_fn = collate_fn_sbic
        if w_double:
            raise NotImplementedError("w_double=True is not supported for cold eval; double-augmentation collate_fn is missing.")
        collate_fn_w_aug = collate_fn_w_aug_sbic_imp_con
    else:   
        raise NotImplementedError

    if w_aug:
        train_iter  = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size,shuffle=True,collate_fn=collate_fn_w_aug,num_workers=0)
    else:
        train_iter  = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size,shuffle=True,collate_fn=collate_fn,num_workers=0)

    valid_iter  = torch.utils.data.DataLoader(valid_dataset, batch_size=eval_batch_size,shuffle=False,collate_fn=collate_fn,num_workers=0)

    test_iter  = torch.utils.data.DataLoader(test_dataset, batch_size=eval_batch_size,shuffle=False,collate_fn=collate_fn,num_workers=0)


    return train_iter,valid_iter,test_iter
