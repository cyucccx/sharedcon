import os
from pathlib import Path

from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


def resolve_local_model_path(model_name):
    if os.path.isdir(model_name):
        return model_name

    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_root / f"models--{model_name.replace('/', '--')}"
    refs_main = repo_dir / "refs" / "main"
    if refs_main.is_file():
        revision = refs_main.read_text().strip()
        snapshot_dir = repo_dir / "snapshots" / revision
        if snapshot_dir.is_dir():
            return str(snapshot_dir)

    snapshots_dir = repo_dir / "snapshots"
    if snapshots_dir.is_dir():
        snapshot_candidates = sorted([path for path in snapshots_dir.iterdir() if path.is_dir()])
        if snapshot_candidates:
            return str(snapshot_candidates[-1])

    raise FileNotFoundError(
        f"Could not resolve local Hugging Face cache for {model_name}. "
        "Download it once online or disable HF_LOCAL_FILES_ONLY."
    )

# Credits https://github.com/varsha33/LCL_loss
class primary_encoder_v2_no_pooler_for_con(nn.Module):

    def __init__(self,hidden_size,emotion_size,encoder_type="xlm-roberta-base"):
        super(primary_encoder_v2_no_pooler_for_con, self).__init__()
        local_files_only = os.environ.get("HF_LOCAL_FILES_ONLY", "0") == "1"

        model_name_map = {
            "bert-base-uncased": "bert-base-uncased",
            "bert-base-multilingual-cased": "bert-base-multilingual-cased",
            "xlmr": "xlm-roberta-base",
            "xlm-r": "xlm-roberta-base",
            "xlm-roberta-base": "xlm-roberta-base",
            "hatebert": "hate_bert",
        }
        if encoder_type not in model_name_map:
            raise NotImplementedError(f"Unsupported encoder_type: {encoder_type}")

        model_source = model_name_map[encoder_type]
        if local_files_only:
            model_source = resolve_local_model_path(model_source)

        self.encoder_supcon = AutoModel.from_pretrained(
            model_source,
            local_files_only=local_files_only,
        )
        if hasattr(self.encoder_supcon, "encoder") and hasattr(self.encoder_supcon.encoder, "config"):
            self.encoder_supcon.encoder.config.gradient_checkpointing = False

        self.pooler_dropout = nn.Dropout(0.1)
        self.label = nn.Linear(hidden_size,emotion_size)

    def pooler(self, features):
        x = features[:, 0, :]
        x = self.pooler_fc(x)
        x = self.pooler_activation(x)
        return x

    def get_cls_features_ptrnsp(self, text, attn_mask):
        supcon_fea = self.encoder_supcon(text,attn_mask,output_hidden_states=True,output_attentions=True,return_dict=True)
        norm_supcon_fea_cls = F.normalize(supcon_fea.hidden_states[-1][:,0,:], dim=1) # normalized last layer's first token ([CLS])
        pooled_supcon_fea_cls = supcon_fea.pooler_output # [huggingface] Last layer hidden-state of the first token of the sequence (classification token) **further processed by a Linear layer and a Tanh activation function.** The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.

        return pooled_supcon_fea_cls, norm_supcon_fea_cls

    def forward(self, pooled_supcon_fea_cls):
        supcon_fea_cls_logits = self.label(self.pooler_dropout(pooled_supcon_fea_cls))

        return supcon_fea_cls_logits
