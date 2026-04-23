import os
import re
import time

import torch

# Credits https://github.com/varsha33/LCL_loss
def load_model(resume,model=None):

    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    return model


def iter_product(*args, repeat=1):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)

def save_checkpoint(state,filename):
    torch.save(state,filename)


RUN_VERSION_PATTERN = re.compile(r"_(\d{6})_v(\d+)(?:$|[_.])")


def get_run_tag():
    return time.strftime("%y%m%d", time.localtime())


def extract_run_version(name):
    match = RUN_VERSION_PATTERN.search(name)
    if not match:
        return None
    return int(match.group(2))


def get_next_global_run_version(search_root="."):
    max_version = 0
    for root, dirs, files in os.walk(search_root):
        for entry_name in list(dirs) + list(files):
            version = extract_run_version(entry_name)
            if version is not None:
                max_version = max(max_version, version)
    return max_version + 1


def get_run_version(reference_name=None, search_root="."):
    env_run_version = os.environ.get("RUN_VERSION")
    if env_run_version:
        return int(env_run_version)

    if reference_name:
        reference_version = extract_run_version(reference_name)
        if reference_version is not None:
            return reference_version

    cached_version = getattr(get_run_version, "_cached_value", None)
    if cached_version is None:
        cached_version = get_next_global_run_version(search_root=search_root)
        setattr(get_run_version, "_cached_value", cached_version)
    return cached_version


def build_versioned_output_path(save_dir, base_name, run_tag, run_version=None):
    stem, ext = os.path.splitext(base_name)
    run_version = run_version or get_run_version()
    candidate = os.path.join(save_dir, f"{stem}_{run_tag}_v{run_version}{ext}")
    if not os.path.exists(candidate):
        return candidate

    duplicate_index = 2
    while True:
        duplicate_candidate = os.path.join(
            save_dir,
            f"{stem}_{run_tag}_v{run_version}_r{duplicate_index}{ext}",
        )
        if not os.path.exists(duplicate_candidate):
            return duplicate_candidate
        duplicate_index += 1


def build_versioned_output_dir(base_dir, run_tag, run_version=None):
    run_version = run_version or get_run_version()
    candidate = f"{base_dir}_{run_tag}_v{run_version}"
    if not os.path.exists(candidate):
        return candidate

    duplicate_index = 2
    while True:
        duplicate_candidate = f"{base_dir}_{run_tag}_v{run_version}_r{duplicate_index}"
        if not os.path.exists(duplicate_candidate):
            return duplicate_candidate
        duplicate_index += 1


def ensure_output_path_is_new(path, label="output path"):
    if os.path.exists(path):
        raise FileExistsError(f"{label} already exists: {path}")
    return path


def ensure_output_dir_is_new(path, label="output directory"):
    if os.path.exists(path):
        raise FileExistsError(f"{label} already exists: {path}")
    return path


def resolve_checkpoint_path(load_dir, preferred_filename=None):
    if not load_dir or not os.path.isdir(load_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {load_dir}")

    if preferred_filename:
        preferred_path = os.path.join(load_dir, preferred_filename)
        if not os.path.isfile(preferred_path):
            raise FileNotFoundError(
                f"Configured checkpoint file not found: {preferred_path}"
            )
        return preferred_path

    checkpoint_candidates = [
        entry.path
        for entry in os.scandir(load_dir)
        if entry.is_file() and entry.name.startswith("model") and entry.name.endswith(".pt")
    ]
    if not checkpoint_candidates:
        raise FileNotFoundError(f"No checkpoint found under: {load_dir}")

    checkpoint_candidates.sort(key=lambda path: (os.path.getmtime(path), path))
    return checkpoint_candidates[-1]


def clip_gradient(model, clip_value):

    for name,param in model.named_parameters():
        param.grad.data.clamp_(-clip_value, clip_value)

def one_hot(labels, class_size):
    if type(labels) is list:
      targets = torch.zeros(len(labels), class_size)
    else:
      targets = torch.zeros(labels.size(0), class_size)

    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets
