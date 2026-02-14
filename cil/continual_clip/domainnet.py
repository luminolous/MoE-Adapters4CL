from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image
import torch
from torch.utils.data import ConcatDataset, Dataset


DOMAINNET_DOMAINS: Tuple[str, ...] = (
    "clipart",
    "infograph",
    "painting",
    "quickdraw",
    "real",
    "sketch",
)


def _read_list_file(txt_path: Union[str, Path]) -> List[Tuple[str, Optional[int]]]:
    """Return list of (relative_path, label).

    The M3SDA DomainNet lists are typically space separated: "path label".
    We keep the label optional to fail with a clear error if a variant is used.
    """

    txt_path = Path(txt_path)
    items: List[Tuple[str, Optional[int]]] = []
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            rel = parts[0]
            lab = int(parts[1]) if len(parts) > 1 and parts[1].lstrip("-").isdigit() else None
            items.append((rel, lab))
    return items


def _infer_class_name_from_relpath(rel_path: str) -> Optional[str]:
    """Infer class folder name from a DomainNet rel path.

    Common pattern:
      clipart/aircraft_carrier/xxx.png
    """
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) >= 3 and parts[0] in DOMAINNET_DOMAINS:
        # Common: <domain>/<class>/<img>
        if parts[1] not in ("train", "test"):
            return parts[1]
        # Variant: <domain>/<train|test>/<class>/...
        if len(parts) >= 4:
            return parts[2]
    if len(parts) >= 2 and parts[0] not in DOMAINNET_DOMAINS:
        # Might be <class>/<img>
        return parts[0]
    return None


def _resolve_abs_path(domainnet_root: Union[str, Path], rel_path: str) -> str:
    """Resolve rel_path to absolute, tolerant to different unzip layouts."""

    root = Path(domainnet_root)
    rel = rel_path.replace("\\", "/")
    p = root / rel
    if p.exists():
        return str(p)

    # Allow an extra nesting level (some users unzip into <root>/DomainNet/...) 
    alt = root / "DomainNet" / rel
    if alt.exists():
        return str(alt)

    # Last resort: if list includes domain prefix but data was unzipped into <root>/<domain>/...
    parts = rel.split("/")
    if parts and parts[0] in DOMAINNET_DOMAINS:
        alt2 = root / parts[0] / "/".join(parts[1:])
        if alt2.exists():
            return str(alt2)

    return str(p)


def _build_label_to_name(
    list_items: Sequence[Tuple[str, Optional[int]]],
    nb_classes_hint: int = 345,
) -> List[str]:
    """Build label->class_name list (size = max(label)+1 or hint)."""

    label_to_name: Dict[int, str] = {}
    max_lab = -1
    for rel, lab in list_items:
        if lab is None:
            continue
        max_lab = max(max_lab, lab)
        if lab not in label_to_name:
            cname = _infer_class_name_from_relpath(rel)
            if cname is not None:
                label_to_name[lab] = cname

    size = max(nb_classes_hint, max_lab + 1)
    out = [f"class_{i}" for i in range(size)]
    for k, v in label_to_name.items():
        if 0 <= k < size:
            out[k] = v
    return out


class DomainNetListDataset(Dataset):
    """List-based dataset returning (image_tensor, label, task_id)."""

    def __init__(
        self,
        abs_paths: Sequence[str],
        labels: Sequence[int],
        task_id: int,
        transform=None,
    ):
        assert len(abs_paths) == len(labels)
        self.abs_paths = list(abs_paths)
        self.labels = list(labels)
        self.task_id = int(task_id)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.abs_paths)

    def __getitem__(self, idx: int):
        path = self.abs_paths[idx]
        y = self.labels[idx]
        with Image.open(path) as im:
            im = im.convert("RGB")
            if self.transform is not None:
                im = self.transform(im)
        return im, torch.tensor(y, dtype=torch.long), torch.tensor(self.task_id, dtype=torch.long)


class SimpleScenario:
    """Minimal scenario wrapper compatible with cil/main.py expectations."""

    def __init__(self, tasks: Sequence[Dataset]):
        self.tasks = list(tasks)

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ConcatDataset(self.tasks[idx])
        return self.tasks[idx]


@dataclass
class DomainNetPaths:
    domainnet_root: str
    txt_root: str


def resolve_domainnet_paths(cfg) -> DomainNetPaths:
    """Resolve DomainNet paths from hydra cfg.

    Expected layout (default):
      <cfg.dataset_root>/domainnet/
        clipart/...  infograph/...  ...
        txt/clipart_train.txt  txt/clipart_test.txt  ...
    """

    dn_root = Path(cfg.dataset_root) / "domainnet"
    txt_root = Path(getattr(cfg, "domainnet_txt_root", "txt"))
    if not txt_root.is_absolute():
        txt_root = dn_root / txt_root
    return DomainNetPaths(domainnet_root=str(dn_root), txt_root=str(txt_root))


def _load_domain_list(paths: DomainNetPaths, domain: str, split: str) -> List[Tuple[str, Optional[int]]]:
    assert split in ("train", "test")
    txt_file = Path(paths.txt_root) / f"{domain}_{split}.txt"
    if not txt_file.exists():
        raise FileNotFoundError(
            f"Missing list file: {txt_file}.\n"
            f"Download the DomainNet split lists and place them under: {paths.txt_root}"
        )
    return _read_list_file(txt_file)


def build_domainnet_scenarios(cfg, is_train: bool, transforms):
    """Entry point used by continual_clip/datasets.py."""
    if cfg.scenario == "class":
        return build_domainnet_class_scenario(cfg, is_train=is_train, transforms=transforms)
    if cfg.scenario == "domain":
        return build_domainnet_domain_scenario(cfg, is_train=is_train, transforms=transforms)
    raise ValueError("DomainNet only supports scenario in {'class','domain'} for now")


def build_domainnet_class_scenario(cfg, is_train: bool, transforms):
    """Scenario 1: one domain, class-incremental tasks on shared class set."""

    domain = getattr(cfg, "domainnet_domain", "clipart")
    if domain not in DOMAINNET_DOMAINS:
        raise ValueError(f"Unknown DomainNet domain '{domain}'. Must be one of {DOMAINNET_DOMAINS}.")

    paths = resolve_domainnet_paths(cfg)
    split = "train" if is_train else "test"
    list_items = _load_domain_list(paths, domain=domain, split=split)

    # Infer class names in the ORIGINAL label space (0..344)
    classes_names = _build_label_to_name(list_items, nb_classes_hint=getattr(cfg, "domainnet_nb_classes", 345))

    # Remap original labels -> new labels (order-based) so tasks become contiguous blocks.
    class_order: List[int] = list(getattr(cfg, "class_order", list(range(len(classes_names)))))
    old_to_new = {old: new for new, old in enumerate(class_order)}

    abs_paths: List[str] = []
    y_old: List[int] = []
    for rel, lab in list_items:
        if lab is None:
            raise ValueError(
                f"List file '{domain}_{split}.txt' has missing labels. "
                f"This implementation expects 'rel_path label' per line."
            )
        abs_paths.append(_resolve_abs_path(paths.domainnet_root, rel))
        y_old.append(int(lab))

    # Build task class splits based on class_order (original ids)
    class_ids_per_task: List[List[int]] = []
    class_ids_per_task.append(class_order[: cfg.initial_increment])
    for i in range(cfg.initial_increment, len(class_order), cfg.increment):
        class_ids_per_task.append(class_order[i : i + cfg.increment])

    tasks: List[Dataset] = []
    for task_id, class_ids in enumerate(class_ids_per_task):
        keep = set(class_ids)
        task_paths: List[str] = []
        task_labels_new: List[int] = []
        for p, lab_old in zip(abs_paths, y_old):
            if lab_old in keep:
                task_paths.append(p)
                task_labels_new.append(old_to_new[lab_old])
        tasks.append(
            DomainNetListDataset(
                abs_paths=task_paths,
                labels=task_labels_new,
                task_id=task_id,
                transform=transforms,
            )
        )

    return SimpleScenario(tasks), classes_names


def build_domainnet_domain_scenario(cfg, is_train: bool, transforms):
    """Scenario 2: tasks are domains (6 tasks)."""

    paths = resolve_domainnet_paths(cfg)
    split = "train" if is_train else "test"

    union_items: List[Tuple[str, Optional[int]]] = []
    domain_items: Dict[str, List[Tuple[str, Optional[int]]]] = {}
    for d in DOMAINNET_DOMAINS:
        items = _load_domain_list(paths, domain=d, split=split)
        domain_items[d] = items
        union_items.extend(items)
    classes_names = _build_label_to_name(union_items, nb_classes_hint=getattr(cfg, "domainnet_nb_classes", 345))

    tasks: List[Dataset] = []
    for task_id, d in enumerate(DOMAINNET_DOMAINS):
        abs_paths: List[str] = []
        labels: List[int] = []
        for rel, lab in domain_items[d]:
            if lab is None:
                raise ValueError(
                    f"List file '{d}_{split}.txt' has missing labels. "
                    f"This implementation expects 'rel_path label' per line."
                )
            abs_paths.append(_resolve_abs_path(paths.domainnet_root, rel))
            labels.append(int(lab))  # keep original label space

        tasks.append(
            DomainNetListDataset(
                abs_paths=abs_paths,
                labels=labels,
                task_id=task_id,
                transform=transforms,
            )
        )

    return SimpleScenario(tasks), classes_names