import re, unicodedata, random, numpy as np
from typing import List
from dataclasses import dataclass

def slugify(text: str) -> str:
    t = unicodedata.normalize("NFKD", text).encode("ascii","ignore").decode("ascii")
    t = re.sub(r"[^a-zA-Z0-9_]+","_", t)
    return re.sub(r"_+","_", t).strip("_").lower()

def clip(s: str, n: int) -> str:
    s = (s or "").strip().replace("\n"," ")
    return s if len(s) <= n else s[:n]

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)

@dataclass
class Pair:
    doc: str
    page: int
