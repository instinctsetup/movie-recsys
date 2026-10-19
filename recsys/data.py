"""MovieLens 100k loading. We use the official u1.base / u1.test split -
the same one the original papers use, so numbers are comparable."""
import re
from pathlib import Path
import numpy as np

DATA = Path(__file__).parent.parent / "data"


def load_titles(path=None):
    titles = {}
    for line in open(path or DATA / "u.item", encoding="latin-1"):
        parts = line.split("|")
        if len(parts) > 1:
            t = re.sub(r"\s*\(\d{4}\)\s*$", "", parts[1]).strip()
            year = re.search(r"\((\d{4})\)", parts[1])
            titles[int(parts[0])] = {"title": t, "year": year.group(1) if year else ""}
    return titles


def load_ratings(path, u_map=None, i_map=None):
    """Returns (user_idx, item_idx, rating) with 0-based contiguous indices
    plus the maps back to MovieLens ids. Pass existing maps so a second
    file (the test split) shares the SAME index space - otherwise a test
    index points at a different user/item than the same index in train."""
    users, items, ratings = [], [], []
    u_map = {} if u_map is None else u_map
    i_map = {} if i_map is None else i_map
    for line in open(path):
        u, i, r, _ = line.split()
        for raw, lst, mp in ((u, users, u_map), (i, items, i_map)):
            mp.setdefault(int(raw), len(mp))
            lst.append(mp[int(raw)])
        ratings.append(float(r))
    return (np.array(users), np.array(items), np.array(ratings),
            u_map, i_map)
