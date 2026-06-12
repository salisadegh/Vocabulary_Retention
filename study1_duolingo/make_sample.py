"""
make_sample.py — deterministic Duolingo dataset sampler
========================================================
Run from the repository root after download_data.sh:
    python study1_duolingo/make_sample.py

What it does (takes a few minutes):
  * Keeps a DETERMINISTIC ~15% of USERS (md5-hash rule, no randomness,
    user-complete: every kept user keeps ALL their rows -> clean
    user-level cross-validation later).
  * Drops the heavy lexeme_string column; keeps the 11 analysis columns.
  * Writes standalone parts of ~700k rows each into data/duolingo_sample/
  * Writes a manifest with row counts + MD5 per part, for verification
    against manifest_reference.txt.
Requires: Python 3 + pandas.
"""
import hashlib, os, sys
import pandas as pd

SRC = "data/duolingo_raw/settles.acl16.learning_traces.13m.csv.gz"
KEEP_PCT = 15
ROWS_PER_PART = 700_000
COLS = ["p_recall", "timestamp", "delta", "user_id", "learning_language",
        "ui_language", "lexeme_id", "history_seen", "history_correct",
        "session_seen", "session_correct"]

os.makedirs("data/duolingo_sample", exist_ok=True)

if not os.path.exists(SRC):
    sys.exit(f"Dataset not found: {SRC} — run study1_duolingo/download_data.sh first.")

def keep_user(uid: str) -> bool:
    return int(hashlib.md5(uid.encode()).hexdigest(), 16) % 100 < KEEP_PCT

def md5(path, bs=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(bs):
            h.update(chunk)
    return h.hexdigest()

buf, part, total_in, total_kept, manifest = [], 0, 0, 0, []

def flush():
    global buf, part
    if not buf:
        return
    part += 1
    name = f"data/duolingo_sample/duolingo_sample_part{part:02d}.csv.gz"
    out = pd.concat(buf, ignore_index=True)
    out.to_csv(name, index=False, compression="gzip")
    manifest.append((name, len(out), md5(name),
                     round(os.path.getsize(name) / 1e6, 1)))
    print(f"  wrote {name}: {len(out):,} rows, "
          f"{os.path.getsize(name)/1e6:.1f} MB")
    buf = []

buffered = 0
for ch in pd.read_csv(SRC, usecols=COLS, chunksize=1_000_000,
                      dtype={"user_id": str}):
    total_in += len(ch)
    kept = ch[ch["user_id"].map(keep_user)]
    total_kept += len(kept)
    buf.append(kept)
    buffered += len(kept)
    print(f"scanned {total_in:,} rows -> kept {total_kept:,}")
    while buffered >= ROWS_PER_PART:
        big = pd.concat(buf, ignore_index=True)
        head, tail = big.iloc[:ROWS_PER_PART], big.iloc[ROWS_PER_PART:]
        buf, buffered = [tail], len(tail)
        part += 1
        name = f"data/duolingo_sample/duolingo_sample_part{part:02d}.csv.gz"
        head.to_csv(name, index=False, compression="gzip")
        manifest.append((name, len(head), md5(name),
                         round(os.path.getsize(name) / 1e6, 1)))
        print(f"  wrote {name}: {len(head):,} rows, "
              f"{os.path.getsize(name)/1e6:.1f} MB")
flush()

with open("data/duolingo_sample/manifest.txt", "w") as f:
    f.write(f"source_rows_scanned={total_in}\nrows_kept={total_kept}\n"
            f"keep_rule=md5(user_id)%100<{KEEP_PCT}\ncolumns={','.join(COLS)}\n")
    for name, n, h, mb in manifest:
        f.write(f"{name}\trows={n}\tmd5={h}\tsize_mb={mb}\n")

print(f"\nDONE. Scanned {total_in:,} rows, kept {total_kept:,} "
      f"({100*total_kept/max(total_in,1):.1f}%) across {part} part(s).")
print("Verify the manifest against study1_duolingo/manifest_reference.txt.")
