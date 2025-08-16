import argparse
import hashlib
import os
import json

def sha256sum(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--index", default="checkpoints/index.json")
    args = ap.parse_args()

    digest = sha256sum(args.file)
    print("SHA256:", digest)

    if os.path.exists(args.index):
        with open(args.index, "r", encoding="utf-8") as f:
            idx = json.load(f)
        matches = [a for a in idx.get("artifacts", [])
                   if a.get("filename") == os.path.basename(args.file)]
        if matches and matches[0].get("sha256") not in (None, "", "<TO_BE_FILLED>"):
            expected = matches[0]["sha256"]
            if expected == digest:
                print("Checksum matches entry in index.json ✅")
            else:
                print("Checksum does NOT match index.json ❌")
        else:
            print("No SHA256 in index.json for this file (or placeholder).")
