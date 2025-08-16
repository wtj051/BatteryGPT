import argparse
import json
import os
import sys
import hashlib
from urllib.request import urlopen, Request

INDEX_PATH = os.path.join("checkpoints", "index.json")
CKPT_DIR = "checkpoints"

def sha256sum(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def choose_artifact(index, dataset, stage):
    cand = [a for a in index["artifacts"]
            if a["dataset"].lower() == dataset.lower()
            and a["stage"].lower() == stage.lower()]
    if not cand:
        raise ValueError(f"No artifact for dataset={dataset}, stage={stage}")
    return cand[0]

def download(url, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as resp, open(out_path, "wb") as f:
        total = int(resp.headers.get("Content-Length", "0"))
        read = 0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            read += len(chunk)
            if total:
                done = int(50 * read / total)
                sys.stdout.write("\r[{}{}] {}/{} bytes".format(
                    "#"*done, "."*(50-done), read, total))
                sys.stdout.flush()
    print("\nDownloaded:", out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="NASA | Oxford | CALCE | SNL | ISU")
    parser.add_argument("--stage", default="finetune", help="pretrain | finetune")
    args = parser.parse_args()

    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        index = json.load(f)

    art = choose_artifact(index, args.dataset, args.stage)
    filename = art["filename"]
    url = art["url"]
    target = os.path.join(CKPT_DIR, filename)

    if os.path.exists(target):
        print("File exists; verifying checksum...")
    else:
        print(f"Downloading {filename} from:\n  {url}")
        download(url, target)

    if art.get("sha256") and art["sha256"] != "<TO_BE_FILLED>":
        got = sha256sum(target)
        assert got == art["sha256"], f"SHA256 mismatch: {got} != {art['sha256']}"
        print("Checksum OK.")
    else:
        print("SHA256 not provided or placeholder; please update 'index.json' after computing it.")

if __name__ == "__main__":
    main()
