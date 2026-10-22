"""CPU-only data expansion and optional index build for a copied environment."""
import argparse
import gzip
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA = 'f51c33217061479f9c95a1068621fcd38e4883ae3d2f6a1627037bea934f2125'
PRODUCT_SHA = '57b10950a0064d16c81535a1d764a75879a508d250dde8a2a1787c5e6045559f'


def sha(path):
    with path.open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('sft', 'flashreinforce'), required=True)
    parser.add_argument('--build-index', action='store_true')
    args = parser.parse_args()
    env = ROOT / args.stage / 'environments/ShopSimulator/shop_env'
    archive, product = env / 'data/fine_items_eval_train_all.json.gz', env / 'data/items_eval_train.json'
    if sha(archive) != ARCHIVE_SHA: raise ValueError('compressed product archive differs')
    if not product.exists():
        temporary = product.with_suffix('.preparing')
        if temporary.exists(): raise ValueError('preserve existing preparation artifact')
        with gzip.open(archive, 'rb') as source, temporary.open('xb') as target: shutil.copyfileobj(source, target)
        if sha(temporary) != PRODUCT_SHA: raise ValueError('expanded product fingerprint differs')
        temporary.replace(product)
    if sha(product) != PRODUCT_SHA: raise ValueError('existing product file differs')
    print('Frozen product data verified:', product)
    if args.build_index:
        subprocess.run([sys.executable, str(env / 'scripts/build_index.py')], cwd=env, check=True)


if __name__ == '__main__':
    main()
