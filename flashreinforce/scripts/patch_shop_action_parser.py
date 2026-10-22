"""Build a hash-pinned ShopSimulator parser repair without changing its source copy."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

VERSION = "shopping-action-parser-v1"
ORIGINAL_SHA256 = "130e0261c5c589ac133f1562f6d718736cda44abe54e2151867f5b17f136a086"
OLD = "    pattern = re.compile(r'(.+)\\[(.+)\\]')\n    m = re.match(pattern, action)"
NEW = "    pattern = re.compile(r'([^\\[\\]]+)\\[(.+)\\]')\n    m = pattern.fullmatch(action)"


def patched_source(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("native engine source differs from the audited Environment v2.1 copy")
    text = source.decode("utf-8")
    if text.count(OLD) != 1:
        raise ValueError("expected exactly one native action parser")
    result = text.replace(OLD, NEW)
    ast.parse(result)
    return result.encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.source.resolve():
        raise SystemExit("choose a new output file; the native source is preserved")
    source = args.source.read_bytes()
    result = patched_source(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(result)
    print(json.dumps({"version": VERSION, "source_sha256": hashlib.sha256(source).hexdigest(),
                      "patched_sha256": hashlib.sha256(result).hexdigest(),
                      "source_preserved": args.source.read_bytes() == source}))


if __name__ == "__main__":
    main()
