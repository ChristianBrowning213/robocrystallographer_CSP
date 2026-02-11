#!/usr/bin/env python3
"""
grab_mp_bulk.py — Bulk-download CIFs from Materials Project with resume + throttling.

Examples:
  # 10k stable materials to a folder
  python grab_mp_bulk.py --out data/mp_stable_10k --max 10000 --stable-only

  # 50k stable materials
  python grab_mp_bulk.py --out data/mp_stable_50k --max 50000 --stable-only

  # domain slice
  python grab_mp_bulk.py --out data/mp_li_o_20k --max 20000 --chemsys Li-O --stable-only

Notes:
- Respects MP rate limit guidance by defaulting to conservative RPS. Docs say rate limits start at 25 req/s.
- Writes manifest.jsonl for checkpointing and reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)


def write_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_done_ids(out_dir: Path) -> Set[str]:
    """Consider an ID done if its CIF exists OR it appears as OK in manifest."""
    done: Set[str] = set()
    for p in out_dir.glob("mp-*.cif"):
        done.add(p.stem)

    manifest = out_dir / "manifest.jsonl"
    if manifest.exists():
        with manifest.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "OK" and rec.get("material_id"):
                        done.add(str(rec["material_id"]))
                except Exception:
                    continue
    return done


def iter_summary_ids(
    mpr: MPRester,
    *,
    stable_only: bool,
    chemsys: Optional[str],
    elements: Optional[List[str]],
    chunk_size: int,
) -> Iterable[List[str]]:
    """
    Yield lists of material_ids from summary.search in chunks.
    We request only material_id fields to minimize payload.
    """
    fields = ["material_id"]

    # mp-api supports chunking via num_chunks/chunk_size in search.
    # We don't know total upfront, so we just stream from the iterator and batch ourselves.
    docs = mpr.materials.summary.search(
        chemsys=chemsys,
        elements=elements,
        is_stable=True if stable_only else None,
        fields=fields,
        # chunking hints: these reduce server load for big pulls
        chunk_size=min(max(chunk_size, 1), 1000),
        num_chunks=None,
    )

    buf: List[str] = []
    for d in docs:
        mid = getattr(d, "material_id", None)
        if not mid:
            continue
        buf.append(str(mid))
        if len(buf) >= chunk_size:
            yield buf
            buf = []
    if buf:
        yield buf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("MP_API_KEY"), help="MP API key or set MP_API_KEY env var")
    ap.add_argument("--out", required=True, help="Output directory (CIFs + manifest.jsonl)")
    ap.add_argument("--max", type=int, default=10000, help="Target number of CIFs to download")
    ap.add_argument("--stable-only", action="store_true", help="If set, restrict to stable materials")
    ap.add_argument("--chemsys", default=None, help='Chemical system e.g. "Li-O" or "Li-Fe-O"')
    ap.add_argument("--elements", nargs="*", default=None, help='Elements list filter e.g. Li O (alternative to chemsys)')
    ap.add_argument("--chunk-size", type=int, default=200, help="How many IDs to batch per summary page buffer")
    ap.add_argument("--rps", type=float, default=8.0, help="Requests per second cap for structure fetches (conservative)")
    ap.add_argument("--resume", action="store_true", help="Skip IDs already downloaded / OK in manifest")
    ap.add_argument("--sleep", type=float, default=0.0, help="Extra sleep per request (in addition to rps pacing)")
    args = ap.parse_args()

    if not args.api_key:
        print("ERROR: set MP_API_KEY or pass --api-key", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    done_ids: Set[str] = load_done_ids(out_dir) if args.resume else set()
    downloaded = 0

    min_interval = 1.0 / max(args.rps, 0.1)
    last_call = 0.0

    def pace():
        nonlocal last_call
        now = time.time()
        dt = now - last_call
        if dt < min_interval:
            time.sleep(min_interval - dt)
        if args.sleep > 0:
            time.sleep(args.sleep)
        last_call = time.time()

    query_meta = {
        "stable_only": bool(args.stable_only),
        "chemsys": args.chemsys,
        "elements": args.elements,
        "max": args.max,
        "chunk_size": args.chunk_size,
        "rps": args.rps,
    }

    with MPRester(args.api_key) as mpr:
        for batch in iter_summary_ids(
            mpr,
            stable_only=args.stable_only,
            chemsys=args.chemsys,
            elements=args.elements,
            chunk_size=args.chunk_size,
        ):
            # Stop if reached target
            if downloaded >= args.max:
                break

            for mid in batch:
                if downloaded >= args.max:
                    break

                if args.resume and mid in done_ids:
                    continue

                cif_path = out_dir / safe_filename(f"{mid}.cif")

                try:
                    pace()
                    # Fetch structure (expensive endpoint). Keep requests paced.
                    structure = mpr.get_structure_by_material_id(mid)
                    cif_text = CifWriter(structure).write_string()

                    cif_path.write_text(cif_text, encoding="utf-8")

                    rec = {
                        "retrieved_at": now_iso(),
                        "status": "OK",
                        "material_id": mid,
                        "cif_path": str(cif_path),
                        "cif_sha256": sha256_text(cif_text),
                        "query": query_meta,
                    }
                    write_jsonl(manifest_path, rec)

                    downloaded += 1
                    if downloaded % 100 == 0:
                        print(f"[grab_mp_bulk] downloaded={downloaded}/{args.max} (last={mid})")

                except Exception as e:
                    err = {
                        "retrieved_at": now_iso(),
                        "status": "ERROR",
                        "material_id": mid,
                        "error": repr(e),
                        "query": query_meta,
                    }
                    write_jsonl(manifest_path, err)
                    # continue; keep going
                    print(f"[grab_mp_bulk] ERROR {mid}: {e}", file=sys.stderr)

    print(f"[grab_mp_bulk] DONE: downloaded={downloaded}, out={out_dir}, manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
