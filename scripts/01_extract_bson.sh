#!/usr/bin/env bash
# Unzip the raw MongoDB dump and convert every .bson collection to JSON Lines.
# Schema-agnostic on purpose: we don't know collection names until we look, so
# this just converts everything under the dump and we inspect afterwards.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"
ZIP="$RAW/ThePublicJiraDataset.zip"
EXTRACT_DIR="$RAW/mongodump"
OUT_DIR="$ROOT/data/processed/jsonl"

if [ ! -f "$ZIP" ]; then
  echo "Missing $ZIP" >&2
  exit 1
fi

mkdir -p "$EXTRACT_DIR" "$OUT_DIR"

if [ -z "$(ls -A "$EXTRACT_DIR" 2>/dev/null)" ]; then
  echo "Unzipping $ZIP ..."
  unzip -q "$ZIP" -d "$EXTRACT_DIR"
else
  echo "Already extracted at $EXTRACT_DIR, skipping unzip."
fi

echo "Looking for .bson files..."
find "$EXTRACT_DIR" -name '*.bson' | while read -r bsonfile; do
  rel="${bsonfile#"$EXTRACT_DIR"/}"
  collection="$(basename "$bsonfile" .bson)"
  db_dir="$(dirname "$rel")"
  out_sub="$OUT_DIR/$db_dir"
  mkdir -p "$out_sub"
  out_file="$out_sub/$collection.jsonl"
  if [ -s "$out_file" ]; then
    echo "skip (exists): $out_file"
    continue
  fi
  echo "bsondump: $rel -> $out_file"
  bsondump --quiet "$bsonfile" > "$out_file"
done

echo "Done. Output under $OUT_DIR"
du -sh "$OUT_DIR"/* 2>/dev/null || true
