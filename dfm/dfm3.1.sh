zcat *.jsonl.gz > dfm-3.1.jsonl
python clean.py dfm-3.1.jsonl
MEMORY=42 SEED=42 terashuf < dfm3.1.jsonl > dfm3.1-shuf.jsonl
head -n 1177737 dfm3.1-shuf.jsonl > dfm3.1-validation.jsonl
tail -n +1177738 dfm3.1-shuf.jsonl > dfm3.1-train.jsonl
split -a 2 -C 1GB -d dfm3.1-validation.jsonl dfm3.1-validation- --additional-suffix=.jsonl
split -a 2 -C 1GB -d dfm3.1-train.jsonl dfm3.1-train- --additional-suffix=.jsonl
