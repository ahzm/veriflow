.PHONY: run bench test

run:
	python -m veriflow.cli verify --input bench/T001/gold.json --prompt "$$(cat bench/T001/prompt.txt)"

bench:
	python -m veriflow.cli bench --glob "bench/*/gold.json" --out experiments/results/report.csv

test:
	pytest -q
