# BIS Standards Recommendation Engine

This project recommends the top BIS building-material standards for a product
description. It is built for the hackathon evaluation command:

```bash
python inference.py --input hidden_private_dataset.json --output team_results.json
```

## Approach

- `src/preprocess.py` extracts SP 21 summaries from `dataset.pdf` into
  `data/processed_data.json`.
- `src/retrever.py` builds a local BM25-style lexical retriever with title,
  standard-number, and material-word boosts.
- `src/pipeline.py` exposes the recommendation pipeline.
- `inference.py` reads input JSON and writes the required output schema.

The inference path is fully local and does not require an LLM API key or model
download, so latency stays well below the 5 second target.

## Rebuild The Index

If `dataset.pdf` changes, rebuild the processed index:

```bash
python src/preprocess.py --pdf dataset.pdf --output data/processed_data.json
```

`pdftotext` from Poppler is preferred for preprocessing because it preserves the
SP 21 heading order better than basic PDF loaders.

## Local Validation

```bash
python inference.py \
  --input "/home/ujjwal/Desktop/BIS Hackathon Materials/public_test_set.json" \
  --output /tmp/bis_public_results.json

python eval_script.py --results /tmp/bis_public_results.json
```

## Demo UI

Install the UI dependency and run the Streamlit interface:

```bash
pip install -r requirements.txt
streamlit run interface.py
```
