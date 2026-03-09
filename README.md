# vidfetch — Lightweight Video Retrieval

Content-based video retrieval using compact color-histogram features. No GPU or heavy deep learning; runs on CPU with minimal dependencies.

## Setup

```bash
cd vidfetch
pip install -r requirements.txt
```

## Quick start

1. **Add videos**  
   Put some videos (e.g. `.mp4`, `.avi`, `.mov`) in a folder. Default folder is `data/`:

   ```
   vidfetch/
   └── data/
       ├── clip1.mp4
       ├── clip2.mp4
       └── ...
   ```

2. **Build the index**

   ```bash
   python -m scripts.build_index data
   ```

   Index is saved under `index_store/` by default.

3. **Query** (find videos similar to a query video)

   ```bash
   python -m scripts.query data/clip1.mp4 --k 5
   ```

4. **Run evaluation** (precision@k, recall@k, retrieval time)

   ```bash
   python -m scripts.run_evaluation --index-dir index_store --k 5
   ```

## Options

- **build_index**
  - `video_dir` — folder with videos (default: `data`)
  - `--index-dir` — where to save index (default: `index_store`)
  - `--fps` — frames per second to sample (default: 1.0)
  - `--max-frames` — max frames per video (default: 50)

- **query**
  - `query_video` — path to the query video
  - `--index-dir` — index folder (default: `index_store`)
  - `--k` — number of results (default: 5)

- **run_evaluation**
  - `--index-dir` — index folder (default: `index_store`)
  - `--k` — k for precision@k and recall@k (default: 5)

## Project layout

- `src/extract.py` — frame sampling and HSV color histogram features
- `src/index.py` — in-memory index, save/load, L2 nearest-neighbor search
- `src/retrieval.py` — load index and run query
- `src/evaluate.py` — precision@k, recall@k, mean retrieval time
- `scripts/build_index.py` — CLI to build index from a video directory
- `scripts/query.py` — CLI to query by video path
- `scripts/run_evaluation.py` — CLI to run self-query evaluation

## For the presentation

- Use a small set of **5–10 short videos** in `data/` so indexing and query are fast.
- Run **build_index** once, then **query** with different videos to show retrieval.
- Run **run_evaluation** to show metrics (precision, recall, retrieval time).
