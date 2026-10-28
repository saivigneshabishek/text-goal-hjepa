import os
import torch
import argparse
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=os.environ.get("HJEPA_HOME"), help="Output directory (default: $HJEPA_HOME)")
    args = parser.parse_args()

    if args.out_dir is None:
        parser.error("Provide --out-dir or set $HJEPA_HOME")

    OUTPUT_DIR = args.out_dir / "antmaze_text_embeddings"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
    PROMPT = ("Instruct: Represent the destination of a maze navigation instruction, preserving its row and column.\nQuery: ")

    BATCH_SIZE = 8
    MAX_LENGTH = 128
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    MEDIUM_MAZE = (
        (1, 1, 1, 1, 1, 1, 1, 1),
        (1, 0, 0, 1, 1, 0, 0, 1),
        (1, 0, 0, 1, 0, 0, 0, 1),
        (1, 1, 0, 0, 0, 1, 1, 1),
        (1, 0, 0, 1, 0, 0, 0, 1),
        (1, 0, 1, 0, 0, 1, 0, 1),
        (1, 0, 0, 0, 1, 0, 0, 1),
        (1, 1, 1, 1, 1, 1, 1, 1),
    )

    WORDS = ("_", "one", "two", "three", "four", "five", "six")
    ORDINALS = ("_", "first", "second", "third", "fourth", "fifth", "sixth")

    TRAIN_TEMPLATES = (
        "Go to maze cell ({row}, {col}).",
        "Navigate to the cell at row {row_word}, column {col_word}.",
        "Reach the maze cell in row {row} and column {col}.",
        "Move the ant to row {row_word}, column {col_word} of the maze.",
        "Your destination is the cell at row {row}, column {col}.",
        "Guide the ant to cell ({row}, {col}) in the maze.",
        "Head to the cell in the {row_ordinal} row and {col_ordinal} column.",
        "Travel to maze row {row_word}, column {col_word}.",
    )

    VAL_TEMPLATES = (
        "Make your way to the cell at row {row}, column {col}.",
        "Bring the ant to the maze cell in the {row_ordinal} row and {col_ordinal} column.",
        "The target cell is at row {row_word} and column {col_word}.",
        "Please get to cell ({row}, {col}) of the maze.",
    )

    def build_records(templates):
        records = []
        for row, vals in enumerate(MEDIUM_MAZE):
            for col, wall in enumerate(vals):
                if wall == 1:
                    continue
                for idx, text in enumerate(templates):
                    records.append({
                        "template_id": idx,
                        "maze_cell": (row, col),
                        "text": text.format(
                            row=row, col=col,
                            row_word=WORDS[row], col_word=WORDS[col],
                            row_ordinal=ORDINALS[row], col_ordinal=ORDINALS[col],
                        )
                    })
        return records

    def save_embeddings(path, records, emb):
        np.savez_compressed(
            path,
            embeddings=emb.astype(np.float32, copy=False),
            texts=np.array([r["text"] for r in records]),
            cells=np.array([r["maze_cell"] for r in records], dtype=np.int32)
        )
        print(f"Saved {path}: {emb.shape}")

    train_records = build_records(TRAIN_TEMPLATES) # 26*8 = 208
    val_records  = build_records(VAL_TEMPLATES) # 26*4 = 104

    all_texts = [r["text"] for r in train_records] + [r["text"] for r in val_records]
    assert len(all_texts) == len(set(all_texts)), "Check for duplicate sentences"

    print("Train:", len(train_records), "| Val:", len(val_records))

    model = SentenceTransformer(
        MODEL_ID,
        device=DEVICE,
        model_kwargs={"dtype": torch.float32, "attn_implementation": "eager"}
    )
    model.max_seq_length = MAX_LENGTH
    model.eval().requires_grad_(False)

    with torch.inference_mode():
        vectors = model.encode(
            all_texts,
            prompt=PROMPT,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_tensor=True
        )
        vectors = vectors.float().cpu().numpy()

    n_train = len(train_records)
    train_emb = vectors[:n_train]
    val_emb  = vectors[n_train:]

    save_embeddings(OUTPUT_DIR/"train_text_embeddings.npz", train_records, train_emb)
    save_embeddings(OUTPUT_DIR/"eval_text_embeddings.npz", val_records, val_emb)

if __name__ == "__main__":
    main()
