import os
import re
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModel


class DataPreprocessor:
    def __init__(self, data_dir, seed=42, clean_text=True):
        """
        Initializes the preprocessor and loads data splits.
        """
        self.seed = seed
        self.clean_text = clean_text
        self.data_dir = data_dir
        np.random.seed(seed)

        # Load each split from the dataset directory
        train_path = os.path.join(data_dir, 'training.csv')
        test_path  = os.path.join(data_dir, 'testing.csv')
        val_path   = os.path.join(data_dir, 'validation.csv')

        # Merge all splits before re-splitting according to Train/Test/RAG strategy
        self.data = pd.concat([
            pd.read_csv(train_path),
            pd.read_csv(test_path),
            pd.read_csv(val_path)
        ], ignore_index=True)
        print(f"Total records loaded: {len(self.data)}")

        if self.clean_text and 'text' in self.data.columns:
            self.data['text'] = self.data['text'].apply(self._clean_text)

    def _clean_text(self, text):
        if pd.isna(text): return ""
        # Standardize whitespace and remove redundant newlines
        text = re.sub(r'\s+', ' ', str(text)).strip()
        return text

    def deduplicate(self):
        """Removes overlapping entries based on content to prevent data leakage."""
        before = len(self.data)
        subset_col = ['text'] if 'text' in self.data.columns else None
        self.data = self.data.drop_duplicates(subset=subset_col).reset_index(drop=True)
        print(f"Dropped {before - len(self.data)} duplicate records.")
        return self

    def split_dataset(self, train_ratio=0.50, test_ratio=0.25):
        """
        1. Training set: for graph optimization.
        2. Test set: for final evaluation.
        3. RAG pool: for retrieval augmentation during inference.
        """
        stratify_col = self.data['label'] if 'label' in self.data.columns else None

        # Carve out training set
        train_df, temp_df = train_test_split(
            self.data,
            test_size=(1 - train_ratio),
            stratify=stratify_col,
            random_state=self.seed
        )

        # Re-stratify the remaining for Test and RAG pool (50/50 split of the remainder)
        temp_strat = temp_df['label'] if stratify_col is not None else None
        test_df, rag_df = train_test_split(
            temp_df,
            test_size=0.5,
            stratify=temp_strat,
            random_state=self.seed
        )

        self.train_set, self.test_set, self.rag_pool = train_df, test_df, rag_df
        print(f"Data split sizes: Train={len(train_df)}, Test={len(test_df)}, RAG={len(rag_df)}")
        return self

    def encode_with_sliding_window(self, text, tokenizer, model, device, max_length=512, stride=256):
        """
        Handles long texts by averaging CLS embeddings over multiple windows.
        """
        tokens = tokenizer.encode(text, add_special_tokens=False)

        # Short text: encode directly
        if len(tokens) <= max_length - 2:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                inputs = tokenizer(text, truncation=True, padding=True,
                                   max_length=max_length, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                return model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()[0]

        # Long text: decode each window back to text and re-tokenize
        # so that attention_mask and token_type_ids are handled correctly
        window_embs = []
        window_size = max_length - 2
        for start in range(0, len(tokens), stride):
            end = min(start + window_size, len(tokens))
            window_text = tokenizer.decode(tokens[start:end], skip_special_tokens=True)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                inputs = tokenizer(window_text, truncation=True, padding=True,
                                   max_length=max_length, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()[0]
                window_embs.append(emb)
            if end >= len(tokens):
                break

        return np.mean(window_embs, axis=0)

    def encode_all(self, model_name='mental/mental-bert-base-uncased'):
        """Computes embeddings for all splits using the specified transformer model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device).eval()

        print(f"Encoding datasets with {model_name}...")
        results = {}
        for name, df in [('train', self.train_set), ('test', self.test_set), ('rag', self.rag_pool)]:
            embs = []
            for i, text in enumerate(df['text'].tolist()):
                embs.append(self.encode_with_sliding_window(text, self.tokenizer, self.model, self.device))
                if (i + 1) % 200 == 0: print(f"  Processed {i + 1} samples in {name}")

            # Normalization is essential for cosine similarity search later
            results[name] = normalize(np.vstack(embs))

        self.train_embs, self.test_embs, self.rag_embs = results['train'], results['test'], results['rag']
        return self

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        # Save split CSVs and embeddings
        self.train_set.to_csv(os.path.join(output_dir, 'train_set.csv'), index=False)
        self.test_set.to_csv(os.path.join(output_dir, 'test_set.csv'), index=False)
        self.rag_pool.to_csv(os.path.join(output_dir, 'rag_pool.csv'), index=False)

        np.save(os.path.join(output_dir, 'train_embeddings.npy'), self.train_embs)
        np.save(os.path.join(output_dir, 'test_embeddings.npy'), self.test_embs)
        np.save(os.path.join(output_dir, 'rag_embeddings.npy'), self.rag_embs)
        print(f"Outputs saved to {output_dir}")


if __name__ == "__main__":
    preprocessor = DataPreprocessor(
        data_dir='<path/to/dataset>',
        seed=42,
        clean_text=True
    )
    preprocessor.deduplicate()
    preprocessor.split_dataset(train_ratio=0.50, test_ratio=0.25)
    preprocessor.encode_all(model_name='mental/mental-bert-base-uncased')
    preprocessor.save(output_dir='<path/to/output/directory>')