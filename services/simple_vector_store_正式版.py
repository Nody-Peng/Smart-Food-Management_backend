# backend/services/simple_vector_store.py
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from pathlib import Path
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import re
import google.generativeai as genai

logger = logging.getLogger(__name__)

class SimpleRecipeVectorStore:
    def __init__(self, api_key: str = None):
        """使用 TF-IDF 的簡單向量化方案，支援中文斷詞與自動翻譯查詢（Gemini）"""
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words=None,
            tokenizer=self.chinese_tokenizer
        )
        self.tfidf_matrix = None
        self.recipes_metadata = []
        self.is_built = False

        # 初始化 Gemini 模型（用於中文翻譯）
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None

    def chinese_tokenizer(self, text: str) -> List[str]:
        """中文分詞器"""
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        tokens = jieba.lcut(text)
        tokens = [token.strip() for token in tokens if len(token) > 1 and not token.isdigit()]
        return tokens

    def build_index_from_chunks(self, processed_data_dir: str = 'data/processed'):
        """從處理後的資料塊建立 TF-IDF 索引"""
        logger.info("開始建立 TF-IDF 索引...")

        index_file = f"{processed_data_dir}/index.json"
        if not Path(index_file).exists():
            raise FileNotFoundError("找不到處理後的資料索引檔案")

        with open(index_file, 'r', encoding='utf-8') as f:
            index_data = json.load(f)

        chunk_files = index_data['chunk_files']
        all_texts = []

        for chunk_file in chunk_files:
            logger.info(f"載入 {chunk_file}")
            with open(chunk_file, 'r', encoding='utf-8') as f:
                recipes_chunk = json.load(f)

            for recipe in recipes_chunk:
                all_texts.append(recipe['full_text'])
                self.recipes_metadata.append(recipe)

        logger.info(f"開始建立 TF-IDF 矩陣，共 {len(all_texts):,} 筆文字...")
        self.tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        self.is_built = True
        logger.info(f"TF-IDF 索引建立完成！矩陣大小: {self.tfidf_matrix.shape}")
        self.save_index()

    def save_index(self, index_dir: str = 'data/embeddings'):
        """儲存索引"""
        Path(index_dir).mkdir(parents=True, exist_ok=True)

        with open(f"{index_dir}/tfidf_vectorizer.pkl", 'wb') as f:
            pickle.dump(self.vectorizer, f)

        with open(f"{index_dir}/tfidf_matrix.pkl", 'wb') as f:
            pickle.dump(self.tfidf_matrix, f)

        with open(f"{index_dir}/recipes_metadata.pkl", 'wb') as f:
            pickle.dump(self.recipes_metadata, f)

        logger.info(f"索引已儲存至 {index_dir}")

    def load_index(self, index_dir: str = 'data/embeddings'):
        """載入索引"""
        try:
            with open(f"{index_dir}/tfidf_vectorizer.pkl", 'rb') as f:
                self.vectorizer = pickle.load(f)

            with open(f"{index_dir}/tfidf_matrix.pkl", 'rb') as f:
                self.tfidf_matrix = pickle.load(f)

            with open(f"{index_dir}/recipes_metadata.pkl", 'rb') as f:
                self.recipes_metadata = pickle.load(f)

            self.is_built = True
            logger.info(f"索引載入成功！矩陣大小: {self.tfidf_matrix.shape}")
            return True

        except Exception as e:
            logger.error(f"載入索引失敗: {e}")
            return False

    def detect_chinese(self, text: str) -> bool:
        """判斷是否包含中文"""
        return any('\u4e00' <= c <= '\u9fff' for c in text)

    def translate_to_english_with_gemini(self, text: str) -> str:
        """使用 Gemini 將中文翻譯為英文"""
        if not self.model:
            logger.warning("⚠️ Gemini 模型尚未初始化，無法翻譯")
            return text
        try:
            prompt = f"請將以下句子翻譯為英文，不要加解釋：{text}"
            response = self.model.generate_content(prompt)
            translated = response.text.strip()
            logger.info(f"🔁 查詢翻譯為英文：{translated}")
            return translated
        except Exception as e:
            logger.warning(f"❌ Gemini 翻譯失敗，改用原始查詢: {e}")
            return text

    def search(self, query: str, k: int = 10, min_similarity: float = 0.1) -> List[Tuple[Dict, float]]:
        """搜尋相似食譜，必要時將中文查詢翻譯為英文"""
        if not self.is_built:
            raise ValueError("索引尚未建立")

        # 如果 query 含中文，翻譯成英文
        if self.detect_chinese(query):
            query = self.translate_to_english_with_gemini(query)

        start_time = time.time()
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-k:][::-1]

        results = []
        for idx in top_indices:
            sim_score = float(similarities[idx])
            if sim_score >= min_similarity:
                recipe = self.recipes_metadata[idx]
                results.append((recipe, sim_score))

        search_time = time.time() - start_time
        logger.info(f"🔍 查詢「{query}」完成，共取得 {len(results)} 筆結果，耗時 {search_time:.3f} 秒")

        if results:
            for i, (r, score) in enumerate(results[:3]):
                logger.info(f"  {i+1}. {r['title']}（score: {score:.4f}）")
        else:
            logger.warning("⚠️ 查無相似結果")

        return results
