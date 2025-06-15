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

logger = logging.getLogger(__name__)

class SimpleRecipeVectorStore:
    def __init__(self):
        """使用 TF-IDF 的簡單向量化方案"""
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words=None,
            tokenizer=self.chinese_tokenizer
        )
        self.tfidf_matrix = None
        self.recipes_metadata = []
        self.is_built = False
        
    def chinese_tokenizer(self, text: str) -> List[str]:
        """中文分詞器"""
        # 清理文字
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        # 使用 jieba 分詞
        tokens = jieba.lcut(text)
        # 過濾短詞和數字
        tokens = [token.strip() for token in tokens if len(token) > 1 and not token.isdigit()]
        return tokens
    
    def build_index_from_chunks(self, processed_data_dir: str = 'data/processed'):
        """從處理後的資料塊建立 TF-IDF 索引"""
        logger.info("開始建立 TF-IDF 索引...")
        
        # 讀取索引檔案
        index_file = f"{processed_data_dir}/index.json"
        if not Path(index_file).exists():
            raise FileNotFoundError("找不到處理後的資料索引檔案")
        
        with open(index_file, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        chunk_files = index_data['chunk_files']
        
        # 收集所有文字
        all_texts = []
        
        for chunk_file in chunk_files:
            logger.info(f"載入 {chunk_file}")
            with open(chunk_file, 'r', encoding='utf-8') as f:
                recipes_chunk = json.load(f)
            
            for recipe in recipes_chunk:
                all_texts.append(recipe['full_text'])
                self.recipes_metadata.append(recipe)
        
        logger.info(f"開始建立 TF-IDF 矩陣，共 {len(all_texts):,} 筆文字...")
        
        # 建立 TF-IDF 矩陣
        self.tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        self.is_built = True
        logger.info(f"TF-IDF 索引建立完成！矩陣大小: {self.tfidf_matrix.shape}")
        
        # 儲存索引
        self.save_index()
    
    def save_index(self, index_dir: str = 'data/embeddings'):
        """儲存索引"""
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        
        # 儲存 TF-IDF 向量化器和矩陣
        with open(f"{index_dir}/tfidf_vectorizer.pkl", 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(f"{index_dir}/tfidf_matrix.pkl", 'wb') as f:
            pickle.dump(self.tfidf_matrix, f)
        
        # 儲存 metadata
        with open(f"{index_dir}/recipes_metadata.pkl", 'wb') as f:
            pickle.dump(self.recipes_metadata, f)
        
        logger.info(f"索引已儲存至 {index_dir}")
    
    def load_index(self, index_dir: str = 'data/embeddings'):
        """載入索引"""
        try:
            # 載入 TF-IDF 向量化器和矩陣
            with open(f"{index_dir}/tfidf_vectorizer.pkl", 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            with open(f"{index_dir}/tfidf_matrix.pkl", 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
            
            # 載入 metadata
            with open(f"{index_dir}/recipes_metadata.pkl", 'rb') as f:
                self.recipes_metadata = pickle.load(f)
            
            self.is_built = True
            logger.info(f"索引載入成功！矩陣大小: {self.tfidf_matrix.shape}")
            return True
            
        except Exception as e:
            logger.error(f"載入索引失敗: {e}")
            return False
    
    def search(self, query: str, k: int = 10) -> List[Tuple[Dict, float]]:
        """搜尋相似食譜"""
        if not self.is_built:
            raise ValueError("索引尚未建立")
        
        start_time = time.time()
        
        # 向量化查詢
        query_vector = self.vectorizer.transform([query])
        
        # 計算相似度
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # 取得前 k 個結果
        top_indices = similarities.argsort()[-k:][::-1]
        
        # 準備結果
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # 只返回有相似度的結果
                recipe = self.recipes_metadata[idx]
                results.append((recipe, float(similarities[idx])))
        
        search_time = time.time() - start_time
        logger.info(f"搜尋完成，耗時: {search_time:.3f}秒")
        
        return results
