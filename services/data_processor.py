# backend/services/data_processor.py
import pandas as pd
import json
import logging
from typing import List, Dict, Any, Generator
import re
from pathlib import Path
from tqdm import tqdm
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedRecipeProcessor:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.chunk_size = 5000  # 每次處理5000筆
    
    def clean_text(self, text: str) -> str:
        """清理文字"""
        if pd.isna(text) or not text:
            return ""
        text = str(text).strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def process_recipe_row(self, row: pd.Series, recipe_id: str) -> Dict[str, Any]:
        """處理單一食譜行"""
        try:
            # 解析 JSON 資料
            ingredients = json.loads(row['ingredients']) if pd.notna(row['ingredients']) else []
            directions = json.loads(row['directions']) if pd.notna(row['directions']) else []
            ner_ingredients = json.loads(row['NER']) if pd.notna(row['NER']) else []
            
            # 清理文字
            title = self.clean_text(row['title'])
            
            # 建立搜尋用文字
            search_text_parts = [
                title,
                ' '.join(ingredients),
                ' '.join(directions),
                ' '.join(ner_ingredients)
            ]
            full_text = ' '.join([part for part in search_text_parts if part])
            
            return {
                'id': recipe_id,
                'title': title,
                'ingredients': ingredients,
                'directions': directions,
                'ner_ingredients': ner_ingredients,
                'source': self.clean_text(row.get('source', '')),
                'link': self.clean_text(row.get('link', '')),
                'full_text': full_text
            }
            
        except Exception as e:
            logger.warning(f"處理食譜 {recipe_id} 失敗: {e}")
            return None
    
    def process_in_chunks(self) -> Generator[List[Dict], None, None]:
        """分塊處理資料"""
        logger.info(f"開始分塊處理 {self.csv_path}")
        
        chunk_number = 0
        for chunk in pd.read_csv(self.csv_path, chunksize=self.chunk_size):
            logger.info(f"處理第 {chunk_number + 1} 塊，共 {len(chunk)} 筆資料")
            
            processed_recipes = []
            for idx, row in chunk.iterrows():
                recipe_id = f"recipe_{chunk_number * self.chunk_size + idx}"
                recipe = self.process_recipe_row(row, recipe_id)
                
                if recipe and recipe['title']:  # 只保留有標題的食譜
                    processed_recipes.append(recipe)
            
            logger.info(f"第 {chunk_number + 1} 塊處理完成，有效食譜: {len(processed_recipes)}")
            chunk_number += 1
            yield processed_recipes
    
    def save_processed_data(self, output_dir: str = 'data/processed'):
        """儲存處理後的資料"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        all_recipes = []
        chunk_files = []
        
        for chunk_idx, recipes_chunk in enumerate(self.process_in_chunks()):
            # 儲存每個塊到單獨檔案
            chunk_file = f"{output_dir}/recipes_chunk_{chunk_idx}.json"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(recipes_chunk, f, ensure_ascii=False, indent=2)
            
            chunk_files.append(chunk_file)
            all_recipes.extend(recipes_chunk)
            
            # 每10萬筆顯示進度
            if len(all_recipes) % 100000 == 0:
                logger.info(f"已處理 {len(all_recipes):,} 筆食譜")
        
        # 儲存索引檔案
        index_data = {
            'total_recipes': len(all_recipes),
            'chunk_files': chunk_files,
            'processing_complete': True
        }
        
        with open(f"{output_dir}/index.json", 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"處理完成！總共 {len(all_recipes):,} 筆有效食譜")
        return all_recipes
