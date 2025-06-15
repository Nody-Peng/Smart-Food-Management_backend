# backend/services/simple_rag_service.py
import logging
from typing import List, Dict, Any
import google.generativeai as genai
from .simple_vector_store import SimpleRecipeVectorStore

logger = logging.getLogger(__name__)

class SimpleRecipeRAGService:
    def __init__(self, api_key: str):
        self.vector_store = SimpleRecipeVectorStore()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # 嘗試載入現有索引
        if not self.vector_store.load_index():
            logger.warning("找不到現有的向量索引，需要先建立索引")
    
    def initialize_vector_store(self):
        """初始化向量資料庫"""
        logger.info("開始初始化向量資料庫...")
        self.vector_store.build_index_from_chunks()
        logger.info("向量資料庫初始化完成")
    
    def search_recipes(self, query: str, max_results: int = 5) -> List[Dict]:
        """搜尋相關食譜"""
        if not self.vector_store.is_built:
            raise ValueError("向量索引尚未建立，請先執行初始化")
        
        results = self.vector_store.search(query, k=max_results)
        
        # 格式化結果
        formatted_results = []
        for recipe, score in results:
            formatted_results.append({
                'recipe': recipe,
                'similarity_score': score
            })
        
        return formatted_results
    
    def generate_enhanced_recipe(self, user_ingredients: List[str], 
                                cuisine: str = "chinese", 
                                cooking_time: str = "30") -> str:
        """使用 RAG 生成增強的食譜"""
        
        # 1. 搜尋相關食譜
        ingredients_query = " ".join(user_ingredients)
        similar_recipes = self.search_recipes(ingredients_query, max_results=3)
        
        # 2. 準備參考食譜文字
        reference_recipes = []
        for item in similar_recipes:
            recipe = item['recipe']
            reference_text = f"""
食譜名稱: {recipe['title']}
食材: {', '.join(recipe['ingredients'][:5])}
烹飪步驟: {' '.join(recipe['directions'][:3])}
"""
            reference_recipes.append(reference_text)
        
        # 3. 建構增強的提示詞
        reference_text = "\n---\n".join(reference_recipes)
        
        cuisine_map = {
            "chinese": "中式",
            "japanese": "日式", 
            "korean": "韓式",
            "western": "西式",
            "thai": "泰式",
            "italian": "義式"
        }
        
        cuisine_name = cuisine_map.get(cuisine, "中式")
        ingredients_text = "、".join(user_ingredients)
        
        prompt = f"""
你是一位專業的料理顧問。請根據用戶提供的食材和以下參考食譜，創作一道新的{cuisine_name}料理。

用戶食材: {ingredients_text}
料理風味: {cuisine_name}
烹飪時間: 約{cooking_time}分鐘

參考食譜:
{reference_text}

請根據參考食譜的靈感，結合用戶的食材，創作一道新的料理。請提供：

1. 🍽️ 料理名稱
2. 📋 完整食材清單（包含份量）
3. 👨‍🍳 詳細烹飪步驟
4. 💡 料理小貼士
5. 🌟 營養價值說明

請用繁體中文回答，格式清楚易讀。
"""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"RAG 食譜生成失敗: {e}")
            return "抱歉，食譜生成失敗，請稍後再試。"
