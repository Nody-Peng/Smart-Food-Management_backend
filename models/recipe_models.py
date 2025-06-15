# backend/models/recipe_models.py
from pydantic import BaseModel
from typing import List, Optional
import json

class Recipe(BaseModel):
    id: str
    title: str
    ingredients: List[str]
    directions: List[str]
    ner_ingredients: List[str]
    source: str
    link: Optional[str] = None
    full_text: str  # 用於向量搜尋的完整文字
    
    @classmethod
    def from_csv_row(cls, row_data: dict, recipe_id: str):
        """從 CSV 行資料建立 Recipe 物件"""
        try:
            # 解析 JSON 格式的食材和步驟
            ingredients = json.loads(row_data.get('ingredients', '[]'))
            directions = json.loads(row_data.get('directions', '[]'))
            ner_ingredients = json.loads(row_data.get('NER', '[]'))
            
            # 建立完整搜尋文字
            full_text = f"{row_data.get('title', '')} {' '.join(ingredients)} {' '.join(directions)} {' '.join(ner_ingredients)}"
            
            return cls(
                id=recipe_id,
                title=row_data.get('title', ''),
                ingredients=ingredients,
                directions=directions,
                ner_ingredients=ner_ingredients,
                source=row_data.get('source', ''),
                link=row_data.get('link', ''),
                full_text=full_text
            )
        except Exception as e:
            print(f"解析食譜失敗: {e}")
            return None

class RecipeSearchRequest(BaseModel):
    query: str
    ingredients: Optional[List[str]] = []
    cuisine_type: Optional[str] = None
    max_results: int = 10
    
class RecipeSearchResponse(BaseModel):
    recipes: List[Recipe]
    total_found: int
    search_time: float
