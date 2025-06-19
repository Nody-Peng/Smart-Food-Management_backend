import google.generativeai as genai
import os
from PIL import Image
import io
from typing import List
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageRecognitionService:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def analyze_food_image(self, image_data: bytes) -> List[str]:
        """分析食物圖片並返回食材清單"""
        try:
            # 將bytes轉換為PIL Image
            image = Image.open(io.BytesIO(image_data))
            logger.info(f"圖片大小: {image.size}")
            
            # 調整圖片大小以提高處理速度
            if image.size[0] > 1024 or image.size[1] > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                logger.info(f"調整後圖片大小: {image.size}")
            
            # 簡化的提示詞
            prompt = """
            請分析這張圖片中的食材，並列出主要的食材名稱。
            請用繁體中文回答，每個食材用逗號分隔。
            例如：番茄,洋蔥,雞蛋,青椒
            只列出可以用來烹飪的食材，最多8個。
            """
            
            # 調用Gemini API
            logger.info("正在調用 Gemini API...")
            response = self.model.generate_content([prompt, image])
            
            # 🔥 加入詳細除錯
            logger.info(f"API 回應狀態: {response}")
            logger.info(f"回應文字: '{response.text}'")
            logger.info(f"回應文字長度: {len(response.text) if response.text else 0}")
            
            # 解析回應
            ingredients_text = response.text.strip()
            logger.info(f"清理後的文字: '{ingredients_text}'")
            
            # 簡單的文字解析
            ingredients = []
            if ',' in ingredients_text:
                ingredients = [item.strip() for item in ingredients_text.split(',')]
            else:
                # 如果沒有逗號，嘗試其他分隔符
                ingredients = [item.strip() for item in ingredients_text.replace('、', ',').replace('\n', ',').split(',')]
            
            logger.info(f"分割後的食材: {ingredients}")
            
            # 過濾和清理
            cleaned_ingredients = []
            for ingredient in ingredients:
                if ingredient and len(ingredient) >= 2 and len(ingredient) <= 10:
                    cleaned_ingredients.append(ingredient)
            
            logger.info(f"最終食材清單: {cleaned_ingredients}")
            return cleaned_ingredients[:8]  # 限制最多8個
                
        except Exception as e:
            logger.error(f"圖片分析失敗: {e}")
            logger.error(f"錯誤類型: {type(e)}")
            import traceback
            logger.error(f"完整錯誤: {traceback.format_exc()}")
            # 回傳一些預設食材作為fallback
            return ["番茄", "洋蔥", "雞蛋"]
