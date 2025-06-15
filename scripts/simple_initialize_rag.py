# backend/scripts/simple_initialize_rag.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.data_processor import OptimizedRecipeProcessor
from services.simple_rag_service import SimpleRecipeRAGService
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """初始化簡化版 RAG 系統"""
    load_dotenv()
    
    # 第一步：處理原始資料
    logger.info("=== 第一步：處理原始 CSV 資料 ===")
    processor = OptimizedRecipeProcessor('data/recipes_data.csv')
    processor.save_processed_data()
    
    # 第二步：建立 TF-IDF 索引
    logger.info("=== 第二步：建立 TF-IDF 索引 ===")
    api_key = os.getenv("GOOGLE_API_KEY")
    rag_service = SimpleRecipeRAGService(api_key)
    rag_service.initialize_vector_store()
    
    logger.info("=== 簡化版 RAG 系統初始化完成！ ===")

if __name__ == "__main__":
    main()
