from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
import logging

# 延遲導入以避免版本衝突
try:
    from services.image_recognition import ImageRecognitionService
except ImportError as e:
    print(f"圖片識別導入錯誤: {e}")
    ImageRecognitionService = None

# 導入 RAG 服務
try:
    from services.simple_rag_service import SimpleRecipeRAGService
except ImportError as e:
    print(f"RAG 服務導入錯誤: {e}")
    SimpleRecipeRAGService = None

load_dotenv()

# 設定日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="剩食幫手 API",
    description="整合圖片識別與智能食譜推薦的完整 API",
    version="2.0.0"
)

# 設定CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全域服務實例
image_service = None
rag_service = None

@app.on_event("startup")
async def startup_event():
    """應用啟動時初始化所有服務"""
    global image_service, rag_service
    
    logger.info("🚀 正在啟動剩食幫手系統...")
    
    # 🔑 統一使用 GEMINI_API_KEY
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        logger.error("❌ 未找到 GEMINI_API_KEY 環境變數")
        logger.error("請在 .env 檔案中設定: GEMINI_API_KEY=your_api_key")
        return
    
    logger.info("✅ 找到 GEMINI_API_KEY，開始初始化服務...")
    
    # 初始化圖片識別服務
    if ImageRecognitionService:
        try:
            image_service = ImageRecognitionService(api_key=api_key)
            logger.info("✅ 圖片識別服務初始化成功")
        except Exception as e:
            logger.error(f"❌ 圖片識別服務初始化失敗: {e}")
    else:
        logger.warning("⚠️ 圖片識別服務類別未找到")
    
    # 初始化 RAG 服務（使用同一個 API Key）
    if SimpleRecipeRAGService:
        try:
            rag_service = SimpleRecipeRAGService(api_key)  # 🔄 使用同一個 key
            
            # 檢查索引是否存在
            if not rag_service.vector_store.is_built:
                logger.error("❌ 向量索引尚未建立！")
                logger.error("請先執行: python scripts/simple_initialize_rag.py")
            else:
                logger.info("✅ RAG 服務初始化成功！")
                logger.info(f"📊 索引大小: {rag_service.vector_store.tfidf_matrix.shape}")
        except Exception as e:
            logger.error(f"❌ RAG 服務初始化失敗: {e}")
    else:
        logger.warning("⚠️ RAG 服務類別未找到")

# API 模型定義
class ImageAnalysisResponse(BaseModel):
    ingredients: List[str]
    success: bool
    message: str

class RecipeRequest(BaseModel):
    ingredients: List[str]
    cuisine: Optional[str] = "chinese"
    cooking_time: Optional[str] = "30"

class SearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5

class RecipeResponse(BaseModel):
    recipe: str
    reference_count: int

class SearchResponse(BaseModel):
    results: List[dict]
    total_found: int

class CompleteFlowResponse(BaseModel):
    ingredients: List[str]
    recipe: str
    reference_count: int
    success: bool
    message: str

# ==================== 基礎端點 ====================
@app.get("/")
async def root():
    return {
        "message": "剩食幫手 API 運行中", 
        "status": "ok",
        "version": "2.0.0",
        "features": ["圖片識別", "食譜推薦", "完整流程"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "image_service_available": image_service is not None,
        "rag_service_available": rag_service is not None and rag_service.vector_store.is_built,
        "api_key_configured": bool(os.getenv("GEMINI_API_KEY")),  # 🔄 只檢查一個 key
        "index_size": rag_service.vector_store.tfidf_matrix.shape if rag_service and rag_service.vector_store.is_built else None
    }

# ==================== 圖片識別端點 ====================
@app.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """分析上傳的食物圖片"""
    if not image_service:
        raise HTTPException(
            status_code=503, 
            detail="圖片識別服務未初始化，請檢查 GEMINI_API_KEY 設定"
        )
    
    try:
        # 檢查檔案類型
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="請上傳圖片檔案")
        
        # 檢查檔案大小 (5MB 限制)
        contents = await file.read()
        if len(contents) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="檔案太大，請上傳小於 5MB 的圖片")
        
        # 調用圖片分析
        logger.info(f"📷 分析圖片，檔案大小: {len(contents)} bytes")
        ingredients = image_service.analyze_food_image(contents)
        
        return ImageAnalysisResponse(
            ingredients=ingredients,
            success=True,
            message=f"成功識別出 {len(ingredients)} 種食材"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"圖片分析錯誤: {e}")
        return ImageAnalysisResponse(
            ingredients=["番茄", "洋蔥", "雞蛋"],  # fallback
            success=False,
            message=f"圖片分析失敗，使用預設食材: {str(e)}"
        )

# ==================== RAG 食譜端點 ====================
@app.post("/generate-recipe", response_model=RecipeResponse)
async def generate_recipe(request: RecipeRequest):
    """根據食材生成智能食譜"""
    if not rag_service or not rag_service.vector_store.is_built:
        raise HTTPException(
            status_code=503, 
            detail="RAG 服務未初始化或索引未建立，請檢查 GEMINI_API_KEY 設定並執行索引建立"
        )
    
    try:
        logger.info(f"🍳 生成食譜請求 - 食材: {request.ingredients}")
        
        recipe = rag_service.generate_enhanced_recipe(
            user_ingredients=request.ingredients,
            cuisine=request.cuisine,
            cooking_time=request.cooking_time
        )
        
        # 搜尋參考食譜數量
        ingredients_query = " ".join(request.ingredients)
        similar_recipes = rag_service.search_recipes(ingredients_query, max_results=3)
        
        return RecipeResponse(
            recipe=recipe,
            reference_count=len(similar_recipes)
        )
        
    except Exception as e:
        logger.error(f"❌ 食譜生成失敗: {e}")
        raise HTTPException(status_code=500, detail=f"食譜生成失敗: {str(e)}")

@app.post("/search-recipes", response_model=SearchResponse)
async def search_recipes(request: SearchRequest):
    """搜尋相似食譜"""
    if not rag_service or not rag_service.vector_store.is_built:
        raise HTTPException(
            status_code=503, 
            detail="RAG 服務未初始化或索引未建立"
        )
    
    try:
        logger.info(f"🔍 搜尋食譜 - 查詢: {request.query}")
        
        results = rag_service.search_recipes(
            query=request.query,
            max_results=request.max_results
        )
        
        return SearchResponse(
            results=results,
            total_found=len(results)
        )
        
    except Exception as e:
        logger.error(f"❌ 食譜搜尋失敗: {e}")
        raise HTTPException(status_code=500, detail=f"食譜搜尋失敗: {str(e)}")

# ==================== 完整流程端點 ====================
@app.post("/image-to-recipe")
async def image_to_recipe(file: UploadFile = File(...)):
    """完整流程：圖片 → 食材識別 → 食譜生成"""
    if not image_service:
        raise HTTPException(status_code=503, detail="圖片識別服務未初始化")
    
    if not rag_service or not rag_service.vector_store.is_built:
        raise HTTPException(status_code=503, detail="RAG 服務未初始化")
    
    try:
        # 1. 圖片分析
        contents = await file.read()
        if len(contents) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="檔案太大")
        
        logger.info("📷 開始完整流程：圖片分析...")
        ingredients = image_service.analyze_food_image(contents)
        
        # 2. 生成食譜
        logger.info(f"🍳 識別到食材: {ingredients}，開始生成食譜...")
        recipe = rag_service.generate_enhanced_recipe(
            user_ingredients=ingredients,
            cuisine="chinese",
            cooking_time="30"
        )
        
        # 3. 搜尋參考數量
        ingredients_query = " ".join(ingredients)
        similar_recipes = rag_service.search_recipes(ingredients_query, max_results=3)
        
        return CompleteFlowResponse(
            ingredients=ingredients,
            recipe=recipe,
            reference_count=len(similar_recipes),
            success=True,
            message=f"成功從圖片識別出 {len(ingredients)} 種食材並生成食譜"
        )
        
    except Exception as e:
        logger.error(f"❌ 完整流程失敗: {e}")
        raise HTTPException(status_code=500, detail=f"流程失敗: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
