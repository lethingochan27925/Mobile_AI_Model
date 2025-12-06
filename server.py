from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
from tensorflow.keras.models import load_model
import joblib

#------------------ DEFINE INPUT SCHEMA ------------------
class UserVector(BaseModel):
    user_vector: List[List[float]]  # dạng [[0,1,0,...]] mỗi user là 1 list

#------------------ INIT APP ------------------
app = FastAPI(title="Sweet Recommendation API", version="1.0")

#------------------ LOAD MODEL + SCALER + PRODUCT LIST ------------------
autoencoder = load_model("autoencoder_model.h5")
encoder = load_model("encoder_model.h5")
scaler = joblib.load("scaler.pkl")
product_columns = joblib.load("product_columns.pkl")

#------------------ HELPER FUNCTION ------------------
def get_top_k_products(user_scaled: np.ndarray, k: int = 4):
    predicted = autoencoder.predict(user_scaled)[0]
    top_idx = np.argsort(predicted)[::-1][:k]
    top_products = [{"product_id": str(product_columns[i]), "score": float(predicted[i])} 
                    for i in top_idx]
    return top_products

#------------------ ROUTES ------------------
@app.post("/recommend", summary="Recommend top 4 products for a user")
async def recommend(data: UserVector):
    user_vector = np.array(data.user_vector)
    
    # Scale
    user_scaled = scaler.transform(user_vector)
    
    # Get top 4 products
    top_products = get_top_k_products(user_scaled, k=4)
    
    return {"top_products": top_products}

#-------------------------RETRAIN MỖI KHI CÓ ĐƠN MỚI
@app.post("/retrain")
async def retrain(data: dict):
    user_id = data.get("userId")
    
    # 1. Build user-item matrix mới (cả user cũ + mới)
    X = build_user_item_matrix()  # trả về toàn bộ user-item matrix
    
    # 2. Train lại model
    autoencoder.fit(X, X, epochs=5, batch_size=32, verbose=0)
    
    # 3. Lưu model
    autoencoder.save("model_autoencoder.h5")
    
    return {"success": True}

#------------------ RUN ------------------
# uvicorn main:app --reload
