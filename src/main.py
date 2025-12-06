from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List
import numpy as np

from models import autoencoder, encoder, scaler, product_columns
from utils import get_top_k_products
from data import retrain_model

from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from collections import Counter

from dotenv import load_dotenv
import os

load_dotenv()

# -------------------- MongoDB --------------------
client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
db = client["SweetDB"]
product_collection = db["SANPHAM"]

# -------------------- Models --------------------
class UserVector(BaseModel):
    user_vector: List[List[float]]

class CartItems(BaseModel):
    cart_product_ids: List[str]  # danh sách product_id trong giỏ hàng


app = FastAPI(title="Sweet Recommendation API", version="1.0")


# -------------------- Helper --------------------
async def get_product_category(product_id: str):

    print("Fetching category for product_id:", product_id)
    product = await product_collection.find_one({"_id": ObjectId(product_id)})
    if not product:
        return None

    return product.get("category")


async def detect_cross_category(cart_ids):
    categories = []

    for pid in cart_ids:
        c = await get_product_category(pid)
        if c:
            categories.append(c)

    if not categories:
        return None

    main_cat = Counter(categories).most_common(1)[0][0]

    # CROSS-CATEGORY MAP
    cross_map = {
        "Bánh kem": ["Đồ uống"],
        "Bánh mì": ["Đồ uống"],
        "Bánh mini": ["Đồ uống"],
        "Bánh ngọt": ["Đồ uống"],
        "Bánh quy": ["Đồ uống"],

        # "Đồ uống": ["Bánh kem"],
        # "Đồ uống": ["Bánh mì"],
        # "Đồ uống": ["Bánh mini"],
        # "Đồ uống": ["Bánh ngọt"],
        # "Đồ uống": ["Bánh quy"],
    }

    return cross_map.get(main_cat, [])

async def detect_cross_category(cart_ids):

    print("\n===== Detecting cross category =====")
    categories = []

    # Lấy category của các sản phẩm trong giỏ hàng
    for pid in cart_ids:
        c = await get_product_category(pid)
        if c:
            categories.append(c)
    
    print("Categories in cart:", categories)

    if not categories:
        return []

    # Lấy category xuất hiện nhiều nhất
    main_cat = Counter(categories).most_common(1)[0][0]

    print("Main category:", main_cat)

    # CROSS-CATEGORY MAP (KHÔNG TRÙNG KEY)
    cross_map = {
        "Bánh kem": ["Đồ uống"],
        "Bánh mì": ["Đồ uống"],
        "Bánh mini": ["Đồ uống"],
        "Bánh ngọt": ["Đồ uống"],
        "Bánh quy": ["Đồ uống"],

        "Đồ uống": ["Bánh kem", "Bánh mì", "Bánh mini", "Bánh ngọt", "Bánh quy"],
    }


    return cross_map.get(main_cat, [])


# -------------------- Recommendation API --------------------
@app.post("/recommend_cart")
async def recommend_cart(data: CartItems):
    """
    Gợi ý sản phẩm mua kèm dựa trên giỏ hàng – CROSS CATEGORY
    """
    print("\n======================")
    print("CART ITEMS:", data.cart_product_ids)
    print("======================")

    # --- 1. Build input vector ---
    input_vector = np.zeros(len(product_columns))

    # print(product_columns)

    for pid in data.cart_product_ids:
        if pid in product_columns:
            idx = product_columns.index(pid)
            input_vector[idx] = 1

    print("Input vector:", input_vector.tolist())

    input_scaled = scaler.transform([input_vector])
    pred = autoencoder.predict(input_scaled)[0]

    print("Raw pred from model:", pred.tolist())

    # --- 2. Lấy cross-category ---
    cross_categories = await detect_cross_category(data.cart_product_ids)

    print("Cross categories final:", cross_categories)

    if not cross_categories:
        print("No cross category → return []")
        return {"top_products": []}
    

    # --- 3. Lọc sản phẩm không thuộc cross-category ---
    # 3. Lọc sản phẩm thuộc cross-category
    print("\nFILTERING…")
    for i, pid in enumerate(product_columns):
        cat = await get_product_category(pid)

        # Nếu sản phẩm không lấy được category → bỏ qua (không loại)
        if not cat:
            print(f"Product {pid} has NO category – skipping filter")
            continue

        # Nếu category của sản phẩm không thuộc cross list → loại
        if cat not in cross_categories:
            pred[i] = -1
            
    print("Pred after filtering:", pred.tolist())

    # --- 4. Loại sản phẩm đã chọn ---
    for pid in data.cart_product_ids:
        if pid in product_columns:
            pred[product_columns.index(pid)] = -1
    
    print("Pred before selecting top:", pred.tolist())


    # --- 5. Lấy top 4 ---
    top_idx = np.argsort(pred)[::-1][:4]
    top_products = [product_columns[i] for i in top_idx]



    return {"top_products": top_products}


# -------------------- Retrain --------------------
@app.post("/retrain")
async def retrain(data: dict, background_tasks: BackgroundTasks):
    background_tasks.add_task(retrain_model)
    return {"success": True, "message": "Retrain đang chạy nền."}


#--------------------- RECOMMEND HOME ----------------
@app.post("/recommend")
async def recommend_home(data: UserVector):
    """
    Nhận user vector và trả về top 4 sản phẩm gợi ý.
    """
    try:
        print("\n===== HOME RECOMMEND CALLED =====")
        print("User vector:", data.user_vector)

        # Predict từ autoencoder
        user_vector = np.array(data.user_vector)

        scaled = scaler.transform(user_vector)
        pred = autoencoder.predict(scaled)[0]

        print("Model prediction:", pred.tolist())

        # Lấy top 4 sản phẩm có score cao nhất
        top_idx = np.argsort(pred)[::-1][:4]

        top_products = []
        for idx in top_idx:
            pid = product_columns[idx]
            score = float(pred[idx])

            top_products.append({
                "product_id": pid,
                "score": score
            })

        print("Top products:", top_products)
        return {"top_products": top_products}

    except Exception as e:
        print("Error in /recommend:", e)
        return {"top_products": []}