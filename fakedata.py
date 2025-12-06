import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras import layers, models, regularizers
import random
from sklearn.preprocessing import MinMaxScaler

#--------------------- FAKE DATA KHÁCH

file_path = r'E:\odoo\customer.xlsx'
customer_data = pd.read_excel(file_path)

# Lấy 1000 khách hàng đầu tiên
fake_customers = customer_data.head(1000).copy()

# Tạo cột ID từ 1 đến 1000
fake_customers['ID'] = range(1, 1001)

# Hiển thị dữ liệu giả
print(fake_customers[['ID', 'Name*', 'Email', 'Phone']])

#---------------- LẤY SẢN PHẨM TRONG DB 
client = MongoClient("mongodb+srv://NgocHan:ngochan6263@sweetdb.coon2qf.mongodb.net/SweetDB?retryWrites=true&w=majority&appName=SweetDB")
db = client["SweetDB"]
products_collection = db["SANPHAM"]

products = list(products_collection.find())
drink_products = [p for p in products if p['category'] == "Đồ uống"]

#Kiểm tra sản phẩm 
for product in products:
    print(product)

product_ids = [(product['_id'], product['name'], product['category']) for product in products]
print(product_ids)

#------------------------TẠO ĐƠN GIẢ

orders = []

for index, customer in fake_customers.iterrows():
    num_orders = random.randint(1, 27)  # Mỗi khách hàng sẽ có từ 1 đến 5 đơn hàng ngẫu nhiên
    for _ in range(num_orders):
        # Chọn ngẫu nhiên sản phẩm từ danh sách sản phẩm
        product = random.choice(products)
        orders.append({
            'customer_id': customer['ID'],
            'product_id': product['_id'],
            'product_name': product['name'],
            'order_date': pd.to_datetime('today').strftime('%Y-%m-%d'),
            'quantity': random.randint(1, 3),  # Mỗi đơn hàng có từ 1 đến 3 sản phẩm
            'order_id': random.randint(100000, 999999)  # ID đơn hàng ngẫu nhiên
        })

        if product['category'] != "Đồ uống":
            if random.random() < 0.5:  # 50% khách mua đồ ăn sẽ mua thêm đồ uống
                drink = random.choice(drink_products)
                orders.append({
                    'customer_id': customer['ID'],
                    'product_id': drink['_id'],
                    'product_name': drink['name'],
                    'order_date': pd.to_datetime('today').strftime('%Y-%m-%d'),
                    'quantity': 1,
                    'order_id': random.randint(100000, 999999)
                })

orders_df = pd.DataFrame(orders)

orders_df.to_excel('orders_27500.xlsx', index=False)



# --------------------------------- MA TRẬN THƯA THỚT 

#đọc file csv 
file_path_orders = 'D:\MOBILE\AI_MOBILE\orders_27500.xlsx'
orders_df = pd.read_excel(file_path_orders)
# Chuyển đổi ma trận user-item
user_item_matrix = orders_df.pivot_table(index='customer_id', columns='product_id', values='quantity', aggfunc='sum', fill_value=0)

# Hiển thị ma trận user-item (có thể rất lớn)
# print(user_item_matrix)

# Lưu ma trận user-item vào file Excel
# user_item_matrix.to_excel('user_item_matrix.xlsx')

#--------------------------------- CHUẨN BỊ DỮ LIỆU 
# Chuyển đổi ma trận thành mảng numpy
X = user_item_matrix.values

# Chuẩn hóa dữ liệu về phạm vi [0, 1] để mô hình học tốt hơn
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Chia dữ liệu thành tập huấn luyện và kiểm tra (80% train, 20% test)
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# Kiểm tra shape của dữ liệu
print(X_train.shape, X_test.shape)

#--------------------------------TÍNH PRECISION K
import numpy as np

def precision_at_k(y_true, y_pred, k=4):
    """
    y_true: vector ground truth (0/1 hoặc số lượng)
    y_pred: vector dự đoán liên tục từ autoencoder
    """
    # Lấy top k sản phẩm dự đoán
    top_k_items = np.argsort(y_pred)[::-1][:k]

    # Những item mà user thực sự có mua
    true_items = np.where(y_true > 0)[0]

    if len(true_items) == 0:
        return None  # user không mua gì → bỏ qua để không ảnh hưởng kết quả

    hits = len(set(top_k_items).intersection(set(true_items)))

    return hits / k



#--------------------------------XÂY DỰNG MÔ HÌNH AUTOENCODER
# Xây dựng mô hình Autoencoder
input_dim = X_train.shape[1]  # Số lượng sản phẩm
encoding_dim = 50  # Số lượng chiều của đặc trưng ẩn

# Encoder
input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)

# Decoder
decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

# Mô hình Autoencoder
autoencoder = models.Model(input_layer, decoded)

# Mô hình Encoder (để sử dụng trong việc tái tạo các đặc trưng ẩn)
encoder = models.Model(input_layer, encoded)

# Biên dịch mô hình
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Huấn luyện mô hình
history = autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, validation_data=(X_test, X_test))

# Đánh giá mô hình
loss = autoencoder.evaluate(X_test, X_test)
print(f"Test loss: {loss}")

#precision k
predicted = autoencoder.predict(X_test)

k = 4
precisions = []

for i in range(len(X_test)):
    score = precision_at_k(X_test[i], predicted[i], k=k)
    if score is not None:
        precisions.append(score)

print(f"Precision@{k}: {np.mean(precisions):.4f}")


# Sử dụng encoder để lấy các đặc trưng ẩn
encoded_data = encoder.predict(X_test)

#------------------------------------- TEST 
user_index = 10  # đổi user muốn xem

user_pred = predicted[user_index]

# Danh sách product_id theo đúng thứ tự cột của pivot
product_columns = user_item_matrix.columns.tolist()

# Map product_id -> tên + category từ MongoDB
product_map = {
    str(p['_id']): (p['name'], p['category']) 
    for p in products
}

# Lấy top 4 sản phẩm được dự đoán cao nhất
top4_idx = np.argsort(user_pred)[::-1][:4]

print("\n===== GỢI Ý TOP 4 SẢN PHẨM =====")
for idx in top4_idx:
    product_id = product_columns[idx]      # product_id gốc
    product_id_str = str(product_id)

    if product_id_str in product_map:
        p_name, p_cat = product_map[product_id_str]
    else:
        p_name, p_cat = ("Không tìm thấy", "Unknown")

    print(f"- {p_name} ({p_cat}) — ID: {product_id_str} — score={user_pred[idx]:.4f}")

# Lưu model
autoencoder.save("autoencoder_model.h5")
encoder.save("encoder_model.h5")

print("Saved model thành công!")

import joblib
joblib.dump(scaler, "scaler.pkl")


product_columns = list(user_item_matrix.columns)
joblib.dump(product_columns, "product_columns.pkl")



