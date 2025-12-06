import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models, regularizers, callbacks
import joblib

#---------------- LẤY DỮ LIỆU TỪ EXCEL
file_path_orders = r''
orders_df = pd.read_excel(file_path_orders)

# Pivot table user-item matrix
user_item_matrix = orders_df.pivot_table(
    index='customer_id',
    columns='product_id',
    values='quantity',
    aggfunc='sum',
    fill_value=0
)

# Chuyển quantity -> log(1+quantity) để giảm ảnh hưởng outlier
X = np.log1p(user_item_matrix.values)

# Scale dữ liệu
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

#---------------- XÂY DỰNG DEEP AUTOENCODER
input_dim = X_train.shape[1]
encoding_dim = 50

input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(128, activation='relu')(input_layer)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(encoding_dim, activation='relu',
                       activity_regularizer=regularizers.l2(1e-5))(encoded)

decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = models.Model(input_layer, decoded)
encoder = models.Model(input_layer, encoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Early stopping
es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=256,
    validation_data=(X_test, X_test),
    callbacks=[es]
)

# Save model
autoencoder.save("autoencoder_model.h5")
encoder.save("encoder_model.h5")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(user_item_matrix.columns), "product_columns.pkl")

print("Saved model thành công!")

#---------------- TEST top 4 recommendation
predicted = autoencoder.predict(X_test)

def get_top_k_products(user_vector, predicted_vector, product_columns, k=4):
    top_idx = np.argsort(predicted_vector)[::-1][:k]
    top_products = [{"product_id": str(product_columns[i]), "score": float(predicted_vector[i])} 
                    for i in top_idx]
    return top_products

user_index = 10
top4 = get_top_k_products(X_test[user_index], predicted[user_index], list(user_item_matrix.columns), k=4)
print("Top 4 products:", top4)
