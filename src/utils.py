import numpy as np

def get_top_k_products(autoencoder, user_scaled, product_columns, k=4):
    predicted = autoencoder.predict(user_scaled)[0]
    top_idx = np.argsort(predicted)[::-1][:k]
    return [{"product_id": str(product_columns[i]), "score": float(predicted[i])} for i in top_idx]
