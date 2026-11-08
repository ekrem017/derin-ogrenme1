import time
import numpy as np # type: ignore
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, ELU

# Sabitler ve Okul Numarası Tabanlı Random Seed
SEED = 8
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Veri Seti Hazırlığı (-2 <= p <= 2 aralığı)
p = np.linspace(-2, 2, 200)
# Hedef fonksiyonu: y = 1 + sin(pi * p / 4)
y_target = 1 + np.sin(np.pi * p / 4)

# Model Oluşturma Fonksiyonu (Esnek Meta Parametreler)
def create_model(hidden_layers, neurons_per_layer, activation_fn, initializer):
    model = Sequential()
    # İlk gizli katman (giriş boyutu belirtilir)
    if activation_fn == 'leaky_relu':
        model.add(Dense(neurons_per_layer, kernel_initializer=initializer, input_shape=(1,)))
        model.add(LeakyReLU(alpha=0.3))
    elif activation_fn == 'elu':
        model.add(Dense(neurons_per_layer, kernel_initializer=initializer, input_shape=(1,)))
        model.add(ELU(alpha=1.0))
    else:
        model.add(Dense(neurons_per_layer, activation=activation_fn, kernel_initializer=initializer, input_shape=(1,)))
        
    # Ek gizli katmanlar
    for _ in range(hidden_layers - 1):
        if activation_fn == 'leaky_relu':
            model.add(Dense(neurons_per_layer, kernel_initializer=initializer))
            model.add(LeakyReLU(alpha=0.3))
        elif activation_fn == 'elu':
            model.add(Dense(neurons_per_layer, kernel_initializer=initializer))
            model.add(ELU(alpha=1.0))
        else:
            model.add(Dense(neurons_per_layer, activation=activation_fn, kernel_initializer=initializer))
            
    # Çıkış katmanı (Lineer / Aktivasyonsuz)
    model.add(Dense(1, activation='linear', kernel_initializer=initializer))
    return model

# Örnek Çalıştırma (Tablo 1 Parametreleri İçin: 1 Gizli Katman, 2 Nöron, SGD, LR: 0.1, Normal Dağılım)
initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=SEED)
model = create_model(hidden_layers=1, neurons_per_layer=2, activation_fn='relu', initializer=initializer)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

start_time = time.time()
history = model.fit(p, y_target, epochs=300, verbose=0)
elapsed_time = time.time() - start_time

# Performans Metrikleri Hesaplama
y_pred = model.predict(p).flatten()
mse = np.mean((y_target - y_pred) ** 2)
rmse = np.sqrt(mse)
ss_res = np.sum((y_target - y_pred) ** 2)
ss_tot = np.sum((y_target - np.mean(y_target)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print(f"MSE: {mse:.5f}, R2: {r2:.5f}, Süre: {elapsed_time:.2f} sn")