# Classification-of-the-Simpsons

The project is under development...

gdown --folder https://drive.google.com/drive/folders/10ET4wN898yG2oiRshYxTH95p-Dpoor1S?usp=sharing

tar -xzvf data.tar.gz


========= 1 ========
conv1
Conv2d(3 → 16, kernel=3, stride=1) → BN(16) → ReLU
Размер: 16x254x254

conv2
Conv2d(16 → 32, kernel=3, stride=1) → BN(32) → ReLU → MaxPool(2x2)
Размер: 32x127x127

conv3
Conv2d(32 → 64, kernel=3, stride=1) → BN(64) → ReLU
Размер: 64x125x125

conv4
Conv2d(64 → 64, kernel=3, stride=1) → BN(64) → ReLU → MaxPool(2x2)
Размер: 64x62x62

conv5
Conv2d(64 → 128, kernel=3, stride=1) → BN(128) → ReLU
Размер: 128x60x60

conv6
Conv2d(128 → 128, kernel=3, stride=1) → BN(128) → ReLU → MaxPool(2x2)
Размер: 128x29x29

conv8 (пропущен conv7)
Conv2d(128 → 256, kernel=3, stride=1) → BN(256) → ReLU → MaxPool(3x3)
Размер: 256x9x9

Переход к FC-слоям:

interpolate(4x4) → Размер: 256x4x4

Flatten() → Размер: 4096 (4*4*256)

Полносвязные слои:

fc1
Linear(4096 → 2048) → BN(2048) → ReLU

fc2
Linear(2048 → 42) (n_classes=42)

test: 88