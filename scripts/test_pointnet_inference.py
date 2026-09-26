import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('models/onnx/pointnet2_semseg.onnx')
pts = np.zeros((1, 2048, 3), dtype=np.float32)
# road
pts[0, :1000, 0] = np.linspace(5, 50, 1000)
pts[0, :1000, 1] = np.random.uniform(-4, 4, 1000)
pts[0, :1000, 2] = -1.6
# vehicle
pts[0, 1000:1500, 0] = np.random.uniform(14, 18, 500)
pts[0, 1000:1500, 1] = np.random.uniform(-1, 1, 500)
pts[0, 1000:1500, 2] = np.random.uniform(-1.2, 0.2, 500)
# static
pts[0, 1500:, 0] = np.random.uniform(5, 50, 548)
pts[0, 1500:, 1] = np.random.uniform(5, 10, 548)
pts[0, 1500:, 2] = np.random.uniform(-1.0, 3.0, 548)

logits = session.run(['logits'], {'points': pts})[0]
preds = np.argmax(logits, axis=-1)[0]
print('Class 0 (terrain):', int(np.sum(preds == 0)))
print('Class 1 (static):', int(np.sum(preds == 1)))
print('Class 2 (dynamic/vehicle):', int(np.sum(preds == 2)))
