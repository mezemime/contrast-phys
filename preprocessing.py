import os
import cv2
import numpy as np
import h5py
import pandas as pd

def openface_h5(video_path, landmark_path, h5_path, store_size=128):
    landmark = pd.read_csv(landmark_path)
    with h5py.File(h5_path, 'w') as f:
        total_num_frame = len(landmark)
        cap = cv2.VideoCapture(video_path)

        for frame_num in range(total_num_frame):
            if landmark[' success'][frame_num]:
                lm_x = [landmark[f' x_{i}'][frame_num] for i in range(68)]
                lm_y = [landmark[f' y_{i}'][frame_num] for i in range(68)]
                lm_x = np.array(lm_x)
                lm_y = np.array(lm_y)

                minx = np.min(lm_x)
                maxx = np.max(lm_x)
                miny = np.min(lm_y)
                maxy = np.max(lm_y)

                y_range_ext = (maxy - miny) * 0.2
                miny -= y_range_ext

                cnt_x = int(np.round((minx + maxx) / 2))
                cnt_y = int(np.round((maxy + miny) / 2))
                break

        bbox_size = int(np.round(1.5 * (maxy - miny)))
        if store_size is None:
            store_size = bbox_size

        imgs = f.create_dataset(
            'imgs',
            shape=(total_num_frame, store_size, store_size, 3),
            dtype='uint8',
            chunks=(1, store_size, store_size, 3),
            compression="gzip",
            compression_opts=4
        )

        for frame_num in range(total_num_frame):
            if landmark[' success'][frame_num]:
                lm_x_ = [landmark[f' x_{i}'][frame_num] for i in range(68)]
                lm_y_ = [landmark[f' y_{i}'][frame_num] for i in range(68)]
                lm_x_ = np.array(lm_x_)
                lm_y_ = np.array(lm_y_)

                lm_x = 0.9 * lm_x + 0.1 * lm_x_
                lm_y = 0.9 * lm_y + 0.1 * lm_y_

                minx = np.min(lm_x)
                maxx = np.max(lm_x)
                miny = np.min(lm_y)
                maxy = np.max(lm_y)

                y_range_ext = (maxy - miny) * 0.2
                miny -= y_range_ext

                cnt_x = int(np.round((minx + maxx) / 2))
                cnt_y = int(np.round((maxy + miny) / 2))

            ret, frame = cap.read()
            if not ret:
                print("⚠️ Fin du flux ou erreur de lecture.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bbox_half_size = bbox_size // 2

            face = np.take(frame, range(cnt_y - bbox_half_size, cnt_y - bbox_half_size + bbox_size), axis=0, mode='clip')
            face = np.take(face, range(cnt_x - bbox_half_size, cnt_x - bbox_half_size + bbox_size), axis=1, mode='clip')

            if store_size == bbox_size:
                imgs[frame_num] = face
            else:
                imgs[frame_num] = cv2.resize(face, (store_size, store_size))

        cap.release()

# === Chemins ===
video_path = r"C:\Users\Dell\Desktop\SD\stage\videos\seance11\vid76.mp4"
landmark_path = r"C:\Users\Dell\Desktop\SD\stage\landmarks\vid76.csv"
h5_output_path = r"C:\Users\Dell\Desktop\SD\stage\data\vid76.h5"

# === Supprimer le .h5 s'il existe pour éviter PermissionError ===
if os.path.exists(h5_output_path):
    try:
        os.remove(h5_output_path)
        print(f"🗑️ Ancien fichier supprimé : {h5_output_path}")
    except Exception as e:
        print(f"❌ Impossible de supprimer {h5_output_path} : {e}")

# === Traitement ===
print(f"🔄 Traitement de : {video_path}")
try:
    openface_h5(video_path, landmark_path, h5_output_path)
    print(f"✅ Fichier .h5 créé : {h5_output_path}")
except Exception as e:
    print(f"❌ Erreur pendant le traitement : {e}")
