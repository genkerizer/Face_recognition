import os
import cv2
from tqdm import tqdm
import numpy as np
from numpy import dot
from numpy.linalg import norm

from face_reg_torch.predict_model import FaceReg
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Evaluation:

    def __init__(self, database_dir, verification_dir, **kwargs):
        self.reg = FaceReg()
        self.database_dir = database_dir
        self.verification_dir = verification_dir
        self.threshold = 0.4


    def _create_database(self):
        database_list = []
        id_database_list = []

        for root, dir_, filenames in tqdm(os.walk(self.database_dir)):
            if len(dir_) != 0:
                continue

            id_ = float(root.split('/')[-1])
            batch = []
            for img_name in filenames:
                img_path = os.path.join(root, img_name)
                img = cv2.imread(img_path)
                batch.append(img)
            embed_vector = self.reg.predict(batch)
            database_list.append(embed_vector)
            id_database_list.append(id_)
        database = np.array(database_list, dtype=np.float32)
        id_database = np.array(id_database_list, dtype=np.float32)
        print("Vector Database:\t", database.shape)
        print("Id Database:\t", id_database.shape)
        np.save("FINAL_EVAL_DATA/dump/vector_database.npy", database)
        np.save("FINAL_EVAL_DATA/dump/id_database.npy", id_database)


    def cosin_similarity(self, feat_1, feat_2):
        feat_1 = feat_1[None]
        feat_2 = feat_2.transpose()
        cos_sim = dot(feat_1, feat_2)/(norm(feat_1, axis=-1)*norm(feat_2, axis=0))
        return cos_sim
        

    def eval(self):
        total = 0
        count_acc = 0
        id_database = np.load("FINAL_EVAL_DATA/dump/id_database.npy")
        database = np.load("FINAL_EVAL_DATA/dump/vector_database.npy")
        print("Database:\t", database.shape)

        for root, dir_, filenames in tqdm(os.walk(self.verification_dir)):
            if len(dir_) != 0:
                continue
            id_ = float(root.split('/')[-1])

            for img_name in filenames:
                img_path = os.path.join(root, img_name)
                img = cv2.imread(img_path)
                verifi_vector = self.reg.predict([img])
                verifi_vector = np.array(verifi_vector, dtype=np.float32)

                mean_embed = []
                for i in range(4):
                    sub_database = database[:, i, :512]
                    cosin_matrix = self.cosin_similarity(verifi_vector, sub_database)
                    mean_embed.append(cosin_matrix.squeeze()[:, None])
                mean_embed = np.concatenate(mean_embed, axis=-1)
                mean_embed = np.max(mean_embed, axis=-1)

                index = np.argmax(mean_embed)
                gt_id = id_database[index]
                if gt_id == id_:
                    count_acc += 1
                total += 1
                
        print("Accuracy:\t", count_acc / total * 100)


        return None


if __name__ == '__main__':
    database_dir = "FINAL_EVAL_DATA/data_base"
    verification_dir = "FINAL_EVAL_DATA/data_verification"
    class_ = Evaluation(database_dir, verification_dir)
    class_._create_database()
    class_.eval()
   