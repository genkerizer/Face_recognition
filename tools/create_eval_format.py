import os
import shutil

if __name__ == '__main__':
    database_dir = "FINAL_EVAL_DATA/data_base"
    for root, dirs, filename in os.walk("DATATEST/cfp-dataset_convert/Data/Images"):
        if len(dirs) != 0:
            continue
        idx_dir = root
        id_ = idx_dir.split('/')[-2]
        tyle_img =  idx_dir.split('/')[-1]
        if 'profile' in tyle_img:
            continue
        for img_name in filename:
            if img_name in ['02.jpg', '03.jpg', '04.jpg', '05.jpg']:
                save_id_dir = os.path.join(database_dir, id_)
                if not os.path.exists(save_id_dir):
                    os.makedirs(save_id_dir, exist_ok=True)
                img_path  = os.path.join(idx_dir, img_name)
                dst_path = os.path.join(save_id_dir, img_name)
                shutil.move(img_path, dst_path)

                