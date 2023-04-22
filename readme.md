# Face Recognition


## Các nội dung chính

- [Chuẩn bị dữ liệu huấn luyện](#Prepare-Data-Training)
- [Hướng dẫn training](#How-to-training-model)
- [Hướng dẫn đánh giá kết quả](#How-to-evaluate-model)
- [Độ chính xác](#Accuracy)
- [Hướng dẫn sử dụng inference module](#Tutorial-to-use-inference-module)
- [Mã nguồn tham khảo](#reference)
- [Thành viên thực hiện](#Contributor)

## Chuẩn bị dữ liệu huấn luyện


Tải về dữ liệu huấn luyện tại đây [Face recognition dataset](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-arcface-85k-ids58m-images-57) 
<hr/>

Cụ thể nhóm dùng data [ArcFace Dataset](https://drive.google.com/u/0/uc?id=1SXS4-Am3bsKSK615qbYdbA_FMVh3sAvR&export=download)

Sau khi giải nén ta được folder chứa data có cấu trúc như sau:

`Data train được copy vào folder DATASET`

### **`Folder structure`**

- agedb_30.bin
- calfw.bin
- cfp_ff.bin
- cfp_fo.bin
- cplfw.bin
- lfw.bin
- property
- train.idx
- train.rec
- vgg2_fp.bin

<hr/>



## Hướng dẫn training
- Cài đặt các thư viên cần thiết

```
    pip install -r requirements.txt
```

- Điều chỉnh các tham số cần thiết trong `config/`
```
Global:
  use_pretrain: False   # Có sử dụng pretrain hay không
  num_epoch: 10     # Số lượng epoch cần training
  warmup_epoch: 0   # Bắt đầu warmup learning tại epoch thứ bao nhiêu
  checkpoints: checkpoints/20230320 # Checkpoint dùng để lưu weight


Architecture: 
  Backbone:
    name: IResnet
    layers: [3, 4, 14, 3]
  Head:
    name: PartialFC_V2
    embedding_size: 512
    num_classes: 93431  # Số lượng id, tùy vào bộ data mà điều chỉnh số classes cho phù hợp
    sample_rate: 1.0    # Tỉ lệ chia cắt data để làm tập training và đánh giá, ở đây nhóm dùng 100% training


Loss:
  margin_list: [1.0, 0.5, 0.0] # Số số của losses
  interclass_filtering_threshold: 0


Save:
  save_iter: 3000   # Số lượng iteration để lưu weights


Optimizer:  # Các tham số optimizer 
  gradient_acc: 1.0
  lr: 0.1
  weight_decay: 0.0005
  last_epoch: -1

Dataloader:
  name: loader
  num_image: 5800000    # Số lượng ảnh có trong datatrain
  batch_size: 64    #
  root_dir: DATASET/faces_emore/ # Đường dẫn đến folder chứa data training



```

### run training file
```python3 -m trainer.train_face_reg_iresnet50```

- Ngoài ra, mã nguồn còn hỗ trợ training với backbone mfacenet và resnet18, tham khảo tại `trainer/`



## Hướng dẫn đánh giá kết quả

- Tải về data đánh giá tại [Face recognition datatest](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-arcface-85k-ids58m-images-57)

- Cấu trục thư mục data đánh giá như sau:

```
  eval_data
    data_base:
      id_01
        img1.jpg
        ...
        img5.jpg
      
      id_02
        img1.jpg
        ...
        img5.jpg
      ...
    
    data_verification
      id_01
        img.jpg

      id_02
        img.jpg
      ...

  
```

- Thay đổi đường dẫn trong file `evaluation/evaluation.py`

```
    database_dir = "eval_data/data_base"
    verification_dir = "eval_data/data_verification"
```



- Chạy lệnh: ```python3 -m evaluation.evaluation```


## Độ chính xác:

| Datasets         | Backbone | Accuracy(%) |                                                                                                                             |
|:-----------------|:------------------|:------------|:-------------------------------------------------------------------------------------------------------------------------------------------|
| CFP-500ID    | MobileFaceNet        | 54.2       | 
| CFP-500ID    | IResnet50        | 54.2       | 


## Hướng dẫn sử dụng inference moudle

+ Bước 1: Convert lấy weight của backbone từ checkpoints. Điều chỉnh đường dẫn checkpoint trong file `tools/get_backbone.py`

+ Bước 2: Chạy lên `python3 -m tools.get_backbone`. Ta sẽ nhận được weight của backbone

+ Bước 3: Copy weight của backbone vào folder `face_reg_torch/models`

+ Bước 4: Cập nhật config tại `face_reg_torch/configs/face_config.json`
    
### **`Input`**

<p align="left">
<img src='assets/01.jpg' width="128" height="128">
<img src='assets/02.jpg' width="128" height="128">
</p>

### **`Output`**
```
[512-dim-vector1, 512-dim-vector2]
```

### Import và sử dụng module

```
    from face_reg_torch.predict_model FaceReg

    face_reg = FaceReg()

    vector = face_reg.predict(img_list)
```

## Mã nguồn tham khảo

+ [InsightFace](https://github.com/deepinsight/insightface)

## Thành viên thực hiện

+ [Nguyễn Y Hợp](22C15006@student.hcmus.edu.vn)

+ [Nguyễn Đăng Khoa](22C15010@student.hcmus.edu.vn)

+ [Phạm Minh Thạch](22C15018@student.hcmus.edu.vn)