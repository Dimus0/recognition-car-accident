import os
import shutil
import random

def split_dataset(base_path,output_path,test_ration=0.2):
    
    classes = os.listdir(base_path)

    for cls in classes:
        class_path = os.path.join(base_path,cls)

        if not os.path.isdir(class_path):
            continue


        train_dir = os.path.join(output_path, "train",cls)
        test_dir = os.path.join(output_path, "test",cls)

        os.makedirs(train_dir,exist_ok=True)
        os.makedirs(test_dir,exist_ok=True)

        images = os.listdir(class_path)
        images = [img for img in images if img.lower().endswith(('.jpg','.png','.jpeg'))]

        random.shuffle(images)

        split_idx = int(len(images) * (1-test_ration))
        train_imgs = images[:split_idx]
        test_imgs = images[:split_idx]

        for img in train_imgs:

            src = os.path.join(class_path,img)
            dst = os.path.join(train_dir,img)

            shutil.copy(src,dst)

        for img in test_imgs:

            src = os.path.join(class_path,img)
            dst = os.path.join(test_dir,img)

            shutil.copy(src,dst)


        print(f"Class {cls}: {len(train_imgs)} train, {len(test_imgs)} test")

    print("Датасет успішно поділений")


split_dataset(
    base_path="D:\project\Accident Images Analysis Dataset\Accident Images Analysis Dataset\Accident -Detection",
    output_path="dataset",
)

