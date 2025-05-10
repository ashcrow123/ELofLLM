from pycocotools.coco import COCO
import os
import random

class coco_loader:
    def __init__(self,task="caption",dataset="train"):
        self.data_dir = './data/coco'
        if task=="caption":
            if dataset=="train":
                ann_file_name='captions_train2017.json'
            elif dataset=="val":
                ann_file_name='captions_val2017.json'
            else:
                raise ValueError("dataset should be train or val")
        elif task=="instance":
            if dataset=="train":
                ann_file_name='instances_train2017.json'
            elif dataset=="val":
                ann_file_name='instances_val2017.json'
            else:
                raise ValueError("dataset should be train or val")
        elif task=="keypoint":
            if dataset=="train":
                ann_file_name='person_keypoints_train2017.json'
            elif dataset=="val":
                ann_file_name='person_keypoints_val2017.json'
            else:
                raise ValueError("dataset should be train or val")
        else:
            raise ValueError("task should be caption, instance or keypoint")
        self.ann_file = os.path.join(self.data_dir, 'annotations', ann_file_name)
        self.coco = COCO(self.ann_file)
        self.image_ids= None
    def sample(self,percentage=1):
        self.image_ids = random.sample(self.coco.getImgIds(),round(percentage*len(self.coco.getImgIds())))
        random.shuffle(self.image_ids)
        return
    def get_image_ids(self):
        if self.image_ids is None:
            raise ValueError("Please sample the image ids first")
        return self.image_ids
    def get_captions(self,image_id):
        if self.image_ids is None:
            raise ValueError("Please sample the image ids first")
        if image_id not in self.image_ids:
            raise ValueError("image id is not in the sampled image ids")
        caption_ids = self.coco.getAnnIds(imgIds=image_id)
        captions = self.coco.loadAnns(caption_ids)
        return captions
    def get_image_info(self,image_id):
        if self.image_ids is None:
            raise ValueError("Please sample the image ids first")
        if image_id not in self.image_ids:
            raise ValueError("image id is not in the sampled image ids")
        image_info = self.coco.loadImgs(image_id)
        return image_info
    def get_random_caption(self,count=3):
        ids_list=random.choices(self.image_ids,k=count)
        captions_list=[random.choice(self.get_captions(id))["caption"] for id in ids_list]
        return captions_list
        
    
    
if __name__=="__main__":
    data_loader=coco_loader(task="caption",dataset="train")
    data_loader.sample()
    image_ids=data_loader.get_image_ids()
    captions=data_loader.get_captions(image_ids[0])
    print(captions)
    print(type(captions))
    
    