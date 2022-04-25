import copy 
import torch
from torch.utils.data import Dataset
from natsort import natsorted
import cv2, glob, os, skimage 
import skimage.transform
import numpy as np 

class NuscenesDataset(Dataset):
    def __init__(self,img_folder, args):
        self.img_channels = args.img_channels
        self.resize_img = args.resize_img
        self.resize_img_ht = args.resize_img_ht
        self.resize_img_wd = args.resize_img_wd
        self.img_folder = img_folder
        file_list = glob.glob(self.img_folder)
        self.data = []
        self.img_seq = ["None"]*args.seq_len

        for i in natsorted(os.listdir(file_list[0])):
            if 'nuscenes' in i :
                file_num = os.path.join(file_list[0],i )
            else:   continue
        
            for foldernum in natsorted(os.listdir(file_num)):
                currentfolder = os.path.join(file_num, foldernum)

                for idximages,image in enumerate(natsorted(os.listdir(currentfolder))):

                    current_image = os.path.join(currentfolder, image)
                    self.img_seq[-1] = current_image

                    if '/1.png' in self.img_seq[0]:  self.data.append(self.img_seq)

                    if (((idximages+1)-args.seq_len) % args.seqimg_gap == 0) :    
                        if ((idximages+1)-args.seq_len > 0 ):   self.data.append(self.img_seq)

                    self.img_seq = self.img_seq[1:] + self.img_seq[:1]

    def __len__(self):
        return len(self.data)

    def resize(self, image, min_side=256, max_side=256):
        rows, cols, _ = image.shape
        smallest_side = min(rows, cols)
        scale = min_side / smallest_side
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows * scale)), int(round((cols * scale)))))
        rows, cols,_ = image.shape

        return image

    def __getitem__(self, idx):
        image_path = self.data[idx]
        #print("\n",image_path)
        for i, img in enumerate(image_path):
            image = cv2.imread(img)

            # Considering static and Semantic objects from the grid
            staticimg = copy.deepcopy(image)
            Semanticimg = copy.deepcopy(image)

            index = np.argmax(staticimg, axis =2)
            # separate the static and semantic information 
            x1,y1 = np.where(  (index!= 0) )
            staticimg[x1,y1,:] = 255
            x1,y1 = np.where(  (index!= 1))
            Semanticimg[x1,y1,:] = 255

            # Converting all images to black and white
            gray_img =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            (_, bwimg) = cv2.threshold(gray_img, 220, 255, cv2.THRESH_BINARY)
            static_gray = cv2.cvtColor(staticimg, cv2.COLOR_BGR2GRAY)
            (_, static_bwimg) = cv2.threshold(static_gray, 220, 255, cv2.THRESH_BINARY)
            Semantic_gray = cv2.cvtColor(Semanticimg, cv2.COLOR_BGR2GRAY)
            (_, Semantic_bwimg) = cv2.threshold(Semantic_gray, 220, 255, cv2.THRESH_BINARY)

            if self.resize_img: 
                image = self.resize(image,self.resize_img_ht, self.resize_img_wd)
                bwimg = np.expand_dims(bwimg, axis=2)
                bwimg = self.resize(bwimg,self.resize_img_ht, self.resize_img_wd)

                staticimg = self.resize(staticimg,self.resize_img_ht, self.resize_img_wd)
                static_bwimg = np.expand_dims(static_bwimg, axis=2)
                static_bwimg = self.resize(static_bwimg,self.resize_img_ht, self.resize_img_wd)

                Semanticimg = self.resize(Semanticimg,self.resize_img_ht, self.resize_img_wd)
                Semantic_bwimg = np.expand_dims(Semantic_bwimg, axis=2)
                Semantic_bwimg = self.resize(Semantic_bwimg,self.resize_img_ht, self.resize_img_wd)

            if i ==0:
                seq_tensor = torch.unsqueeze(torch.from_numpy(image), 0)
                bw_tensor = torch.unsqueeze(torch.from_numpy(bwimg), 0)
                staticgrid_tensor = torch.unsqueeze(torch.from_numpy(static_bwimg), 0)
                Semanticgrid_tensor = torch.unsqueeze(torch.from_numpy(Semantic_bwimg), 0)

            else:   
                seq_tensor = torch.cat((seq_tensor, torch.unsqueeze(torch.from_numpy(image), 0)),0)
                bw_tensor = torch.cat((bw_tensor, torch.unsqueeze(torch.from_numpy(bwimg), 0)),0)
                staticgrid_tensor = torch.cat((staticgrid_tensor, torch.unsqueeze(torch.from_numpy(static_bwimg), 0)),0)
                Semanticgrid_tensor = torch.cat((Semanticgrid_tensor, torch.unsqueeze(torch.from_numpy(Semantic_bwimg), 0)),0)

        seqDict = { "seqTensor": seq_tensor.float(), "bwTensor": bw_tensor.float(), "staticgridTensor": staticgrid_tensor.float(), "SemanticgridTensor": Semanticgrid_tensor.float()}
        return seqDict