import sys
import cv2
import os
from PIL import Image, ExifTags
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from utils.utils import compute_time_execution

class HDR():
    def __init__(self, path_images: str):

        self._path_images = path_images
        file_list = os.listdir(path_images)
        image_extensions = ['.jpg', '.jpeg', '.png']
        self.exif_list = []
        self.img_list = [cv2.imread(os.path.join(path_images, fn))
                         for fn in file_list if os.path.splitext(fn)[1].lower() in image_extensions]
        size_sum = sum(sys.getsizeof(i) for i in self.img_list)
        print("size img_list: ", size_sum)
        print(f"max of images: {max(self.img_list[0].flatten())}")
        self.img_list_pil = [Image.open(os.path.join(
            path_images, fn)) for fn in file_list if os.path.splitext(fn)[1].lower() in image_extensions]
        size_sum = sum(sys.getsizeof(i) for i in self.img_list_pil)
        print("size img_list_pil: ", size_sum)

        self.calibrate_debevec_obj = cv2.createCalibrateDebevec()
        print("size calibrate_debevec_obj: ", sys.getsizeof(self.calibrate_debevec_obj))
        self.calibrate_robertson_obj = cv2.createCalibrateRobertson()
        print("size calibrate_robertson_obj: ", sys.getsizeof(self.calibrate_robertson_obj))
        self.merge_debevec_obj = cv2.createMergeDebevec()
        print("size merge_debevec_obj: ", sys.getsizeof(self.merge_debevec_obj))
        self.merge_robertson_obj = cv2.createMergeRobertson()
        print("size merge_robertson_obj: ", sys.getsizeof(self.merge_robertson_obj))
        self.merge_mertens_obj = cv2.createMergeMertens()
        print("size merge_mertens_obj: ", sys.getsizeof(self.merge_mertens_obj))
        self.hdr_debevec = None
        self.response_debevec = None
        self.aligned_img_list = self.img_list.copy()
        self.fusion_img = None
        self.tonemap_reinhard = cv2.createTonemap(2.2)
        print("size tonemap_reinhard: ", sys.getsizeof(self.tonemap_reinhard))
        self.ldr_reinhard = None
        self.hdr_robertson = None
    def extract_exif(self):

        for img in self.img_list_pil:
            exif = {ExifTags.TAGS[k]: v for k,
                    v in img._getexif().items() if k in ExifTags.TAGS}
            self.exif_list.append(exif)
        delattr(self, 'img_list_pil') 

    def reset_exif(self):
        self.exif_list = []

    def get_exif(self):
        return self.exif_list

    def get_images(self):
        return self.img_list

    def get_images_pil(self):
        return self.img_list_pil

    @compute_time_execution
    def align_images(self):
        cv2.createAlignMTB().process(self.img_list.copy(), self.aligned_img_list)
        self.aligned_img_list = np.array(self.aligned_img_list)
        print("Images aligned")
        
    @compute_time_execution
    def extract_exposure_times(self):
        if (len(self.exif_list) == 0):
            self.extract_exif()

        self.exposure_times = np.array(
            [exif['ExposureTime'] for exif in self.exif_list], dtype=np.float32)

    def get_exposure_times(self):
        return self.exposure_times

    @compute_time_execution
    def calibrate(self):
        self.response_robertson = self.calibrate_robertson_obj.process(
            self.img_list.copy(), self.exposure_times.copy())
        self.response_debevec = self.calibrate_debevec_obj.process(
            self.img_list.copy(), self.exposure_times.copy())
        print("Calibrated")
    @compute_time_execution
    def merge(self, method: str):
        if (len(self.exposure_times) == 0):
            self.extract_exposure_times()
     
        
            
        self.hdr_debevec = self.merge_debevec_obj.process(
            self.img_list, self.exposure_times)
        

        self.hdr_robertson = self.merge_robertson_obj.process(
            self.img_list, self.exposure_times)
        print("Merged")
        if(method == "debevec"):
            return self.hdr_debevec
        elif(method == "robertson"):
            return self.hdr_robertson
        else:
            raise Exception("Invalid method")

        
        
    @compute_time_execution
    def process_tone_map_reinhard(self, method : str):
        if(method == "debevec"):
            
            self.ldr_reinhard = self.tonemap_reinhard.process(self.hdr_debevec.copy())
            res_8bit = np.clip(self.ldr_reinhard*255, 0, 255).astype('uint8')
        elif(method == "robertson"):
            self.ldr_reinhard = self.tonemap_reinhard.process(self.hdr_robertson.copy())
            res_8bit = np.clip(self.ldr_reinhard*255, 0, 255).astype('uint8')
        else:
            raise Exception("Invalid method")
        return res_8bit

    @compute_time_execution
    def process_exposure_fusion(self):

        resFusion = self.merge_mertens_obj.process(self.aligned_img_list)
        self.fusion_img = np.clip(resFusion*255, 0, 255).astype('uint8')

    def get_fusion_img(self):
        return self.fusion_img

    def plot_camera_inv_response(self, method: str):
        if (method == "debevec"):
            response = self.response_debevec
        elif (method == "robertson"):
            response = self.response_robertson
        else:
            raise Exception("Invalid method")
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.set_xlabel("calibrated Intensity")
        ax.set_ylabel("Measured Intensity")
        ax.set_title(f"Inverse Response Function, method: {method}")
        for i, color in zip(range(3), ["blue", "green", "red"]):
            new_response = MinMaxScaler().fit_transform(
                response[:, 0, i].copy().reshape(-1, 1))
            ax.plot(new_response,  c=color)

    def plot_camera_response(self, method : str):
        if (method == "debevec"):
            response = self.response_debevec
        elif (method == "robertson"):
            response = self.response_robertson
        else:
            raise Exception("Invalid method")
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.set_xlabel("calibrated Intensity")
        ax.set_ylabel("Measured Intensity")
        ax.set_title("Response Function")
        for i, color in zip(range(3), ["blue", "green", "red"]):
            new_response = MinMaxScaler().fit_transform(
                response[:, 0, i].copy().reshape(-1, 1))
            ax.plot(np.arange(0, 1, 1/256), new_response,  c=color)

    def show_fusion_image(self):

        rgb = cv2.cvtColor(self.fusion_img.copy(), cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb)
        plt.xticks([])
        plt.yticks([])
        plt.show()
