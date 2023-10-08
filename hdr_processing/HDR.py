import cv2
import os
from PIL import Image, ExifTags
import numpy as np
from utils.utils import compute_size
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class HDR():
    def __init__(self, path_images: str):

        self._path_images = path_images
        file_list = os.listdir(path_images)
        image_extensions = ['.jpg']
        self.exif_list = []
        self.img_list = [cv2.imread(os.path.join(path_images, fn))
                         for fn in file_list if os.path.splitext(fn)[1].lower() in image_extensions]
        self.img_list_pil = [Image.open(os.path.join(
            path_images, fn)) for fn in file_list if os.path.splitext(fn)[1].lower() in image_extensions]
        self.calibrate_debevec_obj = cv2.createCalibrateDebevec()
        self.merge_debevec_obj = cv2.createMergeDebevec()
        self.merge_mertens_obj = cv2.createMergeMertens()
        self.hdr_debevec = None
        self.response_debevec = None
        self.aligned_img_list = self.img_list.copy()
        self.fusion_img = None
        self.tonemap_reinhard = cv2.createTonemapReinhard(2.2)
        self.ldr_reinhard = None
    def extract_exif(self):

        for img in self.img_list_pil:
            exif = {ExifTags.TAGS[k]: v for k,
                    v in img._getexif().items() if k in ExifTags.TAGS}
            self.exif_list.append(exif)

    def reset_exif(self):
        self.exif_list = []

    def get_exif(self):
        return self.exif_list

    def get_images(self):
        return self.img_list

    def get_images_pil(self):
        return self.img_list_pil

    def align_images(self):
        cv2.createAlignMTB().process(self.img_list.copy(), self.aligned_img_list)
        self.aligned_img_list = np.array(self.aligned_img_list)
        print("Images aligned")

    def extract_exposure_times(self):
        if (len(self.exif_list) == 0):
            self.extract_exif()

        self.exposure_times = np.array(
            [exif['ExposureTime'] for exif in self.exif_list], dtype=np.float32)

    def get_exposure_times(self):
        return self.exposure_times

    @compute_size
    def calibrate_debevec(self):

        self.response_debevec = self.calibrate_debevec_obj.process(
            self.img_list.copy(), self.exposure_times.copy())
        print("Calibrated")

    def merge_debevec(self):
        if (len(self.exposure_times) == 0):
            self.extract_exposure_times()
        self.hdr_debevec = self.merge_debevec_obj.process(
            self.img_list, self.exposure_times, self.response_debevec)
        print("Merged")
        return self.hdr_debevec
    
    def process_tone_map_reinhard(self):
        if(self.hdr_debevec is None):
            self.merge_debevec()
        self.ldr_reinhard = self.tonemap_reinhard.process(self.hdr_debevec.copy())
        res_debevec_8bit = np.clip(self.ldr_reinhard*255, 0, 255).astype('uint8')
        return res_debevec_8bit

    
    def process_exposure_fusion(self):

        resFusion = self.merge_mertens_obj.process(self.aligned_img_list)
        self.fusion_img = np.clip(resFusion*255, 0, 255).astype('uint8')

    def get_fusion_img(self):
        return self.fusion_img

    def plot_camera_inv_response(self):
        if (self.response_debevec is None):
            self.calibrate_debevec()
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(1, 2, 1)
        ax.set_xlabel("calibrated Intensity")
        ax.set_ylabel("Measured Intensity")
        ax.set_title("Inverse Response Function")
        for i, color in zip(range(3), ["blue", "green", "red"]):
            new_response = MinMaxScaler().fit_transform(
                self.response_debevec[:, 0, i].copy().reshape(-1, 1))
            ax.plot(new_response,  c=color)

    def plot_camera_response(self):
        if (len(self.response_debevec) == 0):
            self.calibrate_debevec()
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(1, 2, 1)
        ax.set_xlabel("calibrated Intensity")
        ax.set_ylabel("Measured Intensity")
        ax.set_title("Response Function")
        for i, color in zip(range(3), ["blue", "green", "red", ]):
            new_response = MinMaxScaler().fit_transform(
                self.response_debevec[:, 0, i].reshape(-1, 1))
            ax.plot(new_response, np.arange(0, 1, 1/256),  c=color)

    def show_fusion_image(self):

        rgb = cv2.cvtColor(self.fusion_img.copy(), cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb)
        plt.xticks([])
        plt.yticks([])
        plt.show()
