import random

import numpy as np
from torch.utils.data import Dataset, ConcatDataset
import torch
import cv2
import math
import torchvision.transforms as transforms


def get_ratio_kept_resized_image(image: np.ndarray, ref_width: int, ref_height: int) -> np.ndarray:
    org_h, org_w, org_b = 0, 0, 0
    if 2 == image.ndim:
        org_w, org_h = image.shape
        org_b = 1
    if 3 == image.ndim:
        org_w, org_h, _ = image.shape
        org_b = 3
    if org_h == org_w:
        dst_h = ref_height
        dst_w = ref_width
    elif org_h > org_w:
        dst_h = ref_height
        dst_w = ref_height * org_w / org_h
    else:
        dst_h = ref_width * org_h / org_w
        dst_w = ref_width
    dst_h = int(dst_h)
    dst_w = int(dst_w)
    image = cv2.resize(image, (dst_h, dst_w), interpolation=cv2.INTER_NEAREST)
    if org_h == org_w:
        image_dst = image
    elif org_h > org_w:
        image_dst = np.zeros((ref_height, ref_width, org_b), np.uint8)
        len_val = (ref_width - dst_w) // 2
        image_dst[len_val: dst_w + len_val, :] = image
    else:
        image_dst = np.zeros((ref_height, ref_width, org_b), np.uint8)
        len_val = (ref_height - dst_h) // 2
        image_dst[:, len_val: dst_h + len_val] = image
    return image_dst


def augment_image_pair(
        image1: np.ndarray,
        image2: np.ndarray,
        zoom=False,
        zoom_range=(0.8, 1.2),
        flip=True,
        rotate=True,
        angle_range=20
) -> ():
    org_shape = image1.shape
    if zoom:
        zoom_scale = np.random.uniform(zoom_range[0], zoom_range[1])
        dst_shape = (int(org_shape[0] * zoom_scale), int(org_shape[1] * zoom_scale))
        image1 = cv2.resize(image1, dst_shape, interpolation=cv2.INTER_NEAREST)
        image2 = cv2.resize(image2, dst_shape, interpolation=cv2.INTER_NEAREST)
    if flip:
        h_flip_prob = np.random.uniform(0, 1)
        v_flip_prob = np.random.uniform(0, 1)
        if h_flip_prob < 0.5:
            image1 = cv2.flip(image1, 1, dst=None)  # horizontal flip
            image2 = cv2.flip(image2, 1, dst=None)  # horizontal flip
        if v_flip_prob < 0.5:
            image1 = cv2.flip(image1, 0, dst=None)  # vertical flip
            image2 = cv2.flip(image2, 0, dst=None)  # vertical flip
    if rotate:
        (h, w) = image1.shape[:2]
        prob_angle = np.random.uniform(-1, 1)
        prob_angle *= angle_range
        abs_angle = math.fabs(prob_angle)
        abs_angle_arc = abs_angle * math.pi / 180.0
        if prob_angle > 0:
            nh = int(w * math.sin(abs_angle_arc) + h * math.cos(abs_angle_arc))
            nw = int(w * math.cos(abs_angle_arc) + h * math.sin(abs_angle_arc))
        else:
            nw = int(h * math.sin(abs_angle_arc) + w * math.cos(abs_angle_arc))
            nh = int(h * math.cos(abs_angle_arc) + w * math.sin(abs_angle_arc))
        center = (w // 2, h // 2)
        mat = cv2.getRotationMatrix2D(center, prob_angle, 1.0)
        mat[0, 2] = mat[0, 2] + (nw // 2 - w // 2)
        mat[1, 2] = mat[1, 2] + (nh // 2 - h // 2)
        image1 = cv2.warpAffine(image1, mat, (nw, nh))
        image2 = cv2.warpAffine(image2, mat, (nw, nh))
    pro_shape = image1.shape
    if pro_shape[0] < org_shape[0]:
        image1 = get_ratio_kept_resized_image(image1, org_shape[0], org_shape[1])
        image2 = get_ratio_kept_resized_image(image2, org_shape[0], org_shape[1])
    elif pro_shape[0] > org_shape[0]:
        dif_vertical = pro_shape[0] - org_shape[0]
        dif_horizontal = pro_shape[1] - org_shape[1]
        crop_top = int(np.random.uniform(0, 1) * dif_vertical)
        crop_left = int(np.random.uniform(0, 1) * dif_horizontal)
        image1 = image1[crop_top: crop_top + org_shape[0], crop_left: crop_left + org_shape[1]]
        image2 = image2[crop_top: crop_top + org_shape[0], crop_left: crop_left + org_shape[1]]
    return image1, image2


def data_extraction(
        image_path_name: str,
        mask_path_name: str,
        dst_size: () = (96, 96),
        data_aug: bool = False
) -> ():
    image = cv2.imread(image_path_name)
    mask = cv2.imread(mask_path_name)
    image = get_ratio_kept_resized_image(image, dst_size[0], dst_size[1])
    mask = get_ratio_kept_resized_image(mask, dst_size[0], dst_size[1])
    if data_aug:
        image, mask = augment_image_pair(image, mask)
    image = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
    # image = transforms.Normalize(0.5, 0.5)(image)
    mask = torch.FloatTensor(mask).permute(2, 0, 1) / 255.0
    return image, mask[0].unsqueeze(0)


# this dataset class is used for semantic segmentation
class SegmentErrorDataset(Dataset):
    def __init__(
            self,
            sample_list_csv: str,
            segment_image_dir: str,
            segment_mask_dir: str,
            segment_image_name: str = 'segment_imagette_head_pixel_x_%d_y_%d.tif',
            segment_mask_name: str = 'segment_imagette_head_pixel_x_%d_y_%d.tif',
            dst_size: () = (96, 96),
            data_aug: bool = False,
            n_class: int = 3,
    ):
        super().__init__()
        self.segment_image_dir = segment_image_dir
        self.segment_mask_dir = segment_mask_dir
        self.sample_list = []
        self.dst_size = dst_size
        self.data_aug = data_aug
        self.n_class = n_class
        with open(sample_list_csv, 'r') as f:
            for line in f:
                line = line.strip('\n')
                # a line includes 3 elements:
                # 1) sample label (1: segment without error, 2: over-segmented segment, 3: under-segmented segment)
                # 2) x coordinate of the head pixel
                # 3) y coordinate of the head pixel
                elements = line.split(',')
                ele_l, ele_x, ele_y = int(elements[0]), int(elements[1]), int(elements[2])
                segment_imagette_file = self.segment_image_dir + segment_image_name % (ele_x, ele_y)
                segment_mask_file = self.segment_mask_dir + segment_mask_name % (ele_x, ele_y)
                self.sample_list.append((ele_l, segment_imagette_file, segment_mask_file))

    def __getitem__(self, index):
        assert 0 <= index < len(self.sample_list), 'index out of range'
        image_path_name = self.sample_list[index][1]
        mask_path_name = self.sample_list[index][2]
        label = torch.zeros(size=(self.n_class,))
        label[self.sample_list[index][0] - 1] = 1
        return label, data_extraction(image_path_name, mask_path_name, self.dst_size, self.data_aug)

    def __len__(self):
        return len(self.sample_list)


def func(
        param1: int or float,
        param2: bool or int or []
):
    print(type(param1))
    print(type(param2))


if __name__ == '__main__':
    # aaa = [1, 2, 3, 4, 5, 6, 7, 8]
    # aaa = np.array(aaa)
    # aaa = aaa.reshape((2, 4))
    # print(aaa)

    # a = 9
    # print(type(a))
    # if type(a) == int:
    #     print('a')

    # func(12.5, bool(0))
    # func(12, 399)
    # func(33.990, [33, 22, 11])
    # func([], (2,))

    ds1 = SegmentErrorDataset(
        sample_list_csv=r'D:\data\data_deep_seg_error\202108_wuyuan_jilin1\segment_sample_scale_50\sample_records.csv',
        segment_image_dir=
        'D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\segment_sample_scale_50\\segment_imagettes\\',
        segment_mask_dir=
        'D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\segment_sample_scale_50\\segment_masks\\',
        data_aug=True,
    )

    # aa, bb = ds.__getitem__(64)
    #
    # print(aa)
    # print(bb[0].shape)
    # print(bb[1].shape)
    #
    # im_recover = bb[0].permute(1, 2, 0).numpy() * 255.0
    # im_recover = im_recover.astype(np.uint8)
    # cv2.imshow('1', im_recover)
    # # cv2.waitKey(0)
    # mk_recover = bb[1].permute(1, 2, 0).numpy() * 255.0
    # mk_recover = mk_recover.astype(np.uint8)
    # cv2.imshow('2', mk_recover)
    # cv2.waitKey(0)

    ds2 = SegmentErrorDataset(
        sample_list_csv=r'D:\data\data_deep_seg_error\202108_wuyuan_jilin1\segment_sample_scale_50\sample_records.csv',
        segment_image_dir=
        'D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\segment_sample_scale_50\\segment_imagettes\\',
        segment_mask_dir=
        'D:\\data\\data_deep_seg_error\\202108_wuyuan_jilin1\\segment_sample_scale_50\\segment_masks\\',
        data_aug=True,
    )

    ds12 = ConcatDataset((ds1, ds2,))

    print(len(ds12))

    idx = random.randint(0, len(ds12) // 2)

    aa, bb = ds12.__getitem__(idx)
    cc, dd = ds12.__getitem__(idx + len(ds12) // 2 + 1)

    im_recover = bb[0].permute(1, 2, 0).numpy() * 255.0
    im_recover = im_recover.astype(np.uint8)
    cv2.imshow('1', im_recover)
    mk_recover = bb[1].permute(1, 2, 0).numpy() * 255.0
    mk_recover = mk_recover.astype(np.uint8)
    cv2.imshow('2', mk_recover)
    im_recover = dd[0].permute(1, 2, 0).numpy() * 255.0
    im_recover = im_recover.astype(np.uint8)
    cv2.imshow('3', im_recover)
    mk_recover = dd[1].permute(1, 2, 0).numpy() * 255.0
    mk_recover = mk_recover.astype(np.uint8)
    cv2.imshow('4', mk_recover)

    cv2.waitKey(0)
