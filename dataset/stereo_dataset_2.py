import os
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image


class StereoDataset(Dataset):
    """
    Stereo dataset from https://www.dropbox.com/sh/ve1yoar9fwrusz0/AAAfu7Fo4NqUB7Dn9AiN8pCca?dl=0
    """

    def __init__(self, config):   ## add split for training/validation??
        path = config['path']

        # rotation/translation vectors should be with negative sign 
        # due to backward transformation from depth2color camera space
        r_vec = -np.array([[0.00531, -0.01196, 0.00301]])
        t_vec = -np.array([-24.0381, -0.4563, -1.2326])
        r_mat, _ = cv2.Rodrigues(r_vec)
        transform_matrix = np.hstack([r_mat, t_vec.reshape((3, 1))])

        images = []
        labels = []
        for scope in ['Random', 'Counting']:
            for chunk in range(1, 7):
                folder_name = f'images/B{chunk}{scope}'
                mat = loadmat(os.path.join(path, 'labels', f'B{chunk}{scope}_SK.mat'))
                kpoints3d = mat['handPara']

                for i in range(1500):
                    img = os.path.join(path, folder_name, f'SK_color_{i}.png')
                    images.append(img)
                    kps3d = kpoints3d[..., i].T
                    kps3d_h = np.hstack([kps3d, np.ones((21, 1))])
                    kps3d = (transform_matrix @ kps3d_h.T).T
                    labels.append(kps3d)

        self.images = images
        self.labels = np.array(labels)
        self.intrinsic = np.array([
            [607.92271, 0, 314.78337],
            [0, 607.88192, 236.42484],
            [0, 0, 1]
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = np.float32(Image.open(img)) / 255  #changed from BGR to RGB using PIL.Image loader
        kpoints3d = self.labels[idx]

        kpoints3d_normed = kpoints3d.copy()

        kpoints2d = (self.intrinsic @ kpoints3d_normed.T).T
        kpoints2d = kpoints2d[:, :2] / kpoints2d[:, 2:]

        kpoints3d_normed[:, :2] = kpoints2d
        kpoints3d_normed[:, 2] = kpoints3d_normed[:, 2] - kpoints3d_normed[0, 2]
        kpoints3d_normed[:, 2] = kpoints3d_normed[:, 2] / np.abs(kpoints3d_normed[0, 2] - kpoints3d_normed[1, 2])

        img, kpoints3d_normed = self._crop_hand(img, kpoints3d_normed)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        kpoints3d_normed = torch.from_numpy(kpoints3d_normed).float()

        return {
            'image': img,
            'target': kpoints3d_normed
        }

    def _crop_hand(self, img, kpoints3d, target_size=128, padding=0.2):
        x_min, y_min = np.min(kpoints3d[:, 0]).astype('int16'), np.min(kpoints3d[:, 1]).astype('int16')
        x_max, y_max = np.max(kpoints3d[:, 0]).astype('int16'), np.max(kpoints3d[:, 1]).astype('int16')

        width = x_max - x_min
        height = y_max - y_min

        if max(width, height) * (1 + padding) <= target_size:
            x_padding = target_size - width
            y_padding = target_size - height
        else:
            pretarget_size = int(max(width, height) * (1 + padding))
            x_padding = pretarget_size - width
            y_padding = pretarget_size - height

        # randomize left-right padding
        left_padding = int(x_padding * np.random.random())
        right_padding = x_padding - left_padding

        up_padding = int(y_padding * np.random.random())
        bottom_padding = y_padding - up_padding

        x0 = x_min - left_padding
        x1 = x_max + right_padding
        y0 = y_min - up_padding
        y1 = y_max + bottom_padding

        if x0 < 0:
            x1 -= x0
            x0 = 0
        if y0 < 0:
            y1 -= y0
            y0 = 0
        if x1 >= img.shape[1]:
            x0 -= x1 - img.shape[1] - 1
            x1 = img.shape[1] - 1
        if y1 >= img.shape[1]:
            y0 -= y1 - img.shape[0] - 1
            y1 = img.shape[0] - 1

        img = img[y0:y1, x0:x1]
        kpoints3d[:, :2] = kpoints3d[:, :2] - [x0, y0]

        if img.shape[0] != target_size:
            scale = target_size / img.shape[0]
            kpoints3d[:, :2] = kpoints3d[:, :2] * scale
            img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
            img = np.clip(img, 0, 1)

        return img, kpoints3d


