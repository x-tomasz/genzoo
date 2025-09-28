import copy
import os
import numpy as np
import torch
from typing import Any, Dict, List
from yacs.config import CfgNode
import braceexpand
import cv2
import smplx
from torch.utils.data import Dataset
from .utils import get_example, expand_to_aspect_ratio, get_bbox
import random
import torchvision.transforms.v2 as T

def expand(s):
    return os.path.expanduser(os.path.expandvars(s))
def expand_urls(urls: str|List[str]):
    if isinstance(urls, str):
        urls = [urls]
    urls = [u for url in urls for u in braceexpand.braceexpand(expand(url))]
    return urls

AIC_TRAIN_CORRUPT_KEYS = {
    '0a047f0124ae48f8eee15a9506ce1449ee1ba669',
    '1a703aa174450c02fbc9cfbf578a5435ef403689',
    '0394e6dc4df78042929b891dbc24f0fd7ffb6b6d',
    '5c032b9626e410441544c7669123ecc4ae077058',
    'ca018a7b4c5f53494006ebeeff9b4c0917a55f07',
    '4a77adb695bef75a5d34c04d589baf646fe2ba35',
    'a0689017b1065c664daef4ae2d14ea03d543217e',
    '39596a45cbd21bed4a5f9c2342505532f8ec5cbb',
    '3d33283b40610d87db660b62982f797d50a7366b',
}
CORRUPT_KEYS = {
    *{f'aic-train/{k}' for k in AIC_TRAIN_CORRUPT_KEYS},
    *{f'aic-train-vitpose/{k}' for k in AIC_TRAIN_CORRUPT_KEYS},
}

body_permutation = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]
extra_permutation = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]
FLIP_KEYPOINT_PERMUTATION = body_permutation + [25 + i for i in extra_permutation]

DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])
DEFAULT_IMG_SIZE = 256

class SMALLayer(smplx.SMPLLayer):
    NUM_JOINTS = 34
    NUM_BODY_JOINTS = 34
    SHAPE_SPACE_DIM = 145

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vertex_joint_selector.extra_joints_idxs = torch.empty(0, dtype=torch.int32)

def symmetric_orthogonalization(x):
    """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

    x: should have size [batch_size, 9]

    Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
    """
    m = x.view(-1, 3, 3)
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    r = torch.matmul(u, vt)
    return r

def weak_perspective_project(verts, scale, tx, ty):
    proj = verts[:, :2] * scale
    proj[:, 0] += tx
    proj[:, 1] += ty
    return proj

def full_perspective_project(X, fov, image):
                cx, cy = (image.shape[0] / 2, image.shape[0] / 2)
                f = (cx / 2) / np.tan(np.deg2rad(fov / 2))
                k = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
                proj = (k @ X.T).T
                proj /= proj[:, 2, None]
                return proj[:, :2]

class ImageDataset(Dataset):
    @staticmethod
    def load_tars_as_webdataset(cfg: CfgNode, urls: str|List[str], train: bool,
            resampled=False,
            epoch_size=None,
            cache_dir=None,
            **kwargs) -> Dataset:
        """
        Loads the dataset from a webdataset tar file.
        """

        IMG_SIZE = cfg.MODEL.IMAGE_SIZE
        BBOX_SHAPE = cfg.MODEL.get('BBOX_SHAPE', None)
        MEAN = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        STD = 255. * np.array(cfg.MODEL.IMAGE_STD)

        # Load the dataset
        if epoch_size is not None:
            resampled = True
        corrupt_filter = lambda sample: (sample['__key__'] not in CORRUPT_KEYS)
        import webdataset as wds
        dataset = wds.WebDataset(expand_urls(urls),
                                nodesplitter=wds.split_by_node,
                                shardshuffle=True,
                                resampled=resampled,
                                # cache_dir=cache_dir,
                                empty_check=False,
                              )#.select(corrupt_filter)
        if train:
            dataset = dataset.shuffle(100)

        dataset = dataset.decode('pil').rename(image='jpg;jpeg;png')
        dataset = dataset.select(lambda x: 'betas' not in x['npz'] or np.abs(x['npz']['betas']).max() < 10)

        use_skimage_antialias = cfg.DATASETS.get('USE_SKIMAGE_ANTIALIAS', False)
        border_mode = {
            'constant': cv2.BORDER_CONSTANT,
            'replicate': cv2.BORDER_REPLICATE,
        }[cfg.DATASETS.get('BORDER_MODE', 'constant')]

        if train:
            transforms = T.Compose([
                T.Resize(256),
                T.RandomOrder([
                    T.RandomResize(min_size=56, max_size=256),
                    T.RandomGrayscale(),
                    T.RandomPhotometricDistort(),
                    T.JPEG((50, 90)),
                ]),
            ])
        else:
            transforms = None

        smal = SMALLayer(model_path=cfg.SMPL.MODEL_PATH, num_betas=SMALLayer.SHAPE_SPACE_DIM)
        # Process the dataset further
        dataset = dataset.map(lambda x: ImageDataset.process_webdataset_tar_item(x, train,
                                                        augm_config=cfg.DATASETS.CONFIG,
                                                        MEAN=MEAN, STD=STD, IMG_SIZE=IMG_SIZE,
                                                        BBOX_SHAPE=BBOX_SHAPE,
                                                        use_skimage_antialias=use_skimage_antialias,
                                                        border_mode=border_mode,
                                                        smal=smal,
                                                        transforms=transforms,
                                                        ))

        if epoch_size is not None:
            dataset = dataset.with_epoch(epoch_size)

        return dataset

    @staticmethod
    def process_webdataset_tar_item(item, train, 
                                    augm_config=None, 
                                    MEAN=DEFAULT_MEAN, 
                                    STD=DEFAULT_STD, 
                                    IMG_SIZE=DEFAULT_IMG_SIZE,
                                    BBOX_SHAPE=None,
                                    use_skimage_antialias=False,
                                    border_mode=cv2.BORDER_CONSTANT,
                                    smal=None,
                                    transforms=None,
                                    ):
        key = item['__key__']
        image = item['image']
        data = item['npz']

        # item = {}
        # item['img'] = np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        # item['keypoints_2d'] = np.zeros((35, 3), dtype=np.float32)
        # item['keypoints_3d'] = np.zeros((35, 4), dtype=np.float32)
        # item['vertices_2d'] = np.zeros((3889, 3), dtype=np.float32)
        # item['vertices'] = np.zeros((3889, 4), dtype=np.float32)
        # item['box_center'] = np.zeros(2, dtype=np.float32)
        # item['box_size'] = np.zeros(2, dtype=np.float32)
        # item['img_size'] = np.array([IMG_SIZE, IMG_SIZE], dtype=np.float32)
        # item['smpl_params'] = {
        #     'betas': np.zeros(145, dtype=np.float32),
        #     'global_orient': np.zeros(9, dtype=np.float32),
        #     'body_pose': np.zeros(34*9, dtype=np.float32),
        # }
        # item['has_smpl_params'] = {
        #     'betas': False,
        #     'global_orient': False,
        #     'body_pose': False,
        # }
        # item['imgname'] = key
        # return item

        if transforms is not None:
            image = transforms(image)
        image = np.array(image)

        keypoints_2d = data.get('keypoints_2d')
        keypoints_3d = data.get('keypoints_3d')
        vertices = data.get('vertices')
        vertices_2d = data.get('vertices_2d')
        pose = data.get('poses')
        if pose is not None:
            pose = symmetric_orthogonalization(torch.tensor(pose)).numpy()
            global_orient = pose[:1]
            body_pose = pose[1:]
        else:
            global_orient = None
            body_pose = None
        betas = data.get('betas')

        ## Weak perspective
        scale = data.get('scales')
        tx = data.get('txs')
        ty = data.get('tys')
        ## Full perspective
        # transl = data.get('transls')
        # fov = data.get('fovs')

        # from PIL import Image
        # image = Image.fromarray(image)
        # image.save(f'aaa_{key}.png')
        # raise Exception()

        ## Weak perspective
        if not any(n is None for n in [global_orient, body_pose, betas, scale, tx, ty]):

        ## Full perspective
        # is_train = False
        # if not any(n is None for n in [global_orient, body_pose, betas, transl, fov]):
            # is_train = True
            with torch.inference_mode():
                smal_output = smal(
                    global_orient=torch.tensor(global_orient)[None],
                    body_pose=torch.tensor(body_pose)[None],
                    betas=torch.tensor(betas)[None],
                )
            vertices = smal_output.vertices.numpy()[0]
            keypoints_3d = smal_output.joints.numpy()[0]

            ## Weak perspective
            keypoints_2d = weak_perspective_project(keypoints_3d, scale, tx, ty)
            vertices_2d = weak_perspective_project(vertices, scale, tx, ty)

            ## Full perspective
            # keypoints_3d_translated = keypoints_3d + transl
            # vertices_translated = vertices + transl
            # keypoints_2d = full_perspective_project(keypoints_3d_translated, fov, image)
            # vertices_2d = full_perspective_project(vertices_translated, fov, image)

            # import matplotlib.pyplot as plt
            # im = image
            # fig, ax = plt.subplots()
            # ax.imshow(im)
            # ax.scatter(keypoints_2d[:, 0], keypoints_2d[:, 1], c='red')
            # ax.scatter(vertices_2d[:, 0], vertices_2d[:, 1], c='blue', alpha=0.5)
            # fig.savefig(f'aaa_{key}.png')
            # raise Exception(keypoints_2d)

        # print(keypoints_2d.shape if keypoints_2d is not None else 'no keypoints_2d', keypoints_3d.shape if keypoints_3d is not None else 'no keypoints_3d', vertices_2d.shape if vertices_2d is not None else 'no vertices_2d', vertices.shape if vertices is not None else 'no vertices', image.shape if image is not None else 'no image')
        # if any is None, print `data`:
        if any(n is None for n in [keypoints_2d, keypoints_3d, vertices_2d, vertices]):
            raise Exception(data)

        ## Weak perspective
        keypoints_2d = np.stack([
            image.shape[1] / 2 + image.shape[1] * keypoints_2d[:, 0] / 2,
            image.shape[0] / 2 + image.shape[0] * keypoints_2d[:, 1] / 2,
            np.ones(keypoints_2d.shape[0]),
        ], axis=1)
        vertices_2d = np.stack([
            image.shape[1] / 2 + image.shape[1] * vertices_2d[:, 0] / 2,
            image.shape[0] / 2 + image.shape[0] * vertices_2d[:, 1] / 2,
            np.ones((vertices_2d.shape[0])),
        ], axis=1)

        ## Full perspective 
        # if not is_train: # validation data is in normalized coordinates, convert to pixel coordinates
        #     keypoints_2d[:, :2] = (keypoints_2d[:, :2] + 1) * image.shape[1] / 2
        #     vertices_2d[:, :2] = (vertices_2d[:, :2] + 1) * image.shape[0] / 2
        # keypoints_2d = np.concatenate([keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=1)
        # vertices_2d = np.concatenate([vertices_2d, np.ones((vertices_2d.shape[0], 1))], axis=1)

        # Add confidence channel
        keypoints_3d = np.concatenate([keypoints_3d, np.ones((keypoints_3d.shape[0], 1))], axis=1)
        vertices = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)

        if random.random() < 0.5:
            center, scale = get_bbox(vertices_2d)
            scale /= 200
        else:
            center = np.array([image.shape[1]/2, image.shape[0]/2])
            scale = np.array([image.shape[1]/200, image.shape[0]/200])

        center_x = center[0]
        center_y = center[1]
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()

        # # set center, scale to fill the image (by width and height)
        # bbox = np.array([center - 100*scale, center + 100*scale])
        # # print(center, scale)
        # # print(bbox)
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.imshow(image)
        # ax.add_patch(plt.Rectangle(bbox[0], bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1], fill=False, edgecolor='red', linewidth=2))
        # # ax.set_title(f'{center=} {scale=} {bbox_size=}')
        # fig.savefig(f'aaa_{key}.png')
        # raise Exception()

        smpl_params = {
            'global_orient': global_orient,
            'body_pose': body_pose,
            'betas': betas,
        }
        has_smpl_params = {
            'global_orient': global_orient is not None,
            'body_pose': body_pose is not None,
            'betas': betas is not None,
        }

        # fill in None values with zeros of appropriate shape
        if smpl_params['global_orient'] is None:
            smpl_params['global_orient'] = np.zeros(1*9, dtype=np.float32)
        if smpl_params['body_pose'] is None:
            smpl_params['body_pose'] = np.zeros(34*9, dtype=np.float32)
        if smpl_params['betas'] is None:
            smpl_params['betas'] = np.zeros(145, dtype=np.float32)

        augm_config = copy.deepcopy(augm_config)
        # Crop image and (possibly) perform data augmentation
        mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
        img_rgba = np.concatenate([image, mask.astype(np.uint8)[:,:,None]*255], axis=2)
        img_patch_rgba, keypoints_2d, keypoints_3d, vertices_2d, vertices, smpl_params, has_smpl_params, img_size, trans = get_example(img_rgba,
                                                                                                    center_x, center_y,
                                                                                                    bbox_size, bbox_size,
                                                                                                    keypoints_2d, keypoints_3d,
                                                                                                    vertices_2d, vertices,
                                                                                                    smpl_params, has_smpl_params,
                                                                                                    FLIP_KEYPOINT_PERMUTATION,
                                                                                                    IMG_SIZE, IMG_SIZE,
                                                                                                    MEAN, STD, train, augm_config,
                                                                                                    is_bgr=False, return_trans=True,
                                                                                                    use_skimage_antialias=use_skimage_antialias,
                                                                                                    border_mode=border_mode,
                                                                                                    )
        img_patch = img_patch_rgba[:3,:,:]

        if has_smpl_params['global_orient']:
            smpl_params['global_orient'] = symmetric_orthogonalization(torch.tensor(smpl_params['global_orient'])).numpy()
        if has_smpl_params['body_pose']:
            smpl_params['body_pose'] = symmetric_orthogonalization(torch.tensor(smpl_params['body_pose'])).numpy()

        item = {}
        item['img'] = img_patch
        item['keypoints_2d'] = keypoints_2d.astype(np.float32)
        item['keypoints_3d'] = keypoints_3d.astype(np.float32)
        item['vertices_2d'] = vertices_2d.astype(np.float32)
        item['vertices'] = vertices.astype(np.float32)
        item['box_center'] = center.copy()
        item['box_size'] = bbox_size
        item['img_size'] = 1.0 * img_size[::-1].copy()
        item['smpl_params'] = smpl_params
        item['has_smpl_params'] = has_smpl_params
        item['imgname'] = key

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 1)
        # ax = [ax]
        # ax[0].imshow(img_patch_rgba[:, :, :3])
        # ax[0].scatter(img_patch.shape[1]/2 + img_patch.shape[1]*vertices_2d[:, 0], img_patch.shape[0]/2 + img_patch.shape[0]*vertices_2d[:, 1], c='blue', alpha=0.5)
        # ax[0].scatter(img_patch.shape[1]/2 + img_patch.shape[1]*keypoints_2d[:, 0], img_patch.shape[0]/2 + img_patch.shape[0]*keypoints_2d[:, 1], c='red')
        
        # # reproject vertices:
        # with torch.inference_mode():
        #     smal_output = smal(
        #         global_orient=torch.tensor(item['smpl_params']['global_orient'])[None],
        #         body_pose=torch.tensor(item['smpl_params']['body_pose'])[None],
        #         betas=torch.tensor(item['smpl_params']['betas'])[None],
        #     )
        # # vertices_2d_reproj = weak_perspective_project(smal_output.vertices.numpy()[0], item['box_center'], item['box_center'][0], item['box_center'][1])
        # # ax[1].scatter(img_patch.shape[1]/2 + img_patch.shape[1]*vertices_2d_reproj[:, 0], img_patch.shape[0]/2 + img_patch.shape[0]*vertices_2d_reproj[:, 1], c='blue', alpha=0.5)
        # # ax[1].invert_yaxis()
        # # ax[1].set_aspect('equal')
        # fig.savefig(f'aaa_{key}.png')
        # raise Exception()
        return item
