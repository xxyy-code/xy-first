"""BraTS (HGG/LGG) -> vol/*_vol.npy + seg/*_seg.npy

Example:
  python preprocess.py \
    --src_path /path/to/BraTS/Training \
    --tar_path /path/to/BraTS_none_npy
"""
import argparse
import os
import numpy as np
import medpy.io as medio

join = os.path.join


def sup_128(xmin, xmax):
    if xmax - xmin < 128:
        print('#' * 100)
        ecart = int((128 - (xmax - xmin)) / 2)
        xmax = xmax + ecart + 1
        xmin = xmin - ecart
    if xmin < 0:
        xmax -= xmin
        xmin = 0
    return xmin, xmax


def crop(vol):
    if len(vol.shape) == 4:
        vol = np.amax(vol, axis=0)
    assert len(vol.shape) == 3

    x_nonzeros, y_nonzeros, z_nonzeros = np.where(vol != 0)
    x_min, x_max = np.amin(x_nonzeros), np.amax(x_nonzeros)
    y_min, y_max = np.amin(y_nonzeros), np.amax(y_nonzeros)
    z_min, z_max = np.amin(z_nonzeros), np.amax(z_nonzeros)

    x_min, x_max = sup_128(x_min, x_max)
    y_min, y_max = sup_128(y_min, y_max)
    z_min, z_max = sup_128(z_min, z_max)
    return x_min, x_max, y_min, y_max, z_min, z_max


def normalize(vol):
    mask = vol.sum(0) > 0
    for k in range(4):
        x = vol[k, ...]
        y = x[mask]
        x = (x - y.mean()) / y.std()
        vol[k, ...] = x
    return vol


def main():
    parser = argparse.ArgumentParser(description='Preprocess BraTS to npy')
    parser.add_argument('--src_path', required=True, type=str,
                        help='Raw BraTS Training root with HGG/ and LGG/')
    parser.add_argument('--tar_path', required=True, type=str,
                        help='Output root; will create vol/ and seg/')
    args = parser.parse_args()

    src_path = args.src_path
    tar_path = args.tar_path

    HGG_list = ['HGG/' + x for x in os.listdir(join(src_path, 'HGG'))]
    LGG_list = ['LGG/' + x for x in os.listdir(join(src_path, 'LGG'))]
    name_list = HGG_list + LGG_list

    os.makedirs(join(tar_path, 'vol'), exist_ok=True)
    os.makedirs(join(tar_path, 'seg'), exist_ok=True)

    for file_name in name_list:
        print(file_name)
        HLG = 'HGG_' if 'HGG' in file_name else 'LGG_'
        case_id = file_name.split('/')[-1]
        flair, _ = medio.load(join(src_path, file_name, case_id + '_flair.nii.gz'))
        t1ce, _ = medio.load(join(src_path, file_name, case_id + '_t1ce.nii.gz'))
        t1, _ = medio.load(join(src_path, file_name, case_id + '_t1.nii.gz'))
        t2, _ = medio.load(join(src_path, file_name, case_id + '_t2.nii.gz'))

        vol = np.stack((flair, t1ce, t1, t2), axis=0).astype(np.float32)
        x_min, x_max, y_min, y_max, z_min, z_max = crop(vol)
        vol1 = normalize(vol[:, x_min:x_max, y_min:y_max, z_min:z_max])
        vol1 = vol1.transpose(1, 2, 3, 0)
        print(vol1.shape)

        seg, _ = medio.load(join(src_path, file_name, case_id + '_seg.nii.gz'))
        seg = seg.astype(np.uint8)
        seg1 = seg[x_min:x_max, y_min:y_max, z_min:z_max]
        seg1[seg1 == 4] = 3

        np.save(join(tar_path, 'vol', HLG + case_id + '_vol.npy'), vol1)
        np.save(join(tar_path, 'seg', HLG + case_id + '_seg.npy'), seg1)


if __name__ == '__main__':
    main()
