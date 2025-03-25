#
import pandas as pd
from joblib import Parallel, delayed
import os
import nibabel as nib
import nibabel.processing as nibp
import numpy as np
from PIL import Image
from radiomics import featureextractor
import shutil


radDBPath = "./data/radDB"
slicesPath = "./slices"


def getData (dataset):
    data = pd.read_csv(f"./data/pinfo_{dataset}.csv")
    for i, (idx, row) in enumerate(data.iterrows()):
        image = os.path.join(radDBPath, row["Patient"], "image.nii.gz")
        if dataset == "CRLM":
            mask = os.path.join(radDBPath, row["Patient"], "segmentation_lesion0_RAD.nii.gz")
        else:
            mask = os.path.join(radDBPath, row["Patient"], "segmentation.nii.gz")
        data.at[idx, "Image"] = image
        data.at[idx, "mask"] = mask
    return data


def processFile (row, dataID):
    f = row["Image"]
    fmask = row["mask"]
    img = nib.load(f)
    seg = nib.load(fmask)

    tmp = seg.get_fdata()
    tmp = np.asarray(tmp > 0, dtype = np.uint8)

    # copy also over affine infos, else we can end up with volumes that are SLIGHTLY different
    new_seg = seg.__class__(tmp, img.affine, img.header)
    rimg = nibp.resample_to_output(img, voxel_sizes = [1, 1, 1], order = 3)
    rSeg = nibp.resample_to_output(new_seg, voxel_sizes = [1, 1, 1], order = 0)

    test = np.median(rSeg.get_fdata())
    assert (test == 0.0 or test == 1.0)
    assert (rimg.shape == rSeg.shape)

    seg_data = rSeg.get_fdata()
    slice_sums = np.sum(seg_data, axis=(0, 1))
    max_idx = np.argmax(slice_sums)

    diff = 3
    slices = [max(0, max_idx - diff), max_idx, min(rimg.shape[2] - 1, max_idx + diff)]
    assert len(set(slices)) == 3

    # extract radiomics features first
    if dataID == "CRLM" or dataID == "GIST":
        params = os.path.join("config/CT.yaml")
    else:
        params = os.path.join("config/MR.yaml")
    eParams = {"binWidth": 25, "force2D": True}
    extractor = featureextractor.RadiomicsFeatureExtractor(params, **eParams)
    extractor.enableImageTypeByName("LBP2D")

    for i, sl in enumerate(slices):
        os.makedirs(f'{slicesPath}/{dataID}/{row["Diagnosis"]}/{row["Patient"]}/', exist_ok = True)
        fvol = rimg.slicer[:, :, sl:sl+1]
        fmask = rSeg.slicer[:, :, sl:sl+1]

        vol_path = f'/tmp/temp_vol_{row["Patient"]}.nii.gz'
        mask_path = f'/tmp/temp_mask_{row["Patient"]}.nii.gz'
        nib.save(fvol, vol_path)
        nib.save(fmask, mask_path)

        mask_data = fmask.get_fdata()
        if len(np.unique(mask_data))  == 1: # for debug
            print ("### ", row)
        f = extractor.execute(vol_path, mask_path, label=1)
        f = {p: f[p] for p in f if "diagnost" not in p}
        pd.DataFrame([f]).to_csv(f'{slicesPath}/{dataID}/{row["Diagnosis"]}/{row["Patient"]}/{row["Patient"]}_{i}.csv')


    for i, idx in enumerate(slices):
        os.makedirs(f'{slicesPath}/{dataID}/{row["Diagnosis"]}/{row["Patient"]}/', exist_ok = True)
        img_name = f'{slicesPath}/{dataID}/{row["Diagnosis"]}/{row["Patient"]}/{row["Patient"]}_{i}.png'

        img_slice = rimg.get_fdata()[:, :, idx]
        seg_slice = seg_data[:, :, idx]

        # normalize mask
        masked_pixels = img_slice[seg_slice > 0]
        min_val = masked_pixels.min()
        max_val = masked_pixels.max()
        img_slice = np.zeros_like(img_slice)
        img_slice[seg_slice > 0] = (masked_pixels - min_val) / (max_val - min_val) * 255

        Image.fromarray(np.uint8(img_slice)).save(img_name)


if __name__ == '__main__':
    try:
        shutil.rmtree(slicesPath)
    except:
        pass
    for d in ["Lipo", "Desmoid", "CRLM", "GIST"]:
        data = getData(d)
        fv = Parallel (n_jobs = 16)(delayed(processFile)(row, d) for (idx, row) in data.iterrows())

#
