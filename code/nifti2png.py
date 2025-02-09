# -*- coding: utf-8 -*-
"""
@file: nifti2png.py
@brief: aligning nifti volume to dicom series, cut into slices, and save as png files.
@author: Luyang Zhang
@contact: luyang.zhang@os.lpixel.net
@created: 2025-02-09
@modified: 2025-02-09
@license: MIT
"""

import os
import numpy as np
import pydicom
import nibabel as nib
import matplotlib.pyplot as plt
import csv

import cv2
import pydicom


def generate_coordinate_matrix(coord_system):
    """
    Generate a transformation matrix from the given coordinate system
    to the world coordinate system.
    :param coord_system: String representing the coordinate system (e.g., 'LSP', 'RAS', etc.)
    :return: 4x4 transformation matrix
    """
    if len(coord_system) != 3:
        raise ValueError("Coordinate system must have exactly 3 characters (e.g., 'RAS', 'LSP').")
    
    # Mapping from coordinate labels to axis directions
    axis_mapping = {
        'R': [1, 0, 0],   # Right (positive X)
        'L': [-1, 0, 0],  # Left (negative X)
        'A': [0, 1, 0],   # Anterior (positive Y)
        'P': [0, -1, 0],  # Posterior (negative Y)
        'S': [0, 0, 1],   # Superior (positive Z)
        'I': [0, 0, -1]   # Inferior (negative Z)
    }

    # Extract the directions for X, Y, Z
    x_axis = axis_mapping[coord_system[0]]
    y_axis = axis_mapping[coord_system[1]]
    z_axis = axis_mapping[coord_system[2]]

    # Ensure the coordinate system is valid 
    if not np.isclose(np.dot(x_axis, y_axis), 0) or not np.isclose(np.dot(x_axis, z_axis), 0) or not np.isclose(np.dot(y_axis, z_axis), 0):
        raise ValueError(f"Invalid coordinate system: {coord_system}. Axes must be orthogonal.")
    
    # Create the transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, 0] = x_axis  # X axis
    transform_matrix[:3, 1] = y_axis  # Y axis
    transform_matrix[:3, 2] = z_axis  # Z axis

    return transform_matrix

def compute_affine(dicom_files,need_sort = True):
    if need_sort:
        dicom_files = sorted(
        [os.path.join(dcm_folder, f) for f in os.listdir(dcm_folder) if f.endswith('.dcm')],
        key=lambda x: pydicom.dcmread(x).InstanceNumber
    )
    
    ds = pydicom.dcmread(dicom_files[0])
    
    image_orientation = np.array(ds.ImageOrientationPatient) 

    row_cosine = image_orientation[3:]
    col_cosine = image_orientation[:3]

    pixel_spacing = np.array(ds.PixelSpacing) 
    
    first_position = np.array(ds.ImagePositionPatient)  
    
    if len(dicom_files) > 1:
        ds_next = pydicom.dcmread(dicom_files[1])
        second_position = np.array(ds_next.ImagePositionPatient)

        
        slice_direction = second_position - first_position
        slice_spacing = np.linalg.norm(slice_direction)
        slice_cosine = slice_direction / slice_spacing

    else:
        slice_cosine = np.cross(col_cosine,row_cosine)
        slice_spacing = 1.0
    

    affine = np.eye(4)
    affine[:3, 0] = row_cosine * pixel_spacing[0]
    affine[:3, 1] = col_cosine * pixel_spacing[1]
    affine[:3, 2] = slice_cosine * slice_spacing * np.sign(ds.InstanceNumber-ds_next.InstanceNumber)
    affine[:3, 3] = first_position

    return affine, dicom_files


def convert_nifti2png(dicom_files, nii_img,need_sort = False, verbose = False):
    if need_sort:
        dicom_files = sorted(
        [os.path.join(dcm_folder, f) for f in os.listdir(dcm_folder) if f.endswith('.dcm')],
        key=lambda x: pydicom.dcmread(x).InstanceNumber
    )

    nii_data = nii_img.get_fdata()
    nii_affine = nii_img.affine
    dicom_affine, dicom_files = compute_affine(dicom_files)
    if verbose:
        print (f'Dicom affine matrix {dicom_affine}')
        print (f'nifti affine matrix {nii_affine}')

    m_lps2ras = calculate_transform_matrix('RAS', 'LPS')
    m_nifti2dicom = np.linalg.inv(m_lps2ras @ dicom_affine) @ nii_affine
    ornt = nib.orientations.io_orientation(m_nifti2dicom)
    reoriented_nifti = nib.orientations.apply_orientation(nii_data, ornt)
    return dico_files, reoriented_nifti


def save_overlay_and_mask(dicom_files, reoriented_nifti, output_overlay_dir, output_mask_dir):#,raw_reoriented_nifti, output_raw_dir):
    if not os.path.exists(output_overlay_dir):
        os.makedirs(output_overlay_dir)
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)

    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)
    '''
    if not os.path.exists(output_raw_dir):
        os.makedirs(output_raw_dir)
    '''
    mask_mapping = []

    for i, dicom_file in enumerate(dicom_files):
        print(f'Processing {dicom_file}, slice {i+1}/{len(dicom_files)}')
        ds = pydicom.dcmread(dicom_file)
        original_image = ds.pixel_array

        # Resize the reoriented NIfTI mask to match the DICOM resolution
        mask_slice = reoriented_nifti[:, :, i]
  

        original_image_normalized = (original_image / original_image.max() * 255).astype(np.uint8)

        # Normalize mask to [0, 255] for overlay
        mask_normalized = (mask_slice / mask_slice.max() * 255).astype(np.uint8)

        # Convert the original image to RGB
        overlay = cv2.cvtColor(original_image_normalized, cv2.COLOR_GRAY2BGR)

        # Create a red mask with the same shape as the overlay
        red_mask = np.zeros_like(overlay, dtype=np.uint8)
        red_mask[..., 2] = mask_normalized  # Red channel

        # Blend the original image and the red mask using addWeighted
        alpha = 0.5  # Transparency factor for the mask
        beta = 1 - alpha
        gamma = 0  # Scalar added to each sum
        overlay = cv2.addWeighted(overlay, beta, red_mask, alpha, gamma)

        # Save the overlay as JPG
        overlay_filename = os.path.join(output_overlay_dir, f'{os.path.splitext(os.path.basename(dicom_file))[0]}.jpg')
        cv2.imwrite(overlay_filename, overlay)


        # Save mask as PNG
        mask_filename = os.path.join(output_mask_dir, f'{os.path.splitext(os.path.basename(dicom_file))[0]}.png')
        plt.imsave(mask_filename, mask_slice, cmap='gray')

        # Append to mask mapping
        mask_mapping.append({'DICOM': dicom_file, 'Mask': mask_filename})
        #print (f'saving to {mask_filename}')

    return mask_mapping


def full_process(dcm_folder,nii_file,out_folder): #,raw_nifti_data):
    
    dataset,basename =  dcm_folder.strip("/").split("/")[-3],dcm_folder.strip("/").split("/")[-1]
    print(f'Processing {dataset} {basename}')
    output_overlay_dir = os.path.join(out_folder,f'{dataset}',f'{basename}','overlay') 
    output_mask_dir = os.path.join(out_folder,f'{dataset}',f'{basename}','mask') 

    output_raw_dir = os.path.join(out_folder,f'{dataset}',f'{basename}','raw') 

    mapping_path = os.path.join(out_folder,f'{dataset}','mapping')
    output_csv_path = os.path.join(mapping_path,f'{basename}.csv') 
    

    dicom_files = sorted(
        [os.path.join(dcm_folder, f) for f in os.listdir(dcm_folder) if f.endswith('.dcm')],
        key=lambda x: pydicom.dcmread(x).InstanceNumber
    )
    try:
        nii_img = nib.load(nii_file)
    except:
        print(f'Segmentation {nii_file} not found')
        return 0
    nii_data = nii_img.get_fdata()
    nii_affine = nii_img.affine

    dicom_affine = compute_affine(dicom_files)
    print (f'Dicom affine matrix {dicom_affine}')
    print (f'nifti affine matrix {nii_affine}')
    m_lps2ras = calculate_transform_matrix('RAS', 'LPS')
    m_nifti2dicom = np.linalg.inv(m_lps2ras @ dicom_affine) @ nii_affine
    ornt = nib.orientations.io_orientation(m_nifti2dicom)
    reoriented_nifti = nib.orientations.apply_orientation(nii_data, ornt)

    mask_mapping = save_overlay_and_mask(dicom_files, reoriented_nifti, output_overlay_dir, output_mask_dir)#, raw_reoriented_nifti, output_raw_dir)




    if not os.path.exists(mapping_path):
        os.makedirs(mapping_path)
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['DICOM', 'Mask'])
        writer.writeheader()
        writer.writerows(mask_mapping)

if __name__ == '__main__':
    d_list = os.listdir('/mnt/project/science_intern/S900006_ct/data/')
    for d in d_list:
        if d in ['D00_public','D01_sample','D02_trk_20241227_cancer']:
            continue
        dcm_folders = os.listdir(f'/mnt/project/science_intern/S900006_ct/data/{d}/dcm_ct')
        dcm_folders = [os.path.join(f'/mnt/project/science_intern/S900006_ct/data/{d}/dcm_ct',poi) for poi in dcm_folders]
        out_folder = r'/home/Zhang/work_process/results'
        for dcm_folder in dcm_folders:
            nii_file = f'/home/Zhang/work_process/2024-12/seg_results/{d}/{os.path.basename(dcm_folder)}/output/pancreas.nii.gz'
            
            #raw_nifti_data = f'/home/Zhang/work_process/nnunet_data/nnUNet_raw_data_base/Dataset007_Pancreas/imagesTs/{os.path.basename(dcm_folder)}.nii.gz'
            full_process(dcm_folder,nii_file,out_folder)#,raw_nifti_data)
