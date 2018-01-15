import os
from typing import Tuple, List

from cv2 import cv2, countNonZero, cvtColor

from PIL import Image, ImageChops
from tqdm import tqdm
import re
import numpy as np


def align_images_with_opencv(path_to_desired_image: str, path_to_image_to_warp: str,
                             output_path: str):
    # See https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/

    # image_sr_path = "C:/Users/Alex/Repositories/CVC-MUSCIMA/CVCMUSCIMA_SR/CvcMuscima-Distortions/ideal/w-50/image/p020.png"
    # image_wi_path = "C:/Users/Alex/Repositories/CVC-MUSCIMA/CVCMUSCIMA_WI/CVCMUSCIMA_WI/PNG_GT_Gray/w-50/p020.png"

    # Read the images to be aligned
    im1 = cv2.imread(path_to_desired_image)
    im2 = cv2.imread(path_to_image_to_warp)

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    im1_gray = cv2.bitwise_not(im1_gray)

    # Find size of image1
    sz = im1.shape

    # Define the motion model
    warp_mode = cv2.MOTION_AFFINE

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 100

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-7

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)

    # Warp BW-image
    bw_image_path = str.replace(path_to_image_to_warp, "PNG_GT_Gray", "PNG_GT_BW")
    bw_image = cv2.imread(bw_image_path)
    bw_image_gray = cv2.cvtColor(bw_image, cv2.COLOR_BGR2GRAY)
    bw_image_aligned = cv2.warpAffine(bw_image_gray, warp_matrix, (sz[1], sz[0]),
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cv2.imwrite(bw_image_path, bw_image_aligned)

    # Warp NoStaff-image
    bw_no_staff_image_path = str.replace(path_to_image_to_warp, "PNG_GT_Gray", "PNG_GT_NoStaff")
    bw_no_staff_image = cv2.imread(bw_no_staff_image_path)
    bw_no_staff_image_gray = cv2.cvtColor(bw_no_staff_image, cv2.COLOR_BGR2GRAY)
    bw_no_staff_image_aligned = cv2.warpAffine(bw_no_staff_image_gray, warp_matrix, (sz[1], sz[0]),
                                               flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cv2.imwrite(bw_no_staff_image_path, bw_no_staff_image_aligned)

    # Warp Gray-Image last, in case user interrupts, to make sure the process continues appropriately and doesn't
    # miss the bw and bw_no_staff images, because the process compares only on grayscale images
    im2_aligned = cv2.warpAffine(im2_gray, warp_matrix, (sz[1], sz[0]),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    cv2.imwrite(output_path, im2_aligned)


def check_staff_removal_dataset_integrity(staff_removal_dataset_dir: str, writers: List[str]) -> List[str]:
    distortions = os.listdir(staff_removal_dataset_dir)
    inconsistent_sr_images = []
    ideal_images = []

    for writer in tqdm(writers, desc="Checking integrity of SR dataset"):
        for distortion in distortions:
            for i in range(1, 21):
                filename = "p{0:03}.png".format(i)
                staff_removal_gt_path = os.path.join(staff_removal_dataset_dir, distortion, writer, "gt", filename)
                staff_removal_image_path = os.path.join(staff_removal_dataset_dir, distortion, writer, "image",
                                                        filename)
                staff_removal_symbol_path = os.path.join(staff_removal_dataset_dir, distortion, writer, "symbol",
                                                         filename)
                if distortion == "ideal":
                    ideal_images.append(staff_removal_image_path)
                image_gt = Image.open(staff_removal_gt_path)
                image_ideal = Image.open(staff_removal_image_path)
                image_symbol = Image.open(staff_removal_symbol_path)
                if image_gt.size != image_ideal.size or image_gt.size != image_symbol.size:
                    inconsistent_sr_images.append(staff_removal_image_path)
    if len(inconsistent_sr_images) > 0:
        print("Found {0} inconsistencies in SR dataset".format(len(inconsistent_sr_images)))

    return ideal_images


def check_writer_identification_dataset_integrity(writer_identification_dataset_dir: str, writers: List[str]):
    inconsistent_wi_images = []
    for writer in tqdm(writers, desc="Checking integrity of WI dataset"):
        for i in range(1, 21):
            filename = "p{0:03}.png".format(i)
            writer_identification_bw = os.path.join(writer_identification_dataset_dir, "PNG_GT_BW",
                                                    writer, filename)
            writer_identification_gray = os.path.join(writer_identification_dataset_dir, "PNG_GT_Gray",
                                                      writer, filename)
            writer_identification_bw_no_staff = os.path.join(writer_identification_dataset_dir, "PNG_GT_NoStaff",
                                                             writer, filename)

            image_bw = Image.open(writer_identification_bw)
            image_gray = Image.open(writer_identification_gray)
            image_bw_no_staff = Image.open(writer_identification_bw_no_staff)

            if image_bw.size != image_gray.size or image_bw.size != image_bw_no_staff.size:
                inconsistent_wi_images.append(writer_identification_gray)
    if len(inconsistent_wi_images) > 0:
        print("Found {0} inconsistencies in WI dataset".format(len(inconsistent_wi_images)))


def fix_inconsistencies_in_sr_dataset(ideal_images: List[str], grayscale_images: List[str],
                                      output_diff_directory: str) -> List[str]:
    inconsistent_images = []
    for image_sr_path, image_wi_path in tqdm(zip(ideal_images, grayscale_images), total=1000,
                                             desc="Checking consistency between datasets"):
        image_sr = Image.open(image_sr_path)  # type: Image.Image
        image_wi = Image.open(image_wi_path)  # type: Image.Image

        if image_sr.size != image_wi.size:
            if image_sr.width < image_wi.width or image_sr.height < image_wi.height:
                # image_sr is always bigger than image_wi - otherwise we would get a print here
                print("{0}x{1} - {2}x{3}".format(image_sr.width, image_sr.height, image_wi.width, image_wi.height))

            inconsistent_images.append(image_sr_path)

            diff_before_cropping = ImageChops.difference(Image.open(image_sr_path), Image.open(image_wi_path))
            # naive_crop_to_smaller_size(image_wi, image_sr, image_sr_path)
            align_images_with_opencv(image_sr_path, image_wi_path, image_wi_path)
            diff_after_cropping = ImageChops.difference(Image.open(image_sr_path), Image.open(image_wi_path))

            writer = re.search("w-\d\d", image_sr_path)[0]

            output_name = os.path.splitext(os.path.basename(image_sr_path))[0]
            diff_before_cropping.save(
                os.path.join(output_diff_directory, "{0}_{1}-before.png".format(writer, output_name)))
            diff_after_cropping.save(
                os.path.join(output_diff_directory, "{0}_{1}-after.png".format(writer, output_name)))

    return inconsistent_images


if __name__ == "__main__":
    base_path = "C:/Users/Alex/Repositories/CVC-MUSCIMA"
    staff_removal_dataset_dir = os.path.join(base_path, "CVCMUSCIMA_SR/CvcMuscima-Distortions")
    writer_identification_dataset_dir = os.path.join(base_path, "CVCMUSCIMA_WI/CVCMUSCIMA_WI")
    output_diff_directory = os.path.join(base_path, "output_diffs")
    os.makedirs(output_diff_directory, exist_ok=True)
    writers = os.listdir(staff_removal_dataset_dir + "/ideal")

    # Check the integrity of the SR dataset, that all versions of the same image have the exact same dimensions (gt, image, symbol).
    ideal_images = check_staff_removal_dataset_integrity(staff_removal_dataset_dir, writers)

    # Check the integrity of WI dataset
    check_writer_identification_dataset_integrity(writer_identification_dataset_dir, writers)

    # Collect all grayscale images
    grayscale_images = []
    for writer in tqdm(writers, desc="Collecting grayscale images from WI dataset"):
        current_directory = os.path.join(base_path, writer_identification_dataset_dir, "PNG_GT_Gray", writer)
        images = [os.path.join(current_directory, p) for p in os.listdir(current_directory) if p.upper() != ".DS_STORE"]
        grayscale_images.extend(images)

    # Compare SR with WI dataset and fix SR dataset
    inconsistent_images = fix_inconsistencies_in_sr_dataset(ideal_images, grayscale_images, output_diff_directory)

    print("Fixed {0} inconsistent images:".format(len(inconsistent_images)))
    for inconsistent_image in inconsistent_images:
        print("Fixed {}".format(inconsistent_image))

    # Check the integrity of the SR dataset, that all versions of the same image have the exact same dimensions (gt, image, symbol).
    check_staff_removal_dataset_integrity(staff_removal_dataset_dir, writers)

    # Check the integrity of WI dataset
    check_writer_identification_dataset_integrity(writer_identification_dataset_dir, writers)

    print("Number on non-black pixels in the diff images:")
    for diff_file in os.listdir(output_diff_directory):
        diff_image = cv2.imread(os.path.join(output_diff_directory, diff_file), cv2.IMREAD_GRAYSCALE) + 1
        non_zeroes = countNonZero(diff_image)
        if non_zeroes > 5000:
            print("{0}: {1}".format(diff_file, non_zeroes))
