# Current format: binary/w-01/p001.png
# Destination format: CVC-MUSCIMA_W-01_N-10_D-binary.png
import os
import shutil
from glob import glob

from tqdm import tqdm

if __name__ == "__main__":
    input_directory = "CVCMUSCIMA_MultiConditionAligned"
    output_directory = "CVCMUSCIMA_MCA_Flat"
    os.makedirs(output_directory, exist_ok=True)

    all_images = glob(input_directory + "/**/*.png", recursive=True)
    for image_path in tqdm(all_images, desc="Flattening images"):
        _, condition, writer, page = image_path.split("\\")
        writer = writer.replace("w-", "")
        page = page.replace("p0", "").replace(".png", "")
        output_name = "CVC-MUSCIMA_W-{0}_N-{1}_D-{2}.png".format(writer,page, condition)
        shutil.copy(image_path, os.path.join(output_directory, output_name))
