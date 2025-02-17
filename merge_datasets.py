# Note: This is just a utility file because we're using two different (but similar folder structure) asl alphabet image
# datasets that need to be merged into a single one, either at file management level or at code level.

import shutil
from pathlib import Path


def merge_datasets(source1, source2, destination):
    source1, source2, destination = Path(source1), Path(source2), Path(destination)

    for dataset in ["asl_alphabet_test", "asl_alphabet_train"]:
        src1_path = source1 / dataset
        src2_path = source2 / dataset
        dest_path = destination / dataset

        dest_path.mkdir(parents=True, exist_ok=True)

        for src in [src1_path, src2_path]:
            if not src.exists():
                continue

            for item in src.iterdir():
                if item.is_file():
                    new_name = item.name
                    counter = 1
                    while (dest_path / new_name).exists():
                        new_name = f"{item.stem}_{counter}{item.suffix}"
                        counter += 1
                    shutil.move(item, dest_path / new_name)
                elif item.is_dir():
                    sub_dest_path = dest_path / item.name
                    sub_dest_path.mkdir(exist_ok=True)
                    for img in item.iterdir():
                        if img.is_file():
                            new_name = img.name
                            counter = 1
                            while (sub_dest_path / new_name).exists():
                                new_name = f"{img.stem}_{counter}{img.suffix}"
                                counter += 1
                            shutil.move(img, sub_dest_path / new_name)
            shutil.rmtree(src)


merge_datasets("asl_alphabet_dataset", "ASL_Alphabet_Dataset", "asl_alphabet_image_dataset")
