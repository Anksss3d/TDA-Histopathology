import cv2
import os
import shutil
import sys
import csv
import glob
import random
import timeit

from feature_extraction import generate_lbp_features, generate_betti_input, generate_gabor_filters
from utils import time_break, tile_content

def check_structure(INPUT_DIR, TILES_DIR, classes):
    '''
    Function to check the direcory of dataset and generate directory structure for output datset
    :param INPUT_DIR: Dataset input directory
    :param TILES_DIR: Output Tiles directory
    :param classes: Names of classes as list
    :return: None
    '''
    if os.path.exists(INPUT_DIR):
        if os.path.exists(TILES_DIR):
            shutil.rmtree(TILES_DIR)
        print("TILES DIR path does not exist, generating one")
        os.mkdir(TILES_DIR)
        for grp in ["train", "validation"]:
            os.mkdir(f"{TILES_DIR}{grp}")
            for clas in classes:
                os.mkdir(f"{TILES_DIR}{grp}/{clas}")
    else:
        print("Dataset dir couldnt found. Exiting")
        sys.exit(0)


def generate_tiles(
        INPUT_DIR,
        TILE_DIR,
        CLASS_TO_INDEX,
        INPUT_HEIGHT,
        INPUT_WIDTH,
        STEP_HEIGHT,
        STEP_WIDTH,
        TILE_SIZE,
        image_format,
        print_frequency=10,
        samples = None,
        thresh = 0.7
    ):
    '''
    Function to generate tiles and all features as CSV file.
    :param INPUT_DIR: Dataset input directory
    :param TILES_DIR: Output Tiles directory
    :param CLASS_TO_INDEX: Class to index dictionary
    :param INPUT_HEIGHT: Input tile height
    :param INPUT_WIDTH: Input tile width
    :param STEP_HEIGHT: Step size for overlap height, same as tile size if no overlap is needed. Smaller than tile size if need overlap.
    :param STEP_WIDTH: Step size for overlap width, same as tile size if no overlap is needed. Smaller than tile size if need overlap.
    :param TILE_SIZE: output tile size.
    :param image_format: Input image format
    :param print_frequency: Print frequency. After how many tiles the analysis should be printed.
    :param samples: Samples from each class for dataset generation. Ignore if want to generate whole dataset. (Dictionary)
    :param thresh: Image content threshold.
    :return: None
    '''
    train_dataset = []
    validation_dataset = []
    check_structure(INPUT_DIR, TILE_DIR, CLASS_TO_INDEX.keys())

    for clas in CLASS_TO_INDEX.keys():
        im_num = 0
        files = glob.glob1(f'{INPUT_DIR}{clas}/', "*."+image_format)
        #
        if samples:
            count  = min(len(files), samples[clas])
        else:
            count = len(files)
        print(f"count for class {clas}: {count}")
        total_tiles = (((INPUT_WIDTH - TILE_SIZE) // STEP_WIDTH + 1) * ((INPUT_HEIGHT - TILE_SIZE) // STEP_HEIGHT + 1)) * count
        print(f'Total Tiles for class{clas}: {total_tiles}')
        random_index = 0
        lst = set(random.sample(range(total_tiles), int(0.3 * total_tiles)))
        t, v = 0, 0
        start = timeit.default_timer()
        for i in range(count):
            img = cv2.imread(f'{INPUT_DIR}{clas}/{files[i]}')
            for p in range(0, INPUT_HEIGHT-TILE_SIZE+1, STEP_HEIGHT):
                for q in range(0, INPUT_WIDTH-TILE_SIZE+1, STEP_WIDTH):
                    im = img[p:p+TILE_SIZE, q:q+TILE_SIZE]
                    white_content = tile_content(im)
                    if white_content<thresh:
                        features = []
                        betti_features_rgb = generate_betti_input(im)
                        features = features + betti_features_rgb

                        betti_features_hsv = generate_betti_input(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
                        features = features + betti_features_hsv


                        features = features + generate_gabor_filters(im)
                        features = features + generate_gabor_filters(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))

                        features = features + generate_lbp_features(im)
                        features = features + generate_lbp_features(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))

                        if random_index in lst:
                            tile_name = f"{v}_{white_content}.jpg"
                            cv2.imwrite(f"{TILE_DIR}validation/{clas}/{tile_name}", im)
                            data = [f"{clas}/{tile_name}"] + features + [CLASS_TO_INDEX[clas]]
                            validation_dataset.append(data)
                            v += 1
                        else:
                            tile_name = f"{t}_{white_content}.jpg"
                            cv2.imwrite(f"{TILE_DIR}train/{clas}/{tile_name}", im)
                            data = [f"{clas}/{tile_name}"] + features + [CLASS_TO_INDEX[clas]]
                            train_dataset.append(data)
                            t += 1
                    random_index+=1
                    im_num += 1
                    if im_num%print_frequency == 0:
                        end = timeit.default_timer()
                        average = (end-start)/im_num
                        remaining = average * (total_tiles-im_num)
                        hrs, mins, secs = time_break(remaining)
                        hrs2, mins2, secs2 = time_break(end-start)
                        print(
                            f"\rClass {clas} done ({im_num}/{total_tiles} ): {round((im_num * 100) / total_tiles, 4)}%",
                            f"\tEstimated Time Remaining for this class : {hrs}:{mins}:{secs}",
                            f"\tAverage Time per Tile: {round(average,2)} seconds, Total Elapsed Time: {hrs2}:{mins2}:{secs2}",
                            end="")

        print(f"\nTiles generated for class: {clas} = {im_num}, train: {t}, val: {v}")

    with open(f'{TILE_DIR}train/data_2800.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(train_dataset)
    with open(f'{TILE_DIR}validation/data_2800.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(validation_dataset)


# Sample Function call for tile generation

generate_tiles(
    INPUT_DIR=r"D://All Datasets/SICAPv2/dataset/",
    TILE_DIR=r"D://All Datasets/SICAPv2/all_tiles_all_features_256x256/",
    CLASS_TO_INDEX={
        "NC": 0,
        "G3": 1,
        "G4": 2,
        "G5": 3,
    },
    INPUT_WIDTH=512,
    INPUT_HEIGHT=512,
    TILE_SIZE=256,
    STEP_WIDTH=256,
    STEP_HEIGHT=256,
    image_format="jpg",
    print_frequency=5,
    samples={
        "NC": 600,
        "G3": 200,
        "G4": 200,
        "G5": 200,
    },
    thresh=0.7
)