import pandas as pd
import os
from PIL import Image
import argparse
from tqdm import tqdm

import hpsv2

def parse_args():
    parser = argparse.ArgumentParser(description='Scores <prompt, image> Pairs From .csv')
    parser.add_argument('path_to_csv', help='.csv file, to score pairs from')
    parser.add_argument('--hps_root', help='Epochs to train the model', default="/app/alexw/Experements/Imaginova/sergey_blip_dataset_creation/HPSv2")
    
    return parser.parse_args()
    
def main():
    args = parse_args()
    os.environ['HPS_ROOT'] = args.hps_root
    
    df = pd.read_csv(args.path_to_csv, index_col=0)
    assert set(df.columns) == {'paths', 'caption'}
    
    items = []
    print(f'Scoring {args.path_to_csv}...')
    for path, capt in tqdm(list(zip(df.paths, df.caption))):
        image = Image.open(path)
        score = hpsv2.score(image, capt)
        items.append((score[0], path, capt))
        image.close()

    items = sorted(items, reverse=True)

    df_new = pd.DataFrame(columns=["paths", "caption", "score"])

    df_new.score = [i[0] for i in items]
    df_new.paths = [i[1] for i in items]
    df_new.caption = [i[2] for i in items]
    
    path_to_new_csv_folder = os.path.join(os.path.dirname(args.path_to_csv), 'scored')
    if not os.path.exists(path_to_new_csv_folder):
        os.makedirs(path_to_new_csv_folder)
    
    df_new.to_csv(os.path.join(path_to_new_csv_folder, os.path.basename(args.path_to_csv)))
    
    
    
if __name__== "__main__":
    main()