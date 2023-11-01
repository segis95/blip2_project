root='/app/alexw/Experements/Imaginova/sergey_blip_dataset_creation/inference_results/'

for filename in "tqmkcowp_rosy-aardvark-52_BLIP2_INFERENCE_RESULTS.csv" \
                "qtdz14mc_desert-shape-51_BLIP2_INFERENCE_RESULTS.csv"
do
    python score_prompt_image_pairs.py $root/$filename
done