# IMAGE_DIRS=(
#         '/app/alexw/Experements/Imaginova/sergey_blip_dataset_creation/data/toy_data/images1_infer'
#         )

#"${IMAGE_DIRS[@]}"
# "/app/alexw/Experements/Imaginova/sergey_blip_dataset_creation/data/toy_data/images1_infer" \
#         "/app/alexw/Experements/Imaginova/sergey_blip_dataset_creation/data/toy_data/images2_infer"
for input_dir in "/app/alexw/Experements/Imaginova/DataPreparation/hentaiii"
do
    accelerate launch scripts/predict_llava.py ++image_folders_list="[$input_dir]"
done

# "/app/alexw/Experements/Imaginova/DataPreparation/a_lot_of_nudes" \
# "/app/alexw/Experements/Imaginova/DataPreparation/aestetic_128ch" \
# "/app/alexw/Experements/Imaginova/DataPreparation/dgavrikovphoto" \
# "/app/alexw/Experements/Imaginova/DataPreparation/0nitebenesvetyat" \
# "/app/alexw/Experements/Imaginova/DataPreparation/kitsuneai" \
# "/app/alexw/Experements/Imaginova/DataPreparation/mental.mental" \
# "/app/alexw/Experements/Imaginova/DataPreparation/nyashky" \
# "/app/alexw/Experements/Imaginova/DataPreparation/pexels" \
# "/app/alexw/Experements/Imaginova/DataPreparation/pexels_clean" \
# "/app/alexw/Experements/Imaginova/DataPreparation/photos" \
# "/app/alexw/Experements/Imaginova/DataPreparation/pixbay" \
# "/app/alexw/Experements/Imaginova/DataPreparation/public65484616" \
# "/app/alexw/Experements/Imaginova/DataPreparation/rrtelochki"
