echo "DOWNLOADING DATASET"
python download_and_format_data.py

echo "ROTATING IMAGES"
python create_datasets.py

echo "RUNNING GPT4O INFERENCE"
python inference.py -nick GPT4o -ds small --run 0

echo "EVALUATING GPT4O PERFORMANCE"
python evaluate.py -nick GPT4o -ds small --runs 0 