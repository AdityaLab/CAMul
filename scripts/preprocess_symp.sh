set -e
python ./symp_extract/create_df.py
mkdir -p ./data/ili_data
wget -O ./data/ili_data/ILINet.csv https://raw.githubusercontent.com/AdityaLab/EpiFNP/master/data/ILINet.csv
python ./symp_extract/combine.py