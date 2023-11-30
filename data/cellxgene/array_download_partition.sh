#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=1-9
#SBATCH --mem=48G
#SBATCH --qos=nopreemption
#SBATCH -p cpu

echo  "------- runnung array_download_partition"

SCRIPT_PATH="/Users/ayudzin/PycharmProjects/scGPT/data/cellxgene"
INDEX_PATH="/Users/ayudzin/PycharmProjects/scGPT/data/cellxgene/index"
QUERY_PATH="/Users/ayudzin/PycharmProjects/scGPT/data/cellxgene/query_list.txt"
DATA_PATH="/Users/ayudzin/PycharmProjects/scGPT/data/cellxgene/expression_data"

SLURM_ARRAY_TASK_ID="2"

cd $DATA_PATH

query_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$QUERY_PATH")

echo " Downloading:"
echo "1 - query_name = ${query_name}"
echo "2 - index_path = ${INDEX_PATH}"
echo "3 - data_path = ${DATA_PATH}"

"${SCRIPT_PATH}/download_partition.sh" "${query_name}" "${INDEX_PATH}" "${DATA_PATH}"