set -e

echo "download MSMARCO document data"

wget -nc --no-check-certificate https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz
gunzip ./msmarco-docs.tsv.gz

wget -nc --no-check-certificate https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz
gunzip ./msmarco-doctrain-queries.tsv.gz

wget -nc --no-check-certificate https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz
gunzip ./msmarco-doctrain-qrels.tsv.gz

wget -nc --no-check-certificate https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz
gunzip ./msmarco-docdev-qrels.tsv.gz

wget -nc --no-check-certificate https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz
gunzip ./msmarco-docdev-queries.tsv.gz

echo "Done!"
