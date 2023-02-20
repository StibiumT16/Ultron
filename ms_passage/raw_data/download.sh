set -e

echo "download MSMARCO passage data"

echo "start downloading corpus, this may take some time depending on the network"
wget -nc --no-check-certificate https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
tar --skip-old-files -zxvf collectionandqueries.tar.gz -C ./

echo "Done"

