```bash
cd msmarco_document

sh download.sh

cd ..

sh ms_top300k_process1.sh

cd qg

sh qg.sh 4 #The number of your gpu

cd ..

sh ms_top300k_process2.sh 4 #The number of your gpu

```
The generated data can be downloaded from [here](https://drive.google.com/drive/folders/1FMZcCXNRMq4ZbutMkANT0RG1Ga3clnte?usp=share_link) and should be put into `./dataset/top300k`

