# get files from server
sudo gsutil cp gs://vid2viddl/test_04/test_04_set_neut.zip dataset/face/test_04/neut/test_04_set_neut.zip 

#unzip
sudo unzip dataset/face/test_04/neut/test_04_set_neut.zip 

# test with real images
python test.py --name test_04_masked_256 \
--dataroot test_img_04/carolina/ --dataset_mode face \
--input_nc 15 --loadSize 256 --how_many 6000 --ntest 8 \
--use_real_img   --results_dir test_04_masked_256_f/

# test with no first image
python test.py --name test_04_masked_256 \
--dataroot test_img_04/carolina/ --dataset_mode face \
--input_nc 15 --loadSize 256 --how_many 600 --ntest 8 \
--no_first_img  --use_single_G --results_dir test_04_masked_256_nf/


# test with real images
python test.py --name test_04_masked \
--dataroot datasets/face/carolina/ --dataset_mode face \
--input_nc 15 --loadSize 256 --how_many 600 --ntest 8 \
--use_real_img  --use_single_G --results_dir test_04_masked_512_f/

# test with no first image
python test.py --name test_04_masked \
--dataroot datasets/face/carolina/--dataset_mode face \
--input_nc 15 --loadSize 256 --how_many 600 --ntest 8 \
--no_first_img  --use_single_G --results_dir test_04_masked_512_nf/