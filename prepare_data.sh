mkdir ./data
wget -O data/all_ids.tar.gz https://umich.box.com/shared/static/4m1mr6aud793gwi7jn266t32w83ml25l.gz
wget -O data/all_viewdata.tar.gz https://umich.box.com/shared/static/ckvihxh4berjzcgd3s8aiu87aurv3gms.gz
wget -O data/all_voxdata.tar.gz https://umich.box.com/shared/static/bwyx8qsby2f38ju1ybcrp50a1q9uzenu.gz
tar xf data/all_ids.tar.gz -C data/
echo "It may take a while, please be patient."
tar xf data/all_viewdata.tar.gz -C data/
tar xf data/all_voxdata.tar.gz -C data/
