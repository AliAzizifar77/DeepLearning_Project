echo Start exclusive representation training
data_base_folder="/teamspace/studios/this_studio/dataset"
xp_name="Exclusive_representation_training"
conf_path="conf/exclusive_conf.yaml"
trained_enc_x_path="/teamspace/studios/this_studio/Learning-Disentangled-Representations-via-Mutual-Information-Estimation/mlruns/287907110944572018/beec59011ec245fdb15b3688e4493207/artifacts/sh_encoder_x/state_dict.pth"
trained_enc_y_path="/teamspace/studios/this_studio/Learning-Disentangled-Representations-via-Mutual-Information-Estimation/mlruns/287907110944572018/beec59011ec245fdb15b3688e4493207/artifacts/sh_encoder_y/state_dict.pth"

PYTHONPATH=$PYTHONPATH:src python src/edim_train.py --xp_name $xp_name --conf_path $conf_path --data_base_folder $data_base_folder --trained_enc_x_path $trained_enc_x_path --trained_enc_y_path $trained_enc_y_path
