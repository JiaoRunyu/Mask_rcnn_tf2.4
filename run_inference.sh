#python run_inference.py \
#	-i /media/DATA/VAD_datasets/taiwan_sa/testing/frames \
#	-o /media/DATA/VAD_datasets/taiwan_sa/testing/mask_rcnn_detections \
#	--for_deepsort \
#	--image_shape 1280 720 3 \
#	-g=0 
#python run_inference.py \
#	-i /media/DATA/AnAnAccident_Detection_Dataset/frames \
#	-o /media/DATA/AnAnAccident_Detection_Dataset/mask_rcnn_detections \
#	--for_deepsort \
#	--image_shape 1280 720 3 \
#	-g=0

#python run_inference.py \
#	-i /home/vad/VAD/Utile/Mask_RCNN_TF2.3_final/frames \
#	-o /home/vad/VAD/Utile/Mask_RCNN_TF2.3_final/mask_rcnn_detections \
#	--for_deepsort \
#	--image_shape 1280 720 3 \
#	-g=0

#For KITTI
python run_inference.py \
	-i /home/vad/VAD/DATA/KITTI_Raw/2011_09_29/2011_09_29_drive_0004_sync/image_02/detection_for_deepsort/frames/ \
	-o /home/vad/VAD/DATA/KITTI_Raw/2011_09_29/2011_09_29_drive_0004_sync/image_02/detection_for_deepsort \
	--for_deepsort \
	--image_shape 1280 384 3 \
	-g=0