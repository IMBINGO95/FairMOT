import cv2
def box_rectify(new_reference_point,channel):
	if len(new_reference_point) != 0:
		print('box of channel : {} exists'.format(channel))
		delet_x = new_reference_point[0] - reference_point[0]
		delet_y = new_reference_point[1] - reference_point[1]
		img_point_of_other = (img_point_of_other[0] + delet_x, img_point_of_other[1] + delet_y)
		sub_imgs.append(sub_img)
		if args.visualization in ['both', 'small']:
			cv2.imwrite(os.path.join(visualization_dir, '{}_{}' + other_channel + '_sub.jpg').format('sub', img_count_),
			            sub_img)