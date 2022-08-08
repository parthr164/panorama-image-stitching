import sys
import cv2
from itertools import combinations
import random
import numpy as np

# Finding matching points using ORB
def ORB(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create()
    
    keypoints_1, descriptors_1 = orb.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1,descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)
    
    matched_img = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)

    common_points = []
    for match in matches:
            x1y1 = keypoints_1[match.queryIdx].pt
            x2y2 = keypoints_2[match.trainIdx].pt
            feature = list(map(int, list(x1y1) + list(x2y2) + [match.distance]))
            common_points.append(feature)
    return common_points

# Find tranformation matrix
def trans_matrix(image_points):

    i_coord = []
    for img1_x1, img1_y1, img2_x1, img2_y1, dist in image_points:
        i_coord.append([img1_x1, img1_y1, 1, 0, 0, 0, -img2_x1*img1_x1, -img2_x1*img1_y1, -img2_x1])
        i_coord.append([0, 0, 0, img1_x1, img1_y1, 1, -img2_y1*img1_x1, -img2_y1*img1_y1, -img2_y1])
    i_coord = np.asarray(i_coord)

    # Taking SVD
    A, B, points = np.linalg.svd(i_coord)
    
    temp_mat = points[-1, :] / points[-1, -1]
    M = temp_mat.reshape(3, 3)

    return M

# Implementation of RANSAC
def RANSAC(common_points, n_best_feat = 40, max_dist= 4):
    feature_subsets = list(combinations(common_points[:n_best_feat], 4))
    best_count = 0
    best_points = 0

    for feature_point in feature_subsets[:5000]:
        inlier_points = []
        inlier_count = 0
        transformation_mat = trans_matrix(feature_point)
        
        for point in common_points[:n_best_feat]:
            initial_point = np.array(point[:2] + [1]).reshape(-1,1)
            final_point = np.array(point[2:4] + [1]).reshape(-1,1)
            
            mapped_point = np.dot(transformation_mat, initial_point)
            mapped_point/= mapped_point[-1,-1]
            distance = np.linalg.norm(mapped_point - final_point)
            
            if distance < max_dist:
                inlier_count+=1
                inlier_points.append(point)
            
        if inlier_count > best_count:
            best_points = inlier_points
            best_count = inlier_count
            
    return trans_matrix(best_points)

# Warping Image
def image_transform(im_arr,M):

    point = np.array([[k for k in range(im_arr.shape[1]+1) for i in range(im_arr.shape[0]+1)],
                  [k for i in range(im_arr.shape[1]+1) for k in range(im_arr.shape[0]+1)],
                  [1 for i in range((im_arr.shape[1]+1)*(im_arr.shape[0]+1))]])
    
    a,b = point[:,0],point[:,-1]
    c,d = np.array([b[0],0,1]),np.array([0,b[1],1])
    boundary = np.array([a,c,b,d]).T
    
    boundary_tr = M @ boundary
    boundary_tr = boundary_tr/boundary_tr[2,:]
    
    col_min,col_max = min(min(boundary_tr[0,:]),0),max(max(boundary_tr[0,:]),im_arr.shape[1])
    row_min,row_max = min(min(boundary_tr[1,:]),0),max(max(boundary_tr[1,:]),im_arr.shape[0])
    
    col_offset = 0 - col_min
    row_offset = 0 - row_min

    width,height = int(round(col_max-col_min)),int(round(row_max-row_min))
    new_arr = np.zeros((height,width,3))
    
    M[:,2] = M @ np.array([-int(col_min),-int(row_min),1]).T
    
    point = np.array([[k for k in range(new_arr.shape[1]) for i in range(new_arr.shape[0])],
                  [k for i in range(new_arr.shape[1]) for k in range(new_arr.shape[0])],
                  [1 for i in range((new_arr.shape[1])*(new_arr.shape[0]))]])
    
    point_tr = np.linalg.inv(M) @ point
    point_tr = point_tr/point_tr[2,:]
    
    k=0
    for col in range(new_arr.shape[1]):
        for row in range(new_arr.shape[0]):

            b,a = point_tr[:,k][0] - np.floor(point_tr[:,k][0]),point_tr[:,k][1] - np.floor(point_tr[:,k][1])
            b_c,a_c = np.int64(np.ceil(point_tr[:,k][0])),np.int64(np.ceil(point_tr[:,k][1]))
            b_f,a_f = np.int64(np.floor(point_tr[:,k][0])),np.int64(np.floor(point_tr[:,k][1]))

            k+=1

            if (a_c>=0 and a_c<im_arr.shape[0] and a_f>=0 and a_f<im_arr.shape[0] and b_c>=0 and b_c<im_arr.shape[1] and b_f>=0 and b_f<im_arr.shape[1] ):

                new_arr[row,col,:] = (1-b)*(1-a)*im_arr[a_f,b_f] +(1-b)*(a)*im_arr[a_c,b_f]+b*(1-a)*im_arr[a_f,b_c] +b*a*im_arr[a_c,b_c]
    
    return int(row_offset), int(col_offset), new_arr

def create_panorama(file_path, row_offset, col_offset, img1, img2):
    new_img = np.zeros((img2.shape[0],img2.shape[1],img2.shape[2]))
    new_img[row_offset:row_offset+img1.shape[0], col_offset:col_offset+img1.shape[1]] = img1
    new_img = new_img.astype(int)

    for j in range(img2.shape[1]):
        for i in range(img2.shape[0]):
            if not np.all((img2[i,j] == 0)) and np.all((new_img[i,j] == 0)):
                new_img[i,j] = img2[i,j]
    cv2.imwrite(file_path, new_img)

if __name__ == '__main__':
    
      im1_name = sys.argv[1]
      im2_name = sys.argv[2]
      file_name = sys.argv[3]

      img1 = cv2.imread(im1_name)
      img2 = cv2.imread(im2_name)

      common_points = ORB(img1, img2)
      print("Common points found!")
      transformation_matrix = RANSAC(common_points)
      row_offset, col_offset, warped_img = image_transform(img1, transformation_matrix)
      print("Image has been warped!")
      create_panorama(file_name, row_offset, col_offset, img2, warped_img)
      print("Output file is created successfully!")