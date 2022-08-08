# panorama-image-stitching

This is my attempt at automatic image matching and transformations of 2 images to create a panorama.

This task involes 4 distinct parts:
1. Extract interest points from each image
2. RANSAC to figure best relative transformation
3. Transform images to common coordinate system
4. Blend the images together

### Interest Points

First, the feature points are extracted by using ORB implementation from OpenCV. Then, BFMatcher is used to extract the top features. The resultant set of pairs of points are sorted by distance and the top 50 best mactched points are used.

### RANSAC
The RANSAC algorithm is implemented as follows:
1. The extracted common points are arranged in combinations of 4 points in order to use them as our hypothesis.
2. Iterate through the sets of 4 point pairs created and for each set (hypothesis) calculate the homography matrix.
3. Using calculated homography matrix calculate the mapped points for the remaining points and compare with the points extracted. If the mapped point falls within a threshold of a certain number of pixels with the points found through ORB, then it is counted as an inlier. 
4. We keep count of total number inliers using a particular hypothesis. We also keep track of the best inliers and hypothesis.
5. Once we have iterated through all our hypothesis we calulate the best homography matrix using the best set of inlier points that we have stored.

### Image Tranformation

Once the best homography matrix is calculated, the warped image transformation of the second image is calculated in the coordinate system of the first image.

### Blending

Here the two images, the original image and the warped image need to be stitched together. The algorithm first places the original image in the canvas, and then places pixels from the warped image only where the pixel value is 0 in the final image. This way a fully stitched image is obtained. 

To try this yourself type this line in your command line:

```
python pano.py image_1.jpg image_2.jpg output.jpg
```

Here are some examples:

Image 1           |  Image 2           | Stitched Image           
:-------------------------:|:-------------------------:|:-------------------------:
![test1](https://user-images.githubusercontent.com/55157425/183475879-a19994ca-7ca8-4701-b613-8235a7b8ea22.png) | ![test2](https://user-images.githubusercontent.com/55157425/183475907-69dbb219-f9e4-478f-8836-787a1839cb63.png) | ![output](https://user-images.githubusercontent.com/55157425/183475951-4bf58ca8-a4a7-45b6-9b08-e89e8d070541.jpg)

Image 1           |  Image 2           | Stitched Image           
:-------------------------:|:-------------------------:|:-------------------------:
![room1](https://user-images.githubusercontent.com/55157425/183476035-28cafe00-972e-4f90-8660-3fec73799fc6.png) | ![room2](https://user-images.githubusercontent.com/55157425/183476054-78feaed8-f429-4d4f-ad1e-049a44930d59.png) | ![output](https://user-images.githubusercontent.com/55157425/183476654-71bc3752-71d6-42c9-8c8d-c7c97f2fe904.jpg)



