#include <iostream>
#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;
const char* keys =
        "{ help h |                          | Print help message. }"
        "{ input1 | ../data/box.png          | Path to input image 1. }"
        "{ input2 | ../data/box_in_scene.png | Path to input image 2. }";
int main( int argc, char* argv[] )
{
    CommandLineParser parser( argc, argv, keys );

    cv::Mat camMatrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distCoeffs = cv::Mat::zeros(1, 5, CV_64F);
// tango at 960p
        //  camMatrix.at<double>(0, 0) = 867.601196289;
        //  camMatrix.at<double>(1, 1) = 867.601196289;
        //  camMatrix.at<double>(0, 2) = 482.6857910;
        //  camMatrix.at<double>(1, 2) = 268.82540;
        //  camMatrix.at<double>(2, 2) = 1;
// pupil at 720p
        camMatrix.at<double>(0, 0) = 1200;
        camMatrix.at<double>(1, 1) = 1200;
        camMatrix.at<double>(0, 2) = 1280/2;
        camMatrix.at<double>(1, 2) = 720/2;
        camMatrix.at<double>(2, 2) = 1;
        
        distCoeffs.at<double>(0, 0) = -0.63037088;
        distCoeffs.at<double>(0, 1) =  0.17767048;
        distCoeffs.at<double>(0, 2) = -0.00489945;
        distCoeffs.at<double>(0, 3) = -0.00192122;
        distCoeffs.at<double>(0, 4) =  0.1757496;





    Mat img_object = imread( parser.get<String>("input1"), IMREAD_GRAYSCALE );

    Mat img_scene_raw = imread( parser.get<String>("input2"), IMREAD_GRAYSCALE );
    Mat img_scene;
    cv::undistort(img_scene_raw, img_scene, camMatrix, distCoeffs);

    if ( img_object.empty() || img_scene.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        parser.printMessage();
        return -1;
    }
    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 400;
    //Ptr<SURF> detector = SURF::create( minHessian );
    Ptr<SIFT> detector = SIFT::create();
    std::vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;
    detector->detectAndCompute( img_object, noArray(), keypoints_object, descriptors_object );
    detector->detectAndCompute( img_scene, noArray(), keypoints_scene, descriptors_scene );
    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors_object, descriptors_scene, knn_matches, 2 );
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.75f;
    std::vector<DMatch> good_matches;
    std::vector<DMatch> inliers;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    //-- Draw matches
    Mat img_matches;
    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    imshow("Good Matches & Object detection", img_matches );
    waitKey(0);             
    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }
    Mat mask;
    Mat H = findHomography( obj, scene, RANSAC, 3.0 , mask );
    int inlierCount = 0;
    for( size_t i = 0; i < good_matches.size(); i++ ) { 
        if (mask.at<char>(i, 0) > 0) {
            inliers.push_back(good_matches[i]);
            inlierCount++;
          }
    }
    std::cout << "homography computed based on " << inlierCount << " inliers" << std::endl;
    Mat img_inliers;
    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene, inliers, img_inliers, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  //  std::cout << "mask: /n" << mask <<std::endl;
    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f( (float)img_object.cols, 0 );
    obj_corners[2] = Point2f( (float)img_object.cols, (float)img_object.rows );
    obj_corners[3] = Point2f( 0, (float)img_object.rows );
    std::vector<Point2f> scene_corners(4);
    perspectiveTransform( obj_corners, scene_corners, H);
    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( img_inliers, scene_corners[0] + Point2f((float)img_object.cols, 0),
          scene_corners[1] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4 );
    line( img_inliers, scene_corners[1] + Point2f((float)img_object.cols, 0),
          scene_corners[2] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_inliers, scene_corners[2] + Point2f((float)img_object.cols, 0),
          scene_corners[3] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_inliers, scene_corners[3] + Point2f((float)img_object.cols, 0),
          scene_corners[0] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    //-- Show detected matches
    imshow("Good Matches & Object detection", img_inliers );
    waitKey();
    return 0;
}
#else
int main()
{
    std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif