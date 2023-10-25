#include <iostream>
#include"ImgPro.h"

using namespace std;
using namespace cv;

int main()
{

	//string filename = "D:\\横州甘蔗地\\IMG_20230518_112045.jpg";//image path
	//string filename = "E:\\泛化能力验证数据集\\20230815中秋玉米苗\\IMG_20230815_095811.jpg";
	//string filename = "E:\\泛化能力验证数据集\\20230803东高玉米苗\\IMG_20230813_180115.jpg";
	string filename = "E:\\泛化能力验证数据集\\20230813东高水稻苗\\IMG_20230813_181840.jpg";
	Mat inputImage = imread(filename);
	CImgPro::imgCols = inputImage.cols;
	CImgPro::imgRows = inputImage.rows;

	CImgPro myImgPro;

	Mat ExGImage(inputImage.size(), CV_8UC1);
	myImgPro.NormalizedExG(inputImage, ExGImage);

	/*
		Median filtering is more effective than Gaussian filtering in dealing with salt-and-pepper noise
	*/
	int MedianBlur_kernel_size = 5;		
	Mat MedianBlurImg = myImgPro.MedianBlur(ExGImage, MedianBlur_kernel_size);
	myImgPro.SaveImg(filename, MedianBlurImg);

	auto result_OTSU = myImgPro.OTSU(MedianBlurImg);
	Mat temp = result_OTSU.first;
	Mat OtsuImg = temp.clone();
	float NonZeroPixelRatio = result_OTSU.second;


	/*
		Morphological operations are helpful for eliminating weeds and side branches, but also reduce crop details
	*/
	Mat MorphImg;
	int flag = 0;
	//if (NonZeroPixelRatio > 0.06 && NonZeroPixelRatio <= 0.1) {
	//	MorphImg = myImgPro.MorphologicalOperation(OtsuImg, 3, 2, 1);
	//	flag = 1;
	//}
	//if (NonZeroPixelRatio > 0.1 && NonZeroPixelRatio < 0.2) {
	//	MorphImg = myImgPro.MorphologicalOperation(OtsuImg, 3, 5, 3);
	//	flag = 1;
	//}
	//if (NonZeroPixelRatio >= 0.2 && NonZeroPixelRatio < 0.3) {
	//	MorphImg = myImgPro.MorphologicalOperation(OtsuImg, 3, 9, 4);
	//	flag = 1;
	//}
	//if (NonZeroPixelRatio >= 0.3 && NonZeroPixelRatio < 0.4) {
	//	MorphImg = myImgPro.MorphologicalOperation(OtsuImg, 3, 11, 5);
	//	flag = 1;
	//}
	//if (NonZeroPixelRatio >= 0.4) {
	//	MorphImg = myImgPro.MorphologicalOperation(OtsuImg, 3, 14, 6);
	//	flag = 1;
	//}

	auto result_open = myImgPro.NZPR_to_Erosion_Dilation(NonZeroPixelRatio);
	if (result_open.first && result_open.second) {
		MorphImg = myImgPro.MorphologicalOperation(OtsuImg, 3, result_open.first, result_open.second);
		flag = 1;
	}
	

	/*
		The eight-connected algorithm can be employed to further eliminate noise and minor connected components
	*/
	pair<Mat, vector<int>> result_EC;
	if (flag == 0) {
		result_EC = myImgPro.EightConnectivity(OtsuImg, 0.7);
	}
	else
	{
		result_EC = myImgPro.EightConnectivity(MorphImg, 0.7);
	}
	Mat ConnectImg = result_EC.first;



	//Calculate the x-coordinate of the center row baseline within the crop based on the histogram analysis
	auto result_VPFCX = myImgPro.verticalProjectionForCenterX(result_EC.second);
	Mat firstHistorImg = result_VPFCX.first;
	int centerX = result_VPFCX.second;//baseline


	/*
		Using windows to extract features points, reducing data size
	*/
	CImgPro::Cluster reduce_points;
	Mat featureImg(ConnectImg.size(), CV_8UC1, Scalar(0));
	myImgPro.processImageWithWindow(ConnectImg, featureImg, reduce_points, 8, 8, 1);



	/*
		Density clustering-based
	*/
	vector<CImgPro::Cluster> first_cluster_points = myImgPro.firstClusterBaseOnDbscan(reduce_points, 110, 50);
	float cof = 0.65;//0.4
	vector<CImgPro::Cluster> second_cluster_points;
	do
	{
		second_cluster_points = myImgPro.secondClusterBaseOnCenterX(first_cluster_points, centerX, cof);
		cof += 0.05;
	} while (second_cluster_points.size() == 0);
	Mat F_ClusterImg = myImgPro.ClusterPointsDrawing(ExGImage, first_cluster_points);
	Mat S_ClusterImg = myImgPro.ClusterPointsDrawing(ExGImage, second_cluster_points);


	//Thresholding segmentation of images
	Mat HistogramImg;
	/*
	if (CImgPro::NonZeroPixelRatio <= 0.1) {
		HistogramImg = myImgPro.verticalProjection(S_ClusterImg, maxPts, 0.8);
	}
	if (CImgPro::NonZeroPixelRatio < 0.2 && CImgPro::NonZeroPixelRatio > 0.1) {
		HistogramImg = myImgPro.verticalProjection(S_ClusterImg, maxPts, 0.8);
	}
	if (CImgPro::NonZeroPixelRatio < 0.3 && CImgPro::NonZeroPixelRatio >= 0.2) {
		HistogramImg = myImgPro.verticalProjection(S_ClusterImg, maxPts, 0.5);
	}
	if (CImgPro::NonZeroPixelRatio >= 0.3) {
		HistogramImg = myImgPro.verticalProjection(S_ClusterImg, maxPts, 0.4);
	}
	*/
	double tsd = myImgPro.thresholdingSigmoid(NonZeroPixelRatio, -8.67, 0.354);//0.1-0.9  0.4-0.4
	//double tsd = myImgPro.thresholdingSigmoid(CImgPro::NonZeroPixelRatio, -4.977, 0.3185);//0.04-0.8  0.4-0.4
	HistogramImg = myImgPro.verticalProjection(S_ClusterImg, second_cluster_points, tsd);
	myImgPro.retainMainStem(second_cluster_points);
	Mat MainStemImg = myImgPro.ClusterPointsDrawing(ExGImage, second_cluster_points);


	/*
		Second extraction
	*/
	CImgPro::Cluster final_points;
	Mat ExtractImg(MainStemImg.size(), CV_8UC3, Scalar(255,255,255));
	myImgPro.processImageWithWindow(MainStemImg, ExtractImg, final_points, 16, 32, 2);


	/*
		fit line
	*/
	Mat RansacImg = inputImage.clone();
	//Mat RansacImg(ConnectImg.size(), CV_8UC3, Scalar(0, 0, 0));
	if (NonZeroPixelRatio >= 0.1) {
		myImgPro.RANSAC(final_points, 0.155, RansacImg);
	}
	else
	{
		myImgPro.RANSAC(final_points, 0.13, RansacImg);
	}
	//myImgPro.SaveImg(filename, RansacImg);



	/********************************************************************************************************************/

	/*    Irrelevant code, kept for future research convenience.   */

	/*
		Skeletonization process is characterized by a lengthy computational time
	*/
	//CImgPro::Cluster skeleton_points;
	//Mat skeletonImg = myImgPro.skeletonization(ConnectImg, skeleton_points);
	/*
	int thresh = 10, k = 18;
	CImgPro::Cluster susan_points;
	Mat TempImg = myImgPro.My_SUSAN(ConnectImg, thresh, k, susan_points);
	Mat SusanImg = TempImg.clone();
	*/

	/*
	  The covariance matrix types in a Gaussian Mixture Model (GMM) that correspond to different data shapes
	  Here, using a full covariance matrix, which is suitable for situations where there is clear correlation present in the data
	*/
	//ml::EM::Types covarianceType = ml::EM::Types::COV_MAT_GENERIC; 
	//vector<CImgPro::Cluster> cluster_points = myImgPro.Gaussian_Mixture_Model(points, 3, covarianceType);
	//Mat ClusterImg = myImgPro.ClusterPointsDrawing(ExGImage, cluster_points);


	//float perCof = 0.8;
	//int cluNum = 3;
	//vector<CImgPro::Cluster> points = myImgPro.Bisecting_Kmeans(susan_points, cluNum, perCof);
	//Mat ClusterImg = myImgPro.ClusterPointsDrawing(ExGImage, points);


	//int areaHeight = 128, areaWidth = 205, areaDegree = 128, areaExtent = 205;		
	//vector<CImgPro::Cluster> points = myImgPro.Cluster_for_Ransac(MorphImg, areaHeight, areaWidth, areaDegree, areaExtent);
	//Mat ClusterImg = myImgPro.ClusterPointsDrawing(ExGImage, points);

	/********************************************************************************************************************/

	myImgPro.ShowImg(featureImg, "feature_Image", 0, 0);
	myImgPro.ShowImg(ExGImage, "ExG_Image", 0, 10);
	myImgPro.ShowImg(MedianBlurImg, "MedianBlur_Img", 500, 0);
	myImgPro.ShowImg(temp, "OTSU_Img", 500, 500);
	if (flag == 1) {
		myImgPro.ShowImg(MorphImg, "Morph_Img", 0, 510);
	}
	myImgPro.ShowImg(ConnectImg, "Connect_Img", 0, 550);
	myImgPro.ShowImg(firstHistorImg, "firstHistor_Img", 0, 300);
	myImgPro.ShowImg(F_ClusterImg, "F_Cluster_Img", 700, 0);
	myImgPro.ShowImg(S_ClusterImg, "S_Cluster_Img", 800, 0);
	myImgPro.ShowImg(HistogramImg, "Histogram_Img", 800, 400);
	myImgPro.ShowImg(MainStemImg, "MainStem_Img", 900, 0);
	myImgPro.ShowImg(ExtractImg, "Extract_Img", 900, 200);

	//myImgPro.ShowImg(skeletonImg, "Skeleton_Img", 500, 700);
	//namedWindow("Skeleton_Img", WINDOW_NORMAL);
	//moveWindow("Skeleton_Img", 500, 700);
	//imshow("Skeleton_Img", skeletonImg);

	myImgPro.ShowImg(RansacImg, "Ransac_Img", 500, 400);

	waitKey(0);

	return 0;
}
