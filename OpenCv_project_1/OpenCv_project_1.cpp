#include <iostream>
#include"ImgPro.h"

using namespace std;
using namespace cv;

int main()
{

	string filename = "D:\\横州甘蔗地\\IMG_20230518_103112.jpg";
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

	auto result_OTSU = myImgPro.OTSU(MedianBlurImg);
	Mat OtsuImg = result_OTSU.first;
	float NonZeroPixelRatio = result_OTSU.second;


	/*
		Morphological operations are helpful for eliminating weeds and side branches, but also reduce crop details
	*/
	Mat MorphImg;
	int flag = 0;
	if (NonZeroPixelRatio > 0.06 && NonZeroPixelRatio <= 0.1) {
		MorphImg = myImgPro.MorphologicalOperation(OtsuImg, 3, 2, 1);
		flag = 1;
	}
	if (NonZeroPixelRatio > 0.1 && NonZeroPixelRatio < 0.2) {
		MorphImg = myImgPro.MorphologicalOperation(OtsuImg, 3, 5, 3);
		flag = 1;
	}
	if (NonZeroPixelRatio >= 0.2 && NonZeroPixelRatio < 0.3) {
		MorphImg = myImgPro.MorphologicalOperation(OtsuImg, 3, 9, 4);
		flag = 1;
	}
	if (NonZeroPixelRatio >= 0.3 && NonZeroPixelRatio < 0.4) {
		MorphImg = myImgPro.MorphologicalOperation(OtsuImg, 3, 11, 5);
		flag = 1;
	}
	if (NonZeroPixelRatio >= 0.4) {
		MorphImg = myImgPro.MorphologicalOperation(OtsuImg, 3, 14, 6);
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
	int centerX = result_VPFCX.second;


	///*
	//	Skeletonization process is characterized by a lengthy computational time
	//*/
	////CImgPro::Cluster skeleton_points;
	////Mat skeletonImg = myImgPro.skeletonization(ConnectImg, skeleton_points);
	//
	///*
	//int thresh = 10, k = 18;		//k为响应阈值的比例系数
	//CImgPro::Cluster susan_points;
	//Mat TempImg = myImgPro.My_SUSAN(ConnectImg, thresh, k, susan_points);
	//Mat SusanImg = TempImg.clone();
	//*/

	/*
		Using windows to extract features points, reducing data size
	*/
	CImgPro::Cluster reduce_points;
	Mat featureImg(ConnectImg.size(), CV_8UC1, Scalar(0));
	myImgPro.processImageWithWindow(ConnectImg, featureImg, reduce_points, 8, 8);


	////高斯混合模型的协方差矩阵类型(对应不同的数据形状)，球状、对角、完全、等方位
	////ml::EM::Types covarianceType = ml::EM::Types::COV_MAT_GENERIC; //这里使用完全协方差矩阵，适用于数据中存在明显的相关性的情况
	////vector<CImgPro::Cluster> cluster_points = myImgPro.Gaussian_Mixture_Model(points, 3, covarianceType);
	////Mat ClusterImg = myImgPro.ClusterPointsDrawing(ExGImage, cluster_points);
	//

	///*
	//	参数perCof很重要，需要调整以划分出指定的簇
	//*/
	////float perCof = 0.8;
	////int cluNum = 3;
	////vector<CImgPro::Cluster> points = myImgPro.Bisecting_Kmeans(susan_points, cluNum, perCof);
	////Mat ClusterImg = myImgPro.ClusterPointsDrawing(ExGImage, points);
	//

	///*
	//	averageD是聚类算法中的重要参数，决定了点与点间是否同属一类，太小会导致同属一类的点聚成两类，太大会聚到无关点
	//*/
	////int areaHeight = 128, areaWidth = 205, areaDegree = 128, areaExtent = 205;		//扫描窗口
	////vector<CImgPro::Cluster> points = myImgPro.Cluster_for_Ransac(MorphImg, areaHeight, areaWidth, areaDegree, areaExtent);
	////Mat ClusterImg = myImgPro.ClusterPointsDrawing(ExGImage, points);


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
	//myImgPro.SaveImg(filename, S_ClusterImg);



	////Thresholding segmentation of images
	//Mat HistogramImg;
	///*
	//if (CImgPro::NonZeroPixelRatio <= 0.1) {
	//	HistogramImg = myImgPro.verticalProjection(S_ClusterImg, maxPts, 0.8);
	//}
	//if (CImgPro::NonZeroPixelRatio < 0.2 && CImgPro::NonZeroPixelRatio > 0.1) {
	//	HistogramImg = myImgPro.verticalProjection(S_ClusterImg, maxPts, 0.8);
	//}
	//if (CImgPro::NonZeroPixelRatio < 0.3 && CImgPro::NonZeroPixelRatio >= 0.2) {
	//	HistogramImg = myImgPro.verticalProjection(S_ClusterImg, maxPts, 0.5);
	//}
	//if (CImgPro::NonZeroPixelRatio >= 0.3) {
	//	HistogramImg = myImgPro.verticalProjection(S_ClusterImg, maxPts, 0.4);
	//}
	//*/
	//double tsd = myImgPro.thresholdingSigmoid(NonZeroPixelRatio, -8.67, 0.354);//0.1-0.9  0.4-0.4
	////double tsd = myImgPro.thresholdingSigmoid(CImgPro::NonZeroPixelRatio, -4.977, 0.3185);//0.04-0.8  0.4-0.4
	//HistogramImg = myImgPro.verticalProjection(S_ClusterImg, second_cluster_points, tsd);
	//myImgPro.retainMainStem(second_cluster_points);
	//Mat MainStemImg = myImgPro.ClusterPointsDrawing(ExGImage, second_cluster_points);


	///*
	//	Second extraction
	//*/
	//CImgPro::Cluster final_points;
	//Mat ExtractImg(MainStemImg.size(), CV_8UC1, Scalar(0));
	//myImgPro.processImageWithWindow(MainStemImg, ExtractImg, final_points, 16, 32);


	///*
	//	fit line
	//*/
	//Mat RansacImg = inputImage.clone();
	////Mat RansacImg(ConnectImg.size(), CV_8UC3, Scalar(0, 0, 0));
	//if (NonZeroPixelRatio >= 0.1) {
	//	myImgPro.RANSAC(final_points, 0.155, RansacImg);
	//}
	//else
	//{
	//	myImgPro.RANSAC(final_points, 0.13, RansacImg);
	//}


	////Mat ProjectedImg = myImgPro.projectedImg(maxPtsImg, maxPts_temp, CImgPro::firstSlope);

	///*vector<CImgPro::Cluster> F_Pts;
	//F_Pts.push_back(final_points);
	//myImgPro.Hough_Line(F_Pts, RansacImg);*/


	//myImgPro.SaveImg(filename, RansacImg);


	/********************************************************************************************************************/

	//namedWindow("feature_Image", WINDOW_NORMAL);
	//moveWindow("feature_Image", 0, 0);		
	//imshow("feature_Image", featureImg);

	namedWindow("ExG_Image", WINDOW_NORMAL);
	moveWindow("ExG_Image", 0, 10);		
	imshow("ExG_Image", ExGImage);

	namedWindow("MedianBlur_Img", WINDOW_NORMAL);
	moveWindow("MedianBlur_Img",500, 0);		
	imshow("MedianBlur_Img", MedianBlurImg);

	namedWindow("OTSU_Img", WINDOW_NORMAL);
	moveWindow("OTSU_Img", 500, 500);
	imshow("OTSU_Img",OtsuImg);

	/*if (flag == 1) {
		namedWindow("Morph_Img", WINDOW_NORMAL);
		moveWindow("Morph_Img", 0, 1000);
		imshow("Morph_Img", MorphImg);
	}*/
	

	//namedWindow("Connect_Img", WINDOW_NORMAL);
	//moveWindow("Connect_Img", 0, 550);
	//imshow("Connect_Img", ConnectImg);

	//namedWindow("firstHistor_Img", WINDOW_NORMAL);
	//moveWindow("firstHistor_Img", 0, 300);
	//imshow("firstHistor_Img", firstHistorImg);

	namedWindow("F_Cluster_Img", WINDOW_NORMAL);
	moveWindow("F_Cluster_Img", 700, 0);
	imshow("F_Cluster_Img", F_ClusterImg);

	namedWindow("S_Cluster_Img", WINDOW_NORMAL);
	moveWindow("S_Cluster_Img", 800, 0);
	imshow("S_Cluster_Img", S_ClusterImg);

	/*namedWindow("Histogram_Img", WINDOW_NORMAL);
	moveWindow("Histogram_Img", 800, 400);
	imshow("Histogram_Img", HistogramImg);

	namedWindow("MainStem_Img", WINDOW_NORMAL);
	moveWindow("MainStem_Img", 900, 0);
	imshow("MainStem_Img", MainStemImg);

	namedWindow("Extract_Img", WINDOW_NORMAL);
	moveWindow("Extract_Img", 900, 200);
	imshow("Extract_Img", ExtractImg);*/

	//namedWindow("Skeleton_Img", WINDOW_NORMAL);
	//moveWindow("Skeleton_Img", 500, 700);
	//imshow("Skeleton_Img", skeletonImg);

	/*namedWindow("Ransac_Img", WINDOW_NORMAL);
	moveWindow("Ransac_Img", 500, 400);
	imshow("Ransac_Img", RansacImg);*/

	/*namedWindow("Projected_Img", WINDOW_NORMAL);
	moveWindow("Projected_Img", 400, 400);
	imshow("Projected_Img", ProjectedImg);*/

	waitKey(0);

	return 0;
}
