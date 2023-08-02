#include <iostream>

#include"ImgPro.h"

using namespace std;
using namespace cv;

int main()
{

	string filename = "D:\\横州甘蔗地\\IMG_20230518_111417.jpg";
	Mat inputImage = imread(filename);

	//// 获取图像尺寸
	//int width = inputImage.cols;
	//int height = inputImage.rows;
	//// 计算裁剪区域的高度
	//int cropHeight = height / 10;
	//// 计算裁剪后图像的起始位置
	//int startY = cropHeight;
	//// 裁剪图像
	//cv::Rect roi(0, startY, width, height - cropHeight);
	//cv::Mat croppedImage = inputImage(roi);

	CImgPro myImgPro;

	Mat ExGImage(inputImage.size(), CV_8UC1);

	myImgPro.NormalizedExG(inputImage, ExGImage);	

	/*
		中值滤波比高斯滤波在处理椒盐噪声上效果更好
	*/
	int MedianBlur_kernel_size = 5;		//卷积核大小
	Mat MedianBlurImg = myImgPro.MedianBlur(ExGImage, MedianBlur_kernel_size);

	
	Mat OtsuImg = myImgPro.OTSU(MedianBlurImg);

	/*
		形态学操作对于消除细小杂草有帮助，但同时也会减少作物细节
	*/
	Mat MorphImg;
	int flag = 0;
	if (CImgPro::NonZeroPixelRatio > 0.1) {
		MorphImg = myImgPro.MorphologicalOperation(OtsuImg, 3, 5);
		flag = 1;
	}


	/*
		使用八连通筛选算法可有效去除噪声和细小杂草，但可进一步优化参数保留更多作物细节
	*/
	float cof = 0.7;
	Mat ConnectImg;
	if (flag == 0) {
		ConnectImg = myImgPro.EightConnectivity(OtsuImg, cof);
	}
	else
	{
		ConnectImg = myImgPro.EightConnectivity(MorphImg, cof);
	}
	
	

	/*
		骨架化，耗费时间长
	*/
	//CImgPro::Cluster skeleton_points;
	//Mat skeletonImg = myImgPro.skeletonization(ConnectImg, skeleton_points);


	/*
		
	*/
	//int thresh = 10, k = 18;		//k为响应阈值的比例系数
	//CImgPro::Cluster susan_points;
	//Mat TempImg = myImgPro.My_SUSAN(ConnectImg, thresh, k, susan_points);
	//Mat SusanImg = TempImg.clone();
	///*int x = FeatureImg.cols, y = FeatureImg.rows;*/
	///*cout << "size:" << x <<"x"<< y << endl;*/


	/*
		目前看使用长条窗口去提取作物特征点比较好，能保留作物主要特征同时减少数据量
	*/
	CImgPro::Cluster reduce_points;
	Mat featureImg(ConnectImg.size(), CV_8UC1, Scalar(0));
	myImgPro.processImageWithWindow(ConnectImg, featureImg, reduce_points, 8, 8);


	/*
	myImgPro.averageCoordinates(susan_points);
	vector< CImgPro::Cluster> points;
	points.push_back(susan_points);
	Mat ClusterImg = myImgPro.ClusterPointsDrawing(ExGImage, points);
	*/

	/*
	vector<int> clusters = myImgPro.spectralClustering(susan_points, 3, 1000.0);
	// 创建一个簇向量，其中每个元素是包含属于该簇的点的向量
	vector<vector<Point>> clusters_points(3);

	// 遍历所有点，将它们添加到相应的簇中
	for (size_t i = 0; i < clusters.size(); ++i) {
		int cluster_label = clusters[i];
		clusters_points[cluster_label].push_back(susan_points.points[i]);
	}

	// 输出每个簇中的点
	for (size_t i = 0; i < clusters_points.size(); ++i) {
		cout << "Cluster " << i << ":" << endl;
		for (const Point& p : clusters_points[i]) {
			cout << "(" << p.x << ", " << p.y << ")" << endl;
		}
	}
	*/


	//高斯混合模型的协方差矩阵类型(对应不同的数据形状)，球状、对角、完全、等方位
	//ml::EM::Types covarianceType = ml::EM::Types::COV_MAT_GENERIC; //这里使用完全协方差矩阵，适用于数据中存在明显的相关性的情况
	//vector<CImgPro::Cluster> cluster_points = myImgPro.Gaussian_Mixture_Model(points, 3, covarianceType);
	//Mat ClusterImg = myImgPro.ClusterPointsDrawing(ExGImage, cluster_points);
	

	/*
		参数perCof很重要，需要调整以划分出指定的簇
	*/
	//float perCof = 0.8;
	//int cluNum = 3;
	//vector<CImgPro::Cluster> points = myImgPro.Bisecting_Kmeans(susan_points, cluNum, perCof);
	//Mat ClusterImg = myImgPro.ClusterPointsDrawing(ExGImage, points);
	

	/*
		averageD是聚类算法中的重要参数，决定了点与点间是否同属一类，太小会导致同属一类的点聚成两类，太大会聚到无关点
	*/
	//int areaHeight = 128, areaWidth = 205, areaDegree = 128, areaExtent = 205;		//扫描窗口
	//vector<CImgPro::Cluster> points = myImgPro.Cluster_for_Ransac(MorphImg, areaHeight, areaWidth, areaDegree, areaExtent);
	//Mat ClusterImg = myImgPro.ClusterPointsDrawing(ExGImage, points);


	/*
		使用dbscan的思想
	*/
	int imgCenterX = inputImage.cols / 2;
	vector<CImgPro::Cluster> first_cluster_points = myImgPro.firstClusterBaseOnDbscan(reduce_points, 110, 50);
	//vector<CImgPro::Cluster> first_cluster_points = myImgPro.firstClusterBaseOnDbscan(reduce_points, 40, 30);
	vector<CImgPro::Cluster> second_cluster_points = myImgPro.secondClusterBaseOnCenterX(first_cluster_points, imgCenterX, 0.65);	
	Mat F_ClusterImg = myImgPro.ClusterPointsDrawing(ExGImage, first_cluster_points);
	Mat S_ClusterImg = myImgPro.ClusterPointsDrawing(ExGImage, second_cluster_points);


	vector<CImgPro::Cluster> maxPts = myImgPro.MaxPoints(second_cluster_points);
	Mat maxPtsImg = myImgPro.ClusterPointsDrawing(ExGImage, maxPts);
	

	Mat HistogramImg = myImgPro.verticalProjection(S_ClusterImg, maxPts);

	myImgPro.retainMainStem(maxPts);
	Mat MainStemImg = myImgPro.ClusterPointsDrawing(ExGImage, maxPts);

	CImgPro::Cluster final_points;
	Mat ExtractImg(MainStemImg.size(), CV_8UC1, Scalar(0));
	myImgPro.processImageWithWindow(MainStemImg, ExtractImg, final_points, 16, 32);

	/*
		目前来看，霍夫变换无法很好处理离群点和噪声
	*/
	///*vector<CCoorTran::LineParameter> linepara;
	//myImgPro.Hough_Line(points, TempImg, linepara);
	//Mat HoughImg = TempImg;*/

	/*
		经过实验，ransac算法能比较好处理离群点和噪声。
		距离阈值需根据数据点调整
		使用改进的最小二乘法处理ransac后的点得到进一步的优化
		数据点规模很大的情况下会造成迭代次数过多，程序报错
	*/
	//float RANSAC_thresh = 0.5;
	//Mat RansacImg = inputImage.clone();
	//myImgPro.RANSAC(points, RANSAC_thresh, RansacImg);

	//保存拟合图像
	//myImgPro.SaveImg(filename, RansacImg);



	namedWindow("feature_Image", WINDOW_NORMAL);
	moveWindow("feature_Image", 0, 0);		// 设置第一个窗口的位置
	imshow("feature_Image", featureImg);

	//namedWindow("ExG_Image", WINDOW_NORMAL);
	//moveWindow("ExG_Image", 0, 10);		
	//imshow("ExG_Image", ExGImage);

	//namedWindow("MedianBlur_Img", WINDOW_NORMAL);
	//moveWindow("MedianBlur_Img",500, 0);		
	//imshow("MedianBlur_Img", MedianBlurImg);

	/*namedWindow("Susan_Img", WINDOW_NORMAL);
	moveWindow("Susan_Img", 0, 500);
	imshow("Susan_Img", TempImg);*/

	namedWindow("OTSU_Img", WINDOW_NORMAL);
	moveWindow("OTSU_Img", 500, 500);		 
	imshow("OTSU_Img",OtsuImg);

	if (flag == 1) {
		namedWindow("Morph_Img", WINDOW_NORMAL);
		moveWindow("Morph_Img", 0, 1000);
		imshow("Morph_Img", MorphImg);
	}
	

	namedWindow("Connect_Img", WINDOW_NORMAL);
	moveWindow("Connect_Img", 0, 550);
	imshow("Connect_Img", ConnectImg);

	namedWindow("F_Cluster_Img", WINDOW_NORMAL);
	moveWindow("F_Cluster_Img", 700, 0);
	imshow("F_Cluster_Img", F_ClusterImg);

	namedWindow("S_Cluster_Img", WINDOW_NORMAL);
	moveWindow("S_Cluster_Img", 800, 0);
	imshow("S_Cluster_Img", S_ClusterImg);

	namedWindow("maxPts_Img", WINDOW_NORMAL);
	moveWindow("maxPts_Img", 800, 200);
	imshow("maxPts_Img", maxPtsImg);

	namedWindow("Histogram_Img", WINDOW_NORMAL);
	moveWindow("Histogram_Img", 800, 400);
	imshow("Histogram_Img", HistogramImg);

	namedWindow("MainStem_Img", WINDOW_NORMAL);
	moveWindow("MainStem_Img", 900, 0);
	imshow("MainStem_Img", MainStemImg);

	namedWindow("Extract_Img", WINDOW_NORMAL);
	moveWindow("Extract_Img", 900, 200);
	imshow("Extract_Img", ExtractImg);

	//namedWindow("Skeleton_Img", WINDOW_NORMAL);
	//moveWindow("Skeleton_Img", 500, 700);
	//imshow("Skeleton_Img", skeletonImg);

	////namedWindow("Hough_Img", WINDOW_NORMAL);
	////moveWindow("Hough_Img", 500, 1000);
	////imshow("Hough_Img", HoughImg);

	//namedWindow("Ransac_Img", WINDOW_NORMAL);
	//moveWindow("Ransac_Img", 500, 1000);
	//imshow("Ransac_Img", RansacImg);


	waitKey(0);

	return 0;
}
