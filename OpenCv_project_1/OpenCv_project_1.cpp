#include <iostream>

#include"ImgPro.h"

using namespace std;
using namespace cv;

int main()
{

	string filename = "D:\\���ݸ����\\IMG_20230518_111417.jpg";
	Mat inputImage = imread(filename);

	//// ��ȡͼ��ߴ�
	//int width = inputImage.cols;
	//int height = inputImage.rows;
	//// ����ü�����ĸ߶�
	//int cropHeight = height / 10;
	//// ����ü���ͼ�����ʼλ��
	//int startY = cropHeight;
	//// �ü�ͼ��
	//cv::Rect roi(0, startY, width, height - cropHeight);
	//cv::Mat croppedImage = inputImage(roi);

	CImgPro myImgPro;

	Mat ExGImage(inputImage.size(), CV_8UC1);

	myImgPro.NormalizedExG(inputImage, ExGImage);	

	/*
		��ֵ�˲��ȸ�˹�˲��ڴ�����������Ч������
	*/
	int MedianBlur_kernel_size = 5;		//����˴�С
	Mat MedianBlurImg = myImgPro.MedianBlur(ExGImage, MedianBlur_kernel_size);

	
	Mat OtsuImg = myImgPro.OTSU(MedianBlurImg);

	/*
		��̬ѧ������������ϸС�Ӳ��а�������ͬʱҲ���������ϸ��
	*/
	Mat MorphImg;
	int flag = 0;
	if (CImgPro::NonZeroPixelRatio > 0.1) {
		MorphImg = myImgPro.MorphologicalOperation(OtsuImg, 3, 5);
		flag = 1;
	}


	/*
		ʹ�ð���ͨɸѡ�㷨����Чȥ��������ϸС�Ӳݣ����ɽ�һ���Ż�����������������ϸ��
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
		�Ǽܻ����ķ�ʱ�䳤
	*/
	//CImgPro::Cluster skeleton_points;
	//Mat skeletonImg = myImgPro.skeletonization(ConnectImg, skeleton_points);


	/*
		
	*/
	//int thresh = 10, k = 18;		//kΪ��Ӧ��ֵ�ı���ϵ��
	//CImgPro::Cluster susan_points;
	//Mat TempImg = myImgPro.My_SUSAN(ConnectImg, thresh, k, susan_points);
	//Mat SusanImg = TempImg.clone();
	///*int x = FeatureImg.cols, y = FeatureImg.rows;*/
	///*cout << "size:" << x <<"x"<< y << endl;*/


	/*
		Ŀǰ��ʹ�ó�������ȥ��ȡ����������ȽϺã��ܱ���������Ҫ����ͬʱ����������
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
	// ����һ��������������ÿ��Ԫ���ǰ������ڸôصĵ������
	vector<vector<Point>> clusters_points(3);

	// �������е㣬��������ӵ���Ӧ�Ĵ���
	for (size_t i = 0; i < clusters.size(); ++i) {
		int cluster_label = clusters[i];
		clusters_points[cluster_label].push_back(susan_points.points[i]);
	}

	// ���ÿ�����еĵ�
	for (size_t i = 0; i < clusters_points.size(); ++i) {
		cout << "Cluster " << i << ":" << endl;
		for (const Point& p : clusters_points[i]) {
			cout << "(" << p.x << ", " << p.y << ")" << endl;
		}
	}
	*/


	//��˹���ģ�͵�Э�����������(��Ӧ��ͬ��������״)����״���Խǡ���ȫ���ȷ�λ
	//ml::EM::Types covarianceType = ml::EM::Types::COV_MAT_GENERIC; //����ʹ����ȫЭ������������������д������Ե�����Ե����
	//vector<CImgPro::Cluster> cluster_points = myImgPro.Gaussian_Mixture_Model(points, 3, covarianceType);
	//Mat ClusterImg = myImgPro.ClusterPointsDrawing(ExGImage, cluster_points);
	

	/*
		����perCof����Ҫ����Ҫ�����Ի��ֳ�ָ���Ĵ�
	*/
	//float perCof = 0.8;
	//int cluNum = 3;
	//vector<CImgPro::Cluster> points = myImgPro.Bisecting_Kmeans(susan_points, cluNum, perCof);
	//Mat ClusterImg = myImgPro.ClusterPointsDrawing(ExGImage, points);
	

	/*
		averageD�Ǿ����㷨�е���Ҫ�����������˵������Ƿ�ͬ��һ�̫࣬С�ᵼ��ͬ��һ��ĵ�۳����̫࣬���۵��޹ص�
	*/
	//int areaHeight = 128, areaWidth = 205, areaDegree = 128, areaExtent = 205;		//ɨ�贰��
	//vector<CImgPro::Cluster> points = myImgPro.Cluster_for_Ransac(MorphImg, areaHeight, areaWidth, areaDegree, areaExtent);
	//Mat ClusterImg = myImgPro.ClusterPointsDrawing(ExGImage, points);


	/*
		ʹ��dbscan��˼��
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
		Ŀǰ����������任�޷��ܺô�����Ⱥ�������
	*/
	///*vector<CCoorTran::LineParameter> linepara;
	//myImgPro.Hough_Line(points, TempImg, linepara);
	//Mat HoughImg = TempImg;*/

	/*
		����ʵ�飬ransac�㷨�ܱȽϺô�����Ⱥ���������
		������ֵ��������ݵ����
		ʹ�øĽ�����С���˷�����ransac��ĵ�õ���һ�����Ż�
		���ݵ��ģ�ܴ������»���ɵ����������࣬���򱨴�
	*/
	//float RANSAC_thresh = 0.5;
	//Mat RansacImg = inputImage.clone();
	//myImgPro.RANSAC(points, RANSAC_thresh, RansacImg);

	//�������ͼ��
	//myImgPro.SaveImg(filename, RansacImg);



	namedWindow("feature_Image", WINDOW_NORMAL);
	moveWindow("feature_Image", 0, 0);		// ���õ�һ�����ڵ�λ��
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
