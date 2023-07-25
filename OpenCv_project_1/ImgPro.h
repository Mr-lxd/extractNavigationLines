#pragma once

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
//使用vcpkg安装eigen库，注意是x64
#include <Eigen/Dense>



using namespace std;
using namespace cv;
using namespace Eigen;

class CImgPro
{
public:

	typedef struct
	{
		vector<Point> points;
		vector<Point> CategID;
		double averageX;
		double averageY;
		double X;
		int count;
		char ID;
		char state;
	}Cluster;

	Mat MedianBlur(Mat srcimg, int kernel_size);
	Mat My_SUSAN(Mat& src, int thresh, int k, Cluster& points);
	Mat OTSU(Mat src);
	Mat MorphologicalOperation(Mat src, int kernel_size, int cycle_num);
	Mat ClusterPointsDrawing(Mat& src, vector<Cluster>& points);
	Mat applyPCA(Cluster& cluster, int num_components);
	void processImageWithWindow(Mat& img, Cluster& points, int windowWidth, int windowHeight);
	vector<int> LFDC(vector<vector<double>>& X, vector<int>& YIni, int dim);
	void averageCoordinates(Cluster& points);
	Mat EightConnectivity(Mat& img, float cof);
	vector<Cluster> Gaussian_Mixture_Model(Cluster& points, int numCluster, ml::EM::Types covarianceType);
	vector<Cluster> firstClusterBaseOnDbscan(Cluster& points, float epsilon, int minPts);
	vector<Cluster> secondClusterBaseOnCenterX(vector<Cluster> cluster_points, int imgCenterX);
	vector<int> spectralClustering(Cluster& points, int k, double sigma);
	void NormalizedExG(Mat srcimg, Mat& outimg);
	vector<Cluster> Cluster_Nearest(Mat& featureimage);
	//void Hough_Line(vector<Cluster> clusters, Mat& outimg, vector<CCoorTran::LineParameter>& linepara);
	//void Least_Square(vector<Cluster> clusters, Mat& outimg, vector<CCoorTran::LineParameter>& linepara);
	void RANSAC(vector<Cluster>& points, float thresh, Mat& outimg);
	vector<Cluster> Cluster_for_Ransac(Mat& featureimage, int areaHeight, int areaWidth, int areaDegree, int areaExtent);
	vector<Cluster> Bisecting_Kmeans(Cluster& points, int k, float perCof);
	void SaveImg(String filename, Mat& img);

private:

	float euclidean_distance(Point a, Point b);
	int calculate_x(Point p, float k, int outimg_rows);
	Point centroid(vector<Point>& points);
	Point min(vector<Point>& points) const;
	Point max(vector<Point>& points) const;
	int isLeftOrRight(const Point& a, const Point& b, const Point& c);
	bool isClusterPassed(const Cluster& cluster, const Point& minPoint, const Point& maxPoint, char ID);
	Mat computeSimilarityMatrix(const Cluster& points, double sigma);
	vector<Cluster> ComparePoints(vector<Cluster>& points);
	vector<int> regionQuery(Cluster& points, Point& point, double epsilon);
	void expandCluster(Cluster& points, vector<int>& clusterIDs, int currentClusterID,
		int pointIndex, double epsilon, int minPts, const vector<int>& neighbours);
	void leastSquaresFit_edit(vector<Cluster>& points, Mat& outimg);

	//	聚类参数
	vector<Cluster> BaseCluster(Mat featureimage, int beginHeight, int areaHeight, int areaWidth);

};