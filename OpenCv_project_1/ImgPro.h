#pragma once

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ximgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
//Install the Eigen library using vcpkg, and make sure it's for the x64 architecture.
#include <Eigen/Dense>



using namespace std;
using namespace cv;
using namespace Eigen;

class CImgPro
{
	public:
		static int imgCols, imgRows;

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

		double thresholdingSigmoid(double NonZeroPixelRatio, double k, double x);
		Mat MedianBlur(Mat srcimg, int kernel_size);
		Mat verticalProjection(Mat& img, const vector<Cluster>& clusters, double cof);
		Mat My_SUSAN(Mat& src, int thresh, int k, Cluster& points);		
		Mat MorphologicalOperation(Mat src, int kernel_size, int cycle_num_e, int cycle_num_d);
		Mat ClusterPointsDrawing(Mat& src, vector<Cluster>& points);
		Mat projectedImg(Mat& img, vector<Cluster>& clusters, float slope);
		Mat skeletonization(Mat& img, Cluster& points);
		pair<Mat, int> verticalProjectionForCenterX(const vector<int>& histogram);
		pair<Mat, vector<int>> EightConnectivity(Mat& img, float cof);
		pair<Mat, float> OTSU(Mat src);
		vector<Cluster> Gaussian_Mixture_Model(Cluster& points, int numCluster, ml::EM::Types covarianceType);
		vector<Cluster> firstClusterBaseOnDbscan(Cluster& points, float epsilon, int minPts);
		vector<Cluster> secondClusterBaseOnCenterX(vector<Cluster>& cluster_points, int imgCenterX, float cof);
		vector<Cluster> Cluster_Nearest(Mat& featureimage);
		vector<Cluster> Bisecting_Kmeans(Cluster& points, int k, float perCof);
		void processImageWithWindow(Mat& srcimg, Mat& outimg, Cluster& points, int windowWidth, int windowHeight);
		void retainMainStem(vector<Cluster>& clusters);
		void NormalizedExG(Mat& srcimg, Mat& outimg);		
		void RANSAC(Cluster& points, float thresh, Mat& outimg);
		void SaveImg(String filename, Mat& img);
		void leastSquaresFit_edit(Cluster& cluster, Mat& outimg);
		void Hough_Line(vector<Cluster>& clusters, Mat& outimg);


	private:

		float k1, k2, k3, k4, k5, k6;		//分别为左中右聚类的最小和最大点对应的范围直线的斜率
		float x_max, x_min;		//the x-coordinate of the intersection between the histogram and the horizontal line

		float euclidean_distance(Point a, Point b);
		float calculateNonZeroPixelRatio(Mat& img);
		int calculate_x(Point p, float k, int outimg_rows);
		bool isClusterPassed(const Cluster& cluster, const Point& minPoint, const Point& maxPoint, char ID);
		int isLeftOrRight(const Point& a, const Point& b, const Point& c);
		Point centroid(vector<Point>& points);
		Point min(vector<Point>& points) const;
		Point max(vector<Point>& points) const;
		vector<Cluster> ComparePoints(vector<Cluster>& points);
		vector<int> regionQuery(Cluster& points, Point& point, double epsilon);
		void expandCluster(Cluster& points, vector<int>& clusterIDs, int currentClusterID,
			int pointIndex, double epsilon, int minPts, const vector<int>& neighbours);

		//	聚类参数
		vector<Cluster> BaseCluster(Mat featureimage, int beginHeight, int areaHeight, int areaWidth);

};