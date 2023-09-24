#include "ImgPro.h"
#include<iostream>
#include <cmath>
#include <numeric>
#include <random> 
#include<opencv2/imgproc/types_c.h>


int CImgPro::imgCols = -1, CImgPro::imgRows = -1;


void CImgPro::NormalizedExG(Mat& srcimg, Mat& outimg)
{
	cvtColor(srcimg, outimg, COLOR_RGB2GRAY);		// convert input image to gray image
	unsigned char* in;		// input image pointer
	unsigned char* out;		
	unsigned char R, G, B;
	uchar temp1;		
	float r, g, b;		// The normalized difference vegetation index
	for (int i = 0; i < srcimg.rows; i++)
	{
		//Obtain the base address of the data located in the i-th row, and retrieve the pointers to the input and output image data of the current row
		//The variable "data" points to the starting address of the image data, variable "step" represents the number of bytes occupied by each row of data
		in = (unsigned char*)(srcimg.data + i * srcimg.step);		
		out = (unsigned char*)(outimg.data + i * outimg.step);
		for (int j = 0; j < srcimg.cols; j++)
		{
			//Retrieve the channel value of the j-th pixel in each row.
			B = in[3 * j];
			G = in[3 * j + 1];
			R = in[3 * j + 2];
			b = (float)B / (B + G + R);
			g = (float)G / (B + G + R);
			r = (float)R / (B + G + R);

			if (2 * g - r - b < 0)		
				temp1 = 0;
			else if (2 * g - b - r > 1)	
				temp1 = 255;
			else
				temp1 = (2 * g - b - r) * 255;

			out[j] = temp1;


			//The current results of the ExG-ExR method are not satisfactory, particularly in cases where the crops have a significant green component.
			/*float ExG = 2*G - R - B;
			float ExR = 1.4*R - G;
			if (G>R && G>B && ExG - ExR > 0)
			{
					out[j] = ExG - ExR;								
			}
			else
			{
				out[j] = 0;
			}*/

		}
	}
}

float CImgPro::euclidean_distance(Point a, Point b)
{
	float dx = a.x - b.x;
	float dy = a.y - b.y;
	return sqrt(pow(dx, 2) + pow(dy, 2));
}

float CImgPro::calculateNonZeroPixelRatio(Mat& img)
{
	int nonZeroPixelCount = 0;
	int totalPixelCount = img.rows * img.cols;

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			if (img.at<uchar>(i, j) != 0) {
				nonZeroPixelCount++;
			}
		}
	}

	// NonZeroPixelRatio
	float ratio = (float) nonZeroPixelCount / totalPixelCount;
	return ratio;
}

double CImgPro::thresholdingSigmoid(double NonZeroPixelRatio, double k, double x)
{
	//threshold = 1 / (1 + exp(-k * (NonZeroPixelRatio - x0)))
	double exp_part = exp(-k * (NonZeroPixelRatio - x));
	
	double numerator = 1;
	
	double denominator = 1 + exp_part;
	
	double result = numerator / denominator;
	
	return result;
}

Point CImgPro::centroid(vector<Point>& points)
{
	float sum_x = 0.0, sum_y = 0.0;
	for (const auto& p : points) {
		sum_x += p.x;
		sum_y += p.y;
	}
	return Point(sum_x / points.size(), sum_y / points.size());
}

int CImgPro::calculate_x(Point p, float k, int outimg_rows)
{
	float b = p.y - k * p.x;
	
	int x = (outimg_rows - b) / k;
	
	return x;
}

Point CImgPro::min(vector<Point>& points) const {
	assert(!points.empty());
	return *std::min_element(points.begin(), points.end(),
		[](const Point& a, const Point& b) { return a.x < b.x; });
}

Point CImgPro::max(vector<Point>& points) const {
	assert(!points.empty());
	return *std::max_element(points.begin(), points.end(),
		[](const Point& a, const Point& b) { return a.x < b.x; });
}

int CImgPro::isLeftOrRight(const Point& a, const Point& b, const Point& c)
{
	float side = ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
	if (side > 0)
		return -1;		//If the cross product is greater than zero, the point is on the left of the line
	else if (side < 0)
		return 1;		//If the cross product is less than zero, the point is on the right of the line
	/*else if (((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)) == 0)
		return 0;*/		//the point is on the the line
}

bool CImgPro::isClusterPassed(const Cluster& cluster, const Point& a, const Point& b, char ID)
{
	int inliers = 0;
	//Point l_min = Point(a.x - 5, a.y + 4), l_max = Point(b.x - 4, b.y + 5);//左
	//Point r_min = Point(a.x + 4, a.y + 5), r_max = Point(b.x + 5, b.y + 4);//右
	Point c_min = Point(a.x - 1, a.y + 12), c_max = Point(b.x + 1, b.y + 12);//中
	//k1 = (float)(l_min.y - a.y) / (l_min.x - a.x), k2 = (float)(l_max.y - b.y) / (l_max.x - b.x), k3 = (float)(r_min.y - a.y) / (r_min.x - a.x);
	//k4 = (float)(r_max.y - b.y) / (r_max.x - b.x), 
	k5 = (float)(c_min.y - a.y) / (c_min.x - a.x), k6 = (float)(c_max.y - b.y) / (c_max.x - b.x);

	//if (ID == 'l') {
	//
	//	for (const Point& p : cluster.points) {
	//		int i = isLeftOrRight(a, l_min, p);
	//		int j = isLeftOrRight(b, l_max, p);
	//		if (i == 1 && j == -1)
	//			inliers++;
	//	}
	//}
	//if (ID == 'r') {
	//	for (const Point& p : cluster.points) {
	//		int i = isLeftOrRight(a, r_min, p);
	//		int j = isLeftOrRight(b, r_max, p);
	//		if (i == 1 && j == -1)
	//			inliers++;
	//	}
	//}
	if (ID == 'c') {
		for (const Point& p : cluster.points) {
			int i = isLeftOrRight(a, c_min, p);
			int j = isLeftOrRight(b, c_max, p);
			if (i == 1 && j == -1)
				inliers++;
		}
	}

	if (inliers != 0) {
		return true;
	}
	else
	{
		return false;
	}
}

Mat CImgPro::MedianBlur(Mat srcimg, int kernel_size)
{
	Mat MedianBlurImg(srcimg.size(), CV_8UC1);
	medianBlur(srcimg, MedianBlurImg, kernel_size);

	return MedianBlurImg;
}

pair<Mat, vector<int>> CImgPro::EightConnectivity(Mat& img, float cof)
{
	Mat labels; // Output labeled image
	int num_labels; // Number of connected components
	Mat stats; // Output statistics for each connected component (bounding box, area, etc.)
	Mat centroids; // Output centroids for each connected component
	num_labels = connectedComponentsWithStats(img, labels, stats, centroids, 8, CV_32S);
	vector<int> firstHistogram(img.cols, 0);

	double sum_area = 0.0;
	double mean_area = 0.0;
	for (int i = 1; i < num_labels; i++) { // Start from 1 to skip the background
		sum_area += stats.at<int>(i, CC_STAT_AREA); //  Accumulate area of each connected component
	}
	mean_area = sum_area / (num_labels - 1); // Calculate mean area, excluding background

	Mat output = Mat::zeros(labels.size(), CV_8UC1);
	for (int i = 0; i < labels.rows; i++) {
		for (int j = 0; j < labels.cols; j++) {
			int label = labels.at<int>(i, j); // Get label of the current pixel
			// Check if area of the connected component containing the current pixel is greater than or equal to the mean area
			if (label > 0 && stats.at<int>(label, CC_STAT_AREA) >= mean_area * cof) { 
				output.at<uchar>(i, j) = 255;
				if (j >= 0.325 * labels.cols && j <= 0.675 * labels.cols) {//// Process region where the image does not follow normal distribution, like: 112212
					firstHistogram[j]++;
				}
				
			}
		}
	}

	return { output,firstHistogram };
}

void CImgPro::processImageWithWindow(Mat& srcimg, Mat& outimg, Cluster& points, int windowWidth, int windowHeight, int flag)
{
	int rows = srcimg.rows;
	int cols = srcimg.cols;
	
	for (int y = 0; y <= rows - windowHeight; y += windowHeight)
	{
		for (int x = 0; x <= cols - windowWidth; x += windowWidth)
		{
			// Calculate the average value of pixel coordinates within the window
			int count = 0;
			int sumX = 0.0, sumY = 0.0;
			float avgX = 0.0, avgY = 0.0;
			for (int wy = 0; wy < windowHeight; ++wy)
			{
				for (int wx = 0; wx < windowWidth; ++wx)
				{
					if (srcimg.channels() == 1) {
						if (srcimg.at<uchar>(y + wy, x + wx) != 0)
						{
							sumY += y + wy;
							sumX += x + wx;
							count++;
							//img.at<uchar>(y + wy, x + wx) = 0;
						}
					}
					if (srcimg.channels() == 3) {
						if (srcimg.at<Vec3b>(y + wy, x + wx) != Vec3b(0, 0, 0))
						{
							sumY += y + wy;
							sumX += x + wx;
							count++;
						}
					}
				}
			}

			if (sumX!=0 || sumY!=0)
			{
				avgX = (float)sumX / count;
				avgY = (float)sumY / count;
				if (flag==1) {
					outimg.at<uchar>(avgY, avgX) = 255;
					
				}
				if(flag==2)
				{
					outimg.at<Vec3b>(avgY, avgX) = Vec3b(0, 0, 0);
				}
				
				points.points.push_back(Point(avgX, avgY));
			}
		}
	}

}

Mat CImgPro::verticalProjection(Mat& img, const vector<Cluster>& clusters, double cof)
{
	vector<int> histogram(img.cols, 0);
	
	for (auto& c : clusters) {
		for (auto& p : c.points) {
			histogram[p.x]++;			
		}
	}	

	int y_max = histogram[0];
	for (auto& y : histogram) {
		if (y > y_max) {
			y_max = y;
		}
	}
	

	Size size(img.cols, 1.2 * y_max);
	Mat histogramImg(size, CV_8UC1, Scalar(0));
	for (int i = 0; i < histogram.size(); i++) {
		// Draw the histogram line
		line(histogramImg, Point(i, 1.2 * y_max), Point(i, 1.2 * y_max - histogram[i]), Scalar(255), 1);
	}

	// Draw a horizontal line for thresholding
	int horizontal_line_height = cof * y_max;
	//line(histogramImg, Point(0, horizontal_line_height), Point(histogramImg.cols - 1, horizontal_line_height), Scalar(255), 1);

	// Find the x-coordinate of the intersection between the histogram and the horizontal line
	bool flag = true;
	x_max = -1, x_min = histogramImg.cols + 1;
	for (int i = 0; i < histogram.size(); i++) {
		if (histogramImg.at<uchar>(horizontal_line_height, i) == 255) {
			if (flag) {
				x_min = i;
				flag = false;
			}
			if (i > x_max) {
				x_max = i;
			}
		}
	}
	int x1 = x_min;
	int x2 = x_max;

	return histogramImg;
}

pair<Mat, int> CImgPro::verticalProjectionForCenterX(const vector<int>& histogram)
{
	int y_max = -1, centerX = -1;
	for (int i = 0; i < histogram.size(); i++)
	{
		if (histogram[i] > y_max) {
			y_max = histogram[i];
			centerX = i;
		}
	}


	Size size(histogram.size(), 1.2 * y_max);
	Mat histogramImg(size, CV_8UC1, Scalar(0));
	for (int i = 0; i < histogram.size(); i++) {
		// Draw the histogram line
		line(histogramImg, Point(i, 1.2 * y_max), Point(i, 1.2 * y_max - histogram[i]), Scalar(255), 1);
	}


	return {histogramImg, centerX};
}

void CImgPro::retainMainStem(vector<Cluster>& clusters)
{
	for (auto& c : clusters) {
		auto it = c.points.begin();
		while (it != c.points.end()) {
			if (it->x < x_min || it->x > x_max) {
				// Remove points outside the desired range
				it = c.points.erase(it);
			}
			else {
				// Keep points inside the desired range
				++it;
			}
		}
	}

}

pair<Mat, float> CImgPro::OTSU(Mat src)
{
	int thresh = 0, PerPixSum[256] = { 0 };
	float PerPixDis[256] = { 0 }, NonZeroPixelRatio = 0.0f;

	//Count the quantity of each grayscale value
	for (int i=0; i<src.rows; i++)
	{
		for (int j=0; j<src.cols; j++)
		{
			PerPixSum[src.at<uchar>(i, j)]++;
		}
	}

	//Calculate the proportion of pixels for each grayscale level compared to the total number of pixels in the image
	for (int i = 0; i < 256; i++)
	{
		PerPixDis[i] = (float)PerPixSum[i] / (src.rows * src.cols);
	}

	//Iterate through all grayscale levels and calculate the threshold corresponding to the maximum between-class variance
	float PixDis_1, PixSum_1, PixDis_2, PixSum_2, avg_1, avg_2, ICV_temp;
	double ICV_max = 0.0;
	for (int i = 0; i < 256; i++)
	{
		PixDis_1 = PixSum_1 = PixDis_2 = PixSum_2 = avg_1 = avg_2 = ICV_temp = 0;
		//Compute the two segments resulting from threshold segmentation
		for (int j = 0; j < 256; j++)
		{
			//first segment
			if (j <= i)
			{
				PixDis_1 += PerPixDis[j];
				PixSum_1 += j * PerPixDis[j];
			}
			//second segment
			else
			{
				PixDis_2 += PerPixDis[j];
				PixSum_2 += j * PerPixDis[j];
			}
		}
		//The grayscale mean values of the two segments
		avg_1 = PixSum_1 / PixDis_1;
		avg_2 = PixSum_2 / PixDis_2;
		ICV_temp = PixDis_1 * PixDis_2 * pow((avg_1 - avg_2), 2);
		//Compare the thresholds
		if (ICV_temp > ICV_max)
		{
			ICV_max = ICV_temp;
			thresh = i;
		}
	}

	//binary
	Mat OtsuImg(src.size(), CV_8UC1, Scalar(0));
	int nonZeroPixelCount = 0;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			//Foreground
			if (src.at<uchar>(i, j) > 0.8 * thresh)//Lower the threshold to prevent filtering out darker green tones.
			{
				OtsuImg.at<uchar>(i, j) = 255;
				nonZeroPixelCount++;
			}
			//background
			//else
			//{
			//	OtsuImg.at<uchar>(i, j) = 0;
			//}
		}
	}

	// NonZeroPixelRatio for MorphologicalOperation
	int totalPixel = src.rows * src.cols;
	NonZeroPixelRatio = (float)nonZeroPixelCount / totalPixel;

	return {OtsuImg, NonZeroPixelRatio};
}

Mat CImgPro::MorphologicalOperation(Mat src, int kernel_size, int cycle_num_e, int cycle_num_d)
{
	Mat kernel;

	kernel = getStructuringElement(MORPH_RECT, Size(kernel_size, kernel_size));

	erode(src, src, kernel, Point(-1, -1), cycle_num_e);
	dilate(src, src, kernel, Point(-1, -1), cycle_num_d);

	return src;
}

Mat CImgPro::ClusterPointsDrawing(Mat& src, vector<Cluster>& points)
{		
	//Mat outimg(src.size(), CV_8UC3, Scalar(0, 0, 0));
	Mat outimg(src.size(), CV_8UC3, Scalar(0, 0, 0));

	vector<Scalar> colors = {
	Scalar(0, 0, 255),     // red
	Scalar(0, 255, 0),     // green
	Scalar(255, 0, 0),     // blue
	Scalar(255, 255, 0),   // yellow
	Scalar(0, 255, 255),   // cyan
	Scalar(255, 0, 255),   // magenta
	Scalar(128, 0, 0),     // deep red
	Scalar(0, 128, 0),     // dark green
	Scalar(0, 0, 128),      // dark blue
	Scalar(128, 128, 0),   // olive green
	Scalar(128, 0, 128),   // purple
	Scalar(0, 128, 128),   // turquoise
	Scalar(0, 165, 255),   // orange
	};

	// Iterate through each set of coordinates and plot them on the image.
	for (int i = 0; i < points.size(); i++) {
		bool firstPoint = true;
		int id = i + 1;
		for (int j = 0; j < points[i].points.size(); j++) {
			int x = points[i].points[j].x;
			int y = points[i].points[j].y;

			// Determine the brush color based on the group to which the point belongs.
			Scalar color = colors[i % colors.size()];
			/*Scalar color = colors[i];*/

			// Plot the point
			circle(outimg, Point(x, y), 1, color, -1);			
			
			//Display the group number next to the first point.
			if (firstPoint)
			{
				Point textPt1(x + 20, y + 90);
				putText(outimg, to_string(id), textPt1, FONT_HERSHEY_SIMPLEX, 4, color, 6);
				firstPoint = false;

				//d1,d2,d3....
				/*Point textPt2((points[i].averageX + 2029)/2, points[i].averageY-10);
				string s = "d" + to_string(id);
				putText(outimg, s, textPt2, FONT_HERSHEY_SIMPLEX, 4, Scalar(255, 255, 255), 8);
				firstPoint = false;*/
			}

			/*arrowedLine(outimg, Point(points[i].averageX, points[i].averageY), Point(2029, points[i].averageY), Scalar(255, 255, 255), 2, 8, 0, 0.1);
			arrowedLine(outimg, Point(2029, points[i].averageY), Point(points[i].averageX, points[i].averageY), Scalar(255, 255, 255), 2, 8, 0, 0.1);*/
		}

	}
	/*line(outimg, Point(1316, 0), Point(1316, outimg.rows), Scalar(255, 255, 255), 4, 8, 0);
	line(outimg, Point(2029, 0), Point(2029, outimg.rows), Scalar(255, 255, 255), 4, 8, 0);
	line(outimg, Point(2742, 0), Point(2742, outimg.rows), Scalar(255, 255, 255), 4, 8, 0);
	arrowedLine(outimg, Point(1316, 3000), Point(2742, 3000), Scalar(255, 255, 255), 2, 8, 0, 0.1);
	arrowedLine(outimg, Point(2742, 3000), Point(1316, 3000), Scalar(255, 255, 255), 2, 8, 0, 0.1);*/

	//ROI
	/*k5 = -12, k6 = 12;
	int x1 = calculate_x(points[0].CategID[0], k5, outimg.rows);
	line(outimg, points[0].CategID[0], Point(x1, outimg.rows), Scalar(255, 255, 255), 2, 8, 0);
	int x2 = calculate_x(points[0].CategID[1], k6, outimg.rows);
	line(outimg, points[0].CategID[1], Point(x2, outimg.rows), Scalar(255, 255, 255), 2, 8, 0);*/

	return outimg;
}

void CImgPro::RANSAC(Cluster& cluster, float thresh, Mat& outimg)
{
	vector<float> dis;
	vector<CImgPro::Cluster> inliers;
	Cluster tempPoints;
	struct Line {
		double slope, intercept;
	};

	random_device rd;		//Local true random number generator
	mt19937 gen(rd());		//Pseudo-random number generator using rd as seed

	int best_inliers = 0;
	float bestSlope = 0.0, bestIntercept = 0.0;

	float iterations = 0.0, ConfidenceLevel = 0.99, Probability = 2.0 / cluster.points.size();
	iterations = log((1 - ConfidenceLevel)) / log((1 - pow(Probability, 2)));
	for (int j = 0; j < iterations; j++)		//Continuously iterate in this cluster
	{		
		// Randomly select two different points
		uniform_int_distribution<> distrib(0, cluster.points.size()-1);		//Integer distribution object 'distrib' with range [0, n]
		int index1 = distrib(gen);
		int index2 = distrib(gen);
		// Prevent selecting the same index
		while (index2 == index1)
		{
			index2 = distrib(gen);
		}
		Point p1 = cluster.points[index1];
		Point p2 = cluster.points[index2];

		float slope = 0, intercept = 0;
		if (p1.x==p2.x)
		{
			slope = 9999999;
		}
		else {
			slope = (float)(p2.y - p1.y) / (p2.x - p1.x);
		}
		intercept = p1.y - slope * p1.x;
		Line l = { slope, intercept };

		// Calculate distances from all points in this cluster to the line excluding points p1 and p2
		float distance = 0;
		for (auto p : cluster.points)
		{
			if (p != p1 && p != p2)
			{
				distance = abs(p.y - l.slope * p.x - l.intercept) / sqrt(1 + l.intercept * l.intercept);
				//Check inliers
				if (distance < thresh)
				{
					tempPoints.points.push_back(p);
					//dis.push_back(distance);
				}
			}
		}	


		if (tempPoints.points.size()>best_inliers)
		{
			inliers.clear();
			inliers.push_back(tempPoints);
			best_inliers = tempPoints.points.size();
			bestSlope = l.slope;
			bestIntercept = l.intercept;

			//Update inlier ratio and iterations 
			Probability = (float)best_inliers / cluster.points.size(); // Calculate inlier ratio based on the current best hypothesis inliers
			iterations = log((1 - ConfidenceLevel)) / log((1 - pow(Probability, 2))); 
			iterations = 300 * iterations;
			j = 0; //Reset iteration counter
		}

		
		tempPoints.points.clear();
		//dis.clear();		
	}

	//outimg = ClusterPointsDrawing(outimg, inliers);

	/*Scalar color = CV_RGB(255, 0, 0);
	line(outimg, Point(-bestIntercept / bestSlope, 0), Point((outimg.rows - bestIntercept) / bestSlope, outimg.rows), color, 10, 8, 0);*/

	//perform least-squares fitting on the point set with the most inliers
	leastSquaresFit_edit(inliers[0], outimg);
	//Hough_Line(inliers, outimg);
}

vector<CImgPro::Cluster> CImgPro::firstClusterBaseOnDbscan(Cluster& points, float epsilon, int minPts)
{
	vector<int> clusterIDs(points.points.size(), -1); // Initialize the cluster labels for all points as -1, indicating noise points
	int currentClusterID = 0; 




	for (int i = 0; i < points.points.size(); ++i) {
		if (clusterIDs[i] == -1) { // unclassified
			Point& p = points.points[i];
			vector<int> neighbours = regionQuery(points, p, epsilon);
			if (neighbours.size() >= minPts) { 
				expandCluster(points, clusterIDs, currentClusterID, i, epsilon, minPts, neighbours); // 扩展聚类
				currentClusterID++; 				
			}
		}

		
	}

	vector<Cluster> cluster_points(currentClusterID);
	for (int i = 0; i < points.points.size(); ++i) {
		int clusterID = clusterIDs[i]; // current point ID
		if (clusterID >= 0) { // remove noise（ID=-1）
			cluster_points[clusterID].points.push_back(Point(points.points[i].x, points.points[i].y)); // 将当前点添加到相应的聚类集合中
		}
	}

	for (Cluster& cluster : cluster_points) {
		if (!cluster.points.empty()) {
			// calculate centroid
			for (Point& point : cluster.points) {
				cluster.averageX += point.x;
				cluster.averageY += point.y;
			}
			cluster.averageX /= cluster.points.size();
			cluster.averageY /= cluster.points.size();

		}
	}

	return cluster_points; 
}

vector<CImgPro::Cluster> CImgPro::secondClusterBaseOnCenterX(vector<Cluster>& cluster_points, int imgCenterX, float cof)
{
	//imageCenterX is the baseline
	vector<float> centroidDistances;
	vector<float> centroidXCoords;
	for (Cluster& cluster : cluster_points) {
		float distance = abs(cluster.averageX - imgCenterX);
		centroidDistances.push_back(distance);
		centroidXCoords.push_back(cluster.averageX);
	}


	float averageDistance = accumulate(centroidDistances.begin(), centroidDistances.end(), 0.0f) / centroidDistances.size();


	vector<Cluster> center;
	for (int i = 0; i < cluster_points.size(); ++i) {
		if (centroidDistances[i] <= cof * averageDistance) {
			center.push_back(cluster_points[i]);
		}
	}

	Cluster temp;

	/*    Irrelevant code, kept for future research convenience.   */
	// lfet side
	/* 
	while (!left.empty())
	{
		Point minPoint = min(left[0].points);
		Point maxPoint = max(left[0].points);

		temp.CategID.push_back(minPoint);
		temp.CategID.push_back(maxPoint);

		temp.points.insert(temp.points.end(), left[0].points.begin(), left[0].points.end());
		移动位置并删除该聚类
		rotate(left.begin(), left.begin() + 1, left.end());
		left.pop_back();
		temp.ID = 'l';

		for (auto it = left.begin(); it != left.end();) {
			Cluster& cluster = *it;
			bool flag = isClusterPassed(cluster, minPoint, maxPoint, cluster.ID);
			if (flag) {
				temp.points.insert(temp.points.end(), cluster.points.begin(), cluster.points.end());
				it = left.erase(it); 
			}
			else {
				++it; 
			}
		}

		final_cluster_points.push_back(temp);
		temp.points.clear();
		temp.CategID.clear();
	}
	*/

	vector<Cluster> final_cluster_points;
	Point minPoint = Point(imgCenterX - 0.07 * imgCols, 0);
	Point maxPoint = Point(imgCenterX + 0.07 * imgCols, 0);

	temp.CategID.push_back(minPoint);
	temp.CategID.push_back(maxPoint);
	temp.ID = 'c';

	for (auto it = center.begin(); it != center.end();) {
		Cluster& cluster = *it;
		cluster.ID = 'c';
		bool flag = isClusterPassed(cluster, minPoint, maxPoint, cluster.ID);
		if (flag) {
			temp.points.insert(temp.points.end(), cluster.points.begin(), cluster.points.end());
			it = center.erase(it); // Delete visited clusters and update the iterator.
		}
		else {
			++it; // Continue iterating to the next cluster.
		}
	}
	final_cluster_points.push_back(temp);

	vector<int> firstHistogram(imgCols, 0);
	for (Cluster& cluster : final_cluster_points) { 
		for (Point& point : cluster.points) { 
			int key = point.x; 
			firstHistogram[key]++;			
		}
	}


	temp.points.clear();
	temp.CategID.clear();

	return final_cluster_points;
}

vector<int> CImgPro::regionQuery(Cluster& points, Point& point, double epsilon)
{
	vector<int> neighbours;
	for (int i = 0; i < points.points.size(); ++i) {
		Point& p = points.points[i];
		if (euclidean_distance(point, p) <= epsilon) { 
			neighbours.push_back(i); // Add the indices of samples within ε distance to the neighborhood.
		}
	}
	return neighbours;
}

void CImgPro::expandCluster(Cluster& points, vector<int>& clusterIDs, int currentClusterID,
	int pointIndex, double epsilon, int minPts, const vector<int>& neighbours)
{
	std::queue<int> processQueue;
	processQueue.push(pointIndex);
	clusterIDs[pointIndex] = currentClusterID; // Mark the current point as the current cluster ID

	while (!processQueue.empty()) {
		int currentIndex = processQueue.front();
		processQueue.pop();
		Point& p = points.points[currentIndex];
		vector<int> newNeighbours = regionQuery(points, p, epsilon); // Neighborhood query of density-connected points
		if (newNeighbours.size() >= minPts) { // Number of samples in the neighborhood of density-connected points is greater than or equal to minPts
			for (int i : newNeighbours) {
				if (clusterIDs[i] == -1) { // Unclassified
					clusterIDs[i] = currentClusterID; // Mark density-connected points as the current cluster ID
					processQueue.push(i); // Add newly found neighboring points to the queue
				}
			}
		}
	}
}

//Improved Least Squares Fitting Capable of Fitting Vertical Lines
void CImgPro::leastSquaresFit_edit(Cluster& cluster, Mat& outimg)
{
	//ax+by+c=0
	//for (int i = 0; i < points.size(); i++)
	//{
		double sumX = 0.0, sumY = 0.0, avgX = 0.0, avgY = 0.0;
		for (auto p : cluster.points)
		{
			sumX += p.x;
			sumY += p.y;
		}
		avgX = sumX / cluster.points.size();
		avgY = sumY / cluster.points.size();

		double L_xx = 0.0, L_yy = 0.0, L_xy = 0.0;
		for (auto p : cluster.points)
		{
			L_xx += (p.x - avgX) * (p.x - avgX);
			L_xy += (p.x - avgX) * (p.y - avgY);
			L_yy += (p.y - avgY) * (p.y - avgY);
		}

		double lamd1 = 0.0, lamd2 = 0.0, lamd = 0.0, m = 0.0, n = 0.0;
		m = L_xx + L_yy;
		n = L_xx * L_yy - L_xy * L_xy;
		lamd1 = (m + sqrt(m * m - 4 * n)) / 2;
		lamd2 = (m - sqrt(m * m - 4 * n)) / 2;
		lamd = lamd1 < lamd2 ? lamd1 : lamd2;
		double d = sqrt((L_xx - lamd) * (L_xx - lamd) + L_xy * L_xy);

		double a, b, c;
		if (abs(d) < 1e-6)
		{
			a = 1;
			b = 0;
			c = -a * avgX - b * avgY;
		}
		else
		{
			if (lamd >= L_xx)
			{
				a = L_xy / d;
				b = (lamd - L_xx) / d;
				c = -a * avgX - b * avgY;
			}
			else
			{
				a = -L_xy / d;
				b = (L_xx - lamd) / d;
				c = -a * avgX - b * avgY;
			}
		}


		if (b != 0 && (-a / b >= 1 || -a / b <= -1))
		{
			Scalar color = CV_RGB(255, 0, 0);
			line(outimg, Point(-c / a, 0), Point((outimg.rows * (-b) - c) / a, outimg.rows), color, 20, 8, 0);
		}
		
		float firstSlope = (float)-a / b;
	//}
}

void CImgPro::SaveImg(String filename, Mat& img)
{
	// Get the full path of the source image file name
	std::string fullpath = cv::samples::findFile(filename, true, true);

	// Get the directory and file name from the full path
	std::string directory = fullpath.substr(0, fullpath.find_last_of("/\\") + 1);
	std::string basename = fullpath.substr(fullpath.find_last_of("/\\") + 1);

	// Remove the file extension from the file name
	std::string filename_without_extension = basename.substr(0, basename.find_last_of("."));

	// Construct the file name and path for saving
	//std::string outfilename = "D:\\ProcessedImg3.0\\" + filename_without_extension + "_slidingWin" + ".jpg";
	std::string outfilename = "D:\\水稻处理图像\\" + filename_without_extension + ".jpg";

	// Save image
	cv::imwrite(outfilename, img);
}




/********************************************************************************************************************/

/*    Irrelevant code, kept for future research convenience.   */

//预分类
vector< CImgPro::Cluster> CImgPro::BaseCluster(Mat featureimage, int beginHeight, int areaHeight, int areaWidth)
{
	vector<Cluster> Categories;		//存储所有聚类
	Cluster tempCateg;		//临时存储一个聚类
	tempCateg.averageX = 0;
	tempCateg.averageY = 0;
	tempCateg.count = 0;
	tempCateg.X = 0;
	int Height = featureimage.rows, Width = featureimage.cols;
	vector<int> column(Width, 0);		//使用静态数组会导致栈溢出
	int i, j, k, l, m, n;
	int tempNum, averageNum, totalNum = 0, areaNum = 0;
	unsigned char* in = (unsigned char*)featureimage.data;
	bool flagPre = false, flagAf = false;		//标记前一窗口和当前窗口

	//统计特征图扫描窗口每次从左到右遍历时的特征点
	for (j = 0; j < Width; j++)
	{
		tempNum = 0;
		//在扫描窗口h范围内统计每列的特征点
		for (i = beginHeight; i > beginHeight - areaHeight; i--)
		{
			if (0 != in[i * Width + j])
				tempNum++;
		}
		column[j] = tempNum;
		totalNum += tempNum;
	}
	averageNum = totalNum * areaWidth / Width;		//第一次遍历的平均特征点=总数/行扫描窗口数

	//从左到右遍历每个窗口，j是窗口左边界
	for (j = 0; j < Width - areaWidth; j++)
	{
		if (j == 0)
			for (l = 0; l < areaWidth; l++)
				areaNum += column[l];		//第一个扫描窗口内特征点总数
		else
		{
			areaNum = areaNum + column[j + areaWidth - 1] - column[j - 1];				//计算某一区域的特征点总数，扫描窗口每次移动一个像素点
		}
		if (areaNum > averageNum * 0.7)		//聚类条件 小于0.7时容易过聚类，大于时容易聚类数量不足
			flagAf = true;
		else
			flagAf = false;

		//当前窗口满足聚类条件而前一个窗口不满足时创建一个新聚类
		if ((true == flagAf) && (flagPre == false))
		{
			for (m = j; m < j + areaWidth; m++)
			{
				for (n = beginHeight; n > beginHeight - areaHeight; n--)
				{
					if (0 != in[n * Width + m])
					{
						//所有特征点加入新聚类，同时更新该聚类平均坐标和总数
						tempCateg.points.push_back(cvPoint(m, n));
						//注意原点是图片左上角
						tempCateg.averageX += m;
						tempCateg.averageY += n;
						tempCateg.count++;
					}
				}
			}
		}
		else if ((true == flagAf) && (true == flagPre))
		{
			for (n = beginHeight; n > beginHeight - areaHeight; n--)
				//当前窗口是一个已有的聚类，只需要遍历扫描窗口内新加入的一列像素
				if (0 != in[n * Width + j + areaWidth - 1])
				{
					tempCateg.points.push_back(cvPoint(j + areaWidth - 1, n));		//j+areaWidth-1表示扫描窗口的右边界，即新加入的那列
					tempCateg.averageX += (j + areaWidth - 1);
					tempCateg.averageY += n;
					tempCateg.count++;
				}
			//当前窗口是最后一个窗口时，将该聚类放入Categories，并清空临时聚类
			if (j == (Width - areaWidth - 1))
			{
				tempCateg.averageX = tempCateg.averageX / tempCateg.count;		//计算当前预分类的新类的平均坐标
				tempCateg.averageY = tempCateg.averageY / tempCateg.count;
				tempCateg.ID++;		//标记当前tempCateg
				Categories.push_back(tempCateg);
				tempCateg.points.clear();
				tempCateg.averageX = 0;
				tempCateg.averageY = 0;
				tempCateg.count = 0;
			}
		}
		//当前窗口不满足时表明当前该新类已聚类完成，更新相关参数
		else if ((false == flagAf) && (true == flagPre))
		{
			tempCateg.averageX = tempCateg.averageX / tempCateg.count;
			tempCateg.averageY = tempCateg.averageY / tempCateg.count;
			tempCateg.ID++;
			Categories.push_back(tempCateg);
			tempCateg.points.clear();
			tempCateg.averageX = 0;
			tempCateg.averageY = 0;
			tempCateg.count = 0;
		}
		flagPre = flagAf;
	}
	return Categories;
}
//合并最近类
vector<CImgPro::Cluster> CImgPro::Cluster_Nearest(Mat& featureimage)
{
	vector<Cluster> Categories;
	vector<Cluster> preCateg;
	vector<Cluster> points;							//用于记录平均点
	Cluster tempPoint;
	int Height = featureimage.rows, Width = featureimage.cols;
	vector<int> column(Width, 0);
	int i, j, k, l, m, n;
	int tempNum, averageNum, totalNum = 0, areaNum = 0;
	int areaHeight = 128, areaWidth = 205, areaDegree = 128, areaExtent = 205;
	unsigned char* in = (unsigned char*)featureimage.data;		//传入特征图像数据指针
	Cluster tempCateg;
	int minD, tempD, ID, averageD, temp = 0;

	//初次聚类，并将初次聚类的平均坐标放入points
	Categories = BaseCluster(featureimage, Height - 1, areaHeight, areaWidth);
	for (i = 0; i < Categories.size(); i++)
	{
		tempPoint.points.push_back(cvPoint(Categories[i].averageX, Categories[i].averageY));
		points.push_back(tempPoint);
		tempPoint.points.clear();
	}

	//按扫描窗口高度进行每一行的扫描并聚类，下半部？
	for (i = Height - areaHeight - 1; i >= Height / 2 + areaHeight - 1; i = i - areaHeight)
	{
		preCateg = BaseCluster(featureimage, i, areaHeight, areaWidth);		/*预分类当前行，窗口相邻类合并
																			（断点调试时此处找不到matrix.cpp文件跳到函数中去执行代码，在opencv库中找到文件并指定路径即可）*/
		if (preCateg.size() != 0)
			averageD = Width / (2 * preCateg.size()) + areaHeight;					//不安全，除数为0

		//不适用，存在preCateg.size()=1的情况
		//if(preCateg.size()!=0 && preCateg.size()!=1)
		//	for (int a = 1; a < preCateg.size(); a++)
		//	{
		//		temp +=sqrt(pow(preCateg[a].averageX-preCateg[a-1].averageX, 2) + pow(preCateg[a].averageY - preCateg[a - 1].averageY, 2))/(preCateg.size()-1) + areaHeight;
		//		averageD = temp;
		//	}

		for (j = 0; j < preCateg.size(); j++)
		{
			minD = 4096;
			for (k = 0; k < Categories.size(); k++)
			{
				//				tempD = abs(preCateg[j].averageX - Categories[k].averageX)+abs(preCateg[j].averageY - Categories[k].averageY);
				//计算当前行聚类中第j个聚类与上次聚类的距离
				tempD = sqrt((preCateg[j].averageX - Categories[k].averageX) * (preCateg[j].averageX - Categories[k].averageX)
					+ (preCateg[j].averageY - Categories[k].averageY) * (preCateg[j].averageY - Categories[k].averageY));
				//找出最小距离和相应的索引ID
				if (tempD < minD)
				{
					ID = k;
					minD = tempD;
				}
			}
			if (minD < averageD)
			{
				//将当前聚类归入Categories中距离最近的索引ID的聚类，并更新points中相应位置的平均点
				Categories[ID].CategID.push_back(cvPoint(preCateg[j].averageX, preCateg[j].averageY));
				points[ID].points.push_back(cvPoint(preCateg[j].averageX, preCateg[j].averageY));
			}
			else
			{
				//将当前preCateg中第j个聚类作为新的类别加入Categories，并在points中添加新的类平均点
				tempPoint.points.clear();
				preCateg[j].ID = Categories.size() + 1;
				Categories.push_back(preCateg[j]);
				tempPoint.points.push_back(cvPoint(preCateg[j].averageX, preCateg[j].averageY));
				points.push_back(tempPoint);
			}

		}

		//更新Categories为当前行聚类的平均坐标，为计算下一次的聚类与该行聚类的距离做准备
		for (j = 0; j < Categories.size(); j++)
		{
			l = 0;
			m = 0;
			if (Categories[j].CategID.size() != 0)
			{
				for (k = 0; k < Categories[j].CategID.size(); k++)
				{
					l += Categories[j].CategID[k].x;
					m += Categories[j].CategID[k].y;
				}
				Categories[j].averageX = l / Categories[j].CategID.size();
				Categories[j].averageY = m / Categories[j].CategID.size();
			}
			Categories[j].CategID.clear();
		}
		preCateg.clear();
	}

	for (i = Height / 2 - 1; i >= areaDegree - 1; i = i - areaDegree)
	{
		preCateg = BaseCluster(featureimage, i, areaDegree, areaExtent);
		averageD = Width / (1.6 * preCateg.size()) + areaDegree;
		for (j = 0; j < preCateg.size(); j++)
		{
			minD = 4096;
			for (k = 0; k < Categories.size(); k++)
			{
				//				tempD = abs(preCateg[j].averageX - Categories[k].averageX)+abs(preCateg[j].averageY - Categories[k].averageY);
				tempD = sqrt((preCateg[j].averageX - Categories[k].averageX) * (preCateg[j].averageX - Categories[k].averageX)
					+ (preCateg[j].averageY - Categories[k].averageY) * (preCateg[j].averageY - Categories[k].averageY));
				if (tempD < minD)
				{
					ID = k;
					minD = tempD;
				}
			}
			if (minD < averageD)
			{
				Categories[ID].CategID.push_back(cvPoint(preCateg[j].averageX, preCateg[j].averageY));
				points[ID].points.push_back(cvPoint(preCateg[j].averageX, preCateg[j].averageY));
			}
			else
			{
				tempPoint.points.clear();
				preCateg[j].ID = Categories.size() + 1;
				Categories.push_back(preCateg[j]);
				tempPoint.points.push_back(cvPoint(preCateg[j].averageX, preCateg[j].averageY));
				points.push_back(tempPoint);
			}

		}
		for (j = 0; j < Categories.size(); j++)
		{
			l = 0;
			m = 0;
			if (Categories[j].CategID.size() != 0)
			{
				for (k = 0; k < Categories[j].CategID.size(); k++)
				{
					l += Categories[j].CategID[k].x;
					m += Categories[j].CategID[k].y;
				}
				Categories[j].averageX = l / Categories[j].CategID.size();
				Categories[j].averageY = m / Categories[j].CategID.size();
			}
			Categories[j].CategID.clear();
		}
		preCateg.clear();
	}
	for (i = 0; i < points.size(); i++)
	{
		k = 0;
		m = 0;
		for (j = 0; j < points[i].points.size(); j++)
		{
			k += points[i].points[j].x;
			m += points[i].points[j].y;
		}
		n = points[i].points.size();
		points[i].averageX = k / n;
		points[i].averageY = m / n;
		points[i].count = n;
	}

	return points;
}

Mat CImgPro::skeletonization(Mat& img, Cluster& points)
{

	Mat outimg;
	thinning(img, outimg, ximgproc::THINNING_ZHANGSUEN);

	for (int i = 0; i < outimg.rows; i++) {
		for (int j = 0; j < outimg.cols; j++) {
			if (outimg.at<uchar>(i, j) != 0) {
				points.points.push_back(Point(j, i));
			}
		}
	}

	return outimg;
}

//////////////////////////////////////////////////
//37 pixel mask:    ooo       3 by 3 mask:  ooo
//                 ooooo                    ooo
//                ooooooo                   ooo
//                ooooooo
//                ooooooo
//                 ooooo
//                  ooo
/////////////////////////////////////////////////
Mat CImgPro::My_SUSAN(Mat& src, int thresh, int k, Cluster& points)
{
	//Mat dst(src.size(), CV_8UC1, Scalar(0));

	Mat grayImg_padded;
	//对灰度图像进行3像素点填充,以便37掩膜遍历完图像上的所有像素点
	copyMakeBorder(src, grayImg_padded, 3, 3, 3, 3, BORDER_REFLECT);
	Mat dst(grayImg_padded.size(), CV_8UC1, Scalar(0));
	//模版 x 和 y的坐标的偏移量
	int OffSetX[37] =
	{
				-3, -3, -3,
			-2, -2, -2, -2, -2,
		-1, -1, -1, -1, -1, -1, -1,
			 0, 0, 0, 0, 0, 0, 0,
			 1, 1, 1, 1, 1, 1, 1,
				2, 2, 2, 2, 2,
				   3, 3, 3
	};

	int OffSetY[37] =
	{
				-1, 0, 1,
			-2, -1, 0, 1, 2,
		-3, -2, -1, 0, 1, 2, 3,
		-3, -2, -1, 0, 1, 2, 3,
		-3, -2, -1, 0, 1, 2, 3,
			-2, -1, 0, 1, 2,
				-1, 0, 1
	};


	for (int i = 3; i < grayImg_padded.rows - 3; i++)
	{
		for (int j = 3; j < grayImg_padded.cols - 3; j++)
		{
			//same表示近似像素的数量
			int same = 0, brightness_differ;
			float last_temp = 0;
			for (int k = 0; k < 37; k++)
			{
				if (OffSetX[k] != 0 && OffSetY[k] != 0)		//去除中心点影响
				{
					brightness_differ = abs(grayImg_padded.at<uchar>(i + OffSetX[k], j + OffSetY[k]) - grayImg_padded.at<uchar>(i, j));

					if (brightness_differ < thresh)		//小于thresh时把其归入USAN区域
					{
						float temp = ((float)brightness_differ) / ((float)thresh);
						temp = temp * temp * temp * temp * temp * temp;
						temp = exp(-temp);
						temp = temp + last_temp;
						last_temp = temp;		//储存上次计算值
					}

				}
			}

			float n = last_temp;		//USAN区域值

			//int g = k * grayImg_padded.at<uchar>(i, j);		//响应阈值
			int g = k;
			if (n < g)
			{
				dst.at<uchar>(i, j) = g - n;
			}
		}
	}

	//非极大值抑制
	int x[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
	int y[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
	for (int i = 3; i < dst.rows - 3; i++)
	{
		for (int j = 3; j < dst.cols - 3; j++)
		{
			int flag = 0;
			for (int k = 0; k < 8; k++)
			{
				if (dst.at<uchar>(i, j) <= dst.at<uchar>(i + x[k], j + y[k]))
				{
					//如果邻域内存在近似像素的值比中心位置的近似像素的值高，则进行极大值抑制
					flag = 1;
					break;
				}
			}

			if (flag == 0)
			{
				dst.at<uchar>(i, j) = 255;
				points.points.push_back(Point(j, i));		//注意！！！！！dst.at<uchar>(i, j)是指图像上第i行第j列的像素值,所以要反过来
			}
			else
			{
				dst.at<uchar>(i, j) = 0;
			}
		}
	}


	// 定义 ROI
	Rect roi(3, 3, src.cols, src.rows);

	// 裁剪填充区域，获取 ROI 区域的 Mat 对象
	Mat featureImg = dst(roi);

	//Rect roi(1, 3, src.cols+4, src.rows);//此处将图像宽扩充四个像素点 1/20*width
	//Mat featureImg = dst(roi);

	return featureImg;
}

vector<CImgPro::Cluster> CImgPro::ComparePoints(vector<Cluster>& points)
{
	Cluster max1, max2, max3;
	vector<Cluster> cmpPoints;
	int a = 0, b = 0, c = 0;
	max1 = points[0];


	if (points.size() > 2)
	{
		for (int i = 1; i < points.size(); i++)
		{
			if (points[i].points.size() > max1.points.size())
			{
				max3 = max2;
				max2 = max1;
				max1 = points[i];

				c = b;
				b = a;
				a = i;
			}
			else if (points[i].points.size() > max2.points.size()) {
				max3 = max2;
				max2 = points[i];
				c = b;
				b = i;
			}
			else if (points[i].points.size() > max3.points.size()) {
				max3 = points[i];
				c = i;
			}
		}

		int min = (a < b ? a : b) < c ? (a < b ? a : b) : c;
		int max = (a > b ? a : b) > c ? (a > b ? a : b) : c;
		int mid = (a + b + c) - min - max;

		cmpPoints.push_back(points[min]);
		cmpPoints.push_back(points[mid]);
		cmpPoints.push_back(points[max]);
	}
	else
	{
		cmpPoints.push_back(points[0]);
		cmpPoints.push_back(points[1]);
	}


	return cmpPoints;
}

vector<CImgPro::Cluster> CImgPro::Bisecting_Kmeans(Cluster& points, int k, float perCof)
{
	vector<Cluster> clusters;
	Cluster tempPoint;

	clusters.push_back(points);

	while (clusters.size() < k)
	{
		// 初始化最大 SSE（平方误差和）为负无穷，以及 SSE 最大的簇的索引为 -1
		double max_sse = -1.0;
		int max_cluster_idx = -1;
		// 对每个簇计算 SSE，并找到 SSE 最大的簇
		for (int i = 0; i < clusters.size(); ++i)
		{
			vector<Point> cur_cluster = clusters[i].points;
			Point c = centroid(cur_cluster); // 计算簇的中心点
			double sse = 0.0;
			for (const auto& p : cur_cluster)
			{
				sse += pow(euclidean_distance(p, c), 2); // 计算该点与中心点之间的距离的平方并累加
			}
			if (sse > max_sse)
			{ // 如果计算得到的 SSE 更大，则更新最大值以及对应的簇的索引
				max_sse = sse;
				max_cluster_idx = i;
			}
		}

		// 获取要被拆分的SSE最大的簇
		vector<Point>& cluster_to_split = clusters[max_cluster_idx].points;// cluster_to_split 的任何修改将直接反映到实际参数 clusters[max_cluster_idx].points 上
		// 初始化新簇
		vector<Point> new_cluster = { cluster_to_split.front() };
		Point c = centroid(cluster_to_split); // 计算要被拆分簇的中心点
		cluster_to_split.erase(cluster_to_split.begin());
		for (const auto& p : cluster_to_split)
		{
			// 将距离新簇中心点更近的点加入新簇
			Point d = centroid(new_cluster);
			double dis1 = euclidean_distance(c, p);
			double dis2 = euclidean_distance(d, p);
			if (dis2 < perCof * dis1)
			{
				new_cluster.push_back(p);
			}
		}
		// 将新簇从要被拆分的簇中移除。注意前面使用了引用，此处的修改会反映到clusters中
		cluster_to_split.erase(remove_if(cluster_to_split.begin(), cluster_to_split.end(),
			[&](const Point& p) { return find(new_cluster.begin(), new_cluster.end(), p) != new_cluster.end(); }),
			cluster_to_split.end());
		// 将新簇添加到簇集合中
		tempPoint.points = new_cluster;
		clusters.push_back(tempPoint);
		new_cluster.clear();
	}

	return clusters;
}

vector<CImgPro::Cluster> CImgPro::Gaussian_Mixture_Model(Cluster& points, int numCluster, ml::EM::Types covarianceType)
{
	int sampleCount = points.points.size();
	Mat matPoints(sampleCount, 2, CV_32FC1);
	Mat labels;

	for (int i = 0; i < sampleCount; i++)
	{
		matPoints.at<float>(i, 0) = points.points[i].y;
		matPoints.at<float>(i, 1) = points.points[i].x;
	}

	Ptr<ml::EM> em_model = ml::EM::create();
	em_model->setClustersNumber(numCluster);
	em_model->setCovarianceMatrixType(covarianceType);
	em_model->setTermCriteria(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.1));
	em_model->trainEM(matPoints, noArray(), labels, noArray());



	vector<Cluster> cluster(numCluster);
	for (int i = 0; i < sampleCount; i++) {
		int index = labels.at<int>(i);
		Point p = Point(matPoints.at<float>(i, 0), matPoints.at<float>(i, 1));
		cluster[index].points.push_back(p);
	}

	return cluster;
}

void CImgPro::Hough_Line(vector<Cluster>& clusters, Mat& outimg)
{
	int i, j, k, l = 0, Maxcount = 0, slopeNum[10] = { 0 }, j1;
	double cenY = 0.0, cenX = 0.0, Lthreshold = 0.0, Hthreshold = 0.0, step, slope, tempslope, sum[10] = { 0 };
	vector<double> finalslope;

	for (i = 0; i < clusters.size(); i++)
	{
		for (auto& p : clusters[i].points) {
			cenX += p.x;
			cenY += p.y;
		}
		cenX = cenX / clusters[i].points.size();
		cenY = cenY / clusters[i].points.size();
		Lthreshold = -2;
		Hthreshold = 4;
		step = (Hthreshold - Lthreshold) / 7;

		do
		{
			for (j = 0; j < clusters[i].points.size(); j++)
			{
				if (clusters[i].points[j].x == cenX)
				{
					tempslope = 2;
				}
				else
				{
					slope = (clusters[i].points[j].y - cenY) / (clusters[i].points[j].x - cenX);
					if (slope > 1 || slope < -1)
					{
						tempslope = 1 / slope + 2;
					}
					else
					{
						tempslope = slope;
					}
				}
				for (k = 0; k < 7; k++)
				{
					if (tempslope >= Lthreshold + k * step && tempslope <= Lthreshold + (k + 1) * step)
					{
						slopeNum[k]++;
						sum[k] += tempslope;
					}
				}
			}
			for (k = 0; k < 7; k++)
			{
				if (Maxcount < slopeNum[k])
				{
					Maxcount = slopeNum[k];
					j1 = k;
				}
			}
			tempslope = sum[j1] / slopeNum[j1];
			if (j1 == 6)
			{
				Hthreshold = Lthreshold + (j1 + 1) * step;
				Lthreshold = Lthreshold + (j1 - 2) * step;
				step = (Hthreshold - Lthreshold) / 7;
			}
			else if (j1 == 0)
			{
				Hthreshold = Lthreshold + (j1 + 3) * step;
				Lthreshold = Lthreshold + (j1)*step;
				step = (Hthreshold - Lthreshold) / 7;
			}
			else
			{
				Hthreshold = Lthreshold + (j1 + 2) * step;
				Lthreshold = Lthreshold + (j1 - 1) * step;
				step = (Hthreshold - Lthreshold) / 7;
			}
			for (k = 0; k < 10; k++)
			{
				slopeNum[k] = 0;
				sum[k] = 0;
			}
		} while (step > 0.6);
		if ((tempslope > 1 && tempslope != 2) || tempslope < -1)
		{
			finalslope.push_back(1 / (tempslope - 2));
		}

		else if (tempslope == 2)
		{
			finalslope.push_back(999999);
		}
		else
		{
			finalslope.push_back(tempslope);;
		}
		Scalar color = CV_RGB(255, 0, 0);
		line(outimg, Point(cenX - cenY / finalslope[l], 0), Point((outimg.rows - cenY) / finalslope[l] + cenX, outimg.rows), color, 10, 8, 0);
		l++;
	}

	finalslope.clear();
}

/********************************************************************************************************************/


