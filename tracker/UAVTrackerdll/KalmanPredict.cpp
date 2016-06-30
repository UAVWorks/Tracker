#include"KalmanPredict.h"
kalman::kalman()
{
	KF.init(stateNum, measureNum, 0);
	KF.transitionMatrix = (Mat_<float>(4, 4) <<
		1, 0, 1, 0,
		0, 1, 0, 1,
		0, 0, 1, 0,
		0, 0, 0, 1);//元素导入矩阵，按行;
	setIdentity(KF.measurementMatrix);		//置为单位矩阵
	setIdentity(KF.processNoiseCov, Scalar::all(1e-5));
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(KF.errorCovPost, Scalar::all(1));
	randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));
}
void kalman::kalmaninit(Point2f& initpoint)
{
	KF.statePost.at<float>(0) =(float) initpoint.x;
	KF.statePost.at<float>(1) = (float)initpoint.y;
}
void kalman::kalmanpredict(Point2f& measurept, Point2f& predictpt)
{

	Mat prediction;
	KF.predict();
	measurement.at<float>(0) = (float)measurept.x;
	measurement.at<float>(1) = (float)measurept.y;
	prediction=KF.correct(measurement);
	predictpt = Point2f(prediction.at<float>(0), prediction.at<float>(1));
	centerpt = Point2f(prediction.at<float>(0), prediction.at<float>(1));
}