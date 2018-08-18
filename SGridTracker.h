#pragma once
#include <opencv2/opencv.hpp>

#define VARIANCE 4
#define MAXTRACKS 800
#define GRIDSIZE 15
#define MASK_RADIUS 15

inline bool rule(const cv::KeyPoint& p1, const cv::KeyPoint& p2)
{
    return p1.response > p2.response;
}

class GridTracker
{
public:

    //mask used for ignoring regions of the image in the detector and for maintaining minimal feature distance
    cv::Mat curMask;

    //tracked feas of current frame
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> trackedFeas;//matched Feas of currFrame

    //all feas of current frame
    std::vector<cv::Point2f> allFeas;
    std::vector<cv::Point2f> preFeas;//matched Feas of preFrame

    int rows, cols;

    //num of feas from last frame
    int numActiveTracks;

    int TRACKING_HSIZE, LK_PYRAMID_LEVEL, MAX_ITER, fealimitGrid;

    float ACCURACY, LAMBDA;

    float usableFrac;

    //store image pyramid for re-utilization
    std::vector<cv::Mat> prevPyr;

    //int overflow;
    int MaxTracks;

    //grids devision
    int hgrids_x;
    int hgrids_y;
    int hgrids_total;

    //records feature number of each grids
    std::vector<int> feanumofGrid;

    float minAddFrac, minToAdd;

    int unusedRoom, gridsHungry;

    std::vector<int> lastNumDetectedGridFeatures;

    //feature detector for Fast feature dtection one each gridCols
	std::vector<cv::Ptr<cv::FastFeatureDetector> > detector;

    //thresholds for Fast detection on each grid
    std::vector<int> hthresholds;

    float DETECT_GAIN;

private:
    bool maskPoint(float x, float y);
    void ParameterInit(cv::Mat& im);
    void SpaceInit(cv::Mat& im);
    void Calc_optical_flow_if_feas_from_last_frame(cv::Mat& im);
    void Sampling(cv::Mat& im);

public:
    
    GridTracker(){}
    
    bool Update(cv::Mat& im1);

    bool trackerInit(cv::Mat& im);
};









