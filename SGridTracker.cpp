#include "SGridTracker.h"
#include <cstdio>

using namespace std;
using namespace cv;

bool GridTracker::maskPoint(float x, float y)
{
    // 0 indicates that this pixel is in the mask, thus is not useable for new features of OF results
    if (curMask.at<unsigned char>(cv::Point(x, y)) == 0)
        return 1;// means that this feature should be killed
    cv::rectangle(curMask, cv::Point(int(x - MASK_RADIUS / 2 + .5), int(y - MASK_RADIUS / 2 + .5)), cv::Point(int(x + MASK_RADIUS / 2 + .5), int(y + MASK_RADIUS / 2 + .5)), cv::Scalar(0), -1);//define a new image patch
    return 0;// means that this feature can be retained
}

void GridTracker::ParameterInit(cv::Mat& im){
    rows = im.rows;
    cols = im.cols;

    numActiveTracks = 0;
    TRACKING_HSIZE = 8;
    LK_PYRAMID_LEVEL = 4;
    MAX_ITER = 10;
    ACCURACY = 0.1;
    LAMBDA = 0.0;
    
    hgrids_x = GRIDSIZE;
    hgrids_y = GRIDSIZE;
    hgrids_total = hgrids_x * hgrids_y;

    usableFrac = 0.02;

    MaxTracks = MAXTRACKS;

    minAddFrac = 0.1;
    minToAdd = minAddFrac * MaxTracks;

    //upper limit of features of each grid
    fealimitGrid = floor( (1.0*MaxTracks) / (1.0 * hgrids_total));
    DETECT_GAIN = 10;
}

void GridTracker::SpaceInit(cv::Mat& im){
    curMask = cv::Mat(rows, cols, CV_8UC1, Scalar(255));
    lastNumDetectedGridFeatures.resize(hgrids_total, 0);

    for (int i = 0;i < hgrids_total;i++)
    {
        hthresholds.push_back(20);
        detector.push_back(cv::FastFeatureDetector::create(hthresholds[i], true));
        feanumofGrid.push_back(0);
    }
    cv::buildOpticalFlowPyramid(im, prevPyr, Size(2 * TRACKING_HSIZE + 1, 2 * TRACKING_HSIZE + 1), LK_PYRAMID_LEVEL, true);
}

void GridTracker::Calc_optical_flow_if_feas_from_last_frame(cv::Mat& im1){
    vector<uchar> status(allFeas.size(), 1);
    vector<float> error(allFeas.size(), -1);

    //image pyramid of curr frame
    std::vector<cv::Mat> nextPyr;
    cv::buildOpticalFlowPyramid(im1, nextPyr, Size(2 * TRACKING_HSIZE + 1, 2 * TRACKING_HSIZE + 1), LK_PYRAMID_LEVEL, true);

    // perform LK tracking from OpenCV, parameters matter a lot
    points1 = allFeas;

    preFeas.clear();
    trackedFeas.clear();

    cv::calcOpticalFlowPyrLK(prevPyr,
        nextPyr,
        cv::Mat(allFeas),
        cv::Mat(points1),
        cv::Mat(status),
        cv::Mat(error),
        Size(2 * TRACKING_HSIZE + 1, 2 * TRACKING_HSIZE + 1),
        LK_PYRAMID_LEVEL,
        TermCriteria(TermCriteria::COUNT | TermCriteria::EPS,
            MAX_ITER,
            ACCURACY),
        1,//enables optical flow initialization
        LAMBDA);//minEigTheshold

    //renew prevPyr
    prevPyr.swap(nextPyr);

    //clear feature counting for each grid
    for (int k = 0;k < hgrids_total; k++)
    {
        feanumofGrid[k] = 0;
    }

    for (size_t i = 0; i < points1.size(); i++)
    {
        if (status[i] && points1[i].x > usableFrac * cols && points1[i].x < (1.0 - usableFrac) * cols && 
        points1[i].y > usableFrac * rows && points1[i].y < (1.0 - usableFrac) * rows)
        {
            bool shouldKill = maskPoint(points1[i].x, points1[i].y);

            if (shouldKill)
            {
                numActiveTracks--;
            }
            else
            {
                preFeas.push_back(allFeas[i]);
                trackedFeas.push_back(points1[i]);

                int hgridIdx =
                    (int)(floor(points1[i].x / (1.0*cols / hgrids_x)) + hgrids_x * floor(points1[i].y / (1.0*rows / hgrids_y)));

                feanumofGrid[hgridIdx]++;
            }
        }
        else
        {
            numActiveTracks--;
        }
    }
}

void GridTracker::Sampling(cv::Mat& im1){
        //unusedRoom sum
        unusedRoom = 0;

        //hungry sum
        gridsHungry = 0;

        //hungry grids
        vector<int> hungryGrid_idx;
        vector<float> hungryGrid_value;
        
        //room for adding featurs to each grid
        int room = 0;

        //the hungry degree of a whole frame 
        int hungry = 0;

        //set a specific cell as ROI
        Mat sub_image;

        //set the corresponding mask for the previously choosen cell
        Mat sub_mask;

        //keypoints detected from each grids
        vector<vector<cv::KeyPoint> > sub_keypoints;
        sub_keypoints.resize(hgrids_total);

        //patch for computing variance
        cv::Mat patch;
        int midGrid = floor((hgrids_total - 1) / 2.0);

        //the first round resampling on each grid
        for (int q = 0; q < hgrids_total && numActiveTracks < MaxTracks; q++)
        {
            int i = q;
            if (q == 0)
                i = midGrid;
            if (q == midGrid)
                i = 0;
            
            room = fealimitGrid - feanumofGrid[i];
            
            if (room > fealimitGrid*minAddFrac)
            {
                int celly = i / hgrids_x;
                int cellx = i - celly * hgrids_x;
                int row_start = (celly * rows) / hgrids_y;
                int row_size = ((celly + 1) * rows) / hgrids_y - row_start;
                int col_start = (cellx * cols) / hgrids_x;
                int col_size = ((cellx + 1) * cols) / hgrids_x - col_start;
                
                sub_image = im1(cv::Rect(col_start,row_start,col_size,row_size));
                sub_mask = curMask(cv::Rect(col_start,row_start,col_size,row_size));
                
                float lastP = ((float)lastNumDetectedGridFeatures[i] - 15.0 * room) / (15.0 * room);
                
                float newThresh = detector[i]->getThreshold();
                
                newThresh = newThresh + ceil(DETECT_GAIN*lastP);

                if (newThresh > 200)newThresh = 200;
                if (newThresh < 5.0)newThresh = 5.0;
                
                detector[i]->setThreshold(newThresh);
				//detect keypoints in this cell
				detector[i]->detect(sub_image, sub_keypoints[i], sub_mask);
                
                lastNumDetectedGridFeatures[i] = sub_keypoints[i].size();
                
                //!!!KeyPointsFilter::retainBest(sub_keypoints[i], 2 * fealimitGrid);

                //sort features
                //!!!std::sort(sub_keypoints[i].begin(), sub_keypoints[i].end(), rule);

                //for each feature ...
                int n = 0;
                int j = 0;    
                //first round
                
                while(1){
                    if(j<sub_keypoints[i].size() && n<room && numActiveTracks < MaxTracks){

                    }
                    else{
                        break;
                    }
                    cv::Point2f pt = sub_keypoints[i][j].pt;
                    pt.x += col_start;
                    pt.y += row_start;
                    
                    //check is features are being too close
                    if (curMask.at<uchar>(int(pt.y), int(pt.x)) != 0){
                        //runs to here means this feature will be kept
                        int u = int(pt.x - MASK_RADIUS / 2 + .5);
                        int v = int(pt.y - MASK_RADIUS / 2 + .5);
                        int w = int(pt.x + MASK_RADIUS / 2 + .5);
                        int z = int(pt.y + MASK_RADIUS / 2 + .5);
                        cv::rectangle(curMask,cv::Point(u, v),cv::Point(w, z),cv::Scalar(0), -1);

                        allFeas.push_back(pt);
                        ++numActiveTracks;
                        n++;
                    }
                    j++;
                }

                //recollects unused room
                if(n <= room)
                {
                    //records hungry grid's index and how hungry they are
                    hungryGrid_idx.push_back(i);
                    hungryGrid_value.push_back(sub_keypoints[i].size() - j);
                    //sums up to get the total hungry degree
                    hungry += hungryGrid_value.back();
                }
            }
        }

        //begin of second round
        unusedRoom = MaxTracks - numActiveTracks;

        //resampling for the second round
        if (unusedRoom > minToAdd)
        {
            for(int i=0;i<hungryGrid_idx.size();i++)
            {
                int first = hungryGrid_idx[i];
                float second = hungryGrid_value[i];
                
                int celly = first / hgrids_x;
                int cellx = first - celly * hgrids_x;
                int row_start = (celly * rows) / hgrids_y;
                int col_start = (cellx * cols) / hgrids_x;
                
                //how much food can we give it
                room = floor((float)(unusedRoom * hungryGrid_value[first]) / (float)hungry);

                int m=0,j=0;
                while(1){
                    if(m<room && j<sub_keypoints[first].size() - second){

                    }else{
                        break;
                    }
                    
                    //check is features are being too close
                    cv::Point2f pt = sub_keypoints[first][j].pt;
                    //transform grid based position to image based position
                    pt.x += col_start;
                    pt.y += row_start;
                    
                    if (curMask.at<uchar>(int(pt.y), int(pt.x)) != 0){
                       int u = int(pt.x - MASK_RADIUS / 2 + .5);
                       int v = int(pt.y - MASK_RADIUS / 2 + .5);
                       int w = int(pt.x + MASK_RADIUS / 2 + .5);
                       int z = int(pt.y + MASK_RADIUS / 2 + .5);
                       cv::rectangle(curMask,cv::Point(u, v),cv::Point(w, z),cv::Scalar(0), -1);
                       allFeas.push_back(pt);
                       m++;
                    }
                    j++;
                }
            }
        }
}

bool GridTracker::trackerInit(cv::Mat& im)
{
    ParameterInit(im);
    SpaceInit(im);    
    Update(im);

    return 1;
}

bool GridTracker::Update(cv::Mat& im1)
{
    //clear the mask to be white everywhere  
    curMask.setTo(Scalar(255));
    
    //num of feas from last frame
    numActiveTracks = allFeas.size();

    //do optical flow if there are feas from last frame
    if (numActiveTracks > 0)
    {
       Calc_optical_flow_if_feas_from_last_frame(im1);
    }
    else
    {
        //clear feature counting for each grid
        for (int k = 0;k < (hgrids_total);k++)
            feanumofGrid[k] = 0;
    }

    allFeas = trackedFeas;
    int ntoadd = MaxTracks - numActiveTracks;
    
    if (ntoadd > minToAdd)
    {
        Sampling(im1);
    }
    return 1;
}





