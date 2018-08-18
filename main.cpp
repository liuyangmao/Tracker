#include <iostream>
#include "SGridTracker.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv){
    
    char name[1024];
    cv::Mat img = cv::imread("../frames/0000.png");
    
    GridTracker gt;
    gt.trackerInit(img);
    
    for(int i=1;i<200;i++){
        sprintf(name,"../frames/%04d.png",i);
        cv::Mat temp = cv::imread(name);
        gt.Update(temp);
        sprintf(name,"../debug/%04d.png",i);

        for(int j=0;j<gt.trackedFeas.size();j++)
            cv::circle(temp,gt.trackedFeas[j],2,cv::Scalar(0,255,0),-1);
        cv::imwrite(name,temp);
    }






    return 0;
}
