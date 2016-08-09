//
// Created by vojirtom on 8/9/16.
//

#ifndef KCF_TRACKER_CNFEAT_H
#define KCF_TRACKER_CNFEAT_H

#include <opencv2/opencv.hpp>

class CNFeat
{
public:
    static std::vector<cv::Mat> extract(const cv::Mat & patch_rgb)
    {
        std::vector<cv::Mat> cn_feat(p_cn_channels);
        for (int i = 0; i < p_cn_channels; ++i) {
            cn_feat[i].create(patch_rgb.size(), CV_32FC1);
        }

        float * ch_ptr[p_cn_channels];
        for (int y = 0; y < patch_rgb.rows; ++y) {
            for (int i = 0; i < p_cn_channels; ++i)
                ch_ptr[i] = cn_feat[i].ptr<float>(y);
            for (int x = 0; x < patch_rgb.cols; ++x) {
                //images in opencv stored in BGR order
                cv::Vec3b bgr_val = patch_rgb.at<cv::Vec3b>(y,x);
                for (int i = 0; i < p_cn_channels; ++i)
                    ch_ptr[i][x] = p_id2feat[rgb2id(bgr_val[2], bgr_val[1], bgr_val[0])][i];
            }
        }
        return cn_feat;
    }

private:
    inline static int rgb2id(int r, int g, int b)
    {   return (r >> 3) + 32*(g >> 3) + 32*32*(b >> 3);     }
    static const int p_cn_channels = 10;
    static float p_id2feat[32768][10];
};


#endif //KCF_TRACKER_CNFEAT_H
