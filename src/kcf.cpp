#include "kcf.h"
#include <numeric>

void KCF_Tracker::init(cv::Mat &img, BBox_c &bbox)
{
    p_pose = bbox;

    cv::Mat input;
    if (img.channels() == 3){
        cv::cvtColor(img, input, CV_BGR2GRAY);
        input.convertTo(input, CV_32FC1);
    }else
        img.convertTo(input, CV_32FC1);

    // don't need too large image
    if (bbox.w*bbox.h > 100.*100.) {
        std::cout << "resizing image by factor of 2" << std::endl;
        p_resize_image = true;
        p_pose.scale(0.5);
        cv::resize(input, input, cv::Size(0,0), 0.5, 0.5, cv::INTER_CUBIC);
    }

    p_windows_size[0] = floor(p_pose.w * (1. + p_padding));
    p_windows_size[1] = floor(p_pose.h * (1. + p_padding));

    p_output_sigma = std::sqrt(p_pose.w*p_pose.h) * p_output_sigma_factor / static_cast<double>(p_cell_size);

    //window weights, i.e. labels
    p_yf = fft2(gaussian_shaped_labels(p_output_sigma, p_windows_size[0]/p_cell_size, p_windows_size[1]/p_cell_size));

    p_cos_window = cosine_window_function(p_yf.cols, p_yf.rows);

    //obtain a sub-window for training initial model
    cv::Mat patch = get_subwindow(input, p_pose.cx, p_pose.cy, p_windows_size[0], p_windows_size[1]);
    p_model_xf = fft2(p_fhog.extract(patch, 2, p_cell_size, 9), p_cos_window);
    //Kernel Ridge Regression, calculate alphas (in Fourier domain)
    ComplexMat kf = gaussian_correlation(p_model_xf, p_model_xf, p_kernel_sigma, true);

    p_model_alphaf = p_yf / (kf + p_lambda);   //equation for fast training

//    p_model_alphaf_num = p_yf * kf;
//    p_model_alphaf_den = kf * (kf + p_lambda);
//    p_model_alphaf = p_model_alphaf_num / p_model_alphaf_den;
}

void KCF_Tracker::setTrackerPose(BBox_c &bbox, cv::Mat & img)
{
    init(img, bbox);
}

void KCF_Tracker::updateTrackerPosition(BBox_c &bbox)
{
    if (p_resize_image) {
        BBox_c tmp = bbox;
        tmp.scale(0.5);
        p_pose.cx = tmp.cx;
        p_pose.cy = tmp.cy;
    } else {
        p_pose.cx = bbox.cx;
        p_pose.cy = bbox.cy;
    }
}

BBox_c KCF_Tracker::getBBox()
{
    if (p_resize_image) {
        BBox_c tmp = p_pose;
        tmp.scale(2);
        return tmp;
    } else
        return p_pose;
}

void KCF_Tracker::track(cv::Mat &img)
{
    cv::Mat input;
    if (img.channels() == 3){
        cv::cvtColor(img, input, CV_BGR2GRAY);
        input.convertTo(input, CV_32FC1);
    }else
        img.convertTo(input, CV_32FC1);

    // don't need too large image
    if (p_resize_image)
        cv::resize(input, input, cv::Size(0,0), 0.5, 0.5, cv::INTER_CUBIC);

    cv::Mat patch = get_subwindow(input, p_pose.cx, p_pose.cy, p_windows_size[0], p_windows_size[1]);
    ComplexMat zf = fft2(p_fhog.extract(patch, 2, p_cell_size, 9), p_cos_window);
    ComplexMat kzf = gaussian_correlation(zf, p_model_xf, p_kernel_sigma);
    cv::Mat response = ifft2(p_model_alphaf*kzf);
    //std::cout << response << std::endl;

    /* target location is at the maximum response. we must take into
    account the fact that, if the target doesn't move, the peak
    will appear at the top-left corner, not at the center (this is
    discussed in the paper). the responses wrap around cyclically. */
    double min_val, max_val;
    cv::Point2i min_loc, max_loc;
    cv::minMaxLoc(response, &min_val, &max_val, &min_loc, &max_loc);

    if (max_loc.y > zf.rows/2) //wrap around to negative half-space of vertical axis
        max_loc.y = max_loc.y - zf.rows;
    if (max_loc.x > zf.cols/2) //same for horizontal axis
        max_loc.x = max_loc.x - zf.cols;

    //shift bbox, no scale change
    p_pose.cx += p_cell_size * max_loc.x;
    p_pose.cy += p_cell_size * max_loc.y;

    //obtain a subwindow for training at newly estimated target position
    patch = get_subwindow(input, p_pose.cx, p_pose.cy, p_windows_size[0], p_windows_size[1]);
    ComplexMat xf = fft2(p_fhog.extract(patch, 2, p_cell_size, 9), p_cos_window);
    //Kernel Ridge Regression, calculate alphas (in Fourier domain)
    ComplexMat kf = gaussian_correlation(xf, xf, p_kernel_sigma, true);

    ComplexMat alphaf = p_yf / (kf + p_lambda); //equation for fast training

    //subsequent frames, interpolate model
    p_model_xf = p_model_xf * (1. - p_interp_factor) + xf * p_interp_factor;
    p_model_alphaf = p_model_alphaf * (1. - p_interp_factor) + alphaf * p_interp_factor;

//    ComplexMat alphaf_num = p_yf * kf;
//    ComplexMat alphaf_den = kf * (kf + p_lambda);
//    p_model_alphaf_num = p_model_alphaf_num * (1. - p_interp_factor) + (p_yf * kf) * p_interp_factor;
//    p_model_alphaf_den = p_model_alphaf_den * (1. - p_interp_factor) + kf * (kf + p_lambda) * p_interp_factor;
//    p_model_alphaf = p_model_alphaf_num / p_model_alphaf_den;
}

// ****************************************************************************

cv::Mat KCF_Tracker::gaussian_shaped_labels(double sigma, int dim1, int dim2)
{
    cv::Mat labels(dim2, dim1, CV_32FC1);
    int range_y[2] = {-dim2 / 2, dim2 - dim2 / 2};
    int range_x[2] = {-dim1 / 2, dim1 - dim1 / 2};

    double sigma_s = sigma*sigma;

    for (int y = range_y[0], j = 0; y < range_y[1]; ++y, ++j){
        float * row_ptr = labels.ptr<float>(j);
        double y_s = y*y;
        for (int x = range_x[0], i = 0; x < range_x[1]; ++x, ++i){
            row_ptr[i] = std::exp(-0.5 * (y_s + x*x) / sigma_s);
        }
    }

    //rotate so that 1 is at top-left corner (see KCF paper for explanation)
    cv::Mat rot_labels = circshift(labels, range_x[0], range_y[0]);
    //sanity check, 1 at top left corner
    assert(rot_labels.at<float>(0,0) >= 1.f - 1e-10f);

    return rot_labels;
}

cv::Mat KCF_Tracker::circshift(const cv::Mat &patch, int x_rot, int y_rot)
{
    cv::Mat rot_patch(patch.size(), CV_32FC1);
    cv::Mat tmp_x_rot(patch.size(), CV_32FC1);

    //circular rotate x-axis
    if (x_rot < 0) {
        //move part that does not rotate over the edge
        cv::Range orig_range(-x_rot, patch.cols);
        cv::Range rot_range(0, patch.cols - (-x_rot));
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));

        //rotated part
        orig_range = cv::Range(0, -x_rot);
        rot_range = cv::Range(patch.cols - (-x_rot), patch.cols);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));
    }else if (x_rot > 0){
        //move part that does not rotate over the edge
        cv::Range orig_range(0, patch.cols - x_rot);
        cv::Range rot_range(x_rot, patch.cols);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));

        //rotated part
        orig_range = cv::Range(patch.cols - x_rot, patch.cols);
        rot_range = cv::Range(0, x_rot);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));
    }else {    //zero rotation
        //move part that does not rotate over the edge
        cv::Range orig_range(0, patch.cols);
        cv::Range rot_range(0, patch.cols);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));
    }

    //circular rotate y-axis
    if (y_rot < 0) {
        //move part that does not rotate over the edge
        cv::Range orig_range(-y_rot, patch.rows);
        cv::Range rot_range(0, patch.rows - (-y_rot));
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));

        //rotated part
        orig_range = cv::Range(0, -y_rot);
        rot_range = cv::Range(patch.rows - (-y_rot), patch.rows);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));
    }else if (y_rot > 0){
        //move part that does not rotate over the edge
        cv::Range orig_range(0, patch.rows - y_rot);
        cv::Range rot_range(y_rot, patch.rows);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));

        //rotated part
        orig_range = cv::Range(patch.rows - y_rot, patch.rows);
        rot_range = cv::Range(0, y_rot);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));
    }else { //zero rotation
        //move part that does not rotate over the edge
        cv::Range orig_range(0, patch.rows);
        cv::Range rot_range(0, patch.rows);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));
    }

    return rot_patch;
}

ComplexMat KCF_Tracker::fft2(const cv::Mat &input)
{
    cv::Mat complex_result;
//    cv::Mat padded;                            //expand input image to optimal size
//    int m = cv::getOptimalDFTSize( input.rows );
//    int n = cv::getOptimalDFTSize( input.cols ); // on the border add zero pixels
//    copyMakeBorder(input, padded, 0, m - input.rows, 0, n - input.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
//    cv::dft(padded, complex_result, cv::DFT_COMPLEX_OUTPUT);
//    return ComplexMat(complex_result(cv::Range(0, input.rows), cv::Range(0, input.cols)));

    cv::dft(input, complex_result, cv::DFT_COMPLEX_OUTPUT);
    return ComplexMat(complex_result);
}

ComplexMat KCF_Tracker::fft2(const std::vector<cv::Mat> &input, const cv::Mat &cos_window)
{
    int n_channels = input.size();
    ComplexMat result(input[0].rows, input[0].cols, n_channels);
    for (int i = 0; i < n_channels; ++i){
        cv::Mat complex_result;
//        cv::Mat padded;                            //expand input image to optimal size
//        int m = cv::getOptimalDFTSize( input[0].rows );
//        int n = cv::getOptimalDFTSize( input[0].cols ); // on the border add zero pixels

//        copyMakeBorder(input[i].mul(cos_window), padded, 0, m - input[0].rows, 0, n - input[0].cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
//        cv::dft(padded, complex_result, cv::DFT_COMPLEX_OUTPUT);
//        result.set_channel(i, complex_result(cv::Range(0, input[0].rows), cv::Range(0, input[0].cols)));

        cv::dft(input[i].mul(cos_window), complex_result, cv::DFT_COMPLEX_OUTPUT);
        result.set_channel(i, complex_result);
    }
    return result;
}

cv::Mat KCF_Tracker::ifft2(const ComplexMat &inputf)
{

    cv::Mat real_result;
    if (inputf.n_channels == 1){
        cv::dft(inputf.to_cv_mat(), real_result, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    } else {
        std::vector<cv::Mat> mat_channels = inputf.to_cv_mat_vector();
        std::vector<cv::Mat> ifft_mats(inputf.n_channels);
        for (int i = 0; i < inputf.n_channels; ++i) {
            cv::dft(mat_channels[i], ifft_mats[i], cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
        }
        cv::merge(ifft_mats, real_result);
    }
    return real_result;
}

//hann window actually (Power-of-cosine windows)
cv::Mat KCF_Tracker::cosine_window_function(int dim1, int dim2)
{
    cv::Mat m1(1, dim1, CV_32FC1), m2(dim2, 1, CV_32FC1);
    double N_inv = 1./(static_cast<double>(dim1)-1.);
    for (int i = 0; i < dim1; ++i)
        m1.at<float>(i) = 0.5*(1. - std::cos(2. * CV_PI * static_cast<double>(i) * N_inv));
    N_inv = 1./(static_cast<double>(dim2)-1.);
    for (int i = 0; i < dim2; ++i)
        m2.at<float>(i) = 0.5*(1. - std::cos(2. * CV_PI * static_cast<double>(i) * N_inv));
    cv::Mat ret = m2*m1;
    return ret;
}

// Returns sub-window of image input centered at [cx, cy] coordinates),
// with size [width, height]. If any pixels are outside of the image,
// they will replicate the values at the borders.
cv::Mat KCF_Tracker::get_subwindow(const cv::Mat &input, int cx, int cy, int width, int height)
{
    cv::Mat patch;

    int x1 = cx - width/2;
    int y1 = cy - height/2;
    int x2 = cx + width/2;
    int y2 = cy + height/2;

    //out of image
    if (x1 >= input.cols || y1 >= input.rows || x2 < 0 || y2 < 0) {
        patch.create(height, width, CV_32FC1);
        patch.setTo(0.f);
        return patch;
    }

    int top = 0, bottom = 0, left = 0, right = 0;

    //fit to image coordinates, set border extensions;
    if (x1 < 0) {
        left = -x1;
        x1 = 0;
    }
    if (y1 < 0) {
        top = -y1;
        y1 = 0;
    }
    if (x2 >= input.cols) {
        right = x2 - input.cols + width % 2;
        x2 = input.cols;
    } else
        x2 += width % 2;

    if (y2 >= input.rows) {
        bottom = y2 - input.rows + height % 2;
        y2 = input.rows;
    } else
        y2 += height % 2;

    if (x2 - x1 == 0 || y2 - y1 == 0)
        patch = cv::Mat::zeros(height, width, CV_32FC1);
    else
        cv::copyMakeBorder(input(cv::Range(y1, y2), cv::Range(x1, x2)), patch, top, bottom, left, right, cv::BORDER_REPLICATE);

    //sanity check
    assert(patch.cols == width && patch.rows == height);

    return patch;
}

ComplexMat KCF_Tracker::gaussian_correlation(const ComplexMat &xf, const ComplexMat &yf, double sigma, bool auto_correlation)
{
    float xf_sqr_norm = xf.sqr_norm();
    float yf_sqr_norm = auto_correlation ? xf_sqr_norm : yf.sqr_norm();

    ComplexMat xyf = auto_correlation ? xf.sqr_mag() : xf * yf.conj();

    //ifft2 and sum over 3rd dimension, we dont care about individual channels
    cv::Mat xy_sum(xf.rows, xf.cols, CV_32FC1);
    xy_sum.setTo(0);
    cv::Mat ifft2_res = ifft2(xyf);
    for (int y = 0; y < xf.rows; ++y) {
        float * row_ptr = ifft2_res.ptr<float>(y);
        float * row_ptr_sum = xy_sum.ptr<float>(y);
        for (int x = 0; x < xf.cols; ++x){
            row_ptr_sum[x] = std::accumulate((row_ptr + x*ifft2_res.channels()), (row_ptr + x*ifft2_res.channels() + ifft2_res.channels()), 0.f);
        }
    }

    float numel_xf_inv = 1.f/(xf.cols * xf.rows * xf.n_channels);
    cv::Mat tmp;
    cv::exp(- 1.f / (sigma * sigma) * cv::max((xf_sqr_norm + yf_sqr_norm - 2 * xy_sum) * numel_xf_inv, 0), tmp);

    return fft2(tmp);
}