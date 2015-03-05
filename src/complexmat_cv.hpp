#ifndef COMPLEX_MAT_CV_HPP_213123048309482094
#define COMPLEX_MAT_CV_HPP_213123048309482094

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

template<typename T> class ComplexMat_
{
public:
    int rows = 0;
    int cols = 0;
    int n_channels = 0;

    ComplexMat_()  {}
    ComplexMat_(int _rows, int _cols, int _n_channels) : rows(_rows), cols(_cols), n_channels(_n_channels)
    {
        p_real.resize(n_channels);
        p_imag.resize(n_channels);
    }

    //assuming that mat has 2 channels (real, img)
    ComplexMat_(const cv::Mat & mat) : n_channels(1)
    {
        cols = mat.cols;
        rows = mat.rows;
        p_real.resize(1);
        p_imag.resize(1);
        std::vector<cv::Mat> channel = {p_real[0], p_imag[0]};
        cv::split(mat, channel);
        p_real[0] = channel[0];
        p_imag[0] = channel[1];
    }

    //assuming that mat has 2 channels (real, imag)
    void set_channel(int idx, const cv::Mat & mat)
    {
        std::vector<cv::Mat> channel = {p_real[idx], p_imag[idx]};
        cv::split(mat, channel);
        p_real[idx] = channel[0];
        p_imag[idx] = channel[1];
    }

    T sqr_norm() const
    {
        T sum_sqr_norm = 0;
        for (int i = 0; i < n_channels; ++i){
            sum_sqr_norm += cv::sum(p_real[i].mul(p_real[i]) + p_imag[i].mul(p_imag[i])).val[0];
        }
        return sum_sqr_norm / static_cast<T>(p_real[0].cols*p_real[0].rows);
    }

    ComplexMat_<T> sqr_mag() const
    {
        ComplexMat_<T> ret(rows, cols, n_channels);
        for (int i = 0; i < n_channels; ++i){
            ret.p_real[i] = p_real[i].mul(p_real[i]) + p_imag[i].mul(p_imag[i]);
            ret.p_imag[i] = cv::Mat::zeros(ret.p_real[i].rows, ret.p_real[i].cols, ret.p_real[i].type());
        }
        return ret;
    }

    ComplexMat_<T> conj() const
    {
        ComplexMat_<T> ret(rows, cols, n_channels);
        for (int i = 0; i < n_channels; ++i){
            ret.p_real[i] = p_real[i].clone();
            ret.p_imag[i] = -p_imag[i].clone();
        }
        return ret;
    }

    //return 2 channels (real, imag) for first complex channel
    cv::Mat to_cv_mat() const
    {
        std::vector<cv::Mat> channel = {p_real[0].clone(), p_imag[0].clone()};
        cv::Mat res;
        cv::merge(channel, res);
        return res;
    }
    //return a vector of 2 channels (real, imag) per one complex channel
    std::vector<cv::Mat> to_cv_mat_vector() const
    {
        std::vector<cv::Mat> result;
        result.reserve(n_channels);

        for (int i = 0; i < n_channels; ++i) {
            std::vector<cv::Mat> channel = {p_real[i].clone(), p_imag[i].clone()};
            cv::Mat res;
            cv::merge(channel, res);
            result.push_back(res);
        }

        return result;
    }

    //element-wise per channel multiplication, division and addition
    ComplexMat_<T> operator*(const ComplexMat_<T> & rhs) const
    {
        ComplexMat_<T> ret(rows, cols, n_channels);
        for (int i = 0; i < n_channels; ++i){
            ret.p_real[i] = p_real[i].mul(rhs.p_real[i]) - p_imag[i].mul(rhs.p_imag[i]);
            ret.p_imag[i] = p_imag[i].mul(rhs.p_real[i]) + p_real[i].mul(rhs.p_imag[i]);
        }
        return ret;
    }
    ComplexMat_<T> operator/(const ComplexMat_<T> & rhs) const
    {
        ComplexMat_<T> ret(rows, cols, n_channels);
        for (int i = 0; i < n_channels; ++i){
            cv::Mat denominator = rhs.p_real[i].mul(rhs.p_real[i]) + rhs.p_imag[i].mul(rhs.p_imag[i]);
            ret.p_real[i] = (p_real[i].mul(rhs.p_real[i]) + p_imag[i].mul(rhs.p_imag[i]))/denominator;
            ret.p_imag[i] = (p_imag[i].mul(rhs.p_real[i]) - p_real[i].mul(rhs.p_imag[i]))/denominator;
        }
        return ret;
    }
    ComplexMat_<T> operator+(const ComplexMat_<T> & rhs) const
    {
        ComplexMat_<T> ret(rows, cols, n_channels);
        for (int i = 0; i < n_channels; ++i){
            ret.p_real[i] = p_real[i] + rhs.p_real[i];
            ret.p_imag[i] = p_imag[i] + rhs.p_imag[i];
        }

        return ret;
    }

    //multiplying or adding constant
    ComplexMat_<T> operator*(const T & rhs) const
    {
        ComplexMat_<T> ret(rows, cols, n_channels);
        for (int i = 0; i < n_channels; ++i){
            ret.p_real[i] = rhs*p_real[i];
            ret.p_imag[i] = rhs*p_imag[i];
        }
        return ret;
    }
    ComplexMat_<T> operator+(const T & rhs) const
    {
        ComplexMat_<T> ret(rows, cols, n_channels);
        for (int i = 0; i < n_channels; ++i){
            ret.p_real[i] = rhs+p_real[i];
            ret.p_imag[i] = p_imag[i].clone();
        }
        return ret;
    }

    //text output
    friend std::ostream & operator<<(std::ostream & os, const ComplexMat_<T> & mat)
    {
        //for (int i = 0; i < mat.n_channels; ++i){
        for (int i = 0; i < 1; ++i){
            os << "Channel " << i << std::endl;
                    }
        return os;
    }


private:
    std::vector<cv::Mat> p_real;
    std::vector<cv::Mat> p_imag;
};

typedef ComplexMat_<float> ComplexMat;


#endif //COMPLEX_MAT_CV_HPP_213123048309482094