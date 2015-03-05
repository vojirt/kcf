#include <stdlib.h>
#include <opencv2/opencv.hpp>

#include "kcf.h"

void run_tests(KCF_Tracker & tracker, const std::vector<bool> & tests)
{
    int dim1 = 5, dim2 = 5;
    cv::Mat input = (cv::Mat_<float>(dim1, dim2) <<
        0.0183,    0.0821,    0.1353,    0.0821,    0.0183,
        0.0821,    0.3679,    0.6065,    0.3679,    0.0821,
        0.1353,    0.6065,    1.0000,    0.6065,    0.1353,
        0.0821,    0.3679,    0.6065,    0.3679,    0.0821,
        0.0183,    0.0821,    0.1353 ,   0.0821,    0.0183);

    cv::Mat correct_output = (cv::Mat_<float>(dim1, dim2) <<
        1.0000,    0.6065,    0.1353,    0.1353,    0.6065,
        0.6065,    0.3679,    0.0821,    0.0821,    0.3679,
        0.1353,    0.0821,    0.0183,    0.0183,    0.0821,
        0.1353,    0.0821,    0.0183,    0.0183,    0.0821,
        0.6065,    0.3679,    0.0821,    0.0821,    0.3679);


    //test for odd dims
    if (tests[0]) {
        std::cout << "circshift : " << std::endl;
        std::cout << (tracker.circshift(input, -dim1 / 2, -dim2 / 2) == correct_output) << std::endl;
        std::cout << "-----------------------------------" << std::endl;

        std::cout << "gaussian_shaped_labels : " << std::endl << "correct labels" << std::endl << input << std::endl;
        std::cout << (tracker.gaussian_shaped_labels(1., dim1, dim2)) << std::endl;
        std::cout << "correct rotated: " << std::endl << correct_output << std::endl;
        std::cout << "-----------------------------------" << std::endl;
    }
    dim1 = 6, dim2 = 6;
    input = (cv::Mat_<float>(dim1, dim2) <<
        0.0001,    0.0015,    0.0067,    0.0111,    0.0067,    0.0015,
        0.0015,    0.0183,    0.0821,    0.1353,    0.0821,    0.0183,
        0.0067,    0.0821,    0.3679,    0.6065,    0.3679,    0.0821,
        0.0111,    0.1353,    0.6065,    1.0000,    0.6065,    0.1353,
        0.0067,    0.0821,    0.3679,    0.6065,    0.3679,    0.0821,
        0.0015,    0.0183,    0.0821,    0.1353,    0.0821,    0.0183);

    correct_output = (cv::Mat_<float>(dim1, dim2) <<
        1.0000,    0.6065,    0.1353,    0.0111,    0.1353,    0.6065,
        0.6065,    0.3679,    0.0821,    0.0067,    0.0821,    0.3679,
        0.1353,    0.0821,    0.0183,    0.0015,    0.0183,    0.0821,
        0.0111,    0.0067,    0.0015,    0.0001,    0.0015,    0.0067,
        0.1353,    0.0821,    0.0183,    0.0015,    0.0183,    0.0821,
        0.6065,    0.3679,    0.0821,    0.0067,    0.0821,    0.3679);

    //test for even dim
    if (tests[1]) {
        std::cout << "circshift : " << std::endl;
        std::cout << (tracker.circshift(input, -dim1 / 2, -dim2 / 2) == correct_output) << std::endl;
        std::cout << "-----------------------------------" << std::endl;

        std::cout << "gaussian_shaped_labels : " << std::endl << "correct labels" << std::endl << input << std::endl;
        std::cout << (tracker.gaussian_shaped_labels(1., dim1, dim2)) << std::endl;
        std::cout << "correct rotated: " << std::endl << correct_output << std::endl;
        std::cout << "-----------------------------------" << std::endl;
    }

    dim1 = 3, dim2 = 5;
    cv::Mat correct_cos_window = (cv::Mat_<float>(dim2, dim1) <<
        0,         0,         0,
        0,    0.5000,         0,
        0,    1.0000,         0,
        0,    0.5000,         0,
        0,         0,         0);

    if (tests[2]) {
        std::cout << "cosine window : " << std::endl << tracker.cosine_window_function(dim1, dim2) << std::endl;
        std::cout << "correct cosine window: " << std::endl << correct_cos_window << std::endl;
        std::cout << "-----------------------------------" << std::endl;
    }

    dim1 = 5, dim2 = 5;
    input = (cv::Mat_<float>(dim2, dim1) <<
            3,    4,    3,    1,    1,
            2,    8,    8,    1,    6,
            3,    6,    8,    5,    5,
            6,    5,    4,    8,    0,
            5,    9,    6,    9,    3);
    correct_output = (cv::Mat_<float>(3, 3) <<
            3,    3,    4,
            3,    3,    4,
            2,    2,    8);

    int cx = 0, cy = 0;
    int w =3, h = 3;
    if (tests[3]) {
        std::cout << "get sub-window : " << std::endl << tracker.get_subwindow(input, cx, cy, w, h) << std::endl;
        std::cout << "correct sub-window: " << std::endl << correct_output << std::endl;
    }


    dim1 = dim2 = 6;

    input = (cv::Mat_<float>(dim2, dim1) <<
        0.8147,    0.2785,    0.9572,    0.7922,    0.6787,    0.7060,
        0.9058,    0.5469,    0.4854,    0.9595,    0.7577,    0.0318,
        0.1270,    0.9575,    0.8003,    0.6557,    0.7431,    0.2769,
        0.9134,    0.9649,    0.1419,    0.0357,    0.3922,    0.0462,
        0.6324,    0.1576,    0.4218,    0.8491,    0.6555,    0.0971,
        0.0975,    0.9706,    0.9157,    0.9340,    0.1712,    0.8235);

    cv::Mat correct_output_c1 = (cv::Mat_<float>(dim2, dim1) <<
        18.3594,   -0.8008,   -0.8645 ,   1.8725 ,  -0.8645  , -0.8008,
        0.3376,   -0.2076,   -0.1915  ,  0.2603  , -1.0988   , 2.7914,
        -0.0476,   -0.6210,   -0.3654 ,   0.6377 ,  -2.5532  ,  1.0623,
        2.5838,    0.7869,    1.6363  ,  0.6739  ,  1.6363   , 0.7869,
        -0.0476,    1.0623,   -2.5532 ,   0.6377 ,  -0.3654  , -0.6210,
        0.3376,    2.7914,   -1.0988  ,  0.2603  , -0.1915   ,-0.2076);

    cv::Mat correct_output_c2 = (cv::Mat_<float>(dim2, dim1) <<
        0   ,-0.2750,   -1.7024,         0,    1.7024,    0.2750,
        0.6036,   -0.4577 ,  -2.2689  ,  0.1396 ,  -1.1823  ,  0.8429,
        -0.8309,   -0.0995  ,  1.4797  ,  0.6993  ,  1.1071,    0.6369,
        0,    1.3954   , 1.0643   ,      0  , -1.0643 ,  -1.3954,
        0.8309,   -0.6369 ,  -1.1071 ,  -0.6993 ,  -1.4797 ,   0.0995,
        -0.6036,   -0.8429  ,  1.1823  , -0.1396  ,  2.2689 ,   0.4577);

    ComplexMat fft_output = tracker.fft2(input);

    if (tests[4]) {
        cv::Mat complex_result;
        cv::dft(input, complex_result, cv::DFT_COMPLEX_OUTPUT);
        std::cout << "FFT2 Output without conversion: " << std::endl << complex_result << std::endl;
        std::cout << "FFT2 Output ComplexMat: " << std::endl << fft_output << std::endl;
        std::cout << "matlab output real" << std::endl << correct_output_c1 << std::endl;
        std::cout << "matlab output im" << std::endl << correct_output_c2 << std::endl;
    }

    if (tests[5]) {
        cv::Mat complex_result, real_result;
        cv::dft(input, complex_result, cv::DFT_COMPLEX_OUTPUT);
        cv::dft(complex_result, real_result, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

        std::cout << "IFFT2 w/o conversion output : " << std::endl << real_result << std::endl;
        std::cout << "IFFT2 w/o correct output : " << std::endl << input << std::endl;

        cv::Mat out = tracker.ifft2(fft_output);
        std::cout << "IFFT2 w/ output : " << std::endl << out << std::endl;
        std::cout << "correct w/ correct output : " << std::endl << input << std::endl;
    }

    correct_output = (cv::Mat_<float>(dim2, dim1) <<
        31.1807 ,   0.0420 ,   0.1827 ,   0.1789 ,   0.1827 ,   0.0420,
        0.0330  ,  0.0199  ,  0.2615  ,  0.0128  ,  0.1332  ,  0.4180,
        0.0435  ,  0.0312  ,  0.1206  ,  0.0515  ,  0.3840  ,  0.0805,
        0.3315  ,  0.1296  ,  0.1921  ,  0.0366  ,  0.1921  ,  0.1296,
        0.0435  ,  0.0805  ,  0.3840  ,  0.0515  ,  0.1206  ,  0.0312,
        0.0330  ,  0.4180  ,  0.1332  ,  0.0128  ,  0.2615  ,  0.0199);


    if (tests[6]) {
        ComplexMat out = tracker.gaussian_correlation(fft_output, fft_output, 1);
        std::cout << "gauss correlation output : " << std::endl << out << std::endl;
        std::cout << "correct output : " << std::endl << correct_output << std::endl;
    }

    dim1 = dim2 = 10;
    input = (cv::Mat_<float>(dim2, dim1) <<
        0.5383,    0.0844,    0.8693,    0.2399,    0.3377,    0.9421,    0.6491,    0.3685,    0.5085,    0.8759,
        0.9961,    0.3998,    0.5797,    0.1233,    0.9001,    0.9561,    0.7317,    0.6256,    0.5108,    0.5502,
        0.0782,    0.2599,    0.5499,    0.1839,    0.3692,    0.5752,    0.6477,    0.7802,    0.8176,    0.6225,
        0.4427,    0.8001,    0.1450,    0.2400,    0.1112,    0.0598,    0.4509,    0.0811,    0.7948,    0.5870,
        0.1067,    0.4314,    0.8530,    0.4173,    0.7803,    0.2348,    0.5470,    0.9294,    0.6443,    0.2077,
        0.9619,    0.9106,    0.6221,    0.0497,    0.3897,    0.3532,    0.2963,    0.7757,    0.3786,    0.3012,
        0.0046,    0.1818,    0.3510,    0.9027,    0.2417,    0.8212,    0.7447,    0.4868,    0.8116,    0.4709,
        0.7749,    0.2638,    0.5132,    0.9448,    0.4039,    0.0154,    0.1890,    0.4359,    0.5328,    0.2305,
        0.8173,    0.1455,    0.4018,    0.4909,    0.0965,    0.0430,    0.6868,    0.4468,    0.3507,    0.8443,
        0.8687,    0.1361,    0.0760,    0.4893,    0.1320,    0.1690,    0.1835,    0.3063,    0.9390,    0.1948);
    //TODO 2015-02-05 17:04:20+01:00 : dodelat test, vypada ze fhog jeste nefunguje
    if (tests[7]) {
        FHoG fhog_class;
        std::vector<cv::Mat> fhog = fhog_class.extract(input);
        for (int i = 0; i < 3; ++i)
            std::cout << fhog[i] << std::endl;
    }

}

int main()
{
/*
0    cv::Mat circshift(const cv::Mat & patch, int x_rot, int y_rot);
1    cv::Mat gaussian_shaped_labels(double sigma, int dim1, int dim2);
2    cv::Mat cosine_window_function(int dim1, int dim2);
3    cv::Mat get_subwindow(const cv::Mat & input, int cx, int cy, int size_x, int size_y);
4    ComplexMat fft2(const cv::Mat & input);
5    cv::Mat ifft2(const ComplexMat & inputf, cv::Mat * sum_channel = nullptr);
6    ComplexMat gaussian_correlation(const ComplexMat & xf, const ComplexMat & yf, double sigma);
7    FHoG::extract(patch, 2, p_cell_size, 9)
*/
    //                         0  1  2  3  4  5  6  7
    std::vector<bool> tests = {1, 0, 0, 0, 0, 0, 0, 0};

    KCF_Tracker test_tracker;
    run_tests(test_tracker, tests);
    return EXIT_SUCCESS;
}