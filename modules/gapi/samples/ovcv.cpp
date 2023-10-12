#include <algorithm>
#include <iostream>
#include <cctype>

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/operators.hpp>
#include <opencv2/gapi/infer/ov.hpp>
#include <opencv2/gapi/fluid/core.hpp>
#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/highgui.hpp>

const std::string about =
    "This is an OVCV example";
const std::string keys =
    "{ h help    |  | Print this help message }"
    "{ i input   |  | Path to the input video file }"
    "{ l overlay |  | Path to the overlay video file }";

int main(int argc, char *argv[])
{
    cv::CommandLineParser cmd(argc, argv, keys);
    cmd.about(about);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }
    const std::string input_path = cmd.get<std::string>("input");
    const std::string ovlay_path = cmd.get<std::string>("overlay");
    const cv::Scalar  CHROMAKEY  = cv::Scalar(27,214,43);
    const cv::Size    OUTSZ      = cv::Size( 853, 480);
    const cv::Size    OVLSZ      = cv::Size( 320, 240);

    cv::GMat input, overlay;
    cv::GMat base_image = cv::gapi::resize(input,   OUTSZ);
    cv::GMat over_patch = cv::gapi::resize(overlay, OVLSZ);

    // Extend Overlay with the chroma key color, shift it to the bottom left
    const int add_top = OUTSZ.height - OVLSZ.height;
    const int ext_rgt = OUTSZ.width  - OVLSZ.width;
    cv::GMat over_image = cv::gapi::pad(over_patch, add_top, 0, 0, ext_rgt, CHROMAKEY);

    // Create base and overlay mask. In OpenCV compare needs to be
    // called per-channel...
    cv::GMat bp, gp, rp;
    std::tie(bp, gp, rp) = cv::gapi::split3(over_image);
    cv::GMat bm = bp == cv::GScalar(CHROMAKEY[0]);
    cv::GMat gm = gp == cv::GScalar(CHROMAKEY[1]);
    cv::GMat rm = rp == cv::GScalar(CHROMAKEY[2]);
    cv::GMat gray_mask = bm & gm & rm;
    cv::GMat base_mask = cv::gapi::merge3(gray_mask, gray_mask, gray_mask);
    cv::GMat over_mask = cv::gapi::bitwise_not(base_mask);
    cv::GMat out = (base_image & base_mask) | (over_image & over_mask);

    try {
        auto pipeline = cv::GComputation(cv::GIn(input, overlay), cv::GOut(out))
            .compileStreaming(cv::compile_args(// cv::gapi::use_only{
                        cv::gapi::ov::kernels()// }
                              ));
        auto input_vid = cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(input_path);
        auto ovlay_vid = cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(ovlay_path);
        pipeline.setSource(cv::gin(input_vid, ovlay_vid));
        pipeline.start();

        cv::Mat pic;
        while (pipeline.pull(cv::gout(pic))) {
            cv::imshow("Out", pic);
            cv::waitKey(1);
        }
    } catch (std::exception &ex) {
        std::cout << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
