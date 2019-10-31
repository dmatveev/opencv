// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019 Intel Corporation

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/fluid/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include "opencv2/gapi/streaming/cap.hpp"

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iomanip>

namespace config
{
constexpr char       kWinFaceBeautification[] = "FaceBeautificator";
constexpr char       kWinInput[]              = "Input";
const     cv::Scalar kClrWhite (255, 255, 255);
const     cv::Scalar kClrGreen (  0, 255,   0);
const     cv::Scalar kClrYellow(  0, 255, 255);

constexpr float      kFaceConfThreshold = 0.7f;

const     cv::Size   kGaussKernelSize(5, 5);
constexpr double     kGaussSigma      = 0.0;
constexpr int        kBilatDiameter   = 9;
constexpr double     kBilatSigmaColor = 30.0;
constexpr double     kBilatSigmaSpace = 30.0;
constexpr int        kUnsharpSigma    = 3;
constexpr float      kUnsharpStrength = 0.7f;
constexpr int        kAngDelta        = 1;
constexpr bool       kClosedLine      = true;

const size_t kNumPointsInHalfEllipse = 180 / config::kAngDelta + 1;
} // namespace config

namespace
{
using VectorROI = std::vector<cv::Rect>;
using GArrayROI = cv::GArray<cv::Rect>;
using Contour   = std::vector<cv::Point>;


// Wrapper function
template<typename Tp> inline int toIntRounded(const Tp x)
{
    return static_cast<int>(std::lround(x));
}

template<typename Tp> inline double toDouble(const Tp x)
{
    return static_cast<double>(x);
}

std::string getWeightsPath(const std::string &mdlXMLPath) // mdlXMLPath =
                                                          // "The/Full/Path.xml"
{
    size_t size = mdlXMLPath.size();
    CV_Assert(mdlXMLPath.substr(size - 4, size)           // The last 4 symbols
                  == ".xml");                             // must be ".xml"
    std::string mdlBinPath(mdlXMLPath);
    return mdlBinPath.replace(size - 3, 3, "bin");        // return
                                                          // "The/Full/Path.bin"
}
} // anonymous namespace



namespace custom
{
using TplPtsFaceElements_Jaw = std::tuple<cv::GArray<std::vector<cv::Point>>,
                                          cv::GArray<Contour>>;

// Wrapper-functions
inline cv::Rect getFaceRect(const cv::Point2f &ptfTopLeft,
                            const cv::Point2f &ptfBotRight,
                            const cv::Size &imgSize);
inline cv::Point getRigthCoordinates(const cv::Point2f &ptFloat,
                                     const cv::Rect &faceCoordinates);
inline int getLineInclinationAngleDegrees(const cv::Point &ptLeft,
                                          const cv::Point &ptRight);
inline Contour getForeheadEllipse(const cv::Point &ptJawLeft,
                                  const cv::Point &ptJawRight,
                                  const cv::Point &ptJawMiddle,
                                  const size_t capacity);
inline Contour getEyeEllipse(const cv::Point &ptLeft, const cv::Point &ptRight,
                             const size_t capacity);
inline Contour getPatchedEllipse(const cv::Point &ptLeft,
                                 const cv::Point &ptRight,
                                 const cv::Point &ptUp,
                                 const cv::Point &ptDown);

// Networks
//! [net_decl]
G_API_NET(FaceDetector, <cv::GMat(cv::GMat)>, "face_detector");
G_API_NET(FacialLandmarksDetector, <cv::GMat(cv::GMat)>, "landm_detector");
//! [net_decl]

// Function kernels
G_TYPED_KERNEL(GBilateralFilter, <cv::GMat(cv::GMat,int,double,double)>,
               "custom.faceb12n.bilateralFilter")
{
    static cv::GMatDesc outMeta(cv::GMatDesc in, int, double, double)
    { return in; }
};

G_TYPED_KERNEL(GLaplacian, <cv::GMat(cv::GMat,int)>,
               "custom.faceb12n.Laplacian")
{
    static cv::GMatDesc outMeta(cv::GMatDesc in, int) { return in; }
};

//! [kern_decl]
G_TYPED_KERNEL(GFillPolyGContours, <cv::GMat(cv::GMat,cv::GArray<Contour>)>,
               "custom.faceb12n.fillPolyGContours")
{
    static cv::GMatDesc outMeta(cv::GMatDesc in, cv::GArrayDesc)
    { return in.withType(CV_8U, 1); }
};
//! [kern_decl]

G_TYPED_KERNEL(GPolyLines, <cv::GMat(cv::GMat,cv::GArray<Contour>,bool,
                                     cv::Scalar)>,
               "custom.faceb12n.polyLines")
{
    static cv::GMatDesc outMeta(cv::GMatDesc in, cv::GArrayDesc,bool,cv::Scalar)
    { return in; }
};

G_TYPED_KERNEL(GRectangle, <cv::GMat(cv::GMat,GArrayROI,cv::Scalar)>,
               "custom.faceb12n.rectangle")
{
    static cv::GMatDesc outMeta(cv::GMatDesc in, cv::GArrayDesc, cv::Scalar)
    { return in; }
};

G_TYPED_KERNEL(GFacePostProc, <GArrayROI(cv::GMat,cv::GMat,float)>,
               "custom.faceb12n.faceDetectPostProc")
{
    static cv::GArrayDesc outMeta(const cv::GMatDesc&, const cv::GMatDesc&,
                                  float)
    { return cv::empty_array_desc(); }
};

G_TYPED_KERNEL_M(GLandmPostProc, <TplPtsFaceElements_Jaw(cv::GArray<cv::GMat>,
                                                         GArrayROI)>,
                 "custom.faceb12n.landmDetectPostProc")
{
    static std::tuple<cv::GArrayDesc,cv::GArrayDesc> outMeta(
                const cv::GArrayDesc&, const cv::GArrayDesc&)
    { return std::make_tuple(cv::empty_array_desc(), cv::empty_array_desc()); }
};

//! [kern_m_decl]
using TplFaces_FaceElements  = std::tuple<cv::GArray<Contour>,
                                          cv::GArray<Contour>>;
G_TYPED_KERNEL_M(GGetContours, <TplFaces_FaceElements
                                (cv::GArray<std::vector<cv::Point>>,
                                 cv::GArray<Contour>)>,
                 "custom.faceb12n.getContours")
{
    static std::tuple<cv::GArrayDesc,cv::GArrayDesc> outMeta(
                const cv::GArrayDesc&, const cv::GArrayDesc&)
    { return std::make_tuple(cv::empty_array_desc(), cv::empty_array_desc()); }
};
//! [kern_m_decl]


// OCV_Kernels
// This kernel applies Bilateral filter to an input src with default
//  "cv::bilateralFilter" border argument
GAPI_OCV_KERNEL(GCPUBilateralFilter, custom::GBilateralFilter)
{
    static void run(const cv::Mat &src, const int diameter,
                    const double sigmaColor,
                    const double sigmaSpace, cv::Mat &out)
    { cv::bilateralFilter(src, out, diameter, sigmaColor, sigmaSpace); }
};

// This kernel applies Laplace operator to an input src with default
//  "cv::Laplacian" arguments
GAPI_OCV_KERNEL(GCPULaplacian, custom::GLaplacian)
{
    static void run(const cv::Mat &src, const int ddepth, cv::Mat &out)
    { cv::Laplacian(src, out, ddepth); }
};

// This kernel draws given white filled contours "cnts" on a clear Mat "out"
//  (defined by a Scalar(0)) with standard "cv::fillPoly" arguments.
//  It should be used to create a mask.
// The input Mat seems unused inside the function "run", but it is used deeper
//  in the kernel to define an output size.
//! [kern_impl]
GAPI_OCV_KERNEL(GCPUFillPolyGContours, custom::GFillPolyGContours)
{
    static void run(const cv::Mat &, const std::vector<Contour> &cnts,
                    cv::Mat &out)
    {
        out = cv::Scalar(0);
        cv::fillPoly(out, cnts, config::kClrWhite);
    }
};
//! [kern_impl]

// This kernel draws given contours on an input src with default "cv::polylines"
//  arguments
GAPI_OCV_KERNEL(GCPUPolyLines, custom::GPolyLines)
{
    static void run(const cv::Mat &src, const std::vector<Contour> &cnts,
                    const bool isClosed, const cv::Scalar &color, cv::Mat &out)
    {
        src.copyTo(out);
        cv::polylines(out, cnts, isClosed, color);
    }
};

// This kernel draws given rectangles on an input src with default
//  "cv::rectangle" arguments
GAPI_OCV_KERNEL(GCPURectangle, custom::GRectangle)
{
    static void run(const cv::Mat &src, const VectorROI &vctFaceBoxes,
                    const cv::Scalar &color, cv::Mat &out)
    {
        src.copyTo(out);
        for (const cv::Rect &box : vctFaceBoxes)
            cv::rectangle(out, box, color);
    }
};

// A face detector outputs a blob with the shape: [1, 1, N, 7], where N is
//  the number of detected bounding boxes. Structure of an output for every
//  detected face is the following:
//  [image_id, label, conf, x_min, y_min, x_max, y_max]; all the seven elements
//  are floating point. For more details please visit:
// https://github.com/opencv/open_model_zoo/blob/master/intel_models/face-detection-adas-0001
// This kernel is the face detection output blob parsing that returns a vector
//  of detected faces' rects:
GAPI_OCV_KERNEL(GCPUFacePostProc, GFacePostProc)
{
    static void run(const cv::Mat &inDetectResult, const cv::Mat &inFrame,
                    const float faceConfThreshold, VectorROI &outFaces)
    {
        const cv::Size imgSize = inFrame.size();
        outFaces.clear();
        const int numOfDetections = inDetectResult.size[2];
        for (int i = 0; i < numOfDetections; i++)
        {
            const float faceId         = inDetectResult.at<float>
                                                      (cv::Vec<int,4>(0,0,i,0));
            const float faceConfidence = inDetectResult.at<float>
                                                      (cv::Vec<int,4>(0,0,i,2));
            if (faceId < 0.f)  // indicates the end of detections
            {
                break;
            }
            if (faceConfidence > faceConfThreshold)
            {
                const cv::Point2f ptfTopLeft( inDetectResult.at<float>
                                              (cv::Vec<int,4>(0,0,i,3)),
                                              inDetectResult.at<float>
                                              (cv::Vec<int,4>(0,0,i,4)));
                const cv::Point2f ptfBotRight(inDetectResult.at<float>
                                              (cv::Vec<int,4>(0,0,i,5)),
                                              inDetectResult.at<float>
                                              (cv::Vec<int,4>(0,0,i,6)));
                outFaces.push_back(getFaceRect(ptfTopLeft, ptfBotRight,
                                               imgSize));
            }
        }
    }
};

// This kernel is the facial landmarks detection output Mat parsing for every
//  detected face; returns a tuple containing a vector of vectors of
//  face elements' Points and a vector of vectors of jaw's Points:
GAPI_OCV_KERNEL(GCPULandmPostProc, GLandmPostProc)
{
    static void run(const std::vector<cv::Mat> &vctDetectResults,
                    const VectorROI &vctRects,
                    std::vector<std::vector<cv::Point>> &vctPtsFaceElems,
                    std::vector<Contour> &vctCntJaw)
    {
        // There are 35 landmarks given by the default detector for each face
        //  in a frame; the first 18 of them are face elements (eyes, eyebrows,
        //  a nose, a mouth) and the last 17 - a jaw contour. The detector gives
        //  floating point values for landmarks' normed coordinates relatively
        //  to an input ROI (not the original frame).
        //  For more details please visit:
// https://github.com/opencv/open_model_zoo/blob/master/intel_models/facial-landmarks-35-adas-0002
        static constexpr int kNumFaceElems = 18;
        static constexpr int kNumTotal     = 35;
        const size_t numFaces = vctRects.size();
        CV_Assert(vctPtsFaceElems.size() == 0ul);
        CV_Assert(vctCntJaw.size()       == 0ul);
        vctPtsFaceElems.reserve(numFaces);
        vctCntJaw.reserve(numFaces);

        std::vector<cv::Point> ptsFaceElems;
        Contour cntJaw;
        ptsFaceElems.reserve(kNumFaceElems);
        cntJaw.reserve(kNumTotal - kNumFaceElems);

        for (size_t i = 0; i < numFaces; i++)
        {
            // The face elements points:
            ptsFaceElems.clear();
            for (int j = 0; j < kNumFaceElems * 2; j += 2)
            {
                cv::Point pt = getRigthCoordinates(
                            cv::Point2f(vctDetectResults[i].at<float>(j),
                                        vctDetectResults[i].at<float>(j + 1)),
                            vctRects[i]);
                ptsFaceElems.push_back(pt);
            }
            vctPtsFaceElems.push_back(ptsFaceElems);

            // The jaw contour points:
            cntJaw.clear();
            for(int j = kNumFaceElems * 2; j < kNumTotal * 2; j += 2)
            {
                cv::Point pt = getRigthCoordinates(
                            cv::Point2f(vctDetectResults[i].at<float>(j),
                                        vctDetectResults[i].at<float>(j + 1)),
                            vctRects[i]);
                cntJaw.push_back(pt);
            }
            vctCntJaw.push_back(cntJaw);
        }
    }
};

// This kernel is the facial landmarks detection post-processing for every face
//  detected before; output is a tuple of vectors of detected face contours and
//  facial elements contours:
//! [kern_m_impl]
GAPI_OCV_KERNEL(GCPUGetContours, GGetContours)
{
    static void run(const std::vector<std::vector<cv::Point>> vctPtsFaceElems,
                    const std::vector<Contour> vctCntJaw,
                    std::vector<Contour> &vctFaceElemsContours,
                    std::vector<Contour> &vctFaceContours)
    {
//! [kern_m_impl]
        size_t numFaces = vctCntJaw.size();
        CV_Assert(numFaces == vctPtsFaceElems.size());
        CV_Assert(vctFaceElemsContours.size() == 0ul);
        CV_Assert(vctFaceContours.size()      == 0ul);
        // vctFaceElemsContours will store all the face elements' contours found
        //  on an input image, namely 4 elements (two eyes, nose, mouth)
        //  for every detected face
        vctFaceElemsContours.reserve(numFaces * 4);
        // vctFaceElemsContours will store all the faces' contours found on
        //  an input image
        vctFaceContours.reserve(numFaces);

        Contour cntFace, cntLeftEye, cntRightEye, cntNose, cntMouth;
        cntNose.reserve(4);

        for (size_t i = 0ul; i < numFaces; i++)
        {
            // The face elements contours
            // A left eye:
            // Approximating the lower eye contour by half-ellipse
            //  (using eye points) and storing in cntLeftEye:
            cntLeftEye = getEyeEllipse(vctPtsFaceElems[i][1],
                                       vctPtsFaceElems[i][0],
                                       config::kNumPointsInHalfEllipse + 3);
            // Pushing the left eyebrow clock-wise:
            cntLeftEye.insert(cntLeftEye.cend(), {vctPtsFaceElems[i][12],
                                                  vctPtsFaceElems[i][13],
                                                  vctPtsFaceElems[i][14]});
            // A right eye:
            // Approximating the lower eye contour by half-ellipse
            //  (using eye points) and storing in vctRightEye:
            cntRightEye = getEyeEllipse(vctPtsFaceElems[i][2],
                                        vctPtsFaceElems[i][3],
                                        config::kNumPointsInHalfEllipse + 3);
            // Pushing the right eyebrow clock-wise:
            cntRightEye.insert(cntRightEye.cend(), {vctPtsFaceElems[i][15],
                                                    vctPtsFaceElems[i][16],
                                                    vctPtsFaceElems[i][17]});
            // A nose:
            // Storing the nose points clock-wise
            cntNose.clear();
            cntNose.insert(cntNose.cend(), {vctPtsFaceElems[i][4],
                                            vctPtsFaceElems[i][7],
                                            vctPtsFaceElems[i][5],
                                            vctPtsFaceElems[i][6]});
            // A mouth:
            // Approximating the mouth contour by two half-ellipses
            //  (using mouth points) and storing in vctMouth:
            cntMouth = getPatchedEllipse(vctPtsFaceElems[i][8],
                                         vctPtsFaceElems[i][9],
                                         vctPtsFaceElems[i][10],
                                         vctPtsFaceElems[i][11]);
            // Storing all the elements in a vector:
            vctFaceElemsContours.insert(vctFaceElemsContours.cend(),
                                        {cntLeftEye, cntRightEye, cntNose,
                                         cntMouth});

            // The face contour:
            // Approximating the forehead contour by half-ellipse
            //  (using jaw points) and storing in vctFace:
            cntFace = getForeheadEllipse(vctCntJaw[i][0], vctCntJaw[i][16],
                                         vctCntJaw[i][8],
                                         config::kNumPointsInHalfEllipse +
                                            vctCntJaw[i].size());
            // The ellipse is drawn clock-wise, but jaw contour points goes
            //  vice versa, so it's necessary to push cntJaw from the end
            //  to the begin using a reverse iterator:
            std::copy(vctCntJaw[i].crbegin(), vctCntJaw[i].crend(),
                      std::back_inserter(cntFace));
            // Storing the face contour in another vector:
            vctFaceContours.push_back(cntFace);
        }
    }
};

// GAPI subgraph functions
inline cv::GMat unsharpMask(const cv::GMat &src, const int sigma,
                            const float strength);
inline cv::GMat mask3C(const cv::GMat &src, const cv::GMat &mask);
} // namespace custom


// Functions implementation:
// Converts output float coordinates of the face rectangle top-left and
//  bottom-right points into a face rectangle:
inline cv::Rect custom::getFaceRect(const cv::Point2f &ptfTopLeft,
                                    const cv::Point2f &ptfBotRight,
                                    const cv::Size &imgSize)
{
    const int imgCols = imgSize.width;
    const int imgRows = imgSize.height;
    const cv::Point tl(std::max(toIntRounded(ptfTopLeft.x  * imgCols), 0),
                       std::max(toIntRounded(ptfTopLeft.y  * imgRows), 0));
    const cv::Point br(std::min(toIntRounded(ptfBotRight.x * imgCols),
                                imgCols - 1),
                       std::min(toIntRounded(ptfBotRight.y * imgRows),
                                imgRows - 1));
    return cv::Rect(tl, br);
}

// The landmarks detector gives floating point values for landmarks' normed
//  coordinates relatively to an input ROI (not the original frame), so this
//  function recounts the coordinates:
inline cv::Point custom::getRigthCoordinates(const cv::Point2f &ptFloat,
                                             const cv::Rect &faceCoordinates)
{
    return cv::Point(toIntRounded(ptFloat.x * faceCoordinates.width
                                                + faceCoordinates.x),
                     toIntRounded(ptFloat.y * faceCoordinates.height
                                                + faceCoordinates.y));
}

// Returns an angle (in degrees) between a line given by two Points and
//  the horison. Note that the result depends on the arguments order:
inline int custom::getLineInclinationAngleDegrees(const cv::Point &ptLeft,
                                                  const cv::Point &ptRight)
{
    const cv::Point residual = ptRight - ptLeft;
    if (residual.y == 0 && residual.x == 0)
        return 0;
    else
        return toIntRounded(atan2(toDouble(residual.y), toDouble(residual.x))
                                * 180.0 / M_PI);
}

// Approximates a forehead by half-ellipse using jaw points and some geometry
//  and then returns points of the contour; "capacity" is used to reserve enough
//  memory as there will be other points inserted.
inline Contour custom::getForeheadEllipse(const cv::Point &ptJawLeft,
                                          const cv::Point &ptJawRight,
                                          const cv::Point &ptJawLower,
                                          const size_t capacity = 0)
{
    Contour cntForehead;
    cntForehead.reserve(std::max(capacity, config::kNumPointsInHalfEllipse));
    // The point amid the top two points of a jaw:
    const cv::Point ptFaceCenter((ptJawLeft + ptJawRight) / 2);
    // This will be the center of the ellipse.

    // The angle between the jaw and the vertical:
    const int angFace = getLineInclinationAngleDegrees(ptJawLeft, ptJawRight);
    // This will be the inclination of the ellipse

    // Counting the half-axis of the ellipse:
    const double jawWidth  = norm(ptJawLeft - ptJawRight);
    // A forehead width equals the jaw width, and we need a half-axis:
    const int axisX        = toIntRounded(jawWidth / 2.0);

    const double jawHeight = norm(ptFaceCenter - ptJawLower);
    // According to research, in average a forehead is approximately 2/3 of
    //  a jaw:
    const int axisY        = toIntRounded(jawHeight * 2 / 3.0);

    // We need the upper part of an ellipse:
    static constexpr int kAngForeheadStart = 180;
    static constexpr int kAngForeheadEnd   = 360;
    cv::ellipse2Poly(ptFaceCenter, cv::Size(axisX, axisY), angFace,
                     kAngForeheadStart, kAngForeheadEnd, config::kAngDelta,
                     cntForehead);
    return cntForehead;
}

// Approximates the lower eye contour by half-ellipse using eye points and some
//  geometry and then returns points of the contour; "capacity" is used
//  to reserve enough memory as there will be other points inserted.
inline Contour custom::getEyeEllipse(const cv::Point &ptLeft,
                                     const cv::Point &ptRight,
                                     const size_t capacity = 0)
{
    Contour cntEyeBottom;
    cntEyeBottom.reserve(std::max(capacity, config::kNumPointsInHalfEllipse));
    const cv::Point ptEyeCenter((ptRight + ptLeft) / 2);
    const int angle = getLineInclinationAngleDegrees(ptLeft, ptRight);
    const int axisX = toIntRounded(norm(ptRight - ptLeft) / 2.0);
    // According to research, in average a Y axis of an eye is approximately
    //  1/3 of an X one.
    const int axisY = axisX / 3;
    // We need the lower part of an ellipse:
    static constexpr int kAngEyeStart = 0;
    static constexpr int kAngEyeEnd   = 180;
    ellipse2Poly(ptEyeCenter, cv::Size(axisX, axisY), angle, kAngEyeStart,
                 kAngEyeEnd, config::kAngDelta, cntEyeBottom);
    return cntEyeBottom;
}

//This function approximates an object (a mouth) by two half-ellipses using
//  4 points of the axes' ends and then returns points of the contour:
inline Contour custom::getPatchedEllipse(const cv::Point &ptLeft,
                                         const cv::Point &ptRight,
                                         const cv::Point &ptUp,
                                         const cv::Point &ptDown)
{
    // Shared characteristics for both half-ellipses:
    const cv::Point ptMouthCenter((ptLeft + ptRight) / 2);
    const int angMouth = getLineInclinationAngleDegrees(ptLeft, ptRight);
    const int axisX    = toIntRounded(norm(ptRight - ptLeft) / 2.0);

    // The top half-ellipse:
    Contour cntMouthTop;
    const int axisYTop = toIntRounded(norm(ptMouthCenter - ptUp));
    // We need the upper part of an ellipse:
    static constexpr int angTopStart = 180;
    static constexpr int angTopEnd   = 360;
    ellipse2Poly(ptMouthCenter, cv::Size(axisX, axisYTop), angMouth,
                 angTopStart, angTopEnd, config::kAngDelta, cntMouthTop);

    // The bottom half-ellipse:
    Contour cntMouth;
    const int axisYBot = toIntRounded(norm(ptMouthCenter - ptDown));
    // We need the lower part of an ellipse:
    static constexpr int angBotStart = 0;
    static constexpr int angBotEnd   = 180;
    ellipse2Poly(ptMouthCenter, cv::Size(axisX, axisYBot), angMouth,
                 angBotStart, angBotEnd, config::kAngDelta, cntMouth);

    // Pushing the upper part to vctOut
    cntMouth.reserve(cntMouth.size() + cntMouthTop.size());
    std::copy(cntMouthTop.cbegin(), cntMouthTop.cend(),
              std::back_inserter(cntMouth));
    return cntMouth;
}

inline cv::GMat custom::unsharpMask(const cv::GMat &src, const int sigma,
                                    const float strength)
{
    cv::GMat blurred   = cv::gapi::medianBlur(src, sigma);
    cv::GMat laplacian = custom::GLaplacian::on(blurred, CV_8U);
    return (src - (laplacian * strength));
}

inline cv::GMat custom::mask3C(const cv::GMat &src, const cv::GMat &mask)
{
    std::tuple<cv::GMat,cv::GMat,cv::GMat> tplIn = cv::gapi::split3(src);
    cv::GMat masked0 = cv::gapi::mask(std::get<0>(tplIn), mask);
    cv::GMat masked1 = cv::gapi::mask(std::get<1>(tplIn), mask);
    cv::GMat masked2 = cv::gapi::mask(std::get<2>(tplIn), mask);
    return cv::gapi::merge3(masked0, masked1, masked2);
}


int main(int argc, char** argv)
{
    cv::namedWindow(config::kWinFaceBeautification, cv::WINDOW_NORMAL);
    cv::namedWindow(config::kWinInput,              cv::WINDOW_NORMAL);

    cv::CommandLineParser parser(argc, argv,
"{ help         h       ||      print the help message. }"

"{ facepath     f       ||      full path to a Face detection model file"
                                " (.xml).}"
"{ facedevice           |GPU|   the face detection computation device.}"

"{ landmpath    l       ||      full path to a facial Landmarks detection model"
                                " file (.xml).}"
"{ landmdevice          |CPU|   the landmarks detection computation device.}"

"{ input        i       ||      full path to an input image or a video file."
                                " Skip this argument to capture frames from"
                                " a camera.}"
"{ boxes        b       |false| set true if want to draw face Boxes in the"
                                " \"Input\" window.}"
"{ landmarks    m       |false| set true if want to draw facial landMarks in"
                                "the \"Input\" window.}"
    );
    parser.about("Use this script to run the face beautification"
                 " algorithm on G-API.");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    // Parsing input arguments
    const std::string faceXmlPath = parser.get<std::string>("facepath");
    const std::string faceBinPath = getWeightsPath(faceXmlPath);
    const std::string faceDevice  = parser.get<std::string>("facedevice");

    const std::string landmXmlPath = parser.get<std::string>("landmpath");
    const std::string landmBinPath = getWeightsPath(landmXmlPath);
    const std::string landmDevice  = parser.get<std::string>("landmdevice");

    // The flags for drawing/not drawing face boxes or/and landmarks in the
    //  \"Input\" window:
    const bool flgBoxes     = parser.get<bool>("boxes");
    const bool flgLandmarks = parser.get<bool>("landmarks");
    // To provide this opportunity, it is necessary to check the flags when
    //  compiling a graph

    // Declaring a graph
    // Streaming-API version of a pipeline expression with a lambda-based
    //  constructor is used to keep all temporary objects in a dedicated scope.
    cv::GComputation pipeline([=]()
    {
        cv::GMat gimgIn;
        // Infering
        cv::GMat facesDetected                          =
                cv::gapi::infer<custom::FaceDetector>(gimgIn);
        GArrayROI garFaceRects                          =
                custom::GFacePostProc::on(facesDetected, gimgIn,
                                          config::kFaceConfThreshold);
        cv::GArray<cv::GMat> garLandmarksDetected       =
                cv::gapi::infer<custom::FacialLandmarksDetector>(garFaceRects,
                                                                 gimgIn);
        cv::GArray<std::vector<cv::Point>> garPtsFaceElems;
        cv::GArray<Contour>                garJawContours;
        std::tie(garPtsFaceElems, garJawContours)       =
                custom::GLandmPostProc::on(garLandmarksDetected, garFaceRects);
        cv::GArray<Contour> garFaceElemsContours;
        cv::GArray<Contour> garFaceContours;
        std::tie(garFaceElemsContours, garFaceContours) =
                custom::GGetContours::on(garPtsFaceElems, garJawContours);
        // Masks drawing
        // All masks are created as CV_8UC1
        cv::GMat mskSharp                               =
                custom::GFillPolyGContours::on(gimgIn, garFaceElemsContours);
        cv::GMat mskSharpGaussed                        =
                cv::gapi::gaussianBlur(mskSharp, config::kGaussKernelSize,
                                       config::kGaussSigma);
        cv::GMat mskBlur                                =
                custom::GFillPolyGContours::on(gimgIn, garFaceContours);
        cv::GMat mskBlurGaussed                         =
                cv::gapi::gaussianBlur(mskBlur, config::kGaussKernelSize,
                                       config::kGaussSigma);
        // The first argument in mask() is Blur as we want to subtract from Blur
        //  the next step
        cv::GMat mskBlurFinal                           =
                mskBlurGaussed - cv::gapi::mask(mskBlurGaussed,
                                                mskSharpGaussed);
        cv::GMat mskFacesGaussed                        =
                mskBlurFinal + mskSharpGaussed;
        cv::GMat mskFacesWhite                          =
                cv::gapi::threshold(mskFacesGaussed, 0, 255, cv::THRESH_BINARY);
        cv::GMat mskNoFaces                             =
                cv::gapi::bitwise_not(mskFacesWhite);
        cv::GMat gimgBilat                              =
                custom::GBilateralFilter::on(gimgIn, config::kBilatDiameter,
                                             config::kBilatSigmaColor,
                                             config::kBilatSigmaSpace);
        cv::GMat gimgSharp                              =
                custom::unsharpMask(gimgIn, config::kUnsharpSigma,
                                    config::kUnsharpStrength);
        // Applying the masks
        // Custom function mask3C() should be used instead of just gapi::mask()
        //  as mask() provides CV_8UC1 source only (and we have CV_8U3C)
        cv::GMat gimgBilatMasked = custom::mask3C(gimgBilat, mskBlurFinal);
        cv::GMat gimgSharpMasked = custom::mask3C(gimgSharp, mskSharpGaussed);
        cv::GMat gimgInMasked    = custom::mask3C(gimgIn,    mskNoFaces);
        cv::GMat gimgBeautif = gimgBilatMasked + gimgSharpMasked + gimgInMasked;
        // Drawing face boxes and landmarks if necessary:
        cv::GMat gimgInShowTemp;
        if (flgLandmarks == true)
        {
            cv::GMat gimgInShowTempTemp =
                    custom::GPolyLines::on(gimgIn, garFaceContours,
                                           config::kClosedLine,
                                           config::kClrYellow);
            gimgInShowTemp              =
                    custom::GPolyLines::on(gimgInShowTempTemp,
                                           garFaceElemsContours,
                                           config::kClosedLine,
                                           config::kClrYellow);
        }
        else
        {
            gimgInShowTemp = gimgIn;
        }
        cv::GMat gimgInShow;
        if (flgBoxes == true)
        {
            gimgInShow =
                    custom::GRectangle::on(gimgInShowTemp, garFaceRects,
                                           config::kClrGreen);
        }
        else
        {
            // This action is necessary because an output node must be a result of
            //  some operations applied to an input node, so it handles the case
            //  when it should be nothing to draw
            gimgInShow = cv::gapi::copy(gimgInShowTemp);
        }
        return cv::GComputation(cv::GIn(gimgIn),
                                cv::GOut(gimgBeautif, gimgInShow));
    });
    // Declaring IE params for networks
    auto faceParams  = cv::gapi::ie::Params<custom::FaceDetector>
    {
        faceXmlPath,
        faceBinPath,
        faceDevice
    };
    auto landmParams = cv::gapi::ie::Params<custom::FacialLandmarksDetector>
    {
        landmXmlPath,
        landmBinPath,
        landmDevice
    };
    auto networks    = cv::gapi::networks(faceParams, landmParams);
    // Declaring custom and fluid kernels have been used:
    auto customKernels = cv::gapi::kernels<custom::GCPUBilateralFilter,
                                           custom::GCPULaplacian,
                                           custom::GCPUFillPolyGContours,
                                           custom::GCPUPolyLines,
                                           custom::GCPURectangle,
                                           custom::GCPUFacePostProc,
                                           custom::GCPULandmPostProc,
                                           custom::GCPUGetContours>();
    auto kernels       = cv::gapi::combine(cv::gapi::core::fluid::kernels(),
                                           customKernels);
    // Now we are ready to compile the pipeline to a stream with specified
    //  kernels, networks and image format expected to process
    auto stream = pipeline.compileStreaming(
                cv::GMatDesc{CV_8U,3,cv::Size(640,480)},
                cv::compile_args(kernels, networks));
    // Setting the source for the stream:
    if (parser.has("input"))
    {
        stream.setSource(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>
                         (parser.get<cv::String>("input")));
    }
    else
    {
        stream.setSource(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>
                         (0));
    }
    // Declaring output variables
    cv::Mat imgShow;
    cv::Mat imgBeautif;
    // Streaming:
    stream.start();
    while (stream.running())
    {
        auto out_vector = cv::gout(imgBeautif, imgShow);
        if (!stream.try_pull(std::move(out_vector)))
        {
            // Use a try_pull() to obtain data.
            // If there's no data, let UI refresh (and handle keypress)
            if (cv::waitKey(1) >= 0) break;
            else continue;
        }
        cv::imshow(config::kWinInput,              imgShow);
        cv::imshow(config::kWinFaceBeautification, imgBeautif);
    }
    stream.stop();
    return 0;
}
