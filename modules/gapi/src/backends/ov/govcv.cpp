// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation

#include "precomp.hpp"

#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/infer/ov.hpp> // ov::kernels()
#include <opencv2/gapi/gcompoundkernel.hpp>
#include "backends/ov/ovdef.hpp"

#ifdef HAVE_OPENVINO_2_0

#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset4.hpp>
#include <openvino/opsets/opset8.hpp>

#include "backends/ov/govcvkernel.hpp"

namespace {
namespace util {

ov::Output<ov::Node> ovBinaryToOCVGray(const ov::Output<ov::Node> &in) {
    auto i32  = std::make_shared<ov::opset1::Convert>(in, ov::element::i32);

    // multiply [0,1] vector to [0,255] to follow OCV conventions
    auto cval = std::make_shared<ov::opset8::Constant>
        (ov::element::i32,
         ov::Shape{1},
         255);
    auto mul = std::make_shared<ov::opset1::Multiply>(i32, cval);
    return std::make_shared<ov::opset1::Convert>(mul, ov::element::u8);
}

}} // namespace <anonymous>::util

////////////////////////////////////////////////////////////////////////////////

GAPI_OVCV_KERNEL(GOVCVAdd, cv::gapi::core::GAdd) {
    static ov::Output<ov::Node>
        run(const ovcv::TensorNode &src1,
            const ovcv::TensorNode &src2,
            int ddepth) {
        return std::make_shared<ov::opset1::Add>(src1.node, src2.node);
        // FIXME: Support ddepth
    };
};

////////////////////////////////////////////////////////////////////////////////

GAPI_OVCV_KERNEL(GOVCVMulCOld, cv::gapi::core::GMulCOld)
{
    static ov::Output<ov::Node>
        run(const ovcv::TensorNode &src,
            double a,
            int ddepth) {
        auto cval = std::make_shared<ov::opset8::Constant>
            (ov::element::f32,
             ov::Shape{1},
             static_cast<float>(a));

        // Its a shame OpenVINO doesn't support mixed precision Multiply
        auto f32 = std::make_shared<ov::opset1::Convert>(src.node, ov::element::f32);
        auto tmp = std::make_shared<ov::opset1::Multiply>(f32, cval);
        return std::make_shared<ov::opset1::Convert>(tmp, ov::element::u8);
        // FIXME: Support ddepth
    }
};

////////////////////////////////////////////////////////////////////////////////

GAPI_OVCV_KERNEL(GOVCVResize, cv::gapi::imgproc::GResize) {
    static ov::Output<ov::Node>
        run(const ovcv::TensorNode &src,
            cv::Size sz,
            double fx,
            double fy,
            int interp) {
        // FIXME: At this point, there's no way to inspect the input
        // image's metadata. It's cv::GMetaArg is not available here.
        // Need to implement the same trick as in Fluid backend...
        // So assume we deal with NHWC images only at this point
        // with N = 1 always.

        auto axes = std::make_shared<ov::opset8::Constant>
            (ov::element::i32,
             ov::Shape{2},
             std::vector<int>{1,2}); // HW in NHWC

        ov::opset4::Interpolate::InterpolateAttrs attrs;
        attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SIZES;

        GAPI_Assert(fx == 0);
        GAPI_Assert(fy == 0);

        std::shared_ptr<ov::opset8::Constant> sizes = std::make_shared<ov::opset8::Constant>
                (ov::element::i32,
                 ov::Shape{2},
                 std::vector<int>{sz.height, sz.width});

        std::shared_ptr<ov::opset8::Constant> scales = std::make_shared<ov::opset8::Constant>
                (ov::element::f32,
                 ov::Shape{2},
                 std::vector<float>{static_cast<float>(fy), static_cast<float>(fx)});

        // FIXME: Support interp
        return std::make_shared<ov::opset4::Interpolate>(src.node, sizes, scales, axes, attrs);
    }
};

////////////////////////////////////////////////////////////////////////////////

GAPI_OVCV_KERNEL(GOVCVSplit3, cv::gapi::core::GSplit3) {
    static std::tuple<ov::Output<ov::Node>, ov::Output<ov::Node>, ov::Output<ov::Node> >
        run(const ovcv::TensorNode &src) {
        auto cval = std::make_shared<ov::opset8::Constant>
            (ov::element::i32,
             ov::Shape{},
             3); // split over C in NHWC
        auto split = std::make_shared<ov::opset1::Split>(src.node, cval, 3);
        return std::make_tuple(split->output(0), split->output(1), split->output(2));
    }
};

GAPI_OVCV_KERNEL(GOVCVMerge3, cv::gapi::core::GMerge3) {
    static ov::Output<ov::Node>
        run(const ovcv::TensorNode &a,
            const ovcv::TensorNode &b,
            const ovcv::TensorNode &c) {
        auto inputs = std::vector<ov::Output<ov::Node> >{a.node, b.node, c.node};
        return std::make_shared<ov::opset1::Concat>(inputs, 3); // C in NHWC
    }
};

////////////////////////////////////////////////////////////////////////////////

GAPI_OVCV_KERNEL(GOVCVConcatHor, cv::gapi::core::GConcatHor) {
    static ov::Output<ov::Node>
        run(const ovcv::TensorNode &a,
            const ovcv::TensorNode &b) {
        auto inputs = std::vector<ov::Output<ov::Node> >{a.node, b.node};
        return std::make_shared<ov::opset1::Concat>(inputs, 2); // W in NHWC
    }
};

GAPI_OVCV_KERNEL(GOVCVConcatVert, cv::gapi::core::GConcatVert) {
    static ov::Output<ov::Node>
        run(const ovcv::TensorNode &a,
            const ovcv::TensorNode &b) {
        auto inputs = std::vector<ov::Output<ov::Node> >{a.node, b.node};
        return std::make_shared<ov::opset1::Concat>(inputs, 1); // H in NHWC
    }
};

////////////////////////////////////////////////////////////////////////////////

// ..because OpenVINO can't pad with 3-ch value
namespace local {
G_API_OP(GPadPlane, <cv::GMat(cv::GMat,int,int,int,int,uint8_t)>, "org.opencv.core.padplane") {
    static cv::GMatDesc outMeta(const cv::GMatDesc &in,
                            int top, int bot, int lef, int rig,
                            uint8_t) {
        GAPI_Assert(top >= 0);
        GAPI_Assert(bot >= 0);
        GAPI_Assert(lef >= 0);
        GAPI_Assert(rig >= 0);
        return in.withSizeDelta(lef + rig, bot + top);
    }
};
} // namespace local

GAPI_OVCV_KERNEL(GOVCVPadPlane, local::GPadPlane) {
    static ov::Output<ov::Node>
        run(const ovcv::TensorNode &src,
            int top, int bot, int lef, int rig,
            uint8_t color) {
        // Assume our image as NHWC tensor, then
        // pads_begin is [  0, top, lef,  0]
        // pads_end   is [  0, bot, rig,  0]
        // constant   is [  1,   1,   1,  3]
        auto pads_begin = std::make_shared<ov::opset8::Constant>
            (ov::element::i32,
             ov::Shape{4},
             std::vector<int>{0, top, lef, 0});
        auto pads_end = std::make_shared<ov::opset8::Constant>
            (ov::element::i32,
             ov::Shape{4},
             std::vector<int>{0, bot, rig, 0});
        auto pad_val = std::make_shared<ov::opset8::Constant>
            (ov::element::u8,
             ov::Shape{},
             color);
        return std::make_shared<ov::opset1::Pad>(src.node,
                                                 pads_begin,
                                                 pads_end,
                                                 pad_val,
                                                 ov::op::PadMode::CONSTANT);
    }
};

GAPI_COMPOUND_KERNEL(GOVCVPad, cv::gapi::core::GPad) {
    static cv::GMat expand(cv:: GMat in, int top, int bot, int lef, int rig, const cv::Scalar &color) {
        // Need to pad planes with individual colors. In OpenVINO,
        // only one constant value can be assigned, so have to
        // decompose in channels & compose back
        cv::GMat b, g, r;
        std::tie(b, g, r) = cv::gapi::split3(in);
        auto pp = [=](cv::GMat plane, double value) {
            return local::GPadPlane::on(plane, top, bot, lef, rig, static_cast<uint8_t>(value));
        };
        using namespace local;
        return cv::gapi::merge3(pp(b, color[0]), pp(g, color[1]), pp(r, color[2]));
    }
};

////////////////////////////////////////////////////////////////////////////////

GAPI_OVCV_KERNEL(GOVCVCmpEQScalar, cv::gapi::core::GCmpEQScalar) {
    static ov::Output<ov::Node>
        run(const ovcv::TensorNode &src, const ovcv::TensorNode &scalar) {

        // OpenCV specifics now. Believe or not, its cv::compare(mat,scalar)
        // takes just 1 compoment from scalar to compare the image against.
        // It is not BGR-BGR comparison, it is BGR-Const comparison.
        // As we're fed with 3-channel (in fact, 4-channel) Scalars frost,
        // we need to extract this first element here to make the comparison.
        // Needless to say the image has to be decomposed at the user side in
        // three channels to amke three channel comparison.
        auto extract = std::make_shared<ov::opset8::Constant>
            (ov::element::i32,
             ov::Shape{},
             3); // split over C in NHWC
        auto element = std::make_shared<ov::opset1::Split>(scalar.node, extract, 3)->output(0);

        // Specifics start here: {{{
        // EQ can't into u8 there
        auto i32_img = std::make_shared<ov::opset1::Convert>(src.node, ov::element::i32);
        auto i32_val = std::make_shared<ov::opset1::Convert>(element,  ov::element::i32);
        // Specifics end here: }}}

        // Actually run Compare, get BOOL tensor (which is still u8 in OpenVINO)
        auto bin = std::make_shared<ov::opset1::Equal>(i32_img, i32_val);
        return util::ovBinaryToOCVGray(bin);
    }
};

////////////////////////////////////////////////////////////////////////////////

GAPI_OVCV_KERNEL(GOVCVInRange, cv::gapi::core::GInRange) {
    static ov::Output<ov::Node>
        run(const ovcv::TensorNode &src,
            const ovcv::TensorNode &lo,
            const ovcv::TensorNode &hi) {
        // With OpenVINO opset, InRange is:
        // 1) a = src >= lo;    % a is [BOOL]
        // 2) b = src <= hi;    % b is [BOOL]
        // 3) c = a & b;        % c is [BOOL]
        // 4) d = reduce(c,&)   $ d is [BOOL]
        // 5) e = bin2gray(e)   $ e is [u8], 0..255
        auto cmp_lo  = std::make_shared<ov::opset1::GreaterEqual>(src.node, lo.node);
        auto cmp_hi  = std::make_shared<ov::opset1::LessEqual>(src.node, hi.node);
        auto cmp_bgr = std::make_shared<ov::opset1::LogicalAnd>(cmp_lo, cmp_hi);

        auto and_axs = std::make_shared<ov::opset8::Constant>
            (ov::element::i32,
             ov::Shape{},
             3); // C in NHWC
        auto cmp_gry = std::make_shared<ov::opset1::ReduceLogicalAnd>(cmp_bgr, and_axs, true);
        return util::ovBinaryToOCVGray(cmp_gry);
    }
};

////////////////////////////////////////////////////////////////////////////////

GAPI_OVCV_KERNEL(GOVCVMask, cv::gapi::core::GMask) {
    static ov::Output<ov::Node>
        run(const ovcv::TensorNode &src,
            const ovcv::TensorNode &mask) {
        // OCV mask tensor is grayscale.
        // Need to convert it to 0..1 range and multiply

        GAPI_Assert(cv::util::get<cv::GMatDesc>(src.meta).chan == 3);

        // be 4-dim - see the ovcv example pipeline for details.
        auto zero = std::make_shared<ov::opset8::Constant>
            (ov::element::u8,
             ov::Shape{},
             0);
        auto mbin = std::make_shared<ov::opset1::Greater>(mask.node, zero);
        auto mu8  = std::make_shared<ov::opset1::Convert>(mbin, ov::element::u8);

        // Need to broadcast the binary mask to three channels...
        auto mu8p = std::vector<ov::Output<ov::Node> >{mu8, mu8, mu8};
        auto mu83 = std::make_shared<ov::opset1::Concat>(mu8p, 3); // C in NHWC
        return std::make_shared<ov::opset1::Multiply>(src.node, mu83);
    }
};

////////////////////////////////////////////////////////////////////////////////

cv::GKernelPackage cv::gapi::ov::kernels() {
    return cv::gapi::kernels
        < GOVCVAdd
        , GOVCVMulCOld
        , GOVCVResize
        , GOVCVSplit3
        , GOVCVMerge3
        , GOVCVConcatHor
        , GOVCVConcatVert
        , GOVCVPadPlane
        , GOVCVPad
        , GOVCVCmpEQScalar
        , GOVCVInRange
        , GOVCVMask
        >();
}

#else // HAVE_OPENVINO_2_0

cv::GKernelPackage cv::gapi::ov::kernels() {
    GAPI_Assert(false && "No OVCV kernels supported");
}

#endif // HAVE_OPENVINO_2_0
