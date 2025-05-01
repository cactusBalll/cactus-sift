// build.rs

use std::env;
use std::path::Path;

macro_rules! libl {
    ($($id: ident); *) => {
        vec![$(stringify!($id)),*]
    };
}
fn main() {
    let out_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    // 仍未解决库的链接问题，在这里链接
    println!(
        "cargo:rustc-link-search={}",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\lib\\x64"
    );
    println!(
        "cargo:rustc-link-search={}",
        "D:\\college2025spr\\GPGPU\\opencv-custom-build\\install\\x64\\vc17\\lib"
    );
    // println!("cargo:rustc-link-lib=static=cuda");
    println!("cargo:rustc-link-lib=static=cudart");
    let opencv_libs = libl!(opencv_calib3d;opencv_core;opencv_dnn;opencv_features2d;opencv_flann;opencv_gapi;opencv_highgui;opencv_imgcodecs;opencv_imgproc;opencv_ml;opencv_objdetect;opencv_photo;opencv_stitching;opencv_video;opencv_videoio;opencv_aruco;opencv_bgsegm;opencv_bioinspired;opencv_ccalib;opencv_cudaarithm;opencv_cudabgsegm;opencv_cudacodec;opencv_cudafeatures2d;opencv_cudafilters;opencv_cudaimgproc;opencv_cudalegacy;opencv_cudaobjdetect;opencv_cudaoptflow;opencv_cudastereo;opencv_cudawarping;opencv_cudev;opencv_datasets;opencv_dnn_objdetect;opencv_dnn_superres;opencv_dpm;opencv_face;opencv_fuzzy;opencv_hfs;opencv_img_hash;opencv_intensity_transform;opencv_line_descriptor;opencv_mcc;opencv_optflow;opencv_phase_unwrapping;opencv_plot;opencv_quality;opencv_rapid;opencv_reg;opencv_rgbd;opencv_saliency;opencv_shape;opencv_signal;opencv_stereo;opencv_structured_light;opencv_superres;opencv_surface_matching;opencv_text;opencv_tracking;opencv_videostab;opencv_wechat_qrcode;opencv_xfeatures2d;opencv_ximgproc;opencv_xobjdetect;opencv_xphoto);
    for lib in opencv_libs.iter() {
        println!("cargo:rustc-link-lib=static={}4110", lib);
    }
    println!("cargo::rustc-link-search=native={}", out_dir);
    println!("cargo::rustc-link-lib=static=gaussian_lib");
    println!("cargo::rustc-link-lib=static=gaussian_cv2");

    // println!("cargo::rustc-link-lib=static=hello");
    // println!("cargo::rerun-if-changed=src/hello.c");
}
