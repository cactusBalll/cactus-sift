use image::{
    GrayImage, ImageBuffer, Luma, Rgb, RgbImage,
    buffer::ConvertBuffer,
    imageops::{self, FilterType::Triangle},
};
use imageproc::{drawing::draw_antialiased_line_segment_mut, pixelops};
use nalgebra::{Matrix3, Vector3, min};
use std::{f32::consts::PI, i64};

type LumaF32Image = ImageBuffer<Luma<f32>, Vec<f32>>;

const SIGMA0: f64 = 1.6;
const SIGMA_INV1: f64 = SIGMA0 / 2.0;
// 输入图像放大了SCALE_INV1 ^ -1 倍
const SCALE_INV1: f64 = 0.5;
/// 输入的图像假设它是做了sigma的高斯模糊，
/// 无限精细的图像不可获得
const SIGMA_IN: f64 = 0.5;
/// 每个octave含多少尺度的高斯模糊图像
const SCALES_PER_OCTAVE: f64 = 3.0;

const CONTRAST_THRESHOLD: f64 = 0.04;

const IMG_BORDER: usize = 5;

#[derive(Default)]
struct DoGComputed {
    octave_cnt: usize,
    /// 高斯模糊金字塔，原图像在最底层
    pyr_g: Vec<Vec<LumaF32Image>>,
    /// DoG（差分高斯）金字塔
    pyr_dog: Vec<Vec<LumaF32Image>>,
}

#[derive(Default)]
pub struct DoGAndKPComputed {
    octave_cnt: usize,
    /// 高斯模糊金字塔，原图像在最底层
    pyr_g: Vec<Vec<LumaF32Image>>,
    /// DoG（差分高斯）金字塔
    pyr_dog: Vec<Vec<LumaF32Image>>,
    pub keypoints: Vec<SIFTKeyPoint>,
}

struct SIFTKeyPoint {
    pub x: u32,
    pub y: u32,
    pub size: f32,
    pub angle: f32,
    pub response: f32,
    pub octave: usize,
    pub scale: usize,
    /// 关键点特征向量
    pub feature: [u8; 128],
}

struct SIFTResult {
    pub keypoints: Vec<SIFTKeyPoint>,
}

pub fn sift(img: &LumaF32Image) -> DoGAndKPComputed {
    let dog = calculate_dog(img);
    let dog = find_keypoints(dog);
    let dog = filter_keypoints(dog);
    let dog = calculate_main_orientation(dog);
    let mut res = calculate_feature(dog);

    for kpt in res.keypoints.iter_mut() {
        let scale = 2_f32.powi(kpt.octave as i32 - 1);
        let x = (kpt.x as f32 * scale) as u32;
        let y = (kpt.y as f32 * scale) as u32;
        kpt.x = x;
        kpt.y = y;
    }
    res
}

#[derive(Debug, Default)]
pub struct PointPair {
    pub x1: u32,
    pub y1: u32,
    pub x2: u32,
    pub y2: u32,
}

fn dist(f1: &[u8; 128], f2: &[u8; 128]) -> i64 {
    f1.iter()
        .zip(f2.iter())
        .fold(0, |acc, (x1, x2)| acc + (*x1 as i64 - *x2 as i64).pow(2))
        .isqrt()
}
pub fn match_1nn(dog1: &DoGAndKPComputed, dog2: &DoGAndKPComputed) -> Vec<PointPair> {
    let mut ret = Vec::new();
    for kpt1 in dog1.keypoints.iter() {
        let mut min_dist = i64::MAX;
        let mut pp = PointPair::default();
        pp.x1 = kpt1.x;
        pp.y1 = kpt1.y;
        for kpt2 in dog2.keypoints.iter() {
            let new_dist = dist(&kpt1.feature, &kpt2.feature);
            // println!("{:?} {:?}", kpt1.feature, kpt2.feature);
            if new_dist < min_dist {
                min_dist = new_dist;
                pp.x2 = kpt2.x;
                pp.y2 = kpt2.y;
                // println!("{} ({}, {})",min_dist ,kpt2.x, kpt2.y);
            }
        }
        ret.push(pp);
    }
    ret
}

pub fn match_2nn(dog1: &DoGAndKPComputed, dog2: &DoGAndKPComputed) -> Vec<PointPair> {
    const RATIO: f64 = 0.80;
    let mut ret = Vec::new();
    for kpt1 in dog1.keypoints.iter() {
        let mut min_dist = i64::MAX;
        let mut min2_dist = i64::MAX;
        let mut pp = PointPair::default();
        pp.x1 = kpt1.x;
        pp.y1 = kpt1.y;
        for kpt2 in dog2.keypoints.iter() {
            let new_dist = dist(&kpt1.feature, &kpt2.feature);
            // println!("{:?} {:?}", kpt1.feature, kpt2.feature);
            if new_dist < min_dist {
                min2_dist = min_dist;
                min_dist = new_dist;
                pp.x2 = kpt2.x;
                pp.y2 = kpt2.y;
                // println!("{} ({}, {})",min_dist ,kpt2.x, kpt2.y);
            }
        }
        if (min_dist as f64) < min2_dist as f64 * RATIO {
            ret.push(pp);
        }
    }
    ret
}

fn calculate_dog(img: &LumaF32Image) -> DoGComputed {
    let img_f32: LumaF32Image = img.convert();
    // 放大2x的f32类型灰度图
    let img_2x = imageops::resize(
        &img_f32,
        img_f32.width() * 2,
        img_f32.height() * 2,
        imageops::FilterType::Triangle,
    );
    let sigma_init = (SIGMA_INV1.powi(2) - SIGMA_IN.powi(2)).sqrt();
    let seed_img = imageops::blur(&img_2x, sigma_init as f32);
    // 用图片最小方向计算octaves数量，它由图片尺度决定，后面减的常数可以任意设定
    let min_axis = seed_img.width().min(seed_img.height());
    let n_octaves = ((min_axis as f32).log2() - 2.0).round() as usize;

    // 构建高斯模糊塔
    // scale间的比例
    let m = 2_f64.powf(2.0 / SCALES_PER_OCTAVE as f64);
    let sigmas: Vec<f64> = (0..(SCALES_PER_OCTAVE as i32 + 3))
        .map(|s| {
            let a = m.powi(s - 1);
            let b = m * a;
            (b - a).sqrt() * SIGMA0
        })
        .collect();

    // 在一个octave中创建一系列scale
    let create_octave = |initial: LumaF32Image| {
        let mut imgs = Vec::with_capacity(SCALES_PER_OCTAVE as usize + 3_usize);
        imgs.push(initial);
        sigmas.iter().skip(1).for_each(|sigma| {
            let prev = imgs.last().unwrap();
            imgs.push(imageops::blur(prev, *sigma as f32));
        });
        imgs
    };

    let mut pyr_g: Vec<Vec<LumaF32Image>> = Vec::with_capacity(n_octaves);
    pyr_g.push(create_octave(seed_img));
    for _ in 1..n_octaves {
        // 每一个octave的第一个scale从上一层得到，最后还有额外2个scale，所以 +3
        let last_octave = &pyr_g.last().unwrap();
        let initial = &last_octave[last_octave.len() - 3];
        let scaled_half =
            imageops::resize(initial, initial.width() / 2, initial.height() / 2, Triangle);
        pyr_g.push(create_octave(scaled_half));
    }

    // 构建dog
    let mut pyr_dog: Vec<Vec<LumaF32Image>> = Vec::with_capacity(n_octaves);
    for octave in pyr_g.iter() {
        pyr_dog.push(Vec::with_capacity(SCALES_PER_OCTAVE as usize + 2));
        for i in 1..octave.len() {
            let img1 = &octave[i - 1];
            let img2 = &octave[i];
            assert_eq!(img1.height(), img2.height());
            assert_eq!(img1.width(), img2.width());
            let mut img = LumaF32Image::new(img1.width(), img1.height());
            // 逐个像素相减
            img1.enumerate_pixels()
                .zip(img2.enumerate_pixels())
                .for_each(|(p1, p2)| {
                    assert_eq!(p1.0, p2.0);
                    assert_eq!(p1.1, p2.1);
                    let luma = Luma::<f32>([p2.2[0] - p1.2[0]]);
                    let pixel = img.get_pixel_mut(p1.0, p1.1);
                    *pixel = luma;
                    // dbg!(luma);
                });
            pyr_dog.last_mut().unwrap().push(img);
        }
    }

    DoGComputed {
        octave_cnt: n_octaves,
        pyr_g,
        pyr_dog,
    }
}

fn find_keypoints(
    DoGComputed {
        octave_cnt,
        pyr_g,
        pyr_dog,
    }: DoGComputed,
) -> DoGAndKPComputed {
    fn around(img: &LumaF32Image, x: usize, y: usize) -> [f32; 9] {
        let x = x as u32;
        let y = y as u32;
        [
            img.get_pixel(x - 1, y - 1).0[0],
            img.get_pixel(x - 1, y).0[0],
            img.get_pixel(x - 1, y + 1).0[0],
            img.get_pixel(x, y - 1).0[0],
            img.get_pixel(x, y).0[0],
            img.get_pixel(x, y + 1).0[0],
            img.get_pixel(x + 1, y - 1).0[0],
            img.get_pixel(x + 1, y).0[0],
            img.get_pixel(x + 1, y + 1).0[0],
        ]
    }
    fn around_max(img: &LumaF32Image, x: usize, y: usize) -> f32 {
        let arr = around(img, x, y);
        arr.into_iter().reduce(f32::max).unwrap_or(0.0)
    }
    fn around_min(img: &LumaF32Image, x: usize, y: usize) -> f32 {
        let arr = around(img, x, y);
        arr.into_iter().reduce(f32::min).unwrap_or(0.0)
    }
    let mut keypoints: Vec<SIFTKeyPoint> = Vec::new();
    let threshold = 0.5 * CONTRAST_THRESHOLD / SCALES_PER_OCTAVE;
    for (i, octave) in pyr_dog.iter().enumerate() {
        for j in 1..octave.len() - 1 {
            let img = &octave[j];
            let prev = &octave[j - 1];
            let next = &octave[j + 1];
            for r in IMG_BORDER..img.height() as usize - IMG_BORDER {
                for c in IMG_BORDER..img.width() as usize - IMG_BORDER {
                    let px = img.get_pixel(c as u32, r as u32);
                    let v = px.0[0];
                    if px.0[0].abs() < threshold as f32 {
                        continue;
                    }
                    if (around_max(img, c, r) <= v
                        && around_max(prev, c, r) <= v
                        && around_max(next, c, r) <= v)
                        || (around_min(img, c, r) >= v
                            && around_min(prev, c, r) >= v
                            && around_min(next, c, r) >= v)
                    {
                        keypoints.push(SIFTKeyPoint {
                            x: c as u32,
                            y: r as u32,
                            size: 0.,
                            angle: 0.,
                            response: 0.,
                            octave: i,
                            scale: j,
                            feature: [0; 128],
                        });
                    }
                }
            }
        }
    }

    DoGAndKPComputed {
        octave_cnt,
        pyr_g,
        pyr_dog,
        keypoints,
    }
}

/// 筛选极值点
fn filter_keypoints(
    DoGAndKPComputed {
        octave_cnt,
        pyr_g,
        pyr_dog,
        mut keypoints,
    }: DoGAndKPComputed,
) -> DoGAndKPComputed {
    const BIAS_THRESHOLD: f32 = 0.5;
    const EDGE_THRESHOLD: f32 = 10.0;
    const TRY_COUNT: usize = 5;
    let mut delete_point_index: Vec<usize> = Vec::new();
    'pt: for (i, kpt) in keypoints.iter_mut().enumerate() {
        let prev = &pyr_dog[kpt.octave][kpt.scale - 1];
        let img = &pyr_dog[kpt.octave][kpt.scale];
        let next = &pyr_dog[kpt.octave][kpt.scale + 1];
        let mut kpt_x = kpt.x;
        let mut kpt_y = kpt.y;
        let mut kpt_scale = kpt.scale;
        let mut tr_h: f32 = 0.0;
        let mut det_h: f32 = 0.0;
        for _ in 0..TRY_COUNT {
            let prev = &pyr_dog[kpt.octave][kpt_scale - 1];
            let img = &pyr_dog[kpt.octave][kpt_scale];
            let next = &pyr_dog[kpt.octave][kpt_scale + 1];
            let (x1, x2) = (
                img.get_pixel(kpt_x - 1, kpt_y).0[0],
                img.get_pixel(kpt_x + 1, kpt_y).0[0],
            );
            let (y1, y2) = (
                img.get_pixel(kpt_x, kpt_y - 1).0[0],
                img.get_pixel(kpt_x, kpt_y + 1).0[0],
            );
            let (s1, s2) = (
                prev.get_pixel(kpt_x, kpt_y).0[0],
                next.get_pixel(kpt_x, kpt_y).0[0],
            );

            let p122 = img.get_pixel(kpt_x + 1, kpt_y + 1).0[0];
            let p112 = img.get_pixel(kpt_x - 1, kpt_y + 1).0[0];
            let p121 = img.get_pixel(kpt_x + 1, kpt_y - 1).0[0];
            let p111 = img.get_pixel(kpt_x - 1, kpt_y - 1).0[0];

            let p222 = next.get_pixel(kpt_x + 1, kpt_y).0[0];
            let p212 = next.get_pixel(kpt_x - 1, kpt_y).0[0];
            let p221 = prev.get_pixel(kpt_x + 1, kpt_y).0[0];
            let p211 = prev.get_pixel(kpt_x - 1, kpt_y).0[0];

            let p022 = next.get_pixel(kpt_x, kpt_y + 1).0[0];
            let p012 = next.get_pixel(kpt_x, kpt_y - 1).0[0];
            let p021 = prev.get_pixel(kpt_x, kpt_y + 1).0[0];
            let p011 = prev.get_pixel(kpt_x, kpt_y - 1).0[0];

            let pt = img.get_pixel(kpt_x, kpt_y).0[0];
            // 一阶导数
            let dD = [(x2 - x1) * 0.5, (y2 - y1) * 0.5, (s2 - s1) * 0.5];
            // 二阶导数
            let dxx = x1 + x2 - 2.0 * pt;
            let dyy = y1 + y2 - 2.0 * pt;
            let dss = s1 + s2 - 2.0 * pt;
            // 混合导数
            let dxy = (p122 - p112 - p121 + p111) * 0.25;
            let dxs = (p222 - p212 - p221 + p211) * 0.25;
            let dys = (p022 - p012 - p021 + p011) * 0.25;

            // 组合Hessian矩阵
            let hessian = Matrix3::<f32>::new(dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss);
            let dD = Vector3::<f32>::new(dD[0], dD[1], dD[2]);
            let x_hat = hessian.try_inverse().unwrap_or(Matrix3::<f32>::identity()) * dD;
            let x_hat = -x_hat;
            kpt_x += x_hat[0].round() as u32;
            kpt_y += x_hat[1].round() as u32;
            kpt_scale += x_hat[2].round() as usize;
            if kpt_x < IMG_BORDER as u32
                || kpt_x > img.width() - IMG_BORDER as u32
                || kpt_y < IMG_BORDER as u32
                || kpt_y > img.height() - IMG_BORDER as u32
                || kpt_scale < 1
                || kpt_scale > SCALES_PER_OCTAVE as usize
            {
                // 越界，删除并跳过此候选点
                delete_point_index.push(i);
                continue 'pt;
            }
            let D_hat = img.get_pixel(kpt_x, kpt_y).0[0] + dD.dot(&x_hat) * 0.5;
            kpt.response = D_hat; // 响应值
            // hessian矩阵的迹和行列式
            tr_h = dxx + dyy;
            det_h = dxx * dyy - dxy * dxy;
            if x_hat[0] < BIAS_THRESHOLD && x_hat[1] < BIAS_THRESHOLD && x_hat[1] < BIAS_THRESHOLD {
                break;
            }
        }
        let small_response =
            kpt.response.abs() * (SCALES_PER_OCTAVE as f32) < CONTRAST_THRESHOLD as f32;
        let edge = det_h <= 0.0
            || (tr_h * tr_h * EDGE_THRESHOLD >= (EDGE_THRESHOLD + 1.0).powi(2) * det_h);
        kpt.x = kpt_x;
        kpt.y = kpt_y;
        kpt.scale = kpt_scale;
        kpt.size = 2_f32
            .powf(1.0 / SCALES_PER_OCTAVE as f32)
            .powi(kpt_scale as i32)
            * SIGMA0 as f32;
        if small_response || edge {
            delete_point_index.push(i);
        }
    }
    let new_pts: Vec<SIFTKeyPoint> = keypoints
        .drain(..)
        .enumerate()
        .filter_map(|(i, pt)| {
            if !delete_point_index.contains(&i) {
                Some(pt)
            } else {
                None
            }
        })
        .collect();
    DoGAndKPComputed {
        octave_cnt,
        pyr_g,
        pyr_dog,
        keypoints: new_pts,
    }
}

/// 计算特征点主方向
fn calculate_main_orientation(
    DoGAndKPComputed {
        octave_cnt,
        pyr_g,
        pyr_dog,
        mut keypoints,
    }: DoGAndKPComputed,
) -> DoGAndKPComputed {
    // 辅方向作为新的关键点加入
    let mut added_keypoints: Vec<SIFTKeyPoint> = Vec::new();
    for kpt in keypoints.iter_mut() {
        let radius = (3.0 * 1.5 * kpt.size).round() as i32;
        let img = &pyr_g[kpt.octave][kpt.scale];
        // 柱状图
        let mut hist: [f32; 40] = [0.0; 40];
        // 遍历radius内所有点，包括边界处理
        for i in -radius..=radius {
            let px = kpt.x as i32 + i;
            if px <= 0 || px >= (img.width() - 1) as i32 {
                continue;
            }
            for j in -radius..=radius {
                let py = kpt.y as i32 + j;
                if py <= 0 || py >= (img.height() - 1) as i32 {
                    continue;
                }
                // 计算梯度
                let dx = img.get_pixel(px as u32 + 1, py as u32).0[0]
                    - img.get_pixel(px as u32 - 1, py as u32).0[0];
                let dy = img.get_pixel(px as u32, py as u32 + 1).0[0]
                    - img.get_pixel(px as u32, py as u32 - 1).0[0];

                let mag = (dx * dx + dy * dy).sqrt();
                let ang = dy.atan2(dx);
                let bin = (ang + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
                let bin = (bin * 36.0).round() as usize;
                // 高斯加权项
                let t0 = -(i * i + j * j) as f32;
                let t1 = 2.0 * (1.5 * kpt.size).powi(2);
                let w_g = (t0 / t1).exp();

                hist[bin + 2] += mag * w_g;
            }
        }
        // 方便实现平滑
        hist[0] = hist[36];
        hist[1] = hist[37];
        hist[38] = hist[2];
        hist[39] = hist[3];
        let mut smth_hist = [0_f32; 36];
        let mut hist_max = 0_f32;
        for k in 2..38 {
            smth_hist[k - 2] = (hist[k - 2] + hist[k + 2]) * 1.0 / 16.0
                + (hist[k - 1] + hist[k + 1]) * 4.0 / 16.0
                + hist[k] * 6.0 / 16.0;
            if smth_hist[k - 2] > hist_max {
                hist_max = smth_hist[k - 2];
            }
        }
        let hist_threshold = 0.8 * hist_max;
        let mut more_orient = false;
        for k in 0..36 {
            let kl = if k > 0 { k - 1 } else { 35 };
            let kr = if k < 35 { k + 1 } else { 0 };
            if smth_hist[k] > smth_hist[kl]
                && smth_hist[k] > smth_hist[kr]
                && smth_hist[k] >= hist_threshold
            {
                // 插值
                let bin = k as f32
                    + 0.5 * (smth_hist[kl] - smth_hist[kr])
                        / (smth_hist[kl] - 2.0 * smth_hist[k] + smth_hist[kr]);
                let bin = if bin < 0.0 {
                    bin + 36.0
                } else if bin >= 36.0 {
                    bin - 36.0
                } else {
                    bin
                };
                let ang = bin * 10.0;
                let ang = if (ang - 360.0).abs() < f32::EPSILON {
                    0.0
                } else {
                    ang
                };

                if more_orient {
                    added_keypoints.push(SIFTKeyPoint { angle: ang, ..*kpt });
                } else {
                    more_orient = true;
                    kpt.angle = ang;
                }
            }
        }
    }
    keypoints.append(&mut added_keypoints);
    DoGAndKPComputed {
        octave_cnt,
        pyr_g,
        pyr_dog,
        keypoints,
    }
}

fn calculate_feature(
    DoGAndKPComputed {
        octave_cnt,
        pyr_g,
        pyr_dog,
        mut keypoints,
    }: DoGAndKPComputed,
) -> DoGAndKPComputed {
    // 4 个子空间
    const D: u32 = 4;
    // 8 个幅角区间
    const N: u32 = 8;
    const HIST_LEN: usize = ((D + 2) * (D + 2) * (N + 2)) as usize;
    for kpt in keypoints.iter_mut() {
        let img = &pyr_g[kpt.octave][kpt.scale];
        let hist_width = 3.0 * kpt.size;
        let cos_t = (kpt.angle * PI / 180.0).cos() / hist_width;
        let sin_t = (kpt.angle * PI / 180.0).sin() / hist_width;

        let radius = (hist_width * 2_f32.sqrt() * (D as f32 + 1.0) * 0.5).round();
        let radius = radius as u32;

        let radius = radius.min((img.width().pow(2) + img.height().pow(2)).isqrt()) as i32;
        let mut hist = [0_f32; HIST_LEN];

        for i in -radius..radius {
            for j in -radius..radius {
                let x_rot = i as f32 * cos_t - j as f32 * sin_t;
                let y_rot = i as f32 * sin_t + j as f32 * cos_t;

                let x_bin = x_rot + D as f32 / 2.0 - 0.5;
                let y_bin = y_rot + D as f32 / 2.0 - 0.5;
                let px_x = kpt.x as i32 + i;
                let px_y = kpt.y as i32 + j;
                if x_bin < -1.0
                    || x_bin >= D as f32
                    || y_bin < -1.0
                    || y_bin >= D as f32
                    || px_x <= 1
                    || px_x >= img.width() as i32 - 1
                    || px_y <= 1
                    || px_y >= img.height() as i32 - 1
                {
                    continue;
                }
                let px_x = px_x as u32;
                let px_y = px_y as u32;
                let dx = img.get_pixel(px_x + 1, px_y).0[0] - img.get_pixel(px_x - 1, px_y).0[0];
                let dy = img.get_pixel(px_x, px_y + 1).0[0] - img.get_pixel(px_x, px_y - 1).0[0];

                let mag = (dx * dx + dy * dy).sqrt();
                let ang = dy.atan2(dx) + PI;

                let obin = ang / PI * 180.0 - kpt.angle;
                let obin = if obin < 0.0 { obin + 360.0 } else { obin };
                let obin = obin / 360.0 * N as f32;

                let w_g = (-(x_rot * x_rot + y_rot * y_rot) / (0.5 * (D as f32).powi(2))).exp();
                let mag = mag * w_g;

                let x0 = x_bin.floor() as i32;
                let y0 = y_bin.floor() as i32;
                let o0 = obin.floor() as i32;

                let x_bin = x_bin - x0 as f32;
                let y_bin = y_bin - y0 as f32;
                let obin = obin - o0 as f32;

                // 三线性插值求贡献

                let v000 = x_bin * y_bin * obin;
                let v001 = x_bin * y_bin * (1.0 - obin);
                let v010 = x_bin * (1.0 - y_bin) * obin;
                let v011 = x_bin * (1.0 - y_bin) * (1.0 - obin);
                let v100 = (1.0 - x_bin) * y_bin * obin;
                let v101 = (1.0 - x_bin) * y_bin * (1.0 - obin);
                let v110 = (1.0 - x_bin) * (1.0 - y_bin) * obin;
                let v111 = (1.0 - x_bin) * (1.0 - y_bin) * (1.0 - obin);

                hist[(60 * (x0 + 1) + 10 * (y0 + 1) + o0) as usize] = mag * v000;
                hist[(60 * (x0 + 1) + 10 * (y0 + 1) + o0 + 1) as usize] = mag * v001;
                hist[(60 * (x0 + 1) + 10 * (y0 + 2) + o0) as usize] = mag * v010;
                hist[(60 * (x0 + 1) + 10 * (y0 + 2) + o0 + 1) as usize] = mag * v011;
                hist[(60 * (x0 + 2) + 10 * (y0 + 1) + o0) as usize] = mag * v100;
                hist[(60 * (x0 + 2) + 10 * (y0 + 1) + o0 + 1) as usize] = mag * v101;
                hist[(60 * (x0 + 2) + 10 * (y0 + 2) + o0) as usize] = mag * v110;
                hist[(60 * (x0 + 2) + 10 * (y0 + 2) + o0 + 1) as usize] = mag * v111;
            }
        }
        let mut k = 0;
        let mut feature = [0_f32; 128];
        for i in 1..=4 {
            for j in 1..=4 {
                hist[60 * i + 10 * j + 0] += hist[60 * i + 10 * j + 8];
                hist[60 * i + 10 * j + 1] += hist[60 * i + 10 * j + 9];

                for l in 0..8 {
                    feature[k] = hist[60 * i + 10 * j + l];
                    k += 1;
                }
            }
        }

        // 优化feature vector
        let mut fvec_norm = 0_f32;
        for k in 0..128 {
            fvec_norm += feature[k] * feature[k];
        }
        let fvec_threshold = 0.2_f32 * fvec_norm.sqrt();
        fvec_norm = 0_f32;
        for k in 0..128 {
            if feature[k] > fvec_threshold {
                feature[k] = fvec_threshold;
            }
            fvec_norm += feature[k] * feature[k];
        }

        let scl = 1_f32 / fvec_norm.sqrt().max(f32::EPSILON);
        for k in 0..128 {
            kpt.feature[k] = ((feature[k] * scl) * 255.0) as u8;
        }
    }

    DoGAndKPComputed {
        octave_cnt,
        pyr_g,
        pyr_dog,
        keypoints,
    }
}

pub fn draw_match(img1: &RgbImage, img2: &RgbImage, ppairs: &Vec<PointPair>) -> RgbImage {
    assert_eq!(img1.width(), img2.width());
    assert_eq!(img1.height(), img2.height());
    let mut result: RgbImage = ImageBuffer::new(img1.width() * 2, img2.height() * 2);
    imageops::replace(&mut result, img1, 0, 0);
    imageops::replace(&mut result, img2, img1.width() as i64, img1.height() as i64);
    for pp in ppairs.iter() {
        draw_antialiased_line_segment_mut(
            &mut result,
            (pp.x1 as i32, pp.y1 as i32),
            (
                (pp.x2 + img1.width()) as i32,
                (pp.y2 + img1.height()) as i32,
            ),
            Rgb::from([255, 0, 0]),
            pixelops::interpolate,
        );
    }
    result
}
impl DoGComputed {}
#[cfg(test)]
mod test {
    use std::f32::consts::PI;

    use image::{
        GrayImage, ImageBuffer, ImageReader, Rgb, RgbImage, buffer::ConvertBuffer, imageops,
    };
    use imageproc::{drawing::draw_antialiased_line_segment_mut, pixelops};

    use super::{
        LumaF32Image, SCALES_PER_OCTAVE, calculate_dog, calculate_feature,
        calculate_main_orientation, draw_match, filter_keypoints, find_keypoints, match_1nn,
        match_2nn, sift,
    };

    #[test]
    fn step1() {
        let image = ImageReader::open("./res/img0.jpg")
            .unwrap()
            .decode()
            .unwrap();
        let image = image.to_luma32f();
        let dog = calculate_dog(&image);

        let max_w = dog.pyr_g[0][0].width();
        let max_h = dog.pyr_g[0][0].height();

        let view_x = max_w * (SCALES_PER_OCTAVE as u32 + 3);
        let view_y = max_h * (dog.octave_cnt as u32).min(3);
        let mut g_view: LumaF32Image = ImageBuffer::new(view_x, view_y);
        let mut dog_view: LumaF32Image = ImageBuffer::new(view_x, view_y);
        for (i, octave) in dog.pyr_g.iter().enumerate() {
            if i >= (dog.octave_cnt).min(3) {
                break;
            }
            let y_offset = i as u32 * max_h;
            for (j, scale) in octave.iter().enumerate() {
                let x_offset = j as u32 * max_w;
                imageops::replace(&mut g_view, scale, x_offset as i64, y_offset as i64);
            }
        }
        for (i, octave) in dog.pyr_dog.iter().enumerate() {
            if i >= (dog.octave_cnt).min(3) {
                break;
            }
            let y_offset = i as u32 * max_h;
            for (j, scale) in octave.iter().enumerate() {
                let x_offset = j as u32 * max_w;
                imageops::replace(&mut dog_view, scale, x_offset as i64, y_offset as i64);
            }
        }
        let g_view: GrayImage = g_view.convert();
        for pix in dog_view.enumerate_pixels_mut() {
            pix.2[0] += 0.5;
        }
        let dog_view: GrayImage = dog_view.convert();
        g_view
            .save_with_format("./gaussian_tower.jpg", image::ImageFormat::Jpeg)
            .unwrap();
        dog_view
            .save_with_format("./dog_tower.jpg", image::ImageFormat::Jpeg)
            .unwrap();
    }
    #[test]
    fn step2() {
        let image = ImageReader::open("./res/img0.jpg")
            .unwrap()
            .decode()
            .unwrap();
        let image1 = image.to_luma32f();
        let dog = calculate_dog(&image1);
        let kpdog = find_keypoints(dog);
        let kpdog = filter_keypoints(kpdog);

        let mut image = image.to_rgb8();

        for kpt in kpdog.keypoints.iter() {
            let scale = 2_f32.powi(kpt.octave as i32 - 1);
            let x = (kpt.x as f32 * scale) as u32;
            let y = (kpt.y as f32 * scale) as u32;
            let pix = image.get_pixel_mut(x, y);
            *pix = Rgb::from([255, 0, 0]);
        }
        image
            .save_with_format("kpt.jpg", image::ImageFormat::Jpeg)
            .unwrap();
    }

    #[test]
    fn step3() {
        let image = ImageReader::open("./res/img0.jpg")
            .unwrap()
            .decode()
            .unwrap();
        let image1 = image.to_luma32f();
        let dog = calculate_dog(&image1);
        let kpdog = find_keypoints(dog);
        let kpdog = filter_keypoints(kpdog);
        let kpdog = calculate_main_orientation(kpdog);
        let kpdog = calculate_feature(kpdog);

        let mut image = image.to_rgb8();

        for kpt in kpdog.keypoints.iter() {
            let scale = 2_f32.powi(kpt.octave as i32 - 1);
            let x = (kpt.x as f32 * scale) as i32;
            let y = (kpt.y as f32 * scale) as i32;
            let tan = (kpt.angle / 180.0 * PI).tan();
            dbg!(kpt.angle);
            if tan > 10.0 {
                for i in 1..5 {
                    let pix = image.get_pixel_mut((x + i) as u32, (y + i) as u32);
                    *pix = Rgb::from([0, 255, 0]);
                }
            } else if tan < 10.0 {
                for i in 1..5 {
                    let pix = image.get_pixel_mut((x + i) as u32, (y - i) as u32);
                    *pix = Rgb::from([0, 255, 0]);
                }
            } else {
                for i in 1..5 {
                    let pix =
                        image.get_pixel_mut((x + i) as u32, (y + (tan * (i as f32)) as i32) as u32);
                    *pix = Rgb::from([0, 255, 0]);
                }
            }
        }
        image
            .save_with_format("kpt_drct.jpg", image::ImageFormat::Jpeg)
            .unwrap();
    }

    #[test]
    fn step4() {
        let image = ImageReader::open("./res/usb1.jpg")
            .unwrap()
            .decode()
            .unwrap();
        let image = image.resize(
            image.width() / 2,
            image.height() / 2,
            imageops::FilterType::Gaussian,
        );
        let image_ref = image.to_rgb8();
        let image_ref_flip = imageops::flip_horizontal(&image_ref);
        let image = image.to_luma32f();
        let image_flip = imageops::flip_horizontal(&image);
        let res = sift(&image);
        let res_flip = sift(&image_flip);

        let point_pairs = match_2nn(&res, &res_flip);
        let mut result: RgbImage = ImageBuffer::new(image.width() * 2, image.height() * 2);
        // let image_ref_flip = imageops::resize(
        //     &image_ref_flip,
        //     image_ref_flip.width() * 4,
        //     image_ref_flip.height() * 4,
        //     imageops::FilterType::Gaussian,
        // );

        imageops::replace(&mut result, &image_ref, 0, 0);
        imageops::replace(
            &mut result,
            &image_ref_flip,
            image_ref.width() as i64,
            image_ref.height() as i64,
        );
        for pp in point_pairs.iter().step_by(16) {
            draw_antialiased_line_segment_mut(
                &mut result,
                (pp.x1 as i32, pp.y1 as i32),
                (
                    (pp.x2 + image_ref.width()) as i32,
                    (pp.y2 + image_ref.height()) as i32,
                ),
                Rgb::from([255, 0, 0]),
                pixelops::interpolate,
            );
        }
        result
            .save_with_format("match4_1.jpg", image::ImageFormat::Jpeg)
            .unwrap();
    }
    #[test]
    fn step4_1() {
        let image1 = ImageReader::open("./res/usb1.jpg")
            .unwrap()
            .decode()
            .unwrap();
        let image2 = ImageReader::open("./res/usb2.jpg")
            .unwrap()
            .decode()
            .unwrap();
        let image1 = image1.resize(
            image1.width() / 2,
            image1.height() / 2,
            imageops::FilterType::Gaussian,
        );
        let image2 = image2.resize(
            image2.width() / 2,
            image2.height() / 2,
            imageops::FilterType::Gaussian,
        );
        let image1_ref = image1.to_rgb8();
        let image2_ref = image2.to_rgb8();
        let image1 = image1.to_luma32f();
        let image2 = image2.to_luma32f();
        let res1 = sift(&image1);
        let res2 = sift(&image2);

        let point_pairs = match_2nn(&res1, &res2);
        let result = draw_match(&image1_ref, &image2_ref, &point_pairs);
        result
            .save_with_format("match4.jpg", image::ImageFormat::Jpeg)
            .unwrap();
    }
}
