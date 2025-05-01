extern crate cactus_sift;

use std::time::Duration;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use image::{ImageReader, Rgb};
use imageproc::drawing::draw_hollow_circle_mut;

use cactus_sift::sift::sift;

#[inline]
fn sift_bench_func() {
    let image = ImageReader::open("./res/arch.jpeg")
        .unwrap()
        .decode()
        .unwrap();
    let image1 = image.to_luma32f();
    let kpdog = sift(&image1);

    let mut image = image.to_rgb8();

    for kpt in kpdog.keypoints.iter() {
        let x = kpt.x as i32;
        let y = kpt.y as i32;
        let scale = 2_f32.powi(kpt.octave as i32 - 1);
        draw_hollow_circle_mut(
            &mut image,
            (x, y),
            (kpt.size * scale) as i32,
            Rgb([255, 0, 0]),
        );
    }
    // 保留副作用，防止优化器过度优化影响benchmark
    image
        .save_with_format("arch_kpt_cuda.jpg", image::ImageFormat::Jpeg)
        .unwrap();
}
pub fn sift_benchmark(c: &mut Criterion) {
    c.bench_function("sift impl gpu", |b| b.iter(|| sift_bench_func()));
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(256)
        .measurement_time(Duration::from_secs(2500));
    targets = sift_benchmark
);
criterion_main!(benches);
