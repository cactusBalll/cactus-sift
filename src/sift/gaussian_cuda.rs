use image::{ImageBuffer, Luma, imageops};

type LumaF32Image = ImageBuffer<Luma<f32>, Vec<f32>>;
type FuncT = fn(dest: *const f32, src: *const f32, image_h: i32, image_w: i32, sigma: f32);
#[link(name="gaussian_lib", kind = "static")]
unsafe extern "C" {
    fn gaussian_gpu(dest: *const f32, src: *const f32, image_h: i32, image_w: i32, sigma: f32);
    fn gaussian_cpu(dest: *const f32, src: *const f32, image_h: i32, image_w: i32, sigma: f32);
}

#[link(name="gaussian_cv2", kind = "static")]
unsafe extern "C" {
    fn gaussian_cv2_simd(dest: *const f32, src: *const f32, image_h: i32, image_w: i32, sigma: f32);
    fn gaussian_cv2_cuda(dest: *const f32, src: *const f32, image_h: i32, image_w: i32, sigma: f32);
}

pub fn blur(image_in: &LumaF32Image, sigma: f32) -> LumaF32Image {
    if image_in.width() > 128 && image_in.height() > 64 {
        let p_src = image_in.as_raw().as_ptr();
        let dest = vec![0.0_f32; (image_in.width() * image_in.height()) as usize];
        let p_dest = dest.as_ptr();
        unsafe {
            gaussian_gpu(
                p_dest,
                p_src,
                image_in.height() as i32,
                image_in.width() as i32,
                sigma,
            );
        }
        LumaF32Image::from_raw(image_in.width(), image_in.height(), dest).unwrap()
    } else {
        imageops::fast_blur(image_in, sigma)
    }
}
