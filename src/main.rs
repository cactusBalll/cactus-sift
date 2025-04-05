use clap::Parser;
use image::ImageReader;
mod sift;
/// Simple SIFT algorithm implementation
#[derive(Parser, Debug)]
struct Args {
    /// 第一张图片
    image1: std::path::PathBuf,
    /// 第二张图片
    image2: std::path::PathBuf,
}

fn main() {
    let args = Args::parse();
    let image1 = ImageReader::open(args.image1).unwrap().decode().unwrap();
    let image2 = ImageReader::open(args.image2).unwrap().decode().unwrap();
}

#[cfg(test)]
mod test {
    use image::ImageBuffer;
    use image::ImageReader;
    use image::Luma;
    type LumaF32Image = ImageBuffer<Luma<f32>, Vec<f32>>;
    #[test]
    fn img_fmt() {
        let image = ImageReader::open("./res/img0.jpg")
            .unwrap()
            .decode()
            .unwrap();
        let image = image.to_luma32f();
        for pix in image.enumerate_pixels() {
            println!("{:?}", pix.2);
        }
    }
}
