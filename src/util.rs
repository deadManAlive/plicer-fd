use std::f32::consts::PI;

// pub fn append_path<'a>(path: &'a Path, new_str: &'a str) -> PathBuf {
//     let parent = path.parent().unwrap();
//     let stem = path.file_stem().unwrap().to_string_lossy();
//     let extension = path.extension().unwrap().to_string_lossy();

//     let new_stem = format!("{}{}", stem, new_str);
//     let new_filename = format!("{}.{}", new_stem, extension);

//     parent.join(new_filename)
// }

pub fn sigmoid_mapper(x: f32, thresh: f32) -> f32 {
    let w = 8f32;

    let res = -2f32 * w * PI * (x - thresh);
    let res = res.exp();
    let res = 1f32 + res;

    res.recip()
}