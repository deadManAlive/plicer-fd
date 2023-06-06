use std::path::{Path, PathBuf};

pub fn append_path<'a>(path: &'a Path, new_str: &'a str) -> PathBuf {
    let parent = path.parent().unwrap();
    let stem = path.file_stem().unwrap().to_string_lossy();
    let extension = path.extension().unwrap().to_string_lossy();

    let new_stem = format!("{}{}", stem, new_str);
    let new_filename = format!("{}.{}", new_stem, extension);

    parent.join(new_filename)
}
