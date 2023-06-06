use std::{path::Path, time::Instant};

use plicer_fd;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let filename = match args.len() {
        1 => return Err("no file provided")?,
        _ => args.get(1),
    };

    let start = Instant::now();
    let direction = plicer_fd::process(Path::new(&filename.unwrap()))?;
    let stop = start.elapsed();

    println!("It took {:.5} s to find out that '{}' was tilted {:?}", stop.as_secs_f32(), filename.unwrap(), direction);

    Ok(())
}