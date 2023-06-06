use std::{path::Path, time::Instant};

use plicer_fd;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let (filename, outputfile) = match args.len() {
        2 => (args.get(1), None),
        3 => (args.get(1), args.get(2)),
        _ => return Err("no file provided")?
    };

    let start = Instant::now();
    plicer_fd::process(Path::new(&filename.unwrap()), outputfile)?;
    let stop = start.elapsed();

    println!("Process took {} us for {}", stop.as_micros(), filename.unwrap());

    Ok(())
}