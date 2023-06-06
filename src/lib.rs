use std::path::{Path, PathBuf};

use image::{GenericImageView, Rgba};
use imageproc::{rect::Rect, drawing::draw_hollow_rect_mut};
use tensorflow::{Graph, ImportGraphDefOptions, Session, SessionOptions, SessionRunArgs, Tensor};

mod util;

#[derive(Debug, Clone, Copy)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub prob: f32,
}

const RED: Rgba<u8> = Rgba([255, 0, 0, 0]);

pub fn process(input: &Path, output: Option<&String>) -> Result<(), Box<dyn std::error::Error>> {
    let model = include_bytes!("mtcnn.pb");

    let mut graph = Graph::new();
    graph.import_graph_def(model, &ImportGraphDefOptions::new())?;

    let mut input_image = image::open(input)?;
    
    // TODO: input_image.rotate to find image possible rotation
    for i in 0..4 {
        let image = input_image.clone();
        let output = match output {
            Some(v) => PathBuf::from(v),
            None => {
                util::append_path(input, format!("-det{}", 90 * i).as_str())
            },
        };

        let mut flattened: Vec<f32> = Vec::new();
    
        for (_, _, rgb) in image.pixels() {
            flattened.push(rgb[2] as f32);
            flattened.push(rgb[1] as f32);
            flattened.push(rgb[0] as f32);
        }
    
        let input = Tensor::new(&[image.height() as u64, image.width() as u64, 3])
            .with_values(&flattened)?;
    
        let session = Session::new(&SessionOptions::new(), &graph)?;
    
        let min_size = Tensor::new(&[]).with_values(&[40f32])?;
        let thresholds = Tensor::new(&[3]).with_values(&[0.6f32, 0.7f32, 0.7f32])?;
        let factor = Tensor::new(&[]).with_values(&[0.709f32])?;
    
        let mut args = SessionRunArgs::new();
    
        args.add_feed(&graph.operation_by_name_required("min_size")?, 0, &min_size);
        args.add_feed(
            &graph.operation_by_name_required("thresholds")?,
            0,
            &thresholds,
        );
        args.add_feed(&graph.operation_by_name_required("factor")?, 0, &factor);
        args.add_feed(&graph.operation_by_name_required("input")?, 0, &input);
    
        let bbox = args.request_fetch(&graph.operation_by_name_required("box")?, 0);
        let prob = args.request_fetch(&graph.operation_by_name_required("prob")?, 0);
    
        session.run(&mut args)?;
    
        let bbox_res: Tensor<f32> = args.fetch(bbox)?;
        let prob_res: Tensor<f32> = args.fetch(prob)?;
    
        let bboxes: Vec<_> = bbox_res
            .chunks_exact(4)
            .zip(prob_res.iter())
            .map(|(bbox, &prob)| BBox {
                y1: bbox[0],
                x1: bbox[1],
                y2: bbox[2],
                x2: bbox[3],
                prob,
            })
            .collect();
    
        println!("BBox Length: {}, BBoxes:{:#?}", bboxes.len(), bboxes);
    
        let mut output_image = image;
    
        for bbox in bboxes {
            let rect = Rect::at(bbox.x1 as i32, bbox.y1 as i32)
                .of_size((bbox.x2 - bbox.x1) as u32, (bbox.y2 - bbox.y1) as u32);
    
            draw_hollow_rect_mut(&mut output_image, rect, RED);
        }
    
        output_image.save(output)?;
        input_image = input_image.rotate90();
    }


    Ok(())
}
