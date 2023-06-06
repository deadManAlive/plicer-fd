use std::path::{Path};

use image::{GenericImageView};
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

#[derive(Debug)]
pub enum Direction {
    Up,
    Right,
    Down,
    Left,
}

use Direction::*;

use crate::util::sigmoid_mapper;

pub fn process(input: &Path) -> Result<Direction, Box<dyn std::error::Error>> {
    let model = include_bytes!("mtcnn.pb");

    let mut graph = Graph::new();
    graph.import_graph_def(model, &ImportGraphDefOptions::new())?;

    let mut input_image = image::open(input)?;

    let mut dsize: Vec<(usize, f32)> = vec![(0, 0f32); 4];

    for (i, _) in [Up, Right, Down, Left].iter().enumerate() {
        let image = input_image.clone();

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

        let dcount = bboxes.len();
        let prbavg = bboxes.iter().fold(0f32, |acc, bbox| acc + sigmoid_mapper(bbox.prob, 0.85));

        dsize[i] = (
            dcount,
            if dcount == 0 {
                0f32
            } else {
                prbavg / (dcount as f32)
            },
        );

        input_image = input_image.rotate90();
    }

    let max_idx = dsize
        .iter()
        .enumerate()
        .max_by(|(_, &(_, a)),(_, &(_, b))| a.total_cmp(&b))
        .map(|(idx, _)| idx)
        .unwrap();

    match max_idx {
        0 => Ok(Up),
        1 => Ok(Left),
        2 => Ok(Down),
        3 => Ok(Right),
        _ => Err("Something is wrong...")?,
    }
}
