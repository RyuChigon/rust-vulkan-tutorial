use std::collections::HashMap;

use tobj::LoadOptions;

use crate::vertex::Vertex;

pub struct Loader;

impl Loader {
    pub fn load_model(file_name: &str) -> (Vec<Vertex>, Vec<u32>) {
        let (models, _) = tobj::load_obj(
            file_name,
            &LoadOptions {
                triangulate: true,
                ..Default::default()
            },
        )
        .unwrap();

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        {
            let vertex_count = models
                .iter()
                .map(|m| m.mesh.positions.len() / 3)
                .sum::<usize>();
            let index_count = models.iter().map(|m| m.mesh.indices.len()).sum::<usize>();
            log::info!(
                "Before deduplicating, vertices len: {}, indices len: {}",
                vertex_count,
                index_count
            );
        }

        let mut unique_vertices = HashMap::new();

        for mesh in models.iter().map(|m| &m.mesh) {
            for index in &mesh.indices {
                let pos_offset = (3 * index) as usize;
                let tex_coord_offset = (2 * index) as usize;

                let vertex = Vertex {
                    pos: [
                        mesh.positions[pos_offset],
                        mesh.positions[pos_offset + 1],
                        mesh.positions[pos_offset + 2],
                    ],
                    color: [1.0, 1.0, 1.0],
                    tex_coord: [
                        mesh.texcoords[tex_coord_offset],
                        1.0 - mesh.texcoords[tex_coord_offset + 1],
                    ],
                };

                if let Some(index) = unique_vertices.get(&vertex) {
                    indices.push(*index as u32);
                } else {
                    let index = vertices.len();
                    unique_vertices.insert(vertex, index);
                    vertices.push(vertex);
                    indices.push(index as u32);
                }
            }
        }

        log::info!(
            "After deduplicating, vertices len: {}, indices len: {}",
            vertices.len(),
            indices.len()
        );

        (vertices, indices)
    }
}
