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
                single_index: true,
                ..Default::default()
            },
        )
        .unwrap();

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for mesh in models.iter().map(|m| &m.mesh) {
            for i in 0..mesh.indices.len() {
                let index = mesh.indices[i] as usize;

                let pos_offset = 3 * index;
                let normal_offset = 3 * index;
                let tex_coord_offset = 2 * index;

                let vertex = Vertex {
                    pos: [
                        mesh.positions[pos_offset],
                        mesh.positions[pos_offset + 1],
                        mesh.positions[pos_offset + 2],
                    ],
                    color: [1.0, 1.0, 1.0],
                    tex_coord: if !mesh.texcoords.is_empty() {
                        [
                            mesh.texcoords[tex_coord_offset],
                            1.0 - mesh.texcoords[tex_coord_offset + 1],
                        ]
                    } else {
                        [0.0, 0.0]
                    },
                    normal: if !mesh.normals.is_empty() {
                        [
                            mesh.normals[normal_offset],
                            mesh.normals[normal_offset + 1],
                            mesh.normals[normal_offset + 2],
                        ]
                    } else {
                        [0.0, 0.0, 0.0]
                    },
                };

                vertices.push(vertex);
                indices.push(i as u32);
            }
        }

        log::info!(
            "After processing, vertices len: {}, indices len: {}",
            vertices.len(),
            indices.len()
        );

        (vertices, indices)
    }
}
