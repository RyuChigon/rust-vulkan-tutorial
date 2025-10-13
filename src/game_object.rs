use std::rc::Rc;

use ash::{Device, vk};
use cgmath::Matrix4;

pub struct Mesh {
    pub device: Rc<Device>,
    pub vertex_buffer: vk::Buffer,
    pub vertex_buffer_memory: vk::DeviceMemory,
    pub index_buffer: vk::Buffer,
    pub index_buffer_memory: vk::DeviceMemory,
    pub index_count: u32,
}

impl Drop for Mesh {
    fn drop(&mut self) {
        unsafe {
            self.device.free_memory(self.index_buffer_memory, None);
            self.device.destroy_buffer(self.index_buffer, None);

            self.device.free_memory(self.vertex_buffer_memory, None);
            self.device.destroy_buffer(self.vertex_buffer, None);
        }
    }
}

pub struct Material {
    pub device: Rc<Device>,
    pub texture_image: vk::Image,
    pub texture_image_memory: vk::DeviceMemory,
    pub texture_image_view: vk::ImageView,
    pub texture_sampler: vk::Sampler,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
}

impl Drop for Material {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_sampler(self.texture_sampler, None);
            self.device
                .destroy_image_view(self.texture_image_view, None);
            self.device.free_memory(self.texture_image_memory, None);
            self.device.destroy_image(self.texture_image, None);
        }
    }
}

pub struct GameObject {
    pub mesh: Rc<Mesh>,
    pub material: Rc<Material>,
    pub transform: Matrix4<f32>,
}
