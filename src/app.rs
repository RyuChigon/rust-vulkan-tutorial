use std::rc::Rc;

use cgmath::Matrix4;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    window::{Window, WindowAttributes},
};

use crate::{game_object::GameObject, loader::Loader, renderer::Renderer};

pub const WIDTH: u32 = 1024;
pub const HEIGHT: u32 = 768;

#[derive(Default)]
pub struct App {
    window: Option<Window>,
    game_objects: Vec<GameObject>,
    renderer: Option<Renderer>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(
                WindowAttributes::default()
                    .with_title("Vulkan tutorial")
                    .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT)),
            )
            .unwrap();

        let renderer = Renderer::new(&window);

        let (vertices, indices) = Loader::load_model("models/viking_room.obj");

        let viking_mesh = Rc::new(renderer.create_mesh(&vertices, &indices));
        let viking_material = Rc::new(renderer.create_material("textures/viking_room.png"));

        let viking_object = GameObject {
            mesh: viking_mesh,
            material: viking_material,
            transform: Matrix4::from_scale(1.0),
        };

        self.game_objects.push(viking_object);
        self.renderer = Some(renderer);
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_dimensions) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.handle_window_resized(new_dimensions.width, new_dimensions.height);
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        let renderer = self.renderer.as_mut().unwrap();
        let window = self.window.as_ref().unwrap();

        if renderer.dirty_swapchain {
            let size = window.inner_size();
            if size.width > 0 && size.height > 0 {
                renderer.recreate_swapchain();
            } else {
                // Handling minimization
                return;
            }
        }

        renderer.dirty_swapchain = renderer.draw_frame(&self.game_objects);
    }

    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(renderer) = self.renderer.as_ref() {
            renderer.wait_device_idle();
        }
    }
}
