mod app;
mod debug;
mod game_object;
mod loader;
mod math;
mod renderer;
mod swapchain_properties;
mod vertex;

use crate::app::App;
use winit::event_loop::EventLoop;

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
