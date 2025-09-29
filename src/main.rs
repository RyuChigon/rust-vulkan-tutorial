mod debug;
mod swapchain_properties;

use std::{
    ffi::CStr,
    fs::File,
    io::{Cursor, Read},
};

use ash::{
    Device, Instance,
    ext::debug_utils,
    vk::{self, DebugUtilsMessengerEXT},
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::EventLoop,
    window::{Window, WindowAttributes},
};

use crate::{
    debug::{
        ENABLE_VALIDATION_LAYERS, check_validation_layer_support, create_debug_create_info,
        get_layer_names_and_pointers, setup_debug_messenger,
    },
    swapchain_properties::{SwapchainProperties, SwapchainSupportDetails},
};

const WIDTH: u32 = 1080;
const HEIGHT: u32 = 720;

#[derive(Default)]
struct App {
    vulkan: Option<VulkanApp>,
    window: Option<Window>,
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

        self.vulkan = Some(VulkanApp::new(&window));
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
            WindowEvent::Resized(_new_dimensions) => {
                // TODO
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let app = self.vulkan.as_mut().unwrap();
        let window = self.window.as_ref().unwrap();
    }
}

struct VulkanApp {
    instance: Instance,
    debug_messenger: Option<(debug_utils::Instance, DebugUtilsMessengerEXT)>,
    physical_device: vk::PhysicalDevice,
    device: Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    surface_instance: ash::khr::surface::Instance,
    surface_khr: vk::SurfaceKHR,

    swapchain_device: ash::khr::swapchain::Device,
    swapchain_khr: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_properties: SwapchainProperties,
    swapchain_image_views: Vec<vk::ImageView>,

    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl VulkanApp {
    fn new(window: &Window) -> Self {
        let entry = unsafe { ash::Entry::load().unwrap() };

        let instance = Self::create_instance(&entry, window);

        let debug_messenger = setup_debug_messenger(&entry, &instance);

        let (surface_instance, surface_khr) = Self::create_surface(&entry, &instance, window);

        let physical_device = Self::pick_physical_device(&instance, &surface_instance, surface_khr);

        let (graphics_index, present_index) =
            Self::find_queue_families(&instance, physical_device, &surface_instance, surface_khr);
        let (graphics_index, present_index) = (graphics_index.unwrap(), present_index.unwrap());

        let (device, graphics_queue, present_queue) = Self::create_logical_device_and_queue(
            &instance,
            physical_device,
            graphics_index,
            present_index,
        );

        let (swapchain_device, swapchain_khr, swapchain_images, swapchain_properties) =
            Self::create_swapchain(
                &instance,
                &device,
                physical_device,
                &surface_instance,
                surface_khr,
                graphics_index,
                present_index,
            );

        let swapchain_image_views =
            Self::create_swapchain_image_views(&device, &swapchain_images, &swapchain_properties);

        let (pipeline, pipeline_layout) = Self::create_pipeline(&device, &swapchain_properties);

        Self {
            instance,
            debug_messenger,
            physical_device,
            device,
            graphics_queue,
            present_queue,
            surface_instance,
            surface_khr,

            swapchain_device,
            swapchain_khr,
            swapchain_images,
            swapchain_properties,
            swapchain_image_views,

            pipeline_layout,
            pipeline,
        }
    }

    fn create_instance(entry: &ash::Entry, window: &Window) -> Instance {
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Vulkan Application")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"No Engine")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, 1, 4, 0));

        let extension_names =
            ash_window::enumerate_required_extensions(window.display_handle().unwrap().as_raw())
                .unwrap();

        let mut extension_names = extension_names.to_vec();
        if ENABLE_VALIDATION_LAYERS {
            extension_names.push(debug_utils::NAME.as_ptr());
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            extension_names.push(ash::khr::portability_enumeration::NAME.as_ptr());
            // Enabling this extension is a requirement when using `VK_KHR_portability_subset`
            extension_names.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
        }

        // _layer_names를 _로 하면 ERROR_LAYER_NOT_PRESENT가 발생.
        // _로 하면 layer_names를 Drop 시키거나 버리기 때문
        let (_layer_names, layer_names_ptrs) = get_layer_names_and_pointers();

        let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::default()
        };

        let mut debug_create_info = create_debug_create_info();

        let mut instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .flags(create_flags);

        if ENABLE_VALIDATION_LAYERS {
            check_validation_layer_support(entry);
            instance_info = instance_info
                .enabled_layer_names(&layer_names_ptrs)
                .push_next(&mut debug_create_info);
        }

        unsafe { entry.create_instance(&instance_info, None).unwrap() }
    }

    fn pick_physical_device(
        instnace: &Instance,
        surface_instance: &ash::khr::surface::Instance,
        surface_khr: vk::SurfaceKHR,
    ) -> vk::PhysicalDevice {
        let devices = unsafe { instnace.enumerate_physical_devices().unwrap() };
        let physical_device = devices
            .into_iter()
            .find(|d| Self::is_device_suitable(instnace, *d, surface_instance, surface_khr))
            .expect("failed to find a suitable GPU.");

        physical_device
    }

    fn is_device_suitable(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface_instance: &ash::khr::surface::Instance,
        surface_khr: vk::SurfaceKHR,
    ) -> bool {
        let (graphics_index, present_index) =
            Self::find_queue_families(instance, physical_device, surface_instance, surface_khr);
        if graphics_index.is_none() {
            log::error!("can't find graphics queue");
            return false;
        }

        if present_index.is_none() {
            log::error!("can't find present queue");
            return false;
        }

        if !Self::check_device_extension_support(instance, physical_device) {
            log::error!("device does not support extensions");
            return false;
        }

        let device_properties = unsafe { instance.get_physical_device_properties(physical_device) };

        if device_properties.api_version < vk::make_api_version(0, 1, 3, 0) {
            return false;
        }

        true
    }

    fn check_device_extension_support(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
    ) -> bool {
        let required_extensions = Self::get_required_device_extensions();

        let extension_props = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device)
                .unwrap()
        };

        for required in required_extensions.iter() {
            let found = extension_props.iter().any(|p| {
                let name = unsafe { CStr::from_ptr(p.extension_name.as_ptr()) };
                *required == name
            });

            if !found {
                return false;
            }
        }

        true
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    fn get_required_device_extensions() -> [&'static CStr; 4] {
        [
            ash::khr::swapchain::NAME,
            ash::khr::spirv_1_4::NAME,
            ash::khr::synchronization2::NAME,
            ash::khr::create_renderpass2::NAME,
        ]
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn get_required_device_extensions() -> [&'static CStr; 5] {
        [
            ash::khr::swapchain::NAME,
            ash::khr::spirv_1_4::NAME,
            ash::khr::synchronization2::NAME,
            ash::khr::create_renderpass2::NAME,
            ash::khr::portability_subset::NAME,
        ]
    }

    fn find_queue_families(
        instnace: &Instance,
        physical_device: vk::PhysicalDevice,
        surface_instance: &ash::khr::surface::Instance,
        surface_khr: vk::SurfaceKHR,
    ) -> (Option<u32>, Option<u32>) {
        let mut graphics = None;
        let mut present = None;

        let queue_family_properties =
            unsafe { instnace.get_physical_device_queue_family_properties(physical_device) };

        for (index, props) in queue_family_properties.iter().enumerate() {
            let index = index as u32;

            if props.queue_flags.contains(vk::QueueFlags::GRAPHICS) && graphics.is_none() {
                graphics = Some(index);
            }

            let present_support = unsafe {
                surface_instance
                    .get_physical_device_surface_support(physical_device, index, surface_khr)
                    .unwrap()
            };

            if present_support && present.is_none() {
                present = Some(index);
            }

            if graphics.is_some() && present.is_some() {
                break;
            }
        }

        (graphics, present)
    }

    fn create_logical_device_and_queue(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        graphics_index: u32,
        present_index: u32,
    ) -> (Device, vk::Queue, vk::Queue) {
        let queue_priorities = &[0.0f32];

        let queue_info = {
            let mut indices = vec![graphics_index, present_index];
            indices.dedup();

            indices
                .iter()
                .map(|index| {
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(*index)
                        .queue_priorities(queue_priorities)
                })
                .collect::<Vec<_>>()
        };

        let mut vulkan_13_features =
            vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
        let mut extended_dynamic_state_features =
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT::default()
                .extended_dynamic_state(true);
        let mut device_features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut vulkan_13_features)
            .push_next(&mut extended_dynamic_state_features);

        let device_extensions_ptrs = Self::get_required_device_extensions()
            .iter()
            .map(|e| e.as_ptr())
            .collect::<Vec<_>>();

        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&device_extensions_ptrs)
            .push_next(&mut device_features);

        let device = unsafe {
            instance
                .create_device(physical_device, &device_info, None)
                .unwrap()
        };

        let graphics_queue = unsafe { device.get_device_queue(graphics_index, 0) };
        let present_queue = unsafe { device.get_device_queue(present_index, 0) };

        (device, graphics_queue, present_queue)
    }

    fn create_surface(
        entry: &ash::Entry,
        instance: &Instance,
        window: &Window,
    ) -> (ash::khr::surface::Instance, vk::SurfaceKHR) {
        // surface 관련 확장 함수 사용을 위한 래퍼
        let surface_instance = ash::khr::surface::Instance::new(entry, instance);

        // 윈도우 핸들
        let surface_khr = unsafe {
            ash_window::create_surface(
                entry,
                instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )
            .unwrap()
        };

        (surface_instance, surface_khr)
    }

    fn create_swapchain(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        surface_instance: &ash::khr::surface::Instance,
        surface_khr: vk::SurfaceKHR,
        graphics_index: u32,
        present_index: u32,
    ) -> (
        ash::khr::swapchain::Device,
        vk::SwapchainKHR,
        Vec<vk::Image>,
        SwapchainProperties,
    ) {
        let swapchain_support_details =
            SwapchainSupportDetails::new(physical_device, surface_instance, surface_khr);

        let swapchain_properties =
            swapchain_support_details.get_ideal_swapchain_properties([WIDTH, HEIGHT]);

        let min_image_count = {
            let max = swapchain_support_details.capabilities.max_image_count;
            let mut preferred = swapchain_support_details.capabilities.min_image_count + 1;
            if max > 0 && preferred > max {
                preferred = max;
            }

            preferred
        };

        let queue_family_indices = &[graphics_index, present_index];

        let mut swapchain_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface_khr)
            .min_image_count(min_image_count)
            .image_format(swapchain_properties.format.format)
            .image_color_space(swapchain_properties.format.color_space)
            .image_extent(swapchain_properties.extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .pre_transform(swapchain_support_details.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(swapchain_properties.present_mode)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());

        if graphics_index == present_index {
            swapchain_info = swapchain_info.image_sharing_mode(vk::SharingMode::EXCLUSIVE);
        } else {
            swapchain_info = swapchain_info
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(queue_family_indices)
        }

        let swapchain_device = ash::khr::swapchain::Device::new(instance, &device);
        let swapchain_khr = unsafe {
            swapchain_device
                .create_swapchain(&swapchain_info, None)
                .unwrap()
        };

        let swapchain_images = unsafe {
            swapchain_device
                .get_swapchain_images(swapchain_khr)
                .unwrap()
        };

        (
            swapchain_device,
            swapchain_khr,
            swapchain_images,
            swapchain_properties,
        )
    }

    fn create_swapchain_image_views(
        device: &Device,
        swapchain_images: &[vk::Image],
        swapchain_properties: &SwapchainProperties,
    ) -> Vec<vk::ImageView> {
        let mut image_view_info = vk::ImageViewCreateInfo::default()
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(swapchain_properties.format.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        swapchain_images
            .iter()
            .map(|image| {
                image_view_info = image_view_info.image(*image);
                unsafe { device.create_image_view(&image_view_info, None).unwrap() }
            })
            .collect::<Vec<_>>()
    }

    fn create_pipeline(
        device: &Device,
        swapchain_properties: &SwapchainProperties,
    ) -> (vk::Pipeline, vk::PipelineLayout) {
        let shader_code = Self::read_shader_from_file("shaders/shader.spv");
        let shader_module = Self::create_shader_module(device, &shader_code);

        let vertex_shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(shader_module)
            .name(c"vertMain");

        let fragment_shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(shader_module)
            .name(c"fragMain");

        let shader_stages = &[vertex_shader_stage_info, fragment_shader_stage_info];

        let dynamic_states = vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        // We'll get back to it in the vertex buffer chapter.
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport_info = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterization_info = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false)
            .depth_bias_slope_factor(1.0)
            .line_width(1.0);

        // We'll revisit multisampling in later chapter
        let multisampling_info = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false);

        // TODO: Depth and stencil testing

        // Alpha blending
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);
        let color_blend_attachment = &[color_blend_attachment];
        let color_blend_info = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(color_blend_attachment);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .unwrap()
        };

        // Dynamic rendering
        let format = &[swapchain_properties.format.format];
        let mut rendering_info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(format);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .push_next(&mut rendering_info)
            .stages(shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterization_info)
            .multisample_state(&multisampling_info)
            .color_blend_state(&color_blend_info)
            .dynamic_state(&dynamic_state_info)
            .layout(pipeline_layout)
            // render_pass is set to null because we're using dynamic rendering instead of a traditional render pass.
            .render_pass(vk::RenderPass::null());
        let pipeline_infos = &[pipeline_info];
        let pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), pipeline_infos, None)
                .unwrap()[0]
        };

        unsafe {
            device.destroy_shader_module(shader_module, None);
        };

        (pipeline, pipeline_layout)
    }

    fn read_shader_from_file<P: AsRef<std::path::Path>>(path: P) -> Vec<u32> {
        log::debug!("Loading shader file: {}", path.as_ref().to_str().unwrap());
        let mut buffer = Vec::new();
        let mut file = File::open(path).unwrap();
        file.read_to_end(&mut buffer).unwrap();
        let mut cursor = Cursor::new(buffer);
        ash::util::read_spv(&mut cursor).unwrap()
    }

    fn create_shader_module(device: &Device, code: &[u32]) -> vk::ShaderModule {
        let shader_module_info = vk::ShaderModuleCreateInfo::default().code(code);
        unsafe {
            device
                .create_shader_module(&shader_module_info, None)
                .unwrap()
        }
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);

            self.swapchain_image_views
                .iter()
                .for_each(|iv| self.device.destroy_image_view(*iv, None));
            self.swapchain_device
                .destroy_swapchain(self.swapchain_khr, None);

            self.device.destroy_device(None);
            self.surface_instance
                .destroy_surface(self.surface_khr, None);
            if let Some((debug_instance, messenger)) = self.debug_messenger.take() {
                debug_instance.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
