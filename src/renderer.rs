use std::{
    ffi::{CStr, c_void},
    fs::File,
    io::{Cursor, Read},
    rc::Rc,
};

use ash::{
    Device, Instance,
    ext::debug_utils,
    vk::{self, DebugUtilsMessengerEXT},
};
use cgmath::{Deg, Matrix4, Point3, Vector3, Vector4};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

use crate::{
    app::{HEIGHT, WIDTH},
    debug::{
        ENABLE_VALIDATION_LAYERS, check_validation_layer_support, create_debug_create_info,
        get_layer_names_and_pointers, setup_debug_messenger,
    },
    game_object::{GameObject, Material, Mesh},
    math,
    swapchain_properties::{SwapchainProperties, SwapchainSupportDetails},
    vertex::Vertex,
};

pub struct Renderer {
    _entry: ash::Entry,
    instance: Instance,
    debug_messenger: Option<(debug_utils::Instance, DebugUtilsMessengerEXT)>,
    physical_device: vk::PhysicalDevice,
    msaa_samples: vk::SampleCountFlags,
    device: Rc<Device>,
    graphics_index: u32,
    present_index: u32,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    surface_instance: ash::khr::surface::Instance,
    surface_khr: vk::SurfaceKHR,

    swapchain_device: ash::khr::swapchain::Device,
    swapchain_khr: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_properties: SwapchainProperties,
    swapchain_image_views: Vec<vk::ImageView>,

    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,

    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memories: Vec<vk::DeviceMemory>,
    uniform_buffers_mapped_ptrs: Vec<*mut c_void>,

    color_image: vk::Image,
    color_image_memory: vk::DeviceMemory,
    color_image_view: vk::ImageView,

    depth_image: vk::Image,
    depth_image_memory: vk::DeviceMemory,
    depth_image_view: vk::ImageView,

    descriptor_pool: vk::DescriptorPool,

    sync_objects: Vec<SyncObjects>,
    current_frame: usize,

    pub dirty_swapchain: bool,
}

impl Renderer {
    pub fn new(window: &Window) -> Self {
        let entry = unsafe { ash::Entry::load().unwrap() };

        let instance = Self::create_instance(&entry, window);

        let debug_messenger = setup_debug_messenger(&entry, &instance);

        let (surface_instance, surface_khr) = Self::create_surface(&entry, &instance, window);

        let physical_device = Self::pick_physical_device(&instance, &surface_instance, surface_khr);
        let msaa_samples = Self::get_max_usable_sample_count(&instance, physical_device);

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
        let swapchain_images_len = swapchain_images.len();

        let swapchain_image_views =
            Self::create_swapchain_image_views(&device, &swapchain_images, &swapchain_properties);

        let descriptor_set_layout = Self::create_descriptor_set_layout(&device);

        let depth_format = Self::find_depth_format(&instance, physical_device);
        let (pipeline, pipeline_layout) = Self::create_pipeline(
            &device,
            &swapchain_properties,
            &[descriptor_set_layout],
            depth_format,
            msaa_samples,
        );

        let command_pool = Self::create_command_pool(&device, graphics_index);
        let command_buffers =
            Self::create_command_buffers(&device, command_pool, swapchain_images_len);

        let (uniform_buffers, uniform_buffers_memories, uniform_buffers_mapped_ptrs) =
            Self::create_uniform_buffers(&instance, &device, physical_device, swapchain_images_len);

        let (color_image, color_image_memory, color_image_view) = Self::create_color_resources(
            &device,
            &instance,
            physical_device,
            &swapchain_properties,
            msaa_samples,
        );

        let (depth_image, depth_image_memory, depth_image_view) = Self::create_depth_resources(
            &device,
            &instance,
            physical_device,
            &swapchain_properties,
            msaa_samples,
        );

        let descriptor_pool = Self::create_descriptor_pool(&device, swapchain_images_len);

        let sync_objects = Self::create_sync_objects(&device, swapchain_images_len);

        Self {
            _entry: entry, // When entry is dropped, getting a surface when recreate swapchain throws error
            instance,
            debug_messenger,
            physical_device,
            msaa_samples,
            device: Rc::new(device),
            graphics_index,
            present_index,
            graphics_queue,
            present_queue,
            surface_instance,
            surface_khr,

            swapchain_device,
            swapchain_khr,
            swapchain_images,
            swapchain_properties,
            swapchain_image_views,

            descriptor_set_layout,
            pipeline_layout,
            pipeline,

            command_pool,
            command_buffers,

            uniform_buffers,
            uniform_buffers_memories,
            uniform_buffers_mapped_ptrs,

            color_image,
            color_image_memory,
            color_image_view,

            depth_image,
            depth_image_memory,
            depth_image_view,

            descriptor_pool,

            sync_objects,
            current_frame: 0,

            dirty_swapchain: false,
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

        // If _layer_names is named with underscore, ERROR_LAYER_NOT_PRESENT occurs.
        // This happens because with an underscore, layer_names gets dropped or discarded.
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

        let supported_features = unsafe { instance.get_physical_device_features(physical_device) };
        if supported_features.sampler_anisotropy == 0 {
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

    fn get_max_usable_sample_count(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
    ) -> vk::SampleCountFlags {
        let physical_device_properties =
            unsafe { instance.get_physical_device_properties(physical_device) };

        let counts = physical_device_properties
            .limits
            .framebuffer_color_sample_counts
            & physical_device_properties
                .limits
                .framebuffer_depth_sample_counts;

        if counts.contains(vk::SampleCountFlags::TYPE_64) {
            return vk::SampleCountFlags::TYPE_64;
        }
        if counts.contains(vk::SampleCountFlags::TYPE_32) {
            return vk::SampleCountFlags::TYPE_32;
        }
        if counts.contains(vk::SampleCountFlags::TYPE_16) {
            return vk::SampleCountFlags::TYPE_16;
        }
        if counts.contains(vk::SampleCountFlags::TYPE_8) {
            return vk::SampleCountFlags::TYPE_8;
        }
        if counts.contains(vk::SampleCountFlags::TYPE_4) {
            return vk::SampleCountFlags::TYPE_4;
        }
        if counts.contains(vk::SampleCountFlags::TYPE_2) {
            return vk::SampleCountFlags::TYPE_2;
        }

        vk::SampleCountFlags::TYPE_1
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

        let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        let mut extended_dynamic_state_features =
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT::default()
                .extended_dynamic_state(true);
        let mut shader_draw_parameters_features =
            vk::PhysicalDeviceShaderDrawParametersFeatures::default().shader_draw_parameters(true);
        let mut device_features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut vulkan_13_features)
            .push_next(&mut extended_dynamic_state_features)
            .push_next(&mut shader_draw_parameters_features)
            .features(vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true));

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
        swapchain_images
            .iter()
            .map(|image| {
                Self::create_image_view(
                    device,
                    *image,
                    swapchain_properties.format.format,
                    vk::ImageAspectFlags::COLOR,
                    1,
                )
            })
            .collect::<Vec<_>>()
    }

    fn create_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
        let descriptor_set_layout_bindings = &[
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];

        let descriptor_set_layout_create_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(descriptor_set_layout_bindings);

        unsafe {
            device
                .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
                .unwrap()
        }
    }

    fn create_pipeline(
        device: &Device,
        swapchain_properties: &SwapchainProperties,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        depth_format: vk::Format,
        msaa_samples: vk::SampleCountFlags,
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

        let vertex_binding_descriptions = &[Vertex::get_binding_description()];
        let vertex_attribute_descriptions = &Vertex::get_attribute_descriptions();
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(vertex_binding_descriptions)
            .vertex_attribute_descriptions(vertex_attribute_descriptions);

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
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            .depth_bias_slope_factor(1.0)
            .line_width(1.0);

        // We'll revisit multisampling in later chapter
        let multisampling_info = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(msaa_samples)
            .sample_shading_enable(false);

        let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0)
            .stencil_test_enable(false);

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

        // Push constant ranges
        let push_constant_ranges = &[vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(size_of::<Matrix4<f32>>() as _)];

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(descriptor_set_layouts)
            .push_constant_ranges(push_constant_ranges);
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .unwrap()
        };

        // Dynamic rendering
        let color_formats = &[swapchain_properties.format.format];
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(color_formats)
            .depth_attachment_format(depth_format);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .push_next(&mut rendering_info)
            .stages(shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterization_info)
            .multisample_state(&multisampling_info)
            .depth_stencil_state(&depth_stencil_info)
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

    fn create_command_pool(device: &Device, graphics_index: u32) -> vk::CommandPool {
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(graphics_index);

        unsafe {
            device
                .create_command_pool(&command_pool_info, None)
                .unwrap()
        }
    }

    fn create_command_buffers(
        device: &Device,
        command_pool: vk::CommandPool,
        swapchain_image_len: usize,
    ) -> Vec<vk::CommandBuffer> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(swapchain_image_len as _);

        unsafe {
            device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .unwrap()
        }
    }

    fn create_color_resources(
        device: &Device,
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        swapchain_properties: &SwapchainProperties,
        msaa_samples: vk::SampleCountFlags,
    ) -> (vk::Image, vk::DeviceMemory, vk::ImageView) {
        let color_format = swapchain_properties.format.format;

        let (image, image_memory) = Self::create_image(
            device,
            instance,
            physical_device,
            swapchain_properties.extent.width,
            swapchain_properties.extent.height,
            color_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            1,
            msaa_samples,
        );

        let image_view =
            Self::create_image_view(device, image, color_format, vk::ImageAspectFlags::COLOR, 1);

        (image, image_memory, image_view)
    }

    fn create_depth_resources(
        device: &Device,
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        swapchain_properties: &SwapchainProperties,
        msaa_samples: vk::SampleCountFlags,
    ) -> (vk::Image, vk::DeviceMemory, vk::ImageView) {
        let depth_format = Self::find_depth_format(instance, physical_device);

        let (image, image_memory) = Self::create_image(
            device,
            instance,
            physical_device,
            swapchain_properties.extent.width,
            swapchain_properties.extent.height,
            depth_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            1,
            msaa_samples,
        );

        let image_view =
            Self::create_image_view(device, image, depth_format, vk::ImageAspectFlags::DEPTH, 1);

        (image, image_memory, image_view)
    }

    fn find_depth_format(instance: &Instance, physical_device: vk::PhysicalDevice) -> vk::Format {
        Self::find_supported_format(
            instance,
            physical_device,
            &[
                vk::Format::D32_SFLOAT,
                vk::Format::D32_SFLOAT_S8_UINT,
                vk::Format::D24_UNORM_S8_UINT,
            ],
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        )
    }

    #[allow(dead_code)]
    fn has_stencil_component(format: vk::Format) -> bool {
        format == vk::Format::D32_SFLOAT_S8_UINT || format == vk::Format::D24_UNORM_S8_UINT
    }

    fn find_supported_format(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        candidates: &[vk::Format],
        tiling: vk::ImageTiling,
        format_features: vk::FormatFeatureFlags,
    ) -> vk::Format {
        for &format in candidates {
            let format_properties =
                unsafe { instance.get_physical_device_format_properties(physical_device, format) };
            log::info!("format: {:?}", format);
            log::info!("format_properties: {:?}", format_properties);

            if tiling == vk::ImageTiling::LINEAR
                && format_properties
                    .linear_tiling_features
                    .contains(format_features)
            {
                return format;
            }

            if tiling == vk::ImageTiling::OPTIMAL
                && format_properties
                    .optimal_tiling_features
                    .contains(format_features)
            {
                log::info!("Selected format: {:?}", format);
                return format;
            }
        }

        panic!("failed to find supported format!")
    }

    fn create_texture_image(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        texture_path: &str,
    ) -> (vk::Image, vk::DeviceMemory, u32) {
        let image = image::open(texture_path).unwrap();
        let image_as_rgb = image.to_rgba8();
        let (width, height) = (image_as_rgb.width(), image_as_rgb.height());
        let pixels = image_as_rgb.into_raw();
        let image_size = (pixels.len() * size_of::<u8>()) as vk::DeviceSize;
        let mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            instance,
            device,
            physical_device,
            image_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        );

        unsafe {
            let data_ptr = device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    image_size,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap();
            let mut align = ash::util::Align::new(data_ptr, align_of::<u8>() as _, image_size);
            align.copy_from_slice(&pixels);
            device.unmap_memory(staging_buffer_memory);
        }

        let (texture_image, texture_image_memory) = Self::create_image(
            device,
            instance,
            physical_device,
            width,
            height,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            mip_levels,
            vk::SampleCountFlags::TYPE_1,
        );

        Self::transition_image_layout_single_time(
            device,
            command_pool,
            queue,
            texture_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            mip_levels,
        );
        Self::copy_buffer_to_image(
            device,
            command_pool,
            queue,
            staging_buffer,
            texture_image,
            width,
            height,
        );
        Self::generate_mipmaps(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            texture_image,
            vk::Format::R8G8B8A8_SRGB,
            width as _,
            height as _,
            mip_levels,
        );

        unsafe {
            device.free_memory(staging_buffer_memory, None);
            device.destroy_buffer(staging_buffer, None);
        }

        (texture_image, texture_image_memory, mip_levels)
    }

    fn generate_mipmaps(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        image: vk::Image,
        image_format: vk::Format,
        width: i32,
        height: i32,
        mip_levels: u32,
    ) {
        let format_properties = unsafe {
            instance.get_physical_device_format_properties(physical_device, image_format)
        };
        if !format_properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
        {
            panic!("texture image format does not support linear blitting!");
        }

        Self::execute_single_time_commands(device, command_pool, queue, |command_buffer| {
            let mut image_memory_barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let (mut mip_width, mut mip_height) = (width, height);

            for i in 1..mip_levels {
                image_memory_barrier.subresource_range.base_mip_level = i - 1;
                image_memory_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                image_memory_barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                image_memory_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
                image_memory_barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

                unsafe {
                    device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[image_memory_barrier],
                    )
                };

                let src_offsets = [
                    vk::Offset3D::default(),
                    vk::Offset3D {
                        x: mip_width,
                        y: mip_height,
                        z: 1,
                    },
                ];
                let dst_offsets = [
                    vk::Offset3D::default(),
                    vk::Offset3D {
                        x: if mip_width > 1 { mip_width / 2 } else { 1 },
                        y: if mip_height > 1 { mip_height / 2 } else { 1 },
                        z: 1,
                    },
                ];

                let image_blit = vk::ImageBlit::default()
                    .src_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(i - 1)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
                    .dst_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(i)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
                    .src_offsets(src_offsets)
                    .dst_offsets(dst_offsets);

                unsafe {
                    device.cmd_blit_image(
                        command_buffer,
                        image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[image_blit],
                        vk::Filter::LINEAR,
                    )
                };

                image_memory_barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                image_memory_barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                image_memory_barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
                image_memory_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

                unsafe {
                    device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[image_memory_barrier],
                    )
                };

                if mip_width > 1 {
                    mip_width /= 2;
                }
                if mip_height > 1 {
                    mip_height /= 2;
                }
            }

            image_memory_barrier.subresource_range.base_mip_level = mip_levels - 1;
            image_memory_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            image_memory_barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            image_memory_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            image_memory_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            unsafe {
                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[image_memory_barrier],
                )
            };
        });
    }

    fn create_texture_image_view(
        device: &Device,
        image: vk::Image,
        mip_levels: u32,
    ) -> vk::ImageView {
        Self::create_image_view(
            device,
            image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageAspectFlags::COLOR,
            mip_levels,
        )
    }

    fn create_texture_sampler(
        device: &Device,
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
    ) -> vk::Sampler {
        let physical_device_properties =
            unsafe { instance.get_physical_device_properties(physical_device) };
        let sampler_create_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            // Mipmapping
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(0.0)
            .anisotropy_enable(true)
            .max_anisotropy(physical_device_properties.limits.max_sampler_anisotropy)
            // If enabled, texels will first be compared to a value, and the result of
            // that comparison is used n filtering operations.
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false);

        unsafe { device.create_sampler(&sampler_create_info, None).unwrap() }
    }

    fn create_image(
        device: &Device,
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        width: u32,
        height: u32,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        memory_properties: vk::MemoryPropertyFlags,
        mip_levels: u32,
        num_samples: vk::SampleCountFlags,
    ) -> (vk::Image, vk::DeviceMemory) {
        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .samples(num_samples)
            .tiling(tiling)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = unsafe { device.create_image(&image_create_info, None).unwrap() };

        let memory_requirements = unsafe { device.get_image_memory_requirements(image) };

        let physical_device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let memory_type_index = Self::find_memory_type_index(
            memory_requirements,
            physical_device_memory_properties,
            memory_properties,
        );
        let memory_allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(memory_type_index);

        let memory = unsafe { device.allocate_memory(&memory_allocate_info, None).unwrap() };
        unsafe { device.bind_image_memory(image, memory, 0).unwrap() }

        (image, memory)
    }

    fn create_image_view(
        device: &Device,
        image: vk::Image,
        format: vk::Format,
        aspect_mask: vk::ImageAspectFlags,
        mip_levels: u32,
    ) -> vk::ImageView {
        let image_view_create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            });

        unsafe {
            device
                .create_image_view(&image_view_create_info, None)
                .unwrap()
        }
    }

    fn create_vertex_buffer(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        graphics_queue: vk::Queue,
        vertices: &[Vertex],
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_size = (size_of::<Vertex>() * vertices.len()) as vk::DeviceSize;
        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            instance,
            device,
            physical_device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data_ptr = device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    buffer_size,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap();
            let mut align =
                ash::util::Align::new(data_ptr, align_of_val(&vertices[0]) as _, buffer_size);
            align.copy_from_slice(&vertices);
            device.unmap_memory(staging_buffer_memory);
        }

        let (vertex_buffer, vertex_buffer_memory) = Self::create_buffer(
            instance,
            device,
            physical_device,
            buffer_size,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        Self::copy_buffer(
            device,
            command_pool,
            graphics_queue,
            staging_buffer,
            vertex_buffer,
            buffer_size,
        );

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }

        (vertex_buffer, vertex_buffer_memory)
    }

    fn create_index_buffer(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        graphics_queue: vk::Queue,
        indices: &[u32],
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_size = (size_of_val(&indices[0]) * indices.len()) as vk::DeviceSize;
        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            instance,
            device,
            physical_device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data_ptr = device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    buffer_size,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap();
            let mut align =
                ash::util::Align::new(data_ptr, align_of_val(&indices[0]) as _, buffer_size);
            align.copy_from_slice(&indices);
            device.unmap_memory(staging_buffer_memory);
        }

        let (index_buffer, index_buffer_memory) = Self::create_buffer(
            instance,
            device,
            physical_device,
            buffer_size,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        Self::copy_buffer(
            device,
            command_pool,
            graphics_queue,
            staging_buffer,
            index_buffer,
            buffer_size,
        );

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }

        (index_buffer, index_buffer_memory)
    }

    fn create_uniform_buffers(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        count: usize,
    ) -> (Vec<vk::Buffer>, Vec<vk::DeviceMemory>, Vec<*mut c_void>) {
        let buffer_size = size_of::<UniformBufferObject>() as u64;

        let mut uniform_buffers = Vec::with_capacity(count);
        let mut uniform_buffers_memories = Vec::with_capacity(count);
        let mut uniform_buffers_mapped_ptrs = Vec::with_capacity(count);

        for _ in 0..count {
            // We're going to copy new data to the uniform buffer every frame, so it doesn't
            // really make any sense to have a staging buffer. It would just add extra overhead
            // in this case and likely degrade performance instead of improving it.

            let (buffer, memory) = Self::create_buffer(
                instance,
                device,
                physical_device,
                buffer_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
            );

            // persistent mapping
            let mapped_ptr = unsafe {
                device
                    .map_memory(memory, 0, buffer_size, vk::MemoryMapFlags::empty())
                    .unwrap()
            };

            uniform_buffers.push(buffer);
            uniform_buffers_memories.push(memory);
            uniform_buffers_mapped_ptrs.push(mapped_ptr);
        }

        (
            uniform_buffers,
            uniform_buffers_memories,
            uniform_buffers_mapped_ptrs,
        )
    }

    fn create_descriptor_pool(device: &Device, count: usize) -> vk::DescriptorPool {
        let descriptor_pool_sizes = &[
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(count as _),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(count as _),
        ];

        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(count as _)
            .pool_sizes(descriptor_pool_sizes);

        unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .unwrap()
        }
    }

    fn create_descriptor_sets(
        device: &Device,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layout: vk::DescriptorSetLayout,
        uniform_buffers: &[vk::Buffer],
        texture_image_view: vk::ImageView,
        texture_sampler: vk::Sampler,
    ) -> Vec<vk::DescriptorSet> {
        let layouts = (0..uniform_buffers.len())
            .map(|_| descriptor_set_layout)
            .collect::<Vec<_>>();
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);

        let descriptor_sets = unsafe {
            device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
                .unwrap()
        };

        for (&descriptor_set, &uniform_buffer) in descriptor_sets.iter().zip(uniform_buffers) {
            let descriptor_buffer_info = &[vk::DescriptorBufferInfo::default()
                .buffer(uniform_buffer)
                .offset(0)
                .range(size_of::<UniformBufferObject>() as _)];

            let descriptor_image_info = &[vk::DescriptorImageInfo::default()
                .sampler(texture_sampler)
                .image_view(texture_image_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];

            let write_descriptor_set = &[
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(descriptor_buffer_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(descriptor_image_info),
            ];

            let descriptor_copies = &[];
            unsafe { device.update_descriptor_sets(write_descriptor_set, descriptor_copies) };
        }

        descriptor_sets
    }

    fn create_buffer(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_property_flags: vk::MemoryPropertyFlags,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let vertex_buffer = unsafe { device.create_buffer(&buffer_info, None).unwrap() };

        let memory_requirements = unsafe { device.get_buffer_memory_requirements(vertex_buffer) };

        let physical_device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let memory_type_index = Self::find_memory_type_index(
            memory_requirements,
            physical_device_memory_properties,
            memory_property_flags,
        );

        let memory_allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(memory_type_index);

        let memory = unsafe { device.allocate_memory(&memory_allocate_info, None).unwrap() };

        unsafe { device.bind_buffer_memory(vertex_buffer, memory, 0).unwrap() };

        (vertex_buffer, memory)
    }

    fn copy_buffer(
        device: &Device,
        command_pool: vk::CommandPool,
        graphics_queue: vk::Queue,
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        size: vk::DeviceSize,
    ) {
        Self::execute_single_time_commands(
            device,
            command_pool,
            graphics_queue,
            |command_buffer| {
                let regions = &[vk::BufferCopy::default().size(size)];
                unsafe { device.cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, regions) };
            },
        );
    }

    fn find_memory_type_index(
        memory_requirements: vk::MemoryRequirements,
        physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
        memory_property_flags: vk::MemoryPropertyFlags,
    ) -> u32 {
        for i in 0..physical_device_memory_properties.memory_type_count {
            if memory_requirements.memory_type_bits & (1 << i) == 0 {
                continue;
            }

            let memory_type = physical_device_memory_properties.memory_types[i as usize];

            if memory_type.property_flags.contains(memory_property_flags) {
                return i;
            }
        }

        panic!("Failed to find suitable memory type!");
    }

    fn record_command_buffer(
        device: &Device,
        command_buffer: vk::CommandBuffer,
        swapchain_images: &[vk::Image],
        swapchain_image_views: &[vk::ImageView],
        swapchain_properties: &SwapchainProperties,
        image_index: usize,
        pipeline: vk::Pipeline,
        pipeline_layout: vk::PipelineLayout,
        current_frame: usize,
        color_image: vk::Image,
        color_image_view: vk::ImageView,
        depth_image: vk::Image,
        depth_image_view: vk::ImageView,
        game_objects: &[GameObject],
    ) {
        unsafe {
            device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
                .unwrap()
        }

        Self::transition_image_layout(
            device,
            command_buffer,
            swapchain_images[image_index],
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::AccessFlags2::empty(),
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::TOP_OF_PIPE,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::ImageAspectFlags::COLOR,
        );

        // Transition the multisampled color image to COLOR_ATTACHMENT_OPTIMAL
        Self::transition_image_layout(
            device,
            command_buffer,
            color_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::AccessFlags2::empty(),
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::TOP_OF_PIPE,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::ImageAspectFlags::COLOR,
        );

        // Transition the depth image to DEPTH_ATTACHMENT_OPTIMAL
        Self::transition_image_layout(
            device,
            command_buffer,
            depth_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            vk::AccessFlags2::empty(),
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::TOP_OF_PIPE,
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
            vk::ImageAspectFlags::DEPTH,
        );

        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };
        let clear_depth = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };

        let color_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(color_image_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            // multisampled with resolve attachment
            .resolve_mode(vk::ResolveModeFlags::AVERAGE)
            .resolve_image_view(swapchain_image_views[image_index])
            .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear_color);
        let depth_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(depth_image_view)
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(clear_depth);

        let color_attachment_infos = &[color_attachment_info];
        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: swapchain_properties.extent,
            })
            .layer_count(1)
            .color_attachments(color_attachment_infos)
            .depth_attachment(&depth_attachment_info);

        unsafe { device.cmd_begin_rendering(command_buffer, &rendering_info) };

        unsafe {
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline)
        };

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: swapchain_properties.extent.width as _,
            height: swapchain_properties.extent.height as _,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: swapchain_properties.extent,
        };
        unsafe { device.cmd_set_viewport(command_buffer, 0, &[viewport]) };
        unsafe { device.cmd_set_scissor(command_buffer, 0, &[scissor]) };

        for game_object in game_objects {
            unsafe {
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_layout,
                    0,
                    &[game_object.material.descriptor_sets[current_frame]],
                    &[],
                )
            };

            unsafe {
                let transform_bytes: &[u8] = {
                    std::slice::from_raw_parts(
                        &game_object.transform as *const Matrix4<f32> as *const u8,
                        size_of::<Matrix4<f32>>(),
                    )
                };
                device.cmd_push_constants(
                    command_buffer,
                    pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    transform_bytes,
                );
            }

            let vertex_buffers = &[game_object.mesh.vertex_buffer];
            let offsets = &[0];

            unsafe {
                device.cmd_bind_vertex_buffers(command_buffer, 0, vertex_buffers, offsets);
                device.cmd_bind_index_buffer(
                    command_buffer,
                    game_object.mesh.index_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
            }

            unsafe {
                device.cmd_draw_indexed(command_buffer, game_object.mesh.index_count, 1, 0, 0, 0)
            };
        }

        unsafe { device.cmd_end_rendering(command_buffer) };

        Self::transition_image_layout(
            device,
            command_buffer,
            swapchain_images[image_index],
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags2::default(),
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            vk::ImageAspectFlags::COLOR,
        );

        unsafe { device.end_command_buffer(command_buffer).unwrap() }
    }

    fn transition_image_layout_single_time(
        device: &Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        mip_levels: u32,
    ) {
        Self::execute_single_time_commands(device, command_pool, queue, |command_buffer| {
            let (src_access_mask, dst_access_mask, src_stage_mask, dst_stage_mask) =
                match (old_layout, new_layout) {
                    (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                    ),
                    (
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    ) => (
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::SHADER_READ,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                    ),
                    _ => panic!("unsupported layout transition!"),
                };

            let image_memory_barrier = vk::ImageMemoryBarrier::default()
                .old_layout(old_layout)
                .new_layout(new_layout)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: mip_levels,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_access_mask(src_access_mask)
                .dst_access_mask(dst_access_mask);

            unsafe {
                device.cmd_pipeline_barrier(
                    command_buffer,
                    src_stage_mask,
                    dst_stage_mask,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[image_memory_barrier],
                )
            };
        });
    }

    fn transition_image_layout(
        device: &Device,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_access_mask: vk::AccessFlags2,
        dst_access_mask: vk::AccessFlags2,
        src_stage_mask: vk::PipelineStageFlags2,
        dst_stage_mask: vk::PipelineStageFlags2,
        aspect_mask: vk::ImageAspectFlags,
    ) {
        let image_memory_barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask)
            .src_stage_mask(src_stage_mask)
            .dst_stage_mask(dst_stage_mask)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        let image_memory_barriers = &[image_memory_barrier];
        let dependency_info = vk::DependencyInfo::default()
            .dependency_flags(vk::DependencyFlags::empty())
            .image_memory_barriers(image_memory_barriers);

        unsafe { device.cmd_pipeline_barrier2(command_buffer, &dependency_info) };
    }

    fn copy_buffer_to_image(
        device: &Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        buffer: vk::Buffer,
        image: vk::Image,
        width: u32,
        height: u32,
    ) {
        Self::execute_single_time_commands(device, command_pool, queue, |command_buffer| {
            let buffer_image_copy = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                });

            unsafe {
                device.cmd_copy_buffer_to_image(
                    command_buffer,
                    buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[buffer_image_copy],
                )
            };
        });
    }

    fn execute_single_time_commands<F: FnOnce(vk::CommandBuffer)>(
        device: &Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        executor: F,
    ) {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = unsafe {
            device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .unwrap()
        }[0];

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .unwrap()
        }

        executor(command_buffer);

        unsafe { device.end_command_buffer(command_buffer).unwrap() }

        let command_buffers = &[command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(command_buffers);

        unsafe {
            device
                .queue_submit(queue, &[submit_info], vk::Fence::null())
                .unwrap();

            device.queue_wait_idle(queue).unwrap();
        }

        unsafe { device.free_command_buffers(command_pool, command_buffers) }
    }

    fn create_sync_objects(device: &Device, swapchain_image_len: usize) -> Vec<SyncObjects> {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let mut sync_objects_vec = Vec::with_capacity(swapchain_image_len as _);
        for _ in 0..swapchain_image_len {
            let present_complete_semaphore =
                unsafe { device.create_semaphore(&semaphore_info, None).unwrap() };
            let render_finished_semaphore =
                unsafe { device.create_semaphore(&semaphore_info, None).unwrap() };

            let in_flight_fence = unsafe { device.create_fence(&fence_info, None).unwrap() };

            let sync_objects = SyncObjects {
                present_complete_semaphore,
                render_finished_semaphore,
                in_flight_fence,
            };
            sync_objects_vec.push(sync_objects);
        }

        sync_objects_vec
    }

    pub fn draw_frame(&mut self, game_objects: &[GameObject]) -> bool /* dirty_swapchain */ {
        let SyncObjects {
            present_complete_semaphore,
            render_finished_semaphore,
            in_flight_fence,
        } = self.sync_objects[self.current_frame];
        let command_buffer = self.command_buffers[self.current_frame];
        unsafe {
            self.device
                .wait_for_fences(&[in_flight_fence], true, u64::MAX)
                .unwrap()
        }

        let result = unsafe {
            self.swapchain_device.acquire_next_image(
                self.swapchain_khr,
                u64::MAX,
                present_complete_semaphore,
                vk::Fence::null(),
            )
        };
        let image_index = match result {
            Ok((image_index, _)) => image_index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                log::debug!("Swapchain is dirty from acquiring next image.");
                return true;
            }
            Err(error) => panic!("Error while acquiring next image. Cause: {}", error),
        };

        unsafe { self.device.reset_fences(&[in_flight_fence]).unwrap() }

        Self::record_command_buffer(
            &self.device,
            command_buffer,
            &self.swapchain_images,
            &self.swapchain_image_views,
            &self.swapchain_properties,
            image_index as _,
            self.pipeline,
            self.pipeline_layout,
            self.current_frame,
            self.color_image,
            self.color_image_view,
            self.depth_image,
            self.depth_image_view,
            game_objects,
        );

        self.update_uniform_buffer(self.current_frame);

        // Submit
        {
            let wait_semaphores = &[present_complete_semaphore];
            let command_buffers = &[command_buffer];
            let signal_semaphores = &[render_finished_semaphore];
            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(wait_semaphores)
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                .command_buffers(command_buffers)
                .signal_semaphores(signal_semaphores);

            unsafe {
                self.device
                    .queue_submit(self.graphics_queue, &[submit_info], in_flight_fence)
                    .unwrap()
            }
        }

        unsafe {
            self.device
                .wait_for_fences(&[in_flight_fence], true, u64::MAX)
                .unwrap()
        }

        // Present
        {
            let wait_semaphores = &[render_finished_semaphore];
            let swapchains = &[self.swapchain_khr];
            let image_indices = &[image_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(wait_semaphores)
                .swapchains(swapchains)
                .image_indices(image_indices);

            let result = unsafe {
                self.swapchain_device
                    .queue_present(self.present_queue, &present_info)
            };
            match result {
                Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    log::debug!("Swapchain is dirty from presenting.");
                    return true;
                }
                Err(error) => panic!("Failed to present queue. Cause: {}", error),
                _ => {}
            }
        }

        self.current_frame = (self.current_frame + 1) % self.swapchain_images.len();

        false
    }

    fn update_uniform_buffer(&mut self, current_image: usize) {
        let aspect = self.swapchain_properties.aspect();

        let view = Matrix4::look_at_rh(
            Point3::new(2.0, 2.0, 2.0),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        );
        let proj = math::perspective(Deg(45.0), aspect, 0.1, 10.0);

        let ubos = [UniformBufferObject {
            view,
            proj,
            light_position: Vector4::new(2.0, 2.0, 2.0, 1.0),
            light_color: Vector4::new(0.0, 1.0, 0.0, 0.0),
            view_position: Vector4::new(2.0, 2.0, 2.0, 1.0),
        }];

        let mapped_ptr = self.uniform_buffers_mapped_ptrs[current_image];
        let mut align = unsafe {
            ash::util::Align::new(
                mapped_ptr,
                align_of::<UniformBufferObject>() as _,
                size_of::<UniformBufferObject>() as _,
            )
        };
        align.copy_from_slice(&ubos);
    }

    pub fn recreate_swapchain(&mut self) {
        log::debug!("recreate swapchain");
        unsafe { self.device.device_wait_idle().unwrap() }

        self.cleanup_swapchain();

        let (swapchain_device, swapchain_khr, swapchain_images, swapchain_properties) =
            Self::create_swapchain(
                &self.instance,
                &self.device,
                self.physical_device,
                &self.surface_instance,
                self.surface_khr,
                self.graphics_index,
                self.present_index,
            );

        let swapchain_image_views = Self::create_swapchain_image_views(
            &self.device,
            &swapchain_images,
            &swapchain_properties,
        );

        let (color_image, color_image_memory, color_image_view) = Self::create_color_resources(
            &self.device,
            &self.instance,
            self.physical_device,
            &swapchain_properties,
            self.msaa_samples,
        );

        let (depth_image, depth_image_memory, depth_image_view) = Self::create_depth_resources(
            &self.device,
            &self.instance,
            self.physical_device,
            &swapchain_properties,
            self.msaa_samples,
        );

        self.swapchain_device = swapchain_device;
        self.swapchain_khr = swapchain_khr;
        self.swapchain_images = swapchain_images;
        self.swapchain_properties = swapchain_properties;
        self.swapchain_image_views = swapchain_image_views;

        self.color_image = color_image;
        self.color_image_memory = color_image_memory;
        self.color_image_view = color_image_view;

        self.depth_image = depth_image;
        self.depth_image_memory = depth_image_memory;
        self.depth_image_view = depth_image_view;
    }

    fn cleanup_swapchain(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.depth_image_view, None);
            self.device.free_memory(self.depth_image_memory, None);
            self.device.destroy_image(self.depth_image, None);

            self.device.destroy_image_view(self.color_image_view, None);
            self.device.free_memory(self.color_image_memory, None);
            self.device.destroy_image(self.color_image, None);

            self.swapchain_image_views
                .iter()
                .for_each(|iv| self.device.destroy_image_view(*iv, None));
            self.swapchain_device
                .destroy_swapchain(self.swapchain_khr, None);
        }
    }

    pub fn wait_device_idle(&self) {
        unsafe { self.device.device_wait_idle().unwrap() }
    }

    pub fn handle_window_resized(&mut self, new_width: u32, new_height: u32) {
        let is_dimensions_changed = new_width != self.swapchain_properties.extent.width
            || new_height != self.swapchain_properties.extent.height;

        if is_dimensions_changed {
            self.dirty_swapchain = true;
        }
    }

    pub fn create_mesh(&self, vertices: &[Vertex], indices: &[u32]) -> Mesh {
        let (vertex_buffer, vertex_buffer_memory) = Self::create_vertex_buffer(
            &self.instance,
            &self.device,
            self.physical_device,
            self.command_pool,
            self.graphics_queue,
            vertices,
        );

        let (index_buffer, index_buffer_memory) = Self::create_index_buffer(
            &self.instance,
            &self.device,
            self.physical_device,
            self.command_pool,
            self.graphics_queue,
            indices,
        );

        Mesh {
            device: self.device.clone(),
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
            index_count: indices.len() as _,
        }
    }

    pub fn create_material(&self, texture_path: &str) -> Material {
        let (texture_image, texture_image_memory, mip_levels) = Self::create_texture_image(
            &self.instance,
            &self.device,
            self.physical_device,
            self.command_pool,
            self.graphics_queue,
            texture_path,
        );

        let texture_image_view =
            Self::create_texture_image_view(&self.device, texture_image, mip_levels);

        let texture_sampler =
            Self::create_texture_sampler(&self.device, &self.instance, self.physical_device);

        let descriptor_sets = Self::create_descriptor_sets(
            &self.device,
            self.descriptor_pool,
            self.descriptor_set_layout,
            &self.uniform_buffers,
            texture_image_view,
            texture_sampler,
        );

        Material {
            device: self.device.clone(),
            texture_image,
            texture_image_memory,
            texture_image_view,
            texture_sampler,
            descriptor_sets,
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.sync_objects
                .iter()
                .for_each(|o| o.destroy(&self.device));

            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            self.uniform_buffers_memories.iter().for_each(|m| {
                self.device.unmap_memory(*m);
                self.device.free_memory(*m, None);
            });
            self.uniform_buffers
                .iter()
                .for_each(|b| self.device.destroy_buffer(*b, None));

            self.device
                .free_command_buffers(self.command_pool, &self.command_buffers);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            self.cleanup_swapchain();

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

#[derive(Clone, Copy)]
struct SyncObjects {
    present_complete_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    in_flight_fence: vk::Fence,
}

impl SyncObjects {
    fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_fence(self.in_flight_fence, None);
            device.destroy_semaphore(self.render_finished_semaphore, None);
            device.destroy_semaphore(self.present_complete_semaphore, None);
        }
    }
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
struct UniformBufferObject {
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
    light_position: Vector4<f32>,
    light_color: Vector4<f32>,
    view_position: Vector4<f32>,
}
