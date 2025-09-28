use ash::vk;

pub struct SwapchainProperties {
    pub format: vk::SurfaceFormatKHR,
    pub present_mode: vk::PresentModeKHR,
    pub extent: vk::Extent2D,
}

pub struct SwapchainSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupportDetails {
    pub fn new(
        physical_device: vk::PhysicalDevice,
        surface_instance: &ash::khr::surface::Instance,
        surface_khr: vk::SurfaceKHR,
    ) -> Self {
        let capabilities = unsafe {
            surface_instance
                .get_physical_device_surface_capabilities(physical_device, surface_khr)
                .unwrap()
        };

        let formats = unsafe {
            surface_instance
                .get_physical_device_surface_formats(physical_device, surface_khr)
                .unwrap()
        };

        let present_modes = unsafe {
            surface_instance
                .get_physical_device_surface_present_modes(physical_device, surface_khr)
                .unwrap()
        };

        Self {
            capabilities,
            formats,
            present_modes,
        }
    }

    pub fn get_ideal_swapchain_properties(
        &self,
        preferred_dimensions: [u32; 2],
    ) -> SwapchainProperties {
        let format = self.choose_swapchain_surface_format();
        let present_mode = Self::choose_swapchain_present_mode(&self);
        let extent = Self::choose_swapchain_extent(&self, preferred_dimensions);

        SwapchainProperties {
            format,
            present_mode,
            extent,
        }
    }

    fn choose_swapchain_surface_format(&self) -> vk::SurfaceFormatKHR {
        let format = self.formats.iter().find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        });

        if let Some(format) = format {
            return *format;
        }

        self.formats[0]
    }

    fn choose_swapchain_present_mode(&self) -> vk::PresentModeKHR {
        if self
            .present_modes
            .iter()
            .any(|p| *p == vk::PresentModeKHR::MAILBOX)
        {
            return vk::PresentModeKHR::MAILBOX;
        }

        if self
            .present_modes
            .iter()
            .any(|p| *p == vk::PresentModeKHR::FIFO)
        {
            return vk::PresentModeKHR::FIFO;
        }

        self.present_modes[0]
    }

    fn choose_swapchain_extent(&self, dimensions: [u32; 2]) -> vk::Extent2D {
        if self.capabilities.current_extent.width != u32::MAX {
            return self.capabilities.current_extent;
        }

        let min = self.capabilities.min_image_extent;
        let max = self.capabilities.max_image_extent;

        let width = dimensions[0].clamp(min.width, max.width);
        let height = dimensions[1].clamp(min.height, max.height);

        vk::Extent2D { width, height }
    }
}
