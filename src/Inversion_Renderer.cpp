#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <format>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <optional>
#include <set>
#include <cstdint> 
#include <limits> 
#include <algorithm> 
#include <fstream>

// Struct for checking our required queue types as we add them
struct QueueFamilyIndices {
    // Each of these indices is wrapped by std::optional, which just lets us check if it was ever assigned or not. Not very performant but this is very rarely used outside of initialization.
    std::optional<uint32_t> graphicsFamily; // Will contain the index for the garphics queue family
    std::optional<uint32_t> presentFamily; // Will contain the index for the presentation queue family

    // Checks that the queue family supports ALL the required functionality
    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

// Describes what our swapchain(s) must support
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities; // Basic info such as min/max number of images in the chain, min/max width and height of the images, etc.
    std::vector<VkSurfaceFormatKHR> formats; // pixel and color formats to support
    std::vector<VkPresentModeKHR> presentModes; // Available presentation modes
};

class HelloTriangleApplication {
public:
    // Executes all of our code
    void run() {
        initWindow();
        std::cout << "Window Initialized!\n";
        initVulkan();
        std::cout << "Main Loop Started!\n";
        mainLoop();
        cleanup();
        std::cout << "Main Loop Cleaned Up!\n";
    }

private:
    // Window Dimensions
    const uint32_t WIDTH = 800;
    const uint32_t HEIGHT = 600;

    // Create a vector of required validation layers
    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation" // This layer has all the 'useful' validation bundled into it. It covers all our basic bases, but more can be added
    };

    // Create a vector of required device extensions
    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME  // Indicates swapchain support, which is a container that is passed between queues and our application containing the images to draw
    };

    // Enable validation layers
    const bool enableValidationLayers = true;

    // Actual window. Use a pointer.
    GLFWwindow* window;

    // Vulkan instance
    VkInstance instance;

    // Handle to the device, which we instantiate as null for now
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

    // Handle to the logical device
    VkDevice logicalDevice;

    // Hanlde to the graphics queue, which is populated with vkGetDeviceQueue(). We can always use index 0 since there is only one queue, but that will obviously change eventually.
    VkQueue graphicsQueue;

    // Handle to the render surface
    VkSurfaceKHR surface;

    // Handle to the presentation queue, which is where we put the stuff we're going to present to the window
    VkQueue presentQueue;

    // Hanlde for the swap chain
    VkSwapchainKHR swapChain;

    // The swapchain image format
    VkFormat swapChainImageFormat;

    // The swapchain extent
    VkExtent2D swapChainExtent;

    // Stores the handles to swapChain images so we can access them
    std::vector<VkImage> swapChainImages;

    // Stores the imageViews associated with the images on the swapchain
    std::vector<VkImageView> swapChainImageViews;

    // Frame buffers for submission to the swap chain
    std::vector<VkFramebuffer> swapChainFramebuffers;

    // Render pass that describes the frame buffer attachment used during rendering. This is tied directly to the shader code
    VkRenderPass renderPass;

    // Stores the layout of our render pipeline
    VkPipelineLayout pipelineLayout;

    // The final graphics Pipeline for rendering
    VkPipeline graphicsPipeline;

    // Memory pool for the command buffers
    VkCommandPool commandPool;

    // Command buffer. We're just going to record over this every frame, but ideally we'd probably have more than one of these
    VkCommandBuffer commandBuffer;

    // Fence used to wait on previous frame to complete rendering
    VkFence inFlightFence;

    // Sempaphore indicating an image is available
    VkSemaphore imageAvailableSemaphore; 

    // Semaphore indicating the renderer has finished rendering a frame
    VkSemaphore renderFinishedSemaphore;

    // Initialized the GLFW window
    void initWindow() {
        // Initialize glfw library
        glfwInit();

        // Tell GLFW not to use OpenGL context
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        // Disable window resizing for now, since I'm a beta who can't support it
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        // Actually create the window
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Test", nullptr, nullptr);
    }

    // Initialze Vulkan
    void initVulkan() {
        createInstance();
        std::cout << "Vulkan instance created!\n";
        createSurface();
        std::cout << "Vulkan surface created!\n";
        pickPhysicalDevice();
        std::cout << "Vulkan has chosen a physical device!\n";
        createLogicalDevice();
        std::cout << "Vulkan has associated a logical handle with the device!\n";
        createSwapChain();
        std::cout << "Vulkan has established a swap chain!\n";
        createImageViews();
        std::cout << "Vulkan has instantiated the images and views for the swap chain!\n";
        createRenderPass();
        std::cout << "Vulkan has built a render pass definition!\n";
        createGraphicsPipeline();
        std::cout << "Vulkan has built the graphics pipeline!\n";
        createFrameBuffers();
        std::cout << "Vulkan has create the frame buffers!\n";
        createCommandPool();
        std::cout << "Vulkan has create the command pool!\n";
        createCommandBuffer();
        std::cout << "Vulkan has created a command buffer!\n";
        createSyncObjects();
        std::cout << "Vulkan has created the synchronization objects!\n";
    }

    // Create a Vulkan instance
    void createInstance() {
        // Before we do anything, make sure our vaildation layers are available if we're using them
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("Not all validation layers are available!");
        }
        else if (enableValidationLayers) {
            std::cout << "All validation layers enabled\n";
        }

        // Create the struct containing the application creation info
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;             // Always have to specify the structure type in sType
        appInfo.pApplicationName = "Vulkan Application";                // Applicatoin Name. This probably doesn't matter much, but let's be descriptive.
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);          // Probably want to use a newer version, but we'll stick with the basics for now.
        appInfo.pEngineName = "No Engine";                              // Apparently not using an engine. Not sure what this is about yet.
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);               // Specifying the engine version, which is kind of irrelevant all things considered.
        appInfo.apiVersion = VK_API_VERSION_1_0;                        // API version, which I assume is super relevant. Using 1.0 for now.

        // We need to tell Vulkan what extensions to use, since it's cross platform. GLFW give us this
        // Make storage variables
        uint32_t glfwExtensionCount = 0;
        uint32_t extensionCount = 0;
        const char** glfwExtensions;

        // Actually check for extensions instead of just leaving it blank
        glfwGetRequiredInstanceExtensions(&glfwExtensionCount);                                 // This gets the GLFW extension count

        // Store the extensions. The function sets glfwExtensionCount to whatever number for us.
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);                // GLFW extensions
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);              // First arg is the validation layer. Second is the extension count and last is the actual extension data, which is null here since we need the extensionCount

        // Now that we have the extension count, we can instantiate the vector
        std::vector<VkExtensionProperties> extensions(extensionCount);

        // Now run enumerate again with this vector as the destination
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

        // Search the list of available Vulkan extensions to make sure GLFW is fully supported
        std::cout << "Required Extensions:\n";
        for (int i = 0; i < glfwExtensionCount; ++i) {
            std::cout << glfwExtensions[i] << ":\t";

            for (const auto& extension : extensions) {
                if (strcmp(glfwExtensions[i], extension.extensionName)) {
                    std::cout << "AVAILABLE\n";
                    break;
                }
                else {
                    throw std::runtime_error(std::format("\nMissing GLFW extension: {}", glfwExtensions[i]));
                }
            }
        }
        // Print success statement. I'll probably use validation layers for this later though.
        std::cout << "All Extensions Supported!\n";

        // Now we create the instance data struct
        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;      // Always have to specify struct type
        createInfo.pApplicationInfo = &appInfo;                         // Pass the appInfo we just made into the instance creation struct
        createInfo.enabledExtensionCount = glfwExtensionCount;          // Set the number of enabled extensions
        createInfo.ppEnabledExtensionNames = glfwExtensions;            // Set the list of enabled extensions
        createInfo.enabledLayerCount = 0;                               // For enabled validation layers

        // If we're using validation layers, modify the struct to include those
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }

        // Create the instance using these structs. Throw error if anything went wrong.
        if (VkResult result = vkCreateInstance(&createInfo, nullptr, &instance)) { // Feed the create function the struct and the target instance. Note the nullptr; this is for a custom memory allocator, which I will probably never use.
            throw std::runtime_error("Failed to create instance!");
        }
    }

    // Creates the render surface
    void createSurface() {
        // GLFW abstracts away the horrors of WIN32_API and X11, so we just use that and throw an exception if anything goes wrong
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) { // As usual, the nullptr is a pointer to a custom allocator
            throw std::runtime_error("Failed to create the window surface!");
        }
    }

    // Find all physical devices that support vulkan and choose one. This would likely be done via launcher menu in practice, but here we just use whatever has the most VRAM
    void pickPhysicalDevice() {
        // As is tradition, we get the count first before rerunning the enumeration to fill out a vector
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        // Throw an exception if there are no graphical devices
        if (deviceCount == 0) {
            throw std::runtime_error("Failed to find a supported GPU or physical device!");
        }

        // Get the devices
        std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data());

        // Evaluate if any of the devices are suitable for our needs and choose the best one
        for (const auto& device : physicalDevices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                std::cout << "Valid discrete GPU found!\n";
                break;
            }
        }

        // If the device is still Null, throw an exception
        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("None of the available GPUs or phsyical devices are suitable!");
        }
    }

    // Function for creating the logical device that interfaces with the phsyical device
    void createLogicalDevice() {
        // Indices of the queue families needed to construct our queues
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        // The queue priority has to be a reference to a constant, so we'll just define it here
        float queuePriority = 1.0f;

        // Since we're making multiple queues, we create a vector to hold them
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

        // Now we create a set of unique Queue families. We use a set to ensure we don't have duplicates
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        // As usual, we create a struct describing the queue we want to create. Since they're all instantiated the same way, we can just iterate over them (for now).
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{}; // Create the struct with default values
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO; // Struct type definition
            queueCreateInfo.queueFamilyIndex = queueFamily; // the indices of the supported family types
            queueCreateInfo.queueCount = 1; // The number of queues to create. In this case 1 since we're only using graphics, but this would really be equal to the number of entries in QueueFamilyIndices.
            queueCreateInfo.pQueuePriorities = &queuePriority; // Queue priority for command scheduling. This is required, even though we only have a single queue. It also has to be a const, which is annoying.
            queueCreateInfos.push_back(queueCreateInfo); // Emplace the queue into the queue vector
        }

        // Create a struct describing teh deviceFeatures we want to implement
        VkPhysicalDeviceFeatures deviceFeatures{}; // Create the struct with default values

        // Create a struct describing the logical device we want to create
        VkDeviceCreateInfo createInfo{}; // Instantiate the struct
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO; // Specify the struct type
        createInfo.pQueueCreateInfos = queueCreateInfos.data(); // Reference to the queueCreateInfo structs in the vector
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()); // Number of queues, same as the queueCreateInfos
        createInfo.pEnabledFeatures = &deviceFeatures; // Reference to the device features struct
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()); // Number of device extensions to suppport
        createInfo.ppEnabledExtensionNames = deviceExtensions.data(); // Pointer to the names of the supported extensions

        // These are depricated, but I'll include the validation layer specifications anyway
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size()); // Number of validation layers
            createInfo.ppEnabledLayerNames = validationLayers.data();  // Names of the validation layers.
        }
        else {
            createInfo.enabledLayerCount = 0;
        }

        // Instantiate the logical device and throw and exception if we fail for some reason
        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &logicalDevice) != VK_SUCCESS) { // Self explanatory. The nullptr is a pointer to a callback memory allocation function, which I'm not using yet.
            throw std::runtime_error("Failed to create the logical device!");
        }

        // Create the queues for teh graphics family and the presentation family
        vkGetDeviceQueue(logicalDevice, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(logicalDevice, indices.presentFamily.value(), 0, &presentQueue);
    }

    // Creates the swapchain for queing images
    void createSwapChain() {
        // First we get the supported swapchain features
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        // Then we call our helper functions to choose the correct format, presentation mode, and extents for the chain
        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        // We must also define how many images we want to keep in the swap chain. We want at least 1 more than the minimum so we don't find ourselves waiting for an image to render
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

        // We have to be careful not to exceed the maximum though, so we clamp.
        imageCount = std::clamp(imageCount, static_cast<uint32_t>(0), swapChainSupport.capabilities.maxImageCount);
        
        // As is tradition, we must define a struct for the swapchain creation
        VkSwapchainCreateInfoKHR createInfo{}; // Always 0 initialize!
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR; // Type of struct
        createInfo.surface = surface; // The surface the swap chain is associated with
        createInfo.minImageCount = imageCount; // The number of images the swapchain can support
        createInfo.imageFormat = surfaceFormat.format; // Swapchain format
        createInfo.imageColorSpace = surfaceFormat.colorSpace; // Swapchain color space
        createInfo.imageExtent = extent; // Extents of the images in the swap chain
        createInfo.imageArrayLayers = 1; // Layers an image can consist of. This is always 1 unless you're doing something weird, like steroscopic images

        // This is used because we're rendering directly to the chain, but if we wanted to do something I.E. post-processing, we'd have to use a different value like VK_IMAGE_USAGE_TRANSFER_DST_BIT
        // This would also involve moving the data around memory, so this is probably a major area to look into, since I will of course include post-processing in the engine
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        // We'll need some info about queue families to sepcify how the swap chain handles multiple queue families sharing an image
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        // We have to specify which queue families a using the swap chain concurrently. However the queue family indices have to be unique, so we must use exclusive mode if all of our features on are implemented on the same queue family as is usually the case
        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT; // Concurrent mode, since we have separate queue families handling graphics and presentation queues
            createInfo.queueFamilyIndexCount = 2; // Hard-coded here since we only care about graphics and presentation, but this would have to be set to however many unique queues we have
            createInfo.pQueueFamilyIndices = queueFamilyIndices; // Indices of the queue families being used
        }
        else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE; // The queue families are all the same, so we operate in exlusive mode
            createInfo.queueFamilyIndexCount = 0; // Not actually required since we 0-initialized anyway, and it's implied by the imageSharingMode
            createInfo.pQueueFamilyIndices = nullptr; // Not actually required since we 0-initialized anyway, and it's implied by the imageSharingMode
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform; // This is for performing transforms like rotations, etc., on the image. We aren't doing this though, so we use the current transform as the pre-transformation to avoid changing anything.
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; // This is for blending alpha with other windows in the window system. Things like fading the search box in notepad++ when you click off for example. We want to disable this by making it opaque.
        createInfo.presentMode = presentMode; // The chosen present mode.
        createInfo.clipped = VK_TRUE; // Vulkan's booleans are acutally ints for compatibiliy with old C styles. This means things are not obscured, i.e., by another window, are ignored. This means we can't access them, which culd be relevent. Right now we enable this.
        createInfo.oldSwapchain = VK_NULL_HANDLE; // This is required if you ever have to destroy and recreate the swap chain. We will have to do this for window resizing for example, but for simplicity we just disable it for now.

        // Finally, create the swap chain and assign the handle!
        if (vkCreateSwapchainKHR(logicalDevice, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create the swap chain!");
        }

        // Here we retrieve an images on the swapchain and add initialize the swapChainImages vector. This should probably be empty since we just made it, but it's more of a sanity thing.
        vkGetSwapchainImagesKHR(logicalDevice, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(logicalDevice, swapChain, &imageCount, swapChainImages.data());

        // Set the swapchain extents and image format on the class member variables
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;

    };

    // Creates image views for the images on the swapchain
    void createImageViews() {
        // Resize the vector to fit the a view for each image on the swapchain
        swapChainImageViews.resize(swapChainImages.size());

        // Iterate through the images to create a view for each
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            // We're creating something, so we must define the associated struct
            VkImageViewCreateInfo createInfo{}; // ALWAYS 0-initialize
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO; // Structure type
            createInfo.image = swapChainImages[i]; // The image this view is associated with
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D; // This specifies the type of texture, i.e., 1D, 2D, 3D, cubemap, etc.
            createInfo.format = swapChainImageFormat;

            // These parameters let us swizzle the color channels around if we want, so we could create effects like color blindness or monochrome. We're setting them to 'identity' so they aren't swizzled.
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY; 
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY; 
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY; 
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY; 

            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; // I believe this just indicates we're this image as a color target
            createInfo.subresourceRange.baseMipLevel = 0; // We're not introducing mipmaps yet, but this would specify the level to use for the chain
            createInfo.subresourceRange.levelCount = 1; // How many mipmap levels are there
            createInfo.subresourceRange.baseArrayLayer = 0; // Which layer to start with, 0-indexed. You'd only have more than 1 for weird stuff like steroscopic rendering
            createInfo.subresourceRange.layerCount = 1; // Same as above, we only use one layer

            // Crate the image view. Since we sized the vector beforehand we can just specify a reference to an index in the vector for the target
            if (vkCreateImageView(logicalDevice, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create image views!");
            }
        }
    }

    // Creates the render pass specification that defines the framebuffer attachments for the pipeline
    void createRenderPass() {
        // This is super simplified, so we only have a single color buffer attachment represented by one of the images in from the swap chain
        VkAttachmentDescription colorAttachment{}; // You know it's 0 initialized
        colorAttachment.format = swapChainImageFormat; // Image format used by the swapChain
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT; // We're not multisampling yet, so we only use a single sample per bit
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // What to do with the data in our frame buffer attachment before rendering. Here we clear to black.
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE; // What to do after. We want to actually render the contents of the frame buffer so we store it.
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; // Not usting stencil data (yet) so we don't care what happens do this data
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; // Ditto
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; // We don't care what the image layout is when we recieve it since we're not doing anything to it, so we set to undefined
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; // We do care about the output though since we want to display it, so we define it as an image to be presented to the swap chain

        // Now we need to describe a reference to our AttachmentDescriptions for use by rendering subpasses. This is thankfully very simple.
        VkAttachmentReference colorAttachmentRef{}; // 0 init
        colorAttachmentRef.attachment = 0; // Attachment index. Since we only have a single VkAttachmentDescription it will always be index 0 once we construct an array of them
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // We're using the attachment as a color buffer, so we use the optimal attachment for color, as the name suggests

        // Next is defining the actual subpass itself
        VkSubpassDescription subpass{}; // 0 initialized
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; // Specify that this is a graphics subpass, since the API can potentially support compute subpasses as well
        subpass.colorAttachmentCount = 1; // We only have the one color attachment defined earlier
        subpass.pColorAttachments = &colorAttachmentRef; // Reference to said color attachment. These could both be arrays of course. Worth noting that this color attachment is the 0th item in the array, which is referenced by the shader: layout(location = 0) out vec4 outColor

        // We're also going to add a subpass dependency. There are two implicit subpasses that handle the transition into and out of the render pass, but they assume our render pass starts at the beginning of the pipeline
        // Problem is, we don't actually start rendering until we get an image from the swapchain in the VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT stage.
        // To deal with this, we are going to add a dependency on that stage.
        VkSubpassDependency dependency{}; // 0 Init
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL; // This refers to the implicit subpass at the beginning or end of a render pass, depending on when it is specified. Here it is specified during srcSubpass, so it's the former 
        dependency.dstSubpass = 0; // This is our only subpass, and usually has to be equal or higher than srcSubpass. VK_SUBPASS_EXTERNAL is an exception though, since it referse to a subpass that is implicitly always first 
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; // This is the pipeline stage we need to wait on, or the 'dependency'. 
        dependency.srcAccessMask = 0; // Not masiking the srcStage. I'm not really sure why you would do this...  
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; // We need to wait on this to complete. So we defined it as the stage we have to wait on to start the subpass, and the one that marks the end of the subpass 
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; // I still don't know what this mask does, but I guess we're masking the write bit of the flag? Whatever that means. 

        // Lastly we create the struct taht defines the render pass for creation
        VkRenderPassCreateInfo renderPassInfo{}; // 0 init
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO; // Struct type
        renderPassInfo.attachmentCount = 1; // Number of attachments
        renderPassInfo.pAttachments = &colorAttachment; // Reference to the attachment or array of attachments
        renderPassInfo.subpassCount = 1; // Number of subpassed
        renderPassInfo.pSubpasses = &subpass; // Reference to the subpass or array of subpasses
        renderPassInfo.dependencyCount = 1; // We've only add the single dependency on the VK_PIPELINE_STAGE_COLOR_ATTACHMENT_BIT 
        renderPassInfo.pDependencies = &dependency; // This dependcy is the only one we care about. If we had defined others, we could include them here. 

        // We then construct our render pass as store it as a member variable
        if (vkCreateRenderPass(logicalDevice, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create render pass!");
        }
    }

    // Creates a graphics pipeline. In theory reuseable for multiple but probably only one kind for this example.
    void createGraphicsPipeline() {
        // Read our compiled vertex and fragment shaders into memory
        auto vertShaderCode = readFile("shaders/vert.spv");
        auto fragShaderCode = readFile("shaders/frag.spv");

        // Create the modules. Note that once they're loaded into the pipeline we don't need them anymore, so they're created in local scope to be automatically freed once the pipeline is created.
        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        // Now we have to create pipeline stages for these shaders. You know the deal, we're creating a Vulkan object sooooo
        VkPipelineShaderStageCreateInfo vertShaderStageInfo{}; // Always 0 initialize
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; // Struct type
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT; // Specific stage of the pipeline this shader is to be applied in
        vertShaderStageInfo.module = vertShaderModule; // The module to be registered with this stage
        vertShaderStageInfo.pName = "main"; // Shader entry point. You could have multiple shaders in a single file with different entry points, but we're using the basic "main" function here
        vertShaderStageInfo.pSpecializationInfo = nullptr; // This lets us specify constants in the shader for compiler optimization. It's intialized to null anyway, but I added for completeness.

        // Next we do the fragment shader. Same exact fields, just using FRAGMENT_BIT and teh fragShaderModule instead of teh vertex equivalents
        VkPipelineShaderStageCreateInfo fragShaderStageInfo{}; // ALways 0 initialize
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";
        fragShaderStageInfo.pSpecializationInfo = nullptr;

        // Pack these into an array for pipeline construction
        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        // Before we can use that however, we need to describe what the vertex data will look like going into the pipeline. This involves creating an info struct. Shocking.
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{}; // Always 0-initialize
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO; // Struct type
        vertexInputInfo.vertexBindingDescriptionCount = 0; // We're saying there is no vertex data to load, since we're hard-coding it, As such all of these are set to 0 for now, but we'll be using them later
        vertexInputInfo.pVertexBindingDescriptions = nullptr; // 
        vertexInputInfo.vertexAttributeDescriptionCount = 0; // 
        vertexInputInfo.pVertexAttributeDescriptions = nullptr; // 

        // Next we describe what this data will be used to draw.
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{}; // Always 0-initialize
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO; // Struct type
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; // We're drawing triangles. There are options for points and strips of various kinds too.
        inputAssembly.primitiveRestartEnable = VK_FALSE; // We're not breaking the tris into strips, so we can disable this. For now.

        // Now that we know what we're building, we need to describe how to rener it with a viewport object.
        VkViewport viewport{}; // Always 0-initialize, although this isn't a struct, so we have no sType
        viewport.x = 0.0f; // Starting value of the viewport along the x-axis. This is almost always 0, but there might be a reason not to do this.
        viewport.y = 0.0f;// Starting value of the veiwport along th y-axis
        viewport.width = (float)swapChainExtent.width; // End point the viewport is rendered to along the x-axis. We just use the extent of the swapchain since that should encompass the whole 'screen'
        viewport.height = (float)swapChainExtent.height; // Ditto for the height or y-axis
        viewport.minDepth = 0.0f; // Must be between 0.0 and 1.0. No reason to do anything special with this unless you have a specific reason
        viewport.maxDepth = 1.0f; // Ditto with minDepth

        // The scissor describes how the pixels are actually stored in the buffer. This basically masks part of the viewport if it's smaller than the viewport, and does some weird stuff if it's larger that I'll have to read up on.
        VkRect2D scissor{}; // Is that a 0-initialization I see?
        scissor.offset = { 0, 0 }; // Offset is 0 since we want to align it with the viewport
        scissor.extent = swapChainExtent; // We make it the same size as the swapChain extents so it doesn't actually cut anything out of the buffer or do anything weird to the final image

        // Now we create the viewport state struct using the scissor and the viewport object. I'm going to make it static for simplicity, but I could have made it dynamic to support fucking with the viewport at without rebuilding the pipeline
        VkPipelineViewportStateCreateInfo viewportState{}; // 0 to the initialize to the zation
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO; // Struct type
        viewportState.viewportCount = 1; // Number of viewports. We're not doing any multi-window shenanigans so we'll stick with 1 for now.
        viewportState.pViewports = &viewport; // Reference to the viewport object we created earlier
        viewportState.scissorCount = 1; // I'm not even sure how multiple scissors would even work. Stacking additively maybe?
        viewportState.pScissors = &scissor; // The scissor we're using ,since we don't have multiple

        // Now we configure the rasterizer, which takes our vertex shader and turns the vetices into fragments. It also handles things like face culling and depth testing, as well as all of the scissor test stuff we just defined
        VkPipelineRasterizationStateCreateInfo rasterizer{}; // Always 0-intialize
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO; // Struct type
        rasterizer.depthClampEnable = VK_FALSE; // This discards anything beyond the near and far planes of the render window. Requires a bit more work, so we'll ignore it for now. Useful for shadows and things of that nature.
        rasterizer.rasterizerDiscardEnable = VK_FALSE; // Disables the rasterizer, which in effect disables output to the frame buffer. We want to render stuff so we set this to false.
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL; // Fills in the area of polygons with fragments. Other options like point and line rendering require GPU options and I probably don't give a shit about wire frame rendering right now anyway
        rasterizer.lineWidth = 1.0f; // Thickest line width possible without additional GPU features. Measured in terms of the number of fragments to use to build a line.
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT; // Enables back-face culling
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE; // What direction the vertices are wound to determine front and back facing
        rasterizer.depthBiasEnable = VK_FALSE; // Offsets depth value based on a constant or shader value. We're disabling this for now since it's used for shadows and advnaced features
        rasterizer.depthBiasConstantFactor = 0.0f; // Not required with depthBias disabled.
        rasterizer.depthBiasClamp = 0.0f; // Not required with depthBias disabled.
        rasterizer.depthBiasSlopeFactor = 0.0f; // Not required with depthBias disabled.

        // Multisampling for i.e. anti-aliasing. It's disabled for now but I'll enable it later.
        VkPipelineMultisampleStateCreateInfo multisampling{}; // See those curly braces? That shit is 0-initialized
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO; // Struct type
        multisampling.sampleShadingEnable = VK_FALSE; // Disables mutlisampled shading
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT; // If only one plygon is mapped to a bit, we're effectively not using multi-sampling
        multisampling.minSampleShading = 1.0f; // Shader is disabled, so this doesn't matter. Still set it to 1 for later use though
        multisampling.pSampleMask = nullptr; // Same deal, this is optional without the other stuff
        multisampling.alphaToCoverageEnable = VK_FALSE; // Same deal, this is optional without the other stuff
        multisampling.alphaToOneEnable = VK_FALSE; // Same deal, this is optional without the other stuff

        // This is where we'd normally put a depth/stencil buffer struct, but we're not using one for now so we'll just pass a nullptr to anything that needs one.

        // Now we have to deal with the color blending. This involves two objects that need info sctructs.
        // The first is configured per attached frame buffer
        VkPipelineColorBlendAttachmentState colorBlendAttachment{}; // 0-initialized
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT; // Mast for the color bits. RGBA is used here. so we OR the masks together
        colorBlendAttachment.blendEnable = VK_FALSE; // This means we just use whatever is in the fragment shader. The following parameters can be combined to perform various color blending techniques if we set this to true
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // ALl of this is optional. The actual use is kind of detailed, so I'll leave it for later. The details are in the specification
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

        // The second is configured globally
        VkPipelineColorBlendStateCreateInfo colorBlending{}; // 0 initialized
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO; // Struct type
        colorBlending.logicOpEnable = VK_FALSE; // This would enable bitwise blending operations and disables the other method from the AttachmentState.
        colorBlending.logicOp = VK_LOGIC_OP_COPY; // We're just copying  the original, so nothing is done here
        colorBlending.attachmentCount = 1; // Single attachment. I'm not actually sure what this means, but I assume it's fragments attached to polys or vis versa
        colorBlending.pAttachments = &colorBlendAttachment; // The per buffer configuration
        colorBlending.blendConstants[0] = 0.0f; // All of these are optional since we aren't using the bitwise blend operation
        colorBlending.blendConstants[1] = 0.0f; // Optional
        colorBlending.blendConstants[2] = 0.0f; // Optional
        colorBlending.blendConstants[3] = 0.0f; // Optional

        // Now that we have defined how our shaders and rasterizer will work, we have to define the layout of the pipeline
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{}; // 0 intitialize
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO; // Struct type
        pipelineLayoutInfo.setLayoutCount = 0; // We're creating an empty pipeline, so all of this is null/0 at creation.
        pipelineLayoutInfo.pSetLayouts = nullptr; // Optional
        pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
        pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

        // Create the pipeline layout and store it on a member variable for later use.
        if (vkCreatePipelineLayout(logicalDevice, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        // Lastly we scribe how we're going to handle the dynamic aspects of the pipeline
        // We're really only looking at the Viewport and Scissor for dynamic states right now
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };

        VkPipelineDynamicStateCreateInfo dynamicState{};// 0 Initialized, as always
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO; // Struct type
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()); // Number of dynamic states
        dynamicState.pDynamicStates = dynamicStates.data(); // Actual vector of dynamic states

        // At long last, we create the sturcture for building the graphics pipeline
        VkGraphicsPipelineCreateInfo pipelineInfo{}; // 0 intialize the struct, as always
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO; // Struct type
        pipelineInfo.stageCount = 2; // We have 2 stages, the vertext and fragment shaders
        pipelineInfo.pStages = shaderStages; // Array containing our stages
        pipelineInfo.pVertexInputState = &vertexInputInfo; // Vertex shader stage
        pipelineInfo.pInputAssemblyState = &inputAssembly; // Descriptor for what the incoming data looks likde
        pipelineInfo.pViewportState = &viewportState; // Descriptor of what the viewport looks like
        pipelineInfo.pRasterizationState = &rasterizer; // Descriptor for what the rasterizer does
        pipelineInfo.pMultisampleState = &multisampling; // What our multisampling will look like
        pipelineInfo.pDepthStencilState = nullptr; // We're not stenciling so we leave this null
        pipelineInfo.pColorBlendState = &colorBlending; // Descriptor for our color blending stage
        pipelineInfo.pDynamicState = &dynamicState; // Describes how we handle the few dynamic parts of the pipeline
        pipelineInfo.layout = pipelineLayout; // Pipeline layout
        pipelineInfo.renderPass = renderPass; // Render pass description
        pipelineInfo.subpass = 0; // Using the 0th render pass, since that is the only one we defined.
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Handle to an existing pipeline, since it's cheaper to modify a similar one than make a new one from scratch. We don't have another one though, so we leave it NULL
        pipelineInfo.basePipelineIndex = -1; // Reference a pipeline that is about to be cleared by index and build off that. Probably useful for rebuilding the pipeline when you change static parts of it

        // Create the graphics pipeline as assign to the member variable
        if (vkCreateGraphicsPipelines(logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        // Clean up the shaders since they've already been registered with the pipeline
        vkDestroyShaderModule(logicalDevice, fragShaderModule, nullptr);
        vkDestroyShaderModule(logicalDevice, vertShaderModule, nullptr);
    }

    // Creates the frame buffers for submission to the swap chain
    void createFrameBuffers() {
        // Resize the array of buffers to fit the swap chain
        swapChainFramebuffers.resize(swapChainImageViews.size());

        // Iterate over the image vies and create a frame buffer for each
        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            VkImageView attachments[] = {
                swapChainImageViews[i]
            };

            // Since we're creating something, we need a struct
            VkFramebufferCreateInfo framebufferInfo{}; // 0 Init
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO; // Struct type
            framebufferInfo.renderPass = renderPass; // Render pass definition
            framebufferInfo.attachmentCount = 1; // Number of attachments. We only have the one color attachment
            framebufferInfo.pAttachments = attachments; // This is the image view that should be found on the associated attachment descriptor
            framebufferInfo.width = swapChainExtent.width; // Width of the swapchain image
            framebufferInfo.height = swapChainExtent.height; // Height of the swapchain image
            framebufferInfo.layers = 1; // Number of layers. We're not doing any weird shit with steroscopy so this is just 1.

            // Insert the frame buffer into the array of frame buffers
            if (vkCreateFramebuffer(logicalDevice, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create framebuffer!");
            }
        }
    }

    // Create the command pool for storing command buffers
    void createCommandPool() {
        // We need to know what the index is for the graphics queue family since we are using render commands, so we query the device. I don't really know why this isn't just captured once though, it seems wasteful to redo this every time.
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        // Command pool needs to be created, so we get a struct
        VkCommandPoolCreateInfo poolInfo{}; // 0 Initialized
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; // Struct type
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // Allows commands to be rerecorded individually. Can also set VK_COMMAND_POOL_CREATE_TRANSIENT_BIT by ORing them to indicate the pool is rerecorded often. We're just recording over it though.
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(); // Index of the graphics queue family

        // Create the summbitch and assign it to the member variable
        if (vkCreateCommandPool(logicalDevice, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    // Create a command buffer for recording commands to
    void createCommandBuffer() {
        // Create a command buffer allocation struct to describe what command pool and number of buffers to use
        VkCommandBufferAllocateInfo allocInfo{}; // 0 init
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; // Struct type
        allocInfo.commandPool = commandPool; // Command pool to use to store the buffer
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; // Primary buffers can be submitted to a queue directly, whereas secondary buffers can only be alled from a primary buffer, and can't be submitted directly
        allocInfo.commandBufferCount = 1; // Number of buffers. We only have the one.

        // Create the buffer and assign to member variable
        if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    // Creates the needed synchronization objects, such as fences, locks, and semaphores
    void createSyncObjects() {
        // Creating a semaphore basically takes a blank struct with only the type defined
        VkSemaphoreCreateInfo semaphoreInfo{}; // 0 Init
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO; // Struct type

        // Ditto for the fence
        VkFenceCreateInfo fenceInfo{}; // 0 Init
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO; // Struct type
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // Start with the signal set so we can execute on the first rendering frame

        // Because of this, we can create both Sempaphores using this struct
        if (vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &imageAvailableSemaphore) != VK_SUCCESS ||
            vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &renderFinishedSemaphore) != VK_SUCCESS ||
            vkCreateFence(logicalDevice, &fenceInfo, nullptr, &inFlightFence) != VK_SUCCESS) {
            throw std::runtime_error("failed to create semaphores!");
        }
    }

    // Record to a command buffer
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        // Struct for creating a command to be recorded into the buffer
        VkCommandBufferBeginInfo beginInfo{}; // 0 Initialized
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO; // Type of struct
        beginInfo.flags = 0; // Not needed, but can be used to specify things like when it will be rerecorded and if you can resubmit it while it is already pending execution
        beginInfo.pInheritanceInfo = nullptr; // Only relevant to secondary buffers. Specifies what sate to inherit from the calling primary buffer

        // Once again, we create it and add it to the commandBuffer. Everything between this and the complete will be recorded.
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        // Color to sue when we clear the render area.
        VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} }; // Black with 100% opacity

        // Now that we've added a command to the command buffer, we can begin a render pass
        VkRenderPassBeginInfo renderPassInfo{}; // 0 init
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO; // Struct type
        renderPassInfo.renderPass = renderPass; // The render pass we are going to begin
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex]; // The frame buffer, which is a color image, to bind to
        renderPassInfo.renderArea.offset = { 0, 0 }; // We don't want an offset since we want to align the render area with the viewport
        renderPassInfo.renderArea.extent = swapChainExtent; // Similarly, we want the render area to match the swap chain extents, which we configured to be the same as the viewport
        renderPassInfo.clearValueCount = 1; // Clear once
        renderPassInfo.pClearValues = &clearColor; // Color to use. In this case black with 100% opacity

        // Command to begin the renderpassInfo with the command buffer
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE); // Last argument specifies if this is executed from primary or secondary command buffer. Inline means primary.

        // Bind the grahics pipeline to the command buffer now that a command is in it and we've begun a render pass
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline); // Second command just tells it this is a graphics pass, not a compute pass

        // Since Scissor and Viewport state are dynamic, we also have to set those in the command buffer
        VkViewport viewport{}; // 0 Init
        viewport.x = 0.0f; // I already defined these when setting up the graphics pipeline, and honestly they're self explanatory anyway
        viewport.y = 0.0f; 
        viewport.width = static_cast<float>(swapChainExtent.width); 
        viewport.height = static_cast<float>(swapChainExtent.height); 
        viewport.minDepth = 0.0f; 
        viewport.maxDepth = 1.0f; 
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport); 

        // Also already defined these
        VkRect2D scissor{}; // But I will continue to compulsive point out that this is 0 initiated, as indicated by the {}
        scissor.offset = { 0, 0 }; 
        scissor.extent = swapChainExtent; 
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor); 

        // At lon last, we issue the draw command to draw the triangle
        vkCmdDraw(commandBuffer, 3, 1, 0, 0); // Vetex count, instance count, first vertex, first instance

        // End the render pass
        vkCmdEndRenderPass(commandBuffer);

        // Complete recording the command buffer.
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }

    }

    // Function for drawing a frame to the window
    void drawFrame() {
        // We need to wait on the fence before we draw this frame. It starts unset so we can begin on the first frame
        vkWaitForFences(logicalDevice, 1, &inFlightFence, VK_TRUE, UINT64_MAX); // device, how many fences, do we wait on all or one, and timeout, which me made comically large at the max value of a 64-bit unsigned integer

        // Immediately reset the fence, since we're going to want to wait on it again next frame
        vkResetFences(logicalDevice, 1, &inFlightFence); // device, number of fences, fence

        // We must first acquire an image from the swap chain, which will be the image's index in the chain
        uint32_t imageIndex; 

        // The device, the swap chain, timout value, the semaphore to wait on, fence (which we're not using), and the variable to store the image index
        vkAcquireNextImageKHR(logicalDevice, swapChain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

        // Now we need to record the command buffer. We first reset it to ensure it's ready to be recorded to
        recordCommandBuffer(commandBuffer, imageIndex);

        // Specify the semaphores to wait on
        VkSemaphore waitSemaphores[] = { imageAvailableSemaphore }; // We only have the one semaphore to wait on

        // Specify the stages to wait at
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT }; // We're just waiting at the stage where we actually write the colors to the image, since the image needs to actually be there to write to

        // We also define the array of semaphores to signal once we complete the command. Once again, we only have the one
        VkSemaphore signalSemaphores[] = { renderFinishedSemaphore }; // Just add the single semaphore to an array

        // Now we define the submit info struct for submitting to the queue
        // Note that the index of the waitSempaphore array corresponds to the stage that will be waiting in the waitStages array, meaning they must both be the same length and they must be coordinated.
        VkSubmitInfo submitInfo{}; // 0 Initialized
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; // Struct type
        submitInfo.waitSemaphoreCount = 1; // Number of semaphores to wait on
        submitInfo.pWaitSemaphores = waitSemaphores; // The array of semaphores
        submitInfo.pWaitDstStageMask = waitStages; // The array of stages to wait at
        submitInfo.commandBufferCount = 1; // We are only submitting one command buffer
        submitInfo.pCommandBuffers = &commandBuffer; // We only have 1 to submit anyway, so we specify it here
        submitInfo.signalSemaphoreCount = 1; // Only signally one semaphore when the command buffer completes
        submitInfo.pSignalSemaphores = signalSemaphores; // It's the one we just put in an array by itself

        // Now we submit that info to the graphics queue and throw and exception if it doesn't work for some reason
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFence) != VK_SUCCESS) { // submitInfo can be an array if you're submitting a bunch of stuff. The last argument is the fence to signal when the command buffer is done executing so we can render our next frame
            throw std::runtime_error("Failed to submit draw command buffer!");
        }

        // The swapchains to return the results to
        VkSwapchainKHR swapChains[] = { swapChain }; // Again, we have only one chain so we just put that into an array by itself

        // Last step is to return the results of this render pass to the swapchain
        VkPresentInfoKHR presentInfo{}; // 0 Initialized
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR; // Struct type
        presentInfo.waitSemaphoreCount = 1; // Number of semaphores to wait of for presentation
        presentInfo.pWaitSemaphores = signalSemaphores; // The semaphores. In this case the one that is signalled when the render pass completes
        presentInfo.swapchainCount = 1; // Number of swapchains to present the results to. Again, we only have the 1. And apparently you generally don't submit to multiple in practice either.
        presentInfo.pSwapchains = swapChains; // Our array of swapchains
        presentInfo.pImageIndices = &imageIndex; // The index in the swapchain we're returning the result to, in this case the same one we grabbed it from, which I assume is common practice
        presentInfo.pResults = nullptr; // Only needed if you have more than one swap chain and you need to check the results of all of them. Otherwise you can just use the same vk*Submit check we've been using

        // Present the result back to the swapchain
        vkQueuePresentKHR(presentQueue, &presentInfo);

    }

    // Check that all of our requested valiation layers are available
    bool checkValidationLayerSupport() {
        // Same deal as instance enumeration. Run this onces with a nullptr to get layer count, then again with the destination vector set
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);  
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        // Iterate over the vector of validation layers and make sure all of them appear in the list of available layers
        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if (std::strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

    // Finds the supported queue familes for the given phsyical device
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        // Struct containing our queue family indices (duh)
        QueueFamilyIndices indices;

        // Needed to query if a queue family supports presenting to a surface or not
        VkBool32 presentSupport = false;
        
        // Unsurprisingly, we query for count then query again to fill in a vector of values
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        // Find a queue family that supports VK_QUEUE_GRAPHICS_BIT, which is required to enable graphics rendering.
        for (int i = 0; i < queueFamilies.size(); i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { // This is a bitmask, so we just logical AND it to make sure the bit is still set.
                indices.graphicsFamily = i;
            }
            
            // Check if 'present' is supported by any queue family
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
            if (presentSupport) {
                indices.presentFamily = i;
            }

        }

        return indices;
    }

    // Evaluates a physical device for suitability. For now it just uses the first discrete GPU we find that supports everything.
    bool isDeviceSuitable(VkPhysicalDevice device) {
        VkPhysicalDeviceProperties deviceProperties{};
        vkGetPhysicalDeviceProperties(device, &deviceProperties);

        // Return false if no discrete GPU was found
        if (!(deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)) {
            return false;
        }

        // Return false if all of our queue families are not supported
        if (!findQueueFamilies(device).isComplete()) {
            return false;
        }

        // Return false if all of our extensions are not supported
        if (!checkDeviceExtensionSupport(device)) {
            return false;
        }

        // Return false if the swap chain doesn't support at least one format and one presentation mode. We can be more specific here later.
        SwapChainSupportDetails swapChainSupportDetails = querySwapChainSupport(device);
        if ((swapChainSupportDetails.formats.size() == 0) || (swapChainSupportDetails.presentModes.size() == 0)) {
            return false;
        }

        return true;
    }

    // Checks that the given device supports our required extensions
    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        // Typical approach for querying stuff in Vulkan
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        for (const char* extensionName : deviceExtensions) {
            bool extensionFound = false;

            for (const auto& extensionProperties : availableExtensions) {
                if (std::strcmp(extensionName, extensionProperties.extensionName) == 0) {
                    extensionFound = true;
                    break;
                }
            }

            if (!extensionFound) {
                return false;
            }
        }
        return true;
    }  

    // Gets the supported features of a given device's swapchain
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;

        // We first query the basic surface capabilities. This returns only a sinlge struct, so we don't have to do the typical call -> create vector of size -> call again process
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        // Formats returns a vector of format strcuts, so we DO have to go through the typical process for that
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        // Only query and resize if there are supported formats
        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        // Same deal for the supported present modes
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        // Only query and resize if there are supported present modes
        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }



        return details;
    }

    // Chooses the most optimal surface format
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        // Check if the prefferred formats are available. Industry standard is SRGB with 8-bit colors
        for (const auto& availableFormat : availableFormats) {
            if ((availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) && (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB)) {
                return availableFormat;
            }
        }

        // If we couldn't find what we wanted, we'll have to fall back to whatever is available
        return availableFormats[0];
    }

    // Choos the most optimal present mode
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        // We use VK_PRESENT_MODE_MAILBOX_KHR on recommendation, but we'll want to actually think about this for a real implementaion
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }

        // We sue VK_PRESENT_MODE_FIFO_KHR as our fallback, as it is guarenteed to be available on anything that supports presentation
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    // Choose the most optimal swap extent, which is the screen resolution in pixels. Note that this can be different from the screen coordinates on high resolutiond displays, so we don't use those.
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) { // If the extents are defines as 0xFFFFFFFF then we know the window manager is handling this for us
            return capabilities.currentExtent;
        }
        else { // Otherwise, we have to explicitly ask the manager for the frame buffer size
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            // Create the VkExtent2D sctruct with these values
            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            // We must then clamp the current extents to whatever our real maximum and minimum are.
            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;

        }
    }

    // Helper function for loading shaders into memory as char arrays
    static std::vector<char> readFile(const std::string& filename) {
        // Open a file stream at the end of the file in binary format
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        // Throw and exception if the file failed to open for any reason
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file!");
        }

        // Preallocate the memory
        size_t fileSize = (size_t)file.tellg(); 
        std::vector<char> buffer(fileSize);

        // Seek back to beginning of file and read the whole thing into memory
        file.seekg(0);
        file.read(buffer.data(), fileSize);

        // Close and return the buffer
        file.close();

        return buffer;
    }

    // Helper function to create shader modules after loading a compiled SPIR-V shader
    VkShaderModule createShaderModule(const std::vector<char>& shaderBinary) {
        // We're creating something in Vulkan, so you know what that means...
        VkShaderModuleCreateInfo createInfo{}; // Always 0-initialize
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO; // Struct type
        createInfo.codeSize = shaderBinary.size(); // Size of the compiled shader binary in bytes

        // Pointer to the shader code. We loaded it in as a character array though, so we have to cast to uint32_t
        // Note that this could violate alignment, since an array of 8-bit char is might not allign with a 32-bit pointer. Thankfully, vector always satisfies worst-case alginment, but keep this in mind if you change the fileRead funtion.
        createInfo.pCode = reinterpret_cast<const uint32_t*>(shaderBinary.data());

        // Create the shader module and return it
        VkShaderModule shaderModule;
        if (vkCreateShaderModule(logicalDevice, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create shader module!");
        }

        return shaderModule;
    }

    // Main execution loop. Keeps the window alive and polls for event.
    void mainLoop() {
        // Loop until we should close the window or we error out.
        while (!glfwWindowShouldClose(window)) {
            // Event polling, which probably becomes super relevant later for resizing and other window shit. Including rendering.
            glfwPollEvents();

            // Draw a frame
            drawFrame();
        }
    }

    // This tuff will probably be implemented using RAII but for now we do it manually to be explicit.
    void cleanup() {
        // Cleanup the synch objects. If I have a lot of these, it would probably be smarter to store them in an array and iterate over that. As it stands, I only have a few though
        vkDestroySemaphore(logicalDevice, imageAvailableSemaphore, nullptr); 
        vkDestroySemaphore(logicalDevice, renderFinishedSemaphore, nullptr); 
        vkDestroyFence(logicalDevice, inFlightFence, nullptr);

        // Destroy the command pool
        vkDestroyCommandPool(logicalDevice, commandPool, nullptr);

        // Destroy all the frame buffers before the pipelines they're associated with
        for (auto framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(logicalDevice, framebuffer, nullptr);
        }

        // Destroy the actual pipeline before destroying it's layout
        vkDestroyPipeline(logicalDevice, graphicsPipeline, nullptr);

        // Destroy the graphics pipeline layout
        vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);

        // Destroy the render pass used to describe the attachments to the pipeline
        vkDestroyRenderPass(logicalDevice, renderPass, nullptr);

        // Destroy the image views before any of the lower lever objects
        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(logicalDevice, imageView, nullptr);
        }

        // Destroy the swap chain
        vkDestroySwapchainKHR(logicalDevice, swapChain, nullptr);

        // Destroy the logical device first since it references the other Vulkun objects
        vkDestroyDevice(logicalDevice, nullptr);

        // Destroy the surface before the instance, since the instance has a handle to the surface.
        vkDestroySurfaceKHR(instance, surface, nullptr);

        // Destroy the vkInstance before the window.
        vkDestroyInstance(instance, nullptr); // ONce again, nullptr for validation layer.

        // Destroy the glfwWindow
        glfwDestroyWindow(window);

        // Terminate GLFW, which is probably the bookend to 'init'. We probably want to make sure the window is gone first.
        glfwTerminate();

    }
};
