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
        std::cout << "Vulkan Initialized!\n";
        mainLoop();
        std::cout << "Main Loop Started!\n";
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
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
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

    // Main execution loop. Keeps the window alive and polls for event.
    void mainLoop() {
        // Loop until we should close the window or we error out.
        while (!glfwWindowShouldClose(window)) {
            // Event polling, which probably becomes super relevant later for resizing and other window shit. Including rendering.
            glfwPollEvents();
        }
    }

    // This tuff will probably be implemented using RAII but for now we do it manually to be explicit.
    void cleanup() {
        // Destroy the swap chain
        //vkDestroySwapchainKHR();

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
