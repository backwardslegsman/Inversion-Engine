Just some notes for reference. Some day this will be a real readme

## Why aren't you using headers?
Because I'm using templates, and making header files with templates is a huge pain in the ass. I'll do it later.

## Why does your renderer just draw a hard coded triangle?
Graphics programming is hard

## TODO List
# Renderer
* TODO: Create a JSON file format that encode ALL of the data required to build a graphics pipeline
* TODO: Implement helper functions for creating each of the structs used in pipeline creation from JSON
* TODO: Create a .pak format that contains a JSON header detailing the offset and size into the file for resources
* TODO: Create a .gltf importer that loads model vertex and material data and adds it to the .pak and modifies the header appropriately
* TODO: Implement helper functions for loading vertex and material data out of the .pak
* TODO: Add logic for allocating memory UBOs and pixel buffers. Similar to what I did for texture and vertex memory allocation.
* TODO: Add logic for MVP matrix push constant
* TODO: Update RenderAsset struct to include all the allocation details for the vertices and material textures, as well as the associated graphics pipeline
* TODO: Bring it all together. A function that reads the .pak header, loads the vertex, texture, shader, and pipeline data. Then loads the asset data into memory and builds/caches the pipelines.
* TODO: Create sync objects and build the draw_frame() function
* TODO: Integrate Dear ImGUI and start messing around with moving/scaling things and moving the camera around
*       This will require a lot of misc. functions to support, and may be implementable using my ECS. I'll probably write a basic object oriented engine for testing
* TODO: REFACTOR! Break this massive class down into some smaller, specialized classes
* TODO: Write unit tests for these smaller, specialized classes
* TODO: Start messing around with shaders. Really beat the hell out of this by trying to load/unload various things, create/destroy/load pipelines, etc.
* TODO: Add support for Ultralight for HTML/CSS rendering support
* TODO: Implement a GUI system using Ultralight and the HomeBrewery Dungeons and Dragons CSS style sheet
* TODO: Probably refactor and unit test again
* TODO: Start working on the VTT project!

# Entity Component System
* TODO: Replace nested vectors with large, single vectors + indexing to avoid pinter invalidation when inner vectors are resized

# Main
* Change the system execution logic to buffer changes, then apply them once all threads have completed. Right now the systems modify the components immediately, meaning they can't operate on the same data without serious race conditions or synchronization
