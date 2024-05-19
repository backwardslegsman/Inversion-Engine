#pragma once
#include <iostream>
#include <unordered_map>
#include <functional>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
#include <future>
#include <glm.hpp>


// Toggle debug messages
//#define M_DEBUG_ENA
//#define DEBUG_ENA
extern std::mutex debugMutex;

// Indicates that an entity or component is invalid
static int INVALID_ID = -1;

// Thread-safe debug macro with mutex lock for general debugging
#ifdef DEBUG_ENA
#define DEBUG(str) do { \
	std::lock_guard<std::mutex> lock(debugMutex); \
	std::cout << str << std::endl; } while( false )
#else
#define DEBUG(str) do { } while ( false )
#endif

// Thread-safe debug macro with mutex lock for memory debugging
#ifdef M_DEBUG_ENA
#define MDEBUG(str) do { \
	std::lock_guard<std::mutex> lock(debugMutex); \
	std::cout << str << std::endl; } while( false )
#else
#define MDEBUG(str) do { } while ( false )
#endif

namespace ECS_Core{
	// Base structure for struct type erasure.
	// 'alive' indicates if the component has been destroyed
	struct baseComponent {
		bool alive = true;
	};

	// Base class for memory pools to be overriden
	class BaseMemoryPool {
	public:
		virtual ~BaseMemoryPool() = default;
		virtual int allocate(baseComponent* initialValue = nullptr) = 0;
		virtual void deallocate(int componentIndex) = 0;
		virtual baseComponent* getComponent(int componentIndex) = 0;
		virtual void expandPool() = 0;
	};

	// Class responsible for managing the memory pool for a given component type
	template<typename T>
	class MemoryPool : public BaseMemoryPool {
	public:

		// Constructor that instantiates the memory pool to have poolSize number of entries
		MemoryPool(size_t initialPoolSize, size_t blockSize) : poolSize(initialPoolSize), blockSize(blockSize), nextIndex(0) {
			pool = new T[initialPoolSize];
		}

		// Destructor to deallocate the pool
		~MemoryPool() override {
			delete[] pool;
		}

		// Checks that there is enough memory for another allocation, then returns the index to the allocation and then increments the index
		int allocate(baseComponent* initialValue = nullptr) override {
			int freeIndex = nextIndex;

			// Expand the pool if we are out of space
			if (nextIndex >= poolSize) {
				expandPool();
			}

			// Check if there are any available entries in the deadPool for reuse. Use the first element in the array if so.
			if (deadPool.size() > 0) {
				freeIndex = deadPool.back();
				deadPool.pop_back();
				MDEBUG("Recycling " << typeid(T).name() << " memory at location " << &pool[freeIndex]);
			}

			// If a component was provided, use that as the initial value.
			if (initialValue) {
				MDEBUG("Copying " << reinterpret_cast<T*>(initialValue)->x << " to memory pool location " << &pool[freeIndex]);
				memcpy(&pool[freeIndex], initialValue, sizeof(T));
			}
			else {
				// Default init the component, set the alive flag when memory is allocate, then return
				pool[freeIndex] = T();
				reinterpret_cast<baseComponent*>(&pool[freeIndex])->alive = true;
				MDEBUG("New memory allocated for " << typeid(T).name() << " at " << &pool[freeIndex]);
			}

			// Increment nextIndex if we didn't reuse memory
			if (freeIndex == nextIndex) {
				return nextIndex++;
			}

			return freeIndex;

		}

		// Sets the 'alive' flag to false, indicating that the component at the given location has been destroyed.
		void deallocate(int componentIndex) override {
			// The index must be less than poolSize to be valid
			if (componentIndex < poolSize) {
				baseComponent* block = reinterpret_cast<baseComponent*>(&pool[componentIndex]);
				block->alive = false;

				// Add the component index to the 'deadpool' buffer for reallocation
				deadPool.emplace_back(componentIndex);
				MDEBUG(typeid(T).name() << " deallocated and " << componentIndex << " added to the deadpool");
				return;
			}
			throw std::range_error("Attempted to deallocate an unallocated memory block for component");
		}

		// Retrieves a pointer to a component in the pool.
		T* getComponent(int componentIndex) override {
			if (componentIndex < poolSize) {
				return &pool[componentIndex];
			}
			throw std::range_error("Attempted to access a component that does not exist!");
		}

	private:
		// Number of blocks comprising the pool
		size_t poolSize;

		// Size of an individual block in the pool, measured in the number of entities
		size_t blockSize;

		// The next available index, not counting the deadPool
		size_t nextIndex;

		// Pointer to the memory location containing the array of components
		T* pool;

		// Keeps track of any deallocated components so their memory locations can be reused instead of destroyed
		std::vector<int> deadPool;

		// Expand the bool by blockSize.
		void expandPool() override {

			// Calculate expanded pool size
			size_t newSize = poolSize + blockSize;
			DEBUG("Resizing " << typeid(T).name() << " memory pool from " << poolSize << " to " << newSize);

			// Create the new pool
			T* newPool = new T[newSize];

			// Copy existing data to the new pool
			memcpy(newPool, pool, newSize);

			// Destroy the old pool
			delete[] pool;

			// Change pool pointer to new pool
			pool = newPool;
			poolSize = newSize;
		}
	};

	// Class responsible for registering components and associating their IDs wtih memory pools
	class ComponentRegistry {
	public:
		// Instantiate generatorID
		ComponentRegistry() {
			generatorID = 0;
		}

		// Accepts any struct and registers it with the system
		template<typename T>
		int registerComponent(int poolSize, int blockSize)
		{

			// create a memory pool for the component, then return an ID to reference it in the map
			int id = getID();
			memoryPools[id] = std::make_unique<MemoryPool<T>>(poolSize, blockSize);
			return id;
		}

		// Returns a pointer to the memory pool for the given component type ID
		BaseMemoryPool* getComponentMemoryPool(int component_id)
		{
			// Iterate over map until you find the component. Return null if we reach the end of the list.
			auto it = memoryPools.find(component_id);
			if (it != memoryPools.end()) {
				return it->second.get();
			}
			return nullptr;
		}

	private:
		// Get a fresh ID
		int getID()
		{
			// Get a new unique numeric ID
			int newID = this->generatorID++;

			// Return successful
			return newID;
		}

		// Increments each time an ID is assigned to a component
		int generatorID;

		// Associates each struct with a unique numeric ID
		std::unordered_map<int, std::unique_ptr<BaseMemoryPool>> memoryPools;
	};

	// Class for registering and managing entities.
	class EntityRegistry {
	public:
		// Constructor preallocates memory for componentSets vector
		EntityRegistry(size_t setSize, size_t entityBlock, size_t archetypeBlock) : setSize_(setSize), entityBlock_(entityBlock), archetypeBlock_(archetypeBlock) {
			entityVector.reserve(entityBlock_);
			componentIndexMap.reserve(entityBlock_);
			allocateComponentBlocks();
		}

		// Destructor
		~EntityRegistry() {
			// Free all the memory associated with the entityLists
		}

		// Adds a component to the entity array and complementary componentSet array. Allocates additional blocks as necessary.
		uint32_t createEntity() {

			// Check if both arrays can accommodate a new entry and allocate additional blocks as needed
			// Both arrays should always be the same size, so we don't have to check both until I add error handling...
			if (entityVectorIndex >= entityVector.capacity()) {
				MDEBUG("Entity Vector expanded from " << entityVector.capacity() << " to " << entityVector.capacity() + entityBlock_);
				entityVector.reserve(entityVector.capacity() + entityBlock_);
				componentIndexMap.reserve(entityVector.capacity() + entityBlock_);
				allocateComponentBlocks();
			}

			// Add the entity
			entityVector.push_back(entityVectorIndex);

			// Add component Map
			componentIndexMap.emplace_back();

			// Return the new entity ID and increment
			DEBUG("Entity created with ID " << entityVectorIndex);
			return entityVectorIndex++;
		}

		// Gets the entity ID by index. -1 indicates a dead or deallocated entity.
		int getEntityID(int index) {
			return entityVector[index];
		}

		// Add a component to the given entity
		int addComponent(int entity, int component_id, ComponentRegistry* compRegistry, baseComponent* initialValue = nullptr) {
			// Check if the entity already has this component
			if (componentIndexMap[entity].find(component_id) != componentIndexMap[entity].end()) {
				return INVALID_ID;
			}

			// Confirm the set has enough space for another entity
			if (componentSets[entity].size() >= setSize_) {
				return INVALID_ID;
			}

			// Create the component in the component registry
			auto pool = compRegistry->getComponentMemoryPool(component_id);
			int comp_index = pool->allocate(initialValue);
			if (initialValue) { MDEBUG("Component Creation initiated for component type " << component_id << " on entity " << entity); };

			// Copy the old set vector into an unordered set
			std::unordered_set<int> oldSet;
			for (auto it : componentSets[entity]) {
				oldSet.insert(it);
			}

			// Add the component ID to the set
			componentSets[entity].emplace_back(component_id);

			// Associate the component with the component's index
			componentIndexMap[entity].emplace(component_id, comp_index);

			// Copy the new set vector into an unordered set
			std::unordered_set<int> newSet;
			for (auto it : componentSets[entity]) {
				oldSet.insert(it);
			}

			// Update archetype lists
			updateArchetypeLists(entity, oldSet, newSet);

			// TODO: Right now just using 0 to indicate nothing went wrong, but will want better error handling later.
			return 0;
		}

		// Access the given component on an entity
		baseComponent* getComponent(int entity, int component_id, ComponentRegistry* compRegistry) {
			// Get the component's index in memory pool
			auto find_comp = componentIndexMap[entity].find(component_id);
			if (find_comp != componentIndexMap[entity].end()) {
				return compRegistry->getComponentMemoryPool(component_id)->getComponent(find_comp->second);
			}
			return nullptr;
		}

		// Query only the entities in the 'entities' vector
		std::vector<int> queryEntitySubset(const std::unordered_set<int>& query, const std::vector<int>& entities) {
			// Vector of entities matching the query
			std::vector<int> results;

			// Loop over the component sets for each of the entities
			for (auto it : entities) {
				// Skip any entries with INVALID_ID, which are dead entities
				if (componentSets[it].empty() || entityVector[it] == INVALID_ID) {
					continue;
				}

				
				// Stop iterating and move to the next query parameter if we find a match
				bool pass = true;
				for (uint32_t q : query) {
					if (!pass) {
						break;
					}

					for (int j = 0; j < componentSets[it].size(); j++) {
						if (componentSets[it][j] == q) {
							break;
						}
						// This first argument assumes the components are sorted numerically from lowest to highest.
						// It's only worth sorting if queries are more common than entity adds, which seems unlikely
						// In that case, we would just remove the '> q' inequality.
						else if ((componentSets[it][j] > q) or (j == componentSets[it].size() - 1)) {
							pass = false;
							break;
						}
					}

					if (!pass) {
						break;
					}
				}
				if (pass) {
					results.emplace_back(it);
				}
			}
			return results;
		}

		// Query for all entities containing the matching component IDs
		std::vector<int> queryAllEntities(const std::unordered_set<int>& query, bool alwaysQuery = false, bool cache = false) {
			// If an existing archetype already exists that matches the query and alwaysQuery is false, just return its entityList.
			if (!alwaysQuery) {
				// Check for equality between the query and all archetypes, returning the entityList if there is a match
				for (int i = 0; i < archetypes.size(); i++) {
					DEBUG("Searching for matching archetype...");
					if (archetypes[i] == query) {
						DEBUG("Matching archetype found!");
						std::vector<int> ret;
						ret.reserve(archetypeLists[i].size());

						for (auto it : archetypeLists[i]) {
							ret.push_back(it);
						}
						return ret;
					}
				}
			}
			DEBUG("No matching archetype found. Running query...");

			// Vector to store matching entity indices
			std::vector<int> results;

			// Loop over all compoonent sets
			for (int i = 0; i < componentSets.size(); i++) {
				// Skip any entries with INVALID_ID, which are dead entities
				if (componentSets[i].empty() || entityVector[i] == INVALID_ID) {
					continue;
				}

				// Stop iterating and move to the next query parameter if we find a match
				bool pass = true;
				for (uint32_t q : query) {
					if (!pass) {
						break;
					}

					for (int j = 0; j < componentSets[i].size(); j++) {
						if (componentSets[i][j] == q) {
							break;
						}
						// This first argument assumes the components are sorted numerically from lowest to highest.
						// It's only worth sorting if queries are more common than entity adds, which seems unlikely
						// In that case, we would just remove the '> q' inequality.
						else if ((componentSets[i][j] > q) or (j == componentSets[i].size() - 1)) {
							pass = false;
							break;
						}
					}

					if (!pass) {
						break;
					}
				}
				if (pass) {
					results.emplace_back(i);
				}
			}

			// Create an archetype for this query if the 'cache' flag is set
			if (cache) {
				createArchetype(query, &results);
				DEBUG("New query cached!");
			}

			return results;
		}

		// Destroy and entity by setting it's value to -1 and marks all of its components as dead
		void destroyEntity(int entity, ComponentRegistry* compRegistry) {
			// Ensure entity is valid
			if (entity >= entityVector.size()) {
				MDEBUG("Attempted to destroy an invalid entity at index " << entity);
				return;
			}

			// Iterate over and remove all entity components. Skip this step if the ID is laready invalid.
			if (entityVector[entity] != INVALID_ID) {
				for (int i = 0; i < componentSets[entity].size(); i++) {
					removeComponent(entity, componentSets[entity][i], compRegistry);
				}

				// Set entity to -1, indicating it is dead.
				entityVector[entity] = INVALID_ID;
			}

		}

		// Deletes a component from an entity's componentSet and Map. Also marks the component as dead in the component registry
		void removeComponent(int entity, int component_id, ComponentRegistry* compRegistry, bool destroyed = false) {
			// If the component is in the map, remove it
			auto find_comp = componentIndexMap[entity].find(component_id);
			if (find_comp != componentIndexMap[entity].end()) {
				// Mark as dead in the Component Registry pool
				compRegistry->getComponentMemoryPool(component_id)->deallocate(componentIndexMap[entity][component_id]);

				// Erase from the map
				componentIndexMap[entity].erase(find_comp);
			}//

			// Copy the set vector into an unordered set before removing it.
			std::unordered_set<int> oldSet;
			for (auto it : componentSets[entity]) {
				oldSet.insert(it);
			}

			// If the component is in the set, remove it
			for (int i = 0; i < componentSets[entity].size(); i++) {
				if (componentSets[entity][i] == component_id) {
					componentSets[entity][i] = INVALID_ID;
					MDEBUG("Component with ID " << component_id << " removed from entity " << entity);
					break;
				}
			}

			// Copy the vector again if we're not destroying. We know this set is empty if we're deleting the entity though.
			std::unordered_set<int> newSet;
			if (!destroyed) {
				for (auto it : componentSets[entity]) {
					newSet.insert(it);
				}
			}

			// Update the archetypeLists
			updateArchetypeLists(entity, oldSet, newSet);
		}

		// Create an archetype, which is simply an int ID associated with a set of component IDs. This should only be called the first a specific type of query is made.
		std::vector<int> createArchetype(const std::unordered_set<int>& archetype, std::vector<int>* entities = nullptr) {
			// Add the unordered set to the archetypes array
			archetypes.emplace_back(archetype);
			size_t archetypeID = archetypes.size() - 1;

			// Populate the archetype entity array with any existing matches and allocate some memory
			std::vector<int> existingEntities;
			if (entities) {
				existingEntities = *entities;
			}
			else {
				existingEntities = queryAllEntities(archetype, true, false);
			}

			// Convert the entity list to a set and add it to archetypes
			std::unordered_set<int> existingEntitiesSet;
			existingEntitiesSet.reserve(existingEntities.size());

			for (auto it : existingEntities) {
				existingEntitiesSet.insert(it);
			}

			archetypeLists.emplace_back(existingEntitiesSet);
			allocateArchetypeBlocks(archetypeID);

			// Return archetype ID
			return existingEntities;

		}

		// Iterate over archetype lists and add/remove the given entity as needed
		void updateArchetypeLists(int entity, std::unordered_set<int> oldSet, std::unordered_set<int> newSet) {
			for (int i = 0; i < archetypes.size(); i++) {
				auto& archetypeSet = archetypes[i];
				if (isSubsetOf(oldSet, archetypeSet)) {
					archetypeLists[i].erase(entity);
				}
				if (isSubsetOf(newSet, archetypeSet)) {
					archetypeLists[i].insert(entity);
				}
			}
		}

	private:
		// Maximum number of components per entity, and the size of the preallocated unordered_sets
		size_t setSize_;

		// Number of entities to preallocate.
		size_t entityBlock_;

		// Number of entries in each archetype vector to reserve per allocation
		size_t archetypeBlock_;

		// Current free index in the entityVecotr
		int entityVectorIndex = 0;

		// Vector of registered entities, where the array index is also the entity ID. Initialized to 1024, but expandable.
		std::vector<uint32_t> entityVector;

		// A vector of preallocated sets. Each set's index matches the entity index in entityVector
		// TODO: This is a vector of vectors, and these aren't actually stored contiguously in memory, even with predefined size. If this becomes a PERFORMANCE issue, we use the method described in the octree
		// TODO: That is, we  define a single large vector and the 'blocks' are just ranges of indices within that vector.
		std::vector<std::vector<uint32_t>> componentSets;

		// A vector of unordered_map, where each index corresponds with an entity index. The keys are component IDs and the values are tha component's index
		std::vector<std::unordered_map<int, int>> componentIndexMap;

		// Vector of archetype*, where each archetype's index is it's ID. Preallocation is not needed as this is only expanded when systems are registered during intialization.
		std::vector<std::unordered_set<int>> archetypes;

		// Vector matching the archetypes vector, this one containing the associated list of entities for each.
		std::vector<std::unordered_set<int>> archetypeLists;

		// Allocates blocks of memory in the componentSets vector
		void allocateComponentBlocks() {
			for (int i = 0; i < entityBlock_; i++) {
				std::vector<uint32_t> tempSet;
				tempSet.reserve(setSize_);
				std::fill(tempSet.begin(), tempSet.end(), INVALID_ID);
				componentSets.emplace_back(tempSet);
			}
		}

		// Reserves memory in an archetype vector
		void allocateArchetypeBlocks(int archetype) {
			if (archetype != INVALID_ID) {
				archetypeLists[archetype].reserve(archetypeLists[archetype].size() + archetypeBlock_);
			}
		}

		// Checks if a set is a subset of another. Generally more efficient to check the smaller set against the larger.
		bool isSubsetOf(std::unordered_set<int> a, std::unordered_set<int> b)
		{
			// If the sets are the same size, use a simple equality
			if (a.size() == b.size()) { return a == b; }

			// If a < b return false
			if (a.size() < b.size()) { return false; }

			// Otherwise iterate through the sets and check
			auto const not_found = b.end();
			for (auto const& element : a)
				if (b.find(element) == not_found)
					return false;

			return true;
		}
	};

	// Thread pool
	class ThreadPool {
	public:
		// Default available threads to hardware_concurency
		ThreadPool(ComponentRegistry& compReg, EntityRegistry& entReg, size_t numThreads = std::thread::hardware_concurrency()) : stop(false), entityRegistry(entReg), componentRegistry(compReg) {
			size_t systemThreads = numThreads;

			// We want to leave 2 threads for the renderer and the asset manager if possible
			if (numThreads > 2) {
				systemThreads = numThreads - 2;
			}

			// Create the lambda function that runs in the thread and checks the queue for work
			for (size_t i = 0; i < systemThreads; ++i) {
				workers.emplace_back([this, &compReg, &entReg] {
					while (true) {
						std::function<void()> task;
						{
							std::unique_lock<std::mutex> lock(this->queueMutex);
							this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
							if (this->stop && this->tasks.empty())
								return;
							task = std::move(this->tasks.front());
							this->tasks.pop();
						}
						task();
					}
					});
			}
		}

		// Accepts any callable and enqueues it for use by a thread. This will only ever be system lambdas
		template<typename Func>
		std::future<void> enqueue(Func&& func) {
			// Get a future so we know when the system has completed execution
			auto task = std::make_shared<std::packaged_task<void()>>(std::forward<Func>(func));

			std::future<void> result = task->get_future();
			{
				std::unique_lock<std::mutex> lock(queueMutex);
				if (stop)
					throw std::runtime_error("enqueue on stopped ThreadPool");
				tasks.emplace(std::forward<Func>(func));
			}
			condition.notify_one();

			return result;
		}

		// Stop all the threads and wait for a join before exiting
		// TODO: Make sure threads are joinable first. Throw an exception if not.
		~ThreadPool() {
			{
				std::unique_lock<std::mutex> lock(queueMutex);
				stop = true;
			}
			condition.notify_all();
			for (std::thread& worker : workers)
				worker.join();
		}

		int getThreads() {
			return workers.size();
		}

	private:
		// Vector of worker threads
		std::vector<std::thread> workers;

		// System Queue
		std::queue<std::function<void()>> tasks;

		// Mutex for locking the queue when a thread is quarying it/when something is being enqueued/dequeued
		std::mutex queueMutex;

		std::condition_variable condition;

		// When set, tells the threads to stop polling and die
		bool stop;

		// Registries used in the lambdas
		ComponentRegistry& componentRegistry;
		EntityRegistry& entityRegistry;
	};

	// Class for registering and orchestrating systems
	class SystemRegistry {
	public:
		// Register a system as a lambda function. Must specify the component and entity registries the system uses.
		int registerSystem(std::function<void()> system, std::string systemName = "") {

			// Add the system to the systems array
			systems.push_back(system);

			// Append to end of execution order
			executionOrder.push_back(systems.size() - 1);

			// If no name was provided, use the index as the name instead.
			if (systemName.empty()) {
				systemNames[std::to_string(systems.size() - 1)] = systems.size() - 1;
				return systems.size() - 1;
			}

			// Add the system and name to the systemNames map
			systemNames[systemName] = systems.size() - 1;

			return systems.size() - 1;
		}

		// Returns the system ID for a given string
		int getSystemID(std::string sysName) {
			auto it = systemNames.find(sysName);

			// -1 indicates a dead or invalid system
			if (it == systemNames.end()) {
				return INVALID_ID;
			}

			return it->second;
		}

		// Replace the existing execution order, assuming you have the indices
		void setExecutionOrderByIndex(std::vector<int>& order) {
			// Execution order is only valid if its contents are a permutation of the registered systems
			if (std::is_permutation(order.begin(), order.end(), executionOrder.begin())) {
				executionOrder = order;
				return;
			}
			// TODO: Fail with some type of error reporting
			return;
		}

		// Replace the existing execution order, assuming you have the names
		void setExecutionOrderByName(std::vector<std::string>& names) {
			// Variable to hold the new array
			std::vector<int> newOrder;

			// Create a vector of indices from a vector of names.
			for (auto& it : names) {
				auto systemIndex = systemNames.find(it);
				if (systemIndex == systemNames.end()) {
					// Couldn't find a name so we do not replace anything. Error Reporting.
					return;
				}
				newOrder.emplace_back(systemIndex->second);
			}

			// Set the new order
			setExecutionOrderByIndex(newOrder);
			return;
		}

		// Executes the specified system
		void executeSystem(int sysID) {
			systems[sysID]();
		}

		// Executes systems in order
		void executeAllSystems() {
			// Iterate over the executionOrder array and execute systems in order
			for (int i = 0; i < executionOrder.size(); i++) {
				int sysID = executionOrder[i];
				executeSystem(sysID);
			}
		}


	private:
		// Array of system lambdas
		std::vector<std::function<void()>> systems;

		// Array of system indices. This is used to define execution order of the systems
		std::vector<int> executionOrder;

		// Maps system name to its numeric ID
		std::unordered_map<std::string, int> systemNames;
	};
}
