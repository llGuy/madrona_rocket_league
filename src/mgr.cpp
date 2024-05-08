#include "mgr.hpp"
#include "sim.hpp"
#include "import.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/physics_loader.hpp>
#include <madrona/tracing.hpp>
#include <madrona/mw_cpu.hpp>
#include <madrona/render/api.hpp>
#include <madrona/physics_assets.hpp>

#include <array>
#include <charconv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

#define MADRONA_VIEWER

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;
using namespace madrona::render;
using namespace madrona::py;

namespace madEscape {

struct RenderGPUState {
    render::APILibHandle apiLib;
    render::APIManager apiMgr;
    render::GPUHandle gpu;
};


static inline Optional<RenderGPUState> initRenderGPUState(
    const Manager::Config &mgr_cfg)
{
#if defined(MADRONA_VIEWER)
    if (mgr_cfg.extRenderDev || !mgr_cfg.enableBatchRenderer) {
        return Optional<RenderGPUState>::none();
    }
#endif

    auto render_api_lib = render::APIManager::loadDefaultLib();
    render::APIManager render_api_mgr(render_api_lib.lib());
    render::GPUHandle gpu = render_api_mgr.initGPU(mgr_cfg.gpuID);

    return RenderGPUState {
        .apiLib = std::move(render_api_lib),
        .apiMgr = std::move(render_api_mgr),
        .gpu = std::move(gpu),
    };
}

static inline Optional<render::RenderManager> initRenderManager(
    const Manager::Config &mgr_cfg,
    const Optional<RenderGPUState> &render_gpu_state)
{
#if defined(MADRONA_VIEWER)
    if (!mgr_cfg.extRenderDev && !mgr_cfg.enableBatchRenderer) {
        return Optional<render::RenderManager>::none();
    }
#endif

    render::APIBackend *render_api;
    render::GPUDevice *render_dev;

    if (render_gpu_state.has_value()) {
        render_api = render_gpu_state->apiMgr.backend();
        render_dev = render_gpu_state->gpu.device();
    } else {
        render_api = mgr_cfg.extRenderAPI;
        render_dev = mgr_cfg.extRenderDev;
    }

    return render::RenderManager(render_api, render_dev, {
        .enableBatchRenderer = mgr_cfg.enableBatchRenderer,
        .agentViewWidth = mgr_cfg.batchRenderViewWidth,
        .agentViewHeight = mgr_cfg.batchRenderViewHeight,
        .numWorlds = mgr_cfg.numWorlds,
        .maxViewsPerWorld = consts::numAgents,
        .maxInstancesPerWorld = 1024,
        .execMode = mgr_cfg.execMode,
        .voxelCfg = {},
    });
}

struct Manager::Impl {
    Config cfg;
    PhysicsLoader physicsLoader;
    WorldReset *worldResetBuffer;
    Action *agentActionsBuffer;
    Optional<RenderGPUState> renderGPUState;
    Optional<render::RenderManager> renderMgr;
    uint32_t raycastOutputResolution;

    inline Impl(const Manager::Config &mgr_cfg,
                PhysicsLoader &&phys_loader,
                WorldReset *reset_buffer,
                Action *action_buffer,
                Optional<RenderGPUState> &&render_gpu_state,
                Optional<render::RenderManager> &&render_mgr,
                uint32_t raycast_output_resolution)
        : cfg(mgr_cfg),
          physicsLoader(std::move(phys_loader)),
          worldResetBuffer(reset_buffer),
          agentActionsBuffer(action_buffer),
          renderGPUState(std::move(render_gpu_state)),
          renderMgr(std::move(render_mgr)),
          raycastOutputResolution(raycast_output_resolution)
    {}

    inline virtual ~Impl() {}

    virtual void run() = 0;

    virtual Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dimensions) const = 0;

    static inline Impl * init(const Config &cfg);
};

struct Manager::CPUImpl final : Manager::Impl {
    using TaskGraphT =
        TaskGraphExecutor<Engine, Sim, Sim::Config, Sim::WorldInit>;

    TaskGraphT cpuExec;

    inline CPUImpl(const Manager::Config &mgr_cfg,
                   PhysicsLoader &&phys_loader,
                   WorldReset *reset_buffer,
                   Action *action_buffer,
                   Optional<RenderGPUState> &&render_gpu_state,
                   Optional<render::RenderManager> &&render_mgr,
                   TaskGraphT &&cpu_exec)
        : Impl(mgr_cfg, std::move(phys_loader),
               reset_buffer, action_buffer,
               std::move(render_gpu_state), std::move(render_mgr),
               mgr_cfg.raycastOutputResolution),
          cpuExec(std::move(cpu_exec))
    {}

    inline virtual ~CPUImpl() final {}

    inline virtual void run()
    {
        cpuExec.run();
    }

    virtual inline Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dims) const final
    {
        void *dev_ptr = cpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, Optional<int>::none());
    }
};

#ifdef MADRONA_CUDA_SUPPORT
struct Manager::CUDAImpl final : Manager::Impl {
    MWCudaExecutor gpuExec;
    MWCudaLaunchGraph stepGraph;

    inline CUDAImpl(const Manager::Config &mgr_cfg,
                   PhysicsLoader &&phys_loader,
                   WorldReset *reset_buffer,
                   Action *action_buffer,
                   Optional<RenderGPUState> &&render_gpu_state,
                   Optional<render::RenderManager> &&render_mgr,
                   MWCudaExecutor &&gpu_exec,
                   MWCudaLaunchGraph &&step_graph)
        : Impl(mgr_cfg, std::move(phys_loader),
               reset_buffer, action_buffer,
               std::move(render_gpu_state), std::move(render_mgr),
               mgr_cfg.raycastOutputResolution),
          gpuExec(std::move(gpu_exec)),
          stepGraph(std::move(step_graph))
    {}

    inline virtual ~CUDAImpl() final {}

    inline virtual void run()
    {
        gpuExec.run(stepGraph);
    }

    virtual inline Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dims) const final
    {
        void *dev_ptr = gpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, cfg.gpuID);
    }
};
#endif

// #define LOAD_ENV 2

static imp::ImportedAssets loadRenderObjects(
        Optional<render::RenderManager> &render_mgr,
        std::vector<ImportedInstance> &imported_instances,
        math::Vector2 *scene_center,
        bool merge_all)
{
    //(void)bvh;
    std::vector<render::MeshBVH::Node> nodes;
    std::vector<render::MeshBVH::LeafGeometry> leafGeos;
    std::vector<render::MeshBVH::LeafMaterial> leafMats;
    std::vector<Vector3> vertices;
    std::vector<render::MeshBVH> bvhs;

    const char *loaded_env = getenv("MADRONA_LOADED_ENV");

    assert(loaded_env != nullptr);

    std::string scene_path;

    if (loaded_env[0] == '0') {
        *scene_center = { -59.872063, 36.738739 };
        // Get the render objects needed from the habitat JSON
        scene_path = std::filesystem::path(DATA_DIR) /
            "hssd-hab/scenes-uncluttered/108736656_177263304.scene_instance.json";
    } else if (loaded_env[0] == '1') {
        *scene_center = { -8.241938, 36.422760 };
        scene_path = std::filesystem::path(DATA_DIR) /
            "hssd-hab/scenes-uncluttered/105515286_173104287.scene_instance.json";
    } else if (loaded_env[0] == '2') {
        *scene_center = { -17.695925, 5.110266 };
        scene_path = std::filesystem::path(DATA_DIR) /
            "hssd-hab/scenes-uncluttered/107734254_176000121.scene_instance.json";
    }

    auto loaded_scene = HabitatJSON::habitatJSONLoad(scene_path);

    std::vector<std::string> render_asset_paths;
    render_asset_paths.resize((size_t)SimObjectDefault::NumObjects);

    render_asset_paths[(size_t)SimObjectDefault::Cube] =
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();
    render_asset_paths[(size_t)SimObjectDefault::Wall] =
        (std::filesystem::path(DATA_DIR) / "wall_render.obj").string();
    render_asset_paths[(size_t)SimObjectDefault::Door] =
        (std::filesystem::path(DATA_DIR) / "wall_render.obj").string();
    render_asset_paths[(size_t)SimObjectDefault::Agent] =
        (std::filesystem::path(DATA_DIR) / "agent_render.obj").string();
    render_asset_paths[(size_t)SimObjectDefault::Button] =
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();
    render_asset_paths[(size_t)SimObjectDefault::Plane] =
        (std::filesystem::path(DATA_DIR) / "plane.obj").string();
    // render_asset_paths[(size_t)SimObjectDefault::Dust2] =
        // (std::filesystem::path(DATA_DIR) / "funky2.obj").string();
    render_asset_paths[(size_t)SimObjectDefault::Dust2] =
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();

    // All models in the habitat thing use the same material for now
    uint32_t habitat_material = 0;

    float height_offset = 0.f;

    float scale = 10.f;

    // All the assets from the habitat JSON scene have object IDs which start at
    // SimObjectDefault::NumObjects
    {
        imported_instances.push_back({
            .position = Quat::angleAxis(pi_d2, { 1.f, 0.f, 0.f }).
                        rotateVec({ 0.f, 0.f, 0.f + height_offset }) * scale,
            .rotation = Quat::angleAxis(pi_d2,{ 1.f, 0.f, 0.f }) *
                        Quat::angleAxis(0.f, math::up),
            .scale = { scale, scale, scale },
            .objectID = (int32_t)render_asset_paths.size(),
        });

        render_asset_paths.push_back(loaded_scene.stagePath.string());

        std::unordered_map<std::string, uint32_t> loaded_gltfs;
        std::unordered_map<uint32_t, uint32_t> object_to_imported_instance;

        for (const HabitatJSON::AdditionalInstance &inst :
                loaded_scene.additionalInstances) {
            auto [iter, insert_success] = loaded_gltfs.emplace(inst.gltfPath.string(), 
                    render_asset_paths.size());
            if (insert_success) {
                auto pos = Quat::angleAxis(pi_d2, { 1.f, 0.f, 0.f }).
                           rotateVec(Vector3{ inst.pos[0], inst.pos[1], 
                                              inst.pos[2] + height_offset });

                auto scale_vec = madrona::math::Diag3x3 {
                    inst.scale[0] * scale,
                    inst.scale[1] * scale,
                    inst.scale[2] * scale
                };
                
                ImportedInstance new_inst = {
                    .position = {pos.x * scale, pos.y * scale, pos.z * scale},
                    .rotation = Quat::angleAxis(pi_d2, { 1.f, 0.f, 0.f }) * 
                                Quat{ inst.rotation[0], inst.rotation[1],
                                      inst.rotation[2], inst.rotation[3] },
                    .scale = scale_vec,
                    .objectID = (int32_t)render_asset_paths.size(),
                };

                imported_instances.push_back(new_inst);
                render_asset_paths.push_back(inst.gltfPath.string());
            } else {
                // Push the instance to the instances array
                auto pos = Quat::angleAxis(pi_d2, { 1.f, 0.f, 0.f }).
                           rotateVec(Vector3{ inst.pos[0], inst.pos[1], 
                                              inst.pos[2] + height_offset });

                auto scale_vec = madrona::math::Diag3x3 {
                    inst.scale[0] * scale,
                    inst.scale[1] * scale,
                    inst.scale[2] * scale
                };

                ImportedInstance new_inst = {
                    .position = {pos.x * scale,pos.y * scale,pos.z * scale},
                    .rotation = Quat::angleAxis(pi_d2,{ 1.f, 0.f, 0.f }) *
                                Quat{ inst.rotation[0], inst.rotation[1],
                                      inst.rotation[2], inst.rotation[3] },
                    .scale = scale_vec,
                    .objectID = (int32_t)iter->second,
                };

                imported_instances.push_back(new_inst);
            }
        }

        printf("Loaded %d render objects\n", (int)loaded_gltfs.size());
    }

    for (size_t i = 0; i < render_asset_paths.size(); ++i) {
        printf("%s\n", render_asset_paths[i].c_str());
    }

    // std::array<const char *, (size_t)SimObjectDefault::NumObjects> render_asset_cstrs;
    std::vector<const char *> render_asset_cstrs;
    for (size_t i = 0; i < render_asset_paths.size()-6; i++) {
        render_asset_cstrs.push_back(render_asset_paths[i].c_str());
    }

    std::array<char, 1024> import_err;
    auto render_assets = imp::ImportedAssets::importFromDisk(
        render_asset_cstrs, Span<char>(import_err.data(), import_err.size()),
        true, true);

    if (!render_assets.has_value()) {
        FATAL("Failed to load render assets: %s", import_err);
    }

    auto materials = std::to_array<imp::SourceMaterial>({
        { render::rgb8ToFloat(191, 108, 10), -1, 0.8f, 0.2f },
        { math::Vector4{0.4f, 0.4f, 0.4f, 0.0f}, -1, 0.8f, 0.2f,},
        { math::Vector4{1.f, 1.f, 1.f, 0.0f}, 1, 0.5f, 1.0f,},
        { render::rgb8ToFloat(230, 230, 230),   -1, 0.8f, 1.0f },
        { math::Vector4{0.5f, 0.3f, 0.3f, 0.0f},  0, 0.8f, 0.2f,},
        { render::rgb8ToFloat(230, 20, 20),   -1, 0.8f, 1.0f },
        { render::rgb8ToFloat(230, 230, 20),   -1, 0.8f, 1.0f },
        { render::rgb8ToFloat(230, 230, 230),   -1, 0.8f, 1.0f },
    });

    habitat_material = 7;

    // Override materials
    render_assets->objects[(CountT)SimObjectDefault::Cube].meshes[0].materialIDX = 0;
    render_assets->objects[(CountT)SimObjectDefault::Wall].meshes[0].materialIDX = 1;
    render_assets->objects[(CountT)SimObjectDefault::Door].meshes[0].materialIDX = 5;
    render_assets->objects[(CountT)SimObjectDefault::Agent].meshes[0].materialIDX = 2;
    render_assets->objects[(CountT)SimObjectDefault::Agent].meshes[1].materialIDX = 3;
    render_assets->objects[(CountT)SimObjectDefault::Agent].meshes[2].materialIDX = 3;
    render_assets->objects[(CountT)SimObjectDefault::Button].meshes[0].materialIDX = 6;
    render_assets->objects[(CountT)SimObjectDefault::Plane].meshes[0].materialIDX = 4;
    for(int i =0; 
            i<render_assets->objects[(CountT)SimObjectDefault::Dust2].meshes.size();
            i++) {
        render_assets->objects[(CountT) SimObjectDefault::Dust2].meshes[i].materialIDX = 0;
    }

    if (merge_all) {
        //Count up total memory requirements
        size_t vertex_size = 0;
        size_t indices_size = 0;

        printf("Render asset size %d\n",render_assets->objects.size());
        for(auto& inst : imported_instances){
            if(inst.objectID >= render_assets->objects.size())
                continue;
            auto& render_asset = render_assets->objects[inst.objectID];
            for (int mesh_i = 0; mesh_i < render_asset.meshes.size(); ++mesh_i) {
                vertex_size += render_asset.meshes[mesh_i].numVertices;
                indices_size += render_asset.meshes[mesh_i].numFaces * 3;
            }
        }

        fprintf(stderr,"Pre alloc %lu,%lu\n",indices_size,vertex_size);
        DynArray<madrona::imp::SourceMesh> dummy_vector(1);
        DynArray<Vector3> positions(vertex_size);
        DynArray<Vector3> normals(vertex_size);
        DynArray<Vector4> tangentsSigns(vertex_size);
        DynArray<Vector2> uvs(vertex_size);
        DynArray<uint32_t> indices(indices_size);

        uint32_t v_index = 0;
        uint32_t i_index = 0;
        for(ImportedInstance& inst : imported_instances) {
            if(inst.objectID >= render_assets->objects.size())
                continue;
            auto& render_asset = render_assets->objects[inst.objectID];
            for (int mesh_i = 0; mesh_i < render_asset.meshes.size(); ++mesh_i) {
                madrona::imp::SourceMesh& mesh = render_asset.meshes[mesh_i];

                for(size_t i = 0; i < mesh.numFaces; i++){
                    indices.push_back(mesh.indices[i*3] + v_index);
                    indices.push_back(mesh.indices[i*3 + 1] + v_index);
                    indices.push_back(mesh.indices[i*3 + 2] + v_index);
                    i_index += 3;
                }

                for(size_t i = 0; i < mesh.numVertices; i++){
                    positions.push_back(inst.rotation.rotateVec(inst.scale * mesh.positions[i]) + inst.position);
                    normals.push_back(inst.rotation.rotateVec((inst.scale * mesh.normals[i]).normalize()).normalize());
                    if(mesh.tangentAndSigns) {
                        tangentsSigns.push_back(mesh.tangentAndSigns[i]);
                    }
                    if(mesh.uvs) {
                        uvs.push_back(mesh.uvs[i]);
                    }else{
                        uvs.push_back(Vector2{0,0});
                    }

                    v_index++;
                }
            }
        }
        printf("Done with merge\n");

        madrona::imp::SourceMesh merged_mesh{
            .positions = positions.data(),
                .normals = normals.data(),
                .tangentAndSigns = nullptr,
                .uvs = uvs.data(),
                .indices = indices.data(),
                .faceCounts = nullptr,
                .faceMaterials = nullptr,
                .numVertices = (uint32_t)vertex_size,
                .numFaces = (uint32_t)indices_size / 3,
                .materialIDX = habitat_material,
        };

        dummy_vector.push_back(merged_mesh);

        madrona::imp::SourceObject object{
            madrona::Span<madrona::imp::SourceMesh>(dummy_vector.data(),1)
        };
        printf("Pre remove objects %p,%p,%p,%p,%p,%p\n",positions.data(),normals.data(),tangentsSigns.data(),indices.data(),
                uvs.data(),dummy_vector.data());
        //Remove the unmerged originals

        uint32_t sub = (loaded_env[0] == '2') ? 9 : 6;

        for (int obj_i = (int)SimObjectDefault::NumObjects; obj_i < render_asset_paths.size()-sub; ++obj_i) {  //Use this for env 0 and 1
                                                                                                               //for (int obj_i = (int)SimObjectDefault::NumObjects; obj_i < render_asset_paths.size()-9; ++obj_i) {  //Use this for env 2
            render_assets->objects.pop_back();
        }
        printf("Past remove objects %d\n",uvs.size());
        render_assets->objects.push_back(object);
        render_assets->geoData.positionArrays.emplace_back(std::move(positions));
        render_assets->geoData.normalArrays.emplace_back(std::move(normals));
        render_assets->geoData.uvArrays.emplace_back(std::move(uvs));
        //render_assets->geoData.tangentAndSignArrays.emplace_back(std::move(tangentsSigns));
        render_assets->geoData.indexArrays.emplace_back(std::move(indices));
        render_assets->geoData.meshArrays.emplace_back(std::move(dummy_vector));
        }

    for (int obj_i = (int)SimObjectDefault::NumObjects;
            obj_i < render_assets->objects.size(); ++obj_i) {
        auto *obj_data = &render_assets->objects[obj_i];

        for (int mesh_i = 0; mesh_i < obj_data->meshes.size(); ++mesh_i) {
            obj_data->meshes[mesh_i].materialIDX = habitat_material;
        }
    }

    if (render_mgr.has_value()) {
        render_mgr->loadObjects(render_assets->objects, materials, {
            { (std::filesystem::path(DATA_DIR) /
               "green_grid.png").string().c_str() },
            { (std::filesystem::path(DATA_DIR) /
               "smile.png").string().c_str() },
        });

        render_mgr->configureLighting({
            { true, math::Vector3{1.0f, -1.0f, -0.05f}, math::Vector3{1.0f, 1.0f, 1.0f} }
        });
    }

    return std::move(*render_assets);
}

static void loadPhysicsObjects(PhysicsLoader &loader)
{
    std::array<std::string, (size_t)SimObjectDefault::NumObjects - 1> asset_paths;
    asset_paths[(size_t)SimObjectDefault::Cube] =
        (std::filesystem::path(DATA_DIR) / "cube_collision.obj").string();
    asset_paths[(size_t)SimObjectDefault::Wall] =
        (std::filesystem::path(DATA_DIR) / "wall_collision.obj").string();
    asset_paths[(size_t)SimObjectDefault::Door] =
        (std::filesystem::path(DATA_DIR) / "wall_collision.obj").string();
    asset_paths[(size_t)SimObjectDefault::Agent] =
        (std::filesystem::path(DATA_DIR) / "agent_collision_simplified.obj").string();
    asset_paths[(size_t)SimObjectDefault::Button] =
        (std::filesystem::path(DATA_DIR) / "cube_collision.obj").string();
    asset_paths[(size_t)SimObjectDefault::Dust2] =
       (std::filesystem::path(DATA_DIR) / "cube_collision.obj").string();

    std::array<const char *, (size_t)SimObjectDefault::NumObjects - 1> asset_cstrs;
    for (size_t i = 0; i < asset_paths.size(); i++) {
        asset_cstrs[i] = asset_paths[i].c_str();
    }

    char import_err_buffer[4096];
    auto imported_src_hulls = imp::ImportedAssets::importFromDisk(
        asset_cstrs, import_err_buffer, true);

    if (!imported_src_hulls.has_value()) {
        FATAL("%s", import_err_buffer);
    }

    DynArray<imp::SourceMesh> src_convex_hulls(
        imported_src_hulls->objects.size());

    DynArray<DynArray<SourceCollisionPrimitive>> prim_arrays(0);
    HeapArray<SourceCollisionObject> src_objs(
        (CountT)SimObjectDefault::NumObjects);

    auto setupHull = [&](SimObjectDefault obj_id,
                         float inv_mass,
                         RigidBodyFrictionData friction) {
        auto meshes = imported_src_hulls->objects[(CountT)obj_id].meshes;
        DynArray<SourceCollisionPrimitive> prims(meshes.size());

        for (const imp::SourceMesh &mesh : meshes) {
            src_convex_hulls.push_back(mesh);
            prims.push_back({
                .type = CollisionPrimitive::Type::Hull,
                .hullInput = {
                    .hullIDX = uint32_t(src_convex_hulls.size() - 1),
                },
            });
        }

        prim_arrays.emplace_back(std::move(prims));

        src_objs[(CountT)obj_id] = SourceCollisionObject {
            .prims = Span<const SourceCollisionPrimitive>(prim_arrays.back()),
            .invMass = inv_mass,
            .friction = friction,
        };
    };

    setupHull(SimObjectDefault::Cube, 0.075f, {
        .muS = 0.5f,
        .muD = 0.75f,
    });

    setupHull(SimObjectDefault::Wall, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });

    setupHull(SimObjectDefault::Door, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });

    setupHull(SimObjectDefault::Agent, 1.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });

    setupHull(SimObjectDefault::Button, 1.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });

    setupHull(SimObjectDefault::Dust2, 1.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });

    SourceCollisionPrimitive plane_prim {
        .type = CollisionPrimitive::Type::Plane,
    };

    src_objs[(CountT)SimObjectDefault::Plane] = {
        .prims = Span<const SourceCollisionPrimitive>(&plane_prim, 1),
        .invMass = 0.f,
        .friction = {
            .muS = 0.5f,
            .muD = 0.5f,
        },
    };

    StackAlloc tmp_alloc;
    RigidBodyAssets rigid_body_assets;
    CountT num_rigid_body_data_bytes;
    void *rigid_body_data = RigidBodyAssets::processRigidBodyAssets(
        src_convex_hulls,
        src_objs,
        false,
        tmp_alloc,
        &rigid_body_assets,
        &num_rigid_body_data_bytes);

    if (rigid_body_data == nullptr) {
        FATAL("Invalid collision hull input");
    }

    // This is a bit hacky, but in order to make sure the agents
    // remain controllable by the policy, they are only allowed to
    // rotate around the Z axis (infinite inertia in x & y axes)
    rigid_body_assets.metadatas[
        (CountT)SimObjectDefault::Agent].mass.invInertiaTensor.x = 0.f;
    rigid_body_assets.metadatas[
        (CountT)SimObjectDefault::Agent].mass.invInertiaTensor.y = 0.f;

    loader.loadRigidBodies(rigid_body_assets);
    free(rigid_body_data);
}

Manager::Impl * Manager::Impl::init(
    const Manager::Config &mgr_cfg)
{
    Sim::Config sim_cfg;
    sim_cfg.autoReset = mgr_cfg.autoReset;
    sim_cfg.initRandKey = rand::initKey(mgr_cfg.randSeed);

#if 0
    std::array<std::string,1> collision_asset_paths;
    collision_asset_paths[0] =
        (std::filesystem::path(DATA_DIR) / "funky2.obj").string();
    std::array<const char *, 1> collision_asset_cstrs;
    for (size_t i = 0; i < collision_asset_paths.size(); i++) {
        collision_asset_cstrs[i] = collision_asset_paths[i].c_str();
    }
    std::array<char, 1024> import_err;
    auto collision_assets = imp::ImportedAssets::importFromDisk(
        collision_asset_cstrs, Span<char>(import_err.data(), import_err.size()));

    if (!collision_assets.has_value()) {
        FATAL("Failed to load collision meshes: %s", import_err);
    }

    auto* bvh = (MeshBVH*)malloc(sizeof(MeshBVH));
    StackAlloc tmp_alloc;
    CountT numBytes;
    MeshBVHBuilder::build(collision_assets->objects[0].meshes,
                             tmp_alloc,
                             bvh,
                             &numBytes);
#endif

    switch (mgr_cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        printf("Cuda init:\n");
        CUcontext cu_ctx = MWCudaExecutor::initCUDA(mgr_cfg.gpuID);
        printf("post Cuda init:\n");
        PhysicsLoader phys_loader(ExecMode::CUDA, 10);
        loadPhysicsObjects(phys_loader);

        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();
        //sim_cfg.rigidBodyObjMgr = phys_obj_mgr;

        Optional<RenderGPUState> render_gpu_state =
            initRenderGPUState(mgr_cfg);

        Optional<render::RenderManager> render_mgr =
            initRenderManager(mgr_cfg, render_gpu_state);

        imp::ImportedAssets::GPUGeometryData *gpu_imported_assets_ptr = nullptr;

        imp::ImportedAssets::GPUGeometryData gpu_imported_assets;
        std::vector<ImportedInstance> imported_instances;

        const char *merge_all_env = getenv("MADRONA_MERGE_ALL");
        assert(merge_all_env);

        bool merge_all = (merge_all_env[0] == '1');
        sim_cfg.mergeAll = merge_all;

        math::Vector2 scene_center;
        auto imported_assets = loadRenderObjects(
                render_mgr, imported_instances, &scene_center, merge_all);

        auto gpu_imported_assets_opt =
            imp::ImportedAssets::makeGPUData(imported_assets);

        assert(gpu_imported_assets_opt.has_value());

        gpu_imported_assets = std::move(*gpu_imported_assets_opt);

        gpu_imported_assets_ptr = &gpu_imported_assets;

        sim_cfg.importedInstances = (ImportedInstance *)cu::allocGPU(
                sizeof(ImportedInstance) * imported_instances.size());
        sim_cfg.numImportedInstances = imported_instances.size();
        sim_cfg.numObjects = gpu_imported_assets.numBVHs;

        sim_cfg.sceneCenter = scene_center;

        REQ_CUDA(cudaMemcpy(sim_cfg.importedInstances, imported_instances.data(),
                            sizeof(ImportedInstance) * imported_instances.size(),
                            cudaMemcpyHostToDevice));

        if (render_mgr.has_value()) {
            sim_cfg.renderBridge = render_mgr->bridge();
        } else {
            sim_cfg.renderBridge = nullptr;
        }

        HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);

        uint32_t raycast_output_resolution = mgr_cfg.raycastOutputResolution;

        // If the rasterizer is enabled, disable the raycaster
        if (mgr_cfg.enableBatchRenderer) {
            raycast_output_resolution = 0;
        }

        printf("Combine compile:\n");
        MWCudaExecutor gpu_exec({
            .worldInitPtr = world_inits.data(),
            .numWorldInitBytes = sizeof(Sim::WorldInit),
            .userConfigPtr = (void *)&sim_cfg,
            .numUserConfigBytes = sizeof(Sim::Config),
            .numWorldDataBytes = sizeof(Sim),
            .worldDataAlignment = alignof(Sim),
            .numWorlds = mgr_cfg.numWorlds,
            .numTaskGraphs = (uint32_t)TaskGraphID::NumTaskGraphs,
            .numExportedBuffers = (uint32_t)ExportID::NumExports, 
            .geometryData = gpu_imported_assets_ptr,
            .raycastOutputResolution = raycast_output_resolution,
        }, {
            { GPU_HIDESEEK_SRC_LIST },
            { GPU_HIDESEEK_COMPILE_FLAGS },
            CompileConfig::OptMode::LTO,
        }, cu_ctx);

        MWCudaLaunchGraph step_graph = gpu_exec.buildLaunchGraph(
                TaskGraphID::Step, !mgr_cfg.enableBatchRenderer);

        printf("Combine postcompile\n");
        WorldReset *world_reset_buffer = 
            (WorldReset *)gpu_exec.getExported((uint32_t)ExportID::Reset);

        Action *agent_actions_buffer = 
            (Action *)gpu_exec.getExported((uint32_t)ExportID::Action);
        fprintf(stderr,"About to exit\n");
        return new CUDAImpl {
            mgr_cfg,
            std::move(phys_loader),
            world_reset_buffer,
            agent_actions_buffer,
            std::move(render_gpu_state),
            std::move(render_mgr),
            std::move(gpu_exec),
            std::move(step_graph)
        };
#else
        FATAL("Madrona was not compiled with CUDA support");
#endif
    } break;
    case ExecMode::CPU: {
#if 0
        PhysicsLoader phys_loader(ExecMode::CPU, 10);
        loadPhysicsObjects(phys_loader);

        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();
        //sim_cfg.rigidBodyObjMgr = phys_obj_mgr;

        Optional<RenderGPUState> render_gpu_state =
            initRenderGPUState(mgr_cfg);

        Optional<render::RenderManager> render_mgr =
            initRenderManager(mgr_cfg, render_gpu_state);

        if (render_mgr.has_value()) {
            std::vector<render::MeshBVH::Node> nodes;
            std::vector<render::MeshBVH::LeafGeometry> leafGeos;
            std::vector<render::MeshBVH::LeafMaterial> leafMats;
            std::vector<Vector3> vertices;
            std::vector<render::MeshBVH> bvhs;

            auto imported_instances = loadRenderObjects(*render_mgr,bvhs,nodes,leafGeos,leafMats,vertices);
            sim_cfg.renderBridge = render_mgr->bridge();

            sim_cfg.importedInstances = (ImportedInstance *)malloc(
                    sizeof(ImportedInstance) * imported_instances.size());
            sim_cfg.numImportedInstances = imported_instances.size();
            memcpy(sim_cfg.importedInstances, imported_instances.data(),
                   sizeof(ImportedInstance) * imported_instances.size());

            auto bvhPtr = (render::MeshBVH*)malloc(bvhs.size()*sizeof(render::MeshBVH));

            auto nodePtr = (render::MeshBVH::Node*)malloc(nodes.size()*sizeof(render::MeshBVH::Node));
            memcpy(nodePtr,nodes.data(),nodes.size()*sizeof(render::MeshBVH::Node));

            auto geoPtr = (render::MeshBVH::LeafGeometry*)malloc(leafGeos.size()*sizeof(render::MeshBVH::LeafGeometry));
            memcpy(geoPtr,leafGeos.data(),leafGeos.size()*sizeof(render::MeshBVH::LeafGeometry));

            auto matPtr = (render::MeshBVH::LeafMaterial*)malloc(leafMats.size()*sizeof(render::MeshBVH::LeafMaterial));
            memcpy(matPtr,leafMats.data(),sizeof(render::MeshBVH::LeafMaterial)*leafMats.size());

            auto vertexPtr = (Vector3*)malloc(vertices.size()*sizeof(Vector3));
            memcpy(vertexPtr,vertices.data(),vertices.size()*sizeof(Vector3));

            //Fix BVH Pointers
            printf("BVHlis %d\n",bvhs.size());
            for(size_t i = 0;i<bvhs.size();i++){
                size_t numLeafs = (size_t)(bvhs[i].leafGeos);
                bvhs[i].nodes = nodePtr + (size_t)(bvhs[i].nodes);
                bvhs[i].leafGeos= geoPtr + numLeafs;
                bvhs[i].leafMats = matPtr + numLeafs;
                bvhs[i].vertices = vertexPtr + (size_t)(bvhs[i].vertices);
            }
            memcpy(bvhPtr,bvhs.data(),sizeof(render::MeshBVH)*bvhs.size());

            for(int i=0;i<bvhs.size();i++){
                float t;
                Vector3 s;
                bvhPtr[i].traceRay({0,0,0},{0,1,0},&t,&s);
                //printf("%x,%x,%x,%x,%x\n",&bvhPtr[4].nodes[i],bvhPtr[4].nodes[i].children[0],bvhPtr[4].nodes[i].children[1],bvhPtr[4].nodes[i].children[2],bvhPtr[4].nodes[i].children[3]);
            }

            sim_cfg.bvhs = (void*)bvhPtr;
        } else {
            sim_cfg.renderBridge = nullptr;
        }

        HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);

        CPUImpl::TaskGraphT cpu_exec {
            ThreadPoolExecutor::Config {
                .numWorlds = mgr_cfg.numWorlds,
                .numExportedBuffers = (uint32_t)ExportID::NumExports,
            },
            sim_cfg,
            world_inits.data(),
        };

        WorldReset *world_reset_buffer = 
            (WorldReset *)cpu_exec.getExported((uint32_t)ExportID::Reset);

        Action *agent_actions_buffer = 
            (Action *)cpu_exec.getExported((uint32_t)ExportID::Action);

        auto cpu_impl = new CPUImpl {
            mgr_cfg,
            std::move(phys_loader),
            world_reset_buffer,
            agent_actions_buffer,
            std::move(render_gpu_state),
            std::move(render_mgr),
            std::move(cpu_exec),
        };

        return cpu_impl;
#endif
        return {};
    } break;
    default: MADRONA_UNREACHABLE();
    }
}

Manager::Manager(const Config &cfg)
    : impl_(Impl::init(cfg))
{
    // Currently, there is no way to populate the initial set of observations
    // without stepping the simulations in order to execute the taskgraph.
    // Therefore, after setup, we step all the simulations with a forced reset
    // that ensures the first real step will have valid observations at the
    // start of a fresh episode in order to compute actions.
    //
    // This will be improved in the future with support for multiple task
    // graphs, allowing a small task graph to be executed after initialization.
    
    for (int32_t i = 0; i < (int32_t)cfg.numWorlds; i++) {
        triggerReset(i);
    }

    step();
}

Manager::~Manager() {}

void Manager::step()
{
    impl_->run();

#if defined(MADRONA_VIEWER)
    if (impl_->renderMgr.has_value()) {
#else
    if (impl_->cfg.enableBatchRenderer) {
#endif
        impl_->renderMgr->readECS();
    }

    if (impl_->cfg.enableBatchRenderer) {
        impl_->renderMgr->batchRender();
    }
}

Tensor Manager::resetTensor() const
{
    return impl_->exportTensor(ExportID::Reset,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   1,
                               });
}

Tensor Manager::actionTensor() const
{
    return impl_->exportTensor(ExportID::Action, TensorElementType::Int32,
        {
            impl_->cfg.numWorlds,
            consts::numAgents,
            4,
        });
}

Tensor Manager::rewardTensor() const
{
    return impl_->exportTensor(ExportID::Reward, TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   consts::numAgents,
                                   1,
                               });
}

Tensor Manager::doneTensor() const
{
    return impl_->exportTensor(ExportID::Done, TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   consts::numAgents,
                                   1,
                               });
}

Tensor Manager::selfObservationTensor() const
{
    return impl_->exportTensor(ExportID::SelfObservation,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   consts::numAgents,
                                   8,
                               });
}

Tensor Manager::partnerObservationsTensor() const
{
    return impl_->exportTensor(ExportID::PartnerObservations,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   consts::numAgents,
                                   consts::numAgents - 1,
                                   3,
                               });
}

Tensor Manager::roomEntityObservationsTensor() const
{
    return impl_->exportTensor(ExportID::RoomEntityObservations,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   consts::numAgents,
                                   consts::maxEntitiesPerRoom,
                                   3,
                               });
}

Tensor Manager::doorObservationTensor() const
{
    return impl_->exportTensor(ExportID::DoorObservation,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   consts::numAgents,
                                   3,
                               });
}

Tensor Manager::lidarTensor() const
{
    return impl_->exportTensor(ExportID::Lidar, TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   consts::numAgents,
                                   consts::numLidarSamples,
                                   2,
                               });
}

Tensor Manager::stepsRemainingTensor() const
{
    return impl_->exportTensor(ExportID::StepsRemaining,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   consts::numAgents,
                                   1,
                               });
}

Tensor Manager::rgbTensor() const
{
    const uint8_t *rgb_ptr = impl_->renderMgr->batchRendererRGBOut();

    return Tensor((void*)rgb_ptr, TensorElementType::UInt8, {
        impl_->cfg.numWorlds,
        consts::numAgents,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        4,
    }, impl_->cfg.gpuID);
}

Tensor Manager::depthTensor() const
{
    const float *depth_ptr = impl_->renderMgr->batchRendererDepthOut();

    return Tensor((void *)depth_ptr, TensorElementType::Float32, {
        impl_->cfg.numWorlds,
        consts::numAgents,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        1,
    }, impl_->cfg.gpuID);
}

Tensor Manager::raycastTensor() const
{
    uint32_t pixels_per_view = impl_->raycastOutputResolution *
        impl_->raycastOutputResolution;
    return impl_->exportTensor(ExportID::Raycast,
                               TensorElementType::UInt8,
                               {
                                   impl_->cfg.numWorlds*consts::numAgents,
                                   pixels_per_view * 3,
                               });
}

void Manager::triggerReset(int32_t world_idx)
{
    WorldReset reset {
        1,
    };

    auto *reset_ptr = impl_->worldResetBuffer + world_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(reset_ptr, &reset, sizeof(WorldReset),
                   cudaMemcpyHostToDevice);
#endif
    }  else {
        *reset_ptr = reset;
    }
}

void Manager::setAction(int32_t world_idx,
                        int32_t agent_idx,
                        int32_t move_amount,
                        int32_t move_angle,
                        int32_t rotate,
                        int32_t grab,
                        int32_t x,
                        int32_t y,
                        int32_t z,
                        int32_t rot,
                        int32_t vrot)
{
    Action action { 
        .moveAmount = move_amount,
        .moveAngle = move_angle,
        .rotate = rotate,
        .grab = grab,
        .x = x,
        .y = y,
        .z = z,
        .rot = rot,
        .vrot = vrot,
    };

    auto *action_ptr = impl_->agentActionsBuffer +
        world_idx * consts::numAgents + agent_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(action_ptr, &action, sizeof(Action),
                   cudaMemcpyHostToDevice);
#endif
    } else {
        *action_ptr = action;
    }
}

render::RenderManager & Manager::getRenderManager()
{
    return *impl_->renderMgr;
}

}
