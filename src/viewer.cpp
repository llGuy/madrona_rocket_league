#include <madrona/viz/viewer.hpp>
#include <madrona/render/render_mgr.hpp>
#include <madrona/window.hpp>

#include "sim.hpp"
#include "mgr.hpp"
#include "types.hpp"

#include <filesystem>
#include <fstream>
#include <imgui.h>

using namespace madrona;
using namespace madrona::viz;

static HeapArray<int32_t> readReplayLog(const char *path)
{
    std::ifstream replay_log(path, std::ios::binary);
    replay_log.seekg(0, std::ios::end);
    int64_t size = replay_log.tellg();
    replay_log.seekg(0, std::ios::beg);

    HeapArray<int32_t> log(size / sizeof(int32_t));

    replay_log.read((char *)log.data(), (size / sizeof(int32_t)) * sizeof(int32_t));

    return log;
}

int main(int argc, char *argv[])
{
    using namespace madEscape;

    constexpr int64_t num_views = 2;

    printf("Started in: \n");
    // Read command line arguments
    uint32_t num_worlds = 1;
    if (argc >= 2) {
        num_worlds = (uint32_t)atoi(argv[1]);
    }

    ExecMode exec_mode = ExecMode::CPU;
    if (argc >= 3) {
        if (!strcmp("--cpu", argv[2])) {
            exec_mode = ExecMode::CPU;
        } else if (!strcmp("--cuda", argv[2])) {
            exec_mode = ExecMode::CUDA;
        }
    }

    // Setup replay log
    const char *replay_log_path = nullptr;
    if (argc >= 4) {
        replay_log_path = argv[3];
    }

    auto replay_log = Optional<HeapArray<int32_t>>::none();
    uint32_t cur_replay_step = 0;
    uint32_t num_replay_steps = 0;
    if (replay_log_path != nullptr) {
        replay_log = readReplayLog(replay_log_path);
        num_replay_steps = replay_log->size() / (num_worlds * num_views * 4);
    }

    // Render mode 0 is no rendering
    // Render mode 1 is rasterization.
    // Render mode 2 is raycasting.
    auto *render_mode = getenv("MADRONA_RENDER_MODE");

    bool enable_batch_renderer =
#ifdef MADRONA_MACOS
        false;
#else
        render_mode[0] == '1';
#endif

    //WindowManager wm {WindowManager::Config{.enableRenderAPIValidation=true,.renderBackendSelect =
    //        render::APIBackendSelect::Auto}};
    WindowManager wm {};
    WindowHandle window = wm.makeWindow("Escape Room", 1408, 1408);
    printf("Here\n");
    render::GPUHandle render_gpu = wm.initGPU(0, { window.get() });


    printf("premanage: \n");

    auto *resolution_str = getenv("MADRONA_RENDER_RESOLUTION");

    uint32_t raycast_output_resolution = 32;

    if (resolution_str[0] == '0') {
        raycast_output_resolution *= 1;
    } else if (resolution_str[0] == '1') {
        raycast_output_resolution *= 2;
    } else if (resolution_str[0] == '2') {
        raycast_output_resolution *= 4;
    } else if (resolution_str[0] == '3') {
        raycast_output_resolution *= 8;
    }

    auto *trace_test = getenv("MADRONA_TRACE_TEST");
    if (trace_test[0] == '1') {
        raycast_output_resolution = 1408;
        printf("I set the resolution!!!\n");
    }

    // Create the simulation manager
    Manager mgr({
        .execMode = exec_mode,
        .gpuID = 0,
        .numWorlds = num_worlds,
        .randSeed = 5,
        .autoReset = replay_log.has_value(),
        .enableBatchRenderer = enable_batch_renderer,
        .batchRenderViewWidth = raycast_output_resolution,
        .batchRenderViewHeight = raycast_output_resolution,
        .extRenderAPI = wm.gpuAPIManager().backend(),
        .extRenderDev = render_gpu.device(),
        .raycastOutputResolution = raycast_output_resolution,
    });
    printf("postmanage: \n");

    float camera_move_speed = 10.f;

    math::Vector3 initial_camera_position = { 0, consts::worldLength / 2.f, 30 };

    math::Quat initial_camera_rotation =
        (math::Quat::angleAxis(-math::pi / 2.f, math::up) *
        math::Quat::angleAxis(-math::pi / 2.f, math::right)).normalize();


    // Create the viewer viewer
    viz::Viewer viewer(mgr.getRenderManager(), window.get(), {
        .numWorlds = num_worlds,
        .simTickRate = 120,
        .cameraMoveSpeed = camera_move_speed,
        .cameraPosition = initial_camera_position,
        .cameraRotation = initial_camera_rotation,
    });

    // Replay step
    auto replayStep = [&]() {
        if (cur_replay_step == num_replay_steps - 1) {
            return true;
        }

        printf("Step: %u\n", cur_replay_step);

        for (uint32_t i = 0; i < num_worlds; i++) {
            for (uint32_t j = 0; j < num_views; j++) {
                uint32_t base_idx = 0;
                base_idx = 4 * (cur_replay_step * num_views * num_worlds +
                    i * num_views + j);

                int32_t move_amount = (*replay_log)[base_idx];
                int32_t move_angle = (*replay_log)[base_idx + 1];
                int32_t turn = (*replay_log)[base_idx + 2];
                int32_t g = (*replay_log)[base_idx + 3];

                printf("%d, %d: %d %d %d %d\n",
                       i, j, move_amount, move_angle, turn, g);
                mgr.setAction(i, j, move_amount, move_angle, turn, g,1,1,1,1,1);
            }
        }

        cur_replay_step++;

        return false;
    };

    // Printers
#if 0
    auto self_printer = mgr.selfObservationTensor().makePrinter();
    auto partner_printer = mgr.partnerObservationsTensor().makePrinter();
    auto room_ent_printer = mgr.roomEntityObservationsTensor().makePrinter();
    auto door_printer = mgr.doorObservationTensor().makePrinter();
    auto lidar_printer = mgr.lidarTensor().makePrinter();
    auto steps_remaining_printer = mgr.stepsRemainingTensor().makePrinter();
    auto reward_printer = mgr.rewardTensor().makePrinter();
#endif

    auto printObs = [&]() {
        /*printf("Self\n");
        self_printer.print();

        printf("Partner\n");
        partner_printer.print();

        printf("Room Entities\n");
        room_ent_printer.print();

        printf("Door\n");
        door_printer.print();

        printf("Lidar\n");
        lidar_printer.print();

        printf("Steps Remaining\n");
        steps_remaining_printer.print();

        printf("Reward\n");
        reward_printer.print();

        printf("\n");*/
    };



    // Main loop for the viewer viewer
    viewer.loop(
    [&mgr](CountT world_idx, const Viewer::UserInput &input)
    {
        // printf("new frame\n");

        using Key = Viewer::KeyboardKey;
        if (input.keyHit(Key::R)) {
            mgr.triggerReset(world_idx);
        }
    },
    [&mgr](CountT world_idx, CountT agent_idx,
           const Viewer::UserInput &input)
    {
        using Key = Viewer::KeyboardKey;

        int32_t x = 0;
        int32_t y = 0;
        int32_t r = 2;
        int32_t g = 0;

        bool shift_pressed = input.keyPressed(Key::Shift);

        if (input.keyPressed(Key::W)) {
            y += 1;
        }
        if (input.keyPressed(Key::S)) {
            y -= 1;
        }

        if (input.keyPressed(Key::D)) {
            x += 1;
        }
        if (input.keyPressed(Key::A)) {
            x -= 1;
        }

        if (input.keyPressed(Key::Q)) {
            r += shift_pressed ? 2 : 1;
        }
        if (input.keyPressed(Key::E)) {
            r -= shift_pressed ? 2 : 1;
        }

        if (input.keyHit(Key::G)) {
            g = 1;
        }

        int32_t move_amount;
        if (x == 0 && y == 0) {
            move_amount = 0;
        } else if (shift_pressed) {
            move_amount = consts::numMoveAmountBuckets - 1;
        } else {
            move_amount = 1;
        }

        int32_t move_angle;
        if (x == 0 && y == 1) {
            move_angle = 0;
        } else if (x == 1 && y == 1) {
            move_angle = 1;
        } else if (x == 1 && y == 0) {
            move_angle = 2;
        } else if (x == 1 && y == -1) {
            move_angle = 3;
        } else if (x == 0 && y == -1) {
            move_angle = 4;
        } else if (x == -1 && y == -1) {
            move_angle = 5;
        } else if (x == -1 && y == 0) {
            move_angle = 6;
        } else if (x == -1 && y == 1) {
            move_angle = 7;
        } else {
            move_angle = 0;
        }

        x = 1;
        if (input.keyPressed(Key::W)) {
            x = 2;
        }
        if (input.keyPressed(Key::S)) {
            x = 0;
        }

        y = 1;
        if (input.keyPressed(Key::D)) {
            y = 2;
        }
        if (input.keyPressed(Key::A)) {
            y = 0;
        }

        int rot=1;
        if (input.keyPressed(Key::Q)) {
            rot = 2;
        }
        if (input.keyPressed(Key::E)) {
            rot = 0;
        }

        int vrot = 1;
        if (input.keyPressed(Key::T)) {
            vrot = 2;
        }
        if (input.keyPressed(Key::F)) {
            vrot = 0;
        }

        int z = 1;
        if (input.keyPressed(Key::Space)) {
            z = 2;
        }

        if (input.keyPressed(Key::Shift)) {
            z = 0;
        }

        mgr.setAction(world_idx, agent_idx, move_amount, move_angle, r, g,x,y,z,rot,vrot);
    }, [&]() {
        if (replay_log.has_value()) {
            bool replay_finished = replayStep();

            if (replay_finished) {
                viewer.stopLoop();
            }
        }

        mgr.step();

        printObs();
    }, [&]() {
        unsigned char* print_ptr;
        #ifdef MADRONA_CUDA_SUPPORT
            int64_t num_bytes = 3 * raycast_output_resolution * raycast_output_resolution;
            print_ptr = (unsigned char*)cu::allocReadback(num_bytes);
        #else
            print_ptr = nullptr;
        #endif

        char *raycast_tensor = (char *)(mgr.raycastTensor().devicePtr());

        uint32_t bytes_per_image = 3 * raycast_output_resolution * raycast_output_resolution;
        uint32_t image_idx = viewer.getCurrentWorldID() * consts::numAgents + 
            std::max(viewer.getCurrentViewID(), (CountT)0);
        raycast_tensor += image_idx * bytes_per_image;

        if(exec_mode == ExecMode::CUDA){
#ifdef MADRONA_CUDA_SUPPORT
            cudaMemcpy(print_ptr, raycast_tensor,
                    bytes_per_image,
                    cudaMemcpyDeviceToHost);
            raycast_tensor = (char *)print_ptr;
#endif
        }

        ImGui::Begin("Raycast");

        auto draw2 = ImGui::GetWindowDrawList();
        ImVec2 windowPos = ImGui::GetWindowPos();
        char *raycasters = raycast_tensor;

        int vertOff = 70;

        float pixScale = 3;
        int extentsX = (int)(pixScale * raycast_output_resolution);
        int extentsY = (int)(pixScale * raycast_output_resolution);

        for (int i = 0; i < raycast_output_resolution; i++) {
            for (int j = 0; j < raycast_output_resolution; j++) {
                uint32_t linear_idx = 3 * (j + i * raycast_output_resolution);

                auto realColor = IM_COL32(
                        raycasters[linear_idx + 0],
                        raycasters[linear_idx + 1],
                        raycasters[linear_idx + 2], 
                        255);

                draw2->AddRectFilled(
                    { (i * pixScale) + windowPos.x, 
                      (j * pixScale) + windowPos.y +vertOff }, 
                    { ((i + 1) * pixScale) + windowPos.x,   
                      ((j + 1) * pixScale)+ +windowPos.y+vertOff },
                    realColor, 0, 0);
            }
        }
        ImGui::End();
    });
}
