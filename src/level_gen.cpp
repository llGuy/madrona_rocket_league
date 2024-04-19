#include "level_gen.hpp"
#include "madrona/render/ecs.hpp"

#define FLOORPLANNER

namespace madEscape {

using namespace madrona;
using namespace madrona::math;

namespace consts {

inline constexpr float doorWidth = consts::worldWidth / 3.f;

}

enum class RoomType : uint32_t {
    SingleButton,
    DoubleButton,
    CubeBlocking,
    CubeButtons,
    NumTypes,
};

static inline float randInRangeCentered(Engine &ctx, float range)
{
    return ctx.data().rng.sampleUniform() * range - range / 2.f;
}

static inline float randBetween(Engine &ctx, float min, float max)
{
    return ctx.data().rng.sampleUniform() * (max - min) + min;
}

// Initialize the basic components needed for physics rigid body entities
static inline void setupRigidBodyEntity(
    Engine &ctx,
    Entity e,
    Vector3 pos,
    Quat rot,
    SimObjectDefault sim_obj,
    EntityType entity_type,
    Diag3x3 scale = {1, 1, 1})
{
    ObjectID obj_id { (int32_t)sim_obj };

    ctx.get<Position>(e) = pos;
    ctx.get<Rotation>(e) = rot;
    ctx.get<Scale>(e) = scale;
    ctx.get<ObjectID>(e) = obj_id;
    ctx.get<EntityType>(e) = entity_type;
}

// Register the entity with the broadphase system
// This is needed for every entity with all the physics components.
// Not registering an entity will cause a crash because the broadphase
// systems will still execute over entities with the physics components.
static void registerRigidBodyEntity(
    Engine &ctx,
    Entity e,
    SimObjectDefault sim_obj)
{
    ObjectID obj_id { (int32_t)sim_obj };
}

// Creates floor, outer walls, and agent entities.
// All these entities persist across all episodes.
void createPersistentEntities(Engine &ctx)
{
#if 1
    printf("there are %d imported instances\n",
            (int)ctx.data().numImportedInstances);

#if defined(FLOORPLANNER)
    if (ctx.data().mergeAll) {

        for (int i = 0; i < 512; ++i) {
            Entity e_inst = ctx.makeEntity<DummyRenderable>();
            ctx.get<Position>(e_inst) = Vector3{(float)i * 100.f,0,0};
            ctx.get<Rotation>(e_inst) = Quat{1,0,0,0};
            ctx.get<Scale>(e_inst) = Diag3x3{1,1,1};
            //ctx.get<ObjectID>(e_inst).idx = (int)SimObjectDefault::NumObjects+67;//ctx.data().numObjects-8;
            ctx.get<ObjectID>(e_inst).idx = (int)SimObjectDefault::NumObjects;
            printf("onum: %d\n",ctx.data().numObjects);
            render::RenderingSystem::makeEntityRenderable(ctx, e_inst);
        }

    } else {
        for (int i = 0; i < (int)ctx.data().numImportedInstances; ++i) {
            ImportedInstance *imp_inst = &ctx.data().importedInstances[i];

            if (imp_inst->objectID < ctx.data().numObjects - 6) {
                Entity e_inst = ctx.makeEntity<DummyRenderable>();
                ctx.get<Position>(e_inst) = imp_inst->position;
                ctx.get<Rotation>(e_inst) = imp_inst->rotation;
                ctx.get<Scale>(e_inst) = imp_inst->scale;
                ctx.get<ObjectID>(e_inst).idx = imp_inst->objectID;

                render::RenderingSystem::makeEntityRenderable(ctx, e_inst);
            }
        }
    }
#endif
#endif

#if 0
    // Create the floor entity, just a simple static plane.
    Entity e = ctx.data().floorPlane = ctx.makeRenderableEntity<DummyRenderable>();
    ctx.get<Position>(e) = Vector3{0,0,0};
    ctx.get<Rotation>(e) = Quat(1,0,0,0);
    ctx.get<Scale>(e) = Diag3x3{1,1,1};
    ctx.get<ObjectID>(e) = {(int32_t)SimObjectDefault::Plane};
#endif


#if 0
    Entity e = ctx.makeRenderableEntity<DummyRenderable>();
    ctx.get<Position>(e) = Vector3{0,0,0};
    ctx.get<Rotation>(e) = Quat(1,0,0,0);
    ctx.get<Scale>(e) = Diag3x3{1,1,1};
    ctx.get<ObjectID>(e) = {(int32_t)SimObjectDefault::Cube};

    e = ctx.makeRenderableEntity<DummyRenderable>();
    ctx.get<Position>(e) = Vector3{0,0,10.f};
    ctx.get<Rotation>(e) = Quat(1,0,0,0);
    ctx.get<Scale>(e) = Diag3x3{1,1,1};
    ctx.get<ObjectID>(e) = {(int32_t)SimObjectDefault::Cube};

    e = ctx.makeRenderableEntity<DummyRenderable>();
    ctx.get<Position>(e) = Vector3{10.f,0,0.f};
    ctx.get<Rotation>(e) = Quat(1,0,0,0);
    ctx.get<Scale>(e) = Diag3x3{1,1,1};
    ctx.get<ObjectID>(e) = {(int32_t)SimObjectDefault::Cube};

    e = ctx.makeRenderableEntity<DummyRenderable>();
    ctx.get<Position>(e) = Vector3{40.f,0.f,0.f};
    ctx.get<Rotation>(e) = Quat(1,0,0,0);
    ctx.get<Scale>(e) = Diag3x3{30,30,30};
    ctx.get<ObjectID>(e) = {(int32_t)SimObjectDefault::Cube};
#endif


#if !defined(FLOORPLANNER) && 0
    // Create the outer wall entities
    // Behind
    ctx.data().borders[0] = ctx.makeRenderableEntity<DummyRenderable>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().borders[0],
        Vector3 {
            0,
            -consts::wallWidth / 2.f,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObjectDefault::Wall,
        EntityType::Wall,
        Diag3x3 {
            consts::worldWidth + consts::wallWidth * 2,
            consts::wallWidth,
            2.f,
        });

    // Right
    ctx.data().borders[1] = ctx.makeRenderableEntity<DummyRenderable>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().borders[1],
        Vector3 {
            consts::worldWidth / 2.f + consts::wallWidth / 2.f,
            consts::worldLength / 2.f,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObjectDefault::Wall,
        EntityType::Wall,
        Diag3x3 {
            consts::wallWidth,
            consts::worldLength,
            2.f,
        });

    // Left
    ctx.data().borders[2] = ctx.makeRenderableEntity<DummyRenderable>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().borders[2],
        Vector3 {
            -consts::worldWidth / 2.f - consts::wallWidth / 2.f,
            consts::worldLength / 2.f,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObjectDefault::Wall,
        EntityType::Wall,
        Diag3x3 {
            consts::wallWidth,
            consts::worldLength,
            2.f,
        });
#endif

     float yaws[2] = {
         1.141593f, -2.641593
     };

     if (ctx.data().mergeAll) {
        yaws[0] = -0.858407f;
     }

    // Create agent entities. Note that this leaves a lot of components
    // uninitialized, these will be set during world generation, which is
    // called for every episode.
    for (CountT i = 0; i < consts::numAgents; ++i) {
        Entity agent = ctx.data().agents[i] =
            ctx.makeRenderableEntity<Agent>();

#if defined(FLOORPLANNER)
        auto camera = ctx.makeEntity<DetatchedCamera>();
            ctx.get<AgentCamera>(agent) = { .camera = camera, .yaw = yaws[i], .pitch = 0 };
#else
        auto camera = ctx.makeEntity<DetatchedCamera>();
            ctx.get<AgentCamera>(agent) = { .camera = camera, .yaw = yaws[i], .pitch = 0 };
#endif

        // Create a render view for the agent
        render::RenderingSystem::attachEntityToView(ctx,
                camera,
                90.f, 0.001f,
                { 0,0,0 });

        ctx.get<ObjectID>(agent) = ObjectID { (int32_t)SimObjectDefault::Agent };

        ctx.get<Scale>(agent) = Diag3x3 { 1, 1, 1 };
        ctx.get<GrabState>(agent).constraintEntity = Entity::none();
        ctx.get<EntityType>(agent) = EntityType::Agent;

        ctx.get<Rotation>(agent) = Quat::angleAxis(
            math::pi/2.f,
            math::up);
        ctx.get<Rotation>(camera) = Quat::angleAxis(
            math::pi/2.f,
            math::up);
    }

    // Populate OtherAgents component, which maintains a reference to the
    // other agents in the world for each agent.
    for (CountT i = 0; i < consts::numAgents; i++) {
        Entity cur_agent = ctx.data().agents[i];

        OtherAgents &other_agents = ctx.get<OtherAgents>(cur_agent);
        CountT out_idx = 0;
        for (CountT j = 0; j < consts::numAgents; j++) {
            if (i == j) {
                continue;
            }

            Entity other_agent = ctx.data().agents[j];
            other_agents.e[out_idx++] = other_agent;
        }
    }
}

// Although agents and walls persist between episodes, we still need to
// re-register them with the broadphase system and, in the case of the agents,
// reset their positions.
static void resetPersistentEntities(Engine &ctx)
{
    // registerRigidBodyEntity(ctx, ctx.data().floorPlane, SimObject::Plane);

     for (CountT i = 0; i < 3; i++) {
         Entity wall_entity = ctx.data().borders[i];
         // registerRigidBodyEntity(ctx, wall_entity, SimObject::Wall);
     }

     Vector3 set_positions[2] = {
         {0,0,0},
         // { -23.289743, 34.212898, 13.000000 },
         { -94.375542, 74.301033, 12.000000 }
     };

     float yaws[2] = {
         1.141593f, -2.641593
     };

     for (CountT i = 0; i < consts::numAgents; i++) {
         Entity agent_entity = ctx.data().agents[i];
         //registerRigidBodyEntity(ctx, agent_entity, SimObject::Agent);

         // Place the agents near the starting wall
#if defined(FLOORPLANNER)
         Vector3 pos {
             -8.f + 5.f * (float)i, 10.f, 15.f
         };

         pos = set_positions[i];
#else
         Vector3 pos {
             -8.f + 5.f * (float)i, 10.f, 6.f
         };
#endif

         ctx.get<Rotation>(agent_entity) = Quat::angleAxis(
             -math::pi,
             math::up);

         auto camera = ctx.get<AgentCamera>(agent_entity).camera;
         ctx.get<Position>(camera) = ctx.get<Position>(agent_entity);
         ctx.get<Rotation>(camera) = ctx.get<Rotation>(agent_entity);
         ctx.get<Scale>(camera) = Diag3x3{ 0.001,0.001,0.001 };

         ctx.get<Position>(agent_entity) = pos;

         auto &grab_state = ctx.get<GrabState>(agent_entity);
         if (grab_state.constraintEntity != Entity::none()) {
             ctx.destroyEntity(grab_state.constraintEntity);
             grab_state.constraintEntity = Entity::none();
         }

         ctx.get<Progress>(agent_entity).maxY = pos.y;

         ctx.get<Action>(agent_entity) = Action {
             .moveAmount = 0,
             .moveAngle = 0,
             .rotate = consts::numTurnBuckets / 2,
             .grab = 0,
             .x = 1,
             .y = 1,
             .z = 1,
             .rot = 1,
             .vrot = 1
         };

         ctx.get<StepsRemaining>(agent_entity).t = consts::episodeLen;
     }
}

// Builds the two walls & door that block the end of the challenge room
static void makeEndWall(Engine &ctx,
                        Room &room,
                        CountT room_idx)
{
    float y_pos = consts::roomLength * (room_idx + 1) -
        consts::wallWidth / 2.f;

    // Quarter door of buffer on both sides, place door and then build walls
    // up to the door gap on both sides
    float door_center = randBetween(ctx, 0.75f * consts::doorWidth, 
        consts::worldWidth - 0.75f * consts::doorWidth);
    float left_len = door_center - 0.5f * consts::doorWidth;
    Entity left_wall = ctx.makeRenderableEntity<DummyRenderable>();
    setupRigidBodyEntity(
        ctx,
        left_wall,
        Vector3 {
            (-consts::worldWidth + left_len) / 2.f,
            y_pos,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObjectDefault::Wall,
        EntityType::Wall,
        Diag3x3 {
            left_len,
            consts::wallWidth,
            1.75f,
        });
    registerRigidBodyEntity(ctx, left_wall, SimObjectDefault::Wall);

    float right_len =
        consts::worldWidth - door_center - 0.5f * consts::doorWidth;
    Entity right_wall = ctx.makeRenderableEntity<DummyRenderable>();
    setupRigidBodyEntity(
        ctx,
        right_wall,
        Vector3 {
            (consts::worldWidth - right_len) / 2.f,
            y_pos,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObjectDefault::Wall,
        EntityType::Wall,
        Diag3x3 {
            right_len,
            consts::wallWidth,
            1.75f,
        });
    registerRigidBodyEntity(ctx, right_wall, SimObjectDefault::Wall);

    Entity door = ctx.makeRenderableEntity<DummyRenderable>();
    setupRigidBodyEntity(
        ctx,
        door,
        Vector3 {
            door_center - consts::worldWidth / 2.f,
            y_pos,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObjectDefault::Door,
        EntityType::Door,
        Diag3x3 {
            consts::doorWidth * 0.8f,
            consts::wallWidth,
            1.75f,
        });
    registerRigidBodyEntity(ctx, door, SimObjectDefault::Door);

    room.walls[0] = left_wall;
    room.walls[1] = right_wall;
    room.door = door;
}

static Entity makeButton(Engine &ctx,
                         float button_x,
                         float button_y)
{
    Entity button = ctx.makeRenderableEntity<DummyRenderable>();
    ctx.get<Position>(button) = Vector3 {
        button_x,
        button_y,
        0.f,
    };
    ctx.get<Rotation>(button) = Quat { 1, 0, 0, 0 };
    ctx.get<Scale>(button) = Diag3x3 {
        consts::buttonWidth,
        consts::buttonWidth,
        0.2f,
    };
    ctx.get<ObjectID>(button) = ObjectID { (int32_t)SimObjectDefault::Button };

    return button;
}

static Entity makeCube(Engine &ctx,
                       float cube_x,
                       float cube_y,
                       float scale = 1.f)
{
    Entity cube = ctx.makeRenderableEntity<DummyRenderable>();
    setupRigidBodyEntity(
        ctx,
        cube,
        Vector3 {
            cube_x,
            cube_y,
            1.f * scale,
        },
        Quat { 1, 0, 0, 0 },
        SimObjectDefault::Cube,
        EntityType::Cube,
        Diag3x3 {
            scale,
            scale,
            scale,
        });
    registerRigidBodyEntity(ctx, cube, SimObjectDefault::Cube);

    return cube;
}

static void setupDoor(Engine &ctx,
                      Entity door,
                      Span<const Entity> buttons,
                      bool is_persistent)
{
#if 0
    DoorProperties &props = ctx.get<DoorProperties>(door);

    for (CountT i = 0; i < buttons.size(); i++) {
        props.buttons[i] = buttons[i];
    }
    props.numButtons = (int32_t)buttons.size();
    props.isPersistent = is_persistent;
#endif
}

// A room with a single button that needs to be pressed, the door stays open.
static CountT makeSingleButtonRoom(Engine &ctx,
                                   Room &room,
                                   float y_min,
                                   float y_max)
{
    float button_x = randInRangeCentered(ctx,
        consts::worldWidth / 2.f - consts::buttonWidth);
    float button_y = randBetween(ctx, y_min + consts::roomLength / 4.f,
        y_max - consts::wallWidth - consts::buttonWidth / 2.f);

    Entity button = makeButton(ctx, button_x, button_y);

    setupDoor(ctx, room.door, { button }, true);

    room.entities[0] = button;

    return 1;
}

// A room with two buttons that need to be pressed simultaneously,
// the door stays open.
static CountT makeDoubleButtonRoom(Engine &ctx,
                                   Room &room,
                                   float y_min,
                                   float y_max)
{
    float a_x = randBetween(ctx,
        -consts::worldWidth / 2.f + consts::buttonWidth,
        -consts::buttonWidth);

    float a_y = randBetween(ctx,
        y_min + consts::roomLength / 4.f,
        y_max - consts::wallWidth - consts::buttonWidth / 2.f);

    Entity a = makeButton(ctx, a_x, a_y);

    float b_x = randBetween(ctx,
        consts::buttonWidth,
        consts::worldWidth / 2.f - consts::buttonWidth);

    float b_y = randBetween(ctx,
        y_min + consts::roomLength / 4.f,
        y_max - consts::wallWidth - consts::buttonWidth / 2.f);

    Entity b = makeButton(ctx, b_x, b_y);

    setupDoor(ctx, room.door, { a, b }, true);

    room.entities[0] = a;
    room.entities[1] = b;

    return 2;
}

// This room has 3 cubes blocking the exit door as well as two buttons.
// The agents either need to pull the middle cube out of the way and
// open the door or open the door with the buttons and push the cubes
// into the next room.
static CountT makeCubeBlockingRoom(Engine &ctx,
                                   Room &room,
                                   float y_min,
                                   float y_max)
{
    float button_a_x = randBetween(ctx,
        -consts::worldWidth / 2.f + consts::buttonWidth,
        -consts::buttonWidth - consts::worldWidth / 4.f);

    float button_a_y = randBetween(ctx,
        y_min + consts::buttonWidth,
        y_max - consts::roomLength / 4.f);

    Entity button_a = makeButton(ctx, button_a_x, button_a_y);

    float button_b_x = randBetween(ctx,
        consts::buttonWidth + consts::worldWidth / 4.f,
        consts::worldWidth / 2.f - consts::buttonWidth);

    float button_b_y = randBetween(ctx,
        y_min + consts::buttonWidth,
        y_max - consts::roomLength / 4.f);

    Entity button_b = makeButton(ctx, button_b_x, button_b_y);

    setupDoor(ctx, room.door, { button_a, button_b }, true);

    Vector3 door_pos = ctx.get<Position>(room.door);

    float cube_a_x = door_pos.x - 3.f;
    float cube_a_y = door_pos.y - 2.f;

    Entity cube_a = makeCube(ctx, cube_a_x, cube_a_y, 1.5f);

    float cube_b_x = door_pos.x;
    float cube_b_y = door_pos.y - 2.f;

    Entity cube_b = makeCube(ctx, cube_b_x, cube_b_y, 1.5f);

    float cube_c_x = door_pos.x + 3.f;
    float cube_c_y = door_pos.y - 2.f;

    Entity cube_c = makeCube(ctx, cube_c_x, cube_c_y, 1.5f);

    room.entities[0] = button_a;
    room.entities[1] = button_b;
    room.entities[2] = cube_a;
    room.entities[3] = cube_b;
    room.entities[4] = cube_c;

    return 5;
}

// This room has 2 buttons and 2 cubes. The buttons need to remain pressed
// for the door to stay open. To progress, the agents must push at least one
// cube onto one of the buttons, or more optimally, both.
static CountT makeCubeButtonsRoom(Engine &ctx,
                                  Room &room,
                                  float y_min,
                                  float y_max)
{
    float button_a_x = randBetween(ctx,
        -consts::worldWidth / 2.f + consts::buttonWidth,
        -consts::buttonWidth - consts::worldWidth / 4.f);

    float button_a_y = randBetween(ctx,
        y_min + consts::buttonWidth,
        y_max - consts::roomLength / 4.f);

    Entity button_a = makeButton(ctx, button_a_x, button_a_y);

    float button_b_x = randBetween(ctx,
        consts::buttonWidth + consts::worldWidth / 4.f,
        consts::worldWidth / 2.f - consts::buttonWidth);

    float button_b_y = randBetween(ctx,
        y_min + consts::buttonWidth,
        y_max - consts::roomLength / 4.f);

    Entity button_b = makeButton(ctx, button_b_x, button_b_y);

    setupDoor(ctx, room.door, { button_a, button_b }, false);

    float cube_a_x = randBetween(ctx,
        -consts::worldWidth / 4.f,
        -1.5f);

    float cube_a_y = randBetween(ctx,
        y_min + 2.f,
        y_max - consts::wallWidth - 2.f);

    Entity cube_a = makeCube(ctx, cube_a_x, cube_a_y, 1.5f);

    float cube_b_x = randBetween(ctx,
        1.5f,
        consts::worldWidth / 4.f);

    float cube_b_y = randBetween(ctx,
        y_min + 2.f,
        y_max - consts::wallWidth - 2.f);

    Entity cube_b = makeCube(ctx, cube_b_x, cube_b_y, 1.5f);

    room.entities[0] = button_a;
    room.entities[1] = button_b;
    room.entities[2] = cube_a;
    room.entities[3] = cube_b;

    return 4;
}

// Make the doors and separator walls at the end of the room
// before delegating to specific code based on room_type.
static void makeRoom(Engine &ctx,
                     LevelState &level,
                     CountT room_idx,
                     RoomType room_type)
{
    Room &room = level.rooms[room_idx];
    makeEndWall(ctx, room, room_idx);

    float room_y_min = room_idx * consts::roomLength;
    float room_y_max = (room_idx + 1) * consts::roomLength;

    CountT num_room_entities;
    switch (room_type) {
    case RoomType::SingleButton: {
        num_room_entities =
            makeSingleButtonRoom(ctx, room, room_y_min, room_y_max);
    } break;
    case RoomType::DoubleButton: {
        num_room_entities =
            makeDoubleButtonRoom(ctx, room, room_y_min, room_y_max);
    } break;
    case RoomType::CubeBlocking: {
        num_room_entities =
            makeCubeBlockingRoom(ctx, room, room_y_min, room_y_max);
    } break;
    case RoomType::CubeButtons: {
        num_room_entities =
            makeCubeButtonsRoom(ctx, room, room_y_min, room_y_max);
    } break;
    default: MADRONA_UNREACHABLE();
    }

    // Need to set any extra entities to type none so random uninitialized data
    // from prior episodes isn't exported to pytorch as agent observations.
    for (CountT i = num_room_entities; i < consts::maxEntitiesPerRoom; i++) {
        room.entities[i] = Entity::none();
    }
}

static void generateLevel(Engine &ctx)
{
    LevelState &level = ctx.singleton<LevelState>();

    // For training simplicity, define a fixed sequence of levels.
    makeRoom(ctx, level, 0, RoomType::DoubleButton);
    makeRoom(ctx, level, 1, RoomType::CubeBlocking);
    makeRoom(ctx, level, 2, RoomType::CubeButtons);

#if 0
    // An alternative implementation could randomly select the type for each
    // room rather than a fixed progression of challenge difficulty
    for (CountT i = 0; i < consts::numRooms; i++) {
        RoomType room_type = (RoomType)(
            ctx.data().rng.sampleI32(0, (uint32_t)RoomType::NumTypes));

        makeRoom(ctx, level, i, room_type);
    }
#endif
}

// Randomly generate a new world for a training episode
void generateWorld(Engine &ctx)
{
    resetPersistentEntities(ctx);

#if defined(FLOORPLANNER) && 0
    generateLevel(ctx);
#endif
}

}
