#pragma once

#include <vector>
#include <string_view>
#include <filesystem>

namespace HabitatJSON {
    
enum class LightType {
    Point,
    Environment,
};

struct Light {
    LightType type;
    float position[3];
    float intensity;
    float color[3];
};

struct AdditionalInstance {
    std::string name;
    std::filesystem::path gltfPath;
    float pos[3];
    float rotation[4];
    float scale[3];
    bool dynamic;
};

struct AdditionalObject {
    std::string name;
    std::filesystem::path gltfPath;
};

struct Scene {
    std::filesystem::path stagePath;
    std::vector<AdditionalInstance> additionalInstances;
    std::vector<AdditionalObject> additionalObjects;
    std::vector<Light> lights;
};

Scene habitatJSONLoad(std::string_view scene_path_name);

}
