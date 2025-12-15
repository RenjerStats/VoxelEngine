#pragma once
struct RenderVoxel {
public:
    RenderVoxel(){};
    RenderVoxel(float x, float y, float z, float colorID):x(x), y(y), z(z), colorID(colorID){}

    float x, y, z;
    float colorID;
};
