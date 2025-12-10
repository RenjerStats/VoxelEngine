#pragma once

struct CudaVoxel {      //offsets
    float x, y, z;      // 0, 4, 8 bytes
    float vx, vy, vz;   // 12, 16, 20 bytes
    float mass;         // 24 bytes
    float friction;     // 28 bytes
    float elasticity;   // 32 bytes
    float colorID;      // 36 bytes
    float oldX, oldY, oldZ;
};
