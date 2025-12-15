#pragma once

#include "core/Types.h"

#include <vector>
#include <cmath>

class PhysicsHelper{
public:
    static std::vector<RenderVoxel> generateSphereVoxels(float diameter, unsigned int colorID=1) {
        std::vector<RenderVoxel> voxels;

        float radius = diameter / 2;
        int r = std::ceil(radius);

        for (int x = -r; x <= r; x++) {
            for (int y = -r; y <= r; y++) {
                for (int z = -r; z <= r; z++) {
                    if (x*x + y*y + z*z <= (radius) * (radius)) {
                        voxels.push_back(RenderVoxel(x, y, z, colorID));
                    }
                }
            }
        }
        return voxels;
    }

    static std::vector<RenderVoxel> generateCubeVoxels(int sizeInVoxels,  unsigned int colorID=1) {
        std::vector<RenderVoxel> voxels;
        int half = sizeInVoxels / 2;

        for (int x = -half; x < half + (sizeInVoxels%2); x++) {
            for (int y = -half; y < half + (sizeInVoxels%2); y++) {
                for (int z = -half; z < half + (sizeInVoxels%2); z++) {
                    voxels.push_back(RenderVoxel(x, y, z, colorID));
                }
            }
        }
        return voxels;
    }
private:
    PhysicsHelper();
};
