#version 450 core

// Входящие атрибуты (как настроено в C++: location 0 и 1)
layout(location = 0) in vec3 inPos;
layout(location = 1) in float inColorIdx;

// Передаем в Geometry Shader
out VS_OUT {
    float colorIdx;
} vs_out;

void main()
{
    gl_Position = vec4(inPos, 1.0);
    vs_out.colorIdx = inColorIdx;
}
