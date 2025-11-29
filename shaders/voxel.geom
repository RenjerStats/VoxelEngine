#version 450 core

layout (points) in;
// Максимум вершин: 6 граней * 4 вершины = 24
layout (triangle_strip, max_vertices = 24) out;

in VS_OUT {
    float colorIdx;
} gs_in[];

out GS_OUT {
    vec3 fragPos;
    vec3 normal;
    flat float colorIdx; // flat, чтобы цвет не интерполировался
} gs_out;

uniform mat4 view;
uniform mat4 proj;
uniform float voxelSize;

// Функция для создания грани
void CreateFace(vec3 center, vec3 offset, vec3 u, vec3 v, vec3 normalDir) {
    gs_out.normal = normalDir;
    gs_out.colorIdx = gs_in[0].colorIdx;

    // Половина размера вокселя
    float halfSize = voxelSize * 0.5;

    // Вершины квадрата (Triangle Strip)
    // Порядок: Bottom-Left, Bottom-Right, Top-Left, Top-Right (для CCW culling)
    vec3 p1 = center + offset - u * halfSize - v * halfSize;
    vec3 p2 = center + offset + u * halfSize - v * halfSize;
    vec3 p3 = center + offset - u * halfSize + v * halfSize;
    vec3 p4 = center + offset + u * halfSize + v * halfSize;

    // 1
    gl_Position = proj * view * vec4(p1, 1.0);
    gs_out.fragPos = p1;
    EmitVertex();

    // 2
    gl_Position = proj * view * vec4(p2, 1.0);
    gs_out.fragPos = p2;
    EmitVertex();

    // 3
    gl_Position = proj * view * vec4(p3, 1.0);
    gs_out.fragPos = p3;
    EmitVertex();

    // 4
    gl_Position = proj * view * vec4(p4, 1.0);
    gs_out.fragPos = p4;
    EmitVertex();

    EndPrimitive();
}

void main() {
    vec3 center = gl_in[0].gl_Position.xyz;
    float hs = voxelSize * 0.5;

    // Векторы направлений для генерации граней
    vec3 up    = vec3(0.0, 1.0, 0.0);
    vec3 right = vec3(1.0, 0.0, 0.0);
    vec3 front = vec3(0.0, 0.0, 1.0);

    // Передняя грань (+Z)
    CreateFace(center, front * hs, right, up, front);

    // Задняя грань (-Z) - меняем u и v местами или знак для сохранения Winding order
    CreateFace(center, -front * hs, -right, up, -front);

    // Правая грань (+X)
    CreateFace(center, right * hs, -front, up, right);

    // Левая грань (-X)
    CreateFace(center, -right * hs, front, up, -right);

    // Верхняя грань (+Y)
    CreateFace(center, up * hs, right, -front, up);

    // Нижняя грань (-Y)
    CreateFace(center, -up * hs, right, front, -up);
}
