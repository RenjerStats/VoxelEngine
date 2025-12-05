#version 450 core

in GS_OUT {
    vec3 fragPos;
    vec3 normal;
    flat float colorIdx;
} fs_in;

out vec4 FragColor;

uniform sampler2D uPalette;
uniform sampler2D shadowMap; // <--- Новая текстура с глубиной

uniform vec3 lightDir;
uniform vec3 viewPos;
uniform float shininess;
uniform mat4 lightSpaceMatrix; // <--- Матрица вида света

// Функция расчета тени
float ShadowCalculation(vec3 fragPosWorld, vec3 normal, vec3 lightDir)
{
    // 1. Переводим позицию фрагмента в пространство света
    vec4 fragPosLightSpace = lightSpaceMatrix * vec4(fragPosWorld, 1.0);

    // 2. Perspective divide (стандартный шаг, хотя для Ortho он не меняет w)
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;

    // 3. Приводим диапазон [-1, 1] к [0, 1] для выборки из текстуры
    projCoords = projCoords * 0.5 + 0.5;

    // 4. Если мы за пределами карты теней (далеко), считаем, что тени нет
    if(projCoords.z > 1.0)
        return 0.0;

    // 5. Текущая глубина фрагмента (расстояние от света)
    float currentDepth = projCoords.z;

    // 6. Bias против "Shadow Acne" (артефакты в виде полос)
    // Чем больше угол падения света, тем больше bias
    float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);

    // 7. PCF (Percentage-closer filtering) для мягких краев тени
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -1; x <= 1; ++x)
    {
        for(int y = -1; y <= 1; ++y)
        {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
        }
    }
    shadow /= 9.0;

    return shadow;
}

void main()
{
    // --- 1. Цвет из палитры ---
    float u = (fs_in.colorIdx + 0.5) / 256.0;
    vec4 objectColor = texture(uPalette, vec2(u, 0.5));
    if(objectColor.a < 0.1) discard;

    // --- 2. Вектора ---
    vec3 norm = normalize(fs_in.normal);
    vec3 lightDirNorm = normalize(lightDir); // Направление НА свет

    // --- 3. Освещение (Blinn-Phong) ---
    // Ambient
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * objectColor.rgb;

    // Diffuse
    float diff = max(dot(norm, lightDirNorm), 0.0);
    vec3 diffuse = diff * objectColor.rgb;

    // Specular
    float specularStrength = 0.2;
    vec3 viewDir = normalize(viewPos - fs_in.fragPos);
    vec3 halfwayDir = normalize(lightDirNorm + viewDir);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), shininess);
    vec3 specular = vec3(1.0) * spec * specularStrength;

    // --- 4. Расчет Тени ---
    float shadow = ShadowCalculation(fs_in.fragPos, norm, lightDirNorm);

    // Итоговый цвет: (Ambient + (1.0 - Shadow) * (Diffuse + Specular))
    vec3 lighting = (ambient + (1.0 - shadow) * (diffuse + specular));

    FragColor = vec4(lighting, 1.0);
}
