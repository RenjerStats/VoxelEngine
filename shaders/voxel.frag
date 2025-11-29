#version 450 core

in GS_OUT {
    vec3 fragPos;
    vec3 normal;
    flat float colorIdx;
} fs_in;

out vec4 FragColor;

uniform sampler2D uPalette; // Ваша палитра 256x1
uniform vec3 lightDir;      // Направление света
uniform vec3 viewPos;       // Позиция камеры
uniform float shininess;

void main()
{
    // --- 1. Получение цвета из палитры ---
    // Текстурные координаты:
    // Индекс 0..255 нужно привести к 0..1.
    // Сдвигаем на 0.5 пикселя, чтобы попасть в центр текселя.
    float u = (fs_in.colorIdx + 0.5) / 256.0;
    vec4 objectColor = texture(uPalette, vec2(u, 0.5));

    // Если альфа 0, можно отбросить фрагмент (опционально)
    if(objectColor.a < 0.1) discard;

    // --- 2. Освещение (Blinn-Phong) ---

    // Ambient (фоновое)
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * objectColor.rgb;

    // Diffuse (рассеянное)
    vec3 norm = normalize(fs_in.normal);
    // lightDir у вас уже normalized в C++, но для надежности:
    vec3 lightDirNorm = normalize(lightDir);
    float diff = max(dot(norm, lightDirNorm), 0.0);
    vec3 diffuse = diff * objectColor.rgb;

    // Specular (блики)
    float specularStrength = 0.2; // Можно вынести в uniform
    vec3 viewDir = normalize(viewPos - fs_in.fragPos);
    vec3 halfwayDir = normalize(lightDirNorm + viewDir);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), shininess);
    vec3 specular = vec3(1.0) * spec * specularStrength; // Белый блик

    vec3 result = ambient + diffuse + specular;

    FragColor = vec4(result, 1.0);
}
