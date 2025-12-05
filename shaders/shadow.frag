#version 450 core

// Нам не нужно выводить цвет, OpenGL сам запишет глубину в буфер
void main()
{
    // gl_FragDepth = gl_FragCoord.z; // Это происходит автоматически
}
