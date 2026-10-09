#include <torch/torch.h>

#include "resnet.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

static constexpr int CANVAS_SIZE = 280;
static constexpr int MNIST_SIZE = 28;

// These must match the preprocessing used when training the model.
static constexpr float MNIST_MEAN = 0.1307f;
static constexpr float MNIST_STD  = 0.3081f;

// -----------------------------------------------------------------------------
// Drawing canvas
// -----------------------------------------------------------------------------

class DigitCanvas {
public:
    DigitCanvas()
        : pixels_(CANVAS_SIZE * CANVAS_SIZE, 0.0f) {
    }

    void clear() {
        std::fill(pixels_.begin(), pixels_.end(), 0.0f);
    }

    const std::vector<float>& pixels() const {
        return pixels_;
    }

    void drawPoint(float x, float y, float radius) {
        const int min_x = std::max(
            0,
            static_cast<int>(std::floor(x - radius))
        );

        const int max_x = std::min(
            CANVAS_SIZE - 1,
            static_cast<int>(std::ceil(x + radius))
        );

        const int min_y = std::max(
            0,
            static_cast<int>(std::floor(y - radius))
        );

        const int max_y = std::min(
            CANVAS_SIZE - 1,
            static_cast<int>(std::ceil(y + radius))
        );

        const float radius_squared = radius * radius;

        for (int py = min_y; py <= max_y; ++py) {
            for (int px = min_x; px <= max_x; ++px) {
                const float dx =
                    static_cast<float>(px) - x;

                const float dy =
                    static_cast<float>(py) - y;

                const float distance_squared =
                    dx * dx + dy * dy;

                if (distance_squared > radius_squared) {
                    continue;
                }

                const float distance =
                    std::sqrt(distance_squared);

                float alpha =
                    1.0f - distance / radius;

                alpha = std::clamp(alpha, 0.0f, 1.0f);

                float& pixel =
                    pixels_[py * CANVAS_SIZE + px];

                pixel = std::max(pixel, alpha);
            }
        }
    }

    void drawLine(
        float x0,
        float y0,
        float x1,
        float y1,
        float radius
    ) {
        const float dx = x1 - x0;
        const float dy = y1 - y0;

        const float distance =
            std::sqrt(dx * dx + dy * dy);

        // Keep the circles close enough together that
        // fast mouse movements don't produce gaps.
        const float step =
            std::max(1.0f, radius * 0.35f);

        const int steps =
            std::max(
                1,
                static_cast<int>(
                    std::ceil(distance / step)
                )
            );

        for (int i = 0; i <= steps; ++i) {
            const float t =
                static_cast<float>(i) /
                static_cast<float>(steps);

            const float x =
                x0 + dx * t;

            const float y =
                y0 + dy * t;

            drawPoint(x, y, radius);
        }
    }

private:
    std::vector<float> pixels_;
};

// -----------------------------------------------------------------------------
// Convert 280x280 canvas to the 28x28 tensor expected by the network
// -----------------------------------------------------------------------------

static torch::Tensor canvasToTensor(
    const DigitCanvas& canvas
) {
    /*
     * The canvas contains values in [0, 1]:
     *
     *   0 = black
     *   1 = white
     *
     * We create:
     *
     *   [1, 1, 280, 280]
     *
     * and then downsample it to:
     *
     *   [1, 1, 28, 28]
     */

    auto image = torch::from_blob(
        const_cast<float*>(canvas.pixels().data()),
        {1, 1, CANVAS_SIZE, CANVAS_SIZE},
        torch::TensorOptions()
            .dtype(torch::kFloat32)
    ).clone();

    // Average every 10x10 region.
    auto small = torch::avg_pool2d(
        image,
        {10, 10},
        {10, 10}
    );

    /*
     * MNIST normalization.
     *
     * This assumes the model was trained using:
     *
     *   transforms.Normalize(
     *       (0.1307,),
     *       (0.3081,)
     *   )
     */
    small =
        (small - MNIST_MEAN) / MNIST_STD;

    return small;
}

// -----------------------------------------------------------------------------
// Run inference
// -----------------------------------------------------------------------------

static int predict(
    Net& net,
    const DigitCanvas& canvas,
    const torch::Device& device
) {
    torch::NoGradGuard no_grad;

    auto input =
        canvasToTensor(canvas).to(device);

    auto output =
        net.forward(input);

    /*
     * Net::forward() returns log_softmax().
     *
     * Therefore argmax() directly gives the predicted
     * class without needing to call softmax().
     */

    const int64_t prediction =
        output.argmax(1).item<int64_t>();

    return static_cast<int>(prediction);
}

// -----------------------------------------------------------------------------
// GLFW error callback
// -----------------------------------------------------------------------------

static void glfwErrorCallback(
    int error,
    const char* description
) {
    std::cerr
        << "GLFW error "
        << error
        << ": "
        << description
        << '\n';
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
    // -------------------------------------------------------------------------
    // GLFW
    // -------------------------------------------------------------------------

    glfwSetErrorCallback(glfwErrorCallback);

    if (!glfwInit()) {
        std::cerr
            << "Failed to initialize GLFW.\n";

        return 1;
    }

#ifdef __APPLE__

    /*
     * macOS requires a forward-compatible core profile
     * for modern OpenGL contexts.
     */

    glfwWindowHint(
        GLFW_CONTEXT_VERSION_MAJOR,
        3
    );

    glfwWindowHint(
        GLFW_CONTEXT_VERSION_MINOR,
        3
    );

    glfwWindowHint(
        GLFW_OPENGL_FORWARD_COMPAT,
        GLFW_TRUE
    );

    glfwWindowHint(
        GLFW_OPENGL_PROFILE,
        GLFW_OPENGL_CORE_PROFILE
    );

#else

    glfwWindowHint(
        GLFW_CONTEXT_VERSION_MAJOR,
        3
    );

    glfwWindowHint(
        GLFW_CONTEXT_VERSION_MINOR,
        3
    );

    glfwWindowHint(
        GLFW_OPENGL_PROFILE,
        GLFW_OPENGL_CORE_PROFILE
    );

#endif

    // -------------------------------------------------------------------------
    // Window
    // -------------------------------------------------------------------------

    GLFWwindow* window =
        glfwCreateWindow(
            900,
            600,
            "MNIST Digit Recognizer",
            nullptr,
            nullptr
        );

    if (!window) {
        std::cerr
            << "Failed to create GLFW window.\n";

        glfwTerminate();

        return 1;
    }

    glfwMakeContextCurrent(window);

    // Enable vsync.
    glfwSwapInterval(1);

    // -------------------------------------------------------------------------
    // Dear ImGui
    // -------------------------------------------------------------------------

    IMGUI_CHECKVERSION();

    ImGui::CreateContext();

    ImGuiIO& io =
        ImGui::GetIO();

    io.ConfigFlags |=
        ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(
        window,
        true
    );

    ImGui_ImplOpenGL3_Init(
        "#version 330"
    );

    // -------------------------------------------------------------------------
    // Load model
    // -------------------------------------------------------------------------

    const std::string model_path =
        argc >= 2
            ? argv[1]
            : "net.pt";

    std::cout
        << "Loading model: "
        << model_path
        << '\n';

    auto net =
        std::make_shared<Net>(10);

    try {
        /*
         * This expects net.pt to have been created with
         * the LibTorch C++ serialization API, e.g.:
         *
         *     torch::save(net, "net.pt");
         */

        torch::load(
            net,
            model_path
        );

        net->eval();

        std::cout
            << "Model loaded successfully.\n";
    }
    catch (const c10::Error& e) {
        std::cerr
            << "Failed to load model '"
            << model_path
            << "':\n"
            << e.what()
            << '\n';

        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();

        ImGui::DestroyContext();

        glfwDestroyWindow(window);
        glfwTerminate();

        return 1;
    }

    // -------------------------------------------------------------------------
    // Select device
    // -------------------------------------------------------------------------

    torch::Device device =
        get_best_device();

    std::cout
        << "Using device: "
        << device
        << '\n';

    try {
        net->to(device);
        net->eval();
    }
    catch (const c10::Error& e) {
        std::cerr
            << "Could not move model to "
            << device
            << ":\n"
            << e.what()
            << '\n';

        std::cerr
            << "Falling back to CPU.\n";

        device =
            torch::Device(torch::kCPU);

        net->to(device);
        net->eval();
    }

    // -------------------------------------------------------------------------
    // Application state
    // -------------------------------------------------------------------------

    DigitCanvas canvas;

    int prediction = -1;

    float brush_radius = 12.0f;

    bool drawing = false;

    float last_x = 0.0f;
    float last_y = 0.0f;

    // -------------------------------------------------------------------------
    // Main loop
    // -------------------------------------------------------------------------

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // ---------------------------------------------------------------------
        // Start ImGui frame
        // ---------------------------------------------------------------------

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();

        ImGui::NewFrame();

        // ---------------------------------------------------------------------
        // Main window
        // ---------------------------------------------------------------------

        ImGui::SetNextWindowSize(
            ImVec2(850.0f, 550.0f),
            ImGuiCond_FirstUseEver
        );

        ImGui::Begin(
            "MNIST Digit Recognizer"
        );

        ImGui::TextUnformatted(
            "Draw a digit:"
        );

        ImGui::SameLine();

        if (ImGui::Button("Clear")) {
            canvas.clear();
            prediction = -1;
        }

        ImGui::SliderFloat(
            "Brush size",
            &brush_radius,
            2.0f,
            30.0f,
            "%.1f"
        );

        ImGui::Spacing();

        // ---------------------------------------------------------------------
        // Canvas position
        // ---------------------------------------------------------------------

        const ImVec2 canvas_pos =
            ImGui::GetCursorScreenPos();

        const ImVec2 canvas_size(
            static_cast<float>(CANVAS_SIZE),
            static_cast<float>(CANVAS_SIZE)
        );

        ImDrawList* draw_list =
            ImGui::GetWindowDrawList();

        // ---------------------------------------------------------------------
        // Canvas background
        // ---------------------------------------------------------------------

        draw_list->AddRectFilled(
            canvas_pos,
            ImVec2(
                canvas_pos.x + canvas_size.x,
                canvas_pos.y + canvas_size.y
            ),
            IM_COL32(
                255,
                255,
                255,
                255
            )
        );

        // ---------------------------------------------------------------------
        // Canvas border
        // ---------------------------------------------------------------------

        draw_list->AddRect(
            canvas_pos,
            ImVec2(
                canvas_pos.x + canvas_size.x,
                canvas_pos.y + canvas_size.y
            ),
            IM_COL32(
                100,
                100,
                100,
                255
            ),
            0.0f,
            0,
            2.0f
        );

        // ---------------------------------------------------------------------
        // Draw rasterized canvas
        // ---------------------------------------------------------------------

        const auto& pixels =
            canvas.pixels();

        for (int y = 0; y < CANVAS_SIZE; ++y) {
            for (int x = 0; x < CANVAS_SIZE; ++x) {
                const float value =
                    pixels[
                        y * CANVAS_SIZE + x
                    ];

                if (value <= 0.01f) {
                    continue;
                }

                const int intensity =
                    255 -
                    static_cast<int>(
                        value * 255.0f
                    );

                draw_list->AddRectFilled(
                    ImVec2(
                        canvas_pos.x +
                            static_cast<float>(x),

                        canvas_pos.y +
                            static_cast<float>(y)
                    ),

                    ImVec2(
                        canvas_pos.x +
                            static_cast<float>(x + 1),

                        canvas_pos.y +
                            static_cast<float>(y + 1)
                    ),

                    IM_COL32(
                        intensity,
                        intensity,
                        intensity,
                        255
                    )
                );
            }
        }

        // ---------------------------------------------------------------------
        // Invisible mouse interaction area
        // ---------------------------------------------------------------------

        ImGui::SetCursorScreenPos(
            canvas_pos
        );

        ImGui::InvisibleButton(
            "drawing_canvas",
            canvas_size,
            ImGuiButtonFlags_MouseButtonLeft
        );

        const bool canvas_hovered =
            ImGui::IsItemHovered();

        // ---------------------------------------------------------------------
        // Drawing
        // ---------------------------------------------------------------------

        if (canvas_hovered) {
            const ImVec2 mouse =
                ImGui::GetIO().MousePos;

            const float x =
                mouse.x - canvas_pos.x;

            const float y =
                mouse.y - canvas_pos.y;

            // Start a new stroke.
            if (ImGui::IsMouseClicked(
                    ImGuiMouseButton_Left)) {

                drawing = true;

                last_x = x;
                last_y = y;

                canvas.drawPoint(
                    x,
                    y,
                    brush_radius
                );

                prediction =
                    predict(
                        *net,
                        canvas,
                        device
                    );
            }

            // Continue current stroke.
            if (drawing &&
                ImGui::IsMouseDown(
                    ImGuiMouseButton_Left)) {

                canvas.drawLine(
                    last_x,
                    last_y,
                    x,
                    y,
                    brush_radius
                );

                last_x = x;
                last_y = y;

                prediction =
                    predict(
                        *net,
                        canvas,
                        device
                    );
            }
        }

        if (ImGui::IsMouseReleased(
                ImGuiMouseButton_Left)) {

            drawing = false;
        }

        // ---------------------------------------------------------------------
        // Recognition result
        // ---------------------------------------------------------------------

        ImGui::SameLine();

        ImGui::BeginGroup();

        ImGui::TextUnformatted(
            "Recognized digit:"
        );

        ImGui::Spacing();

        if (prediction >= 0) {
            ImGui::Text(
                "%d",
                prediction
            );
        }
        else {
            ImGui::TextUnformatted(
                "-"
            );
        }

        ImGui::Spacing();

        ImGui::TextUnformatted(
            "Input:"
        );

        ImGui::Text(
            "%dx%d",
            CANVAS_SIZE,
            CANVAS_SIZE
        );

        ImGui::TextUnformatted(
            "Network input:"
        );

        ImGui::Text(
            "%dx%d",
            MNIST_SIZE,
            MNIST_SIZE
        );

        ImGui::Spacing();

        ImGui::TextWrapped(
            "Draw a handwritten digit in the "
            "white area."
        );

        ImGui::EndGroup();

        ImGui::End();

        // ---------------------------------------------------------------------
        // Render
        // ---------------------------------------------------------------------

        ImGui::Render();

        int display_width = 0;
        int display_height = 0;

        glfwGetFramebufferSize(
            window,
            &display_width,
            &display_height
        );

        glViewport(
            0,
            0,
            display_width,
            display_height
        );

        glClearColor(
            0.12f,
            0.12f,
            0.12f,
            1.0f
        );

        glClear(
            GL_COLOR_BUFFER_BIT
        );

        ImGui_ImplOpenGL3_RenderDrawData(
            ImGui::GetDrawData()
        );

        glfwSwapBuffers(window);
    }

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------

    ImGui_ImplOpenGL3_Shutdown();

    ImGui_ImplGlfw_Shutdown();

    ImGui::DestroyContext();

    glfwDestroyWindow(window);

    glfwTerminate();

    return 0;
}
