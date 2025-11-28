#include <arv.h>
#include <zmq.h>
#include <csignal>
#include <cstring>
#include <iostream>
#include <stdexcept>

static bool running = true;

#pragma pack(push, 1)
struct FrameHeader {
    uint32_t width;
    uint32_t height;
    uint32_t channels;        // 1 = mono, 3 = RGB
    uint32_t bytes_per_pixel; // 1,2,4,...
    uint64_t pixel_format;    // Aravis pixel format ID
    uint64_t timestamp_ns;
};
#pragma pack(pop)

void signal_handler(int) {
    running = false;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "cam_server <port> <pixel_format> <channels>\n";
        std::cerr << "RGB:   cam_server 5555 RGB8_PACKED 3\n";
        std::cerr << "depth: cam_server 5556 Mono16 1\n";
        return 1;
    }

    int port = std::stoi(argv[1]);
    std::string pixfmt_str = argv[2];
    int channels = std::stoi(argv[3]);

    // Inicializar Aravis
    arv_update_device_list();
    int n_dev = arv_get_n_devices();
    if (n_dev == 0) {
        std::cerr << "No hay dispositivos GigE Vision\n";
        return 1;
    }

    const char* device_id = arv_get_device_id(0);
    ArvCamera* camera = arv_camera_new(device_id);
    if (!camera) {
        std::cerr << "Error creando cámara\n";
        return 1;
    }

    // Pixel format
    ArvPixelFormat pixel_format = ARV_PIXEL_FORMAT_RGB8_PACKED;
    if (pixfmt_str == "RGB8_PACKED") {
        pixel_format = ARV_PIXEL_FORMAT_RGB8_PACKED;
    } else if (pixfmt_str == "Mono8") {
        pixel_format = ARV_PIXEL_FORMAT_MONO_8;
    } else if (pixfmt_str == "Mono16") {
        pixel_format = ARV_PIXEL_FORMAT_MONO_16;
    } else {
        std::cerr << "PixelFormat no soportado en el ejemplo: " << pixfmt_str << "\n";
        return 1;
    }

    arv_camera_set_pixel_format(camera, pixel_format);

    // Resolución 
    int width = 0, height = 0;
    arv_camera_get_width(camera, &width);
    arv_camera_get_height(camera, &height);

    std::cout << "Cámara " << arv_camera_get_model_name(camera)
              << " " << width << "x" << height << ", pixfmt=" << pixfmt_str << "\n";

    // Crear stream
    ArvStream* stream = arv_camera_create_stream(camera, nullptr, nullptr, nullptr);
    if (!stream) {
        std::cerr << "Error creando stream\n";
        return 1;
    }

    // Asignar buffers
    for (int i = 0; i < 8; ++i) {
        size_t size = width * height * channels * ((pixfmt_str == "Mono16") ? 2 : 1);
        ArvBuffer* buffer = arv_buffer_new_allocate(size);
        arv_stream_push_buffer(stream, buffer);
    }

    arv_camera_set_acquisition_mode(camera, ARV_ACQUISITION_MODE_CONTINUOUS);
    arv_camera_start_acquisition(camera);

    // ZeroMQ PUB
    void* zmq_ctx = zmq_ctx_new();
    void* pub = zmq_socket(zmq_ctx, ZMQ_PUB);
    std::string endpoint = "tcp://*:" + std::to_string(port);
    if (zmq_bind(pub, endpoint.c_str()) != 0) {
        std::cerr << "Error bind ZMQ: " << zmq_strerror(errno) << "\n";
        return 1;
    }

    std::cout << "cam_server publicando en " << endpoint << "\n";

    std::signal(SIGINT, signal_handler);

    while (running) {
        ArvBuffer* buffer = arv_stream_timeout_pop_buffer(stream, 200000);
        if (!buffer) continue;

        if (arv_buffer_get_status(buffer) == ARV_BUFFER_STATUS_SUCCESS) {
            size_t data_size = 0;
            void* data = arv_buffer_get_data(buffer, &data_size);
            int img_w = arv_buffer_get_image_width(buffer);
            int img_h = arv_buffer_get_image_height(buffer);
            uint64_t ts = arv_buffer_get_timestamp(buffer);

            FrameHeader header;
            header.width = img_w;
            header.height = img_h;
            header.channels = channels;
            header.bytes_per_pixel = (pixfmt_str == "Mono16") ? 2 : 1;
            header.pixel_format = static_cast<uint64_t>(pixel_format);
            header.timestamp_ns = ts;

            size_t msg_size = sizeof(FrameHeader) + data_size;
            zmq_msg_t msg;
            zmq_msg_init_size(&msg, msg_size);
            std::memcpy(zmq_msg_data(&msg), &header, sizeof(FrameHeader));
            std::memcpy((char*)zmq_msg_data(&msg) + sizeof(FrameHeader), data, data_size);

            zmq_msg_send(&msg, pub, 0);
            zmq_msg_close(&msg);
        }

        arv_stream_push_buffer(stream, buffer);
    }

    arv_camera_stop_acquisition(camera);
    g_object_unref(stream);
    g_object_unref(camera);
    zmq_close(pub);
    zmq_ctx_term(zmq_ctx);

    return 0;
}
