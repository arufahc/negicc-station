#ifndef CAMERA_SESSION_H
#define CAMERA_SESSION_H

#include <string>
#include <vector>
#include <cstdint>

/**
 * @brief Enum defining the types of captures supported.
 */
enum class CaptureType {
    /**
     * @brief A single standard shot.
     */
    SINGLE,

    /**
     * @brief A 4-shot pixel shift sequence.
     * This implies that 4 separate images are captured with sensor shifting.
     * The output will contain exactly 4 file paths.
     */
    SONY_PIXEL_SHIFT_4
};

/**
 * @brief Structure containing the output of a capture operation.
 */
struct CaptureOutput {
    CaptureType type;
    std::vector<std::string> filepaths;
};

/**
 * @brief Abstract interface representing a generic camera session.
 * Agnostic to specific camera manufacturers (e.g. Sony).
 */
class CameraSession {
public:
    virtual ~CameraSession() = default;

    /**
     * @brief Connects to the camera and initializes the session.
     * @return true if successful, false otherwise.
     */
    virtual bool initialize() = 0;

    /**
     * @brief Configures standard camera settings (ISO, focus mode, store destination).
     * @return true if successful, false otherwise.
     */
    virtual bool configure_settings() = 0;

    /**
     * @brief Sets the camera's shutter speed.
     * @param val The shutter speed code in the camera's format.
     * @return true if successful, false otherwise.
     */
    virtual bool set_shutter_speed(uint32_t val) = 0;

    /**
     * @brief Triggers a capture sequence of the specified type.
     * @param type The type of capture to perform (e.g., SINGLE or SONY_PIXEL_SHIFT_4).
     * @param output The output structure to be filled with the capture details and file paths.
     * @return true if the capture succeeded and all files were downloaded, false otherwise.
     */
    virtual bool capture(CaptureType type, CaptureOutput& output) = 0;

    /**
     * @brief Closes the connection and shuts down the camera session.
     */
    virtual void close() = 0;
};

#endif // CAMERA_SESSION_H
