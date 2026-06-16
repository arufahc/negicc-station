#ifndef SONY_CAMERA_SESSION_H
#define SONY_CAMERA_SESSION_H

#include "camera_session.h"
#include <memory>

/**
 * @brief Sony-specific implementation of the CameraSession interface.
 * Uses the Sony Camera Remote SDK to interact with the A7R4.
 */
class SonyCameraSession : public CameraSession {
public:
    SonyCameraSession();
    virtual ~SonyCameraSession();

    virtual bool initialize() override;
    virtual bool configure_settings() override;
    virtual bool set_shutter_speed(uint32_t val) override;
    virtual bool capture(CaptureType type, CaptureOutput& output) override;
    virtual void close() override;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif // SONY_CAMERA_SESSION_H
