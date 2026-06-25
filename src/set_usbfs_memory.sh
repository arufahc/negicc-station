#!/bin/bash
# Adjust the USB memory limit to 1024MB to prevent connection drop / ENOMEM during capture
echo 1024 > /sys/module/usbcore/parameters/usbfs_memory_mb
echo "[USB] usbfs_memory_mb successfully set to 1024"
