# hailo_tappas_baseline — vendored reference postprocess

This directory vendors Hailo's official non-GStreamer C++ reference for YOLOv8/v5
instance segmentation, used by `ara2-validator` as a **baseline backend** for
head-to-head accuracy and performance comparison against the EdgeFirst HAL
decoder. It is not used in production inference paths.

## Provenance

| Field | Value |
|---|---|
| Upstream repo | https://github.com/hailo-ai/Hailo-Application-Code-Examples |
| Upstream path | `runtime/cpp/instance_segmentation/` + `runtime/cpp/common/` |
| Vendored at commit | `7b850e0444a142ada88443d675e3c3ddda18bd74` (2026-02-18) |
| Upstream license | MIT — see `LICENSE.upstream` |
| Vendored on | 2026-05-06 |

The `upstream/` subtree mirrors the upstream layout exactly so future syncs are a
straightforward `cp -r` from a fresh clone. No upstream files were modified.

## Why vendor instead of submodule

- The directory is small (~2,800 LOC) and changes infrequently.
- Pinning by copy lets the validator's CI build reproducibly without a network
  fetch step at build time.
- Future syncs are explicit (a commit that updates `upstream/` and bumps the
  hash above) rather than silent submodule pointer drift.

## Layout

```
hailo_tappas_baseline/
├── README.md                 # this file
├── LICENSE.upstream          # MIT license from upstream
└── upstream/                 # exact mirror of hailo-ai/Hailo-Application-Code-Examples runtime/cpp/
    ├── instance_segmentation/
    │   ├── instance_segmentation.cpp
    │   ├── instance_seg_postprocess.{cpp,hpp}
    │   ├── CMakeLists.txt
    │   ├── build.sh
    │   └── README.md
    └── common/
        ├── hailo_infer.{cpp,hpp}     # HailoRT VDevice/InferModel driver
        ├── toolbox.{cpp,hpp}         # Image I/O, OpenCV helpers, timing
        ├── general/                  # HailoROI/HailoTensor data classes, NMS, xtensor glue
        │   ├── hailo_common.hpp
        │   ├── hailo_objects.hpp
        │   ├── hailo_tensors.hpp
        │   ├── hailo_xtensor.hpp
        │   ├── math.hpp
        │   ├── nms.hpp
        │   └── tensors.hpp
        └── labels/
            ├── coco_eighty.hpp
            └── dota_fifteen.hpp
```

## Building (standalone sanity check)

This is the upstream's own build path — useful as a smoke test to confirm
HailoRT + OpenCV are correctly installed before attempting to wire the
postprocess into a Python module.

Prerequisites on aarch64 (RPi5) or x86_64 host:

```bash
sudo apt install gcc-11 g++-11 cmake libopencv-dev
# HailoRT debs from Hailo Developer Zone (provides HailoRTConfig.cmake):
sudo dpkg -i hailort_*_arm64.deb hailort-pcie-driver_*_all.deb
```

Build:

```bash
cd upstream/instance_segmentation
./build.sh
# Binary: build/x86_64/instance_segmentation  (or aarch64 on RPi5)
```

Run against one of our v0.3.0 HEFs:

```bash
./build/x86_64/instance_segmentation \
    -n yolov8m_seg \
    -i ../../../../assets/test.jpg \
    -h /path/to/yolov8n-seg-from-our-converter.hef
```

> **Caveat — HEF compatibility check.** The upstream postprocess assumes the
> standard YOLOv8-seg tensor naming/ordering. Our v0.3.0 HEFs ship the
> canonical schema-v2 metadata, but the upstream code reads tensors by name,
> not from `edgefirst.json`. Before assuming it just works, dump the output
> tensor names HailoRT enumerates for our HEF and confirm they match what
> `instance_seg_postprocess.cpp` expects.

## Future work in this directory

The next step (not yet committed) is to add a sibling `pybind11/` directory
containing a thin shim that exposes the postprocess as a Python module
importable by `ara2_validator.hailo_pipeline`. The upstream code in
`upstream/` will remain untouched so that diffs against future ACE updates
stay clean.
