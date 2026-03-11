# Cubemap Converter

This project includes a local utility for converting six cubemap face images into one equirectangular panorama that can be uploaded into the existing tour pipeline.

## Supported input

Place the six source files in one folder and name them exactly:

- `front.tif`
- `back.tif`
- `left.tif`
- `right.tif`
- `top.tif`
- `down.tif`

All six faces must be square and have the same resolution.

## Command

Run from the project root:

```powershell
python scripts\cubemap_to_equirect.py `
  --faces-dir C:\Users\Michael\Downloads `
  --output C:\Users\Michael\lokalny_obiektyw\output\cubemap_equirect_8192.jpg `
  --width 8192
```

## Parameters

- `--faces-dir`: folder that contains the six cubemap files.
- `--output`: output file path. Supported formats depend on the extension, for example `.jpg`, `.png`, `.tif`.
- `--width`: output panorama width. Height is always `width / 2`.
- `--chunk-rows`: optional processing chunk size used to limit RAM usage. Default is `256`.

## Recommended sizes

- `8192`: good default for web delivery and editor testing.
- `12288`: higher quality if the source faces are large and the target machine has enough RAM.

## Output

The script produces one standard equirectangular panorama. After conversion, upload that single file into the product as a normal 360 panorama scene.

## Notes

- The current product does not yet import cubemap faces directly in Studio.
- This utility is the intended bridge until native cubemap import is added.
