cwExtract (Matchbox) -- object mattes from a cwSegment "Object IDs" map

cwSegment with Output = Object IDs writes object n as the value n x 0.02 (up to
50 objects, background 0, one object per pixel). cwExtract decodes each pixel to
the nearest ID, so float16 and 8-bit maps work, and outputs:
  RGB + Matte  one object each in R, G, B and on the Matte output
  One Object   that object in every channel
  Range        objects From..To as one matte
Feed the ID map unfiltered: no resize, blur or colour transform in between.

Install: keep cwExtract.xml + cwExtract.1.glsl together in one folder and load the
.xml from the Matchbox browser (or copy both into /opt/Autodesk/presets/<version>/matchbox/shaders).
Test (headless GL 4.3, moderngl + EGL): python test_cwextract_gl.py
