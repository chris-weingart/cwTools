<p><picture><source media="(prefers-color-scheme: light)" srcset="img/cwSegment-light.svg"><img src="img/cwSegment.svg" alt="cwSegment" height="56"></picture></p>

# cwSegment

Prompt-based object segmentation for Flame Batch: tell it **what** to
separate and it returns clean object masks. Based on SAM3.

<!-- screenshot -->

## Prompts

- **Text Prompt:** one or more concepts, comma separated: `person, sofa, dog`.
  Each concept can return several instances; **Best Instance Only** keeps the
  highest-scoring one.
- **Auto Concepts (optional):** a vision language model looks at the frame and
  fills in the concepts for you. It works with any OpenAI-compatible endpoint
  (**VLM URL**, a local Ollama by default), because the frame is sent to it.
- **Pick R / G / B / Matte:** press it, and the viewer shows every object (ID
  Map) inside a frame in that channel's colour; click one to put it in that
  channel. The view then returns to Object Channels, ready for the next pick.
- **R / G / B / Matte Number:** with the slot on *Object #*, which object goes
  there (#1 = first in the list); on *Concept #*, which typed concept. While fewer
  than four concepts are typed, a Matte slot on *Concept #* 4 carries the full
  matte of all objects; from the fourth concept on it is that concept (black
  when it isn't found, e.g. off frame).

## Output

- **Mask:** all found objects as one mask.
- **Object Channels** (default): up to four objects in R, G, B and A (the
  plate passes through until an object is found), so one node feeds
  several refiners. Objects are numbered by the **Number By** rule.
- **ID Map:** every object in its own colour, optionally with its number
  stamped in, to see which object is which.
- **Object IDs:** every object as one value (object n = n × 0.02, up to 50)
  for the **cwExtract** Matchbox, which turns it back into mattes: four objects
  per node (R, G, B, Matte), one object, or a range. Keep the IDs unfiltered
  between the two nodes: no resize, blur or colour transform.
- **Overlay / Source + Mask:** for checking.

Without tracking, objects are found on every frame anew, so their numbers can
swap between frames when the scene changes. **Track** keeps them.

## Track

Tracking follows chosen objects through the shot with SAM3's video memory
(Device = Local):

1. Type the concepts (e.g. `person, dog`), set Device to Local and go to the
   frame to start from.
2. Press **Pick Objects**: the viewer shows every object, numbered, in an
   orange frame. Click the ones to follow; each click adds its number to
   **Tracked Objects** (or type the numbers, e.g. `2, 5`, and press **Anchor
   Here**). The frame you pick on becomes the anchor frame.
3. Press **Pick Objects** again to finish. **Track** turns on (Forward; also
   Backward or Both). The output now carries only the picked objects.
4. Press **Track Range**: the frames from the anchor to **Track To**
   (Forward / Both) and back to **Track From** (Backward / Both) are fetched
   and then tracked in the background, so Flame stays usable. Reading the
   frames pauses Flame briefly (about 0.25 s a frame); for 10 frames or more a
   dialog says how long first. Track From / To
   are filled with the whole clip; type a shorter range if you like. Tracked
   frames show as you scrub; the node message counts the frames done.
   Pressing Track Range during a run stops it; pressing again continues.
   Stepping one frame past the tracked range tracks that frame too.
5. **Batch Start** (default 1001) is the number Flame shows for the clip's
   first frame. Set it when your Batch starts elsewhere: Track From / To and
   the messages use these numbers, and changing it refills Track From / To
   with the whole clip. (OpenFX gives the plugin only clip times from 0, so it
   can't read Batch's start frame itself.)

Tracked objects are numbered in list order: with Object Channels, tracked
object 1 is R, 2 is G, 3 is B and 4 is the Matte. An object that leaves the
frame is reported in the node message and its channel stays black until it
returns. The node message shows the tracked range; **Reset Track** starts over; changing the objects, prompt or anchor
starts over by itself.

## Local or Remote

- **Local** runs SAM3 on the workstation GPU through Flame's ONNX Runtime
  (one mask, or one object per output channel). With *Free VRAM/Frame* on it
  holds about 1.5 GB between frames and peaks near 4 GB during a frame (fp16
  encoder, 24 GB NVIDIA GPU: about 0.45 s a frame). *Release GPU Memory* unloads it.
  Without a GPU it runs on the CPU, much slower.
- **Remote** sends the frame to a SAM3 server (**Server URL**, or set once at
  install). The SAM3 server is not part of this release yet; until it is, use
  Local.

## Models

The SAM3 ONNX files come in four downloads: `segment-sam3` (fp16 image
encoder for the GPU and the decoder), `segment-sam3-text` (text encoder,
detector and tokenizer files, needed for text prompts) and, for CPU-only
machines, `segment-sam3-cpu` (fp32 encoder). The installer picks the first
two by default. Tracking needs `segment-sam3-track` (86 MB, the tracker's
memory networks), also a default. SAM3 is published by Meta under its own licence (SAM License,
see `licenses/`); the installer asks you to accept it.

## Tips

- A short, concrete text prompt works best ("woman" rather than "the woman
  on the left in a red coat"). Use **Pick** or **Best Instance Only** to get
  one instance.
- Objects are numbered per frame. **Concept #** (the default for the output
  channels) keeps each concept in its channel even when objects come and go.
- Feed the mask into [cwRefine](cwRefine.md) for hair and soft edges.

## Download and install

**[cwSegment for Linux](../../../releases/latest/download/cwSegment-linux.tar.gz)** ·
**[cwSegment for macOS](../../../releases/latest/download/cwSegment-macos.tar.gz)** ·
[all releases](../../../releases)

Unpack and run `./install.sh`. It asks where to install, with a default
for every answer, and adds cwSegment to an existing cwTools install.
The installer then downloads the models it needs and checks them, or uses
a [shared models folder](../README.md#models).

Prefer everything at once? Get the [full suite](../README.md#full-suite).

[← all tools](../README.md)
