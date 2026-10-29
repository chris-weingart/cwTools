// cwExtract -- object mattes from a cwSegment "Object IDs" map.
//
// The ID map stores object k (1..50) as the value k * 0.02 in R (= G = B),
// background 0: hard, one object per pixel. Each pixel is decoded to the
// nearest ID, so small value drift (float16, 8-bit, a lossless re-render)
// doesn't matter. Filtering the ID map (resize, blur, motion blur, colour
// transforms) blends neighbouring IDs and is NOT allowed upstream.
//
// Mode 0 RGB + Matte: R / G / B / Matte = the objects picked for each channel.
// Mode 1 One Object:  that object in R, G, B and Matte.
// Mode 2 Range:       objects From..To (inclusive) as one matte in all channels.
#version 430
layout ( location = 0 ) out vec4 fragColor;

layout( binding = 1 ) uniform AdskUniformBlock
{
   float adsk_result_w;
   float adsk_result_h;
};

layout( binding = 3 ) uniform sampler2D idTex;   // input 1 (Front): cwSegment Object IDs

layout( binding = 2 ) uniform UniformBlock
{
   int mode;
   int objR;
   int objG;
   int objB;
   int objMatte;
   int objOne;
   int rangeFrom;
   int rangeTo;
   bool invertOut;
};

const float ID_SCALE = 50.0;   // object k is stored as k / 50 (cwSegment Object IDs)

int idAt( ivec2 p )
{
   ivec2 sz = textureSize( idTex, 0 );
   float v;
   if ( sz == ivec2( int( adsk_result_w ), int( adsk_result_h ) ) )
      v = texelFetch( idTex, p, 0 ).r;
   else  // different size: nearest sample at this pixel's position (never interpolate IDs)
      v = texelFetch( idTex, clamp( ivec2( ( vec2( p ) + 0.5 ) * vec2( sz ) / vec2( adsk_result_w, adsk_result_h ) ), ivec2( 0 ), sz - 1 ), 0 ).r;
   return int( floor( v * ID_SCALE + 0.5 ) );
}

float is( int id, int want )
{
   return ( want > 0 && id == want ) ? 1.0 : 0.0;
}

void main()
{
   int id = idAt( ivec2( gl_FragCoord.xy ) );
   vec4 o;
   if ( mode == 1 )
      o = vec4( is( id, objOne ) );
   else if ( mode == 2 )
   {
      int a = min( rangeFrom, rangeTo );
      int b = max( rangeFrom, rangeTo );
      o = vec4( ( id > 0 && id >= a && id <= b ) ? 1.0 : 0.0 );
   }
   else
      o = vec4( is( id, objR ), is( id, objG ), is( id, objB ), is( id, objMatte ) );
   if ( invertOut )
      o = 1.0 - o;
   fragColor = o;
}
