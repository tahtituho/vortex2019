#version 330 core

out vec4 FragColor;

uniform float act;
uniform vec3 fade;
uniform float time;
uniform vec2 resolution;

uniform vec3 cameraPosition;
uniform vec3 cameraLookAt;
uniform float cameraFov;

uniform float rayMaxSteps;
uniform float rayThreshold;

uniform vec3 lightPosition;

uniform vec3 fogColor;
uniform float fogIntensity;

uniform vec3 groupNamePosition;
uniform vec3 groupNameRotation;

uniform vec3 demoNamePosition;
uniform vec3 demoNameRotation;

uniform sampler2D demoName;
uniform sampler2D groupLogo;

uniform vec3 makersPosition;
uniform vec3 makersOffset;
uniform float makersTexture;
uniform vec3 demonamePosition;
uniform vec3 demonameOffset;

// Scene 10
uniform float terrainType;
uniform float ripplePos;
uniform float snakeLength;
uniform float snakePos;
uniform float applePos;
uniform float randPopper;

uniform sampler2D tunnelTex;
uniform sampler2D tunnelTexNm;
uniform sampler2D bogdanLogo;
uniform sampler2D bogdan;
uniform sampler2D cassetteLabel;
uniform sampler2D adaptGreets;
uniform sampler2D convergeGreets;
uniform sampler2D exoticMenGreets;
uniform sampler2D wideLoadGreets;
uniform sampler2D codeColho;
uniform sampler2D codeHelgrima;
uniform sampler2D musicMajaniemi;
uniform sampler2D gfxWartti;

in float[12] sines;
in float[12] coses;
in float random;


#define PI 3.14159265359

struct vec2Tuple {
    vec2 first;
    vec2 second;
};

struct vec3Tuple {
    vec3 first;
    vec3 second;
};

struct textureOptions {
    int index;
    vec3 offset;
    vec3 scale;
    bool normalMap;
};

struct material {
    vec3 ambient;
    float ambientStrength;

    vec3 diffuse;
    float diffuseStrength;

    vec3 specular;
    float specularStrength;
    float shininess;

    float shadowHardness;
    bool receiveShadows;
    float shadowLowerLimit;
    float shadowUpperLimit;

    textureOptions textureOptions;
};

struct entity {
    float dist;
    vec3 point;
    bool needNormals;
    material material;
};

struct hit {
    vec3 point;
    vec3 normal;

    float steps;
    float dist;
    
    float last;

    entity entity;
};

//
// Description : Array and textureless GLSL 2D/3D/4D simplex 
//               noise functions.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : stegu
//     Lastmod : 20110822 (ijm)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//               https://github.com/stegu/webgl-noise
// 

float rand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

float hash(vec2 p) {
     return fract(sin(1.0 + dot(p, vec2(127.1, 311.7))) * 43758.545);
}

vec3 mod289(vec3 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289(vec4 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x) {
     return mod289(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt(vec4 r) {
  return 1.79284291400159 - 0.85373472095314 * r;
}

float snoise(vec3 v) { 
    const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
    const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

    // First corner
    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 =   v - i + dot(i, C.xxx) ;

    // Other corners
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );

    //   x0 = x0 - 0.0 + 0.0 * C.xxx;
    //   x1 = x0 - i1  + 1.0 * C.xxx;
    //   x2 = x0 - i2  + 2.0 * C.xxx;
    //   x3 = x0 - 1.0 + 3.0 * C.xxx;
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
    vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

    // Permutations
    i = mod289(i); 
    vec4 p = permute( permute( permute( 
                i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
            + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
            + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
    float n_ = 0.142857142857; // 1.0/7.0
    vec3  ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );

    //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
    //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);

    //Normalise gradients
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix final noise value
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                dot(p2,x2), dot(p3,x3) ) );
}

//https://computergraphics.stackexchange.com/questions/4686/advice-on-how-to-create-glsl-2d-soft-smoke-cloud-shader
float fbm3D(vec3 P, float frequency, float lacunarity, int octaves, float addition)
{
    float t = 0.0f;
    float amplitude = 1.0;
    float amplitudeSum = 0.0;
    for(int k = 0; k < octaves; ++k)
    {
        t += min(snoise(P * frequency)+addition, 1.0) * amplitude;
        amplitudeSum += amplitude;
        amplitude *= 0.5;
        frequency *= lacunarity;
    }
    return t/amplitudeSum;
}

//Source http://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
float opSmoothUnion(float d1, float d2, float k) {
    float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

float opSmoothSubtraction(float d1, float d2, float k) {
    float h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    return mix(d2, -d1, h) + k * h * (1.0 - h);
}

float opSmoothIntersection(float d1, float d2, float k) {
    float h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0 );
    return mix(d2, d1, h) + k * h * (1.0 - h);
}

entity opSmoothUnion(entity m1, entity m2, float k, float threshold) {
    float h = opSmoothUnion(m1.dist, m2.dist, k);
    if (smoothstep(m1.dist, m2.dist, h + threshold) > 0.5) {
        m2.dist = h;
        return m2;
    }
    else {
        m1.dist = h;
        return m1;
    }
}

entity opSmoothSubtraction(entity m1, entity m2, float k, float threshold) {
    float h = opSmoothSubtraction(m1.dist, m2.dist, k);
    if (smoothstep(m1.dist, m2.dist, h + threshold) > 0.5) {
        m2.dist = h;
        return m2;
    }
    else {
        m1.dist = h;
        return m1;
    }
}

entity opSmoothIntersection(entity m1, entity m2, float k, float threshold) {
    float h = opSmoothIntersection(m1.dist, m2.dist, k);
    if (smoothstep(m1.dist, m2.dist, h + threshold) > 0.5) {
        m2.dist = h;
        return m2;
    }
    else {
        m1.dist = h;
        return m1;
    }
}

vec3 opTwist(vec3 p, float angle)
{
    float c = cos(angle * p.y);
    float s = sin(angle * p.y);
    mat2 m = mat2(c, -s, s, c);
    vec3 q = vec3(m * p.xz, p.y);
    return q;
}

vec3 opBend(vec3 p, float angle)
{
    float c = cos(angle * p.y);
    float s = sin(angle * p.y);
    mat2 m = mat2(c, -s, s, c);
    vec3 q = vec3(m * p.xy, p.z);
    return q;
}

float opRound(float p, float rad)
{
    return p - rad;
}

//Distance functions to creat primitives to 3D world
//Source http://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
float sdPlane(vec3 p, vec3 pos, vec4 n)
{
  // n must be normalized
    vec3 p1 = vec3(p) + pos;
    return dot(p1, n.xyz) + n.w;
}

float sdSphere(vec3 p, vec3 pos, float radius)
{
    vec3 p1 = vec3(p) + pos;
    return length(p1) - radius;
}

float sdEllipsoid(vec3 p, vec3 pos, vec3 r)
{
    vec3 p1 = p + pos;
    float k0 = length(p1 / r);
    float k1 = length(p1 / (r * r));
    return k0 * (k0 - 1.0) / k1;
}

float sdBox(vec3 p, vec3 pos, vec3 b, float r)
{   
    vec3 p1 = vec3(p) + pos;
    vec3 d = abs(p1) - b;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0)) - r;
}

float sdTorus(vec3 p, vec3 pos, vec2 t)
{   
    vec3 p1 = vec3(p) + pos;
    vec2 q = vec2(length(p1.xz)-t.x,p1.y);
    return length(q)-t.y;
}

float sdCylinder(vec3 p, vec3 c, float r)
{
    return length(p.xz - c.xy) - c.z - r;
}

float sdCappedCylinder(vec3 p, vec2 size, float r)
{
  vec2 d = abs(vec2(length(p.xz), p.y)) - size;
  return min(max(d.x ,d.y), 0.0) + length(max(d, 0.0)) - r;
}

float sdRoundCone(in vec3 p, vec3 pos,in float r1, float r2, float h)
{    
    vec3 p1 = vec3(p) + pos;
    vec2 q = vec2( length(p1.xz), p1.y );
    
    float b = (r1-r2)/h;
    float a = sqrt(1.0-b*b);
    float k = dot(q,vec2(-b,a));
    
    if( k < 0.0 ) return length(q) - r1;
    if( k > a*h ) return length(q-vec2(0.0,h)) - r2;
        
    return dot(q, vec2(a,b) ) - r1;
}

float sdCapsule(vec3 p, vec3 pos, vec3 a, vec3 b, float r)
{   
    vec3 p1 = vec3(p) + pos;
    vec3 pa = p1 - a, ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h ) - r;
}

float sdHexPrism(vec3 p, vec2 h)
{
    vec3 q = abs(p);
    return max(q.z - h.y, max((q.x * 0.866025 + q.y * 0.5), q.y) - h.x);
}

float sdPyramid(vec3 p, float h)
{
    float m2 = h * h + 0.25;

    p.xz = abs(p.xz);
    p.xz = (p.z > p.x) ? p.zx : p.xz;
    p.xz -= 0.5;

    vec3 q = vec3(p.z, h * p.y - 0.5 * p.x, h * p.x + 0.5 * p.y);

    float s = max(-q.x, 0.0);
    float t = clamp((q.y - 0.5 * p.z) / (m2 + 0.25), 0.0, 1.0);

    float a = m2 * (q.x + s) * (q.x + s) + q.y * q.y;
    float b = m2 * (q.x + 0.5 * t) * (q.x + 0.5 * t) + (q.y - m2 * t) * (q.y - m2 * t);

    float d2 = min(q.y, -q.x * m2 - q.y * 0.5) > 0.0 ? 0.0 : min(a,b);

    return sqrt((d2 + q.z * q.z) / m2) * sign(max(q.z, -p.y));
}

float sdOctahedron(vec3 p, float s)
{
    p = abs(p);
    float m = p.x + p.y + p.z - s;
    vec3 q;
    if(3.0 * p.x < m ) q = p.xyz;
    else if(3.0 * p.y < m ) q = p.yzx;
    else if(3.0 * p.z < m ) q = p.zxy;
    else return m*0.57735027;

    float k = clamp(0.5 * (q.z - q.y + s), 0.0, s); 
    return length(vec3(q.x, q.y - s + k, q.z - k)); 
}

entity mMandleBox(vec3 path, material material, float size, float scale, float minrad, float limit, float factor, int iterations, float foldingLimit, float radClamp1, float radClamp2)
{
    vec4 scalev = vec4(size) / minrad;
    float absScalem1 = abs(scale - 1.0);
    float absScaleRaisedTo1mIters = pow(abs(scale), float(1 - iterations));
    vec4 p = vec4(path, 1.0), p0 = p;
 
    for (int i = 0; i < iterations; i++)
    {
        p.xyz = clamp(p.xyz, -limit, limit) * factor - p.xyz;
        float r2 = dot(p.xyz, p.xyz);
        p *= clamp(max(minrad / r2, minrad), radClamp1, radClamp2);
        p = p * scalev + p0;
        if (r2 > foldingLimit) {
            break;
        } 
   }
   entity e;
   e.dist =  ((length(p.xyz) - absScalem1) / p.w - absScaleRaisedTo1mIters);
   e.material = material;
   e.point = p.xyz;
   return e;
}

float sdMandlebulb(vec3 p, vec3 pos, float pwr, float dis, float bail, int it) {
    vec3 z = p + pos;
 
    float dr = 1.0;
    float r = 0.0;
    float power = pwr + dis;
    for (int i = 0; i < it; i++) {
        r = length(z);
        if (r > bail) break;
        
        // convert to polar coordinates
        float theta = acos(z.z/r);
        float phi = atan(z.y,z.x);
        dr =  pow(r, power - 1.0) * power * dr + 1.0;
        
        // scale and rotate the point
        float zr = pow(r, power);
        theta = theta*power;
        phi = phi*power;
        
        // convert back to cartesian coordinates
        z = zr * vec3(sin(theta)*cos(phi), sin(phi)*sin(theta), cos(theta));
        
        z += (p + pos);
    }
    return (0.5 * log(r) * r / dr);
}

float displacement(vec3 p, vec3 m)
{
    return sin(m.x * p.x) * sin(m.y * p.y) * sin(m.z * p.z);
}

float impulse(float x, float k)
{
    float h = k * x;
    return h * exp(1.0 - h);
}

float sinc(float x, float k)
{
    float a = PI * k * x - 1.0;
    return sin(a) / a;
}

vec3 rotX(vec3 p, float a)
{
    float s = sin(a);
    float c = cos(a);
    return vec3(
        p.x,
        c*p.y-s*p.z,
        s*p.y+c*p.z
    );
}

vec3 rotY(vec3 p, float a)
{
    float s = sin(a);
    float c = cos(a);
    return vec3(
        c*p.x+s*p.z,
        p.y,
        -s*p.x+c*p.z
    );
}
 
vec3 rotZ(vec3 p, float a)
{
    float s = sin(a);
    float c = cos(a);
    return vec3(
        c*p.x-s*p.y,
        s*p.x+c*p.y,
        p.z
    );
}

vec3 rot(vec3 p, vec3 a) {
    return rotX(rotY(rotZ(p, a.z), a.y), a.x);
}

vec3 twistX(vec3 p, float angle) {
    return rotX(p, p.x * angle);
}

vec3 twistY(vec3 p, float angle) {
    return rotY(p, p.y * angle);
}

vec3 twistZ(vec3 p, float angle) {
    return rotZ(p, p.z * angle);
}

vec3 translate(vec3 p, vec3 p1) {
    return p + (p1 * -1.0);
}

vec3 scale(vec3 p, float s) {
    vec3 p1 = p;
    p1 /= s;
    return p1;
} 

float rand(float n){return fract(sin(n) * 43758.5453123);}

float noise(float p){
	float fl = floor(p);
    float fc = fract(p);
	return mix(rand(fl), rand(fl + 1.0), fc);
}

vec3Tuple repeat(vec3 p, vec3 size) {
    vec3 c = floor((p + size * 0.5 ) / size);
    vec3 path1 = mod(p + size * 0.5, size) - size * 0.5;
    return vec3Tuple(path1, c);
}

vec2Tuple repeatPolar(vec2 p, float repetitions) {
	float angle = 2 * PI / repetitions;
	float a = atan(p.y, p.x) + angle / 2.0;
	float r = length(p);
	float c = floor(a / angle);
	a = mod(a, angle) - angle / 2.0;
	vec2 path = vec2(cos(a), sin(a)) * r;
	// For an odd number of repetitions, fix cell index of the cell in -x direction
	// (cell index would be e.g. -5 and 5 in the two halves of the cell):
	if (abs(c) >= (repetitions / 2.0)) {
        c = abs(c);
    } 
	return vec2Tuple(path, vec2(c));
}

entity opUnion(entity m1, entity m2) {
    return m1.dist < m2.dist ? m1 : m2;
}

entity opSubtraction(entity m1, entity m2) {
    if(-m1.dist > m2.dist) {
        m1.dist *= -1.0;
        return m1;
    }
    else {
        return m2;
    }
    
}

entity opIntersection(entity m1, entity m2) {
    return m1.dist > m2.dist ? m1 : m2;
}

vec3 planeFold(vec3 z, vec3 n, float d) {
    vec3 z1 = z;
	z1.xyz -= 2.0 * min(0.0, dot(z1.xyz, n) - d) * n;
    return z1;
}

vec3 absFold(vec3 z, vec3 c) {
    vec3 z1 = z;
	z1.xyz = abs(z1.xyz - c) + c;
    return z1;
}

vec3 sierpinskiFold(vec3 z) {
    vec3 z1 = z;
	z1.xy -= min(z1.x + z1.y, 0.0);
	z1.xz -= min(z1.x + z1.z, 0.0);
	z1.yz -= min(z1.y + z1.z, 0.0);
    return z1;
}

vec3 mengerFold(vec3 z) {
    vec3 z1 = z;
	float a = min(z1.x - z1.y, 0.0);
	z1.x -= a;
	z1.y += a;
	a = min(z1.x - z1.z, 0.0);
	z1.x -= a;
	z1.z += a;
	a = min(z1.y - z1.z, 0.0);
	z1.y -= a;
	z1.z += a;
    return z1;
}

vec3 sphereFold(vec3 z, float minR, float maxR) {
    vec3 z1 = z;
	float r2 = dot(z1.xyz, z1.xyz);
	z1 *= max(maxR / max(minR, r2), 1.0);
    return z1;
}

vec3 boxFold(vec3 z, vec3 r) {
    vec3 z1 = z;
	z1.xyz = clamp(z1.xyz, -r, r) * 2.0 - z1.xyz;
    return z1;
}

vec2 spiral(float n) {
    float k=ceil((sqrt(n)-1)/2);
    float t=2*k+1;
    float m=pow(t, 2);
    t=t-1;
    if (n>=m-t) {
        return vec2(k-(m-n),-k);
    }
    else {
        m = m-t;
    }
    if (n>=m-t) {
        return vec2(-k,-k+(m-n));
    }   
    else {
        m = m-t;
    }
    if (n>=m-t) {
        return vec2(-k+(m-n),k);
    } 
    else {
        return vec2(k,k-(m-n-t));
    }
}

entity mPlane(vec3 p, vec3 pos, vec4 n, material material)
{
    entity m;
    vec3 p1 = p;
    m.dist = sdPlane(p, pos, n);
    m.point = p1;
    m.material = material;
    return m;
}

entity mSphere(vec3 path, float radius, float scale, material material) {
    entity m;
    vec3 p1 = path / scale;
    m.dist = sdSphere(p1, vec3(0.0), radius) * scale;
    m.point = p1;
    m.material = material;
    return m;
}

entity mBox(vec3 path, vec3 size, float r, float scale, material material) {
    entity m;
    vec3 p1 = path / scale;
    m.dist = sdBox(p1, vec3(0.0), size, r) * scale;
    m.point = p1;
    m.material = material;
    return m;
}

entity mCapsule(vec3 p, vec3 pos, vec3 a, vec3 b, float r, material material) {
    entity m;
    vec3 p1 = p;
    m.dist = sdCapsule(p, pos, a, b, r);
    m.point = p1;
    m.material = material;
    return m;
}

float vmax(vec3 v) {
	return max(max(v.x, v.y), v.z);
}

entity mBoxCheap(vec3 p, vec3 b, material material) {
    entity m;
    m.dist = vmax(abs(p) - b);
    m.point = p;
    m.material = material;
    return m;
}

entity mCappedCylinder(vec3 path, vec2 size, float r, float scale, material material) {
    entity m;
    vec3 p1 = path / scale;
    m.dist = sdCappedCylinder(p1, size, r) * scale;
    m.point = p1;
    m.material = material;
    return m;
}

entity mHexPrim(vec3 path, vec2 size, float scale, material material) {
    entity m;
    vec3 p1 = path / scale;
    m.dist = sdHexPrism(p1, size) * scale;
    m.point = p1;
    m.material = material;
    return m;
}

entity mPyramid(vec3 path, float height, float scale, material material) {
    entity m;
    vec3 p1 = path / scale;
    m.dist = sdPyramid(p1, height) * scale;
    m.point = p1;
    m.material = material;
    return m;
}

entity mCylinder(vec3 path, vec3 size, float r, material material) {
    entity m;
    vec3 p1 = path;
    m.dist = sdCylinder(path, size, r);
    m.point = p1;
    m.material = material;
    return m;
}


entity mTerrain(vec3 path, vec3 par, material material) {
    entity m;
    float s = 5.0;
    vec3Tuple p1 = repeat(path, vec3(s * 2.5, 0.0, s * 2.5));

    // Ripple effect
    if (terrainType == 1) {
        float timer = floor(ripplePos);
        float midtotimer = floor(length(p1.second.xz));
        float ramp = midtotimer + 1;
        float ramp2 = midtotimer - 1;
        if (midtotimer == timer && timer > 0) {
            material.ambient = vec3(1.0, 1.0, 1.0);
            m = mBox(translate(p1.first, vec3(0.0, 10, 0.0)), vec3(s, s, s), 0.05, 1.0, material);
        }
        else if ((midtotimer == (timer - 5) || midtotimer == (timer + 5)) && timer > 0) {
            material.ambient = vec3(1.0, 0.0, 0.0);
            m = mBox(translate(p1.first, vec3(0.0, 5, 0.0)), vec3(s, s, s), 0.05, 1.0, material);
        }
        else if (ramp == timer || ramp2 == timer && timer > 0) {
            material.ambient = vec3(1.0, 0.0, 0.0);
            m = mBox(translate(p1.first, vec3(0.0, 2.0, 0.0)), vec3(s, s, s), 0.05, 1.0, material);
        }
        else {
            material.ambient = vec3(0.2, 0.2, 0.2);
            m = mBox(p1.first, vec3(s, s, s), 0.05, 1.0, material);
        }
    }
    // Random boxes
    else if (terrainType == 2) {
        vec2 randomizer = vec2(rand(p1.second.xz), randPopper);
        if ( rand(randomizer) < 0.1) {
            material.ambient = vec3(1.0, 0.0, 0.0);
            float coeff = 1;
            m = mBox(translate(p1.first, sin(time * randPopper)*vec3(0.0, rand(p1.second.xz) * 5, 0.0)), vec3(s, s, s), 0.05, 1.0, material);
        }
        else {
            material.ambient = vec3(0.2, 0.2, 0.2);
            m = mBox(p1.first, vec3(s, s, s), 0.05, 1.0, material);
        }
    }
    // Spiral worm
    else if (terrainType == 3) {
        material.ambient = vec3(0.2, 0.2, 0.2);
        m = mBox(p1.first, vec3(s, s, s), 0.05, 1.0, material);
        vec2 applePosV = vec2(0.0, 0.0);
        if (applePos > 0) {
            applePosV = spiral(floor(applePos));
        }
        if (fade.x == 2) {
            s = 1;
        }
        for (int y = 0; y < snakeLength; y++) {
            vec2 spiralPos = spiral(floor(snakePos) - y);
            if (spiralPos == p1.second.xz) {
                material.ambient = vec3(1.0, 0.0, 0.0);
                m = mBox(translate(p1.first, vec3(0.0, 5, 0.0)), vec3(s, s, s), 0.05, 1.0, material);
            }
            else if (applePosV == p1.second.xz && applePos > 0) {
                material.ambient = vec3(0.0, 1.0, 0.0);
                m = mBox(translate(p1.first, vec3(0.0, 5, 0.0)), vec3(s, s, s), 0.05, 1.0, material);
            }
        }
    }
    m.point = p1.first;
    return m;
}


entity tunnelSegment(vec3 path, float r1, float r2, float h, float notch, int numberOfNotches, float scale) {
    material m1 = material(
        vec3(1.0, 1.0, 0.0),
        20.0,

        vec3(0.3, 0.3, 0.0),
        100.0,

        vec3(1.0, 1.0, 1.0),
        50000.0,
        80.0,

        0.3,
        true,
        0.25,
        10.0,
        textureOptions(
            0,
            vec3(1.5, 1.5, 1.5),
            vec3(2.0, 2.0, 2.0),
            false
        )
    );
    entity cs1 = mCappedCylinder(path, vec2(r1, h), 0.0, scale, m1);
    cs1.needNormals = true;
    entity cs2 = mCappedCylinder(path, vec2(r2, h + 1.0), 0.0, scale, m1);
    cs1.needNormals = true;
    vec2Tuple notchBoxes = repeatPolar(path.xz, numberOfNotches);
    entity e3 = mBox(vec3(notchBoxes.first, path.y) - vec3(r2, 0.0, 0.0), vec3(notch), 0.0, 1.0, m1);
    e3.needNormals = true;
    return opSmoothSubtraction(e3, opSmoothSubtraction(cs2, cs1, 0.05, 0.5), 0.05, 0.5);

}  

entity tunnel(vec3 path, float time) {
    vec3Tuple repeated = repeat(path, vec3(0.0, 1.0, 0.0));
    vec3 tunnelPosition = repeated.first;
    tunnelPosition = translate(tunnelPosition, vec3(0.0, 0.0, sin(repeated.second.y / 5.0) * 1.0));
    return tunnelSegment(rot(tunnelPosition, vec3(0.0, mod(repeated.second.y, 8) * (time / 20.0), 0.0)), 2.0, 1.5, 0.2, 0.25, 6 + int(mod(repeated.second.y, 5) * 2.0), 1.0);   
}


entity mCappedCylinder(vec3 path, vec2 size, float r, material material) {
    entity m;
    vec3 p1 = path;
    m.dist = sdCappedCylinder(path, size, r);
    m.point = p1;
    m.material = material;
    return m;
}


entity mCasette(vec3 path, float scale, float time) {
    material bodyMat = material(
        vec3(0.25, 0.25, 0.25),
        0.2,

        vec3(0.25, 0.25, 0.25),
        0.2,

        vec3(0.0, 1.0, 0.2),
        0.0,
        0.0,

        0.2,
        false,
        1.5,
        5.5,
        textureOptions(
            0,
            vec3(1.5, 1.5, 1.5),
            vec3(2.0, 2.0, 2.0),
            false
        )
    );
    material labelMat = material(
        vec3(1.00, 1.00, 1.00),
        1.0,

        vec3(1.00, 1.00, 1.00),
        2.2,

        vec3(1.0, 1.0, 1.0),
        0.0,
        0.0,

        0.2,
        false,
        1.5,
        5.5,
        textureOptions(
            51,
            vec3(0.65, 0.45, 0.0),
            vec3(88.0, 150.0, 5.0),
            false
        )
    );
    material tapeMat = material(
        vec3(1.0, 1.0, 1.0),
        1.0,

        vec3(1.0, 1.0, 1.0),
        0.0,

        vec3(1.0, 1.0, 1.0),
        0.0,
        0.0,

        0.2,
        false,
        1.5,
        5.5,
        textureOptions(
            52,
            vec3(0.5, 0.5, 0.5),
            vec3(0.2, 0.2, 0.2),
            false
        )
    );
    vec3 sPath = path / scale;
    float gearDir = sign(sPath.x);
    sPath.x = -abs(sPath.x);
    float bodyBase = sdBox(translate(sPath, vec3(-50.1, 0.0, 0.0)), vec3(0.0, 0.0, 0.0), vec3(50.1, 63.8, 8.6), 1.8);
    float bodySideOverHang = sdBox(translate(sPath, vec3(-100.2 + 2.5, -41.0, 0.0)), vec3(0.0, 0.0, 0.0), vec3(5.0, 17.7, 2.7), 1.8);
    float bodyLowerOverHang = sdBox(translate(sPath, vec3(-34.5, -63.8 + 15.5, 0.0)), vec3(0.0), vec3(34.5, 15.5, 12.0), 1.8);
    float bodyLowerOverHangDel = sdBox(rotZ(translate(sPath, vec3(-79.0, -63.8 + 15.5, 0.0)), 0.17), vec3(0.0), vec3(15.0, 20.0, 20.0), 0.0);
    float bodyLowerOverHangUnion = opSmoothSubtraction(bodyLowerOverHangDel, bodyLowerOverHang, 0.0);

    float centerHole = sdBox(translate(sPath, vec3(-11.5, 12.9, 0.0)), vec3(0.0), vec3(12.5, 8.4, 15.0), 1.8);
    float gearHole = sdCappedCylinder(translate(rotX(sPath, 1.5708), vec3(-21.3 - 22.0, 0.0, 11.0)), vec2(11.0, 15.0), 0.0); 
    float pinHole1 = sdCappedCylinder(translate(rotX(sPath, 1.5708), vec3(-48.5, 0.0, -59.0)), vec2(4.6, 40.0), 0.0);
    float pinHole2 = sdBox(translate(rotX(sPath, 1.5708), vec3(-30.0, 0.0, -55.0)), vec3(0.0), vec3(4.2, 40.0, 4.2), 0.0);
    float bodyLowerEmpty = sdBox(translate(sPath, vec3(-27.0, -59.0, 0.0)), vec3(0.0), vec3(37.0, 8.0, 10.0), 0.0);
    float copyHole = sdBox(translate(sPath, vec3(-90.0, 65.0, 0.0)), vec3(0.0), vec3(6.25, 5.0, 5.0), 0.0);
   
    float bottom = sdBox(translate(sPath, vec3(-25.0, -63.6, 0.0)), vec3(0.0, 0.0, 0.0), vec3(13.0, 2.0, 12.0), 0.0);
    vec2Tuple gearRepeat = repeatPolar(rotZ(translate(sPath, vec3(-45.0, 11.0, 0.0)), time * gearDir).xy, 6);
    float gears = sdBox(vec3(gearRepeat.first, sPath.z) - vec3(9.0, 0.0, 0.0), vec3(0.0), vec3(1.5, 1.5, 1.5), 0.0); 
    float body = opSmoothUnion(bottom, opSmoothSubtraction(copyHole, opSmoothSubtraction(bodyLowerEmpty, opSmoothUnion(gears, opSmoothSubtraction(pinHole2, opSmoothSubtraction(pinHole1, opSmoothSubtraction(gearHole, opSmoothSubtraction(centerHole, opSmoothUnion(opSmoothUnion(bodyBase, bodySideOverHang, 0.0), bodyLowerOverHangUnion, 0.0), 0.0), 0.0), 0.0), 0.0), 0.0), 0.0), 0.0), 0.0) * scale;
   
    entity label;
    float labelPaper = sdBox(translate(sPath, vec3(0.0, 16.0, -11.5)), vec3(0.0), vec3(93.0, 42.0, 2.0), 0.5);
    float labelHole = sdBox(translate(sPath, vec3(0.0, 12.0, -11.5)), vec3(0.0), vec3(53.0, 9.0, 2.0), 6.0);
    label.dist = opSmoothSubtraction(labelHole, labelPaper, 0.0) * scale;
    label.point = rotZ(path / scale, 1.5708);
    label.material = labelMat;
    label.needNormals = true;
   
    entity cass;
    cass.dist = body;
    cass.point = sPath;
    cass.material = bodyMat;
    cass.needNormals = true;
   
    entity tape;
    vec3 tapePath = path / scale;
    float tapeX = smoothstep(-65.0, 65.0, tapePath.x);
    //tapePath = rotZ(tapePath, sin(tapeX * 5.0) * cos(tapeX * 2.2));
    //tapePath = rotZ(tapePath, cos(tapeX * 2.0) * cos(tapeX * 4.2));
    //tapePath = rotZ(tapePath, pow(tapeX, 2.0)); 
    tapePath = translate(tapePath, vec3(0.0, (pow(tapeX - 0.5, 2.0)) * 150.0, 0.0));
   
    tape.dist = sdBox(translate(tapePath, vec3(0.0, -99.6, 0.0)), vec3(0.0), vec3(70.0, 0.2, 8.6), 0.0) * scale;
    tape.material = tapeMat;
    tape.needNormals = true;
    //return tape;
    return opSmoothUnion(tape, opSmoothSubtraction(label, cass, 0.0, 0.0), 0.0, 0.0);
}

entity mMandleMaze(vec3 path, float time, float scale) {
    material m1 = material(
        vec3(0.3, 0.3, 0.3),
        5.0,

        vec3(1.0, 1.0, 1.0),
        5.0,

        vec3(0.0, 0.0, 1.0),
        500.0,
        5.0,

        0.9,
        true,
        0.5,
        5.5,
        textureOptions(
            0,
            vec3(1.5, 1.5, 1.5),
            vec3(2.0, 2.0, 2.0),
            false
        )
    );

    vec3Tuple rPath = repeat(path, vec3(4.0));
    vec3 sPath = rPath.first / scale;
    float offset = rPath.second.z / 10.0;// + (sin(time) / 10.0);
    entity maze = mMandleBox(sPath, m1, 2.0, 2.2, 0.15, 2.6  + offset, 1.6, 15, 100.0, 0.18 + offset, 1.0);
    maze.dist *= scale;
    maze.needNormals = true;
    maze.point = sPath;

    entity mazeCut = mSphere(sPath, 13.5, 1.0, m1);
    mazeCut.dist *= scale;
    mazeCut.needNormals = true;
    mazeCut.point = sPath;
    return opSmoothSubtraction(mazeCut, maze, 0.0, 0.0);
}

entity banner(vec3 path, float scale, int texture, vec3 offset) {
    material m1 = material(
        vec3(0.5, 0.5, 0.5),
        1.0,

        vec3(1.0, 1.0, 1.0),
        1.2,

        vec3(1.0, 1.0, 1.0),
        1.0,
        20.0,

        0.8,
        false,
        1.5,
        2.0,
        textureOptions(
            texture,
            offset,
            vec3(1.0, 1.0, 1.0),
            false
        )
    );
    entity p;
    vec3 size = vec3(1.0, 1.0, 1.0);
    float p1 = sdBox(path / scale, vec3(0.0), size, 0.0);

    p.point = path;
    p.dist = p1 * scale;
    p.material = m1;
    return p;
}

entity scene(vec3 path, vec2 uv)
{   
    int a = int(act);
    if(a == 1) {
        vec3 groupPath = translate(rot(path, groupNameRotation), groupNamePosition);
        material groupMat = material(
            vec3(1.0, 1.0, 1.0),
            1.0,

            vec3(1.0, 1.0, 1.0),
            1.0,

            vec3(1.0, 1.0, 1.0),
            0.0,
            0.0,

            0.2,
            false,
            1.5,
            5.5,
            textureOptions(
                10,
                vec3(0.5, 0.5, 0.5),
                vec3(2.5, 2.5 * 1.8333, 1.0),
                false
            )
        );

        entity e1 = mBox(rotZ(groupPath, 1.5708), vec3(1080.0, 1920.0, 0.1), 0.0, 1.0, groupMat);
        e1.needNormals = true;  

        vec3 demoPath = rot(translate(path, demoNamePosition), demoNameRotation);
        material demoMat = material(
            vec3(1.0, 1.0, 1.0),
            1.0,

            vec3(1.0, 1.0, 1.0),
            2.2,

            vec3(1.0, 1.0, 1.0),
            0.0,
            0.0,

            0.2,
            false,
            1.5,
            5.5,
            textureOptions(
                11,
                vec3(0.5, 0.5, 0.5),
                vec3(0.5, 0.5 * 1.8333, 1.0),
                false
            )
        );

        entity e2 = mBox(rotZ(demoPath, 1.5708), vec3(1.0 / 5.0, 1.83 / 5.0, 0.1), 0.0, 1.0, demoMat);
        e2.needNormals = true;
       
        entity comb = opSmoothUnion(e1, e2, 0.0, 0.0);
        //comb.dist += displacement(r, vec3(3.0));
        comb.needNormals = true;
        //return e2;
        return comb;
    }
    else if (a == 10) {
        material planemat = material(
            vec3(0.0, 0.0, 0.0),
            1.0,
            vec3(0.5, 0.5, 0.5),
            1.3,
            vec3(0.0, 0.0, 0.5),
            10.0,
            0.4,
            1.0, 
            true,
            2.5,
            5.5,
            textureOptions(
                0,
                vec3(0.0),
                vec3(0.0),
                false
            )
        );

        entity terrain = mTerrain(path, vec3(100.0, 20.0, 1.0), planemat);
        float saizu = 1;
        entity guard = mBoxCheap(path, vec3(saizu), planemat);
        guard.dist = -guard.dist;
        guard.dist = abs(guard.dist) + saizu * 0.1;

        return opUnion(terrain, guard);
    }
    else if (a == 11) {
        material testmat = material(
            vec3(0.9, 0.1, 0.1),
            1.0,
            vec3(0.5, 0.5, 0.5),
            1.3,
            vec3(0.0, 0.0, 0.5),
            10.0,
            0.4,
            1.0, 
            true,
            2.0,
            5.5,
            textureOptions(
                0,
                vec3(0.0),
                vec3(0.0),
                false
            )
        );
        float s = 1.0;
        vec3Tuple p1 = repeat(path, vec3(s * 3.8, 0.0, s * 3.8));

        entity guard = mBoxCheap(path, vec3(s), testmat);
        guard.dist = -guard.dist;
        guard.dist = abs(guard.dist) + s* 0.1;
        vec2 randomizer = vec2(rand(p1.second.xz), 100);
        vec3 rot = rotY(p1.first, sin(time * 2.5)* rand(p1.second.xz) * 1.5);
        entity dickles;
        if (rand(p1.second.xz) < 0.5) {
            dickles = mCapsule(rotY(rotX(rot, rand(p1.second.xz) * 0.3  * (sin(time * rand(p1.second.xz) * 26.5)) * smoothstep(0, 4, p1.first.y)), 2 * rand(p1.second.xz)), vec3(1.0, 1.0, 1.0), vec3(1.0, 4.0, 1.0), vec3(1, 0, 0), 1.0, testmat);
        
        }
        else {
            dickles = mCapsule(rotY(rotZ(rot, rand(p1.second.xz) * 0.3  * (sin(time * rand(p1.second.xz) * 26.5)) * smoothstep(0, 4, p1.first.y)), 2 * rand(p1.second.xz)), vec3(1.0, 1.0, 1.0), vec3(1.0, 4.0, 1.0), vec3(1, 0, 0), 1.0, testmat);
        
        }
         
        entity floor = mPlane(path, vec3(0, -0.2, 0), vec4(0, 1, 0, 1), testmat);

        return opSmoothUnion(floor, dickles, 2, 0.01);
    }
    else if(a == 2) {
        entity mandel = mMandleMaze(path, time, 0.2);
        return mandel;
    }
    else if(a == 4) {
        vec3 r = rot(path, vec3(PI / 2.0, 0.0, time / 10.0));
        entity e = tunnel(r - vec3(0.0, time / 0.5, 0.0), time);
        entity m = banner(translate(rot(path, vec3(time)), makersPosition), 0.5, int(makersTexture), makersOffset);
        return opSmoothUnion(m, e, 0.25, 0.0);
    }
    else if(a == 5) {
        entity cass = mCasette(rot(path, vec3(-0.35, 0.21, -0.45)), 0.02, time);
        return cass;
    }
 
} 

hit raymarch(vec3 rayOrigin, vec3 rayDirection, vec2 uv) {
    hit h;
    h.steps = 0.0;
    h.last = 100.0;
    
    for(float i = 0.0; i <= rayMaxSteps; i++) {
        h.point = rayOrigin + rayDirection * h.dist;
        h.entity = scene(h.point, uv);
        h.steps += 1.0;
        h.last = min(h.entity.dist, h.last);
        if(abs(h.entity.dist) < rayThreshold) {
            if(h.entity.needNormals == true) {
                vec2 eps = vec2(0.01, 0.0);
                h.normal = normalize(vec3(
                    scene(h.point + eps.xyy, uv).dist - h.entity.dist,
                    scene(h.point + eps.yxy, uv).dist - h.entity.dist,
                    scene(h.point + eps.yyx, uv).dist - h.entity.dist
                ));
            }
            break;
        }
        h.dist += (h.entity.dist);

    }
    
    return h;
}

vec3 ambient(vec3 color, float strength) {
    return color * strength;
} 

vec3 diffuse(vec3 normal, vec3 hit, vec3 pos, vec3 color, float strength) {
    if(strength <= 0.0) {
        return vec3(0.0);
    }
    vec3 lightDir = normalize(pos - hit);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * color * strength;
    return diffuse;
}

vec3 specular(vec3 normal, vec3 eye, vec3 hit, vec3 pos, vec3 color, float strength, float shininess) {
    if(strength <= 0.0) {
        return vec3(0.0);
    }
    vec3 lightDir = normalize(pos - hit);
    vec3 viewDir = normalize(eye - hit);
    vec3 halfwayDir = normalize(lightDir + viewDir);

    float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);
    vec3 specular = strength * spec * color;
    return specular;
} 

float shadows(vec3 ro, vec3 rd, float mint, float maxt, float k, vec2 uv) {
    float res = 1.0;
    float ph = 1e20;
    for(float t = mint; t < maxt;)
    {
        float h = scene(ro + (rd * t), uv).dist;
        if(h < rayThreshold)
            return 0.0;
        float y = h * h / (2.0 * ph);
        float d = sqrt(h * h - y * y);
        res = min(res, k * d / max(0.0, t - y));
        ph = h;
        t += h;    
    }
    return res;
}

vec4 textureCube(sampler2D sam, in vec3 p, in vec3 n)
{
	vec4 x = texture(sam, p.yz);
	vec4 y = texture(sam, p.zx);
	vec4 z = texture(sam, p.yx);
    vec3 a = abs(n);
	return (x*a.x + y*a.y + z*a.z) / (a.x + a.y + a.z);
}

vec2 planarMapping(vec3 p) {
    vec3 p1 = normalize(p);
    vec2 r = vec2(0.0);
    if(abs(p1.x) == 1.0) {
        r = vec2((p1.z + 1.0) / 2.0, (p1.y + 1.0) / 2.0);
    }
    else if(abs(p1.y) == 1.0) {
        r = vec2((p1.x + 1.0) / 2.0, (p1.z + 1.0) / 2.0);
    }
    else {
        r = vec2((p1.x + 1.0) / 2.0, (p1.y + 1.0) / 2.0);
    }
    return r;
}

vec2 cylindiricalMapping(vec3 p) {
    return vec2(atan(p.y / p.x), p.z);
}

vec2 scaledMapping(vec2 t, vec2 o, vec2 s) {
    return -vec2((t.x / o.x) + s.x, (t.y / o.y) + s.y);
}

float noise(float v, float amplitude, float frequency, float time) {
    float r = sin(v * frequency);
    float t = 0.01*(-time*130.0);
    r += sin(v*frequency*2.1 + t)*4.5;
    r += sin(v*frequency*1.72 + t*1.121)*4.0;
    r += sin(v*frequency*2.221 + t*0.437)*5.0;
    r += sin(v*frequency*3.1122+ t*4.269)*2.5;
    r *= amplitude*0.06;
    
    return r;
}

float plot(float pct, float thickness, vec2 position) {
    return smoothstep(pct - thickness, pct, position.x) - smoothstep(pct, pct + thickness, position.x);
}

vec4 background(vec2 uv) {
    int a = int(act);
    vec4 r = vec4(0.0);
    if(a == 4) {
        r.xyz = vec3(1.0);
        r.w = 0.0;
    }

    return r;
}

vec3 generateTexture(int index, vec3 point, vec3 offset, vec3 scale) {
    vec3 r = vec3(1.0);
    switch(index) {
        case 10: {
            vec3 rp = vec3((point.x / scale.x) + offset.x, (point.y / scale.y) + offset.y, (point.z / scale.z) + offset.z);
            r = textureCube(groupLogo, rp, vec3(0.0, 0.0, 0.1)).xyz;
            break;
        }
        case 11: {
            vec3 rp = vec3((point.x / scale.x) + offset.x, (point.y / scale.y) + offset.y, (point.z / scale.z) + offset.z);
            r = textureCube(demoName, rp, vec3(0.0, 0.0, 0.1)).xyz;
            break;
        }
         case 40: {
            vec3 rp = vec3((point.x / scale.x) + offset.x, (point.y / scale.y) + offset.y, (point.z / scale.z) + offset.z);
            r = textureCube(tunnelTex, rp, vec3(0.0, 0.0, 0.1)).xyz;
            break;
        }
        case 41: {
            vec3 rp = vec3((point.x / scale.x) + offset.x, (point.y / scale.y) + offset.y, (point.z / scale.z) + offset.z);
            r = textureCube(codeColho, rp, vec3(0.0, 0.0, 0.1)).xyz;
            break;
        }
        case 42: {
            vec3 rp = vec3((point.x / scale.x) + offset.x, (point.y / scale.y) + offset.y, (point.z / scale.z) + offset.z);
            r = textureCube(codeHelgrima, rp, vec3(0.0, 0.0, 0.1)).xyz;
            break;
        }
        case 43: {
            vec3 rp = vec3((point.x / scale.x) + offset.x, (point.y / scale.y) + offset.y, (point.z / scale.z) + offset.z);
            r = textureCube(gfxWartti, rp, vec3(0.0, 0.0, 0.1)).xyz;
            break;
        }
        case 44: {
            vec3 rp = vec3((point.x / scale.x) + offset.x, (point.y / scale.y) + offset.y, (point.z / scale.z) + offset.z);
            r = textureCube(musicMajaniemi, rp, vec3(0.0, 0.0, 0.1)).xyz;
            break;
        }
        case 51: {
            vec3 rp = vec3((point.x / scale.x) + offset.x, (point.y / scale.y) + offset.y, (point.z / scale.z) + offset.z);
            r = textureCube(cassetteLabel, rp, vec3(0.0, 0.0, 0.1)).xyz;
            break;
        }
        case 52: {
            vec3 rp = vec3((point.x / scale.x) + offset.x, (point.y / scale.y) + offset.y, (point.z / scale.z) + offset.z);
            r = textureCube(adaptGreets, rp, vec3(0.0, 0.0, 1.0)).xyz;
            break;
        }
        case 53: {
            vec3 rp = vec3((point.x / scale.x) + offset.x, (point.y / scale.y) + offset.y, (point.z / scale.z) + offset.z);
            r = textureCube(convergeGreets, rp, vec3(0.0, 0.0, 1.0)).xyz;
            break;
        }
        case 54: {
            vec3 rp = vec3((point.x / scale.x) + offset.x, (point.y / scale.y) + offset.y, (point.z / scale.z) + offset.z);
            r = textureCube(exoticMenGreets, rp, vec3(0.0, 0.0, 1.0)).xyz;
            break;
        }
        case 55: {
            vec3 rp = vec3((point.x / scale.x) + offset.x, (point.y / scale.y) + offset.y, (point.z / scale.z) + offset.z);
            r = textureCube(wideLoadGreets, rp, vec3(0.0, 0.0, 1.0)).xyz;
            break;
        }
       
    }  
    return r;
}

vec3 determinePixelBaseColor(float steps, float dist, entity e) {
    vec3 base = vec3(1.0 - smoothstep(0, rayMaxSteps, steps));
    if(e.material.textureOptions.normalMap == false) {
        base *= generateTexture(e.material.textureOptions.index, e.point, e.material.textureOptions.offset, e.material.textureOptions.scale);
    }
    return base;
}

vec3 calculateNormal(in vec3 n, in entity e) {
    vec3 normal = n;
    if(e.material.textureOptions.normalMap == true) {
        normal *= generateTexture(e.material.textureOptions.index, e.point, e.material.textureOptions.offset, e.material.textureOptions.scale);
    }
    return normal;
}

vec3 calculateLights(in vec3 normal, in vec3 eye, in vec3 lp, in vec3 origin, entity entity, vec2 uv) {
    vec3 lights = vec3(0.0);
    vec3 ambient = ambient(entity.material.ambient, entity.material.ambientStrength);
    vec3 diffuse = diffuse(normal, origin, lp, entity.material.diffuse, entity.material.diffuseStrength);
    vec3 specular = specular(normal, eye, origin, lp, entity.material.specular, entity.material.specularStrength, entity.material.shininess);
    float shadow = 1.0;
    if(entity.material.receiveShadows == true) {
        shadow = shadows(origin, normalize(lp - origin), entity.material.shadowLowerLimit, entity.material.shadowUpperLimit, entity.material.shadowHardness, uv);
    }

    lights += ambient;
    lights += diffuse;
    lights += specular;
    lights *= vec3(shadow);
    return lights;
}

vec3 fog(vec3 original, vec3 color, float dist, float b) {
    return mix(original, color, 1.0 - exp(-dist * b));
}

vec3 processColor(hit h, vec3 rd, vec3 eye, vec2 uv, vec3 lp)
{
    vec4 bg = (h.steps >= rayMaxSteps) ? background(uv) : vec4(0.0);
    vec3 base = determinePixelBaseColor(h.steps, h.dist, h.entity);
    vec3 normal = calculateNormal(h.normal, h.entity);
    vec3 lights = calculateLights(normal, eye, lp, h.point, h.entity, uv);

    vec3 result = base;
    result *= lights;
    
    result = fog(result, fogColor, h.dist, fogIntensity);
    result = mix(result, bg.rgb, bg.w);
    
    float gamma = 2.2;

    vec3 correct = pow(result, vec3(1.0 / gamma));
   
    return vec3(correct);
}

vec4 postProcess(vec3 original, vec2 uv) {
    return vec4(original * fade, 1.0);
}

vec3 drawMarching(vec2 uv) {
    vec3 camPos = vec3(cameraPosition.x, cameraPosition.y, cameraPosition.z);
    vec3 cameraLookAt = cameraLookAt;
    vec3 forward = normalize(cameraLookAt - camPos); 
    vec3 right = normalize(vec3(forward.z, 0.0, -forward.x));
    vec3 up = normalize(cross(forward, right)); 
    
    vec3 rd = normalize(forward + cameraFov * uv.x * right + cameraFov * uv.y * up);
    
    vec3 ro = vec3(camPos);
 
    hit tt = raymarch(ro, rd, uv);
    return processColor(tt, rd, ro, uv, camPos); 
}

void main() {
    float aspectRatio = resolution.x / resolution.y;
    vec2 uv = (gl_FragCoord.xy / resolution.xy) * 2.0 - 1.0;
    
    uv.x *= aspectRatio;
    vec3 o = drawMarching(uv);
    FragColor = postProcess(o, uv);
}