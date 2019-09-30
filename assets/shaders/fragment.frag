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

uniform sampler2D bogdan;

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
    float h = clamp( 0.5 - 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix(d2, d1, h) + k*h*(1.0-h);
}

entity opSmoothUnion(entity m1, entity m2, float k, float threshold) {
    float h = opSmoothUnion(m1.dist, m2.dist, k);
    // Adding a 0.1 fixes the union, could be that
    // the operation changes the original shape minimally
   
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
    // Adding a 0.1 fixes the union, could be that
    // the operation changes the original shape minimally
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
    // Adding a 0.1 fixes the union, could be that
    // the operation changes the original shape minimally
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

float sdCylinder(vec3 p, vec3 pos, vec3 c )
{
    vec3 p1 = p + pos;
    return length(p1.xz - c.xy) - c.z;
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

float sdHexPrism(vec3 p, vec3 pos, vec2 h)
{
    vec3 p1 = p + pos;
    vec3 q = abs(p1);
    return max(q.z-h.y,max((q.x*0.866025+q.y*0.5),q.y)-h.x);
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
    return (m.x * sin(p.x)) + (m.y * sin(p.y)) + (m.z * sin(p.z));
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

vec3 translate(vec3 p, vec3 p1) {
    return p + (p1 * -1.0);
}

vec3 scale(vec3 p, float s) {
    vec3 p1 = p;
    p1 /= s;
    return p1;
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

entity mPlane(vec3 p, vec3 pos, vec4 n, material material)
{
    entity m;
    vec3 p1 = p;
    m.dist = sdPlane(p, pos, n);
    m.point = p1;
    m.material = material;
    return m;
}

entity mSphere(vec3 path, float radius, material material) {
    entity m;
    vec3 p1 = path;
    m.dist = sdSphere(path, vec3(0.0), radius);
    m.point = p1;
    m.material = material;
    return m;
}

entity mBox(vec3 path, vec3 size, float r, material material) {
    entity m;
    vec3 p1 = path;
    m.dist = sdBox(path, vec3(0.0), size, r);
    m.point = p1;
    m.material = material;
    return m;
}

entity mTorus(vec3 path, vec2 dim, material material) {
    entity m;
    vec3 p1 = path;
    m.dist = sdTorus(path, vec3(0.0), dim);
    m.point = p1;
    m.material = material;
    return m;
}

entity scene(vec3 path, vec2 uv)
{   
    int a = int(act);
    if(a == 1) {
        vec3 r = rot(path, vec3(time));
        vec3Tuple rr = repeat(r, vec3(10.0, 10.0, 10.0));
        //r = rr.first;
        material m1 = material(
            vec3(1.0, 0.0, 0.0),
            1.0,

            vec3(1.0, 0.0, 0.0),
            2.2,

            vec3(1.0, 1.0, 1.0),
            50.0,
            100.0,

            0.2,
            true,
            1.5,
            5.5,
            textureOptions(
                0,
                vec3(1.5, 1.5, 1.5),
                vec3(2.0, 2.0, 2.0),
                false
            )
        );

        entity e1 = mBox(r, vec3(1.0), 0.1, m1);
        e1.needNormals = true;  

        material m2 = material(
            vec3(0.0, 1.0, 0.0),
            1.0,

            vec3(0.0, 1.0, 0.0),
            2.2,

            vec3(1.0, 1.0, 1.0),
            50.0,
            100.0,

            0.2,
            true,
            1.5,
            5.5,
            textureOptions(
                0,
                vec3(1.5, 1.5, 1.5),
                vec3(2.0, 2.0, 2.0),
                false
            )
        );

        entity e2 = mBox(rotZ(translate(r, vec3(0.5, 1.0, 1.0)), -0.7), vec3(1.0), 0.1, m2);
        e2.needNormals = true;

        material m3 = material(
            vec3(0.0, 0.0, 1.0),
            1.0,

            vec3(0.0, 0.0, 1.0),
            2.2,

            vec3(1.0, 1.0, 1.0),
            50.0,
            100.0,

            0.2,
            true,
            2.5,
            5.5,
            textureOptions(
                0,
                vec3(1.5, 1.5, 1.5),
                vec3(2.0, 2.0, 2.0),
                false
            )
        );
        
        entity e3 = mBox(rotZ(translate(r, vec3(-0.5, -1.0, -1.0)), 0.7), vec3(1.0), 0.1, m3);
        e3.needNormals = true;  
        return opSmoothUnion(opSmoothUnion(e1, e2, 0.5, 0.0), e3, 0.5, 0.0);
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
        if(h.entity.dist < rayThreshold) {
            if(h.entity.needNormals == true) {
                vec2 eps = vec2(0.001, 0.0);
                h.normal = normalize(vec3(
                    scene(h.point + eps.xyy, uv).dist - h.entity.dist,
                    scene(h.point + eps.yxy, uv).dist - h.entity.dist,
                    scene(h.point + eps.yyx, uv).dist - h.entity.dist
                ));
            }
            break;
        }
        h.dist += h.entity.dist;

    }
    
    return h;
}

vec3 ambient(vec3 color, float strength) {
    return color * strength;
} 

vec3 diffuse(vec3 normal, vec3 hit, vec3 pos, vec3 color, float strength) {
    vec3 lightDir = normalize(pos - hit);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * color * strength;
    return diffuse;
}

vec3 specular(vec3 normal, vec3 eye, vec3 hit, vec3 pos, vec3 color, float strength, float shininess) {
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
        if(h < 0.01)
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
    if(a == 1) {
        r.xyz = vec3(0.0);
        r.w = 1.0;
    }

    return r;
}

vec3 generateTexture(int index, vec3 point, vec3 offset, vec3 scale) {
    vec3 r = vec3(1.0);
    switch(index) {
        case 1: {
            vec3 rp = vec3((point.x / scale.x) + offset.x, (point.y / scale.y) + offset.y, (point.z / scale.z) + offset.z);
            r = textureCube(bogdan, rp, vec3(0.0, 0.0, 0.1)).xyz;
            break;
        }
       
    }  
    return r;
}

vec3 determinePixelBaseColor(float steps, float dist, entity e) {
    vec3 base = vec3(1.0 - smoothstep(0.0, rayMaxSteps, steps));
    base *= generateTexture(e.material.textureOptions.index, e.point, e.material.textureOptions.offset, e.material.textureOptions.scale);
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

vec3 processColor(hit h, vec3 rd, vec3 eye, vec2 uv, vec3 lp)
{
    vec4 bg = (h.steps >= rayMaxSteps) ? background(uv) : vec4(0.0);
    vec3 base = determinePixelBaseColor(h.steps, h.dist, h.entity);
    vec3 normal = calculateNormal(h.normal, h.entity);
    vec3 lights = calculateLights(normal, eye, lp, h.point, h.entity, uv);

    vec3 result = base;
    result *= lights;
    
    result = mix(result, bg.rgb, h.last * bg.w);
    float gamma = 2.2;
    vec3 correct = pow(result, vec3(1.0 / gamma));
   
    return vec3(correct);
}

vec4 postProcess(vec3 original, vec2 uv) {
    return vec4(original * fade, 1.0);
}

vec3 drawMarching(vec2 uv) {
    vec3 camPos = vec3(cameraPosition.x, cameraPosition.y, cameraPosition.z);
    vec3 forward = normalize(cameraLookAt - camPos); 
    vec3 right = normalize(vec3(forward.z, 0.0, -forward.x));
    vec3 up = normalize(cross(forward, right)); 
    
    vec3 rd = normalize(forward + cameraFov * uv.x * right + cameraFov * uv.y * up);
    
    vec3 ro = vec3(camPos);
 
    hit tt = raymarch(ro, rd, uv);
    return processColor(tt, rd, ro, uv, lightPosition); 
}

void main() {
    float aspectRatio = resolution.x / resolution.y;
    vec2 uv = (gl_FragCoord.xy / resolution.xy) * 2.0 - 1.0;
    
    uv.x *= aspectRatio;
    vec3 o = drawMarching(uv);
    FragColor = postProcess(o, uv);
}