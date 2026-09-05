
#pragma GCC optimize("O3")

#include "render.h"
#include <algorithm>
#include <cstring>

namespace flipcraft {

static constexpr float PI = 3.14159265358979323846f;

static float kFsin[16], kFcos[16];
static int8_t kSinYaw[16], kCosYaw[16];

// Per-block face textures for full cubes: [block][0 top, 1 bottom, 2 side].
struct FaceTex { uint8_t tex, set; bool valid; };
static FaceTex gFaceTex[32][3];

// Pre-flattened mesh quads for the non-full blocks (sapling cross, chest box),
// packed: gNonFullIdx maps a block id to its entry, 0xFF = none.
struct NonFullQuad { uint8_t quad, tex, set; };
struct NonFullMesh { uint8_t count; NonFullQuad q[8]; };
constexpr int NONFULL_MESHES = __builtin_popcount(BLOCKS_NOT_FULL) - 1;   // air has none
static NonFullMesh gNonFull[NONFULL_MESHES];
static uint8_t gNonFullIdx[32];

// Sunlight every quad template can receive: 0 faces away from the sun,
// 1 grazing light, 2 direct light. The sun is fixed (flipcraft.h), so the
// table is built once and never touched again.
static uint8_t gQuadLit[QUAD_COUNT];
static const float gSun[3] = {SUN_DIR_X, SUN_DIR_Y, SUN_DIR_Z};
// A shadow ray only ever moves towards +x, +y, +z (see sunVisible).
static_assert(SUN_DIR_X > 0 && SUN_DIR_Y > 0 && SUN_DIR_Z > 0, "sunVisible assumes a +x +y +z sun");

// Per rebuild: the highest block in the resident ring, and per slot the
// highest block in any resident chunk a ray can still enter from that chunk
// (chunks at >= cx and >= cz). A ray above that level meets nothing but air.
static int8_t gReach[WINDOW_CHUNKS][WINDOW_CHUNKS];
static int8_t gRingMaxY = WORLD_SY - 1;

// Outward unit normal of a quad template.
static void quadNormal(int q, float n[3]) {
    const int (*t)[3] = quadTemplate(q);
    const float e0[3] = {(float)(t[1][0]-t[0][0]), (float)(t[1][1]-t[0][1]), (float)(t[1][2]-t[0][2])};
    const float e1[3] = {(float)(t[2][0]-t[1][0]), (float)(t[2][1]-t[1][1]), (float)(t[2][2]-t[1][2])};
    n[0] = e0[1]*e1[2] - e0[2]*e1[1];
    n[1] = e0[2]*e1[0] - e0[0]*e1[2];
    n[2] = e0[0]*e1[1] - e0[1]*e1[0];
    float len = sqrtf(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    if (len < 1e-6f) len = 1.0f;
    for (int k = 0; k < 3; k++) n[k] = n[k] / len;
}

// Axis-aligned faces are invisible unless the camera is on their front side;
// this rejects roughly half of all cached faces with one compare, before any
// vertex transform. axis<0 marks quads with no single facing plane.
struct FaceCull { int8_t axis; int8_t neg; uint8_t off; };
static const FaceCull kCull[QUAD_COUNT] = {
    {0,1,0}, {0,0,16}, {2,1,0}, {2,0,16}, {1,1,0}, {1,0,16},   // full cube
    {-1,0,0}, {-1,0,0},                                        // cross
    {0,1,1}, {0,0,15}, {2,1,1}, {2,0,15}, {1,1,0}, {1,0,14},   // small cube
    {-1,0,0},                                                  // item shadow
    {-1,0,0}, {-1,0,0}, {-1,0,0}, {-1,0,0}, {-1,0,0}, {-1,0,0},// block item
    {-1,0,0}, {-1,0,0}, {-1,0,0}, {-1,0,0},                    // cross item
    {1,0,0},                                                   // bedrock
};

static bool gTablesReady = false;

static void initTables() {
    if (gTablesReady) return;

    for (int i = 0; i < 16; i++) {
        float a = PI * 2.0f * (i / 16.0f);
        kFsin[i] = sinf(a);
        kFcos[i] = cosf(a);
        kSinYaw[i] = (int8_t)floorf(-sinf(a) * 64.0f);
        kCosYaw[i] = (int8_t)floorf( cosf(a) * 64.0f);
    }

    for (int q = 0; q < QUAD_COUNT; q++) {
        float n[3];
        quadNormal(q, n);
        float dot = 0;
        for (int k = 0; k < 3; k++) dot += n[k] * gSun[k];
        gQuadLit[q] = dot >= 0.65f ? 2 : (dot > 0.05f ? 1 : 0);
    }

    int nfCount = 0;
    for (int id = 0; id < 32; id++) {
        const MeshEntry& m = meshBlock((uint8_t)id);
        for (int f = 0; f < 3; f++) {
            bool valid = m.exists && f < m.texCount;
            gFaceTex[id][f] = {valid ? m.textures[f].id : (uint8_t)0,
                               valid ? m.textures[f].settings : (uint8_t)0, valid};
        }
        gNonFullIdx[id] = 0xFF;
        if (m.exists && !blockIsFull((uint8_t)id) && nfCount < NONFULL_MESHES) {
            NonFullMesh& nf = gNonFull[nfCount];
            nf.count = 0;
            for (int qi = 0; qi < m.quadCount && nf.count < 8; qi++) {
                const MeshQuadRef& q = m.quads[qi];
                if (q.texIndex >= m.texCount) continue;
                nf.q[nf.count++] = {q.quadId, m.textures[q.texIndex].id,
                                    m.textures[q.texIndex].settings};
            }
            gNonFullIdx[id] = (uint8_t)nfCount++;
        }
    }
    gTablesReady = true;
}

Renderer::Renderer() {
    initTables();
    camRotToMatrix(0, 0);
    clearBuffer();
}

void Renderer::camRotToMatrix(int pitch, int yaw) {
    yawIndex = yaw; pitchIndex = pitch;
    float sc = kFsin[yaw & 0xF],   cc = kFcos[yaw & 0xF];
    float sb = kFsin[pitch & 0xF], cb = kFcos[pitch & 0xF];
    matrix[0][0]=cc;       matrix[0][1]=0;  matrix[0][2]=sc;
    matrix[1][0]=sb*sc;    matrix[1][1]=cb; matrix[1][2]=-sb*cc;
    matrix[2][0]=-cb*sc;   matrix[2][1]=sb; matrix[2][2]=cb*cc;
}
void Renderer::setCamRot(uint8_t data) { camRotToMatrix(data >> 4, data & 0xF); }

void Renderer::setShaders(bool on) {
    if (shaders == on) return;
    shaders = on;
    invalidateChunkMeshes();   // every cached mesh was baked the other way
}

int Renderer::sinYaw() const { return kSinYaw[yawIndex & 0xF]; }
int Renderer::cosYaw() const { return kCosYaw[yawIndex & 0xF]; }
float Renderer::camDir(int axis) const { return floorf(matrix[2][axis]*64.0f); }

void Renderer::invalidateChunkMeshes() {
    for (auto& col : chunkMesh)
        for (auto& cm : col) {
            cm.cx = cm.cz = -1;
            cm.faces.clear();
            cm.faces.shrink_to_fit();   // release, don't keep the old world's capacity
            cm.masks.clear();
            cm.masks.shrink_to_fit();
        }
}

// matrix[0][1] is 0 by construction (no roll), so row 0 skips the oy term.
Vertex Renderer::worldToCam(const Vertex& v) const {
    float ox = v.x - camPos[0], oy = v.y - camPos[1], oz = v.z - camPos[2];
    Vertex r;
    r.x = matrix[0][0]*ox + matrix[0][2]*oz;
    r.y = matrix[1][0]*ox + matrix[1][1]*oy + matrix[1][2]*oz;
    r.z = matrix[2][0]*ox + matrix[2][1]*oy + matrix[2][2]*oz;
    r.u = v.u; r.v = v.v;
    return r;
}

Vertex Renderer::camToScreen(const Vertex& v) const {
    const float invZ = 1.0f / v.z;
    const float persp = (float)LENS * invZ;
    constexpr float HALF_W   = SCREEN_WIDTH / 2;                        // 64
    constexpr float CENTER_Y = (SCREEN_HEIGHT - 1) - SCREEN_HEIGHT / 2; // 31
    Vertex r;
    r.x = std::clamp(v.x * persp + HALF_W, -255.0f, 255.0f);
    r.y = std::clamp(CENTER_Y - v.y * persp, -255.0f, 255.0f);
    r.z = invZ;
    r.u = v.u * invZ;
    r.v = v.v * invZ;
    return r;
}

bool Renderer::isBackfacing(const Vertex& v1,const Vertex& v2,const Vertex& v3) const {
    float cross = (v3.x - v1.x)*(v1.y - v2.y) - (v1.y - v3.y)*(v2.x - v1.x);
    return cross < 0.0f;
}

void Renderer::clearBuffer() {
    if(zbuf) memset(zbuf, 0, (size_t)SCREEN_HEIGHT * SCREEN_WIDTH);
}

// Pixels between perspective-correct samples. Texture coords are interpolated
// affinely inside a run of this many pixels, so one reciprocal serves PERSP_STEP
// pixels instead of one per pixel. For voxel-sized faces the drift is invisible.
static constexpr int PERSP_STEP = 8;

void Renderer::rasterTri(const Vertex& A,const Vertex& B,const Vertex& C) {
    const float area = (B.x-A.x)*(C.y-A.y) - (B.y-A.y)*(C.x-A.x);
    if (fabsf(area) < 1e-9f) return;

    int minY = ifloor(std::min({A.y,B.y,C.y}));
    int maxY = -ifloor(-std::max({A.y,B.y,C.y}));   // == ceil
    int minX = ifloor(std::min({A.x,B.x,C.x}));
    int maxX = -ifloor(-std::max({A.x,B.x,C.x}));
    minY=std::max(minY,0); maxY=std::min(maxY,SCREEN_HEIGHT-1);
    minX=std::max(minX,0); maxX=std::min(maxX,SCREEN_WIDTH-1);
    if (minX>maxX || minY>maxY) return;

    const uint8_t* trow = texturePacked(texture);
    const bool skipZero   = settings.transparent;
    const uint8_t invMask = settings.inverted ? 1 : 0;
    const bool useOverlay = settings.overlay;
    // On a transparent texture the surviving ink IS the object -- a glass
    // frame, a sapling cross -- not a shaded surface, so the lit dither would
    // punch half of it away and the block would vanish into the ground behind
    // it. Those quads keep their plain texture at every shader level.
    const uint8_t lit     = settings.transparent ? 0 : litLevel;
    const uint8_t* lmask  = litMask;

    const float e0dx = B.y-C.y, e0dy = C.x-B.x, e0c = B.x*C.y - B.y*C.x;
    const float e1dx = C.y-A.y, e1dy = A.x-C.x, e1c = C.x*A.y - C.y*A.x;

    const float ia = 1.0f/area;
    const float zA=A.z*ia, zB=B.z*ia, zC=C.z*ia;     // vertex 1/z, weighted
    const float uA=A.u*ia, uB=B.u*ia, uC=C.u*ia;     // vertex u/z, weighted
    const float vA=A.v*ia, vB=B.v*ia, vC=C.v*ia;     // vertex v/z, weighted
    const bool posArea = area > 0.0f;

    // Per-pixel deltas of the screen-linear quantities (constant for the tri):
    // invZ, S=u/z, T=v/z all step by these as x advances by one.
    const float de2dx = -(e0dx + e1dx);
    const float dInvZ = e0dx*zA + e1dx*zB + de2dx*zC;
    const float dS    = e0dx*uA + e1dx*uB + de2dx*uC;
    const float dT    = e0dx*vA + e1dx*vB + de2dx*vC;
    const float dDepth = 512.0f * dInvZ;

    for (int y=minY;y<=maxY;y++) {
        const float py = y+0.5f, px0 = minX+0.5f;
        float e0 = e0c + e0dx*px0 + e0dy*py;
        float e1 = e1c + e1dx*px0 + e1dy*py;
        // Seed the linear accumulators at the row's first pixel.
        float e2    = area - e0 - e1;
        float invZ  = e0*zA + e1*zB + e2*zC;
        float S     = e0*uA + e1*uB + e2*uC;
        float T     = e0*vA + e1*vB + e2*vC;
        float depthAcc = 512.0f * invZ;
        uint8_t* row = zbuf[y];

        bool wasIn = false;     // the row's span is contiguous: leave -> done
        int sub = 0;            // pixels left in the current affine run
        float fu=0, fv=0, dfu=0, dfv=0;   // 8*u, 8*v and their per-pixel steps

        for (int x=minX;x<=maxX;x++,
             e0+=e0dx, e1+=e1dx, e2+=de2dx, invZ+=dInvZ, S+=dS, T+=dT, depthAcc+=dDepth) {
            if ((posArea ? (e0<0||e1<0||e2<0) : (e0>0||e1>0||e2>0)) || invZ <= 0) {
                if (wasIn) break;
                sub = 0;
                continue;
            }
            wasIn = true;

            int depth = (int)depthAcc;              // invZ>0 -> trunc == floor
            if (depth > 127) depth = 127;
            if ((row[x] >> 1) > depth) {            // occluded
                if (sub > 0) { fu += dfu; fv += dfv; sub--; }
                continue;
            }

            if (sub == 0) {                         // perspective-correct sample
                const float rz = 1.0f/invZ;
                fu = 8.0f * S * rz;
                fv = 8.0f * T * rz;
                // local affine gradient: d(u)/dx = rz*(dS - u*dInvZ)
                dfu = 8.0f * rz * (dS - (S*rz)*dInvZ);
                dfv = 8.0f * rz * (dT - (T*rz)*dInvZ);
                sub = PERSP_STEP;
            }

            int a = (int)fu; if (a<0) a=0; else if (a>7) a=7;
            int b = (int)fv; if (b<0) b=0; else if (b>7) b=7;
            uint8_t color = (uint8_t)((trow[b] >> a) & 1);

            fu += dfu; fv += dfv; sub--;

            if (skipZero && color==0) continue;
            color ^= invMask;
            // Sunlit texels lose ink to a dither, so a lit face reads brighter
            // than the same face in shadow: 50% for direct light, 25% grazing.
            if (lit && (!lmask || ((lmask[b] >> a) & 1)))
                color &= (uint8_t)(lit == 2 ? ((x ^ y) & 1) : ((x | y) & 1));
            if (useOverlay) color ^= row[x] & 1;
            row[x] = (uint8_t)((depth << 1) | color);
        }
    }
}

static int clipNear(const Vertex* in, int n, Vertex* out) {
    int m = 0;
    for (int i=0;i<n;i++) {
        const Vertex& cur = in[i];
        const Vertex& nxt = in[(i+1)%n];
        bool curIn = cur.z >= CLIP, nxtIn = nxt.z >= CLIP;
        if (curIn) out[m++] = cur;
        if (curIn != nxtIn) {
            float t = ((float)CLIP - cur.z) / (nxt.z - cur.z);
            Vertex& e = out[m++];
            e.x = cur.x + t*(nxt.x-cur.x);
            e.y = cur.y + t*(nxt.y-cur.y);
            e.z = CLIP;
            e.u = cur.u + t*(nxt.u-cur.u);
            e.v = cur.v + t*(nxt.v-cur.v);
        }
    }
    return m;
}

void Renderer::drawQuadCam(Vertex q[4]) {
    if (q[0].z < CLIP && q[1].z < CLIP && q[2].z < CLIP && q[3].z < CLIP) return;
    Vertex clipped[8];
    int n = clipNear(q, 4, clipped);
    if (n < 3) return;

    Vertex scr[8];
    for (int i=0;i<n;i++) scr[i] = camToScreen(clipped[i]);

    if (isBackfacing(scr[0], scr[1], scr[2])) {
        if (settings.cullBackface) return;
        for (int i=0, j=n-1; i<j; i++, j--) std::swap(scr[i], scr[j]);
    }

    for (int i=1;i+1<n;i++) rasterTri(scr[0], scr[i], scr[i+1]);
}

static const float kQuadUvs[4][2] = {{0,0},{0,1},{1,1},{1,0}};

// Float-position path used for item entities and overlays; block faces from
// the chunk meshes go through drawBlockQuad instead.
void Renderer::renderQuad(float x,float y,float z,int quadId,uint8_t texId,int texSettings) {
    int bx = ifloor(x), bz = ifloor(z);
    if (bx < winX0 || bx > winX1 || bz < winZ0 || bz > winZ1) return;
    const int (*tmpl)[3] = quadTemplate(quadId);
    Vertex cam[4];
    for (int i=0;i<4;i++) {
        Vertex world;
        world.x = x*16.0f + tmpl[i][0];
        world.y = y*16.0f + tmpl[i][1];
        world.z = z*16.0f + tmpl[i][2];
        world.u = kQuadUvs[i][0]; world.v = kQuadUvs[i][1];
        cam[i] = worldToCam(world);
    }
    texture = (Texture)texId;
    settings.cullBackface = (texSettings & TS_CULLBACK) != 0;
    settings.transparent  = (texSettings & TS_TRANSPARENT) != 0;
    settings.inverted     = (texSettings & TS_INVERTED) != 0;
    settings.overlay      = (texSettings & TS_OVERLAY) != 0;
    litLevel = 0; litMask = nullptr;
    drawQuadCam(cam);
}

void Renderer::drawBlockQuad(int x,int y,int z,int quadId,uint8_t texId,int texSettings,
                             uint8_t lit,const uint8_t* mask) {
    const int (*tmpl)[3] = quadTemplate(quadId);
    const float bx = (float)(x << 4), by = (float)(y << 4), bz = (float)(z << 4);
    Vertex cam[4];
    for (int i=0;i<4;i++) {
        Vertex world;
        world.x = bx + tmpl[i][0];
        world.y = by + tmpl[i][1];
        world.z = bz + tmpl[i][2];
        world.u = kQuadUvs[i][0]; world.v = kQuadUvs[i][1];
        cam[i] = worldToCam(world);
    }
    texture = (Texture)texId;
    settings.cullBackface = (texSettings & TS_CULLBACK) != 0;
    settings.transparent  = (texSettings & TS_TRANSPARENT) != 0;
    settings.inverted     = (texSettings & TS_INVERTED) != 0;
    settings.overlay      = (texSettings & TS_OVERLAY) != 0;
    litLevel = lit; litMask = mask;
    drawQuadCam(cam);
}

void Renderer::renderFace(int x,int y,int z,uint8_t texId,int direction,bool small_) {
    int quadId = direction + (small_ ? 8 : 0);
    renderQuad(x, y, z, quadId, texId, TS_CULLBACK);
}

void Renderer::renderOverlay(const World& w,int x,int y,int z,int breakPhase) {
    int texId = TEX_BREAK0 + breakPhase;
    static const int Faces[6][3] = {{-1,0,0},{1,0,0},{0,0,-1},{0,0,1},{0,-1,0},{0,1,0}};
    static const int BlockQuads[6] = {QUAD_FULL_NEGX,QUAD_FULL_POSX,QUAD_FULL_NEGZ,QUAD_FULL_POSZ,QUAD_FULL_NEGY,QUAD_FULL_POSY};
    for (int i=0;i<6;i++) {
        uint8_t adj = w.getBlock(x+Faces[i][0], y+Faces[i][1], z+Faces[i][2]);
        if (blockIsTransparent(adj))
            renderQuad(x, y, z, BlockQuads[i], texId, TS_CULLBACK|TS_TRANSPARENT|TS_OVERLAY);
    }
}

void Renderer::renderItem(float x,float y,float z,uint8_t itemId,uint8_t inv) {
    const MeshEntry& it = meshItem(itemId);
    if (it.exists && itemIsBlockItem(itemId)) {
        static const int ItemQuads[6] = {QUAD_BLOCKITEM_NEGY,QUAD_BLOCKITEM_POSY,QUAD_BLOCKITEM_NEGX,
                                         QUAD_BLOCKITEM_POSX,QUAD_BLOCKITEM_NEGZ,QUAD_BLOCKITEM_POSZ};
        static const int TexIndices[6] = {1,0,2,2,3,2};
        for (int i=0;i<6;i++) {
            if (TexIndices[i] >= it.texCount) continue;
            const MeshTex& t = it.textures[TexIndices[i]];
            renderQuad(x, y, z, ItemQuads[i], t.id, t.settings ^ inv);
        }
    } else if (it.exists) {
        for (int qi=0; qi<it.quadCount; qi++) {
            const MeshQuadRef& q = it.quads[qi];
            if (q.texIndex >= it.texCount) continue;
            const MeshTex& t = it.textures[q.texIndex];
            renderQuad(x, y, z, q.quadId, t.id, t.settings ^ inv);
        }
    }
    renderQuad(x, y, z, QUAD_ITEMSHADOW, TEX_SHADOW, TS_CULLBACK|TS_TRANSPARENT|TS_INVERTED);
}

// tex[6] = negx,posx,negz,posz,negy,posy; headDir = world side the body's +Z
// points at, top/bottom faces are sampled in body space (v=0 at the head end)
// per face+vertex corner: bit0 pick x1, bit1 pick y1, bit2 pick z1
static const uint8_t kCorner[6][4] = {
    {4,6,2,0}, {1,3,7,5}, {0,2,3,1}, {5,7,6,4}, {4,0,1,5}, {2,6,7,3},
};

void Renderer::renderBox(float x0,float y0,float z0,float x1,float y1,float z1,
                         const uint8_t tex[6],int texSettings,uint8_t headDir) {
    const float px[2]={x0,x1}, py[2]={y0,y1}, pz[2]={z0,z1};
    settings.cullBackface = (texSettings & TS_CULLBACK) != 0;
    settings.transparent  = (texSettings & TS_TRANSPARENT) != 0;
    settings.inverted     = (texSettings & TS_INVERTED) != 0;
    settings.overlay      = (texSettings & TS_OVERLAY) != 0;
    litMask = nullptr;
    for (int f=0;f<6;f++) {
        Vertex cam[4];
        for (int i=0;i<4;i++) {
            uint8_t c = kCorner[f][i];
            Vertex w;
            w.x = px[c&1]; w.y = py[(c>>1)&1]; w.z = pz[(c>>2)&1];
            if (f>=4) {
                const float cxb=(float)(c&1), czb=(float)((c>>2)&1);
                switch (headDir&3) {
                    case 0:  w.u=czb; w.v=cxb;      break;
                    case 1:  w.u=czb; w.v=1.0f-cxb; break;
                    case 2:  w.u=cxb; w.v=czb;      break;
                    default: w.u=cxb; w.v=1.0f-czb; break;
                }
            } else { w.u = kQuadUvs[i][0]; w.v = kQuadUvs[i][1]; }
            cam[i] = worldToCam(w);
        }
        texture = (Texture)tex[f];
        // The box is not in the voxel grid, so it gets no shadow rays: light
        // it from the face normal alone (kCorner faces match QUAD_FULL_*).
        litLevel = shaders ? gQuadLit[f] : 0;
        drawQuadCam(cam);
    }
}

// Lit dynamite entity: full block-size cube, (x,z) centre / y bottom in world
// sub-pixels; inv flashes the fuse
void Renderer::renderDynamite(float x,float y,float z,uint8_t inv) {
    const int bxc = ifloor(x*(1.0f/16.0f)), bzc = ifloor(z*(1.0f/16.0f));
    if (bxc < winX0 || bxc > winX1 || bzc < winZ0 || bzc > winZ1) return;
    static const uint8_t tex[6] = {TEX_DYNAMITE,TEX_DYNAMITE,TEX_DYNAMITE,TEX_DYNAMITE,
                                   TEX_DYNAMITETOP,TEX_DYNAMITETOP};
    renderBox(x-8.0f, y, z-8.0f, x+8.0f, y+16.0f, z+8.0f, tex, TS_CULLBACK ^ inv, 2);
}

// (x,y,z) feet centre in world sub-pixels; yaw = 16-step heading, camera
// convention fwd=(-sin,cos); inv = 0 or TS_INVERTED; sc16 = scale*16
void Renderer::renderMob(float x,float y,float z,uint8_t species,uint8_t yaw,uint8_t inv,uint8_t sc16) {
    const int bxc = ifloor(x*(1.0f/16.0f)), bzc = ifloor(z*(1.0f/16.0f));
    if (bxc < winX0 || bxc > winX1 || bzc < winZ0 || bzc > winZ1) return;
    const MobSpec& s = mobSpec(species);
    // local (lx,lz) -> world: {wx = c*lx - s*lz, wz = s*lx + c*lz}
    const float sn = kFsin[yaw & 0xF], cs = kFcos[yaw & 0xF];
    const float k = sc16*(1.0f/16.0f);
    settings.cullBackface = true;
    settings.transparent  = false;
    settings.inverted     = (inv & TS_INVERTED) != 0;
    settings.overlay      = false;
    litMask = nullptr;

    int n;
    const MobBox* boxes = mobBoxes(species, n);
    for (int i=0;i<n;i++) {
        const MobBox& bx = boxes[i];
        const float lx[2]={bx.ox*k,(bx.ox+bx.sx)*k};
        const float ly[2]={y+bx.oy*k,y+(bx.oy+bx.sy)*k};
        const float lz[2]={bx.oz*k,(bx.oz+bx.sz)*k};
        for (int f=0;f<6;f++) {
            Vertex cam[4];
            for (int j=0;j<4;j++) {
                uint8_t c = kCorner[f][j];
                const float px=lx[c&1], pz=lz[(c>>2)&1];
                Vertex w;
                w.x = x + cs*px - sn*pz;
                w.y = ly[(c>>1)&1];
                w.z = z + sn*px + cs*pz;
                // top/bottom sampled in body space, v=0 at the head end
                if (f>=4) { w.u=(float)(c&1); w.v=1.0f-(float)((c>>2)&1); }
                else { w.u = kQuadUvs[j][0]; w.v = kQuadUvs[j][1]; }
                cam[j] = worldToCam(w);
            }
            texture = (Texture)((f==3 && (bx.flags&1)) ? s.texFront :
                                f>=4 ? s.texTop : s.texSide);
            // A creature turns, so its side faces do not keep a fixed normal;
            // only the flat top is lit, which still separates it from the ground.
            litLevel = (shaders && f >= 4) ? gQuadLit[f] : 0;
            drawQuadCam(cam);
        }
    }
    renderQuad((x-4.0f)/16.0f, y/16.0f, (z-4.0f)/16.0f, QUAD_ITEMSHADOW, TEX_SHADOW,
               TS_CULLBACK|TS_TRANSPARENT|TS_INVERTED);
}

// One shadow ray, marched through the voxel grid in block units towards the
// sun (the Raymarcher's shadow() step, on a grid instead of a distance field).
// A block stops the light unless its texture is drawn with TS_TRANSPARENT, in
// which case only its ink texels do: glass throws the shadow of its frame and
// lets the rest of the light through. `viaTex` reports that such a block was
// crossed, so the caller knows the face needs a per-texel bake.
__attribute__((noinline)) static bool sunVisible(const World& w, float px, float py, float pz,
                                                 const int own[3], bool& viaTex) {
    constexpr float FAR = 1e9f;
    int vx = ifloor(px), vy = ifloor(py), vz = ifloor(pz);

    // The sample point can already sit inside a block (a floor right under a
    // glass pane, or the inset faces of the chest, which stay in their own
    // cell). Its own cell never shades it; anything else does.
    if (vx != own[0] || vy != own[1] || vz != own[2]) {
        uint8_t id0 = w.getBlock(vx, vy, vz);
        if (id0 != BLOCK_AIR) {
            const FaceTex& ft = gFaceTex[id0 & 0x1F][1];
            if (!ft.valid || !(ft.set & TS_TRANSPARENT)) return false;
            viaTex = true;
            int tu = (int)((px - vx)*8.0f); if (tu<0) tu=0; else if (tu>7) tu=7;
            int tv = (int)((pz - vz)*8.0f); if (tv<0) tv=0; else if (tv>7) tv=7;
            uint8_t ink = (uint8_t)((texturePacked(ft.tex)[tv] >> tu) & 1);
            if (ft.set & TS_INVERTED) ink ^= 1;
            if (ink) return false;
        }
    }

    const int sx = gSun[0] > 0 ? 1 : -1, sy = 1, sz = gSun[2] > 0 ? 1 : -1;
    const float tdx = gSun[0] != 0 ? fabsf(1.0f/gSun[0]) : FAR;
    const float tdy = fabsf(1.0f/gSun[1]);
    const float tdz = gSun[2] != 0 ? fabsf(1.0f/gSun[2]) : FAR;
    float tmx = gSun[0] != 0 ? (gSun[0] > 0 ? (vx+1-px) : (px-vx)) * tdx : FAR;
    float tmy = (vy+1-py) * tdy;
    float tmz = gSun[2] != 0 ? (gSun[2] > 0 ? (vz+1-pz) : (pz-vz)) * tdz : FAR;

    // The ray never comes back down, so once it is above every block it can
    // still reach it is in open sky. The chunk it is in is resolved once per
    // chunk crossed, not per step, and looked into only below that chunk's
    // own top.
    int ccx = INT32_MIN, ccz = INT32_MIN, cmaxY = -1, reachY = gRingMaxY;
    const uint8_t* base = nullptr;
    for (int i = 0; i < SHADOW_MAX_STEPS; i++) {
        float t;
        int axis;
        if (tmx <= tmy && tmx <= tmz)      { t = tmx; tmx += tdx; vx += sx; axis = 0; }
        else if (tmy <= tmz)               { t = tmy; tmy += tdy; vy += sy; axis = 1; }
        else                               { t = tmz; tmz += tdz; vz += sz; axis = 2; }
        if (vy >= WORLD_SY || vy > reachY) return true;
        const int cx = vx >> CHUNK_SHIFT, cz = vz >> CHUNK_SHIFT;
        if (cx != ccx || cz != ccz) {
            ccx = cx; ccz = cz;
            int csx, csz;
            base = w.chunkData(cx, cz, csx, csz);
            if (base) { cmaxY = w.slotMaxY[csx][csz]; reachY = gReach[csx][csz]; }
            else      { reachY = gRingMaxY; }
            if (vy > reachY) return true;
        }
        if (!base || vy > cmaxY || vy < 0) continue;
        uint8_t id = base[(vy * CHUNK_SIZE + (vz & CHUNK_MASK)) * CHUNK_SIZE + (vx & CHUNK_MASK)];
        if (id == BLOCK_AIR) continue;

        // Entered through the face the ray crossed: bottom face on a vertical
        // step, side texture otherwise.
        const FaceTex& ft = gFaceTex[id & 0x1F][axis == 1 ? 1 : 2];
        if (!ft.valid || !(ft.set & TS_TRANSPARENT)) return false;
        viaTex = true;

        const float hx = px + gSun[0]*t, hy = py + gSun[1]*t, hz = pz + gSun[2]*t;
        float fu, fv;
        if (axis == 0)      { fu = hz - ifloor(hz); fv = hy - ifloor(hy); }
        else if (axis == 1) { fu = hx - ifloor(hx); fv = hz - ifloor(hz); }
        else                { fu = hx - ifloor(hx); fv = hy - ifloor(hy); }
        int tu = (int)(fu*8.0f); if (tu<0) tu=0; else if (tu>7) tu=7;
        int tv = (int)(fv*8.0f); if (tv<0) tv=0; else if (tv>7) tv=7;
        uint8_t ink = (uint8_t)((texturePacked(ft.tex)[tv] >> tu) & 1);
        if (ft.set & TS_INVERTED) ink ^= 1;
        if (ink) return false;
    }
    return true;
}

// The bake bookkeeping from here on runs once per face, not per ray or per
// pixel: size matters more than speed in a .fal that lives in RAM.
#pragma GCC push_options
#pragma GCC optimize("Os")

// Sun state of one face: 0 lit, 1 shadowed, 2 mixed (mask filled, bit u of
// byte v set = that texel sees the sun). Five probe rays settle the uniform
// cases; the other faces pay for all 64. Only the Quality level gets here.
__attribute__((noinline)) static int bakeFaceShadow(const World& w, int gx, int y, int gz, int quad, uint8_t* mask) {
    if (!gQuadLit[quad]) return 1;

    const int (*t)[3] = quadTemplate(quad);
    float n[3];
    quadNormal(quad, n);
    float base[3], du[3], dv[3];
    for (int k = 0; k < 3; k++) {
        const float org = (float)(k == 0 ? gx : (k == 1 ? y : gz));
        base[k] = org + t[0][k]*(1.0f/16.0f) + n[k]*0.03f;   // lift off the surface
        du[k] = (float)(t[3][k] - t[0][k]) * (1.0f/16.0f);
        dv[k] = (float)(t[1][k] - t[0][k]) * (1.0f/16.0f);
    }
    const int own[3] = {gx, y, gz};
    auto rayAt = [&](int u, int v, bool& viaTex) {
        const float fu = (u + 0.5f) * (1.0f/8.0f), fv = (v + 0.5f) * (1.0f/8.0f);
        return sunVisible(w, base[0] + du[0]*fu + dv[0]*fv,
                             base[1] + du[1]*fu + dv[1]*fv,
                             base[2] + du[2]*fu + dv[2]*fv, own, viaTex);
    };

    static const uint8_t kProbe[5][2] = {{0,0},{7,0},{0,7},{7,7},{4,4}};
    bool viaTex = false;
    int litProbes = 0;
    uint8_t probeBits[8] = {0};   // probe results in mask layout, reused below
    for (int i = 0; i < 5; i++)
        if (rayAt(kProbe[i][0], kProbe[i][1], viaTex)) {
            litProbes++;
            probeBits[kProbe[i][1]] |= (uint8_t)(1u << kProbe[i][0]);
        }
    // All five agree and nothing see-through was crossed: the face is uniform.
    if ((litProbes == 0 || litProbes == 5) && !viaTex) return litProbes ? 0 : 1;

    for (int v = 0; v < 8; v++) {
        const uint8_t probed = (v == 0 || v == 7) ? 0x81 : (v == 4 ? 0x10 : 0);
        uint8_t bits = probeBits[v];
        for (int u = 0; u < 8; u++)
            if (!((probed >> u) & 1) && rayAt(u, v, viaTex)) bits |= (uint8_t)(1u << u);
        mask[v] = bits;
    }
    return 2;
}

// Face word bits 27-28 hold the sun state; bit 29 marks a face that wanted a
// per-texel mask and was refused one by the chunk budget.
constexpr uint32_t FACE_KEY = 0x07FFFFFF;
constexpr uint32_t FACE_NOMASK = 1u << 29;

// The chunk's previous bake, walked in step with the new scan: both lists
// are in scan order, so a face is found by advancing a cursor, never searched.
struct BakeCarry {
    const uint32_t* faces; size_t count;
    const uint8_t* masks; size_t maskBytes;
    size_t at, maskAt;          // cursor: next old face and its mask offset
    World::DirtyBox box;        // cells changed since that bake
};

// Can a shadow ray leaving a face of voxel (gx,y,gz) cross a cell of `box`?
// A ray starts within one block of its voxel and climbs 0.7885 blocks in x
// and 0.287 in z per block of y; both bounds are rounded up.
static bool rayCanReach(const World::DirtyBox& b, int gx, int y, int gz) {
    if (b.x0 > b.x1) return true;
    const int dyMax = b.y1 - y;
    if (dyMax < -1 || b.x1 - gx < -1 || b.z1 - gz < -1) return false;
    const int r = dyMax + 2;
    return b.x0 - gx <= (r * 4 + 4) / 5 + 1 && b.z0 - gz <= (r * 3 + 9) / 10 + 1;
}

__attribute__((noinline)) static bool carryFace(BakeCarry& c, uint32_t key, int& state, const uint8_t*& mask) {
    const uint32_t kv = key & 0x3FF;   // voxel: the scan order
    while (c.at < c.count && (c.faces[c.at] & 0x3FF) < kv) {
        if (((c.faces[c.at] >> 27) & 3) == 2) c.maskAt += 8;
        c.at++;
    }
    size_t m = c.maskAt;
    for (size_t i = c.at; i < c.count && (c.faces[i] & 0x3FF) == kv; i++) {
        const uint32_t f = c.faces[i];
        const int s = (f >> 27) & 3;
        if ((f & FACE_KEY) == key && !(f & FACE_NOMASK)) {
            if (s == 2) {
                if (m + 8 > c.maskBytes) return false;
                mask = c.masks + m;
            }
            state = s;
            return true;
        }
        if (s == 2) m += 8;
    }
    return false;
}

// Sun bits of one face word (already shifted): state 0 take the quad's own
// light, 1 none (plain texture), 2 per-texel mask appended to `scratch`. A
// face whose rays cannot reach any changed cell keeps its previous bake.
// Deliberately out of line -- its caller is inlined at every emit site in the
// scan loop, and inlining this with it costs about 4 KB in a .fal that runs
// from RAM.
__attribute__((noinline)) static uint32_t faceSun(
    const World& w, bool shaders, int gx, int y, int gz, uint32_t key,
    uint8_t* scratch, int& maskCount, BakeCarry* carry) {
    if (!shaders) return 1u << 27;

    uint8_t mask[8];
    const uint8_t* src = mask;
    int state;
    if (!(carry && !rayCanReach(carry->box, gx, y, gz) && carryFace(*carry, key, state, src)))
        state = bakeFaceShadow(w, gx, y, gz, (key >> 10) & 31, mask);
    if (state != 2) return (uint32_t)state << 27;
    if (maskCount < SHADOW_MASKS_PER_CHUNK) {
        memcpy(scratch + maskCount*8, src, 8);
        maskCount++;
        return 2u << 27;
    }
    // Out of budget: keep whichever uniform state covers most of the face.
    int litTexels = 0;
    for (int i = 0; i < 8; i++) litTexels += __builtin_popcount(src[i]);
    return (litTexels >= 32 ? 0u : 1u << 27) | FACE_NOMASK;
}

__attribute__((noinline)) static void computeReach(const World& w) {
    int m = -1;
    for (int a = 0; a < WINDOW_CHUNKS; a++)
        for (int b = 0; b < WINDOW_CHUNKS; b++)
            if (w.slotCX[a][b] >= 0 && w.slotMaxY[a][b] > m) m = w.slotMaxY[a][b];
    gRingMaxY = (int8_t)m;
    for (int a = 0; a < WINDOW_CHUNKS; a++)
        for (int b = 0; b < WINDOW_CHUNKS; b++) {
            int r = m;
            if (w.slotCX[a][b] >= 0) {
                r = -1;
                for (int c = 0; c < WINDOW_CHUNKS; c++)
                    for (int d = 0; d < WINDOW_CHUNKS; d++)
                        if (w.slotCX[c][d] >= w.slotCX[a][b] && w.slotCZ[c][d] >= w.slotCZ[a][b] &&
                            w.slotMaxY[c][d] > r)
                            r = w.slotMaxY[c][d];
            }
            gReach[a][b] = (int8_t)r;
        }
}
#pragma GCC pop_options

// Rebuild the packed face list for the chunk resident in window slot (sx,sz).
// Two passes over the chunk's voxels up to its highest non-air layer: the
// first only counts, so the list is allocated once at exactly its final size
// (no push_back doubling, no high-water capacity kept between rebuilds). With
// shaders on the second pass also settles the sun state of every face it
// emits -- carried over from the previous bake of the same chunk when no ray
// of the face can cross a changed cell, cast afresh otherwise -- which is why
// it runs only when the chunk (or a chunk whose shadow reaches it) changed,
// never per frame.
void Renderer::buildChunkMesh(const World& w, int sx, int sz) {
    uint8_t maskScratch[SHADOW_MASKS_PER_CHUNK * 8];

    ChunkMesh& cm = chunkMesh[sx][sz];
    const int cx = w.slotCX[sx][sz], cz = w.slotCZ[sx][sz];
    computeReach(w);

    // Same chunk, same shader setting: the old bake stays until the new list
    // is settled. Otherwise free it before counting so the peak is one list.
    BakeCarry carry;
    BakeCarry* carryPtr = nullptr;
    if (shaders && cm.cx == cx && cm.cz == cz && !cm.faces.empty()) {
        carry = {cm.faces.data(), cm.faces.size(), cm.masks.data(), cm.masks.size(),
                 0, 0, w.slotBox[sx][sz]};
        carryPtr = &carry;
    } else {
        cm.faces.clear();
        cm.faces.shrink_to_fit();
        cm.masks.clear();
        cm.masks.shrink_to_fit();
    }
    w.slotBox[sx][sz] = {1, 0, 1, 0, 1, 0};   // consumed
    cm.cx = cx; cm.cz = cz; cm.gen = w.slotGen[sx][sz];

    const uint8_t (*B)[CHUNK_SIZE][CHUNK_SIZE] = w.slot[sx][sz];
    const int bx0 = cx << CHUNK_SHIFT, bz0 = cz << CHUNK_SHIFT;
    // An all-air chunk still owns its bedrock floor, so scan at least y == 0.
    const int yTop = w.slotMaxY[sx][sz] < 0 ? 0 : w.slotMaxY[sx][sz];

    uint32_t* out = nullptr;   // null on the counting pass
    int count = 0, maskCount = 0;
    const bool bake = shaders;
    auto emit = [&](int lx, int y, int lz, int quad, uint8_t tex, uint8_t set) {
        if (out) {
            const uint32_t key = (uint32_t)lx | ((uint32_t)lz << 3) | ((uint32_t)y << 6) |
                                 ((uint32_t)quad << 10) | ((uint32_t)tex << 15) |
                                 ((uint32_t)set << 23);
            out[count] = key | faceSun(w, bake, bx0 + lx, y, bz0 + lz, key,
                                       maskScratch, maskCount, carryPtr);
        }
        count++;
    };
    // A face is visible when the neighbour is a different, see-through block.
    auto shows = [](uint8_t id, uint8_t n) { return n != id && blockIsTransparent(n); };

    auto scan = [&]() {
        for (int y = 0; y <= yTop; y++) {
            for (int lz = 0; lz < CHUNK_SIZE; lz++) {
                const uint8_t* row = B[y][lz];
                for (int lx = 0; lx < CHUNK_SIZE; lx++) {
                    const uint8_t id = row[lx];
                    if (y == 0 && blockIsTransparent(id))
                        emit(lx, 0, lz, QUAD_BEDROCK, TEX_STONE, TS_CULLBACK|TS_INVERTED);
                    if (id == BLOCK_AIR) continue;

                    if (!blockIsFull(id)) {
                        const uint8_t ni = gNonFullIdx[id & 0x1F];
                        if (ni != 0xFF) {
                            const NonFullMesh& nf = gNonFull[ni];
                            for (int i = 0; i < nf.count; i++)
                                emit(lx, y, lz, nf.q[i].quad, nf.q[i].tex, nf.q[i].set);
                        }
                        continue;
                    }

                    const FaceTex* ft = gFaceTex[id];
                    uint8_t n;
                    if (ft[2].valid) {  // side faces
                        n = lx > 0 ? row[lx-1] : w.getBlock(bx0-1, y, bz0+lz);
                        if (shows(id, n)) emit(lx, y, lz, QUAD_FULL_NEGX, ft[2].tex, ft[2].set);
                        n = lx < CHUNK_MASK ? row[lx+1] : w.getBlock(bx0+CHUNK_SIZE, y, bz0+lz);
                        if (shows(id, n)) emit(lx, y, lz, QUAD_FULL_POSX, ft[2].tex, ft[2].set);
                        n = lz > 0 ? B[y][lz-1][lx] : w.getBlock(bx0+lx, y, bz0-1);
                        if (shows(id, n)) emit(lx, y, lz, QUAD_FULL_NEGZ, ft[2].tex, ft[2].set);
                        n = lz < CHUNK_MASK ? B[y][lz+1][lx] : w.getBlock(bx0+lx, y, bz0+CHUNK_SIZE);
                        if (shows(id, n)) emit(lx, y, lz, QUAD_FULL_POSZ, ft[2].tex, ft[2].set);
                    }
                    // Down face: y == 0 can never be seen from below, skip it.
                    if (y > 0 && ft[1].valid && shows(id, B[y-1][lz][lx]))
                        emit(lx, y, lz, QUAD_FULL_NEGY, ft[1].tex, ft[1].set);
                    n = y < WORLD_SY - 1 ? B[y+1][lz][lx] : (uint8_t)BLOCK_AIR;
                    if (ft[0].valid && shows(id, n))
                        emit(lx, y, lz, QUAD_FULL_POSY, ft[0].tex, ft[0].set);
                }
            }
        }
    };

    scan();                     // pass 1: count faces
    std::vector<uint32_t> fresh(count);   // exactly count, no doubling
    out = fresh.data();
    count = 0;
    scan();                     // pass 2: fill and bake; same input, same counts
    cm.faces.swap(fresh);
    std::vector<uint8_t>(maskScratch, maskScratch + (size_t)maskCount*8).swap(cm.masks);
}

void Renderer::renderScene(const World& w) {
    const int camBX = ifloor(camPos[0] * (1.0f / (float)BLOCKSIZE));
    const int camBZ = ifloor(camPos[2] * (1.0f / (float)BLOCKSIZE));
    ActiveWindow win = activeWindowAround(camBX, camBZ, w.worldSX(), w.worldSZ());
    winX0 = win.x0; winX1 = win.x1; winZ0 = win.z0; winZ1 = win.z1;

    // Near-only draw distance: the window shrinks to the single chunk the
    // camera stands in, and the other eight slots hand their face lists and
    // shadow masks back to the allocator. That is where the RAM and most of
    // the raster time go, so this is the cheap mode in both.
    int onlySX = -1, onlySZ = -1;
    if (nearOnly) {
        const int ccx = camBX >> CHUNK_SHIFT, ccz = camBZ >> CHUNK_SHIFT;
        onlySX = ((ccx % 3) + 3) % 3;
        onlySZ = ((ccz % 3) + 3) % 3;
        const int x0 = ccx << CHUNK_SHIFT, z0 = ccz << CHUNK_SHIFT;
        if (x0 > winX0) winX0 = x0;
        if (x0 + CHUNK_MASK < winX1) winX1 = x0 + CHUNK_MASK;
        if (z0 > winZ0) winZ0 = z0;
        if (z0 + CHUNK_MASK < winZ1) winZ1 = z0 + CHUNK_MASK;
    }

    const float cpx = camPos[0], cpy = camPos[1], cpz = camPos[2];
    int budget = shaders ? REBUILDS_PER_FRAME : WINDOW_CHUNKS * WINDOW_CHUNKS;
    meshPending = false;

    for (int sz = 0; sz < WINDOW_CHUNKS; sz++)
        for (int sx = 0; sx < WINDOW_CHUNKS; sx++) {
            ChunkMesh& cm = chunkMesh[sx][sz];
            if (nearOnly && (sx != onlySX || sz != onlySZ)) {
                if (!cm.faces.empty() || !cm.masks.empty()) {
                    cm.cx = cm.cz = -1;
                    cm.faces.clear();
                    cm.faces.shrink_to_fit();
                    cm.masks.clear();
                    cm.masks.shrink_to_fit();
                }
                continue;
            }
            const int cx = w.slotCX[sx][sz], cz = w.slotCZ[sx][sz];
            if (cx < 0) continue;
            const int bx0 = cx << CHUNK_SHIFT, bz0 = cz << CHUNK_SHIFT;
            if (bx0 > winX1 || bx0 + CHUNK_MASK < winX0 ||
                bz0 > winZ1 || bz0 + CHUNK_MASK < winZ0) continue;

            if (cm.cx != cx || cm.cz != cz || cm.gen != w.slotGen[sx][sz]) {
                if (budget > 0) {
                    buildChunkMesh(w, sx, sz);
                    budget--;
                } else {
                    meshPending = true;
                    // A stale bake still draws; another chunk's faces cannot.
                    if (cm.cx != cx || cm.cz != cz) continue;
                }
            }

            // Chunks fully inside the window skip the per-face window test.
            const bool clip = bx0 < winX0 || bx0 + CHUNK_MASK > winX1 ||
                              bz0 < winZ0 || bz0 + CHUNK_MASK > winZ1;

            // Masked faces consume their mask in face order, culled or not.
            size_t maskAt = 0;
            for (uint32_t f : cm.faces) {
                const uint32_t sun = (f >> 27) & 3;
                const uint8_t* mask = nullptr;
                if (sun == 2) {
                    if (maskAt + 8 <= cm.masks.size()) mask = cm.masks.data() + maskAt;
                    maskAt += 8;
                }
                const int gx = bx0 + (f & 7), gz = bz0 + ((f >> 3) & 7);
                if (clip && (gx < winX0 || gx > winX1 || gz < winZ0 || gz > winZ1))
                    continue;
                const int y = (f >> 6) & 15, quad = (f >> 10) & 31;
                const FaceCull& fc = kCull[quad];
                if (fc.axis >= 0) {
                    const float cam = fc.axis == 0 ? cpx : (fc.axis == 1 ? cpy : cpz);
                    const float plane =
                        (float)(((fc.axis == 0 ? gx : (fc.axis == 1 ? y : gz)) << 4) + fc.off);
                    if (fc.neg ? cam >= plane : cam <= plane) continue;
                }
                drawBlockQuad(gx, y, gz, quad, (uint8_t)((f >> 15) & 0xFF), (f >> 23) & 0xF,
                              sun == 1 ? 0 : gQuadLit[quad], mask);
            }
        }
}

}
