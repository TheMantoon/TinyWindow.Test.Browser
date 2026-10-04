import { dotnet } from './_framework/dotnet.js';

const canvas = document.getElementById('canvas');
const loadingOverlay = document.getElementById('loading-overlay');
const webgpuWarning = document.getElementById('webgpu-warning');
const fpsCounter = document.getElementById('fps-counter');
const debugOverlay = document.getElementById('debug-overlay');
const debugLog = document.getElementById('debug-log');

function logToScreen(type, ...args) {
    const text = args.map(a => typeof a === 'object' ? (a?.stack || a?.message || JSON.stringify(a)) : String(a)).join(' ');
    if (debugLog && debugOverlay) {
        const item = document.createElement('div');
        item.style.color = type === 'error' ? '#f87171' : type === 'warn' ? '#fbbf24' : '#94a3b8';
        item.textContent = `[${type.toUpperCase()}] ${text}`;
        debugLog.appendChild(item);
        debugOverlay.style.display = 'block';
        debugOverlay.scrollTop = debugOverlay.scrollHeight;
    }
}

window.addEventListener('error', (e) => logToScreen('error', e.message, `(${e.filename}:${e.lineno})`));
window.addEventListener('unhandledrejection', (e) => logToScreen('error', 'Unhandled Promise:', e.reason));

const originalConsoleError = console.error;
console.error = function(...args) {
    originalConsoleError.apply(console, args);
    logToScreen('error', ...args);
};

const originalConsoleWarn = console.warn;
console.warn = function(...args) {
    originalConsoleWarn.apply(console, args);
    logToScreen('warn', ...args);
};

// ==========================================
// WebGPU Shaders (WGSL)
// ==========================================

const rectShaderWgsl = `
struct Uniforms {
    projection: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = uniforms.projection * vec4<f32>(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
`;

const sdfShaderWgsl = `
struct Uniforms {
    projection: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) localPos: vec2<f32>,
    @location(3) halfSize: vec2<f32>,
    @location(4) radius: f32,
    @location(5) borderThickness: f32,
    @location(6) borderColor: vec4<f32>,
    @location(7) blur: f32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) localPos: vec2<f32>,
    @location(2) halfSize: vec2<f32>,
    @location(3) radius: f32,
    @location(4) borderThickness: f32,
    @location(5) borderColor: vec4<f32>,
    @location(6) blur: f32,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = uniforms.projection * vec4<f32>(input.position, 0.0, 1.0);
    output.color = input.color;
    output.localPos = input.localPos;
    output.halfSize = input.halfSize;
    output.radius = input.radius;
    output.borderThickness = input.borderThickness;
    output.borderColor = input.borderColor;
    output.blur = input.blur;
    return output;
}

fn roundedBoxSDF(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + vec2<f32>(r, r);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0, 0.0))) - r;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let r = min(input.radius, min(input.halfSize.x, input.halfSize.y));
    let dist = roundedBoxSDF(input.localPos, input.halfSize, r);
    let edgeSoftness = max(fwidth(dist), 1.0);

    if (input.blur > 0.0) {
        let shadowAlpha = 1.0 - smoothstep(-input.blur, input.blur, dist);
        return vec4<f32>(input.color.rgb, input.color.a * shadowAlpha);
    }

    let outerAlpha = 1.0 - smoothstep(-0.5 * edgeSoftness, 0.5 * edgeSoftness, dist);
    if (outerAlpha <= 0.0) {
        discard;
    }

    if (input.borderThickness > 0.0) {
        let innerDist = dist + input.borderThickness;
        let innerAlpha = 1.0 - smoothstep(-0.5 * edgeSoftness, 0.5 * edgeSoftness, innerDist);
        let borderAlpha = max(outerAlpha - innerAlpha, 0.0);
        return input.borderColor * borderAlpha + input.color * innerAlpha;
    } else {
        return vec4<f32>(input.color.rgb, input.color.a * outerAlpha);
    }
}
`;

const imageShaderWgsl = `
struct Uniforms {
    projection: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var t_texture: texture_2d<f32>;
@group(1) @binding(1) var s_sampler: sampler;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = uniforms.projection * vec4<f32>(input.position, 0.0, 1.0);
    output.uv = input.uv;
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let texColor = textureSample(t_texture, s_sampler, input.uv);
    return texColor * input.color;
}
`;

const textShaderWgsl = `
struct Uniforms {
    projection: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var t_font: texture_2d<f32>;
@group(1) @binding(1) var s_sampler: sampler;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = uniforms.projection * vec4<f32>(input.position, 0.0, 1.0);
    output.uv = input.uv;
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let alpha = textureSample(t_font, s_sampler, input.uv).r;
    if (alpha <= 0.005) {
        discard;
    }
    return vec4<f32>(input.color.rgb, input.color.a * alpha);
}
`;

// ==========================================
// WebGPU Engine State
// ==========================================

let adapter = null;
let device = null;
let context = null;
let format = null;

let uniformBuffer = null;
let uniformBindGroupLayout = null;
let uniformBindGroup = null;

let textureBindGroupLayout = null;
let linearSampler = null;

let rectPipeline = null;
let sdfPipeline = null;
let imagePipeline = null;
let textPipeline = null;

const RECT_BUFFER_SIZE = 2 * 1024 * 1024;
const SDF_BUFFER_SIZE = 4 * 1024 * 1024;
const IMAGE_BUFFER_SIZE = 2 * 1024 * 1024;
const TEXT_BUFFER_SIZE = 4 * 1024 * 1024;

let rectVbo = null;
let sdfVbo = null;
let imageVbo = null;
let textVbo = null;

let rectBufferOffset = 0;
let sdfBufferOffset = 0;
let imageBufferOffset = 0;
let textBufferOffset = 0;

let currentEncoder = null;
let currentPass = null;
let currentWidth = 1280;
let currentHeight = 720;

const textures = new Map();
let nextTextureId = 1;
let bridge = null;

// ==========================================
// WebGPU Initialization
// ==========================================

async function checkShader(module, name) {
    if (module.getCompilationInfo) {
        const info = await module.getCompilationInfo();
        for (const msg of info.messages) {
            console[msg.type === 'error' ? 'error' : 'warn'](`[WGSL ${name}]: ${msg.message} (line ${msg.lineNum})`);
        }
    }
}

async function initWebGpu(width, height) {
    if (!navigator.gpu) {
        console.error("WebGPU is not supported on this browser.");
        if (loadingOverlay) loadingOverlay.style.display = 'none';
        if (webgpuWarning) webgpuWarning.style.display = 'block';
        return false;
    }

    try {
        adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) {
            console.error("Failed to request WebGPU adapter.");
            if (loadingOverlay) loadingOverlay.style.display = 'none';
            if (webgpuWarning) webgpuWarning.style.display = 'block';
            return false;
        }

        device = await adapter.requestDevice();
        device.addEventListener('uncapturederror', (event) => {
            console.error('[WebGPU Uncaptured Error]:', event.error.message);
        });

        context = canvas.getContext('webgpu');
        format = navigator.gpu.getPreferredCanvasFormat();

        context.configure({
            device,
            format,
            alphaMode: 'opaque'
        });

        currentWidth = width;
        currentHeight = height;

        // Uniform Buffer (mat4 = 16 floats = 64 bytes)
        uniformBuffer = device.createBuffer({
            size: 64,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        uniformBindGroupLayout = device.createBindGroupLayout({
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.VERTEX,
                buffer: { type: 'uniform' }
            }]
        });

        uniformBindGroup = device.createBindGroup({
            layout: uniformBindGroupLayout,
            entries: [{
                binding: 0,
                resource: { buffer: uniformBuffer }
            }]
        });

        // Texture Bind Group Layout & Sampler
        textureBindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: 'float' }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: { type: 'filtering' }
                }
            ]
        });

        linearSampler = device.createSampler({
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
            magFilter: 'linear',
            minFilter: 'linear'
        });

        // Vertex Buffers
        rectVbo = device.createBuffer({
            size: RECT_BUFFER_SIZE,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });

        sdfVbo = device.createBuffer({
            size: SDF_BUFFER_SIZE,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });

        imageVbo = device.createBuffer({
            size: IMAGE_BUFFER_SIZE,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });

        textVbo = device.createBuffer({
            size: TEXT_BUFFER_SIZE,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });

        const blendState = {
            color: {
                srcFactor: 'src-alpha',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add'
            },
            alpha: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add'
            }
        };

        // 1. Rect Pipeline
        const rectModule = device.createShaderModule({ code: rectShaderWgsl });
        await checkShader(rectModule, "Rect");
        rectPipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [uniformBindGroupLayout] }),
            vertex: {
                module: rectModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 24, // 6 floats
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x2' },
                        { shaderLocation: 1, offset: 8, format: 'float32x4' }
                    ]
                }]
            },
            fragment: {
                module: rectModule,
                entryPoint: 'fs_main',
                targets: [{ format, blend: blendState }]
            },
            primitive: { topology: 'triangle-list' }
        });

        // 2. SDF Pipeline
        const sdfModule = device.createShaderModule({ code: sdfShaderWgsl });
        await checkShader(sdfModule, "SDF");
        sdfPipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [uniformBindGroupLayout] }),
            vertex: {
                module: sdfModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 68, // 17 floats
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x2' },
                        { shaderLocation: 1, offset: 8, format: 'float32x4' },
                        { shaderLocation: 2, offset: 24, format: 'float32x2' },
                        { shaderLocation: 3, offset: 32, format: 'float32x2' },
                        { shaderLocation: 4, offset: 40, format: 'float32' },
                        { shaderLocation: 5, offset: 44, format: 'float32' },
                        { shaderLocation: 6, offset: 48, format: 'float32x4' },
                        { shaderLocation: 7, offset: 64, format: 'float32' }
                    ]
                }]
            },
            fragment: {
                module: sdfModule,
                entryPoint: 'fs_main',
                targets: [{ format, blend: blendState }]
            },
            primitive: { topology: 'triangle-list' }
        });

        // 3. Image Pipeline
        const imageModule = device.createShaderModule({ code: imageShaderWgsl });
        await checkShader(imageModule, "Image");
        imagePipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [uniformBindGroupLayout, textureBindGroupLayout] }),
            vertex: {
                module: imageModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 32, // 8 floats
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x2' },
                        { shaderLocation: 1, offset: 8, format: 'float32x2' },
                        { shaderLocation: 2, offset: 16, format: 'float32x4' }
                    ]
                }]
            },
            fragment: {
                module: imageModule,
                entryPoint: 'fs_main',
                targets: [{ format, blend: blendState }]
            },
            primitive: { topology: 'triangle-list' }
        });

        // 4. Text Pipeline
        const textModule = device.createShaderModule({ code: textShaderWgsl });
        await checkShader(textModule, "Text");
        textPipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [uniformBindGroupLayout, textureBindGroupLayout] }),
            vertex: {
                module: textModule,
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 32, // 8 floats
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x2' },
                        { shaderLocation: 1, offset: 8, format: 'float32x2' },
                        { shaderLocation: 2, offset: 16, format: 'float32x4' }
                    ]
                }]
            },
            fragment: {
                module: textModule,
                entryPoint: 'fs_main',
                targets: [{ format, blend: blendState }]
            },
            primitive: { topology: 'triangle-list' }
        });

        updateProjection(width, height);

        if (loadingOverlay) {
            loadingOverlay.style.opacity = '0';
            setTimeout(() => { loadingOverlay.style.display = 'none'; }, 400);
        }

        console.log(`[WebGPU] Initialized successfully. Resolution: ${width}x${height}`);
        return true;
    } catch (err) {
        console.error("WebGPU setup failed:", err);
        if (loadingOverlay) loadingOverlay.style.display = 'none';
        if (webgpuWarning) webgpuWarning.style.display = 'block';
        return false;
    }
}

function updateProjection(width, height) {
    if (!device || !uniformBuffer) return;
    const proj = new Float32Array([
        2.0 / width,  0.0,           0.0, 0.0,
        0.0,         -2.0 / height,  0.0, 0.0,
        0.0,          0.0,           1.0, 0.0,
        0.0,          0.0,           0.0, 1.0
    ]);
    device.queue.writeBuffer(uniformBuffer, 0, proj);
}

// ==========================================
// Frame Drawing & Bridge Functions
// ==========================================

function beginFrame(width, height, r, g, b, a) {
    if (!device || !context) return;
    if (width !== currentWidth || height !== currentHeight) {
        currentWidth = width;
        currentHeight = height;
        updateProjection(width, height);
    }

    rectBufferOffset = 0;
    sdfBufferOffset = 0;
    imageBufferOffset = 0;
    textBufferOffset = 0;

    currentEncoder = device.createCommandEncoder();
    const textureView = context.getCurrentTexture().createView();

    currentPass = currentEncoder.beginRenderPass({
        colorAttachments: [{
            view: textureView,
            clearValue: { r, g, b, a },
            loadOp: 'clear',
            storeOp: 'store'
        }]
    });
}

function toUint8Array(mem) {
    if (!mem) return new Uint8Array(0);
    if (mem instanceof Uint8Array) return mem;
    if (typeof mem._unsafe_create_view === 'function') {
        return mem._unsafe_create_view();
    }
    if (typeof mem.slice === 'function') {
        return mem.slice();
    }
    if (mem.buffer instanceof ArrayBuffer) {
        return new Uint8Array(mem.buffer, mem.byteOffset || 0, mem.byteLength ?? mem.length ?? 0);
    }
    return new Uint8Array(mem);
}

function safeWriteBuffer(targetBuffer, targetOffset, srcView, byteLength) {
    if (!device || !targetBuffer || !srcView || byteLength <= 0) return;
    try {
        const u8 = toUint8Array(srcView);
        device.queue.writeBuffer(targetBuffer, targetOffset, u8.buffer, u8.byteOffset, byteLength);
    } catch (e) {
        console.error('[safeWriteBuffer error]:', e);
    }
}

function drawRects(vertexBytes, vertexCount) {
    if (!currentPass || !rectPipeline || vertexCount === 0) return;
    const byteLength = vertexBytes.byteLength;
    if (rectBufferOffset + byteLength > RECT_BUFFER_SIZE) return;

    safeWriteBuffer(rectVbo, rectBufferOffset, vertexBytes, byteLength);
    currentPass.setPipeline(rectPipeline);
    currentPass.setBindGroup(0, uniformBindGroup);
    currentPass.setVertexBuffer(0, rectVbo, rectBufferOffset);
    currentPass.draw(vertexCount);
    rectBufferOffset += Math.ceil(byteLength / 4) * 4;
}

function drawSdf(vertexBytes, vertexCount) {
    if (!currentPass || !sdfPipeline || vertexCount === 0) return;
    const byteLength = vertexBytes.byteLength;
    if (sdfBufferOffset + byteLength > SDF_BUFFER_SIZE) return;

    safeWriteBuffer(sdfVbo, sdfBufferOffset, vertexBytes, byteLength);
    currentPass.setPipeline(sdfPipeline);
    currentPass.setBindGroup(0, uniformBindGroup);
    currentPass.setVertexBuffer(0, sdfVbo, sdfBufferOffset);
    currentPass.draw(vertexCount);
    sdfBufferOffset += Math.ceil(byteLength / 4) * 4;
}

function drawImage(vertexBytes, vertexCount, textureId) {
    if (!currentPass || !imagePipeline || vertexCount === 0) return;
    const texObj = textures.get(textureId);
    if (!texObj) return;

    const byteLength = vertexBytes.byteLength;
    if (imageBufferOffset + byteLength > IMAGE_BUFFER_SIZE) return;

    safeWriteBuffer(imageVbo, imageBufferOffset, vertexBytes, byteLength);
    currentPass.setPipeline(imagePipeline);
    currentPass.setBindGroup(0, uniformBindGroup);
    currentPass.setBindGroup(1, texObj.bindGroup);
    currentPass.setVertexBuffer(0, imageVbo, imageBufferOffset);
    currentPass.draw(vertexCount);
    imageBufferOffset += Math.ceil(byteLength / 4) * 4;
}

function drawText(vertexBytes, vertexCount, fontTextureId) {
    if (!currentPass || !textPipeline || vertexCount === 0) return;
    const texObj = textures.get(fontTextureId);
    if (!texObj) return;

    const byteLength = vertexBytes.byteLength;
    if (textBufferOffset + byteLength > TEXT_BUFFER_SIZE) return;

    safeWriteBuffer(textVbo, textBufferOffset, vertexBytes, byteLength);
    currentPass.setPipeline(textPipeline);
    currentPass.setBindGroup(0, uniformBindGroup);
    currentPass.setBindGroup(1, texObj.bindGroup);
    currentPass.setVertexBuffer(0, textVbo, textBufferOffset);
    currentPass.draw(vertexCount);
    textBufferOffset += Math.ceil(byteLength / 4) * 4;
}

function setScissor(x, y, w, h) {
    if (!currentPass) return;
    const targetW = canvas.width;
    const targetH = canvas.height;
    const x0 = Math.max(0, Math.min(targetW - 1, Math.floor(x)));
    const y0 = Math.max(0, Math.min(targetH - 1, Math.floor(y)));
    const x1 = Math.max(x0 + 1, Math.min(targetW, Math.floor(x + w)));
    const y1 = Math.max(y0 + 1, Math.min(targetH, Math.floor(y + h)));
    currentPass.setScissorRect(x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0));
}

function resetScissor() {
    if (!currentPass) return;
    currentPass.setScissorRect(0, 0, canvas.width, canvas.height);
}

function endFrame() {
    if (!currentPass || !currentEncoder) return;
    currentPass.end();
    device.queue.submit([currentEncoder.finish()]);
    currentPass = null;
    currentEncoder = null;
}

function createTexture(width, height, formatMode, pixelData) {
    if (!device) return 0;
    const id = nextTextureId++;
    const isR8 = formatMode === 1;
    const texFormat = isR8 ? 'r8unorm' : 'rgba8unorm';
    const bpp = isR8 ? 1 : 4;

    const texture = device.createTexture({
        size: [width, height],
        format: texFormat,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
    });

    const unpaddedBytesPerRow = width * bpp;
    const bytesPerRow = Math.ceil(unpaddedBytesPerRow / 256) * 256;

    const u8 = toUint8Array(pixelData);
    let uploadData;
    if (bytesPerRow === unpaddedBytesPerRow) {
        uploadData = new Uint8Array(u8.buffer, u8.byteOffset, u8.byteLength);
    } else {
        uploadData = new Uint8Array(bytesPerRow * height);
        for (let y = 0; y < height; y++) {
            const row = new Uint8Array(u8.buffer, u8.byteOffset + y * unpaddedBytesPerRow, unpaddedBytesPerRow);
            uploadData.set(row, y * bytesPerRow);
        }
    }

    device.queue.writeTexture(
        { texture },
        uploadData,
        { bytesPerRow },
        { width, height }
    );

    const view = texture.createView();
    const bindGroup = device.createBindGroup({
        layout: textureBindGroupLayout,
        entries: [
            { binding: 0, resource: view },
            { binding: 1, resource: linearSampler }
        ]
    });

    textures.set(id, { id, texture, view, bindGroup, width, height });
    return id;
}

function destroyTexture(textureId) {
    const texObj = textures.get(textureId);
    if (texObj) {
        texObj.texture.destroy();
        textures.delete(textureId);
    }
}

let inMemoryClipboard = "";
function getClipboard() {
    return inMemoryClipboard;
}

function setClipboard(text) {
    inMemoryClipboard = text;
    navigator.clipboard?.writeText(text).catch(() => {});
}

function setCursor(cursorCss) {
    canvas.style.cursor = cursorCss;
}

function setTitle(title) {
    document.title = title;
}

// ==========================================
// Setup DotNet Runtime & Module Imports
// ==========================================

const { setModuleImports, getAssemblyExports, getConfig } = await dotnet
    .withDiagnosticTracing(false)
    .withApplicationArgumentsFromQuery()
    .create();

setModuleImports('tinywindow_webgpu', {
    InitWebGPU: (w, h) => initWebGpu(w, h),
    BeginFrame: (w, h, r, g, b, a) => beginFrame(w, h, r, g, b, a),
    DrawRects: (data, count) => drawRects(data, count),
    DrawSdf: (data, count) => drawSdf(data, count),
    DrawImage: (data, count, texId) => drawImage(data, count, texId),
    DrawText: (data, count, fontTexId) => drawText(data, count, fontTexId),
    SetScissor: (x, y, w, h) => setScissor(x, y, w, h),
    ResetScissor: () => resetScissor(),
    EndFrame: () => endFrame(),
    CreateTexture: (w, h, fmt, data) => createTexture(w, h, fmt, data),
    DestroyTexture: (id) => destroyTexture(id),
    SetCursor: (css) => setCursor(css),
    SetTitle: (title) => setTitle(title),
    GetClipboard: () => getClipboard(),
    SetClipboard: (txt) => setClipboard(txt),
    ShowKeyboard: () => showVirtualKeyboard(),
    HideKeyboard: () => hideVirtualKeyboard()
});

function resolveBridge(obj) {
    if (!obj) return null;
    if (typeof obj.OnAnimationFrame === 'function') return obj;
    if (obj.WebGPUBridge && typeof obj.WebGPUBridge.OnAnimationFrame === 'function') return obj.WebGPUBridge;
    if (obj.BrowserBridge && typeof obj.BrowserBridge.OnAnimationFrame === 'function') return obj.BrowserBridge;
    if (obj.TinyWindow?.Browser?.WebGPUBridge && typeof obj.TinyWindow.Browser.WebGPUBridge.OnAnimationFrame === 'function') {
        return obj.TinyWindow.Browser.WebGPUBridge;
    }
    if (obj.TinyWindow?.Test?.Browser?.BrowserBridge && typeof obj.TinyWindow.Test.Browser.BrowserBridge.OnAnimationFrame === 'function') {
        return obj.TinyWindow.Test.Browser.BrowserBridge;
    }
    for (const key of Object.keys(obj)) {
        if (typeof obj[key] === 'object' && obj[key] !== null) {
            const found = resolveBridge(obj[key]);
            if (found) return found;
        }
    }
    return null;
}

try {
    const rawExports = await getAssemblyExports("TinyWindow.Browser");
    bridge = resolveBridge(rawExports);
    console.log('[TinyWindow] Initial bridge lookup:', bridge);
} catch (e) {
    console.warn('[TinyWindow] getAssemblyExports before run:', e);
}

// ==========================================
// Canvas Events & Input Handling
// ==========================================

function getCanvasCoords(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
        x: (clientX - rect.left) * scaleX,
        y: (clientY - rect.top) * scaleY
    };
}

function resizeCanvas(force = false) {
    const dpr = window.devicePixelRatio || 1;
    const displayWidth = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const displayHeight = Math.max(1, Math.round(canvas.clientHeight * dpr));

    const changed = canvas.width !== displayWidth || canvas.height !== displayHeight;
    if (changed) {
        canvas.width = displayWidth;
        canvas.height = displayHeight;
    }

    if ((changed || force) && bridge) {
        bridge.OnResize(displayWidth, displayHeight);
    }
}

window.addEventListener('resize', () => resizeCanvas(false));
resizeCanvas(false);

// Block browser context menu (right click) so in-app ContextMenu works without interference
window.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    return false;
});

// Prevent file/text drag-and-drop into browser window
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => e.preventDefault());

// Synchronize clipboard with system
window.addEventListener('paste', (e) => {
    const text = e.clipboardData?.getData('text');
    if (text) inMemoryClipboard = text;
});
window.addEventListener('copy', (e) => {
    if (inMemoryClipboard) {
        e.clipboardData?.setData('text/plain', inMemoryClipboard);
        e.preventDefault();
    }
});
window.addEventListener('cut', (e) => {
    if (inMemoryClipboard) {
        e.clipboardData?.setData('text/plain', inMemoryClipboard);
        e.preventDefault();
    }
});

canvas.addEventListener('mousemove', (e) => {
    const { x, y } = getCanvasCoords(e.clientX, e.clientY);
    bridge?.OnMouseMove(x, y);
});

canvas.addEventListener('mousedown', (e) => {
    canvas.focus();
    // Prevent middle-click autoscroll and browser right-click gestures
    if (e.button === 1 || e.button === 2) {
        e.preventDefault();
    }
    bridge?.OnMouseDown(e.button);
});

canvas.addEventListener('mouseup', (e) => {
    bridge?.OnMouseUp(e.button);
});

canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    let dx = e.deltaX;
    let dy = e.deltaY;

    // Convert lines/pages to approximate pixels
    if (e.deltaMode === 1) { // DOM_DELTA_LINE
        dx *= 33.33;
        dy *= 33.33;
    } else if (e.deltaMode === 2) { // DOM_DELTA_PAGE
        dx *= 100.0;
        dy *= 100.0;
    }

    // Standard mouse notch is ~100px.
    // Invert sign so scrolling UP is positive (+1.0) and DOWN is negative (-1.0),
    // matching GLFW / SilkNET conventions in TinyWindow.
    const normX = -dx / 100.0;
    const normY = -dy / 100.0;

    bridge?.OnMouseWheel(normX, normY);
}, { passive: false });

// Touch Support
canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    if (e.touches.length > 0) {
        const touch = e.touches[0];
        const { x, y } = getCanvasCoords(touch.clientX, touch.clientY);
        bridge?.OnTouchStart(x, y);
    }
}, { passive: false });

canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (e.touches.length > 0) {
        const touch = e.touches[0];
        const { x, y } = getCanvasCoords(touch.clientX, touch.clientY);
        bridge?.OnTouchMove(x, y);
    }
}, { passive: false });

// ==========================================
// Virtual / Mobile Soft Keyboard Support
// ==========================================

const virtualInput = document.createElement('input');
virtualInput.type = 'text';
virtualInput.id = 'tinywindow-virtual-input';
virtualInput.setAttribute('autocomplete', 'off');
virtualInput.setAttribute('autocorrect', 'off');
virtualInput.setAttribute('autocapitalize', 'off');
virtualInput.setAttribute('spellcheck', 'false');

Object.assign(virtualInput.style, {
    position: 'fixed',
    top: '0px',
    left: '0px',
    width: '1px',
    height: '1px',
    opacity: '0.001',
    pointerEvents: 'none',
    border: 'none',
    margin: '0',
    padding: '0',
    zIndex: '-1',
    fontSize: '16px', // 16px prevents iOS Safari from automatically zooming the page
    outline: 'none',
    webkitTapHighlightColor: 'transparent'
});
document.body.appendChild(virtualInput);

const ZERO_WIDTH_SPACE = '\u200B';
virtualInput.value = ZERO_WIDTH_SPACE;

let isVirtualKeyboardActive = false;
let lastInteractionWasTouch = false;

window.addEventListener('touchstart', () => { lastInteractionWasTouch = true; }, { passive: true });
window.addEventListener('mousedown', (e) => {
    if (e.sourceCapabilities && e.sourceCapabilities.firesTouchEvents) return;
    lastInteractionWasTouch = false;
}, { passive: true });

function isMobileOrTouchUser() {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)
        || (navigator.maxTouchPoints > 0 && lastInteractionWasTouch)
        || ('ontouchstart' in window && lastInteractionWasTouch);
}

function showVirtualKeyboard() {
    isVirtualKeyboardActive = true;
    if (!isMobileOrTouchUser()) return;

    virtualInput.value = ZERO_WIDTH_SPACE;
    try {
        virtualInput.focus({ preventScroll: true });
    } catch {
        virtualInput.focus();
    }
}

function hideVirtualKeyboard() {
    isVirtualKeyboardActive = false;
    virtualInput.blur();
}

let lastBackspaceTime = 0;
function dispatchBackspace() {
    const now = performance.now();
    if (now - lastBackspaceTime < 30) return;
    lastBackspaceTime = now;
    bridge?.OnKeyDown('Backspace', 8);
    bridge?.OnKeyUp('Backspace', 8);
}

let lastEnterTime = 0;
function dispatchEnter() {
    const now = performance.now();
    if (now - lastEnterTime < 30) return;
    lastEnterTime = now;
    bridge?.OnKeyDown('Enter', 13);
    bridge?.OnKeyUp('Enter', 13);
}

virtualInput.addEventListener('keydown', (e) => {
    if (e.key === 'Backspace' || e.code === 'Backspace' || e.keyCode === 8) {
        dispatchBackspace();
        virtualInput.value = ZERO_WIDTH_SPACE;
        e.preventDefault();
        return;
    }
    if (e.key === 'Enter' || e.code === 'Enter' || e.keyCode === 13) {
        dispatchEnter();
        virtualInput.value = ZERO_WIDTH_SPACE;
        e.preventDefault();
        return;
    }
});

virtualInput.addEventListener('input', (e) => {
    if (e.inputType === 'deleteContentBackward') {
        dispatchBackspace();
        virtualInput.value = ZERO_WIDTH_SPACE;
        return;
    }
    if (e.inputType === 'insertLineBreak') {
        dispatchEnter();
        virtualInput.value = ZERO_WIDTH_SPACE;
        return;
    }

    const val = virtualInput.value;
    const cleanText = val.replaceAll(ZERO_WIDTH_SPACE, '');
    if (cleanText.length > 0) {
        for (let i = 0; i < cleanText.length; i++) {
            bridge?.OnTextInput(cleanText.charCodeAt(i));
        }
    } else if (val.length === 0) {
        dispatchBackspace();
    }
    virtualInput.value = ZERO_WIDTH_SPACE;
});

canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    if (e.changedTouches.length > 0) {
        const touch = e.changedTouches[0];
        const { x, y } = getCanvasCoords(touch.clientX, touch.clientY);
        bridge?.OnTouchEnd(x, y);
    }
    // Mobile virtual keyboard safety: if keyboard was requested, ensure input is focused
    if (isVirtualKeyboardActive && isMobileOrTouchUser() && document.activeElement !== virtualInput) {
        try {
            virtualInput.focus({ preventScroll: true });
        } catch { }
    }
}, { passive: false });

canvas.addEventListener('touchcancel', () => {
    bridge?.OnTouchCancel();
});

// Keyboard Support - Intercept all browser hotkeys & forward to MiniUI
window.addEventListener('keydown', (e) => {
    if (e.target === virtualInput) return;

    // 1. Function keys: F1..F12 (F12 DevTools, F5 Reload, F11 Fullscreen, F1 Help, F3 Find, F7 Caret, etc.)
    if (/^F([1-9]|1[0-2])$/.test(e.key) || (e.keyCode >= 112 && e.keyCode <= 123)) {
        e.preventDefault();
    }
    // 2. Navigation, scrolling & editing control keys
    else if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space', 'Tab', 'PageUp', 'PageDown', 'Home', 'End', 'Backspace', 'Escape'].includes(e.code)) {
        e.preventDefault();
    }
    // 3. Browser accelerator combinations (Ctrl+*, Alt+*, Meta+*)
    // Blocks Ctrl+S, Ctrl+P, Ctrl+F, Ctrl+G, Ctrl+O, Ctrl+U, Ctrl+D, Ctrl+H, Ctrl+J, Ctrl+R, Ctrl+Shift+I, etc.
    else if (e.ctrlKey || e.metaKey || e.altKey) {
        e.preventDefault();
    }

    bridge?.OnKeyDown(e.code, e.keyCode);
});

window.addEventListener('keyup', (e) => {
    if (e.target === virtualInput) return;
    bridge?.OnKeyUp(e.code, e.keyCode);
});

window.addEventListener('keypress', (e) => {
    if (e.target === virtualInput) return;
    if (e.charCode && e.charCode > 0) {
        bridge?.OnTextInput(e.charCode);
    }
});

// ==========================================
// Animation Loop & FPS Tracking
// ==========================================

let lastTime = 0;
let frameCount = 0;
let lastFpsTime = performance.now();

let initialResizeDone = false;

function animate(time) {
    if (lastTime === 0) lastTime = time;
    const dt = Math.min((time - lastTime) / 1000.0, 0.1);
    lastTime = time;

    frameCount++;
    const now = performance.now();
    if (now - lastFpsTime >= 1000) {
        const fps = Math.round((frameCount * 1000) / (now - lastFpsTime));
        if (fpsCounter) fpsCounter.textContent = `${fps} FPS`;
        frameCount = 0;
        lastFpsTime = now;
    }

    if (bridge) {
        if (!initialResizeDone) {
            initialResizeDone = true;
            resizeCanvas(true);
        }
        bridge.OnAnimationFrame(dt);
    }

    requestAnimationFrame(animate);
}

requestAnimationFrame(animate);

// ==========================================
// Start DotNet Program
// ==========================================

const runPromise = dotnet.run();

// Post-run bridge resolution
if (!bridge) {
    try {
        const browserExports = await getAssemblyExports("TinyWindow.Browser");
        bridge = resolveBridge(browserExports);
        console.log('[TinyWindow] Post-run browser bridge lookup:', bridge);
    } catch (e) {
        console.warn('[TinyWindow] Error getting TinyWindow.Browser exports:', e);
    }
}

if (!bridge) {
    try {
        const config = getConfig();
        const mainExports = await getAssemblyExports(config.mainAssemblyName);
        bridge = resolveBridge(mainExports);
        console.log('[TinyWindow] Post-run main bridge lookup:', bridge);
    } catch (e) {
        console.warn('[TinyWindow] Error getting main assembly exports:', e);
    }
}

if (bridge) {
    console.log('[TinyWindow] Bridge active and connected! Triggering initial resize...');
    resizeCanvas(true);
} else {
    console.error('[TinyWindow] FATAL: Failed to resolve WebGPUBridge exports!');
}

await runPromise;
