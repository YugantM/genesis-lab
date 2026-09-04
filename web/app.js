const TILE = 128;
const GRID = 3;
const COUNT = 9;
const ATLAS = TILE * GRID;
const canvas = document.querySelector('#zoo');
const gl = canvas.getContext('webgl2', { antialias: false, preserveDrawingBuffer: true });
if (!gl) throw new Error('Genesis Zoo requires WebGL 2');
gl.getExtension('EXT_color_buffer_float');

const vertex = `#version 300 es
in vec2 p; out vec2 uv;
void main(){ uv=(p+1.0)*.5; gl_Position=vec4(p,0,1); }`;
const update = `#version 300 es
precision highp float; out vec4 outColor; uniform sampler2D state;
uniform float mu; uniform float sigma; uniform ivec2 origin;
const int TILE=128;
void main(){
  ivec2 pixel=ivec2(gl_FragCoord.xy), local=pixel-origin;
  float sum=0.0, weight=0.0;
  for(int y=-13;y<=13;y++) for(int x=-13;x<=13;x++){
    float r=length(vec2(x,y))/13.0;
    if(r>0.0 && r<1.0){
      float k=exp(4.0-1.0/(r*(1.0-r)));
      ivec2 q=ivec2((local.x+x+TILE)%TILE,(local.y+y+TILE)%TILE)+origin;
      sum+=texelFetch(state,q,0).r*k; weight+=k;
    }
  }
  float u=sum/weight;
  float growth=2.0*exp(-pow(u-mu,2.0)/(2.0*sigma*sigma))-1.0;
  outColor=vec4(clamp(texelFetch(state,pixel,0).r+.1*growth,0.0,1.0),0,0,1);
}`;
const display = `#version 300 es
precision highp float; in vec2 uv; out vec4 outColor; uniform sampler2D state;
void main(){
  float v=texture(state,uv).r; vec2 g=uv*3.0;
  float zone=floor(g.x)+floor(g.y)*3.0;
  float halo=smoothstep(.008,.25,v), core=smoothstep(.3,.88,v);
  vec3 deep=vec3(.008,.025,.020);
  vec3 moss=mix(vec3(.055,.25,.15),vec3(.08,.19,.25),fract(zone*.381));
  vec3 acid=mix(vec3(.62,1.0,.36),vec3(.35,1.0,.72),fract(zone*.217));
  vec3 c=mix(deep,moss,halo); c=mix(c,acid,smoothstep(.1,.68,v));
  c=mix(c,vec3(.9,1.0,.84),core*.75); outColor=vec4(c,1);
}`;

function shader(type, source) {
  const value = gl.createShader(type);
  gl.shaderSource(value, source);
  gl.compileShader(value);
  if (!gl.getShaderParameter(value, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(value));
  return value;
}
function program(fragment) {
  const value = gl.createProgram();
  gl.attachShader(value, shader(gl.VERTEX_SHADER, vertex));
  gl.attachShader(value, shader(gl.FRAGMENT_SHADER, fragment));
  gl.linkProgram(value);
  if (!gl.getProgramParameter(value, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(value));
  return value;
}
const updateProgram = program(update);
const displayProgram = program(display);
const quad = gl.createBuffer();
const framebuffer = gl.createFramebuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quad);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]), gl.STATIC_DRAW);
function bindQuad(value) {
  const location = gl.getAttribLocation(value, 'p');
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.enableVertexAttribArray(location);
  gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 0, 0);
}
function createTexture() {
  const value = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, value);
  // R32F linear filtering is an optional extension in WebGL 2. NEAREST keeps
  // the simulation texture complete on every conforming implementation.
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, ATLAS, ATLAS, 0, gl.RED, gl.FLOAT, null);
  return value;
}

const textures = [createTexture(), createTexture()];
let front = 0;
let generation = 0;
let paused = false;
let records = [];
let benchmark = {};
let selected = 0;
let lastFrame = 0;
let noticeTimer;
let updateCursor = 0;
let labVisible = true;
let serverMode = false;

function cellValue(code) {
  if (code === '.' || code === 'b') return 0;
  if (code === 'o') return 1;
  if (code.length === 1) return (code.charCodeAt(0) - 64) / 255;
  return ((code.charCodeAt(0) - 112) * 24 + code.charCodeAt(1) - 65 + 25) / 255;
}
function decodeRle(encoded) {
  const rows = [];
  let row = [], count = '', prefix = '';
  for (const char of encoded.replace(/!$/, '') + '$') {
    if (/\d/.test(char)) count += char;
    else if ('pqrstuvwxy@'.includes(char)) prefix = char;
    else if (char === '$') {
      rows.push(row);
      for (let i = 1; i < (count ? +count : 1); i++) rows.push([]);
      row = []; count = ''; prefix = '';
    } else {
      for (let i = 0; i < (count ? +count : 1); i++) row.push(cellValue(prefix + char));
      count = ''; prefix = '';
    }
  }
  return rows;
}
function physicalTile(index) {
  return { x: (index % GRID) * TILE, y: (GRID - 1 - Math.floor(index / GRID)) * TILE };
}
function initialAtlas() {
  const data = new Float32Array(ATLAS * ATLAS);
  records.forEach((record, index) => {
    const rows = decodeRle(record.initial_state.cells);
    const width = Math.max(...rows.map(row => row.length));
    if (rows.length > TILE || width > TILE) throw new Error(`${record.name} exceeds habitat size`);
    const tile = physicalTile(index);
    const top = Math.floor((TILE - rows.length) / 2);
    const left = Math.floor((TILE - width) / 2);
    rows.forEach((row, y) => row.forEach((value, x) => {
      data[(tile.y + top + y) * ATLAS + tile.x + left + x] = value;
    }));
  });
  return data;
}
function upload(data) {
  for (const texture of textures) {
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, ATLAS, ATLAS, gl.RED, gl.FLOAT, data);
  }
  front = 0;
  generation = 0;
  updateCursor = 0;
}
function stepHabitat(index) {
  const source = textures[front];
  const target = textures[1-front];
  // Preserve the eight habitats not being advanced this frame. Updating two
  // tiles at a time avoids the former nine-habitat shader burst.
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, source, 0);
  gl.bindTexture(gl.TEXTURE_2D, target);
  gl.copyTexSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 0, 0, ATLAS, ATLAS);
  gl.useProgram(updateProgram);
  bindQuad(updateProgram);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, target, 0);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, source);
  gl.uniform1i(gl.getUniformLocation(updateProgram, 'state'), 0);
  const record = records[index];
  const tile = physicalTile(index);
  gl.viewport(tile.x, tile.y, TILE, TILE);
  gl.uniform1f(gl.getUniformLocation(updateProgram, 'mu'), record.parameters.growth_center);
  gl.uniform1f(gl.getUniformLocation(updateProgram, 'sigma'), record.parameters.growth_width);
  gl.uniform2i(gl.getUniformLocation(updateProgram, 'origin'), tile.x, tile.y);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  front = 1 - front;
  updateCursor = (updateCursor + 1) % COUNT;
  if (updateCursor === 0) generation++;
}
function step() { for (let index = 0; index < COUNT; index++) stepHabitat(index); }
function draw() {
  gl.useProgram(displayProgram);
  bindQuad(displayProgram);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, textures[front]);
  gl.uniform1i(gl.getUniformLocation(displayProgram, 'state'), 0);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  document.querySelector('#generation').textContent = `GEN ${String(generation).padStart(6, '0')}`;
}
function readAtlas() {
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, textures[front], 0);
  const pixels = new Float32Array(ATLAS * ATLAS);
  gl.readPixels(0, 0, ATLAS, ATLAS, gl.RED, gl.FLOAT, pixels);
  return pixels;
}
function habitatValues(pixels, index) {
  const tile = physicalTile(index);
  const values = [];
  for (let y = 0; y < TILE; y++) for (let x = 0; x < TILE; x++) {
    values.push(pixels[(tile.y + y) * ATLAS + tile.x + x]);
  }
  return values;
}
function measure() {
  const pixels = readAtlas();
  records.forEach((_, index) => {
    const total = habitatValues(pixels, index).reduce((a, b) => a + b, 0);
    const node = document.querySelector(`[data-mass="${index}"]`);
    if (node) node.textContent = total < .5 ? 'extinct' : `mass ${total.toFixed(0)}`;
    document.querySelector(`[data-index="${index}"]`)?.classList.toggle('extinct', total < .5);
  });
}
function showEvent(message) {
  const node = document.querySelector('#eventMessage');
  node.textContent = message;
  node.classList.add('visible');
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => node.classList.remove('visible'), 2600);
}
function injure(indexes) {
  const pixels = readAtlas();
  for (const index of indexes) {
    const tile = physicalTile(index);
    const cells = [];
    let total = 0, sx = 0, cx = 0, sy = 0, cy = 0;
    for (let y = 0; y < TILE; y++) for (let x = 0; x < TILE; x++) {
      const offset = (tile.y + y) * ATLAS + tile.x + x;
      const value = pixels[offset], ax = x * 2 * Math.PI / TILE, ay = y * 2 * Math.PI / TILE;
      total += value; sx += value * Math.sin(ax); cx += value * Math.cos(ax);
      sy += value * Math.sin(ay); cy += value * Math.cos(ay);
    }
    if (total < 1) continue;
    const centerX = (Math.atan2(sx, cx) / (2*Math.PI) + 1) % 1 * TILE;
    const centerY = (Math.atan2(sy, cy) / (2*Math.PI) + 1) % 1 * TILE;
    for (let y = 0; y < TILE; y++) for (let x = 0; x < TILE; x++) {
      const dx = Math.min(Math.abs(x-centerX), TILE-Math.abs(x-centerX));
      const dy = Math.min(Math.abs(y-centerY), TILE-Math.abs(y-centerY));
      const offset = (tile.y+y)*ATLAS+tile.x+x;
      cells.push({ distance: Math.round(Math.hypot(dx,dy)*1e6)/1e6, offset, value: pixels[offset] });
    }
    cells.sort((a,b) => a.distance-b.distance);
    let removed = 0, cursor = 0;
    const target = total * .05;
    while (cursor < cells.length && removed < target) {
      const distance = cells[cursor].distance, ring = [];
      while (cursor < cells.length && cells[cursor].distance === distance) ring.push(cells[cursor++]);
      const ringMass = ring.reduce((sum, cell) => sum + cell.value, 0);
      if (removed + ringMass >= target && ringMass > 0) {
        const fraction = (target - removed) / ringMass;
        ring.forEach(cell => pixels[cell.offset] *= 1 - fraction);
        removed = target;
      } else {
        ring.forEach(cell => pixels[cell.offset] = 0);
        removed += ringMass;
      }
    }
    const habitat = document.querySelector(`[data-index="${index}"]`);
    habitat?.classList.remove('pulse');
    requestAnimationFrame(() => habitat?.classList.add('pulse'));
  }
  gl.bindTexture(gl.TEXTURE_2D, textures[front]);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, ATLAS, ATLAS, gl.RED, gl.FLOAT, pixels);
  measure();
  showEvent(indexes.length === 1 ? `${records[indexes[0]].name} · 5% central injury` : 'Extinction pulse delivered · 5% mass per habitat');
}
function classificationText(item) {
  if (!item) return 'Catalogue ancestor';
  const delta = Math.round(item.difference * 100);
  const label = item.classification.replaceAll('-', ' ');
  return `${label}${delta ? ` · ${delta > 0 ? '+' : ''}${delta} points` : ''}`;
}
function selectHabitat(index) {
  selected = index;
  document.querySelectorAll('.habitat').forEach((node, i) => node.setAttribute('aria-pressed', String(i === index)));
  const record = records[index];
  const code = record.provenance?.catalogue_code || 'GEN–001';
  const isGenesis = record.id === 'specimen-001-genesis';
  const item = benchmark[isGenesis ? 'O2u' : code];
  const parent = item?.parent_recovery_rate || 0;
  const transfer = item?.genesis_transfer_recovery_rate || 0;
  document.querySelector('#selectedKind').textContent = isGenesis ? 'Local descendant · candidate 942' : 'Pinned catalogue ancestor';
  document.querySelector('#selectedName').textContent = record.name;
  document.querySelector('#selectedSummary').textContent = isGenesis
    ? 'The first organism discovered here: an Orbium descendant with a narrow but reproducible central-injury advantage.'
    : 'A stable parental species in the normalized injury benchmark. Its habitat uses the species’ native rule—not Genesis 001’s rule.';
  document.querySelector('#selectedCode').textContent = isGenesis ? 'GEN–001' : code;
  document.querySelector('#selectedMu').textContent = record.parameters.growth_center.toFixed(4);
  document.querySelector('#selectedSigma').textContent = record.parameters.growth_width.toFixed(4);
  document.querySelector('#selectedLineage').textContent = isGenesis ? 'O2u → #942' : 'original catalogue';
  document.querySelector('#parentRate').textContent = `${Math.round(parent*100)}%`;
  document.querySelector('#transferRate').textContent = `${Math.round(transfer*100)}%`;
  document.querySelector('#parentBar').style.width = `${parent*100}%`;
  document.querySelector('#transferBar').style.width = `${transfer*100}%`;
  document.querySelector('#classification').textContent = isGenesis
    ? 'Beneficial · central recovery 0% → 100%'
    : classificationText(item);
}
function buildHabitats() {
  const grid = document.querySelector('#habitatGrid');
  records.forEach((record, index) => {
    const code = record.id === 'specimen-001-genesis' ? 'GEN–001' : record.provenance.catalogue_code;
    const button = document.createElement('button');
    button.type = 'button'; button.className = 'habitat'; button.dataset.index = index;
    button.setAttribute('aria-pressed', String(index === 0));
    button.setAttribute('aria-label', `Inspect ${record.name}`);
    button.innerHTML = `<span class="habitat-top"><b class="habitat-name">${record.name}</b><span class="habitat-state">${index === 0 ? 'born here' : code}</span></span><span class="habitat-bottom"><i></i><span data-mass="${index}">stabilising</span></span>`;
    button.onclick = () => selectHabitat(index);
    grid.appendChild(button);
  });
}
function reset() {
  upload(initialAtlas()); measure(); showEvent('Population restored from genomic records');
}
function loop(now) {
  if (!serverMode && labVisible && now - lastFrame > 32) {
    if (!paused) {
      stepHabitat(updateCursor);
      stepHabitat(updateCursor);
    }
    draw();
    if (updateCursor === 0 && generation % 20 === 0) measure();
    lastFrame = now;
  }
  requestAnimationFrame(loop);
}

if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    labVisible = !serverMode && entries[0].isIntersecting;
    if (labVisible && records.length) draw();
  }, { rootMargin: '160px' });
  observer.observe(document.querySelector('.zoo-shell'));
}

window.genesisZooClient = {
  enterServerMode: () => {
    serverMode = true;
    labVisible = false;
  },
  getSelected: () => selected,
};

document.querySelector('#pauseAll').onclick = event => {
  paused = !paused;
  event.currentTarget.textContent = paused ? 'Resume' : 'Pause';
  showEvent(paused ? 'All habitats paused' : 'Time resumed');
};
document.querySelector('#resetAll').onclick = reset;
document.querySelector('#stressAll').onclick = () => injure([...Array(COUNT).keys()]);
document.querySelector('#stressSelected').onclick = () => injure([selected]);

Promise.all([
  'specimens/genesis-001.json',
  'specimens/species-benchmark.json',
  'data/multispecies-transfer.json',
].map(path => fetch(path).then(response => {
  if (!response.ok) throw new Error(`Failed to load ${path}`);
  return response.json();
}))).then(async ([genesis, manifest, results]) => {
  const ancestors = await Promise.all(manifest.entries.map(entry =>
    fetch(`specimens/${entry.path}`).then(response => response.json())
  ));
  records = [genesis, ...ancestors];
  benchmark = Object.fromEntries(results.species_comparison.map(item => [item.species_code, item]));
  buildHabitats(); upload(initialAtlas()); selectHabitat(0); measure(); requestAnimationFrame(loop);
}).catch(error => { showEvent(error.message); throw error; });
