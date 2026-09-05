(() => {
  const canvas = document.querySelector('#ecosystem');
  if (!canvas) return;
  const disclosure = document.querySelector('#exploratoryWorld');
  const ctx = canvas.getContext('2d', { alpha: false });
  const W = canvas.width;
  const H = canvas.height;
  const TWO_PI = Math.PI * 2;
  const founderNames = [
    'Genesis', 'Orbium', 'Gyrorbium', 'Synorbium', 'Scutium',
    'Paraptera', 'Helicium S', 'Helicium C', 'Circium',
  ];
  const founderHues = [92, 154, 187, 38, 273, 18, 323, 211, 61];

  let seed = 0x9e3779b9;
  let epoch = 0;
  let elapsed = 0;
  let births = 0;
  let deaths = 0;
  let nextId = 1;
  let nextLineage = 10;
  let paused = false;
  let agents = [];
  let nutrients = [];
  let events = [];
  let selectedId = null;
  let last = performance.now();
  let uiAccumulator = 0;
  let shock = null;
  let ecosystemVisible = !('IntersectionObserver' in window);
  let initialized = false;
  let frameId = null;

  function random() {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let value = seed;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  }

  function between(min, max) { return min + random() * (max - min); }
  function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
  function wrap(value, max) { return (value + max) % max; }
  function delta(a, b, max) {
    let value = b - a;
    if (value > max / 2) value -= max;
    if (value < -max / 2) value += max;
    return value;
  }
  function distance(a, b) {
    const dx = delta(a.x, b.x, W);
    const dy = delta(a.y, b.y, H);
    return { dx, dy, length: Math.hypot(dx, dy) };
  }

  function makeGenome(index) {
    return {
      speed: clamp(.42 + (index % 3) * .14 + between(-.05, .05), .22, .92),
      efficiency: clamp(.58 + ((index * 2) % 5) * .07 + between(-.04, .04), .4, .95),
      fertility: clamp(.48 + ((index * 3) % 4) * .1 + between(-.04, .04), .35, .9),
      sense: 85 + (index % 4) * 19 + between(-8, 8),
      social: between(-.65, .75),
      aggression: clamp(.12 + (index % 5) * .12, .08, .72),
      size: between(.78, 1.25),
      lobes: 3 + (index % 5),
      turn: between(.55, 1.25),
      longevity: between(78, 132),
    };
  }

  function mutateGenome(parent) {
    const gene = { ...parent };
    const scale = .075;
    gene.speed = clamp(gene.speed + between(-scale, scale), .18, 1);
    gene.efficiency = clamp(gene.efficiency + between(-scale, scale), .3, 1);
    gene.fertility = clamp(gene.fertility + between(-scale, scale), .25, 1);
    gene.sense = clamp(gene.sense + between(-14, 14), 55, 175);
    gene.social = clamp(gene.social + between(-.13, .13), -.9, .9);
    gene.aggression = clamp(gene.aggression + between(-.1, .1), 0, .9);
    gene.size = clamp(gene.size + between(-.09, .09), .58, 1.48);
    gene.turn = clamp(gene.turn + between(-.1, .1), .35, 1.55);
    gene.longevity = clamp(gene.longevity + between(-9, 9), 62, 160);
    if (random() < .08) gene.lobes = clamp(gene.lobes + (random() < .5 ? -1 : 1), 3, 8);
    return gene;
  }

  function geneDistance(a, b) {
    return Math.abs(a.speed - b.speed) + Math.abs(a.efficiency - b.efficiency)
      + Math.abs(a.fertility - b.fertility) + Math.abs(a.social - b.social)
      + Math.abs(a.aggression - b.aggression) + Math.abs(a.size - b.size);
  }

  function createAgent({ lineage, lineageName, hue, generation = 0, parentId = null, genome, x, y, energy = 88 }) {
    const angle = between(0, TWO_PI);
    const agent = {
      id: nextId++, lineage, lineageName, hue, generation, parentId, genome,
      x, y, vx: Math.cos(angle) * 12, vy: Math.sin(angle) * 12,
      energy, age: 0, phase: between(0, TWO_PI), history: [],
    };
    agents.push(agent);
    return agent;
  }

  function addNutrient(x = between(0, W), y = between(0, H), value = between(.35, 1)) {
    nutrients.push({ x, y, value, phase: between(0, TWO_PI) });
  }

  function note(text, tone = 'normal') {
    events.unshift({ text, tone, time: elapsed });
    events = events.slice(0, 7);
    renderEvents();
  }

  function resetWorld() {
    epoch++;
    elapsed = 0;
    births = 0;
    deaths = 0;
    nextId = 1;
    nextLineage = 10;
    agents = [];
    nutrients = [];
    events = [];
    seed = (0x9e3779b9 + epoch * 7919) >>> 0;
    for (let i = 0; i < 300; i++) addNutrient();
    founderNames.forEach((name, index) => {
      const angle = index / founderNames.length * TWO_PI - Math.PI / 2;
      const cx = W / 2 + Math.cos(angle) * 250;
      const cy = H / 2 + Math.sin(angle) * 205;
      const genome = makeGenome(index);
      for (let pair = 0; pair < 2; pair++) {
        createAgent({
          lineage: index + 1,
          lineageName: name,
          hue: founderHues[index],
          genome: { ...genome },
          x: wrap(cx + between(-24, 24), W),
          y: wrap(cy + between(-24, 24), H),
          energy: between(92, 106),
        });
      }
    });
    selectedId = agents[0].id;
    shock = null;
    ctx.fillStyle = '#020807';
    ctx.fillRect(0, 0, W, H);
    note(`Epoch ${String(epoch).padStart(2, '0')} seeded with nine founding lineages`, 'system');
    updateUI();
  }

  function nearestNutrient(agent) {
    let best = null;
    let bestScore = Infinity;
    for (const nutrient of nutrients) {
      if (nutrient.value < .06) continue;
      const d = distance(agent, nutrient);
      if (d.length < agent.genome.sense) {
        const score = d.length / (.2 + nutrient.value);
        if (score < bestScore) { best = { ...d, nutrient }; bestScore = score; }
      }
    }
    return best;
  }

  function reproduce(parent) {
    const genome = mutateGenome(parent.genome);
    const divergence = geneDistance(parent.genome, genome);
    let lineage = parent.lineage;
    let lineageName = parent.lineageName;
    let hue = parent.hue + between(-4, 4);
    if (divergence > .31 && random() < .32) {
      lineage = nextLineage++;
      lineageName = `${parent.lineageName.split(' ')[0]}-${String.fromCharCode(64 + ((lineage - 9) % 26 || 26))}`;
      hue = (parent.hue + between(16, 34)) % 360;
      note(`Branch L${lineage} split from ${parent.lineageName}`, 'mutation');
    }
    parent.energy *= .53;
    const child = createAgent({
      lineage, lineageName, hue, generation: parent.generation + 1,
      parentId: parent.id, genome,
      x: wrap(parent.x + between(-22, 22), W),
      y: wrap(parent.y + between(-22, 22), H),
      energy: parent.energy * .88,
    });
    births++;
    if (births < 5 || births % 5 === 0) note(`${lineageName} gave rise to #${child.id} · generation ${child.generation}`, 'birth');
  }

  function updateAgent(agent, dt) {
    agent.age += dt;
    agent.phase += dt * (1.1 + agent.genome.turn);
    const target = nearestNutrient(agent);
    let ax = Math.cos(agent.phase * .73 + agent.id) * 5;
    let ay = Math.sin(agent.phase * .61 + agent.id) * 5;
    if (target) {
      const pull = 28 + agent.genome.sense * .08;
      ax += target.dx / Math.max(target.length, 1) * pull;
      ay += target.dy / Math.max(target.length, 1) * pull;
    }

    for (const other of agents) {
      if (other === agent) continue;
      const d = distance(agent, other);
      if (d.length > 0 && d.length < 48) {
        const kin = other.lineage === agent.lineage;
        const social = kin ? agent.genome.social : -(.4 + agent.genome.aggression);
        ax += d.dx / d.length * social * 13;
        ay += d.dy / d.length * social * 13;
        if (!kin && d.length < 14 * agent.genome.size) {
          const contest = dt * agent.genome.aggression * .85;
          other.energy -= contest;
          agent.energy += contest * .42;
        }
      }
    }

    const maxSpeed = 24 + agent.genome.speed * 54;
    agent.vx = (agent.vx + ax * dt) * Math.pow(.88, dt);
    agent.vy = (agent.vy + ay * dt) * Math.pow(.88, dt);
    const velocity = Math.hypot(agent.vx, agent.vy);
    if (velocity > maxSpeed) {
      agent.vx = agent.vx / velocity * maxSpeed;
      agent.vy = agent.vy / velocity * maxSpeed;
    }
    agent.x = wrap(agent.x + agent.vx * dt, W);
    agent.y = wrap(agent.y + agent.vy * dt, H);

    const bodyRadius = 7 + agent.genome.size * 8;
    for (const nutrient of nutrients) {
      if (nutrient.value <= .01) continue;
      const d = distance(agent, nutrient);
      if (d.length < bodyRadius + 5) {
        const eaten = Math.min(nutrient.value, dt * (1.1 + agent.genome.efficiency));
        nutrient.value -= eaten;
        agent.energy += eaten * (7.5 + agent.genome.efficiency * 7);
      }
    }
    agent.energy -= dt * (.44 + agent.genome.speed * .34 + agent.genome.size * .18);
    agent.energy = Math.min(agent.energy, 158);
    agent.history.unshift({ x: agent.x, y: agent.y });
    agent.history = agent.history.slice(0, 18);

    if (agent.energy > 112 && agent.age > 4 && agents.length < 78
      && random() < dt * (.055 + agent.genome.fertility * .075)) reproduce(agent);
  }

  function update(dt) {
    elapsed += dt;
    for (const nutrient of nutrients) nutrient.value = Math.min(1, nutrient.value + dt * .018);
    if (nutrients.length < 330 && random() < dt * 1.4) addNutrient();
    for (const agent of [...agents]) updateAgent(agent, dt);
    const gone = agents.filter(agent => agent.energy <= 0 || agent.age > agent.genome.longevity);
    for (const agent of gone) {
      deaths++;
      if (agent.id === selectedId) selectedId = null;
      if (deaths < 4 || deaths % 5 === 0) note(`#${agent.id} of ${agent.lineageName} ended after ${agent.age.toFixed(0)} world-seconds`, 'death');
    }
    agents = agents.filter(agent => agent.energy > 0 && agent.age <= agent.genome.longevity);
    if (!selectedId && agents.length) selectedId = agents[0].id;
    if (agents.length < 5 && random() < dt * .12) {
      const index = Math.floor(random() * founderNames.length);
      const revived = createAgent({
        lineage: index + 1, lineageName: founderNames[index], hue: founderHues[index],
        genome: makeGenome(index), x: between(0, W), y: between(0, H), energy: 96,
      });
      note(`${revived.lineageName} returned from the genomic archive`, 'system');
    }
    if (shock) {
      shock.life -= dt;
      if (shock.life <= 0) shock = null;
    }
  }

  function hsl(hue, saturation, light, alpha = 1) {
    return `hsla(${hue} ${saturation}% ${light}% / ${alpha})`;
  }

  function drawWrappedLine(a, b, color, width) {
    const d = distance(a, b);
    if (Math.abs(d.dx) > W * .35 || Math.abs(d.dy) > H * .35) return;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(a.x + d.dx, a.y + d.dy);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke();
  }

  function drawAgent(agent, selected) {
    const radius = (7 + agent.genome.size * 8) * (1 + Math.sin(agent.phase * 2) * .05);
    for (let i = agent.history.length - 1; i > 1; i--) {
      const a = agent.history[i];
      const b = agent.history[i - 1];
      if (Math.abs(a.x - b.x) < W / 2 && Math.abs(a.y - b.y) < H / 2) {
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
        ctx.strokeStyle = hsl(agent.hue, 90, 64, (agent.history.length - i) / 160);
        ctx.lineWidth = 1 + (agent.history.length - i) * .08;
        ctx.stroke();
      }
    }

    const angle = Math.atan2(agent.vy, agent.vx);
    ctx.save();
    ctx.translate(agent.x, agent.y);
    ctx.rotate(angle);
    ctx.shadowColor = hsl(agent.hue, 95, 62, .9);
    ctx.shadowBlur = selected ? 28 : 14;
    ctx.beginPath();
    const points = 28;
    for (let i = 0; i <= points; i++) {
      const theta = i / points * TWO_PI;
      const ripple = 1 + Math.sin(theta * agent.genome.lobes + agent.phase) * .22
        + Math.cos(theta * 2 - agent.phase * .6) * .08;
      const x = Math.cos(theta) * radius * ripple * 1.22;
      const y = Math.sin(theta) * radius * ripple * .82;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.closePath();
    const gradient = ctx.createRadialGradient(-radius * .25, -radius * .2, 1, 0, 0, radius * 1.35);
    gradient.addColorStop(0, hsl(agent.hue + 18, 95, 86, .96));
    gradient.addColorStop(.42, hsl(agent.hue, 86, 62, .78));
    gradient.addColorStop(1, hsl(agent.hue - 18, 82, 26, .38));
    ctx.fillStyle = gradient;
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = hsl(agent.hue, 100, selected ? 88 : 69, selected ? .95 : .45);
    ctx.lineWidth = selected ? 1.8 : .7;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(radius * .24, 0, Math.max(1.7, radius * .12), 0, TWO_PI);
    ctx.fillStyle = '#efffe7';
    ctx.fill();
    if (selected) {
      ctx.beginPath(); ctx.arc(0, 0, radius * 1.75, 0, TWO_PI);
      ctx.strokeStyle = hsl(agent.hue, 100, 72, .45 + Math.sin(agent.phase) * .15);
      ctx.setLineDash([3, 7]); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#e9f7ee';
      ctx.font = '10px DM Mono, monospace';
      ctx.fillText(`#${agent.id} · ${agent.lineageName}`, radius * 2, 4);
    }
    ctx.restore();
  }

  function render() {
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = 'rgba(2, 8, 7, .20)';
    ctx.fillRect(0, 0, W, H);

    ctx.globalCompositeOperation = 'lighter';
    for (const nutrient of nutrients) {
      if (nutrient.value < .025) continue;
      const glow = 1.2 + nutrient.value * 2.5 + Math.sin(elapsed * 1.7 + nutrient.phase) * .45;
      ctx.beginPath(); ctx.arc(nutrient.x, nutrient.y, Math.max(.5, glow), 0, TWO_PI);
      ctx.fillStyle = `rgba(170, 255, 116, ${.06 + nutrient.value * .22})`;
      ctx.shadowColor = '#9eff70'; ctx.shadowBlur = 5 + nutrient.value * 8; ctx.fill();
    }
    ctx.shadowBlur = 0;

    for (let i = 0; i < agents.length; i++) {
      for (let j = i + 1; j < agents.length; j++) {
        const a = agents[i], b = agents[j];
        const d = distance(a, b);
        if (d.length < 92 && a.lineage === b.lineage) {
          drawWrappedLine(a, b, hsl(a.hue, 80, 62, (1 - d.length / 92) * .11), .7);
        } else if (d.length < 34 && a.lineage !== b.lineage) {
          drawWrappedLine(a, b, 'rgba(255,120,86,.13)', .7);
        }
      }
    }
    const subject = agents.find(agent => agent.id === selectedId);
    for (const agent of agents) drawAgent(agent, agent === subject);

    if (shock) {
      ctx.globalCompositeOperation = 'source-over';
      ctx.beginPath(); ctx.arc(shock.x, shock.y, shock.radius * (1 + (1.4 - shock.life) * .16), 0, TWO_PI);
      ctx.strokeStyle = `rgba(255, 112, 83, ${Math.max(0, shock.life / 1.4)})`;
      ctx.lineWidth = 2; ctx.stroke();
    }
    ctx.globalCompositeOperation = 'source-over';
    ctx.shadowBlur = 0;
  }

  function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  }

  function renderEvents() {
    const log = document.querySelector('#ecoEventLog');
    if (!log) return;
    log.replaceChildren(...events.map(event => {
      const li = document.createElement('li');
      li.className = event.tone;
      const time = document.createElement('time');
      time.textContent = formatTime(event.time);
      const span = document.createElement('span');
      span.textContent = event.text;
      li.append(time, span);
      return li;
    }));
  }

  function updateUI() {
    const lineageCounts = new Map();
    for (const agent of agents) {
      const item = lineageCounts.get(agent.lineage) || { count: 0, name: agent.lineageName, hue: agent.hue };
      item.count++;
      lineageCounts.set(agent.lineage, item);
    }
    const available = nutrients.reduce((sum, item) => sum + item.value, 0) / Math.max(1, nutrients.length);
    document.querySelector('#ecoPopulation').textContent = String(agents.length).padStart(2, '0');
    document.querySelector('#ecoBirths').textContent = String(births).padStart(2, '0');
    document.querySelector('#ecoDeaths').textContent = String(deaths).padStart(2, '0');
    document.querySelector('#ecoLineages').textContent = String(lineageCounts.size).padStart(2, '0');
    document.querySelector('#ecoEnergy').textContent = `${Math.round(available * 100)}%`;
    document.querySelector('#ecoClock').textContent = `EPOCH ${String(epoch).padStart(2, '0')} · ${formatTime(elapsed)}`;

    const subject = agents.find(agent => agent.id === selectedId);
    if (subject) {
      const mark = document.querySelector('#ecoSubjectMark');
      mark.textContent = `L${subject.lineage}`;
      mark.style.setProperty('--subject', hsl(subject.hue, 90, 64));
      document.querySelector('.eco-subject').style.setProperty('--subject', hsl(subject.hue, 90, 64));
      document.querySelector('#ecoSubjectName').textContent = `${subject.lineageName} / #${subject.id}`;
      document.querySelector('#ecoSubjectMeta').textContent = `Generation ${subject.generation} · ${subject.parentId ? `child of #${subject.parentId}` : 'founder'}`;
      document.querySelector('#traitSpeed').style.width = `${subject.genome.speed * 100}%`;
      document.querySelector('#traitEfficiency').style.width = `${subject.genome.efficiency * 100}%`;
      document.querySelector('#traitFertility').style.width = `${subject.genome.fertility * 100}%`;
    }

    const ribbon = document.querySelector('#lineageRibbon');
    ribbon.replaceChildren(...[...lineageCounts.entries()].sort((a, b) => b[1].count - a[1].count).map(([id, item]) => {
      const element = document.createElement('button');
      element.type = 'button';
      element.innerHTML = `<i></i><span>L${id} · ${item.name}</span><b>${item.count}</b>`;
      element.style.setProperty('--lineage', hsl(item.hue, 88, 62));
      element.onclick = () => {
        const target = agents.find(agent => agent.lineage === id);
        if (target) { selectedId = target.id; updateUI(); }
      };
      return element;
    }));
  }

  function bloom() {
    const x = between(W * .18, W * .82);
    const y = between(H * .18, H * .82);
    for (let i = 0; i < 82; i++) {
      const angle = between(0, TWO_PI);
      const radius = Math.sqrt(random()) * 105;
      addNutrient(wrap(x + Math.cos(angle) * radius, W), wrap(y + Math.sin(angle) * radius, H), between(.72, 1));
    }
    if (nutrients.length > 520) nutrients = nutrients.sort((a, b) => b.value - a.value).slice(0, 520);
    note(`Nutrient bloom opened at ${Math.round(x)}, ${Math.round(y)}`, 'bloom');
    updateUI();
  }

  function climateShock() {
    const x = between(W * .2, W * .8);
    const y = between(H * .2, H * .8);
    const radius = between(125, 175);
    shock = { x, y, radius, life: 1.4 };
    for (const nutrient of nutrients) {
      if (distance({ x, y }, nutrient).length < radius) nutrient.value *= .08;
    }
    let exposed = 0;
    for (const agent of agents) {
      if (distance({ x, y }, agent).length < radius) { agent.energy -= 27; exposed++; }
    }
    note(`Climate shock crossed ${exposed} organisms; nearby energy collapsed`, 'death');
    updateUI();
  }

  canvas.addEventListener('pointerdown', event => {
    const box = canvas.getBoundingClientRect();
    const point = { x: (event.clientX - box.left) / box.width * W, y: (event.clientY - box.top) / box.height * H };
    let nearest = null;
    let best = 46;
    for (const agent of agents) {
      const d = distance(point, agent).length;
      if (d < best) { nearest = agent; best = d; }
    }
    if (nearest) { selectedId = nearest.id; updateUI(); }
  });

  document.querySelector('#ecoPause').addEventListener('click', event => {
    paused = !paused;
    event.currentTarget.textContent = paused ? 'Resume time' : 'Pause time';
    note(paused ? 'Ecosystem time suspended' : 'Ecosystem time resumed', 'system');
    syncRuntime();
  });
  document.querySelector('#ecoBloom').addEventListener('click', () => { bloom(); render(); });
  document.querySelector('#ecoShock').addEventListener('click', () => { climateShock(); render(); });
  document.querySelector('#ecoReset').addEventListener('click', resetWorld);

  function ensureInitialized() {
    if (initialized) return;
    initialized = true;
    resetWorld();
  }

  function canRun() {
    return initialized && (!disclosure || disclosure.open) && ecosystemVisible && !document.hidden && !paused;
  }

  function syncRuntime() {
    if (!disclosure || disclosure.open) ensureInitialized();
    if (canRun() && frameId === null) {
      last = performance.now();
      frameId = requestAnimationFrame(frame);
    } else if (!canRun() && frameId !== null) {
      cancelAnimationFrame(frameId);
      frameId = null;
    }
  }

  function frame(now) {
    frameId = null;
    if (!canRun()) return;
    const dt = Math.min(.04, (now - last) / 1000 || .016);
    last = now;
    update(dt);
    render();
    uiAccumulator += dt;
    if (uiAccumulator > .25) { updateUI(); uiAccumulator = 0; }
    frameId = requestAnimationFrame(frame);
  }

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      ecosystemVisible = entries[0].isIntersecting;
      syncRuntime();
    }, { rootMargin: '160px' });
    observer.observe(canvas);
  }
  disclosure?.addEventListener('toggle', syncRuntime);
  document.addEventListener('visibilitychange', syncRuntime);

  window.genesisEcosystem = {
    getState: () => ({ epoch, elapsed, births, deaths, population: agents.length, lineages: new Set(agents.map(a => a.lineage)).size, paused, initialized, running: frameId !== null }),
    bloom: () => { ensureInitialized(); bloom(); render(); },
    climateShock: () => { ensureInitialized(); climateShock(); render(); },
    reset: () => { if (initialized) resetWorld(); else ensureInitialized(); },
    advance: seconds => { ensureInitialized(); for (let time = 0; time < seconds; time += .025) update(.025); updateUI(); render(); },
  };

  syncRuntime();
})();
