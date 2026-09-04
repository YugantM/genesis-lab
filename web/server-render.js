(() => {
  const image = document.querySelector('#serverZoo');
  const canvas = document.querySelector('#zoo');
  const shell = document.querySelector('.zoo-shell');
  const mode = document.querySelector('#renderMode');
  const rate = document.querySelector('#renderRate');
  if (!image || !canvas) return;

  function notice(message) {
    const node = document.querySelector('#eventMessage');
    node.textContent = message;
    node.classList.add('visible');
    window.setTimeout(() => node.classList.remove('visible'), 2200);
  }

  async function status() {
    const response = await fetch('api/status', { cache: 'no-store' });
    if (!response.ok) throw new Error('local render server unavailable');
    return response.json();
  }

  function update(payload) {
    document.querySelector('#generation').textContent = `GEN ${String(payload.generation).padStart(6, '0')}`;
    mode.textContent = `MLX server · ${payload.device}`;
    rate.textContent = `${payload.render_fps.toFixed(1)} fps · ${payload.simulation_steps_per_second.toFixed(0)} steps/s`;
    document.querySelector('#pauseAll').textContent = payload.paused ? 'Resume' : 'Pause';
    payload.masses.forEach((mass, index) => {
      const node = document.querySelector(`[data-mass="${index}"]`);
      if (node) node.textContent = mass < 0.5 ? 'extinct' : `mass ${mass.toFixed(0)}`;
      document.querySelector(`[data-index="${index}"]`)?.classList.toggle('extinct', mass < 0.5);
    });
  }

  async function control(action, index) {
    const response = await fetch('api/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action, index }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'control failed');
    update(payload);
    return payload;
  }

  status().then(payload => {
    window.genesisZooClient?.enterServerMode();
    canvas.hidden = true;
    image.hidden = false;
    image.src = `stream/zoo.mjpeg?started=${Date.now()}`;
    shell.classList.add('server-rendering');
    update(payload);
    notice('MLX server renderer connected');

    document.querySelector('#pauseAll').onclick = () => control('toggle_pause').catch(error => notice(error.message));
    document.querySelector('#resetAll').onclick = () => control('reset').then(() => notice('Population restored on MLX')).catch(error => notice(error.message));
    document.querySelector('#stressAll').onclick = () => control('stress_all').then(() => notice('Server applied exact 5% injury')).catch(error => notice(error.message));
    document.querySelector('#stressSelected').onclick = () => control('stress_one', window.genesisZooClient?.getSelected() || 0).then(() => notice('Selected creature injured on MLX')).catch(error => notice(error.message));

    window.setInterval(() => status().then(update).catch(() => {
      mode.textContent = 'MLX server disconnected';
      rate.textContent = '';
    }), 500);
  }).catch(() => {
    mode.textContent = 'WebGL renderer';
    rate.textContent = 'browser fallback';
  });
})();
