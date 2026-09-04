(() => {
  const matrix = document.querySelector('#couplingMatrix');
  if (!matrix) return;

  const codes = ['GEN001', 'O2u', 'OG2g', 'O4i', 'S1s', 'P4cl', 'H3s', 'H3cp', 'C0v'];
  const names = {
    GEN001: 'Genesis 001', O2u: 'Orbium unicaudatus', OG2g: 'Gyrorbium gyrans',
    O4i: 'Synorbium ignis', S1s: 'Scutium solidus', P4cl: 'Paraptera cavus labens',
    H3s: 'Helicium solidus', H3cp: 'Helicium cavus pedes', C0v: 'Circium ventilans',
  };
  let discovery;
  let validation;
  let evolution;
  let interaction;
  let strength = 1;
  let selected = ['GEN001', 'S1s'];

  function percent(value, digits = 0) {
    return `${(value * 100).toFixed(digits)}%`;
  }

  function findPair(left, right) {
    return discovery.pair_summary.find(item =>
      item.competition === strength
      && ((item.left_code === left && item.right_code === right)
        || (item.left_code === right && item.right_code === left))
    );
  }

  function updatePair(item) {
    if (!item) return;
    selected = [item.left_code, item.right_code];
    document.querySelector('#pairCodes').textContent = `${item.left_code} × ${item.right_code}`;
    document.querySelector('#pairNames').innerHTML = `${names[item.left_code]}<br>with ${names[item.right_code]}`;
    document.querySelector('#pairRate').textContent = percent(item.coexistence_rate);
    document.querySelector('#pairCoexist').textContent = item.coexistence_trials;
    document.querySelector('#pairDominance').textContent = item.dominance_trials;
    document.querySelector('#pairCollapse').textContent = item.mutual_collapse_trials;
    document.querySelector('#pairSimilarity').textContent = item.median_minimum_similarity.toFixed(2);
    document.querySelectorAll('.matrix-cell').forEach(cell => {
      const pair = cell.dataset.pair?.split('|') || [];
      cell.classList.toggle('selected', pair.includes(item.left_code) && pair.includes(item.right_code));
    });
  }

  function renderMatrix() {
    const nodes = [];
    const corner = document.createElement('span');
    corner.className = 'matrix-label corner';
    corner.textContent = 'λ';
    nodes.push(corner);
    for (const code of codes) {
      const label = document.createElement('span');
      label.className = 'matrix-label top';
      label.textContent = code;
      nodes.push(label);
    }
    for (const left of codes) {
      const label = document.createElement('span');
      label.className = 'matrix-label side';
      label.textContent = left;
      nodes.push(label);
      for (const right of codes) {
        if (left === right) {
          const diagonal = document.createElement('span');
          diagonal.className = 'matrix-diagonal';
          diagonal.textContent = '—';
          nodes.push(diagonal);
          continue;
        }
        const item = findPair(left, right);
        const cell = document.createElement('button');
        cell.type = 'button';
        cell.className = 'matrix-cell';
        cell.dataset.pair = `${left}|${right}`;
        cell.style.setProperty('--rate', item.coexistence_rate);
        cell.style.backgroundColor = `rgba(186,255,114,${.035 + item.coexistence_rate * .66})`;
        cell.textContent = Math.round(item.coexistence_rate * 100);
        cell.setAttribute('aria-label', `${names[left]} with ${names[right]}: ${percent(item.coexistence_rate)} coexistence`);
        cell.onclick = () => updatePair(item);
        nodes.push(cell);
      }
    }
    matrix.replaceChildren(...nodes);
    updatePair(findPair(...selected) || findPair('GEN001', 'S1s'));
  }

  function renderOverview() {
    const summary = discovery.summary.strengths.find(item => item.competition === strength);
    document.querySelector('#couplingRate').textContent = percent(summary.coexistence_rate);
    document.querySelector('#couplingOutcome').textContent = `${summary.coexistence_trials} of ${summary.valid_trials} matched encounters at ${strength === .25 ? 'low' : strength === .5 ? 'medium' : 'high'} competition.`;
    for (const geometry of ['near', 'contact', 'overlap']) {
      const item = discovery.summary.geometries.find(value => value.competition === strength && value.geometry === geometry);
      document.querySelector(`#${geometry}Rate`).textContent = percent(item.coexistence_rate);
      document.querySelector(`#${geometry}Bar`).style.width = percent(item.coexistence_rate);
    }
    document.querySelectorAll('[data-coupling]').forEach(button => {
      button.setAttribute('aria-pressed', String(Number(button.dataset.coupling) === strength));
    });
  }

  function renderValidation() {
    const result = validation.comparison;
    document.querySelector('#genesisEcoRate').textContent = percent(result.genesis_coexistence_rate, 1);
    document.querySelector('#parentEcoRate').textContent = percent(result.parent_coexistence_rate, 1);
    const delta = (result.difference * 100).toFixed(1);
    document.querySelector('#validationReason').textContent = `+${delta} points · below the locked +5 point gate · p=${result.two_sided_exact_sign_p.toFixed(4)}`;
  }

  function renderEvolution() {
    const summary = evolution.summary;
    const training = summary.selected_training;
    const held = summary.held_out_candidate;
    const baseline = summary.held_out_baseline;
    document.querySelector('#evoSearched').textContent = evolution.protocol.candidate_count.toLocaleString();
    document.querySelector('#evoViable').textContent = summary.solo_viable_candidates;
    document.querySelector('#evoTraining').textContent = `${training.coexistence_trials} / ${training.trials}`;
    document.querySelector('#evoParentRate').textContent = `${baseline.coexistence_trials} / ${baseline.trials}`;
    document.querySelector('#evoCandidateRate').textContent = `${held.coexistence_trials} / ${held.trials}`;
    document.querySelector('#evoCandidateBar').style.width = percent(held.coexistence_rate);
  }

  function renderInteraction() {
    const summary = interaction.summary;
    const training = summary.selected_training;
    const held = summary.held_out_candidate;
    const baseline = summary.held_out_baseline;
    document.querySelector('#interactionSearched').textContent = interaction.protocol.candidate_count.toLocaleString();
    document.querySelector('#interactionViable').textContent = summary.identity_safe_candidates.toLocaleString();
    document.querySelector('#interactionTraining').textContent = `${training.coexistence_trials} / ${training.trials}`;
    document.querySelector('#interactionCandidateLabel').textContent = `candidate ${training.candidate}`;
    document.querySelector('#interactionParentRate').textContent = `${baseline.coexistence_trials} / ${baseline.trials}`;
    document.querySelector('#interactionCandidateRate').textContent = `${held.coexistence_trials} / ${held.trials}`;
    document.querySelector('#interactionParentBar').style.width = percent(baseline.coexistence_rate);
    document.querySelector('#interactionCandidateBar').style.width = percent(held.coexistence_rate);
  }

  document.querySelectorAll('[data-coupling]').forEach(button => {
    button.onclick = () => {
      strength = Number(button.dataset.coupling);
      renderOverview();
      renderMatrix();
    };
  });

  Promise.all([
    fetch('data/coupled-ecology.json').then(response => {
      if (!response.ok) throw new Error('Coupling experiment unavailable');
      return response.json();
    }),
    fetch('data/coupled-ecology-validation.json').then(response => {
      if (!response.ok) throw new Error('Coupling validation unavailable');
      return response.json();
    }),
    fetch('data/coexistence-evolution.json').then(response => {
      if (!response.ok) throw new Error('Evolution run unavailable');
      return response.json();
    }),
    fetch('data/interaction-evolution.json').then(response => {
      if (!response.ok) throw new Error('Interaction evolution run unavailable');
      return response.json();
    }),
  ]).then(([discoveryRecord, validationRecord, evolutionRecord, interactionRecord]) => {
    discovery = discoveryRecord;
    validation = validationRecord;
    evolution = evolutionRecord;
    interaction = interactionRecord;
    renderOverview();
    renderValidation();
    renderEvolution();
    renderInteraction();
    renderMatrix();
    window.genesisCoupling = {
      getState: () => ({ strength, selected, trialCount: discovery.summary.trial_count, validated: validation.comparison.validated }),
    };
  }).catch(error => {
    document.querySelector('.coupling-section').classList.add('data-error');
    document.querySelector('#couplingOutcome').textContent = error.message;
  });
})();
