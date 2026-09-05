(() => {
  const rows = document.querySelector('#comparisonRows');
  const status = document.querySelector('#comparisonStatus');
  if (!rows || !status) return;
  const names = {
    O2u: 'Orbium', OG2g: 'Gyrorbium', O4i: 'Synorbium', S1s: 'Scutium',
    P4cl: 'Paraptera', H3s: 'Helicium solidus', H3cp: 'Helicium cavus pedes', C0v: 'Circium',
  };
  const methods = ['performance', 'map-elites', 'direct-regeneration'];
  const percent = value => `${(value * 100).toFixed(1)}%`;
  fetch('data/charter-analysis.json').then(response => {
    if (!response.ok) throw new Error('Analysis unavailable');
    return response.json();
  }).then(record => {
    if (record.schema !== 'genesis.charter-analysis/v1') throw new Error('Unsupported analysis');
    Object.entries(names).forEach(([code, name]) => {
      const tr = document.createElement('tr');
      const label = document.createElement('th');
      label.scope = 'row';
      label.textContent = name;
      tr.append(label);
      methods.forEach(method => {
        const item = record.summary.find(row => row.species === code && row.method === method);
        const td = document.createElement('td');
        if (item) {
          td.textContent = percent(item.functional_recovery_rate);
          const validity = document.createElement('small');
          validity.textContent = `control valid ${percent(item.control_validity)}`;
          td.append(validity);
        } else td.textContent = 'Excluded';
        tr.append(td);
      });
      rows.append(tr);
    });
    status.textContent = 'Success requires a valid undamaged control and retained function after intervention. Each percentage averages ten interventions within each search seed.';
  }).catch(() => {
    status.textContent = 'The analysis could not be loaded. Open the evidence links below to inspect the saved records.';
  });
})();
