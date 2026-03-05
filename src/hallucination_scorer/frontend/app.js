async function scoreClaim() {
  const claim = document.getElementById("claim").value.trim();
  const contextRaw = document.getElementById("context").value.trim();
  const btn = document.getElementById("score-btn");
  const results = document.getElementById("results");

  if (!claim) {
    alert("Please enter a claim.");
    return;
  }

  const contextChunks = contextRaw
    ? contextRaw.split("\n").map((line) => line.trim()).filter(Boolean)
    : [];

  btn.disabled = true;
  btn.textContent = "Scoring...";
  results.classList.add("hidden");
  results.innerHTML = "";

  try {
    const response = await fetch("/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ claim, context_chunks: contextChunks }),
    });

    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const data = await response.json();
    renderResults(data);
  } catch (err) {
    console.error(err);
    results.classList.remove("hidden");
    results.innerHTML = `<p>Request failed: ${err.message}</p>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Score claim";
  }
}

function renderResults(data) {
  const results = document.getElementById("results");
  results.classList.remove("hidden");

  const overall = data.overall_grounding_score;
  const unsupportedCount = data.unsupported_sentence_count;

  const header = document.createElement("div");
  const percent = (overall * 100).toFixed(1);
  header.innerHTML = `
    <h2>Grounding score: ${percent}%</h2>
    <p>${unsupportedCount} unsupported sentences</p>
    <div class="score-bar">
      <div class="score-bar-fill" style="width: ${percent}%;"></div>
    </div>
  `;
  results.appendChild(header);

  data.per_sentence.forEach((s) => {
    const div = document.createElement("div");
    div.className = "sentence " + (s.unsupported ? "unsupported" : "supported");

    const confidence = s.best_chunk_score != null
      ? (s.best_chunk_score * 100).toFixed(1) + "%"
      : "n/a";

    const headerHtml = `
      <div class="sentence-header">
        <span>Sentence ${s.sentence_index + 1}</span>
        <span>${s.unsupported ? "Unsupported" : "Supported"} • score ${confidence}</span>
      </div>
    `;

    const sentenceHtml = `<div class="sentence-text">${s.sentence}</div>`;

    let evidenceHtml = "";
    if (s.evidence_span_text && data.retrieved_chunks) {
      // Try to find the parent chunk text for nicer display (optional).
      evidenceHtml = `<div class="evidence">Evidence: <span class="highlight">${s.evidence_span_text}</span></div>`;
    } else if (s.evidence_span_text) {
      evidenceHtml = `<div class="evidence">Evidence: <span class="highlight">${s.evidence_span_text}</span></div>`;
    }

    div.innerHTML = headerHtml + sentenceHtml + evidenceHtml;
    results.appendChild(div);
  });
}

document.getElementById("score-btn").addEventListener("click", scoreClaim);

