"use strict";

module.exports = async function handler(req, res) {
  if (req.method !== "GET") {
    res.setHeader("Allow", "GET");
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  const endpoint = process.env.MODAL_RESULT_URL;
  const proxyKey = process.env.MODAL_PROXY_KEY;
  const proxySecret = process.env.MODAL_PROXY_SECRET;
  const jobId = String(req.query.job_id || "");
  if (!endpoint || !proxyKey || !proxySecret) {
    res.status(500).json({ error: "Server is missing configuration" });
    return;
  }
  if (!/^[0-9a-f-]{36}$/i.test(jobId)) {
    res.status(400).json({ error: "Bad job id" });
    return;
  }

  const upstream = await fetch(endpoint + "?job_id=" + encodeURIComponent(jobId), {
    headers: {
      "Modal-Key": proxyKey,
      "Modal-Secret": proxySecret,
    },
    signal: AbortSignal.timeout(10000),
  });

  if (upstream.status === 204) {
    res.status(204).end();
    return;
  }
  if (!upstream.ok) {
    res.status(502).json({ error: "Generation failed" });
    return;
  }

  const bytes = Buffer.from(await upstream.arrayBuffer());
  res.setHeader("Content-Type", "image/png");
  res.setHeader("Cache-Control", "no-store");
  res.setHeader("X-Content-Type-Options", "nosniff");
  res.status(200).send(bytes);
};
