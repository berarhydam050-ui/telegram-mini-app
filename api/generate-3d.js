export const config = {
  maxDuration: 300, // allow long GPU runs
};

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,POST');
  res.setHeader(
    'Access-Control-Allow-Headers',
    'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version, X-API-Key'
  );

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ success: false, error: 'Method Not Allowed' });
  }

  const modalUrl =
    process.env.MODAL_API_URL ||
    'https://berarhydam050--sf3d-backend-generate.modal.run';

  try {
    const body = req.body || {};
    const image = body.image || (body.input && body.input.image);

    if (!image) {
      return res.status(400).json({ success: false, error: 'No image provided in request body' });
    }

    console.log('Forwarding generation request to Modal GPU backend...');

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 280000);

    const headers = { 'Content-Type': 'application/json' };
    // Pass API key if configured (same value as Modal secret MODAL_API_KEY)
    if (process.env.MODAL_API_KEY) {
      headers['X-API-Key'] = process.env.MODAL_API_KEY;
    }

    const response = await fetch(modalUrl, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        image: image,
        texture_resolution: body.texture_resolution || 2048,
        remesh: body.remesh || body.remesh_option || 'triangle',
        return_base64: false   // prefer lightweight response
      }),
      signal: controller.signal
    });

    clearTimeout(timeout);

    const data = await response.json();

    if (!response.ok) {
      console.error('Modal execution error:', data);
      return res.status(response.status).json({
        success: false,
        error: data.detail || data.error || 'Modal rejected the request'
      });
    }

    console.log('3D Model generated successfully via Modal.');

    return res.status(200).json({
      success: true,
      ...data,
      model: data.model || null,
      asset_id: data.asset_id || null
    });

  } catch (error) {
    console.error('Generate 3D Handler Exception:', error);
    const msg =
      error.name === 'AbortError'
        ? 'Generation timed out. Please try again.'
        : (error.message || 'Proxy error');
    return res.status(500).json({ success: false, error: msg });
  }
}
