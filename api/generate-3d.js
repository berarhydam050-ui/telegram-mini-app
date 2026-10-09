export default async function handler(req, res) {
  // CORS Headers for Telegram Mini App
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,POST');
  res.setHeader('Access-Control-Allow-Headers', 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ success: false, error: 'Method Not Allowed' });
  }

  // Load Modal API URL from Vercel environment variables
  const modalUrl = process.env.MODAL_API_URL;
  if (!modalUrl) {
    return res.status(500).json({ success: false, error: 'Missing MODAL_API_URL environment variable on Vercel' });
  }

  try {
    const body = req.body || {};
    const image = body.image || (body.input && body.input.image);

    if (!image) {
      return res.status(400).json({ success: false, error: 'No image provided in request body' });
    }

    // Call Modal Backend API
    const response = await fetch(modalUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        image: image,
        texture_resolution: body.texture_resolution || 1024,
        remesh: body.remesh_option || 'triangle'
      })
    });

    const data = await response.json();

    if (!response.ok) {
      return res.status(response.status).json({ success: false, error: data.error || 'Modal rejected the request' });
    }

    // Return the successful result to your Telegram Mini App frontend
    return res.status(200).json({
      success: true,
      model: data.model || data
    });

  } catch (error) {
    console.error('Generate 3D Handler Error:', error);
    return res.status(500).json({ success: false, error: error.message });
  }
}
