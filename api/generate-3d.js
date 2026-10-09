export default async function handler(req, res) {
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

  const modalUrl = process.env.MODAL_API_URL;
  if (!modalUrl) {
    console.error('Missing MODAL_API_URL environment variable');
    return res.status(500).json({ success: false, error: 'Missing MODAL_API_URL environment variable on Vercel' });
  }

  try {
    const body = req.body || {};
    const image = body.image || (body.input && body.input.image);

    if (!image) {
      return res.status(400).json({ success: false, error: 'No image provided in request body' });
    }

    console.log('Forwarding generation request to Modal GPU backend...');

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
      console.error('Modal execution error:', data);
      return res.status(response.status).json({ success: false, error: data.error || 'Modal rejected the request' });
    }

    console.log('3D Model generated successfully via Modal.');
    return res.status(200).json({
      success: true,
      model: data.model || data
    });

  } catch (error) {
    console.error('Generate 3D Handler Exception:', error);
    return res.status(500).json({ success: false, error: error.message });
  }
}
