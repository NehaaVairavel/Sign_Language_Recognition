const API_BASE_URL = '/api'; // Use relative path to leverage Vite proxy

/**
 * Predict sign from landmarks
 * @param {Array<number>} landmarks - Flat array of 126 numbers (21 points * 3 coords * 2 hands? No, model says 126 inputs. 21*3=63. 63*2=126. So both hands or 2D+visibility?
 * Let's assume the model expects 63*2 = 126 (x,y,z for 42 points? or x,y,z for 21 points * 2 hands).
 * The python script `extract_landmarks_to_csv.py` would confirm this.
 * For now, we'll assume the frontend will produce the correct array.
 */
export const predictSign = async (landmarks) => {
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ landmarks }),
        });

        if (!response.ok) {
            throw new Error(`Prediction failed: ${response.statusText}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Error predicting sign:', error);
        throw error;
    }
};

/**
 * Verify sign (for Learn/Practice mode)
 * @param {Array<number>} landmarks 
 * @param {number|string} expectedClass - The ID of the expected letter
 */
export const verifySign = async (landmarks, expectedClass) => {
    try {
        const response = await fetch(`${API_BASE_URL}/verify`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                landmarks,
                expected_class: expectedClass
            }),
        });

        if (!response.ok) {
            throw new Error(`Verification failed: ${response.statusText}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Error verifying sign:', error);
        throw error;
    }
};

/**
 * Convert text to speech using backend TTS
 * @param {string} text - Text to speak
 * @returns {Promise<Blob>} - Audio blob
 */
export const speakText = async (text) => {
    try {
        const response = await fetch(`${API_BASE_URL}/speak`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text }),
        });

        if (!response.ok) {
            throw new Error(`TTS failed: ${response.statusText}`);
        }

        return await response.blob();
    } catch (error) {
        console.error('Error with TTS:', error);
        throw error;
    }
};
