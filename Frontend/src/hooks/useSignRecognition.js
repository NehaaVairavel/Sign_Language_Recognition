import { useState, useCallback, useRef, useEffect } from 'react';
import { Hands } from '@mediapipe/hands';
import { Camera } from '@mediapipe/camera_utils';
import { predictSign } from '@/lib/api';
import { TAMIL_LETTERS } from '@/lib/tamilLetters';

export const useSignRecognition = (videoRef) => {
    const [isDetecting, setIsDetecting] = useState(false);
    const [detectedLetter, setDetectedLetter] = useState(null);
    const [confidence, setConfidence] = useState(0);
    const [isHandPresent, setIsHandPresent] = useState(false);

    const handsRef = useRef(null);
    const cameraRef = useRef(null);
    const lastPredictionTime = useRef(0);
    const predictionBuffer = useRef([]);
    const noHandCount = useRef(0);
    const BUFFER_SIZE = 5;
    const RESET_THRESHOLD = 3;



    // Initialize MediaPipe Hands
    useEffect(() => {
        const hands = new Hands({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
            }
        });

        hands.setOptions({
            maxNumHands: 2,
            modelComplexity: 1,
            minDetectionConfidence: 0.7,
            minTrackingConfidence: 0.5
        });

        hands.onResults(onResults);
        handsRef.current = hands;

        return () => {
            if (handsRef.current) {
                handsRef.current.close();
            }
        };
    }, []);

    const onResults = useCallback(async (results) => {
        const now = Date.now();
        // Limit prediction rate to avoid overloading backend (e.g., every 200ms)
        if (now - lastPredictionTime.current < 200) return;

        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            setIsHandPresent(true);

            // Prepare landmarks for backend (Flattening logic matches extract_landmarks_to_csv.py)
            // Backend expects 126 features. (2 hands * 21 points * 3 coords)
            // If only 1 hand, we might need to pad or duplicate? 
            // The training script strictly expected 2 hands. 
            // We will try to send what we have, but formatted correctly.

            // Logic: 
            // 1. Get up to 2 hands.
            // 2. Normalize each hand to its wrist.
            // 3. Flatten.

            let allFeatures = [];

            // Limit to 2 hands to match model expectation
            const handsToProcess = results.multiHandLandmarks.slice(0, 2);

            handsToProcess.forEach(landmarks => {
                const wrist = landmarks[0];
                const normalizedLandmarks = landmarks.map(lm => ({
                    x: lm.x - wrist.x,
                    y: lm.y - wrist.y,
                    z: lm.z - wrist.z
                }));

                // Flatten x, y, z
                normalizedLandmarks.forEach(lm => {
                    allFeatures.push(lm.x, lm.y, lm.z);
                });
            });

            // If we have less than 126 features (i.e., only 1 hand detected), 
            // the model might fail or give bad results. 
            // For now, let's pad with zeros to ensure 126 length.
            while (allFeatures.length < 126) {
                allFeatures.push(0);
            }

            // Truncate if somehow > 126
            if (allFeatures.length > 126) {
                allFeatures = allFeatures.slice(0, 126);
            }

            try {
                lastPredictionTime.current = now;
                const response = await predictSign(allFeatures);

                if (response.valid && response.class_id) {
                    // Add to buffer
                    predictionBuffer.current.push(response.class_id);
                    if (predictionBuffer.current.length > BUFFER_SIZE) {
                        predictionBuffer.current.shift();
                    }

                    // Get most frequent prediction in buffer
                    const counts = {};
                    predictionBuffer.current.forEach(id => {
                        counts[id] = (counts[id] || 0) + 1;
                    });

                    const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
                    const mostFrequentId = parseInt(sorted[0][0]);
                    const frequency = sorted[0][1];

                    // Majority vote: require > 50% agreement (e.g., 2 out of 3)
                    if (frequency >= Math.ceil(BUFFER_SIZE * 0.5)) {
                        const letter = TAMIL_LETTERS.find(l => l.id === mostFrequentId);
                        if (letter) {
                            setDetectedLetter(letter);
                            setConfidence(response.confidence * 100);
                        }
                    }
                } else {
                    // Optional: Handle low confidence? 
                    // For now, keep last stable prediction or do nothing
                }
            } catch (err) {
                console.error("Prediction error:", err);
            }

        } else {
            setIsHandPresent(false);
            noHandCount.current += 1;

            // Reset if no hand for multiple consecutive checks
            if (noHandCount.current >= RESET_THRESHOLD) {
                predictionBuffer.current = [];
                setDetectedLetter(null);
                setConfidence(0);
            }
        }
    }, []);

    const startDetection = useCallback(async () => {
        if (!videoRef?.current || !handsRef.current) return;
        setIsDetecting(true);

        // We can use Camera utils or manual requestAnimationFrame
        // Using Camera utils is easier for MediaPipe
        if (!cameraRef.current) {
            cameraRef.current = new Camera(videoRef.current, {
                onFrame: async () => {
                    if (videoRef.current && handsRef.current) {
                        await handsRef.current.send({ image: videoRef.current });
                    }
                },
                width: 640,
                height: 480
            });
            await cameraRef.current.start();
        }
    }, [videoRef]);

    const stopDetection = useCallback(() => {
        setIsDetecting(false);
        setDetectedLetter(null);
        setConfidence(0);
        setIsHandPresent(false);

        if (cameraRef.current) {
            cameraRef.current.stop();
            cameraRef.current = null;
        }
    }, []);

    // Manual simulation for demo purposes (Keep it for fallback/testing)
    const simulateDetection = useCallback((letterId) => {
        const letter = TAMIL_LETTERS.find(l => l.id === letterId);
        if (letter) {
            setIsHandPresent(true);
            setDetectedLetter(letter);
            setConfidence(95);
        }
    }, []);

    return {
        isDetecting,
        detectedLetter,
        confidence,
        startDetection,
        stopDetection,
        simulateDetection,
        isHandPresent
    };
};
