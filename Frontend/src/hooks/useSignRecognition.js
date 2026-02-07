import { useState, useCallback, useRef, useEffect } from 'react';
import { TAMIL_LETTERS } from '@/lib/tamilLetters';

// Simulated sign recognition for demo purposes
// In production, this would integrate with TensorFlow.js or a similar ML library
export const useSignRecognition = () => {
    const [isDetecting, setIsDetecting] = useState(false);
    const [detectedLetter, setDetectedLetter] = useState(null);
    const [confidence, setConfidence] = useState(0);
    const [isHandPresent, setIsHandPresent] = useState(false);

    const detectionIntervalRef = useRef(null);
    const lastDetectedRef = useRef(null);

    const startDetection = useCallback(() => {
        setIsDetecting(true);

        // Simulate detection process
        // In real implementation, this would process video frames
        detectionIntervalRef.current = setInterval(() => {
            // Randomly simulate hand presence (70% chance of hand being present when detecting)
            const handPresent = Math.random() > 0.3;
            setIsHandPresent(handPresent);

            if (!handPresent) {
                setDetectedLetter(null);
                setConfidence(0);
                lastDetectedRef.current = null;
            }
        }, 500);
    }, []);

    const stopDetection = useCallback(() => {
        setIsDetecting(false);
        setDetectedLetter(null);
        setConfidence(0);
        setIsHandPresent(false);

        if (detectionIntervalRef.current) {
            clearInterval(detectionIntervalRef.current);
            detectionIntervalRef.current = null;
        }
    }, []);

    // Manual simulation for demo purposes
    const simulateDetection = useCallback((letterId) => {
        const letter = TAMIL_LETTERS.find(l => l.id === letterId);
        if (letter && lastDetectedRef.current !== letterId) {
            setIsHandPresent(true);
            setDetectedLetter(letter);
            setConfidence(Math.random() * 20 + 80); // 80-100% confidence
            lastDetectedRef.current = letterId;
        }
    }, []);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (detectionIntervalRef.current) {
                clearInterval(detectionIntervalRef.current);
            }
        };
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
