import { useState, useRef, useCallback, useEffect } from 'react';

export const useWebcam = () => {
    const videoRef = useRef(null);
    const streamRef = useRef(null);
    const [isStreaming, setIsStreaming] = useState(false);
    const [error, setError] = useState(null);
    const [hasPermission, setHasPermission] = useState(null);

    const startCamera = useCallback(async () => {
        try {
            setError(null);

            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            });

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                streamRef.current = stream;
                setIsStreaming(true);
                setHasPermission(true);
            }
        } catch (err) {
            const errorMessage = err instanceof Error ? err.message : 'Failed to access camera';

            if (errorMessage.includes('Permission denied') || errorMessage.includes('NotAllowedError')) {
                setError('Camera permission denied. Please allow camera access to use this feature.');
                setHasPermission(false);
            } else if (errorMessage.includes('NotFoundError')) {
                setError('No camera found. Please connect a camera and try again.');
            } else {
                setError(`Camera error: ${errorMessage}`);
            }

            setIsStreaming(false);
        }
    }, []);

    const stopCamera = useCallback(() => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }

        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }

        setIsStreaming(false);
    }, []);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            stopCamera();
        };
    }, [stopCamera]);

    return {
        videoRef,
        isStreaming,
        error,
        startCamera,
        stopCamera,
        hasPermission
    };
};
