import { useEffect, useState } from 'react';
import { Clock } from 'lucide-react';
import { cn } from '@/lib/utils';

export const Timer = ({
    duration,
    isRunning,
    onTimeUp,
    className
}) => {
    const [timeLeft, setTimeLeft] = useState(duration);

    useEffect(() => {
        setTimeLeft(duration);
    }, [duration]);

    useEffect(() => {
        if (!isRunning) return;

        if (timeLeft <= 0) {
            onTimeUp?.();
            return;
        }

        const timer = setInterval(() => {
            setTimeLeft((prev) => prev - 1);
        }, 1000);

        return () => clearInterval(timer);
    }, [isRunning, timeLeft, onTimeUp]);

    const progress = (timeLeft / duration) * 100;
    const isLow = timeLeft <= 5;

    return (
        <div className={cn("flex items-center gap-3", className)}>
            <div className={cn(
                "w-10 h-10 rounded-xl flex items-center justify-center transition-colors",
                isLow ? "bg-destructive/10" : "bg-primary/10"
            )}>
                <Clock className={cn(
                    "w-5 h-5 transition-colors",
                    isLow ? "text-destructive animate-pulse" : "text-primary"
                )} />
            </div>

            <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                    <span className={cn(
                        "text-2xl font-bold tabular-nums",
                        isLow ? "text-destructive" : "text-foreground"
                    )}>
                        {timeLeft}s
                    </span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                    <div
                        className={cn(
                            "h-full transition-all duration-1000",
                            isLow ? "bg-destructive" : "gradient-primary"
                        )}
                        style={{ width: `${progress}%` }}
                    />
                </div>
            </div>
        </div>
    );
};
