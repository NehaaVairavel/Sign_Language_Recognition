import { Check, X, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

export const FeedbackDisplay = ({
    isCorrect,
    message,
    onTryAgain,
    className
}) => {
    if (isCorrect === null) return null;

    return (
        <div className={cn(
            "flex flex-col items-center gap-4 animate-bounce-in",
            className
        )}>
            <div className={cn(
                "w-20 h-20 rounded-full flex items-center justify-center",
                isCorrect ? "bg-success/20" : "bg-destructive/20"
            )}>
                {isCorrect ? (
                    <Check className="w-10 h-10 text-success" />
                ) : (
                    <X className="w-10 h-10 text-destructive" />
                )}
            </div>

            <div className={cn(
                "feedback-badge",
                isCorrect ? "correct" : "incorrect"
            )}>
                {isCorrect ? (
                    <>
                        <Check className="w-4 h-4" />
                        Correct!
                    </>
                ) : (
                    <>
                        <X className="w-4 h-4" />
                        Try Again
                    </>
                )}
            </div>

            {message && (
                <p className="text-muted-foreground text-center text-sm">
                    {message}
                </p>
            )}

            {!isCorrect && onTryAgain && (
                <Button onClick={onTryAgain} variant="outline" size="sm">
                    <RefreshCw className="w-4 h-4" />
                    Try Again
                </Button>
            )}
        </div>
    );
};
