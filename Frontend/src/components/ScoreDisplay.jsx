import { Trophy, Target, Percent } from 'lucide-react';
import { cn } from '@/lib/utils';

export const ScoreDisplay = ({
    correct,
    total,
    showPercentage = true,
    className
}) => {
    const percentage = total > 0 ? Math.round((correct / total) * 100) : 0;
    const incorrect = total - correct;

    return (
        <div className={cn(
            "elevated-card p-4 sm:p-6",
            className
        )}>
            <div className="flex items-center justify-between gap-4 flex-wrap">
                {/* Score */}
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-success/10 flex items-center justify-center">
                        <Trophy className="w-5 h-5 text-success" />
                    </div>
                    <div>
                        <p className="text-2xl font-bold text-foreground">
                            {correct}/{total}
                        </p>
                        <p className="text-xs text-muted-foreground">Correct</p>
                    </div>
                </div>

                {/* Incorrect */}
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-destructive/10 flex items-center justify-center">
                        <Target className="w-5 h-5 text-destructive" />
                    </div>
                    <div>
                        <p className="text-2xl font-bold text-foreground">
                            {incorrect}
                        </p>
                        <p className="text-xs text-muted-foreground">Incorrect</p>
                    </div>
                </div>

                {/* Percentage */}
                {showPercentage && (
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
                            <Percent className="w-5 h-5 text-primary" />
                        </div>
                        <div>
                            <p className="text-2xl font-bold text-foreground">
                                {percentage}%
                            </p>
                            <p className="text-xs text-muted-foreground">Accuracy</p>
                        </div>
                    </div>
                )}
            </div>

            {/* Progress bar */}
            <div className="mt-4 h-2 bg-muted rounded-full overflow-hidden">
                <div
                    className="h-full gradient-success transition-all duration-500"
                    style={{ width: `${percentage}%` }}
                />
            </div>
        </div>
    );
};
