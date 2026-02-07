import { Hand, Volume2 } from 'lucide-react';

export const Header = () => {
    return (
        <header className="sticky top-0 z-50 glass-card border-b border-border/50">
            <div className="container mx-auto px-4 py-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="gradient-primary p-2.5 rounded-xl shadow-glow">
                            <Hand className="w-6 h-6 text-primary-foreground" />
                        </div>
                        <div>
                            <h1 className="text-xl font-bold text-foreground">
                                Tamil Sign Language
                            </h1>
                            <p className="text-xs text-muted-foreground">
                                Learn • Practice • Speak
                            </p>
                        </div>
                    </div>

                    <div className="flex items-center gap-2 text-muted-foreground">
                        <Volume2 className="w-4 h-4" />
                        <span className="text-sm font-medium">Audio Enabled</span>
                    </div>
                </div>
            </div>
        </header>
    );
};
