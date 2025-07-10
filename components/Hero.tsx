import React from 'react';

const Hero: React.FC = () => {
  return (
    <section className="text-center pt-24 md:pt-32 pb-12 md:pb-16">
      <h1 className="text-4xl md:text-6xl font-extrabold text-white leading-tight bg-clip-text text-transparent bg-gradient-to-b from-white to-slate-400">
        Stop Drowning in Noise.
        <br />
        Start Finding Signals.
      </h1>
      <p className="mt-6 text-lg md:text-xl text-slate-400 max-w-3xl mx-auto">
        ManagerFM connects your communication tools and uses AI to generate your daily executive summary. Save time, reduce tool fatigue, and focus on what truly matters.
      </p>
      <div className="mt-10">
        <a
          href="#join-waitlist"
          className="bg-brand-blue text-white font-bold py-4 px-8 rounded-lg text-lg hover:bg-sky-400 transition-all duration-300 shadow-lg shadow-sky-500/20 hover:shadow-xl hover:shadow-sky-500/30 transform hover:-translate-y-1"
        >
          Get Early Access
        </a>
      </div>
      <div className="mt-16 w-full max-w-4xl mx-auto opacity-50" aria-hidden="true">
        <svg viewBox="0 0 1360 200" xmlns="http://www.w3.org/2000/svg" className="w-full h-auto">
            <defs>
                <linearGradient x1="50%" y1="0%" x2="50%" y2="100%" id="hero-sound-wave">
                    <stop stopColor="#0ea5e9" offset="0%"></stop>
                    <stop stopColor="#0f172a" stopOpacity="0" offset="100%"></stop>
                </linearGradient>
            </defs>
            <path fill="url(#hero-sound-wave)" d="M0 100C113.333 100 226.667 100 340 100C453.333 100 566.667 100 680 100C793.333 100 906.667 100 1020 100C1133.33 100 1246.67 100 1360 100V200H0V100Z">
                <animate 
                  attributeName="d" 
                  dur="2s" 
                  repeatCount="indefinite" 
                  values="M0 100C113.333 100 226.667 100 340 100C453.333 100 566.667 100 680 100C793.333 100 906.667 100 1020 100C1133.33 100 1246.67 100 1360 100V200H0V100Z; M0 100C113.333 127.333 226.667 141 340 141C453.333 141 566.667 127.333 680 100C793.333 72.667 906.667 59 1020 59C1133.33 59 1246.67 72.667 1360 100V200H0V100Z; M0 100C113.333 100 226.667 100 340 100C453.333 100 566.667 100 680 100C793.333 100 906.667 100 1020 100C1133.33 100 1246.67 100 1360 100V200H0V100Z" />
            </path>
        </svg>
      </div>
    </section>
  );
};

export default Hero;