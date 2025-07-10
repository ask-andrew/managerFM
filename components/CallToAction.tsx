
import React, { useState } from 'react';
import { CheckCircleIcon } from './Icons';

const CallToAction: React.FC = () => {
  const [email, setEmail] = useState('');
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (email && email.includes('@')) {
      console.log('Submitted email:', email);
      setSubmitted(true);
    }
  };

  return (
    <section id="join-waitlist" className="py-20">
      <div className="bg-gradient-to-r from-brand-blue to-sky-400 p-8 md:p-12 rounded-2xl text-center">
        <h2 className="text-3xl md:text-4xl font-extrabold text-white mb-4">
          Be the First to Try ManagerFM
        </h2>
        <p className="text-lg text-sky-100 max-w-2xl mx-auto mb-8">
          Join our waitlist for early access and help shape the future of management.
        </p>

        {submitted ? (
          <div className="flex items-center justify-center gap-3 bg-green-500/20 text-white p-4 rounded-lg max-w-md mx-auto animate-fade-in-up">
            <CheckCircleIcon className="h-7 w-7" />
            <p className="font-semibold">Thanks for joining! We'll be in touch.</p>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="max-w-md mx-auto flex flex-col sm:flex-row gap-3">
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="your.email@company.com"
              required
              className="w-full px-5 py-3 rounded-lg bg-white/90 text-slate-900 placeholder-slate-500 focus:outline-none focus:ring-4 focus:ring-white/50 transition duration-300"
            />
            <button
              type="submit"
              className="bg-brand-dark text-white font-bold py-3 px-6 rounded-lg hover:bg-slate-800 transition-colors duration-300 flex-shrink-0"
            >
              Join Waitlist
            </button>
          </form>
        )}
      </div>
    </section>
  );
};

export default CallToAction;
