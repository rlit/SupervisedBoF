function T = timerange(s, L)

% s = time points/octave

% Logarithmic time sampling s points/octave
if length(L)==2,
    Tmin = L(1);
    Tmax = L(2);
else
    Tmin = 1/L(end) * log(1/0.5);
    Tmax = 1/L(2)   * log(1/0.5);
end
%Smin = 2^floor(log2(sqrt(Tmin)));
%Smax = 2^ceil( log2(sqrt(Tmax)));
%logS = log2(Smin):1/s:log2(Smax);
%T    = (2.^logS).^2;


Tmin = 2^floor(log2(Tmin));
Tmax = 2^ceil( log2(Tmax));
Smin = sqrt(Tmin);
Smax = sqrt(Tmax);
logS = log2(Smin):1/s:log2(Smax);
T    = (2.^logS).^2;