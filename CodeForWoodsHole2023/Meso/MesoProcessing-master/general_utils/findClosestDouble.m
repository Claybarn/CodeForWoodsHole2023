function ind = findClosestDouble(vec, x)

[~,ind] = min(abs(vec-x));