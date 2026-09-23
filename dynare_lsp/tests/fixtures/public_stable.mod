// Small deterministic fixture; no data, MATLAB code, or external includes.
var y;
model(linear);
    y = 0.5*y(-1);
end;
