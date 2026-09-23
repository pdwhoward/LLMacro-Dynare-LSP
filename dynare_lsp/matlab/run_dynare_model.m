function run_dynare_model(model_path, out_json, dynare_root)
%RUN_DYNARE_MODEL  Run full Dynare on one .mod model; write a JSON verdict.
%
%   Captures success/error, the computed steady state, and an
%   eigenvalue/Blanchard-Kahn summary.  Never throws: every outcome is
%   written to OUT_JSON so the caller can tell a model error apart from a
%   harness/MATLAB crash (in which case OUT_JSON is simply absent).
%
%   Non-finite numbers (Inf/NaN eigenvalues, unsolved steady-state entries)
%   are written as JSON null so the cache is always valid JSON.

global M_ oo_ options_ %#ok<GVMIS>

r = struct();
r.status       = 'unknown';
r.error        = '';
r.error_id     = '';
r.endo_names   = {};
r.endo_nbr     = [];
r.steady_state = {};
r.eigval_abs   = {};
r.n_explosive  = [];
r.nstatic = []; r.npred = []; r.nboth = []; r.nfwrd = [];
r.bk_rank_ok = [];
r.has_dr = false;

try
    if nargin >= 3 && ~isempty(dynare_root) && exist(dynare_root, 'dir')
        if isempty(which('dynare')); addpath(dynare_root); end
    end
    if isempty(which('dynare'))
        r.status = 'no_dynare';
        local_write(r, out_json);
        quit('force');
    end

    [mdir, mname] = fileparts(model_path);
    if isempty(mdir); mdir = pwd; end
    old = cd(mdir);
    restore = onCleanup(@() cd(old)); %#ok<NASGU>

    % Run the model. evalc suppresses (and discards) Dynare's console output
    % so the batch stdout stays clean; errors still propagate to catch.
    % 'nostrict' lets a model that declares an unused exogenous shock run
    % standalone -- notably the MMB modelbase interface shocks (interest_,
    % fiscal_, ...) that are only wired into equations by the MMB common policy
    % block; without it Dynare's strict mode aborts preprocessing.
    try
        evalc("dynare('" + string(mname) + "','noclearall','nograph','nointeractive','nostrict')");
        r.status = 'success';
    catch ME
        r.status   = 'dynare_error';
        r.error    = local_trim(ME.message, 1200);
        r.error_id = ME.identifier;
    end

    % --- model structure (best effort, even on partial failure) ---
    try
        if ~isempty(M_) && isstruct(M_)
            if isfield(M_, 'endo_names'); r.endo_names = M_.endo_names; end
            if isfield(M_, 'endo_nbr');   r.endo_nbr   = M_.endo_nbr;   end
            if isfield(M_, 'nstatic');    r.nstatic    = M_.nstatic;    end
            if isfield(M_, 'npred');      r.npred      = M_.npred;      end
            if isfield(M_, 'nboth');      r.nboth      = M_.nboth;      end
            if isfield(M_, 'nfwrd');      r.nfwrd      = M_.nfwrd;      end
        end
    catch
    end

    % --- steady state ---
    try
        if ~isempty(oo_) && isstruct(oo_) && isfield(oo_, 'steady_state')
            r.steady_state = local_finite_cell(oo_.steady_state);
        end
    catch
    end

    % --- eigenvalues / Blanchard-Kahn ---
    try
        if ~isempty(oo_) && isstruct(oo_) && isfield(oo_, 'dr') ...
                && isstruct(oo_.dr) && isfield(oo_.dr, 'eigval')
            ev = oo_.dr.eigval;
            absev = abs(ev(:)');
            r.eigval_abs  = local_finite_cell(absev);
            r.n_explosive = sum(absev > 1 + 1e-6);   % Inf counts as explosive
            r.has_dr = true;
            if ~isempty(r.nfwrd) && ~isempty(r.nboth)
                r.bk_rank_ok = (r.n_explosive == (r.nfwrd + r.nboth));
            end
        end
    catch
    end

catch ME2
    r.status = 'harness_error';
    r.error  = local_trim(ME2.message, 1200);
end

local_write(r, out_json);
% Force-terminate immediately: some heavy models (high-order perturbation,
% stochastic volatility) compute and write their result fine, then hang in
% MATLAB's normal shutdown and wedge on an uninterruptible I/O wait, leaking
% an unkillable ~650MB process per model. quit('force') skips that teardown.
quit('force');
end


function local_write(r, out_json)
try
    txt = jsonencode(r);
    fid = fopen(out_json, 'w');
    if fid ~= -1
        fwrite(fid, txt, 'char');
        fclose(fid);
    end
catch
end
end


function s = local_trim(s, n)
if ischar(s) && numel(s) > n
    s = s(1:n);
end
end


function c = local_finite_cell(v)
% Numeric vector -> cell row; non-finite entries become [] (JSON null).
v = double(v(:)');
c = num2cell(v);
for i = 1:numel(c)
    if ~isfinite(c{i})
        c{i} = [];
    end
end
end
