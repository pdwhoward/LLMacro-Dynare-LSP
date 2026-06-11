function dynare_session_server(workdir, dynare_root)
%DYNARE_SESSION_SERVER  Persistent Dynare runner for one reused MATLAB session.
%
%   Launched once via ``matlab -batch
%   "dynare_session_server('<workdir>','<dynare_matlab_dir>')"``.  It watches
%   WORKDIR for ``<id>.job`` files (JSON ``{mod, out}``), runs each model in
%   this single session, writes the verdict to ``<out>`` atomically, and loops
%   until a ``STOP`` file appears.  Reusing one session avoids paying MATLAB's
%   ~7s startup on every model run.
%
%   The per-model verdict format is identical to ``run_dynare_model.m`` so the
%   Python side (``matlab_runner._shape_from_record``) is unchanged.

global M_ oo_ options_ %#ok<GVMIS>

try
    if nargin >= 2 && ~isempty(dynare_root) && exist(dynare_root, 'dir')
        if isempty(which('dynare')); addpath(dynare_root); end
    end
catch
end

% File-based readiness signal (robust to -batch stdout buffering).
try
    fid = fopen(fullfile(workdir, 'READY'), 'w'); if fid ~= -1; fclose(fid); end
catch
end

while true
    if exist(fullfile(workdir, 'STOP'), 'file'); break; end
    jobs = dir(fullfile(workdir, '*.job'));
    if isempty(jobs)
        pause(0.03);
        continue;
    end
    [~, ord] = sort([jobs.datenum]);
    jobname = jobs(ord(1)).name;
    jobpath = fullfile(workdir, jobname);
    spec = [];
    try
        spec = jsondecode(fileread(jobpath));
    catch
    end
    delete(jobpath);
    if isempty(spec) || ~isfield(spec, 'mod') || ~isfield(spec, 'out')
        continue;
    end
    run_one(spec.mod, spec.out, dynare_root);
end
quit('force');
end


function run_one(model_path, out_json, dynare_root)
% Run one .mod end-to-end; never throws.  Mirrors run_dynare_model.m but does
% NOT quit (the session is reused) and resets Dynare globals first so a prior
% model cannot leak into this one.
clear global M_ oo_ options_ estim_params_ bayestopt_ dataset_ oo_recursive_;
global M_ oo_ options_ %#ok<GVMIS>

r = struct();
r.status = 'unknown'; r.error = ''; r.error_id = '';
r.endo_names = {}; r.endo_nbr = [];
r.steady_state = {}; r.eigval_abs = {}; r.n_explosive = [];
r.nstatic = []; r.npred = []; r.nboth = []; r.nfwrd = [];
r.bk_rank_ok = []; r.has_dr = false;

try
    if isempty(which('dynare')) && ~isempty(dynare_root) && exist(dynare_root, 'dir')
        addpath(dynare_root);
    end
    if isempty(which('dynare'))
        r.status = 'no_dynare';
        local_write(r, out_json);
        return;
    end

    [mdir, mname] = fileparts(model_path);
    if isempty(mdir); mdir = pwd; end
    old = cd(mdir);
    restore = onCleanup(@() cd(old)); %#ok<NASGU>

    try
        evalc("dynare('" + string(mname) + "','noclearall','nograph','nointeractive','nostrict')");
        r.status = 'success';
    catch ME
        r.status = 'dynare_error';
        r.error = local_trim(ME.message, 1200);
        r.error_id = ME.identifier;
    end

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

    try
        if ~isempty(oo_) && isstruct(oo_) && isfield(oo_, 'steady_state')
            r.steady_state = local_finite_cell(oo_.steady_state);
        end
    catch
    end

    try
        if ~isempty(oo_) && isstruct(oo_) && isfield(oo_, 'dr') ...
                && isstruct(oo_.dr) && isfield(oo_.dr, 'eigval')
            ev = oo_.dr.eigval;
            absev = abs(ev(:)');
            r.eigval_abs  = local_finite_cell(absev);
            r.n_explosive = sum(absev > 1 + 1e-6);
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
end


function local_write(r, out_json)
% Write JSON to a temp file then rename, so the caller never reads a partial
% file (the renamed <out> is the completion signal).
try
    txt = jsonencode(r);
    tmp = [out_json '.tmp'];
    fid = fopen(tmp, 'w');
    if fid ~= -1
        fwrite(fid, txt, 'char');
        fclose(fid);
        movefile(tmp, out_json, 'f');
    end
catch
end
end


function s = local_trim(s, n)
if ischar(s) && numel(s) > n; s = s(1:n); end
end


function c = local_finite_cell(v)
v = double(v(:)');
c = num2cell(v);
for i = 1:numel(c)
    if ~isfinite(c{i}); c{i} = []; end
end
end
