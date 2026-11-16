abstract type MultistartExecution end

"""Refine starts in input order in the current Julia process (default)."""
struct SerialMultistart <: MultistartExecution end

"""
    ProcessMultistart(; workers)

Refine in at most `workers` solve-owned local Julia processes. Each uses one
Julia computation thread, sets BLAS threads to one and requests one OpenMP
thread through its environment. The parent schedules starts
until its cooperative deadline, finishes outstanding attempts and returns
records in start order. Worker launch and cleanup are included in wall time.
"""
struct ProcessMultistart <: MultistartExecution
    workers::Int
    function ProcessMultistart(; workers::Integer)
        !(workers isa Bool) && 0 < workers <= typemax(Int) ||
            throw(ArgumentError("workers must be a positive machine integer"))
        new(Int(workers))
    end
end

function _execute_population(::SerialMultistart, refiner, model, starts, info, clock; structural)
    records = _refine_population(refiner, model, starts, info, clock; structural)
    return records, (; mode=:serial, requested_workers=0, worker_pids=Int[])
end

# Serialization is private to a single solve and Julia executable. Workers
# only read the snapshot written in the parent's fresh temporary directory.
# Atomic publication keeps a partial response from being read after a crash.
function _publish_refinement(path, data)
    serialize(path * ".partial", data)
    mv(path * ".partial", path)
end

function _multistart_worker(directory, slot, blas_threads)
    model, refiner, starts = deserialize(joinpath(directory, "request"))
    worker = (; pid=Int(getpid()), julia_threads=Threads.nthreads(), blas_threads)
    worker.julia_threads == worker.blas_threads == 1 || error("unexpected refinement thread budget")
    _publish_refinement(joinpath(directory, "ready-$slot"), worker)
    for line in eachline(stdin)
        ordinal = parse(Int, line)
        start = starts[ordinal]
        started = time_ns()
        outcome = _run_refiner(refiner, model, start)
        reply = (; ordinal, start_id=start.start_id, outcome, seconds=_elapsed(started))
        _publish_refinement(joinpath(directory, "result-$ordinal"), reply)
    end
end

function _multistart_command(refiner, directory, slot)
    project = Base.active_project()
    isnothing(project) && _invalid_option(:refinement, ArgumentError("process multistart requires an active Julia project"))
    imports = if refiner isa IpoptRefiner
        isnothing(Base.get_extension(@__MODULE__, :QHDOPTIpoptExt)) &&
            _invalid_option(:refinement, ArgumentError("load the optional Ipopt package before requesting Ipopt process multistart"))
        "using QHDOPT, Ipopt"
    elseif refiner isa TNCRefiner
        slot == 1 && _tnc_installation(refiner)  # report configuration errors in the parent
        "using QHDOPT"
    else
        _invalid_option(:refinement, ArgumentError("process execution supports the QHDOPT Ipopt and TNC adapters"))
    end
    # Ipopt's initializer can attach its LP64 BLAS. Configure libblastrampoline
    # after all solver imports so both LP64 and ILP64 providers are present.
    bootstrap = "using LinearAlgebra; " * imports * "; BLAS.set_num_threads(1); " *
                "QHDOPT._multistart_worker(ARGS[1], ARGS[2], BLAS.get_num_threads())"
    cmd = `$(Base.julia_cmd()) --startup-file=no --threads=1 --project=$(dirname(project)) -e $bootstrap $directory $slot`
    separator = Sys.iswindows() ? ';' : ':'
    return addenv(cmd, "JULIA_LOAD_PATH"=>join(Base.load_path(), separator),
                  "JULIA_DEPOT_PATH"=>join(DEPOT_PATH, separator),
                  "OPENBLAS_NUM_THREADS"=>"1", "OMP_NUM_THREADS"=>"1", "MKL_NUM_THREADS"=>"1",
                  "BLIS_NUM_THREADS"=>"1", "VECLIB_MAXIMUM_THREADS"=>"1")
end

function _check_multistart_child(process, directory, slot)
    process_exited(process) || return
    log = joinpath(directory, "stderr-$slot")
    throw(WorkerError("Refinement worker $slot exited before a committed response",
                      isfile(log) ? read(log, String) : ""))
end

function _process_records!(records, children, directory, model, starts, info, clock)
    # Wait until all children are ready before assigning any native call.
    # Thus a setup failure cannot leave a partially dispatched population.
    metadata = NamedTuple[]
    for slot in eachindex(children)
        ready = joinpath(directory, "ready-$slot")
        while !isfile(ready)
            for (id, child) in enumerate(children)
                _check_multistart_child(child, directory, id)
            end
            _remaining(clock) > 0 || return 1
            sleep(0.01)
        end
        push!(metadata, deserialize(ready))
    end
    inflight = zeros(Int, length(children))
    next = 1
    while next <= length(starts) || any(!iszero, inflight)
        for (slot, process) in enumerate(children)
            _check_multistart_child(process, directory, slot)
            i = inflight[slot]
            if i != 0
                path = joinpath(directory, "result-$i")
                isfile(path) || continue
                reply = deserialize(path)
                reply.ordinal == i && reply.start_id == starts[i].start_id ||
                    error("refinement worker returned a different start identity")
                records[i] = _refinement_record(model, starts[i], info, reply.outcome, reply.seconds;
                                                worker=metadata[slot])
                inflight[slot] = 0
            end
            if next <= length(starts) && _remaining(clock) > 0
                println(process.in, next)
                flush(process.in)
                inflight[slot] = next
                next += 1
            end
        end
        all(iszero, inflight) && break
        sleep(0.01)
    end
    return next
end

function _execute_population(execution::ProcessMultistart, refiner, model, starts, info, clock; structural)
    worker_pids = Int[]
    provenance = (; mode=:process, requested_workers=execution.workers, worker_pids,
                   julia_threads_per_worker=1, blas_threads_per_worker=1)
    if isempty(starts) || structural || refiner isa NoRefiner || _remaining(clock) <= 0
        return _refine_population(refiner, model, starts, info, clock; structural), provenance
    end
    records = Vector{RefinementRecord}(undef, length(starts))
    mktempdir() do directory
        serialize(joinpath(directory, "request"), (model, refiner, starts))
        children = Base.Process[]
        try
            for slot in 1:min(execution.workers, length(starts))
                _remaining(clock) > 0 || break
                cmd = _multistart_command(refiner, directory, slot)
                process = Base.disable_sigint() do
                    p = open(pipeline(cmd; stderr=joinpath(directory, "stderr-$slot")), "w")
                    push!(children, p)
                    p
                end
                push!(worker_pids, Int(getpid(process)))
            end
            next = _process_records!(records, children, directory, model, starts, info, clock)
            for i in next:length(starts)
                skipped = RefinerOutcome(status=:not_attempted_deadline, solver_called=false)
                records[i] = _refinement_record(model, starts[i], info, skipped, 0.0)
            end
            if _remaining(clock) > 0
                foreach(p -> close(p.in), children)
                foreach(wait, children)
                for (slot, process) in enumerate(children)
                    success(process) || _check_multistart_child(process, directory, slot)
                end
            end
        finally
            # Includes startup errors, caller interruption, and a deadline
            # during bootstrap. Only processes created for this solve are reaped.
            Base.disable_sigint() do
                @sync for process in children
                    @async try
                        _force_stop(process)
                    finally
                        close(process.in)
                    end
                end
            end
        end
    end
    return records, provenance
end
