### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 76730d06-06da-4466-8814-2096b221090f
begin
	# Packages for Notebook experience
	using PlutoUI, PlutoLinks, PlutoTeachingTools, PlutoTest
	using Plots

	# Packages for parallelization
	#using SharedArrays
	using CpuId, Hwloc
	using ThreadsX, OhMyThreads, Polyester
	using FLoops
	using StructArrays
	# Packages for benchmarking
	using BenchmarkTools

	# Packages needed by model
	using Distributions, Random
	using QuadGK
	using StaticArrays
	Random.seed!(42)

	nb_link_prefix = PlutoRunner.notebook_id[] |>string; # for making urls to notebook
end

# ╔═╡ 85aad005-eac0-4f71-a32c-c8361c31813b
md"""
# Lab 6, Exercise 1
## Parallelization: Shared-memory model & Multi-threading
"""

# ╔═╡ bdf61711-36e0-40d5-b0c5-3bac20a25aa3
md"""
In this lab, we'll explore multiple different ways that we can parallelize calculations across multiple cores of a single workstation or server.
This exercise will focus on parallelization using multiple *threads*.
A separate exercise will focus on parallelization using multiple *processes*.
"""

# ╔═╡ 629442ba-a968-4e35-a7cb-d42a0a8783b4
protip(md"""
In my experience, parallelization via multiple threads tend to be more efficient than using multiple processes for small to mid-scale parallelization (several to tens of cores).  For larger-scale parallelization (hundreds or thousands of cores), using multiple processes can be advantagous or even required.   For me, multi-threading is my "go-to" method for an initial parallelization.  That said, it's good to be aware of some of the reasons that some projects choose to parallelize their code over multiple processes (e.g., if you're concerned about security of data, robustness to errors in one process).  For me, the main advantage of using multiple processes is that multiple processes will be necessary once we transition to distributed memory computing.  Therefore, parallelizing your code using multiple processes can make it easier to scale up when you want to use more cores or more memory than is avaliable in a single node.

Near the end of this exercise we'll see an example of how a programming interface makes it easy to transition code between multi-threaded and multi-process models.
""")

# ╔═╡ b133064a-d02e-4a71-88d7-e430b80d24b1
md"""
## Hardware & Pluto server configuration
"""

# ╔═╡ 571cab3f-771e-4464-959e-f351194049e2
md"""
Most modern workstations and even laptops have multiple processor cores.
Before we get started, let's get some information about the processor that our Pluto server is running on and double check that we're set to use an appropriate number of threads.
"""

# ╔═╡ 0c775b35-702e-4664-bd23-7557e4e189f4
with_terminal() do
	Sys.cpu_summary()
end

# ╔═╡ 3059f3c2-cabf-4e20-adaa-9b6d0c07184f
md"""
If you're running this notebook on your own computer, then we'll want to make sure that we set the number of threads to be no more than the number of processor cores listed above. It's very likely that you might be better off requesting only half the number of processors as listed above. (Many processors present themselves as having more cores than they actually do. For some applications, this can be useful.  For many scientific applications it's better to only use as many threads as physical cores that are avaliable.
"""

# ╔═╡ 4fa907d0-c556-45df-8056-72041edcf430
md"""
The [CpuId.jl](https://github.com/m-j-w/CpuId.jl) package provides some useful functions to query the properties of the processor you're running on.  It provides functions to easily find out the physical capabilities of the computer you're using.
"""

# ╔═╡ 73e5e40a-1e59-41ed-a48d-7fb99f5a6755
cpucores()   # query number of physical cores

# ╔═╡ f97f1815-50a2-46a9-ac20-e4a3e34d898c
cputhreads() # query number of logical cores

# ╔═╡ 53da8d7a-8620-4fe5-81ba-f615d2d4ed2a
if cpucores() < cputhreads()
	warning_box(md"""Your processor is presenting itself as having $(cputhreads()) cores, when it really only has $(cpucores()) cores.  I suggest limiting the number of threads to no more than $(cpucores()).  
	
	If you're running on Lynx or Roar Collab, then you should also limit the number of threads you use to the number of CPU cores assigned to your job by the slurm workload manager.
	""")
end

# ╔═╡ 8df4e8d2-b955-424d-a7ec-264ce2b0e506
md"""
Just because you're running on a server with $(cpucores()) cores doesn't mean that you should use them all for yourself.  
If you're using the Lynx portal and BYOE JupyterLab server, then you should limit the number of threads you use to the number of CPU cores that have been allocated to your job.  
For this lab, you should request at least three CPU cores be allocated to your session when you first submit the request for the BYOE JupyterLab server using the box labeled "Number of Cores", i.e. before you open this notebook and even before you start your Pluto session.  I'd suggest asking for 4 CPU cores, so you can notice more significant speed-ups from parallelization.
"""

# ╔═╡ 826d2312-0803-4c36-bb72-df6d8241910c
md"""
We can look up the parameters for your job using **environment variables**.  In Julia, they are accessible using the dictionary, `ENV`.  For example, the environment variable `ENV["SLURM_CPUS_PER_TASK"]` tells us how many CPU cores were allocated for our task.
"""

# ╔═╡ f76f329a-8dde-4790-96f2-ade735643aeb
if haskey(ENV,"PBS_NUM_PPN")
	procs_per_node = parse(Int64,ENV["PBS_NUM_PPN"])
	md"Your PBS job was allocated $procs_per_node CPU cores per node."
elseif haskey(ENV,"SLURM_CPUS_PER_TASK") && haskey(ENV,"SLURM_TASKS_PER_NODE")
    procs_per_task = parse(Int64,ENV["SLURM_CPUS_PER_TASK"])
    tasks_per_node = parse(Int64,ENV["SLURM_TASKS_PER_NODE"])
	procs_per_node = procs_per_task * tasks_per_node
	md"Your Slurm job was allocated $procs_per_node CPU cores per node."
else
	procs_per_node = cpucores()
	md"It appears you're not running this on the Lynx or Roar Collab clusters."
end

# ╔═╡ 0e4d7808-47e2-4740-ab93-5d3973eecaa8
if !ismissing(procs_per_node)
	if procs_per_node > 4
		warning_box(md"""While we're in class (and the afternoon/evening before labs are due), please ask for just 4 cores, so there will be enough to go around.

		If you return to working on the lab outside of class, then feel free to try benchmarking the code using 8 cores or even 16 cores. Anytime you ask for more than four cores, then please be extra diligent about closing your session when you're done.""")
	end		
end

# ╔═╡ 410b6052-6d6d-4fd8-844e-8941845d8d90
md"""
This Pluto notebook has **$(Threads.nthreads()) threads** available for multithreaded computations.
"""

# ╔═╡ 8a50e9fa-031c-4912-8a2d-466e6a9a9935
md"""
Even when you have a JupyterLab server (or remote desktop or job scheduled by Slurm, PBS or HTCondor) that has been allocated multiple CPU cores, that doesn't mean that any code will make use of more than one core.  For code to make use of those cores, it has to be written in a way that indicates which portions of the code may execute in parallel.  This exercise will demonstrate several ways that you can write parallel code.  You don't need to master the syntax for all of them.  As always, the concepts are more important.  When it comes time to write parallel code for your project, you can come back here to remind yourself of the syntax specific to whichever package you are using to express your parallel computations.
"""

# ╔═╡ 7df5fc86-889f-4a5e-ac2b-8c6f68d7c32e
warning_box(md"""The Lynx Portal's Pluto server for this class has been configured to start notebooks with as many threads as physical cores that were allocated to the parent job.  

However, if you start julia manually (e.g., from the command line or remote desktop), then you should check that its using the desired number of threads.  You can control this by either setting the `JULIA_NUM_THREADS` environment variable before you start julia or by adding the `-t` option on the command line when you start julia.  Somewhat confusingly, even if you start julia using multiple threads, that doesn't mean that the Pluto server will assign that many threads to each notebook.  If you run your own Pluto server, then you can control the number of threads used within a notebook by starting it with
```julia
using Pluto
Pluto.run(threads=4)
```""")

# ╔═╡ cc1418c8-3261-4c70-bc19-2921695570a6
Threads.nthreads()  # Number of threads available to this Pluto notebook

# ╔═╡ 7f724449-e90e-4f8b-b13c-9640a498893c
@test 1 <= Threads.nthreads() <= cpucores()

# ╔═╡ c85e51b2-2d3d-46a2-8f3f-03b289cab288
 @test !ismissing(procs_per_node) && 1 <= Threads.nthreads() <= procs_per_node

# ╔═╡ 907766c5-f084-4ddc-bb52-336cb037d521
md"1a.  How many threads is your notebook using?  (Please enter it as an integer rather than a function call, so that it gets stored in your notebook.  That way the TA and instructor will be able to interpret the speed-up factors you get below.)"

# ╔═╡ 0bcde4df-1e31-4774-a31f-bd451bb6f758
response_1a = missing # Insert response as a simple integer, and not as a variable for the function

# ╔═╡ c41d65e3-ea35-4f97-90a1-bfeaeaf927ad
begin
    if !@isdefined(response_1a)
		var_not_defined(:response_1a)
    elseif ismissing(response_1a)
    	still_missing()
	elseif !(typeof(response_1a) <: Integer)
		warning_box(md"response_1a should be an Integer")
	elseif !(1<(response_1a))
		warning_box(md"Please restart your JupyterLab session and use at least 2 cores.")
	elseif (response_1a) != Threads.nthreads()
		warning_box(md"That's not what I was expecting.  Please double check your response.")
	else
		correct(md"Thank you.")
	end
end


# ╔═╡ 6e617a7c-a640-4cb3-9451-28a0036d8fdc
md"# Calculation to parallelize"

# ╔═╡ 5e6c430a-cd2f-4169-a5c7-a92acef813ac
md"""
For this lab, I've written several functions that will be used to generate simulated spectra with multiple absorption lines.  This serves a couple of purposes.
First, you'll use the code in the exercise, so you have a calculation that's big enough to be worth parallelizing.  For the purposes of this exercise, it's not essential that you review the code I provided in the `src/*.jl` files.  However, the second purpose of this example is providing code that demonstrates several of the programming patterns that we've discussed in class.  For example, the code in the `ModelSpectrum` module
- is in the form of several small functions, each of which does one specific task.
- has been moved out of the notebook and into `.jl` files in the `src` directory.
- creates objects that compute a model spectrum and a convolution kernel.
- uses [abstract types](https://docs.julialang.org/en/v1/manual/types/#Abstract-Types-1) and [parametric types](https://docs.julialang.org/en/v1/manual/types/#Parametric-Types-1), so as to create type-stable functions.
- has been put into a Julia [module](https://docs.julialang.org/en/v1/manual/modules/index.html), so that it can be easily loaded and so as to limit potential for namespace conflicts.

You don't need to read all of this code right now.  But, when you're writing code for your class project, you're likely to want to make use of some of these same programming patterns.   It may be useful to refer back to this code later to help see examples of how to apply these design patterns in practice.

In the Helper code section at the bottom of the notebook, we read the code in `src/model_spectrum.jl` and place it in a module named ModelSpectrum.  Note that this implicitly includes the code from other files: `continuum.jl`, `spectrum.jl` and `convolution_kernels.jl`.
Then we'll bring several of the custom types into scope, so we can use them easily below.
"""

# ╔═╡ c31cf36c-21ec-46f1-96aa-b014ff094f8a
md"""
## Synthetic Spectrum
In this exercise, we're going to create a model spectrum consisting of continuum, stellar absorption lines, and telluric absorption lines.
The `ModelSpectrum` module provides a `SimulatedSpectrum` type.
We need to create a `SimulatedSpectrum` object that contains specific parameter values.  The function below will do that for us.
"""

# ╔═╡ 7026e51d-c3e4-4503-9f35-71074b0c2f1a
md"""
Next, we specify a set of wavelengths where the spectrum will be defined,
and create a functor (or function-like object) that contains all the line properties and can compute the synthetic spectrum.
"""

# ╔═╡ ad302f2b-69dc-4559-ba12-d7fb2e8e689e
begin  # Pick range of of wavelength to work on.
	lambda_min = 5000
	lambda_max = 6000
end;

# ╔═╡ 16ad0225-c7d6-455b-8eb0-3e93c9f9f91a
md"## Convolved spectrum

Next, we will create an object containing a model for the point spread function (implemented as a mixture of multiple Gaussians).
Then we create a [functor](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects) that can compute the convolution of our spectral model with the point spread function model.
"

# ╔═╡ 324a9a25-1ec4-4dc2-a7ca-e0f1f56dbf66
md"""
## Visualize the models
Before going further, it's probably useful to plot both the raw spectrum and the convolved spectrum.
"""

# ╔═╡ 52127f57-9a07-451a-bb24-c1f3c5581f0a
begin 	# You may want to adjust the num_lambda to make things more/less computationally intensive
	num_lambda = 16*1024
	lambdas = range(lambda_min,stop=lambda_max, length=num_lambda)
	lambdas = collect(lambdas) # to make an actual array
end;

# ╔═╡ 75948469-1347-45e2-9281-f366b41d0e04
md"""
That's fairly crowded, you it may be useful to zoom in on a narrower range.
"""

# ╔═╡ 4d1cf57f-b394-4f37-98c3-0d765f4ee635
md"""
Plot width:
$(@bind idx_plt_width Slider(8:min(1024,length(lambdas)), default=min(128,floor(Int,length(lambdas)//2)) ) )
center:
  $(@bind idx_plt_center Slider(1:length(lambdas), default = floor(Int,length(lambdas)//2)) )

"""

# ╔═╡ cddd761a-f051-4338-9e40-d35e050060d3
begin
		idx_plt_lo = max(1,idx_plt_center - idx_plt_width)
		idx_plt_hi = min(length(lambdas),idx_plt_center + idx_plt_width)
		idx_plot = idx_plt_lo:idx_plt_hi
end;

# ╔═╡ ee96411d-e3fa-442b-b0fe-10d6ede37b6a
md"""
You can adjust the sliders to interactively explore our model spectra.
"""

# ╔═╡ b92aad2e-8a3b-4edf-ae7e-6e3cff6eead4
protip(md"Feel free to look at the hidden code in the cells above for the lower plot and slider bars, as well as the documentation at [PlutoUI.jl](https://docs.juliahub.com/PlutoUI/abXFp/0.7.52/) or the [example notebook](https://featured.plutojl.org/basic/plutoui.jl) for examples of how to make interactive widgets in your notebooks.")

# ╔═╡ e5f9fa06-9fbb-40a8-92de-71523775d257
md"""
# Serial implementations
## Benchmarking spectrum (w/o convolution)

Before we parallelize anything, we want to benchmark the calculation of spectra on a single processor.  To avoid an annoying lag when using the notebook, we won't use the `@benchmark` script.  Instead, we'll run each calculation just twice, once to ensure it's compiled and a second time for benchmarking it with the `@timed` macro.  When it comes time to benchmark your project code, you'll want to collect multiple samples to get accurate benchmarking results.
"""

# ╔═╡ b195ebd2-9584-40b8-ae3e-6d9ce88b5398
md"""
Let's think about what's happening with the serial version.
With `raw_spectrum(lambdas)` or `raw_spectrum.(lambdas)` we will evaluate the spectrum model at each of the specified wavelengths using a few different syntaxes.
"""

# ╔═╡ d6d3a2d1-241e-44c1-a11b-5bfb2b3c5f4b
md"""
As expected, the different versions perform very similarly in terms of wall-clock time and memory allocated.
"""

# ╔═╡ 0344a74d-456b-44f0-84dc-c2fdbd41a379
md"""
## Benchmarking convolved spectrum

Next, we'll evaluate the convolution of the raw spectrum with the PSF model at each of the wavelengths, using `conv_spectrum`.
"""

# ╔═╡ 51adffd7-8fb6-4ed2-8510-303a37d6efc3
md"""
Now, the two implementations performed very differently.  Let's think about what's causing that difference.
In each case, the convolution integral is being computed numerically by [QuadGK.jl](https://github.com/JuliaMath/QuadGK.jl).  On one hand, it's impressive that QuadGK.jl was written in a generic way, so that it can compute an integral of a scalar (when we used the broadcasting notation) or an integral of vectors (when we passed the vector of wavelengths without broadcasting).
On the other hand, there's a significant difference in the wall clock time and lots more memory being allocated when we pass the vector, instead of using broadcasting.
When we pass a vector, the `quadgk` is computing the convolution integral using vectors.  Since the size of the vectors isn't known at compile time they must be allocated  on the heap.  This results in many unnecessary memory allocations (compared to if the calculations were done one wavelength at a time).

We can get around this problem by using broadcasting or a map, so the convolution integral is performed on scalars, once for each wavelength.  This significantly reduces the number of memory allocations and the runtime.  This also has the advantage that we've broken up the work into many independent calculations that could be performed in parallel.
"""

# ╔═╡ 71d943e3-761a-4337-b412-b0b768483bc2
protip(md"Interestingly, there's actually more work to do in the case of computing integrals of scalars, since the adaptive quadrature algorithm chooses how many points and where to evaluate the integrand separately for each wavelength.  However, the added cost of memory allocations is much more expensive than the cost of the added calculations.

Another complicating factor, the answers aren't identical.  This is because the criteria used by `quadgk` for when to stop evaluating the integrand at more points changes depending on whether it's deciding when to stop for each wavelength separately or for the entire vector at once.

In principle, we could further optimize the serial version to avoid unnecessary memory allocations.  QuadGK.jl provides a function `quadgk!` that writes the output into a preallocated space.  Even `quadgk!` needs some memory to compute intermediate values.  Normally,  `quadgk` or `quadgk!` will allocate a buffer for segments automatically.  However, you can instead allocate a buffer using `alloc_segbuf(...)` and pass the preallocated buffer as the `segbuf` argument.  When using multiple threads, we'd need to allocate a separate buffer for each thread and make sure that each thread uses only its own buffer.  However, it would take some time to figure out how to do that and to test the resulting code.  In practice, it's often a better use of our time to make a pretty good serial code that can be parallelized well and to use our time to parallelize that, rather than making the most efficient serial code possible.")

# ╔═╡ db1583f4-61cb-43e0-9326-d6c15d8fad5a
md"""
## Map
Our calculation is one example of a very useful programming pattern, known as **map**.  The map pattern corresponds to problems where the total work can be organized as doing one smaller calculation many times with different input values.
Julia provides a [`map`](https://docs.julialang.org/en/v1/base/collections/#Base.map) function that can be quite useful.
`map(func,collection)` applies func to every element of the collection and returns a collection similar in size to collection.
There is also a `map!` function for writing the results into memory that's been preallocated.
In our example, each input wavelength is mapped to our output flux.
"""

# ╔═╡ f108d26b-6c75-4eb6-9e88-a60ec038a73c
md"""
As expected, the map versions perform very similarly in terms of wall-clock time and memory allocated to the broadcasted versions for both the raw and convolved spectra.
"""

# ╔═╡ 21f305db-24e1-47d1-b1f4-be04ca91780e
protip(md"""
It is possible to have each function return an array.  Then the output is an array of arrays.  In that case we could use `stack` to return a 2-d array. 

However, if each function returns a NamedTuple (or a custom struct), then the output of `map` is an array of NamedTuples (or an array of structs).  
These can be converted into a `StructArray` using the [StructArrays.jl](https://juliaarrays.github.io/StructArrays.jl/stable/) package or a DataFrame using the [DataFrames.jl](https://dataframes.juliadata.org/latest/) package.  However, sometimes getting the outputs into a format we want to use for subsequent calculations (e.g., arrays for each output, rather than an array of structs) is often a bit of a hassle and more error prone than just writing our code in terms of either a `for` loop or a broadcasted function.""")

# ╔═╡ e71cede9-382e-47e2-953a-2fa96ed50002
md"## Loop (serial)"

# ╔═╡ 4d54b6a7-3fc0-4c63-8a9d-d683aa4ecefe
md"""
Sometimes it's cumbersome to write code in terms of `map` functions.  For example, you might be computing multiple quantities during one pass of your data (e.g., calculating a sample variance in lab 1).  In these cases, it's often more natural to write your code as a `for` loop.
"""

# ╔═╡ a44a3478-541d-40d6-9d99-04b918c16bfb
md"""We'll implement a serial version as a starting point and comparison.
"""

# ╔═╡ 96914ff8-56c8-4cc8-96bc-fd3d13f7e4ce
md"As expected the performance is very similar to the versions using broadcasting or `map`."

# ╔═╡ 32685a28-54d9-4c0d-8940-e82843d2cab2
md"# Parallelization via multiple threads"

# ╔═╡ 3717d201-0bc3-4e3c-8ecd-d835e58f6821
md"""
Julia has native support for using multiple **threads**.  This is useful when you have one computer with multiple processor cores.  Then each thread can execute on a separate processor core.  Because the threads are part of the same **process**, every thread has access to all the memory used by every other thread.  Programming with threads requires being careful to avoid undefined behavior because threads read and write to the same memory location in an unexpected order.  In general, multi-threaded programming can be intimidating, since arbitrary parallel code is hard to write, read, debug and maintain.  One way to keep things manageable is to stick with some common programming patterns which are relatively easy to work with.  In this exercise, we'll explore using multi-threading for a parallel for loop, parallel map, parallel reduce, and parallel mapreduce.
"""

# ╔═╡ 04bcafcd-1d2f-4ce5-893f-7ec5bb05f9ed
md"""
1b.  Given that this notebook is using $(Threads.nthreads()) threads, what is the theoretical maximum improvement in the performance of calculating the spectrum when using multi-threading relative to calculating the spectrum in serial?  """

# ╔═╡ ca8ceb27-86ea-4b90-a1ae-86d794c9fc98
response_1b = missing  # md"Insert your response"

# ╔═╡ 8b61fca1-2f89-4c74-bca8-c6cc70ba62ad
begin
    if !@isdefined(response_1b)
		var_not_defined(:response_1b)
    elseif ismissing(response_1b)
    	still_missing()
	end
end

# ╔═╡ bd81357b-c461-458e-801c-610893dd5ea1
md"## Parallel Loop"

# ╔═╡ 0e0c25d4-35b9-429b-8223-90e9e8be90f9
md"""
It is possible to parallelize for loops using multiple threads.  Julia's built-in `Threads` module provides one implementation.
"""

# ╔═╡ 3604a697-8f21-447f-bf75-b3af989e9896
md"""
Now, we'll time `calc_spectrum_threaded_for_loop` to compare to the serial version and theoretical maximum improvement.
"""

# ╔═╡ 791041e9-d277-4cac-a5ac-1d6ec52e0287
md"""
While Threads.@threads can be useful for some simple tasks, there is active development of packages that provide additional high-level functions and macros to make multi-threaded programming easier.  For example, the [ThreadsX.jl](https://github.com/tkf/ThreadsX.jl) package provides a `foreach` function, the [OhMyThreads.jl](https://juliafolds2.github.io/OhMyThreads.jl/stable/) package provides a `tforeach`, the [Polyester.jl](https://github.com/JuliaSIMD/Polyester.jl) package provides a `@batch` macro, and the FLoops package provides a `@floop` macro. I'll demonstrate each below.
"""

# ╔═╡ f9b3f5ce-cfc1-4d59-b21e-2fd07075f036
md"""
At first, it may seem like the above examples are just alternative syntaxes for writing a loop parallelized over multiple threads.  Each package has made slightly different implementation choices optimized for different circumstances and considerations.  For now, I just wanted to demonstrate the syntax of each in case it would be useful to you for your project.  
"""

# ╔═╡ 7c367a0b-c5b9-459b-9ccf-e07c84e0b32a
protip(md"""
Inevitably, one package/pattern for parallelizing your code will be a little more efficient than the others. But there are often multiple ways to implement parallelism that result in comparable in run-time. When the performance is similar, other considerations (e.g., ease of programming, quality of documentation, ease swapping out different parallelization strategies) may play a major role in your decision of how to implement parallelism.
	   
[`ThreadsX.jl`](https://github.com/tkf/ThreadsX.jl) provides a drop-in replacement for several functions from Base.  The common interface makes it easy to swap in for serial code quickly.  

[`OhMyThreads.jl`](https://juliafolds2.github.io/OhMyThreads.jl/stable/) provides very similar syntax with just slightly different names.  It's also being actively developed and improved.  One distinguishing feature is that it aims to allow nested threaded operations to result in good performance.  If I had to recommend just one package for multi-threading, this is what I'd currently  recommend.   

In terms of raw performance, [`Polyester.jl`](https://github.com/JuliaSIMD/Polyester.jl) appears to be the fastest at the moment, particularly for loops that have little computation.  However, it can only use a static schedule. Therefore, nested parallel loops don't work well with Polyester.

While [`FLoops.jl`](https://github.com/JuliaFolds/FLoops.jl) requires a somewhat different syntax, it can make it easier to swap between multiple forms of parallelism.  Therefore, writing your code so it can be multi-threaded using FLoops is likely to make it very easy to parallelize your code for a distributed memory architecture.  FLoops can even make it easy to parallelize code using a GPU.  

It's worth keeping these tradeoffs in mind when planning your project.  
""")

# ╔═╡ 496e8c5e-251b-4448-8c59-541877d752c1
md"""
## Parallel Map

If you can write your computations in terms of calling **`map`**, then one easy way to parallelize your code is to replace the call to `map` with a call to a parallel map that makes use of multiple threads, such as `ThreadsX.map` or `OhMyThreads.tmap`.
If your julia kernel has only a single thread, then it will still run in serial.  But if you have multiple threads, then `ThreadsX.map` or `OhMyThreads.tmap` will parallelize your code.
"""

# ╔═╡ 263e96b7-e659-468d-ba97-ca9832f6ea4d
md"""
**Q1c:**  How much faster do you expect the `conv_spectrum` code to run when using map with multiple threads relative to the serial version?
"""

# ╔═╡ 86e7d984-c128-4d2e-8599-3bc70db87a1d
response_1c = missing # md"Insert your response"

# ╔═╡ c69c0a4a-b90b-414c-883d-3aa50c04b5e1
begin
    if !@isdefined(response_1c)
		var_not_defined(:response_1c)
    elseif ismissing(response_1c)
    	still_missing()
	end
end

# ╔═╡ 4ad081a2-b5c2-48ff-9a28-ec9c8d9f0d0e
begin
    if !@isdefined(response_1c)
		var_not_defined(:response_1c)
    elseif ismissing(response_1c)
    	still_missing()
	end
end

# ╔═╡ 2399ce76-b6da-4a61-bcda-aee22dd275f8
md"""
1d. How did the performance improvement compare to the theoretical maximum speed-up factor and your expectations?
"""

# ╔═╡ a25c6705-54f4-4bad-966e-a8f13ae4c711
response_1d = missing  # md"Insert your response"

# ╔═╡ 739136b1-6b01-44c0-bbfd-dcb490d1e191
begin
    if !@isdefined(response_1d)
		var_not_defined(:response_1d)
    elseif ismissing(response_1d)
    	still_missing()
	end
end

# ╔═╡ dcce9a84-a9b1-47c1-8e08-7575cb299b56
md"""
Depending on the computer being used, you might be a little disappointed in the speed-up factor for one or both of those calls.  What could have gone wrong?
"""

# ╔═╡ bd185f74-d666-42b4-8da1-c768217f7782
hint(md"""  In this case, we have a non-trivial, but still modest amount of work to do for each wavelength.  `map` distributes the work one element at a time.  The overhead in distributing the work and assembling the pieces likely ate into the potential performance gains.  To improve on this, we can tell `map` to distribute the work in batches.  Below, we'll specify an optional named parameter (e.g., `basesize` or `chucksize` depending on the library).  (Feel free to try changing the size of batches to see how that affects the runtime.)""")

# ╔═╡ 2c9cf709-9bc8-48bf-9e19-db8cf7c8690b
md"""
Let's try adjusting the size of each batch of wavelength processes within one task.
"""

# ╔═╡ b4cbb43d-6e4c-4917-99a4-03d13e736144
chunksize_for_onmythreads = 4

# ╔═╡ fb063bc5-22bc-4b32-8fcb-5fbc4765c8b5
batchsize_for_ThreadsXmap = 4

# ╔═╡ 90c9d079-4bbc-4609-aa12-afa41a74b2fb
md"""
1e.  After specifying the size of each batch or chunk of work, did either the `OhMyThreads.tmap` and `ThreadsX.map` perform noticeably better than when using their default behavior?  How does the speed up using a batch or chunk size larger than 1 compare to the theoretical maximum speed-up factor?   
"""

# ╔═╡ 0edbb2db-4db8-4dc4-9a73-f7ff86e6f577
response_1e = missing  # md"Insert your response"

# ╔═╡ a944fdea-f41b-4a5f-95ac-e5f4074d4290
begin
    if !@isdefined(response_1e)
		var_not_defined(:response_1e)
    elseif ismissing(response_1e)
    	still_missing()
	end
end

# ╔═╡ eb57f7bb-1bff-471f-a599-d1d7d8f771ad
md"""

The results to the question above will depend on details like the type of CPU being used and the number of CPU cores and threads in use.  Before starting large calculations, it's good to test the effects of parameters like `basesize` or `chunksize` with the specific hardware and configuration parameters that you'll be using for your big runs.
"""

# ╔═╡ d43525da-e0a2-4d2f-9dbb-bf187eebf6c1
tip(md"""
## ''Embarassingly'' parallel is good

So far, we've demonstrated parallelizing a computation that can be easily broken into smaller tasks that do not need to communicate with each other.  This is often called an called *embarassingly parallel* computation.  Don't let the name mislead you.  While it could be embarrassing if a Computer Science graduate student tried to make a Ph.D. thesis out of parallelizing an embarassingly parallel problem, that doesn't mean that programmers shouldn't take advantage of opportunities to use embarrassingly parallel techniques when they can.  If you can parallelize your code using embarassingly parallel techniques, then you should almost always parallelize it that way, instead of (or at least before) trying to parallelize it at a finer grained level.

Next, we'll consider problems that do require some communication between tasks, but in a very structured manner.
""")

# ╔═╡ 547ad5ba-06ad-4707-a7ef-e444cf88ae53
md"""
# Reductions
Many common calculations can be formulated as a [**reduction operation**](https://en.wikipedia.org/wiki/Reduction_operator), where many inputs are transformed into one output.  Common examples would be `sum` or `maximum`.  One key property of reduction operations is that they are associative, meaning it's ok for the computer to change the order in which inputs are reduced.  (Thinking back to our lesson about floating point arithmetic, many operations aren't formally associative or commutative, but are still close enough that we're willing to let the computer reorder calculations.)

When we have multiple processors, the input can be divided into subsets and each processor reduces each subset separately.  Then each processor only needs to communicate one value of the variable being reduced to another processor, even if the input is quite large.  For some problems, reductions also reduce the amount of memory allocations necessary.
"""

# ╔═╡ 7ba35a63-ac61-434b-b759-95d505f62d9e
md"""
We'll explore different ways to perform reductions on an example problem where we calculate the mean squared error between the model and the model Doppler shifted by a velocity, $v$. First, let's write a vanilla serial version, where we first compute an array of squared residuals and pass that to the `sum` function.
"""

# ╔═╡ cee9c93d-cf7b-4da1-b4bb-b544b7cc104c
v = 10.0

# ╔═╡ 3ac01c04-52e3-497e-8c29-8c704e23ae39
md"## Serial loop with reduction"

# ╔═╡ 790377a7-1301-44a8-b300-418567737373
md"""
Now we'll write a version of the function using a serial for loop.  Note that we no longer need to allocate an output array, since `calc_mse_loop` only needs to return the reduced mean squared error and not the value of the spectrum at every wavelength.
"""

# ╔═╡ 8f56a866-a141-4275-9769-957ed5834afe
md"""
## Parallel loop with separate reduction
"""

# ╔═╡ cf2938cc-d3f0-4077-9262-3d51866df2cf
md"""
First, we'll try applying a parallel map followed by a parallel reduce.
"""

# ╔═╡ bd75a60b-ca34-4211-ac35-8325102cff68
md"""
**Q1f:** How did the performance of `calc_mse_ohmythreads_map_reduce_separate` compare to the serial versions above?  How does this ratio compare to the theoretical maximum speed-up factor?

"""

# ╔═╡ 5f379c7a-9713-45f7-9a5e-57b8197332c3
response_1f = missing # md"Insert your response"

# ╔═╡ 8d7c27d5-4a07-4ab4-9ece-94fdb7053f73
begin
    if !@isdefined(response_1f)
		var_not_defined(:response_1f)
    elseif ismissing(response_1f)
    	still_missing()
	end
end

# ╔═╡ 161ea6af-5661-44e1-ae40-1b581b636c25
md"""
## Parallel loop with simultaneous reduction 
Next, we'll use parallel loop macros to compute the mean squared error using multiple threads.  
When using [FLoops.jl](https://github.com/JuliaFolds/FLoops.jl), we need to use the `@floop` macro around the loop  *and* the `@reduce` macro to indicate which variables are part of the reduction.

When using [OhMyThreads.jl](), we need to use the `@tasks` macro around the loop and the `@set reducer` instructions inside the loop.  We also use the `@local` macro to specify that some variables can be allocated a single time per thread rather than once per iteration.
"""

# ╔═╡ 3183c6ac-5acd-4770-a638-c4c6ba3f7c4f
md"""
**Q1g:**  How did the performance of `calc_mse_flloop` or `calc_mse_ohmythreads` to the performance of the serial versions (e.g., `calc_mse_loop` or `calc_mse_broadcasted`)?   How does that compare to the theoretical maximum speed-up.
"""

# ╔═╡ 8e9b1e02-2bc0-49d2-b7ed-38de877ebe77
response_1g = missing  # md"Insert your response"

# ╔═╡ ba62f716-b1b5-4d11-91f2-ed121b48216c
begin
    if !@isdefined(response_1g)
		var_not_defined(:response_1g)
    elseif ismissing(response_1g)
    	still_missing()
	end
end

# ╔═╡ bbdd495c-f2c6-4264-a4e9-5083753eb410
md"""
One advantage of parallelizing your code with [FLoops.jl](https://juliafolds.github.io/FLoops.jl/dev/) is that it then becomes very easy to compare the performance of a calculation in serial and in parallel using different **[executors](https://juliafolds.github.io/FLoops.jl/dev/tutorials/parallel/#tutorials-executor)** that specify how the calculation should be implemented.  There are different parallel executor for shared-memory parallelism (via multi-threading this exercise), distributed-memory parallelism, and even for parallelizing code over GPUs (although there are some restrictions on what code can be run on the GPU, that we'll see in a [Lab 8](https://github.com/PsuAstro528/lab8)).
"""

# ╔═╡ 383aa611-e115-482e-873c-4487e53d457f
md"# Mapreduce

We can combine `map` and `reduce` into one function **`mapreduce`**.  There are opportunities for some increased efficiencies when merging the two, since the amount of communications between threads can be significantly decreased thanks to the reduction operator.  Mapreduce is a common, powerful and efficient programming pattern.  For example, we often want to evaluate a model for many input values, compare the results of the model to data and compute some statistic about how much the model and data differ.

In this exercise, we'll demonstrate using `mapreduce` for calculating the mean squared error between the model and the model Doppler shifted by a velocity, $v$.  First, we'll
"

# ╔═╡ 2c6fa743-3dec-417b-b05a-17bb52b5d39d
 md"## Mapreduce (serial)"

# ╔═╡ ac1ffdbf-de6f-48cd-af7c-99528ef26dc0
md"""
**Q1h:** How does the performance of `calc_mse_mapreduce` compare to the performance of `calc_mse_loop` (or `calc_mse_broadcasted`)?
"""

# ╔═╡ e331d501-71ed-4d93-8498-5c1193776865
response_1h = missing

# ╔═╡ ae47ef38-e8d0-40b9-9e61-3ab3ca7e7a49
md"## Parallel mapreduce"

# ╔═╡ aad94861-e2b3-417d-b640-b821e53adb23
md"""
The OhMyThreads and ThreadsX packages provide multi-threaded versions of mapreduce that we can easily drop in.
"""

# ╔═╡ f1c0321b-7811-42b1-9d0c-9c69f43d7e1a
md"""
Similar to before, we may be able to reduce the overhead associated with distributing work across threads by grouping the calculations into batches.  
"""

# ╔═╡ df044a68-605f-4347-832a-68090ee07950
mapreduce_batchsize = 16

# ╔═╡ 3f01d534-b01d-4ab4-b3cd-e809b02563a9
md"""
**Q1i:**  How did the performance of `calc_mse_mapreduce_threadsx` or `calc_mse_mapreduce_ohmythreads` compare to the performance of `calc_mse_loop`?  
Were the speed-up factors larger than when performing parallel map-type operations?  
Why?  
Can you explain why the batchsize had a bigger effect for the mapreduce calculations than for the map operations?
"""

# ╔═╡ d16adf94-72c3-480d-bd92-738e806068f8
response_1i = missing #= md"""
Insert your
multi-line
response
"""
=#

# ╔═╡ 56c5b496-a063-459a-8686-22fc70b6a214
begin
    if !@isdefined(response_1i)
		var_not_defined(:response_1i)
    elseif ismissing(response_1i)
    	still_missing()
	end
end

# ╔═╡ c4ff4add-ab3c-4585-900e-41f17e905ac5
md"""
**Q1j:**  Think about how you will parallelize your class project code.  The first parallelization typically uses a shared-memory model.  Which of these programming patterns would be a good fit for your project?  Can your project calculation be formulated as a `map` or `mapreduce` problem?  If not, then could it be implemented as a series of multiple maps/reductions/mapreduces?

Which of the parallel programming strategies are well-suited for your project?

After having worked through this lab, do you anticipate any barriers to applying one of these techniques to your project?

"""

# ╔═╡ ac18f1ca-0f60-4436-9d8a-797b3dfd8657
response_1j = missing  #= md"""
Insert your
multi-line
response
"""
=#

# ╔═╡ e8082779-143d-4562-81f3-d493679cf3c7
begin
    if !@isdefined(response_1j)
		var_not_defined(:response_1j)
    elseif ismissing(response_1j)
    	still_missing()
	end
end

# ╔═╡ bd77bc71-ffdf-4ba1-b1ee-6f2a69044e6f
begin
    σ_obs1 = 0.02*ones(size(lambdas))
    σ_obs2 = 0.02*ones(size(lambdas))
end;

# ╔═╡ 3b50062c-99c1-4f68-aabe-2d40d4ad7504
md"# Helper code"

# ╔═╡ d83a282e-cb2b-4837-bfd4-8404b3722e3a
WidthOverDocs()

# ╔═╡ c9cf6fb3-0146-42e6-aaae-24e97254c805
TableOfContents(aside=true)

# ╔═╡ 73358bcf-4129-46be-bef4-f623b11e245b
begin
	# Code for our model
	ModelSpectrum = @ingredients "./src/model_spectrum.jl"
	import .ModelSpectrum:AbstractSpectrum, SimulatedSpectrum, ConvolvedSpectrum, GaussianMixtureConvolutionKernel, doppler_shifted_spectrum
end

# ╔═╡ 4effbde2-2764-4c51-a9d0-a2db82f60862
"Create an object that provides a model for the raw spectrum (i.e., before entering the telescope)"
function make_spectrum_object(;lambda_min = 4500, lambda_max = 7500, flux_scale = 1.0,
        num_star_lines = 200, num_telluric_lines = 100, limit_line_effect = 10.0)

    continuum_param = flux_scale .* [1.0, 1e-5, -2e-8]

    star_line_locs = rand(Uniform(lambda_min,lambda_max),num_star_lines)
    star_line_widths = fill(1.0,num_star_lines)
    star_line_depths = rand(Uniform(0,1.0),num_star_lines)

    telluric_line_locs = rand(Uniform(lambda_min,lambda_max),num_telluric_lines)
    telluric_line_widths = fill(0.2,num_telluric_lines)
    telluric_line_depths = rand(Uniform(0,0.4),num_telluric_lines)

	SimulatedSpectrum(star_line_locs,star_line_widths,star_line_depths,telluric_line_locs,telluric_line_widths,telluric_line_depths,continuum_param=continuum_param,lambda_mid=0.5*(lambda_min+lambda_max),limit_line_effect=limit_line_effect)
end

# ╔═╡ 86b8dd31-1261-4fb9-bfd3-13f6f01e7790
# Create a functor (function-like object) that computes a model spectrum that we'll analyze below
raw_spectrum = make_spectrum_object(lambda_min=lambda_min,lambda_max=lambda_max)

# ╔═╡ 1c069610-4468-4d10-98f7-99662c26bdda
if true
	raw_spectrum.(lambdas)
	stats_broadcasted_serial_raw = @timed raw_spectrum.(lambdas)
	(;  time=stats_broadcasted_serial_raw.time, bytes=stats_broadcasted_serial_raw.bytes)
end

# ╔═╡ ca9c7d9e-e6cc-46cc-8a9b-ccda123591a2
if true
	map(raw_spectrum,lambdas)
	stats_map_serial_raw = @timed map(raw_spectrum,lambdas)
	(;  time=stats_map_serial_raw.time, bytes=stats_map_serial_raw.bytes)
end

# ╔═╡ 65398796-73ab-4d98-9851-3bb162ac8cbc
begin      # Create a model for the point spread function (PSF)
	psf_widths  = [0.5, 1.0, 2.0]
	psf_weights = [0.8, 0.15, 0.05]
	psf_model = GaussianMixtureConvolutionKernel(psf_widths,psf_weights)
end

# ╔═╡ 0aafec61-ff44-49e2-95e9-d3506ac6afa7
# Create a functor (function-like object) that computes a model for the convolution of the raw spectrum with the PSF model
conv_spectrum = ConvolvedSpectrum(raw_spectrum,psf_model)

# ╔═╡ dbf05374-1d89-4f30-b4b4-6cf57631f8b7
begin
	plot(lambdas,raw_spectrum.(lambdas),xlabel="λ", ylabel="Flux", label="Raw spectrum", legend=:bottomright)
	plot!(lambdas,conv_spectrum.(lambdas), label="Convolved spectrum")
end

# ╔═╡ f2b23082-98bc-4be1-bb6d-cac8facb8a46
let
	plt = plot(view(lambdas,idx_plot),raw_spectrum.(view(lambdas,idx_plot)),xlabel="λ", ylabel="Flux", label="Raw spectrum", legend=:bottomright)
	plot!(plt,view(lambdas,idx_plot),conv_spectrum.(view(lambdas,idx_plot)), label="Convolved spectrum")
	ylims!(plt,0,1.01)
end

# ╔═╡ 6ccce964-0439-4707-adf9-e171fd703609
if true
	result_spec_vec_serial =  conv_spectrum(lambdas)
	stats_spec_vec_serial = @timed conv_spectrum(lambdas)
	(;  time=stats_spec_vec_serial.time, bytes=stats_spec_vec_serial.bytes)
end

# ╔═╡ a172be44-1ac0-4bd8-a3d1-bac5666ab68e
if true
 	result_spec_serial_broadcast = conv_spectrum.(lambdas)
	stats_spec_serial_broadcast = @timed conv_spectrum.(lambdas)
	(;  time=stats_spec_serial_broadcast.time,
		bytes=stats_spec_serial_broadcast.bytes )
end

# ╔═╡ 215011e0-5977-43f8-bb65-83c09b3c07d8
if true
	result_spec_serial_map = map(conv_spectrum,lambdas)
	stats_spec_serial_map = @timed map(conv_spectrum,lambdas)
	(;  time=stats_spec_serial_map.time, bytes=stats_spec_serial_map.bytes )
end

# ╔═╡ 658f73c3-1e7a-47da-9130-06673f484ba1
if true
	ModelSpectrum.eval_spectrum_using_array(raw_spectrum,lambdas)
	stats_serial_raw = @timed ModelSpectrum.eval_spectrum_using_array(raw_spectrum,lambdas)
	(;  time=stats_serial_raw.time, bytes=stats_serial_raw.bytes)
end

# ╔═╡ 4b9a98ba-1731-4707-89a3-db3b5ac3a79b
function calc_spectrum_loop(x::AbstractArray, spectrum::T) where T<:AbstractSpectrum
    out = zeros(length(x))
    for i in 1:length(x)
        @inbounds out[i] = spectrum(x[i])
    end
    return out
end

# ╔═╡ 9941061a-ad42-46b0-9d0f-7584ebca7c62
if true
	result_spec_serial_loop = calc_spectrum_loop(lambdas,conv_spectrum)
	stats_spec_serial_loop = @timed calc_spectrum_loop(lambdas,conv_spectrum)
	(;  time=stats_spec_serial_loop.time,
		bytes=stats_spec_serial_loop.bytes )
end

# ╔═╡ 5acc645a-0d32-4f51-8aa6-063725b83fa8
if !ismissing(response_1c)
	result_spec_ohmythreads = tmap(conv_spectrum,lambdas)
	stats_spec_ohmythreads_map = @timed tmap(conv_spectrum,lambdas)
	(;  time=stats_spec_ohmythreads_map.time,
		bytes=stats_spec_ohmythreads_map.bytes,
		speedup=stats_spec_serial_loop.time/stats_spec_ohmythreads_map.time)
end

# ╔═╡ c7121d63-b1ff-4c38-8579-e1adbfef48ef
if !ismissing(response_1c)
	result_spec_ThreadsXmap = ThreadsX.map(conv_spectrum,lambdas)
	stats_spec_ThreadsXmap = @timed ThreadsX.map(conv_spectrum,lambdas)
	(;  time=stats_spec_ThreadsXmap.time,
		bytes=stats_spec_ThreadsXmap.bytes,
	speedup=stats_spec_serial_loop.time/stats_spec_ThreadsXmap.time)
end

# ╔═╡ af161709-46d2-4c73-b10d-90acb1e85189
if !ismissing(response_1d)
	tmap(conv_spectrum,lambdas; scheduler=:static, chunksize=chunksize_for_onmythreads)
	walltime_OhMyThreadsmap_static_batched = @elapsed tmap(conv_spectrum,lambdas; scheduler=:static, chunksize=chunksize_for_onmythreads)
	(;speedup=stats_spec_serial_loop.time/walltime_OhMyThreadsmap_static_batched)
end

# ╔═╡ 1e93928c-7f4c-4295-b755-4e5e16adbd8e
if !ismissing(response_1d)
	tmap(conv_spectrum,lambdas; scheduler=:dynamic, chunksize=chunksize_for_onmythreads)
	walltime_OhMyThreadsmap_dynamic = @elapsed tmap(conv_spectrum,lambdas; scheduler=:dynamic, chunksize=chunksize_for_onmythreads)
	(;speedup=stats_spec_serial_loop.time/walltime_OhMyThreadsmap_dynamic)
end

# ╔═╡ 0e9664ec-98d8-49d4-a376-24d4770c4c8f
if !ismissing(response_1d)
	ThreadsX.map(conv_spectrum,lambdas,basesize=batchsize_for_ThreadsXmap)
	walltime_ThreadsXmap_batched = @elapsed ThreadsX.map(conv_spectrum,lambdas,basesize=batchsize_for_ThreadsXmap)
	(;speedup=stats_spec_serial_loop.time/walltime_ThreadsXmap_batched)
end

# ╔═╡ e55c802d-7923-458f-af42-d951e82e029b
function calc_spectrum_threaded_for_loop(x::AbstractArray, spectrum::T) where T<:AbstractSpectrum
    out = zeros(length(x))
    Threads.@threads for i in 1:length(x)
        @inbounds out[i] = spectrum(x[i])
    end
    return out
end

# ╔═╡ b3a6004f-9d10-4582-832a-8917b701f2ad
if !ismissing(response_1c)
	result_spec_threaded_loop = calc_spectrum_threaded_for_loop(lambdas,conv_spectrum)
	stats_spec_threaded_loop = @timed calc_spectrum_threaded_for_loop(lambdas,conv_spectrum)
	(;  time=stats_spec_threaded_loop.time,
		bytes=stats_spec_threaded_loop.bytes, 
	 	speedup = stats_spec_serial_loop.time/stats_spec_threaded_loop.time  )
end

# ╔═╡ f272eab0-8a33-4161-bf9e-e378255631dd
function calc_spectrum_ohmythreads_foreach(x::AbstractArray, spectrum::T ) where { T<:AbstractSpectrum }
    out = zeros(length(x))
	OhMyThreads.tforeach(eachindex(out, x)) do i
           @inbounds out[i] = spectrum(x[i])
    end
    return out
end

# ╔═╡ d43540d3-beb9-45aa-98cd-1a77f6b8db10
if true
	result_spec_ohmythreads_foreach = calc_spectrum_ohmythreads_foreach(lambdas,conv_spectrum)
	stats_spec_ohmythreads_foreach = @timed calc_spectrum_ohmythreads_foreach(lambdas,conv_spectrum)
	(;  time=stats_spec_ohmythreads_foreach.time,
		bytes=stats_spec_ohmythreads_foreach.bytes,
		speedup=stats_spec_serial_loop.time/stats_spec_ohmythreads_foreach.time )
end

# ╔═╡ c65aa7b6-d85e-4efa-a2ee-1b615155796e
function calc_spectrum_threadsX_foreach(x::AbstractArray, spectrum::T ) where { T<:AbstractSpectrum }
    out = zeros(length(x))
	ThreadsX.foreach(eachindex(out, x)) do I
           @inbounds out[I] = spectrum(x[I])
    end
    return out
end

# ╔═╡ d1beea61-776f-4841-97e4-8d423ac22820
if true
	result_spec_threadsX_foreach = calc_spectrum_threadsX_foreach(lambdas,conv_spectrum)
	stats_spec_threadsX_foreach = @timed calc_spectrum_threadsX_foreach(lambdas,conv_spectrum)
	(;  time=stats_spec_threadsX_foreach.time,
		bytes=stats_spec_threadsX_foreach.bytes,
		speedup = stats_spec_serial_loop.time/stats_spec_threadsX_foreach.time)
end

# ╔═╡ 48b1fc18-eaaf-4f5e-9275-1e942dfbd643
function calc_spectrum_polyester_foreach(x::AbstractArray, spectrum::T ) where { T<:AbstractSpectrum }
    out = zeros(length(x))
	@batch for i in eachindex(out, x)
           @inbounds out[i] = spectrum(x[i])
    end
    return out
end

# ╔═╡ 6a09fa9c-f04e-466f-9746-beb2a7cfdfa3
if true
	result_spec_polyester_foreach = calc_spectrum_polyester_foreach(lambdas,conv_spectrum)
	stats_spec_polyester_foreach = @timed calc_spectrum_polyester_foreach(lambdas,conv_spectrum)
	(;  time=stats_spec_polyester_foreach.time,
		bytes=stats_spec_polyester_foreach.bytes,
		speedup=stats_spec_serial_loop.time/stats_spec_polyester_foreach.time)
end

# ╔═╡ 9b734e9c-f571-4a09-9744-221dcd55b4bf
function calc_spectrum_flloop(x::AbstractArray, spectrum::T; basesize::Integer=div(Threads.nthreads(),length(x)),  ex::FLoops.Executor = ThreadedEx(basesize=basesize) ) where { T<:AbstractSpectrum }
    out = zeros(length(x))
     @floop ex for i in eachindex(out, x)
        @inbounds out[i] = spectrum(x[i])
    end
    return out
end

# ╔═╡ c2c68b93-1cd4-4a38-9dd9-47ce2d591907
if true
	result_spec_flloop = calc_spectrum_flloop(lambdas,conv_spectrum)
	stats_spec_flloop = @timed calc_spectrum_flloop(lambdas,conv_spectrum) 
	(;  time=stats_spec_flloop.time, bytes=stats_spec_flloop.bytes,
	speedup=stats_spec_serial_loop.time/stats_spec_flloop.time)
end

# ╔═╡ 398ba928-899f-4843-ad58-25df67c81ffe
function calc_mse_broadcasted(lambdas::AbstractArray, spec1::AbstractSpectrum, spec2::AbstractSpectrum, v::Number)
	c = ModelSpectrum.speed_of_light
	z = v/c
	spec2_shifted = doppler_shifted_spectrum(spec2,z)
	abs2_diff_spectra = abs2.(spec1.(lambdas) .- spec2_shifted.(lambdas))
	mse = sum(abs2_diff_spectra)
	mse /= length(lambdas)
end

# ╔═╡ 9f8667f3-4104-4642-b2d9-a6d12a6fa5d3
begin
	result_mse_broadcasted = calc_mse_broadcasted(lambdas,conv_spectrum,conv_spectrum,v)
	stats_mse_broadcasted = @timed calc_mse_broadcasted(lambdas,conv_spectrum,conv_spectrum,v)
	(;  time=stats_mse_broadcasted.time, bytes=stats_mse_broadcasted.bytes)
end

# ╔═╡ 536fe0c4-567c-4bda-8c95-347f183c007b
function calc_mse_loop(lambdas::AbstractArray, spec1::AbstractSpectrum, spec2::AbstractSpectrum,  v::Number; basesize::Integer=div(Threads.nthreads(),length(lambdas)),  ex::FLoops.Executor = ThreadedEx(basesize=basesize))
	c = ModelSpectrum.speed_of_light
	z = v/c
	spec2_shifted = doppler_shifted_spectrum(spec2,z)
	eltype_spec1 = first(Base.return_types(spec1,(eltype(lambdas),)))
    eltype_spec2_shifted = first(Base.return_types(spec2_shifted,(eltype(lambdas),)))
	mse = zero(promote_type(eltype_spec1,eltype_spec2_shifted))
	for i in eachindex(lambdas)
        l = lambdas[i]
		flux1 = spec1(l)
        flux2 = spec2_shifted(l)
		mse += (flux1-flux2)^2
    end
	mse /= length(lambdas)
    return mse
end

# ╔═╡ db96a6c9-8352-47f3-8319-9c373aa03ff4
if true
	result_mse_loop = calc_mse_loop(lambdas,conv_spectrum,conv_spectrum,v)
	stats_mse_loop = @timed calc_mse_loop(lambdas,conv_spectrum,conv_spectrum,v)
	(;  time=stats_mse_loop.time, bytes=stats_mse_loop.bytes )
end

# ╔═╡ 6e52c719-e9fc-478a-9709-49e250a27d6b
md"""
As expected, the $(floor(Int,stats_mse_loop.bytes//1024^2)) MB allocated when we compute the mean squared error between *two* simulated spectra is very nearly twice the $(floor(Int,stats_spec_serial_loop.bytes//1024^2)) MB allocated by the serial for loop to compute the one spectrum at each wavelength.
"""

# ╔═╡ e36cda69-d300-4156-9bef-a372f94306d9
md"""
Similarly, it's likely that the wall time for the serial loop to compute the mean squared error $(round(stats_mse_loop.time,digits=3)) sec
is nearly twice that of the serial loop to compute one spectrum $(round(stats_spec_serial_loop.time,digits=3)) sec.
So far it doesn't seem particularly interesting.
"""

# ╔═╡ 01418f56-432c-458a-82ea-f2a5c75eb405
function calc_mse_ohmythreads_map_reduce_separate(lambdas::AbstractArray, spec1::AbstractSpectrum, spec2::AbstractSpectrum, v::Number)
	c = ModelSpectrum.speed_of_light
	z = v/c
	spec2_shifted = doppler_shifted_spectrum(spec2,z)
	s1 = tmap(spec1,lambdas)
	s2 = tmap(spec2_shifted,lambdas)
	sa_of_spectra = StructArray(; s1=s1, s2=s2 )
	abs2_diff_spectra = tmap(x->abs2(x.s1 - x.s2),sa_of_spectra)
	mse = treduce(+,abs2_diff_spectra)
	mse /= length(lambdas)
end
	

# ╔═╡ d3995cc5-f804-4591-926e-b358a8068221
if true
	result_mse_ohmythreads_map_reduce_separate = calc_mse_ohmythreads_map_reduce_separate(lambdas,conv_spectrum,conv_spectrum,v)
	stats_mse_ohmythreads_map_reduce_separate = @timed calc_mse_ohmythreads_map_reduce_separate(lambdas,conv_spectrum,conv_spectrum,v)
	(;  time=stats_mse_ohmythreads_map_reduce_separate.time, bytes=stats_mse_ohmythreads_map_reduce_separate.bytes,
	speedup=stats_mse_loop.time/stats_mse_ohmythreads_map_reduce_separate.time)
end

# ╔═╡ cda32841-0e89-4019-abdc-cf7b0377aa48
hint(md"The speed-up was $(round( stats_mse_loop.time / stats_mse_ohmythreads_map_reduce_separate.time /  Threads.nthreads(), digits=3)) of the maximum theoretical speed-up.")

# ╔═╡ 293ad084-6d6a-4401-9819-53a24646d2c9
function calc_mse_ohmythreads(lambdas::AbstractArray, spec1::AbstractSpectrum, spec2::AbstractSpectrum,  v::Number)
	c = ModelSpectrum.speed_of_light
	z = v/c
	spec2_shifted = doppler_shifted_spectrum(spec2,z)
	tmp1 = spec1(first(lambdas))
    tmp2 = spec2_shifted(first(lambdas))
	#mse = zero(promote_type(typeof(tmp1),typeof(tmp2)))
	mse = OhMyThreads.@tasks for i in eachindex(lambdas)
        @set reducer = +
		@local begin
			l = zero(eltype(lambdas))
			diffsq = zero(promote_type(typeof(tmp1),typeof(tmp2)))
		end
		l = lambdas[i]
		diffsq = (spec1(l)-spec2_shifted(l))^2
    end
	mse /= length(lambdas)
    return mse
end

# ╔═╡ 3231b010-718a-4863-be43-1f0326451e96
if !ismissing(response_1f)
	result_mse_ohmythreads = calc_mse_ohmythreads(lambdas,conv_spectrum,conv_spectrum,v)
	stats_mse_ohmythreads = @timed calc_mse_ohmythreads(lambdas,conv_spectrum,conv_spectrum,v)
	(;  time=stats_mse_ohmythreads.time, bytes=stats_mse_ohmythreads.bytes,
	speedup=stats_mse_loop.time/stats_mse_ohmythreads.time)
end

# ╔═╡ 1c1ccc51-e32a-4881-b892-095d2be55916
function calc_mse_flloop(lambdas::AbstractArray, spec1::AbstractSpectrum, spec2::AbstractSpectrum,  v::Number; basesize::Integer=div(Threads.nthreads(),length(lambdas)), ex::FLoops.Executor = ThreadedEx(basesize=basesize))
	c = ModelSpectrum.speed_of_light
	z = v/c
	spec2_shifted = doppler_shifted_spectrum(spec2,z)
	tmp1 = spec1(first(lambdas))
    tmp2 = spec2_shifted(first(lambdas))
	mse = zero(promote_type(typeof(tmp1),typeof(tmp2)))
	@floop ex for i in eachindex(lambdas)
        l = lambdas[i]
		flux1 = spec1(l)
        flux2 = spec2_shifted(l)
		@reduce(mse += (flux1-flux2)^2)
    end
	mse /= length(lambdas)
    return mse
end

# ╔═╡ b0e08212-7e12-4d54-846f-5b0863c37236
if !ismissing(response_1f)
	result_mse_flloop = calc_mse_flloop(lambdas,conv_spectrum,conv_spectrum,v)
	stats_mse_flloop = @timed calc_mse_flloop(lambdas,conv_spectrum,conv_spectrum,v)
	(;  time=stats_mse_flloop.time, bytes=stats_mse_flloop.bytes,
	speedup=stats_mse_loop.time/stats_mse_flloop.time)
end

# ╔═╡ bad94aca-f77e-417e-be32-0840a3e5c958
if !ismissing(response_1f)
	hint(md"The speed-ups were $(round(stats_mse_loop.time / stats_mse_flloop.time / Threads.nthreads(), digits=3)) and $(round(stats_mse_loop.time /stats_mse_ohmythreads.time / Threads.nthreads(), digits=3)) of the maximum theoretical speed-up.")
end

# ╔═╡ 17659ddb-d4e0-4a4b-b34c-8ac52d5dad45
function calc_mse_mapreduce(lambdas::AbstractArray, spec1::AbstractSpectrum, spec2::AbstractSpectrum, v::Number)
	c = ModelSpectrum.speed_of_light
	z = v/c
	spec2_shifted = doppler_shifted_spectrum(spec2,z)
	mse = mapreduce(λ->(spec1.(λ) .- spec2_shifted.(λ)).^2, +, lambdas)
	mse /= length(lambdas)
end

# ╔═╡ 2ef9e7e0-c856-4ef3-a08f-89817fc5fd60
begin
	result_mse_mapreduce_serial = calc_mse_mapreduce(lambdas,conv_spectrum,conv_spectrum,v)
	stats_mse_mapreduce_serial = @timed calc_mse_mapreduce(lambdas, conv_spectrum,conv_spectrum,v)
	(;  time=stats_mse_mapreduce_serial.time, bytes=stats_mse_mapreduce_serial.bytes)
end

# ╔═╡ ab886349-5f3f-45e9-a6e1-a81fdfafa72f
function calc_mse_mapreduce_ohmythreads(lambdas::AbstractArray, spec1::AbstractSpectrum, spec2::AbstractSpectrum,  v::Number; basesize::Integer = 1)
	c = ModelSpectrum.speed_of_light
	z = v/c
	spec2_shifted = doppler_shifted_spectrum(spec2,z)
	mse = tmapreduce(λ->(spec1.(λ) .- spec2_shifted.(λ)).^2, +, lambdas, chunksize=basesize)
	mse /= length(lambdas)
end

# ╔═╡ 19052549-3c5d-4b49-b708-05eac0a2a0ac
begin
	result_mse_mapreduce_ohmythreads = calc_mse_mapreduce_ohmythreads(lambdas,conv_spectrum,conv_spectrum,v)
	stats_mse_mapreduce_ohmythreads = @timed calc_mse_mapreduce_ohmythreads(lambdas,conv_spectrum,conv_spectrum,v)
	(;  time=stats_mse_mapreduce_ohmythreads.time, bytes=stats_mse_mapreduce_ohmythreads.bytes,
	speedup=stats_mse_loop.time/stats_mse_mapreduce_ohmythreads.time )

end

# ╔═╡ 7eccc74d-9a49-44d9-9e43-cbb3c8ad7ce5
begin
	result_mse_mapreduce_ohmythreads_batched = calc_mse_mapreduce_ohmythreads(lambdas,conv_spectrum,conv_spectrum,v; basesize=mapreduce_batchsize)
	stats_mse_mapreduce_ohmythreads_batched = @timed calc_mse_mapreduce_ohmythreads(lambdas,conv_spectrum,conv_spectrum,v; basesize=mapreduce_batchsize)
	(;  time=stats_mse_mapreduce_ohmythreads_batched.time, bytes=stats_mse_mapreduce_ohmythreads_batched.bytes,
	speedup=stats_mse_loop.time/stats_mse_mapreduce_ohmythreads_batched.time)
end

# ╔═╡ 1778899b-8f05-4b1f-acb5-32af1ace08ee
function calc_mse_mapreduce_threadsx(lambdas::AbstractArray, spec1::AbstractSpectrum, spec2::AbstractSpectrum,  v::Number; basesize::Integer = 1)
	c = ModelSpectrum.speed_of_light
	z = v/c
	spec2_shifted = doppler_shifted_spectrum(spec2,z)
	mse = ThreadsX.mapreduce(λ->(spec1.(λ) .- spec2_shifted.(λ)).^2, +, lambdas, basesize=basesize)
	mse /= length(lambdas)
end

# ╔═╡ 9e78bfc1-fb4e-4626-b387-c2f83bed6ef0
begin
	result_mse_mapreduce_threadsx = calc_mse_mapreduce_threadsx(lambdas,conv_spectrum,conv_spectrum,v)
	stats_mse_mapreduce_threadsx = @timed calc_mse_mapreduce_threadsx(lambdas,conv_spectrum,conv_spectrum,v)
	(;  time=stats_mse_mapreduce_threadsx.time, bytes=stats_mse_mapreduce_threadsx.bytes,
	speedup=stats_mse_loop.time/stats_mse_mapreduce_threadsx.time )

end

# ╔═╡ a661d895-d3d7-4e96-a08f-55b125ed1d40
begin
	result_mse_mapreduce_threadsx_batched = calc_mse_mapreduce_threadsx(lambdas,conv_spectrum,conv_spectrum,v; basesize=mapreduce_batchsize)
	stats_mse_mapreduce_threadsx_batched = @timed calc_mse_mapreduce_threadsx(lambdas,conv_spectrum,conv_spectrum,v; basesize=mapreduce_batchsize)
	(;  time=stats_mse_mapreduce_threadsx_batched.time, bytes=stats_mse_mapreduce_threadsx_batched.bytes,
	speedup=stats_mse_loop.time/stats_mse_mapreduce_threadsx_batched.time)
end

# ╔═╡ 87df5b25-0d2f-4f81-80f1-aaf6c9f89ce3
# For response_1k edit the following function:
function calc_χ²_my_way(lambdas::AbstractArray, spec1::AbstractSpectrum, spec2::AbstractSpectrum, σ1::AbstractArray, σ2::AbstractArray, v::Number; #= any optional parameters? =# )
    # INSERT YOUR CODE HERE
    return missing
end

# ╔═╡ 4dec4888-08db-4965-b27a-fc44f316b529
begin
    if !@isdefined(calc_χ²_my_way)
		func_not_defined(:calc_χ²_my_way)
    elseif ismissing(calc_χ²_my_way(lambdas,conv_spectrum, conv_spectrum, σ_obs1, σ_obs2, 0.0))
    	still_missing()
	else
		md"I've provided some tests below to help you recognize if your parallelized version is working well."
	end
end

# ╔═╡ 6f411bcc-7084-43c3-a88b-b56ba77b5732
begin
    calc_χ²_my_way, lambdas, conv_spectrum, σ_obs1, σ_obs2
    @test abs(calc_χ²_my_way(lambdas,conv_spectrum, conv_spectrum, σ_obs1, σ_obs2, 0.0 )) < 1e-8
end

# ╔═╡ a9601654-8263-425e-8d8f-c5bbeacbbe06
begin
function calc_χ²_loop(lambdas::AbstractArray, spec1::AbstractSpectrum, spec2::AbstractSpectrum, σ1::AbstractArray, σ2::AbstractArray, v::Number )
    @assert size(lambdas) == size(σ1) == size(σ2)
    c = ModelSpectrum.speed_of_light
    z = v/c
    spec2_shifted = doppler_shifted_spectrum(spec2,z)
	eltype_spec1 = first(Base.return_types(spec1,(eltype(lambdas),)))
    eltype_spec2_shifted = first(Base.return_types(spec2_shifted,(eltype(lambdas),)))
    χ² = zero(promote_type(eltype_spec1,eltype_spec2_shifted))
    for i in eachindex(lambdas)
        @inbounds l = lambdas[i]
        flux1 = spec1(l)
        flux2 = spec2_shifted(l)
        @inbounds χ² += (flux1-flux2)^2/(σ1[i]^2+σ2[i]^2)
    end
    return χ²
end
	# for making urls to this cell
	linkto_calc_χ²_loop = "#" * (PlutoRunner.currently_running_cell_id[] |> string) 
end

# ╔═╡ 8737797c-6563-4513-a5fc-fde9681b4c63
Markdown.parse("""
**Q1k:**  Before parallelizing your project code for shared memory, it may be good to get some practice parallelizing a simple function very similar to what's already been done above.  Try parallelizing the function `calc_χ²` by writing a function `calc_χ²_my_way` in the cell below.   You can parallelize the calculation of χ² using any one of the parallelization strategies demonstrated above.  I'd suggest trying to use the one that you plan to use for your project.  Feel free to refer to the serial function [`calc_χ²` at the bottom of the notebook]($linkto_calc_χ²_loop).
""")

# ╔═╡ 3c5ee822-b938-4848-b2b0-f0de2e65b4db
begin
    calc_χ²_my_way, lambdas, conv_spectrum, σ_obs1, σ_obs2
    @test calc_χ²_my_way(lambdas,conv_spectrum, conv_spectrum, σ_obs1, σ_obs2, 10.0 ) ≈ calc_χ²_loop(lambdas,conv_spectrum, conv_spectrum, σ_obs1, σ_obs2, 10.0 )
end


# ╔═╡ 35b1b5b1-2a23-44bb-bc19-805393d18d8a
md"""
# Extra information about your hardware
"""

# ╔═╡ ec08aa84-3f63-441a-a31c-85b7a82412d1
Hwloc.topology_info()

# ╔═╡ 21b3c52e-533f-488f-a2eb-602600b66738
Hwloc.cachesize()

# ╔═╡ e38e0b91-dbd3-4cc6-87ac-add4953411d1
Hwloc.cachelinesize()

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CpuId = "adafc99b-e345-5852-983c-f28acb93d879"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
FLoops = "cc61a311-1640-44b5-9fba-1b764f453329"
Hwloc = "0e44f5e4-bd66-52a0-8798-143a42290a1d"
OhMyThreads = "67456a42-1dca-4109-a031-0a68de7e3ad5"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoLinks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoTest = "cb4044da-4d16-4ffa-a6a3-8cad7f73ebdc"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Polyester = "f517fe37-dbe3-4b94-8317-1923a5111588"
QuadGK = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
ThreadsX = "ac1d9e8a-700a-412c-b207-f0111f4b6c0d"

[compat]
BenchmarkTools = "~1.6.0"
CpuId = "~0.3.1"
Distributions = "~0.25.120"
FLoops = "~0.2.2"
Hwloc = "~3.3.0"
OhMyThreads = "~0.8.3"
Plots = "~1.40.19"
PlutoLinks = "~0.1.6"
PlutoTeachingTools = "~0.4.5"
PlutoTest = "~0.2.2"
PlutoUI = "~0.7.71"
Polyester = "~0.7.18"
QuadGK = "~2.11.2"
StaticArrays = "~1.9.14"
StructArrays = "~0.7.1"
ThreadsX = "~0.1.12"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.2"
manifest_format = "2.0"
project_hash = "92000de1c0fc1a2c8e82fbf1ba77ef29c0198c7d"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "3b86719127f50670efe356bc11073d84b4ed7a5d"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.42"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "f7817e2e585aa6d924fd714df1e2a84be7896c60"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.3.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgCheck]]
git-tree-sha1 = "f9e9a66c9b7be1ad7372bbd9b062d9230c30c5ce"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "dbd8c3bbbdbb5c2778f85f4422c39960eac65a42"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.20.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceMetalExt = "Metal"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "26f41e1df02c330c4fa1e98d4aa2168fdafc9b1f"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.4"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "e38fbc49a620f5d0b660d7f543db1009fe0f8336"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "f21cfd4950cb9f0587d5067e69405ad2acd27b87"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.6"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Preferences", "Static"]
git-tree-sha1 = "f3a21d7fc84ba618a779d1ed2fcca2e682865bab"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.7"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

[[deps.ChunkSplitters]]
git-tree-sha1 = "63a3903063d035260f0f6eab00f517471c5dc784"
uuid = "ae650224-84b6-46f8-82ea-d812ca08434e"
version = "3.1.2"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "05ba0d07cd4fd8b7a39541e31a7b0254704ea581"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.13"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "062c5e1a5bf6ada13db96a4ae4749a4c2234f521"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.9"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "a656525c8b46aa6a1c76891552ed5381bb32ae7b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.30.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "0037835448781bb46feb39866934e243886d756a"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "76b3b7c3925d943edf158ddb7f693ba54eb297a5"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.0"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3e6d038b77f22791b8e3472b7c633acea1ecac06"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.120"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7bb1361afdb33c7f2b085aa49ea8fe1b0fb14e58"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.1+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "83dc665d0312b41367b7263e8a4d172eac1897f4"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.4"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "3a948313e7a41eb1db7a1e733e6335f17b4ab3c4"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "7.1.1+0"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "0a2e5873e9a5f54abb06418d57a8df689336a660"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.2"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "1828eb7275491981fa5f1752a5e126e8f26f8741"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.17"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "27299071cc29e409488ada41ec7643e0ab19091f"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.17+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "35fbd0cefb04a516104b8e183ce0df11b70a3f1a"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.84.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "ed5e9c58612c4e081aecdb6e1a479e18462e041e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.Hwloc]]
deps = ["CEnum", "Hwloc_jll", "Printf"]
git-tree-sha1 = "6a3d80f31ff87bc94ab22a7b8ec2f263f9a6a583"
uuid = "0e44f5e4-bd66-52a0-8798-143a42290a1d"
version = "3.3.0"

    [deps.Hwloc.extensions]
    HwlocTrees = "AbstractTrees"

    [deps.Hwloc.weakdeps]
    AbstractTrees = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dd3b49277ec2bb2c6b94eb1604d4d0616016f7a6"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.11.2+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e95866623950267c1e4878846f848d94810de475"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.2+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "c47892541d03e5dc63467f8964c9f2b415dfe718"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.46"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "52e1296ebbde0db845b356abbbe67fb82a0a116c"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.9"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "a9eaadb366f5493a5654e843864c13d8b107548c"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.17"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "706dfd3c0dd56ca090e86884db6eda70fa7dd4af"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.1+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d3c8af829abaeba27181db4acb485b18d15d89c6"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.1+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "39240b5f66956acfa462d7fe12efe08e26d6d70d"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.2.2"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OhMyThreads]]
deps = ["BangBang", "ChunkSplitters", "ScopedValues", "StableTasks", "TaskLocalValues"]
git-tree-sha1 = "e0a1a8b92f6c6538b2763196f66417dddb54ac0c"
uuid = "67456a42-1dca-4109-a031-0a68de7e3ad5"
version = "0.8.3"
weakdeps = ["Markdown"]

    [deps.OhMyThreads.extensions]
    MarkdownExt = "Markdown"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "f1a7e086c677df53e064e0fdd2c9d0b0833e3f6e"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.5.0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2ae7d4ddec2e13ad3bddf5c0796f7547cf682391"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.2+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c392fc5dd032381919e3b22dd32d6443760ce7ea"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.5.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f07c06228a1c670ae4c87d1276b92c7c597fdda0"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.35"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "275a9a6d85dc86c24d03d1837a0010226a96f540"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.3+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "0c5a5b7e440c008fe31416a3ac9e0d2057c81106"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.19"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoUI"]
git-tree-sha1 = "85778cdf2bed372008e6646c64340460764a5b85"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.4.5"

[[deps.PlutoTest]]
deps = ["HypertextLiteral", "InteractiveUtils", "Markdown", "Test"]
git-tree-sha1 = "17aa9b81106e661cffa1c4c36c17ee1c50a86eda"
uuid = "cb4044da-4d16-4ffa-a6a3-8cad7f73ebdc"
version = "0.2.2"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "8329a3a4f75e178c11c1ce2342778bcbbbfa7e3c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.71"

[[deps.Polyester]]
deps = ["ArrayInterface", "BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "ManualMemory", "PolyesterWeave", "Static", "StaticArrayInterface", "StrideArraysCore", "ThreadingUtilities"]
git-tree-sha1 = "6f7cd22a802094d239824c57d94c8e2d0f7cfc7d"
uuid = "f517fe37-dbe3-4b94-8317-1923a5111588"
version = "0.7.18"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "645bed98cd47f72f67316fd42fc47dee771aefcd"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "eb38d376097f47316fe089fc62cb7c6d85383a52"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.8.2+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "da7adf145cce0d44e892626e647f9dcbe9cb3e10"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.8.2+1"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "9eca9fc3fe515d619ce004c83c31ffd3f85c7ccf"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.8.2+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "e1d5e16d0f65762396f9ca4644a5f4ddab8d452b"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.8.2+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Referenceables]]
deps = ["Adapt"]
git-tree-sha1 = "02d31ad62838181c1a3a5fd23a1ce5914a643601"
uuid = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"
version = "0.1.3"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Revise]]
deps = ["CodeTracking", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "9bb80533cb9769933954ea4ffbecb3025a783198"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.7.2"
weakdeps = ["Distributed"]

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "c3b2323466378a2ba15bea4b2f73b081e022f473"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.5.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "41852b8679f78c8d8961eeadc8f62cef861a52e3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "95af145932c2ed859b63329952ce8d633719f091"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.3"

[[deps.StableTasks]]
git-tree-sha1 = "c4f6610f85cb965bee5bfafa64cbeeda55a4e0b2"
uuid = "91464d47-22a1-43fe-8b7f-2d57ee82463f"
version = "0.1.7"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools"]
git-tree-sha1 = "f737d444cb0ad07e61b3c1bef8eb91203c321eff"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.2.0"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Static"]
git-tree-sha1 = "96381d50f1ce85f2663584c8e886a6ca97e60554"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.8.0"

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

    [deps.StaticArrayInterface.weakdeps]
    OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "cbea8a6bd7bed51b1619658dec70035e07b8502f"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.14"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2c962245732371acd51700dbb268af311bddd719"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.6"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "8e45cecc66f3b42633b8ce14d431e8e57a3e242e"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StrideArraysCore]]
deps = ["ArrayInterface", "CloseOpenIntervals", "IfElse", "LayoutPointers", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface", "ThreadingUtilities"]
git-tree-sha1 = "83151ba8065a73f53ca2ae98bc7274d817aa30f2"
uuid = "7792a7ef-975c-4747-a70f-980b88e8d1da"
version = "0.5.8"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "8ad2e38cbb812e29348719cc63580ec1dfeb9de4"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.1"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TaskLocalValues]]
git-tree-sha1 = "67e469338d9ce74fc578f7db1736a74d93a49eb8"
uuid = "ed4db957-447d-4319-bfb6-7fa9ae7ecf34"
version = "0.1.3"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "d969183d3d244b6c33796b5ed01ab97328f2db85"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.5"

[[deps.ThreadsX]]
deps = ["Accessors", "ArgCheck", "BangBang", "ConstructionBase", "InitialValues", "MicroCollections", "Referenceables", "SplittablesBase", "Transducers"]
git-tree-sha1 = "70bd8244f4834d46c3d68bd09e7792d8f571ef04"
uuid = "ac1d9e8a-700a-412c-b207-f0111f4b6c0d"
version = "0.1.12"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "SplittablesBase", "Tables"]
git-tree-sha1 = "7deeab4ff96b85c5f72c824cae53a1398da3d1cb"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.84"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "372b90fe551c019541fafc6ff034199dc19c8436"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.12"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "6258d453843c466d84c17a58732dda5deeb8d3af"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.24.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    PrintfExt = "Printf"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"
    Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "af305cc62419f9bd61b6644d19170a4d258c7967"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.7.0"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "9caba99d38404b285db8801d5c45ef4f4f425a6d"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.1+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a5bc75478d323358a90dc36766f3c99ba7feb024"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.6+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "aff463c82a773cb86061bce8d53a0d976854923e"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.5+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "e3150c7400c41e207012b41659591f083f3ef795"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.3+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "c5bf2dad6a03dfef57ea0a170a1fe493601603f2"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.5+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "00af7ebdc563c9217ecc67776d1bbf037dbcebf4"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.44.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4bba74fa59ab0755167ad24f98800fe5d727175b"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.12.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "07b6a107d926093898e82b3b1db657ebe33134ec"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.50+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "fbf139bce07a534df0e699dbb5f5cc9346f95cc1"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.9.2+0"
"""

# ╔═╡ Cell order:
# ╟─85aad005-eac0-4f71-a32c-c8361c31813b
# ╟─bdf61711-36e0-40d5-b0c5-3bac20a25aa3
# ╟─629442ba-a968-4e35-a7cb-d42a0a8783b4
# ╟─b133064a-d02e-4a71-88d7-e430b80d24b1
# ╟─571cab3f-771e-4464-959e-f351194049e2
# ╠═0c775b35-702e-4664-bd23-7557e4e189f4
# ╟─3059f3c2-cabf-4e20-adaa-9b6d0c07184f
# ╟─4fa907d0-c556-45df-8056-72041edcf430
# ╠═73e5e40a-1e59-41ed-a48d-7fb99f5a6755
# ╠═f97f1815-50a2-46a9-ac20-e4a3e34d898c
# ╟─53da8d7a-8620-4fe5-81ba-f615d2d4ed2a
# ╟─8df4e8d2-b955-424d-a7ec-264ce2b0e506
# ╟─826d2312-0803-4c36-bb72-df6d8241910c
# ╟─f76f329a-8dde-4790-96f2-ade735643aeb
# ╟─0e4d7808-47e2-4740-ab93-5d3973eecaa8
# ╟─410b6052-6d6d-4fd8-844e-8941845d8d90
# ╟─8a50e9fa-031c-4912-8a2d-466e6a9a9935
# ╟─7df5fc86-889f-4a5e-ac2b-8c6f68d7c32e
# ╠═cc1418c8-3261-4c70-bc19-2921695570a6
# ╠═7f724449-e90e-4f8b-b13c-9640a498893c
# ╠═c85e51b2-2d3d-46a2-8f3f-03b289cab288
# ╟─907766c5-f084-4ddc-bb52-336cb037d521
# ╠═0bcde4df-1e31-4774-a31f-bd451bb6f758
# ╟─c41d65e3-ea35-4f97-90a1-bfeaeaf927ad
# ╟─6e617a7c-a640-4cb3-9451-28a0036d8fdc
# ╟─5e6c430a-cd2f-4169-a5c7-a92acef813ac
# ╟─c31cf36c-21ec-46f1-96aa-b014ff094f8a
# ╠═4effbde2-2764-4c51-a9d0-a2db82f60862
# ╟─7026e51d-c3e4-4503-9f35-71074b0c2f1a
# ╠═ad302f2b-69dc-4559-ba12-d7fb2e8e689e
# ╠═86b8dd31-1261-4fb9-bfd3-13f6f01e7790
# ╟─16ad0225-c7d6-455b-8eb0-3e93c9f9f91a
# ╠═65398796-73ab-4d98-9851-3bb162ac8cbc
# ╠═0aafec61-ff44-49e2-95e9-d3506ac6afa7
# ╟─324a9a25-1ec4-4dc2-a7ca-e0f1f56dbf66
# ╠═52127f57-9a07-451a-bb24-c1f3c5581f0a
# ╟─dbf05374-1d89-4f30-b4b4-6cf57631f8b7
# ╟─75948469-1347-45e2-9281-f366b41d0e04
# ╟─f2b23082-98bc-4be1-bb6d-cac8facb8a46
# ╟─4d1cf57f-b394-4f37-98c3-0d765f4ee635
# ╟─cddd761a-f051-4338-9e40-d35e050060d3
# ╟─ee96411d-e3fa-442b-b0fe-10d6ede37b6a
# ╟─b92aad2e-8a3b-4edf-ae7e-6e3cff6eead4
# ╟─e5f9fa06-9fbb-40a8-92de-71523775d257
# ╟─b195ebd2-9584-40b8-ae3e-6d9ce88b5398
# ╠═658f73c3-1e7a-47da-9130-06673f484ba1
# ╠═1c069610-4468-4d10-98f7-99662c26bdda
# ╟─d6d3a2d1-241e-44c1-a11b-5bfb2b3c5f4b
# ╟─0344a74d-456b-44f0-84dc-c2fdbd41a379
# ╠═6ccce964-0439-4707-adf9-e171fd703609
# ╠═a172be44-1ac0-4bd8-a3d1-bac5666ab68e
# ╟─51adffd7-8fb6-4ed2-8510-303a37d6efc3
# ╟─71d943e3-761a-4337-b412-b0b768483bc2
# ╟─db1583f4-61cb-43e0-9326-d6c15d8fad5a
# ╠═ca9c7d9e-e6cc-46cc-8a9b-ccda123591a2
# ╠═215011e0-5977-43f8-bb65-83c09b3c07d8
# ╟─f108d26b-6c75-4eb6-9e88-a60ec038a73c
# ╟─21f305db-24e1-47d1-b1f4-be04ca91780e
# ╟─e71cede9-382e-47e2-953a-2fa96ed50002
# ╟─4d54b6a7-3fc0-4c63-8a9d-d683aa4ecefe
# ╟─a44a3478-541d-40d6-9d99-04b918c16bfb
# ╠═4b9a98ba-1731-4707-89a3-db3b5ac3a79b
# ╠═9941061a-ad42-46b0-9d0f-7584ebca7c62
# ╟─96914ff8-56c8-4cc8-96bc-fd3d13f7e4ce
# ╟─32685a28-54d9-4c0d-8940-e82843d2cab2
# ╟─3717d201-0bc3-4e3c-8ecd-d835e58f6821
# ╟─04bcafcd-1d2f-4ce5-893f-7ec5bb05f9ed
# ╠═ca8ceb27-86ea-4b90-a1ae-86d794c9fc98
# ╟─8b61fca1-2f89-4c74-bca8-c6cc70ba62ad
# ╟─bd81357b-c461-458e-801c-610893dd5ea1
# ╟─0e0c25d4-35b9-429b-8223-90e9e8be90f9
# ╠═e55c802d-7923-458f-af42-d951e82e029b
# ╟─c69c0a4a-b90b-414c-883d-3aa50c04b5e1
# ╟─3604a697-8f21-447f-bf75-b3af989e9896
# ╟─b3a6004f-9d10-4582-832a-8917b701f2ad
# ╟─791041e9-d277-4cac-a5ac-1d6ec52e0287
# ╠═f272eab0-8a33-4161-bf9e-e378255631dd
# ╟─d43540d3-beb9-45aa-98cd-1a77f6b8db10
# ╠═c65aa7b6-d85e-4efa-a2ee-1b615155796e
# ╟─d1beea61-776f-4841-97e4-8d423ac22820
# ╠═48b1fc18-eaaf-4f5e-9275-1e942dfbd643
# ╟─6a09fa9c-f04e-466f-9746-beb2a7cfdfa3
# ╠═9b734e9c-f571-4a09-9744-221dcd55b4bf
# ╟─c2c68b93-1cd4-4a38-9dd9-47ce2d591907
# ╟─f9b3f5ce-cfc1-4d59-b21e-2fd07075f036
# ╟─7c367a0b-c5b9-459b-9ccf-e07c84e0b32a
# ╟─496e8c5e-251b-4448-8c59-541877d752c1
# ╟─263e96b7-e659-468d-ba97-ca9832f6ea4d
# ╠═86e7d984-c128-4d2e-8599-3bc70db87a1d
# ╟─4ad081a2-b5c2-48ff-9a28-ec9c8d9f0d0e
# ╠═5acc645a-0d32-4f51-8aa6-063725b83fa8
# ╠═c7121d63-b1ff-4c38-8579-e1adbfef48ef
# ╟─2399ce76-b6da-4a61-bcda-aee22dd275f8
# ╠═a25c6705-54f4-4bad-966e-a8f13ae4c711
# ╟─739136b1-6b01-44c0-bbfd-dcb490d1e191
# ╟─dcce9a84-a9b1-47c1-8e08-7575cb299b56
# ╟─bd185f74-d666-42b4-8da1-c768217f7782
# ╟─2c9cf709-9bc8-48bf-9e19-db8cf7c8690b
# ╠═b4cbb43d-6e4c-4917-99a4-03d13e736144
# ╠═af161709-46d2-4c73-b10d-90acb1e85189
# ╠═1e93928c-7f4c-4295-b755-4e5e16adbd8e
# ╠═fb063bc5-22bc-4b32-8fcb-5fbc4765c8b5
# ╠═0e9664ec-98d8-49d4-a376-24d4770c4c8f
# ╟─90c9d079-4bbc-4609-aa12-afa41a74b2fb
# ╠═0edbb2db-4db8-4dc4-9a73-f7ff86e6f577
# ╟─a944fdea-f41b-4a5f-95ac-e5f4074d4290
# ╟─eb57f7bb-1bff-471f-a599-d1d7d8f771ad
# ╟─d43525da-e0a2-4d2f-9dbb-bf187eebf6c1
# ╟─547ad5ba-06ad-4707-a7ef-e444cf88ae53
# ╟─7ba35a63-ac61-434b-b759-95d505f62d9e
# ╠═398ba928-899f-4843-ad58-25df67c81ffe
# ╠═cee9c93d-cf7b-4da1-b4bb-b544b7cc104c
# ╠═9f8667f3-4104-4642-b2d9-a6d12a6fa5d3
# ╟─3ac01c04-52e3-497e-8c29-8c704e23ae39
# ╟─790377a7-1301-44a8-b300-418567737373
# ╠═536fe0c4-567c-4bda-8c95-347f183c007b
# ╟─db96a6c9-8352-47f3-8319-9c373aa03ff4
# ╟─6e52c719-e9fc-478a-9709-49e250a27d6b
# ╟─e36cda69-d300-4156-9bef-a372f94306d9
# ╟─8f56a866-a141-4275-9769-957ed5834afe
# ╟─cf2938cc-d3f0-4077-9262-3d51866df2cf
# ╠═01418f56-432c-458a-82ea-f2a5c75eb405
# ╟─d3995cc5-f804-4591-926e-b358a8068221
# ╟─bd75a60b-ca34-4211-ac35-8325102cff68
# ╠═5f379c7a-9713-45f7-9a5e-57b8197332c3
# ╟─8d7c27d5-4a07-4ab4-9ece-94fdb7053f73
# ╟─cda32841-0e89-4019-abdc-cf7b0377aa48
# ╟─161ea6af-5661-44e1-ae40-1b581b636c25
# ╠═293ad084-6d6a-4401-9819-53a24646d2c9
# ╟─3231b010-718a-4863-be43-1f0326451e96
# ╠═1c1ccc51-e32a-4881-b892-095d2be55916
# ╟─b0e08212-7e12-4d54-846f-5b0863c37236
# ╟─3183c6ac-5acd-4770-a638-c4c6ba3f7c4f
# ╠═8e9b1e02-2bc0-49d2-b7ed-38de877ebe77
# ╟─ba62f716-b1b5-4d11-91f2-ed121b48216c
# ╟─bad94aca-f77e-417e-be32-0840a3e5c958
# ╟─bbdd495c-f2c6-4264-a4e9-5083753eb410
# ╟─383aa611-e115-482e-873c-4487e53d457f
# ╟─2c6fa743-3dec-417b-b05a-17bb52b5d39d
# ╠═17659ddb-d4e0-4a4b-b34c-8ac52d5dad45
# ╠═2ef9e7e0-c856-4ef3-a08f-89817fc5fd60
# ╟─ac1ffdbf-de6f-48cd-af7c-99528ef26dc0
# ╠═e331d501-71ed-4d93-8498-5c1193776865
# ╟─ae47ef38-e8d0-40b9-9e61-3ab3ca7e7a49
# ╟─aad94861-e2b3-417d-b640-b821e53adb23
# ╠═ab886349-5f3f-45e9-a6e1-a81fdfafa72f
# ╟─19052549-3c5d-4b49-b708-05eac0a2a0ac
# ╠═1778899b-8f05-4b1f-acb5-32af1ace08ee
# ╟─9e78bfc1-fb4e-4626-b387-c2f83bed6ef0
# ╟─f1c0321b-7811-42b1-9d0c-9c69f43d7e1a
# ╠═df044a68-605f-4347-832a-68090ee07950
# ╠═7eccc74d-9a49-44d9-9e43-cbb3c8ad7ce5
# ╠═a661d895-d3d7-4e96-a08f-55b125ed1d40
# ╟─3f01d534-b01d-4ab4-b3cd-e809b02563a9
# ╠═d16adf94-72c3-480d-bd92-738e806068f8
# ╟─56c5b496-a063-459a-8686-22fc70b6a214
# ╟─c4ff4add-ab3c-4585-900e-41f17e905ac5
# ╠═ac18f1ca-0f60-4436-9d8a-797b3dfd8657
# ╟─e8082779-143d-4562-81f3-d493679cf3c7
# ╟─8737797c-6563-4513-a5fc-fde9681b4c63
# ╠═87df5b25-0d2f-4f81-80f1-aaf6c9f89ce3
# ╟─4dec4888-08db-4965-b27a-fc44f316b529
# ╠═bd77bc71-ffdf-4ba1-b1ee-6f2a69044e6f
# ╠═6f411bcc-7084-43c3-a88b-b56ba77b5732
# ╠═3c5ee822-b938-4848-b2b0-f0de2e65b4db
# ╟─3b50062c-99c1-4f68-aabe-2d40d4ad7504
# ╟─d83a282e-cb2b-4837-bfd4-8404b3722e3a
# ╟─c9cf6fb3-0146-42e6-aaae-24e97254c805
# ╠═76730d06-06da-4466-8814-2096b221090f
# ╠═73358bcf-4129-46be-bef4-f623b11e245b
# ╠═a9601654-8263-425e-8d8f-c5bbeacbbe06
# ╟─35b1b5b1-2a23-44bb-bc19-805393d18d8a
# ╠═ec08aa84-3f63-441a-a31c-85b7a82412d1
# ╠═21b3c52e-533f-488f-a2eb-602600b66738
# ╠═e38e0b91-dbd3-4cc6-87ac-add4953411d1
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
