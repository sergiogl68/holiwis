%% Examples of benchmarks for different input formats
addpath benchmarks
clear all;close all;clc;

%% 2.   morphological version for :boundary benchmark for results stored as contour images
% 
% imgDir = 'data/images';
% gtDir = 'data/groundTruth';
% pbDir = 'data/png';
% outDir = 'eval/test_bdry_fast';
% mkdir(outDir);
% nthresh = 99;
% 
% tic;
% boundaryBench_fast(imgDir, gtDir, pbDir, outDir, nthresh);
% toc;


%% 4. morphological version for : all the benchmarks for results stored as a cell of segmentations

imgDir = 'BSR/BSDS500/data/images/val';
gtDir = 'BSR/BSDS500/data/groundTruth/val';
inDir = 'results_lab_val';
outDir = 'eval_val_lab';
mkdir(outDir);
nthresh = 10;

tic;
allBench_fast(imgDir, gtDir, inDir, outDir, nthresh);
toc;

