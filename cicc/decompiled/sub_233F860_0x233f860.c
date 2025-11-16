// Function: sub_233F860
// Address: 0x233f860
//
__int64 __fastcall sub_233F860(char *a1, size_t a2, __int64 *a3)
{
  size_t v5; // rbx
  char *v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rcx
  _QWORD *v9; // rsi
  __int64 v10; // rdx
  unsigned int v11; // r15d
  unsigned int v12; // eax
  __int64 v13; // rdi
  _QWORD *v14; // rcx
  __int64 v15; // r14
  unsigned __int8 v16; // al
  _QWORD *v18; // [rsp+0h] [rbp-90h]
  unsigned __int8 v19; // [rsp+8h] [rbp-88h]
  _QWORD v20[2]; // [rsp+10h] [rbp-80h] BYREF
  _QWORD v21[2]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v22[12]; // [rsp+30h] [rbp-60h] BYREF

  v5 = a2;
  if ( a2 )
  {
    v6 = a1;
    while ( !sub_2304E40((__int64)v22, *v6) )
    {
      v6 = (char *)(v8 + 1);
      if ( v7 == 1 )
        goto LABEL_7;
    }
    a2 -= v7;
    if ( v5 - v7 > v5 )
      a2 = v5;
  }
  else
  {
    a2 = 0;
  }
LABEL_7:
  if ( sub_9691B0(a1, a2, "function", 8) )
    return 1;
  if ( sub_9691B0(a1, v5, "loop", 4) )
    return 1;
  if ( sub_9691B0(a1, v5, "loop-mssa", 9) )
    return 1;
  if ( sub_9691B0(a1, v5, "machine-function", 16) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<aa>", 11) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<aa>", 14) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<access-info>", 20) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<access-info>", 23) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<assumptions>", 20) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<assumptions>", 23) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<bb-sections-profile-reader>", 35) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<bb-sections-profile-reader>", 38) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<block-freq>", 19) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<block-freq>", 22) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<branch-prob>", 20) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<branch-prob>", 23) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<cycles>", 15) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<cycles>", 18) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<da>", 11) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<da>", 14) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<debug-ata>", 18) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<debug-ata>", 21) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<demanded-bits>", 22) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<demanded-bits>", 25) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<domfrontier>", 20) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<domfrontier>", 23) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<domtree>", 16) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<domtree>", 19) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<func-properties>", 24) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<func-properties>", 27) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<machine-function-info>", 30) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<machine-function-info>", 33) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<gc-function>", 20) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<gc-function>", 23) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<inliner-size-estimator>", 31) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<inliner-size-estimator>", 34) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<last-run-tracking>", 26) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<last-run-tracking>", 29) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<lazy-value-info>", 24) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<lazy-value-info>", 27) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<loops>", 14) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<loops>", 17) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<memdep>", 15) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<memdep>", 18) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<memoryssa>", 18) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<memoryssa>", 21) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<no-op-function>", 23) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<no-op-function>", 26) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<opt-remark-emit>", 24) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<opt-remark-emit>", 27) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<pass-instrumentation>", 29) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<pass-instrumentation>", 32) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<phi-values>", 19) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<phi-values>", 22) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<postdomtree>", 20) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<postdomtree>", 23) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<regions>", 16) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<regions>", 19) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<scalar-evolution>", 25) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<scalar-evolution>", 28) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<should-not-run-function-passes>", 39) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<should-not-run-function-passes>", 42) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<should-run-extra-vector-passes>", 39) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<should-run-extra-vector-passes>", 42) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<ssp-layout>", 19) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<ssp-layout>", 22) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<stack-safety-local>", 27) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<stack-safety-local>", 30) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<target-ir>", 18) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<target-ir>", 21) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<target-lib-info>", 24) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<target-lib-info>", 27) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<uniformity>", 19) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<uniformity>", 22) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<verify>", 15) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<verify>", 18) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<rpa>", 12) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<rpa>", 15) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<merge-sets>", 19) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<merge-sets>", 22) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<basic-aa>", 17) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<basic-aa>", 20) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<objc-arc-aa>", 20) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<objc-arc-aa>", 23) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<scev-aa>", 16) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<scev-aa>", 19) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<scoped-noalias-aa>", 26) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<scoped-noalias-aa>", 29) )
    return 1;
  if ( sub_9691B0(a1, v5, "require<tbaa>", 13) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<tbaa>", 16) )
    return 1;
  if ( sub_9691B0(a1, v5, "aa-eval", 7) )
    return 1;
  if ( sub_9691B0(a1, v5, "adce", 4) )
    return 1;
  if ( sub_9691B0(a1, v5, "add-discriminators", 18) )
    return 1;
  if ( sub_9691B0(a1, v5, "aggressive-instcombine", 22) )
    return 1;
  if ( sub_9691B0(a1, v5, "alignment-from-assumptions", 26) )
    return 1;
  if ( sub_9691B0(a1, v5, "annotation-remarks", 18) )
    return 1;
  if ( sub_9691B0(a1, v5, "assume-builder", 14) )
    return 1;
  if ( sub_9691B0(a1, v5, "assume-simplify", 15) )
    return 1;
  if ( sub_9691B0(a1, v5, "atomic-expand", 13) )
    return 1;
  if ( sub_9691B0(a1, v5, "bdce", 4) )
    return 1;
  if ( sub_9691B0(a1, v5, "break-crit-edges", 16) )
    return 1;
  if ( sub_9691B0(a1, v5, "callbr-prepare", 14) )
    return 1;
  if ( sub_9691B0(a1, v5, "callsite-splitting", 18) )
    return 1;
  if ( sub_9691B0(a1, v5, "chr", 3) )
    return 1;
  if ( sub_9691B0(a1, v5, "codegenprepare", 14) )
    return 1;
  if ( sub_9691B0(a1, v5, "complex-deinterleaving", 22) )
    return 1;
  if ( sub_9691B0(a1, v5, "consthoist", 10) )
    return 1;
  if ( sub_9691B0(a1, v5, "constraint-elimination", 22) )
    return 1;
  if ( sub_9691B0(a1, v5, "coro-elide", 10) )
    return 1;
  if ( sub_9691B0(a1, v5, "correlated-propagation", 22) )
    return 1;
  if ( sub_9691B0(a1, v5, "count-visits", 12) )
    return 1;
  if ( sub_9691B0(a1, v5, "dce", 3) )
    return 1;
  if ( sub_9691B0(a1, v5, "declare-to-assign", 17) )
    return 1;
  if ( sub_9691B0(a1, v5, "dfa-jump-threading", 18) )
    return 1;
  if ( sub_9691B0(a1, v5, "div-rem-pairs", 13) )
    return 1;
  if ( sub_9691B0(a1, v5, "dot-cfg", 7) )
    return 1;
  if ( sub_9691B0(a1, v5, "dot-cfg-only", 12) )
    return 1;
  if ( sub_9691B0(a1, v5, "dot-dom", 7) )
    return 1;
  if ( sub_9691B0(a1, v5, "dot-dom-only", 12) )
    return 1;
  if ( sub_9691B0(a1, v5, "dot-post-dom", 12) )
    return 1;
  if ( sub_9691B0(a1, v5, "dot-post-dom-only", 17) )
    return 1;
  if ( sub_9691B0(a1, v5, "dse", 3) )
    return 1;
  if ( sub_9691B0(a1, v5, "dwarf-eh-prepare", 16) )
    return 1;
  if ( sub_9691B0(a1, v5, "expand-large-div-rem", 20) )
    return 1;
  if ( sub_9691B0(a1, v5, "expand-large-fp-convert", 23) )
    return 1;
  if ( sub_9691B0(a1, v5, "expand-memcmp", 13) )
    return 1;
  if ( sub_9691B0(a1, v5, "extra-vector-passes", 19) )
    return 1;
  if ( sub_9691B0(a1, v5, "fix-irreducible", 15) )
    return 1;
  if ( sub_9691B0(a1, v5, "flatten-cfg", 11) )
    return 1;
  if ( sub_9691B0(a1, v5, "float2int", 9) )
    return 1;
  if ( sub_9691B0(a1, v5, "gc-lowering", 11) )
    return 1;
  if ( sub_9691B0(a1, v5, "guard-widening", 14) )
    return 1;
  if ( sub_9691B0(a1, v5, "gvn-hoist", 9) )
    return 1;
  if ( sub_9691B0(a1, v5, "gvn-sink", 8) )
    return 1;
  if ( sub_9691B0(a1, v5, "helloworld", 10) )
    return 1;
  if ( sub_9691B0(a1, v5, "indirectbr-expand", 17) )
    return 1;
  if ( sub_9691B0(a1, v5, "infer-address-spaces", 20) )
    return 1;
  if ( sub_9691B0(a1, v5, "infer-alignment", 15) )
    return 1;
  if ( sub_9691B0(a1, v5, "inject-tli-mappings", 19) )
    return 1;
  if ( sub_9691B0(a1, v5, "instcount", 9) )
    return 1;
  if ( sub_9691B0(a1, v5, "instnamer", 9) )
    return 1;
  if ( sub_9691B0(a1, v5, "instsimplify", 12) )
    return 1;
  if ( sub_9691B0(a1, v5, "interleaved-access", 18) )
    return 1;
  if ( sub_9691B0(a1, v5, "interleaved-load-combine", 24) )
    return 1;
  if ( sub_9691B0(a1, v5, "invalidate<all>", 15) )
    return 1;
  if ( sub_9691B0(a1, v5, "irce", 4) )
    return 1;
  if ( sub_9691B0(a1, v5, "jump-threading", 14) )
    return 1;
  if ( sub_9691B0(a1, v5, "jump-table-to-switch", 20) )
    return 1;
  if ( sub_9691B0(a1, v5, "kcfi", 4) )
    return 1;
  if ( sub_9691B0(a1, v5, "kernel-info", 11) )
    return 1;
  if ( sub_9691B0(a1, v5, "lcssa", 5) )
    return 1;
  if ( sub_9691B0(a1, v5, "libcalls-shrinkwrap", 19) )
    return 1;
  if ( sub_9691B0(a1, v5, "lint", 4) )
    return 1;
  if ( sub_9691B0(a1, v5, "load-store-vectorizer", 21) )
    return 1;
  if ( sub_9691B0(a1, v5, "loop-data-prefetch", 18) )
    return 1;
  if ( sub_9691B0(a1, v5, "loop-distribute", 15) )
    return 1;
  if ( sub_9691B0(a1, v5, "loop-fusion", 11) )
    return 1;
  if ( sub_9691B0(a1, v5, "loop-load-elim", 14) )
    return 1;
  if ( sub_9691B0(a1, v5, "loop-simplify", 13) )
    return 1;
  if ( sub_9691B0(a1, v5, "loop-sink", 9) )
    return 1;
  if ( sub_9691B0(a1, v5, "loop-versioning", 15) )
    return 1;
  if ( sub_9691B0(a1, v5, "lower-atomic", 12) )
    return 1;
  if ( sub_9691B0(a1, v5, "lower-constant-intrinsics", 25) )
    return 1;
  if ( sub_9691B0(a1, v5, "lower-expect", 12) )
    return 1;
  if ( sub_9691B0(a1, v5, "lower-guard-intrinsic", 21) )
    return 1;
  if ( sub_9691B0(a1, v5, "lower-invoke", 12) )
    return 1;
  if ( sub_9691B0(a1, v5, "lower-widenable-condition", 25) )
    return 1;
  if ( sub_9691B0(a1, v5, "make-guards-explicit", 20) )
    return 1;
  if ( sub_9691B0(a1, v5, "mem2reg", 7) )
    return 1;
  if ( sub_9691B0(a1, v5, "memcpyopt", 9) )
    return 1;
  if ( sub_9691B0(a1, v5, "memprof", 7) )
    return 1;
  if ( sub_9691B0(a1, v5, "mergeicmps", 10) )
    return 1;
  if ( sub_9691B0(a1, v5, "mergereturn", 11) )
    return 1;
  if ( sub_9691B0(a1, v5, "move-auto-init", 14) )
    return 1;
  if ( sub_9691B0(a1, v5, "nary-reassociate", 16) )
    return 1;
  if ( sub_9691B0(a1, v5, "newgvn", 6) )
    return 1;
  if ( sub_9691B0(a1, v5, "no-op-function", 14) )
    return 1;
  if ( sub_9691B0(a1, v5, "normalize", 9) )
    return 1;
  if ( sub_9691B0(a1, v5, "objc-arc", 8) )
    return 1;
  if ( sub_9691B0(a1, v5, "objc-arc-contract", 17) )
    return 1;
  if ( sub_9691B0(a1, v5, "objc-arc-expand", 15) )
    return 1;
  if ( sub_9691B0(a1, v5, "pa-eval", 7) )
    return 1;
  if ( sub_9691B0(a1, v5, "partially-inline-libcalls", 25) )
    return 1;
  if ( sub_9691B0(a1, v5, "pgo-memop-opt", 13) )
    return 1;
  if ( sub_9691B0(a1, v5, "place-safepoints", 16) )
    return 1;
  if ( sub_9691B0(a1, v5, "print", 5) )
    return 1;
  if ( sub_9691B0(a1, v5, "print-alias-sets", 16) )
    return 1;
  if ( sub_9691B0(a1, v5, "print-cfg-sccs", 14) )
    return 1;
  if ( sub_9691B0(a1, v5, "print-memderefs", 15) )
    return 1;
  if ( sub_9691B0(a1, v5, "print-mustexecute", 17) )
    return 1;
  if ( sub_9691B0(a1, v5, "print-predicateinfo", 19) )
    return 1;
  if ( sub_9691B0(a1, v5, "print<access-info>", 18) )
    return 1;
  if ( sub_9691B0(a1, v5, "print<assumptions>", 18) )
    return 1;
  if ( sub_9691B0(a1, v5, "print<block-freq>", 17) )
    return 1;
  if ( sub_9691B0(a1, v5, "print<branch-prob>", 18) )
    return 1;
  if ( sub_9691B0(a1, v5, "print<cost-model>", 17) )
    return 1;
  if ( sub_9691B0(a1, v5, "print<cycles>", 13) )
    return 1;
  if ( sub_9691B0(a1, v5, "print<da>", 9) )
    return 1;
  if ( sub_9691B0(a1, v5, "print<debug-ata>", 16) )
    return 1;
  if ( sub_9691B0(a1, v5, "print<delinearization>", 22) )
    return 1;
  if ( sub_9691B0(a1, v5, "print<demanded-bits>", 20) )
    return 1;
  if ( sub_9691B0(a1, v5, "print<domfrontier>", 18) )
    return 1;
  if ( sub_9691B0(a1, v5, "print<domtree>", 14) )
    return 1;
  if ( sub_9691B0(a1, v5, "print<func-properties>", 22) )
    return 1;
  if ( sub_9691B0(a1, v5, "print<inline-cost>", 18) )
    return 1;
  if ( sub_9691B0(a1, v5, "print<inliner-size-estimator>", 29) )
    return 1;
  if ( sub_9691B0(a1, v5, "print<lazy-value-info>", 22) )
    return 1;
  if ( sub_9691B0(a1, v5, "print<loops>", 12) )
    return 1;
  if ( sub_9691B0(a1, v5, "print<memoryssa-walker>", 23) )
    return 1;
  if ( sub_9691B0(a1, v5, "print<phi-values>", 17) )
    return 1;
  if ( sub_9691B0(a1, v5, "print<postdomtree>", 18) )
    return 1;
  if ( sub_9691B0(a1, v5, "print<regions>", 14) )
    return 1;
  if ( sub_9691B0(a1, v5, "print<scalar-evolution>", 23) )
    return 1;
  if ( sub_9691B0(a1, v5, "print<stack-safety-local>", 25) )
    return 1;
  if ( sub_9691B0(a1, v5, "print<uniformity>", 17) )
    return 1;
  if ( sub_9691B0(a1, v5, "reassociate", 11) )
    return 1;
  if ( sub_9691B0(a1, v5, "redundant-dbg-inst-elim", 23) )
    return 1;
  if ( sub_9691B0(a1, v5, "reg2mem", 7) )
    return 1;
  if ( sub_9691B0(a1, v5, "safe-stack", 10) )
    return 1;
  if ( sub_9691B0(a1, v5, "sandbox-vectorizer", 18) )
    return 1;
  if ( sub_9691B0(a1, v5, "scalarize-masked-mem-intrin", 27) )
    return 1;
  if ( sub_9691B0(a1, v5, "sccp", 4) )
    return 1;
  if ( sub_9691B0(a1, v5, "select-optimize", 15) )
    return 1;
  if ( sub_9691B0(a1, v5, "separate-const-offset-from-gep", 30) )
    return 1;
  if ( sub_9691B0(a1, v5, "sink", 4) )
    return 1;
  if ( sub_9691B0(a1, v5, "sjlj-eh-prepare", 15) )
    return 1;
  if ( sub_9691B0(a1, v5, "slp-vectorizer", 14) )
    return 1;
  if ( sub_9691B0(a1, v5, "slsr", 4) )
    return 1;
  if ( sub_9691B0(a1, v5, "stack-protector", 15) )
    return 1;
  if ( sub_9691B0(a1, v5, "strip-gc-relocates", 18) )
    return 1;
  if ( sub_9691B0(a1, v5, "tailcallelim", 12) )
    return 1;
  if ( sub_9691B0(a1, v5, "transform-warning", 17) )
    return 1;
  if ( sub_9691B0(a1, v5, "trigger-crash-function", 22) )
    return 1;
  if ( sub_9691B0(a1, v5, "trigger-verifier-error", 22) )
    return 1;
  if ( sub_9691B0(a1, v5, "tsan", 4) )
    return 1;
  if ( sub_9691B0(a1, v5, "unify-loop-exits", 16) )
    return 1;
  if ( sub_9691B0(a1, v5, "vector-combine", 14) )
    return 1;
  if ( sub_9691B0(a1, v5, "verify", 6) )
    return 1;
  if ( sub_9691B0(a1, v5, "verify<cycles>", 14) )
    return 1;
  if ( sub_9691B0(a1, v5, "verify<domtree>", 15) )
    return 1;
  if ( sub_9691B0(a1, v5, "verify<loops>", 13) )
    return 1;
  if ( sub_9691B0(a1, v5, "verify<memoryssa>", 17) )
    return 1;
  if ( sub_9691B0(a1, v5, "verify<regions>", 15) )
    return 1;
  if ( sub_9691B0(a1, v5, "verify<safepoint-ir>", 20) )
    return 1;
  if ( sub_9691B0(a1, v5, "verify<scalar-evolution>", 24) )
    return 1;
  if ( sub_9691B0(a1, v5, "view-cfg", 8) )
    return 1;
  if ( sub_9691B0(a1, v5, "view-cfg-only", 13) )
    return 1;
  if ( sub_9691B0(a1, v5, "view-dom", 8) )
    return 1;
  if ( sub_9691B0(a1, v5, "view-dom-only", 13) )
    return 1;
  if ( sub_9691B0(a1, v5, "view-post-dom", 13) )
    return 1;
  if ( sub_9691B0(a1, v5, "view-post-dom-only", 18) )
    return 1;
  if ( sub_9691B0(a1, v5, "wasm-eh-prepare", 15) )
    return 1;
  if ( sub_9691B0(a1, v5, "basic-dbe", 9) )
    return 1;
  if ( sub_9691B0(a1, v5, "branch-dist", 11) )
    return 1;
  if ( sub_9691B0(a1, v5, "byval-mem2reg", 13) )
    return 1;
  if ( sub_9691B0(a1, v5, "bypass-slow-division", 20) )
    return 1;
  if ( sub_9691B0(a1, v5, "normalize-gep", 13) )
    return 1;
  if ( sub_9691B0(a1, v5, "nvvm-reflect-pp", 15) )
    return 1;
  if ( sub_9691B0(a1, v5, "nvvm-peephole-optimizer", 23) )
    return 1;
  if ( sub_9691B0(a1, v5, "old-load-store-vectorizer", 25) )
    return 1;
  if ( sub_9691B0(a1, v5, "print<merge-sets>", 17) )
    return 1;
  if ( sub_9691B0(a1, v5, "remat", 5) )
    return 1;
  if ( sub_9691B0(a1, v5, "print<rpa>", 10) )
    return 1;
  if ( sub_9691B0(a1, v5, "propagate-alignment", 19) )
    return 1;
  if ( sub_9691B0(a1, v5, "reuse-local-memory", 18) )
    return 1;
  if ( sub_9691B0(a1, v5, "set-local-array-alignment", 25) )
    return 1;
  if ( sub_9691B0(a1, v5, "sinking2", 8) )
    return 1;
  if ( sub_9691B0(a1, v5, "d2ir-scalarizer", 15) )
    return 1;
  if ( sub_9691B0(a1, v5, "sink<rp-aware>", 14) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, v5, "cfguard", 7u) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, v5, "early-cse", 9u) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, v5, "ee-instrument", 0xDu) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, v5, "function-simplification", 0x17u) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, v5, "gvn", 3u) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, v5, "instcombine", 0xBu) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, v5, "loop-unroll", 0xBu) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, v5, "loop-vectorize", 0xEu) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, v5, "lower-allow-check", 0x11u) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, v5, "lower-matrix-intrinsics", 0x17u) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, v5, "lower-switch", 0xCu) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, v5, "mldst-motion", 0xCu) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, v5, "print<da>", 9u) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, v5, "print<memoryssa>", 0x10u) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, v5, "print<stack-lifetime>", 0x15u) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, v5, "scalarizer", 0xAu) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, v5, "separate-const-offset-from-gep", 0x1Eu) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, v5, "simplifycfg", 0xBu) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, v5, "speculative-execution", 0x15u) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, v5, "sroa", 4u) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, v5, "structurizecfg", 0xEu) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, v5, "win-eh-prepare", 0xEu) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, v5, "bounds-checking", 0xFu) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, v5, "memory-space-opt", 0x10u) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, v5, "lower-aggr-copies", 0x11u) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, v5, "lower-struct-args", 0x11u) )
    return 1;
  v9 = (_QWORD *)v5;
  v11 = sub_2337DE0(a1, v5, "process-restrict", 0x10u);
  if ( (_BYTE)v11 )
  {
    return 1;
  }
  else
  {
    v12 = *((_DWORD *)a3 + 2);
    if ( v12 )
    {
      v13 = *a3;
      memset(v22, 0, 40);
      v14 = v21;
      v15 = v13 + 32LL * v12;
      while ( v13 != v15 )
      {
        v20[0] = a1;
        v20[1] = v5;
        v21[0] = 0;
        v21[1] = 0;
        if ( !*(_QWORD *)(v13 + 16) )
          sub_4263D6(v13, v9, v10);
        v18 = v14;
        v9 = v20;
        v16 = (*(__int64 (__fastcall **)(__int64, _QWORD *, _QWORD *))(v13 + 24))(v13, v20, v22);
        v14 = v18;
        v13 += 32;
        if ( v16 )
        {
          v19 = v16;
          sub_233F7F0((__int64)v22);
          return v19;
        }
      }
      sub_233F7F0((__int64)v22);
    }
  }
  return v11;
}
