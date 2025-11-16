// Function: sub_2337E30
// Address: 0x2337e30
//
__int64 __fastcall sub_2337E30(char *a1, _QWORD *a2)
{
  _QWORD *v3; // r12
  __int64 *v4; // rdx
  __int64 *v6; // r14
  char *v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rcx
  size_t v10; // r15
  __int64 v11; // rdx
  unsigned int v12; // r15d
  unsigned int v13; // eax
  __int64 v14; // rdi
  _QWORD *v15; // rcx
  unsigned int v16; // eax
  unsigned int v17; // r14d
  _QWORD *v18; // r12
  _QWORD *i; // rbx
  _QWORD *v20; // r12
  _QWORD *j; // rbx
  _QWORD *v22; // [rsp+8h] [rbp-98h]
  __int64 v23; // [rsp+18h] [rbp-88h]
  _QWORD v24[2]; // [rsp+20h] [rbp-80h] BYREF
  _QWORD v25[2]; // [rsp+30h] [rbp-70h] BYREF
  _QWORD *v26; // [rsp+40h] [rbp-60h] BYREF
  _QWORD *v27; // [rsp+48h] [rbp-58h]
  __int64 v28; // [rsp+50h] [rbp-50h]
  __int64 v29; // [rsp+58h] [rbp-48h]
  __int64 v30; // [rsp+60h] [rbp-40h]

  v3 = a2;
  if ( (unsigned __int8)sub_2306580((__int64)a1, (unsigned __int64)a2) )
    return sub_C89090(qword_4FDC370, a1, (__int64)a2, 0, 0);
  v6 = v4;
  if ( a2 )
  {
    v7 = a1;
    while ( !sub_22F7B60((__int64)&v26, *v7) )
    {
      v7 = (char *)(v9 + 1);
      if ( v8 == 1 )
      {
        v10 = (size_t)a2;
        goto LABEL_9;
      }
    }
    v10 = (size_t)a2 - v8;
    if ( (_QWORD *)((char *)a2 - v8) > a2 )
      v10 = (size_t)a2;
  }
  else
  {
    v10 = 0;
  }
LABEL_9:
  if ( sub_9691B0(a1, (size_t)a2, "module", 6) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "cgscc", 5) )
    return 1;
  if ( sub_9691B0(a1, v10, "function", 8) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "coro-cond", 9) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "require<callgraph>", 18) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "invalidate<callgraph>", 21) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "require<collector-metadata>", 27) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "invalidate<collector-metadata>", 30) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "require<ctx-prof-analysis>", 26) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "invalidate<ctx-prof-analysis>", 29) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "require<dxil-metadata>", 22) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "invalidate<dxil-metadata>", 25) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "require<dxil-resource-binding>", 30) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "invalidate<dxil-resource-binding>", 33) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "require<dxil-resource-type>", 27) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "invalidate<dxil-resource-type>", 30) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "require<inline-advisor>", 23) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "invalidate<inline-advisor>", 26) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "require<ir-similarity>", 22) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "invalidate<ir-similarity>", 25) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "require<last-run-tracking>", 26) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "invalidate<last-run-tracking>", 29) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "require<lcg>", 12) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "invalidate<lcg>", 15) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "require<module-summary>", 23) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "invalidate<module-summary>", 26) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "require<no-op-module>", 21) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "invalidate<no-op-module>", 24) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "require<pass-instrumentation>", 29) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "invalidate<pass-instrumentation>", 32) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "require<profile-summary>", 24) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "invalidate<profile-summary>", 27) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "require<reg-usage>", 18) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "invalidate<reg-usage>", 21) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "require<stack-safety>", 21) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "invalidate<stack-safety>", 24) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "require<verify>", 15) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "invalidate<verify>", 18) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "require<globals-aa>", 19) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "invalidate<globals-aa>", 22) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "always-inline", 13) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "annotation2metadata", 19) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "assign-guid", 11) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "attributor", 10) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "attributor-light", 16) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "called-value-propagation", 24) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "canonicalize-aliases", 20) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "check-debugify", 14) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "constmerge", 10) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "coro-cleanup", 12) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "coro-early", 10) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "cross-dso-cfi", 13) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "ctx-instr-gen", 13) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "ctx-prof-flatten", 16) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "noinline-nonprevailing", 22) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "deadargelim", 11) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "debugify", 8) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "dfsan", 5) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "dot-callgraph", 13) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "dxil-upgrade", 12) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "elim-avail-extern", 17) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "extract-blocks", 14) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "expand-variadics", 16) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "forceattrs", 10) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "function-import", 15) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "global-merge-func", 17) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "globalopt", 9) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "globalsplit", 11) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "hotcoldsplit", 12) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "inferattrs", 10) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "inliner-ml-advisor-release", 26) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "inliner-wrapper", 15) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "inliner-wrapper-no-mandatory-first", 34) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "insert-gcov-profiling", 21) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "instrorderfile", 14) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "instrprof", 9) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "ctx-instr-lower", 15) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "print<ctx-prof-analysis>", 24) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "invalidate<all>", 15) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "iroutliner", 10) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "jmc-instrumenter", 16) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "lower-emutls", 12) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "lower-global-dtors", 18) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "lower-ifunc", 11) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "lowertypetests", 14) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "fatlto-cleanup", 14) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "pgo-force-function-attrs", 24) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "memprof-context-disambiguation", 30) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "memprof-module", 14) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "mergefunc", 9) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "metarenamer", 11) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "module-inline", 13) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "name-anon-globals", 17) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "no-op-module", 12) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "nsan", 4) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "objc-arc-apelim", 15) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "openmp-opt", 10) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "openmp-opt-postlink", 19) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "partial-inliner", 15) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "pgo-icall-prom", 14) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "pgo-instr-gen", 13) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "pgo-instr-use", 13) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "pre-isel-intrinsic-lowering", 27) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "print", 5) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "print-callgraph", 15) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "print-callgraph-sccs", 20) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "print-ir-similarity", 19) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "print-lcg", 9) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "print-lcg-dot", 13) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "print-must-be-executed-contexts", 31) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "print-profile-summary", 21) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "print-stack-safety", 18) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "print<dxil-metadata>", 20) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "print<dxil-resource-binding>", 28) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "print<inline-advisor>", 21) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "print<module-debuginfo>", 23) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "print<reg-usage>", 16) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "pseudo-probe", 12) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "pseudo-probe-update", 19) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "recompute-globalsaa", 19) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "rel-lookup-table-converter", 26) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "rewrite-statepoints-for-gc", 26) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "rewrite-symbols", 15) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "rpo-function-attrs", 18) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "rtsan", 5) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "sample-profile", 14) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "sancov-module", 13) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "sanmd-module", 12) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "scc-oz-module-inliner", 21) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "shadow-stack-gc-lowering", 24) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "strip", 5) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "strip-dead-debug-info", 21) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "strip-dead-prototypes", 21) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "strip-debug-declare", 19) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "strip-nondebug", 14) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "strip-nonlinetable-debuginfo", 28) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "trigger-crash-module", 20) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "trigger-verifier-error", 22) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "tsan-module", 11) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "tysan", 5) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "verify", 6) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "view-callgraph", 14) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "wholeprogramdevirt", 18) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "check-gep-index", 15) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "check-kernel-functions", 22) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "cnp-launch-check", 16) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "ipmsp", 5) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "nv-early-inliner", 16) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "nv-inline-must", 14) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "nvvm-pretreat", 13) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "nvvm-verify", 11) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "printf-lowering", 15) )
    return 1;
  if ( sub_9691B0(a1, (size_t)a2, "select-kernels", 14) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, (__int64)a2, "asan", 4u) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, (__int64)a2, "cg-profile", 0xAu) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, (__int64)a2, "global-merge", 0xCu) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, (__int64)a2, "embed-bitcode", 0xDu) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, (__int64)a2, "globaldce", 9u) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, (__int64)a2, "hwasan", 6u) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, (__int64)a2, "internalize", 0xBu) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, (__int64)a2, "ipsccp", 6u) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, (__int64)a2, "loop-extract", 0xCu) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, (__int64)a2, "memprof-use", 0xBu) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, (__int64)a2, "msan", 4u) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, (__int64)a2, "print<structural-hash>", 0x16u) )
    return 1;
  if ( (unsigned __int8)sub_2337DE0(a1, (__int64)a2, "lower-ops", 9u) )
    return 1;
  v12 = sub_2337DE0(a1, (__int64)a2, "set-global-array-alignment", 0x1Au);
  if ( (_BYTE)v12 )
  {
    return 1;
  }
  else
  {
    v13 = *((_DWORD *)v6 + 2);
    if ( v13 )
    {
      v14 = *v6;
      v26 = 0;
      v15 = v25;
      v27 = 0;
      v28 = 0;
      v29 = 0;
      v30 = 0;
      v23 = v14 + 32LL * v13;
      while ( v14 != v23 )
      {
        v24[0] = a1;
        v24[1] = v3;
        v25[0] = 0;
        v25[1] = 0;
        if ( !*(_QWORD *)(v14 + 16) )
          sub_4263D6(v14, a2, v11);
        v22 = v15;
        a2 = v24;
        v16 = (*(__int64 (__fastcall **)(__int64, _QWORD *, _QWORD **))(v14 + 24))(v14, v24, &v26);
        v15 = v22;
        v17 = v16;
        v14 += 32;
        if ( (_BYTE)v16 )
        {
          v18 = v27;
          for ( i = v26; v18 != i; ++i )
          {
            if ( *i )
              (*(void (__fastcall **)(_QWORD, _QWORD *, __int64, _QWORD *))(*(_QWORD *)*i + 8LL))(*i, v24, v11, v15);
          }
          if ( v26 )
            j_j___libc_free_0((unsigned __int64)v26);
          return v17;
        }
      }
      v20 = v27;
      for ( j = v26; v20 != j; ++j )
      {
        if ( *j )
          (*(void (__fastcall **)(_QWORD, _QWORD *, __int64, _QWORD *))(*(_QWORD *)*j + 8LL))(*j, a2, v11, v15);
      }
      if ( v26 )
        j_j___libc_free_0((unsigned __int64)v26);
    }
  }
  return v12;
}
