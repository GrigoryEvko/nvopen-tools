// Function: sub_2394710
// Address: 0x2394710
//
__int64 *__fastcall sub_2394710(__int64 *a1, __int64 a2, __int64 a3, _BYTE *a4, unsigned __int64 a5)
{
  const __m128i *v8; // rax
  const __m128i *v9; // rax
  unsigned __int64 v10; // rax
  const __m128i *v11; // rax
  const __m128i *v12; // rax
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  unsigned int v15; // eax
  unsigned int v16; // ebx
  __int64 v17; // rdx
  __int64 v18; // r14
  unsigned int v19; // edx
  __int64 v20; // rcx
  const char *v22; // rdi
  const __m128i *v23; // rsi
  const __m128i *v24; // rcx
  __int64 v25; // rdx
  unsigned __int64 v26; // rax
  __m128i v27; // rdi
  const __m128i *v28; // rdx
  __int64 v29; // rax
  __int64 *v30; // rcx
  char v31; // al
  _QWORD **i; // rbx
  __int64 v33; // rbx
  const __m128i *v34; // r14
  unsigned int v35; // eax
  __int64 v36; // rdx
  unsigned int v37; // ebx
  bool v38; // zf
  char *v39; // rax
  _QWORD **j; // rbx
  __int64 *v41; // [rsp+0h] [rbp-160h]
  __int64 v42; // [rsp+10h] [rbp-150h]
  __int64 v43; // [rsp+18h] [rbp-148h]
  __int64 v44; // [rsp+18h] [rbp-148h]
  __int64 v45; // [rsp+18h] [rbp-148h]
  __int64 v46; // [rsp+18h] [rbp-148h]
  _QWORD v47[3]; // [rsp+20h] [rbp-140h] BYREF
  __m128i v48; // [rsp+40h] [rbp-120h] BYREF
  __m128i v49; // [rsp+50h] [rbp-110h] BYREF
  const __m128i *v50; // [rsp+60h] [rbp-100h] BYREF
  const __m128i *v51; // [rsp+68h] [rbp-F8h]
  unsigned __int64 v52; // [rsp+70h] [rbp-F0h]
  const __m128i *v53; // [rsp+80h] [rbp-E0h] BYREF
  const __m128i *v54; // [rsp+88h] [rbp-D8h]
  unsigned __int64 v55; // [rsp+90h] [rbp-D0h]
  char v56; // [rsp+98h] [rbp-C8h]
  __m128i v57; // [rsp+A0h] [rbp-C0h] BYREF
  unsigned __int64 v58[4]; // [rsp+B0h] [rbp-B0h] BYREF
  __m128i v59; // [rsp+D0h] [rbp-90h] BYREF
  __int64 *v60; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v61; // [rsp+E8h] [rbp-78h]
  unsigned __int64 v62; // [rsp+F0h] [rbp-70h]
  void *v63; // [rsp+F8h] [rbp-68h] BYREF
  __m128i *v64; // [rsp+100h] [rbp-60h]
  _QWORD v65[2]; // [rsp+108h] [rbp-58h] BYREF
  _QWORD v66[9]; // [rsp+118h] [rbp-48h] BYREF

  v47[0] = a4;
  v47[1] = a5;
  sub_2352D90((__int64)&v53, a4, a5);
  if ( !v56 || v53 == v54 )
  {
    v15 = sub_C63BB0();
    v59.m128i_i64[1] = 22;
    v16 = v15;
    v18 = v17;
    v59.m128i_i64[0] = (__int64)"invalid pipeline '{0}'";
    v60 = v65;
    v61 = 1;
    LOBYTE(v62) = 1;
    v63 = &unk_49DB108;
    v64 = (__m128i *)v47;
    v65[0] = &v63;
    sub_23328D0((__int64)&v57, (__int64)&v59);
    v19 = v16;
    v20 = v18;
    goto LABEL_13;
  }
  v48 = _mm_loadu_si128(v53);
  if ( (unsigned __int8)sub_2337E30((char *)v48.m128i_i64[0], (_QWORD *)v48.m128i_i64[1]) )
    goto LABEL_9;
  if ( !(unsigned __int8)sub_2339450((char *)v48.m128i_i64[0], (_QWORD *)v48.m128i_i64[1], (__int64 *)(a2 + 1568)) )
  {
    if ( (unsigned __int8)sub_233F860((char *)v48.m128i_i64[0], v48.m128i_u64[1], (__int64 *)(a2 + 1728)) )
    {
      v59.m128i_i64[1] = 8;
      v59.m128i_i64[0] = (__int64)"function";
      goto LABEL_6;
    }
    if ( (unsigned __int8)sub_2337DE0((char *)v48.m128i_i64[0], v48.m128i_i64[1], "lnicm", 5u) )
      goto LABEL_29;
    if ( v48.m128i_i64[1] == 12 )
    {
      if ( *(_QWORD *)v48.m128i_i64[0] == 0x616C662D706F6F6CLL && *(_DWORD *)(v48.m128i_i64[0] + 8) == 1852142708 )
        goto LABEL_24;
    }
    else if ( v48.m128i_i64[1] == 16
           && !(*(_QWORD *)v48.m128i_i64[0] ^ 0x746E692D706F6F6CLL
              | *(_QWORD *)(v48.m128i_i64[0] + 8) ^ 0x65676E6168637265LL) )
    {
      goto LABEL_24;
    }
    if ( sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "loop-unroll-and-jam", 19)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "no-op-loopnest", 14)
      || (unsigned __int8)sub_2337CB0(v48.m128i_i64[0], (_QWORD *)v48.m128i_i64[1], (__int64 *)(a2 + 1888)) )
    {
LABEL_24:
      v22 = "loop";
      v59.m128i_i64[1] = 8;
      v59.m128i_i64[0] = (__int64)"function";
LABEL_25:
      v57.m128i_i64[0] = (__int64)v22;
      v57.m128i_i64[1] = strlen(v22);
LABEL_26:
      v58[0] = (unsigned __int64)v53;
      v58[1] = (unsigned __int64)v54;
      v54 = 0;
      v58[2] = v55;
      v55 = 0;
      v53 = 0;
      sub_2366A40((__int64 *)&v60, &v57, 1);
      sub_2366A40((__int64 *)&v50, &v59, 1);
      sub_234A860((__int64)&v53, (__int64 *)&v50);
      sub_234A6B0((unsigned __int64 *)&v50);
      sub_234A6B0((unsigned __int64 *)&v60);
      sub_234A6B0(v58);
      goto LABEL_9;
    }
    if ( (unsigned __int8)sub_2337DE0((char *)v48.m128i_i64[0], v48.m128i_i64[1], "licm", 4u) )
    {
LABEL_29:
      v22 = "loop-mssa";
      v59.m128i_i64[1] = 8;
      v59.m128i_i64[0] = (__int64)"function";
      goto LABEL_25;
    }
    if ( sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "require<ddg>", 12)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "invalidate<ddg>", 15)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "require<iv-users>", 17)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "invalidate<iv-users>", 20)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "require<no-op-loop>", 19)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "invalidate<no-op-loop>", 22)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "require<pass-instrumentation>", 29)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "invalidate<pass-instrumentation>", 32)
      || sub_9691B0(
           (const void *)v48.m128i_i64[0],
           v48.m128i_u64[1],
           "require<should-run-extra-simple-loop-unswitch>",
           46)
      || sub_9691B0(
           (const void *)v48.m128i_i64[0],
           v48.m128i_u64[1],
           "invalidate<should-run-extra-simple-loop-unswitch>",
           49)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "canon-freeze", 12)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "dot-ddg", 7)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "guard-widening", 14)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "extra-simple-loop-unswitch-passes", 33)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "indvars", 7)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "invalidate<all>", 15)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "loop-bound-split", 16)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "loop-deletion", 13)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "loop-idiom", 10)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "loop-idiom-vectorize", 20)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "loop-instsimplify", 17)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "loop-predication", 16)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "loop-reduce", 11)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "loop-term-fold", 14)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "loop-simplifycfg", 16)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "loop-unroll-full", 16)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "loop-versioning-licm", 20)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "no-op-loop", 10)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "print", 5)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "print<ddg>", 10)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "print<iv-users>", 15)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "print<loop-cache-cost>", 22)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "print<loopnest>", 15)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "loop-index-split", 16)
      || (unsigned __int8)sub_2337DE0((char *)v48.m128i_i64[0], v48.m128i_i64[1], "licm", 4u)
      || (unsigned __int8)sub_2337DE0((char *)v48.m128i_i64[0], v48.m128i_i64[1], "lnicm", 5u)
      || (unsigned __int8)sub_2337DE0((char *)v48.m128i_i64[0], v48.m128i_i64[1], "loop-rotate", 0xBu)
      || (unsigned __int8)sub_2337DE0((char *)v48.m128i_i64[0], v48.m128i_i64[1], "simple-loop-unswitch", 0x14u)
      || (unsigned __int8)sub_2337CB0(v48.m128i_i64[0], (_QWORD *)v48.m128i_i64[1], (__int64 *)(a2 + 1888)) )
    {
      goto LABEL_24;
    }
    if ( sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "machine-function", 16)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "require<edge-bundles>", 21)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "invalidate<edge-bundles>", 24)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "require<livedebugvars>", 22)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "invalidate<livedebugvars>", 25)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "require<live-intervals>", 23)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "invalidate<live-intervals>", 26)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "require<live-reg-matrix>", 24)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "invalidate<live-reg-matrix>", 27)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "require<live-stacks>", 20)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "invalidate<live-stacks>", 23)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "require<live-vars>", 18)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "invalidate<live-vars>", 21)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "require<machine-block-freq>", 27)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "invalidate<machine-block-freq>", 30)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "require<machine-branch-prob>", 28)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "invalidate<machine-branch-prob>", 31)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "require<machine-cycles>", 23)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "invalidate<machine-cycles>", 26)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "require<machine-dom-tree>", 25)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "invalidate<machine-dom-tree>", 28)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "require<machine-loops>", 22)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "invalidate<machine-loops>", 25)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "require<machine-opt-remark-emitter>", 35)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "invalidate<machine-opt-remark-emitter>", 38)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "require<machine-post-dom-tree>", 30)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "invalidate<machine-post-dom-tree>", 33)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "require<machine-trace-metrics>", 30)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "invalidate<machine-trace-metrics>", 33)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "require<pass-instrumentation>", 29)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "invalidate<pass-instrumentation>", 32)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "require<regalloc-evict>", 23)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "invalidate<regalloc-evict>", 26)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "require<regalloc-priority>", 26)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "invalidate<regalloc-priority>", 29)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "require<slot-indexes>", 21)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "invalidate<slot-indexes>", 24)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "require<spill-code-placement>", 29)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "invalidate<spill-code-placement>", 32)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "require<virtregmap>", 19)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "invalidate<virtregmap>", 22)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "dead-mi-elimination", 19)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "detect-dead-lanes", 17)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "early-ifcvt", 11)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "early-machinelicm", 17)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "early-tailduplication", 21)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "finalize-isel", 13)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "fixup-statepoint-caller-saved", 29)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "localstackalloc", 15)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "machine-cp", 10)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "machine-cse", 11)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "machine-latecleanup", 19)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "machine-scheduler", 17)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "machinelicm", 11)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "no-op-machine-function", 22)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "opt-phis", 8)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "patchable-function", 18)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "peephole-opt", 12)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "phi-node-elimination", 20)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "post-RA-sched", 13)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "postmisched", 11)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "post-ra-pseudos", 15)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "print", 5)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "print<livedebugvars>", 20)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "print<live-intervals>", 21)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "print<live-stacks>", 18)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "print<live-vars>", 16)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "print<machine-block-freq>", 25)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "print<machine-branch-prob>", 26)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "print<machine-cycles>", 21)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "print<machine-dom-tree>", 23)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "print<machine-loops>", 20)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "print<machine-post-dom-tree>", 28)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "print<slot-indexes>", 19)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "print<virtregmap>", 17)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "reg-usage-collector", 19)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "reg-usage-propagation", 21)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "register-coalescer", 18)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "rename-independent-subregs", 26)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "remove-redundant-debug-values", 29)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "require-all-machine-function-properties", 39)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "stack-coloring", 14)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "stack-slot-coloring", 19)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "tailduplication", 15)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "trigger-verifier-error", 22)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "two-address-instruction", 23)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "verify", 6)
      || sub_9691B0((const void *)v48.m128i_i64[0], v48.m128i_u64[1], "verify<machine-trace-metrics>", 29)
      || (unsigned __int8)sub_2337DE0((char *)v48.m128i_i64[0], v48.m128i_i64[1], "machine-sink", 0xCu)
      || (unsigned __int8)sub_2337DE0((char *)v48.m128i_i64[0], v48.m128i_i64[1], "regallocfast", 0xCu)
      || (v27 = v48, (unsigned __int8)sub_2337DE0((char *)v48.m128i_i64[0], v48.m128i_i64[1], "greedy", 6u)) )
    {
LABEL_189:
      v59.m128i_i64[1] = 8;
      v59.m128i_i64[0] = (__int64)"function";
      v57.m128i_i64[0] = (__int64)"machine-function";
      v57.m128i_i64[1] = 16;
      goto LABEL_26;
    }
    v29 = *(unsigned int *)(a2 + 2136);
    if ( (_DWORD)v29 )
    {
      v27.m128i_i64[0] = *(_QWORD *)(a2 + 2128);
      v59 = 0u;
      v30 = (__int64 *)&v50;
      v60 = 0;
      v61 = 0;
      v62 = 0;
      v42 = v27.m128i_i64[0] + 32 * v29;
      while ( v42 != v27.m128i_i64[0] )
      {
        v50 = 0;
        v49 = v48;
        v51 = 0;
        if ( !*(_QWORD *)(v27.m128i_i64[0] + 16) )
LABEL_190:
          sub_4263D6(v27.m128i_i64[0], v27.m128i_i64[1], v28);
        v41 = v30;
        v27.m128i_i64[1] = (__int64)&v49;
        v31 = (*(__int64 (__fastcall **)(__int64, __m128i *, __m128i *))(v27.m128i_i64[0] + 24))(
                v27.m128i_i64[0],
                &v49,
                &v59);
        v30 = v41;
        v27.m128i_i64[0] += 32;
        if ( v31 )
        {
          v43 = v59.m128i_i64[1];
          for ( i = (_QWORD **)v59.m128i_i64[0]; (_QWORD **)v43 != i; ++i )
          {
            if ( *i )
              (*(void (__fastcall **)(_QWORD, __m128i *, _QWORD, __int64 *))(**i + 8LL))(*i, &v49, **i, v30);
          }
          if ( v59.m128i_i64[0] )
            j_j___libc_free_0(v59.m128i_u64[0]);
          goto LABEL_189;
        }
      }
      v46 = v59.m128i_i64[1];
      for ( j = (_QWORD **)v59.m128i_i64[0]; (_QWORD **)v46 != j; ++j )
      {
        if ( *j )
          (*(void (__fastcall **)(_QWORD, __int64, _QWORD, __int64 *))(**j + 8LL))(*j, v27.m128i_i64[1], **j, v30);
      }
      v27.m128i_i64[0] = v59.m128i_i64[0];
      v27.m128i_i64[1] = (__int64)v60 - v59.m128i_i64[0];
      if ( v59.m128i_i64[0] )
        j_j___libc_free_0(v59.m128i_u64[0]);
    }
    v33 = *(_QWORD *)(a2 + 1408);
    v44 = v33 + 32LL * *(unsigned int *)(a2 + 1416);
    if ( v44 != v33 )
    {
      while ( 1 )
      {
        v28 = v53;
        v57.m128i_i64[0] = (__int64)v53;
        v57.m128i_i64[1] = 0xCCCCCCCCCCCCCCCDLL * (((char *)v54 - (char *)v53) >> 3);
        if ( !*(_QWORD *)(v33 + 16) )
          goto LABEL_190;
        v27.m128i_i64[1] = a3;
        v27.m128i_i64[0] = v33;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, __m128i *))(v33 + 24))(v33, a3, &v57) )
          break;
        v33 += 32;
        if ( v33 == v44 )
          goto LABEL_184;
      }
      *a1 = 1;
      v59.m128i_i64[0] = 0;
      sub_9C66B0(v59.m128i_i64);
LABEL_14:
      if ( !v56 )
        return a1;
      goto LABEL_11;
    }
LABEL_184:
    v34 = v53;
    v35 = sub_C63BB0();
    v45 = v36;
    v37 = v35;
    v38 = v34[1].m128i_i64[1] == v34[1].m128i_i64[0];
    v39 = "pass";
    v59.m128i_i64[0] = (__int64)"unknown {0} name '{1}'";
    if ( !v38 )
      v39 = "pipeline";
    v61 = 2;
    v60 = v66;
    v65[1] = v39;
    v66[0] = v65;
    v63 = &unk_49DB108;
    v64 = &v48;
    v59.m128i_i64[1] = 22;
    LOBYTE(v62) = 1;
    v65[0] = &unk_49E6678;
    v66[1] = &v63;
    sub_23328D0((__int64)&v57, (__int64)&v59);
    v20 = v45;
    v19 = v37;
LABEL_13:
    sub_23058C0(a1, (__int64)&v57, v19, v20);
    sub_2240A30((unsigned __int64 *)&v57);
    goto LABEL_14;
  }
  v59.m128i_i64[1] = 5;
  v59.m128i_i64[0] = (__int64)"cgscc";
LABEL_6:
  v8 = v53;
  v53 = 0;
  v60 = (__int64 *)v8;
  v9 = v54;
  v54 = 0;
  v61 = (__int64)v9;
  v10 = v55;
  v55 = 0;
  v62 = v10;
  sub_2366A40((__int64 *)&v50, &v59, 1);
  if ( v56 )
  {
    v23 = v50;
    v24 = v53;
    v50 = 0;
    v25 = (__int64)v54;
    v26 = v55;
    v53 = v23;
    v57.m128i_i64[0] = (__int64)v24;
    v54 = v51;
    v57.m128i_i64[1] = v25;
    v55 = v52;
    v58[0] = v26;
    v51 = 0;
    v52 = 0;
    sub_234A6B0((unsigned __int64 *)&v57);
  }
  else
  {
    v11 = v50;
    v56 = 1;
    v50 = 0;
    v53 = v11;
    v12 = v51;
    v51 = 0;
    v54 = v12;
    v13 = v52;
    v52 = 0;
    v55 = v13;
  }
  sub_234A6B0((unsigned __int64 *)&v50);
  sub_234A6B0((unsigned __int64 *)&v60);
LABEL_9:
  sub_2394660((unsigned __int64 *)&v59, a2, a3, (__int64)v53, 0xCCCCCCCCCCCCCCCDLL * (((char *)v54 - (char *)v53) >> 3));
  v14 = v59.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v59.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v59.m128i_i64[0] = 0;
    *a1 = v14 | 1;
    sub_9C66B0(v59.m128i_i64);
    goto LABEL_14;
  }
  v59.m128i_i64[0] = 0;
  sub_9C66B0(v59.m128i_i64);
  *a1 = 1;
  sub_9C66B0(v59.m128i_i64);
  if ( v56 )
  {
LABEL_11:
    v56 = 0;
    sub_234A6B0((unsigned __int64 *)&v53);
  }
  return a1;
}
