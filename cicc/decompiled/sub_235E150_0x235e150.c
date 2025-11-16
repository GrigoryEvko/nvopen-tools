// Function: sub_235E150
// Address: 0x235e150
//
__int64 *__fastcall sub_235E150(__int64 *a1, _QWORD *a2, unsigned __int64 *a3, const __m128i *a4)
{
  __int64 v5; // rax
  unsigned int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // r13
  unsigned int v9; // r14d
  __int64 v10; // rax
  __int64 v11; // rbx
  _QWORD *v15; // rax
  _QWORD *v16; // rax
  _QWORD *v17; // rax
  _QWORD *v18; // rax
  _QWORD *v19; // rax
  _QWORD *v20; // rax
  _QWORD *v21; // rax
  _QWORD *v22; // rax
  _QWORD *v23; // rax
  _QWORD *v24; // rax
  char v25; // dl
  __int8 v26; // bl
  __int64 v27; // rax
  __m128i v28; // xmm3
  __m128i v29; // rdi
  __int64 v30; // r15
  __int64 v31; // rbx
  __int64 v32; // rdx
  __m128i v33; // xmm6
  __int64 v34; // rax
  char v35; // al
  _QWORD *v36; // rax
  _QWORD *v37; // rax
  unsigned int v38; // eax
  unsigned int v39; // ebx
  __int64 v40; // rdx
  __int64 v41; // r14
  __m128i v42; // xmm0
  __m128i v43; // xmm5
  __int64 v44; // rdx
  __m128i v45; // xmm6
  char v46; // dl
  __int64 v47; // rbx
  __m128i v48; // xmm0
  __m128i v49; // xmm7
  __m128i v50; // xmm2
  __int64 v51; // rax
  __int64 v52; // rax
  __m128i v53; // xmm3
  __m128i v54; // xmm4
  void (__fastcall *v55)(__m128i *, __m128i *, __int64); // rax
  unsigned int v56; // eax
  unsigned int v57; // ebx
  __int64 v58; // rdx
  void *v59; // rbx
  _QWORD *v60; // rax
  void *v61; // rbx
  _QWORD *v62; // rax
  void *v63; // rbx
  _QWORD *v64; // rax
  void *v65; // rbx
  _QWORD *v66; // rax
  void *v67; // rbx
  _QWORD *v68; // rax
  void *v69; // rbx
  _QWORD *v70; // rax
  _QWORD *v71; // rax
  _QWORD *v72; // rax
  __int64 v73; // r14
  __int64 v74; // rbx
  _QWORD *v75; // rax
  _QWORD *v76; // rax
  _QWORD *v77; // rax
  _QWORD *v78; // rax
  _QWORD *v79; // rax
  _QWORD *v80; // rax
  _QWORD *v81; // rax
  _QWORD *v82; // rax
  __int64 v83; // r14
  _QWORD *v84; // rax
  _QWORD *v85; // rax
  __int64 v86; // rax
  _QWORD *v87; // rax
  _QWORD *v88; // rax
  _QWORD *v89; // rax
  _QWORD *v90; // rax
  _QWORD *v91; // rax
  _QWORD *v92; // rax
  _QWORD *v93; // rax
  _QWORD *v94; // rax
  _QWORD *v95; // rax
  _QWORD *v96; // rax
  _QWORD *v97; // rax
  _QWORD *v98; // rax
  _QWORD *v99; // rax
  _QWORD *v100; // rax
  _QWORD *v101; // rax
  _QWORD *v102; // rax
  _QWORD *v103; // rax
  _QWORD *v104; // rax
  _QWORD *v105; // rax
  _QWORD *v106; // rax
  _QWORD *v107; // rax
  _QWORD *v108; // rax
  void *v109; // rbx
  _QWORD *v110; // rax
  void *v111; // rbx
  _QWORD *v112; // rax
  void *v113; // rbx
  _QWORD *v114; // rax
  void *v115; // rbx
  _QWORD *v116; // rax
  void *v117; // rbx
  _QWORD *v118; // rax
  void *v119; // rbx
  _QWORD *v120; // rax
  void *v121; // rbx
  _QWORD *v122; // rax
  _QWORD *v123; // rax
  _QWORD *v124; // rax
  _QWORD *v125; // rax
  _QWORD *v126; // rax
  _QWORD *v127; // rax
  _QWORD *v128; // rax
  _QWORD *v129; // rax
  _QWORD *v130; // rax
  _QWORD *v131; // rax
  _QWORD *v132; // rax
  _QWORD *v133; // rax
  _QWORD *v134; // rax
  unsigned int v135; // eax
  __int64 v136; // rdx
  char v137; // dl
  unsigned int v138; // eax
  __int64 v139; // rdx
  __m128i v140; // xmm6
  __int64 v141; // rdx
  __m128i v142; // xmm7
  __m128i v143; // xmm2
  __m128i v144; // xmm3
  void (__fastcall *v145)(__m128i *, __m128i *, __int64); // rax
  __int64 v146; // rax
  __m128i v147; // xmm0
  __m128i v148; // xmm4
  __int64 v149; // rax
  __int64 v150; // [rsp+10h] [rbp-1B0h]
  __int64 v151; // [rsp+10h] [rbp-1B0h]
  __int64 v153; // [rsp+18h] [rbp-1A8h]
  __int64 v154; // [rsp+18h] [rbp-1A8h]
  unsigned int v155; // [rsp+18h] [rbp-1A8h]
  unsigned int v156; // [rsp+18h] [rbp-1A8h]
  __int64 v157; // [rsp+28h] [rbp-198h] BYREF
  __m128i v158; // [rsp+30h] [rbp-190h] BYREF
  __m128i v159; // [rsp+40h] [rbp-180h] BYREF
  __m128i v160; // [rsp+50h] [rbp-170h] BYREF
  __m128i v161; // [rsp+60h] [rbp-160h] BYREF
  __m128i v162; // [rsp+70h] [rbp-150h] BYREF
  __m128i v163; // [rsp+90h] [rbp-130h] BYREF
  __m128i v164; // [rsp+A0h] [rbp-120h]
  __m128i v165; // [rsp+B0h] [rbp-110h] BYREF
  char v166; // [rsp+C0h] [rbp-100h]
  __m128i v167; // [rsp+D0h] [rbp-F0h] BYREF
  __m128i v168; // [rsp+E0h] [rbp-E0h]
  __m128i v169; // [rsp+F0h] [rbp-D0h] BYREF
  char v170; // [rsp+100h] [rbp-C0h]
  __m128i v171; // [rsp+110h] [rbp-B0h] BYREF
  unsigned __int128 v172; // [rsp+120h] [rbp-A0h] BYREF
  __m128i v173; // [rsp+130h] [rbp-90h] BYREF
  char v174; // [rsp+140h] [rbp-80h]
  char v175; // [rsp+148h] [rbp-78h]
  __m128i v176; // [rsp+150h] [rbp-70h] BYREF
  __m128i v177; // [rsp+160h] [rbp-60h] BYREF
  __m128i v178; // [rsp+170h] [rbp-50h] BYREF
  __m128i *v179; // [rsp+180h] [rbp-40h]
  _QWORD v180[7]; // [rsp+188h] [rbp-38h] BYREF

  v5 = a4[1].m128i_i64[1];
  v158 = _mm_loadu_si128(a4);
  if ( a4[1].m128i_i64[0] != v5 )
  {
    v6 = sub_C63BB0();
    v176.m128i_i64[0] = (__int64)"invalid pipeline";
    v8 = v7;
    v9 = v6;
    v178.m128i_i16[0] = 259;
    v10 = sub_22077B0(0x40u);
    v11 = v10;
    if ( v10 )
      sub_C63EB0(v10, (__int64)&v176, v9, v8);
    *a1 = v11 | 1;
    return a1;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "require<edge-bundles>", 21) )
  {
    v16 = (_QWORD *)sub_22077B0(0x10u);
    if ( v16 )
      *v16 = &unk_4A140F8;
    goto LABEL_142;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "invalidate<edge-bundles>", 24) )
  {
    v16 = (_QWORD *)sub_22077B0(0x10u);
    if ( v16 )
      *v16 = &unk_4A14138;
    goto LABEL_142;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "require<livedebugvars>", 22) )
  {
    v16 = (_QWORD *)sub_22077B0(0x10u);
    if ( v16 )
      *v16 = &unk_4A14178;
    goto LABEL_142;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "invalidate<livedebugvars>", 25) )
  {
    v16 = (_QWORD *)sub_22077B0(0x10u);
    if ( v16 )
      *v16 = &unk_4A141B8;
    goto LABEL_142;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "require<live-intervals>", 23) )
  {
    v16 = (_QWORD *)sub_22077B0(0x10u);
    if ( v16 )
      *v16 = &unk_4A141F8;
    goto LABEL_142;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "invalidate<live-intervals>", 26) )
  {
    v16 = (_QWORD *)sub_22077B0(0x10u);
    if ( v16 )
      *v16 = &unk_4A14238;
    goto LABEL_142;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "require<live-reg-matrix>", 24) )
  {
    v16 = (_QWORD *)sub_22077B0(0x10u);
    if ( v16 )
      *v16 = &unk_4A14278;
    goto LABEL_142;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "invalidate<live-reg-matrix>", 27) )
  {
    v16 = (_QWORD *)sub_22077B0(0x10u);
    if ( v16 )
      *v16 = &unk_4A142B8;
    goto LABEL_142;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "require<live-stacks>", 20) )
  {
    v16 = (_QWORD *)sub_22077B0(0x10u);
    if ( v16 )
      *v16 = &unk_4A142F8;
    goto LABEL_142;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "invalidate<live-stacks>", 23) )
  {
    v16 = (_QWORD *)sub_22077B0(0x10u);
    if ( v16 )
      *v16 = &unk_4A14338;
    goto LABEL_142;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "require<live-vars>", 18) )
  {
    v16 = (_QWORD *)sub_22077B0(0x10u);
    if ( v16 )
      *v16 = &unk_4A14378;
    goto LABEL_142;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "invalidate<live-vars>", 21) )
  {
    v16 = (_QWORD *)sub_22077B0(0x10u);
    if ( v16 )
      *v16 = &unk_4A143B8;
    goto LABEL_142;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "require<machine-block-freq>", 27) )
  {
    v16 = (_QWORD *)sub_22077B0(0x10u);
    if ( v16 )
      *v16 = &unk_4A143F8;
    goto LABEL_142;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "invalidate<machine-block-freq>", 30) )
  {
    v16 = (_QWORD *)sub_22077B0(0x10u);
    if ( v16 )
      *v16 = &unk_4A14438;
    goto LABEL_142;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "require<machine-branch-prob>", 28) )
  {
    v16 = (_QWORD *)sub_22077B0(0x10u);
    if ( v16 )
      *v16 = &unk_4A14478;
LABEL_142:
    v176.m128i_i64[0] = (__int64)v16;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    sub_233F0A0(v176.m128i_i64);
    *a1 = 1;
    v176.m128i_i64[0] = 0;
    sub_9C66B0(v176.m128i_i64);
    return a1;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "invalidate<machine-branch-prob>", 31) )
  {
    v17 = (_QWORD *)sub_22077B0(0x10u);
    if ( v17 )
      *v17 = &unk_4A144B8;
    v176.m128i_i64[0] = (__int64)v17;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "require<machine-cycles>", 23) )
  {
    v18 = (_QWORD *)sub_22077B0(0x10u);
    if ( v18 )
      *v18 = &unk_4A144F8;
    v176.m128i_i64[0] = (__int64)v18;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "invalidate<machine-cycles>", 26) )
  {
    v19 = (_QWORD *)sub_22077B0(0x10u);
    if ( v19 )
      *v19 = &unk_4A14538;
    v176.m128i_i64[0] = (__int64)v19;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "require<machine-dom-tree>", 25) )
  {
    v20 = (_QWORD *)sub_22077B0(0x10u);
    if ( v20 )
      *v20 = &unk_4A14578;
    v176.m128i_i64[0] = (__int64)v20;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "invalidate<machine-dom-tree>", 28) )
  {
    v22 = (_QWORD *)sub_22077B0(0x10u);
    if ( v22 )
      *v22 = &unk_4A145B8;
    v176.m128i_i64[0] = (__int64)v22;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "require<machine-loops>", 22) )
  {
    v21 = (_QWORD *)sub_22077B0(0x10u);
    if ( v21 )
      *v21 = &unk_4A145F8;
    v176.m128i_i64[0] = (__int64)v21;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "invalidate<machine-loops>", 25) )
  {
    v24 = (_QWORD *)sub_22077B0(0x10u);
    if ( v24 )
      *v24 = &unk_4A14638;
    v176.m128i_i64[0] = (__int64)v24;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "require<machine-opt-remark-emitter>", 35) )
  {
    v23 = (_QWORD *)sub_22077B0(0x10u);
    if ( v23 )
      *v23 = &unk_4A14678;
    v176.m128i_i64[0] = (__int64)v23;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "invalidate<machine-opt-remark-emitter>", 38) )
  {
    v134 = (_QWORD *)sub_22077B0(0x10u);
    if ( v134 )
      *v134 = &unk_4A146B8;
    v176.m128i_i64[0] = (__int64)v134;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "require<machine-post-dom-tree>", 30) )
  {
    v133 = (_QWORD *)sub_22077B0(0x10u);
    if ( v133 )
      *v133 = &unk_4A146F8;
    v176.m128i_i64[0] = (__int64)v133;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "invalidate<machine-post-dom-tree>", 33) )
  {
    v132 = (_QWORD *)sub_22077B0(0x10u);
    if ( v132 )
      *v132 = &unk_4A14738;
    v176.m128i_i64[0] = (__int64)v132;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "require<machine-trace-metrics>", 30) )
  {
    v131 = (_QWORD *)sub_22077B0(0x10u);
    if ( v131 )
      *v131 = &unk_4A14778;
    v176.m128i_i64[0] = (__int64)v131;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "invalidate<machine-trace-metrics>", 33) )
  {
    v130 = (_QWORD *)sub_22077B0(0x10u);
    if ( v130 )
      *v130 = &unk_4A147B8;
    v176.m128i_i64[0] = (__int64)v130;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "require<pass-instrumentation>", 29) )
  {
    v129 = (_QWORD *)sub_22077B0(0x10u);
    if ( v129 )
      *v129 = &unk_4A147F8;
    v176.m128i_i64[0] = (__int64)v129;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "invalidate<pass-instrumentation>", 32) )
  {
    v128 = (_QWORD *)sub_22077B0(0x10u);
    if ( v128 )
      *v128 = &unk_4A14838;
    v176.m128i_i64[0] = (__int64)v128;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "require<regalloc-evict>", 23) )
  {
    v127 = (_QWORD *)sub_22077B0(0x10u);
    if ( v127 )
      *v127 = &unk_4A14878;
    v176.m128i_i64[0] = (__int64)v127;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "invalidate<regalloc-evict>", 26) )
  {
    v126 = (_QWORD *)sub_22077B0(0x10u);
    if ( v126 )
      *v126 = &unk_4A148B8;
    v176.m128i_i64[0] = (__int64)v126;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "require<regalloc-priority>", 26) )
  {
    v125 = (_QWORD *)sub_22077B0(0x10u);
    if ( v125 )
      *v125 = &unk_4A148F8;
    v176.m128i_i64[0] = (__int64)v125;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "invalidate<regalloc-priority>", 29) )
  {
    v124 = (_QWORD *)sub_22077B0(0x10u);
    if ( v124 )
      *v124 = &unk_4A14938;
    v176.m128i_i64[0] = (__int64)v124;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "require<slot-indexes>", 21) )
  {
    v123 = (_QWORD *)sub_22077B0(0x10u);
    if ( v123 )
      *v123 = &unk_4A14978;
    v176.m128i_i64[0] = (__int64)v123;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "invalidate<slot-indexes>", 24) )
  {
    v99 = (_QWORD *)sub_22077B0(0x10u);
    if ( v99 )
      *v99 = &unk_4A149B8;
    v176.m128i_i64[0] = (__int64)v99;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "require<spill-code-placement>", 29) )
  {
    v98 = (_QWORD *)sub_22077B0(0x10u);
    if ( v98 )
      *v98 = &unk_4A149F8;
    v176.m128i_i64[0] = (__int64)v98;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "invalidate<spill-code-placement>", 32) )
  {
    v97 = (_QWORD *)sub_22077B0(0x10u);
    if ( v97 )
      *v97 = &unk_4A14A38;
    v176.m128i_i64[0] = (__int64)v97;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "require<virtregmap>", 19) )
  {
    v96 = (_QWORD *)sub_22077B0(0x10u);
    if ( v96 )
      *v96 = &unk_4A14A78;
    v176.m128i_i64[0] = (__int64)v96;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "invalidate<virtregmap>", 22) )
  {
    v95 = (_QWORD *)sub_22077B0(0x10u);
    if ( v95 )
      *v95 = &unk_4A14AB8;
    v176.m128i_i64[0] = (__int64)v95;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "dead-mi-elimination", 19) )
  {
    v94 = (_QWORD *)sub_22077B0(0x10u);
    if ( v94 )
      *v94 = &unk_4A14AF8;
    v176.m128i_i64[0] = (__int64)v94;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "detect-dead-lanes", 17) )
  {
    v93 = (_QWORD *)sub_22077B0(0x10u);
    if ( v93 )
      *v93 = &unk_4A14B38;
    v176.m128i_i64[0] = (__int64)v93;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "early-ifcvt", 11) )
  {
    v92 = (_QWORD *)sub_22077B0(0x10u);
    if ( v92 )
      *v92 = &unk_4A14B78;
    v176.m128i_i64[0] = (__int64)v92;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "early-machinelicm", 17) )
  {
    v91 = (_QWORD *)sub_22077B0(0x10u);
    if ( v91 )
      *v91 = &unk_4A14BB8;
    v176.m128i_i64[0] = (__int64)v91;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "early-tailduplication", 21) )
  {
    v90 = (_QWORD *)sub_22077B0(0x10u);
    if ( v90 )
    {
      v90[1] = 0;
      *v90 = &unk_4A14BF8;
    }
    v176.m128i_i64[0] = (__int64)v90;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "finalize-isel", 13) )
  {
    v89 = (_QWORD *)sub_22077B0(0x10u);
    if ( v89 )
      *v89 = &unk_4A14C38;
    v176.m128i_i64[0] = (__int64)v89;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "fixup-statepoint-caller-saved", 29) )
  {
    v88 = (_QWORD *)sub_22077B0(0x10u);
    if ( v88 )
      *v88 = &unk_4A14C78;
    v176.m128i_i64[0] = (__int64)v88;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "localstackalloc", 15) )
  {
    v87 = (_QWORD *)sub_22077B0(0x10u);
    if ( v87 )
      *v87 = &unk_4A14CB8;
    v176.m128i_i64[0] = (__int64)v87;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "machine-cp", 10) )
  {
    v86 = sub_22077B0(0x10u);
    if ( v86 )
    {
      *(_BYTE *)(v86 + 8) = 0;
      *(_QWORD *)v86 = &unk_4A14CF8;
    }
    v176.m128i_i64[0] = v86;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "machine-cse", 11) )
  {
    v85 = (_QWORD *)sub_22077B0(0x10u);
    if ( v85 )
      *v85 = &unk_4A14D38;
    v176.m128i_i64[0] = (__int64)v85;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "machine-latecleanup", 19) )
  {
    v84 = (_QWORD *)sub_22077B0(0x10u);
    if ( v84 )
      *v84 = &unk_4A14D78;
    v176.m128i_i64[0] = (__int64)v84;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "machine-scheduler", 17) )
  {
    sub_2EC58D0(&v171, *a2);
    sub_2EC5950(&v176, &v171);
    v82 = (_QWORD *)sub_22077B0(0x18u);
    v83 = (__int64)v82;
    if ( v82 )
    {
      *v82 = &unk_4A14DB8;
      sub_2EC5950(v82 + 1, &v176);
    }
    v167.m128i_i64[0] = v83;
    sub_235DE40(a3, (unsigned __int64 *)&v167);
    if ( v167.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v167.m128i_i64[0] + 8LL))(v167.m128i_i64[0]);
    sub_2EC5930(&v176);
    sub_2EC5930(&v171);
    *a1 = 1;
    v176.m128i_i64[0] = 0;
    sub_9C66B0(v176.m128i_i64);
    return a1;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "machinelicm", 11) )
  {
    v81 = (_QWORD *)sub_22077B0(0x10u);
    if ( v81 )
      *v81 = &unk_4A14DF8;
    v176.m128i_i64[0] = (__int64)v81;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "no-op-machine-function", 22) )
  {
    v80 = (_QWORD *)sub_22077B0(0x10u);
    if ( v80 )
      *v80 = &unk_4A14E38;
    v176.m128i_i64[0] = (__int64)v80;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "opt-phis", 8) )
  {
    v79 = (_QWORD *)sub_22077B0(0x10u);
    if ( v79 )
      *v79 = &unk_4A14E78;
    v176.m128i_i64[0] = (__int64)v79;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "patchable-function", 18) )
  {
    v78 = (_QWORD *)sub_22077B0(0x10u);
    if ( v78 )
      *v78 = &unk_4A14EB8;
    v176.m128i_i64[0] = (__int64)v78;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "peephole-opt", 12) )
  {
    v77 = (_QWORD *)sub_22077B0(0x10u);
    if ( v77 )
      *v77 = &unk_4A14EF8;
    v176.m128i_i64[0] = (__int64)v77;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "phi-node-elimination", 20) )
  {
    v76 = (_QWORD *)sub_22077B0(0x10u);
    if ( v76 )
      *v76 = &unk_4A14F38;
    v176.m128i_i64[0] = (__int64)v76;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "post-RA-sched", 13) )
  {
    v74 = *a2;
    v75 = (_QWORD *)sub_22077B0(0x10u);
    if ( v75 )
    {
      v75[1] = v74;
      *v75 = &unk_4A14F78;
    }
    v176.m128i_i64[0] = (__int64)v75;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "postmisched", 11) )
  {
    sub_2EC5970(&v171, *a2);
    sub_2EC59D0(&v176, &v171);
    v72 = (_QWORD *)sub_22077B0(0x18u);
    v73 = (__int64)v72;
    if ( v72 )
    {
      *v72 = &unk_4A14FB8;
      sub_2EC59D0(v72 + 1, &v176);
    }
    v167.m128i_i64[0] = v73;
    sub_235DE40(a3, (unsigned __int64 *)&v167);
    if ( v167.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v167.m128i_i64[0] + 8LL))(v167.m128i_i64[0]);
    sub_2EC59F0(&v176);
    sub_2EC59F0(&v171);
    *a1 = 1;
    v176.m128i_i64[0] = 0;
    sub_9C66B0(v176.m128i_i64);
    return a1;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "post-ra-pseudos", 15) )
  {
    v71 = (_QWORD *)sub_22077B0(0x10u);
    if ( v71 )
      *v71 = &unk_4A14FF8;
    v176.m128i_i64[0] = (__int64)v71;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "print", 5) )
  {
    v69 = sub_CB72A0();
    v70 = (_QWORD *)sub_22077B0(0x10u);
    if ( v70 )
    {
      v70[1] = v69;
      *v70 = &unk_4A15038;
    }
    v176.m128i_i64[0] = (__int64)v70;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "print<livedebugvars>", 20) )
  {
    v67 = sub_CB72A0();
    v68 = (_QWORD *)sub_22077B0(0x10u);
    if ( v68 )
    {
      v68[1] = v67;
      *v68 = &unk_4A15078;
    }
    v176.m128i_i64[0] = (__int64)v68;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "print<live-intervals>", 21) )
  {
    v65 = sub_CB72A0();
    v66 = (_QWORD *)sub_22077B0(0x10u);
    if ( v66 )
    {
      v66[1] = v65;
      *v66 = &unk_4A150B8;
    }
    v176.m128i_i64[0] = (__int64)v66;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "print<live-stacks>", 18) )
  {
    v63 = sub_CB72A0();
    v64 = (_QWORD *)sub_22077B0(0x10u);
    if ( v64 )
    {
      v64[1] = v63;
      *v64 = &unk_4A150F8;
    }
    v176.m128i_i64[0] = (__int64)v64;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "print<live-vars>", 16) )
  {
    v61 = sub_CB72A0();
    v62 = (_QWORD *)sub_22077B0(0x10u);
    if ( v62 )
    {
      v62[1] = v61;
      *v62 = &unk_4A15138;
    }
    v176.m128i_i64[0] = (__int64)v62;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "print<machine-block-freq>", 25) )
  {
    v59 = sub_CB72A0();
    v60 = (_QWORD *)sub_22077B0(0x10u);
    if ( v60 )
    {
      v60[1] = v59;
      *v60 = &unk_4A15178;
    }
    v176.m128i_i64[0] = (__int64)v60;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "print<machine-branch-prob>", 26) )
  {
    v121 = sub_CB72A0();
    v122 = (_QWORD *)sub_22077B0(0x10u);
    if ( v122 )
    {
      v122[1] = v121;
      *v122 = &unk_4A151B8;
    }
    v176.m128i_i64[0] = (__int64)v122;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "print<machine-cycles>", 21) )
  {
    v119 = sub_CB72A0();
    v120 = (_QWORD *)sub_22077B0(0x10u);
    if ( v120 )
    {
      v120[1] = v119;
      *v120 = &unk_4A151F8;
    }
    v176.m128i_i64[0] = (__int64)v120;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "print<machine-dom-tree>", 23) )
  {
    v117 = sub_CB72A0();
    v118 = (_QWORD *)sub_22077B0(0x10u);
    if ( v118 )
    {
      v118[1] = v117;
      *v118 = &unk_4A15238;
    }
    v176.m128i_i64[0] = (__int64)v118;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "print<machine-loops>", 20) )
  {
    v115 = sub_CB72A0();
    v116 = (_QWORD *)sub_22077B0(0x10u);
    if ( v116 )
    {
      v116[1] = v115;
      *v116 = &unk_4A15278;
    }
    v176.m128i_i64[0] = (__int64)v116;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "print<machine-post-dom-tree>", 28) )
  {
    v113 = sub_CB72A0();
    v114 = (_QWORD *)sub_22077B0(0x10u);
    if ( v114 )
    {
      v114[1] = v113;
      *v114 = &unk_4A152B8;
    }
    v176.m128i_i64[0] = (__int64)v114;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "print<slot-indexes>", 19) )
  {
    v111 = sub_CB72A0();
    v112 = (_QWORD *)sub_22077B0(0x10u);
    if ( v112 )
    {
      v112[1] = v111;
      *v112 = &unk_4A152F8;
    }
    v176.m128i_i64[0] = (__int64)v112;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "print<virtregmap>", 17) )
  {
    v109 = sub_CB72A0();
    v110 = (_QWORD *)sub_22077B0(0x10u);
    if ( v110 )
    {
      v110[1] = v109;
      *v110 = &unk_4A15338;
    }
    v176.m128i_i64[0] = (__int64)v110;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "reg-usage-collector", 19) )
  {
    v108 = (_QWORD *)sub_22077B0(0x10u);
    if ( v108 )
      *v108 = &unk_4A15378;
    v176.m128i_i64[0] = (__int64)v108;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "reg-usage-propagation", 21) )
  {
    v107 = (_QWORD *)sub_22077B0(0x10u);
    if ( v107 )
      *v107 = &unk_4A153B8;
    v176.m128i_i64[0] = (__int64)v107;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "register-coalescer", 18) )
  {
    v106 = (_QWORD *)sub_22077B0(0x10u);
    if ( v106 )
      *v106 = &unk_4A153F8;
    v176.m128i_i64[0] = (__int64)v106;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "rename-independent-subregs", 26) )
  {
    v105 = (_QWORD *)sub_22077B0(0x10u);
    if ( v105 )
      *v105 = &unk_4A15438;
    v176.m128i_i64[0] = (__int64)v105;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "remove-redundant-debug-values", 29) )
  {
    v104 = (_QWORD *)sub_22077B0(0x10u);
    if ( v104 )
      *v104 = &unk_4A15478;
    v176.m128i_i64[0] = (__int64)v104;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "require-all-machine-function-properties", 39) )
  {
    v103 = (_QWORD *)sub_22077B0(0x10u);
    if ( v103 )
      *v103 = off_49D2AE8;
    v176.m128i_i64[0] = (__int64)v103;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "stack-coloring", 14) )
  {
    v102 = (_QWORD *)sub_22077B0(0x10u);
    if ( v102 )
      *v102 = &unk_4A154B8;
    v176.m128i_i64[0] = (__int64)v102;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "stack-slot-coloring", 19) )
  {
    v101 = (_QWORD *)sub_22077B0(0x10u);
    if ( v101 )
      *v101 = &unk_4A154F8;
    v176.m128i_i64[0] = (__int64)v101;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "tailduplication", 15) )
  {
    v100 = (_QWORD *)sub_22077B0(0x10u);
    if ( v100 )
    {
      v100[1] = 0;
      *v100 = &unk_4A15538;
    }
    v176.m128i_i64[0] = (__int64)v100;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "trigger-verifier-error", 22) )
  {
    v37 = (_QWORD *)sub_22077B0(0x10u);
    if ( v37 )
      *v37 = off_49D2B28;
    v176.m128i_i64[0] = (__int64)v37;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "two-address-instruction", 23) )
  {
    v36 = (_QWORD *)sub_22077B0(0x10u);
    if ( v36 )
      *v36 = &unk_4A15578;
    v176.m128i_i64[0] = (__int64)v36;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
    goto LABEL_547;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "verify", 6) )
  {
    LOBYTE(v172) = 0;
    v171 = (__m128i)(unsigned __int64)&v172;
    sub_2241BD0(v176.m128i_i64, (__int64)&v171);
    sub_235E050(a3, &v176);
    sub_2240A30((unsigned __int64 *)&v176);
    sub_2240A30((unsigned __int64 *)&v171);
    *a1 = 1;
    v176.m128i_i64[0] = 0;
    sub_9C66B0(v176.m128i_i64);
    return a1;
  }
  if ( sub_9691B0((const void *)v158.m128i_i64[0], v158.m128i_u64[1], "verify<machine-trace-metrics>", 29) )
  {
    v15 = (_QWORD *)sub_22077B0(0x10u);
    if ( v15 )
      *v15 = &unk_4A155F8;
    v176.m128i_i64[0] = (__int64)v15;
    sub_235DE40(a3, (unsigned __int64 *)&v176);
    if ( v176.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v176.m128i_i64[0] + 8LL))(v176.m128i_i64[0]);
LABEL_547:
    *a1 = 1;
    v176.m128i_i64[0] = 0;
    sub_9C66B0(v176.m128i_i64);
    return a1;
  }
  if ( (unsigned __int8)sub_2337DE0((char *)v158.m128i_i64[0], v158.m128i_i64[1], "machine-sink", 0xCu) )
  {
    sub_234B110(
      (__int64)&v176,
      (void (__fastcall *)(__int64, const void *, __int64))sub_233A830,
      (const void *)v158.m128i_i64[0],
      v158.m128i_i64[1],
      "machine-sink",
      0xCu);
    v25 = v176.m128i_i8[8] & 1;
    v176.m128i_i8[8] = (2 * (v176.m128i_i8[8] & 1)) | v176.m128i_i8[8] & 0xFD;
    if ( v25 )
    {
      sub_234B1B0(a1, v176.m128i_i64);
    }
    else
    {
      v26 = v176.m128i_i8[0];
      v27 = sub_22077B0(0x10u);
      if ( v27 )
      {
        *(_BYTE *)(v27 + 8) = v26;
        *(_QWORD *)v27 = &unk_4A15638;
      }
      v171.m128i_i64[0] = v27;
      sub_235DE40(a3, (unsigned __int64 *)&v171);
      if ( v171.m128i_i64[0] )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v171.m128i_i64[0] + 8LL))(v171.m128i_i64[0]);
      *a1 = 1;
      v171.m128i_i64[0] = 0;
      sub_9C66B0(v171.m128i_i64);
    }
    sub_2351C10(&v176);
    return a1;
  }
  if ( (unsigned __int8)sub_2337DE0((char *)v158.m128i_i64[0], v158.m128i_i64[1], "regallocfast", 0xCu) )
  {
    v159 = v158;
    if ( (unsigned __int8)sub_95CB50((const void **)&v159, "regallocfast", 0xCu) )
    {
      if ( v159.m128i_i64[1]
        && (!(unsigned __int8)sub_95CB50((const void **)&v159, "<", 1u) || !(unsigned __int8)sub_232E070(&v159, ">", 1u)) )
      {
        BUG();
      }
      v177.m128i_i64[0] = 0;
      v170 = 1;
      v160 = v159;
      v168.m128i_i64[0] = 0;
      v169.m128i_i64[0] = (__int64)"all";
      v169.m128i_i64[1] = 3;
      sub_A17130((__int64)&v176);
      while ( 1 )
      {
        while ( 1 )
        {
          if ( !v160.m128i_i64[1] )
          {
            v140 = _mm_loadu_si128(&v171);
            v141 = *((_QWORD *)&v172 + 1);
            v142 = _mm_loadu_si128(&v169);
            v171 = _mm_loadu_si128(&v167);
            v167 = v140;
            v175 = v175 & 0xFC | 2;
            v173 = v142;
            v172 = (unsigned __int128)v168;
            v168.m128i_i64[0] = 0;
            v168.m128i_i64[1] = v141;
            v174 = v170;
            goto LABEL_549;
          }
          v171.m128i_i8[0] = 59;
          v161 = 0u;
          sub_232E160(&v176, &v160, &v171, 1u);
          v28 = _mm_loadu_si128(&v177);
          v161 = _mm_loadu_si128(&v176);
          v160 = v28;
          if ( (unsigned __int8)sub_95CB50((const void **)&v161, "filter=", 7u) )
            break;
          if ( !sub_9691B0((const void *)v161.m128i_i64[0], v161.m128i_u64[1], "no-clear-vregs", 14) )
          {
            v135 = sub_C63BB0();
            v176.m128i_i64[1] = 42;
            v150 = v136;
            v176.m128i_i64[0] = (__int64)"invalid regallocfast pass parameter '{0}' ";
            v177.m128i_i64[0] = (__int64)v180;
            v155 = v135;
            v177.m128i_i64[1] = 1;
            v178.m128i_i64[1] = (__int64)&unk_49DB108;
            v180[0] = &v178.m128i_i64[1];
            v178.m128i_i8[0] = 1;
            v179 = &v161;
            sub_23328D0((__int64)&v163, (__int64)&v176);
            sub_23058C0(v162.m128i_i64, (__int64)&v163, v155, v150);
            v175 |= 3u;
            v171.m128i_i64[0] = v162.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
            sub_2240A30((unsigned __int64 *)&v163);
            goto LABEL_549;
          }
          v170 = 0;
        }
        sub_233C300(&v163, (__int64)a2, v161.m128i_i64[0], v161.m128i_i64[1]);
        if ( !v165.m128i_i8[0] )
          break;
        sub_23425F0((__int64)&v176, (__int64)&v163);
        v146 = v177.m128i_i64[0];
        v147 = _mm_loadu_si128(&v176);
        v148 = _mm_loadu_si128(&v167);
        v177.m128i_i64[0] = v168.m128i_i64[0];
        v168.m128i_i64[0] = v146;
        v149 = v177.m128i_i64[1];
        v176 = v148;
        v177.m128i_i64[1] = v168.m128i_i64[1];
        v168.m128i_i64[1] = v149;
        v167 = v147;
        sub_A17130((__int64)&v176);
        v169 = _mm_loadu_si128(&v161);
        if ( v165.m128i_i8[0] )
        {
          v165.m128i_i8[0] = 0;
          sub_A17130((__int64)&v163);
        }
      }
      v138 = sub_C63BB0();
      v178.m128i_i8[0] = 1;
      v151 = v139;
      v176.m128i_i64[0] = (__int64)"invalid regallocfast register filter '{0}' ";
      v177.m128i_i64[0] = (__int64)v180;
      v156 = v138;
      v176.m128i_i64[1] = 43;
      v178.m128i_i64[1] = (__int64)&unk_49DB108;
      v180[0] = &v178.m128i_i64[1];
      v177.m128i_i64[1] = 1;
      v179 = &v161;
      sub_23328D0((__int64)&v162, (__int64)&v176);
      sub_23058C0(&v157, (__int64)&v162, v156, v151);
      v175 |= 3u;
      v171.m128i_i64[0] = v157 & 0xFFFFFFFFFFFFFFFELL;
      sub_2240A30((unsigned __int64 *)&v162);
      if ( v165.m128i_i8[0] )
      {
        v165.m128i_i8[0] = 0;
        sub_A17130((__int64)&v163);
      }
LABEL_549:
      sub_A17130((__int64)&v167);
      v137 = v175 & 1;
      v175 = (2 * (v175 & 1)) | v175 & 0xFD;
      if ( v137 )
      {
        *a1 = v171.m128i_i64[0] | 1;
      }
      else
      {
        sub_23425F0((__int64)&v163, (__int64)&v171);
        v143 = _mm_loadu_si128(&v173);
        v177.m128i_i64[0] = 0;
        v166 = v174;
        v165 = v143;
        if ( v164.m128i_i64[0] )
        {
          ((void (__fastcall *)(__m128i *, __m128i *, __int64))v164.m128i_i64[0])(&v176, &v163, 2);
          v177 = v164;
        }
        v144 = _mm_loadu_si128(&v165);
        v168.m128i_i64[0] = 0;
        LOBYTE(v179) = v166;
        v145 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v177.m128i_i64[0];
        v178 = v144;
        if ( v177.m128i_i64[0] )
        {
          ((void (__fastcall *)(__m128i *, __m128i *, __int64))v177.m128i_i64[0])(&v167, &v176, 2);
          v145 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v177.m128i_i64[0];
          v168 = v177;
        }
        v169 = _mm_loadu_si128(&v178);
        v170 = (char)v179;
        if ( v145 )
          v145(&v176, &v176, 3);
        sub_235DF60(a3, &v167);
        sub_A17130((__int64)&v167);
        sub_A17130((__int64)&v163);
        *a1 = 1;
        v176.m128i_i64[0] = 0;
        sub_9C66B0(v176.m128i_i64);
        if ( (v175 & 2) != 0 )
          sub_2352990(&v171, (__int64)&v167);
        if ( (v175 & 1) != 0 )
        {
          if ( v171.m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v171.m128i_i64[0] + 8LL))(v171.m128i_i64[0]);
        }
        else if ( (_QWORD)v172 )
        {
          ((void (__fastcall *)(__m128i *, __m128i *, __int64))v172)(&v171, &v171, 3);
        }
      }
      return a1;
    }
LABEL_571:
    BUG();
  }
  v29 = v158;
  if ( (unsigned __int8)sub_2337DE0((char *)v158.m128i_i64[0], v158.m128i_i64[1], "greedy", 6u) )
  {
    v161 = v158;
    if ( (unsigned __int8)sub_95CB50((const void **)&v161, "greedy", 6u) )
    {
      if ( v161.m128i_i64[1]
        && (!(unsigned __int8)sub_95CB50((const void **)&v161, "<", 1u) || !(unsigned __int8)sub_232E070(&v161, ">", 1u)) )
      {
        BUG();
      }
      v162 = v161;
      if ( v161.m128i_i64[1] && !sub_9691B0((const void *)v161.m128i_i64[0], v161.m128i_u64[1], "all", 3) )
      {
        sub_233C300(&v167, (__int64)a2, v162.m128i_i64[0], v162.m128i_i64[1]);
        if ( v169.m128i_i8[0] )
        {
          sub_23425F0((__int64)&v163, (__int64)&v167);
          v47 = v162.m128i_i64[1];
          v153 = v162.m128i_i64[0];
          sub_23425F0((__int64)&v176, (__int64)&v163);
          v178.m128i_i64[1] = v47;
          v48 = _mm_loadu_si128(&v176);
          v49 = _mm_loadu_si128(&v171);
          v178.m128i_i64[0] = v153;
          v50 = _mm_loadu_si128(&v178);
          v176 = v49;
          v171 = v48;
          v173 = v50;
          v174 = v174 & 0xFC | 2;
          v51 = v177.m128i_i64[0];
          v177.m128i_i64[0] = 0;
          *(_QWORD *)&v172 = v51;
          v52 = v177.m128i_i64[1];
          v177.m128i_i64[1] = *((_QWORD *)&v172 + 1);
          *((_QWORD *)&v172 + 1) = v52;
          sub_A17130((__int64)&v176);
          sub_A17130((__int64)&v163);
        }
        else
        {
          v56 = sub_C63BB0();
          v176.m128i_i64[1] = 45;
          v57 = v56;
          v176.m128i_i64[0] = (__int64)"invalid regallocgreedy register filter '{0}' ";
          v177.m128i_i64[0] = (__int64)v180;
          v154 = v58;
          v177.m128i_i64[1] = 1;
          v178.m128i_i64[1] = (__int64)&unk_49DB108;
          v179 = &v162;
          v180[0] = &v178.m128i_i64[1];
          v178.m128i_i8[0] = 1;
          sub_23328D0((__int64)&v163, (__int64)&v176);
          sub_23058C0(v160.m128i_i64, (__int64)&v163, v57, v154);
          v174 |= 3u;
          v171.m128i_i64[0] = v160.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
          sub_2240A30((unsigned __int64 *)&v163);
        }
        if ( v169.m128i_i8[0] )
        {
          v169.m128i_i8[0] = 0;
          sub_A17130((__int64)&v167);
        }
      }
      else
      {
        v42 = _mm_loadu_si128(&v176);
        v43 = _mm_loadu_si128(&v171);
        v178.m128i_i64[1] = 3;
        v178.m128i_i64[0] = (__int64)"all";
        v44 = *((_QWORD *)&v172 + 1);
        v45 = _mm_loadu_si128(&v178);
        v168.m128i_i64[0] = 0;
        v177.m128i_i64[0] = 0;
        v176 = v43;
        v174 = v174 & 0xFC | 2;
        v172 = __PAIR128__(v177.m128i_u64[1], 0);
        v177.m128i_i64[1] = v44;
        v171 = v42;
        v173 = v45;
        sub_A17130((__int64)&v176);
        sub_A17130((__int64)&v167);
      }
      v46 = v174 & 1;
      v174 = (2 * (v174 & 1)) | v174 & 0xFD;
      if ( v46 )
      {
        *a1 = v171.m128i_i64[0] | 1;
      }
      else
      {
        sub_23425F0((__int64)&v163, (__int64)&v171);
        v53 = _mm_loadu_si128(&v173);
        v177.m128i_i64[0] = 0;
        v165 = v53;
        if ( v164.m128i_i64[0] )
        {
          ((void (__fastcall *)(__m128i *, __m128i *, __int64))v164.m128i_i64[0])(&v176, &v163, 2);
          v177 = v164;
        }
        v54 = _mm_loadu_si128(&v165);
        v55 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v177.m128i_i64[0];
        v168.m128i_i64[0] = 0;
        v178 = v54;
        if ( v177.m128i_i64[0] )
        {
          ((void (__fastcall *)(__m128i *, __m128i *, __int64))v177.m128i_i64[0])(&v167, &v176, 2);
          v55 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v177.m128i_i64[0];
          v168 = v177;
        }
        v169 = _mm_loadu_si128(&v178);
        if ( v55 )
          v55(&v176, &v176, 3);
        sub_235DE80(a3, &v167);
        sub_A17130((__int64)&v167);
        sub_A17130((__int64)&v163);
        *a1 = 1;
        v176.m128i_i64[0] = 0;
        sub_9C66B0(v176.m128i_i64);
        if ( (v174 & 2) != 0 )
          sub_2352AA0(&v171, (__int64)&v167);
        if ( (v174 & 1) != 0 )
        {
          if ( v171.m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v171.m128i_i64[0] + 8LL))(v171.m128i_i64[0]);
        }
        else if ( (_QWORD)v172 )
        {
          ((void (__fastcall *)(__m128i *, __m128i *, __int64))v172)(&v171, &v171, 3);
        }
      }
      return a1;
    }
    goto LABEL_571;
  }
  v30 = a2[266];
  v31 = v30 + 32LL * *((unsigned int *)a2 + 534);
  while ( v31 != v30 )
  {
    v32 = a4[1].m128i_i64[0];
    v33 = _mm_loadu_si128(&v158);
    v34 = a4[1].m128i_i64[1] - v32;
    v176.m128i_i64[0] = v32;
    v171 = v33;
    v176.m128i_i64[1] = 0xCCCCCCCCCCCCCCCDLL * (v34 >> 3);
    if ( !*(_QWORD *)(v30 + 16) )
      sub_4263D6(v29.m128i_i64[0], v29.m128i_i64[1], v32);
    v29.m128i_i64[0] = v30;
    v29.m128i_i64[1] = (__int64)&v171;
    v35 = (*(__int64 (__fastcall **)(__int64, __m128i *, unsigned __int64 *, __m128i *))(v30 + 24))(
            v30,
            &v171,
            a3,
            &v176);
    v30 += 32;
    if ( v35 )
    {
      *a1 = 1;
      v176.m128i_i64[0] = 0;
      sub_9C66B0(v176.m128i_i64);
      return a1;
    }
  }
  v38 = sub_C63BB0();
  v176.m128i_i64[1] = 26;
  v39 = v38;
  v41 = v40;
  v176.m128i_i64[0] = (__int64)"unknown machine pass '{0}'";
  v177.m128i_i64[0] = (__int64)v180;
  v177.m128i_i64[1] = 1;
  v178.m128i_i8[0] = 1;
  v178.m128i_i64[1] = (__int64)&unk_49DB108;
  v179 = &v158;
  v180[0] = &v178.m128i_i64[1];
  sub_23328D0((__int64)&v171, (__int64)&v176);
  sub_23058C0(a1, (__int64)&v171, v39, v41);
  sub_2240A30((unsigned __int64 *)&v171);
  return a1;
}
