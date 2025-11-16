// Function: sub_19BE360
// Address: 0x19be360
//
__int64 __fastcall sub_19BE360(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        __int64 *a7,
        unsigned __int8 a8,
        int a9,
        __int64 a10,
        __int64 a11,
        _BYTE *a12,
        _BYTE *a13,
        _BYTE *a14,
        __int8 *a15)
{
  __int64 v15; // rbx
  _QWORD *v16; // rax
  unsigned int v17; // r14d
  __int64 v18; // rax
  __int64 v19; // rax
  char *v20; // rax
  size_t v21; // rdx
  __int64 v22; // rax
  __m128i v23; // xmm0
  __m128i v24; // xmm1
  __m128i *v25; // r12
  __m128i *v26; // r13
  __m128i *v27; // rdi
  bool v28; // r13
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdi
  __int64 v37; // rax
  __m128i *v38; // rbx
  __m128i *v39; // r12
  __m128i *v40; // rdi
  unsigned int v41; // r13d
  const __m128i *v42; // rbx
  const __m128i *v43; // r12
  const __m128i *v44; // rdi
  __int64 v46; // rax
  __int64 v47; // r13
  unsigned __int64 v48; // rdi
  unsigned __int64 v49; // rax
  int v50; // ecx
  __int64 v51; // r14
  __int64 v52; // rbx
  unsigned int v53; // r12d
  int v54; // r13d
  __int64 v55; // rax
  char v56; // r12
  int v57; // edx
  __int64 v58; // rax
  __m128i *v59; // r14
  __m128i *v60; // r13
  __m128i *v61; // rdi
  __int64 v62; // rax
  __m128i *v63; // r14
  __m128i *v64; // r13
  __m128i *v65; // rdi
  unsigned int v66; // eax
  __int64 v67; // rax
  __int64 v68; // rdx
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // rax
  __int64 v72; // rax
  char *v73; // rsi
  size_t v74; // rdx
  __int64 v75; // rax
  __int64 v76; // r12
  __int64 v77; // rax
  __int64 v78; // rax
  __m128i *v79; // r13
  __m128i *v80; // r12
  __m128i *v81; // rdi
  __int64 v82; // r13
  __int64 v83; // rax
  __int64 v84; // rax
  __int64 v85; // rax
  __m128i *v86; // rbx
  __m128i *v87; // r12
  __m128i *v88; // rdi
  __int64 v89; // rax
  __m128i *v90; // rbx
  __m128i *v91; // rdi
  __int64 v92; // rsi
  __m128i *v93; // r13
  const __m128i *v94; // rbx
  const __m128i *v95; // r12
  __int64 v96; // rax
  __m128i *v97; // rbx
  __m128i *v98; // rdi
  __int64 v99; // rax
  __int64 v100; // rax
  __int64 v101; // rax
  __m128i *v102; // rbx
  __m128i *v103; // rdi
  unsigned int v104; // r13d
  unsigned int v105; // eax
  __int64 v106; // rax
  __int64 v107; // rax
  __int64 v108; // r13
  char *v109; // rsi
  size_t v110; // rdx
  __int64 v111; // rdi
  __int64 v112; // rdi
  __int64 v113; // r14
  __int64 v114; // rbx
  __int64 v115; // r13
  __int64 v116; // rdi
  __int64 v117; // rax
  __int64 v118; // rax
  __int64 v119; // rax
  __int64 v120; // rax
  __int64 v121; // rax
  __int64 v122; // rax
  __int64 v123; // rax
  __int64 v124; // rax
  __int64 v125; // rax
  char *v126; // rsi
  size_t v127; // r8
  __int64 v128; // rax
  __int64 v129; // rax
  __int64 v130; // [rsp-10h] [rbp-690h]
  unsigned int v131; // [rsp+4h] [rbp-67Ch]
  __int64 v132; // [rsp+8h] [rbp-678h]
  unsigned int v133; // [rsp+10h] [rbp-670h]
  __int64 v134; // [rsp+18h] [rbp-668h]
  char v135; // [rsp+18h] [rbp-668h]
  int v136; // [rsp+20h] [rbp-660h]
  unsigned int v137; // [rsp+20h] [rbp-660h]
  __int64 v138; // [rsp+20h] [rbp-660h]
  __int64 v139; // [rsp+28h] [rbp-658h]
  unsigned int v140; // [rsp+28h] [rbp-658h]
  unsigned int v141; // [rsp+30h] [rbp-650h]
  char v142; // [rsp+37h] [rbp-649h]
  unsigned __int32 v143; // [rsp+48h] [rbp-638h]
  unsigned int v144; // [rsp+50h] [rbp-630h]
  __int64 v150; // [rsp+90h] [rbp-5F0h]
  char v151; // [rsp+A1h] [rbp-5DFh] BYREF
  char v152; // [rsp+A2h] [rbp-5DEh] BYREF
  char v153; // [rsp+A3h] [rbp-5DDh] BYREF
  unsigned int v154; // [rsp+A4h] [rbp-5DCh] BYREF
  __int64 v155; // [rsp+A8h] [rbp-5D8h] BYREF
  __int64 v156; // [rsp+B0h] [rbp-5D0h] BYREF
  __int64 v157; // [rsp+B8h] [rbp-5C8h] BYREF
  __int64 v158; // [rsp+C0h] [rbp-5C0h] BYREF
  int v159; // [rsp+CCh] [rbp-5B4h]
  unsigned int v160; // [rsp+D4h] [rbp-5ACh]
  unsigned int v161; // [rsp+D8h] [rbp-5A8h]
  unsigned int v162; // [rsp+E8h] [rbp-598h]
  char v163; // [rsp+ECh] [rbp-594h]
  unsigned __int8 v164; // [rsp+EDh] [rbp-593h]
  unsigned __int8 v165; // [rsp+EFh] [rbp-591h]
  unsigned __int8 v166; // [rsp+F0h] [rbp-590h]
  char v167; // [rsp+F1h] [rbp-58Fh]
  unsigned __int8 v168; // [rsp+F3h] [rbp-58Dh]
  __m128i v169; // [rsp+100h] [rbp-580h] BYREF
  _QWORD v170[2]; // [rsp+110h] [rbp-570h] BYREF
  _QWORD *v171; // [rsp+120h] [rbp-560h]
  _QWORD v172[6]; // [rsp+130h] [rbp-550h] BYREF
  __m128i v173; // [rsp+160h] [rbp-520h] BYREF
  unsigned __int64 v174[2]; // [rsp+170h] [rbp-510h] BYREF
  __int64 *v175; // [rsp+180h] [rbp-500h]
  char v176; // [rsp+188h] [rbp-4F8h] BYREF
  __int64 v177; // [rsp+190h] [rbp-4F0h] BYREF
  void *v178; // [rsp+290h] [rbp-3F0h] BYREF
  __int32 v179; // [rsp+298h] [rbp-3E8h]
  __int8 v180; // [rsp+29Ch] [rbp-3E4h]
  __int64 v181; // [rsp+2A0h] [rbp-3E0h]
  __m128i v182; // [rsp+2A8h] [rbp-3D8h] BYREF
  __int64 v183; // [rsp+2B8h] [rbp-3C8h]
  __int64 v184; // [rsp+2C0h] [rbp-3C0h]
  __m128i v185; // [rsp+2C8h] [rbp-3B8h] BYREF
  __int64 v186; // [rsp+2D8h] [rbp-3A8h]
  char v187; // [rsp+2E0h] [rbp-3A0h]
  const __m128i *v188; // [rsp+2E8h] [rbp-398h]
  unsigned int v189; // [rsp+2F0h] [rbp-390h]
  _BYTE v190[356]; // [rsp+2F8h] [rbp-388h] BYREF
  int v191; // [rsp+45Ch] [rbp-224h]
  __int64 v192; // [rsp+460h] [rbp-220h]
  __m128i v193; // [rsp+470h] [rbp-210h] BYREF
  __int64 v194; // [rsp+480h] [rbp-200h] BYREF
  __m128i v195; // [rsp+488h] [rbp-1F8h]
  __int64 v196; // [rsp+498h] [rbp-1E8h]
  __int64 v197; // [rsp+4A0h] [rbp-1E0h] BYREF
  __m128i v198; // [rsp+4A8h] [rbp-1D8h]
  __int64 v199; // [rsp+4B8h] [rbp-1C8h]
  char v200; // [rsp+4C0h] [rbp-1C0h]
  __m128i *v201; // [rsp+4C8h] [rbp-1B8h] BYREF
  __int64 v202; // [rsp+4D0h] [rbp-1B0h]
  _BYTE v203[356]; // [rsp+4D8h] [rbp-1A8h] BYREF
  int v204; // [rsp+63Ch] [rbp-44h]
  __int64 v205; // [rsp+640h] [rbp-40h]

  v15 = a1;
  v16 = *(_QWORD **)a1;
  if ( !*(_QWORD *)a1 )
  {
    sub_13FD840(&v155, a1);
    v17 = 1;
    v82 = **(_QWORD **)(a1 + 32);
    v150 = v82;
    sub_15C9090((__int64)&v193, &v155);
    sub_15CA330((__int64)&v178, (__int64)"loop-unroll", (__int64)"tryToUnrollLoop", 15, &v193, v82);
LABEL_167:
    sub_15CAB20((__int64)&v178, "Starting analysis in loop", 0x19u);
    goto LABEL_9;
  }
  v17 = 1;
  do
  {
    v16 = (_QWORD *)*v16;
    ++v17;
  }
  while ( v16 );
  sub_13FD840(&v155, a1);
  v150 = **(_QWORD **)(a1 + 32);
  sub_15C9090((__int64)&v193, &v155);
  sub_15CA330((__int64)&v178, (__int64)"loop-unroll", (__int64)"tryToUnrollLoop", 15, &v193, v150);
  if ( v17 <= 1 )
    goto LABEL_167;
  sub_15CAB20((__int64)&v178, "Starting analysis in nested loop (loop depth : ", 0x2Fu);
  sub_15C9C50((__int64)&v193, "LoopDepth", 9, v17);
  v18 = sub_17C2270((__int64)&v178, (__int64)&v193);
  sub_15CAB20(v18, ")", 1u);
  if ( (__int64 *)v195.m128i_i64[1] != &v197 )
    j_j___libc_free_0(v195.m128i_i64[1], v197 + 1);
  if ( (__int64 *)v193.m128i_i64[0] != &v194 )
    j_j___libc_free_0(v193.m128i_i64[0], v194 + 1);
LABEL_9:
  v19 = **(_QWORD **)(a1 + 32);
  if ( v19 && *(_QWORD *)(v19 + 56) )
  {
    sub_15CAB20((__int64)&v178, ", in function F[", 0x10u);
    v20 = (char *)sub_1649960(*(_QWORD *)(**(_QWORD **)(a1 + 32) + 56LL));
    sub_15CAB20((__int64)&v178, v20, v21);
    sub_15CAB20((__int64)&v178, "]", 1u);
  }
  v22 = sub_15E0530(*a7);
  if ( sub_1602790(v22)
    || (v83 = sub_15E0530(*a7),
        v84 = sub_16033E0(v83),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v84 + 48LL))(v84)) )
  {
    sub_15CAB20((__int64)&v178, "...", 3u);
    v23 = _mm_loadu_si128(&v182);
    v24 = _mm_loadu_si128(&v185);
    v193.m128i_i32[2] = v179;
    v195 = v23;
    v193.m128i_i8[12] = v180;
    v198 = v24;
    v194 = v181;
    v196 = v183;
    v193.m128i_i64[0] = (__int64)&unk_49ECF68;
    v197 = v184;
    v200 = v187;
    if ( v187 )
      v199 = v186;
    v201 = (__m128i *)v203;
    v202 = 0x400000000LL;
    v144 = v189;
    if ( v189 )
    {
      v92 = v189;
      if ( v189 > 4 )
      {
        sub_14B3F20((__int64)&v201, v189);
        v93 = v201;
        v92 = v189;
      }
      else
      {
        v93 = (__m128i *)v203;
      }
      if ( v188 != (const __m128i *)((char *)v188 + 88 * v92) )
      {
        v94 = v188;
        v95 = (const __m128i *)((char *)v188 + 88 * v92);
        do
        {
          if ( v93 )
          {
            v93->m128i_i64[0] = (__int64)v93[1].m128i_i64;
            sub_19B5D20(v93->m128i_i64, v94->m128i_i64[0], v94->m128i_i64[0] + v94->m128i_i64[1]);
            v93[2].m128i_i64[0] = (__int64)v93[3].m128i_i64;
            sub_19B5D20(v93[2].m128i_i64, (_BYTE *)v94[2].m128i_i64[0], v94[2].m128i_i64[0] + v94[2].m128i_i64[1]);
            v93[4] = _mm_loadu_si128(v94 + 4);
            v93[5].m128i_i64[0] = v94[5].m128i_i64[0];
          }
          v94 = (const __m128i *)((char *)v94 + 88);
          v93 = (__m128i *)((char *)v93 + 88);
        }
        while ( v95 != v94 );
        v15 = a1;
      }
      LODWORD(v202) = v144;
    }
    v203[352] = v190[352];
    v204 = v191;
    v205 = v192;
    v193.m128i_i64[0] = (__int64)&unk_49ECF98;
    sub_143AA50(a7, (__int64)&v193);
    v25 = v201;
    v193.m128i_i64[0] = (__int64)&unk_49ECF68;
    v26 = (__m128i *)((char *)v201 + 88 * (unsigned int)v202);
    if ( v201 != v26 )
    {
      do
      {
        v26 = (__m128i *)((char *)v26 - 88);
        v27 = (__m128i *)v26[2].m128i_i64[0];
        if ( v27 != &v26[3] )
          j_j___libc_free_0(v27, v26[3].m128i_i64[0] + 1);
        if ( (__m128i *)v26->m128i_i64[0] != &v26[1] )
          j_j___libc_free_0(v26->m128i_i64[0], v26[1].m128i_i64[0] + 1);
      }
      while ( v25 != v26 );
      v26 = v201;
    }
    if ( v26 != (__m128i *)v203 )
      _libc_free((unsigned __int64)v26);
  }
  sub_13FD840(&v169, v15);
  if ( v169.m128i_i64[0] )
  {
    v28 = 0;
    sub_13FD840(&v173, v15);
    v29 = sub_15C70A0((__int64)&v173);
    if ( *(_DWORD *)(v29 + 8) == 2 && *(_QWORD *)(v29 - 8) )
    {
      sub_13FD840(&v193, v15);
      v30 = sub_15C70A0((__int64)&v193);
      while ( 1 )
      {
        v31 = v30;
        v32 = *(unsigned int *)(v30 + 8);
        if ( (_DWORD)v32 != 2 )
          break;
        v30 = *(_QWORD *)(v31 - 8);
        if ( !v30 )
        {
          v33 = -16;
          goto LABEL_125;
        }
      }
      v33 = -8 * v32;
LABEL_125:
      v28 = *(_QWORD *)(v31 + v33) != 0;
      if ( v193.m128i_i64[0] )
        sub_161E7C0((__int64)&v193, v193.m128i_i64[0]);
    }
    if ( v173.m128i_i64[0] )
      sub_161E7C0((__int64)&v173, v173.m128i_i64[0]);
    if ( v169.m128i_i64[0] )
      sub_161E7C0((__int64)&v169, v169.m128i_i64[0]);
    if ( v28 )
    {
      v67 = sub_15E0530(*a7);
      if ( sub_1602790(v67)
        || (v34 = sub_15E0530(*a7),
            v35 = sub_16033E0(v34),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v35 + 48LL))(v35)) )
      {
        sub_15C9090((__int64)&v173, &v155);
        sub_15CA330((__int64)&v193, (__int64)"loop-unroll", (__int64)"UnrollLoop", 10, &v173, v150);
        sub_15CAB20((__int64)&v193, "  Loop is from an inlined function: inlined into \"", 0x32u);
        sub_13FD840(&v158, v15);
        v68 = sub_15C70A0((__int64)&v158);
        v69 = *(unsigned int *)(v68 + 8);
        if ( (_DWORD)v69 == 2 )
        {
          while ( 1 )
          {
            v70 = *(_QWORD *)(v68 - 8);
            if ( !v70 )
              break;
            v68 = *(_QWORD *)(v68 - 8);
            v69 = *(unsigned int *)(v70 + 8);
            if ( (_DWORD)v69 != 2 )
              goto LABEL_136;
          }
          v71 = -16;
        }
        else
        {
LABEL_136:
          v71 = -8 * v69;
        }
        v72 = *(_QWORD *)(v68 + v71);
        if ( *(_BYTE *)v72 == 15 || (v72 = *(_QWORD *)(v72 - 8LL * *(unsigned int *)(v72 + 8))) != 0 )
        {
          v73 = *(char **)(v72 - 8LL * *(unsigned int *)(v72 + 8));
          if ( v73 )
            v73 = (char *)sub_161E970(*(_QWORD *)(v72 - 8LL * *(unsigned int *)(v72 + 8)));
          else
            v74 = 0;
        }
        else
        {
          v74 = 0;
          v73 = (char *)byte_3F871B3;
        }
        sub_15CAB20((__int64)&v193, v73, v74);
        sub_15CAB20((__int64)&v193, ":", 1u);
        sub_13FD840(&v157, v15);
        v75 = sub_15C70A0((__int64)&v157);
        if ( *(_DWORD *)(v75 + 8) != 2 )
          BUG();
        sub_15C9C50((__int64)&v173, "LineNumber", 10, *(_DWORD *)(*(_QWORD *)(v75 - 8) + 4LL));
        v76 = sub_17C2270((__int64)&v193, (__int64)&v173);
        sub_15CAB20(v76, ":", 1u);
        sub_13FD840(&v156, v15);
        v77 = sub_15C70A0((__int64)&v156);
        if ( *(_DWORD *)(v77 + 8) != 2 )
          BUG();
        sub_15C9C50((__int64)&v169, "ColumnNumber", 12, *(unsigned __int16 *)(*(_QWORD *)(v77 - 8) + 2LL));
        v78 = sub_17C2270(v76, (__int64)&v169);
        sub_15CAB20(v78, "\"", 1u);
        if ( v171 != v172 )
          j_j___libc_free_0(v171, v172[0] + 1LL);
        if ( (_QWORD *)v169.m128i_i64[0] != v170 )
          j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
        if ( v156 )
          sub_161E7C0((__int64)&v156, v156);
        if ( v175 != &v177 )
          j_j___libc_free_0(v175, v177 + 1);
        if ( (unsigned __int64 *)v173.m128i_i64[0] != v174 )
          j_j___libc_free_0(v173.m128i_i64[0], v174[0] + 1);
        if ( v157 )
          sub_161E7C0((__int64)&v157, v157);
        if ( v158 )
          sub_161E7C0((__int64)&v158, v158);
        sub_143AA50(a7, (__int64)&v193);
        v79 = v201;
        v193.m128i_i64[0] = (__int64)&unk_49ECF68;
        v80 = (__m128i *)((char *)v201 + 88 * (unsigned int)v202);
        if ( v201 != v80 )
        {
          do
          {
            v80 = (__m128i *)((char *)v80 - 88);
            v81 = (__m128i *)v80[2].m128i_i64[0];
            if ( v81 != &v80[3] )
              j_j___libc_free_0(v81, v80[3].m128i_i64[0] + 1);
            if ( (__m128i *)v80->m128i_i64[0] != &v80[1] )
              j_j___libc_free_0(v80->m128i_i64[0], v80[1].m128i_i64[0] + 1);
          }
          while ( v79 != v80 );
          v80 = v201;
        }
        if ( v80 != (__m128i *)v203 )
          _libc_free((unsigned __int64)v80);
      }
    }
  }
  v36 = sub_13FD000(v15);
  if ( v36 && sub_1AFD990(v36, "llvm.loop.unroll.disable", 24) )
  {
    v37 = sub_15E0530(*a7);
    if ( !sub_1602790(v37) )
    {
      v99 = sub_15E0530(*a7);
      v100 = sub_16033E0(v99);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v100 + 48LL))(v100) )
      {
LABEL_45:
        v41 = 0;
        goto LABEL_46;
      }
    }
    sub_15C9090((__int64)&v173, &v155);
    sub_15CA330((__int64)&v193, (__int64)"loop-unroll", (__int64)"tryToUnrollLoop", 15, &v173, v150);
    sub_15CAB20((__int64)&v193, "  Not unrolling : loop has unroll disable pragma", 0x30u);
    sub_143AA50(a7, (__int64)&v193);
    v38 = v201;
    v193.m128i_i64[0] = (__int64)&unk_49ECF68;
    v39 = (__m128i *)((char *)v201 + 88 * (unsigned int)v202);
    if ( v201 != v39 )
    {
      do
      {
        v39 = (__m128i *)((char *)v39 - 88);
        v40 = (__m128i *)v39[2].m128i_i64[0];
        if ( v40 != &v39[3] )
          j_j___libc_free_0(v40, v39[3].m128i_i64[0] + 1);
        if ( (__m128i *)v39->m128i_i64[0] != &v39[1] )
          j_j___libc_free_0(v39->m128i_i64[0], v39[1].m128i_i64[0] + 1);
      }
      while ( v38 != v39 );
LABEL_42:
      v39 = v201;
      goto LABEL_43;
    }
    goto LABEL_43;
  }
  if ( !(unsigned __int8)sub_13FCBF0(v15) )
  {
    v96 = sub_15E0530(*a7);
    if ( !sub_1602790(v96) )
    {
      v117 = sub_15E0530(*a7);
      v118 = sub_16033E0(v117);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v118 + 48LL))(v118) )
        goto LABEL_45;
    }
    sub_15C9090((__int64)&v173, &v155);
    sub_15CA330((__int64)&v193, (__int64)"loop-unroll", (__int64)"tryToUnrollLoop", 15, &v173, v150);
    sub_15CAB20((__int64)&v193, "  Not unrolling : loop not in normal form", 0x29u);
    sub_143AA50(a7, (__int64)&v193);
    v97 = v201;
    v193.m128i_i64[0] = (__int64)&unk_49ECF68;
    v39 = (__m128i *)((char *)v201 + 88 * (unsigned int)v202);
    if ( v201 != v39 )
    {
      do
      {
        v39 = (__m128i *)((char *)v39 - 88);
        v98 = (__m128i *)v39[2].m128i_i64[0];
        if ( v98 != &v39[3] )
          j_j___libc_free_0(v98, v39[3].m128i_i64[0] + 1);
        if ( (__m128i *)v39->m128i_i64[0] != &v39[1] )
          j_j___libc_free_0(v39->m128i_i64[0], v39[1].m128i_i64[0] + 1);
      }
      while ( v97 != v39 );
      goto LABEL_42;
    }
    goto LABEL_43;
  }
  v169.m128i_i8[1] = a15[1];
  if ( v169.m128i_i8[1] )
    v169.m128i_i8[0] = *a15;
  BYTE1(v157) = a14[1];
  if ( BYTE1(v157) )
    LOBYTE(v157) = *a14;
  BYTE1(v156) = a13[1];
  if ( BYTE1(v156) )
    LOBYTE(v156) = *a13;
  BYTE1(v154) = a12[1];
  if ( BYTE1(v154) )
    LOBYTE(v154) = *a12;
  v193.m128i_i8[4] = *(_BYTE *)(a10 + 4);
  if ( v193.m128i_i8[4] )
    v193.m128i_i32[0] = *(_DWORD *)a10;
  v173.m128i_i8[4] = *(_BYTE *)(a11 + 4);
  if ( v173.m128i_i8[4] )
    v173.m128i_i32[0] = *(_DWORD *)a11;
  sub_19B6690((__int64)&v158, v15, a4, (__int64)a5, a9, v173.m128i_i32, (__int64)&v193, &v154, &v156, &v157, &v169);
  if ( !(_DWORD)v158 && (!v163 || !v159) )
  {
    v89 = sub_15E0530(*a7);
    if ( !sub_1602790(v89) )
    {
      v106 = sub_15E0530(*a7);
      v107 = sub_16033E0(v106);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v107 + 48LL))(v107) )
        goto LABEL_45;
    }
    sub_15C9090((__int64)&v173, &v155);
    sub_15CA330((__int64)&v193, (__int64)"loop-unroll", (__int64)"tryToUnrollLoop", 15, &v173, v150);
    sub_15CAB20((__int64)&v193, "  Not unrolling : unrolling is disabled", 0x27u);
    sub_143AA50(a7, (__int64)&v193);
    v90 = v201;
    v193.m128i_i64[0] = (__int64)&unk_49ECF68;
    v39 = (__m128i *)((char *)v201 + 88 * (unsigned int)v202);
    if ( v201 != v39 )
    {
      do
      {
        v39 = (__m128i *)((char *)v39 - 88);
        v91 = (__m128i *)v39[2].m128i_i64[0];
        if ( v91 != &v39[3] )
          j_j___libc_free_0(v91, v39[3].m128i_i64[0] + 1);
        if ( (__m128i *)v39->m128i_i64[0] != &v39[1] )
          j_j___libc_free_0(v39->m128i_i64[0], v39[1].m128i_i64[0] + 1);
      }
      while ( v90 != v39 );
      goto LABEL_42;
    }
LABEL_43:
    if ( v39 != (__m128i *)v203 )
      _libc_free((unsigned __int64)v39);
    goto LABEL_45;
  }
  v173.m128i_i64[1] = (__int64)&v176;
  v174[0] = (unsigned __int64)&v176;
  v173.m128i_i64[0] = 0;
  v174[1] = 32;
  LODWORD(v175) = 0;
  sub_14D04F0(v15, a6, (__int64)&v173);
  v143 = sub_19B7070(v15, &v154, &v151, &v152, a5, (__int64)&v173, v162);
  v142 = v151;
  if ( v151 )
  {
    v101 = sub_15E0530(*a7);
    if ( !sub_1602790(v101) )
    {
      v128 = sub_15E0530(*a7);
      v129 = sub_16033E0(v128);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v129 + 48LL))(v129, v130) )
        goto LABEL_180;
    }
    sub_15C9090((__int64)&v169, &v155);
    sub_15CA330((__int64)&v193, (__int64)"loop-unroll", (__int64)"tryToUnrollLoop", 15, &v169, v150);
    sub_15CAB20((__int64)&v193, "  Not unrolling : loop contains non-duplicatable instructions", 0x3Du);
    sub_143AA50(a7, (__int64)&v193);
    v102 = v201;
    v193.m128i_i64[0] = (__int64)&unk_49ECF68;
    v87 = (__m128i *)((char *)v201 + 88 * (unsigned int)v202);
    if ( v201 != v87 )
    {
      do
      {
        v87 = (__m128i *)((char *)v87 - 88);
        v103 = (__m128i *)v87[2].m128i_i64[0];
        if ( v103 != &v87[3] )
          j_j___libc_free_0(v103, v87[3].m128i_i64[0] + 1);
        if ( (__m128i *)v87->m128i_i64[0] != &v87[1] )
          j_j___libc_free_0(v87->m128i_i64[0], v87[1].m128i_i64[0] + 1);
      }
      while ( v102 != v87 );
      goto LABEL_177;
    }
LABEL_178:
    if ( v87 != (__m128i *)v203 )
      _libc_free((unsigned __int64)v87);
    goto LABEL_180;
  }
  v141 = v154;
  if ( v154 )
  {
    v85 = sub_15E0530(*a7);
    if ( !sub_1602790(v85) )
    {
      v124 = sub_15E0530(*a7);
      v125 = sub_16033E0(v124);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v125 + 48LL))(v125, v130) )
        goto LABEL_180;
    }
    sub_15C9090((__int64)&v169, &v155);
    sub_15CA330((__int64)&v193, (__int64)"loop-unroll", (__int64)"tryToUnrollLoop", 15, &v169, v150);
    sub_15CAB20((__int64)&v193, "  Not unrolling : loop contains function calls that may be inlined later", 0x48u);
    sub_143AA50(a7, (__int64)&v193);
    v86 = v201;
    v193.m128i_i64[0] = (__int64)&unk_49ECF68;
    v87 = (__m128i *)((char *)v201 + 88 * (unsigned int)v202);
    if ( v201 != v87 )
    {
      do
      {
        v87 = (__m128i *)((char *)v87 - 88);
        v88 = (__m128i *)v87[2].m128i_i64[0];
        if ( v88 != &v87[3] )
          j_j___libc_free_0(v88, v87[3].m128i_i64[0] + 1);
        if ( (__m128i *)v87->m128i_i64[0] != &v87[1] )
          j_j___libc_free_0(v87->m128i_i64[0], v87[1].m128i_i64[0] + 1);
      }
      while ( v86 != v87 );
LABEL_177:
      v87 = v201;
      goto LABEL_178;
    }
    goto LABEL_178;
  }
  LODWORD(v156) = 0;
  LODWORD(v157) = 1;
  v46 = sub_13FCB50(v15);
  v139 = v46;
  v47 = v46;
  if ( v46 )
  {
    v48 = sub_157EBA0(v46);
    if ( v48 )
    {
      v136 = sub_15F4D60(v48);
      v49 = sub_157EBA0(v47);
      v50 = v136;
      if ( v136 )
      {
        v137 = v17;
        v51 = v49;
        v134 = v15;
        v52 = v15 + 56;
        v53 = 0;
        v54 = v50;
        do
        {
          v55 = sub_15F4DF0(v51, v53);
          if ( !sub_1377F70(v52, v55) )
          {
            v17 = v137;
            v15 = v134;
            goto LABEL_83;
          }
          ++v53;
        }
        while ( v54 != v53 );
        v17 = v137;
        v15 = v134;
      }
    }
  }
  v139 = sub_13F9E70(v15);
  if ( v139 )
  {
LABEL_83:
    LODWORD(v156) = sub_1474190(a4, v15, v139);
    LODWORD(v157) = sub_147DD60(a4, v15, v139);
  }
  if ( !(_DWORD)v156 )
  {
    v140 = sub_1474290(a4, v15);
    v135 = sub_1474320(a4, v15);
    v131 = dword_4FB2920;
    v104 = sub_19B5DD0(v15);
    if ( v140 <= dword_4FB2840 && (v112 = sub_13FD000(v15)) != 0 && sub_1AFD990(v112, "llvm.loop.unroll.full", 21)
      || v140 <= v104 )
    {
      v138 = *(_QWORD *)(v15 + 40);
      if ( *(_QWORD *)(v15 + 32) == v138 )
      {
LABEL_253:
        v167 = 1;
        v142 = v135;
        v141 = v140;
        goto LABEL_85;
      }
      v132 = v15;
      v133 = v17;
      v113 = *(_QWORD *)(v15 + 32);
      while ( 1 )
      {
        v114 = *(_QWORD *)(*(_QWORD *)v113 + 48LL);
        v115 = *(_QWORD *)v113 + 40LL;
        if ( v115 != v114 )
          break;
LABEL_251:
        v113 += 8;
        if ( v138 == v113 )
        {
          v17 = v133;
          v15 = v132;
          goto LABEL_253;
        }
      }
      while ( 1 )
      {
        v116 = v114 - 24;
        if ( !v114 )
          v116 = 0;
        if ( (unsigned __int8)sub_1C30710(v116) )
          break;
        v114 = *(_QWORD *)(v114 + 8);
        if ( v115 == v114 )
          goto LABEL_251;
      }
      v17 = v133;
      v15 = v132;
    }
    if ( v167 || v135 )
    {
      v142 = v135;
      v105 = 0;
      if ( v140 <= v131 )
        v105 = v140;
      v141 = v105;
    }
  }
LABEL_85:
  v153 = 0;
  v56 = sub_19BB5C0(
          v15,
          (__int64 **)a5,
          a2,
          a3,
          a4,
          (__int64)&v173,
          a7,
          (unsigned int *)&v156,
          v141,
          (unsigned int *)&v157,
          v143,
          (int *)&v158,
          (bool *)&v153);
  if ( !v160 )
    goto LABEL_180;
  if ( (_DWORD)v156 && v160 > (unsigned int)v156 )
    v160 = v156;
  if ( v17 > 1 )
  {
    v57 = 3;
    if ( v17 >= 3 )
      v57 = v17;
    if ( (unsigned int)(v159 * v57) > v162 + v160 * (unsigned __int64)(v143 - v162) )
      v168 = 1;
  }
  v58 = sub_15E0530(*a7);
  if ( sub_1602790(v58)
    || (v121 = sub_15E0530(*a7),
        v122 = sub_16033E0(v121),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v122 + 48LL))(v122)) )
  {
    sub_15C9090((__int64)&v169, &v155);
    sub_15CA330((__int64)&v193, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v169, v150);
    sub_15CAB20((__int64)&v193, "    Success! Unrolling strategy :", 0x21u);
    sub_143AA50(a7, (__int64)&v193);
    v59 = v201;
    v193.m128i_i64[0] = (__int64)&unk_49ECF68;
    v60 = (__m128i *)((char *)v201 + 88 * (unsigned int)v202);
    if ( v201 != v60 )
    {
      do
      {
        v60 = (__m128i *)((char *)v60 - 88);
        v61 = (__m128i *)v60[2].m128i_i64[0];
        if ( v61 != &v60[3] )
          j_j___libc_free_0(v61, v60[3].m128i_i64[0] + 1);
        if ( (__m128i *)v60->m128i_i64[0] != &v60[1] )
          j_j___libc_free_0(v60->m128i_i64[0], v60[1].m128i_i64[0] + 1);
      }
      while ( v59 != v60 );
      v60 = v201;
    }
    if ( v60 != (__m128i *)v203 )
      _libc_free((unsigned __int64)v60);
  }
  v62 = sub_15E0530(*a7);
  if ( sub_1602790(v62)
    || (v119 = sub_15E0530(*a7),
        v120 = sub_16033E0(v119),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v120 + 48LL))(v120)) )
  {
    sub_15C9090((__int64)&v169, &v155);
    sub_15CA330((__int64)&v193, (__int64)"loop-unroll", (__int64)"tryToUnrollLoop", 15, &v169, v150);
    if ( v161 )
    {
      sub_15CAB20((__int64)&v193, "      loop peeling by ", 0x16u);
      sub_15C9C50((__int64)&v169, "UP.PeelCount", 12, v161);
      v123 = sub_17C2270((__int64)&v193, (__int64)&v169);
      v109 = " iterations";
      v110 = 11;
      v111 = v123;
    }
    else
    {
      if ( (_DWORD)v156 )
      {
        if ( (_DWORD)v156 == v160 )
        {
          if ( v153 )
            sub_15CAB20((__int64)&v193, "      fully unroll with side exits by known upper bound", 0x37u);
          else
            sub_15CAB20((__int64)&v193, "      fully unroll to straight-line code", 0x28u);
LABEL_110:
          sub_143AA50(a7, (__int64)&v193);
          v63 = v201;
          v193.m128i_i64[0] = (__int64)&unk_49ECF68;
          v64 = (__m128i *)((char *)v201 + 88 * (unsigned int)v202);
          if ( v201 != v64 )
          {
            do
            {
              v64 = (__m128i *)((char *)v64 - 88);
              v65 = (__m128i *)v64[2].m128i_i64[0];
              if ( v65 != &v64[3] )
                j_j___libc_free_0(v65, v64[3].m128i_i64[0] + 1);
              if ( (__m128i *)v64->m128i_i64[0] != &v64[1] )
                j_j___libc_free_0(v64->m128i_i64[0], v64[1].m128i_i64[0] + 1);
            }
            while ( v63 != v64 );
            v64 = v201;
          }
          if ( v64 != (__m128i *)v203 )
            _libc_free((unsigned __int64)v64);
          goto LABEL_119;
        }
        sub_15CAB20((__int64)&v193, "      partially unroll by factor of ", 0x24u);
        sub_15C9C50((__int64)&v169, "UP.Count", 8, v160);
        v126 = " with remainder loop";
        v108 = sub_17C2270((__int64)&v193, (__int64)&v169);
        if ( !((unsigned int)v156 % v160) )
          v126 = (char *)byte_3F871B3;
        sub_15CAB20(v108, v126, (unsigned int)v156 % v160 != 0 ? 0x14 : 0);
        v127 = 0;
        v109 = (char *)byte_3F871B3;
        if ( (unsigned int)v156 % v160 )
        {
          v127 = v168 != 0 ? 0x2A : 0;
          if ( v168 )
            v109 = " and remainder loop will be fully unrolled";
        }
        v110 = v127;
      }
      else
      {
        sub_15CAB20((__int64)&v193, "      runtime unroll by factor of ", 0x22u);
        sub_15C9C50((__int64)&v169, "UP.Count", 8, v160);
        v108 = sub_17C2270((__int64)&v193, (__int64)&v169);
        sub_15CAB20(v108, " with remainder loop", 0x14u);
        v109 = " and remainder loop will be fully unrolled";
        v110 = v168 != 0 ? 0x2A : 0;
        if ( !v168 )
          v109 = (char *)byte_3F871B3;
      }
      v111 = v108;
    }
    sub_15CAB20(v111, v109, v110);
    if ( v171 != v172 )
      j_j___libc_free_0(v171, v172[0] + 1LL);
    if ( (_QWORD *)v169.m128i_i64[0] != v170 )
      j_j___libc_free_0(v169.m128i_i64[0], v170[0] + 1LL);
    goto LABEL_110;
  }
LABEL_119:
  v66 = sub_1B01A40(v15, v160, v156, v166, v164, v165, v153, v142, v157, v161, v168, a3, a4, a2, a6, (__int64)a7, a8);
  v41 = v66;
  if ( v66 )
  {
    if ( v66 != 2 && (v56 || v161) )
      sub_13FD1C0(v15);
    goto LABEL_181;
  }
LABEL_180:
  v41 = 0;
LABEL_181:
  if ( v174[0] != v173.m128i_i64[1] )
    _libc_free(v174[0]);
LABEL_46:
  v42 = v188;
  v178 = &unk_49ECF68;
  v43 = (const __m128i *)((char *)v188 + 88 * v189);
  if ( v188 != v43 )
  {
    do
    {
      v43 = (const __m128i *)((char *)v43 - 88);
      v44 = (const __m128i *)v43[2].m128i_i64[0];
      if ( v44 != &v43[3] )
        j_j___libc_free_0(v44, v43[3].m128i_i64[0] + 1);
      if ( (const __m128i *)v43->m128i_i64[0] != &v43[1] )
        j_j___libc_free_0(v43->m128i_i64[0], v43[1].m128i_i64[0] + 1);
    }
    while ( v42 != v43 );
    v43 = v188;
  }
  if ( v43 != (const __m128i *)v190 )
    _libc_free((unsigned __int64)v43);
  if ( v155 )
    sub_161E7C0((__int64)&v155, v155);
  return v41;
}
