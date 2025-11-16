// Function: sub_E66E10
// Address: 0xe66e10
//
__int64 __fastcall sub_E66E10(__int64 a1)
{
  __int64 **v2; // r13
  __int64 *v3; // r14
  char *v4; // r12
  __int64 *v5; // r14
  __int64 *v6; // r12
  __int64 *v7; // rdi
  __int64 v8; // rax
  __m128i v9; // xmm1
  void (__fastcall *v10)(_QWORD, _QWORD, _QWORD); // rax
  __int64 v11; // rdx
  __m128i v12; // xmm0
  __int64 *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rsi
  unsigned int v16; // ecx
  __int64 v17; // r12
  _QWORD *v18; // rcx
  unsigned __int64 v19; // r12
  _QWORD *v20; // r15
  _QWORD *v21; // r14
  __int64 v22; // r15
  unsigned __int64 v23; // r12
  _QWORD *v24; // rcx
  _QWORD *v25; // r15
  _QWORD *v26; // r14
  __int64 *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rsi
  unsigned int v30; // ecx
  __int64 v31; // r12
  _QWORD *v32; // rcx
  unsigned __int64 v33; // r12
  _QWORD *v34; // r15
  _QWORD *v35; // r14
  __int64 v36; // r15
  unsigned __int64 v37; // r12
  _QWORD *v38; // rcx
  _QWORD *v39; // r15
  _QWORD *v40; // r14
  __int64 *v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rsi
  unsigned int v44; // ecx
  __int64 v45; // r12
  _QWORD *v46; // rcx
  unsigned __int64 v47; // r12
  _QWORD *v48; // r15
  _QWORD *v49; // r14
  __int64 v50; // r15
  unsigned __int64 v51; // r12
  _QWORD *v52; // rcx
  _QWORD *v53; // r15
  _QWORD *v54; // r14
  __int64 *v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rsi
  unsigned int v58; // ecx
  __int64 v59; // r12
  _QWORD *v60; // rcx
  unsigned __int64 v61; // r12
  _QWORD *v62; // r15
  _QWORD *v63; // r14
  __int64 v64; // r15
  unsigned __int64 v65; // r12
  _QWORD *v66; // rcx
  _QWORD *v67; // r15
  _QWORD *v68; // r14
  __int64 *v69; // rax
  __int64 v70; // rsi
  __int64 v71; // rdx
  unsigned int v72; // ecx
  __int64 v73; // r12
  bool v74; // cf
  __int64 v75; // rcx
  unsigned __int64 v76; // rax
  unsigned __int64 v77; // r12
  unsigned __int64 n; // r15
  __int64 v79; // rdi
  unsigned __int64 v80; // rdi
  __int64 v81; // r15
  unsigned __int64 v82; // r12
  unsigned __int64 jj; // r15
  __int64 v84; // rdi
  unsigned __int64 v85; // rdi
  __int64 *v86; // rax
  __int64 v87; // rdx
  __int64 v88; // rsi
  unsigned int v89; // ecx
  __int64 v90; // r12
  _QWORD *v91; // rcx
  unsigned __int64 v92; // r12
  _QWORD *v93; // r15
  _QWORD *v94; // r14
  __int64 v95; // r15
  unsigned __int64 v96; // r12
  _QWORD *v97; // rcx
  _QWORD *v98; // r15
  _QWORD *v99; // r14
  __int64 *v100; // r13
  __int64 v101; // rsi
  __int64 *v102; // r14
  __int64 v103; // rdx
  unsigned int v104; // ecx
  __int64 v105; // r12
  __int64 v106; // rcx
  unsigned __int64 v107; // rax
  unsigned __int64 v108; // r12
  unsigned __int64 mm; // r15
  unsigned __int64 v110; // rdi
  _QWORD *v111; // r14
  _QWORD *nn; // r15
  unsigned __int64 v113; // r13
  unsigned __int64 i1; // r12
  unsigned __int64 v115; // rdi
  __int64 *v116; // r13
  __int64 v117; // rsi
  __int64 *v118; // r14
  __int64 v119; // rdx
  unsigned int v120; // ecx
  __int64 v121; // rax
  __int64 v122; // rcx
  unsigned __int64 v123; // r12
  unsigned __int64 v124; // r15
  __int64 v125; // rdi
  _QWORD *v126; // r14
  _QWORD *i2; // r15
  unsigned __int64 v128; // r13
  unsigned __int64 v129; // r12
  __int64 v130; // rdi
  __int64 *v131; // rax
  __int64 v132; // rdx
  __int64 v133; // rsi
  unsigned int v134; // ecx
  __int64 v135; // r12
  _QWORD *v136; // rcx
  unsigned __int64 v137; // r12
  _QWORD *v138; // r15
  _QWORD *v139; // r14
  __int64 v140; // r15
  unsigned __int64 v141; // r12
  _QWORD *v142; // rcx
  _QWORD *v143; // r15
  _QWORD *v144; // r14
  __int64 *v145; // r13
  __int64 v146; // rsi
  __int64 v147; // rdx
  unsigned int v148; // ecx
  __int64 v149; // r12
  __int64 v150; // rcx
  unsigned __int64 v151; // rax
  unsigned __int64 v152; // r12
  unsigned __int64 i4; // r15
  __int64 v154; // rdi
  _QWORD *v155; // r14
  _QWORD *v156; // r14
  __int64 v157; // r15
  unsigned __int64 v158; // r13
  unsigned __int64 v159; // r12
  __int64 v160; // rdi
  _QWORD *v161; // r15
  __int64 v162; // r13
  __int64 v163; // r14
  __int64 v164; // r12
  __int64 v165; // rsi
  __int64 v166; // rdi
  __int64 v167; // rdi
  __int64 v168; // rdi
  __int64 v169; // rdi
  __int64 v170; // r8
  __int64 v171; // rax
  __int64 v172; // r14
  __int64 v173; // r12
  _QWORD *v174; // rdi
  __int64 v175; // rax
  __int64 v176; // rdx
  __int64 v177; // rax
  __int64 v178; // rdx
  int v179; // eax
  __int64 v180; // rdx
  _DWORD *v181; // rax
  _DWORD *i6; // rdx
  _BYTE *v183; // rax
  int v184; // eax
  __int64 v185; // rdx
  _QWORD *v186; // rax
  _QWORD *i8; // rdx
  __int64 v188; // rax
  int v189; // r8d
  __int64 v190; // r12
  __int64 v191; // rax
  __int64 v192; // r14
  _QWORD **v193; // r13
  _QWORD *v194; // rdi
  __int64 v195; // r12
  __int64 v196; // rax
  __int64 v197; // r14
  _QWORD **v198; // r13
  _QWORD *v199; // rdi
  _QWORD *v200; // rdi
  _QWORD *v201; // rdi
  _QWORD *v202; // rdi
  int v203; // esi
  __int64 v204; // r12
  __int64 v205; // rax
  __int64 v206; // r14
  _QWORD **v207; // r13
  _QWORD *v208; // rdi
  int v209; // eax
  __int64 v210; // rdx
  __int64 v211; // rax
  __int64 i10; // rdx
  int v213; // eax
  __int64 v214; // rdx
  _QWORD *v215; // rax
  _QWORD *i12; // rdx
  unsigned int v218; // ecx
  unsigned int v219; // eax
  _QWORD *v220; // rdi
  int v221; // r12d
  _QWORD *v222; // rax
  unsigned int v223; // ecx
  unsigned int v224; // eax
  __int64 v225; // rdi
  int v226; // r12d
  __int64 v227; // rax
  unsigned int v228; // ecx
  unsigned int v229; // eax
  _QWORD *v230; // rdi
  int v231; // r12d
  _QWORD *v232; // rax
  unsigned int v233; // ecx
  unsigned int v234; // eax
  _DWORD *v235; // rdi
  int v236; // r12d
  unsigned __int64 v237; // rdx
  unsigned __int64 v238; // rax
  _DWORD *v239; // rax
  __int64 v240; // rdx
  _DWORD *i7; // rdx
  unsigned __int64 v242; // rdx
  unsigned __int64 v243; // rax
  _QWORD *v244; // rax
  __int64 v245; // rdx
  _QWORD *i9; // rdx
  unsigned __int64 v247; // rdx
  unsigned __int64 v248; // rax
  _QWORD *v249; // rax
  __int64 v250; // rdx
  _QWORD *i13; // rdx
  unsigned __int64 v252; // rdx
  unsigned __int64 v253; // rax
  __int64 v254; // rax
  __int64 v255; // rdx
  __int64 i11; // rdx
  _DWORD *v257; // rax
  __int64 *v258; // [rsp+0h] [rbp-60h]
  _QWORD *i; // [rsp+0h] [rbp-60h]
  __int64 *v260; // [rsp+0h] [rbp-60h]
  _QWORD *j; // [rsp+0h] [rbp-60h]
  __int64 *v262; // [rsp+0h] [rbp-60h]
  _QWORD *k; // [rsp+0h] [rbp-60h]
  __int64 *v264; // [rsp+0h] [rbp-60h]
  _QWORD *m; // [rsp+0h] [rbp-60h]
  __int64 *v266; // [rsp+0h] [rbp-60h]
  _QWORD *ii; // [rsp+0h] [rbp-60h]
  __int64 *v268; // [rsp+0h] [rbp-60h]
  _QWORD *kk; // [rsp+0h] [rbp-60h]
  __int64 *v270; // [rsp+0h] [rbp-60h]
  _QWORD *i3; // [rsp+0h] [rbp-60h]
  __int64 *v272; // [rsp+8h] [rbp-58h]
  _QWORD *v273; // [rsp+8h] [rbp-58h]
  __int64 *v274; // [rsp+8h] [rbp-58h]
  _QWORD *v275; // [rsp+8h] [rbp-58h]
  __int64 *v276; // [rsp+8h] [rbp-58h]
  _QWORD *v277; // [rsp+8h] [rbp-58h]
  __int64 *v278; // [rsp+8h] [rbp-58h]
  _QWORD *v279; // [rsp+8h] [rbp-58h]
  __int64 *v280; // [rsp+8h] [rbp-58h]
  _QWORD *v281; // [rsp+8h] [rbp-58h]
  __int64 *v282; // [rsp+8h] [rbp-58h]
  _QWORD *v283; // [rsp+8h] [rbp-58h]
  __int64 *v284; // [rsp+8h] [rbp-58h]
  _QWORD *v285; // [rsp+8h] [rbp-58h]
  __int64 *v286; // [rsp+8h] [rbp-58h]
  _QWORD *i5; // [rsp+8h] [rbp-58h]
  __m128i v288; // [rsp+10h] [rbp-50h] BYREF
  void (__fastcall *v289)(_QWORD, _QWORD, _QWORD); // [rsp+20h] [rbp-40h]
  __int64 v290; // [rsp+28h] [rbp-38h]

  v2 = *(__int64 ***)(a1 + 88);
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  if ( v2 )
  {
    v3 = v2[4];
    v4 = (char *)v2[3];
    if ( v3 != (__int64 *)v4 )
    {
      do
      {
        if ( *(char **)v4 != v4 + 16 )
          j_j___libc_free_0(*(_QWORD *)v4, *((_QWORD *)v4 + 2) + 1LL);
        v4 += 32;
      }
      while ( v3 != (__int64 *)v4 );
      v4 = (char *)v2[3];
    }
    if ( v4 )
      j_j___libc_free_0(v4, (char *)v2[5] - v4);
    v5 = v2[1];
    v6 = *v2;
    if ( v5 != *v2 )
    {
      do
      {
        v7 = v6;
        v6 += 3;
        sub_C8EE20(v7);
      }
      while ( v5 != v6 );
      v6 = *v2;
    }
    if ( v6 )
      j_j___libc_free_0(v6, (char *)v2[2] - (char *)v6);
    j_j___libc_free_0(v2, 64);
  }
  v8 = *(_QWORD *)(a1 + 96);
  if ( v8 != *(_QWORD *)(a1 + 104) )
    *(_QWORD *)(a1 + 104) = v8;
  v9 = _mm_loadu_si128((const __m128i *)(a1 + 120));
  v288.m128i_i64[0] = (__int64)sub_E62B10;
  v10 = *(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(a1 + 136);
  *(_QWORD *)(a1 + 136) = sub_E62AE0;
  v11 = *(_QWORD *)(a1 + 144);
  v12 = _mm_loadu_si128(&v288);
  v289 = v10;
  v290 = v11;
  v288 = v9;
  *(_QWORD *)(a1 + 144) = sub_E62AC0;
  *(__m128i *)(a1 + 120) = v12;
  if ( v10 )
    v10(&v288, &v288, 3);
  v13 = *(__int64 **)(a1 + 400);
  v14 = *(unsigned int *)(a1 + 408);
  v272 = v13;
  v258 = &v13[v14];
  if ( v13 != v258 )
  {
    while ( 1 )
    {
      v15 = *v272;
      v16 = (unsigned int)(v272 - v13) >> 7;
      v17 = 4096LL << v16;
      if ( v16 >= 0x1E )
        v17 = 0x40000000000LL;
      v18 = (_QWORD *)((*v272 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      v19 = v15 + v17;
      if ( v15 == v13[v14 - 1] )
        v19 = *(_QWORD *)(a1 + 384);
      v20 = v18 + 22;
      if ( v19 >= (unsigned __int64)(v18 + 22) )
      {
        while ( 1 )
        {
          *v18 = &unk_49E35B0;
          v21 = v18;
          sub_E92880(v18);
          v18 = v20;
          if ( v19 < (unsigned __int64)(v21 + 44) )
            break;
          v20 += 22;
        }
      }
      if ( v258 == ++v272 )
        break;
      v13 = *(__int64 **)(a1 + 400);
      v14 = *(unsigned int *)(a1 + 408);
    }
  }
  v22 = 2LL * *(unsigned int *)(a1 + 456);
  v273 = *(_QWORD **)(a1 + 448);
  for ( i = &v273[v22]; i != v273; v273 += 2 )
  {
    v23 = *v273 + v273[1];
    v24 = (_QWORD *)((*v273 + 7LL) & 0xFFFFFFFFFFFFFFF8LL);
    v25 = v24 + 22;
    if ( v23 >= (unsigned __int64)(v24 + 22) )
    {
      while ( 1 )
      {
        *v24 = &unk_49E35B0;
        v26 = v24;
        sub_E92880(v24);
        v24 = v25;
        if ( v23 < (unsigned __int64)(v26 + 44) )
          break;
        v25 += 22;
      }
    }
  }
  sub_E66D20(a1 + 384);
  v27 = *(__int64 **)(a1 + 496);
  v28 = *(unsigned int *)(a1 + 504);
  v274 = v27;
  v260 = &v27[v28];
  if ( v27 != v260 )
  {
    while ( 1 )
    {
      v29 = *v274;
      v30 = (unsigned int)(v274 - v27) >> 7;
      v31 = 4096LL << v30;
      if ( v30 >= 0x1E )
        v31 = 0x40000000000LL;
      v32 = (_QWORD *)((*v274 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      v33 = v29 + v31;
      if ( v29 == v27[v28 - 1] )
        v33 = *(_QWORD *)(a1 + 480);
      v34 = v32 + 19;
      if ( v33 >= (unsigned __int64)(v32 + 19) )
      {
        while ( 1 )
        {
          *v32 = &unk_49E35D8;
          v35 = v32;
          sub_E92880(v32);
          v32 = v34;
          if ( v33 < (unsigned __int64)(v35 + 38) )
            break;
          v34 += 19;
        }
      }
      if ( v260 == ++v274 )
        break;
      v27 = *(__int64 **)(a1 + 496);
      v28 = *(unsigned int *)(a1 + 504);
    }
  }
  v36 = 2LL * *(unsigned int *)(a1 + 552);
  v275 = *(_QWORD **)(a1 + 544);
  for ( j = &v275[v36]; j != v275; v275 += 2 )
  {
    v37 = *v275 + v275[1];
    v38 = (_QWORD *)((*v275 + 7LL) & 0xFFFFFFFFFFFFFFF8LL);
    v39 = v38 + 19;
    if ( v37 >= (unsigned __int64)(v38 + 19) )
    {
      while ( 1 )
      {
        *v38 = &unk_49E35D8;
        v40 = v38;
        sub_E92880(v38);
        v38 = v39;
        if ( v37 < (unsigned __int64)(v40 + 38) )
          break;
        v39 += 19;
      }
    }
  }
  sub_E66D20(a1 + 480);
  v41 = *(__int64 **)(a1 + 592);
  v42 = *(unsigned int *)(a1 + 600);
  v276 = v41;
  v262 = &v41[v42];
  if ( v41 != v262 )
  {
    while ( 1 )
    {
      v43 = *v276;
      v44 = (unsigned int)(v276 - v41) >> 7;
      v45 = 4096LL << v44;
      if ( v44 >= 0x1E )
        v45 = 0x40000000000LL;
      v46 = (_QWORD *)((*v276 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      v47 = v43 + v45;
      if ( v43 == v41[v42 - 1] )
        v47 = *(_QWORD *)(a1 + 576);
      v48 = v46 + 25;
      if ( v47 >= (unsigned __int64)(v46 + 25) )
      {
        while ( 1 )
        {
          *v46 = &unk_49E3600;
          v49 = v46;
          sub_E92880(v46);
          v46 = v48;
          if ( v47 < (unsigned __int64)(v49 + 50) )
            break;
          v48 += 25;
        }
      }
      if ( v262 == ++v276 )
        break;
      v41 = *(__int64 **)(a1 + 592);
      v42 = *(unsigned int *)(a1 + 600);
    }
  }
  v50 = 2LL * *(unsigned int *)(a1 + 648);
  v277 = *(_QWORD **)(a1 + 640);
  for ( k = &v277[v50]; k != v277; v277 += 2 )
  {
    v51 = *v277 + v277[1];
    v52 = (_QWORD *)((*v277 + 7LL) & 0xFFFFFFFFFFFFFFF8LL);
    v53 = v52 + 25;
    if ( v51 >= (unsigned __int64)(v52 + 25) )
    {
      while ( 1 )
      {
        *v52 = &unk_49E3600;
        v54 = v52;
        sub_E92880(v52);
        v52 = v53;
        if ( v51 < (unsigned __int64)(v54 + 50) )
          break;
        v53 += 25;
      }
    }
  }
  sub_E66D20(a1 + 576);
  v55 = *(__int64 **)(a1 + 784);
  v56 = *(unsigned int *)(a1 + 792);
  v278 = v55;
  v264 = &v55[v56];
  if ( v55 != v264 )
  {
    while ( 1 )
    {
      v57 = *v278;
      v58 = (unsigned int)(v278 - v55) >> 7;
      v59 = 4096LL << v58;
      if ( v58 >= 0x1E )
        v59 = 0x40000000000LL;
      v60 = (_QWORD *)((*v278 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      v61 = v57 + v59;
      if ( v57 == v55[v56 - 1] )
        v61 = *(_QWORD *)(a1 + 768);
      v62 = v60 + 21;
      if ( v61 >= (unsigned __int64)(v60 + 21) )
      {
        while ( 1 )
        {
          *v60 = &unk_49E1A10;
          v63 = v60;
          sub_E92880(v60);
          v60 = v62;
          if ( v61 < (unsigned __int64)(v63 + 42) )
            break;
          v62 += 21;
        }
      }
      if ( v264 == ++v278 )
        break;
      v55 = *(__int64 **)(a1 + 784);
      v56 = *(unsigned int *)(a1 + 792);
    }
  }
  v64 = 2LL * *(unsigned int *)(a1 + 840);
  v279 = *(_QWORD **)(a1 + 832);
  for ( m = &v279[v64]; m != v279; v279 += 2 )
  {
    v65 = *v279 + v279[1];
    v66 = (_QWORD *)((*v279 + 7LL) & 0xFFFFFFFFFFFFFFF8LL);
    v67 = v66 + 21;
    if ( v65 >= (unsigned __int64)(v66 + 21) )
    {
      while ( 1 )
      {
        *v66 = &unk_49E1A10;
        v68 = v66;
        sub_E92880(v66);
        v66 = v67;
        if ( v65 < (unsigned __int64)(v68 + 42) )
          break;
        v67 += 21;
      }
    }
  }
  sub_E66D20(a1 + 768);
  v69 = *(__int64 **)(a1 + 688);
  v70 = *(unsigned int *)(a1 + 696);
  v280 = v69;
  v266 = &v69[v70];
  if ( v69 != v266 )
  {
    v71 = *(_QWORD *)(a1 + 688);
    while ( 1 )
    {
      v72 = (unsigned int)(((__int64)v280 - v71) >> 3) >> 7;
      v73 = 4096LL << v72;
      v74 = v72 < 0x1E;
      v75 = *v280;
      if ( !v74 )
        v73 = 0x40000000000LL;
      v76 = (v75 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      v77 = v75 + v73;
      if ( v75 == *(_QWORD *)(v71 + 8 * v70 - 8) )
        v77 = *(_QWORD *)(a1 + 672);
      for ( n = v76 + 192; n <= v77; n += 192LL )
      {
        *(_QWORD *)(n - 192) = &unk_49E3628;
        v79 = *(_QWORD *)(n - 192 + 176);
        if ( v79 != n )
          _libc_free(v79, v70);
        v80 = n - 192;
        sub_E92880(v80);
      }
      if ( v266 == ++v280 )
        break;
      v71 = *(_QWORD *)(a1 + 688);
      v70 = *(unsigned int *)(a1 + 696);
    }
  }
  v81 = 2LL * *(unsigned int *)(a1 + 744);
  v281 = *(_QWORD **)(a1 + 736);
  for ( ii = &v281[v81]; ii != v281; v281 += 2 )
  {
    v82 = *v281 + v281[1];
    for ( jj = ((*v281 + 7LL) & 0xFFFFFFFFFFFFFFF8LL) + 192; v82 >= jj; jj += 192LL )
    {
      *(_QWORD *)(jj - 192) = &unk_49E3628;
      v84 = *(_QWORD *)(jj - 192 + 176);
      if ( v84 != jj )
        _libc_free(v84, v281);
      v85 = jj - 192;
      sub_E92880(v85);
    }
  }
  sub_E66D20(a1 + 672);
  v86 = *(__int64 **)(a1 + 976);
  v87 = *(unsigned int *)(a1 + 984);
  v282 = v86;
  v268 = &v86[v87];
  if ( v86 != v268 )
  {
    while ( 1 )
    {
      v88 = *v282;
      v89 = (unsigned int)(v282 - v86) >> 7;
      v90 = 4096LL << v89;
      if ( v89 >= 0x1E )
        v90 = 0x40000000000LL;
      v91 = (_QWORD *)((*v282 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      v92 = v88 + v90;
      if ( v88 == v86[v87 - 1] )
        v92 = *(_QWORD *)(a1 + 960);
      v93 = v91 + 23;
      if ( v92 >= (unsigned __int64)(v91 + 23) )
      {
        while ( 1 )
        {
          *v91 = &unk_49E3650;
          v94 = v91;
          sub_E92880(v91);
          v91 = v93;
          if ( v92 < (unsigned __int64)(v94 + 46) )
            break;
          v93 += 23;
        }
      }
      if ( v268 == ++v282 )
        break;
      v86 = *(__int64 **)(a1 + 976);
      v87 = *(unsigned int *)(a1 + 984);
    }
  }
  v95 = 2LL * *(unsigned int *)(a1 + 1032);
  v283 = *(_QWORD **)(a1 + 1024);
  for ( kk = &v283[v95]; kk != v283; v283 += 2 )
  {
    v96 = *v283 + v283[1];
    v97 = (_QWORD *)((*v283 + 7LL) & 0xFFFFFFFFFFFFFFF8LL);
    v98 = v97 + 23;
    if ( v96 >= (unsigned __int64)(v97 + 23) )
    {
      while ( 1 )
      {
        *v97 = &unk_49E3650;
        v99 = v97;
        sub_E92880(v97);
        v97 = v98;
        if ( v96 < (unsigned __int64)(v99 + 46) )
          break;
        v98 += 23;
      }
    }
  }
  sub_E66D20(a1 + 960);
  v100 = *(__int64 **)(a1 + 1072);
  v101 = *(unsigned int *)(a1 + 1080);
  v102 = &v100[v101];
  if ( v100 != v102 )
  {
    v103 = *(_QWORD *)(a1 + 1072);
    while ( 1 )
    {
      v104 = (unsigned int)(((__int64)v100 - v103) >> 3) >> 7;
      v105 = 4096LL << v104;
      v74 = v104 < 0x1E;
      v106 = *v100;
      if ( !v74 )
        v105 = 0x40000000000LL;
      v107 = (v106 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      v108 = v106 + v105;
      if ( v106 == *(_QWORD *)(v103 + 8 * v101 - 8) )
        v108 = *(_QWORD *)(a1 + 1056);
      for ( mm = v107 + 192; mm <= v108; mm += 192LL )
      {
        v110 = mm - 192;
        sub_E96DA0(v110);
      }
      if ( v102 == ++v100 )
        break;
      v103 = *(_QWORD *)(a1 + 1072);
      v101 = *(unsigned int *)(a1 + 1080);
    }
  }
  v111 = *(_QWORD **)(a1 + 1120);
  for ( nn = &v111[2 * *(unsigned int *)(a1 + 1128)]; nn != v111; v111 += 2 )
  {
    v113 = *v111 + v111[1];
    for ( i1 = ((*v111 + 7LL) & 0xFFFFFFFFFFFFFFF8LL) + 192; v113 >= i1; i1 += 192LL )
    {
      v115 = i1 - 192;
      sub_E96DA0(v115);
    }
  }
  sub_E66D20(a1 + 1056);
  v116 = *(__int64 **)(a1 + 1168);
  v117 = *(unsigned int *)(a1 + 1176);
  v118 = &v116[v117];
  if ( v116 != v118 )
  {
    v119 = *(_QWORD *)(a1 + 1168);
    while ( 1 )
    {
      v120 = (unsigned int)(((__int64)v116 - v119) >> 3) >> 7;
      v121 = 4096LL << v120;
      v74 = v120 < 0x1E;
      v122 = *v116;
      if ( !v74 )
        v121 = 0x40000000000LL;
      v123 = (v122 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      v124 = v122 + v121;
      if ( v122 == *(_QWORD *)(v119 + 8 * v117 - 8) )
        v124 = *(_QWORD *)(a1 + 1152);
      while ( 1 )
      {
        v123 += 128LL;
        if ( v124 < v123 )
          break;
        while ( 1 )
        {
          v125 = *(_QWORD *)(v123 - 112);
          if ( v125 == v123 - 96 )
            break;
          _libc_free(v125, v117);
          v123 += 128LL;
          if ( v124 < v123 )
            goto LABEL_145;
        }
      }
LABEL_145:
      if ( v118 == ++v116 )
        break;
      v119 = *(_QWORD *)(a1 + 1168);
      v117 = *(unsigned int *)(a1 + 1176);
    }
  }
  v126 = *(_QWORD **)(a1 + 1216);
  for ( i2 = &v126[2 * *(unsigned int *)(a1 + 1224)]; i2 != v126; v126 += 2 )
  {
    v128 = *v126 + v126[1];
    v129 = (*v126 + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
    while ( 1 )
    {
      v129 += 128LL;
      if ( v128 < v129 )
        break;
      while ( 1 )
      {
        v130 = *(_QWORD *)(v129 - 112);
        if ( v130 == v129 - 96 )
          break;
        _libc_free(v130, v117);
        v129 += 128LL;
        if ( v128 < v129 )
          goto LABEL_152;
      }
    }
LABEL_152:
    ;
  }
  sub_E66D20(a1 + 1152);
  v131 = *(__int64 **)(a1 + 880);
  v132 = *(unsigned int *)(a1 + 888);
  v284 = v131;
  v270 = &v131[v132];
  if ( v131 != v270 )
  {
    while ( 1 )
    {
      v133 = *v284;
      v134 = (unsigned int)(v284 - v131) >> 7;
      v135 = 4096LL << v134;
      if ( v134 >= 0x1E )
        v135 = 0x40000000000LL;
      v136 = (_QWORD *)((v133 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      v137 = v133 + v135;
      if ( v133 == v131[v132 - 1] )
        v137 = *(_QWORD *)(a1 + 864);
      v138 = v136 + 19;
      if ( v137 >= (unsigned __int64)(v136 + 19) )
      {
        while ( 1 )
        {
          *v136 = &unk_49E1A38;
          v139 = v136;
          sub_E92880(v136);
          v136 = v138;
          if ( v137 < (unsigned __int64)(v139 + 38) )
            break;
          v138 += 19;
        }
      }
      if ( v270 == ++v284 )
        break;
      v131 = *(__int64 **)(a1 + 880);
      v132 = *(unsigned int *)(a1 + 888);
    }
  }
  v140 = 2LL * *(unsigned int *)(a1 + 936);
  v285 = *(_QWORD **)(a1 + 928);
  for ( i3 = &v285[v140]; i3 != v285; v285 += 2 )
  {
    v141 = *v285 + v285[1];
    v142 = (_QWORD *)((*v285 + 7LL) & 0xFFFFFFFFFFFFFFF8LL);
    v143 = v142 + 19;
    if ( v141 >= (unsigned __int64)(v142 + 19) )
    {
      while ( 1 )
      {
        *v142 = &unk_49E1A38;
        v144 = v142;
        sub_E92880(v142);
        v142 = v143;
        if ( v141 < (unsigned __int64)(v144 + 38) )
          break;
        v143 += 19;
      }
    }
  }
  sub_E66D20(a1 + 864);
  v145 = *(__int64 **)(a1 + 1264);
  v146 = *(unsigned int *)(a1 + 1272);
  v286 = &v145[v146];
  if ( v145 != v286 )
  {
    v147 = *(_QWORD *)(a1 + 1264);
    while ( 1 )
    {
      v148 = (unsigned int)(((__int64)v145 - v147) >> 3) >> 7;
      v149 = 4096LL << v148;
      v74 = v148 < 0x1E;
      v150 = *v145;
      if ( !v74 )
        v149 = 0x40000000000LL;
      v151 = (v150 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      v152 = v150 + v149;
      if ( v150 == *(_QWORD *)(v147 + 8 * v146 - 8) )
        v152 = *(_QWORD *)(a1 + 1248);
      for ( i4 = v151 + 64; i4 <= v152; i4 += 64LL )
      {
        v154 = *(_QWORD *)(i4 - 40);
        v155 = (_QWORD *)(i4 - 64);
        if ( v154 != i4 - 24 )
          _libc_free(v154, v146);
        if ( *v155 != i4 - 48 )
          _libc_free(*v155, v146);
      }
      if ( v286 == ++v145 )
        break;
      v147 = *(_QWORD *)(a1 + 1264);
      v146 = *(unsigned int *)(a1 + 1272);
    }
  }
  v156 = *(_QWORD **)(a1 + 1312);
  v157 = 2LL * *(unsigned int *)(a1 + 1320);
  for ( i5 = &v156[v157]; v156 != i5; v156 += 2 )
  {
    v158 = *v156 + v156[1];
    v159 = (*v156 + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
    while ( 1 )
    {
      v159 += 64LL;
      if ( v158 < v159 )
        break;
      while ( 1 )
      {
        v160 = *(_QWORD *)(v159 - 40);
        v161 = (_QWORD *)(v159 - 64);
        if ( v160 != v159 - 24 )
          _libc_free(v160, v146);
        if ( *v161 == v159 - 48 )
          break;
        _libc_free(*v161, v146);
        v159 += 64LL;
        if ( v158 < v159 )
          goto LABEL_191;
      }
    }
LABEL_191:
    ;
  }
  sub_E66D20(a1 + 1248);
  v162 = *(_QWORD *)(a1 + 184);
  *(_QWORD *)(a1 + 184) = 0;
  if ( v162 )
  {
    v163 = *(_QWORD *)(v162 + 288);
    v164 = *(_QWORD *)(v162 + 280);
    if ( v163 != v164 )
    {
      do
      {
        v165 = *(unsigned int *)(v164 + 48);
        v166 = *(_QWORD *)(v164 + 32);
        v164 += 56;
        v146 = 16 * v165;
        sub_C7D6A0(v166, v146, 4);
      }
      while ( v163 != v164 );
      v164 = *(_QWORD *)(v162 + 280);
    }
    if ( v164 )
    {
      v146 = *(_QWORD *)(v162 + 296) - v164;
      j_j___libc_free_0(v164, v146);
    }
    v167 = *(_QWORD *)(v162 + 256);
    if ( v167 )
    {
      v146 = *(_QWORD *)(v162 + 272) - v167;
      j_j___libc_free_0(v167, v146);
    }
    sub_E62F00(*(_QWORD *)(v162 + 224));
    v168 = *(_QWORD *)(v162 + 64);
    if ( v168 != v162 + 80 )
      _libc_free(v168, v146);
    v169 = *(_QWORD *)(v162 + 40);
    if ( v162 + 64 != v169 )
      _libc_free(v169, v146);
    v170 = *(_QWORD *)(v162 + 8);
    if ( *(_DWORD *)(v162 + 20) )
    {
      v171 = *(unsigned int *)(v162 + 16);
      if ( (_DWORD)v171 )
      {
        v172 = 8 * v171;
        v173 = 0;
        do
        {
          v174 = *(_QWORD **)(v170 + v173);
          if ( v174 && v174 != (_QWORD *)-8LL )
          {
            v146 = *v174 + 17LL;
            sub_C7D6A0((__int64)v174, v146, 8);
            v170 = *(_QWORD *)(v162 + 8);
          }
          v173 += 8;
        }
        while ( v172 != v173 );
      }
    }
    _libc_free(v170, v146);
    v146 = 312;
    j_j___libc_free_0(v162, 312);
  }
  sub_E66970(a1 + 2264);
  if ( *(_DWORD *)(a1 + 1420) )
  {
    v175 = 0;
    v176 = *(unsigned int *)(a1 + 1416);
    if ( (_DWORD)v176 )
    {
      do
      {
        *(_QWORD *)(*(_QWORD *)(a1 + 1408) + v175) = 0;
        v175 += 8;
      }
      while ( 8 * v176 != v175 );
    }
    *(_QWORD *)(a1 + 1420) = 0;
  }
  if ( *(_DWORD *)(a1 + 1356) )
  {
    v177 = 0;
    v178 = *(unsigned int *)(a1 + 1352);
    if ( (_DWORD)v178 )
    {
      do
      {
        *(_QWORD *)(*(_QWORD *)(a1 + 1344) + v177) = 0;
        v177 += 8;
      }
      while ( 8 * v178 != v177 );
    }
    *(_QWORD *)(a1 + 1356) = 0;
  }
  sub_E66D20(a1 + 192);
  sub_E66D20(a1 + 288);
  v179 = *(_DWORD *)(a1 + 1456);
  ++*(_QWORD *)(a1 + 1440);
  if ( !v179 )
  {
    if ( !*(_DWORD *)(a1 + 1460) )
      goto LABEL_226;
    v180 = *(unsigned int *)(a1 + 1464);
    if ( (unsigned int)v180 > 0x40 )
    {
      v146 = 16 * v180;
      sub_C7D6A0(*(_QWORD *)(a1 + 1448), 16 * v180, 8);
      *(_QWORD *)(a1 + 1448) = 0;
      *(_QWORD *)(a1 + 1456) = 0;
      *(_DWORD *)(a1 + 1464) = 0;
      goto LABEL_226;
    }
    goto LABEL_223;
  }
  v233 = 4 * v179;
  v146 = 64;
  v180 = *(unsigned int *)(a1 + 1464);
  if ( (unsigned int)(4 * v179) < 0x40 )
    v233 = 64;
  if ( (unsigned int)v180 <= v233 )
  {
LABEL_223:
    v181 = *(_DWORD **)(a1 + 1448);
    for ( i6 = &v181[4 * v180]; i6 != v181; v181 += 4 )
      *v181 = -1;
    *(_QWORD *)(a1 + 1456) = 0;
    goto LABEL_226;
  }
  v234 = v179 - 1;
  if ( !v234 )
  {
    v235 = *(_DWORD **)(a1 + 1448);
    v236 = 64;
LABEL_311:
    sub_C7D6A0((__int64)v235, 16 * v180, 8);
    v146 = 8;
    v237 = ((((((((4 * v236 / 3u + 1) | ((unsigned __int64)(4 * v236 / 3u + 1) >> 1)) >> 2)
              | (4 * v236 / 3u + 1)
              | ((unsigned __int64)(4 * v236 / 3u + 1) >> 1)) >> 4)
            | (((4 * v236 / 3u + 1) | ((unsigned __int64)(4 * v236 / 3u + 1) >> 1)) >> 2)
            | (4 * v236 / 3u + 1)
            | ((unsigned __int64)(4 * v236 / 3u + 1) >> 1)) >> 8)
          | (((((4 * v236 / 3u + 1) | ((unsigned __int64)(4 * v236 / 3u + 1) >> 1)) >> 2)
            | (4 * v236 / 3u + 1)
            | ((unsigned __int64)(4 * v236 / 3u + 1) >> 1)) >> 4)
          | (((4 * v236 / 3u + 1) | ((unsigned __int64)(4 * v236 / 3u + 1) >> 1)) >> 2)
          | (4 * v236 / 3u + 1)
          | ((unsigned __int64)(4 * v236 / 3u + 1) >> 1)) >> 16;
    v238 = (v237
          | (((((((4 * v236 / 3u + 1) | ((unsigned __int64)(4 * v236 / 3u + 1) >> 1)) >> 2)
              | (4 * v236 / 3u + 1)
              | ((unsigned __int64)(4 * v236 / 3u + 1) >> 1)) >> 4)
            | (((4 * v236 / 3u + 1) | ((unsigned __int64)(4 * v236 / 3u + 1) >> 1)) >> 2)
            | (4 * v236 / 3u + 1)
            | ((unsigned __int64)(4 * v236 / 3u + 1) >> 1)) >> 8)
          | (((((4 * v236 / 3u + 1) | ((unsigned __int64)(4 * v236 / 3u + 1) >> 1)) >> 2)
            | (4 * v236 / 3u + 1)
            | ((unsigned __int64)(4 * v236 / 3u + 1) >> 1)) >> 4)
          | (((4 * v236 / 3u + 1) | ((unsigned __int64)(4 * v236 / 3u + 1) >> 1)) >> 2)
          | (4 * v236 / 3u + 1)
          | ((unsigned __int64)(4 * v236 / 3u + 1) >> 1))
         + 1;
    *(_DWORD *)(a1 + 1464) = v238;
    v239 = (_DWORD *)sub_C7D670(16 * v238, 8);
    v240 = *(unsigned int *)(a1 + 1464);
    *(_QWORD *)(a1 + 1456) = 0;
    *(_QWORD *)(a1 + 1448) = v239;
    for ( i7 = &v239[4 * v240]; i7 != v239; v239 += 4 )
    {
      if ( v239 )
        *v239 = -1;
    }
    goto LABEL_226;
  }
  _BitScanReverse(&v234, v234);
  v235 = *(_DWORD **)(a1 + 1448);
  v236 = 1 << (33 - (v234 ^ 0x1F));
  if ( v236 < 64 )
    v236 = 64;
  if ( (_DWORD)v180 != v236 )
    goto LABEL_311;
  *(_QWORD *)(a1 + 1456) = 0;
  v257 = &v235[4 * (unsigned int)v180];
  do
  {
    if ( v235 )
      *v235 = -1;
    v235 += 4;
  }
  while ( v257 != v235 );
LABEL_226:
  v183 = *(_BYTE **)(a1 + 1696);
  *(_QWORD *)(a1 + 1536) = 0;
  *(_QWORD *)(a1 + 1704) = 0;
  *v183 = 0;
  sub_E63CD0(*(_QWORD **)(a1 + 1744), v146);
  ++*(_QWORD *)(a1 + 1800);
  *(_QWORD *)(a1 + 1752) = a1 + 1736;
  *(_QWORD *)(a1 + 1760) = a1 + 1736;
  v184 = *(_DWORD *)(a1 + 1816);
  *(_QWORD *)(a1 + 1744) = 0;
  *(_QWORD *)(a1 + 1768) = 0;
  if ( !v184 )
  {
    if ( !*(_DWORD *)(a1 + 1820) )
      goto LABEL_232;
    v185 = *(unsigned int *)(a1 + 1824);
    if ( (unsigned int)v185 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 1808), 8 * v185, 8);
      *(_QWORD *)(a1 + 1808) = 0;
      *(_QWORD *)(a1 + 1816) = 0;
      *(_DWORD *)(a1 + 1824) = 0;
      goto LABEL_232;
    }
    goto LABEL_229;
  }
  v228 = 4 * v184;
  v185 = *(unsigned int *)(a1 + 1824);
  if ( (unsigned int)(4 * v184) < 0x40 )
    v228 = 64;
  if ( v228 >= (unsigned int)v185 )
  {
LABEL_229:
    v186 = *(_QWORD **)(a1 + 1808);
    for ( i8 = &v186[v185]; i8 != v186; ++v186 )
      *v186 = -4096;
    *(_QWORD *)(a1 + 1816) = 0;
    goto LABEL_232;
  }
  v229 = v184 - 1;
  if ( v229 )
  {
    _BitScanReverse(&v229, v229);
    v230 = *(_QWORD **)(a1 + 1808);
    v231 = 1 << (33 - (v229 ^ 0x1F));
    if ( v231 < 64 )
      v231 = 64;
    if ( (_DWORD)v185 == v231 )
    {
      *(_QWORD *)(a1 + 1816) = 0;
      v232 = &v230[v185];
      do
      {
        if ( v230 )
          *v230 = -4096;
        ++v230;
      }
      while ( v232 != v230 );
      goto LABEL_232;
    }
  }
  else
  {
    v230 = *(_QWORD **)(a1 + 1808);
    v231 = 64;
  }
  sub_C7D6A0((__int64)v230, 8 * v185, 8);
  v242 = ((((((((4 * v231 / 3u + 1) | ((unsigned __int64)(4 * v231 / 3u + 1) >> 1)) >> 2)
            | (4 * v231 / 3u + 1)
            | ((unsigned __int64)(4 * v231 / 3u + 1) >> 1)) >> 4)
          | (((4 * v231 / 3u + 1) | ((unsigned __int64)(4 * v231 / 3u + 1) >> 1)) >> 2)
          | (4 * v231 / 3u + 1)
          | ((unsigned __int64)(4 * v231 / 3u + 1) >> 1)) >> 8)
        | (((((4 * v231 / 3u + 1) | ((unsigned __int64)(4 * v231 / 3u + 1) >> 1)) >> 2)
          | (4 * v231 / 3u + 1)
          | ((unsigned __int64)(4 * v231 / 3u + 1) >> 1)) >> 4)
        | (((4 * v231 / 3u + 1) | ((unsigned __int64)(4 * v231 / 3u + 1) >> 1)) >> 2)
        | (4 * v231 / 3u + 1)
        | ((unsigned __int64)(4 * v231 / 3u + 1) >> 1)) >> 16;
  v243 = (v242
        | (((((((4 * v231 / 3u + 1) | ((unsigned __int64)(4 * v231 / 3u + 1) >> 1)) >> 2)
            | (4 * v231 / 3u + 1)
            | ((unsigned __int64)(4 * v231 / 3u + 1) >> 1)) >> 4)
          | (((4 * v231 / 3u + 1) | ((unsigned __int64)(4 * v231 / 3u + 1) >> 1)) >> 2)
          | (4 * v231 / 3u + 1)
          | ((unsigned __int64)(4 * v231 / 3u + 1) >> 1)) >> 8)
        | (((((4 * v231 / 3u + 1) | ((unsigned __int64)(4 * v231 / 3u + 1) >> 1)) >> 2)
          | (4 * v231 / 3u + 1)
          | ((unsigned __int64)(4 * v231 / 3u + 1) >> 1)) >> 4)
        | (((4 * v231 / 3u + 1) | ((unsigned __int64)(4 * v231 / 3u + 1) >> 1)) >> 2)
        | (4 * v231 / 3u + 1)
        | ((unsigned __int64)(4 * v231 / 3u + 1) >> 1))
       + 1;
  *(_DWORD *)(a1 + 1824) = v243;
  v244 = (_QWORD *)sub_C7D670(8 * v243, 8);
  v245 = *(unsigned int *)(a1 + 1824);
  *(_QWORD *)(a1 + 1816) = 0;
  *(_QWORD *)(a1 + 1808) = v244;
  for ( i9 = &v244[v245]; i9 != v244; ++v244 )
  {
    if ( v244 )
      *v244 = -4096;
  }
LABEL_232:
  *(_DWORD *)(a1 + 1840) = 0;
  v188 = *(_QWORD *)(a1 + 1848);
  if ( v188 != *(_QWORD *)(a1 + 1856) )
    *(_QWORD *)(a1 + 1856) = v188;
  v189 = *(_DWORD *)(a1 + 1988);
  *(_QWORD *)(a1 + 1872) = 0;
  *(_QWORD *)(a1 + 1880) = 0;
  *(_DWORD *)(a1 + 1912) = 0;
  *(_QWORD *)(a1 + 1776) = 0;
  *(_QWORD *)(a1 + 1784) = 0x10000;
  if ( v189 )
  {
    v190 = 0;
    v191 = *(unsigned int *)(a1 + 1984);
    v192 = 8 * v191;
    if ( (_DWORD)v191 )
    {
      do
      {
        v193 = (_QWORD **)(v190 + *(_QWORD *)(a1 + 1976));
        v194 = *v193;
        if ( *v193 && v194 != (_QWORD *)-8LL )
          sub_C7D6A0((__int64)v194, *v194 + 17LL, 8);
        v190 += 8;
        *v193 = 0;
      }
      while ( v190 != v192 );
    }
    *(_QWORD *)(a1 + 1988) = 0;
  }
  if ( *(_DWORD *)(a1 + 2060) )
  {
    v195 = 0;
    v196 = *(unsigned int *)(a1 + 2056);
    v197 = 8 * v196;
    if ( (_DWORD)v196 )
    {
      do
      {
        v198 = (_QWORD **)(v195 + *(_QWORD *)(a1 + 2048));
        v199 = *v198;
        if ( *v198 && v199 != (_QWORD *)-8LL )
          sub_C7D6A0((__int64)v199, *v199 + 17LL, 8);
        v195 += 8;
        *v198 = 0;
      }
      while ( v197 != v195 );
    }
    *(_QWORD *)(a1 + 2060) = 0;
  }
  sub_E630D0(*(_QWORD **)(a1 + 2088));
  *(_QWORD *)(a1 + 2088) = 0;
  v200 = *(_QWORD **)(a1 + 2016);
  *(_QWORD *)(a1 + 2096) = a1 + 2080;
  *(_QWORD *)(a1 + 2104) = a1 + 2080;
  *(_QWORD *)(a1 + 2112) = 0;
  sub_E639D0(v200);
  *(_QWORD *)(a1 + 2016) = 0;
  v201 = *(_QWORD **)(a1 + 2136);
  *(_QWORD *)(a1 + 2024) = a1 + 2008;
  *(_QWORD *)(a1 + 2032) = a1 + 2008;
  *(_QWORD *)(a1 + 2040) = 0;
  sub_E633D0(v201);
  *(_QWORD *)(a1 + 2136) = 0;
  v202 = *(_QWORD **)(a1 + 2184);
  *(_QWORD *)(a1 + 2144) = a1 + 2128;
  *(_QWORD *)(a1 + 2152) = a1 + 2128;
  *(_QWORD *)(a1 + 2160) = 0;
  sub_E636D0(v202);
  v203 = *(_DWORD *)(a1 + 2228);
  *(_QWORD *)(a1 + 2184) = 0;
  *(_QWORD *)(a1 + 2192) = a1 + 2176;
  *(_QWORD *)(a1 + 2200) = a1 + 2176;
  *(_QWORD *)(a1 + 2208) = 0;
  if ( v203 )
  {
    v204 = 0;
    v205 = *(unsigned int *)(a1 + 2224);
    v206 = 8 * v205;
    if ( (_DWORD)v205 )
    {
      do
      {
        v207 = (_QWORD **)(v204 + *(_QWORD *)(a1 + 2216));
        v208 = *v207;
        if ( *v207 && v208 != (_QWORD *)-8LL )
          sub_C7D6A0((__int64)v208, *v208 + 17LL, 8);
        v204 += 8;
        *v207 = 0;
      }
      while ( v204 != v206 );
    }
    *(_QWORD *)(a1 + 2228) = 0;
  }
  v209 = *(_DWORD *)(a1 + 2424);
  ++*(_QWORD *)(a1 + 2408);
  if ( !v209 )
  {
    if ( !*(_DWORD *)(a1 + 2428) )
      goto LABEL_261;
    v210 = *(unsigned int *)(a1 + 2432);
    if ( (unsigned int)v210 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 2416), 32 * v210, 8);
      *(_QWORD *)(a1 + 2416) = 0;
      *(_QWORD *)(a1 + 2424) = 0;
      *(_DWORD *)(a1 + 2432) = 0;
      goto LABEL_261;
    }
    goto LABEL_258;
  }
  v223 = 4 * v209;
  v210 = *(unsigned int *)(a1 + 2432);
  if ( (unsigned int)(4 * v209) < 0x40 )
    v223 = 64;
  if ( (unsigned int)v210 <= v223 )
  {
LABEL_258:
    v211 = *(_QWORD *)(a1 + 2416);
    for ( i10 = v211 + 32 * v210; i10 != v211; *(_DWORD *)(v211 - 32) = -1 )
    {
      *(_QWORD *)(v211 + 8) = -1;
      v211 += 32;
      *(_QWORD *)(v211 - 16) = 0;
      *(_DWORD *)(v211 - 28) = -1;
    }
    *(_QWORD *)(a1 + 2424) = 0;
    goto LABEL_261;
  }
  v224 = v209 - 1;
  if ( v224 )
  {
    _BitScanReverse(&v224, v224);
    v225 = *(_QWORD *)(a1 + 2416);
    v226 = 1 << (33 - (v224 ^ 0x1F));
    if ( v226 < 64 )
      v226 = 64;
    if ( (_DWORD)v210 == v226 )
    {
      *(_QWORD *)(a1 + 2424) = 0;
      v227 = v225 + 32LL * (unsigned int)v210;
      do
      {
        if ( v225 )
        {
          *(_DWORD *)v225 = -1;
          *(_DWORD *)(v225 + 4) = -1;
          *(_QWORD *)(v225 + 8) = -1;
          *(_QWORD *)(v225 + 16) = 0;
        }
        v225 += 32;
      }
      while ( v227 != v225 );
      goto LABEL_261;
    }
  }
  else
  {
    v225 = *(_QWORD *)(a1 + 2416);
    v226 = 64;
  }
  sub_C7D6A0(v225, 32 * v210, 8);
  v252 = ((((((((4 * v226 / 3u + 1) | ((unsigned __int64)(4 * v226 / 3u + 1) >> 1)) >> 2)
            | (4 * v226 / 3u + 1)
            | ((unsigned __int64)(4 * v226 / 3u + 1) >> 1)) >> 4)
          | (((4 * v226 / 3u + 1) | ((unsigned __int64)(4 * v226 / 3u + 1) >> 1)) >> 2)
          | (4 * v226 / 3u + 1)
          | ((unsigned __int64)(4 * v226 / 3u + 1) >> 1)) >> 8)
        | (((((4 * v226 / 3u + 1) | ((unsigned __int64)(4 * v226 / 3u + 1) >> 1)) >> 2)
          | (4 * v226 / 3u + 1)
          | ((unsigned __int64)(4 * v226 / 3u + 1) >> 1)) >> 4)
        | (((4 * v226 / 3u + 1) | ((unsigned __int64)(4 * v226 / 3u + 1) >> 1)) >> 2)
        | (4 * v226 / 3u + 1)
        | ((unsigned __int64)(4 * v226 / 3u + 1) >> 1)) >> 16;
  v253 = (v252
        | (((((((4 * v226 / 3u + 1) | ((unsigned __int64)(4 * v226 / 3u + 1) >> 1)) >> 2)
            | (4 * v226 / 3u + 1)
            | ((unsigned __int64)(4 * v226 / 3u + 1) >> 1)) >> 4)
          | (((4 * v226 / 3u + 1) | ((unsigned __int64)(4 * v226 / 3u + 1) >> 1)) >> 2)
          | (4 * v226 / 3u + 1)
          | ((unsigned __int64)(4 * v226 / 3u + 1) >> 1)) >> 8)
        | (((((4 * v226 / 3u + 1) | ((unsigned __int64)(4 * v226 / 3u + 1) >> 1)) >> 2)
          | (4 * v226 / 3u + 1)
          | ((unsigned __int64)(4 * v226 / 3u + 1) >> 1)) >> 4)
        | (((4 * v226 / 3u + 1) | ((unsigned __int64)(4 * v226 / 3u + 1) >> 1)) >> 2)
        | (4 * v226 / 3u + 1)
        | ((unsigned __int64)(4 * v226 / 3u + 1) >> 1))
       + 1;
  *(_DWORD *)(a1 + 2432) = v253;
  v254 = sub_C7D670(32 * v253, 8);
  v255 = *(unsigned int *)(a1 + 2432);
  *(_QWORD *)(a1 + 2424) = 0;
  *(_QWORD *)(a1 + 2416) = v254;
  for ( i11 = v254 + 32 * v255; i11 != v254; v254 += 32 )
  {
    if ( v254 )
    {
      *(_DWORD *)v254 = -1;
      *(_DWORD *)(v254 + 4) = -1;
      *(_QWORD *)(v254 + 8) = -1;
      *(_QWORD *)(v254 + 16) = 0;
    }
  }
LABEL_261:
  v213 = *(_DWORD *)(a1 + 2456);
  ++*(_QWORD *)(a1 + 2440);
  if ( !v213 )
  {
    if ( !*(_DWORD *)(a1 + 2460) )
      goto LABEL_267;
    v214 = *(unsigned int *)(a1 + 2464);
    if ( (unsigned int)v214 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 2448), 16 * v214, 8);
      *(_QWORD *)(a1 + 2448) = 0;
      *(_QWORD *)(a1 + 2456) = 0;
      *(_DWORD *)(a1 + 2464) = 0;
      goto LABEL_267;
    }
    goto LABEL_264;
  }
  v218 = 4 * v213;
  v214 = *(unsigned int *)(a1 + 2464);
  if ( (unsigned int)(4 * v213) < 0x40 )
    v218 = 64;
  if ( v218 >= (unsigned int)v214 )
  {
LABEL_264:
    v215 = *(_QWORD **)(a1 + 2448);
    for ( i12 = &v215[2 * v214]; i12 != v215; *(v215 - 1) = 0 )
    {
      *v215 = -1;
      v215 += 2;
    }
    *(_QWORD *)(a1 + 2456) = 0;
    goto LABEL_267;
  }
  v219 = v213 - 1;
  if ( v219 )
  {
    _BitScanReverse(&v219, v219);
    v220 = *(_QWORD **)(a1 + 2448);
    v221 = 1 << (33 - (v219 ^ 0x1F));
    if ( v221 < 64 )
      v221 = 64;
    if ( (_DWORD)v214 == v221 )
    {
      *(_QWORD *)(a1 + 2456) = 0;
      v222 = &v220[2 * (unsigned int)v214];
      do
      {
        if ( v220 )
        {
          *v220 = -1;
          v220[1] = 0;
        }
        v220 += 2;
      }
      while ( v222 != v220 );
      goto LABEL_267;
    }
  }
  else
  {
    v220 = *(_QWORD **)(a1 + 2448);
    v221 = 64;
  }
  sub_C7D6A0((__int64)v220, 16 * v214, 8);
  v247 = ((((((((4 * v221 / 3u + 1) | ((unsigned __int64)(4 * v221 / 3u + 1) >> 1)) >> 2)
            | (4 * v221 / 3u + 1)
            | ((unsigned __int64)(4 * v221 / 3u + 1) >> 1)) >> 4)
          | (((4 * v221 / 3u + 1) | ((unsigned __int64)(4 * v221 / 3u + 1) >> 1)) >> 2)
          | (4 * v221 / 3u + 1)
          | ((unsigned __int64)(4 * v221 / 3u + 1) >> 1)) >> 8)
        | (((((4 * v221 / 3u + 1) | ((unsigned __int64)(4 * v221 / 3u + 1) >> 1)) >> 2)
          | (4 * v221 / 3u + 1)
          | ((unsigned __int64)(4 * v221 / 3u + 1) >> 1)) >> 4)
        | (((4 * v221 / 3u + 1) | ((unsigned __int64)(4 * v221 / 3u + 1) >> 1)) >> 2)
        | (4 * v221 / 3u + 1)
        | ((unsigned __int64)(4 * v221 / 3u + 1) >> 1)) >> 16;
  v248 = (v247
        | (((((((4 * v221 / 3u + 1) | ((unsigned __int64)(4 * v221 / 3u + 1) >> 1)) >> 2)
            | (4 * v221 / 3u + 1)
            | ((unsigned __int64)(4 * v221 / 3u + 1) >> 1)) >> 4)
          | (((4 * v221 / 3u + 1) | ((unsigned __int64)(4 * v221 / 3u + 1) >> 1)) >> 2)
          | (4 * v221 / 3u + 1)
          | ((unsigned __int64)(4 * v221 / 3u + 1) >> 1)) >> 8)
        | (((((4 * v221 / 3u + 1) | ((unsigned __int64)(4 * v221 / 3u + 1) >> 1)) >> 2)
          | (4 * v221 / 3u + 1)
          | ((unsigned __int64)(4 * v221 / 3u + 1) >> 1)) >> 4)
        | (((4 * v221 / 3u + 1) | ((unsigned __int64)(4 * v221 / 3u + 1) >> 1)) >> 2)
        | (4 * v221 / 3u + 1)
        | ((unsigned __int64)(4 * v221 / 3u + 1) >> 1))
       + 1;
  *(_DWORD *)(a1 + 2464) = v248;
  v249 = (_QWORD *)sub_C7D670(16 * v248, 8);
  v250 = *(unsigned int *)(a1 + 2464);
  *(_QWORD *)(a1 + 2456) = 0;
  *(_QWORD *)(a1 + 2448) = v249;
  for ( i13 = &v249[2 * v250]; i13 != v249; v249 += 2 )
  {
    if ( v249 )
    {
      *v249 = -1;
      v249[1] = 0;
    }
  }
LABEL_267:
  *(_BYTE *)(a1 + 2376) = 0;
  *(_WORD *)(a1 + 1792) = 0;
  *(_DWORD *)(a1 + 1796) = 0;
  return 0;
}
