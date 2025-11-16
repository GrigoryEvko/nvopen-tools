// Function: sub_E68A10
// Address: 0xe68a10
//
__int64 __fastcall sub_E68A10(__int64 a1)
{
  __int64 v2; // rsi
  __int64 v3; // rax
  __int64 v4; // rdi
  _QWORD *v5; // r12
  _QWORD *v6; // r14
  _QWORD *v7; // r15
  _QWORD *v8; // r15
  __int64 v9; // r13
  __int64 v10; // r14
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // r13
  __int64 v15; // r12
  _QWORD *v16; // rdi
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // r13
  __int64 v21; // r12
  _QWORD *v22; // rdi
  __int64 v23; // rax
  __int64 v24; // r8
  __int64 v25; // r13
  __int64 v26; // r12
  _QWORD *v27; // rdi
  __int64 v28; // rax
  __int64 v29; // r8
  __int64 v30; // r13
  __int64 v31; // r12
  _QWORD *v32; // rdi
  _QWORD *v33; // r14
  __int64 v34; // rdi
  _QWORD *v35; // r12
  _QWORD *v36; // r15
  __int64 v37; // r13
  _QWORD *v38; // rdi
  __int64 v39; // rsi
  __int64 v40; // rdi
  __int64 v41; // rax
  __int64 v42; // rdi
  __int64 v43; // rdi
  __int64 v44; // rsi
  __int64 v45; // rdi
  _QWORD *v46; // r13
  _QWORD *v47; // r12
  _QWORD *v48; // rdi
  __int64 v49; // rdi
  __int64 v50; // rdi
  __int64 v51; // rdi
  __int64 v52; // rsi
  __int64 v53; // rsi
  __int64 *v54; // r13
  __int64 v55; // rsi
  __int64 v56; // rdx
  unsigned int v57; // ecx
  __int64 v58; // r12
  bool v59; // cf
  __int64 v60; // rcx
  unsigned __int64 v61; // rax
  unsigned __int64 v62; // r12
  unsigned __int64 i; // r15
  __int64 v64; // rdi
  _QWORD *v65; // r14
  _QWORD *v66; // r14
  __int64 v67; // r15
  unsigned __int64 v68; // r13
  unsigned __int64 v69; // r12
  __int64 v70; // rdi
  _QWORD *v71; // r15
  _QWORD *v72; // r13
  __int64 v73; // rsi
  _QWORD *v74; // r14
  __int64 v75; // rdx
  unsigned int v76; // ecx
  __int64 v77; // rax
  unsigned __int64 v78; // r15
  unsigned __int64 v79; // r12
  __int64 v80; // rdi
  _QWORD *v81; // r14
  _QWORD *k; // r15
  unsigned __int64 v83; // r13
  unsigned __int64 v84; // r12
  __int64 v85; // rdi
  __int64 *v86; // r13
  __int64 v87; // rsi
  __int64 *v88; // r14
  __int64 v89; // rdx
  unsigned int v90; // ecx
  __int64 v91; // r12
  __int64 v92; // rcx
  unsigned __int64 v93; // rax
  unsigned __int64 v94; // r12
  unsigned __int64 m; // r15
  unsigned __int64 v96; // rdi
  _QWORD *v97; // r14
  _QWORD *n; // r15
  unsigned __int64 v99; // r13
  unsigned __int64 ii; // r12
  unsigned __int64 v101; // rdi
  __int64 *v102; // rax
  __int64 v103; // rdx
  __int64 v104; // rsi
  unsigned int v105; // ecx
  __int64 v106; // r12
  _QWORD *v107; // rcx
  unsigned __int64 v108; // r12
  _QWORD *v109; // r15
  _QWORD *v110; // r14
  __int64 v111; // r15
  unsigned __int64 v112; // r12
  _QWORD *v113; // rcx
  _QWORD *v114; // r15
  _QWORD *v115; // r14
  __int64 *v116; // rax
  __int64 v117; // rdx
  __int64 v118; // rsi
  unsigned int v119; // ecx
  __int64 v120; // r12
  _QWORD *v121; // rcx
  unsigned __int64 v122; // r12
  _QWORD *v123; // r15
  _QWORD *v124; // r14
  __int64 v125; // r15
  unsigned __int64 v126; // r12
  _QWORD *v127; // rcx
  _QWORD *v128; // r15
  _QWORD *v129; // r14
  __int64 *v130; // r14
  __int64 *v131; // r12
  __int64 mm; // rax
  __int64 v133; // rdi
  unsigned int v134; // ecx
  __int64 *v135; // r12
  __int64 *v136; // r13
  __int64 v137; // rdi
  __int64 v138; // rdi
  __int64 *v139; // rax
  __int64 v140; // rdx
  __int64 v141; // rsi
  unsigned int v142; // ecx
  __int64 v143; // r12
  _QWORD *v144; // rcx
  unsigned __int64 v145; // r12
  _QWORD *v146; // r15
  _QWORD *v147; // r14
  __int64 v148; // r15
  unsigned __int64 v149; // r12
  _QWORD *v150; // rcx
  _QWORD *v151; // r15
  _QWORD *v152; // r14
  _QWORD *v153; // rax
  __int64 v154; // rsi
  __int64 v155; // rdx
  unsigned int v156; // ecx
  __int64 v157; // r12
  unsigned __int64 v158; // r12
  unsigned __int64 v159; // rax
  unsigned __int64 i1; // r15
  __int64 v161; // rdi
  unsigned __int64 v162; // rdi
  __int64 v163; // r15
  unsigned __int64 v164; // r12
  unsigned __int64 i3; // r15
  __int64 v166; // rdi
  unsigned __int64 v167; // rdi
  __int64 *v168; // rax
  __int64 v169; // rdx
  __int64 v170; // rsi
  unsigned int v171; // ecx
  __int64 v172; // r12
  _QWORD *v173; // rcx
  unsigned __int64 v174; // r12
  _QWORD *v175; // r15
  _QWORD *v176; // r14
  __int64 v177; // r15
  unsigned __int64 v178; // r12
  _QWORD *v179; // rcx
  _QWORD *v180; // r15
  _QWORD *v181; // r14
  __int64 *v182; // rax
  __int64 v183; // rdx
  __int64 v184; // rsi
  unsigned int v185; // ecx
  __int64 v186; // r12
  _QWORD *v187; // rcx
  unsigned __int64 v188; // r12
  _QWORD *v189; // r15
  _QWORD *v190; // r14
  __int64 v191; // r15
  unsigned __int64 v192; // r12
  _QWORD *v193; // rcx
  _QWORD *v194; // r15
  _QWORD *v195; // r14
  __int64 *v196; // rax
  __int64 v197; // rdx
  __int64 v198; // rsi
  unsigned int v199; // ecx
  __int64 v200; // r12
  _QWORD *v201; // rcx
  unsigned __int64 v202; // r12
  _QWORD *v203; // r15
  _QWORD *v204; // r14
  __int64 v205; // r15
  unsigned __int64 v206; // r12
  _QWORD *v207; // rcx
  _QWORD *v208; // r15
  _QWORD *v209; // r14
  __int64 v210; // r13
  __int64 v211; // r14
  __int64 v212; // r12
  __int64 v213; // rsi
  __int64 v214; // rdi
  __int64 v215; // rdi
  __int64 v216; // rdi
  __int64 v217; // rdi
  __int64 v218; // r8
  void (__fastcall *v219)(__int64, __int64, __int64); // rax
  __int64 v220; // rdi
  __int64 **v221; // r13
  __int64 *v222; // r14
  char *v223; // r12
  __int64 *v224; // r14
  __int64 *v225; // r12
  __int64 *v226; // rdi
  __int64 v227; // rdi
  __int64 result; // rax
  __int64 v229; // rax
  __int64 v230; // r14
  __int64 v231; // r12
  _QWORD *v232; // rdi
  __int64 v233; // [rsp+0h] [rbp-50h]
  __int64 v234; // [rsp+8h] [rbp-48h]
  _QWORD *v235; // [rsp+10h] [rbp-40h]
  __int64 *v236; // [rsp+10h] [rbp-40h]
  _QWORD *jj; // [rsp+10h] [rbp-40h]
  __int64 *v238; // [rsp+10h] [rbp-40h]
  _QWORD *kk; // [rsp+10h] [rbp-40h]
  __int64 *v240; // [rsp+10h] [rbp-40h]
  _QWORD *nn; // [rsp+10h] [rbp-40h]
  _QWORD *v242; // [rsp+10h] [rbp-40h]
  _QWORD *i2; // [rsp+10h] [rbp-40h]
  __int64 *v244; // [rsp+10h] [rbp-40h]
  _QWORD *i4; // [rsp+10h] [rbp-40h]
  __int64 *v246; // [rsp+10h] [rbp-40h]
  _QWORD *i5; // [rsp+10h] [rbp-40h]
  __int64 *v248; // [rsp+10h] [rbp-40h]
  _QWORD *i6; // [rsp+10h] [rbp-40h]
  __int64 v250; // [rsp+18h] [rbp-38h]
  _QWORD *v251; // [rsp+18h] [rbp-38h]
  __int64 *v252; // [rsp+18h] [rbp-38h]
  _QWORD *j; // [rsp+18h] [rbp-38h]
  __int64 *v254; // [rsp+18h] [rbp-38h]
  _QWORD *v255; // [rsp+18h] [rbp-38h]
  __int64 *v256; // [rsp+18h] [rbp-38h]
  _QWORD *v257; // [rsp+18h] [rbp-38h]
  __int64 *v258; // [rsp+18h] [rbp-38h]
  _QWORD *v259; // [rsp+18h] [rbp-38h]
  _QWORD *v260; // [rsp+18h] [rbp-38h]
  _QWORD *v261; // [rsp+18h] [rbp-38h]
  __int64 *v262; // [rsp+18h] [rbp-38h]
  _QWORD *v263; // [rsp+18h] [rbp-38h]
  __int64 *v264; // [rsp+18h] [rbp-38h]
  _QWORD *v265; // [rsp+18h] [rbp-38h]
  __int64 *v266; // [rsp+18h] [rbp-38h]
  _QWORD *v267; // [rsp+18h] [rbp-38h]

  if ( *(_BYTE *)(a1 + 2360) )
    sub_E66E10(a1);
  sub_C7D6A0(*(_QWORD *)(a1 + 2448), 16LL * *(unsigned int *)(a1 + 2464), 8);
  v2 = 32LL * *(unsigned int *)(a1 + 2432);
  sub_C7D6A0(*(_QWORD *)(a1 + 2416), v2, 8);
  if ( *(_DWORD *)(a1 + 2396) )
  {
    v3 = *(unsigned int *)(a1 + 2392);
    v4 = *(_QWORD *)(a1 + 2384);
    if ( (_DWORD)v3 )
    {
      v250 = 0;
      v234 = 8 * v3;
      do
      {
        v5 = *(_QWORD **)(v4 + v250);
        if ( v5 && v5 != (_QWORD *)-8LL )
        {
          v6 = (_QWORD *)v5[9];
          v7 = (_QWORD *)v5[8];
          v233 = *v5 + 97LL;
          if ( v6 != v7 )
          {
            do
            {
              if ( (_QWORD *)*v7 != v7 + 2 )
                j_j___libc_free_0(*v7, v7[2] + 1LL);
              v7 += 4;
            }
            while ( v6 != v7 );
            v7 = (_QWORD *)v5[8];
          }
          if ( v7 )
            j_j___libc_free_0(v7, v5[10] - (_QWORD)v7);
          v8 = (_QWORD *)v5[5];
          v235 = (_QWORD *)v5[6];
          if ( v235 != v8 )
          {
            do
            {
              v9 = v8[3];
              v10 = v8[2];
              if ( v9 != v10 )
              {
                do
                {
                  if ( *(_DWORD *)(v10 + 32) > 0x40u )
                  {
                    v11 = *(_QWORD *)(v10 + 24);
                    if ( v11 )
                      j_j___libc_free_0_0(v11);
                  }
                  v10 += 40;
                }
                while ( v9 != v10 );
                v10 = v8[2];
              }
              if ( v10 )
                j_j___libc_free_0(v10, v8[4] - v10);
              v8 += 6;
            }
            while ( v235 != v8 );
            v8 = (_QWORD *)v5[5];
          }
          if ( v8 )
            j_j___libc_free_0(v8, v5[7] - (_QWORD)v8);
          v2 = v233;
          sub_C7D6A0((__int64)v5, v233, 8);
          v4 = *(_QWORD *)(a1 + 2384);
        }
        v250 += 8;
      }
      while ( v234 != v250 );
    }
  }
  else
  {
    v4 = *(_QWORD *)(a1 + 2384);
  }
  _libc_free(v4, v2);
  sub_E66970(a1 + 2264);
  sub_B72320(a1 + 2264, v2);
  if ( *(_DWORD *)(a1 + 2252) )
  {
    v12 = *(unsigned int *)(a1 + 2248);
    v13 = *(_QWORD *)(a1 + 2240);
    if ( (_DWORD)v12 )
    {
      v14 = 8 * v12;
      v15 = 0;
      do
      {
        v16 = *(_QWORD **)(v13 + v15);
        if ( v16 != (_QWORD *)-8LL && v16 )
        {
          v2 = *v16 + 17LL;
          sub_C7D6A0((__int64)v16, v2, 8);
          v13 = *(_QWORD *)(a1 + 2240);
        }
        v15 += 8;
      }
      while ( v14 != v15 );
    }
  }
  else
  {
    v13 = *(_QWORD *)(a1 + 2240);
  }
  _libc_free(v13, v2);
  v17 = *(unsigned int *)(a1 + 2228);
  if ( (_DWORD)v17 )
  {
    v18 = *(unsigned int *)(a1 + 2224);
    v19 = *(_QWORD *)(a1 + 2216);
    if ( (_DWORD)v18 )
    {
      v20 = 8 * v18;
      v21 = 0;
      do
      {
        v22 = *(_QWORD **)(v19 + v21);
        if ( v22 && v22 != (_QWORD *)-8LL )
        {
          v17 = *v22 + 17LL;
          sub_C7D6A0((__int64)v22, v17, 8);
          v19 = *(_QWORD *)(a1 + 2216);
        }
        v21 += 8;
      }
      while ( v20 != v21 );
    }
  }
  else
  {
    v19 = *(_QWORD *)(a1 + 2216);
  }
  _libc_free(v19, v17);
  sub_E636D0(*(_QWORD **)(a1 + 2184));
  sub_E633D0(*(_QWORD **)(a1 + 2136));
  sub_E630D0(*(_QWORD **)(a1 + 2088));
  if ( *(_DWORD *)(a1 + 2060) )
  {
    v23 = *(unsigned int *)(a1 + 2056);
    v24 = *(_QWORD *)(a1 + 2048);
    if ( (_DWORD)v23 )
    {
      v25 = 8 * v23;
      v26 = 0;
      do
      {
        v27 = *(_QWORD **)(v24 + v26);
        if ( v27 && v27 != (_QWORD *)-8LL )
        {
          v17 = *v27 + 17LL;
          sub_C7D6A0((__int64)v27, v17, 8);
          v24 = *(_QWORD *)(a1 + 2048);
        }
        v26 += 8;
      }
      while ( v25 != v26 );
    }
  }
  else
  {
    v24 = *(_QWORD *)(a1 + 2048);
  }
  _libc_free(v24, v17);
  sub_E639D0(*(_QWORD **)(a1 + 2016));
  if ( *(_DWORD *)(a1 + 1988) )
  {
    v28 = *(unsigned int *)(a1 + 1984);
    v29 = *(_QWORD *)(a1 + 1976);
    if ( (_DWORD)v28 )
    {
      v30 = 8 * v28;
      v31 = 0;
      do
      {
        v32 = *(_QWORD **)(v29 + v31);
        if ( v32 && v32 != (_QWORD *)-8LL )
        {
          v17 = *v32 + 17LL;
          sub_C7D6A0((__int64)v32, v17, 8);
          v29 = *(_QWORD *)(a1 + 1976);
        }
        v31 += 8;
      }
      while ( v30 != v31 );
    }
  }
  else
  {
    v29 = *(_QWORD *)(a1 + 1976);
  }
  _libc_free(v29, v17);
  v251 = *(_QWORD **)(a1 + 1936);
  while ( v251 )
  {
    v33 = v251;
    v34 = v251[9];
    v251 = (_QWORD *)*v251;
    if ( v34 )
      j_j___libc_free_0(v34, v33[11] - v34);
    v35 = (_QWORD *)v33[4];
    while ( v35 )
    {
      v36 = v35;
      v35 = (_QWORD *)*v35;
      v37 = v36[3];
      if ( v37 )
      {
        sub_E63F00((_QWORD *)v36[3]);
        j_j___libc_free_0(v37, 96);
      }
      j_j___libc_free_0(v36, 40);
    }
    memset((void *)v33[2], 0, 8LL * v33[3]);
    v38 = (_QWORD *)v33[2];
    v33[5] = 0;
    v33[4] = 0;
    if ( v38 != v33 + 8 )
      j_j___libc_free_0(v38, 8LL * v33[3]);
    j_j___libc_free_0(v33, 112);
  }
  v39 = 0;
  memset(*(void **)(a1 + 1920), 0, 8LL * *(_QWORD *)(a1 + 1928));
  v40 = *(_QWORD *)(a1 + 1920);
  *(_QWORD *)(a1 + 1944) = 0;
  *(_QWORD *)(a1 + 1936) = 0;
  v41 = *(_QWORD *)(a1 + 1928);
  if ( v40 != a1 + 1968 )
  {
    v39 = 8 * v41;
    j_j___libc_free_0(v40, 8 * v41);
  }
  v42 = *(_QWORD *)(a1 + 1848);
  if ( v42 )
  {
    v39 = *(_QWORD *)(a1 + 1864) - v42;
    j_j___libc_free_0(v42, v39);
  }
  v43 = *(_QWORD *)(a1 + 1832);
  if ( a1 + 1848 != v43 )
    _libc_free(v43, v39);
  v44 = 8LL * *(unsigned int *)(a1 + 1824);
  sub_C7D6A0(*(_QWORD *)(a1 + 1808), v44, 8);
  sub_E63CD0(*(_QWORD **)(a1 + 1744), v44);
  v45 = *(_QWORD *)(a1 + 1696);
  if ( v45 != a1 + 1712 )
  {
    v44 = *(_QWORD *)(a1 + 1712) + 1LL;
    j_j___libc_free_0(v45, v44);
  }
  v46 = *(_QWORD **)(a1 + 1680);
  v47 = &v46[8 * (unsigned __int64)*(unsigned int *)(a1 + 1688)];
  if ( v46 != v47 )
  {
    do
    {
      v47 -= 8;
      v48 = (_QWORD *)v47[4];
      if ( v48 != v47 + 6 )
      {
        v44 = v47[6] + 1LL;
        j_j___libc_free_0(v48, v44);
      }
      if ( (_QWORD *)*v47 != v47 + 2 )
      {
        v44 = v47[2] + 1LL;
        j_j___libc_free_0(*v47, v44);
      }
    }
    while ( v46 != v47 );
    v47 = *(_QWORD **)(a1 + 1680);
  }
  if ( v47 != (_QWORD *)(a1 + 1696) )
    _libc_free(v47, v44);
  v49 = *(_QWORD *)(a1 + 1528);
  if ( v49 != a1 + 1552 )
    _libc_free(v49, v44);
  v50 = *(_QWORD *)(a1 + 1512);
  if ( v50 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v50 + 8LL))(v50);
  v51 = *(_QWORD *)(a1 + 1480);
  if ( v51 != a1 + 1496 )
    j_j___libc_free_0(v51, *(_QWORD *)(a1 + 1496) + 1LL);
  v52 = 16LL * *(unsigned int *)(a1 + 1464);
  sub_C7D6A0(*(_QWORD *)(a1 + 1448), v52, 8);
  _libc_free(*(_QWORD *)(a1 + 1408), v52);
  v53 = 16LL * *(unsigned int *)(a1 + 1400);
  sub_C7D6A0(*(_QWORD *)(a1 + 1384), v53, 8);
  _libc_free(*(_QWORD *)(a1 + 1344), v53);
  v54 = *(__int64 **)(a1 + 1264);
  v55 = *(unsigned int *)(a1 + 1272);
  v252 = &v54[v55];
  if ( v54 != v252 )
  {
    v56 = *(_QWORD *)(a1 + 1264);
    while ( 1 )
    {
      v57 = (unsigned int)(((__int64)v54 - v56) >> 3) >> 7;
      v58 = 4096LL << v57;
      v59 = v57 < 0x1E;
      v60 = *v54;
      if ( !v59 )
        v58 = 0x40000000000LL;
      v61 = (v60 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      v62 = v60 + v58;
      if ( v60 == *(_QWORD *)(v56 + 8 * v55 - 8) )
        v62 = *(_QWORD *)(a1 + 1248);
      for ( i = v61 + 64; v62 >= i; i += 64LL )
      {
        v64 = *(_QWORD *)(i - 40);
        v65 = (_QWORD *)(i - 64);
        if ( v64 != i - 24 )
          _libc_free(v64, v55);
        if ( *v65 != i - 48 )
          _libc_free(*v65, v55);
      }
      if ( v252 == ++v54 )
        break;
      v56 = *(_QWORD *)(a1 + 1264);
      v55 = *(unsigned int *)(a1 + 1272);
    }
  }
  v66 = *(_QWORD **)(a1 + 1312);
  v67 = 2LL * *(unsigned int *)(a1 + 1320);
  for ( j = &v66[v67]; j != v66; v66 += 2 )
  {
    v68 = *v66 + v66[1];
    v69 = (*v66 + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
    while ( 1 )
    {
      v69 += 64LL;
      if ( v68 < v69 )
        break;
      while ( 1 )
      {
        v70 = *(_QWORD *)(v69 - 40);
        v71 = (_QWORD *)(v69 - 64);
        if ( v70 != v69 - 24 )
          _libc_free(v70, v55);
        if ( *v71 == v69 - 48 )
          break;
        _libc_free(*v71, v55);
        v69 += 64LL;
        if ( v68 < v69 )
          goto LABEL_112;
      }
    }
LABEL_112:
    ;
  }
  sub_E66D20(a1 + 1248);
  sub_B72320(a1 + 1248, v55);
  v72 = *(_QWORD **)(a1 + 1168);
  v73 = *(unsigned int *)(a1 + 1176);
  v74 = &v72[v73];
  if ( v72 != v74 )
  {
    v75 = *(_QWORD *)(a1 + 1168);
    while ( 1 )
    {
      v76 = (unsigned int)(((__int64)v72 - v75) >> 3) >> 7;
      v77 = 4096LL << v76;
      if ( v76 >= 0x1E )
        v77 = 0x40000000000LL;
      v78 = *v72 + v77;
      v79 = (*v72 + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      if ( *v72 == *(_QWORD *)(v75 + 8 * v73 - 8) )
        v78 = *(_QWORD *)(a1 + 1152);
      while ( 1 )
      {
        v79 += 128LL;
        if ( v78 < v79 )
          break;
        while ( 1 )
        {
          v80 = *(_QWORD *)(v79 - 112);
          if ( v80 == v79 - 96 )
            break;
          _libc_free(v80, v73);
          v79 += 128LL;
          if ( v78 < v79 )
            goto LABEL_122;
        }
      }
LABEL_122:
      if ( v74 == ++v72 )
        break;
      v75 = *(_QWORD *)(a1 + 1168);
      v73 = *(unsigned int *)(a1 + 1176);
    }
  }
  v81 = *(_QWORD **)(a1 + 1216);
  for ( k = &v81[2 * *(unsigned int *)(a1 + 1224)]; k != v81; v81 += 2 )
  {
    v83 = *v81 + v81[1];
    v84 = (*v81 + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
    while ( 1 )
    {
      v84 += 128LL;
      if ( v83 < v84 )
        break;
      while ( 1 )
      {
        v85 = *(_QWORD *)(v84 - 112);
        if ( v85 == v84 - 96 )
          break;
        _libc_free(v85, v73);
        v84 += 128LL;
        if ( v83 < v84 )
          goto LABEL_129;
      }
    }
LABEL_129:
    ;
  }
  sub_E66D20(a1 + 1152);
  sub_B72320(a1 + 1152, v73);
  v86 = *(__int64 **)(a1 + 1072);
  v87 = *(unsigned int *)(a1 + 1080);
  v88 = &v86[v87];
  if ( v86 != v88 )
  {
    v89 = *(_QWORD *)(a1 + 1072);
    while ( 1 )
    {
      v90 = (unsigned int)(((__int64)v86 - v89) >> 3) >> 7;
      v91 = 4096LL << v90;
      v59 = v90 < 0x1E;
      v92 = *v86;
      if ( !v59 )
        v91 = 0x40000000000LL;
      v93 = (v92 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      v94 = v92 + v91;
      if ( v92 == *(_QWORD *)(v89 + 8 * v87 - 8) )
        v94 = *(_QWORD *)(a1 + 1056);
      for ( m = v93 + 192; v94 >= m; m += 192LL )
      {
        v96 = m - 192;
        sub_E96DA0(v96);
      }
      if ( v88 == ++v86 )
        break;
      v89 = *(_QWORD *)(a1 + 1072);
      v87 = *(unsigned int *)(a1 + 1080);
    }
  }
  v97 = *(_QWORD **)(a1 + 1120);
  for ( n = &v97[2 * *(unsigned int *)(a1 + 1128)]; n != v97; v97 += 2 )
  {
    v99 = *v97 + v97[1];
    for ( ii = ((*v97 + 7LL) & 0xFFFFFFFFFFFFFFF8LL) + 192; v99 >= ii; ii += 192LL )
    {
      v101 = ii - 192;
      sub_E96DA0(v101);
    }
  }
  sub_E66D20(a1 + 1056);
  sub_B72320(a1 + 1056, v87);
  v102 = *(__int64 **)(a1 + 976);
  v103 = *(unsigned int *)(a1 + 984);
  v104 = (__int64)&v102[v103];
  v254 = v102;
  v236 = (__int64 *)v104;
  if ( v102 != (__int64 *)v104 )
  {
    while ( 1 )
    {
      v104 = *v254;
      v105 = (unsigned int)(v254 - v102) >> 7;
      v106 = 4096LL << v105;
      if ( v105 >= 0x1E )
        v106 = 0x40000000000LL;
      v107 = (_QWORD *)((v104 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      v108 = v104 + v106;
      if ( v104 == v102[v103 - 1] )
        v108 = *(_QWORD *)(a1 + 960);
      v109 = v107 + 23;
      if ( v108 >= (unsigned __int64)(v107 + 23) )
      {
        while ( 1 )
        {
          *v107 = &unk_49E3650;
          v110 = v107;
          sub_E92880(v107);
          v107 = v109;
          if ( v108 < (unsigned __int64)(v110 + 46) )
            break;
          v109 += 23;
        }
      }
      if ( v236 == ++v254 )
        break;
      v102 = *(__int64 **)(a1 + 976);
      v103 = *(unsigned int *)(a1 + 984);
    }
  }
  v111 = 2LL * *(unsigned int *)(a1 + 1032);
  v255 = *(_QWORD **)(a1 + 1024);
  for ( jj = &v255[v111]; jj != v255; v255 += 2 )
  {
    v112 = *v255 + v255[1];
    v113 = (_QWORD *)((*v255 + 7LL) & 0xFFFFFFFFFFFFFFF8LL);
    v114 = v113 + 23;
    if ( v112 >= (unsigned __int64)(v113 + 23) )
    {
      while ( 1 )
      {
        *v113 = &unk_49E3650;
        v115 = v113;
        sub_E92880(v113);
        v113 = v114;
        if ( v112 < (unsigned __int64)(v115 + 46) )
          break;
        v114 += 23;
      }
    }
  }
  sub_E66D20(a1 + 960);
  sub_B72320(a1 + 960, v104);
  v116 = *(__int64 **)(a1 + 880);
  v117 = *(unsigned int *)(a1 + 888);
  v118 = (__int64)&v116[v117];
  v256 = v116;
  v238 = (__int64 *)v118;
  if ( v116 != (__int64 *)v118 )
  {
    while ( 1 )
    {
      v118 = *v256;
      v119 = (unsigned int)(v256 - v116) >> 7;
      v120 = 4096LL << v119;
      if ( v119 >= 0x1E )
        v120 = 0x40000000000LL;
      v121 = (_QWORD *)((v118 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      v122 = v118 + v120;
      if ( v118 == v116[v117 - 1] )
        v122 = *(_QWORD *)(a1 + 864);
      v123 = v121 + 19;
      if ( v122 >= (unsigned __int64)(v121 + 19) )
      {
        while ( 1 )
        {
          *v121 = &unk_49E1A38;
          v124 = v121;
          sub_E92880(v121);
          v121 = v123;
          if ( v122 < (unsigned __int64)(v124 + 38) )
            break;
          v123 += 19;
        }
      }
      if ( v238 == ++v256 )
        break;
      v116 = *(__int64 **)(a1 + 880);
      v117 = *(unsigned int *)(a1 + 888);
    }
  }
  v125 = 2LL * *(unsigned int *)(a1 + 936);
  v257 = *(_QWORD **)(a1 + 928);
  for ( kk = &v257[v125]; kk != v257; v257 += 2 )
  {
    v126 = *v257 + v257[1];
    v127 = (_QWORD *)((*v257 + 7LL) & 0xFFFFFFFFFFFFFFF8LL);
    v128 = v127 + 19;
    if ( v126 >= (unsigned __int64)(v127 + 19) )
    {
      while ( 1 )
      {
        *v127 = &unk_49E1A38;
        v129 = v127;
        sub_E92880(v127);
        v127 = v128;
        if ( v126 < (unsigned __int64)(v129 + 38) )
          break;
        v128 += 19;
      }
    }
  }
  sub_E66D20(a1 + 864);
  v130 = *(__int64 **)(a1 + 880);
  v131 = &v130[*(unsigned int *)(a1 + 888)];
  if ( v130 != v131 )
  {
    for ( mm = *(_QWORD *)(a1 + 880); ; mm = *(_QWORD *)(a1 + 880) )
    {
      v133 = *v130;
      v134 = (unsigned int)(((__int64)v130 - mm) >> 3) >> 7;
      v118 = 4096LL << v134;
      if ( v134 >= 0x1E )
        v118 = 0x40000000000LL;
      ++v130;
      sub_C7D6A0(v133, v118, 16);
      if ( v131 == v130 )
        break;
    }
  }
  v135 = *(__int64 **)(a1 + 928);
  v136 = &v135[2 * *(unsigned int *)(a1 + 936)];
  if ( v135 != v136 )
  {
    do
    {
      v118 = v135[1];
      v137 = *v135;
      v135 += 2;
      sub_C7D6A0(v137, v118, 16);
    }
    while ( v136 != v135 );
    v136 = *(__int64 **)(a1 + 928);
  }
  if ( v136 != (__int64 *)(a1 + 944) )
    _libc_free(v136, v118);
  v138 = *(_QWORD *)(a1 + 880);
  if ( v138 != a1 + 896 )
    _libc_free(v138, v118);
  v139 = *(__int64 **)(a1 + 784);
  v140 = *(unsigned int *)(a1 + 792);
  v141 = (__int64)&v139[v140];
  v258 = v139;
  v240 = (__int64 *)v141;
  if ( v139 != (__int64 *)v141 )
  {
    while ( 1 )
    {
      v141 = *v258;
      v142 = (unsigned int)(v258 - v139) >> 7;
      v143 = 4096LL << v142;
      if ( v142 >= 0x1E )
        v143 = 0x40000000000LL;
      v144 = (_QWORD *)((v141 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      v145 = v141 + v143;
      if ( v141 == v139[v140 - 1] )
        v145 = *(_QWORD *)(a1 + 768);
      v146 = v144 + 21;
      if ( v145 >= (unsigned __int64)(v144 + 21) )
      {
        while ( 1 )
        {
          *v144 = &unk_49E1A10;
          v147 = v144;
          sub_E92880(v144);
          v144 = v146;
          if ( v145 < (unsigned __int64)(v147 + 42) )
            break;
          v146 += 21;
        }
      }
      if ( v240 == ++v258 )
        break;
      v139 = *(__int64 **)(a1 + 784);
      v140 = *(unsigned int *)(a1 + 792);
    }
  }
  v148 = 2LL * *(unsigned int *)(a1 + 840);
  v259 = *(_QWORD **)(a1 + 832);
  for ( nn = &v259[v148]; nn != v259; v259 += 2 )
  {
    v149 = *v259 + v259[1];
    v150 = (_QWORD *)((*v259 + 7LL) & 0xFFFFFFFFFFFFFFF8LL);
    v151 = v150 + 21;
    if ( v149 >= (unsigned __int64)(v150 + 21) )
    {
      while ( 1 )
      {
        *v150 = &unk_49E1A10;
        v152 = v150;
        sub_E92880(v150);
        v150 = v151;
        if ( v149 < (unsigned __int64)(v152 + 42) )
          break;
        v151 += 21;
      }
    }
  }
  sub_E66D20(a1 + 768);
  sub_B72320(a1 + 768, v141);
  v153 = *(_QWORD **)(a1 + 688);
  v154 = *(unsigned int *)(a1 + 696);
  v260 = v153;
  v242 = &v153[v154];
  if ( v153 != v242 )
  {
    v155 = *(_QWORD *)(a1 + 688);
    while ( 1 )
    {
      v156 = (unsigned int)(((__int64)v260 - v155) >> 3) >> 7;
      v157 = 4096LL << v156;
      if ( v156 >= 0x1E )
        v157 = 0x40000000000LL;
      v158 = *v260 + v157;
      v159 = (*v260 + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      if ( *v260 == *(_QWORD *)(v155 + 8 * v154 - 8) )
        v158 = *(_QWORD *)(a1 + 672);
      for ( i1 = v159 + 192; i1 <= v158; i1 += 192LL )
      {
        *(_QWORD *)(i1 - 192) = &unk_49E3628;
        v161 = *(_QWORD *)(i1 - 192 + 176);
        if ( v161 != i1 )
          _libc_free(v161, v154);
        v162 = i1 - 192;
        sub_E92880(v162);
      }
      if ( v242 == ++v260 )
        break;
      v155 = *(_QWORD *)(a1 + 688);
      v154 = *(unsigned int *)(a1 + 696);
    }
  }
  v163 = 2LL * *(unsigned int *)(a1 + 744);
  v261 = *(_QWORD **)(a1 + 736);
  for ( i2 = &v261[v163]; i2 != v261; v261 += 2 )
  {
    v164 = *v261 + v261[1];
    for ( i3 = ((*v261 + 7LL) & 0xFFFFFFFFFFFFFFF8LL) + 192; v164 >= i3; i3 += 192LL )
    {
      *(_QWORD *)(i3 - 192) = &unk_49E3628;
      v166 = *(_QWORD *)(i3 - 192 + 176);
      if ( v166 != i3 )
        _libc_free(v166, v154);
      v167 = i3 - 192;
      sub_E92880(v167);
    }
  }
  sub_E66D20(a1 + 672);
  sub_B72320(a1 + 672, v154);
  v168 = *(__int64 **)(a1 + 592);
  v169 = *(unsigned int *)(a1 + 600);
  v170 = (__int64)&v168[v169];
  v262 = v168;
  v244 = (__int64 *)v170;
  if ( v168 != (__int64 *)v170 )
  {
    while ( 1 )
    {
      v170 = *v262;
      v171 = (unsigned int)(v262 - v168) >> 7;
      v172 = 4096LL << v171;
      if ( v171 >= 0x1E )
        v172 = 0x40000000000LL;
      v173 = (_QWORD *)((v170 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      v174 = v170 + v172;
      if ( v170 == v168[v169 - 1] )
        v174 = *(_QWORD *)(a1 + 576);
      v175 = v173 + 25;
      if ( v174 >= (unsigned __int64)(v173 + 25) )
      {
        while ( 1 )
        {
          *v173 = &unk_49E3600;
          v176 = v173;
          sub_E92880(v173);
          v173 = v175;
          if ( v174 < (unsigned __int64)(v176 + 50) )
            break;
          v175 += 25;
        }
      }
      if ( v244 == ++v262 )
        break;
      v168 = *(__int64 **)(a1 + 592);
      v169 = *(unsigned int *)(a1 + 600);
    }
  }
  v177 = 2LL * *(unsigned int *)(a1 + 648);
  v263 = *(_QWORD **)(a1 + 640);
  for ( i4 = &v263[v177]; i4 != v263; v263 += 2 )
  {
    v178 = *v263 + v263[1];
    v179 = (_QWORD *)((*v263 + 7LL) & 0xFFFFFFFFFFFFFFF8LL);
    v180 = v179 + 25;
    if ( v178 >= (unsigned __int64)(v179 + 25) )
    {
      while ( 1 )
      {
        *v179 = &unk_49E3600;
        v181 = v179;
        sub_E92880(v179);
        v179 = v180;
        if ( v178 < (unsigned __int64)(v181 + 50) )
          break;
        v180 += 25;
      }
    }
  }
  sub_E66D20(a1 + 576);
  sub_B72320(a1 + 576, v170);
  v182 = *(__int64 **)(a1 + 496);
  v183 = *(unsigned int *)(a1 + 504);
  v184 = (__int64)&v182[v183];
  v264 = v182;
  v246 = (__int64 *)v184;
  if ( v182 != (__int64 *)v184 )
  {
    while ( 1 )
    {
      v184 = *v264;
      v185 = (unsigned int)(v264 - v182) >> 7;
      v186 = 4096LL << v185;
      if ( v185 >= 0x1E )
        v186 = 0x40000000000LL;
      v187 = (_QWORD *)((v184 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      v188 = v184 + v186;
      if ( v184 == v182[v183 - 1] )
        v188 = *(_QWORD *)(a1 + 480);
      v189 = v187 + 19;
      if ( v188 >= (unsigned __int64)(v187 + 19) )
      {
        while ( 1 )
        {
          *v187 = &unk_49E35D8;
          v190 = v187;
          sub_E92880(v187);
          v187 = v189;
          if ( v188 < (unsigned __int64)(v190 + 38) )
            break;
          v189 += 19;
        }
      }
      if ( v246 == ++v264 )
        break;
      v182 = *(__int64 **)(a1 + 496);
      v183 = *(unsigned int *)(a1 + 504);
    }
  }
  v191 = 2LL * *(unsigned int *)(a1 + 552);
  v265 = *(_QWORD **)(a1 + 544);
  for ( i5 = &v265[v191]; i5 != v265; v265 += 2 )
  {
    v192 = *v265 + v265[1];
    v193 = (_QWORD *)((*v265 + 7LL) & 0xFFFFFFFFFFFFFFF8LL);
    v194 = v193 + 19;
    if ( v192 >= (unsigned __int64)(v193 + 19) )
    {
      while ( 1 )
      {
        *v193 = &unk_49E35D8;
        v195 = v193;
        sub_E92880(v193);
        v193 = v194;
        if ( v192 < (unsigned __int64)(v195 + 38) )
          break;
        v194 += 19;
      }
    }
  }
  sub_E66D20(a1 + 480);
  sub_B72320(a1 + 480, v184);
  v196 = *(__int64 **)(a1 + 400);
  v197 = *(unsigned int *)(a1 + 408);
  v198 = (__int64)&v196[v197];
  v266 = v196;
  v248 = (__int64 *)v198;
  if ( v196 != (__int64 *)v198 )
  {
    while ( 1 )
    {
      v198 = *v266;
      v199 = (unsigned int)(v266 - v196) >> 7;
      v200 = 4096LL << v199;
      if ( v199 >= 0x1E )
        v200 = 0x40000000000LL;
      v201 = (_QWORD *)((v198 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      v202 = v198 + v200;
      if ( v198 == v196[v197 - 1] )
        v202 = *(_QWORD *)(a1 + 384);
      v203 = v201 + 22;
      if ( v202 >= (unsigned __int64)(v201 + 22) )
      {
        while ( 1 )
        {
          *v201 = &unk_49E35B0;
          v204 = v201;
          sub_E92880(v201);
          v201 = v203;
          if ( v202 < (unsigned __int64)(v204 + 44) )
            break;
          v203 += 22;
        }
      }
      if ( v248 == ++v266 )
        break;
      v196 = *(__int64 **)(a1 + 400);
      v197 = *(unsigned int *)(a1 + 408);
    }
  }
  v205 = 2LL * *(unsigned int *)(a1 + 456);
  v267 = *(_QWORD **)(a1 + 448);
  for ( i6 = &v267[v205]; i6 != v267; v267 += 2 )
  {
    v206 = *v267 + v267[1];
    v207 = (_QWORD *)((*v267 + 7LL) & 0xFFFFFFFFFFFFFFF8LL);
    v208 = v207 + 22;
    if ( v206 >= (unsigned __int64)(v207 + 22) )
    {
      while ( 1 )
      {
        *v207 = &unk_49E35B0;
        v209 = v207;
        sub_E92880(v207);
        v207 = v208;
        if ( v206 < (unsigned __int64)(v209 + 44) )
          break;
        v208 += 22;
      }
    }
  }
  sub_E66D20(a1 + 384);
  sub_B72320(a1 + 384, v198);
  sub_B72320(a1 + 288, v198);
  sub_B72320(a1 + 192, v198);
  v210 = *(_QWORD *)(a1 + 184);
  if ( v210 )
  {
    v211 = *(_QWORD *)(v210 + 288);
    v212 = *(_QWORD *)(v210 + 280);
    if ( v211 != v212 )
    {
      do
      {
        v213 = *(unsigned int *)(v212 + 48);
        v214 = *(_QWORD *)(v212 + 32);
        v212 += 56;
        v198 = 16 * v213;
        sub_C7D6A0(v214, v198, 4);
      }
      while ( v211 != v212 );
      v212 = *(_QWORD *)(v210 + 280);
    }
    if ( v212 )
    {
      v198 = *(_QWORD *)(v210 + 296) - v212;
      j_j___libc_free_0(v212, v198);
    }
    v215 = *(_QWORD *)(v210 + 256);
    if ( v215 )
    {
      v198 = *(_QWORD *)(v210 + 272) - v215;
      j_j___libc_free_0(v215, v198);
    }
    sub_E62F00(*(_QWORD *)(v210 + 224));
    v216 = *(_QWORD *)(v210 + 64);
    if ( v216 != v210 + 80 )
      _libc_free(v216, v198);
    v217 = *(_QWORD *)(v210 + 40);
    if ( v210 + 64 != v217 )
      _libc_free(v217, v198);
    v218 = *(_QWORD *)(v210 + 8);
    if ( *(_DWORD *)(v210 + 20) )
    {
      v229 = *(unsigned int *)(v210 + 16);
      if ( (_DWORD)v229 )
      {
        v230 = 8 * v229;
        v231 = 0;
        do
        {
          v232 = *(_QWORD **)(v218 + v231);
          if ( v232 != (_QWORD *)-8LL && v232 )
          {
            v198 = *v232 + 17LL;
            sub_C7D6A0((__int64)v232, v198, 8);
            v218 = *(_QWORD *)(v210 + 8);
          }
          v231 += 8;
        }
        while ( v230 != v231 );
      }
    }
    _libc_free(v218, v198);
    j_j___libc_free_0(v210, 312);
  }
  v219 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 136);
  if ( v219 )
    v219(a1 + 120, a1 + 120, 3);
  v220 = *(_QWORD *)(a1 + 96);
  if ( v220 )
    j_j___libc_free_0(v220, *(_QWORD *)(a1 + 112) - v220);
  v221 = *(__int64 ***)(a1 + 88);
  if ( v221 )
  {
    v222 = v221[4];
    v223 = (char *)v221[3];
    if ( v222 != (__int64 *)v223 )
    {
      do
      {
        if ( *(char **)v223 != v223 + 16 )
          j_j___libc_free_0(*(_QWORD *)v223, *((_QWORD *)v223 + 2) + 1LL);
        v223 += 32;
      }
      while ( v222 != (__int64 *)v223 );
      v223 = (char *)v221[3];
    }
    if ( v223 )
      j_j___libc_free_0(v223, (char *)v221[5] - v223);
    v224 = v221[1];
    v225 = *v221;
    if ( v224 != *v221 )
    {
      do
      {
        v226 = v225;
        v225 += 3;
        sub_C8EE20(v226);
      }
      while ( v224 != v225 );
      v225 = *v221;
    }
    if ( v225 )
      j_j___libc_free_0(v225, (char *)v221[2] - (char *)v225);
    j_j___libc_free_0(v221, 64);
  }
  v227 = *(_QWORD *)(a1 + 24);
  result = a1 + 40;
  if ( v227 != a1 + 40 )
    return j_j___libc_free_0(v227, *(_QWORD *)(a1 + 40) + 1LL);
  return result;
}
