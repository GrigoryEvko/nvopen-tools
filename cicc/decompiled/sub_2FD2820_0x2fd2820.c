// Function: sub_2FD2820
// Address: 0x2fd2820
//
__int64 __fastcall sub_2FD2820(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 k, unsigned __int64 a6)
{
  _QWORD *v6; // r15
  int v7; // ebx
  unsigned __int64 v8; // rax
  __int64 v9; // r13
  unsigned __int64 v10; // rdx
  __int64 v11; // rax
  __int64 i; // rdx
  void **p_src; // r13
  __int64 v14; // r12
  __int64 v15; // rbx
  __int64 v16; // r14
  int v17; // edi
  __int64 v18; // r10
  __int64 ***v19; // rax
  __int64 **v20; // rsi
  __int64 v21; // rax
  int *v22; // rdx
  int v23; // eax
  __int64 **v24; // rbx
  __int64 v25; // r8
  __int64 *v26; // r14
  __int64 v27; // rax
  unsigned __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 m; // rdx
  unsigned __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  unsigned __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 n; // rdx
  unsigned __int64 v40; // rax
  _BYTE *v41; // rax
  _BYTE *ii; // rdx
  unsigned __int64 v43; // rax
  __int64 v44; // rdx
  _DWORD *v45; // rax
  _DWORD *jj; // rdx
  __int64 v47; // rbx
  int v48; // ecx
  int v49; // edx
  unsigned __int64 v50; // rdi
  unsigned __int64 v51; // rax
  int v52; // edx
  __int64 v53; // rbx
  int v54; // ecx
  int v55; // edx
  unsigned __int64 v56; // rax
  int v57; // edx
  unsigned __int64 v58; // rax
  __int64 v59; // rsi
  __int64 v60; // r14
  char *kk; // rdx
  __int64 v62; // rax
  unsigned __int64 v63; // rdx
  _QWORD *v64; // rbx
  __int64 v65; // rax
  __int64 v66; // rdx
  int **v67; // r14
  __int64 v68; // rbx
  int **v69; // r13
  unsigned __int64 v70; // rax
  _QWORD *v71; // r12
  char *v72; // r15
  __int64 v73; // rdx
  __int64 v74; // r9
  int v75; // r14d
  __int64 v76; // rcx
  _BYTE *v77; // rsi
  __int64 v78; // r8
  __int64 v79; // rbx
  unsigned __int64 v80; // r13
  unsigned __int64 v81; // rax
  __int64 v82; // rcx
  unsigned __int64 v83; // rdx
  __int64 v84; // rdx
  __int64 v85; // rax
  __int64 mm; // rdx
  unsigned __int64 v87; // rax
  __int64 v88; // rcx
  unsigned __int64 v89; // rdx
  __int64 v90; // rdx
  __int64 v91; // rax
  __int64 nn; // rdx
  __int64 v93; // rbx
  __int64 v94; // r13
  int v95; // ecx
  int v96; // edx
  unsigned __int64 v97; // rax
  int v98; // edx
  __int64 v99; // rbx
  __int64 v100; // r13
  int v101; // ecx
  int v102; // edx
  unsigned __int64 v103; // rax
  int v104; // edx
  float v105; // xmm0_4
  __int64 v106; // r12
  __int64 v107; // r12
  int **v108; // rbx
  int *v109; // rcx
  int *v110; // rdx
  int **v111; // rax
  int **v112; // rsi
  _QWORD *v113; // rbx
  const __m128i *v114; // rax
  __m128i *v115; // rdx
  __int64 v116; // rcx
  __int64 v117; // r14
  unsigned __int64 v118; // r13
  unsigned __int64 v119; // r8
  _QWORD *v120; // r15
  __int64 v121; // rbx
  _QWORD *v122; // rax
  int v123; // eax
  __int64 i1; // rdx
  __int64 v125; // rcx
  __int64 v126; // r8
  __int64 v127; // r9
  unsigned __int64 v128; // r12
  unsigned __int64 v129; // rax
  unsigned int v130; // ebx
  __int64 v131; // rdx
  _DWORD *v132; // rax
  __int64 v133; // r10
  __int64 *v134; // rdx
  __int64 v135; // r12
  __int64 v136; // rax
  unsigned __int64 v139; // r8
  unsigned __int64 v140; // r9
  unsigned int v141; // r12d
  size_t v142; // rdx
  __int64 v143; // rax
  __int64 v144; // rax
  __int64 v145; // rdx
  unsigned __int64 *v146; // r13
  unsigned __int64 *v147; // rbx
  __int64 v148; // rax
  unsigned __int64 *v149; // r13
  unsigned __int64 *v150; // rbx
  __int64 v151; // rax
  unsigned __int64 *v152; // r13
  unsigned __int64 *v153; // rbx
  __int64 v154; // r13
  __int64 v155; // rcx
  __int64 v156; // r13
  unsigned int v157; // r15d
  __int64 v158; // r12
  __int64 v159; // r14
  __int64 v160; // rbx
  _QWORD *v161; // r14
  _QWORD *v162; // rax
  unsigned __int64 v164; // rdx
  __int64 v165; // rbx
  __int64 v166; // rdx
  unsigned __int64 *v167; // rax
  unsigned __int64 *v168; // rdx
  unsigned __int8 v169; // r13
  unsigned __int64 *v170; // rbx
  unsigned __int64 *v171; // r12
  __int64 v172; // rdx
  unsigned __int64 *v173; // r13
  unsigned __int64 *v174; // rdx
  unsigned __int8 v175; // r12
  unsigned __int64 *v176; // rbx
  int **v177; // rsi
  __int64 v178; // r12
  unsigned __int64 *v179; // rbx
  unsigned __int64 *v180; // r12
  __int64 v181; // rbx
  __int64 v182; // r13
  void **v183; // rcx
  __int64 v184; // r14
  _QWORD *v185; // r12
  __int64 v186; // rdi
  _QWORD *v187; // r14
  _QWORD *v188; // rax
  __int64 v189; // r12
  unsigned __int64 *v190; // rbx
  unsigned __int64 *v191; // r12
  unsigned __int64 *v192; // r13
  unsigned __int64 *v193; // r12
  _QWORD *v194; // [rsp+0h] [rbp-130h]
  __int64 v196; // [rsp+10h] [rbp-120h]
  _QWORD *v197; // [rsp+10h] [rbp-120h]
  __int64 v198; // [rsp+18h] [rbp-118h]
  __int64 v199; // [rsp+20h] [rbp-110h]
  unsigned __int64 v200; // [rsp+28h] [rbp-108h]
  char **v201; // [rsp+30h] [rbp-100h]
  unsigned __int64 v202; // [rsp+30h] [rbp-100h]
  unsigned int v203; // [rsp+38h] [rbp-F8h]
  __int64 v204; // [rsp+40h] [rbp-F0h]
  int v205; // [rsp+40h] [rbp-F0h]
  unsigned __int64 v206; // [rsp+40h] [rbp-F0h]
  __int64 v207; // [rsp+40h] [rbp-F0h]
  int v208; // [rsp+40h] [rbp-F0h]
  __int64 j; // [rsp+48h] [rbp-E8h]
  char *v210; // [rsp+48h] [rbp-E8h]
  void **v211; // [rsp+48h] [rbp-E8h]
  int v212; // [rsp+50h] [rbp-E0h]
  unsigned __int64 v213; // [rsp+50h] [rbp-E0h]
  __int64 v214; // [rsp+50h] [rbp-E0h]
  unsigned __int64 v215; // [rsp+58h] [rbp-D8h]
  _QWORD *v216; // [rsp+58h] [rbp-D8h]
  __int64 v217; // [rsp+68h] [rbp-C8h] BYREF
  void *src; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v219; // [rsp+78h] [rbp-B8h]
  _BYTE v220[176]; // [rsp+80h] [rbp-B0h] BYREF

  v6 = (_QWORD *)a1;
  v7 = -858993459 * ((__int64)(*(_QWORD *)(*(_QWORD *)a1 + 16LL) - *(_QWORD *)(*(_QWORD *)a1 + 8LL)) >> 3)
     - *(_DWORD *)(*(_QWORD *)a1 + 32LL);
  v8 = *(unsigned int *)(a1 + 72);
  if ( v7 != v8 )
  {
    v9 = 80LL * v7;
    if ( v7 < v8 )
    {
      v192 = (unsigned __int64 *)(*(_QWORD *)(a1 + 64) + v9);
      v193 = (unsigned __int64 *)(*(_QWORD *)(a1 + 64) + 80 * v8);
      while ( v192 != v193 )
      {
        v193 -= 10;
        if ( (unsigned __int64 *)*v193 != v193 + 2 )
          _libc_free(*v193);
      }
    }
    else
    {
      v10 = *(unsigned int *)(a1 + 76);
      if ( v7 > v10 )
      {
        sub_2FD09D0(a1 + 64, v7, v10, a4, k, a6);
        v8 = *(unsigned int *)(a1 + 72);
      }
      v11 = *(_QWORD *)(a1 + 64) + 80 * v8;
      for ( i = v9 + *(_QWORD *)(a1 + 64); i != v11; v11 += 80 )
      {
        if ( v11 )
        {
          a4 = v11 + 16;
          *(_DWORD *)(v11 + 8) = 0;
          *(_QWORD *)v11 = v11 + 16;
          *(_DWORD *)(v11 + 12) = 8;
        }
      }
    }
    *(_DWORD *)(a1 + 72) = v7;
  }
  p_src = &src;
  for ( j = *(_QWORD *)(a2 + 328); a2 + 320 != j; j = *(_QWORD *)(j + 8) )
  {
    v14 = *(_QWORD *)(j + 56);
    if ( v14 == j + 48 )
      continue;
    do
    {
      v15 = *(_QWORD *)(v14 + 32);
      v16 = v15 + 40LL * (*(_DWORD *)(v14 + 40) & 0xFFFFFF);
      if ( v15 == v16 )
        goto LABEL_26;
      do
      {
LABEL_13:
        if ( *(_BYTE *)v15 != 5 )
          goto LABEL_25;
        v17 = *(_DWORD *)(v15 + 24);
        if ( v17 < 0 )
          goto LABEL_25;
        v18 = v6[2];
        k = *(_QWORD *)(v18 + 112);
        v19 = *(__int64 ****)(*(_QWORD *)(v18 + 104) + 8 * (v17 % k));
        if ( !v19 )
          goto LABEL_25;
        v20 = *v19;
        if ( !*v19 )
          goto LABEL_25;
        a6 = *((unsigned int *)v20 + 2);
        a4 = 0;
        while ( v17 == (_DWORD)a6 )
        {
          v20 = (__int64 **)*v20;
          ++a4;
          if ( !v20 )
            goto LABEL_23;
LABEL_20:
          a6 = *((int *)v20 + 2);
          if ( v17 % k != a6 % k )
            goto LABEL_23;
        }
        if ( a4 )
          goto LABEL_24;
        v20 = (__int64 **)*v20;
        if ( v20 )
          goto LABEL_20;
LABEL_23:
        if ( !a4 )
          goto LABEL_25;
LABEL_24:
        LODWORD(src) = *(_DWORD *)(v15 + 24);
        a6 = (unsigned __int64)sub_2FD0320((_QWORD *)(v18 + 104), (int *)&src);
        if ( (unsigned __int16)(*(_WORD *)(v14 + 68) - 14) <= 4u )
        {
LABEL_25:
          v15 += 40;
          if ( v16 == v15 )
            break;
          goto LABEL_13;
        }
        v213 = a6;
        v15 += 40;
        v105 = sub_2E13950(0, 1u, v6[3], v14, 0);
        a6 = v213;
        *(float *)(v213 + 132) = v105 + *(float *)(v213 + 132);
      }
      while ( v16 != v15 );
LABEL_26:
      v21 = *(_QWORD *)(v14 + 48);
      v22 = (int *)(v21 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v21 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v23 = v21 & 7;
        if ( v23 )
        {
          if ( v23 == 3 )
          {
            v24 = (__int64 **)(v22 + 4);
            v25 = *v22;
            goto LABEL_29;
          }
        }
        else
        {
          *(_QWORD *)(v14 + 48) = v22;
          v24 = (__int64 **)(v14 + 48);
          v25 = 1;
LABEL_29:
          for ( k = (unsigned __int64)&v24[v25]; (__int64 **)k != v24; ++*(_DWORD *)(v30 + 8) )
          {
            while ( 1 )
            {
              v26 = *v24;
              v27 = **v24;
              if ( v27 )
              {
                if ( (v27 & 4) != 0 )
                {
                  v28 = v27 & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v28 )
                  {
                    if ( *(_DWORD *)(v28 + 8) == 4 )
                    {
                      v29 = *(int *)(v28 + 16);
                      if ( (int)v29 >= 0 )
                        break;
                    }
                  }
                }
              }
              if ( (__int64 **)k == ++v24 )
                goto LABEL_40;
            }
            v30 = v6[8] + 80 * v29;
            v31 = *(unsigned int *)(v30 + 8);
            a6 = v31 + 1;
            if ( v31 + 1 > (unsigned __int64)*(unsigned int *)(v30 + 12) )
            {
              v206 = k;
              v214 = v30;
              sub_C8D5F0(v30, (const void *)(v30 + 16), v31 + 1, 8u, k, a6);
              v30 = v214;
              k = v206;
              v31 = *(unsigned int *)(v214 + 8);
            }
            a4 = *(_QWORD *)v30;
            ++v24;
            *(_QWORD *)(*(_QWORD *)v30 + 8 * v31) = v26;
          }
        }
      }
LABEL_40:
      if ( (*(_BYTE *)v14 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v14 + 44) & 8) != 0 )
          v14 = *(_QWORD *)(v14 + 8);
      }
      v14 = *(_QWORD *)(v14 + 8);
    }
    while ( j + 48 != v14 );
  }
  m = *v6;
  v212 = -858993459 * ((__int64)(*(_QWORD *)(*v6 + 16LL) - *(_QWORD *)(*v6 + 8LL)) >> 3) - *(_DWORD *)(*v6 + 32LL);
  v199 = (__int64)(v6 + 185);
  v33 = *((unsigned int *)v6 + 372);
  if ( v33 != 1 )
  {
    if ( v33 > 1 )
    {
      v178 = v6[185];
      v179 = (unsigned __int64 *)(v178 + 72 * v33);
      v180 = (unsigned __int64 *)(v178 + 72);
      while ( v180 != v179 )
      {
        v179 -= 9;
        if ( (unsigned __int64 *)*v179 != v179 + 2 )
          _libc_free(*v179);
      }
    }
    else
    {
      if ( !*((_DWORD *)v6 + 373) )
      {
        sub_2FD0B90(v199, 1u, m, a4, k, a6);
        v33 = *((unsigned int *)v6 + 372);
      }
      v34 = v6[185];
      v35 = v34 + 72 * v33;
      for ( m = v34 + 72; m != v35; v35 += 72 )
      {
        if ( v35 )
        {
          a4 = v35 + 16;
          *(_QWORD *)(v35 + 64) = 0;
          *(_QWORD *)v35 = v35 + 16;
          *(_DWORD *)(v35 + 8) = 0;
          *(_DWORD *)(v35 + 12) = 6;
          *(_OWORD *)(v35 + 16) = 0;
          *(_OWORD *)(v35 + 32) = 0;
          *(_OWORD *)(v35 + 48) = 0;
        }
      }
    }
    *((_DWORD *)v6 + 372) = 1;
  }
  v198 = (__int64)(v6 + 208);
  v36 = *((unsigned int *)v6 + 418);
  if ( v36 != 1 )
  {
    if ( v36 > 1 )
    {
      v189 = v6[208];
      v190 = (unsigned __int64 *)(v189 + 72 * v36);
      v191 = (unsigned __int64 *)(v189 + 72);
      while ( v191 != v190 )
      {
        v190 -= 9;
        if ( (unsigned __int64 *)*v190 != v190 + 2 )
          _libc_free(*v190);
      }
    }
    else
    {
      if ( !*((_DWORD *)v6 + 419) )
      {
        sub_2FD0B90(v198, 1u, m, a4, k, a6);
        v36 = *((unsigned int *)v6 + 418);
      }
      v37 = v6[208];
      v38 = v37 + 72 * v36;
      for ( n = v37 + 72; n != v38; v38 += 72 )
      {
        if ( v38 )
        {
          *(_QWORD *)(v38 + 64) = 0;
          *(_QWORD *)v38 = v38 + 16;
          *(_DWORD *)(v38 + 8) = 0;
          *(_DWORD *)(v38 + 12) = 6;
          *(_OWORD *)(v38 + 16) = 0;
          *(_OWORD *)(v38 + 32) = 0;
          *(_OWORD *)(v38 + 48) = 0;
        }
      }
    }
    *((_DWORD *)v6 + 418) = 1;
  }
  v40 = v6[171];
  if ( v212 != v40 )
  {
    if ( v212 >= v40 )
    {
      if ( (unsigned __int64)v212 > v6[172] )
      {
        sub_C8D290((__int64)(v6 + 170), v6 + 173, v212, 1u, k, a6);
        v40 = v6[171];
      }
      v41 = (_BYTE *)(v6[170] + v40);
      for ( ii = (_BYTE *)(v212 + v6[170]); ii != v41; ++v41 )
      {
        if ( v41 )
          *v41 = 0;
      }
    }
    v6[171] = v212;
  }
  v43 = *((unsigned int *)v6 + 352);
  if ( v212 != v43 )
  {
    if ( v212 >= v43 )
    {
      if ( v212 > (unsigned __int64)*((unsigned int *)v6 + 353) )
      {
        sub_C8D5F0((__int64)(v6 + 175), v6 + 177, v212, 4u, k, a6);
        v43 = *((unsigned int *)v6 + 352);
      }
      v44 = v6[175];
      v45 = (_DWORD *)(v44 + 4 * v43);
      for ( jj = (_DWORD *)(v44 + 4LL * v212); jj != v45; ++v45 )
      {
        if ( v45 )
          *v45 = 0;
      }
    }
    *((_DWORD *)v6 + 352) = v212;
  }
  v47 = v6[185];
  v48 = *(_DWORD *)(v47 + 64) & 0x3F;
  if ( v48 )
    *(_QWORD *)(*(_QWORD *)v47 + 8LL * *(unsigned int *)(v47 + 8) - 8) &= ~(-1LL << v48);
  LOBYTE(v49) = v212;
  *(_DWORD *)(v47 + 64) = v212;
  v50 = (unsigned int)(v212 + 63) >> 6;
  v51 = *(unsigned int *)(v47 + 8);
  v203 = (unsigned int)(v212 + 63) >> 6;
  v215 = v50;
  if ( v50 != v51 )
  {
    if ( v50 >= v51 )
    {
      v107 = v50 - v51;
      if ( v50 > *(unsigned int *)(v47 + 12) )
      {
        sub_C8D5F0(v47, (const void *)(v47 + 16), v50, 8u, k, a6);
        v51 = *(unsigned int *)(v47 + 8);
      }
      if ( 8 * v107 )
      {
        memset((void *)(*(_QWORD *)v47 + 8 * v51), 0, 8 * v107);
        LODWORD(v51) = *(_DWORD *)(v47 + 8);
      }
      v49 = *(_DWORD *)(v47 + 64);
      *(_DWORD *)(v47 + 8) = v107 + v51;
    }
    else
    {
      *(_DWORD *)(v47 + 8) = v50;
    }
  }
  v52 = v49 & 0x3F;
  if ( v52 )
    *(_QWORD *)(*(_QWORD *)v47 + 8LL * *(unsigned int *)(v47 + 8) - 8) &= ~(-1LL << v52);
  v53 = v6[208];
  v54 = *(_DWORD *)(v53 + 64) & 0x3F;
  if ( v54 )
    *(_QWORD *)(*(_QWORD *)v53 + 8LL * *(unsigned int *)(v53 + 8) - 8) &= ~(-1LL << v54);
  LOBYTE(v55) = v212;
  v56 = *(unsigned int *)(v53 + 8);
  *(_DWORD *)(v53 + 64) = v212;
  if ( v50 != v56 )
  {
    if ( v50 >= v56 )
    {
      v106 = v50 - v56;
      if ( v50 > *(unsigned int *)(v53 + 12) )
      {
        sub_C8D5F0(v53, (const void *)(v53 + 16), v50, 8u, k, a6);
        v56 = *(unsigned int *)(v53 + 8);
      }
      if ( 8 * v106 )
      {
        memset((void *)(*(_QWORD *)v53 + 8 * v56), 0, 8 * v106);
        LODWORD(v56) = *(_DWORD *)(v53 + 8);
      }
      v55 = *(_DWORD *)(v53 + 64);
      *(_DWORD *)(v53 + 8) = v106 + v56;
    }
    else
    {
      *(_DWORD *)(v53 + 8) = v203;
    }
  }
  v57 = v55 & 0x3F;
  if ( v57 )
    *(_QWORD *)(*(_QWORD *)v53 + 8LL * *(unsigned int *)(v53 + 8) - 8) &= ~(-1LL << v57);
  v58 = *((unsigned int *)v6 + 484);
  if ( v212 != v58 )
  {
    v59 = 232LL * v212;
    if ( v212 >= v58 )
    {
      if ( v212 > (unsigned __int64)*((unsigned int *)v6 + 485) )
      {
        v113 = v6 + 243;
        v60 = sub_C8D7D0((__int64)(v6 + 241), (__int64)(v6 + 243), v212, 0xE8u, (unsigned __int64 *)&src, a6);
        v114 = (const __m128i *)v6[241];
        k = (unsigned __int64)v114 + 232 * *((unsigned int *)v6 + 484);
        if ( v114 != (const __m128i *)k )
        {
          v115 = (__m128i *)v60;
          do
          {
            if ( v115 )
            {
              *v115 = _mm_loadu_si128(v114);
              v115[1] = _mm_loadu_si128(v114 + 1);
              v115[2] = _mm_loadu_si128(v114 + 2);
              v115[3] = _mm_loadu_si128(v114 + 3);
              v115[4] = _mm_loadu_si128(v114 + 4);
              v115[5] = _mm_loadu_si128(v114 + 5);
              v115[6] = _mm_loadu_si128(v114 + 6);
              v115[7] = _mm_loadu_si128(v114 + 7);
              v115[8] = _mm_loadu_si128(v114 + 8);
              v115[9] = _mm_loadu_si128(v114 + 9);
              v115[10] = _mm_loadu_si128(v114 + 10);
              v115[11] = _mm_loadu_si128(v114 + 11);
              v115[12] = _mm_loadu_si128(v114 + 12);
              v115[13] = _mm_loadu_si128(v114 + 13);
              v115[14].m128i_i64[0] = v114[14].m128i_i64[0];
            }
            v114 = (const __m128i *)((char *)v114 + 232);
            v115 = (__m128i *)((char *)v115 + 232);
          }
          while ( (const __m128i *)k != v114 );
          a6 = v6[241];
          k = a6 + 232LL * *((unsigned int *)v6 + 484);
          if ( a6 != k )
          {
            v116 = v60;
            v117 = v6[241];
            a6 = (unsigned __int64)&src;
            v118 = k;
            v119 = (unsigned __int64)v6;
            v120 = v6 + 243;
            do
            {
              while ( 1 )
              {
                v121 = *(_QWORD *)(v118 - 224);
                v118 -= 232LL;
                if ( v121 )
                {
                  if ( *(_DWORD *)(v121 + 200) )
                    break;
                }
                if ( v118 == v117 )
                  goto LABEL_188;
              }
              v200 = a6;
              v202 = v119;
              v207 = v116;
              sub_2E19AD0(v121 + 8, (char *)sub_2E199D0, 0, v116, v119, a6);
              v116 = v207;
              v119 = v202;
              v122 = (_QWORD *)(v121 + 8);
              a6 = v200;
              do
              {
                *v122 = 0;
                v122 += 2;
                *(v122 - 1) = 0;
              }
              while ( (_QWORD *)(v121 + 136) != v122 );
            }
            while ( v118 != v117 );
LABEL_188:
            v113 = v120;
            v6 = (_QWORD *)v119;
            k = *(_QWORD *)(v119 + 1928);
            v60 = v116;
            p_src = (void **)a6;
          }
        }
        v123 = (int)src;
        if ( v113 != (_QWORD *)k )
        {
          v208 = (int)src;
          _libc_free(k);
          v123 = v208;
        }
        *((_DWORD *)v6 + 485) = v123;
        v58 = *((unsigned int *)v6 + 484);
        v6[241] = v60;
      }
      else
      {
        v60 = v6[241];
      }
      for ( kk = (char *)(v60 + 232 * v58); (char *)(v60 + v59) != kk; kk += 232 )
      {
        if ( kk )
          memset(kk, 0, 0xE8u);
      }
      goto LABEL_101;
    }
    v181 = v6[241] + 232 * v58;
    if ( v181 == v59 + v6[241] )
    {
LABEL_101:
      *((_DWORD *)v6 + 484) = v212;
      goto LABEL_102;
    }
    v182 = v59 + v6[241];
    v183 = &src;
    do
    {
      while ( 1 )
      {
        v184 = *(_QWORD *)(v181 - 224);
        v181 -= 232;
        if ( v184 )
        {
          if ( *(_DWORD *)(v184 + 200) )
            break;
        }
        if ( v182 == v181 )
          goto LABEL_286;
      }
      v185 = (_QWORD *)(v184 + 8);
      v211 = v183;
      v186 = v184 + 8;
      v187 = (_QWORD *)(v184 + 136);
      sub_2E19AD0(v186, (char *)sub_2E199D0, 0, (__int64)v183, k, a6);
      v183 = v211;
      v188 = v185;
      do
      {
        *v188 = 0;
        v188 += 2;
        *(v188 - 1) = 0;
      }
      while ( v187 != v188 );
    }
    while ( v182 != v181 );
LABEL_286:
    p_src = v183;
    *((_DWORD *)v6 + 484) = v212;
  }
LABEL_102:
  src = v220;
  v219 = 0x1000000000LL;
  v62 = v6[2];
  v63 = *(unsigned int *)(v62 + 128);
  if ( v63 > 0x10 )
  {
    sub_C8D5F0((__int64)p_src, v220, v63, 8u, k, a6);
    v65 = (unsigned int)v219;
    v64 = *(_QWORD **)(v6[2] + 120LL);
    v66 = (unsigned int)v219;
    if ( !v64 )
      goto LABEL_108;
  }
  else
  {
    v64 = *(_QWORD **)(v62 + 120);
    v65 = 0;
    if ( !v64 )
    {
LABEL_255:
      sub_2FD1370((char **)v6 + 5);
      goto LABEL_194;
    }
  }
  do
  {
    if ( v65 + 1 > (unsigned __int64)HIDWORD(v219) )
    {
      sub_C8D5F0((__int64)p_src, v220, v65 + 1, 8u, k, a6);
      v65 = (unsigned int)v219;
    }
    *((_QWORD *)src + v65) = v64 + 1;
    v65 = (unsigned int)(v219 + 1);
    LODWORD(v219) = v219 + 1;
    v64 = (_QWORD *)*v64;
  }
  while ( v64 );
  v66 = (unsigned int)v65;
LABEL_108:
  v67 = (int **)src;
  v68 = 8 * v66;
  v69 = (int **)((char *)src + 8 * v66);
  if ( src == v69 )
    goto LABEL_255;
  _BitScanReverse64(&v70, v68 >> 3);
  sub_2FCED60((__int64)src, (unsigned int **)src + v66, 2LL * (int)(63 - (v70 ^ 0x3F)));
  if ( (unsigned __int64)v68 > 0x80 )
  {
    v108 = v67 + 16;
    sub_2FCEB80(v67, v67 + 16);
    if ( v69 != v67 + 16 )
    {
      do
      {
        while ( 1 )
        {
          v109 = *v108;
          v110 = *(v108 - 1);
          v111 = v108 - 1;
          if ( *v110 > **v108 )
            break;
          v177 = v108++;
          *v177 = v109;
          if ( v69 == v108 )
            goto LABEL_111;
        }
        do
        {
          v111[1] = v110;
          v112 = v111;
          v110 = *--v111;
        }
        while ( *v109 < *v110 );
        ++v108;
        *v112 = v109;
      }
      while ( v69 != v108 );
    }
  }
  else
  {
    sub_2FCEB80(v67, v69);
  }
LABEL_111:
  v210 = (char *)src + 8 * (unsigned int)v219;
  v201 = (char **)(v6 + 5);
  if ( src != v210 )
  {
    v71 = v6;
    v72 = (char *)src;
    do
    {
      v73 = *v71;
      v74 = *(_QWORD *)(*v71 + 8LL);
      v75 = *(_DWORD *)(*(_QWORD *)v72 + 120LL) - 0x40000000;
      if ( *(_QWORD *)(v74 + 40LL * (unsigned int)(*(_DWORD *)(*v71 + 32LL) + v75) + 8) != -1 )
      {
        v76 = *(_QWORD *)v72 + 8LL;
        v77 = (_BYTE *)v71[6];
        v217 = v76;
        if ( v77 == (_BYTE *)v71[7] )
        {
          sub_2FD0840((__int64)v201, v77, &v217);
          v73 = *v71;
        }
        else
        {
          if ( v77 )
          {
            *(_QWORD *)v77 = v76;
            v77 = (_BYTE *)v71[6];
            v73 = *v71;
          }
          v71[6] = v77 + 8;
        }
        v204 = 0;
        *(_BYTE *)(v71[170] + v75) = *(_BYTE *)(*(_QWORD *)(v73 + 8)
                                              + 40LL * (unsigned int)(*(_DWORD *)(v73 + 32) + v75)
                                              + 16);
        *(_DWORD *)(v71[175] + 4LL * v75) = *(_QWORD *)(*(_QWORD *)(*v71 + 8LL)
                                                      + 40LL * (unsigned int)(*(_DWORD *)(*v71 + 32LL) + v75)
                                                      + 8);
        v78 = *(_QWORD *)(*v71 + 8LL);
        v79 = *(unsigned __int8 *)(v78 + 40LL * (unsigned int)(*(_DWORD *)(*v71 + 32LL) + v75) + 20);
        if ( (_BYTE)v79 )
        {
          v205 = (unsigned __int8)v79 + 1;
          v80 = v205;
          v81 = *((unsigned int *)v71 + 372);
          if ( v205 != v81 )
          {
            v82 = 72LL * v205;
            if ( v205 < v81 )
            {
              v166 = v71[185];
              v167 = (unsigned __int64 *)(v166 + 72 * v81);
              v168 = (unsigned __int64 *)(v82 + v166);
              if ( v167 != v168 )
              {
                v169 = *(_BYTE *)(v78 + 40LL * (unsigned int)(*(_DWORD *)(*v71 + 32LL) + v75) + 20);
                v170 = v167;
                v194 = v71;
                v171 = v168;
                do
                {
                  v170 -= 9;
                  if ( (unsigned __int64 *)*v170 != v170 + 2 )
                    _libc_free(*v170);
                }
                while ( v171 != v170 );
                v79 = v169;
                v71 = v194;
                v80 = v205;
              }
            }
            else
            {
              v83 = *((unsigned int *)v71 + 373);
              if ( v205 > v83 )
              {
                sub_2FD0B90(v199, v205, v83, v82, v78, v74);
                v81 = *((unsigned int *)v71 + 372);
                v82 = 72LL * v205;
              }
              v84 = v71[185];
              v85 = v84 + 72 * v81;
              for ( mm = v82 + v84; mm != v85; v85 += 72 )
              {
                if ( v85 )
                {
                  *(_QWORD *)(v85 + 64) = 0;
                  *(_QWORD *)v85 = v85 + 16;
                  *(_DWORD *)(v85 + 8) = 0;
                  *(_DWORD *)(v85 + 12) = 6;
                  *(_OWORD *)(v85 + 16) = 0;
                  *(_OWORD *)(v85 + 32) = 0;
                  *(_OWORD *)(v85 + 48) = 0;
                }
              }
            }
            *((_DWORD *)v71 + 372) = v205;
          }
          v87 = *((unsigned int *)v71 + 418);
          if ( v80 != v87 )
          {
            v88 = 72 * v80;
            if ( v80 < v87 )
            {
              v172 = v71[208];
              v173 = (unsigned __int64 *)(v172 + 72 * v87);
              v174 = (unsigned __int64 *)(v88 + v172);
              if ( v173 != v174 )
              {
                v197 = v71;
                v175 = v79;
                v176 = v174;
                do
                {
                  v173 -= 9;
                  if ( (unsigned __int64 *)*v173 != v173 + 2 )
                    _libc_free(*v173);
                }
                while ( v176 != v173 );
                v79 = v175;
                v71 = v197;
              }
            }
            else
            {
              v89 = *((unsigned int *)v71 + 419);
              if ( v80 > v89 )
              {
                sub_2FD0B90(v198, v80, v89, v88, v78, v74);
                v87 = *((unsigned int *)v71 + 418);
                v88 = 72 * v80;
              }
              v90 = v71[208];
              v91 = v90 + 72 * v87;
              for ( nn = v88 + v90; nn != v91; v91 += 72 )
              {
                if ( v91 )
                {
                  *(_QWORD *)(v91 + 64) = 0;
                  *(_QWORD *)v91 = v91 + 16;
                  *(_DWORD *)(v91 + 8) = 0;
                  *(_DWORD *)(v91 + 12) = 6;
                  *(_OWORD *)(v91 + 16) = 0;
                  *(_OWORD *)(v91 + 32) = 0;
                  *(_OWORD *)(v91 + 48) = 0;
                }
              }
            }
            *((_DWORD *)v71 + 418) = v205;
          }
          v93 = 72 * v79;
          v94 = v93 + v71[185];
          v204 = v93;
          v95 = *(_DWORD *)(v94 + 64) & 0x3F;
          if ( v95 )
            *(_QWORD *)(*(_QWORD *)v94 + 8LL * *(unsigned int *)(v94 + 8) - 8) &= ~(-1LL << v95);
          LOBYTE(v96) = v212;
          v97 = *(unsigned int *)(v94 + 8);
          *(_DWORD *)(v94 + 64) = v212;
          if ( v215 != v97 )
          {
            if ( v215 >= v97 )
            {
              v196 = v215 - v97;
              if ( v215 > *(unsigned int *)(v94 + 12) )
              {
                sub_C8D5F0(v94, (const void *)(v94 + 16), v215, 8u, v78, v74);
                v97 = *(unsigned int *)(v94 + 8);
              }
              if ( 8 * v196 )
              {
                memset((void *)(*(_QWORD *)v94 + 8 * v97), 0, 8 * v196);
                LODWORD(v97) = *(_DWORD *)(v94 + 8);
              }
              v96 = *(_DWORD *)(v94 + 64);
              *(_DWORD *)(v94 + 8) = v196 + v97;
            }
            else
            {
              *(_DWORD *)(v94 + 8) = v203;
            }
          }
          v98 = v96 & 0x3F;
          if ( v98 )
            *(_QWORD *)(*(_QWORD *)v94 + 8LL * *(unsigned int *)(v94 + 8) - 8) &= ~(-1LL << v98);
          v99 = v71[208] + v93;
          v100 = v99;
          v101 = *(_DWORD *)(v99 + 64) & 0x3F;
          if ( v101 )
            *(_QWORD *)(*(_QWORD *)v99 + 8LL * *(unsigned int *)(v99 + 8) - 8) &= ~(-1LL << v101);
          LOBYTE(v102) = v212;
          v103 = *(unsigned int *)(v99 + 8);
          *(_DWORD *)(v99 + 64) = v212;
          if ( v215 != v103 )
          {
            if ( v215 >= v103 )
            {
              v164 = *(unsigned int *)(v99 + 12);
              v165 = v215 - v103;
              if ( v215 > v164 )
              {
                sub_C8D5F0(v100, (const void *)(v100 + 16), v215, 8u, v78, v74);
                v103 = *(unsigned int *)(v100 + 8);
              }
              if ( 8 * v165 )
              {
                memset((void *)(*(_QWORD *)v100 + 8 * v103), 0, 8 * v165);
                LODWORD(v103) = *(_DWORD *)(v100 + 8);
              }
              v102 = *(_DWORD *)(v100 + 64);
              *(_DWORD *)(v100 + 8) = v165 + v103;
            }
            else
            {
              *(_DWORD *)(v99 + 8) = v203;
            }
          }
          v104 = v102 & 0x3F;
          if ( v104 )
            *(_QWORD *)(*(_QWORD *)v100 + 8LL * *(unsigned int *)(v100 + 8) - 8) &= ~(-1LL << v104);
        }
        *(_QWORD *)(*(_QWORD *)(v71[185] + v204) + 8LL * ((unsigned int)v75 >> 6)) |= 1LL << v75;
      }
      v72 += 8;
    }
    while ( v210 != v72 );
    v6 = v71;
  }
  sub_2FD1370(v201);
LABEL_194:
  v128 = *((unsigned int *)v6 + 372);
  v129 = *((unsigned int *)v6 + 412);
  v130 = *((_DWORD *)v6 + 372);
  if ( v128 != v129 )
  {
    if ( v128 < v129 )
    {
      *((_DWORD *)v6 + 412) = v128;
    }
    else
    {
      if ( v128 > *((unsigned int *)v6 + 413) )
      {
        sub_C8D5F0((__int64)(v6 + 205), v6 + 207, *((unsigned int *)v6 + 372), 4u, v126, v127);
        v129 = *((unsigned int *)v6 + 412);
      }
      v131 = v6[205];
      v132 = (_DWORD *)(v131 + 4 * v129);
      for ( i1 = v131 + 4 * v128; (_DWORD *)i1 != v132; ++v132 )
      {
        if ( v132 )
          *v132 = 0;
      }
      *((_DWORD *)v6 + 412) = v128;
      v130 = *((_DWORD *)v6 + 372);
    }
  }
  if ( v130 )
  {
    v127 = 0;
    v133 = 0;
    do
    {
      v134 = (__int64 *)(v133 + v6[185]);
      v125 = *((unsigned int *)v134 + 16);
      if ( (_DWORD)v125 )
      {
        v135 = *v134;
        v126 = (unsigned int)(v125 - 1) >> 6;
        v136 = 0;
        while ( 1 )
        {
          _RDX = *(_QWORD *)(v135 + 8 * v136);
          if ( v136 == v126 )
            break;
          if ( _RDX )
            goto LABEL_211;
          if ( ++v136 == (_DWORD)v126 + 1 )
            goto LABEL_243;
        }
        v125 = (unsigned int)-(int)v125;
        _RDX &= 0xFFFFFFFFFFFFFFFFLL >> v125;
        if ( !_RDX )
          goto LABEL_243;
LABEL_211:
        __asm { tzcnt   rdx, rdx }
        i1 = (unsigned int)(((_DWORD)v136 << 6) + _RDX);
      }
      else
      {
LABEL_243:
        i1 = 0xFFFFFFFFLL;
      }
      v133 += 72;
      *(_DWORD *)(v6[205] + v127) = i1;
      v127 += 4;
    }
    while ( 4LL * v130 != v127 );
  }
  if ( src != v220 )
    _libc_free((unsigned __int64)src);
  v141 = sub_2FD1410((char **)v6, a2, i1, v125, v126, v127);
  v142 = 4LL * *((unsigned int *)v6 + 412);
  if ( v142 )
    memset((void *)v6[205], 255, v142);
  v143 = v6[5];
  if ( v143 != v6[6] )
    v6[6] = v143;
  v144 = v6[8];
  v145 = v144 + 80LL * *((unsigned int *)v6 + 18);
  if ( v144 != v145 )
  {
    do
    {
      *(_DWORD *)(v144 + 8) = 0;
      v144 += 80;
    }
    while ( v145 != v144 );
    v146 = (unsigned __int64 *)v6[8];
    v147 = &v146[10 * *((unsigned int *)v6 + 18)];
    while ( v146 != v147 )
    {
      while ( 1 )
      {
        v147 -= 10;
        if ( (unsigned __int64 *)*v147 == v147 + 2 )
          break;
        _libc_free(*v147);
        if ( v146 == v147 )
          goto LABEL_225;
      }
    }
  }
LABEL_225:
  v148 = *((unsigned int *)v6 + 372);
  v149 = (unsigned __int64 *)v6[185];
  *((_DWORD *)v6 + 18) = 0;
  v6[171] = 0;
  *((_DWORD *)v6 + 352) = 0;
  v150 = &v149[9 * v148];
  while ( v149 != v150 )
  {
    v150 -= 9;
    if ( (unsigned __int64 *)*v150 != v150 + 2 )
      _libc_free(*v150);
  }
  v151 = *((unsigned int *)v6 + 418);
  v152 = (unsigned __int64 *)v6[208];
  *((_DWORD *)v6 + 372) = 0;
  v153 = &v152[9 * v151];
  while ( v152 != v153 )
  {
    v153 -= 9;
    if ( (unsigned __int64 *)*v153 != v153 + 2 )
      _libc_free(*v153);
  }
  v154 = *((unsigned int *)v6 + 484);
  v155 = v6[241];
  *((_DWORD *)v6 + 418) = 0;
  v156 = v155 + 232 * v154;
  if ( v155 != v156 )
  {
    v216 = v6;
    v157 = v141;
    v158 = v155;
    do
    {
      while ( 1 )
      {
        v159 = *(_QWORD *)(v156 - 224);
        v156 -= 232;
        if ( v159 )
        {
          if ( *(_DWORD *)(v159 + 200) )
            break;
        }
        if ( v158 == v156 )
          goto LABEL_241;
      }
      v160 = v159 + 8;
      v161 = (_QWORD *)(v159 + 136);
      sub_2E19AD0(v160, (char *)sub_2E199D0, 0, v155, v139, v140);
      v162 = (_QWORD *)v160;
      do
      {
        *v162 = 0;
        v162 += 2;
        *(v162 - 1) = 0;
      }
      while ( v161 != v162 );
    }
    while ( v158 != v156 );
LABEL_241:
    v141 = v157;
    v6 = v216;
  }
  *((_DWORD *)v6 + 484) = 0;
  return v141;
}
