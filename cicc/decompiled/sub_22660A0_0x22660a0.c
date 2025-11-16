// Function: sub_22660A0
// Address: 0x22660a0
//
unsigned __int64 *__fastcall sub_22660A0(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // ecx
  __int64 *v6; // rsi
  __int64 v8; // rax
  __int64 *v9; // rax
  __int64 v10; // rdx
  __int64 *v11; // r12
  __int64 *v12; // r13
  __m128i *v13; // rcx
  __m128i *v14; // rsi
  void *v15; // rdx
  __int64 v16; // rax
  __int64 *v17; // rax
  __int64 v18; // rdx
  const void **v19; // r13
  const __m128i *v20; // r15
  __int64 v21; // r12
  unsigned __int64 v22; // rax
  const __m128i *v23; // r15
  const void *v24; // r13
  size_t v25; // r12
  __int64 v26; // rax
  const void *v27; // r14
  int v28; // eax
  int v29; // eax
  __int64 v30; // rax
  int v31; // eax
  int v32; // eax
  __m128i *v33; // r12
  __m128i *v34; // r13
  __int64 v35; // rdx
  __int64 v36; // r12
  char *v37; // rax
  void *v38; // r14
  char *v39; // r13
  char *v40; // r12
  __int64 v41; // rax
  int v42; // eax
  char *v43; // rsi
  size_t v44; // r14
  const void *v45; // r15
  int v46; // eax
  int v47; // eax
  __int64 v48; // r14
  unsigned __int64 v49; // rcx
  unsigned __int64 v50; // r13
  _QWORD *v51; // r8
  unsigned __int64 v52; // rbx
  __int64 v53; // r8
  __int64 v54; // r11
  __int64 v55; // r12
  const __m128i *v56; // rcx
  unsigned int v57; // edx
  int v58; // eax
  __m128i *v59; // rax
  int v60; // eax
  __int64 v61; // rcx
  const void **v62; // r11
  unsigned __int64 v63; // r8
  int v64; // r9d
  char *v65; // rdi
  size_t v66; // rdx
  const void **v67; // r10
  const void *v68; // rcx
  bool v69; // al
  int v70; // eax
  __int64 v71; // rcx
  unsigned int v72; // r9d
  int v73; // eax
  __m128i *v74; // rax
  unsigned __int64 v75; // r15
  __m128i *v76; // rbx
  __int64 v77; // rbx
  int v78; // eax
  int v79; // eax
  __int64 v80; // rax
  size_t v81; // rbx
  int v82; // eax
  __int64 v83; // rcx
  _QWORD *v84; // r9
  __int64 v85; // r15
  int v86; // r8d
  _QWORD *v87; // rbx
  int v88; // esi
  int v89; // edx
  __m128i *v90; // rax
  _QWORD *v91; // r12
  unsigned int v92; // eax
  void *v93; // rax
  __int64 v94; // rdx
  _BYTE *v95; // rsi
  _DWORD *v96; // rdx
  unsigned __int64 v97; // rdi
  _QWORD *v98; // rax
  _QWORD *v99; // rbx
  unsigned __int64 v100; // r12
  __int64 v101; // rsi
  __int64 v102; // rdi
  int v103; // eax
  __m128i *v104; // rcx
  int v105; // r9d
  char *v106; // rdi
  size_t v107; // rdx
  const void **v108; // r10
  const void *v109; // rcx
  bool v110; // al
  int v111; // eax
  const void *v112; // r12
  int v113; // eax
  int v114; // eax
  __int64 v115; // rax
  size_t v116; // r12
  int v117; // eax
  unsigned int v118; // eax
  _QWORD *v119; // r11
  _QWORD *v120; // r12
  __int64 v121; // rax
  unsigned int v122; // r10d
  _QWORD *v123; // r11
  void *v124; // r8
  _QWORD *v125; // rcx
  unsigned int v126; // eax
  __int64 *v127; // rax
  __int64 *v128; // rax
  __int64 v129; // rax
  unsigned int v130; // r8d
  _QWORD *v131; // r9
  _QWORD *v132; // r15
  __int64 *v133; // rax
  __int64 *v134; // rax
  int v135; // eax
  const void *v136; // rdi
  size_t v137; // rdx
  char *v138; // r11
  int v139; // ecx
  int v140; // r8d
  unsigned int j; // r9d
  const void *v142; // rsi
  unsigned int v143; // r9d
  unsigned int v144; // eax
  int v145; // eax
  int v146; // eax
  __m128i *v147; // r10
  __int64 i; // rcx
  __int64 v149; // [rsp+8h] [rbp-168h]
  __int64 v150; // [rsp+10h] [rbp-160h]
  const void **v151; // [rsp+18h] [rbp-158h]
  size_t v152; // [rsp+18h] [rbp-158h]
  int v153; // [rsp+20h] [rbp-150h]
  const void *v154; // [rsp+20h] [rbp-150h]
  unsigned __int64 v155; // [rsp+28h] [rbp-148h]
  size_t v156; // [rsp+30h] [rbp-140h]
  __m128i *v157; // [rsp+30h] [rbp-140h]
  const void *v158; // [rsp+38h] [rbp-138h]
  int v159; // [rsp+38h] [rbp-138h]
  int v160; // [rsp+38h] [rbp-138h]
  __int64 v161; // [rsp+40h] [rbp-130h]
  __int64 v162; // [rsp+40h] [rbp-130h]
  void *v163; // [rsp+40h] [rbp-130h]
  char *v164; // [rsp+40h] [rbp-130h]
  unsigned __int64 *v165; // [rsp+48h] [rbp-128h]
  unsigned __int64 v166; // [rsp+50h] [rbp-120h]
  void *v167; // [rsp+50h] [rbp-120h]
  _QWORD *v168; // [rsp+50h] [rbp-120h]
  unsigned int v169; // [rsp+50h] [rbp-120h]
  __int64 v170; // [rsp+58h] [rbp-118h]
  int v171; // [rsp+58h] [rbp-118h]
  int v172; // [rsp+58h] [rbp-118h]
  __int64 v173; // [rsp+58h] [rbp-118h]
  unsigned __int64 v174; // [rsp+58h] [rbp-118h]
  _QWORD *v175; // [rsp+58h] [rbp-118h]
  int v176; // [rsp+58h] [rbp-118h]
  unsigned __int64 v177; // [rsp+58h] [rbp-118h]
  __int64 v179; // [rsp+60h] [rbp-110h]
  __int64 v180; // [rsp+60h] [rbp-110h]
  __m128i *v181; // [rsp+60h] [rbp-110h]
  const void **v182; // [rsp+60h] [rbp-110h]
  __int64 v183; // [rsp+60h] [rbp-110h]
  __int64 v184; // [rsp+60h] [rbp-110h]
  unsigned int v185; // [rsp+60h] [rbp-110h]
  _QWORD *v186; // [rsp+60h] [rbp-110h]
  size_t v187; // [rsp+60h] [rbp-110h]
  __int64 v188; // [rsp+60h] [rbp-110h]
  __int64 v189; // [rsp+60h] [rbp-110h]
  void *v191; // [rsp+68h] [rbp-108h]
  int v192; // [rsp+68h] [rbp-108h]
  unsigned int v193; // [rsp+68h] [rbp-108h]
  void *v194; // [rsp+68h] [rbp-108h]
  char *v195; // [rsp+68h] [rbp-108h]
  void *v196; // [rsp+68h] [rbp-108h]
  void *v197; // [rsp+68h] [rbp-108h]
  void *v198; // [rsp+68h] [rbp-108h]
  unsigned int v199; // [rsp+68h] [rbp-108h]
  int v200; // [rsp+68h] [rbp-108h]
  void *v201; // [rsp+68h] [rbp-108h]
  __m128i *v202; // [rsp+70h] [rbp-100h]
  __m128i *v203; // [rsp+70h] [rbp-100h]
  void *v204; // [rsp+70h] [rbp-100h]
  void *v205; // [rsp+70h] [rbp-100h]
  int v206; // [rsp+70h] [rbp-100h]
  unsigned int v207; // [rsp+70h] [rbp-100h]
  void *v208; // [rsp+70h] [rbp-100h]
  _DWORD *v209; // [rsp+70h] [rbp-100h]
  char *v210; // [rsp+70h] [rbp-100h]
  __int64 *v211; // [rsp+78h] [rbp-F8h]
  unsigned int v212; // [rsp+78h] [rbp-F8h]
  size_t v213; // [rsp+80h] [rbp-F0h]
  unsigned int v214; // [rsp+88h] [rbp-E8h]
  __m128i *v215; // [rsp+98h] [rbp-D8h] BYREF
  __m128i *v216; // [rsp+A0h] [rbp-D0h] BYREF
  __m128i *v217; // [rsp+A8h] [rbp-C8h]
  __m128i *v218; // [rsp+B0h] [rbp-C0h]
  void *v219; // [rsp+C0h] [rbp-B0h] BYREF
  char *v220; // [rsp+C8h] [rbp-A8h]
  char *v221; // [rsp+D0h] [rbp-A0h]
  _QWORD *v222; // [rsp+E0h] [rbp-90h] BYREF
  _QWORD *v223; // [rsp+E8h] [rbp-88h]
  _QWORD *v224; // [rsp+F0h] [rbp-80h]
  unsigned __int64 v225; // [rsp+100h] [rbp-70h] BYREF
  _BYTE *v226; // [rsp+108h] [rbp-68h]
  _BYTE *v227; // [rsp+110h] [rbp-60h]
  void *src[2]; // [rsp+120h] [rbp-50h] BYREF
  __int64 v229; // [rsp+130h] [rbp-40h]
  __int64 v230; // [rsp+138h] [rbp-38h]

  v4 = *(_DWORD *)(a3 + 8);
  v165 = a1;
  v216 = 0;
  v217 = 0;
  v218 = 0;
  if ( !v4 )
  {
    a1[2] = 0;
    *(_OWORD *)a1 = 0;
    return v165;
  }
  v6 = *(__int64 **)a3;
  v8 = **(_QWORD **)a3;
  if ( v8 != -8 && v8 )
  {
    v11 = *(__int64 **)a3;
  }
  else
  {
    v9 = v6 + 1;
    do
    {
      do
      {
        v10 = *v9;
        v11 = v9++;
      }
      while ( v10 == -8 );
    }
    while ( !v10 );
  }
  v12 = &v6[v4];
  if ( v12 == v11 )
  {
    for ( i = 6; i; --i )
    {
      *(_DWORD *)a1 = 0;
      a1 = (unsigned __int64 *)((char *)a1 + 4);
    }
    return v165;
  }
  v13 = 0;
  v14 = 0;
  while ( 1 )
  {
    v15 = *(void **)*v11;
    src[0] = (void *)(*v11 + 16);
    src[1] = v15;
    if ( v14 == v13 )
    {
      sub_C677B0((const __m128i **)&v216, v14, (const __m128i *)src);
      v14 = v217;
    }
    else
    {
      if ( v14 )
      {
        *v14 = _mm_loadu_si128((const __m128i *)src);
        v14 = v217;
      }
      v217 = ++v14;
    }
    v16 = v11[1];
    if ( v16 != -8 && v16 )
    {
      ++v11;
    }
    else
    {
      v17 = v11 + 2;
      do
      {
        do
        {
          v18 = *v17;
          v11 = v17++;
        }
        while ( v18 == -8 );
      }
      while ( !v18 );
    }
    if ( v11 == v12 )
      break;
    v13 = v218;
  }
  v19 = (const void **)v216;
  v20 = v14;
  if ( v216 == v14 )
  {
    a1[2] = 0;
    *(_OWORD *)a1 = 0;
    goto LABEL_126;
  }
  v21 = (char *)v14 - (char *)v216;
  _BitScanReverse64(&v22, v14 - v216);
  sub_2261690(v216, v14, 2LL * (int)(63 - (v22 ^ 0x3F)), a3);
  if ( v21 <= 256 )
  {
    sub_2260CB0(v19, (const void **)v14, a3);
    goto LABEL_36;
  }
  sub_2260CB0(v19, v19 + 32, a3);
  v202 = (__m128i *)(v19 + 32);
  if ( v19 + 32 == (const void **)v14 )
    goto LABEL_36;
  do
  {
    v23 = v202;
    v24 = (const void *)v202->m128i_i64[0];
    v25 = v202->m128i_u64[1];
    while ( 1 )
    {
      v211 = (__int64 *)v23;
      v31 = sub_C92610();
      v32 = sub_C92860((__int64 *)a3, v24, v25, v31);
      if ( v32 == -1 || (v26 = *(_QWORD *)a3 + 8LL * v32, v26 == *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8)) )
        v214 = 0;
      else
        v214 = *(_DWORD *)(*(_QWORD *)v26 + 8LL);
      v27 = (const void *)v23[-1].m128i_i64[0];
      v213 = v23[-1].m128i_u64[1];
      v28 = sub_C92610();
      v29 = sub_C92860((__int64 *)a3, v27, v213, v28);
      if ( v29 == -1 )
        break;
      v30 = *(_QWORD *)a3 + 8LL * v29;
      if ( v30 == *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8) )
        break;
      --v23;
      if ( *(_DWORD *)(*(_QWORD *)v30 + 8LL) >= v214 )
        goto LABEL_185;
LABEL_32:
      v23[1] = _mm_loadu_si128(v23);
    }
    --v23;
    if ( v214 )
      goto LABEL_32;
LABEL_185:
    ++v202;
    *v211 = (__int64)v24;
    v211[1] = v25;
  }
  while ( v14 != v202 );
LABEL_36:
  v33 = v217;
  v34 = v216;
  v219 = 0;
  v220 = 0;
  v221 = 0;
  v35 = v217 - v216;
  if ( (char *)v217 - (char *)v216 < 0 )
    sub_4262D8((__int64)"vector::reserve");
  if ( v35 )
  {
    v36 = 4 * v35;
    v37 = (char *)sub_22077B0(4 * v35);
    v38 = v219;
    v39 = v37;
    if ( v220 - (_BYTE *)v219 > 0 )
    {
      memmove(v37, v219, v220 - (_BYTE *)v219);
    }
    else if ( !v219 )
    {
      goto LABEL_40;
    }
    j_j___libc_free_0((unsigned __int64)v38);
LABEL_40:
    v40 = &v39[v36];
    v219 = v39;
    v220 = v39;
    v34 = v216;
    v221 = v40;
    v33 = v217;
  }
  if ( v34 == v33 )
  {
    v147 = v33;
  }
  else
  {
    do
    {
      while ( 1 )
      {
        v44 = v34->m128i_u64[1];
        v45 = (const void *)v34->m128i_i64[0];
        v46 = sub_C92610();
        v47 = sub_C92860((__int64 *)a3, v45, v44, v46);
        if ( v47 == -1 || (v41 = *(_QWORD *)a3 + 8LL * v47, v41 == *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8)) )
          v42 = 0;
        else
          v42 = *(_DWORD *)(*(_QWORD *)v41 + 8LL);
        LODWORD(src[0]) = v42;
        v43 = v220;
        if ( v220 != v221 )
          break;
        ++v34;
        sub_C88AB0((__int64)&v219, v220, src);
        if ( v33 == v34 )
          goto LABEL_53;
      }
      if ( v220 )
      {
        *(_DWORD *)v220 = v42;
        v43 = v220;
      }
      ++v34;
      v220 = v43 + 4;
    }
    while ( v33 != v34 );
LABEL_53:
    v33 = v217;
    v147 = v216;
  }
  if ( *(_BYTE *)(a2 + 1520) )
  {
    v212 = *((_DWORD *)v220 - 1);
  }
  else
  {
    v144 = *(_DWORD *)(a2 + 1440);
    if ( *(_DWORD *)v219 >= v144 )
      v144 = *(_DWORD *)v219;
    v212 = v144;
  }
  v222 = 0;
  v223 = 0;
  v224 = 0;
  v225 = 0;
  v226 = 0;
  v227 = 0;
  if ( v33 == v147 )
  {
    *a1 = 0;
    a1[1] = 0;
    a1[2] = 0;
    goto LABEL_123;
  }
  v48 = a4;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  while ( 2 )
  {
    v52 = 0;
    v53 = (__int64)((__int64)v51 - v49) >> 5;
    if ( !v53 )
      goto LABEL_86;
    while ( *(_DWORD *)(v225 + 4 * v52) > v212 - *((_DWORD *)v219 + v50) )
    {
      if ( ++v52 == v53 )
        goto LABEL_86;
    }
    v54 = 16 * v50;
    v55 = v49 + 32 * v52;
    v56 = &v147[v50];
    v57 = *(_DWORD *)(v55 + 24);
    if ( !v57 )
    {
      src[0] = 0;
      ++*(_QWORD *)v55;
      goto LABEL_64;
    }
    v162 = v53;
    v206 = *(_DWORD *)(v55 + 24);
    v181 = &v147[v50];
    v195 = *(char **)(v55 + 8);
    v103 = sub_C94890(v56->m128i_i64[0], v56->m128i_i64[1]);
    v104 = v181;
    v159 = 1;
    v54 = 16 * v50;
    v53 = v162;
    v182 = 0;
    v105 = v206 - 1;
    v157 = v104;
    v106 = (char *)v104->m128i_i64[0];
    v107 = v104->m128i_u64[1];
    v207 = (v206 - 1) & v103;
    while ( 2 )
    {
      v108 = (const void **)&v195[16 * v207];
      v109 = *v108;
      v110 = v106 + 1 == 0;
      if ( *v108 != (const void *)-1LL )
      {
        v110 = v106 + 2 == 0;
        if ( v109 != (const void *)-2LL )
        {
          if ( (const void *)v107 != v108[1] )
          {
LABEL_132:
            if ( v109 != (const void *)-2LL || v182 )
              v108 = v182;
            v182 = v108;
            v207 = v105 & (v159 + v207);
            ++v159;
            continue;
          }
          v172 = v105;
          if ( !v107 )
            goto LABEL_138;
          v149 = v53;
          v150 = v54;
          v152 = v107;
          v154 = *v108;
          v111 = memcmp(v106, v109, v107);
          v109 = v154;
          v107 = v152;
          v54 = v150;
          v53 = v149;
          v105 = v172;
          v110 = v111 == 0;
          v108 = (const void **)&v195[16 * v207];
        }
      }
      break;
    }
    if ( v110 )
      goto LABEL_138;
    if ( v109 != (const void *)-1LL )
      goto LABEL_132;
    v57 = *(_DWORD *)(v55 + 24);
    v56 = v157;
    if ( v182 )
      v108 = v182;
    src[0] = v108;
    v146 = *(_DWORD *)(v55 + 16);
    ++*(_QWORD *)v55;
    v58 = v146 + 1;
    if ( 4 * v58 < 3 * v57 )
    {
      if ( v57 - (v58 + *(_DWORD *)(v55 + 20)) <= v57 >> 3 )
      {
        v188 = v53;
        v201 = (void *)v54;
        sub_BA8070(v55, v57);
        sub_B9B010(v55, v157, src);
        v53 = v188;
        v54 = (__int64)v201;
        v56 = v157;
        v58 = *(_DWORD *)(v55 + 16) + 1;
      }
      goto LABEL_65;
    }
LABEL_64:
    v179 = v53;
    v191 = (void *)v54;
    v203 = (__m128i *)v56;
    sub_BA8070(v55, 2 * v57);
    sub_B9B010(v55, v203, src);
    v56 = v203;
    v54 = (__int64)v191;
    v53 = v179;
    v58 = *(_DWORD *)(v55 + 16) + 1;
LABEL_65:
    *(_DWORD *)(v55 + 16) = v58;
    v59 = (__m128i *)src[0];
    if ( *(_QWORD *)src[0] != -1 )
      --*(_DWORD *)(v55 + 20);
    *v59 = _mm_loadu_si128(v56);
LABEL_138:
    v173 = v53;
    v183 = v54;
    v208 = (void *)(*(_QWORD *)v48 + 8LL * *(unsigned int *)(v48 + 8));
    v112 = *(const void **)((char *)v216->m128i_i64 + v54);
    v196 = *(void **)((char *)&v216->m128i_i64[1] + v54);
    v113 = sub_C92610();
    v114 = sub_C92860((__int64 *)v48, v112, (size_t)v196, v113);
    v63 = v173;
    if ( v114 == -1 )
      v115 = *(_QWORD *)v48 + 8LL * *(unsigned int *)(v48 + 8);
    else
      v115 = *(_QWORD *)v48 + 8LL * v114;
    if ( v208 == (void *)v115 )
      goto LABEL_85;
    v116 = *(unsigned __int64 *)((char *)&v216->m128i_u64[1] + v183);
    v209 = &v222[4 * v52];
    v197 = *(void **)((char *)v216->m128i_i64 + v183);
    v117 = sub_C92610();
    v118 = sub_C92740(v48, v197, v116, v117);
    v63 = v173;
    v119 = (_QWORD *)(*(_QWORD *)v48 + 8LL * v118);
    v71 = *v119;
    if ( *v119 )
    {
      if ( v71 != -8 )
        goto LABEL_143;
      --*(_DWORD *)(v48 + 16);
    }
    v167 = (void *)v173;
    v175 = v119;
    v185 = v118;
    v121 = sub_C7D670(v116 + 25, 8);
    v122 = v185;
    v123 = v175;
    v124 = v167;
    v125 = (_QWORD *)v121;
    if ( v116 )
    {
      v163 = v167;
      v168 = (_QWORD *)v121;
      memcpy((void *)(v121 + 24), v197, v116);
      v122 = v185;
      v123 = v175;
      v125 = v168;
      v124 = v163;
    }
    *((_BYTE *)v125 + v116 + 24) = 0;
    *v125 = v116;
    v125[1] = 0;
    v125[2] = 0;
    *v123 = v125;
    ++*(_DWORD *)(v48 + 12);
    v198 = v124;
    v126 = sub_C929D0((__int64 *)v48, v122);
    v63 = (unsigned __int64)v198;
    v127 = (__int64 *)(*(_QWORD *)v48 + 8LL * v126);
    v71 = *v127;
    if ( *v127 == -8 || !v71 )
    {
      v128 = v127 + 1;
      do
      {
        do
          v71 = *v128++;
        while ( v71 == -8 );
      }
      while ( !v71 );
    }
LABEL_143:
    v120 = (_QWORD *)(v71 + 8);
    v72 = v209[6];
    if ( !v72 )
    {
      src[0] = 0;
      ++*(_QWORD *)v209;
LABEL_145:
      v174 = v63;
      v184 = v71;
      sub_BA8070((__int64)v209, 2 * v72);
      sub_B9B010((__int64)v209, v120, src);
      v71 = v184;
      v63 = v174;
      v73 = v209[4] + 1;
      goto LABEL_82;
    }
    v166 = v63;
    v192 = v209[6];
    v170 = v71;
    v180 = *((_QWORD *)v209 + 1);
    v60 = sub_C94890(*(_QWORD **)(v71 + 8), *(_QWORD *)(v71 + 16));
    v61 = v170;
    v62 = 0;
    v171 = 1;
    v63 = v166;
    v64 = v192 - 1;
    v161 = v61;
    v65 = *(char **)(v61 + 8);
    v66 = *(_QWORD *)(v61 + 16);
    v193 = (v192 - 1) & v60;
    while ( 2 )
    {
      v67 = (const void **)(v180 + 16LL * v193);
      v68 = *v67;
      if ( *v67 != (const void *)-1LL )
      {
        v69 = v65 + 2 == 0;
        if ( v68 == (const void *)-2LL )
        {
LABEL_74:
          if ( v69 )
            goto LABEL_85;
          if ( v68 == (const void *)-1LL )
          {
            v71 = v161;
            goto LABEL_77;
          }
        }
        else if ( v67[1] == (const void *)v66 )
        {
          v151 = v62;
          v153 = v64;
          if ( !v66 )
            goto LABEL_85;
          v155 = v63;
          v156 = v66;
          v158 = *v67;
          v70 = memcmp(v65, v68, v66);
          v68 = v158;
          v66 = v156;
          v63 = v155;
          v64 = v153;
          v62 = v151;
          v69 = v70 == 0;
          v67 = (const void **)(v180 + 16LL * v193);
          goto LABEL_74;
        }
        if ( v68 != (const void *)-2LL || v62 )
          v67 = v62;
        v62 = v67;
        v193 = v64 & (v171 + v193);
        ++v171;
        continue;
      }
      break;
    }
    v71 = v161;
    if ( v65 == (char *)-1LL )
      goto LABEL_85;
LABEL_77:
    if ( !v62 )
      v62 = v67;
    v72 = v209[6];
    src[0] = v62;
    ++*(_QWORD *)v209;
    v73 = v209[4] + 1;
    if ( 4 * v73 >= 3 * v72 )
      goto LABEL_145;
    if ( v72 - (v73 + v209[5]) <= v72 >> 3 )
    {
      v177 = v63;
      v189 = v71;
      sub_BA8070((__int64)v209, v72);
      sub_B9B010((__int64)v209, v120, src);
      v63 = v177;
      v71 = v189;
      v73 = v209[4] + 1;
    }
LABEL_82:
    v209[4] = v73;
    v74 = (__m128i *)src[0];
    if ( *(_QWORD *)src[0] != -1 )
      --v209[5];
    *v74 = _mm_loadu_si128((const __m128i *)(v71 + 8));
LABEL_85:
    *(_DWORD *)(4 * v52 + v225) += *((_DWORD *)v219 + v50);
    if ( v63 <= v52 )
    {
LABEL_86:
      v75 = v50;
      src[0] = (void *)1;
      v76 = &v216[v50];
      src[1] = 0;
      v215 = 0;
      v229 = 0;
      v230 = 0;
      sub_BA8070((__int64)src, 0);
      sub_B9B010((__int64)src, v76, &v215);
      LODWORD(v229) = v229 + 1;
      if ( v215->m128i_i64[0] != -1 )
        --HIDWORD(v229);
      *v215 = _mm_loadu_si128(v76);
      v77 = *(_QWORD *)v48 + 8LL * *(unsigned int *)(v48 + 8);
      v194 = (void *)v216[v50].m128i_i64[0];
      v204 = (void *)v216[v75].m128i_i64[1];
      v78 = sub_C92610();
      v79 = sub_C92860((__int64 *)v48, v194, (size_t)v204, v78);
      if ( v79 == -1 )
        v80 = *(_QWORD *)v48 + 8LL * *(unsigned int *)(v48 + 8);
      else
        v80 = *(_QWORD *)v48 + 8LL * v79;
      if ( v80 != v77 )
      {
        v81 = v216[v75].m128i_u64[1];
        v205 = (void *)v216[v75].m128i_i64[0];
        v82 = sub_C92610();
        v83 = (unsigned int)sub_C92740(v48, v205, v81, v82);
        v84 = (_QWORD *)(*(_QWORD *)v48 + 8 * v83);
        v85 = *v84;
        if ( *v84 )
        {
          if ( v85 != -8 )
            goto LABEL_93;
          --*(_DWORD *)(v48 + 16);
        }
        v186 = v84;
        v199 = v83;
        v129 = sub_C7D670(v81 + 25, 8);
        v130 = v199;
        v131 = v186;
        v132 = (_QWORD *)v129;
        if ( v81 )
        {
          memcpy((void *)(v129 + 24), v205, v81);
          v130 = v199;
          v131 = v186;
        }
        *((_BYTE *)v132 + v81 + 24) = 0;
        *v132 = v81;
        v132[1] = 0;
        v132[2] = 0;
        *v131 = v132;
        ++*(_DWORD *)(v48 + 12);
        v133 = (__int64 *)(*(_QWORD *)v48 + 8LL * (unsigned int)sub_C929D0((__int64 *)v48, v130));
        v85 = *v133;
        if ( *v133 == -8 || !v85 )
        {
          v134 = v133 + 1;
          do
          {
            do
              v85 = *v134++;
            while ( !v85 );
          }
          while ( v85 == -8 );
          v86 = v230;
          v87 = (_QWORD *)(v85 + 8);
          if ( (_DWORD)v230 )
            goto LABEL_167;
LABEL_94:
          ++src[0];
          v215 = 0;
LABEL_95:
          v88 = 2 * v86;
LABEL_96:
          sub_BA8070((__int64)src, v88);
          sub_B9B010((__int64)src, v87, &v215);
          v89 = v229 + 1;
          v90 = v215;
          goto LABEL_102;
        }
LABEL_93:
        v86 = v230;
        v87 = (_QWORD *)(v85 + 8);
        if ( !(_DWORD)v230 )
          goto LABEL_94;
LABEL_167:
        v200 = v86;
        v210 = (char *)src[1];
        v135 = sub_C94890(*(_QWORD **)(v85 + 8), v87[1]);
        v136 = *(const void **)(v85 + 8);
        v137 = *(_QWORD *)(v85 + 16);
        v138 = 0;
        v139 = 1;
        v140 = v200 - 1;
        for ( j = (v200 - 1) & v135; ; j = v140 & v143 )
        {
          v90 = (__m128i *)&v210[16 * j];
          v142 = (const void *)v90->m128i_i64[0];
          if ( v90->m128i_i64[0] == -1 )
            break;
          if ( v142 == (const void *)-2LL )
          {
            if ( v136 == (const void *)-2LL )
              goto LABEL_105;
            if ( !v138 )
              v138 = &v210[16 * j];
          }
          else if ( v90->m128i_i64[1] == v137 )
          {
            v160 = v139;
            v164 = v138;
            v169 = j;
            v176 = v140;
            if ( !v137 )
              goto LABEL_105;
            v187 = v137;
            v145 = memcmp(v136, v142, v137);
            v137 = v187;
            v140 = v176;
            j = v169;
            v138 = v164;
            v139 = v160;
            if ( !v145 )
              goto LABEL_105;
          }
          v143 = v139 + j;
          ++v139;
        }
        if ( v136 != (const void *)-1LL )
        {
          v86 = v230;
          if ( v138 )
            v90 = (__m128i *)v138;
          ++src[0];
          v89 = v229 + 1;
          v215 = v90;
          if ( 4 * ((int)v229 + 1) >= (unsigned int)(3 * v230) )
            goto LABEL_95;
          if ( (int)v230 - (v89 + HIDWORD(v229)) <= (unsigned int)v230 >> 3 )
          {
            v88 = v230;
            goto LABEL_96;
          }
LABEL_102:
          LODWORD(v229) = v89;
          if ( v90->m128i_i64[0] != -1 )
            --HIDWORD(v229);
          *v90 = _mm_loadu_si128((const __m128i *)(v85 + 8));
        }
      }
LABEL_105:
      v91 = v223;
      if ( v223 == v224 )
      {
        sub_2265400((unsigned __int64 *)&v222, (__int64)v223, (__int64)src);
      }
      else
      {
        if ( v223 )
        {
          *v223 = 0;
          v91[1] = 0;
          v91[2] = 0;
          *((_DWORD *)v91 + 6) = 0;
          sub_C7D6A0(0, 0, 8);
          v92 = v230;
          *((_DWORD *)v91 + 6) = v230;
          if ( v92 )
          {
            v93 = (void *)sub_C7D670(16LL * v92, 8);
            v94 = *((unsigned int *)v91 + 6);
            v91[1] = v93;
            v91[2] = v229;
            memcpy(v93, src[1], 16 * v94);
          }
          else
          {
            v91[1] = 0;
            v91[2] = 0;
          }
          v91 = v223;
        }
        v223 = v91 + 4;
      }
      v95 = v226;
      v96 = (char *)v219 + 4 * v50;
      if ( v226 == v227 )
      {
        sub_12DD6D0((__int64)&v225, v226, v96);
      }
      else
      {
        if ( v226 )
        {
          *(_DWORD *)v226 = *v96;
          v95 = v226;
        }
        v226 = v95 + 4;
      }
      sub_C7D6A0((__int64)src[1], 16LL * (unsigned int)v230, 8);
    }
    v147 = v216;
    ++v50;
    v49 = (unsigned __int64)v222;
    v51 = v223;
    if ( v50 < v217 - v216 )
      continue;
    break;
  }
  v97 = v225;
  v223 = 0;
  v222 = 0;
  *v165 = v49;
  v165[1] = (unsigned __int64)v51;
  v98 = v224;
  v224 = 0;
  v165[2] = (unsigned __int64)v98;
  if ( v97 )
  {
    j_j___libc_free_0(v97);
    v99 = v223;
    v100 = (unsigned __int64)v222;
    if ( v223 != v222 )
    {
      do
      {
        v101 = *(unsigned int *)(v100 + 24);
        v102 = *(_QWORD *)(v100 + 8);
        v100 += 32LL;
        sub_C7D6A0(v102, 16 * v101, 8);
      }
      while ( v99 != (_QWORD *)v100 );
      v100 = (unsigned __int64)v222;
    }
    if ( v100 )
      j_j___libc_free_0(v100);
  }
LABEL_123:
  if ( v219 )
    j_j___libc_free_0((unsigned __int64)v219);
  v20 = v216;
LABEL_126:
  if ( v20 )
    j_j___libc_free_0((unsigned __int64)v20);
  return v165;
}
