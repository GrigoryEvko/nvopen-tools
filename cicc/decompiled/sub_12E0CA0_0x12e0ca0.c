// Function: sub_12E0CA0
// Address: 0x12e0ca0
//
_QWORD *__fastcall sub_12E0CA0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // ecx
  __int64 *v6; // rsi
  __int64 v8; // rax
  __int64 *v9; // rax
  __int64 v10; // rdx
  __int64 *v11; // rbx
  __int64 *v12; // r12
  __m128i *v13; // rcx
  __m128i *v14; // rsi
  void *v15; // rdx
  __int64 v16; // rax
  __int64 *v17; // rax
  __int64 v18; // rdx
  __m128i *v19; // r12
  __m128i *v20; // r15
  __int64 v21; // rbx
  unsigned __int64 v22; // rax
  const __m128i *v23; // r12
  __int64 v24; // r15
  __int64 v25; // r13
  __int64 v26; // rax
  unsigned int v27; // ebx
  int v28; // eax
  __int64 v29; // rax
  int v30; // eax
  __m128i *v31; // r12
  __m128i *v32; // rbx
  __int64 v33; // rax
  int v34; // eax
  _BYTE *v35; // rsi
  int v36; // eax
  bool v37; // zf
  __int64 v38; // r8
  _QWORD *v39; // r12
  __int64 v40; // r15
  unsigned __int64 v41; // r12
  unsigned int v42; // eax
  unsigned __int64 v43; // rbx
  __int64 v44; // r13
  unsigned int v45; // r11d
  __int64 v46; // r8
  const __m128i *v47; // r10
  const void **v48; // rax
  int v49; // edx
  unsigned int v50; // r10d
  _QWORD *v51; // r11
  __m128i *v52; // rcx
  int v53; // eax
  int v54; // eax
  __int64 v55; // r13
  __m128i *v56; // rbx
  __int64 v57; // rbx
  int v58; // eax
  __int64 v59; // rax
  const __m128i *v60; // rcx
  const void *v61; // r13
  size_t v62; // rbx
  __int64 v63; // rdx
  __int64 v64; // r8
  __int64 v65; // r9
  _QWORD *v66; // rcx
  __int64 v67; // r14
  int v68; // r13d
  _QWORD *v69; // rbx
  int v70; // esi
  __m128i *v71; // rax
  int v72; // edx
  _QWORD *v73; // r12
  unsigned int v74; // eax
  void *v75; // rax
  __int64 v76; // rdx
  _BYTE *v77; // rsi
  unsigned int *v78; // rdx
  int v79; // eax
  const void *v80; // rdi
  int v81; // r11d
  size_t v82; // r9
  unsigned int v83; // ecx
  const void *v84; // rsi
  unsigned int v85; // ecx
  __int64 v86; // rax
  __int64 v87; // rdx
  __int64 v88; // r9
  unsigned int v89; // r8d
  _QWORD *v90; // rcx
  _QWORD *v91; // r14
  void *v92; // rdi
  __int64 *v93; // rax
  __int64 *v94; // rax
  int v95; // r13d
  char *v96; // rbx
  int v97; // eax
  const void *v98; // rdi
  size_t v99; // rdx
  __m128i *v100; // rcx
  int v101; // r8d
  unsigned int j; // r9d
  const void *v103; // rsi
  unsigned int v104; // r9d
  __int64 v105; // rdi
  _BYTE *v106; // rsi
  _QWORD *v107; // rax
  _BYTE *v108; // rsi
  bool v109; // dl
  __int64 v110; // r13
  int v111; // eax
  __int64 v112; // rax
  const void *v113; // rsi
  __int64 v114; // r13
  unsigned int v115; // eax
  __int64 v116; // rcx
  __int64 v117; // r9
  size_t v118; // rdx
  void **v119; // r11
  char *v120; // r8
  int v121; // esi
  __int64 v122; // rax
  __int64 v123; // rcx
  __int64 v124; // r9
  size_t v125; // rdx
  unsigned int v126; // r10d
  _QWORD *v127; // r11
  _QWORD *v128; // r8
  void *v129; // rdi
  _QWORD *v130; // rax
  void **v131; // rax
  int v132; // eax
  int v133; // r9d
  unsigned int v134; // r10d
  const void *v135; // rdi
  size_t v136; // rdx
  unsigned int i; // r11d
  __m128i *v138; // rax
  const void *v139; // rsi
  unsigned int v140; // r11d
  __int64 v141; // rax
  void *v142; // rax
  unsigned int v143; // eax
  __int64 v144; // rax
  void *v145; // rax
  int v146; // eax
  int v147; // ecx
  __m128i *v148; // r10
  unsigned int v149; // [rsp+8h] [rbp-168h]
  int v150; // [rsp+Ch] [rbp-164h]
  const __m128i *v151; // [rsp+10h] [rbp-160h]
  __int64 v152; // [rsp+18h] [rbp-158h]
  __m128i *v153; // [rsp+20h] [rbp-150h]
  size_t v154; // [rsp+20h] [rbp-150h]
  int n; // [rsp+28h] [rbp-148h]
  size_t nb; // [rsp+28h] [rbp-148h]
  unsigned int na; // [rsp+28h] [rbp-148h]
  size_t nc; // [rsp+28h] [rbp-148h]
  size_t nd; // [rsp+28h] [rbp-148h]
  int v160; // [rsp+30h] [rbp-140h]
  bool v161; // [rsp+30h] [rbp-140h]
  unsigned int v162; // [rsp+30h] [rbp-140h]
  unsigned int v163; // [rsp+30h] [rbp-140h]
  _QWORD *v164; // [rsp+30h] [rbp-140h]
  _QWORD *v165; // [rsp+30h] [rbp-140h]
  size_t v166; // [rsp+38h] [rbp-138h]
  char *v167; // [rsp+38h] [rbp-138h]
  _QWORD *v168; // [rsp+38h] [rbp-138h]
  _QWORD *v169; // [rsp+38h] [rbp-138h]
  int v170; // [rsp+38h] [rbp-138h]
  _QWORD *v171; // [rsp+38h] [rbp-138h]
  __int64 v172; // [rsp+40h] [rbp-130h]
  const __m128i *v173; // [rsp+40h] [rbp-130h]
  const void **v174; // [rsp+40h] [rbp-130h]
  _QWORD *v175; // [rsp+40h] [rbp-130h]
  size_t v176; // [rsp+40h] [rbp-130h]
  unsigned int v177; // [rsp+40h] [rbp-130h]
  size_t v178; // [rsp+40h] [rbp-130h]
  unsigned int v179; // [rsp+40h] [rbp-130h]
  __int64 v180; // [rsp+40h] [rbp-130h]
  __m128i *v181; // [rsp+40h] [rbp-130h]
  unsigned int v182; // [rsp+40h] [rbp-130h]
  unsigned int v184; // [rsp+58h] [rbp-118h]
  _QWORD *v185; // [rsp+58h] [rbp-118h]
  char *v186; // [rsp+58h] [rbp-118h]
  unsigned int v187; // [rsp+58h] [rbp-118h]
  _QWORD *v188; // [rsp+58h] [rbp-118h]
  __m128i *v190; // [rsp+60h] [rbp-110h]
  __int64 v191; // [rsp+60h] [rbp-110h]
  __int64 v192; // [rsp+60h] [rbp-110h]
  size_t v193; // [rsp+60h] [rbp-110h]
  char *v194; // [rsp+60h] [rbp-110h]
  unsigned __int64 v195; // [rsp+60h] [rbp-110h]
  __int64 v196; // [rsp+60h] [rbp-110h]
  size_t v197; // [rsp+60h] [rbp-110h]
  _QWORD *v198; // [rsp+60h] [rbp-110h]
  __m128i *v199; // [rsp+60h] [rbp-110h]
  size_t v200; // [rsp+60h] [rbp-110h]
  unsigned int v201; // [rsp+60h] [rbp-110h]
  bool v202; // [rsp+68h] [rbp-108h]
  int v203; // [rsp+68h] [rbp-108h]
  unsigned int v204; // [rsp+6Ch] [rbp-104h]
  unsigned __int64 v205; // [rsp+70h] [rbp-100h]
  __int64 v207; // [rsp+78h] [rbp-F8h]
  const __m128i *v208; // [rsp+80h] [rbp-F0h]
  __int64 v209; // [rsp+80h] [rbp-F0h]
  unsigned int v210; // [rsp+80h] [rbp-F0h]
  __int64 *v211; // [rsp+88h] [rbp-E8h]
  unsigned int v212; // [rsp+88h] [rbp-E8h]
  __m128i *v213; // [rsp+98h] [rbp-D8h] BYREF
  __m128i *v214; // [rsp+A0h] [rbp-D0h] BYREF
  const __m128i *v215; // [rsp+A8h] [rbp-C8h]
  __m128i *v216; // [rsp+B0h] [rbp-C0h]
  unsigned int *v217; // [rsp+C0h] [rbp-B0h] BYREF
  _BYTE *v218; // [rsp+C8h] [rbp-A8h]
  _BYTE *v219; // [rsp+D0h] [rbp-A0h]
  __int64 v220; // [rsp+E0h] [rbp-90h] BYREF
  _BYTE *v221; // [rsp+E8h] [rbp-88h]
  _BYTE *v222; // [rsp+F0h] [rbp-80h]
  __int64 v223; // [rsp+100h] [rbp-70h] BYREF
  _QWORD *v224; // [rsp+108h] [rbp-68h]
  _QWORD *v225; // [rsp+110h] [rbp-60h]
  void *src[2]; // [rsp+120h] [rbp-50h] BYREF
  __int64 v227; // [rsp+130h] [rbp-40h]
  __int64 v228; // [rsp+138h] [rbp-38h]

  v4 = *(_DWORD *)(a3 + 8);
  v214 = 0;
  v215 = 0;
  v216 = 0;
  if ( !v4 )
    goto LABEL_2;
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
  if ( v11 == v12 )
  {
LABEL_2:
    *a1 = 0;
    a1[1] = 0;
    a1[2] = 0;
    return a1;
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
      sub_12DD210((const __m128i **)&v214, v14, (const __m128i *)src);
      v14 = (__m128i *)v215;
    }
    else
    {
      if ( v14 )
      {
        *v14 = _mm_loadu_si128((const __m128i *)src);
        v14 = (__m128i *)v215;
      }
      v215 = ++v14;
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
    v13 = v216;
  }
  v19 = v214;
  v20 = v14;
  if ( v214 == v14 )
  {
    *a1 = 0;
    a1[1] = 0;
    a1[2] = 0;
    goto LABEL_131;
  }
  v21 = (char *)v14 - (char *)v214;
  _BitScanReverse64(&v22, v14 - v214);
  sub_12D57D0(v214, v14, 2LL * (int)(63 - (v22 ^ 0x3F)), a3);
  if ( v21 <= 256 )
  {
    sub_12D48A0(v19, v14, a3);
    goto LABEL_36;
  }
  sub_12D48A0(v19, v19 + 16, a3);
  v208 = v19 + 16;
  if ( &v19[16] == v14 )
    goto LABEL_36;
  do
  {
    v23 = v208;
    v24 = v208->m128i_i64[0];
    v25 = v208->m128i_i64[1];
    while ( 1 )
    {
      v211 = (__int64 *)v23;
      v30 = sub_16D1B30(a3, v24, v25);
      if ( v30 == -1 || (v26 = *(_QWORD *)a3 + 8LL * v30, v26 == *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8)) )
        v27 = 0;
      else
        v27 = *(_DWORD *)(*(_QWORD *)v26 + 8LL);
      v28 = sub_16D1B30(a3, v23[-1].m128i_i64[0], v23[-1].m128i_i64[1]);
      if ( v28 == -1 )
        break;
      v29 = *(_QWORD *)a3 + 8LL * v28;
      if ( v29 == *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8) )
        break;
      --v23;
      if ( *(_DWORD *)(*(_QWORD *)v29 + 8LL) >= v27 )
        goto LABEL_180;
LABEL_32:
      v23[1] = _mm_loadu_si128(v23);
    }
    --v23;
    if ( v27 )
      goto LABEL_32;
LABEL_180:
    ++v208;
    *v211 = v24;
    v211[1] = v25;
  }
  while ( v14 != v208 );
LABEL_36:
  v31 = v214;
  v32 = (__m128i *)v215;
  v217 = 0;
  v218 = 0;
  v219 = 0;
  if ( v214 == v215 )
  {
    v148 = (__m128i *)v215;
  }
  else
  {
    do
    {
      while ( 1 )
      {
        v36 = sub_16D1B30(a3, v31->m128i_i64[0], v31->m128i_i64[1]);
        if ( v36 == -1 || (v33 = *(_QWORD *)a3 + 8LL * v36, v33 == *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8)) )
          v34 = 0;
        else
          v34 = *(_DWORD *)(*(_QWORD *)v33 + 8LL);
        LODWORD(src[0]) = v34;
        v35 = v218;
        if ( v218 != v219 )
          break;
        ++v31;
        sub_C88AB0((__int64)&v217, v218, src);
        if ( v32 == v31 )
          goto LABEL_47;
      }
      if ( v218 )
      {
        *(_DWORD *)v218 = v34;
        v35 = v218;
      }
      ++v31;
      v218 = v35 + 4;
    }
    while ( v32 != v31 );
LABEL_47:
    v32 = (__m128i *)v215;
    v148 = v214;
  }
  v222 = 0;
  v220 = 0;
  v37 = *(_BYTE *)(a2 + 4064) == 0;
  v223 = 0;
  v221 = 0;
  v224 = 0;
  v225 = 0;
  if ( v37 )
  {
    v143 = *(_DWORD *)(a2 + 3984);
    if ( *v217 >= v143 )
      v143 = *v217;
    v204 = v143;
  }
  else
  {
    v204 = *((_DWORD *)v218 - 1);
  }
  v205 = v32 - v148;
  if ( !v205 )
  {
    *a1 = 0;
    a1[1] = 0;
    a1[2] = 0;
    goto LABEL_128;
  }
  v38 = 0;
  v39 = 0;
  v40 = a4;
  v212 = 0;
  v209 = 0;
  while ( 2 )
  {
    v207 = v209;
    v41 = ((__int64)v39 - v38) >> 5;
    if ( !v41 )
      goto LABEL_73;
    v42 = 0;
    v43 = 0;
    while ( *(_DWORD *)(v220 + 4 * v43) > v204 - v217[v209] )
    {
      v43 = ++v42;
      if ( v42 >= v41 )
        goto LABEL_73;
    }
    v44 = v38 + 32 * v43;
    v45 = *(_DWORD *)(v44 + 24);
    v46 = 16 * v209;
    v47 = &v148[v209];
    if ( !v45 )
    {
      ++*(_QWORD *)v44;
      goto LABEL_58;
    }
    v160 = *(_DWORD *)(v44 + 24);
    v173 = v47;
    v191 = *(_QWORD *)(v44 + 8);
    v79 = sub_16D3930(v47->m128i_i64[0], v47->m128i_i64[1]);
    v47 = v173;
    n = 1;
    v46 = 16 * v209;
    v174 = 0;
    v80 = (const void *)v47->m128i_i64[0];
    v81 = v160 - 1;
    v82 = v47->m128i_u64[1];
    v83 = (v160 - 1) & v79;
    v202 = v47->m128i_i64[0] == -2;
    v161 = v47->m128i_i64[0] == -1;
    while ( 2 )
    {
      v48 = (const void **)(v191 + 16LL * v83);
      v84 = *v48;
      if ( *v48 == (const void *)-1LL )
      {
        v109 = v161;
      }
      else
      {
        if ( v84 != (const void *)-2LL )
        {
          if ( (const void *)v82 == v48[1] )
          {
            v149 = v83;
            v150 = v81;
            v151 = v47;
            if ( !v82 )
              goto LABEL_136;
            v152 = v46;
            v154 = v82;
            v146 = memcmp(v80, v84, v82);
            v46 = v152;
            if ( !v146 )
              goto LABEL_136;
            v82 = v154;
            v47 = v151;
            v81 = v150;
            v83 = v149;
          }
          goto LABEL_109;
        }
        v109 = v202;
      }
      if ( v109 )
        goto LABEL_136;
      if ( v84 != (const void *)-1LL )
      {
        if ( v174 )
          v48 = v174;
        v174 = v48;
LABEL_109:
        v85 = n + v83;
        ++n;
        v83 = v81 & v85;
        continue;
      }
      break;
    }
    v45 = *(_DWORD *)(v44 + 24);
    if ( v174 )
      v48 = v174;
    v147 = *(_DWORD *)(v44 + 16);
    ++*(_QWORD *)v44;
    v49 = v147 + 1;
    if ( 4 * (v147 + 1) < 3 * v45 )
    {
      if ( v45 - (v49 + *(_DWORD *)(v44 + 20)) <= v45 >> 3 )
      {
        v180 = v46;
        v199 = (__m128i *)v47;
        sub_12E0A40(v44, v45);
        sub_12DDDE0(v44, v199, src);
        v48 = (const void **)src[0];
        v46 = v180;
        v47 = v199;
        v49 = *(_DWORD *)(v44 + 16) + 1;
      }
      goto LABEL_59;
    }
LABEL_58:
    v172 = v46;
    v190 = (__m128i *)v47;
    sub_12E0A40(v44, 2 * v45);
    sub_12DDDE0(v44, v190, src);
    v48 = (const void **)src[0];
    v47 = v190;
    v46 = v172;
    v49 = *(_DWORD *)(v44 + 16) + 1;
LABEL_59:
    *(_DWORD *)(v44 + 16) = v49;
    if ( *v48 != (const void *)-1LL )
      --*(_DWORD *)(v44 + 20);
    *(__m128i *)v48 = _mm_loadu_si128(v47);
LABEL_136:
    v192 = v46;
    v110 = *(_QWORD *)v40 + 8LL * *(unsigned int *)(v40 + 8);
    v111 = sub_16D1B30(
             v40,
             *(__int64 *)((char *)v214->m128i_i64 + v46),
             *(__int64 *)((char *)&v214->m128i_i64[1] + v46));
    if ( v111 == -1 )
      v112 = *(_QWORD *)v40 + 8LL * *(unsigned int *)(v40 + 8);
    else
      v112 = *(_QWORD *)v40 + 8LL * v111;
    if ( v112 == v110 )
      goto LABEL_71;
    v113 = *(const void **)((char *)v214->m128i_i64 + v192);
    v114 = v223 + 32 * v43;
    v193 = *(unsigned __int64 *)((char *)&v214->m128i_u64[1] + v192);
    v115 = sub_16D19C0(v40, v113, v193);
    v118 = v193;
    v119 = (void **)(*(_QWORD *)v40 + 8LL * v115);
    v120 = (char *)*v119;
    if ( *v119 )
    {
      if ( v120 != (char *)-8LL )
        goto LABEL_141;
      --*(_DWORD *)(v40 + 16);
    }
    nb = (size_t)v119;
    v162 = v115;
    v166 = v193;
    v176 = v193 + 25;
    v195 = v193 + 1;
    v122 = malloc(v118 + 25, v118 + 25, v118, v116, v120, v117);
    v125 = v166;
    v126 = v162;
    v127 = (_QWORD *)nb;
    v128 = (_QWORD *)v122;
    if ( !v122 )
    {
      if ( v176
        || (nc = v166,
            v168 = v127,
            v141 = malloc(1, 0, v125, v123, 0, v124),
            v126 = v162,
            v127 = v168,
            v128 = 0,
            v125 = nc,
            !v141) )
      {
        nd = v125;
        v165 = v128;
        v171 = v127;
        v182 = v126;
        sub_16BD1C0("Allocation failed");
        v126 = v182;
        v127 = v171;
        v128 = v165;
        v125 = nd;
        goto LABEL_148;
      }
      v129 = (void *)(v141 + 24);
      v128 = (_QWORD *)v141;
      goto LABEL_168;
    }
LABEL_148:
    v129 = v128 + 3;
    if ( v195 > 1 )
    {
LABEL_168:
      v164 = v128;
      v169 = v127;
      v179 = v126;
      v197 = v125;
      v142 = memcpy(v129, v113, v125);
      v128 = v164;
      v127 = v169;
      v126 = v179;
      v125 = v197;
      v129 = v142;
    }
    *((_BYTE *)v129 + v125) = 0;
    *v128 = v125;
    v128[1] = 0;
    v128[2] = 0;
    *v127 = v128;
    ++*(_DWORD *)(v40 + 12);
    v130 = (_QWORD *)(*(_QWORD *)v40 + 8LL * (unsigned int)sub_16D1CD0(v40, v126));
    v120 = (char *)*v130;
    if ( !*v130 || v120 == (char *)-8LL )
    {
      v131 = (void **)(v130 + 1);
      do
      {
        do
          v120 = (char *)*v131++;
        while ( v120 == (char *)-8LL );
      }
      while ( !v120 );
      v50 = *(_DWORD *)(v114 + 24);
      if ( v50 )
        goto LABEL_155;
LABEL_142:
      ++*(_QWORD *)v114;
      v51 = v120 + 8;
LABEL_143:
      v185 = v51;
      v121 = 2 * v50;
      v194 = v120;
LABEL_144:
      sub_12E0A40(v114, v121);
      sub_12DDDE0(v114, v185, src);
      v52 = (__m128i *)src[0];
      v120 = v194;
      v54 = *(_DWORD *)(v114 + 16) + 1;
      goto LABEL_68;
    }
LABEL_141:
    v50 = *(_DWORD *)(v114 + 24);
    if ( !v50 )
      goto LABEL_142;
LABEL_155:
    v177 = v50;
    v186 = v120;
    v196 = *(_QWORD *)(v114 + 8);
    v132 = sub_16D3930(*((_QWORD *)v120 + 1), *((_QWORD *)v120 + 2));
    v120 = v186;
    v52 = 0;
    v133 = 1;
    v134 = v177 - 1;
    v135 = (const void *)*((_QWORD *)v186 + 1);
    v136 = *((_QWORD *)v186 + 2);
    for ( i = (v177 - 1) & v132; ; i = v134 & v140 )
    {
      v138 = (__m128i *)(v196 + 16LL * i);
      v139 = (const void *)v138->m128i_i64[0];
      if ( v138->m128i_i64[0] == -1 )
        break;
      if ( v139 == (const void *)-2LL )
      {
        if ( v135 == (const void *)-2LL )
          goto LABEL_71;
        if ( !v52 )
          v52 = (__m128i *)(v196 + 16LL * i);
      }
      else if ( v138->m128i_i64[1] == v136 )
      {
        v153 = v52;
        v203 = v133;
        na = i;
        v163 = v134;
        v167 = v120;
        if ( !v136 )
          goto LABEL_71;
        v178 = v136;
        if ( !memcmp(v135, v139, v136) )
          goto LABEL_71;
        v136 = v178;
        v120 = v167;
        v134 = v163;
        i = na;
        v133 = v203;
        v52 = v153;
      }
      v140 = v133 + i;
      ++v133;
    }
    if ( v135 != (const void *)-1LL )
    {
      v50 = *(_DWORD *)(v114 + 24);
      v51 = v120 + 8;
      if ( !v52 )
        v52 = v138;
      v53 = *(_DWORD *)(v114 + 16);
      ++*(_QWORD *)v114;
      v54 = v53 + 1;
      if ( 4 * v54 >= 3 * v50 )
        goto LABEL_143;
      if ( v50 - (v54 + *(_DWORD *)(v114 + 20)) <= v50 >> 3 )
      {
        v185 = v120 + 8;
        v121 = v50;
        v194 = v120;
        goto LABEL_144;
      }
LABEL_68:
      *(_DWORD *)(v114 + 16) = v54;
      if ( v52->m128i_i64[0] != -1 )
        --*(_DWORD *)(v114 + 20);
      *v52 = _mm_loadu_si128((const __m128i *)(v120 + 8));
    }
LABEL_71:
    *(_DWORD *)(v220 + 4 * v43) += v217[v209];
    if ( v41 > v43 )
      goto LABEL_103;
    v148 = v214;
LABEL_73:
    src[0] = (void *)1;
    v55 = v209;
    src[1] = 0;
    v56 = &v148[v209];
    v227 = 0;
    v228 = 0;
    sub_12E0A40((__int64)src, 0);
    sub_12DDDE0((__int64)src, v56, &v213);
    LODWORD(v227) = v227 + 1;
    if ( v213->m128i_i64[0] != -1 )
      --HIDWORD(v227);
    *v213 = _mm_loadu_si128(v56);
    v57 = *(_QWORD *)v40 + 8LL * *(unsigned int *)(v40 + 8);
    v58 = sub_16D1B30(v40, v214[v209].m128i_i64[0], v214[v209].m128i_i64[1]);
    if ( v58 == -1 )
      v59 = *(_QWORD *)v40 + 8LL * *(unsigned int *)(v40 + 8);
    else
      v59 = *(_QWORD *)v40 + 8LL * v58;
    if ( v59 == v57 )
      goto LABEL_92;
    v60 = &v214[v55];
    v61 = (const void *)v214[v55].m128i_i64[0];
    v62 = v60->m128i_u64[1];
    v64 = (unsigned int)sub_16D19C0(v40, v61, v62);
    v66 = (_QWORD *)(*(_QWORD *)v40 + 8 * v64);
    v67 = *v66;
    if ( *v66 )
    {
      if ( v67 != -8 )
        goto LABEL_80;
      --*(_DWORD *)(v40 + 16);
    }
    v175 = v66;
    v184 = v64;
    v86 = malloc(v62 + 25, v62 + 25, v63, v66, v64, v65);
    v89 = v184;
    v90 = v175;
    v91 = (_QWORD *)v86;
    if ( !v86 )
    {
      if ( v62 != -25 || (v144 = malloc(1, 0, v87, v175, v184, v88), v89 = v184, v90 = v175, !v144) )
      {
        v188 = v90;
        v201 = v89;
        sub_16BD1C0("Allocation failed");
        v89 = v201;
        v90 = v188;
        goto LABEL_114;
      }
      v92 = (void *)(v144 + 24);
      v91 = (_QWORD *)v144;
      goto LABEL_184;
    }
LABEL_114:
    v92 = v91 + 3;
    if ( v62 + 1 > 1 )
    {
LABEL_184:
      v198 = v90;
      v210 = v89;
      v145 = memcpy(v92, v61, v62);
      v90 = v198;
      v89 = v210;
      v92 = v145;
    }
    *((_BYTE *)v92 + v62) = 0;
    *v91 = v62;
    v91[1] = 0;
    v91[2] = 0;
    *v90 = v91;
    ++*(_DWORD *)(v40 + 12);
    v93 = (__int64 *)(*(_QWORD *)v40 + 8LL * (unsigned int)sub_16D1CD0(v40, v89));
    v67 = *v93;
    if ( *v93 == -8 || !v67 )
    {
      v94 = v93 + 1;
      do
      {
        do
          v67 = *v94++;
        while ( !v67 );
      }
      while ( v67 == -8 );
      v68 = v228;
      if ( (_DWORD)v228 )
        goto LABEL_121;
LABEL_81:
      ++src[0];
      v69 = (_QWORD *)(v67 + 8);
LABEL_82:
      v70 = 2 * v68;
LABEL_83:
      sub_12E0A40((__int64)src, v70);
      sub_12DDDE0((__int64)src, v69, &v213);
      v71 = v213;
      v72 = v227 + 1;
      goto LABEL_89;
    }
LABEL_80:
    v68 = v228;
    if ( !(_DWORD)v228 )
      goto LABEL_81;
LABEL_121:
    v95 = v68 - 1;
    v96 = (char *)src[1];
    v97 = sub_16D3930(*(_QWORD *)(v67 + 8), *(_QWORD *)(v67 + 16));
    v98 = *(const void **)(v67 + 8);
    v99 = *(_QWORD *)(v67 + 16);
    v100 = 0;
    v101 = 1;
    for ( j = v95 & v97; ; j = v95 & v104 )
    {
      v71 = (__m128i *)&v96[16 * j];
      v103 = (const void *)v71->m128i_i64[0];
      if ( v71->m128i_i64[0] == -1 )
        break;
      if ( v103 == (const void *)-2LL )
      {
        if ( v98 == (const void *)-2LL )
          goto LABEL_92;
        if ( !v100 )
          v100 = (__m128i *)&v96[16 * j];
      }
      else if ( v99 == v71->m128i_i64[1] )
      {
        v170 = v101;
        v181 = v100;
        v187 = j;
        if ( !v99 )
          goto LABEL_92;
        v200 = v99;
        if ( !memcmp(v98, v103, v99) )
          goto LABEL_92;
        v99 = v200;
        j = v187;
        v100 = v181;
        v101 = v170;
      }
      v104 = v101 + j;
      ++v101;
    }
    if ( v98 != (const void *)-1LL )
    {
      v68 = v228;
      v69 = (_QWORD *)(v67 + 8);
      if ( v100 )
        v71 = v100;
      ++src[0];
      v72 = v227 + 1;
      if ( 4 * ((int)v227 + 1) >= (unsigned int)(3 * v228) )
        goto LABEL_82;
      if ( (int)v228 - (v72 + HIDWORD(v227)) <= (unsigned int)v228 >> 3 )
      {
        v70 = v228;
        goto LABEL_83;
      }
LABEL_89:
      LODWORD(v227) = v72;
      if ( v71->m128i_i64[0] != -1 )
        --HIDWORD(v227);
      *v71 = _mm_loadu_si128((const __m128i *)(v67 + 8));
    }
LABEL_92:
    v73 = v224;
    if ( v224 == v225 )
    {
      sub_12DD390(&v223, (__int64)v224, (__int64)src);
    }
    else
    {
      if ( v224 )
      {
        *v224 = 0;
        v73[1] = 0;
        v73[2] = 0;
        *((_DWORD *)v73 + 6) = 0;
        j___libc_free_0(0);
        v74 = v228;
        *((_DWORD *)v73 + 6) = v228;
        if ( v74 )
        {
          v75 = (void *)sub_22077B0(16LL * v74);
          v76 = *((unsigned int *)v73 + 6);
          v73[1] = v75;
          v73[2] = v227;
          memcpy(v75, src[1], 16 * v76);
        }
        else
        {
          v73[1] = 0;
          v73[2] = 0;
        }
        v73 = v224;
      }
      v224 = v73 + 4;
    }
    v77 = v221;
    v78 = &v217[v207];
    if ( v221 == v222 )
    {
      sub_12DD6D0((__int64)&v220, v221, v78);
    }
    else
    {
      if ( v221 )
      {
        *(_DWORD *)v221 = *v78;
        v77 = v221;
      }
      v221 = v77 + 4;
    }
    j___libc_free_0(src[1]);
LABEL_103:
    ++v212;
    v38 = v223;
    v39 = v224;
    v209 = v212;
    if ( v212 < v205 )
    {
      v148 = v214;
      continue;
    }
    break;
  }
  v105 = v220;
  v106 = v222;
  v107 = v225;
  *a1 = v223;
  a1[1] = v39;
  v108 = &v106[-v105];
  a1[2] = v107;
  if ( v105 )
    j_j___libc_free_0(v105, v108);
LABEL_128:
  if ( v217 )
    j_j___libc_free_0(v217, v219 - (_BYTE *)v217);
  v20 = v214;
LABEL_131:
  if ( v20 )
    j_j___libc_free_0(v20, (char *)v216 - (char *)v20);
  return a1;
}
