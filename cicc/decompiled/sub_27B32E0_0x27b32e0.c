// Function: sub_27B32E0
// Address: 0x27b32e0
//
__int64 __fastcall sub_27B32E0(__int64 a1, __int64 a2, __int64 a3, int *a4, int *a5, __int64 a6, __int64 a7)
{
  __int64 v7; // r14
  unsigned __int8 **v8; // rdi
  __int64 v9; // r13
  unsigned __int8 **v10; // rbx
  unsigned __int8 **v12; // r12
  int v13; // r11d
  unsigned int v14; // ecx
  int *v15; // rdi
  int *v16; // rdx
  int v17; // r8d
  int v18; // eax
  int v19; // esi
  int v20; // edi
  _DWORD *v21; // rdx
  __int64 v22; // rdi
  __int64 v23; // r8
  int v24; // esi
  int *v25; // rcx
  int *v26; // r9
  int v27; // eax
  int v28; // r11d
  unsigned int v29; // edx
  __int64 v30; // r9
  _DWORD *v31; // rcx
  int v32; // r10d
  int *v34; // rdx
  int *v35; // rax
  __int64 *v36; // r15
  void **v37; // rsi
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r12
  unsigned int v41; // edx
  void **v42; // rax
  void *v43; // r10
  __int64 v44; // rax
  unsigned __int64 v45; // rdx
  __int64 *v46; // rdi
  __int64 v47; // r12
  __int64 *v48; // r11
  __int64 *v49; // rax
  unsigned int v50; // ecx
  unsigned __int8 v51; // dl
  int v52; // eax
  __int64 v53; // rdx
  __int64 v54; // rax
  __m128i v55; // rcx
  __int64 *v56; // rdx
  __m128i v57; // xmm1
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // r9
  char *v62; // r12
  _QWORD *v63; // rdi
  _QWORD *v64; // rsi
  bool v65; // al
  const void *v66; // rsi
  const void *v67; // r15
  __int64 v68; // rax
  __int64 v69; // rsi
  __int64 v70; // rdx
  __int64 *v71; // rdi
  int v72; // edx
  unsigned int v73; // eax
  __int64 *v74; // r10
  __int64 v75; // r11
  int v76; // edx
  int v77; // r10d
  __int64 v78; // rax
  __int64 v79; // r12
  __int64 *v80; // rax
  __int64 v81; // rdx
  __int64 v82; // rcx
  __int64 v83; // r8
  __int64 v84; // r9
  __int64 *v85; // rax
  __int64 v86; // rdx
  __int64 v87; // rcx
  __int64 v88; // r8
  __int64 v89; // r9
  __int64 v90; // r8
  __int64 v91; // r9
  __int64 v92; // rcx
  __int64 v93; // rdx
  __int64 v94; // r13
  __int64 v95; // rbx
  __int64 v96; // r15
  __int64 v97; // r14
  _QWORD *v98; // r12
  _QWORD *v99; // rdi
  __int64 v100; // rdx
  __int64 v101; // rcx
  __int64 v102; // r8
  __int64 v103; // r9
  __int64 v104; // rcx
  __int64 v105; // r15
  __int64 v106; // rax
  __int64 *v107; // rbx
  __int64 v108; // r13
  __int64 p_dest; // rsi
  __int64 v110; // rdx
  __int64 v111; // rcx
  __int64 v112; // r8
  __int64 v113; // r9
  __int64 v114; // rbx
  __int64 v115; // r8
  __int64 v116; // r9
  __int64 *v117; // rax
  __int64 v118; // rdx
  __int64 v119; // rcx
  __int64 v120; // r8
  __int64 v121; // r9
  __int64 v122; // rdx
  __int64 v123; // rcx
  __int64 v124; // r8
  __int64 v125; // r9
  __int64 v126; // rdx
  __int64 v127; // rcx
  __int64 v128; // r8
  __int64 v129; // r9
  unsigned int v130; // eax
  __int64 v131; // rdx
  __int64 v132; // rcx
  __int64 *v133; // rdx
  __int64 v134; // rbx
  __m128i v135; // xmm3
  __int64 v136; // rax
  __int64 *v137; // r15
  __int64 v138; // r14
  char v139; // di
  __int64 *v140; // rax
  __int64 *v141; // rbx
  __int64 *v142; // r12
  __int64 v143; // rsi
  _QWORD *v144; // rax
  _QWORD *v145; // rdx
  __int64 *v146; // rbx
  __int64 v147; // rax
  __int64 v148; // r15
  __int64 *v149; // rdi
  __int64 v150; // rdx
  __int64 v151; // rax
  __int64 v152; // rsi
  __int64 *v153; // r12
  __int64 *v154; // rbx
  __int64 v155; // r12
  __int64 v156; // rcx
  __int64 *v157; // r15
  void **v158; // rbx
  __m128i *v159; // r14
  __int64 v160; // r13
  __m128i *v161; // rsi
  __int64 v162; // rax
  __int64 v163; // r14
  __int64 *v164; // rbx
  __int64 v165; // r12
  __int64 v166; // rdx
  __int64 v167; // rdx
  __int64 v168; // r15
  __int64 v169; // r9
  __int64 v170; // r8
  unsigned __int64 *v171; // rax
  unsigned __int64 *v172; // r12
  unsigned __int64 *v173; // rbx
  unsigned __int64 v174; // rdi
  __int64 v175; // rax
  __int64 v176; // rax
  __int64 v177; // rdx
  __int64 v178; // rcx
  __int64 v179; // r8
  __int64 v180; // r9
  char *v181; // rcx
  __int64 v182; // r8
  char *v183; // r9
  __int64 v184; // rdx
  __int64 v185; // r8
  __int64 v186; // rdx
  char *v187; // rax
  __int64 v188; // rcx
  int v189; // eax
  __int64 v190; // rdx
  unsigned __int64 v191; // rax
  unsigned __int64 v192; // rax
  unsigned int v193; // esi
  __int64 *v194; // r12
  __m128i *v195; // r15
  void **v196; // r14
  __int64 *v197; // rbx
  __int64 v198; // rsi
  __int64 v199; // rcx
  __int64 v200; // r8
  __int64 v201; // r9
  int v202; // eax
  __int64 v203; // rdx
  int v204; // eax
  unsigned __int64 v205; // rax
  const void *v206; // r14
  size_t v207; // r13
  int v208; // r12d
  int v209; // r12d
  char *v210; // rdi
  __int64 v211; // rdx
  int v212; // r15d
  char *v213; // rdi
  signed __int64 v214; // rax
  int v215; // [rsp+Ch] [rbp-224h]
  unsigned __int8 *v216; // [rsp+18h] [rbp-218h]
  int v217; // [rsp+30h] [rbp-200h]
  __int64 v219; // [rsp+48h] [rbp-1E8h]
  __int64 v220; // [rsp+48h] [rbp-1E8h]
  char v222; // [rsp+58h] [rbp-1D8h]
  unsigned int v223; // [rsp+58h] [rbp-1D8h]
  __int64 *v224; // [rsp+60h] [rbp-1D0h]
  __int64 v225; // [rsp+60h] [rbp-1D0h]
  int v226; // [rsp+68h] [rbp-1C8h]
  __int64 v228; // [rsp+70h] [rbp-1C0h]
  __int64 v229; // [rsp+70h] [rbp-1C0h]
  unsigned __int8 **v231; // [rsp+80h] [rbp-1B0h]
  __int64 v232; // [rsp+80h] [rbp-1B0h]
  __int64 v233; // [rsp+80h] [rbp-1B0h]
  __int64 *v234; // [rsp+80h] [rbp-1B0h]
  int v236; // [rsp+9Ch] [rbp-194h] BYREF
  __int64 v237; // [rsp+A0h] [rbp-190h] BYREF
  int *v238; // [rsp+A8h] [rbp-188h]
  __int64 v239; // [rsp+B0h] [rbp-180h]
  unsigned int v240; // [rsp+B8h] [rbp-178h]
  __int64 v241; // [rsp+C0h] [rbp-170h] BYREF
  __int64 v242; // [rsp+C8h] [rbp-168h]
  __int64 v243; // [rsp+D0h] [rbp-160h]
  __int64 v244; // [rsp+D8h] [rbp-158h]
  __m128i v245; // [rsp+E0h] [rbp-150h] BYREF
  __m128i v246; // [rsp+F0h] [rbp-140h]
  __int64 *v247; // [rsp+110h] [rbp-120h] BYREF
  __int64 v248; // [rsp+118h] [rbp-118h]
  _BYTE v249[32]; // [rsp+120h] [rbp-110h] BYREF
  __int64 *v250; // [rsp+140h] [rbp-F0h] BYREF
  __int64 v251; // [rsp+148h] [rbp-E8h]
  _BYTE v252[32]; // [rsp+150h] [rbp-E0h] BYREF
  _BYTE *v253; // [rsp+170h] [rbp-C0h] BYREF
  __int64 v254; // [rsp+178h] [rbp-B8h]
  _BYTE v255[32]; // [rsp+180h] [rbp-B0h] BYREF
  void *s2[2]; // [rsp+1A0h] [rbp-90h] BYREF
  __m128i v257; // [rsp+1B0h] [rbp-80h] BYREF
  __int64 v258; // [rsp+1C0h] [rbp-70h]
  char v259[8]; // [rsp+1C8h] [rbp-68h] BYREF
  void *dest; // [rsp+1D0h] [rbp-60h] BYREF
  __int64 v261; // [rsp+1D8h] [rbp-58h]
  _BYTE v262[80]; // [rsp+1E0h] [rbp-50h] BYREF
  __int64 v263; // [rsp+240h] [rbp+10h]
  __int64 v264; // [rsp+240h] [rbp+10h]

  v7 = a3;
  v8 = *(unsigned __int8 ***)(a3 + 96);
  v9 = a7;
  v10 = &v8[*(unsigned int *)(a3 + 104)];
  v12 = v8;
  v231 = v8;
  v237 = 0;
  v238 = 0;
  v239 = 0;
  v240 = 0;
  if ( v8 == v10 )
  {
    v168 = 0;
    v236 = MEMORY[0];
LABEL_170:
    v24 = 0;
    s2[0] = 0;
    v237 = v168 + 1;
LABEL_171:
    v24 *= 2;
    goto LABEL_172;
  }
  do
  {
    while ( 1 )
    {
      v18 = sub_27B2E30(a2, *v12);
      LODWORD(v250) = v18;
      if ( v18 == -1 )
      {
        v22 = (__int64)v238;
        *(_BYTE *)(a1 + 72) = 0;
        goto LABEL_30;
      }
      v19 = v240;
      if ( !v240 )
      {
        ++v237;
        s2[0] = 0;
LABEL_8:
        v19 = 2 * v240;
LABEL_9:
        sub_A09770((__int64)&v237, v19);
        sub_A1A0F0((__int64)&v237, (int *)&v250, s2);
        v18 = (int)v250;
        v16 = (int *)s2[0];
        v20 = v239 + 1;
        goto LABEL_20;
      }
      v13 = 1;
      v14 = (v240 - 1) & (37 * v18);
      v15 = &v238[2 * v14];
      v16 = 0;
      v17 = *v15;
      if ( v18 != *v15 )
        break;
LABEL_4:
      ++v12;
      ++v15[1];
      if ( v10 == v12 )
        goto LABEL_23;
    }
    while ( v17 != -1 )
    {
      if ( !v16 && v17 == -2 )
        v16 = v15;
      v14 = (v240 - 1) & (v13 + v14);
      v15 = &v238[2 * v14];
      v17 = *v15;
      if ( v18 == *v15 )
        goto LABEL_4;
      ++v13;
    }
    if ( !v16 )
      v16 = v15;
    ++v237;
    v20 = v239 + 1;
    s2[0] = v16;
    if ( 4 * ((int)v239 + 1) >= 3 * v240 )
      goto LABEL_8;
    if ( v240 - HIDWORD(v239) - v20 <= v240 >> 3 )
      goto LABEL_9;
LABEL_20:
    LODWORD(v239) = v20;
    if ( *v16 != -1 )
      --HIDWORD(v239);
    *v16 = v18;
    ++v12;
    v21 = v16 + 1;
    *v21 = 0;
    *v21 = 1;
  }
  while ( v10 != v12 );
LABEL_23:
  v22 = (__int64)v238;
  v23 = (unsigned int)v239;
  v168 = v237;
  v24 = v240;
  v25 = &v238[2 * v240];
  if ( !(_DWORD)v239 || v238 == v25 )
    goto LABEL_24;
  v26 = v238;
  while ( (unsigned int)*v26 > 0xFFFFFFFD )
  {
    v26 += 2;
    if ( v25 == v26 )
      goto LABEL_24;
  }
  if ( v25 == v26 )
  {
LABEL_24:
    v26 = &v238[2 * v240];
  }
  else
  {
LABEL_36:
    v34 = v26;
    while ( 1 )
    {
      v35 = v34 + 2;
      if ( v25 == v34 + 2 )
        break;
      for ( v34 += 2; (unsigned int)*v35 > 0xFFFFFFFD; v34 = v35 )
      {
        v35 += 2;
        if ( v25 == v35 )
          goto LABEL_25;
      }
      if ( v25 == v35 )
        break;
      if ( v26[1] < (unsigned int)v35[1] )
      {
        v26 = v35;
        goto LABEL_36;
      }
    }
  }
LABEL_25:
  v27 = *v26;
  v236 = *v26;
  if ( !v240 )
    goto LABEL_170;
  v28 = 1;
  v29 = (v240 - 1) & (37 * v27);
  v30 = (__int64)&v238[2 * v29];
  v31 = 0;
  v32 = *(_DWORD *)v30;
  if ( v27 == *(_DWORD *)v30 )
  {
LABEL_27:
    if ( *(_DWORD *)(v30 + 4) == 1 )
    {
      *(_BYTE *)(a1 + 72) = 0;
      goto LABEL_30;
    }
    v226 = *(_DWORD *)(v7 + 40);
    v247 = (__int64 *)v249;
    v248 = 0x400000000LL;
    goto LABEL_45;
  }
  while ( v32 != -1 )
  {
    if ( !v31 && v32 == -2 )
      v31 = (_DWORD *)v30;
    v29 = (v240 - 1) & (v28 + v29);
    v30 = (__int64)&v238[2 * v29];
    v32 = *(_DWORD *)v30;
    if ( v27 == *(_DWORD *)v30 )
      goto LABEL_27;
    ++v28;
  }
  if ( !v31 )
    v31 = (_DWORD *)v30;
  v23 = (unsigned int)(v239 + 1);
  ++v237;
  s2[0] = v31;
  if ( 4 * (int)v23 >= 3 * v240 )
    goto LABEL_171;
  v38 = v240 - HIDWORD(v239) - (unsigned int)v23;
  if ( (unsigned int)v38 <= v240 >> 3 )
  {
LABEL_172:
    sub_A09770((__int64)&v237, v24);
    sub_A1A0F0((__int64)&v237, &v236, s2);
    v27 = v236;
    v31 = s2[0];
  }
  LODWORD(v239) = v239 + 1;
  if ( *v31 != -1 )
    --HIDWORD(v239);
  *v31 = v27;
  v31[1] = 0;
  v226 = *(_DWORD *)(v7 + 40);
  v247 = (__int64 *)v249;
  v248 = 0x400000000LL;
  if ( v231 == v10 )
  {
    v222 = 0;
    v47 = (unsigned int)v248;
    v46 = (__int64 *)v249;
    goto LABEL_106;
  }
LABEL_45:
  v36 = (__int64 *)v231;
  while ( 2 )
  {
    while ( 2 )
    {
      v39 = *(unsigned int *)(a2 + 24);
      v40 = *v36;
      v37 = *(void ***)(a2 + 8);
      if ( (_DWORD)v39 )
      {
        v41 = (v39 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
        v42 = &v37[2 * v41];
        v43 = *v42;
        if ( (void *)v40 == *v42 )
          goto LABEL_49;
        v52 = 1;
        while ( v43 != (void *)-4096LL )
        {
          v23 = (unsigned int)(v52 + 1);
          v175 = ((_DWORD)v39 - 1) & (v41 + v52);
          v41 = v175;
          v42 = &v37[2 * v175];
          v43 = *v42;
          if ( (void *)v40 == *v42 )
            goto LABEL_49;
          v52 = v23;
        }
      }
      v42 = &v37[2 * v39];
LABEL_49:
      if ( v236 != *((_DWORD *)v42 + 2) )
      {
        v37 = s2;
        ++v36;
        s2[0] = *(void **)(v40 + 40);
        sub_27AC510(v7, (__int64 *)s2);
        if ( v10 == (unsigned __int8 **)v36 )
          goto LABEL_53;
        continue;
      }
      break;
    }
    v44 = (unsigned int)v248;
    v45 = (unsigned int)v248 + 1LL;
    if ( v45 > HIDWORD(v248) )
    {
      v37 = (void **)v249;
      sub_C8D5F0((__int64)&v247, v249, v45, 8u, v23, v30);
      v44 = (unsigned int)v248;
    }
    v38 = (__int64)v247;
    ++v36;
    v247[v44] = v40;
    LODWORD(v248) = v248 + 1;
    if ( v10 != (unsigned __int8 **)v36 )
      continue;
    break;
  }
LABEL_53:
  v46 = v247;
  v47 = (unsigned int)v248;
  v9 = a7;
  v23 = (__int64)&v247[(unsigned int)v248];
  v48 = v247;
  if ( v247 != (__int64 *)v23 )
  {
    v49 = v247;
    v30 = 0x100060000000001LL;
    do
    {
      v37 = (void **)*v49;
      v51 = *(_BYTE *)*v49;
      if ( v51 == 84
        || (v50 = v51 - 39, v50 <= 0x38) && ((1LL << v50) & 0x100060000000001LL) != 0
        || v51 == 60
        || (v38 = (__int64)v37[1], *(_BYTE *)(v38 + 8) == 11) )
      {
        *(_BYTE *)(a1 + 72) = 0;
        goto LABEL_62;
      }
      ++v49;
    }
    while ( (__int64 *)v23 != v49 );
  }
  if ( v226 == *(_DWORD *)(v7 + 40) )
  {
    v222 = 0;
    goto LABEL_106;
  }
  v241 = 0;
  v242 = 0;
  v53 = *(unsigned int *)(a6 + 24);
  v55.m128i_i64[0] = *(_QWORD *)(a6 + 8);
  v243 = 0;
  v244 = 0;
  v54 = v53;
  v55.m128i_i64[1] = v55.m128i_i64[0] + 96 * v53;
  v56 = *(__int64 **)a6;
  v224 = (__int64 *)v55.m128i_i64[1];
  if ( *(_DWORD *)(a6 + 16) )
  {
    s2[1] = *(void **)a6;
    v257 = v55;
    s2[0] = (void *)a6;
    sub_27ADFC0((__int64)s2, (__int64)v37, (__int64)v56, v55.m128i_i64[0], v23, v30);
    v54 = *(unsigned int *)(a6 + 24);
    v56 = (__int64 *)(*(_QWORD *)(a6 + 8) + 96 * v54);
    v224 = v56;
  }
  else
  {
    s2[0] = (void *)a6;
    s2[1] = v56;
    v257.m128i_i64[0] = v55.m128i_i64[1];
    v257.m128i_i64[1] = v55.m128i_i64[1];
  }
  v57 = _mm_loadu_si128(&v257);
  v55.m128i_i64[1] = v257.m128i_i64[0];
  v245 = _mm_loadu_si128((const __m128i *)s2);
  v246 = v57;
  if ( v224 == (__int64 *)v257.m128i_i64[0] )
    goto LABEL_97;
  while ( 2 )
  {
    s2[0] = &v257;
    s2[1] = (void *)0x400000000LL;
    if ( *(_DWORD *)(v55.m128i_i64[1] + 8) )
      sub_27ABF90((__int64)s2, v55.m128i_i64[1], (__int64)v56, v55.m128i_i64[0], v23, v30);
    dest = v262;
    v261 = 0x400000000LL;
    if ( *(_DWORD *)(v55.m128i_i64[1] + 56) )
    {
      sub_27AC1D0((__int64)&dest, v55.m128i_i64[1] + 48, (__int64)v56, v55.m128i_i64[0], v23, v30);
      v30 = (__int64)dest;
      v62 = (char *)s2[0];
      v23 = (__int64)dest + 8 * (unsigned int)v261;
      v55.m128i_i64[1] = (__int64)dest;
      while ( v23 != v55.m128i_i64[1] )
      {
        if ( *(_DWORD *)(v7 + 16) )
        {
          v69 = *(_QWORD *)(v7 + 8);
          v70 = *(unsigned int *)(v7 + 24);
          v55.m128i_i64[0] = *(_QWORD *)v55.m128i_i64[1];
          v71 = (__int64 *)(v69 + 8 * v70);
          if ( !(_DWORD)v70 )
          {
            v66 = (const void *)(v55.m128i_i64[1] + 8);
            v67 = v62 + 8;
LABEL_88:
            if ( (const void *)v23 != v66 )
              memmove((void *)v55.m128i_i64[1], v66, v23 - (_QWORD)v66);
            v68 = (unsigned int)(v261 - 1);
            v76 = (int)s2[1];
            v55.m128i_i64[0] = (__int64)s2[0] + 8 * LODWORD(s2[1]);
            LODWORD(v261) = v261 - 1;
            if ( (const void *)v55.m128i_i64[0] != v67 )
            {
              memmove(v62, v67, v55.m128i_i64[0] - (_QWORD)v67);
              v76 = (int)s2[1];
              v68 = (unsigned int)v261;
            }
            v30 = (__int64)dest;
            LODWORD(s2[1]) = v76 - 1;
            goto LABEL_82;
          }
          v72 = v70 - 1;
          v73 = v72 & (((unsigned __int32)v55.m128i_i32[0] >> 9) ^ ((unsigned __int32)v55.m128i_i32[0] >> 4));
          v74 = (__int64 *)(v69 + 8LL * v73);
          v75 = *v74;
          if ( v55.m128i_i64[0] == *v74 )
          {
LABEL_86:
            v65 = v71 != v74;
          }
          else
          {
            v77 = 1;
            while ( v75 != -4096 )
            {
              v212 = v77 + 1;
              v73 = v72 & (v73 + v77);
              v74 = (__int64 *)(v69 + 8LL * v73);
              v75 = *v74;
              if ( v55.m128i_i64[0] == *v74 )
                goto LABEL_86;
              v77 = v212;
            }
            v65 = 0;
          }
        }
        else
        {
          v63 = *(_QWORD **)(v7 + 32);
          v64 = &v63[*(unsigned int *)(v7 + 40)];
          v65 = v64 != sub_27ABE10(v63, (__int64)v64, (__int64 *)v55.m128i_i64[1]);
        }
        v66 = (const void *)(v55.m128i_i64[1] + 8);
        v67 = v62 + 8;
        if ( !v65 )
          goto LABEL_88;
        v68 = (unsigned int)v261;
        v62 += 8;
        v55.m128i_i64[1] += 8;
LABEL_82:
        v23 = v30 + 8 * v68;
      }
    }
    sub_27B1110((__int64)&v250, (__int64)&v241, (const void **)s2, v55.m128i_i64[0], v23, v30);
    sub_27ABC80(s2);
    v246.m128i_i64[0] += 96;
    sub_27ADFC0((__int64)&v245, (__int64)&v241, v58, v59, v60, v61);
    v55.m128i_i64[1] = v246.m128i_i64[0];
    if ( (__int64 *)v246.m128i_i64[0] != v224 )
      continue;
    break;
  }
  v9 = a7;
  v54 = *(unsigned int *)(a6 + 24);
LABEL_97:
  if ( (_DWORD)v54 )
  {
    if ( !byte_4FFC580 && (unsigned int)sub_2207590((__int64)&byte_4FFC580) )
    {
      qword_4FFC5B0 = 0;
      qword_4FFC5A0 = (__int64)&qword_4FFC5B0;
      qword_4FFC5D0 = (__int64)algn_4FFC5E0;
      qword_4FFC5D8 = 0x400000000LL;
      qword_4FFC5A8 = 0x400000001LL;
      __cxa_atexit((void (*)(void *))sub_27ABC80, &qword_4FFC5A0, &qword_4A427C0);
      sub_2207640((__int64)&byte_4FFC580);
    }
    v169 = (unsigned int)qword_4FFC5A8;
    v250 = (__int64 *)v252;
    v251 = 0x400000000LL;
    if ( (_DWORD)qword_4FFC5A8 )
      sub_27ABF90(
        (__int64)&v250,
        (__int64)&qword_4FFC5A0,
        (__int64)v56,
        v55.m128i_i64[0],
        v23,
        (unsigned int)qword_4FFC5A8);
    v170 = (unsigned int)qword_4FFC5D8;
    v253 = v255;
    v254 = 0x400000000LL;
    if ( (_DWORD)qword_4FFC5D8 )
      sub_27AC1D0(
        (__int64)&v253,
        (__int64)&qword_4FFC5D0,
        (__int64)v56,
        v55.m128i_i64[0],
        (unsigned int)qword_4FFC5D8,
        v169);
    if ( !byte_4FFC508 && (unsigned int)sub_2207590((__int64)&byte_4FFC508) )
    {
      qword_4FFC530 = 1;
      qword_4FFC520 = (__int64)&qword_4FFC530;
      qword_4FFC550 = (__int64)algn_4FFC560;
      qword_4FFC558 = 0x400000000LL;
      qword_4FFC528 = 0x400000001LL;
      __cxa_atexit((void (*)(void *))sub_27ABC80, &qword_4FFC520, &qword_4A427C0);
      sub_2207640((__int64)&byte_4FFC508);
    }
    s2[0] = &v257;
    s2[1] = (void *)0x400000000LL;
    if ( (_DWORD)qword_4FFC528 )
      sub_27ABF90((__int64)s2, (__int64)&qword_4FFC520, (__int64)v56, v55.m128i_i64[0], v170, v169);
    dest = v262;
    v261 = 0x400000000LL;
    if ( (_DWORD)qword_4FFC558 )
    {
      sub_27AC1D0((__int64)&dest, (__int64)&qword_4FFC550, (__int64)v56, v55.m128i_i64[0], v170, v169);
      v171 = *(unsigned __int64 **)(a6 + 8);
      v172 = &v171[12 * *(unsigned int *)(a6 + 24)];
      if ( v171 == v172 )
      {
LABEL_193:
        if ( dest != v262 )
          _libc_free((unsigned __int64)dest);
        goto LABEL_195;
      }
    }
    else
    {
      v171 = *(unsigned __int64 **)(a6 + 8);
      v172 = &v171[12 * *(unsigned int *)(a6 + 24)];
      if ( v172 == v171 )
      {
LABEL_195:
        if ( s2[0] != &v257 )
          _libc_free((unsigned __int64)s2[0]);
        if ( v253 != v255 )
          _libc_free((unsigned __int64)v253);
        if ( v250 != (__int64 *)v252 )
          _libc_free((unsigned __int64)v250);
        v54 = *(unsigned int *)(a6 + 24);
        goto LABEL_98;
      }
    }
    v173 = v171;
    do
    {
      v174 = v173[6];
      if ( (unsigned __int64 *)v174 != v173 + 8 )
        _libc_free(v174);
      if ( (unsigned __int64 *)*v173 != v173 + 2 )
        _libc_free(*v173);
      v173 += 12;
    }
    while ( v173 != v172 );
    goto LABEL_193;
  }
LABEL_98:
  sub_C7D6A0(*(_QWORD *)(a6 + 8), 96 * v54, 8);
  v78 = (unsigned int)v244;
  *(_DWORD *)(a6 + 24) = v244;
  if ( (_DWORD)v78 )
  {
    v219 = sub_C7D670(96 * v78, 8);
    *(_QWORD *)(a6 + 8) = v219;
    v79 = *(unsigned int *)(a6 + 24);
    *(_QWORD *)(a6 + 16) = v243;
    v232 = v242;
    v80 = sub_27AC980();
    sub_27AC440(&v250, (__int64)v80, v81, v82, v83, v84);
    v85 = sub_27ACA30();
    sub_27AC440(s2, (__int64)v85, v86, v87, v88, v89);
    v92 = 0;
    v93 = v232;
    if ( v79 )
    {
      v263 = v9;
      v94 = v232;
      v95 = 0;
      v96 = v7;
      v97 = v79;
      v98 = (_QWORD *)v219;
      do
      {
        v99 = v98;
        ++v95;
        v98 += 12;
        sub_27AC440(v99, v94, v93, v92, v90, v91);
        v94 += 96;
      }
      while ( v97 != v95 );
      v7 = v96;
      v9 = v263;
    }
    sub_27ABC80(s2);
    sub_27ABC80(&v250);
  }
  else
  {
    *(_QWORD *)(a6 + 8) = 0;
    *(_QWORD *)(a6 + 16) = 0;
  }
  sub_27AFDA0(v7, v7);
  sub_27AE500((__int64)&v241, v7, v100, v101, v102, v103);
  v222 = 1;
  v47 = (unsigned int)v248;
  v46 = v247;
LABEL_106:
  v104 = (__int64)v255;
  v250 = (__int64 *)v252;
  v253 = v255;
  v251 = 0x400000000LL;
  v254 = 0x400000000LL;
  if ( v47 )
  {
    v30 = *v46;
    v38 = (__int64)v252;
    v105 = v9;
    v106 = 0;
    v107 = v46;
    v108 = *v46;
    while ( 1 )
    {
      *(_QWORD *)(v38 + 8 * v106) = v108;
      ++v107;
      v106 = (unsigned int)(v251 + 1);
      LODWORD(v251) = v251 + 1;
      if ( !--v47 )
        break;
      v104 = HIDWORD(v251);
      v108 = *v107;
      if ( v106 + 1 > (unsigned __int64)HIDWORD(v251) )
      {
        sub_C8D5F0((__int64)&v250, v252, v106 + 1, 8u, v23, v30);
        v106 = (unsigned int)v251;
      }
      v38 = (__int64)v250;
    }
    v9 = v105;
  }
  sub_27AFD10(v7, (__int64)&v253, v38, v104, v23, v30);
  s2[0] = 0;
  s2[1] = 0;
  v257.m128i_i64[0] = 0;
  v257.m128i_i32[2] = 0;
  sub_27B1670((__int64)s2, a2 + 568);
  sub_C7D6A0((__int64)s2[1], 16LL * v257.m128i_u32[2], 8);
  p_dest = (__int64)&v250;
  v114 = sub_27B1390(a6, (const void **)&v250, v110, v111, v112, v113);
  if ( v114 )
  {
    v117 = sub_27ACA30();
    sub_27AC440(s2, (__int64)v117, v118, v119, v120, v121);
    sub_27ABF90(v114, (__int64)s2, v122, v123, v124, v125);
    p_dest = (__int64)&dest;
    sub_27AC1D0(v114 + 48, (__int64)&dest, v126, v127, v128, v129);
    sub_27ABC80(s2);
    --*(_DWORD *)(a6 + 16);
    ++*(_DWORD *)(a6 + 20);
    goto LABEL_115;
  }
  if ( v222 )
  {
LABEL_115:
    ++*(_QWORD *)v9;
    if ( *(_BYTE *)(v9 + 28) )
      goto LABEL_120;
    v130 = 4 * (*(_DWORD *)(v9 + 20) - *(_DWORD *)(v9 + 24));
    v131 = *(unsigned int *)(v9 + 16);
    if ( v130 < 0x20 )
      v130 = 32;
    if ( v130 < (unsigned int)v131 )
    {
      sub_C8C990(v9, p_dest);
    }
    else
    {
      p_dest = 0xFFFFFFFFLL;
      memset(*(void **)(v9 + 8), -1, 8 * v131);
LABEL_120:
      *(_QWORD *)(v9 + 20) = 0;
    }
    v132 = *(unsigned int *)(a6 + 16);
    v133 = *(__int64 **)a6;
    v134 = *(_QWORD *)(a6 + 8) + 96LL * *(unsigned int *)(a6 + 24);
    if ( (_DWORD)v132 )
    {
      v257.m128i_i64[0] = *(_QWORD *)(a6 + 8);
      v257.m128i_i64[1] = v134;
      s2[0] = (void *)a6;
      s2[1] = v133;
      sub_27ADFC0((__int64)s2, p_dest, (__int64)v133, v132, v115, v116);
      v134 = *(_QWORD *)(a6 + 8) + 96LL * *(unsigned int *)(a6 + 24);
    }
    else
    {
      s2[0] = (void *)a6;
      s2[1] = v133;
      v257.m128i_i64[0] = v134;
      v257.m128i_i64[1] = v134;
    }
    v135 = _mm_loadu_si128(&v257);
    v136 = v257.m128i_i64[0];
    v245 = _mm_loadu_si128((const __m128i *)s2);
    v246 = v135;
    if ( v257.m128i_i64[0] != v134 )
    {
      v233 = v7;
      while ( 1 )
      {
        v137 = *(__int64 **)v136;
        v138 = *(_QWORD *)v136 + 8LL * *(unsigned int *)(v136 + 8);
        if ( v138 != *(_QWORD *)v136 )
          break;
LABEL_133:
        v246.m128i_i64[0] += 96;
        sub_27ADFC0((__int64)&v245, p_dest, (__int64)v133, v132, v115, v116);
        v136 = v246.m128i_i64[0];
        if ( v246.m128i_i64[0] == v134 )
        {
          v7 = v233;
          goto LABEL_135;
        }
      }
      v139 = *(_BYTE *)(v9 + 28);
      while ( 2 )
      {
        p_dest = *v137;
        if ( v139 )
        {
          v140 = *(__int64 **)(v9 + 8);
          v132 = *(unsigned int *)(v9 + 20);
          v133 = &v140[v132];
          if ( v140 != v133 )
          {
            while ( p_dest != *v140 )
            {
              if ( v133 == ++v140 )
                goto LABEL_144;
            }
LABEL_132:
            if ( (__int64 *)v138 == ++v137 )
              goto LABEL_133;
            continue;
          }
LABEL_144:
          if ( (unsigned int)v132 < *(_DWORD *)(v9 + 16) )
          {
            v132 = (unsigned int)(v132 + 1);
            *(_DWORD *)(v9 + 20) = v132;
            *v133 = p_dest;
            v139 = *(_BYTE *)(v9 + 28);
            ++*(_QWORD *)v9;
            goto LABEL_132;
          }
        }
        break;
      }
      sub_C8CC70(v9, p_dest, (__int64)v133, v132, v115, v116);
      v139 = *(_BYTE *)(v9 + 28);
      goto LABEL_132;
    }
  }
LABEL_135:
  v141 = v250;
  v142 = &v250[(unsigned int)v251];
  if ( v142 != v250 )
  {
    do
    {
      v143 = *v141;
      if ( *(_BYTE *)(v9 + 28) )
      {
        v144 = *(_QWORD **)(v9 + 8);
        v145 = &v144[*(unsigned int *)(v9 + 20)];
        if ( v144 != v145 )
        {
          while ( v143 != *v144 )
          {
            if ( v145 == ++v144 )
              goto LABEL_147;
          }
LABEL_141:
          *(_BYTE *)(a1 + 72) = 0;
          goto LABEL_142;
        }
      }
      else if ( sub_C8CA60(v9, v143) )
      {
        goto LABEL_141;
      }
LABEL_147:
      ++v141;
    }
    while ( v142 != v141 );
  }
  v146 = v247;
  v228 = (unsigned int)v248;
  v147 = 8LL * (unsigned int)v248;
  v148 = *v247;
  v149 = &v247[(unsigned __int64)v147 / 8];
  v150 = v147 >> 3;
  v151 = v147 >> 5;
  if ( v151 )
  {
    v234 = v247;
    v152 = *v247;
    v153 = v247;
    v154 = &v247[4 * v151];
    while ( 1 )
    {
      if ( !(unsigned __int8)sub_B46250(v148, v152, 0) )
      {
        v146 = v234;
        goto LABEL_157;
      }
      if ( !(unsigned __int8)sub_B46250(v148, v153[1], 0) )
      {
        v146 = v234;
        ++v153;
        goto LABEL_157;
      }
      if ( !(unsigned __int8)sub_B46250(v148, v153[2], 0) )
        break;
      if ( !(unsigned __int8)sub_B46250(v148, v153[3], 0) )
      {
        v146 = v234;
        v153 += 3;
        goto LABEL_157;
      }
      v153 += 4;
      if ( v153 == v154 )
      {
        v146 = v234;
        v150 = v149 - v153;
        goto LABEL_283;
      }
      v152 = *v153;
    }
    v146 = v234;
    v153 += 2;
    goto LABEL_157;
  }
  v153 = v247;
LABEL_283:
  if ( v150 == 2 )
    goto LABEL_305;
  if ( v150 != 3 )
  {
    if ( v150 == 1 )
      goto LABEL_286;
    goto LABEL_158;
  }
  if ( !(unsigned __int8)sub_B46250(v148, *v153, 0) )
    goto LABEL_157;
  ++v153;
LABEL_305:
  if ( !(unsigned __int8)sub_B46250(v148, *v153, 0) )
    goto LABEL_157;
  ++v153;
LABEL_286:
  if ( !(unsigned __int8)sub_B46250(v148, *v153, 0) )
  {
LABEL_157:
    if ( v149 != v153 )
      goto LABEL_141;
  }
LABEL_158:
  v217 = *(_DWORD *)(v148 + 4) & 0x7FFFFFF;
  if ( !v217 )
    goto LABEL_260;
  v264 = v9;
  v155 = 0;
  v156 = (__int64)&v245;
  v215 = (*(_DWORD *)(v148 + 4) & 0x7FFFFFF) - 1;
  v216 = (unsigned __int8 *)v148;
  v157 = v146;
  v158 = s2;
  v220 = v7;
  v159 = &v257;
  while ( 2 )
  {
    v223 = v155;
    s2[0] = v159;
    dest = v262;
    s2[1] = (void *)0x400000000LL;
    v261 = 0x400000000LL;
    sub_27AFD10(v220, (__int64)&dest, v150, v156, v115, v116);
    v115 = (__int64)&v157[v228];
    if ( (__int64 *)v115 == v157 )
    {
      v162 = LODWORD(s2[1]);
    }
    else
    {
      v160 = (__int64)v158;
      v161 = v159;
      v162 = LODWORD(s2[1]);
      v163 = v155;
      v164 = &v157[v228];
      v165 = 32 * v155;
      do
      {
        v167 = *v157;
        if ( (*(_BYTE *)(*v157 + 7) & 0x40) != 0 )
          v166 = *(_QWORD *)(v167 - 8);
        else
          v166 = v167 - 32LL * (*(_DWORD *)(v167 + 4) & 0x7FFFFFF);
        v156 = HIDWORD(s2[1]);
        v116 = *(_QWORD *)(v166 + v165);
        if ( v162 + 1 > (unsigned __int64)HIDWORD(s2[1]) )
        {
          v225 = *(_QWORD *)(v166 + v165);
          sub_C8D5F0(v160, v161, v162 + 1, 8u, v115, v116);
          v162 = LODWORD(s2[1]);
          v116 = v225;
        }
        v150 = (__int64)s2[0];
        ++v157;
        *((_QWORD *)s2[0] + v162) = v116;
        v162 = (unsigned int)++LODWORD(s2[1]);
      }
      while ( v164 != v157 );
      v155 = v163;
      v158 = (void **)v160;
      v159 = v161;
    }
    v176 = 8 * v162;
    if ( !v176 )
      goto LABEL_210;
    v150 = v176 - 8;
    if ( v176 == 8 || !memcmp((char *)s2[0] + 8, s2[0], v150) )
      goto LABEL_210;
    if ( !sub_F58730(v216, v223) )
      goto LABEL_244;
    if ( sub_27B1390(a6, (const void **)v158, v177, v178, v179, v180) )
    {
LABEL_210:
      if ( dest != v262 )
        _libc_free((unsigned __int64)dest);
      if ( s2[0] != v159 )
        _libc_free((unsigned __int64)s2[0]);
      goto LABEL_214;
    }
    v181 = (char *)s2[0];
    v182 = 8LL * LODWORD(s2[1]);
    v183 = (char *)s2[0] + v182;
    v184 = v182 >> 3;
    v185 = v182 >> 5;
    if ( v185 )
    {
      v186 = *(_QWORD *)(*(_QWORD *)s2[0] + 8LL);
      v187 = (char *)s2[0];
      while ( v186 == *(_QWORD *)(*((_QWORD *)v187 + 1) + 8LL) )
      {
        if ( v186 != *(_QWORD *)(*((_QWORD *)v187 + 2) + 8LL) )
        {
          v187 += 16;
          goto LABEL_223;
        }
        if ( v186 != *(_QWORD *)(*((_QWORD *)v187 + 3) + 8LL) )
        {
          v187 += 24;
          goto LABEL_223;
        }
        v187 += 32;
        if ( v187 == (char *)s2[0] + 32 * v185 )
        {
          v184 = (v183 - v187) >> 3;
          goto LABEL_270;
        }
        if ( *(_QWORD *)(*(_QWORD *)v187 + 8LL) != v186 )
          goto LABEL_223;
      }
      v187 += 8;
      goto LABEL_223;
    }
    v187 = (char *)s2[0];
LABEL_270:
    if ( v184 == 2 )
    {
      v211 = *(_QWORD *)(*(_QWORD *)s2[0] + 8LL);
      goto LABEL_299;
    }
    if ( v184 != 3 )
    {
      if ( v184 == 1 )
      {
        v211 = *(_QWORD *)(*(_QWORD *)s2[0] + 8LL);
        goto LABEL_274;
      }
      goto LABEL_224;
    }
    v211 = *(_QWORD *)(*(_QWORD *)s2[0] + 8LL);
    if ( *(_QWORD *)(*(_QWORD *)v187 + 8LL) != v211 )
      goto LABEL_223;
    v187 += 8;
LABEL_299:
    if ( v211 != *(_QWORD *)(*(_QWORD *)v187 + 8LL) )
      goto LABEL_223;
    v187 += 8;
LABEL_274:
    if ( v211 != *(_QWORD *)(*(_QWORD *)v187 + 8LL) )
    {
LABEL_223:
      if ( v183 != v187 )
        goto LABEL_244;
    }
LABEL_224:
    if ( *v216 != 34 && *v216 != 85 || v215 != v223 )
      goto LABEL_226;
    if ( v185 )
    {
      while ( **(_BYTE **)v181 > 0x15u )
      {
        if ( **((_BYTE **)v181 + 1) <= 0x15u )
        {
          v181 += 8;
          goto LABEL_243;
        }
        if ( **((_BYTE **)v181 + 2) <= 0x15u )
        {
          v181 += 16;
          goto LABEL_243;
        }
        if ( **((_BYTE **)v181 + 3) <= 0x15u )
        {
          v181 += 24;
          goto LABEL_243;
        }
        v181 += 32;
        if ( (char *)s2[0] + 32 * v185 == v181 )
          goto LABEL_313;
      }
      goto LABEL_243;
    }
LABEL_313:
    v214 = v183 - v181;
    if ( v183 - v181 != 16 )
    {
      if ( v214 != 24 )
      {
        if ( v214 == 8 )
          goto LABEL_316;
        goto LABEL_226;
      }
      if ( **(_BYTE **)v181 > 0x15u )
      {
        v181 += 8;
        goto LABEL_322;
      }
LABEL_243:
      if ( v183 != v181 )
      {
LABEL_244:
        *(_BYTE *)(a1 + 72) = 0;
        sub_27ABC80(s2);
        goto LABEL_142;
      }
      goto LABEL_226;
    }
LABEL_322:
    if ( **(_BYTE **)v181 <= 0x15u )
      goto LABEL_243;
    v181 += 8;
LABEL_316:
    if ( **(_BYTE **)v181 <= 0x15u )
      goto LABEL_243;
LABEL_226:
    v188 = *(_QWORD *)a6;
    v189 = *(_DWORD *)(a6 + 16);
    v190 = *(_QWORD *)a6 + 1LL;
    if ( v189 )
    {
      *(_QWORD *)a6 = v190;
      v191 = (4 * v189 / 3u + 1) | ((unsigned __int64)(4 * v189 / 3u + 1) >> 1);
      v192 = (((v191 >> 2) | v191) >> 4) | (v191 >> 2) | v191;
      v188 = v192 >> 8;
      v193 = ((((v192 >> 8) | v192) >> 16) | (v192 >> 8) | v192) + 1;
      if ( *(_DWORD *)(a6 + 24) < v193 )
        sub_27B06D0(a6, v193);
    }
    else
    {
      *(_QWORD *)a6 = v190;
    }
    sub_27B1110((__int64)&v245, a6, (const void **)v158, v188, v185, (__int64)v183);
    if ( (char *)s2[0] + 8 * LODWORD(s2[1]) != s2[0] )
    {
      v229 = v155;
      v194 = (__int64 *)s2[0];
      v195 = v159;
      v196 = v158;
      v197 = (__int64 *)((char *)s2[0] + 8 * LODWORD(s2[1]));
      do
      {
        v198 = *v194++;
        sub_AE6EC0(v264, v198);
      }
      while ( v197 != v194 );
      v155 = v229;
      v158 = v196;
      v159 = v195;
    }
    sub_27ABC80(v158);
LABEL_214:
    v157 = v247;
    if ( v217 != (_DWORD)++v155 )
    {
      v228 = (unsigned int)v248;
      continue;
    }
    break;
  }
  v7 = v220;
  v148 = *v247;
LABEL_260:
  if ( (unsigned __int8)sub_27ABDB0((char *)v148) )
    ++*a5;
  v257.m128i_i32[0] = -1;
  v258 = 0x400000000LL;
  v202 = *a4;
  v257.m128i_i64[1] = (__int64)v259;
  *a4 = ++v202;
  HIDWORD(s2[0]) = v202;
  v203 = *(unsigned int *)(a6 + 16);
  v204 = *a5;
  LODWORD(s2[1]) = *(_DWORD *)(a6 + 16);
  HIDWORD(s2[1]) = v204;
  v205 = *(unsigned int *)(v7 + 40);
  v206 = *(const void **)(v7 + 32);
  LODWORD(s2[0]) = v205;
  v207 = 8 * v205;
  v208 = v205;
  if ( v205 > 4 )
  {
    sub_C8D5F0((__int64)&v257.m128i_i64[1], v259, v205, 8u, v200, v201);
    v213 = (char *)(v257.m128i_i64[1] + 8LL * (unsigned int)v258);
  }
  else
  {
    if ( !v207 )
      goto LABEL_264;
    v213 = v259;
  }
  memcpy(v213, v206, v207);
  LODWORD(v207) = v258;
LABEL_264:
  v209 = v207 + v208;
  LODWORD(v258) = v209;
  *(void **)a1 = s2[0];
  *(void **)(a1 + 8) = s2[1];
  *(_DWORD *)(a1 + 16) = v257.m128i_i32[0];
  *(_QWORD *)(a1 + 24) = a1 + 40;
  *(_QWORD *)(a1 + 32) = 0x400000000LL;
  if ( v209 )
    sub_27AC070(a1 + 24, (char **)&v257.m128i_i64[1], v203, v199, v200, v201);
  v210 = (char *)v257.m128i_i64[1];
  *(_BYTE *)(a1 + 72) = 1;
  if ( v210 != v259 )
    _libc_free((unsigned __int64)v210);
LABEL_142:
  sub_27ABC80(&v250);
  v48 = v247;
LABEL_62:
  if ( v48 != (__int64 *)v249 )
    _libc_free((unsigned __int64)v48);
  v22 = (__int64)v238;
LABEL_30:
  sub_C7D6A0(v22, 8LL * v240, 4);
  return a1;
}
