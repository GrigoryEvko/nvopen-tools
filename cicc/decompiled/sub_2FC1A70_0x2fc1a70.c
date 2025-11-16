// Function: sub_2FC1A70
// Address: 0x2fc1a70
//
__int64 __fastcall sub_2FC1A70(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rcx
  int v7; // eax
  __int64 v9; // rax
  int v10; // eax
  int v11; // eax
  __int64 v12; // rax
  int v13; // eax
  __int64 v14; // rsi
  _BYTE *v15; // rsi
  char *v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 v21; // r9
  __int64 v22; // rcx
  __int64 v23; // r8
  unsigned __int64 v24; // r12
  __int64 v25; // rax
  unsigned __int64 v26; // rdi
  __m128i *v27; // rdx
  const __m128i *v28; // rax
  const __m128i *v29; // rcx
  unsigned __int64 v30; // r8
  unsigned __int64 v31; // r12
  __int64 v32; // rax
  unsigned __int64 v33; // rdi
  __m128i *v34; // rdx
  const __m128i *v35; // rax
  unsigned __int64 v36; // rax
  unsigned int v37; // r14d
  unsigned __int64 v38; // rcx
  unsigned __int64 v39; // r8
  _BYTE *v40; // r9
  char v41; // cl
  _BYTE *v42; // rdi
  int v43; // ecx
  __int64 *v44; // r13
  __int64 *i; // r15
  __int64 v46; // rsi
  unsigned int v47; // edx
  _QWORD *v48; // r12
  __int64 v49; // r10
  unsigned int v50; // eax
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rsi
  _QWORD *v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r12
  unsigned __int64 j; // r13
  int v58; // edx
  int v59; // r15d
  unsigned int v60; // r13d
  unsigned int v61; // ecx
  unsigned __int64 v62; // rax
  __int64 *v63; // rdx
  __int64 v64; // rdi
  unsigned __int64 v65; // rcx
  unsigned int v66; // eax
  __int64 v67; // r13
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // rsi
  __int64 v71; // rax
  unsigned int v72; // edx
  __int64 v73; // rdi
  __int64 v74; // rdx
  unsigned int v75; // eax
  unsigned int v76; // eax
  unsigned int v77; // edx
  __int64 v78; // rsi
  __int64 v79; // rax
  __int64 v80; // rcx
  __int64 v81; // rdx
  __int64 v82; // rax
  __int64 v83; // rcx
  _QWORD *v84; // rax
  char *v85; // rcx
  __int64 v86; // rax
  __int64 k; // rdx
  int v88; // ecx
  __int64 v89; // rsi
  __int64 v90; // rcx
  __int64 v91; // rcx
  unsigned __int64 v92; // r11
  int v93; // eax
  int v94; // r12d
  int v95; // r11d
  unsigned __int64 v96; // r10
  unsigned __int64 v97; // rax
  char v98; // si
  __int64 v99; // rax
  unsigned int v100; // ecx
  __int64 v101; // rsi
  __int64 v102; // rax
  __int64 v103; // r8
  __int64 v104; // r10
  __int64 v105; // rax
  __int64 m; // rsi
  unsigned int v107; // ecx
  __int64 v108; // rcx
  __int64 v109; // r8
  __int64 v110; // r9
  __int64 v111; // r9
  __int64 v112; // rcx
  __int64 v113; // r8
  unsigned __int64 v114; // r12
  __int64 v115; // rax
  unsigned __int64 v116; // rdi
  __m128i *v117; // rdx
  const __m128i *v118; // rax
  const __m128i *v119; // rcx
  unsigned __int64 v120; // r8
  unsigned __int64 v121; // r12
  __int64 v122; // rax
  unsigned __int64 v123; // rdi
  __m128i *v124; // rdx
  const __m128i *v125; // rax
  unsigned __int64 v126; // rax
  unsigned __int64 v127; // rcx
  unsigned int v128; // esi
  __int64 v129; // r13
  __int64 v130; // rdi
  int v131; // r12d
  __int64 v132; // r8
  __int64 v133; // r11
  int v134; // r10d
  __int64 v135; // r9
  unsigned int v136; // r14d
  unsigned int v137; // edx
  __int64 v138; // rax
  __int64 v139; // rcx
  __int64 v140; // rax
  __int64 v141; // rdi
  __int64 v142; // rsi
  __int64 v143; // r8
  int v144; // r14d
  __int64 v145; // r9
  __int64 v146; // rax
  __int64 v147; // rdx
  __int64 v148; // rcx
  unsigned __int64 v149; // rax
  __int64 v150; // r14
  int v151; // ecx
  int v152; // ecx
  int v153; // ecx
  int v154; // ecx
  int v155; // ecx
  unsigned __int64 v156; // rax
  __int64 v157; // rcx
  __int64 v158; // r13
  __int64 v159; // r12
  __int64 v160; // rax
  int v161; // ecx
  unsigned int v162; // eax
  unsigned __int64 v163; // r12
  int v164; // edx
  __int64 v165; // rdi
  int v166; // r11d
  unsigned __int64 v167; // rsi
  __int64 v168; // r13
  int v169; // r10d
  __int64 v170; // rsi
  __int64 v171; // r8
  __int64 v172; // r9
  unsigned __int64 v173; // rax
  char v174; // si
  __int64 v175; // rax
  _QWORD *v176; // rbx
  _QWORD *v177; // r12
  unsigned __int64 v178; // rdi
  __int64 v180; // rdi
  __int64 v181; // rax
  int v182; // eax
  int v183; // eax
  unsigned __int64 v184; // r12
  __int64 v185; // r12
  int v186; // ecx
  int v187; // edx
  int v188; // r11d
  int v189; // r11d
  __int64 v190; // r10
  __int64 v191; // rcx
  unsigned int v192; // r12d
  __int64 v193; // rdi
  int v194; // r12d
  int v195; // r12d
  __int64 v196; // r11
  unsigned int v197; // ecx
  int v198; // edi
  int v199; // eax
  __int64 v200; // r11
  __int64 v201; // rcx
  int v202; // r10d
  __int64 v203; // rsi
  __int64 v204; // rdi
  int v205; // eax
  __int64 v206; // r11
  __int64 v207; // rcx
  __int64 v208; // rdi
  int v209; // r10d
  __int64 v210; // rdx
  __int64 v211; // [rsp+18h] [rbp-4E8h]
  _BYTE *v212; // [rsp+40h] [rbp-4C0h]
  unsigned int v213; // [rsp+40h] [rbp-4C0h]
  unsigned __int64 v214; // [rsp+40h] [rbp-4C0h]
  unsigned __int64 v215; // [rsp+48h] [rbp-4B8h]
  _BYTE *v216; // [rsp+48h] [rbp-4B8h]
  unsigned __int64 v217; // [rsp+48h] [rbp-4B8h]
  _BYTE *v218; // [rsp+48h] [rbp-4B8h]
  _BYTE *v219; // [rsp+48h] [rbp-4B8h]
  unsigned __int64 v220; // [rsp+48h] [rbp-4B8h]
  unsigned __int64 v221; // [rsp+50h] [rbp-4B0h]
  int v222; // [rsp+50h] [rbp-4B0h]
  unsigned __int64 v223; // [rsp+50h] [rbp-4B0h]
  _BYTE *v224; // [rsp+50h] [rbp-4B0h]
  _BYTE *v225; // [rsp+50h] [rbp-4B0h]
  unsigned __int64 v226; // [rsp+50h] [rbp-4B0h]
  unsigned __int64 v227; // [rsp+50h] [rbp-4B0h]
  _BYTE *v228; // [rsp+50h] [rbp-4B0h]
  unsigned __int64 v229; // [rsp+58h] [rbp-4A8h]
  unsigned int v230; // [rsp+60h] [rbp-4A0h]
  __int64 v232; // [rsp+70h] [rbp-490h] BYREF
  _QWORD *v233; // [rsp+78h] [rbp-488h]
  __int64 v234; // [rsp+80h] [rbp-480h]
  unsigned int v235; // [rsp+88h] [rbp-478h]
  void *v236; // [rsp+90h] [rbp-470h] BYREF
  __int64 v237; // [rsp+98h] [rbp-468h]
  _DWORD v238[8]; // [rsp+A0h] [rbp-460h] BYREF
  void *v239; // [rsp+C0h] [rbp-440h] BYREF
  __int64 v240; // [rsp+C8h] [rbp-438h]
  _DWORD v241[8]; // [rsp+D0h] [rbp-430h] BYREF
  char v242[8]; // [rsp+F0h] [rbp-410h] BYREF
  unsigned __int64 v243; // [rsp+F8h] [rbp-408h]
  char v244; // [rsp+10Ch] [rbp-3F4h]
  _BYTE v245[64]; // [rsp+110h] [rbp-3F0h] BYREF
  unsigned __int64 v246; // [rsp+150h] [rbp-3B0h]
  unsigned __int64 v247; // [rsp+158h] [rbp-3A8h]
  unsigned __int64 v248; // [rsp+160h] [rbp-3A0h]
  char v249[8]; // [rsp+170h] [rbp-390h] BYREF
  unsigned __int64 v250; // [rsp+178h] [rbp-388h]
  char v251; // [rsp+18Ch] [rbp-374h]
  _BYTE v252[64]; // [rsp+190h] [rbp-370h] BYREF
  unsigned __int64 v253; // [rsp+1D0h] [rbp-330h]
  unsigned __int64 v254; // [rsp+1D8h] [rbp-328h]
  unsigned __int64 v255; // [rsp+1E0h] [rbp-320h]
  char v256[8]; // [rsp+1F0h] [rbp-310h] BYREF
  unsigned __int64 v257; // [rsp+1F8h] [rbp-308h]
  char v258; // [rsp+20Ch] [rbp-2F4h]
  _BYTE v259[64]; // [rsp+210h] [rbp-2F0h] BYREF
  unsigned __int64 v260; // [rsp+250h] [rbp-2B0h]
  unsigned __int64 v261; // [rsp+258h] [rbp-2A8h]
  unsigned __int64 v262; // [rsp+260h] [rbp-2A0h]
  char v263[8]; // [rsp+270h] [rbp-290h] BYREF
  unsigned __int64 v264; // [rsp+278h] [rbp-288h]
  char v265; // [rsp+28Ch] [rbp-274h]
  _BYTE v266[64]; // [rsp+290h] [rbp-270h] BYREF
  unsigned __int64 v267; // [rsp+2D0h] [rbp-230h]
  unsigned __int64 v268; // [rsp+2D8h] [rbp-228h]
  unsigned __int64 v269; // [rsp+2E0h] [rbp-220h]
  _BYTE *v270; // [rsp+2F0h] [rbp-210h] BYREF
  unsigned __int64 v271; // [rsp+2F8h] [rbp-208h]
  _BYTE v272[80]; // [rsp+300h] [rbp-200h] BYREF
  unsigned __int64 v273; // [rsp+350h] [rbp-1B0h]
  __int64 v274; // [rsp+358h] [rbp-1A8h]
  char v275[8]; // [rsp+368h] [rbp-198h] BYREF
  unsigned __int64 v276; // [rsp+370h] [rbp-190h]
  char v277; // [rsp+384h] [rbp-17Ch]
  const __m128i *v278; // [rsp+3C8h] [rbp-138h]
  const __m128i *v279; // [rsp+3D0h] [rbp-130h]
  _BYTE *v280; // [rsp+3E0h] [rbp-120h] BYREF
  unsigned __int64 v281; // [rsp+3E8h] [rbp-118h]
  _BYTE s[48]; // [rsp+3F0h] [rbp-110h] BYREF
  unsigned int v283; // [rsp+420h] [rbp-E0h]
  unsigned __int64 v284; // [rsp+440h] [rbp-C0h]
  __int64 v285; // [rsp+448h] [rbp-B8h]
  char v286[8]; // [rsp+458h] [rbp-A8h] BYREF
  unsigned __int64 v287; // [rsp+460h] [rbp-A0h]
  char v288; // [rsp+474h] [rbp-8Ch]
  const __m128i *v289; // [rsp+4B8h] [rbp-48h]
  const __m128i *v290; // [rsp+4C0h] [rbp-40h]

  v6 = (a2 + 63) >> 6;
  LOBYTE(v7) = a2;
  v232 = 0;
  v233 = 0;
  v234 = 0;
  v235 = 0;
  *(_DWORD *)(a1 + 1280) = 0;
  *(_DWORD *)(a1 + 1336) = a2;
  v230 = (a2 + 63) >> 6;
  v229 = v6;
  if ( (_DWORD)v6 )
  {
    v9 = 0;
    if ( v6 > *(unsigned int *)(a1 + 1284) )
    {
      sub_C8D5F0(a1 + 1272, (const void *)(a1 + 1288), v6, 8u, a5, a6);
      v9 = 8LL * *(unsigned int *)(a1 + 1280);
    }
    memset((void *)(*(_QWORD *)(a1 + 1272) + v9), 0, 8 * v229);
    *(_DWORD *)(a1 + 1280) += v230;
    v7 = *(_DWORD *)(a1 + 1336);
  }
  v10 = v7 & 0x3F;
  if ( v10 )
    *(_QWORD *)(*(_QWORD *)(a1 + 1272) + 8LL * *(unsigned int *)(a1 + 1280) - 8) &= ~(-1LL << v10);
  LOBYTE(v11) = a2;
  *(_DWORD *)(a1 + 1352) = 0;
  *(_DWORD *)(a1 + 1408) = a2;
  if ( v230 )
  {
    v12 = 0;
    if ( v229 > *(unsigned int *)(a1 + 1356) )
    {
      sub_C8D5F0(a1 + 1344, (const void *)(a1 + 1360), v229, 8u, a5, a6);
      v12 = 8LL * *(unsigned int *)(a1 + 1352);
    }
    memset((void *)(*(_QWORD *)(a1 + 1344) + v12), 0, 8 * v229);
    *(_DWORD *)(a1 + 1352) += v230;
    v11 = *(_DWORD *)(a1 + 1408);
  }
  v13 = v11 & 0x3F;
  if ( v13 )
    *(_QWORD *)(*(_QWORD *)(a1 + 1344) + 8LL * *(unsigned int *)(a1 + 1352) - 8) &= ~(-1LL << v13);
  v211 = a2;
  v236 = v238;
  v237 = 0x800000000LL;
  if ( a2 > 8 )
  {
    sub_C8D5F0((__int64)&v236, v238, a2, 4u, a5, a6);
    memset(v236, 0, 4LL * a2);
    v240 = 0x800000000LL;
    LODWORD(v237) = a2;
    v239 = v241;
    sub_C8D5F0((__int64)&v239, v241, a2, 4u, v171, v172);
    memset(v239, 0, 4LL * a2);
  }
  else if ( a2 )
  {
    v71 = 4LL * a2;
    if ( v71 )
    {
      if ( (unsigned int)v71 < 8 )
      {
        if ( (v71 & 4) != 0 )
        {
          v238[0] = 0;
          *(_DWORD *)((char *)&v238[-1] + (unsigned int)v71) = 0;
        }
        else if ( (_DWORD)v71 )
        {
          LOBYTE(v238[0]) = 0;
        }
      }
      else
      {
        *(_QWORD *)((char *)&v238[-2] + (unsigned int)v71) = 0;
        if ( (unsigned int)(v71 - 1) >= 8 )
        {
          v72 = 0;
          do
          {
            v73 = v72;
            v72 += 8;
            *(_QWORD *)((char *)v238 + v73) = 0;
          }
          while ( v72 < (((_DWORD)v71 - 1) & 0xFFFFFFF8) );
        }
      }
    }
    HIDWORD(v240) = 8;
    LODWORD(v237) = a2;
    v239 = v241;
    if ( v71 )
    {
      if ( (unsigned int)v71 < 8 )
      {
        if ( (v71 & 4) != 0 )
        {
          v241[0] = 0;
          *(_DWORD *)((char *)&v241[-1] + (unsigned int)v71) = 0;
        }
        else if ( (_DWORD)v71 )
        {
          LOBYTE(v241[0]) = 0;
        }
      }
      else
      {
        v74 = (unsigned int)v71;
        v75 = v71 - 1;
        *(_QWORD *)((char *)&v241[-2] + v74) = 0;
        if ( v75 >= 8 )
        {
          v76 = v75 & 0xFFFFFFF8;
          v77 = 0;
          do
          {
            v78 = v77;
            v77 += 8;
            *(_QWORD *)((char *)v241 + v78) = 0;
          }
          while ( v77 < v76 );
        }
      }
    }
  }
  else
  {
    LODWORD(v237) = 0;
    v239 = v241;
    HIDWORD(v240) = 8;
  }
  v14 = *(_QWORD *)(a1 + 8);
  LODWORD(v240) = a2;
  sub_2FC1490(&v270, v14);
  v15 = v245;
  v16 = v242;
  sub_C8CD80((__int64)v242, (__int64)v245, (__int64)&v270, v17, v18, v19);
  v22 = v274;
  v23 = v273;
  v246 = 0;
  v247 = 0;
  v248 = 0;
  v24 = v274 - v273;
  if ( v274 == v273 )
  {
    v24 = 0;
    v26 = 0;
  }
  else
  {
    if ( v24 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_382;
    v25 = sub_22077B0(v274 - v273);
    v22 = v274;
    v23 = v273;
    v26 = v25;
  }
  v246 = v26;
  v247 = v26;
  v248 = v26 + v24;
  if ( v23 != v22 )
  {
    v27 = (__m128i *)v26;
    v28 = (const __m128i *)v23;
    do
    {
      if ( v27 )
      {
        *v27 = _mm_loadu_si128(v28);
        v27[1].m128i_i64[0] = v28[1].m128i_i64[0];
      }
      v28 = (const __m128i *)((char *)v28 + 24);
      v27 = (__m128i *)((char *)v27 + 24);
    }
    while ( v28 != (const __m128i *)v22 );
    v26 += 8 * (((unsigned __int64)&v28[-2].m128i_u64[1] - v23) >> 3) + 24;
  }
  v247 = v26;
  v15 = v252;
  v16 = v249;
  sub_C8CD80((__int64)v249, (__int64)v252, (__int64)v275, v22, v23, v21);
  v29 = v279;
  v30 = (unsigned __int64)v278;
  v253 = 0;
  v254 = 0;
  v255 = 0;
  v31 = (char *)v279 - (char *)v278;
  if ( v279 == v278 )
  {
    v31 = 0;
    v33 = 0;
  }
  else
  {
    if ( v31 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_382;
    v32 = sub_22077B0((char *)v279 - (char *)v278);
    v29 = v279;
    v30 = (unsigned __int64)v278;
    v33 = v32;
  }
  v253 = v33;
  v254 = v33;
  v255 = v33 + v31;
  if ( (const __m128i *)v30 == v29 )
  {
    v36 = v33;
  }
  else
  {
    v34 = (__m128i *)v33;
    v35 = (const __m128i *)v30;
    do
    {
      if ( v34 )
      {
        *v34 = _mm_loadu_si128(v35);
        v34[1].m128i_i64[0] = v35[1].m128i_i64[0];
      }
      v35 = (const __m128i *)((char *)v35 + 24);
      v34 = (__m128i *)((char *)v34 + 24);
    }
    while ( v35 != v29 );
    v36 = v33 + 8 * (((unsigned __int64)&v35[-2].m128i_u64[1] - v30) >> 3) + 24;
  }
  v254 = v36;
  v37 = 0;
  while ( 1 )
  {
    v38 = v246;
    if ( v247 - v246 != v36 - v33 )
      goto LABEL_36;
    if ( v246 == v247 )
      break;
    v97 = v33;
    while ( *(_QWORD *)v38 == *(_QWORD *)v97 )
    {
      v98 = *(_BYTE *)(v38 + 16);
      if ( v98 != *(_BYTE *)(v97 + 16) || v98 && *(_QWORD *)(v38 + 8) != *(_QWORD *)(v97 + 8) )
        break;
      v38 += 24LL;
      v97 += 24LL;
      if ( v247 == v38 )
        goto LABEL_116;
    }
LABEL_36:
    v39 = *(_QWORD *)(v247 - 24);
    v40 = s;
    v281 = 0x600000000LL;
    v280 = s;
    v283 = a2;
    v41 = a2;
    if ( v230 )
    {
      v42 = s;
      if ( v229 > 6 )
      {
        v220 = v39;
        sub_C8D5F0((__int64)&v280, s, v229, 8u, v39, (__int64)s);
        v39 = v220;
        v42 = &v280[8 * (unsigned int)v281];
      }
      v221 = v39;
      memset(v42, 0, 8 * v229);
      LODWORD(v281) = v230 + v281;
      v41 = v283;
      v40 = s;
      v39 = v221;
    }
    v43 = v41 & 0x3F;
    if ( v43 )
      *(_QWORD *)&v280[8 * (unsigned int)v281 - 8] &= ~(-1LL << v43);
    v44 = *(__int64 **)(v39 + 64);
    for ( i = &v44[*(unsigned int *)(v39 + 72)]; i != v44; ++v44 )
    {
      v46 = *v44;
      if ( v235 )
      {
        v47 = (v235 - 1) & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
        v48 = &v233[10 * v47];
        v49 = *v48;
        if ( v46 == *v48 )
        {
LABEL_45:
          if ( v48 != &v233[10 * v235] )
          {
            v50 = *((_DWORD *)v48 + 18);
            if ( v283 < v50 )
            {
              if ( (v283 & 0x3F) != 0 )
                *(_QWORD *)&v280[8 * (unsigned int)v281 - 8] &= ~(-1LL << (v283 & 0x3F));
              v91 = (unsigned int)v281;
              v283 = v50;
              v92 = (v50 + 63) >> 6;
              if ( v92 != (unsigned int)v281 )
              {
                if ( v92 >= (unsigned int)v281 )
                {
                  v96 = v92 - (unsigned int)v281;
                  if ( v92 > HIDWORD(v281) )
                  {
                    v214 = v39;
                    v217 = v92 - (unsigned int)v281;
                    v225 = v40;
                    sub_C8D5F0((__int64)&v280, v40, v92, 8u, v39, (__int64)v40);
                    v91 = (unsigned int)v281;
                    v39 = v214;
                    v96 = v217;
                    v40 = v225;
                  }
                  if ( 8 * v96 )
                  {
                    v212 = v40;
                    v215 = v39;
                    v222 = v96;
                    memset(&v280[8 * v91], 0, 8 * v96);
                    LODWORD(v91) = v281;
                    v40 = v212;
                    v39 = v215;
                    LODWORD(v96) = v222;
                  }
                  LOBYTE(v50) = v283;
                  LODWORD(v281) = v96 + v91;
                }
                else
                {
                  LODWORD(v281) = (v50 + 63) >> 6;
                }
              }
              v93 = v50 & 0x3F;
              if ( v93 )
                *(_QWORD *)&v280[8 * (unsigned int)v281 - 8] &= ~(-1LL << v93);
            }
            v51 = 0;
            v52 = *((unsigned int *)v48 + 4);
            v53 = 8 * v52;
            if ( (_DWORD)v52 )
            {
              do
              {
                v54 = &v280[v51];
                v55 = *(_QWORD *)(v48[1] + v51);
                v51 += 8;
                *v54 |= v55;
              }
              while ( v51 != v53 );
            }
          }
        }
        else
        {
          v94 = 1;
          while ( v49 != -4096 )
          {
            v95 = v94 + 1;
            v47 = (v235 - 1) & (v94 + v47);
            v48 = &v233[10 * v47];
            v49 = *v48;
            if ( v46 == *v48 )
              goto LABEL_45;
            v94 = v95;
          }
        }
      }
    }
    v56 = *(_QWORD *)(v39 + 56);
    for ( j = v39 + 48; j != v56; v56 = *(_QWORD *)(v56 + 8) )
    {
      while ( 1 )
      {
        v58 = *(unsigned __int16 *)(v56 + 68);
        if ( (unsigned __int16)(v58 - 14) > 4u )
        {
          v79 = *(_QWORD *)(v56 + 32);
          if ( (unsigned int)(v58 - 22) > 1 )
          {
            for ( k = v79 + 40LL * (*(_DWORD *)(v56 + 40) & 0xFFFFFF); k != v79; v79 += 40 )
            {
              if ( *(_BYTE *)v79 == 5 )
              {
                v88 = *(_DWORD *)(v79 + 24);
                if ( v88 >= 0 )
                {
                  v89 = 1LL << v88;
                  v90 = (unsigned int)v88 >> 6;
                  if ( (*(_QWORD *)&v280[8 * v90] & v89) == 0 )
                    *(_QWORD *)(*(_QWORD *)(a1 + 1344) + 8 * v90) |= v89;
                }
              }
            }
          }
          else
          {
            v80 = *(int *)(v79 + 24);
            if ( (int)v80 >= 0 )
            {
              v81 = 1LL << v80;
              v82 = 8LL * ((unsigned int)v80 >> 6);
              v83 = 4 * v80;
              *(_QWORD *)(v82 + *(_QWORD *)(a1 + 1272)) |= v81;
              v84 = &v280[v82];
              if ( *(_WORD *)(v56 + 68) == 22 )
              {
                *v84 |= v81;
                v85 = (char *)v236 + v83;
              }
              else
              {
                *v84 &= ~v81;
                v85 = (char *)v239 + v83;
              }
              ++*(_DWORD *)v85;
              v86 = *(unsigned int *)(a1 + 1200);
              if ( v86 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 1204) )
              {
                v216 = v40;
                v223 = v39;
                sub_C8D5F0(a1 + 1192, (const void *)(a1 + 1208), v86 + 1, 8u, v39, (__int64)v40);
                v86 = *(unsigned int *)(a1 + 1200);
                v40 = v216;
                v39 = v223;
              }
              ++v37;
              *(_QWORD *)(*(_QWORD *)(a1 + 1192) + 8 * v86) = v56;
              ++*(_DWORD *)(a1 + 1200);
            }
          }
        }
        if ( (*(_BYTE *)v56 & 4) == 0 )
          break;
        v56 = *(_QWORD *)(v56 + 8);
        if ( j == v56 )
          goto LABEL_56;
      }
      while ( (*(_BYTE *)(v56 + 44) & 8) != 0 )
        v56 = *(_QWORD *)(v56 + 8);
    }
LABEL_56:
    if ( !v235 )
    {
      ++v232;
LABEL_230:
      v218 = v40;
      v226 = v39;
      sub_2FC1820((__int64)&v232, 2 * v235);
      if ( !v235 )
        goto LABEL_394;
      v39 = v226;
      v40 = v218;
      v164 = v234 + 1;
      v65 = (v235 - 1) & (((unsigned int)v226 >> 9) ^ ((unsigned int)v226 >> 4));
      v62 = (unsigned __int64)&v233[10 * v65];
      v165 = *(_QWORD *)v62;
      if ( v226 != *(_QWORD *)v62 )
      {
        v166 = 1;
        v167 = 0;
        while ( v165 != -4096 )
        {
          if ( !v167 && v165 == -8192 )
            v167 = v62;
          v65 = (v235 - 1) & ((_DWORD)v65 + v166);
          v62 = (unsigned __int64)&v233[10 * v65];
          v165 = *(_QWORD *)v62;
          if ( v226 == *(_QWORD *)v62 )
            goto LABEL_223;
          ++v166;
        }
        if ( v167 )
          v62 = v167;
      }
      goto LABEL_223;
    }
    v59 = 1;
    v60 = ((unsigned int)v39 >> 4) ^ ((unsigned int)v39 >> 9);
    v61 = (v235 - 1) & v60;
    v62 = 0;
    v63 = &v233[10 * v61];
    v64 = *v63;
    if ( v39 == *v63 )
    {
LABEL_58:
      v65 = *((unsigned int *)v63 + 18);
      v66 = v283;
      v67 = (__int64)(v63 + 1);
      if ( v283 <= (unsigned int)v65 )
        goto LABEL_59;
      v161 = v63[9] & 0x3F;
      if ( v161 )
        *(_QWORD *)(v63[1] + 8LL * *((unsigned int *)v63 + 4) - 8) &= ~(-1LL << v161);
LABEL_203:
      v65 = *(unsigned int *)(v67 + 8);
      *(_DWORD *)(v67 + 64) = v66;
      v39 = (v66 + 63) >> 6;
      if ( v39 != v65 )
      {
        if ( v39 >= v65 )
        {
          v163 = v39 - v65;
          if ( v39 > *(unsigned int *)(v67 + 12) )
          {
            v228 = v40;
            sub_C8D5F0(v67, (const void *)(v67 + 16), v39, 8u, v39, (__int64)v40);
            v65 = *(unsigned int *)(v67 + 8);
            v40 = v228;
          }
          if ( 8 * v163 )
          {
            v224 = v40;
            memset((void *)(*(_QWORD *)v67 + 8 * v65), 0, 8 * v163);
            v65 = *(unsigned int *)(v67 + 8);
            v40 = v224;
          }
          v65 += v163;
          v66 = *(_DWORD *)(v67 + 64);
          *(_DWORD *)(v67 + 8) = v65;
        }
        else
        {
          *(_DWORD *)(v67 + 8) = (v66 + 63) >> 6;
        }
      }
      v162 = v66 & 0x3F;
      if ( v162 )
      {
        v65 = v162;
        *(_QWORD *)(*(_QWORD *)v67 + 8LL * *(unsigned int *)(v67 + 8) - 8) &= ~(-1LL << v162);
      }
      goto LABEL_59;
    }
    while ( v64 != -4096 )
    {
      if ( !v62 && v64 == -8192 )
        v62 = (unsigned __int64)v63;
      v61 = (v235 - 1) & (v59 + v61);
      v63 = &v233[10 * v61];
      v64 = *v63;
      if ( v39 == *v63 )
        goto LABEL_58;
      ++v59;
    }
    if ( !v62 )
      v62 = (unsigned __int64)v63;
    ++v232;
    v164 = v234 + 1;
    if ( 4 * ((int)v234 + 1) >= 3 * v235 )
      goto LABEL_230;
    v65 = v235 - HIDWORD(v234) - v164;
    if ( (unsigned int)v65 <= v235 >> 3 )
    {
      v219 = v40;
      v227 = v39;
      sub_2FC1820((__int64)&v232, v235);
      if ( !v235 )
      {
LABEL_394:
        LODWORD(v234) = v234 + 1;
        BUG();
      }
      v65 = 0;
      LODWORD(v168) = (v235 - 1) & v60;
      v39 = v227;
      v40 = v219;
      v169 = 1;
      v164 = v234 + 1;
      v62 = (unsigned __int64)&v233[10 * (unsigned int)v168];
      v170 = *(_QWORD *)v62;
      if ( v227 != *(_QWORD *)v62 )
      {
        while ( v170 != -4096 )
        {
          if ( v170 == -8192 && !v65 )
            v65 = v62;
          v168 = (v235 - 1) & ((_DWORD)v168 + v169);
          v62 = (unsigned __int64)&v233[10 * v168];
          v170 = *(_QWORD *)v62;
          if ( v227 == *(_QWORD *)v62 )
            goto LABEL_223;
          ++v169;
        }
        if ( v65 )
          v62 = v65;
      }
    }
LABEL_223:
    LODWORD(v234) = v164;
    if ( *(_QWORD *)v62 != -4096 )
      --HIDWORD(v234);
    *(_QWORD *)v62 = v39;
    v67 = v62 + 8;
    *(_QWORD *)(v62 + 72) = 0;
    *(_QWORD *)(v62 + 8) = v62 + 24;
    *(_QWORD *)(v62 + 16) = 0x600000000LL;
    *(_OWORD *)(v62 + 24) = 0;
    *(_OWORD *)(v62 + 40) = 0;
    *(_OWORD *)(v62 + 56) = 0;
    v66 = v283;
    if ( v283 )
      goto LABEL_203;
LABEL_59:
    v68 = 0;
    v69 = (unsigned int)v281;
    v70 = 8LL * (unsigned int)v281;
    if ( (_DWORD)v281 )
    {
      do
      {
        v69 = v68 + *(_QWORD *)v67;
        v65 = *(_QWORD *)&v280[v68];
        v68 += 8;
        *(_QWORD *)v69 |= v65;
      }
      while ( v70 != v68 );
    }
    if ( v280 != v40 )
      _libc_free((unsigned __int64)v280);
    sub_2FC1390((__int64)v242, v70, v69, v65, v39, (__int64)v40);
    v33 = v253;
    v36 = v254;
  }
LABEL_116:
  v213 = v37;
  if ( v33 )
    j_j___libc_free_0(v33);
  if ( !v251 )
    _libc_free(v250);
  if ( v246 )
    j_j___libc_free_0(v246);
  if ( !v244 )
    _libc_free(v243);
  if ( v278 )
    j_j___libc_free_0((unsigned __int64)v278);
  if ( !v277 )
    _libc_free(v276);
  if ( v273 )
    j_j___libc_free_0(v273);
  if ( !v272[12] )
    _libc_free(v271);
  if ( v37 )
  {
    v99 = 0;
    if ( a2 )
    {
      do
      {
        while ( 1 )
        {
          v100 = v99;
          if ( *((int *)v236 + v99) <= 1 && *((int *)v239 + v99) <= 1 )
            break;
          ++v99;
          *(_QWORD *)(*(_QWORD *)(a1 + 1344) + 8LL * (v100 >> 6)) |= 1LL << v100;
          if ( v99 == v211 )
            goto LABEL_139;
        }
        ++v99;
      }
      while ( v99 != v211 );
    }
LABEL_139:
    v101 = *(_QWORD *)(a1 + 8);
    v102 = *(_QWORD *)(v101 + 88);
    if ( v102 )
    {
      v103 = *(_QWORD *)(v102 + 240);
      v104 = v103 + ((unsigned __int64)*(unsigned int *)(v102 + 248) << 6);
      if ( v104 != v103 )
      {
        do
        {
          v105 = *(_QWORD *)(v103 + 16);
          for ( m = v105 + 32LL * *(unsigned int *)(v103 + 24); v105 != m; v105 += 32 )
          {
            v107 = *(_DWORD *)(v105 + 8);
            if ( v107 <= 0x7FFFFFFE )
              *(_QWORD *)(*(_QWORD *)(a1 + 1344) + 8LL * (v107 >> 6)) |= 1LL << v107;
          }
          v103 += 64;
        }
        while ( v103 != v104 );
        v101 = *(_QWORD *)(a1 + 8);
      }
    }
    sub_2FC1490(&v280, v101);
    v15 = v259;
    v16 = v256;
    sub_C8CD80((__int64)v256, (__int64)v259, (__int64)&v280, v108, v109, v110);
    v112 = v285;
    v113 = v284;
    v260 = 0;
    v261 = 0;
    v262 = 0;
    v114 = v285 - v284;
    if ( v285 != v284 )
    {
      if ( v114 <= 0x7FFFFFFFFFFFFFF8LL )
      {
        v115 = sub_22077B0(v285 - v284);
        v112 = v285;
        v113 = v284;
        v116 = v115;
        goto LABEL_150;
      }
LABEL_382:
      sub_4261EA(v16, v15, v20);
    }
    v114 = 0;
    v116 = 0;
LABEL_150:
    v260 = v116;
    v261 = v116;
    v262 = v116 + v114;
    if ( v113 != v112 )
    {
      v117 = (__m128i *)v116;
      v118 = (const __m128i *)v113;
      do
      {
        if ( v117 )
        {
          *v117 = _mm_loadu_si128(v118);
          v117[1].m128i_i64[0] = v118[1].m128i_i64[0];
        }
        v118 = (const __m128i *)((char *)v118 + 24);
        v117 = (__m128i *)((char *)v117 + 24);
      }
      while ( (const __m128i *)v112 != v118 );
      v116 += 8 * ((unsigned __int64)(v112 - 24 - v113) >> 3) + 24;
    }
    v261 = v116;
    v16 = v263;
    v15 = v266;
    sub_C8CD80((__int64)v263, (__int64)v266, (__int64)v286, v112, v113, v111);
    v119 = v290;
    v120 = (unsigned __int64)v289;
    v267 = 0;
    v268 = 0;
    v269 = 0;
    v121 = (char *)v290 - (char *)v289;
    if ( v290 == v289 )
    {
      v123 = 0;
    }
    else
    {
      if ( v121 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_382;
      v122 = sub_22077B0((char *)v290 - (char *)v289);
      v119 = v290;
      v120 = (unsigned __int64)v289;
      v123 = v122;
    }
    v267 = v123;
    v268 = v123;
    v269 = v123 + v121;
    if ( (const __m128i *)v120 == v119 )
    {
      v126 = v123;
    }
    else
    {
      v124 = (__m128i *)v123;
      v125 = (const __m128i *)v120;
      do
      {
        if ( v124 )
        {
          *v124 = _mm_loadu_si128(v125);
          v124[1].m128i_i64[0] = v125[1].m128i_i64[0];
        }
        v125 = (const __m128i *)((char *)v125 + 24);
        v124 = (__m128i *)((char *)v124 + 24);
      }
      while ( v119 != v125 );
      v126 = v123 + 8 * (((unsigned __int64)&v119[-2].m128i_u64[1] - v120) >> 3) + 24;
    }
    v268 = v126;
    while ( 2 )
    {
      v127 = v260;
      if ( v261 - v260 == v126 - v123 )
      {
        if ( v260 == v261 )
        {
LABEL_257:
          if ( v123 )
            j_j___libc_free_0(v123);
          if ( !v265 )
            _libc_free(v264);
          if ( v260 )
            j_j___libc_free_0(v260);
          if ( !v258 )
            _libc_free(v257);
          if ( v289 )
            j_j___libc_free_0((unsigned __int64)v289);
          if ( !v288 )
            _libc_free(v287);
          if ( v284 )
            j_j___libc_free_0(v284);
          if ( !s[12] )
            _libc_free(v281);
          goto LABEL_273;
        }
        v173 = v123;
        while ( *(_QWORD *)v127 == *(_QWORD *)v173 )
        {
          v174 = *(_BYTE *)(v127 + 16);
          if ( v174 != *(_BYTE *)(v173 + 16) || v174 && *(_QWORD *)(v127 + 8) != *(_QWORD *)(v173 + 8) )
            break;
          v127 += 24LL;
          v173 += 24LL;
          if ( v261 == v127 )
            goto LABEL_257;
        }
      }
      v128 = *(_DWORD *)(a1 + 72);
      v129 = *(_QWORD *)(v261 - 24);
      v130 = a1 + 48;
      v131 = *(_DWORD *)(a1 + 88);
      if ( v128 )
      {
        v132 = v128 - 1;
        v133 = *(_QWORD *)(a1 + 56);
        v134 = 1;
        v135 = 0;
        v136 = ((unsigned int)v129 >> 9) ^ ((unsigned int)v129 >> 4);
        v137 = v132 & v136;
        v138 = v133 + 16LL * ((unsigned int)v132 & v136);
        v139 = *(_QWORD *)v138;
        if ( v129 == *(_QWORD *)v138 )
          goto LABEL_169;
        while ( v139 != -4096 )
        {
          if ( v139 == -8192 && !v135 )
            v135 = v138;
          v137 = v132 & (v134 + v137);
          v138 = v133 + 16LL * v137;
          v139 = *(_QWORD *)v138;
          if ( v129 == *(_QWORD *)v138 )
            goto LABEL_169;
          ++v134;
        }
        v186 = *(_DWORD *)(a1 + 64);
        if ( v135 )
          v138 = v135;
        ++*(_QWORD *)(a1 + 48);
        v187 = v186 + 1;
        if ( 4 * (v186 + 1) < 3 * v128 )
        {
          v132 = v128 >> 3;
          if ( v128 - *(_DWORD *)(a1 + 68) - v187 <= (unsigned int)v132 )
          {
            sub_2E3ADF0(v130, v128);
            v199 = *(_DWORD *)(a1 + 72);
            if ( !v199 )
              goto LABEL_395;
            v135 = (unsigned int)(v199 - 1);
            v200 = *(_QWORD *)(a1 + 56);
            LODWORD(v201) = v135 & v136;
            v202 = 1;
            v187 = *(_DWORD *)(a1 + 64) + 1;
            v203 = 0;
            v138 = v200 + 16LL * ((unsigned int)v135 & v136);
            v204 = *(_QWORD *)v138;
            if ( v129 != *(_QWORD *)v138 )
            {
              while ( v204 != -4096 )
              {
                if ( !v203 && v204 == -8192 )
                  v203 = v138;
                v132 = (unsigned int)(v202 + 1);
                v201 = (unsigned int)v135 & ((_DWORD)v201 + v202);
                v138 = v200 + 16 * v201;
                v204 = *(_QWORD *)v138;
                if ( v129 == *(_QWORD *)v138 )
                  goto LABEL_328;
                ++v202;
              }
              goto LABEL_360;
            }
          }
          goto LABEL_328;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 48);
      }
      sub_2E3ADF0(v130, 2 * v128);
      v205 = *(_DWORD *)(a1 + 72);
      if ( !v205 )
      {
LABEL_395:
        ++*(_DWORD *)(a1 + 64);
        BUG();
      }
      v135 = (unsigned int)(v205 - 1);
      v206 = *(_QWORD *)(a1 + 56);
      LODWORD(v207) = v135 & (((unsigned int)v129 >> 9) ^ ((unsigned int)v129 >> 4));
      v187 = *(_DWORD *)(a1 + 64) + 1;
      v138 = v206 + 16LL * (unsigned int)v207;
      v208 = *(_QWORD *)v138;
      if ( v129 != *(_QWORD *)v138 )
      {
        v209 = 1;
        v203 = 0;
        while ( v208 != -4096 )
        {
          if ( v208 == -8192 && !v203 )
            v203 = v138;
          v132 = (unsigned int)(v209 + 1);
          v207 = (unsigned int)v135 & ((_DWORD)v207 + v209);
          v138 = v206 + 16 * v207;
          v208 = *(_QWORD *)v138;
          if ( v129 == *(_QWORD *)v138 )
            goto LABEL_328;
          ++v209;
        }
LABEL_360:
        if ( v203 )
          v138 = v203;
      }
LABEL_328:
      *(_DWORD *)(a1 + 64) = v187;
      if ( *(_QWORD *)v138 != -4096 )
        --*(_DWORD *)(a1 + 68);
      *(_QWORD *)v138 = v129;
      *(_DWORD *)(v138 + 8) = 0;
LABEL_169:
      *(_DWORD *)(v138 + 8) = v131;
      v140 = *(unsigned int *)(a1 + 88);
      if ( v140 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 92) )
      {
        sub_C8D5F0(a1 + 80, (const void *)(a1 + 96), v140 + 1, 8u, v132, v135);
        v140 = *(unsigned int *)(a1 + 88);
      }
      v141 = a1 + 16;
      *(_QWORD *)(*(_QWORD *)(a1 + 80) + 8 * v140) = v129;
      v142 = *(unsigned int *)(a1 + 40);
      ++*(_DWORD *)(a1 + 88);
      if ( (_DWORD)v142 )
      {
        v143 = *(_QWORD *)(a1 + 24);
        v144 = 1;
        v145 = 0;
        LODWORD(v146) = (v142 - 1) & (((unsigned int)v129 >> 4) ^ ((unsigned int)v129 >> 9));
        v147 = v143 + 296LL * (unsigned int)v146;
        v148 = *(_QWORD *)v147;
        if ( v129 == *(_QWORD *)v147 )
        {
LABEL_173:
          v149 = *(unsigned int *)(v147 + 16);
          v150 = v147 + 8;
          v151 = *(_DWORD *)(v147 + 72) & 0x3F;
          if ( v151 )
          {
            v142 = ~(-1LL << v151);
            *(_QWORD *)(*(_QWORD *)(v147 + 8) + 8 * v149 - 8) &= v142;
            v149 = *(unsigned int *)(v147 + 16);
          }
          *(_DWORD *)(v147 + 72) = a2;
          LOBYTE(v152) = a2;
          if ( v229 == v149 )
            goto LABEL_178;
          if ( v229 < v149 )
          {
            *(_DWORD *)(v147 + 16) = v230;
LABEL_178:
            v153 = v152 & 0x3F;
            if ( v153 )
            {
              v142 = *(unsigned int *)(v150 + 8);
              v147 = *(_QWORD *)v150;
              *(_QWORD *)(*(_QWORD *)v150 + 8 * v142 - 8) &= ~(-1LL << v153);
            }
            v154 = *(_DWORD *)(v150 + 136) & 0x3F;
            if ( v154 )
            {
              v142 = *(unsigned int *)(v150 + 80);
              v147 = *(_QWORD *)(v150 + 72);
              *(_QWORD *)(v147 + 8 * v142 - 8) &= ~(-1LL << v154);
            }
            LOBYTE(v155) = a2;
            v156 = *(unsigned int *)(v150 + 80);
            *(_DWORD *)(v150 + 136) = a2;
            if ( v229 != v156 )
            {
              if ( v229 >= v156 )
              {
                v185 = v229 - v156;
                if ( v229 > *(unsigned int *)(v150 + 84) )
                {
                  v142 = v150 + 88;
                  sub_C8D5F0(v150 + 72, (const void *)(v150 + 88), v229, 8u, v143, v145);
                  v156 = *(unsigned int *)(v150 + 80);
                }
                v147 = 8 * v185;
                if ( 8 * v185 )
                {
                  v142 = 0;
                  memset((void *)(*(_QWORD *)(v150 + 72) + 8 * v156), 0, v147);
                  LODWORD(v156) = *(_DWORD *)(v150 + 80);
                }
                v155 = *(_DWORD *)(v150 + 136);
                *(_DWORD *)(v150 + 80) = v185 + v156;
              }
              else
              {
                *(_DWORD *)(v150 + 80) = v230;
              }
            }
            v157 = v155 & 0x3F;
            if ( (_DWORD)v157 )
            {
              v142 = *(unsigned int *)(v150 + 80);
              v147 = *(_QWORD *)(v150 + 72);
              *(_QWORD *)(v147 + 8 * v142 - 8) &= ~(-1LL << v157);
            }
            v158 = v129 + 48;
            v270 = v272;
            v271 = 0x400000000LL;
            v159 = *(_QWORD *)(v158 + 8);
            if ( v158 != v159 )
            {
              do
              {
                while ( 1 )
                {
                  v142 = v159;
                  v249[0] = 0;
                  LODWORD(v271) = 0;
                  if ( (unsigned __int8)sub_2FBF8B0(a1, v159, (__int64)&v270, v249) )
                  {
                    v147 = (__int64)v270;
                    if ( v249[0] )
                    {
                      v142 = (__int64)&v270[4 * (unsigned int)v271];
                      if ( (_BYTE *)v142 != v270 )
                      {
                        do
                        {
                          v180 = 1LL << *(_DWORD *)v147;
                          v181 = 8LL * (*(_DWORD *)v147 >> 6);
                          v157 = v181 + *(_QWORD *)(v150 + 72);
                          v143 = *(_QWORD *)v157;
                          if ( (*(_QWORD *)v157 & v180) != 0 )
                          {
                            v145 = ~v180;
                            v143 &= ~v180;
                            *(_QWORD *)v157 = v143;
                          }
                          v147 += 4;
                          *(_QWORD *)(*(_QWORD *)v150 + v181) |= v180;
                        }
                        while ( v147 != v142 );
                      }
                    }
                    else
                    {
                      v147 = 1LL << *(_DWORD *)v270;
                      v160 = 8LL * (*(_DWORD *)v270 >> 6);
                      v157 = v160 + *(_QWORD *)v150;
                      v142 = *(_QWORD *)v157;
                      if ( (*(_QWORD *)v157 & v147) != 0 )
                      {
                        v142 &= ~v147;
                        *(_QWORD *)v157 = v142;
                      }
                      *(_QWORD *)(*(_QWORD *)(v150 + 72) + v160) |= v147;
                    }
                  }
                  if ( !v159 )
                    BUG();
                  if ( (*(_BYTE *)v159 & 4) == 0 )
                    break;
                  v159 = *(_QWORD *)(v159 + 8);
                  if ( v158 == v159 )
                    goto LABEL_198;
                }
                while ( (*(_BYTE *)(v159 + 44) & 8) != 0 )
                  v159 = *(_QWORD *)(v159 + 8);
                v159 = *(_QWORD *)(v159 + 8);
              }
              while ( v158 != v159 );
LABEL_198:
              if ( v270 != v272 )
                _libc_free((unsigned __int64)v270);
            }
            sub_2FC1390((__int64)v256, v142, v147, v157, v143, v145);
            v123 = v267;
            v126 = v268;
            continue;
          }
LABEL_308:
          v184 = v229 - v149;
          if ( v229 > *(unsigned int *)(v150 + 12) )
          {
            v142 = v150 + 16;
            sub_C8D5F0(v150, (const void *)(v150 + 16), v229, 8u, v143, v145);
            v149 = *(unsigned int *)(v150 + 8);
          }
          v147 = 8 * v184;
          if ( 8 * v184 )
          {
            v142 = 0;
            memset((void *)(*(_QWORD *)v150 + 8 * v149), 0, v147);
            LODWORD(v149) = *(_DWORD *)(v150 + 8);
          }
          v152 = *(_DWORD *)(v150 + 64);
          *(_DWORD *)(v150 + 8) = v184 + v149;
          goto LABEL_178;
        }
        while ( v148 != -4096 )
        {
          if ( !v145 && v148 == -8192 )
            v145 = v147;
          v146 = ((_DWORD)v142 - 1) & (unsigned int)(v146 + v144);
          v147 = v143 + 296 * v146;
          v148 = *(_QWORD *)v147;
          if ( v129 == *(_QWORD *)v147 )
            goto LABEL_173;
          ++v144;
        }
        v182 = *(_DWORD *)(a1 + 32);
        if ( v145 )
          v147 = v145;
        ++*(_QWORD *)(a1 + 16);
        v183 = v182 + 1;
        if ( 4 * v183 < (unsigned int)(3 * v142) )
        {
          v143 = (unsigned int)v142 >> 3;
          if ( (int)v142 - *(_DWORD *)(a1 + 36) - v183 <= (unsigned int)v143 )
          {
            sub_2FBFCC0(v141, v142);
            v188 = *(_DWORD *)(a1 + 40);
            if ( !v188 )
              goto LABEL_396;
            v189 = v188 - 1;
            v190 = *(_QWORD *)(a1 + 24);
            v142 = 1;
            v191 = 0;
            v192 = v189 & (((unsigned int)v129 >> 4) ^ ((unsigned int)v129 >> 9));
            v147 = v190 + 296LL * v192;
            v193 = *(_QWORD *)v147;
            v183 = *(_DWORD *)(a1 + 32) + 1;
            if ( v129 != *(_QWORD *)v147 )
            {
              while ( v193 != -4096 )
              {
                if ( !v191 && v193 == -8192 )
                  v191 = v147;
                v143 = (unsigned int)(v142 + 1);
                v210 = v189 & (v192 + (unsigned int)v142);
                v142 = 9 * v210;
                v192 = v210;
                v147 = v190 + 296 * v210;
                v193 = *(_QWORD *)v147;
                if ( v129 == *(_QWORD *)v147 )
                  goto LABEL_304;
                v142 = (unsigned int)v143;
              }
              if ( v191 )
                v147 = v191;
            }
          }
LABEL_304:
          *(_DWORD *)(a1 + 32) = v183;
          if ( *(_QWORD *)v147 != -4096 )
            --*(_DWORD *)(a1 + 36);
          *(_QWORD *)v147 = v129;
          memset((void *)(v147 + 8), 0, 0x120u);
          v150 = v147 + 8;
          *(_QWORD *)(v147 + 8) = v147 + 24;
          *(_QWORD *)(v147 + 80) = v147 + 96;
          *(_QWORD *)(v147 + 152) = v147 + 168;
          *(_QWORD *)(v147 + 224) = v147 + 240;
          *(_QWORD *)(v147 + 16) = 0x600000000LL;
          *(_DWORD *)(v147 + 72) = a2;
          LOBYTE(v152) = a2;
          *(_QWORD *)(v147 + 88) = 0x600000000LL;
          *(_QWORD *)(v147 + 160) = 0x600000000LL;
          *(_QWORD *)(v147 + 232) = 0x600000000LL;
          if ( !v230 )
            goto LABEL_178;
          v149 = 0;
          goto LABEL_308;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 16);
      }
      break;
    }
    v142 = (unsigned int)(2 * v142);
    sub_2FBFCC0(v141, v142);
    v194 = *(_DWORD *)(a1 + 40);
    if ( !v194 )
    {
LABEL_396:
      ++*(_DWORD *)(a1 + 32);
      BUG();
    }
    v195 = v194 - 1;
    v196 = *(_QWORD *)(a1 + 24);
    v197 = v195 & (((unsigned int)v129 >> 9) ^ ((unsigned int)v129 >> 4));
    v147 = v196 + 296LL * v197;
    v143 = *(_QWORD *)v147;
    v183 = *(_DWORD *)(a1 + 32) + 1;
    if ( v129 != *(_QWORD *)v147 )
    {
      v198 = 1;
      v142 = 0;
      while ( v143 != -4096 )
      {
        if ( v143 == -8192 && !v142 )
          v142 = v147;
        v145 = (unsigned int)(v198 + 1);
        v197 = v195 & (v197 + v198);
        v147 = v196 + 296LL * v197;
        v143 = *(_QWORD *)v147;
        if ( v129 == *(_QWORD *)v147 )
          goto LABEL_304;
        ++v198;
      }
      if ( v142 )
        v147 = v142;
    }
    goto LABEL_304;
  }
LABEL_273:
  if ( v239 != v241 )
    _libc_free((unsigned __int64)v239);
  if ( v236 != v238 )
    _libc_free((unsigned __int64)v236);
  v175 = v235;
  if ( v235 )
  {
    v176 = v233;
    v177 = &v233[10 * v235];
    do
    {
      if ( *v176 != -8192 && *v176 != -4096 )
      {
        v178 = v176[1];
        if ( (_QWORD *)v178 != v176 + 3 )
          _libc_free(v178);
      }
      v176 += 10;
    }
    while ( v177 != v176 );
    v175 = v235;
  }
  sub_C7D6A0((__int64)v233, 80 * v175, 8);
  return v213;
}
