// Function: sub_1A70620
// Address: 0x1a70620
//
__int64 __fastcall sub_1A70620(__int64 a1, double a2, double a3, double a4)
{
  int v5; // eax
  __int64 v6; // rdx
  _QWORD *v7; // rax
  _QWORD *i; // rdx
  void *v9; // rdi
  unsigned int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // rbx
  __int64 result; // rax
  __int64 v14; // r14
  unsigned int v15; // esi
  unsigned __int64 v16; // r10
  __int64 v17; // rcx
  unsigned int v18; // ebx
  unsigned int v19; // edx
  unsigned __int64 v20; // rax
  unsigned int v21; // esi
  __int64 v22; // rcx
  unsigned int v23; // edx
  __int64 v24; // rax
  __int64 v25; // r15
  _QWORD *v26; // rax
  _QWORD *v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 *v30; // rcx
  unsigned __int64 v31; // rbx
  unsigned int v32; // esi
  __int64 v33; // r9
  unsigned __int64 v34; // rdi
  unsigned int v35; // edx
  unsigned __int64 *v36; // rax
  unsigned __int64 v37; // rcx
  __int64 v38; // rdi
  unsigned __int64 *v39; // rax
  unsigned __int64 v40; // rsi
  __int64 v41; // rbx
  unsigned __int64 v42; // rax
  _QWORD *v43; // r12
  unsigned __int64 v44; // r14
  int v45; // ebx
  int v46; // r13d
  __int64 v47; // r11
  _BOOL4 v48; // eax
  unsigned __int64 v49; // rdi
  unsigned __int64 v50; // rsi
  __int64 v51; // rcx
  unsigned int v52; // eax
  __int64 *v53; // rdx
  __int64 v54; // r8
  __int64 v55; // rdi
  unsigned int v56; // esi
  __int64 v57; // rdi
  unsigned int v58; // edx
  unsigned __int64 *v59; // rax
  unsigned __int64 v60; // rcx
  __int64 v61; // r8
  unsigned int v62; // eax
  __int64 v63; // rdi
  __int64 v64; // rax
  unsigned __int64 v65; // r13
  int v66; // eax
  unsigned int v67; // r15d
  __int64 v68; // rbx
  unsigned int v69; // esi
  __int64 v70; // r9
  unsigned int v71; // r8d
  __int64 *v72; // rax
  __int64 v73; // rdi
  unsigned __int64 *v74; // r8
  unsigned int v75; // r9d
  unsigned __int64 *v76; // rcx
  int v77; // r10d
  __int64 v78; // r9
  int v79; // r10d
  unsigned int v80; // edx
  __int64 v81; // rcx
  int v82; // r8d
  int v83; // r10d
  __int64 v84; // rdx
  __int64 v85; // rcx
  int v86; // r9d
  int v87; // edx
  __int64 v88; // r9
  __int64 v89; // rax
  unsigned int v90; // esi
  __int64 v91; // rdi
  unsigned int v92; // edx
  __int64 *v93; // rax
  __int64 v94; // rcx
  __int64 *v95; // r10
  int v96; // eax
  int v97; // eax
  int v98; // eax
  __int64 v99; // rdi
  int v100; // r8d
  unsigned __int64 v101; // r10
  int v102; // r11d
  __int64 *v103; // r9
  int v104; // edx
  __int64 *v105; // r11
  int v106; // edi
  int v107; // edi
  int v108; // eax
  int v109; // r8d
  unsigned __int64 v110; // r10
  int v111; // r11d
  int v112; // r9d
  unsigned __int64 v113; // r8
  int v114; // eax
  int v115; // edx
  int v116; // r9d
  unsigned __int64 *v117; // r8
  int v118; // eax
  int v119; // edx
  int v120; // edi
  int v121; // edi
  __int64 v122; // r8
  unsigned int v123; // eax
  __int64 v124; // rsi
  _QWORD *v125; // r10
  int v126; // ecx
  _QWORD *v127; // r9
  int v128; // r13d
  unsigned __int64 *v129; // r10
  int v130; // edx
  int v131; // r11d
  unsigned __int64 *v132; // r9
  int v133; // edx
  int v134; // r11d
  __int64 *v135; // r10
  int v136; // eax
  int v137; // eax
  int v138; // edx
  __int64 v139; // r11
  unsigned int v140; // ecx
  __int64 v141; // r9
  int v142; // r8d
  __int64 *v143; // rsi
  unsigned int v144; // ecx
  _QWORD *v145; // rdi
  unsigned int v146; // eax
  int v147; // eax
  unsigned __int64 v148; // rax
  unsigned __int64 v149; // rax
  int v150; // ebx
  __int64 v151; // r12
  _QWORD *v152; // rax
  __int64 v153; // rdx
  _QWORD *j; // rdx
  int v155; // eax
  int v156; // edx
  __int64 v157; // r11
  int v158; // r8d
  unsigned int v159; // ecx
  __int64 v160; // r9
  int v161; // r8d
  int v162; // r8d
  __int64 v163; // r9
  unsigned int v164; // eax
  unsigned __int64 v165; // rdi
  unsigned __int64 *v166; // r10
  int v167; // esi
  unsigned __int64 *v168; // rcx
  int v169; // eax
  int v170; // r8d
  __int64 v171; // r11
  unsigned int v172; // r9d
  int v173; // r10d
  __int64 *v174; // rsi
  int v175; // r11d
  int v176; // r11d
  __int64 v177; // r10
  unsigned int v178; // ecx
  unsigned __int64 v179; // r8
  int v180; // edi
  unsigned __int64 *v181; // rsi
  int v182; // r11d
  int v183; // r11d
  unsigned __int64 v184; // r10
  unsigned int v185; // ecx
  unsigned __int64 v186; // r8
  int v187; // edi
  unsigned __int64 *v188; // rsi
  int v189; // r10d
  int v190; // r10d
  __int64 v191; // r9
  unsigned __int64 *v192; // rcx
  unsigned int v193; // r12d
  int v194; // esi
  unsigned __int64 v195; // rdi
  int v196; // edi
  int v197; // edi
  __int64 v198; // r8
  unsigned int v199; // ebx
  __int64 v200; // rcx
  _QWORD *v201; // r9
  int v202; // eax
  _QWORD *v203; // rsi
  int v204; // edi
  int v205; // edi
  __int64 v206; // r8
  unsigned int v207; // ebx
  unsigned __int64 v208; // rcx
  unsigned __int64 *v209; // r9
  int v210; // eax
  unsigned __int64 *v211; // rsi
  int v212; // r10d
  int v213; // r10d
  unsigned __int64 v214; // r9
  unsigned __int64 *v215; // rcx
  unsigned int v216; // r12d
  int v217; // esi
  unsigned __int64 v218; // rdi
  int v219; // eax
  int v220; // r8d
  __int64 v221; // r11
  int v222; // r10d
  __int64 *v223; // rsi
  unsigned int v224; // r9d
  __int64 v225; // rdi
  int v226; // r11d
  __int64 *v227; // r10
  int v228; // ecx
  int v229; // edx
  int v230; // r10d
  int v231; // r10d
  __int64 v232; // r8
  __int64 *v233; // rcx
  unsigned int v234; // r12d
  int v235; // esi
  __int64 v236; // rdi
  int v237; // r11d
  int v238; // r11d
  __int64 v239; // r9
  unsigned int v240; // ecx
  __int64 v241; // r8
  int v242; // edi
  __int64 *v243; // rsi
  _QWORD *v244; // rax
  __int64 v245; // [rsp+8h] [rbp-B8h]
  __int64 v246; // [rsp+10h] [rbp-B0h]
  __int64 v247; // [rsp+18h] [rbp-A8h]
  __int64 v248; // [rsp+20h] [rbp-A0h]
  __int64 v249; // [rsp+28h] [rbp-98h]
  __int64 *v250; // [rsp+30h] [rbp-90h]
  int v251; // [rsp+30h] [rbp-90h]
  __int64 *v252; // [rsp+30h] [rbp-90h]
  __int64 v253; // [rsp+40h] [rbp-80h]
  __int64 v254; // [rsp+48h] [rbp-78h]
  __int64 v255; // [rsp+50h] [rbp-70h]
  unsigned __int64 *v256; // [rsp+58h] [rbp-68h]
  __int64 *v257; // [rsp+60h] [rbp-60h]
  unsigned __int64 v258; // [rsp+68h] [rbp-58h]
  unsigned __int64 v259; // [rsp+68h] [rbp-58h]
  __int64 v260; // [rsp+70h] [rbp-50h]
  unsigned __int64 v261; // [rsp+78h] [rbp-48h]
  int v262; // [rsp+78h] [rbp-48h]
  __int64 v263; // [rsp+80h] [rbp-40h] BYREF
  __int64 v264[7]; // [rsp+88h] [rbp-38h] BYREF

  v246 = a1 + 504;
  sub_1A6DA40(a1 + 504);
  ++*(_QWORD *)(a1 + 616);
  v247 = a1 + 616;
  v5 = *(_DWORD *)(a1 + 632);
  if ( !v5 )
  {
    if ( !*(_DWORD *)(a1 + 636) )
      goto LABEL_7;
    v6 = *(unsigned int *)(a1 + 640);
    if ( (unsigned int)v6 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 624));
      *(_QWORD *)(a1 + 624) = 0;
      *(_QWORD *)(a1 + 632) = 0;
      *(_DWORD *)(a1 + 640) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v144 = 4 * v5;
  v6 = *(unsigned int *)(a1 + 640);
  if ( (unsigned int)(4 * v5) < 0x40 )
    v144 = 64;
  if ( (unsigned int)v6 <= v144 )
  {
LABEL_4:
    v7 = *(_QWORD **)(a1 + 624);
    for ( i = &v7[2 * v6]; i != v7; v7 += 2 )
      *v7 = -8;
    *(_QWORD *)(a1 + 632) = 0;
    goto LABEL_7;
  }
  v145 = *(_QWORD **)(a1 + 624);
  v146 = v5 - 1;
  if ( !v146 )
  {
    v151 = 2048;
    v150 = 128;
LABEL_201:
    j___libc_free_0(v145);
    *(_DWORD *)(a1 + 640) = v150;
    v152 = (_QWORD *)sub_22077B0(v151);
    v153 = *(unsigned int *)(a1 + 640);
    *(_QWORD *)(a1 + 632) = 0;
    *(_QWORD *)(a1 + 624) = v152;
    for ( j = &v152[2 * v153]; j != v152; v152 += 2 )
    {
      if ( v152 )
        *v152 = -8;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v146, v146);
  v147 = 1 << (33 - (v146 ^ 0x1F));
  if ( v147 < 64 )
    v147 = 64;
  if ( (_DWORD)v6 != v147 )
  {
    v148 = (4 * v147 / 3u + 1) | ((unsigned __int64)(4 * v147 / 3u + 1) >> 1);
    v149 = ((v148 | (v148 >> 2)) >> 4) | v148 | (v148 >> 2) | ((((v148 | (v148 >> 2)) >> 4) | v148 | (v148 >> 2)) >> 8);
    v150 = (v149 | (v149 >> 16)) + 1;
    v151 = 16 * ((v149 | (v149 >> 16)) + 1);
    goto LABEL_201;
  }
  *(_QWORD *)(a1 + 632) = 0;
  v244 = &v145[2 * (unsigned int)v6];
  do
  {
    if ( v145 )
      *v145 = -8;
    v145 += 2;
  }
  while ( v244 != v145 );
LABEL_7:
  v245 = a1 + 648;
  sub_1A6DA40(a1 + 648);
  ++*(_QWORD *)(a1 + 312);
  v9 = *(void **)(a1 + 328);
  v260 = a1 + 312;
  if ( v9 == *(void **)(a1 + 320) )
  {
LABEL_12:
    *(_QWORD *)(a1 + 340) = 0;
    goto LABEL_13;
  }
  v10 = 4 * (*(_DWORD *)(a1 + 340) - *(_DWORD *)(a1 + 344));
  v11 = *(unsigned int *)(a1 + 336);
  if ( v10 < 0x20 )
    v10 = 32;
  if ( (unsigned int)v11 <= v10 )
  {
    memset(v9, -1, 8 * v11);
    goto LABEL_12;
  }
  sub_16CC920(v260);
LABEL_13:
  v12 = *(_QWORD *)(a1 + 232);
  result = v12 + 8LL * *(unsigned int *)(a1 + 240);
  v248 = v12;
  v253 = result;
  if ( v12 != result )
  {
    v14 = a1;
    while ( 1 )
    {
      v257 = *(__int64 **)(v253 - 8);
      v255 = *(_QWORD *)(*(_QWORD *)(v14 + 200) + 16LL);
      v15 = *(_DWORD *)(v14 + 528);
      v16 = *v257 & 0xFFFFFFFFFFFFFFF8LL;
      v261 = v16;
      if ( !v15 )
        break;
      v17 = *(_QWORD *)(v14 + 512);
      v18 = ((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4);
      v19 = (v15 - 1) & v18;
      v256 = (unsigned __int64 *)(v17 + 40LL * v19);
      v20 = *v256;
      if ( v16 == *v256 )
      {
LABEL_17:
        v21 = *(_DWORD *)(v14 + 672);
        if ( !v21 )
          goto LABEL_150;
        goto LABEL_18;
      }
      v116 = 1;
      v117 = 0;
      while ( v20 != -8 )
      {
        if ( v20 == -16 && !v117 )
          v117 = v256;
        v19 = (v15 - 1) & (v116 + v19);
        v256 = (unsigned __int64 *)(v17 + 40LL * v19);
        v20 = *v256;
        if ( v16 == *v256 )
          goto LABEL_17;
        ++v116;
      }
      v118 = *(_DWORD *)(v14 + 520);
      if ( !v117 )
        v117 = v256;
      ++*(_QWORD *)(v14 + 504);
      v119 = v118 + 1;
      v256 = v117;
      if ( 4 * (v118 + 1) >= 3 * v15 )
        goto LABEL_215;
      if ( v15 - *(_DWORD *)(v14 + 524) - v119 <= v15 >> 3 )
      {
        sub_1A6F390(v246, v15);
        v204 = *(_DWORD *)(v14 + 528);
        if ( !v204 )
        {
LABEL_432:
          ++*(_DWORD *)(v14 + 520);
          BUG();
        }
        v205 = v204 - 1;
        v206 = *(_QWORD *)(v14 + 512);
        v207 = v205 & v18;
        v256 = (unsigned __int64 *)(v206 + 40LL * v207);
        v208 = *v256;
        v119 = *(_DWORD *)(v14 + 520) + 1;
        if ( v261 != *v256 )
        {
          v209 = (unsigned __int64 *)(v206 + 40LL * v207);
          v210 = 1;
          v211 = 0;
          while ( v208 != -8 )
          {
            if ( v208 == -16 && !v211 )
              v211 = v209;
            v207 = v205 & (v210 + v207);
            v209 = (unsigned __int64 *)(v206 + 40LL * v207);
            v208 = *v209;
            if ( v261 == *v209 )
            {
              v256 = (unsigned __int64 *)(v206 + 40LL * v207);
              goto LABEL_147;
            }
            ++v210;
          }
          if ( !v211 )
            v211 = v209;
          v256 = v211;
        }
      }
LABEL_147:
      *(_DWORD *)(v14 + 520) = v119;
      if ( *v256 != -8 )
        --*(_DWORD *)(v14 + 524);
      v256[1] = 0;
      *v256 = v261;
      v256[2] = 0;
      v256[3] = 0;
      *((_DWORD *)v256 + 8) = 0;
      v21 = *(_DWORD *)(v14 + 672);
      if ( !v21 )
      {
LABEL_150:
        ++*(_QWORD *)(v14 + 648);
        goto LABEL_151;
      }
LABEL_18:
      v22 = *(_QWORD *)(v14 + 656);
      v23 = (v21 - 1) & (((unsigned int)v261 >> 9) ^ ((unsigned int)v261 >> 4));
      v258 = v22 + 40LL * v23;
      v24 = *(_QWORD *)v258;
      if ( v261 == *(_QWORD *)v258 )
        goto LABEL_19;
      v112 = 1;
      v113 = 0;
      while ( v24 != -8 )
      {
        if ( !v113 && v24 == -16 )
          v113 = v258;
        v23 = (v21 - 1) & (v112 + v23);
        v258 = v22 + 40LL * v23;
        v24 = *(_QWORD *)v258;
        if ( v261 == *(_QWORD *)v258 )
          goto LABEL_19;
        ++v112;
      }
      v114 = *(_DWORD *)(v14 + 664);
      if ( !v113 )
        v113 = v258;
      ++*(_QWORD *)(v14 + 648);
      v115 = v114 + 1;
      v258 = v113;
      if ( 4 * (v114 + 1) < 3 * v21 )
      {
        if ( v21 - *(_DWORD *)(v14 + 668) - v115 <= v21 >> 3 )
        {
          sub_1A6F390(v245, v21);
          v196 = *(_DWORD *)(v14 + 672);
          if ( !v196 )
          {
LABEL_437:
            ++*(_DWORD *)(v14 + 664);
            BUG();
          }
          v197 = v196 - 1;
          v198 = *(_QWORD *)(v14 + 656);
          v199 = v197 & (((unsigned int)v261 >> 9) ^ ((unsigned int)v261 >> 4));
          v258 = v198 + 40LL * v199;
          v200 = *(_QWORD *)v258;
          v115 = *(_DWORD *)(v14 + 664) + 1;
          if ( v261 != *(_QWORD *)v258 )
          {
            v201 = (_QWORD *)(v198 + 40LL * v199);
            v202 = 1;
            v203 = 0;
            while ( v200 != -8 )
            {
              if ( !v203 && v200 == -16 )
                v203 = v201;
              v199 = v197 & (v202 + v199);
              v201 = (_QWORD *)(v198 + 40LL * v199);
              v200 = *v201;
              if ( v261 == *v201 )
              {
                v258 = v198 + 40LL * v199;
                goto LABEL_138;
              }
              ++v202;
            }
            if ( !v203 )
              v203 = v201;
            v258 = (unsigned __int64)v203;
          }
        }
        goto LABEL_138;
      }
LABEL_151:
      sub_1A6F390(v245, 2 * v21);
      v120 = *(_DWORD *)(v14 + 672);
      if ( !v120 )
        goto LABEL_437;
      v121 = v120 - 1;
      v122 = *(_QWORD *)(v14 + 656);
      v123 = v121 & (((unsigned int)v261 >> 9) ^ ((unsigned int)v261 >> 4));
      v115 = *(_DWORD *)(v14 + 664) + 1;
      v258 = v122 + 40LL * v123;
      v124 = *(_QWORD *)v258;
      if ( v261 != *(_QWORD *)v258 )
      {
        v125 = (_QWORD *)(v122 + 40LL * (v121 & (((unsigned int)v261 >> 9) ^ ((unsigned int)v261 >> 4))));
        v126 = 1;
        v127 = 0;
        while ( v124 != -8 )
        {
          if ( v124 == -16 && !v127 )
            v127 = v125;
          v123 = v121 & (v126 + v123);
          v125 = (_QWORD *)(v122 + 40LL * v123);
          v124 = *v125;
          if ( v261 == *v125 )
          {
            v258 = v122 + 40LL * v123;
            goto LABEL_138;
          }
          ++v126;
        }
        if ( !v127 )
          v127 = v125;
        v258 = (unsigned __int64)v127;
      }
LABEL_138:
      *(_DWORD *)(v14 + 664) = v115;
      if ( *(_QWORD *)v258 != -8 )
        --*(_DWORD *)(v14 + 668);
      *(_QWORD *)(v258 + 8) = 0;
      *(_QWORD *)v258 = v261;
      *(_QWORD *)(v258 + 16) = 0;
      *(_QWORD *)(v258 + 24) = 0;
      *(_DWORD *)(v258 + 32) = 0;
LABEL_19:
      v25 = *(_QWORD *)(v261 + 8);
      if ( v25 )
      {
        while ( 1 )
        {
          v26 = sub_1648700(v25);
          if ( (unsigned __int8)(*((_BYTE *)v26 + 16) - 25) <= 9u )
            break;
          v25 = *(_QWORD *)(v25 + 8);
          if ( !v25 )
            goto LABEL_33;
        }
        v249 = v258 + 8;
        while ( 1 )
        {
          v27 = *(_QWORD **)(v14 + 200);
          v263 = v26[5];
          if ( !(unsigned __int8)sub_1443560(v27, v263) )
            goto LABEL_30;
          v28 = sub_1443F20(v255, v263);
          v29 = *(_QWORD *)(v14 + 200);
          if ( v28 == v29 )
            break;
          do
          {
            v30 = (__int64 *)v28;
            v28 = *(_QWORD *)(v28 + 8);
          }
          while ( v29 != v28 );
          if ( v257 == v30 )
            goto LABEL_30;
          v31 = *v30 & 0xFFFFFFFFFFFFFFF8LL;
          if ( sub_183E920(v260, v31) )
          {
            v32 = *((_DWORD *)v256 + 8);
            v33 = (__int64)(v256 + 1);
            if ( v32 )
            {
              v34 = v256[2];
              v35 = (v32 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
              v36 = (unsigned __int64 *)(v34 + 16LL * v35);
              v37 = *v36;
              if ( v31 == *v36 )
                goto LABEL_29;
              v128 = 1;
              v129 = 0;
              while ( v37 != -8 )
              {
                if ( v37 != -16 || v129 )
                  v36 = v129;
                v35 = (v32 - 1) & (v128 + v35);
                v37 = *(_QWORD *)(v34 + 16LL * v35);
                if ( v31 == v37 )
                {
                  v36 = (unsigned __int64 *)(v34 + 16LL * v35);
                  goto LABEL_29;
                }
                ++v128;
                v129 = v36;
                v36 = (unsigned __int64 *)(v34 + 16LL * v35);
              }
              if ( v129 )
                v36 = v129;
              ++v256[1];
              v130 = *((_DWORD *)v256 + 6) + 1;
              if ( 4 * v130 < 3 * v32 )
              {
                if ( v32 - *((_DWORD *)v256 + 7) - v130 <= v32 >> 3 )
                {
                  sub_141A900(v33, v32);
                  v212 = *((_DWORD *)v256 + 8);
                  if ( !v212 )
                  {
LABEL_436:
                    ++*((_DWORD *)v256 + 6);
                    BUG();
                  }
                  v213 = v212 - 1;
                  v214 = v256[2];
                  v215 = 0;
                  v216 = v213 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
                  v130 = *((_DWORD *)v256 + 6) + 1;
                  v217 = 1;
                  v36 = (unsigned __int64 *)(v214 + 16LL * v216);
                  v218 = *v36;
                  if ( v31 != *v36 )
                  {
                    while ( v218 != -8 )
                    {
                      if ( !v215 && v218 == -16 )
                        v215 = v36;
                      v216 = v213 & (v217 + v216);
                      v36 = (unsigned __int64 *)(v214 + 16LL * v216);
                      v218 = *v36;
                      if ( v31 == *v36 )
                        goto LABEL_164;
                      ++v217;
                    }
                    if ( v215 )
                      v36 = v215;
                  }
                }
                goto LABEL_164;
              }
            }
            else
            {
              ++v256[1];
            }
            sub_141A900(v33, 2 * v32);
            v182 = *((_DWORD *)v256 + 8);
            if ( !v182 )
              goto LABEL_436;
            v183 = v182 - 1;
            v184 = v256[2];
            v185 = v183 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
            v130 = *((_DWORD *)v256 + 6) + 1;
            v36 = (unsigned __int64 *)(v184 + 16LL * v185);
            v186 = *v36;
            if ( v31 != *v36 )
            {
              v187 = 1;
              v188 = 0;
              while ( v186 != -8 )
              {
                if ( v186 == -16 && !v188 )
                  v188 = v36;
                v185 = v183 & (v187 + v185);
                v36 = (unsigned __int64 *)(v184 + 16LL * v185);
                v186 = *v36;
                if ( v31 == *v36 )
                  goto LABEL_164;
                ++v187;
              }
              if ( v188 )
                v36 = v188;
            }
LABEL_164:
            *((_DWORD *)v256 + 6) = v130;
            if ( *v36 != -8 )
              --*((_DWORD *)v256 + 7);
            *v36 = v31;
            v36[1] = 0;
LABEL_29:
            v36[1] = *(_QWORD *)(v14 + 168);
            goto LABEL_30;
          }
          v56 = *(_DWORD *)(v258 + 32);
          if ( v56 )
          {
            v57 = *(_QWORD *)(v258 + 16);
            v58 = (v56 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
            v59 = (unsigned __int64 *)(v57 + 16LL * v58);
            v60 = *v59;
            if ( v31 == *v59 )
              goto LABEL_57;
            v131 = 1;
            v132 = 0;
            while ( v60 != -8 )
            {
              if ( v132 || v60 != -16 )
                v59 = v132;
              v58 = (v56 - 1) & (v131 + v58);
              v60 = *(_QWORD *)(v57 + 16LL * v58);
              if ( v31 == v60 )
              {
                v59 = (unsigned __int64 *)(v57 + 16LL * v58);
                goto LABEL_57;
              }
              ++v131;
              v132 = v59;
              v59 = (unsigned __int64 *)(v57 + 16LL * v58);
            }
            if ( v132 )
              v59 = v132;
            ++*(_QWORD *)(v258 + 8);
            v133 = *(_DWORD *)(v258 + 24) + 1;
            if ( 4 * v133 < 3 * v56 )
            {
              if ( v56 - *(_DWORD *)(v258 + 28) - v133 <= v56 >> 3 )
              {
                sub_141A900(v249, v56);
                v189 = *(_DWORD *)(v258 + 32);
                if ( !v189 )
                {
LABEL_435:
                  ++*(_DWORD *)(v258 + 24);
                  BUG();
                }
                v190 = v189 - 1;
                v191 = *(_QWORD *)(v258 + 16);
                v192 = 0;
                v193 = v190 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
                v133 = *(_DWORD *)(v258 + 24) + 1;
                v194 = 1;
                v59 = (unsigned __int64 *)(v191 + 16LL * v193);
                v195 = *v59;
                if ( v31 != *v59 )
                {
                  while ( v195 != -8 )
                  {
                    if ( v195 == -16 && !v192 )
                      v192 = v59;
                    v193 = v190 & (v194 + v193);
                    v59 = (unsigned __int64 *)(v191 + 16LL * v193);
                    v195 = *v59;
                    if ( v31 == *v59 )
                      goto LABEL_173;
                    ++v194;
                  }
                  if ( v192 )
                    v59 = v192;
                }
              }
              goto LABEL_173;
            }
          }
          else
          {
            ++*(_QWORD *)(v258 + 8);
          }
          sub_141A900(v249, 2 * v56);
          v175 = *(_DWORD *)(v258 + 32);
          if ( !v175 )
            goto LABEL_435;
          v176 = v175 - 1;
          v177 = *(_QWORD *)(v258 + 16);
          v178 = v176 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
          v133 = *(_DWORD *)(v258 + 24) + 1;
          v59 = (unsigned __int64 *)(v177 + 16LL * v178);
          v179 = *v59;
          if ( v31 != *v59 )
          {
            v180 = 1;
            v181 = 0;
            while ( v179 != -8 )
            {
              if ( !v181 && v179 == -16 )
                v181 = v59;
              v178 = v176 & (v180 + v178);
              v59 = (unsigned __int64 *)(v177 + 16LL * v178);
              v179 = *v59;
              if ( v31 == *v59 )
                goto LABEL_173;
              ++v180;
            }
            if ( v181 )
              v59 = v181;
          }
LABEL_173:
          *(_DWORD *)(v258 + 24) = v133;
          if ( *v59 != -8 )
            --*(_DWORD *)(v258 + 28);
          *v59 = v31;
          v59[1] = 0;
LABEL_57:
          v59[1] = *(_QWORD *)(v14 + 176);
LABEL_30:
          v25 = *(_QWORD *)(v25 + 8);
          if ( !v25 )
            goto LABEL_33;
          while ( 1 )
          {
            v26 = sub_1648700(v25);
            if ( (unsigned __int8)(*((_BYTE *)v26 + 16) - 25) <= 9u )
              break;
            v25 = *(_QWORD *)(v25 + 8);
            if ( !v25 )
              goto LABEL_33;
          }
        }
        v42 = sub_157EBA0(v263);
        v254 = v14;
        v43 = (_QWORD *)(v42 - 24);
        v44 = v42;
        v45 = 0;
        v46 = ((*(_DWORD *)(v42 + 20) & 0xFFFFFFF) == 3) + 1;
        while ( 1 )
        {
          while ( 1 )
          {
            if ( *v43 != v261 || !*v43 )
              goto LABEL_42;
            if ( !sub_183E920(v260, v263) )
              break;
            v47 = (__int64)(v256 + 1);
            if ( (*(_DWORD *)(v44 + 20) & 0xFFFFFFF) != 3
              || (v264[0] = *(_QWORD *)(v44 - 24LL * (v45 == 0) - 24),
                  v48 = sub_183E920(v260, v264[0]),
                  v47 = (__int64)(v256 + 1),
                  !v48) )
            {
              v49 = v256[2];
              v50 = *((unsigned int *)v256 + 8);
LABEL_49:
              if ( (_DWORD)v50 )
              {
                v51 = v263;
                v52 = (v50 - 1) & (((unsigned int)v263 >> 9) ^ ((unsigned int)v263 >> 4));
                v53 = (__int64 *)(v49 + 16LL * v52);
                v54 = *v53;
                if ( v263 == *v53 )
                  goto LABEL_51;
                v251 = 1;
                v95 = 0;
                while ( v54 != -8 )
                {
                  if ( !v95 && v54 == -16 )
                    v95 = v53;
                  v52 = (v50 - 1) & (v251 + v52);
                  v53 = (__int64 *)(v49 + 16LL * v52);
                  v54 = *v53;
                  if ( v263 == *v53 )
                    goto LABEL_51;
                  ++v251;
                }
                v96 = *((_DWORD *)v256 + 6);
                if ( v95 )
                  v53 = v95;
                ++v256[1];
                v97 = v96 + 1;
                if ( 4 * v97 < (unsigned int)(3 * v50) )
                {
                  if ( (int)v50 - (v97 + *((_DWORD *)v256 + 7)) <= (unsigned int)v50 >> 3 )
                  {
                    sub_141A900(v47, v50);
                    v108 = *((_DWORD *)v256 + 8);
                    if ( !v108 )
                    {
LABEL_439:
                      ++*((_DWORD *)v256 + 6);
                      BUG();
                    }
                    v99 = v263;
                    v109 = v108 - 1;
                    v110 = v256[2];
                    v103 = 0;
                    v111 = 1;
                    v50 = (v108 - 1) & (((unsigned int)v263 >> 9) ^ ((unsigned int)v263 >> 4));
                    v97 = *((_DWORD *)v256 + 6) + 1;
                    v53 = (__int64 *)(v110 + 16 * v50);
                    v51 = *v53;
                    if ( v263 != *v53 )
                    {
                      while ( v51 != -8 )
                      {
                        if ( v51 == -16 && !v103 )
                          v103 = v53;
                        v50 = v109 & (unsigned int)(v111 + v50);
                        v53 = (__int64 *)(v110 + 16LL * (unsigned int)v50);
                        v51 = *v53;
                        if ( v263 == *v53 )
                          goto LABEL_102;
                        ++v111;
                      }
                      goto LABEL_110;
                    }
                  }
                  goto LABEL_102;
                }
              }
              else
              {
                ++v256[1];
              }
              sub_141A900(v47, 2 * v50);
              v98 = *((_DWORD *)v256 + 8);
              if ( !v98 )
                goto LABEL_439;
              v99 = v263;
              v100 = v98 - 1;
              v101 = v256[2];
              v50 = (v98 - 1) & (((unsigned int)v263 >> 9) ^ ((unsigned int)v263 >> 4));
              v97 = *((_DWORD *)v256 + 6) + 1;
              v53 = (__int64 *)(v101 + 16 * v50);
              v51 = *v53;
              if ( v263 != *v53 )
              {
                v102 = 1;
                v103 = 0;
                while ( v51 != -8 )
                {
                  if ( v51 == -16 && !v103 )
                    v103 = v53;
                  v50 = v100 & (unsigned int)(v102 + v50);
                  v53 = (__int64 *)(v101 + 16LL * (unsigned int)v50);
                  v51 = *v53;
                  if ( v263 == *v53 )
                    goto LABEL_102;
                  ++v102;
                }
LABEL_110:
                v51 = v99;
                if ( v103 )
                  v53 = v103;
              }
LABEL_102:
              *((_DWORD *)v256 + 6) = v97;
              if ( *v53 != -8 )
                --*((_DWORD *)v256 + 7);
              *v53 = v51;
              v53[1] = 0;
LABEL_51:
              if ( (*(_DWORD *)(v44 + 20) & 0xFFFFFFF) == 3 )
              {
                v55 = *(_QWORD *)(v44 - 72);
                if ( v45 )
                  goto LABEL_63;
              }
              else
              {
                v55 = *(_QWORD *)(v254 + 168);
              }
              goto LABEL_53;
            }
            v49 = v256[2];
            v50 = *((unsigned int *)v256 + 8);
            v77 = *(_DWORD *)(v254 + 640);
            if ( v77 )
            {
              v78 = *(_QWORD *)(v254 + 624);
              v79 = v77 - 1;
              v80 = v79 & ((LODWORD(v264[0]) >> 9) ^ (LODWORD(v264[0]) >> 4));
              v81 = *(_QWORD *)(v78 + 16LL * v80);
              if ( v264[0] == v81 )
                goto LABEL_49;
              v82 = 1;
              while ( v81 != -8 )
              {
                v80 = v79 & (v82 + v80);
                v81 = *(_QWORD *)(v78 + 16LL * v80);
                if ( v264[0] == v81 )
                  goto LABEL_49;
                ++v82;
              }
            }
            if ( (_DWORD)v50 )
            {
              v83 = v50 - 1;
              LODWORD(v84) = (v50 - 1) & ((LODWORD(v264[0]) >> 9) ^ (LODWORD(v264[0]) >> 4));
              v85 = *(_QWORD *)(v49 + 16LL * (unsigned int)v84);
              if ( v264[0] == v85 )
                goto LABEL_49;
              v86 = 1;
              while ( v85 != -8 )
              {
                v84 = v83 & (unsigned int)(v84 + v86);
                v85 = *(_QWORD *)(v49 + 16 * v84);
                if ( v264[0] == v85 )
                  goto LABEL_49;
                ++v86;
              }
              v87 = 1;
              v88 = v83 & (((unsigned int)v263 >> 4) ^ ((unsigned int)v263 >> 9));
              v89 = *(_QWORD *)(v49 + 16 * v88);
              if ( v89 == v263 )
                goto LABEL_49;
              while ( v89 != -8 )
              {
                v88 = v83 & (unsigned int)(v88 + v87);
                v89 = *(_QWORD *)(v49 + 16 * v88);
                if ( v89 == v263 )
                  goto LABEL_49;
                ++v87;
              }
            }
            sub_1A703E0(v47, v264)[1] = *(_QWORD *)(v254 + 176);
            sub_1A703E0((__int64)(v256 + 1), &v263)[1] = *(_QWORD *)(v254 + 168);
LABEL_42:
            ++v45;
            v43 -= 3;
            if ( v46 == v45 )
              goto LABEL_54;
          }
          v50 = *(unsigned int *)(v258 + 32);
          if ( (_DWORD)v50 )
          {
            v51 = v263;
            v61 = *(_QWORD *)(v258 + 16);
            v62 = (v50 - 1) & (((unsigned int)v263 >> 9) ^ ((unsigned int)v263 >> 4));
            v53 = (__int64 *)(v61 + 16LL * v62);
            v63 = *v53;
            if ( v263 == *v53 )
              goto LABEL_60;
            v134 = 1;
            v135 = 0;
            while ( v63 != -8 )
            {
              if ( v135 || v63 != -16 )
                v53 = v135;
              v62 = (v50 - 1) & (v134 + v62);
              v252 = (__int64 *)(v61 + 16LL * v62);
              v63 = *v252;
              if ( v263 == *v252 )
              {
                v53 = (__int64 *)(v61 + 16LL * v62);
                goto LABEL_60;
              }
              ++v134;
              v135 = v53;
              v53 = (__int64 *)(v61 + 16LL * v62);
            }
            if ( v135 )
              v53 = v135;
            ++*(_QWORD *)(v258 + 8);
            v136 = *(_DWORD *)(v258 + 24) + 1;
            if ( 4 * v136 < (unsigned int)(3 * v50) )
            {
              if ( (int)v50 - *(_DWORD *)(v258 + 28) - v136 <= (unsigned int)v50 >> 3 )
              {
                sub_141A900(v249, v50);
                v219 = *(_DWORD *)(v258 + 32);
                if ( !v219 )
                {
LABEL_434:
                  ++*(_DWORD *)(v258 + 24);
                  BUG();
                }
                v51 = v263;
                v220 = v219 - 1;
                v221 = *(_QWORD *)(v258 + 16);
                v222 = 1;
                v223 = 0;
                v224 = (v219 - 1) & (((unsigned int)v263 >> 9) ^ ((unsigned int)v263 >> 4));
                v136 = *(_DWORD *)(v258 + 24) + 1;
                v53 = (__int64 *)(v221 + 16LL * v224);
                v225 = *v53;
                if ( *v53 != v263 )
                {
                  while ( v225 != -8 )
                  {
                    if ( v225 == -16 && !v223 )
                      v223 = v53;
                    v224 = v220 & (v222 + v224);
                    v53 = (__int64 *)(v221 + 16LL * v224);
                    v225 = *v53;
                    if ( v263 == *v53 )
                      goto LABEL_182;
                    ++v222;
                  }
                  if ( v223 )
                    v53 = v223;
                }
              }
              goto LABEL_182;
            }
          }
          else
          {
            ++*(_QWORD *)(v258 + 8);
          }
          sub_141A900(v249, 2 * v50);
          v169 = *(_DWORD *)(v258 + 32);
          if ( !v169 )
            goto LABEL_434;
          v170 = v169 - 1;
          v171 = *(_QWORD *)(v258 + 16);
          v172 = (v169 - 1) & (((unsigned int)v263 >> 9) ^ ((unsigned int)v263 >> 4));
          v136 = *(_DWORD *)(v258 + 24) + 1;
          v53 = (__int64 *)(v171 + 16LL * v172);
          v51 = *v53;
          if ( v263 != *v53 )
          {
            v173 = 1;
            v174 = 0;
            while ( v51 != -8 )
            {
              if ( v51 == -16 && !v174 )
                v174 = v53;
              v172 = v170 & (v173 + v172);
              v53 = (__int64 *)(v171 + 16LL * v172);
              v51 = *v53;
              if ( v263 == *v53 )
                goto LABEL_182;
              ++v173;
            }
            v51 = v263;
            if ( v174 )
              v53 = v174;
          }
LABEL_182:
          v50 = v258;
          *(_DWORD *)(v258 + 24) = v136;
          if ( *v53 != -8 )
            --*(_DWORD *)(v258 + 28);
          *v53 = v51;
          v53[1] = 0;
LABEL_60:
          if ( (*(_DWORD *)(v44 + 20) & 0xFFFFFFF) == 3 )
          {
            v55 = *(_QWORD *)(v44 - 72);
            if ( v45 == 1 )
              goto LABEL_53;
LABEL_63:
            v250 = v53;
            v64 = sub_1A6E4A0(v55, v50, a2, a3, a4, (__int64)v53, v51);
            v53 = v250;
            v55 = v64;
            goto LABEL_53;
          }
          v55 = *(_QWORD *)(v254 + 176);
LABEL_53:
          ++v45;
          v53[1] = v55;
          v43 -= 3;
          if ( v46 == v45 )
          {
LABEL_54:
            v14 = v254;
            goto LABEL_30;
          }
        }
      }
LABEL_33:
      v38 = *v257;
      v39 = *(unsigned __int64 **)(v14 + 320);
      v40 = *v257 & 0xFFFFFFFFFFFFFFF8LL;
      if ( *(unsigned __int64 **)(v14 + 328) != v39 )
      {
LABEL_34:
        sub_16CCBA0(v260, v40);
        v38 = *v257;
        goto LABEL_35;
      }
      v74 = &v39[*(unsigned int *)(v14 + 340)];
      v75 = *(_DWORD *)(v14 + 340);
      if ( v39 == v74 )
      {
LABEL_130:
        if ( v75 >= *(_DWORD *)(v14 + 336) )
          goto LABEL_34;
        *(_DWORD *)(v14 + 340) = v75 + 1;
        *v74 = v40;
        ++*(_QWORD *)(v14 + 312);
        v38 = *v257;
      }
      else
      {
        v76 = 0;
        while ( v40 != *v39 )
        {
          if ( *v39 == -2 )
            v76 = v39;
          if ( v74 == ++v39 )
          {
            if ( !v76 )
              goto LABEL_130;
            *v76 = v40;
            --*(_DWORD *)(v14 + 344);
            ++*(_QWORD *)(v14 + 312);
            v38 = *v257;
            break;
          }
        }
      }
LABEL_35:
      if ( (v38 & 4) == 0 )
      {
        v259 = v38 & 0xFFFFFFFFFFFFFFF8LL;
        v65 = sub_157EBA0(v38 & 0xFFFFFFFFFFFFFFF8LL);
        v66 = sub_15F4D60(v65);
        if ( !v66 )
          goto LABEL_37;
        v262 = v66;
        v67 = 0;
        while ( 2 )
        {
          v68 = sub_15F4DF0(v65, v67);
          if ( sub_183E920(v260, v68) )
          {
            v69 = *(_DWORD *)(v14 + 640);
            if ( v69 )
            {
              v70 = *(_QWORD *)(v14 + 624);
              v71 = (v69 - 1) & (((unsigned int)v68 >> 9) ^ ((unsigned int)v68 >> 4));
              v72 = (__int64 *)(v70 + 16LL * v71);
              v73 = *v72;
              if ( v68 == *v72 )
                goto LABEL_70;
              v104 = 1;
              v105 = 0;
              while ( v73 != -8 )
              {
                if ( v105 || v73 != -16 )
                  v72 = v105;
                v71 = (v69 - 1) & (v104 + v71);
                v73 = *(_QWORD *)(v70 + 16LL * v71);
                if ( v68 == v73 )
                {
                  v72 = (__int64 *)(v70 + 16LL * v71);
                  goto LABEL_70;
                }
                ++v104;
                v105 = v72;
                v72 = (__int64 *)(v70 + 16LL * v71);
              }
              v106 = *(_DWORD *)(v14 + 632);
              if ( v105 )
                v72 = v105;
              ++*(_QWORD *)(v14 + 616);
              v107 = v106 + 1;
              if ( 4 * v107 < 3 * v69 )
              {
                if ( v69 - *(_DWORD *)(v14 + 636) - v107 <= v69 >> 3 )
                {
                  sub_1447B20(v247, v69);
                  v155 = *(_DWORD *)(v14 + 640);
                  if ( !v155 )
                  {
LABEL_438:
                    ++*(_DWORD *)(v14 + 632);
                    BUG();
                  }
                  v156 = v155 - 1;
                  v157 = *(_QWORD *)(v14 + 624);
                  v158 = 1;
                  v159 = (v155 - 1) & (((unsigned int)v68 >> 9) ^ ((unsigned int)v68 >> 4));
                  v107 = *(_DWORD *)(v14 + 632) + 1;
                  v143 = 0;
                  v72 = (__int64 *)(v157 + 16LL * v159);
                  v160 = *v72;
                  if ( v68 != *v72 )
                  {
                    while ( v160 != -8 )
                    {
                      if ( !v143 && v160 == -16 )
                        v143 = v72;
                      v159 = v156 & (v158 + v159);
                      v72 = (__int64 *)(v157 + 16LL * v159);
                      v160 = *v72;
                      if ( v68 == *v72 )
                        goto LABEL_119;
                      ++v158;
                    }
                    goto LABEL_190;
                  }
                }
                goto LABEL_119;
              }
            }
            else
            {
              ++*(_QWORD *)(v14 + 616);
            }
            sub_1447B20(v247, 2 * v69);
            v137 = *(_DWORD *)(v14 + 640);
            if ( !v137 )
              goto LABEL_438;
            v138 = v137 - 1;
            v139 = *(_QWORD *)(v14 + 624);
            v140 = (v137 - 1) & (((unsigned int)v68 >> 9) ^ ((unsigned int)v68 >> 4));
            v107 = *(_DWORD *)(v14 + 632) + 1;
            v72 = (__int64 *)(v139 + 16LL * v140);
            v141 = *v72;
            if ( *v72 != v68 )
            {
              v142 = 1;
              v143 = 0;
              while ( v141 != -8 )
              {
                if ( !v143 && v141 == -16 )
                  v143 = v72;
                v140 = v138 & (v142 + v140);
                v72 = (__int64 *)(v139 + 16LL * v140);
                v141 = *v72;
                if ( v68 == *v72 )
                  goto LABEL_119;
                ++v142;
              }
LABEL_190:
              if ( v143 )
                v72 = v143;
            }
LABEL_119:
            *(_DWORD *)(v14 + 632) = v107;
            if ( *v72 != -8 )
              --*(_DWORD *)(v14 + 636);
            *v72 = v68;
            v72[1] = 0;
LABEL_70:
            v72[1] = v259;
          }
          if ( v262 == ++v67 )
            goto LABEL_37;
          continue;
        }
      }
      v41 = v257[4];
      if ( sub_183E920(v260, v41) )
      {
        v90 = *(_DWORD *)(v14 + 640);
        if ( v90 )
        {
          v91 = *(_QWORD *)(v14 + 624);
          v92 = (v90 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
          v93 = (__int64 *)(v91 + 16LL * v92);
          v94 = *v93;
          if ( v41 == *v93 )
          {
LABEL_95:
            v93[1] = *v257 & 0xFFFFFFFFFFFFFFF8LL;
            goto LABEL_37;
          }
          v226 = 1;
          v227 = 0;
          while ( v94 != -8 )
          {
            if ( v94 == -16 && !v227 )
              v227 = v93;
            v92 = (v90 - 1) & (v226 + v92);
            v93 = (__int64 *)(v91 + 16LL * v92);
            v94 = *v93;
            if ( v41 == *v93 )
              goto LABEL_95;
            ++v226;
          }
          v228 = *(_DWORD *)(v14 + 632);
          if ( v227 )
            v93 = v227;
          ++*(_QWORD *)(v14 + 616);
          v229 = v228 + 1;
          if ( 4 * (v228 + 1) < 3 * v90 )
          {
            if ( v90 - *(_DWORD *)(v14 + 636) - v229 <= v90 >> 3 )
            {
              sub_1447B20(v247, v90);
              v230 = *(_DWORD *)(v14 + 640);
              if ( !v230 )
                goto LABEL_433;
              v231 = v230 - 1;
              v232 = *(_QWORD *)(v14 + 624);
              v233 = 0;
              v234 = v231 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
              v229 = *(_DWORD *)(v14 + 632) + 1;
              v235 = 1;
              v93 = (__int64 *)(v232 + 16LL * v234);
              v236 = *v93;
              if ( v41 != *v93 )
              {
                while ( v236 != -8 )
                {
                  if ( !v233 && v236 == -16 )
                    v233 = v93;
                  v234 = v231 & (v235 + v234);
                  v93 = (__int64 *)(v232 + 16LL * v234);
                  v236 = *v93;
                  if ( v41 == *v93 )
                    goto LABEL_314;
                  ++v235;
                }
                if ( v233 )
                  v93 = v233;
              }
            }
LABEL_314:
            *(_DWORD *)(v14 + 632) = v229;
            if ( *v93 != -8 )
              --*(_DWORD *)(v14 + 636);
            *v93 = v41;
            v93[1] = 0;
            goto LABEL_95;
          }
        }
        else
        {
          ++*(_QWORD *)(v14 + 616);
        }
        sub_1447B20(v247, 2 * v90);
        v237 = *(_DWORD *)(v14 + 640);
        if ( !v237 )
        {
LABEL_433:
          ++*(_DWORD *)(v14 + 632);
          BUG();
        }
        v238 = v237 - 1;
        v239 = *(_QWORD *)(v14 + 624);
        v240 = v238 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
        v229 = *(_DWORD *)(v14 + 632) + 1;
        v93 = (__int64 *)(v239 + 16LL * v240);
        v241 = *v93;
        if ( v41 != *v93 )
        {
          v242 = 1;
          v243 = 0;
          while ( v241 != -8 )
          {
            if ( v241 == -16 && !v243 )
              v243 = v93;
            v240 = v238 & (v242 + v240);
            v93 = (__int64 *)(v239 + 16LL * v240);
            v241 = *v93;
            if ( v41 == *v93 )
              goto LABEL_314;
            ++v242;
          }
          if ( v243 )
            v93 = v243;
        }
        goto LABEL_314;
      }
LABEL_37:
      v253 -= 8;
      result = v253;
      if ( v248 == v253 )
        return result;
    }
    ++*(_QWORD *)(v14 + 504);
LABEL_215:
    sub_1A6F390(v246, 2 * v15);
    v161 = *(_DWORD *)(v14 + 528);
    if ( !v161 )
      goto LABEL_432;
    v162 = v161 - 1;
    v163 = *(_QWORD *)(v14 + 512);
    v164 = v162 & (((unsigned int)v261 >> 9) ^ ((unsigned int)v261 >> 4));
    v256 = (unsigned __int64 *)(v163 + 40LL * v164);
    v165 = *v256;
    v119 = *(_DWORD *)(v14 + 520) + 1;
    if ( v261 != *v256 )
    {
      v166 = (unsigned __int64 *)(v163 + 40LL * (v162 & (((unsigned int)v261 >> 9) ^ ((unsigned int)v261 >> 4))));
      v167 = 1;
      v168 = 0;
      while ( v165 != -8 )
      {
        if ( !v168 && v165 == -16 )
          v168 = v166;
        v164 = v162 & (v167 + v164);
        v166 = (unsigned __int64 *)(v163 + 40LL * v164);
        v165 = *v166;
        if ( v261 == *v166 )
        {
          v256 = (unsigned __int64 *)(v163 + 40LL * v164);
          goto LABEL_147;
        }
        ++v167;
      }
      if ( !v168 )
        v168 = v166;
      v256 = v168;
    }
    goto LABEL_147;
  }
  return result;
}
