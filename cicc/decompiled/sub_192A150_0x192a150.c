// Function: sub_192A150
// Address: 0x192a150
//
unsigned __int64 __fastcall sub_192A150(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rsi
  _BYTE *v12; // rsi
  _QWORD *v13; // rdi
  __int64 v14; // rcx
  __int64 v15; // rdx
  unsigned __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rcx
  char v19; // si
  __int64 v20; // rax
  __int64 v21; // rcx
  unsigned __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // rax
  char v27; // si
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rbx
  __int64 v31; // r12
  int v32; // edx
  int v33; // eax
  int v34; // r15d
  int *v35; // r9
  int v36; // r10d
  unsigned __int64 v37; // rax
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // rax
  unsigned __int64 v40; // r13
  unsigned int j; // edx
  int v42; // r8d
  int *v43; // r14
  int v44; // eax
  unsigned int v45; // edx
  __int64 v46; // rax
  __int64 *v47; // rax
  __int64 v48; // r13
  int v49; // r15d
  int v50; // eax
  int v51; // r11d
  int *v52; // r10
  int v53; // r9d
  unsigned __int64 v54; // rcx
  unsigned __int64 v55; // rdx
  unsigned __int64 v56; // r14
  unsigned int kk; // ecx
  int v58; // r8d
  int *v59; // r13
  int v60; // edx
  __int64 v61; // rax
  int v62; // eax
  __int64 v63; // rax
  unsigned int v64; // r13d
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // r15
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // r15
  __int64 v73; // rdx
  __int64 v74; // rax
  int v75; // esi
  int v76; // r10d
  unsigned __int64 v77; // rcx
  unsigned __int64 v78; // rcx
  unsigned int ii; // eax
  int v80; // ecx
  unsigned int v81; // eax
  unsigned int v82; // eax
  int v83; // esi
  int v84; // edx
  int *v85; // r10
  int v86; // r11d
  unsigned __int64 v87; // rax
  unsigned __int64 v88; // rax
  unsigned __int64 v89; // rdi
  unsigned int v90; // eax
  int v91; // edi
  int v92; // esi
  int v93; // r10d
  unsigned __int64 v94; // rcx
  unsigned __int64 v95; // rcx
  unsigned int m; // eax
  int v97; // ecx
  __int64 v98; // rax
  __int64 *v99; // rax
  __int64 v100; // rsi
  char v101; // al
  char v102; // r8
  bool v103; // al
  double v104; // xmm4_8
  double v105; // xmm5_8
  unsigned __int64 v106; // rax
  unsigned __int64 *v107; // r14
  unsigned __int64 v108; // r13
  unsigned __int64 *v109; // r12
  unsigned __int64 v110; // rdi
  __int64 v111; // rbx
  __int64 v112; // r12
  unsigned __int64 v113; // rdi
  __int64 v114; // rbx
  __int64 v115; // r12
  unsigned __int64 v116; // rdi
  __int64 v117; // rbx
  __int64 v118; // r12
  unsigned __int64 v119; // rdi
  __int64 v120; // rbx
  __int64 v121; // r12
  unsigned __int64 v122; // rdi
  __int64 v123; // rbx
  __int64 v124; // r12
  unsigned __int64 v125; // rdi
  __int64 v126; // rbx
  __int64 v127; // r12
  unsigned __int64 v128; // rdi
  unsigned int v130; // esi
  __int64 v131; // r9
  __int64 v132; // rdi
  unsigned int v133; // ecx
  _QWORD *v134; // rax
  __int64 v135; // rdx
  int v136; // r11d
  _QWORD *v137; // r10
  int v138; // edx
  unsigned int v139; // eax
  int *v140; // rdi
  int v141; // r9d
  unsigned __int64 v142; // rsi
  unsigned __int64 v143; // rsi
  unsigned int i; // eax
  int v145; // esi
  unsigned int v146; // eax
  __int64 v147; // rax
  __int64 v148; // rdx
  __int64 v149; // r14
  __int64 v150; // rax
  int v151; // esi
  int v152; // r10d
  unsigned __int64 v153; // rcx
  unsigned __int64 v154; // rcx
  unsigned int n; // eax
  int v156; // ecx
  unsigned int v157; // eax
  int *v158; // r9
  int v159; // r10d
  unsigned __int64 v160; // rdi
  unsigned __int64 v161; // rdi
  unsigned int jj; // edx
  int v163; // edi
  unsigned int v164; // edx
  int v165; // edx
  __int64 v166; // rax
  int v167; // edi
  int *v168; // rsi
  unsigned int k; // r13d
  int v170; // edx
  unsigned int v171; // r13d
  unsigned int v172; // ecx
  int v173; // ecx
  int v174; // ecx
  unsigned int v175; // eax
  int v176; // edx
  int v177; // eax
  int *v178; // rdi
  unsigned int v179; // r14d
  int mm; // r9d
  int v181; // ecx
  unsigned int v182; // r14d
  int v183; // edx
  int v184; // edx
  int v185; // r9d
  int v186; // r9d
  __int64 v187; // r10
  unsigned int v188; // ecx
  __int64 v189; // rdi
  int v190; // r8d
  _QWORD *v191; // rsi
  int v192; // esi
  int v193; // esi
  __int64 v194; // r9
  int v195; // r8d
  unsigned int v196; // ebx
  _QWORD *v197; // rcx
  __int64 v198; // rdi
  __int64 v199; // [rsp+10h] [rbp-320h]
  __int64 v200; // [rsp+18h] [rbp-318h]
  __int64 v201; // [rsp+20h] [rbp-310h]
  int v202; // [rsp+28h] [rbp-308h]
  int v203; // [rsp+28h] [rbp-308h]
  int v204; // [rsp+2Ch] [rbp-304h]
  int *v205; // [rsp+30h] [rbp-300h] BYREF
  unsigned __int64 v206; // [rsp+38h] [rbp-2F8h] BYREF
  __int64 v207; // [rsp+40h] [rbp-2F0h] BYREF
  __int64 v208; // [rsp+48h] [rbp-2E8h]
  __int64 v209; // [rsp+50h] [rbp-2E0h]
  unsigned int v210; // [rsp+58h] [rbp-2D8h]
  __int64 v211; // [rsp+60h] [rbp-2D0h] BYREF
  __int64 v212; // [rsp+68h] [rbp-2C8h]
  __int64 v213; // [rsp+70h] [rbp-2C0h]
  unsigned int v214; // [rsp+78h] [rbp-2B8h]
  __int64 v215; // [rsp+80h] [rbp-2B0h] BYREF
  __int64 v216; // [rsp+88h] [rbp-2A8h]
  __int64 v217; // [rsp+90h] [rbp-2A0h]
  unsigned int v218; // [rsp+98h] [rbp-298h]
  __int64 v219; // [rsp+A0h] [rbp-290h] BYREF
  __int64 v220; // [rsp+A8h] [rbp-288h]
  __int64 v221; // [rsp+B0h] [rbp-280h]
  unsigned int v222; // [rsp+B8h] [rbp-278h]
  __int64 v223; // [rsp+C0h] [rbp-270h] BYREF
  __int64 v224; // [rsp+C8h] [rbp-268h]
  __int64 v225; // [rsp+D0h] [rbp-260h]
  unsigned int v226; // [rsp+D8h] [rbp-258h]
  __int64 v227; // [rsp+E0h] [rbp-250h] BYREF
  __int64 v228; // [rsp+E8h] [rbp-248h]
  __int64 v229; // [rsp+F0h] [rbp-240h]
  unsigned int v230; // [rsp+F8h] [rbp-238h]
  _QWORD v231[2]; // [rsp+100h] [rbp-230h] BYREF
  unsigned __int64 v232; // [rsp+110h] [rbp-220h]
  _BYTE v233[64]; // [rsp+128h] [rbp-208h] BYREF
  __int64 v234; // [rsp+168h] [rbp-1C8h]
  __int64 v235; // [rsp+170h] [rbp-1C0h]
  unsigned __int64 v236; // [rsp+178h] [rbp-1B8h]
  _QWORD v237[2]; // [rsp+180h] [rbp-1B0h] BYREF
  unsigned __int64 v238; // [rsp+190h] [rbp-1A0h]
  _BYTE v239[64]; // [rsp+1A8h] [rbp-188h] BYREF
  __int64 v240; // [rsp+1E8h] [rbp-148h]
  __int64 v241; // [rsp+1F0h] [rbp-140h]
  unsigned __int64 v242; // [rsp+1F8h] [rbp-138h]
  unsigned __int64 *v243; // [rsp+200h] [rbp-130h] BYREF
  __int64 v244; // [rsp+208h] [rbp-128h]
  unsigned __int64 v245[11]; // [rsp+210h] [rbp-120h] BYREF
  __int64 v246; // [rsp+268h] [rbp-C8h]
  __int64 v247; // [rsp+270h] [rbp-C0h]
  __int64 v248; // [rsp+278h] [rbp-B8h]
  char v249[8]; // [rsp+280h] [rbp-B0h] BYREF
  __int64 v250; // [rsp+288h] [rbp-A8h]
  unsigned __int64 v251; // [rsp+290h] [rbp-A0h]
  __int64 v252; // [rsp+2E8h] [rbp-48h]
  __int64 v253; // [rsp+2F0h] [rbp-40h]
  __int64 v254; // [rsp+2F8h] [rbp-38h]

  v11 = *(_QWORD *)(a2 + 80);
  v207 = 0;
  v208 = 0;
  v209 = 0;
  v210 = 0;
  if ( v11 )
    v11 -= 24;
  v211 = 0;
  v212 = 0;
  v213 = 0;
  v214 = 0;
  v215 = 0;
  v216 = 0;
  v217 = 0;
  v218 = 0;
  v219 = 0;
  v220 = 0;
  v221 = 0;
  v222 = 0;
  v223 = 0;
  v224 = 0;
  v225 = 0;
  v226 = 0;
  v227 = 0;
  v228 = 0;
  v229 = 0;
  v230 = 0;
  sub_19235E0(&v243, v11);
  v12 = v233;
  v13 = v231;
  sub_16CCCB0(v231, (__int64)v233, (__int64)&v243);
  v14 = v247;
  v15 = v246;
  v234 = 0;
  v235 = 0;
  v236 = 0;
  v16 = v247 - v246;
  if ( v247 == v246 )
  {
    v17 = 0;
  }
  else
  {
    if ( v16 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_423;
    v17 = sub_22077B0(v247 - v246);
    v14 = v247;
    v15 = v246;
  }
  v234 = v17;
  v235 = v17;
  v236 = v17 + v16;
  if ( v15 == v14 )
  {
    v18 = v17;
  }
  else
  {
    v18 = v17 + v14 - v15;
    do
    {
      if ( v17 )
      {
        *(_QWORD *)v17 = *(_QWORD *)v15;
        v19 = *(_BYTE *)(v15 + 24);
        *(_BYTE *)(v17 + 24) = v19;
        if ( v19 )
        {
          a3 = (__m128)_mm_loadu_si128((const __m128i *)(v15 + 8));
          *(__m128 *)(v17 + 8) = a3;
        }
      }
      v17 += 32;
      v15 += 32;
    }
    while ( v17 != v18 );
  }
  v13 = v237;
  v235 = v18;
  v12 = v239;
  sub_16CCCB0(v237, (__int64)v239, (__int64)v249);
  v20 = v253;
  v21 = v252;
  v240 = 0;
  v241 = 0;
  v242 = 0;
  v22 = v253 - v252;
  if ( v253 == v252 )
  {
    v24 = 0;
    goto LABEL_15;
  }
  if ( v22 > 0x7FFFFFFFFFFFFFE0LL )
LABEL_423:
    sub_4261EA(v13, v12, v15);
  v23 = sub_22077B0(v253 - v252);
  v21 = v252;
  v24 = v23;
  v20 = v253;
LABEL_15:
  v240 = v24;
  v241 = v24;
  v242 = v24 + v22;
  if ( v21 == v20 )
  {
    v26 = v24;
  }
  else
  {
    v25 = v24;
    v26 = v24 + v20 - v21;
    do
    {
      if ( v25 )
      {
        *(_QWORD *)v25 = *(_QWORD *)v21;
        v27 = *(_BYTE *)(v21 + 24);
        *(_BYTE *)(v25 + 24) = v27;
        if ( v27 )
        {
          a4 = (__m128)_mm_loadu_si128((const __m128i *)(v21 + 8));
          *(__m128 *)(v25 + 8) = a4;
        }
      }
      v25 += 32;
      v21 += 32;
    }
    while ( v25 != v26 );
  }
  v241 = v26;
  v200 = a1;
LABEL_22:
  v28 = v234;
  if ( v235 - v234 != v26 - v24 )
  {
LABEL_23:
    v29 = *(_QWORD *)(v235 - 32);
    v30 = *(_QWORD *)(v29 + 48);
    v199 = v29;
    v201 = v29 + 40;
    if ( v30 == v29 + 40 )
      goto LABEL_49;
    v204 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v31 = 0;
        if ( v30 )
          v31 = v30 - 24;
        if ( !(unsigned __int8)sub_14AE440(v31) )
        {
          v130 = *(_DWORD *)(v200 + 352);
          v131 = v200 + 328;
          if ( v130 )
          {
            v132 = *(_QWORD *)(v200 + 336);
            v133 = (v130 - 1) & (((unsigned int)v199 >> 9) ^ ((unsigned int)v199 >> 4));
            v134 = (_QWORD *)(v132 + 8LL * v133);
            v135 = *v134;
            if ( v199 == *v134 )
              goto LABEL_49;
            v136 = 1;
            v137 = 0;
            while ( v135 != -8 )
            {
              if ( !v137 && v135 == -16 )
                v137 = v134;
              v133 = (v130 - 1) & (v136 + v133);
              v134 = (_QWORD *)(v132 + 8LL * v133);
              v135 = *v134;
              if ( v199 == *v134 )
                goto LABEL_49;
              ++v136;
            }
            if ( v137 )
              v134 = v137;
            ++*(_QWORD *)(v200 + 328);
            v138 = *(_DWORD *)(v200 + 344) + 1;
            if ( 4 * v138 < 3 * v130 )
            {
              if ( v130 - *(_DWORD *)(v200 + 348) - v138 > v130 >> 3 )
                goto LABEL_216;
              sub_163D380(v131, v130);
              v192 = *(_DWORD *)(v200 + 352);
              if ( v192 )
              {
                v193 = v192 - 1;
                v194 = *(_QWORD *)(v200 + 336);
                v195 = 1;
                v196 = v193 & (((unsigned int)v199 >> 9) ^ ((unsigned int)v199 >> 4));
                v138 = *(_DWORD *)(v200 + 344) + 1;
                v197 = 0;
                v134 = (_QWORD *)(v194 + 8LL * v196);
                v198 = *v134;
                if ( v199 != *v134 )
                {
                  while ( v198 != -8 )
                  {
                    if ( !v197 && v198 == -16 )
                      v197 = v134;
                    v196 = v193 & (v195 + v196);
                    v134 = (_QWORD *)(v194 + 8LL * v196);
                    v198 = *v134;
                    if ( v199 == *v134 )
                      goto LABEL_216;
                    ++v195;
                  }
                  if ( v197 )
                    v134 = v197;
                }
                goto LABEL_216;
              }
              goto LABEL_436;
            }
          }
          else
          {
            ++*(_QWORD *)(v200 + 328);
          }
          sub_163D380(v131, 2 * v130);
          v185 = *(_DWORD *)(v200 + 352);
          if ( v185 )
          {
            v186 = v185 - 1;
            v187 = *(_QWORD *)(v200 + 336);
            v138 = *(_DWORD *)(v200 + 344) + 1;
            v188 = v186 & (((unsigned int)v199 >> 9) ^ ((unsigned int)v199 >> 4));
            v134 = (_QWORD *)(v187 + 8LL * v188);
            v189 = *v134;
            if ( v199 != *v134 )
            {
              v190 = 1;
              v191 = 0;
              while ( v189 != -8 )
              {
                if ( v189 == -16 && !v191 )
                  v191 = v134;
                v188 = v186 & (v190 + v188);
                v134 = (_QWORD *)(v187 + 8LL * v188);
                v189 = *v134;
                if ( v199 == *v134 )
                  goto LABEL_216;
                ++v190;
              }
              if ( v191 )
                v134 = v191;
            }
LABEL_216:
            *(_DWORD *)(v200 + 344) = v138;
            if ( *v134 != -8 )
              --*(_DWORD *)(v200 + 348);
            *v134 = v199;
            goto LABEL_49;
          }
LABEL_436:
          ++*(_DWORD *)(v200 + 344);
          BUG();
        }
        if ( dword_4FAF140 != -1 )
        {
          if ( dword_4FAF140 <= v204 )
            goto LABEL_49;
          ++v204;
        }
        v32 = *(unsigned __int8 *)(v31 + 16);
        if ( (unsigned int)(v32 - 25) <= 9 )
        {
LABEL_49:
          sub_17D3A30((__int64)v231);
          v24 = v240;
          v26 = v241;
          goto LABEL_22;
        }
        if ( (_BYTE)v32 != 54 )
          break;
        if ( sub_15F32D0(v31) || (*(_BYTE *)(v31 + 18) & 1) != 0 )
          goto LABEL_48;
        v33 = sub_1911FD0(v200, *(_QWORD *)(v31 - 24));
        v34 = v33;
        if ( !v214 )
        {
          ++v211;
LABEL_224:
          sub_1923E80((__int64)&v211, 2 * v214);
          if ( v214 )
          {
            v140 = 0;
            v141 = 1;
            v142 = (((((unsigned __int64)(unsigned int)(37 * v34) << 32) | 0xFFFFFF91) + 0x6EFFFFFFFFLL) >> 22)
                 ^ ((((unsigned __int64)(unsigned int)(37 * v34) << 32) | 0xFFFFFF91) + 0x6EFFFFFFFFLL);
            v143 = ((9 * (((v142 - 1 - (v142 << 13)) >> 8) ^ (v142 - 1 - (v142 << 13)))) >> 15)
                 ^ (9 * (((v142 - 1 - (v142 << 13)) >> 8) ^ (v142 - 1 - (v142 << 13))));
            for ( i = (v214 - 1) & (((v143 - 1 - (v143 << 27)) >> 31) ^ (v143 - 1 - ((_DWORD)v143 << 27)));
                  ;
                  i = (v214 - 1) & v146 )
            {
              v43 = (int *)(v212 + 56LL * i);
              v145 = *v43;
              if ( v34 == *v43 && v43[1] == -3 )
                break;
              if ( v145 == -1 )
              {
                if ( v43[1] == -1 )
                {
                  if ( v140 )
                    v43 = v140;
                  v165 = v213 + 1;
                  goto LABEL_265;
                }
              }
              else if ( v145 == -2 && v43[1] == -2 && !v140 )
              {
                v140 = (int *)(v212 + 56LL * i);
              }
              v146 = v141 + i;
              ++v141;
            }
            goto LABEL_302;
          }
LABEL_434:
          LODWORD(v213) = v213 + 1;
          BUG();
        }
        v35 = 0;
        v36 = 1;
        v37 = (((unsigned __int64)(unsigned int)(37 * v33) << 32) | 0xFFFFFF91) + 0x6EFFFFFFFFLL;
        v38 = ((v37 >> 22) ^ v37) - 1 - (((v37 >> 22) ^ v37) << 13);
        v39 = ((9 * ((v38 >> 8) ^ v38)) >> 15) ^ (9 * ((v38 >> 8) ^ v38));
        v40 = ((v39 - 1 - (v39 << 27)) >> 31) ^ (v39 - 1 - (v39 << 27));
        for ( j = v40 & (v214 - 1); ; j = (v214 - 1) & v45 )
        {
          v42 = j;
          v43 = (int *)(v212 + 56LL * j);
          v44 = *v43;
          if ( v34 == *v43 && v43[1] == -3 )
            goto LABEL_45;
          if ( v44 == -1 )
            break;
          if ( v44 == -2 && v43[1] == -2 && !v35 )
            v35 = (int *)(v212 + 56LL * j);
LABEL_43:
          v45 = v36 + j;
          ++v36;
        }
        if ( v43[1] != -1 )
          goto LABEL_43;
        if ( v35 )
          v43 = v35;
        ++v211;
        v165 = v213 + 1;
        if ( 4 * ((int)v213 + 1) >= 3 * v214 )
          goto LABEL_224;
        if ( v214 - HIDWORD(v213) - v165 <= v214 >> 3 )
        {
          sub_1923E80((__int64)&v211, v214);
          if ( v214 )
          {
            v167 = 1;
            v168 = 0;
            for ( k = (v214 - 1) & v40; ; k = (v214 - 1) & v171 )
            {
              v43 = (int *)(v212 + 56LL * k);
              v170 = *v43;
              if ( v34 == *v43 && v43[1] == -3 )
                break;
              if ( v170 == -1 )
              {
                if ( v43[1] == -1 )
                {
                  if ( v168 )
                    v43 = v168;
                  v165 = v213 + 1;
                  goto LABEL_265;
                }
              }
              else if ( v170 == -2 && v43[1] == -2 && !v168 )
              {
                v168 = (int *)(v212 + 56LL * k);
              }
              v171 = v167 + k;
              ++v167;
            }
LABEL_302:
            v165 = v213 + 1;
            goto LABEL_265;
          }
          goto LABEL_434;
        }
LABEL_265:
        LODWORD(v213) = v165;
        if ( *v43 != -1 || v43[1] != -1 )
          --HIDWORD(v213);
        v47 = (__int64 *)(v43 + 6);
        *v43 = v34;
        v43[1] = -3;
        *((_QWORD *)v43 + 1) = v43 + 6;
        *((_QWORD *)v43 + 2) = 0x400000000LL;
LABEL_47:
        *v47 = v31;
        ++v43[4];
LABEL_48:
        v30 = *(_QWORD *)(v30 + 8);
        if ( v201 == v30 )
          goto LABEL_49;
      }
      if ( (_BYTE)v32 != 55 )
      {
        if ( (_BYTE)v32 == 78 )
        {
          v61 = *(_QWORD *)(v31 - 24);
          if ( !*(_BYTE *)(v61 + 16) && (*(_BYTE *)(v61 + 33) & 0x20) != 0 )
          {
            v62 = *(_DWORD *)(v61 + 36);
            if ( v62 == 191 || v62 == 4 || (unsigned int)(v62 - 35) <= 3 )
              goto LABEL_48;
          }
          if ( (unsigned __int8)sub_15F3040(v31) )
            goto LABEL_49;
          if ( sub_15F3330(v31) )
            goto LABEL_49;
          if ( (unsigned __int8)sub_1560260((_QWORD *)(v31 + 56), -1, 8) )
            goto LABEL_49;
          v63 = *(_QWORD *)(v31 - 24);
          if ( !*(_BYTE *)(v63 + 16) )
          {
            v206 = *(_QWORD *)(v63 + 112);
            if ( (unsigned __int8)sub_1560260(&v206, -1, 8) )
              goto LABEL_49;
          }
          v64 = sub_1911FD0(v200, v31);
          if ( (unsigned __int8)sub_1560260((_QWORD *)(v31 + 56), -1, 36) )
            goto LABEL_106;
          if ( *(char *)(v31 + 23) >= 0 )
            goto LABEL_439;
          v65 = sub_1648A40(v31);
          v67 = v65 + v66;
          v68 = 0;
          if ( *(char *)(v31 + 23) < 0 )
            v68 = sub_1648A40(v31);
          if ( !(unsigned int)((v67 - v68) >> 4) )
          {
LABEL_439:
            v69 = *(_QWORD *)(v31 - 24);
            if ( !*(_BYTE *)(v69 + 16) )
            {
              v206 = *(_QWORD *)(v69 + 112);
              if ( (unsigned __int8)sub_1560260(&v206, -1, 36) )
              {
LABEL_106:
                v92 = v222;
                v206 = v64 | 0xFFFFFFFD00000000LL;
                if ( v222 )
                {
                  v35 = 0;
                  v93 = 1;
                  v94 = (((((unsigned __int64)(37 * v64) << 32) | 0xFFFFFF91) + 0x6EFFFFFFFFLL) >> 22)
                      ^ ((((unsigned __int64)(37 * v64) << 32) | 0xFFFFFF91) + 0x6EFFFFFFFFLL);
                  v95 = ((9 * (((v94 - 1 - (v94 << 13)) >> 8) ^ (v94 - 1 - (v94 << 13)))) >> 15)
                      ^ (9 * (((v94 - 1 - (v94 << 13)) >> 8) ^ (v94 - 1 - (v94 << 13))));
                  for ( m = (v222 - 1) & (((v95 - 1 - (v95 << 27)) >> 31) ^ (v95 - 1 - ((_DWORD)v95 << 27)));
                        ;
                        m = (v222 - 1) & v175 )
                  {
                    v42 = m;
                    v43 = (int *)(v220 + 56LL * m);
                    v97 = *v43;
                    if ( v64 == *v43 && v43[1] == -3 )
                    {
LABEL_45:
                      v46 = (unsigned int)v43[4];
                      if ( (unsigned int)v46 >= v43[5] )
                      {
                        sub_16CD150((__int64)(v43 + 2), v43 + 6, 0, 8, v42, (int)v35);
                        v47 = (__int64 *)(*((_QWORD *)v43 + 1) + 8LL * (unsigned int)v43[4]);
                      }
                      else
                      {
                        v47 = (__int64 *)(*((_QWORD *)v43 + 1) + 8 * v46);
                      }
                      goto LABEL_47;
                    }
                    if ( v97 == -1 )
                    {
                      if ( v43[1] == -1 )
                      {
                        if ( v35 )
                          v43 = v35;
                        ++v219;
                        v176 = v221 + 1;
                        if ( 4 * ((int)v221 + 1) < 3 * v222 )
                        {
                          if ( v222 - HIDWORD(v221) - v176 <= v222 >> 3 )
                          {
LABEL_328:
                            sub_1923E80((__int64)&v219, v92);
                            sub_1923330((__int64)&v219, (int *)&v206, &v205);
                            v43 = v205;
                            v64 = v206;
                            v176 = v221 + 1;
                          }
                          LODWORD(v221) = v176;
                          if ( *v43 != -1 || v43[1] != -1 )
                            --HIDWORD(v221);
                          goto LABEL_331;
                        }
LABEL_327:
                        v92 = 2 * v222;
                        goto LABEL_328;
                      }
                    }
                    else if ( v97 == -2 && v43[1] == -2 && !v35 )
                    {
                      v35 = (int *)(v220 + 56LL * m);
                    }
                    v175 = v93 + m;
                    ++v93;
                  }
                }
                ++v219;
                goto LABEL_327;
              }
            }
          }
          if ( !(unsigned __int8)sub_1560260((_QWORD *)(v31 + 56), -1, 36) )
          {
            if ( *(char *)(v31 + 23) < 0 )
            {
              v70 = sub_1648A40(v31);
              v72 = v70 + v71;
              v73 = 0;
              if ( *(char *)(v31 + 23) < 0 )
                v73 = sub_1648A40(v31);
              if ( (unsigned int)((v72 - v73) >> 4) )
                goto LABEL_440;
            }
            v74 = *(_QWORD *)(v31 - 24);
            if ( *(_BYTE *)(v74 + 16) || (v206 = *(_QWORD *)(v74 + 112), !(unsigned __int8)sub_1560260(&v206, -1, 36)) )
            {
LABEL_440:
              if ( !(unsigned __int8)sub_1560260((_QWORD *)(v31 + 56), -1, 37) )
              {
                if ( *(char *)(v31 + 23) < 0 )
                {
                  v147 = sub_1648A40(v31);
                  v149 = v147 + v148;
                  v150 = *(char *)(v31 + 23) >= 0 ? 0LL : sub_1648A40(v31);
                  if ( v150 != v149 )
                  {
                    while ( *(_DWORD *)(*(_QWORD *)v150 + 8LL) <= 1u )
                    {
                      v150 += 16;
                      if ( v149 == v150 )
                        goto LABEL_270;
                    }
LABEL_239:
                    v151 = v230;
                    v206 = v64 | 0xFFFFFFFD00000000LL;
                    if ( v230 )
                    {
                      v35 = 0;
                      v152 = 1;
                      v153 = (((((unsigned __int64)(37 * v64) << 32) | 0xFFFFFF91) + 0x6EFFFFFFFFLL) >> 22)
                           ^ ((((unsigned __int64)(37 * v64) << 32) | 0xFFFFFF91) + 0x6EFFFFFFFFLL);
                      v154 = ((9 * (((v153 - 1 - (v153 << 13)) >> 8) ^ (v153 - 1 - (v153 << 13)))) >> 15)
                           ^ (9 * (((v153 - 1 - (v153 << 13)) >> 8) ^ (v153 - 1 - (v153 << 13))));
                      for ( n = (v230 - 1) & (((v154 - 1 - (v154 << 27)) >> 31) ^ (v154 - 1 - ((_DWORD)v154 << 27)));
                            ;
                            n = (v230 - 1) & v157 )
                      {
                        v42 = n;
                        v43 = (int *)(v228 + 56LL * n);
                        v156 = *v43;
                        if ( v64 == *v43 && v43[1] == -3 )
                          goto LABEL_45;
                        if ( v156 == -1 )
                        {
                          if ( v43[1] == -1 )
                          {
                            if ( v35 )
                              v43 = v35;
                            ++v227;
                            v183 = v229 + 1;
                            if ( 4 * ((int)v229 + 1) < 3 * v230 )
                            {
                              if ( v230 - HIDWORD(v229) - v183 <= v230 >> 3 )
                              {
LABEL_353:
                                sub_1923E80((__int64)&v227, v151);
                                sub_1923330((__int64)&v227, (int *)&v206, &v205);
                                v43 = v205;
                                v64 = v206;
                                v183 = v229 + 1;
                              }
                              LODWORD(v229) = v183;
                              if ( *v43 != -1 || v43[1] != -1 )
                                --HIDWORD(v229);
LABEL_331:
                              *v43 = v64;
                              v177 = HIDWORD(v206);
                              *((_QWORD *)v43 + 2) = 0x400000000LL;
                              v43[1] = v177;
                              v47 = (__int64 *)(v43 + 6);
                              *((_QWORD *)v43 + 1) = v43 + 6;
                              goto LABEL_47;
                            }
LABEL_352:
                            v151 = 2 * v230;
                            goto LABEL_353;
                          }
                        }
                        else if ( v156 == -2 && v43[1] == -2 && !v35 )
                        {
                          v35 = (int *)(v228 + 56LL * n);
                        }
                        v157 = v152 + n;
                        ++v152;
                      }
                    }
                    ++v227;
                    goto LABEL_352;
                  }
                }
LABEL_270:
                v166 = *(_QWORD *)(v31 - 24);
                if ( *(_BYTE *)(v166 + 16) )
                  goto LABEL_239;
                v206 = *(_QWORD *)(v166 + 112);
                if ( !(unsigned __int8)sub_1560260(&v206, -1, 37) )
                  goto LABEL_239;
              }
            }
          }
          v75 = v226;
          v206 = v64 | 0xFFFFFFFD00000000LL;
          if ( v226 )
          {
            v35 = 0;
            v76 = 1;
            v77 = (((((unsigned __int64)(37 * v64) << 32) | 0xFFFFFF91) + 0x6EFFFFFFFFLL) >> 22)
                ^ ((((unsigned __int64)(37 * v64) << 32) | 0xFFFFFF91) + 0x6EFFFFFFFFLL);
            v78 = ((9 * (((v77 - 1 - (v77 << 13)) >> 8) ^ (v77 - 1 - (v77 << 13)))) >> 15)
                ^ (9 * (((v77 - 1 - (v77 << 13)) >> 8) ^ (v77 - 1 - (v77 << 13))));
            for ( ii = (v226 - 1) & (((v78 - 1 - (v78 << 27)) >> 31) ^ (v78 - 1 - ((_DWORD)v78 << 27)));
                  ;
                  ii = (v226 - 1) & v81 )
            {
              v42 = ii;
              v43 = (int *)(v224 + 56LL * ii);
              v80 = *v43;
              if ( v64 == *v43 && v43[1] == -3 )
                goto LABEL_45;
              if ( v80 == -1 )
              {
                if ( v43[1] == -1 )
                {
                  if ( v35 )
                    v43 = v35;
                  ++v223;
                  v184 = v225 + 1;
                  if ( 4 * ((int)v225 + 1) < 3 * v226 )
                  {
                    if ( v226 - HIDWORD(v225) - v184 <= v226 >> 3 )
                    {
LABEL_360:
                      sub_1923E80((__int64)&v223, v75);
                      sub_1923330((__int64)&v223, (int *)&v206, &v205);
                      v43 = v205;
                      v64 = v206;
                      v184 = v225 + 1;
                    }
                    LODWORD(v225) = v184;
                    if ( *v43 != -1 || v43[1] != -1 )
                      --HIDWORD(v225);
                    goto LABEL_331;
                  }
LABEL_359:
                  v75 = 2 * v226;
                  goto LABEL_360;
                }
              }
              else if ( v80 == -2 && v43[1] == -2 && !v35 )
              {
                v35 = (int *)(v224 + 56LL * ii);
              }
              v81 = v76 + ii;
              ++v76;
            }
          }
          ++v223;
          goto LABEL_359;
        }
        if ( !*(_BYTE *)(v200 + 636) && (_BYTE)v32 == 56 )
          goto LABEL_48;
        v82 = sub_1911FD0(v200, v31);
        v83 = v210;
        v206 = v82 | 0xFFFFFFFD00000000LL;
        v84 = v82;
        if ( !v210 )
        {
          ++v207;
          goto LABEL_293;
        }
        v53 = v210 - 1;
        v85 = 0;
        v86 = 1;
        v87 = (((unsigned __int64)(37 * v82) << 32) | 0xFFFFFF91) + 0x6EFFFFFFFFLL;
        v88 = ((v87 >> 22) ^ v87) - 1 - (((v87 >> 22) ^ v87) << 13);
        v89 = ((9 * ((v88 >> 8) ^ v88)) >> 15) ^ (9 * ((v88 >> 8) ^ v88));
        v90 = (v210 - 1) & (((v89 - 1 - (v89 << 27)) >> 31) ^ (v89 - 1 - ((_DWORD)v89 << 27)));
        while ( 2 )
        {
          v58 = v90;
          v59 = (int *)(v208 + 56LL * v90);
          v91 = *v59;
          if ( v84 == *v59 )
          {
            if ( v59[1] == -3 )
            {
LABEL_116:
              v98 = (unsigned int)v59[4];
              if ( (unsigned int)v98 >= v59[5] )
              {
                sub_16CD150((__int64)(v59 + 2), v59 + 6, 0, 8, v58, v53);
                v99 = (__int64 *)(*((_QWORD *)v59 + 1) + 8LL * (unsigned int)v59[4]);
              }
              else
              {
                v99 = (__int64 *)(*((_QWORD *)v59 + 1) + 8 * v98);
              }
              goto LABEL_118;
            }
            if ( v91 == -1 )
              goto LABEL_221;
LABEL_101:
            if ( v91 == -2 && v59[1] == -2 && !v85 )
              v85 = (int *)(v208 + 56LL * v90);
          }
          else
          {
            if ( v91 != -1 )
              goto LABEL_101;
LABEL_221:
            if ( v59[1] == -1 )
            {
              if ( v85 )
                v59 = v85;
              ++v207;
              v173 = v209 + 1;
              if ( 4 * ((int)v209 + 1) < 3 * v210 )
              {
                if ( v210 - HIDWORD(v209) - v173 <= v210 >> 3 )
                {
LABEL_294:
                  sub_1923E80((__int64)&v207, v83);
                  sub_1923330((__int64)&v207, (int *)&v206, &v205);
                  v59 = v205;
                  v84 = v206;
                  v173 = v209 + 1;
                }
                LODWORD(v209) = v173;
                if ( *v59 != -1 || v59[1] != -1 )
                  --HIDWORD(v209);
                *v59 = v84;
                v50 = HIDWORD(v206);
                goto LABEL_298;
              }
LABEL_293:
              v83 = 2 * v210;
              goto LABEL_294;
            }
          }
          v139 = v86 + v90;
          ++v86;
          v90 = v53 & v139;
          continue;
        }
      }
      if ( sub_15F32D0(v31) || (*(_BYTE *)(v31 + 18) & 1) != 0 )
        goto LABEL_48;
      v48 = *(_QWORD *)(v31 - 48);
      v49 = sub_1911FD0(v200, *(_QWORD *)(v31 - 24));
      v50 = sub_1911FD0(v200, v48);
      if ( !v218 )
      {
        ++v215;
LABEL_249:
        v202 = v50;
        sub_1923E80((__int64)&v215, 2 * v218);
        if ( !v218 )
          goto LABEL_435;
        v50 = v202;
        v158 = 0;
        v159 = 1;
        v160 = ((((unsigned int)(37 * v202) | ((unsigned __int64)(unsigned int)(37 * v49) << 32))
               - 1
               - ((unsigned __int64)(unsigned int)(37 * v202) << 32)) >> 22)
             ^ (((unsigned int)(37 * v202) | ((unsigned __int64)(unsigned int)(37 * v49) << 32))
              - 1
              - ((unsigned __int64)(unsigned int)(37 * v202) << 32));
        v161 = ((9 * (((v160 - 1 - (v160 << 13)) >> 8) ^ (v160 - 1 - (v160 << 13)))) >> 15)
             ^ (9 * (((v160 - 1 - (v160 << 13)) >> 8) ^ (v160 - 1 - (v160 << 13))));
        for ( jj = (v218 - 1) & (((v161 - 1 - (v161 << 27)) >> 31) ^ (v161 - 1 - ((_DWORD)v161 << 27)));
              ;
              jj = (v218 - 1) & v164 )
        {
          v59 = (int *)(v216 + 56LL * jj);
          v163 = *v59;
          if ( v49 == *v59 && v202 == v59[1] )
            break;
          if ( v163 == -1 )
          {
            if ( v59[1] == -1 )
            {
              if ( v158 )
                v59 = v158;
              v174 = v217 + 1;
              goto LABEL_307;
            }
          }
          else if ( v163 == -2 && v59[1] == -2 && !v158 )
          {
            v158 = (int *)(v216 + 56LL * jj);
          }
          v164 = v159 + jj;
          ++v159;
        }
LABEL_306:
        v174 = v217 + 1;
        goto LABEL_307;
      }
      v51 = 1;
      v52 = 0;
      v53 = v216;
      v54 = ((((unsigned int)(37 * v50) | ((unsigned __int64)(unsigned int)(37 * v49) << 32))
            - 1
            - ((unsigned __int64)(unsigned int)(37 * v50) << 32)) >> 22)
          ^ (((unsigned int)(37 * v50) | ((unsigned __int64)(unsigned int)(37 * v49) << 32))
           - 1
           - ((unsigned __int64)(unsigned int)(37 * v50) << 32));
      v55 = ((v54 - 1 - (v54 << 13)) >> 8) ^ (v54 - 1 - (v54 << 13));
      v56 = (((((9 * v55) >> 15) ^ (9 * v55)) - 1 - ((((9 * v55) >> 15) ^ (9 * v55)) << 27)) >> 31)
          ^ ((((9 * v55) >> 15) ^ (9 * v55)) - 1 - ((((9 * v55) >> 15) ^ (9 * v55)) << 27));
      for ( kk = v56 & (v218 - 1); ; kk = (v218 - 1) & v172 )
      {
        v58 = kk;
        v59 = (int *)(v216 + 56LL * kk);
        v60 = *v59;
        if ( v49 == *v59 && v50 == v59[1] )
          goto LABEL_116;
        if ( v60 == -1 )
          break;
        if ( v60 == -2 && v59[1] == -2 && !v52 )
          v52 = (int *)(v216 + 56LL * kk);
LABEL_291:
        v172 = v51 + kk;
        ++v51;
      }
      if ( v59[1] != -1 )
        goto LABEL_291;
      if ( v52 )
        v59 = v52;
      ++v215;
      v174 = v217 + 1;
      if ( 4 * ((int)v217 + 1) >= 3 * v218 )
        goto LABEL_249;
      if ( v218 - HIDWORD(v217) - v174 <= v218 >> 3 )
      {
        v203 = v50;
        sub_1923E80((__int64)&v215, v218);
        if ( v218 )
        {
          v50 = v203;
          v178 = 0;
          v179 = (v218 - 1) & v56;
          for ( mm = 1; ; ++mm )
          {
            v59 = (int *)(v216 + 56LL * v179);
            v181 = *v59;
            if ( v49 == *v59 && v203 == v59[1] )
              break;
            if ( v181 == -1 )
            {
              if ( v59[1] == -1 )
              {
                if ( v178 )
                  v59 = v178;
                v174 = v217 + 1;
                goto LABEL_307;
              }
            }
            else if ( v181 == -2 && v59[1] == -2 && !v178 )
            {
              v178 = (int *)(v216 + 56LL * v179);
            }
            v182 = mm + v179;
            v179 = (v218 - 1) & v182;
          }
          goto LABEL_306;
        }
LABEL_435:
        LODWORD(v217) = v217 + 1;
        BUG();
      }
LABEL_307:
      LODWORD(v217) = v174;
      if ( *v59 != -1 || v59[1] != -1 )
        --HIDWORD(v217);
      *v59 = v49;
LABEL_298:
      v59[1] = v50;
      v99 = (__int64 *)(v59 + 6);
      *((_QWORD *)v59 + 1) = v59 + 6;
      *((_QWORD *)v59 + 2) = 0x400000000LL;
LABEL_118:
      *v99 = v31;
      ++v59[4];
      v30 = *(_QWORD *)(v30 + 8);
      if ( v201 == v30 )
        goto LABEL_49;
    }
  }
  if ( v234 != v235 )
  {
    v100 = v24;
    while ( *(_QWORD *)v28 == *(_QWORD *)v100 )
    {
      v101 = *(_BYTE *)(v28 + 24);
      v102 = *(_BYTE *)(v100 + 24);
      if ( v101 && v102 )
        v103 = *(_DWORD *)(v28 + 16) == *(_DWORD *)(v100 + 16);
      else
        v103 = v101 == v102;
      if ( !v103 )
        break;
      v28 += 32;
      v100 += 32;
      if ( v235 == v28 )
        goto LABEL_129;
    }
    goto LABEL_23;
  }
LABEL_129:
  if ( v24 )
    j_j___libc_free_0(v24, v242 - v24);
  if ( v238 != v237[1] )
    _libc_free(v238);
  if ( v234 )
    j_j___libc_free_0(v234, v236 - v234);
  if ( v232 != v231[1] )
    _libc_free(v232);
  if ( v252 )
    j_j___libc_free_0(v252, v254 - v252);
  if ( v251 != v250 )
    _libc_free(v251);
  if ( v246 )
    j_j___libc_free_0(v246, v248 - v246);
  if ( v245[0] != v244 )
    _libc_free(v245[0]);
  v244 = 0x400000000LL;
  v243 = v245;
  sub_1928BF0(v200, (__int64)&v207, (__int64)&v243, 1);
  sub_1928BF0(v200, (__int64)&v211, (__int64)&v243, 2);
  sub_1928BF0(v200, (__int64)&v215, (__int64)&v243, 3);
  sub_1928BF0(v200, (__int64)&v219, (__int64)&v243, 1);
  sub_1928BF0(v200, (__int64)&v223, (__int64)&v243, 2);
  sub_1928BF0(v200, (__int64)&v227, (__int64)&v243, 3);
  v106 = sub_19241A0(v200, (__int64)&v243, a3, *(double *)a4.m128_u64, a5, a6, v104, v105, a9, a10);
  v107 = v243;
  v108 = v106;
  v109 = &v243[7 * (unsigned int)v244];
  if ( v243 != v109 )
  {
    do
    {
      v109 -= 7;
      v110 = v109[1];
      if ( (unsigned __int64 *)v110 != v109 + 3 )
        _libc_free(v110);
    }
    while ( v107 != v109 );
    v109 = v243;
  }
  if ( v109 != v245 )
    _libc_free((unsigned __int64)v109);
  if ( v230 )
  {
    v111 = v228;
    v112 = v228 + 56LL * v230;
    do
    {
      while ( *(_DWORD *)v111 == -1 )
      {
        if ( *(_DWORD *)(v111 + 4) != -1 )
          goto LABEL_155;
        v111 += 56;
        if ( v112 == v111 )
          goto LABEL_161;
      }
      if ( *(_DWORD *)v111 != -2 || *(_DWORD *)(v111 + 4) != -2 )
      {
LABEL_155:
        v113 = *(_QWORD *)(v111 + 8);
        if ( v113 != v111 + 24 )
          _libc_free(v113);
      }
      v111 += 56;
    }
    while ( v112 != v111 );
  }
LABEL_161:
  j___libc_free_0(v228);
  if ( v226 )
  {
    v114 = v224;
    v115 = v224 + 56LL * v226;
    do
    {
      while ( *(_DWORD *)v114 == -1 )
      {
        if ( *(_DWORD *)(v114 + 4) != -1 )
          goto LABEL_164;
        v114 += 56;
        if ( v115 == v114 )
          goto LABEL_170;
      }
      if ( *(_DWORD *)v114 != -2 || *(_DWORD *)(v114 + 4) != -2 )
      {
LABEL_164:
        v116 = *(_QWORD *)(v114 + 8);
        if ( v116 != v114 + 24 )
          _libc_free(v116);
      }
      v114 += 56;
    }
    while ( v115 != v114 );
  }
LABEL_170:
  j___libc_free_0(v224);
  if ( v222 )
  {
    v117 = v220;
    v118 = v220 + 56LL * v222;
    do
    {
      while ( *(_DWORD *)v117 == -1 )
      {
        if ( *(_DWORD *)(v117 + 4) != -1 )
          goto LABEL_173;
        v117 += 56;
        if ( v118 == v117 )
          goto LABEL_179;
      }
      if ( *(_DWORD *)v117 != -2 || *(_DWORD *)(v117 + 4) != -2 )
      {
LABEL_173:
        v119 = *(_QWORD *)(v117 + 8);
        if ( v119 != v117 + 24 )
          _libc_free(v119);
      }
      v117 += 56;
    }
    while ( v118 != v117 );
  }
LABEL_179:
  j___libc_free_0(v220);
  if ( v218 )
  {
    v120 = v216;
    v121 = v216 + 56LL * v218;
    do
    {
      while ( *(_DWORD *)v120 == -1 )
      {
        if ( *(_DWORD *)(v120 + 4) != -1 )
          goto LABEL_182;
        v120 += 56;
        if ( v121 == v120 )
          goto LABEL_188;
      }
      if ( *(_DWORD *)v120 != -2 || *(_DWORD *)(v120 + 4) != -2 )
      {
LABEL_182:
        v122 = *(_QWORD *)(v120 + 8);
        if ( v122 != v120 + 24 )
          _libc_free(v122);
      }
      v120 += 56;
    }
    while ( v121 != v120 );
  }
LABEL_188:
  j___libc_free_0(v216);
  if ( v214 )
  {
    v123 = v212;
    v124 = v212 + 56LL * v214;
    do
    {
      while ( *(_DWORD *)v123 == -1 )
      {
        if ( *(_DWORD *)(v123 + 4) != -1 )
          goto LABEL_191;
        v123 += 56;
        if ( v124 == v123 )
          goto LABEL_197;
      }
      if ( *(_DWORD *)v123 != -2 || *(_DWORD *)(v123 + 4) != -2 )
      {
LABEL_191:
        v125 = *(_QWORD *)(v123 + 8);
        if ( v125 != v123 + 24 )
          _libc_free(v125);
      }
      v123 += 56;
    }
    while ( v124 != v123 );
  }
LABEL_197:
  j___libc_free_0(v212);
  if ( v210 )
  {
    v126 = v208;
    v127 = v208 + 56LL * v210;
    do
    {
      while ( *(_DWORD *)v126 == -1 )
      {
        if ( *(_DWORD *)(v126 + 4) != -1 )
          goto LABEL_200;
        v126 += 56;
        if ( v127 == v126 )
          goto LABEL_206;
      }
      if ( *(_DWORD *)v126 != -2 || *(_DWORD *)(v126 + 4) != -2 )
      {
LABEL_200:
        v128 = *(_QWORD *)(v126 + 8);
        if ( v128 != v126 + 24 )
          _libc_free(v128);
      }
      v126 += 56;
    }
    while ( v127 != v126 );
  }
LABEL_206:
  j___libc_free_0(v208);
  return v108;
}
