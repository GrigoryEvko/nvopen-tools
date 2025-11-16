// Function: sub_27A9280
// Address: 0x27a9280
//
__int64 __fastcall sub_27A9280(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  _BYTE *v4; // rsi
  char *v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r8
  __int64 v10; // r9
  const __m128i *v11; // rcx
  const __m128i *v12; // rdx
  unsigned __int64 v13; // rbx
  __m128i *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  const __m128i *v18; // rax
  const __m128i *v19; // rcx
  unsigned __int64 v20; // rbx
  __int64 v21; // rax
  unsigned __int64 v22; // rdi
  __m128i *v23; // rdx
  __m128i *v24; // rax
  unsigned __int64 v25; // rcx
  __int64 v26; // rax
  __int64 v27; // r12
  int v28; // r13d
  char *v29; // rbx
  int v30; // edx
  int v31; // eax
  __int64 v32; // rdx
  int v33; // r15d
  __int64 v34; // r9
  int v35; // r11d
  unsigned __int64 v36; // rcx
  unsigned __int64 v37; // r14
  int *v38; // rax
  unsigned int nn; // edi
  __int64 v40; // r8
  int v41; // ecx
  unsigned int v42; // edi
  __int64 v43; // rax
  int *v44; // r14
  unsigned __int64 v45; // rdx
  __int64 v46; // r14
  unsigned int v47; // eax
  int v48; // r11d
  unsigned __int64 v49; // rcx
  unsigned __int64 v50; // r14
  unsigned int i; // edi
  int v52; // ecx
  __int64 v53; // rax
  int v54; // r14d
  __int64 v55; // r9
  __int64 v56; // r8
  unsigned int ii; // eax
  __int64 v58; // r15
  int v59; // edx
  int v60; // eax
  int v61; // eax
  int v62; // r15d
  int v63; // r10d
  unsigned __int64 v64; // r14
  int *v65; // rax
  unsigned int v66; // ecx
  int v67; // edx
  char v68; // al
  __int64 *v69; // rdi
  __int64 v70; // r8
  __int64 v71; // r9
  unsigned __int64 v72; // rax
  char v73; // si
  __int64 v74; // r8
  __int64 v75; // r9
  __int64 v76; // r8
  __int64 v77; // r9
  __int64 v78; // r8
  __int64 v79; // r9
  __int64 v80; // r8
  __int64 v81; // r9
  __int64 v82; // r8
  __int64 v83; // r9
  __int64 v84; // rax
  __int64 v85; // r14
  __int64 v86; // r13
  unsigned __int64 v87; // r12
  unsigned __int64 v88; // rdi
  unsigned int v89; // eax
  __int64 v90; // rbx
  __int64 v91; // r12
  unsigned __int64 v92; // rdi
  unsigned int v93; // eax
  __int64 v94; // rbx
  __int64 v95; // r12
  unsigned __int64 v96; // rdi
  unsigned int v97; // eax
  __int64 v98; // rbx
  __int64 v99; // r12
  unsigned __int64 v100; // rdi
  unsigned int v101; // eax
  __int64 v102; // rbx
  __int64 v103; // r12
  unsigned __int64 v104; // rdi
  unsigned int v105; // eax
  __int64 v106; // rbx
  __int64 v107; // r12
  unsigned __int64 v108; // rdi
  unsigned int v109; // eax
  __int64 v110; // rbx
  __int64 v111; // r12
  unsigned __int64 v112; // rdi
  int v114; // eax
  unsigned int v115; // esi
  __int64 v116; // r9
  __int64 v117; // rdi
  unsigned int v118; // ebx
  unsigned int v119; // eax
  _QWORD *v120; // rcx
  __int64 v121; // rdx
  int v122; // r11d
  _QWORD *v123; // r10
  int v124; // edx
  unsigned int v125; // ecx
  int v126; // r9d
  int *v127; // r8
  unsigned int mm; // ecx
  int v129; // r10d
  unsigned int v130; // ecx
  int v131; // r9d
  int *v132; // r8
  unsigned int k; // ecx
  int v134; // r10d
  unsigned int v135; // ecx
  int v136; // edi
  int v137; // edi
  unsigned int i1; // r14d
  int v139; // r9d
  unsigned int v140; // r14d
  unsigned int v141; // edi
  int v142; // r8d
  int *v143; // rdi
  unsigned int n; // edx
  int v145; // r9d
  unsigned int v146; // edx
  int v147; // edi
  int v148; // ecx
  int v149; // edx
  int v150; // edi
  __int64 v151; // rcx
  unsigned int kk; // eax
  int v153; // r8d
  unsigned int v154; // eax
  int v155; // edi
  unsigned int j; // r14d
  int v157; // r9d
  unsigned int v158; // r14d
  int v159; // r9d
  int v160; // r9d
  __int64 v161; // r10
  unsigned int v162; // eax
  __int64 v163; // rdi
  int v164; // r8d
  _QWORD *v165; // rsi
  int v166; // r8d
  int v167; // r8d
  __int64 v168; // r10
  unsigned int v169; // eax
  int v170; // r9d
  __int64 v171; // rdi
  int v172; // esi
  int *v173; // rdx
  unsigned int m; // r14d
  int v175; // r8d
  unsigned int v176; // r14d
  int v177; // esi
  unsigned int jj; // r10d
  int *v179; // rax
  int v180; // edi
  unsigned int v181; // r10d
  __int64 v182; // [rsp+10h] [rbp-320h]
  __int64 v183; // [rsp+10h] [rbp-320h]
  __int64 v184; // [rsp+10h] [rbp-320h]
  __int64 v185; // [rsp+10h] [rbp-320h]
  __int64 v186; // [rsp+10h] [rbp-320h]
  __int64 v187; // [rsp+28h] [rbp-308h]
  __int64 v188; // [rsp+30h] [rbp-300h]
  __int64 v189; // [rsp+38h] [rbp-2F8h]
  int v190; // [rsp+40h] [rbp-2F0h] BYREF
  __int64 v191; // [rsp+48h] [rbp-2E8h]
  __int64 v192; // [rsp+50h] [rbp-2E0h] BYREF
  __int64 v193; // [rsp+58h] [rbp-2D8h]
  __int64 v194; // [rsp+60h] [rbp-2D0h]
  unsigned int v195; // [rsp+68h] [rbp-2C8h]
  __int64 v196; // [rsp+70h] [rbp-2C0h] BYREF
  __int64 v197; // [rsp+78h] [rbp-2B8h]
  __int64 v198; // [rsp+80h] [rbp-2B0h]
  unsigned int v199; // [rsp+88h] [rbp-2A8h]
  __int64 v200; // [rsp+90h] [rbp-2A0h] BYREF
  __int64 v201; // [rsp+98h] [rbp-298h]
  __int64 v202; // [rsp+A0h] [rbp-290h]
  unsigned int v203; // [rsp+A8h] [rbp-288h]
  __int64 v204; // [rsp+B0h] [rbp-280h] BYREF
  __int64 v205; // [rsp+B8h] [rbp-278h]
  __int64 v206; // [rsp+C0h] [rbp-270h]
  unsigned int v207; // [rsp+C8h] [rbp-268h]
  __int64 v208; // [rsp+D0h] [rbp-260h] BYREF
  __int64 v209; // [rsp+D8h] [rbp-258h]
  __int64 v210; // [rsp+E0h] [rbp-250h]
  unsigned int v211; // [rsp+E8h] [rbp-248h]
  __int64 v212; // [rsp+F0h] [rbp-240h] BYREF
  __int64 v213; // [rsp+F8h] [rbp-238h]
  __int64 v214; // [rsp+100h] [rbp-230h]
  unsigned int v215; // [rsp+108h] [rbp-228h]
  char v216[8]; // [rsp+110h] [rbp-220h] BYREF
  unsigned __int64 v217; // [rsp+118h] [rbp-218h]
  char v218; // [rsp+12Ch] [rbp-204h]
  _BYTE v219[64]; // [rsp+130h] [rbp-200h] BYREF
  __m128i *v220; // [rsp+170h] [rbp-1C0h]
  __int64 v221; // [rsp+178h] [rbp-1B8h]
  __int8 *v222; // [rsp+180h] [rbp-1B0h]
  char v223[8]; // [rsp+190h] [rbp-1A0h] BYREF
  unsigned __int64 v224; // [rsp+198h] [rbp-198h]
  char v225; // [rsp+1ACh] [rbp-184h]
  _BYTE v226[64]; // [rsp+1B0h] [rbp-180h] BYREF
  unsigned __int64 v227; // [rsp+1F0h] [rbp-140h]
  unsigned __int64 v228; // [rsp+1F8h] [rbp-138h]
  unsigned __int64 v229; // [rsp+200h] [rbp-130h]
  _BYTE *v230; // [rsp+210h] [rbp-120h] BYREF
  unsigned __int64 v231; // [rsp+218h] [rbp-118h]
  _BYTE v232[80]; // [rsp+220h] [rbp-110h] BYREF
  const __m128i *v233; // [rsp+270h] [rbp-C0h]
  const __m128i *v234; // [rsp+278h] [rbp-B8h]
  char v235[8]; // [rsp+288h] [rbp-A8h] BYREF
  unsigned __int64 v236; // [rsp+290h] [rbp-A0h]
  char v237; // [rsp+2A4h] [rbp-8Ch]
  const __m128i *v238; // [rsp+2E8h] [rbp-48h]
  const __m128i *v239; // [rsp+2F0h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 80);
  v192 = 0;
  v193 = 0;
  v194 = 0;
  v195 = 0;
  if ( v3 )
    v3 -= 24;
  v196 = 0;
  v197 = 0;
  v198 = 0;
  v199 = 0;
  v200 = 0;
  v201 = 0;
  v202 = 0;
  v203 = 0;
  v204 = 0;
  v205 = 0;
  v206 = 0;
  v207 = 0;
  v208 = 0;
  v209 = 0;
  v210 = 0;
  v211 = 0;
  v212 = 0;
  v213 = 0;
  v214 = 0;
  v215 = 0;
  sub_27A4830(&v230, v3);
  v4 = v219;
  v5 = v216;
  sub_C8CD80((__int64)v216, (__int64)v219, (__int64)&v230, v6, v7, v8);
  v11 = v234;
  v12 = v233;
  v220 = 0;
  v221 = 0;
  v222 = 0;
  v13 = (char *)v234 - (char *)v233;
  if ( v234 == v233 )
  {
    v13 = 0;
    v14 = 0;
  }
  else
  {
    if ( v13 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_411;
    v14 = (__m128i *)sub_22077B0((char *)v234 - (char *)v233);
    v11 = v234;
    v12 = v233;
  }
  v220 = v14;
  v221 = (__int64)v14;
  v222 = &v14->m128i_i8[v13];
  if ( v11 == v12 )
  {
    v15 = (__int64)v14;
  }
  else
  {
    v15 = (__int64)v14->m128i_i64 + (char *)v11 - (char *)v12;
    do
    {
      if ( v14 )
      {
        *v14 = _mm_loadu_si128(v12);
        v14[1] = _mm_loadu_si128(v12 + 1);
      }
      v14 += 2;
      v12 += 2;
    }
    while ( v14 != (__m128i *)v15 );
  }
  v5 = v223;
  v221 = v15;
  v4 = v226;
  sub_C8CD80((__int64)v223, (__int64)v226, (__int64)v235, v15, v9, v10);
  v18 = v239;
  v19 = v238;
  v227 = 0;
  v228 = 0;
  v229 = 0;
  v20 = (char *)v239 - (char *)v238;
  if ( v239 != v238 )
  {
    if ( v20 <= 0x7FFFFFFFFFFFFFE0LL )
    {
      v21 = sub_22077B0((char *)v239 - (char *)v238);
      v19 = v238;
      v22 = v21;
      v18 = v239;
      goto LABEL_14;
    }
LABEL_411:
    sub_4261EA(v5, v4, v12);
  }
  v20 = 0;
  v22 = 0;
LABEL_14:
  v227 = v22;
  v228 = v22;
  v229 = v22 + v20;
  if ( v19 == v18 )
  {
    v24 = (__m128i *)v22;
  }
  else
  {
    v23 = (__m128i *)v22;
    v24 = (__m128i *)(v22 + (char *)v18 - (char *)v19);
    do
    {
      if ( v23 )
      {
        *v23 = _mm_loadu_si128(v19);
        v23[1] = _mm_loadu_si128(v19 + 1);
      }
      v23 += 2;
      v19 += 2;
    }
    while ( v23 != v24 );
  }
  v228 = (unsigned __int64)v24;
  v188 = a1;
LABEL_20:
  v25 = (unsigned __int64)v220;
  if ( (__m128i *)(v221 - (_QWORD)v220) != (__m128i *)((char *)v24 - v22) )
  {
LABEL_21:
    v26 = *(_QWORD *)(v221 - 32);
    v27 = *(_QWORD *)(v26 + 56);
    v187 = v26;
    v189 = v26 + 48;
    if ( v27 == v26 + 48 )
      goto LABEL_46;
    v28 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v29 = (char *)(v27 - 24);
        if ( !v27 )
          v29 = 0;
        if ( !(unsigned __int8)sub_98CD80(v29) )
        {
          v115 = *(_DWORD *)(v188 + 352);
          v116 = v188 + 328;
          if ( v115 )
          {
            v117 = *(_QWORD *)(v188 + 336);
            v118 = ((unsigned int)v187 >> 9) ^ ((unsigned int)v187 >> 4);
            v119 = (v115 - 1) & v118;
            v120 = (_QWORD *)(v117 + 8LL * v119);
            v121 = *v120;
            if ( v187 == *v120 )
              goto LABEL_46;
            v122 = 1;
            v123 = 0;
            while ( v121 != -4096 )
            {
              if ( v121 == -8192 && !v123 )
                v123 = v120;
              v119 = (v115 - 1) & (v122 + v119);
              v120 = (_QWORD *)(v117 + 8LL * v119);
              v121 = *v120;
              if ( v187 == *v120 )
                goto LABEL_46;
              ++v122;
            }
            if ( v123 )
              v120 = v123;
            ++*(_QWORD *)(v188 + 328);
            v124 = *(_DWORD *)(v188 + 344) + 1;
            if ( 4 * v124 < 3 * v115 )
            {
              if ( v115 - *(_DWORD *)(v188 + 348) - v124 > v115 >> 3 )
              {
LABEL_199:
                *(_DWORD *)(v188 + 344) = v124;
                if ( *v120 != -4096 )
                  --*(_DWORD *)(v188 + 348);
                *v120 = v187;
                goto LABEL_46;
              }
              sub_E3B4A0(v116, v115);
              v166 = *(_DWORD *)(v188 + 352);
              if ( v166 )
              {
                v167 = v166 - 1;
                v168 = *(_QWORD *)(v188 + 336);
                v169 = v167 & v118;
                v170 = 1;
                v120 = (_QWORD *)(v168 + 8LL * (v167 & v118));
                v124 = *(_DWORD *)(v188 + 344) + 1;
                v165 = 0;
                v171 = *v120;
                if ( v187 == *v120 )
                  goto LABEL_199;
                while ( v171 != -4096 )
                {
                  if ( !v165 && v171 == -8192 )
                    v165 = v120;
                  v169 = v167 & (v170 + v169);
                  v120 = (_QWORD *)(v168 + 8LL * v169);
                  v171 = *v120;
                  if ( v187 == *v120 )
                    goto LABEL_199;
                  ++v170;
                }
                goto LABEL_334;
              }
              goto LABEL_413;
            }
          }
          else
          {
            ++*(_QWORD *)(v188 + 328);
          }
          sub_E3B4A0(v116, 2 * v115);
          v159 = *(_DWORD *)(v188 + 352);
          if ( v159 )
          {
            v160 = v159 - 1;
            v161 = *(_QWORD *)(v188 + 336);
            v162 = v160 & (((unsigned int)v187 >> 9) ^ ((unsigned int)v187 >> 4));
            v120 = (_QWORD *)(v161 + 8LL * v162);
            v124 = *(_DWORD *)(v188 + 344) + 1;
            v163 = *v120;
            if ( v187 == *v120 )
              goto LABEL_199;
            v164 = 1;
            v165 = 0;
            while ( v163 != -4096 )
            {
              if ( v163 == -8192 && !v165 )
                v165 = v120;
              v162 = v160 & (v164 + v162);
              v120 = (_QWORD *)(v161 + 8LL * v162);
              v163 = *v120;
              if ( v187 == *v120 )
                goto LABEL_199;
              ++v164;
            }
LABEL_334:
            if ( v165 )
              v120 = v165;
            goto LABEL_199;
          }
LABEL_413:
          ++*(_DWORD *)(v188 + 344);
          BUG();
        }
        if ( (_DWORD)qword_4FFC308 != -1 )
        {
          if ( (int)qword_4FFC308 <= v28 )
            goto LABEL_46;
          ++v28;
        }
        v30 = (unsigned __int8)*v29;
        if ( (unsigned int)(v30 - 30) <= 0xA )
        {
LABEL_46:
          sub_23EC7E0((__int64)v216);
          v22 = v227;
          v24 = (__m128i *)v228;
          goto LABEL_20;
        }
        if ( (_BYTE)v30 == 61 )
          break;
        if ( (_BYTE)v30 == 62 )
        {
          if ( sub_B46500((unsigned __int8 *)v29) || (v29[2] & 1) != 0 )
            goto LABEL_45;
          v46 = *((_QWORD *)v29 - 8);
          v33 = sub_2792F80(v188, *((_QWORD *)v29 - 4));
          v47 = sub_2792F80(v188, v46);
          v32 = v47;
          if ( !v203 )
          {
            ++v200;
            goto LABEL_217;
          }
          v34 = v203 - 1;
          v48 = 1;
          v49 = (0xBF58476D1CE4E5B9LL * v47) >> 31;
          v50 = ((0xBF58476D1CE4E5B9LL
                * ((unsigned int)v49 ^ (484763065 * v47) | ((unsigned __int64)(unsigned int)(37 * v33) << 32))) >> 31)
              ^ (0xBF58476D1CE4E5B9LL
               * ((unsigned int)v49 ^ (484763065 * v47) | ((unsigned __int64)(unsigned int)(37 * v33) << 32)));
          v38 = 0;
          for ( i = (((0xBF58476D1CE4E5B9LL
                     * ((unsigned int)v49 ^ (484763065 * (_DWORD)v32)
                      | ((unsigned __int64)(unsigned int)(37 * v33) << 32))) >> 31)
                   ^ (484763065 * (v49 ^ (484763065 * v32))))
                  & (v203 - 1); ; i = v34 & v141 )
          {
            v40 = v201 + ((unsigned __int64)i << 6);
            v52 = *(_DWORD *)v40;
            if ( v33 == *(_DWORD *)v40 && v32 == *(_QWORD *)(v40 + 8) )
              break;
            if ( v52 == -1 )
            {
              if ( *(_QWORD *)(v40 + 8) == -1 )
              {
                if ( !v38 )
                  v38 = (int *)(v201 + ((unsigned __int64)i << 6));
                ++v200;
                v147 = v202 + 1;
                if ( 4 * ((int)v202 + 1) < 3 * v203 )
                {
                  if ( v203 - HIDWORD(v202) - v147 > v203 >> 3 )
                    goto LABEL_268;
                  v186 = v32;
                  sub_27A3B90((__int64)&v200, v203);
                  if ( v203 )
                  {
                    v155 = 1;
                    v132 = 0;
                    v32 = v186;
                    for ( j = (v203 - 1) & v50; ; j = (v203 - 1) & v158 )
                    {
                      v38 = (int *)(v201 + ((unsigned __int64)j << 6));
                      v157 = *v38;
                      if ( v33 == *v38 && v186 == *((_QWORD *)v38 + 1) )
                        break;
                      if ( v157 == -1 )
                      {
                        if ( *((_QWORD *)v38 + 1) == -1 )
                          goto LABEL_364;
                      }
                      else if ( v157 == -2 && *((_QWORD *)v38 + 1) == -2 && !v132 )
                      {
                        v132 = (int *)(v201 + ((unsigned __int64)j << 6));
                      }
                      v158 = v155 + j;
                      ++v155;
                    }
LABEL_267:
                    v147 = v202 + 1;
LABEL_268:
                    LODWORD(v202) = v147;
                    if ( *v38 != -1 || *((_QWORD *)v38 + 1) != -1 )
                      --HIDWORD(v202);
                    goto LABEL_236;
                  }
LABEL_415:
                  LODWORD(v202) = v202 + 1;
                  BUG();
                }
LABEL_217:
                v184 = v32;
                sub_27A3B90((__int64)&v200, 2 * v203);
                if ( v203 )
                {
                  v32 = v184;
                  v131 = 1;
                  v132 = 0;
                  for ( k = (v203 - 1)
                          & (((0xBF58476D1CE4E5B9LL
                             * ((unsigned int)((0xBF58476D1CE4E5B9LL * v184) >> 31) ^ (484763065 * (_DWORD)v184)
                              | ((unsigned __int64)(unsigned int)(37 * v33) << 32))) >> 31)
                           ^ (484763065 * (((0xBF58476D1CE4E5B9LL * v184) >> 31) ^ (484763065 * v184))));
                        ;
                        k = (v203 - 1) & v135 )
                  {
                    v38 = (int *)(v201 + ((unsigned __int64)k << 6));
                    v134 = *v38;
                    if ( v33 == *v38 && v184 == *((_QWORD *)v38 + 1) )
                      goto LABEL_267;
                    if ( v134 == -1 )
                    {
                      if ( *((_QWORD *)v38 + 1) == -1 )
                      {
LABEL_364:
                        if ( v132 )
                          v38 = v132;
                        v147 = v202 + 1;
                        goto LABEL_268;
                      }
                    }
                    else if ( v134 == -2 && *((_QWORD *)v38 + 1) == -2 && !v132 )
                    {
                      v132 = (int *)(v201 + ((unsigned __int64)k << 6));
                    }
                    v135 = v131 + k;
                    ++v131;
                  }
                }
                goto LABEL_415;
              }
            }
            else if ( v52 == -2 && *(_QWORD *)(v40 + 8) == -2 && !v38 )
            {
              v38 = (int *)(v201 + ((unsigned __int64)i << 6));
            }
            v141 = v48 + i;
            ++v48;
          }
          v43 = *(unsigned int *)(v40 + 24);
          v44 = (int *)(v40 + 16);
          v45 = v43 + 1;
          if ( *(unsigned int *)(v40 + 28) >= (unsigned __int64)(v43 + 1) )
            goto LABEL_44;
LABEL_89:
          v182 = v40;
          sub_C8D5F0((__int64)v44, (const void *)(v40 + 32), v45, 8u, v40, v34);
          v43 = *(unsigned int *)(v182 + 24);
          goto LABEL_44;
        }
        if ( (_BYTE)v30 != 85 )
        {
          if ( !*(_BYTE *)(v188 + 636) && (_BYTE)v30 == 63 )
            goto LABEL_45;
          v61 = sub_2792F80(v188, (__int64)v29);
          v62 = v61;
          if ( !v195 )
          {
            ++v192;
            goto LABEL_253;
          }
          v34 = v193;
          v63 = 1;
          v64 = ((0xBF58476D1CE4E5B9LL * (((unsigned __int64)(unsigned int)(37 * v61) << 32) | 0x2ABF1DA4)) >> 31)
              ^ (0xBF58476D1CE4E5B9LL * (((unsigned __int64)(unsigned int)(37 * v61) << 32) | 0x2ABF1DA4));
          v65 = 0;
          v66 = v64 & (v195 - 1);
          while ( 2 )
          {
            v40 = v193 + ((unsigned __int64)v66 << 6);
            v67 = *(_DWORD *)v40;
            if ( v62 == *(_DWORD *)v40 )
            {
              if ( *(_QWORD *)(v40 + 8) == -3 )
              {
LABEL_43:
                v43 = *(unsigned int *)(v40 + 24);
                v44 = (int *)(v40 + 16);
                v45 = v43 + 1;
                if ( v43 + 1 <= (unsigned __int64)*(unsigned int *)(v40 + 28) )
                  goto LABEL_44;
                goto LABEL_89;
              }
              if ( v67 == -1 )
                goto LABEL_204;
LABEL_82:
              if ( v67 == -2 && *(_QWORD *)(v40 + 8) == -2 && !v65 )
                v65 = (int *)(v193 + ((unsigned __int64)v66 << 6));
            }
            else
            {
              if ( v67 != -1 )
                goto LABEL_82;
LABEL_204:
              if ( *(_QWORD *)(v40 + 8) == -1 )
              {
                if ( !v65 )
                  v65 = (int *)(v193 + ((unsigned __int64)v66 << 6));
                ++v192;
                v148 = v194 + 1;
                if ( 4 * ((int)v194 + 1) < 3 * v195 )
                {
                  if ( v195 - HIDWORD(v194) - v148 > v195 >> 3 )
                    goto LABEL_341;
                  sub_27A3B90((__int64)&v192, v195);
                  if ( v195 )
                  {
                    v172 = 1;
                    v173 = 0;
                    for ( m = (v195 - 1) & v64; ; m = (v195 - 1) & v176 )
                    {
                      v65 = (int *)(v193 + ((unsigned __int64)m << 6));
                      v175 = *v65;
                      if ( v62 == *v65 && *((_QWORD *)v65 + 1) == -3 )
                        goto LABEL_271;
                      if ( v175 == -1 )
                      {
                        if ( *((_QWORD *)v65 + 1) == -1 )
                        {
                          if ( v173 )
                            v65 = v173;
                          v148 = v194 + 1;
                          goto LABEL_341;
                        }
                      }
                      else if ( v175 == -2 && *((_QWORD *)v65 + 1) == -2 && !v173 )
                      {
                        v173 = (int *)(v193 + ((unsigned __int64)m << 6));
                      }
                      v176 = v172 + m;
                      ++v172;
                    }
                  }
                  goto LABEL_416;
                }
LABEL_253:
                sub_27A3B90((__int64)&v192, 2 * v195);
                if ( v195 )
                {
                  v142 = 1;
                  v143 = 0;
                  for ( n = (v195 - 1)
                          & (((0xBF58476D1CE4E5B9LL * (((unsigned __int64)(unsigned int)(37 * v62) << 32) | 0x2ABF1DA4)) >> 31)
                           ^ 0x2FB01F84); ; n = (v195 - 1) & v146 )
                  {
                    v65 = (int *)(v193 + ((unsigned __int64)n << 6));
                    v145 = *v65;
                    if ( v62 == *v65 && *((_QWORD *)v65 + 1) == -3 )
                      break;
                    if ( v145 == -1 )
                    {
                      if ( *((_QWORD *)v65 + 1) == -1 )
                      {
                        if ( v143 )
                          v65 = v143;
                        v148 = v194 + 1;
                        goto LABEL_341;
                      }
                    }
                    else if ( v145 == -2 && *((_QWORD *)v65 + 1) == -2 && !v143 )
                    {
                      v143 = (int *)(v193 + ((unsigned __int64)n << 6));
                    }
                    v146 = v142 + n;
                    ++v142;
                  }
LABEL_271:
                  v148 = v194 + 1;
LABEL_341:
                  LODWORD(v194) = v148;
                  if ( *v65 != -1 || *((_QWORD *)v65 + 1) != -1 )
                    --HIDWORD(v194);
                  *v65 = v62;
                  v44 = v65 + 4;
                  *((_QWORD *)v65 + 1) = -3;
                  *((_QWORD *)v65 + 2) = v65 + 8;
                  *((_QWORD *)v65 + 3) = 0x400000000LL;
                  v43 = 0;
                  goto LABEL_44;
                }
LABEL_416:
                LODWORD(v194) = v194 + 1;
                BUG();
              }
            }
            v125 = v63 + v66;
            ++v63;
            v66 = (v195 - 1) & v125;
            continue;
          }
        }
        v53 = *((_QWORD *)v29 - 4);
        if ( !v53
          || *(_BYTE *)v53
          || *(_QWORD *)(v53 + 24) != *((_QWORD *)v29 + 10)
          || (*(_BYTE *)(v53 + 33) & 0x20) == 0 )
        {
          goto LABEL_419;
        }
        v114 = *(_DWORD *)(v53 + 36);
        if ( (unsigned int)(v114 - 68) <= 3 || v114 == 11 )
          goto LABEL_45;
        if ( v114 != 324 )
        {
LABEL_419:
          if ( (unsigned __int8)sub_B46970((unsigned __int8 *)v29)
            || (unsigned __int8)sub_A73ED0((_QWORD *)v29 + 9, 6)
            || (unsigned __int8)sub_B49560((__int64)v29, 6) )
          {
            goto LABEL_46;
          }
          v54 = sub_2792F80(v188, (__int64)v29);
          if ( !sub_B49E00((__int64)v29) )
          {
            v68 = sub_B49E20((__int64)v29);
            v190 = v54;
            v69 = &v208;
            v191 = -3;
            if ( !v68 )
              v69 = &v212;
            v44 = sub_27A3E70((__int64)v69, &v190);
            v43 = (unsigned int)v44[2];
            if ( v43 + 1 > (unsigned __int64)(unsigned int)v44[3] )
            {
              sub_C8D5F0((__int64)v44, v44 + 4, v43 + 1, 8u, v70, v71);
              v43 = (unsigned int)v44[2];
            }
            goto LABEL_44;
          }
          if ( v207 )
          {
            v55 = 1;
            v56 = 0;
            for ( ii = (((0xBF58476D1CE4E5B9LL * (((unsigned __int64)(unsigned int)(37 * v54) << 32) | 0x2ABF1DA4)) >> 31)
                      ^ 0x2FB01F84)
                     & (v207 - 1); ; ii = (v207 - 1) & v60 )
            {
              v58 = v205 + ((unsigned __int64)ii << 6);
              v59 = *(_DWORD *)v58;
              if ( v54 == *(_DWORD *)v58 && *(_QWORD *)(v58 + 8) == -3 )
              {
                v43 = *(unsigned int *)(v58 + 24);
                v44 = (int *)(v58 + 16);
                if ( v43 + 1 > (unsigned __int64)*(unsigned int *)(v58 + 28) )
                {
                  sub_C8D5F0(v58 + 16, (const void *)(v58 + 32), v43 + 1, 8u, v56, v55);
                  v43 = *(unsigned int *)(v58 + 24);
                }
                goto LABEL_44;
              }
              if ( v59 == -1 )
              {
                if ( *(_QWORD *)(v58 + 8) == -1 )
                {
                  if ( v56 )
                    v58 = v56;
                  ++v204;
                  v149 = v206 + 1;
                  if ( 4 * ((int)v206 + 1) < 3 * v207 )
                  {
                    if ( v207 - HIDWORD(v206) - v149 > v207 >> 3 )
                      goto LABEL_277;
                    sub_27A3B90((__int64)&v204, v207);
                    if ( v207 )
                    {
                      v177 = 1;
                      v58 = 0;
                      for ( jj = (v207 - 1)
                               & (((0xBF58476D1CE4E5B9LL
                                  * (((unsigned __int64)(unsigned int)(37 * v54) << 32) | 0x2ABF1DA4)) >> 31)
                                ^ 0x2FB01F84); ; jj = (v207 - 1) & v181 )
                      {
                        v179 = (int *)(v205 + ((unsigned __int64)jj << 6));
                        v180 = *v179;
                        if ( v54 == *v179 && *((_QWORD *)v179 + 1) == -3 )
                          break;
                        if ( v180 == -1 )
                        {
                          if ( *((_QWORD *)v179 + 1) == -1 )
                          {
                            if ( !v58 )
                              v58 = v205 + ((unsigned __int64)jj << 6);
                            v149 = v206 + 1;
                            goto LABEL_277;
                          }
                        }
                        else if ( v180 == -2 && *((_QWORD *)v179 + 1) == -2 && !v58 )
                        {
                          v58 = v205 + ((unsigned __int64)jj << 6);
                        }
                        v181 = v177 + jj;
                        ++v177;
                      }
                      v58 = v205 + ((unsigned __int64)jj << 6);
                      v149 = v206 + 1;
LABEL_277:
                      LODWORD(v206) = v149;
                      if ( *(_DWORD *)v58 != -1 || *(_QWORD *)(v58 + 8) != -1 )
                        --HIDWORD(v206);
                      *(_DWORD *)v58 = v54;
                      v44 = (int *)(v58 + 16);
                      *(_QWORD *)(v58 + 16) = v58 + 32;
                      *(_QWORD *)(v58 + 24) = 0x400000000LL;
                      v43 = 0;
                      *(_QWORD *)(v58 + 8) = -3;
                      goto LABEL_44;
                    }
                    goto LABEL_414;
                  }
LABEL_295:
                  sub_27A3B90((__int64)&v204, 2 * v207);
                  if ( v207 )
                  {
                    v150 = 1;
                    v151 = 0;
                    for ( kk = (v207 - 1)
                             & (((0xBF58476D1CE4E5B9LL
                                * (((unsigned __int64)(unsigned int)(37 * v54) << 32) | 0x2ABF1DA4)) >> 31)
                              ^ 0x2FB01F84); ; kk = (v207 - 1) & v154 )
                    {
                      v58 = v205 + ((unsigned __int64)kk << 6);
                      v153 = *(_DWORD *)v58;
                      if ( v54 == *(_DWORD *)v58 && *(_QWORD *)(v58 + 8) == -3 )
                        break;
                      if ( v153 == -1 )
                      {
                        if ( *(_QWORD *)(v58 + 8) == -1 )
                        {
                          if ( v151 )
                            v58 = v151;
                          v149 = v206 + 1;
                          goto LABEL_277;
                        }
                      }
                      else if ( v153 == -2 && *(_QWORD *)(v58 + 8) == -2 && !v151 )
                      {
                        v151 = v205 + ((unsigned __int64)kk << 6);
                      }
                      v154 = v150 + kk;
                      ++v150;
                    }
                    v149 = v206 + 1;
                    goto LABEL_277;
                  }
LABEL_414:
                  LODWORD(v206) = v206 + 1;
                  BUG();
                }
              }
              else if ( v59 == -2 && *(_QWORD *)(v58 + 8) == -2 && !v56 )
              {
                v56 = v205 + ((unsigned __int64)ii << 6);
              }
              v60 = v55 + ii;
              v55 = (unsigned int)(v55 + 1);
            }
          }
          ++v204;
          goto LABEL_295;
        }
        v27 = *(_QWORD *)(v27 + 8);
        if ( v189 == v27 )
          goto LABEL_46;
      }
      if ( sub_B46500((unsigned __int8 *)v29) || (v29[2] & 1) != 0 )
        goto LABEL_45;
      v31 = sub_2792F80(v188, *((_QWORD *)v29 - 4));
      v32 = *((_QWORD *)v29 + 1);
      v33 = v31;
      if ( !v199 )
      {
        ++v196;
LABEL_207:
        v183 = v32;
        sub_27A3B90((__int64)&v196, 2 * v199);
        if ( v199 )
        {
          v32 = v183;
          v126 = 1;
          v127 = 0;
          for ( mm = (v199 - 1)
                   & (((0xBF58476D1CE4E5B9LL
                      * ((unsigned int)((0xBF58476D1CE4E5B9LL * v183) >> 31) ^ (484763065 * (_DWORD)v183)
                       | ((unsigned __int64)(unsigned int)(37 * v33) << 32))) >> 31)
                    ^ (484763065 * (((0xBF58476D1CE4E5B9LL * v183) >> 31) ^ (484763065 * v183)))); ; mm = (v199 - 1) & v130 )
          {
            v38 = (int *)(v197 + ((unsigned __int64)mm << 6));
            v129 = *v38;
            if ( v33 == *v38 && v183 == *((_QWORD *)v38 + 1) )
              break;
            if ( v129 == -1 )
            {
              if ( *((_QWORD *)v38 + 1) == -1 )
              {
LABEL_358:
                if ( v127 )
                  v38 = v127;
                v136 = v198 + 1;
                goto LABEL_234;
              }
            }
            else if ( v129 == -2 && *((_QWORD *)v38 + 1) == -2 && !v127 )
            {
              v127 = (int *)(v197 + ((unsigned __int64)mm << 6));
            }
            v130 = v126 + mm;
            ++v126;
          }
          goto LABEL_263;
        }
LABEL_412:
        LODWORD(v198) = v198 + 1;
        BUG();
      }
      v34 = v199 - 1;
      v35 = 1;
      v36 = (0xBF58476D1CE4E5B9LL * v32) >> 31;
      v37 = ((0xBF58476D1CE4E5B9LL
            * ((unsigned int)v36 ^ (484763065 * (_DWORD)v32) | ((unsigned __int64)(unsigned int)(37 * v31) << 32))) >> 31)
          ^ (0xBF58476D1CE4E5B9LL
           * ((unsigned int)v36 ^ (484763065 * (_DWORD)v32) | ((unsigned __int64)(unsigned int)(37 * v31) << 32)));
      v38 = 0;
      for ( nn = (((0xBF58476D1CE4E5B9LL
                  * ((unsigned int)v36 ^ (484763065 * (_DWORD)v32) | ((unsigned __int64)(unsigned int)(37 * v33) << 32))) >> 31)
                ^ (484763065 * (v36 ^ (484763065 * v32))))
               & (v199 - 1); ; nn = v34 & v42 )
      {
        v40 = v197 + ((unsigned __int64)nn << 6);
        v41 = *(_DWORD *)v40;
        if ( v33 == *(_DWORD *)v40 && v32 == *(_QWORD *)(v40 + 8) )
          goto LABEL_43;
        if ( v41 == -1 )
          break;
        if ( v41 == -2 && *(_QWORD *)(v40 + 8) == -2 && !v38 )
          v38 = (int *)(v197 + ((unsigned __int64)nn << 6));
LABEL_41:
        v42 = v35 + nn;
        ++v35;
      }
      if ( *(_QWORD *)(v40 + 8) != -1 )
        goto LABEL_41;
      if ( !v38 )
        v38 = (int *)(v197 + ((unsigned __int64)nn << 6));
      ++v196;
      v136 = v198 + 1;
      if ( 4 * ((int)v198 + 1) >= 3 * v199 )
        goto LABEL_207;
      if ( v199 - HIDWORD(v198) - v136 > v199 >> 3 )
        goto LABEL_234;
      v185 = v32;
      sub_27A3B90((__int64)&v196, v199);
      if ( !v199 )
        goto LABEL_412;
      v137 = 1;
      v127 = 0;
      v32 = v185;
      for ( i1 = (v199 - 1) & v37; ; i1 = (v199 - 1) & v140 )
      {
        v38 = (int *)(v197 + ((unsigned __int64)i1 << 6));
        v139 = *v38;
        if ( v33 == *v38 && v185 == *((_QWORD *)v38 + 1) )
          break;
        if ( v139 == -1 )
        {
          if ( *((_QWORD *)v38 + 1) == -1 )
            goto LABEL_358;
        }
        else if ( v139 == -2 && *((_QWORD *)v38 + 1) == -2 && !v127 )
        {
          v127 = (int *)(v197 + ((unsigned __int64)i1 << 6));
        }
        v140 = v137 + i1;
        ++v137;
      }
LABEL_263:
      v136 = v198 + 1;
LABEL_234:
      LODWORD(v198) = v136;
      if ( *v38 != -1 || *((_QWORD *)v38 + 1) != -1 )
        --HIDWORD(v198);
LABEL_236:
      *((_QWORD *)v38 + 1) = v32;
      v44 = v38 + 4;
      *v38 = v33;
      *((_QWORD *)v38 + 2) = v38 + 8;
      *((_QWORD *)v38 + 3) = 0x400000000LL;
      v43 = 0;
LABEL_44:
      *(_QWORD *)(*(_QWORD *)v44 + 8 * v43) = v29;
      ++v44[2];
LABEL_45:
      v27 = *(_QWORD *)(v27 + 8);
      if ( v189 == v27 )
        goto LABEL_46;
    }
  }
  if ( v220 != (__m128i *)v221 )
  {
    v72 = v22;
    while ( *(_QWORD *)v25 == *(_QWORD *)v72 )
    {
      v73 = *(_BYTE *)(v25 + 24);
      if ( v73 != *(_BYTE *)(v72 + 24) || v73 && *(_DWORD *)(v25 + 16) != *(_DWORD *)(v72 + 16) )
        break;
      v25 += 32LL;
      v72 += 32LL;
      if ( v221 == v25 )
        goto LABEL_101;
    }
    goto LABEL_21;
  }
LABEL_101:
  if ( v22 )
    j_j___libc_free_0(v22);
  if ( !v225 )
    _libc_free(v224);
  if ( v220 )
    j_j___libc_free_0((unsigned __int64)v220);
  if ( !v218 )
    _libc_free(v217);
  if ( v238 )
    j_j___libc_free_0((unsigned __int64)v238);
  if ( !v237 )
    _libc_free(v236);
  if ( v233 )
    j_j___libc_free_0((unsigned __int64)v233);
  if ( !v232[12] )
    _libc_free(v231);
  v231 = 0x400000000LL;
  v230 = v232;
  sub_27A8390(v188, (__int64)&v192, (__int64)&v230, 1, v16, v17);
  sub_27A8390(v188, (__int64)&v196, (__int64)&v230, 2, v74, v75);
  sub_27A8390(v188, (__int64)&v200, (__int64)&v230, 3, v76, v77);
  sub_27A8390(v188, (__int64)&v204, (__int64)&v230, 1, v78, v79);
  sub_27A8390(v188, (__int64)&v208, (__int64)&v230, 2, v80, v81);
  sub_27A8390(v188, (__int64)&v212, (__int64)&v230, 3, v82, v83);
  v84 = sub_27A4BB0(v188, (__int64 *)&v230);
  v85 = (__int64)v230;
  v86 = v84;
  v87 = (unsigned __int64)&v230[56 * (unsigned int)v231];
  if ( v230 != (_BYTE *)v87 )
  {
    do
    {
      v87 -= 56LL;
      v88 = *(_QWORD *)(v87 + 8);
      if ( v88 != v87 + 24 )
        _libc_free(v88);
    }
    while ( v85 != v87 );
    v87 = (unsigned __int64)v230;
  }
  if ( (_BYTE *)v87 != v232 )
    _libc_free(v87);
  v89 = v215;
  if ( v215 )
  {
    v90 = v213;
    v91 = v213 + ((unsigned __int64)v215 << 6);
    while ( 1 )
    {
      while ( *(_DWORD *)v90 == -1 )
      {
        if ( *(_QWORD *)(v90 + 8) != -1 )
          goto LABEL_127;
        v90 += 64;
        if ( v91 == v90 )
        {
LABEL_133:
          v89 = v215;
          goto LABEL_134;
        }
      }
      if ( *(_DWORD *)v90 != -2 || *(_QWORD *)(v90 + 8) != -2 )
      {
LABEL_127:
        v92 = *(_QWORD *)(v90 + 16);
        if ( v92 != v90 + 32 )
          _libc_free(v92);
      }
      v90 += 64;
      if ( v91 == v90 )
        goto LABEL_133;
    }
  }
LABEL_134:
  sub_C7D6A0(v213, (unsigned __int64)v89 << 6, 8);
  v93 = v211;
  if ( !v211 )
    goto LABEL_144;
  v94 = v209;
  v95 = v209 + ((unsigned __int64)v211 << 6);
  do
  {
    while ( *(_DWORD *)v94 != -1 )
    {
      if ( *(_DWORD *)v94 != -2 || *(_QWORD *)(v94 + 8) != -2 )
      {
LABEL_137:
        v96 = *(_QWORD *)(v94 + 16);
        if ( v96 != v94 + 32 )
          _libc_free(v96);
      }
      v94 += 64;
      if ( v95 == v94 )
        goto LABEL_143;
    }
    if ( *(_QWORD *)(v94 + 8) != -1 )
      goto LABEL_137;
    v94 += 64;
  }
  while ( v95 != v94 );
LABEL_143:
  v93 = v211;
LABEL_144:
  sub_C7D6A0(v209, (unsigned __int64)v93 << 6, 8);
  v97 = v207;
  if ( !v207 )
    goto LABEL_154;
  v98 = v205;
  v99 = v205 + ((unsigned __int64)v207 << 6);
  while ( 2 )
  {
    while ( 2 )
    {
      if ( *(_DWORD *)v98 != -1 )
      {
        if ( *(_DWORD *)v98 == -2 && *(_QWORD *)(v98 + 8) == -2 )
          goto LABEL_149;
        goto LABEL_147;
      }
      if ( *(_QWORD *)(v98 + 8) != -1 )
      {
LABEL_147:
        v100 = *(_QWORD *)(v98 + 16);
        if ( v100 != v98 + 32 )
          _libc_free(v100);
LABEL_149:
        v98 += 64;
        if ( v99 == v98 )
          goto LABEL_153;
        continue;
      }
      break;
    }
    v98 += 64;
    if ( v99 != v98 )
      continue;
    break;
  }
LABEL_153:
  v97 = v207;
LABEL_154:
  sub_C7D6A0(v205, (unsigned __int64)v97 << 6, 8);
  v101 = v203;
  if ( !v203 )
    goto LABEL_164;
  v102 = v201;
  v103 = v201 + ((unsigned __int64)v203 << 6);
  while ( 2 )
  {
    while ( 2 )
    {
      if ( *(_DWORD *)v102 != -1 )
      {
        if ( *(_DWORD *)v102 == -2 && *(_QWORD *)(v102 + 8) == -2 )
          goto LABEL_159;
        goto LABEL_157;
      }
      if ( *(_QWORD *)(v102 + 8) != -1 )
      {
LABEL_157:
        v104 = *(_QWORD *)(v102 + 16);
        if ( v104 != v102 + 32 )
          _libc_free(v104);
LABEL_159:
        v102 += 64;
        if ( v103 == v102 )
          goto LABEL_163;
        continue;
      }
      break;
    }
    v102 += 64;
    if ( v103 != v102 )
      continue;
    break;
  }
LABEL_163:
  v101 = v203;
LABEL_164:
  sub_C7D6A0(v201, (unsigned __int64)v101 << 6, 8);
  v105 = v199;
  if ( !v199 )
    goto LABEL_174;
  v106 = v197;
  v107 = v197 + ((unsigned __int64)v199 << 6);
  while ( 2 )
  {
    while ( 2 )
    {
      if ( *(_DWORD *)v106 != -1 )
      {
        if ( *(_DWORD *)v106 == -2 && *(_QWORD *)(v106 + 8) == -2 )
          goto LABEL_169;
        goto LABEL_167;
      }
      if ( *(_QWORD *)(v106 + 8) != -1 )
      {
LABEL_167:
        v108 = *(_QWORD *)(v106 + 16);
        if ( v108 != v106 + 32 )
          _libc_free(v108);
LABEL_169:
        v106 += 64;
        if ( v107 == v106 )
          goto LABEL_173;
        continue;
      }
      break;
    }
    v106 += 64;
    if ( v107 != v106 )
      continue;
    break;
  }
LABEL_173:
  v105 = v199;
LABEL_174:
  sub_C7D6A0(v197, (unsigned __int64)v105 << 6, 8);
  v109 = v195;
  if ( v195 )
  {
    v110 = v193;
    v111 = v193 + ((unsigned __int64)v195 << 6);
    while ( 1 )
    {
      while ( *(_DWORD *)v110 == -1 )
      {
        if ( *(_QWORD *)(v110 + 8) != -1 )
          goto LABEL_177;
        v110 += 64;
        if ( v111 == v110 )
        {
LABEL_183:
          v109 = v195;
          goto LABEL_184;
        }
      }
      if ( *(_DWORD *)v110 != -2 || *(_QWORD *)(v110 + 8) != -2 )
      {
LABEL_177:
        v112 = *(_QWORD *)(v110 + 16);
        if ( v112 != v110 + 32 )
          _libc_free(v112);
      }
      v110 += 64;
      if ( v111 == v110 )
        goto LABEL_183;
    }
  }
LABEL_184:
  sub_C7D6A0(v193, (unsigned __int64)v109 << 6, 8);
  return v86;
}
