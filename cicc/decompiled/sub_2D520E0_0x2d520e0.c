// Function: sub_2D520E0
// Address: 0x2d520e0
//
__int64 *__fastcall sub_2D520E0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  bool v4; // zf
  __int64 v5; // r15
  __int64 v6; // r14
  size_t v7; // r13
  char *v8; // rdx
  unsigned __int64 v9; // rax
  char v10; // bl
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // rax
  __int64 v15; // r9
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  const __m128i *v21; // rdi
  unsigned __int64 v22; // r12
  char *v23; // rbx
  __int64 v24; // r13
  char v25; // dl
  char v26; // al
  __int64 v27; // rsi
  const char **v28; // r11
  int v29; // r15d
  int j; // eax
  const char **v31; // rdx
  int v32; // eax
  __int64 v33; // rdi
  __int64 v34; // rsi
  unsigned __int64 v36; // r15
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdx
  size_t v40; // r12
  int v41; // eax
  int v42; // eax
  __int64 v43; // rdx
  __int64 *v44; // rax
  __int64 v45; // rax
  unsigned __int64 v46; // r12
  unsigned __int64 v47; // rbx
  const __m128i *v48; // r13
  const __m128i *v49; // rax
  const void *v50; // r12
  size_t v51; // r15
  int v52; // eax
  unsigned int v53; // r8d
  _QWORD *v54; // r9
  __int64 v55; // rax
  unsigned int v56; // r8d
  _QWORD *v57; // r9
  char *v58; // rcx
  char *v59; // r12
  size_t v60; // r13
  int v61; // eax
  unsigned int v62; // r15d
  int v63; // eax
  int v64; // eax
  __int64 v65; // rdx
  __int64 *v66; // rax
  __int64 v67; // rax
  int v68; // eax
  int v69; // eax
  __int64 v70; // rdx
  __int64 *v71; // rax
  __int64 v72; // rax
  int v73; // eax
  int v74; // eax
  __int64 v75; // rdx
  __int64 *v76; // rax
  __int64 v77; // rax
  __int64 v78; // rbx
  __int64 v79; // rax
  unsigned __int64 v80; // rsi
  __int64 v81; // rcx
  __int64 v82; // rdx
  const char **v83; // rsi
  _QWORD *v84; // rdi
  unsigned __int64 v85; // r14
  __int64 *m128i_i64; // rax
  __int64 v87; // rdi
  __int64 v88; // rsi
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // r8
  __int64 v92; // r9
  int v93; // r13d
  unsigned __int64 v94; // rbx
  __int64 v95; // rax
  __int64 v96; // rax
  _QWORD *v97; // r8
  _QWORD *v98; // rbx
  __int64 *v99; // rax
  __int64 v100; // rdx
  __int64 v101; // rdx
  _DWORD *v102; // rax
  _DWORD *v103; // rdx
  const char **v104; // rax
  __int64 *v105; // r14
  unsigned __int64 v106; // rbx
  unsigned __int64 v107; // rdi
  char *v108; // rax
  size_t v109; // rdx
  int v110; // eax
  __int64 v111; // rbx
  unsigned __int64 v112; // rcx
  int v113; // r13d
  int v114; // r15d
  unsigned __int64 v115; // rax
  unsigned __int64 v116; // rdx
  _DWORD *v117; // rax
  char v118; // al
  bool v119; // al
  unsigned int v120; // ecx
  _DWORD *v121; // rdi
  __int64 v122; // rsi
  unsigned int v123; // eax
  int v124; // ebx
  unsigned __int64 v125; // rdx
  _DWORD *v126; // rax
  _DWORD *i; // rdx
  const char *v128; // r13
  const char **v129; // rax
  size_t v130; // r15
  const void *v131; // rbx
  int v132; // eax
  int v133; // eax
  __int64 v134; // rdx
  __int64 *v135; // rax
  __int64 v136; // rax
  size_t v137; // r15
  const void *v138; // rbx
  int v139; // eax
  int v140; // eax
  __int64 v141; // rdx
  __int64 *v142; // rax
  __int64 v143; // rax
  size_t v144; // r12
  const void *v145; // rbx
  int v146; // eax
  int v147; // eax
  __int64 v148; // rdx
  __int64 *v149; // rax
  __int64 v150; // rax
  unsigned __int64 v151; // rcx
  __int64 v152; // rdi
  const char **v153; // rcx
  int v154; // eax
  int v155; // r10d
  int v156; // eax
  unsigned __int64 v157; // rax
  int k; // eax
  const char **v159; // rcx
  int v160; // r10d
  int v161; // eax
  _DWORD *v162; // rsi
  __int128 v163; // [rsp-30h] [rbp-2B0h]
  __int128 v164; // [rsp-30h] [rbp-2B0h]
  __int128 v165; // [rsp-30h] [rbp-2B0h]
  __int128 v166; // [rsp-30h] [rbp-2B0h]
  __int128 v167; // [rsp-20h] [rbp-2A0h]
  __int128 v168; // [rsp-20h] [rbp-2A0h]
  __int128 v169; // [rsp-20h] [rbp-2A0h]
  __int128 v170; // [rsp-20h] [rbp-2A0h]
  __int64 v171; // [rsp-10h] [rbp-290h]
  __int64 v172; // [rsp-10h] [rbp-290h]
  size_t v173; // [rsp+8h] [rbp-278h]
  size_t v174; // [rsp+8h] [rbp-278h]
  const __m128i *v175; // [rsp+10h] [rbp-270h]
  const __m128i *v176; // [rsp+10h] [rbp-270h]
  __int64 *v177; // [rsp+18h] [rbp-268h]
  unsigned int v178; // [rsp+24h] [rbp-25Ch]
  const char *s2; // [rsp+30h] [rbp-250h]
  char *s2a; // [rsp+30h] [rbp-250h]
  __int64 *v182; // [rsp+40h] [rbp-240h]
  const __m128i *v183; // [rsp+40h] [rbp-240h]
  _QWORD *v184; // [rsp+40h] [rbp-240h]
  __int64 *v185; // [rsp+40h] [rbp-240h]
  __int64 v186; // [rsp+40h] [rbp-240h]
  __int64 v187; // [rsp+60h] [rbp-220h]
  unsigned int v188; // [rsp+60h] [rbp-220h]
  const void *v189; // [rsp+68h] [rbp-218h]
  __int64 *v190; // [rsp+68h] [rbp-218h]
  size_t v191; // [rsp+68h] [rbp-218h]
  size_t v192; // [rsp+68h] [rbp-218h]
  size_t v193; // [rsp+68h] [rbp-218h]
  _QWORD *v194; // [rsp+68h] [rbp-218h]
  __int64 *v195; // [rsp+68h] [rbp-218h]
  char *v196; // [rsp+70h] [rbp-210h] BYREF
  unsigned __int64 v197; // [rsp+78h] [rbp-208h]
  __int64 v198; // [rsp+80h] [rbp-200h] BYREF
  _DWORD *v199; // [rsp+88h] [rbp-1F8h]
  __int64 v200; // [rsp+90h] [rbp-1F0h]
  __int64 v201; // [rsp+98h] [rbp-1E8h]
  __int64 v202; // [rsp+A8h] [rbp-1D8h]
  char *v203; // [rsp+B0h] [rbp-1D0h]
  __int64 v204; // [rsp+B8h] [rbp-1C8h]
  __int64 v205; // [rsp+C0h] [rbp-1C0h]
  __int64 v206; // [rsp+D8h] [rbp-1A8h]
  char *v207; // [rsp+E0h] [rbp-1A0h]
  __int64 v208; // [rsp+E8h] [rbp-198h]
  __int64 v209; // [rsp+F0h] [rbp-190h]
  unsigned __int64 v210[4]; // [rsp+100h] [rbp-180h] BYREF
  __int64 v211; // [rsp+120h] [rbp-160h]
  _QWORD v212[2]; // [rsp+130h] [rbp-150h] BYREF
  char *v213; // [rsp+140h] [rbp-140h]
  __int64 v214; // [rsp+148h] [rbp-138h]
  __int64 v215; // [rsp+150h] [rbp-130h]
  const char *v216; // [rsp+160h] [rbp-120h] BYREF
  __int64 v217; // [rsp+168h] [rbp-118h]
  char *v218; // [rsp+170h] [rbp-110h] BYREF
  __int64 v219; // [rsp+178h] [rbp-108h]
  __int64 v220; // [rsp+180h] [rbp-100h]
  const __m128i *v221; // [rsp+1A0h] [rbp-E0h] BYREF
  unsigned __int64 v222; // [rsp+1A8h] [rbp-D8h]
  _BYTE v223[64]; // [rsp+1B0h] [rbp-D0h] BYREF
  char *v224; // [rsp+1F0h] [rbp-90h] BYREF
  unsigned __int64 v225; // [rsp+1F8h] [rbp-88h]
  char *v226; // [rsp+200h] [rbp-80h] BYREF
  __int64 v227; // [rsp+208h] [rbp-78h]
  __int64 v228; // [rsp+210h] [rbp-70h]
  int v229; // [rsp+220h] [rbp-60h] BYREF
  unsigned __int64 v230; // [rsp+228h] [rbp-58h]
  int *v231; // [rsp+230h] [rbp-50h]
  int *v232; // [rsp+238h] [rbp-48h]
  __int64 v233; // [rsp+240h] [rbp-40h]

  v2 = *(unsigned int *)(a2 + 104);
  v177 = (__int64 *)(a2 + 96);
  v3 = *(_QWORD *)(a2 + 96);
  v4 = *(_BYTE *)(a2 + 40) == 0;
  v198 = 0;
  v199 = 0;
  v5 = v3 + 8 * v2;
  v200 = 0;
  v201 = 0;
  if ( v4 )
  {
    v33 = 0;
    v34 = 0;
    goto LABEL_36;
  }
  v6 = a2;
  v7 = 0;
  s2 = 0;
  v178 = 0;
  while ( 1 )
  {
    v8 = *(char **)(v6 + 56);
    v9 = *(_QWORD *)(v6 + 64);
    v196 = v8;
    v197 = v9;
    v10 = *v8;
    if ( v9 )
    {
      --v9;
      ++v8;
    }
    v221 = (const __m128i *)v8;
    v222 = v9;
    v11 = 0;
    v12 = sub_C935B0(&v221, byte_3F15413, 6, 0);
    v13 = v222;
    if ( v12 < v222 )
    {
      v11 = v222 - v12;
      v13 = v12;
    }
    v224 = &v221->m128i_i8[v13];
    v225 = v11;
    v14 = sub_C93740((__int64 *)&v224, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL) + 1;
    v196 = v224;
    if ( v14 > v225 )
      v14 = v225;
    v16 = v225 - v11 + v14;
    if ( v16 > v225 )
      v16 = v225;
    v197 = v16;
    v221 = (const __m128i *)v223;
    v222 = 0x400000000LL;
    sub_C93960(&v196, (__int64)&v221, 32, -1, 1, v15);
    if ( v10 == 102 )
    {
      v36 = (unsigned __int64)v221;
      v37 = 16LL * (unsigned int)v222;
      v176 = &v221[(unsigned __int64)v37 / 0x10];
      v38 = v37 >> 4;
      v39 = v37 >> 6;
      if ( v39 )
      {
        v183 = &v221[4 * v39];
        do
        {
          v40 = *(_QWORD *)(v36 + 8);
          v189 = *(const void **)v36;
          v41 = sub_C92610();
          v42 = sub_C92860((__int64 *)(v6 + 72), v189, v40, v41);
          if ( v42 != -1 )
          {
            v43 = *(_QWORD *)(v6 + 72);
            v44 = (__int64 *)(v43 + 8LL * v42);
            if ( v44 != (__int64 *)(v43 + 8LL * *(unsigned int *)(v6 + 80)) )
            {
              if ( !v7 )
                goto LABEL_49;
              v45 = *v44;
              if ( *(_QWORD *)(v45 + 16) == v7 && !memcmp(*(const void **)(v45 + 8), s2, v7) )
                goto LABEL_49;
            }
          }
          v46 = v36 + 16;
          v187 = *(_QWORD *)(v36 + 16);
          v191 = *(_QWORD *)(v36 + 24);
          v63 = sub_C92610();
          v64 = sub_C92860((__int64 *)(v6 + 72), (const void *)v187, v191, v63);
          if ( v64 != -1 )
          {
            v65 = *(_QWORD *)(v6 + 72);
            v66 = (__int64 *)(v65 + 8LL * v64);
            if ( v66 != (__int64 *)(v65 + 8LL * *(unsigned int *)(v6 + 80)) )
            {
              if ( !v7 )
                goto LABEL_50;
              v67 = *v66;
              if ( *(_QWORD *)(v67 + 16) == v7 && !memcmp(*(const void **)(v67 + 8), s2, v7) )
                goto LABEL_50;
            }
          }
          v46 = v36 + 32;
          v187 = *(_QWORD *)(v36 + 32);
          v192 = *(_QWORD *)(v36 + 40);
          v68 = sub_C92610();
          v69 = sub_C92860((__int64 *)(v6 + 72), (const void *)v187, v192, v68);
          if ( v69 != -1 )
          {
            v70 = *(_QWORD *)(v6 + 72);
            v71 = (__int64 *)(v70 + 8LL * v69);
            if ( v71 != (__int64 *)(v70 + 8LL * *(unsigned int *)(v6 + 80)) )
            {
              if ( !v7 )
                goto LABEL_50;
              v72 = *v71;
              if ( *(_QWORD *)(v72 + 16) == v7 && !memcmp(*(const void **)(v72 + 8), s2, v7) )
                goto LABEL_50;
            }
          }
          v46 = v36 + 48;
          v187 = *(_QWORD *)(v36 + 48);
          v193 = *(_QWORD *)(v36 + 56);
          v73 = sub_C92610();
          v74 = sub_C92860((__int64 *)(v6 + 72), (const void *)v187, v193, v73);
          if ( v74 != -1 )
          {
            v75 = *(_QWORD *)(v6 + 72);
            v76 = (__int64 *)(v75 + 8LL * v74);
            if ( v76 != (__int64 *)(v75 + 8LL * *(unsigned int *)(v6 + 80)) )
            {
              if ( !v7 )
                goto LABEL_50;
              v77 = *v76;
              if ( *(_QWORD *)(v77 + 16) == v7 && !memcmp(*(const void **)(v77 + 8), s2, v7) )
                goto LABEL_50;
            }
          }
          v36 += 64LL;
        }
        while ( v183 != (const __m128i *)v36 );
        v38 = (__int64)((__int64)v176->m128i_i64 - v36) >> 4;
      }
      switch ( v38 )
      {
        case 2LL:
          v46 = v36;
          v195 = (__int64 *)(v6 + 72);
          break;
        case 3LL:
          v144 = *(_QWORD *)(v36 + 8);
          v145 = *(const void **)v36;
          v146 = sub_C92610();
          v195 = (__int64 *)(v6 + 72);
          v147 = sub_C92860((__int64 *)(v6 + 72), v145, v144, v146);
          if ( v147 != -1 )
          {
            v148 = *(_QWORD *)(v6 + 72);
            v149 = (__int64 *)(v148 + 8LL * v147);
            if ( v149 != (__int64 *)(v148 + 8LL * *(unsigned int *)(v6 + 80)) )
            {
              if ( !v7 )
              {
LABEL_49:
                v46 = v36;
                goto LABEL_50;
              }
              v150 = *v149;
              if ( *(_QWORD *)(v150 + 16) == v7 )
              {
                v46 = v36;
                if ( !memcmp(*(const void **)(v150 + 8), s2, v7) )
                {
LABEL_50:
                  if ( v176 == (const __m128i *)v46 )
                  {
LABEL_83:
                    v5 = *(_QWORD *)(v6 + 96) + 8LL * *(unsigned int *)(v6 + 104);
LABEL_84:
                    v21 = v221;
                    v7 = 0;
                    s2 = byte_3F871B3;
                    goto LABEL_32;
                  }
                  v47 = 1;
                  v190 = (__int64 *)(v6 + 120);
                  if ( (unsigned int)v222 <= 1 )
                  {
LABEL_60:
                    v59 = (char *)v221->m128i_i64[0];
                    v60 = v221->m128i_u64[1];
                    v61 = sub_C92610();
                    v62 = sub_C92740((__int64)v177, v59, v60, v61);
                    v19 = *(_QWORD *)(v6 + 96) + 8LL * v62;
                    if ( *(_QWORD *)v19 )
                    {
                      if ( *(_QWORD *)v19 != -8 )
                      {
                        LOWORD(v228) = 1283;
                        v224 = "duplicate profile for function '";
                        v17 = v221->m128i_i64[0];
                        LOWORD(v209) = 770;
                        v226 = (char *)v17;
                        v171 = v209;
                        v227 = v221->m128i_i64[1];
                        *((_QWORD *)&v168 + 1) = v208;
                        v207 = "'";
                        *(_QWORD *)&v168 = "'";
                        *((_QWORD *)&v164 + 1) = v206;
                        goto LABEL_87;
                      }
                      --*(_DWORD *)(v6 + 112);
                    }
                    v194 = (_QWORD *)v19;
                    v96 = sub_C7D670(v60 + 153, 8);
                    v97 = v194;
                    v98 = (_QWORD *)v96;
                    if ( v60 )
                    {
                      memcpy((void *)(v96 + 152), v59, v60);
                      v97 = v194;
                    }
                    *((_BYTE *)v98 + v60 + 152) = 0;
                    *v98 = v60;
                    memset(v98 + 1, 0, 0x90u);
                    v98[1] = v98 + 3;
                    v98[2] = 0x300000000LL;
                    v98[9] = v98 + 11;
                    v98[10] = 0x100000000LL;
                    *v97 = v98;
                    ++*(_DWORD *)(v6 + 108);
                    v5 = *(_QWORD *)(v6 + 96) + 8LL * (unsigned int)sub_C929D0(v177, v62);
                    if ( !*(_QWORD *)v5 || *(_QWORD *)v5 == -8 )
                    {
                      v99 = (__int64 *)(v5 + 8);
                      do
                      {
                        do
                        {
                          v100 = *v99;
                          v5 = (__int64)v99++;
                        }
                        while ( v100 == -8 );
                      }
                      while ( !v100 );
                    }
                    ++v198;
                    if ( (_DWORD)v200 )
                    {
                      v120 = 4 * v200;
                      v101 = (unsigned int)v201;
                      if ( (unsigned int)(4 * v200) < 0x40 )
                        v120 = 64;
                      if ( v120 >= (unsigned int)v201 )
                        goto LABEL_119;
                      v121 = v199;
                      v122 = 2LL * (unsigned int)v201;
                      if ( (_DWORD)v200 == 1 )
                      {
                        v124 = 64;
                      }
                      else
                      {
                        _BitScanReverse(&v123, v200 - 1);
                        v124 = 1 << (33 - (v123 ^ 0x1F));
                        if ( v124 < 64 )
                          v124 = 64;
                        if ( (_DWORD)v201 == v124 )
                        {
                          v200 = 0;
                          v162 = &v199[v122];
                          do
                          {
                            if ( v121 )
                            {
                              *v121 = -1;
                              v121[1] = -1;
                            }
                            v121 += 2;
                          }
                          while ( v162 != v121 );
                          goto LABEL_122;
                        }
                      }
                      sub_C7D6A0((__int64)v199, v122 * 4, 4);
                      v125 = ((((((((4 * v124 / 3u + 1) | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 2)
                                | (4 * v124 / 3u + 1)
                                | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 4)
                              | (((4 * v124 / 3u + 1) | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 2)
                              | (4 * v124 / 3u + 1)
                              | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 8)
                            | (((((4 * v124 / 3u + 1) | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 2)
                              | (4 * v124 / 3u + 1)
                              | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 4)
                            | (((4 * v124 / 3u + 1) | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 2)
                            | (4 * v124 / 3u + 1)
                            | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 16;
                      LODWORD(v201) = (v125
                                     | (((((((4 * v124 / 3u + 1) | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 2)
                                         | (4 * v124 / 3u + 1)
                                         | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 4)
                                       | (((4 * v124 / 3u + 1) | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 2)
                                       | (4 * v124 / 3u + 1)
                                       | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 8)
                                     | (((((4 * v124 / 3u + 1) | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 2)
                                       | (4 * v124 / 3u + 1)
                                       | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 4)
                                     | (((4 * v124 / 3u + 1) | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 2)
                                     | (4 * v124 / 3u + 1)
                                     | ((4 * v124 / 3u + 1) >> 1))
                                    + 1;
                      v126 = (_DWORD *)sub_C7D670(
                                         8
                                       * ((v125
                                         | (((((((4 * v124 / 3u + 1) | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 2)
                                             | (4 * v124 / 3u + 1)
                                             | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 4)
                                           | (((4 * v124 / 3u + 1) | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 2)
                                           | (4 * v124 / 3u + 1)
                                           | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 8)
                                         | (((((4 * v124 / 3u + 1) | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 2)
                                           | (4 * v124 / 3u + 1)
                                           | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 4)
                                         | (((4 * v124 / 3u + 1) | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1)) >> 2)
                                         | (4 * v124 / 3u + 1)
                                         | ((unsigned __int64)(4 * v124 / 3u + 1) >> 1))
                                        + 1),
                                         4);
                      v200 = 0;
                      v199 = v126;
                      for ( i = &v126[2 * (unsigned int)v201]; i != v126; v126 += 2 )
                      {
                        if ( v126 )
                        {
                          *v126 = -1;
                          v126[1] = -1;
                        }
                      }
                    }
                    else if ( HIDWORD(v200) )
                    {
                      v101 = (unsigned int)v201;
                      if ( (unsigned int)v201 <= 0x40 )
                      {
LABEL_119:
                        v102 = v199;
                        v103 = &v199[2 * v101];
                        if ( v199 != v103 )
                        {
                          do
                          {
                            *v102 = -1;
                            v102 += 2;
                            *(v102 - 1) = -1;
                          }
                          while ( v103 != v102 );
                        }
                        v200 = 0;
                        goto LABEL_122;
                      }
                      sub_C7D6A0((__int64)v199, 8LL * (unsigned int)v201, 4);
                      v199 = 0;
                      v200 = 0;
                      LODWORD(v201) = 0;
                    }
LABEL_122:
                    v178 = 0;
                    goto LABEL_84;
                  }
                  while ( 1 )
                  {
                    while ( 1 )
                    {
                      v48 = v221;
                      v49 = &v221[v47];
                      v50 = (const void *)v49->m128i_i64[0];
                      v51 = v49->m128i_u64[1];
                      v52 = sub_C92610();
                      v53 = sub_C92740((__int64)v190, v50, v51, v52);
                      v54 = (_QWORD *)(*(_QWORD *)(v6 + 120) + 8LL * v53);
                      if ( *v54 )
                        break;
LABEL_57:
                      v184 = v54;
                      v188 = v53;
                      v55 = sub_C7D670(v51 + 25, 8);
                      v56 = v188;
                      v57 = v184;
                      v58 = (char *)v55;
                      if ( v51 )
                      {
                        s2a = (char *)v55;
                        memcpy((void *)(v55 + 24), v50, v51);
                        v56 = v188;
                        v57 = v184;
                        v58 = s2a;
                      }
                      v58[v51 + 24] = 0;
                      ++v47;
                      *(_QWORD *)v58 = v51;
                      *(__m128i *)(v58 + 8) = _mm_loadu_si128(v48);
                      *v57 = v58;
                      ++*(_DWORD *)(v6 + 132);
                      sub_C929D0(v190, v56);
                      if ( (unsigned int)v222 <= v47 )
                        goto LABEL_60;
                    }
                    if ( *v54 == -8 )
                    {
                      --*(_DWORD *)(v6 + 136);
                      goto LABEL_57;
                    }
                    if ( (unsigned int)v222 <= ++v47 )
                      goto LABEL_60;
                  }
                }
              }
            }
          }
          v46 = v36 + 16;
          break;
        case 1LL:
          v46 = v36;
          v195 = (__int64 *)(v6 + 72);
LABEL_178:
          v137 = *(_QWORD *)(v46 + 8);
          v138 = *(const void **)v46;
          v139 = sub_C92610();
          v140 = sub_C92860(v195, v138, v137, v139);
          if ( v140 == -1 )
            goto LABEL_83;
          v141 = *(_QWORD *)(v6 + 72);
          v142 = (__int64 *)(v141 + 8LL * v140);
          if ( v142 == (__int64 *)(v141 + 8LL * *(unsigned int *)(v6 + 80)) )
            goto LABEL_83;
          if ( v7 )
          {
            v143 = *v142;
            if ( *(_QWORD *)(v143 + 16) != v7 || memcmp(*(const void **)(v143 + 8), s2, v7) )
              goto LABEL_83;
          }
          goto LABEL_50;
        default:
          goto LABEL_83;
      }
      v130 = *(_QWORD *)(v46 + 8);
      v131 = *(const void **)v46;
      v132 = sub_C92610();
      v133 = sub_C92860(v195, v131, v130, v132);
      if ( v133 != -1 )
      {
        v134 = *(_QWORD *)(v6 + 72);
        v135 = (__int64 *)(v134 + 8LL * v133);
        if ( v135 != (__int64 *)(v134 + 8LL * *(unsigned int *)(v6 + 80)) )
        {
          if ( !v7 )
            goto LABEL_50;
          v136 = *v135;
          if ( *(_QWORD *)(v136 + 16) == v7 && !memcmp(*(const void **)(v136 + 8), s2, v7) )
            goto LABEL_50;
        }
      }
      v46 += 16LL;
      goto LABEL_178;
    }
    if ( v10 <= 102 )
    {
      if ( v10 != 64 )
      {
        if ( v10 != 99 )
          goto LABEL_37;
        v21 = v221;
        if ( v5 != *(_QWORD *)(v6 + 96) + 8LL * *(unsigned int *)(v6 + 104) )
        {
          v175 = &v221[(unsigned int)v222];
          if ( v175 != v221 )
          {
            v182 = (__int64 *)v5;
            v22 = (unsigned __int64)v221;
            LODWORD(v187) = 0;
            v173 = v7;
            while ( 1 )
            {
              v23 = *(char **)v22;
              v24 = *(_QWORD *)(v22 + 8);
              sub_2D50FB0((__int64)&v216, (__int64 *)v6, *(char **)v22, v24, v19, v20);
              v25 = v217 & 1;
              v26 = (2 * (v217 & 1)) | v217 & 0xFD;
              LOBYTE(v217) = v26;
              if ( v25 )
              {
                LOBYTE(v217) = v26 & 0xFD;
                v157 = (unsigned __int64)v216;
                v216 = 0;
                *a1 = v157 | 1;
                goto LABEL_38;
              }
              v27 = (unsigned int)v201;
              if ( !(_DWORD)v201 )
              {
                ++v198;
                goto LABEL_196;
              }
              v28 = 0;
              v20 = (unsigned int)(v201 - 1);
              v29 = 1;
              for ( j = v20
                      & (((0xBF58476D1CE4E5B9LL
                         * ((unsigned int)(37 * HIDWORD(v216))
                          | ((unsigned __int64)(unsigned int)(37 * (_DWORD)v216) << 32))) >> 31)
                       ^ (756364221 * HIDWORD(v216))); ; j = v20 & v32 )
              {
                v31 = (const char **)&v199[2 * j];
                v19 = *(unsigned int *)v31;
                if ( v216 == *v31 )
                {
                  v226 = v23;
                  v27 = v6;
                  v224 = "duplicate basic block id found '";
                  v210[2] = (unsigned __int64)"'";
                  LOWORD(v211) = 770;
                  *((_QWORD *)&v170 + 1) = v210[3];
                  *(_QWORD *)&v170 = "'";
                  *((_QWORD *)&v166 + 1) = v210[1];
                  *(_QWORD *)&v166 = &v224;
                  v210[0] = (unsigned __int64)&v224;
                  v227 = v24;
                  LOWORD(v228) = 1283;
                  sub_2D507F0(a1, (__int64 *)v6, (__int64)v31, (unsigned int)v216, v19, v20, v166, v170, v211);
                  if ( (v217 & 2) != 0 )
                    goto LABEL_152;
                  if ( (v217 & 1) != 0 && v216 )
                    (*(void (__fastcall **)(const char *))(*(_QWORD *)v216 + 8LL))(v216);
                  goto LABEL_38;
                }
                if ( (_DWORD)v19 == -1 )
                  break;
                if ( (_DWORD)v19 == -2 && *((_DWORD *)v31 + 1) == -2 && !v28 )
                  v28 = (const char **)&v199[2 * j];
LABEL_27:
                v32 = v29 + j;
                ++v29;
              }
              if ( *((_DWORD *)v31 + 1) != -1 )
                goto LABEL_27;
              if ( v28 )
                v31 = v28;
              ++v198;
              v110 = v200 + 1;
              if ( 4 * ((int)v200 + 1) < (unsigned int)(3 * v201) )
              {
                if ( (int)v201 - HIDWORD(v200) - v110 > (unsigned int)v201 >> 3 )
                  goto LABEL_139;
                sub_2D51B90((__int64)&v198, v201);
                if ( (_DWORD)v201 )
                {
                  v19 = (unsigned int)v216;
                  v27 = (unsigned int)(v201 - 1);
                  v20 = 1;
                  v31 = 0;
                  for ( k = v27
                          & (((0xBF58476D1CE4E5B9LL
                             * ((unsigned int)(37 * HIDWORD(v216))
                              | ((unsigned __int64)(unsigned int)(37 * (_DWORD)v216) << 32))) >> 31)
                           ^ (756364221 * HIDWORD(v216))); ; k = v27 & v161 )
                  {
                    v159 = (const char **)&v199[2 * k];
                    v160 = *(_DWORD *)v159;
                    if ( v216 == *v159 )
                    {
                      v31 = (const char **)&v199[2 * k];
                      v110 = v200 + 1;
                      goto LABEL_139;
                    }
                    if ( v160 == -1 )
                    {
                      if ( *((_DWORD *)v159 + 1) == -1 )
                      {
                        v110 = v200 + 1;
                        if ( !v31 )
                          v31 = v159;
                        goto LABEL_139;
                      }
                    }
                    else if ( v160 == -2 && *((_DWORD *)v159 + 1) == -2 && !v31 )
                    {
                      v31 = (const char **)&v199[2 * k];
                    }
                    v161 = v20 + k;
                    v20 = (unsigned int)(v20 + 1);
                  }
                }
LABEL_233:
                LODWORD(v200) = v200 + 1;
                BUG();
              }
LABEL_196:
              sub_2D51B90((__int64)&v198, 2 * v201);
              if ( !(_DWORD)v201 )
                goto LABEL_233;
              v19 = (unsigned int)v216;
              v27 = (unsigned int)(v201 - 1);
              v153 = 0;
              v20 = 1;
              v154 = v27
                   & (((0xBF58476D1CE4E5B9LL
                      * ((unsigned int)(37 * HIDWORD(v216)) | ((unsigned __int64)(unsigned int)(37 * (_DWORD)v216) << 32))) >> 31)
                    ^ (756364221 * HIDWORD(v216)));
              while ( 2 )
              {
                v31 = (const char **)&v199[2 * v154];
                v155 = *(_DWORD *)v31;
                if ( v216 == *v31 )
                {
                  v110 = v200 + 1;
                  goto LABEL_139;
                }
                if ( v155 != -1 )
                {
                  if ( v155 == -2 && *((_DWORD *)v31 + 1) == -2 && !v153 )
                    v153 = (const char **)&v199[2 * v154];
                  goto LABEL_204;
                }
                if ( *((_DWORD *)v31 + 1) != -1 )
                {
LABEL_204:
                  v156 = v20 + v154;
                  v20 = (unsigned int)(v20 + 1);
                  v154 = v27 & v156;
                  continue;
                }
                break;
              }
              v110 = v200 + 1;
              if ( v153 )
                v31 = v153;
LABEL_139:
              LODWORD(v200) = v110;
              if ( *(_DWORD *)v31 != -1 || *((_DWORD *)v31 + 1) != -1 )
                --HIDWORD(v200);
              *v31 = v216;
              v111 = *v182;
              if ( (v217 & 2) != 0 )
                goto LABEL_152;
              v112 = *(unsigned int *)(v111 + 20);
              v27 = (unsigned int)v216;
              v113 = HIDWORD(v216);
              v114 = v187 + 1;
              v115 = *(unsigned int *)(v111 + 16);
              v116 = v115;
              if ( v115 >= v112 )
              {
                v128 = v216;
                v19 = (v187 << 32) | v178;
                v116 = v115 + 1;
                if ( v112 < v115 + 1 )
                {
                  v27 = v111 + 24;
                  v187 = (v187 << 32) | v178;
                  sub_C8D5F0(v111 + 8, (const void *)(v111 + 24), v116, 0x10u, v19, v20);
                  v115 = *(unsigned int *)(v111 + 16);
                  v19 = v187;
                }
                v129 = (const char **)(*(_QWORD *)(v111 + 8) + 16 * v115);
                *v129 = v128;
                v129[1] = (const char *)v19;
                ++*(_DWORD *)(v111 + 16);
                v119 = (v217 & 2) != 0;
LABEL_145:
                if ( !v119 )
                  goto LABEL_146;
LABEL_152:
                sub_2D51670(&v216, v27);
              }
              v117 = (_DWORD *)(*(_QWORD *)(v111 + 8) + 16 * v115);
              if ( v117 )
              {
                *v117 = (_DWORD)v216;
                v117[1] = v113;
                v117[2] = v178;
                v27 = (unsigned int)v187;
                v117[3] = v187;
                v118 = v217;
                ++*(_DWORD *)(v111 + 16);
                v119 = (v118 & 2) != 0;
                goto LABEL_145;
              }
              v116 = (unsigned int)(v116 + 1);
              *(_DWORD *)(v111 + 16) = v116;
LABEL_146:
              if ( (v217 & 1) != 0 && v216 )
                (*(void (__fastcall **)(const char *, __int64, unsigned __int64))(*(_QWORD *)v216 + 8LL))(
                  v216,
                  v27,
                  v116);
              v22 += 16LL;
              if ( v175 == (const __m128i *)v22 )
              {
                v5 = (__int64)v182;
                v7 = v173;
                v21 = v221;
                break;
              }
              LODWORD(v187) = v114;
            }
          }
          ++v178;
        }
        goto LABEL_32;
      }
      goto LABEL_31;
    }
    if ( v10 == 109 )
    {
      if ( (_DWORD)v222 == 1 )
      {
        v108 = sub_C81F40((char *)v221->m128i_i64[0], v221->m128i_u64[1], 0);
        v21 = v221;
        s2 = v108;
        v7 = v109;
        goto LABEL_32;
      }
      v224 = "invalid module name value: '";
      v226 = v196;
      v227 = v197;
      LOWORD(v228) = 1283;
      v203 = "'";
      LOWORD(v205) = 770;
      v171 = v205;
      *((_QWORD *)&v168 + 1) = v204;
      *(_QWORD *)&v168 = "'";
      *((_QWORD *)&v164 + 1) = v202;
LABEL_87:
      *(_QWORD *)&v164 = &v224;
      sub_2D507F0(a1, (__int64 *)v6, v17, v18, v19, v20, v164, v168, v171);
      goto LABEL_38;
    }
    if ( v10 != 112 )
    {
LABEL_37:
      LOBYTE(v218) = v10;
      v216 = "invalid specifier: '";
      LOWORD(v220) = 2051;
      v226 = "'";
      LOWORD(v228) = 770;
      *((_QWORD *)&v167 + 1) = v227;
      *(_QWORD *)&v167 = "'";
      *((_QWORD *)&v163 + 1) = v225;
      *(_QWORD *)&v163 = &v216;
      v224 = (char *)&v216;
      sub_2D507F0(a1, (__int64 *)v6, 770, v18, v19, v20, v163, v167, v228);
      goto LABEL_38;
    }
    if ( v5 != *(_QWORD *)(v6 + 96) + 8LL * *(unsigned int *)(v6 + 104) )
      break;
LABEL_31:
    v21 = v221;
LABEL_32:
    if ( v21 != (const __m128i *)v223 )
      _libc_free((unsigned __int64)v21);
    sub_C7C5C0(v6 + 8);
    if ( !*(_BYTE *)(v6 + 40) )
    {
      v33 = (__int64)v199;
      v34 = 8LL * (unsigned int)v201;
LABEL_36:
      *a1 = 1;
      goto LABEL_41;
    }
  }
  v229 = 0;
  v224 = (char *)&v226;
  v187 = (__int64)&v226;
  v225 = 0x500000000LL;
  v231 = &v229;
  v232 = &v229;
  v230 = 0;
  v233 = 0;
  v78 = *(_QWORD *)v5;
  v216 = (const char *)&v218;
  v217 = 0xC00000000LL;
  v79 = *(unsigned int *)(v78 + 80);
  v80 = v79 + 1;
  v81 = v79;
  if ( v79 + 1 > (unsigned __int64)*(unsigned int *)(v78 + 84) )
  {
    v151 = *(_QWORD *)(v78 + 72);
    v152 = v78 + 72;
    if ( v151 > (unsigned __int64)&v216 || (v186 = *(_QWORD *)(v78 + 72), (unsigned __int64)&v216 >= v151 + (v79 << 6)) )
    {
      sub_2D516E0(v152, v80, (__int64)&v216, v151, v19, v20);
      v79 = *(unsigned int *)(v78 + 80);
      v82 = *(_QWORD *)(v78 + 72);
      v83 = &v216;
      v81 = v79;
    }
    else
    {
      sub_2D516E0(v152, v80, (__int64)&v216, v151, v19, v20);
      v82 = *(_QWORD *)(v78 + 72);
      v79 = *(unsigned int *)(v78 + 80);
      v83 = (const char **)((char *)&v216 + v82 - v186);
      v81 = v79;
    }
  }
  else
  {
    v82 = *(_QWORD *)(v78 + 72);
    v83 = &v216;
  }
  v84 = (_QWORD *)((v79 << 6) + v82);
  if ( v84 )
  {
    *v84 = v84 + 2;
    v84[1] = 0xC00000000LL;
    if ( *((_DWORD *)v83 + 2) )
      sub_2D50200((__int64)v84, (char **)v83, v82, v81, v19, v20);
    LODWORD(v81) = *(_DWORD *)(v78 + 80);
  }
  *(_DWORD *)(v78 + 80) = v81 + 1;
  if ( v216 != (const char *)&v218 )
    _libc_free((unsigned __int64)v216);
  if ( !(_DWORD)v222 )
  {
LABEL_104:
    sub_2D50620(v230);
    if ( v224 != (char *)&v226 )
      _libc_free((unsigned __int64)v224);
    goto LABEL_31;
  }
  v174 = v7;
  v185 = (__int64 *)v6;
  v85 = 0;
  while ( 1 )
  {
    m128i_i64 = v221[v85].m128i_i64;
    v87 = *m128i_i64;
    v88 = m128i_i64[1];
    v210[0] = 0;
    if ( sub_C93C90(v87, v88, 0xAu, v210) )
      break;
    if ( v85 )
    {
      LODWORD(v212[0]) = v210[0];
      sub_2D51E10((__int64)&v216, (__int64)&v224, (unsigned int *)v212, v90, v91);
      if ( !(_BYTE)v218 )
      {
        v214 = v88;
        v90 = 1283;
        v212[0] = "duplicate cloned block in path: '";
        v104 = (const char **)v212;
        v218 = "'";
        v105 = v185;
        LOWORD(v220) = 770;
        v213 = (char *)v87;
        v172 = v220;
        LOWORD(v215) = 1283;
        *((_QWORD *)&v169 + 1) = v219;
        v216 = (const char *)v212;
        *(_QWORD *)&v169 = "'";
        *((_QWORD *)&v165 + 1) = v217;
        goto LABEL_125;
      }
    }
    v93 = v210[0];
    v94 = *(_QWORD *)(*(_QWORD *)v5 + 72LL) + ((unsigned __int64)*(unsigned int *)(*(_QWORD *)v5 + 80LL) << 6) - 64;
    v95 = *(unsigned int *)(v94 + 8);
    if ( v95 + 1 > (unsigned __int64)*(unsigned int *)(v94 + 12) )
    {
      sub_C8D5F0(
        *(_QWORD *)(*(_QWORD *)v5 + 72LL) + ((unsigned __int64)*(unsigned int *)(*(_QWORD *)v5 + 80LL) << 6) - 64,
        (const void *)(v94 + 16),
        v95 + 1,
        4u,
        v91,
        v92);
      v95 = *(unsigned int *)(v94 + 8);
    }
    ++v85;
    *(_DWORD *)(*(_QWORD *)v94 + 4 * v95) = v93;
    ++*(_DWORD *)(v94 + 8);
    if ( v85 >= (unsigned int)v222 )
    {
      v7 = v174;
      v6 = (__int64)v185;
      goto LABEL_104;
    }
  }
  v91 = 770;
  v219 = v88;
  v216 = "unsigned integer expected: '";
  v104 = &v216;
  v213 = "'";
  v105 = v185;
  LOWORD(v215) = 770;
  v218 = (char *)v87;
  v172 = v215;
  LOWORD(v220) = 1283;
  *((_QWORD *)&v169 + 1) = v214;
  v212[0] = &v216;
  *(_QWORD *)&v169 = "'";
  *((_QWORD *)&v165 + 1) = v212[1];
LABEL_125:
  *(_QWORD *)&v165 = v104;
  sub_2D507F0(a1, v105, v89, v90, v91, v92, v165, v169, v172);
  v106 = v230;
  while ( v106 )
  {
    sub_2D50620(*(_QWORD *)(v106 + 24));
    v107 = v106;
    v106 = *(_QWORD *)(v106 + 16);
    j_j___libc_free_0(v107);
  }
  if ( v224 != (char *)&v226 )
    _libc_free((unsigned __int64)v224);
LABEL_38:
  if ( v221 != (const __m128i *)v223 )
    _libc_free((unsigned __int64)v221);
  v33 = (__int64)v199;
  v34 = 8LL * (unsigned int)v201;
LABEL_41:
  sub_C7D6A0(v33, v34, 4);
  return a1;
}
