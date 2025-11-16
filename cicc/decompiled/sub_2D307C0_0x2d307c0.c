// Function: sub_2D307C0
// Address: 0x2d307c0
//
__int64 __fastcall sub_2D307C0(
        __int64 a1,
        __int64 a2,
        _QWORD *a3,
        __int64 a4,
        __int64 a5,
        _DWORD *a6,
        __m128i a7,
        __m128i a8)
{
  __int64 v8; // r15
  _QWORD *v9; // rax
  _QWORD *i; // r15
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r12
  __int64 v15; // rbx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // r13
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 v21; // rsi
  char v22; // al
  __int64 v23; // r14
  unsigned __int64 v24; // rbx
  unsigned __int64 n; // r12
  __int64 v26; // r8
  __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // r8
  __int64 *v30; // rdi
  __int64 *v31; // r14
  __int64 *v32; // rbx
  __int64 v33; // r13
  __int64 v34; // rsi
  __int64 v35; // r8
  __int64 v36; // rax
  _QWORD *v37; // rbx
  _QWORD *v38; // r12
  unsigned __int64 v39; // rdi
  __int64 v41; // rdx
  __int64 v42; // rdi
  int v43; // r11d
  unsigned int j; // eax
  _QWORD *v45; // rcx
  unsigned int v46; // eax
  int v47; // esi
  __int64 v48; // rcx
  unsigned int k; // edx
  __m128i *v50; // rax
  __int64 v51; // r11
  __int64 v52; // rax
  unsigned int v53; // eax
  __int64 v54; // rbx
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // rax
  unsigned __int64 v58; // rdx
  const __m128i *v59; // rbx
  __int64 v60; // rdx
  __int64 v61; // rsi
  int v62; // r10d
  unsigned int m; // eax
  _QWORD *v64; // r8
  unsigned int v65; // eax
  __int64 v66; // rax
  __int64 v67; // r13
  __int64 v68; // rdi
  __int64 v69; // rax
  __int64 v70; // rdx
  __int64 v71; // rbx
  __int64 v72; // rax
  __m128i v73; // xmm0
  __m128i v74; // rax
  __int64 v75; // rax
  _QWORD *v76; // r8
  __m128i v77; // xmm0
  int v78; // eax
  __int64 v79; // r9
  __m128i v80; // xmm0
  __int64 v81; // r8
  __int64 v82; // rdx
  __int64 v83; // rcx
  const __m128i *v84; // rax
  __m128i *v85; // rdx
  __int64 v86; // r15
  int v87; // eax
  char v88; // al
  __int64 v89; // rax
  __int64 v90; // rax
  __m128i *v91; // rsi
  _BYTE *v92; // rdi
  __int64 *v93; // r13
  __int64 v94; // rax
  __m128i v95; // xmm0
  __int64 v96; // rsi
  __int64 v97; // r14
  __int64 v98; // rax
  __int64 v99; // rax
  __int64 v100; // rbx
  __m128i v101; // xmm7
  __int64 v102; // r8
  __m128i v103; // xmm0
  __m128i v104; // xmm6
  __int64 v105; // rdx
  __int64 v106; // rcx
  const __m128i *v107; // rax
  __m128i *v108; // rdx
  __int64 v109; // rbx
  int v110; // eax
  __int64 v111; // rdx
  unsigned __int64 v112; // r8
  __m128i *v113; // r13
  const __m128i *v114; // rcx
  unsigned __int64 v115; // r9
  __m128i *v116; // rdx
  __int64 v117; // r9
  __int64 v118; // r13
  __int64 v119; // rax
  _QWORD *v120; // r12
  unsigned __int64 v121; // rdx
  __int64 v122; // rax
  __int64 v123; // rdx
  __m128i *v124; // rax
  __m128i *v125; // rsi
  __int64 v126; // rdx
  unsigned int v127; // esi
  int v128; // eax
  _QWORD *v129; // r8
  int v130; // eax
  _QWORD *v131; // rax
  unsigned int v132; // esi
  int v133; // eax
  _QWORD *v134; // rbx
  int v135; // eax
  _QWORD *v136; // rax
  __int64 *v137; // rax
  __int64 v138; // rdx
  __int64 v139; // rcx
  __int64 v140; // r9
  __int64 *v141; // rax
  __int64 v142; // rdx
  __int64 v143; // rcx
  __int64 v144; // r8
  __int64 v145; // r9
  int v146; // edx
  int v147; // esi
  _QWORD *v148; // r12
  int v149; // eax
  _QWORD *v150; // rax
  unsigned int v151; // edx
  __int64 v152; // r8
  __int64 v153; // r9
  __int64 v154; // rax
  __int64 v155; // rcx
  __int64 v156; // rdx
  _QWORD *v157; // rbx
  __int64 v158; // rax
  __int64 v159; // rdi
  __int64 v160; // r12
  const __m128i *v161; // r13
  __m128i *v162; // rsi
  unsigned __int64 v163; // rax
  const __m128i *v164; // r12
  __int64 v165; // rdi
  __int64 *v166; // rax
  int v167; // ecx
  __int64 v168; // rdx
  _QWORD *v169; // r14
  __int64 v170; // r15
  __int64 v171; // rax
  const __m128i *v172; // rbx
  __int8 v173; // si
  __int64 v174; // rcx
  unsigned __int64 v175; // r15
  __int64 v176; // rax
  __int64 v177; // r9
  int v178; // r12d
  __m128i *v179; // rsi
  unsigned int v180; // esi
  __int64 v181; // rdi
  int v182; // r11d
  __int64 v183; // r9
  __int64 v184; // rcx
  _DWORD *v185; // r8
  _DWORD *v186; // rax
  int v187; // edx
  __int64 v188; // rax
  _DWORD *v189; // r12
  char v190; // di
  unsigned __int64 v191; // r8
  __int64 v192; // rax
  unsigned __int64 v193; // rax
  int v194; // edi
  int v195; // ecx
  int v196; // r11d
  int v197; // r11d
  __int64 v198; // r10
  __int64 v199; // r8
  int v200; // esi
  int v201; // r9d
  _DWORD *v202; // rdx
  int v203; // r11d
  int v204; // r11d
  __int64 v205; // r10
  int v206; // r8d
  __int64 v207; // r9
  int v208; // esi
  _QWORD *v209; // rax
  const void *v210; // rsi
  __int64 v211; // r9
  const void *v212; // rsi
  unsigned __int64 v213; // r14
  const void *v214; // rsi
  __int64 v215; // rdi
  __int64 v216; // [rsp+0h] [rbp-2A0h]
  __int64 v217; // [rsp+0h] [rbp-2A0h]
  __int64 v218; // [rsp+8h] [rbp-298h]
  __int64 v220; // [rsp+18h] [rbp-288h]
  _QWORD *v223; // [rsp+30h] [rbp-270h]
  __int64 v226; // [rsp+48h] [rbp-258h]
  _QWORD *v227; // [rsp+50h] [rbp-250h]
  const __m128i *v228; // [rsp+50h] [rbp-250h]
  _QWORD *v229; // [rsp+58h] [rbp-248h]
  int v230; // [rsp+58h] [rbp-248h]
  int v231; // [rsp+58h] [rbp-248h]
  __int64 v232; // [rsp+60h] [rbp-240h]
  __int64 v233; // [rsp+68h] [rbp-238h]
  __int64 v234; // [rsp+68h] [rbp-238h]
  __int64 v235; // [rsp+70h] [rbp-230h]
  __int64 v236; // [rsp+70h] [rbp-230h]
  unsigned __int64 v237; // [rsp+78h] [rbp-228h]
  __int64 src; // [rsp+80h] [rbp-220h]
  __int64 srce; // [rsp+80h] [rbp-220h]
  __int64 srcf; // [rsp+80h] [rbp-220h]
  __int64 srcg; // [rsp+80h] [rbp-220h]
  int srca; // [rsp+80h] [rbp-220h]
  __m128i *srcb; // [rsp+80h] [rbp-220h]
  __m128i *srcc; // [rsp+80h] [rbp-220h]
  __int64 srci; // [rsp+80h] [rbp-220h]
  __int64 srcj; // [rsp+80h] [rbp-220h]
  _DWORD *srch; // [rsp+80h] [rbp-220h]
  int srcd; // [rsp+80h] [rbp-220h]
  __int64 v249; // [rsp+90h] [rbp-210h]
  const __m128i *v250; // [rsp+90h] [rbp-210h]
  __int64 v252; // [rsp+98h] [rbp-208h]
  __int64 v253; // [rsp+98h] [rbp-208h]
  __int64 v254; // [rsp+98h] [rbp-208h]
  __int64 v255; // [rsp+A0h] [rbp-200h] BYREF
  _QWORD *v256; // [rsp+A8h] [rbp-1F8h] BYREF
  __int64 v257[2]; // [rsp+B0h] [rbp-1F0h] BYREF
  __m128i v258; // [rsp+C0h] [rbp-1E0h] BYREF
  __int64 v259; // [rsp+D0h] [rbp-1D0h]
  __int64 v260; // [rsp+E0h] [rbp-1C0h] BYREF
  __int64 v261; // [rsp+E8h] [rbp-1B8h]
  __int64 v262; // [rsp+F0h] [rbp-1B0h]
  __int64 v263; // [rsp+F8h] [rbp-1A8h]
  __int64 v264; // [rsp+100h] [rbp-1A0h] BYREF
  _QWORD *v265; // [rsp+108h] [rbp-198h]
  __int64 v266; // [rsp+110h] [rbp-190h]
  unsigned int v267; // [rsp+118h] [rbp-188h]
  __m128i v268; // [rsp+120h] [rbp-180h] BYREF
  __m128i v269; // [rsp+130h] [rbp-170h] BYREF
  char v270; // [rsp+140h] [rbp-160h]
  _BYTE v271[24]; // [rsp+150h] [rbp-150h] BYREF
  __int64 v272; // [rsp+168h] [rbp-138h]
  __int64 v273; // [rsp+170h] [rbp-130h]
  _BYTE v274[40]; // [rsp+180h] [rbp-120h] BYREF
  _BYTE *v275; // [rsp+1B0h] [rbp-F0h] BYREF
  __int64 v276; // [rsp+1B8h] [rbp-E8h]
  _BYTE v277[48]; // [rsp+1C0h] [rbp-E0h] BYREF
  __int64 *v278; // [rsp+1F0h] [rbp-B0h] BYREF
  __int64 v279; // [rsp+1F8h] [rbp-A8h]
  _BYTE v280[48]; // [rsp+200h] [rbp-A0h] BYREF
  _BYTE v281[80]; // [rsp+230h] [rbp-70h] BYREF

  v8 = (__int64)a3;
  v275 = v277;
  v276 = 0x600000000LL;
  v279 = 0x600000000LL;
  v9 = *(_QWORD **)(a2 + 80);
  v260 = 0;
  v261 = 0;
  v262 = 0;
  v263 = 0;
  v264 = 0;
  v265 = 0;
  v266 = 0;
  v267 = 0;
  v278 = (__int64 *)v280;
  v227 = v9;
  v220 = a2 + 72;
  if ( v9 == (_QWORD *)(a2 + 72) )
    goto LABEL_23;
  do
  {
    if ( !v227 )
      BUG();
    for ( i = (_QWORD *)v227[4]; v227 + 3 != i; i = (_QWORD *)i[1] )
    {
      if ( !i )
        BUG();
      v11 = i[5];
      v249 = (__int64)(i - 3);
      if ( v11 )
      {
        v12 = sub_B14240(v11);
        v14 = v13;
        v15 = v12;
        if ( v13 != v12 )
        {
          while ( *(_BYTE *)(v15 + 32) )
          {
            v15 = *(_QWORD *)(v15 + 8);
            if ( v13 == v15 )
              goto LABEL_17;
          }
          if ( v13 != v15 )
          {
            v18 = sub_2D284E0(v15);
            if ( v18 )
            {
LABEL_12:
              v19 = (unsigned int)v279;
              v20 = (unsigned int)v279 + 1LL;
              if ( v20 > HIDWORD(v279) )
              {
                sub_C8D5F0((__int64)&v278, v280, v20, 8u, v16, v17);
                v19 = (unsigned int)v279;
              }
              v278[v19] = v18;
              LODWORD(v279) = v279 + 1;
              goto LABEL_16;
            }
            while ( 1 )
            {
              sub_AF48C0(v274, v15);
              v41 = *(unsigned int *)(a4 + 24);
              v42 = *(_QWORD *)(a4 + 8);
              *(_QWORD *)v271 = *(_QWORD *)v274;
              *(_QWORD *)&v271[8] = *(_QWORD *)&v274[32];
              if ( (_DWORD)v41 )
              {
                v43 = 1;
                for ( j = (v41 - 1)
                        & (((0xBF58476D1CE4E5B9LL
                           * ((*(_DWORD *)&v274[32] >> 9) ^ (*(_DWORD *)&v274[32] >> 4)
                            | ((unsigned __int64)((*(_DWORD *)v274 >> 9) ^ (*(_DWORD *)v274 >> 4)) << 32))) >> 31)
                         ^ (484763065 * ((*(_DWORD *)&v274[32] >> 9) ^ (*(_DWORD *)&v274[32] >> 4)))); ; j = (v41 - 1) & v46 )
                {
                  v45 = (_QWORD *)(v42 + 16LL * j);
                  if ( *(_QWORD *)v274 == *v45 && *(_QWORD *)&v274[32] == v45[1] )
                    break;
                  if ( *v45 == -4096 && v45[1] == -4096 )
                    goto LABEL_16;
                  v46 = v43 + j;
                  ++v43;
                }
                if ( v45 != (_QWORD *)(v42 + 16 * v41) )
                {
                  sub_F3D270((__int64)v281, (__int64)&v260, (const __m128i *)v274);
                  if ( v281[32] )
                  {
                    v47 = v267;
                    if ( !v267 )
                    {
                      ++v264;
                      *(_QWORD *)v281 = 0;
LABEL_167:
                      v47 = 2 * v267;
                      goto LABEL_168;
                    }
                    v48 = *(_QWORD *)v271;
                    srca = 1;
                    for ( k = (v267 - 1)
                            & (((0xBF58476D1CE4E5B9LL
                               * ((*(_DWORD *)&v271[8] >> 9) ^ (*(_DWORD *)&v271[8] >> 4)
                                | ((unsigned __int64)((*(_DWORD *)v271 >> 9) ^ (*(_DWORD *)v271 >> 4)) << 32))) >> 31)
                             ^ (484763065 * ((*(_DWORD *)&v271[8] >> 9) ^ (*(_DWORD *)&v271[8] >> 4))));
                          ;
                          k = (v267 - 1) & v151 )
                    {
                      v50 = (__m128i *)&v265[44 * k];
                      v51 = v50->m128i_i64[0];
                      if ( *v50 == *(_OWORD *)v271 )
                      {
                        v111 = v50[1].m128i_u32[2];
                        v112 = v50[1].m128i_u64[0];
                        v113 = v50 + 1;
                        v114 = (const __m128i *)v274;
                        v115 = v111 + 1;
                        v116 = (__m128i *)(v112 + 40 * v111);
                        if ( v50[1].m128i_u32[3] < v115 )
                        {
                          v236 = v50[1].m128i_i64[0];
                          v125 = v50 + 2;
                          srcb = v50;
                          if ( v112 > (unsigned __int64)v274 || v116 <= (__m128i *)v274 )
                          {
                            sub_C8D5F0((__int64)v50[1].m128i_i64, v125, v115, 0x28u, v112, v115);
                            v114 = (const __m128i *)v274;
                            v116 = (__m128i *)(srcb[1].m128i_i64[0] + 40LL * srcb[1].m128i_u32[2]);
                          }
                          else
                          {
                            sub_C8D5F0((__int64)v50[1].m128i_i64, v125, v115, 0x28u, v112, v115);
                            v126 = srcb[1].m128i_i64[0];
                            v114 = (const __m128i *)&v274[v126 - v236];
                            v116 = (__m128i *)(v126 + 40LL * srcb[1].m128i_u32[2]);
                          }
                        }
                        goto LABEL_142;
                      }
                      if ( v51 == -4096 )
                      {
                        if ( v50->m128i_i64[1] == -4096 )
                        {
                          if ( v18 )
                            v50 = (__m128i *)v18;
                          ++v264;
                          v146 = v266 + 1;
                          *(_QWORD *)v281 = v50;
                          if ( 4 * ((int)v266 + 1) >= 3 * v267 )
                            goto LABEL_167;
                          if ( v267 - HIDWORD(v266) - v146 > v267 >> 3 )
                            goto LABEL_169;
LABEL_168:
                          sub_2D2E6F0((__int64)&v264, v47);
                          sub_2D28EE0((__int64)&v264, (__int64 *)v271, (__int64 **)v281);
                          v48 = *(_QWORD *)v271;
                          v146 = v266 + 1;
                          v50 = *(__m128i **)v281;
LABEL_169:
                          LODWORD(v266) = v146;
                          if ( v50->m128i_i64[0] != -4096 || v50->m128i_i64[1] != -4096 )
                            --HIDWORD(v266);
                          v50->m128i_i64[0] = v48;
                          v113 = v50 + 1;
                          v114 = (const __m128i *)v274;
                          v50->m128i_i64[1] = *(_QWORD *)&v271[8];
                          v116 = v50 + 2;
                          v50[1].m128i_i64[0] = (__int64)v50[2].m128i_i64;
                          v50[1].m128i_i64[1] = 0x800000000LL;
LABEL_142:
                          *v116 = _mm_loadu_si128(v114);
                          v116[1] = _mm_loadu_si128(v114 + 1);
                          v116[2].m128i_i64[0] = v114[2].m128i_i64[0];
                          ++v113->m128i_i32[2];
                          break;
                        }
                      }
                      else if ( v51 == -8192 && v50->m128i_i64[1] == -8192 && !v18 )
                      {
                        v18 = (__int64)&v265[44 * k];
                      }
                      v151 = srca + k;
                      ++srca;
                    }
                  }
                }
              }
LABEL_16:
              v15 = *(_QWORD *)(v15 + 8);
              if ( v14 == v15 )
                break;
              if ( *(_BYTE *)(v15 + 32) )
                goto LABEL_16;
              if ( v14 == v15 )
                break;
              v18 = sub_2D284E0(v15);
              if ( v18 )
                goto LABEL_12;
            }
          }
        }
      }
LABEL_17:
      if ( *((_BYTE *)i - 24) == 85 )
      {
        v52 = *(i - 7);
        if ( v52 )
        {
          if ( !*(_BYTE *)v52 && *(_QWORD *)(v52 + 24) == i[7] && (*(_BYTE *)(v52 + 33) & 0x20) != 0 )
          {
            v53 = *(_DWORD *)(v52 + 36);
            if ( v53 > 0x45 )
            {
              if ( v53 == 71 )
              {
LABEL_78:
                v54 = sub_2D284A0(v249);
                if ( v54 )
                {
                  v57 = (unsigned int)v276;
                  v58 = (unsigned int)v276 + 1LL;
                  if ( v58 > HIDWORD(v276) )
                  {
                    sub_C8D5F0((__int64)&v275, v277, v58, 8u, v55, v56);
                    v57 = (unsigned int)v276;
                  }
                  *(_QWORD *)&v275[8 * v57] = v54;
                  LODWORD(v276) = v276 + 1;
                  continue;
                }
                v59 = (const __m128i *)v274;
                sub_AF4850((__int64)v274, v249);
                v60 = *(unsigned int *)(a4 + 24);
                v61 = *(_QWORD *)(a4 + 8);
                *(_QWORD *)v271 = *(_QWORD *)v274;
                *(_QWORD *)&v271[8] = *(_QWORD *)&v274[32];
                if ( !(_DWORD)v60 )
                  continue;
                v62 = 1;
                for ( m = (v60 - 1)
                        & (((0xBF58476D1CE4E5B9LL
                           * ((*(_DWORD *)&v274[32] >> 9) ^ (*(_DWORD *)&v274[32] >> 4)
                            | ((unsigned __int64)((*(_DWORD *)v274 >> 9) ^ (*(_DWORD *)v274 >> 4)) << 32))) >> 31)
                         ^ (484763065 * ((*(_DWORD *)&v274[32] >> 9) ^ (*(_DWORD *)&v274[32] >> 4)))); ; m = (v60 - 1) & v65 )
                {
                  v64 = (_QWORD *)(v61 + 16LL * m);
                  if ( *(_QWORD *)v274 == *v64 && *(_QWORD *)&v274[32] == v64[1] )
                    break;
                  if ( *v64 == -4096 && v64[1] == -4096 )
                    goto LABEL_20;
                  v65 = v62 + m;
                  ++v62;
                }
                if ( v64 == (_QWORD *)(v61 + 16 * v60) )
                  continue;
                sub_F3D270((__int64)v281, (__int64)&v260, (const __m128i *)v274);
                if ( !v281[32] )
                  continue;
                if ( (unsigned __int8)sub_2D28EE0((__int64)&v264, (__int64 *)v271, (__int64 **)&v268) )
                {
                  v118 = v268.m128i_i64[0];
                  v119 = *(unsigned int *)(v268.m128i_i64[0] + 24);
                  v120 = (_QWORD *)(v268.m128i_i64[0] + 16);
                  v121 = v119 + 1;
                  v122 = 40 * v119;
                  if ( *(unsigned int *)(v268.m128i_i64[0] + 28) >= v121 )
                    goto LABEL_150;
                  v213 = *(_QWORD *)(v268.m128i_i64[0] + 16);
                  v214 = (const void *)(v268.m128i_i64[0] + 32);
                  v215 = v268.m128i_i64[0] + 16;
                  if ( v213 > (unsigned __int64)v274 || (unsigned __int64)v274 >= v213 + v122 )
                  {
                    sub_C8D5F0(v215, v214, v121, 0x28u, (__int64)v281, v117);
                    v123 = *(_QWORD *)(v118 + 16);
                    v122 = 40LL * *(unsigned int *)(v118 + 24);
                  }
                  else
                  {
                    sub_C8D5F0(v215, v214, v121, 0x28u, (__int64)v281, v117);
                    v123 = *(_QWORD *)(v118 + 16);
                    v59 = (const __m128i *)&v274[v123 - v213];
                    v122 = 40LL * *(unsigned int *)(v118 + 24);
                  }
                  goto LABEL_151;
                }
                v147 = v267;
                v148 = (_QWORD *)v268.m128i_i64[0];
                ++v264;
                v149 = v266 + 1;
                *(_QWORD *)v281 = v268.m128i_i64[0];
                if ( 4 * ((int)v266 + 1) >= 3 * v267 )
                {
                  v147 = 2 * v267;
                }
                else if ( v267 - HIDWORD(v266) - v149 > v267 >> 3 )
                {
LABEL_176:
                  LODWORD(v266) = v149;
                  if ( *v148 != -4096 || v148[1] != -4096 )
                    --HIDWORD(v266);
                  *(_OWORD *)v148 = *(_OWORD *)v271;
                  v150 = v148 + 4;
                  v120 = v148 + 2;
                  *v120 = v150;
                  v120[1] = 0x800000000LL;
                  v122 = 0;
LABEL_150:
                  v123 = *v120;
LABEL_151:
                  v124 = (__m128i *)(v123 + v122);
                  *v124 = _mm_loadu_si128(v59);
                  a7 = _mm_loadu_si128(v59 + 1);
                  v124[1] = a7;
                  v124[2].m128i_i64[0] = v59[2].m128i_i64[0];
                  ++*((_DWORD *)v120 + 2);
                  continue;
                }
                sub_2D2E6F0((__int64)&v264, v147);
                sub_2D28EE0((__int64)&v264, (__int64 *)v271, (__int64 **)v281);
                v148 = *(_QWORD **)v281;
                v149 = v266 + 1;
                goto LABEL_176;
              }
            }
            else if ( v53 > 0x43 )
            {
              goto LABEL_78;
            }
          }
        }
      }
      v21 = sub_B2BEC0(a2);
      v22 = *((_BYTE *)i - 24);
      if ( v22 == 62 )
      {
        sub_AE9D00((__int64)&v268, v21, v249);
        v23 = v268.m128i_i64[1];
        src = v269.m128i_i64[0];
        if ( !v270 )
          continue;
      }
      else
      {
        if ( v22 != 85 )
          continue;
        v66 = *(i - 7);
        if ( !v66 )
          continue;
        if ( *(_BYTE *)v66 )
          continue;
        if ( *(_QWORD *)(v66 + 24) != i[7] )
          continue;
        if ( (*(_BYTE *)(v66 + 33) & 0x20) == 0 )
          continue;
        if ( (unsigned int)(*(_DWORD *)(v66 + 36) - 238) > 7 )
          continue;
        if ( ((1LL << (*(_BYTE *)(v66 + 36) + 18)) & 0xAD) == 0 )
          continue;
        sub_AE9C80((__int64)&v268, v21, v249);
        v23 = v268.m128i_i64[1];
        src = v269.m128i_i64[0];
        if ( !v270 )
          continue;
      }
      v67 = v268.m128i_i64[0];
      if ( (*(_BYTE *)(v268.m128i_i64[0] + 7) & 0x20) != 0 )
      {
        v68 = sub_B91C10(v268.m128i_i64[0], 38);
        if ( !v68 || (v69 = sub_AE94B0(v68), v232 = v70, v71 = v69, v69 == v70) )
        {
          v88 = *(_BYTE *)(v67 + 7) & 0x20;
          goto LABEL_116;
        }
        v229 = i;
        while ( 1 )
        {
LABEL_111:
          v86 = *(_QWORD *)(v71 + 24);
          a7 = 0;
          memset(v271, 0, sizeof(v271));
          v87 = sub_B43CC0(v249);
          if ( !(unsigned __int8)sub_AEA6D0(v87, v67, v23, src, v86, (__int64)v271) )
            goto LABEL_110;
          if ( !v271[16] )
            break;
          if ( *(_QWORD *)v271 )
            goto LABEL_102;
          v71 = *(_QWORD *)(v71 + 8);
          if ( v71 == v232 )
          {
LABEL_115:
            i = v229;
            v88 = *(_BYTE *)(v67 + 7) & 0x20;
LABEL_116:
            if ( !v88 )
              goto LABEL_20;
            v89 = sub_B91C10(v67, 38);
            if ( !v89 )
              goto LABEL_20;
            v90 = *(_QWORD *)(v89 + 8);
            v91 = (__m128i *)(v90 & 0xFFFFFFFFFFFFFFF8LL);
            if ( (v90 & 4) == 0 )
              v91 = 0;
            sub_B967C0((__m128i *)v281, v91);
            v92 = *(_BYTE **)v281;
            v233 = *(_QWORD *)v281 + 8LL * *(unsigned int *)&v281[8];
            if ( v233 == *(_QWORD *)v281 )
            {
LABEL_139:
              if ( v92 != &v281[16] )
                _libc_free((unsigned __int64)v92);
              goto LABEL_20;
            }
            v230 = v67;
            v93 = *(__int64 **)v281;
            v235 = v23;
            while ( 1 )
            {
LABEL_134:
              v109 = *v93;
              a7 = 0;
              v259 = 0;
              v258 = 0;
              v110 = sub_B43CC0(v249);
              if ( !(unsigned __int8)sub_AEA880(v110, v230, v235, src, v109, (__int64)&v258) )
                goto LABEL_133;
              if ( !(_BYTE)v259 )
                break;
              if ( v258.m128i_i64[0] )
                goto LABEL_123;
              if ( (__int64 *)v233 == ++v93 )
              {
LABEL_138:
                v92 = *(_BYTE **)v281;
                goto LABEL_139;
              }
            }
            v94 = sub_B11F60(v109 + 80);
            sub_AF47B0((__int64)v274, *(unsigned __int64 **)(v94 + 16), *(unsigned __int64 **)(v94 + 24));
            v95 = _mm_loadu_si128((const __m128i *)v274);
            v259 = *(_QWORD *)&v274[16];
            v258 = v95;
LABEL_123:
            v96 = *(_QWORD *)(v109 + 24);
            v257[0] = v96;
            if ( v96 )
              sub_B96E90((__int64)v257, v96, 1);
            v97 = sub_B10D40((__int64)v257);
            v98 = sub_B12000(v109 + 72);
            a7 = _mm_loadu_si128(&v258);
            v273 = v97;
            *(_QWORD *)v271 = v98;
            *(_QWORD *)&v274[16] = v259;
            v272 = v259;
            *(__m128i *)v274 = a7;
            *(__m128i *)&v271[8] = a7;
            sub_9C6650(v257);
            v257[0] = *(_QWORD *)v271;
            v257[1] = v273;
            v99 = sub_2D2BDF0(a4, v257);
            if ( !v99 || v99 == *(_QWORD *)(a4 + 8) + 16LL * *(unsigned int *)(a4 + 24) )
              goto LABEL_133;
            v255 = v249;
            if ( (unsigned __int8)sub_2D28FE0(a5, &v255, &v256) )
            {
              v100 = (__int64)(v256 + 1);
              goto LABEL_129;
            }
            v132 = *(_DWORD *)(a5 + 24);
            v133 = *(_DWORD *)(a5 + 16);
            v134 = v256;
            ++*(_QWORD *)a5;
            v135 = v133 + 1;
            *(_QWORD *)v274 = v134;
            if ( 4 * v135 >= 3 * v132 )
            {
              sub_2D2EAE0(a5, 2 * v132);
            }
            else
            {
              if ( v132 - *(_DWORD *)(a5 + 20) - v135 > v132 >> 3 )
              {
LABEL_163:
                *(_DWORD *)(a5 + 16) = v135;
                if ( *v134 != -4096 )
                  --*(_DWORD *)(a5 + 20);
                *v134 = v255;
                v136 = v134 + 3;
                v100 = (__int64)(v134 + 1);
                *(_QWORD *)v100 = v136;
                *(_QWORD *)(v100 + 8) = 0x100000000LL;
LABEL_129:
                v101 = _mm_loadu_si128((const __m128i *)&v271[16]);
                *(__m128i *)v274 = _mm_loadu_si128((const __m128i *)v271);
                *(_QWORD *)&v274[32] = v273;
                *(__m128i *)&v274[16] = v101;
                *(_DWORD *)v274 = sub_2D2C1F0(a3, (const __m128i *)v274);
                v268.m128i_i64[1] = v235;
                v103 = _mm_loadu_si128(&v268);
                v269.m128i_i64[0] = src;
                v104 = _mm_loadu_si128(&v269);
                *(__m128i *)&v274[8] = v103;
                *(__m128i *)&v274[24] = v104;
                v105 = *(unsigned int *)(v100 + 8);
                if ( v105 + 1 > (unsigned __int64)*(unsigned int *)(v100 + 12) )
                {
                  v211 = *(_QWORD *)v100;
                  v212 = (const void *)(v100 + 16);
                  if ( *(_QWORD *)v100 > (unsigned __int64)v274
                    || (v217 = *(_QWORD *)v100, (unsigned __int64)v274 >= v211 + 40 * v105) )
                  {
                    sub_C8D5F0(v100, v212, v105 + 1, 0x28u, v102, v211);
                    v106 = *(_QWORD *)v100;
                    v105 = *(unsigned int *)(v100 + 8);
                    v107 = (const __m128i *)v274;
                  }
                  else
                  {
                    sub_C8D5F0(v100, v212, v105 + 1, 0x28u, v102, v211);
                    v106 = *(_QWORD *)v100;
                    v105 = *(unsigned int *)(v100 + 8);
                    v107 = (const __m128i *)&v274[*(_QWORD *)v100 - v217];
                  }
                }
                else
                {
                  v106 = *(_QWORD *)v100;
                  v107 = (const __m128i *)v274;
                }
                v108 = (__m128i *)(v106 + 40 * v105);
                *v108 = _mm_loadu_si128(v107);
                a7 = _mm_loadu_si128(v107 + 1);
                v108[1] = a7;
                v108[2].m128i_i64[0] = v107[2].m128i_i64[0];
                ++*(_DWORD *)(v100 + 8);
                sub_F3D270((__int64)v274, (__int64)&v260, (const __m128i *)v271);
                if ( v274[32] )
                {
                  v141 = sub_2D2E9F0((__int64)&v264, v257);
                  sub_2D28790((__int64)v141, (const __m128i *)v271, v142, v143, v144, v145);
                }
LABEL_133:
                if ( (__int64 *)v233 == ++v93 )
                  goto LABEL_138;
                goto LABEL_134;
              }
              sub_2D2EAE0(a5, v132);
            }
            sub_2D28FE0(a5, &v255, v274);
            v134 = *(_QWORD **)v274;
            v135 = *(_DWORD *)(a5 + 16) + 1;
            goto LABEL_163;
          }
        }
        v72 = *(_QWORD *)(*(_QWORD *)(v86 + 32 * (2LL - (*(_DWORD *)(v86 + 4) & 0x7FFFFFF))) + 24LL);
        sub_AF47B0((__int64)v281, *(unsigned __int64 **)(v72 + 16), *(unsigned __int64 **)(v72 + 24));
        v73 = _mm_loadu_si128((const __m128i *)v281);
        *(_QWORD *)&v271[16] = *(_QWORD *)&v281[16];
        *(__m128i *)v271 = v73;
LABEL_102:
        v74.m128i_i64[1] = sub_B10D40(v86 + 48);
        a7 = _mm_loadu_si128((const __m128i *)v271);
        v74.m128i_i64[0] = *(_QWORD *)(*(_QWORD *)(v86 + 32 * (1LL - (*(_DWORD *)(v86 + 4) & 0x7FFFFFF))) + 24LL);
        *(_QWORD *)&v281[16] = *(_QWORD *)&v271[16];
        *(_QWORD *)&v274[24] = *(_QWORD *)&v271[16];
        *(_QWORD *)&v274[32] = v74.m128i_i64[1];
        *(_QWORD *)v274 = v74.m128i_i64[0];
        v258 = v74;
        *(__m128i *)v281 = a7;
        *(__m128i *)&v274[8] = a7;
        v75 = sub_2D2BDF0(a4, v258.m128i_i64);
        if ( !v75 || v75 == *(_QWORD *)(a4 + 8) + 16LL * *(unsigned int *)(a4 + 24) )
          goto LABEL_110;
        v256 = (_QWORD *)v249;
        if ( (unsigned __int8)sub_2D28FE0(a5, (__int64 *)&v256, v257) )
        {
          v76 = (_QWORD *)(v257[0] + 8);
          goto LABEL_106;
        }
        v127 = *(_DWORD *)(a5 + 24);
        v128 = *(_DWORD *)(a5 + 16);
        v129 = (_QWORD *)v257[0];
        ++*(_QWORD *)a5;
        v130 = v128 + 1;
        *(_QWORD *)v281 = v129;
        if ( 4 * v130 >= 3 * v127 )
        {
          sub_2D2EAE0(a5, 2 * v127);
        }
        else
        {
          if ( v127 - *(_DWORD *)(a5 + 20) - v130 > v127 >> 3 )
          {
LABEL_158:
            *(_DWORD *)(a5 + 16) = v130;
            if ( *v129 != -4096 )
              --*(_DWORD *)(a5 + 20);
            *v129 = v256;
            v131 = v129 + 3;
            v76 = v129 + 1;
            *v76 = v131;
            v76[1] = 0x100000000LL;
LABEL_106:
            v218 = (__int64)v76;
            *(__m128i *)v281 = _mm_loadu_si128((const __m128i *)v274);
            v77 = _mm_loadu_si128((const __m128i *)&v274[16]);
            *(_QWORD *)&v281[32] = *(_QWORD *)&v274[32];
            *(__m128i *)&v281[16] = v77;
            v78 = sub_2D2C1F0(a3, (const __m128i *)v281);
            v268.m128i_i64[1] = v23;
            v80 = _mm_loadu_si128(&v268);
            *(_DWORD *)v281 = v78;
            *(__m128i *)&v281[8] = v80;
            v81 = v218;
            v269.m128i_i64[0] = src;
            a7 = _mm_loadu_si128(&v269);
            *(__m128i *)&v281[24] = a7;
            v82 = *(unsigned int *)(v218 + 8);
            if ( v82 + 1 > (unsigned __int64)*(unsigned int *)(v218 + 12) )
            {
              v210 = (const void *)(v218 + 16);
              if ( *(_QWORD *)v218 > (unsigned __int64)v281
                || (v216 = *(_QWORD *)v218, (unsigned __int64)v281 >= *(_QWORD *)v218 + 40 * v82) )
              {
                sub_C8D5F0(v218, v210, v82 + 1, 0x28u, v218, v79);
                v81 = v218;
                v84 = (const __m128i *)v281;
                v83 = *(_QWORD *)v218;
                v82 = *(unsigned int *)(v218 + 8);
              }
              else
              {
                sub_C8D5F0(v218, v210, v82 + 1, 0x28u, v218, v79);
                v81 = v218;
                v83 = *(_QWORD *)v218;
                v82 = *(unsigned int *)(v218 + 8);
                v84 = (const __m128i *)&v281[*(_QWORD *)v218 - v216];
              }
            }
            else
            {
              v83 = *(_QWORD *)v218;
              v84 = (const __m128i *)v281;
            }
            v85 = (__m128i *)(v83 + 40 * v82);
            *v85 = _mm_loadu_si128(v84);
            v85[1] = _mm_loadu_si128(v84 + 1);
            v85[2].m128i_i64[0] = v84[2].m128i_i64[0];
            ++*(_DWORD *)(v81 + 8);
            sub_F3D270((__int64)v281, (__int64)&v260, (const __m128i *)v274);
            if ( v281[32] )
            {
              v137 = sub_2D2E9F0((__int64)&v264, v258.m128i_i64);
              sub_2D28790((__int64)v137, (const __m128i *)v274, v138, v139, (__int64)v274, v140);
            }
LABEL_110:
            v71 = *(_QWORD *)(v71 + 8);
            if ( v71 == v232 )
              goto LABEL_115;
            goto LABEL_111;
          }
          sub_2D2EAE0(a5, v127);
        }
        sub_2D28FE0(a5, (__int64 *)&v256, v281);
        v129 = *(_QWORD **)v281;
        v130 = *(_DWORD *)(a5 + 16) + 1;
        goto LABEL_158;
      }
LABEL_20:
      ;
    }
    v227 = (_QWORD *)v227[1];
  }
  while ( (_QWORD *)v220 != v227 );
  v8 = (__int64)a3;
  if ( !(_DWORD)v266 )
  {
LABEL_23:
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_DWORD *)(a1 + 24) = 0;
    goto LABEL_24;
  }
  *(_QWORD *)&v281[24] = &v265[44 * v267];
  *(_QWORD *)v281 = &v264;
  *(_QWORD *)&v281[8] = v264;
  *(_QWORD *)&v281[16] = v265;
  sub_2D290A0((__int64)v281);
  v154 = (__int64)v265;
  v155 = 5LL * v267;
  v156 = *(_QWORD *)&v281[16];
  v157 = &v265[44 * v267];
  if ( *(_QWORD **)&v281[16] == v157 )
  {
    v167 = v266;
  }
  else
  {
    do
    {
      v158 = *(unsigned int *)(v156 + 24);
      v159 = *(_QWORD *)(v156 + 16);
      v160 = 40 * v158;
      v161 = (const __m128i *)(v159 + 40 * v158);
      if ( (const __m128i *)v159 != v161 )
      {
        v162 = (__m128i *)(v159 + 40 * v158);
        srcc = *(__m128i **)(v156 + 16);
        _BitScanReverse64(&v163, 0xCCCCCCCCCCCCCCCDLL * (v160 >> 3));
        sub_2D25A70(v159, v162, 2LL * (int)(63 - (v163 ^ 0x3F)), v155, v152, v153, a7, a8);
        if ( (unsigned __int64)v160 <= 0x280 )
        {
          sub_2D22F20(srcc, v161);
        }
        else
        {
          v164 = srcc + 40;
          sub_2D22F20(srcc, srcc + 40);
          if ( v161 != &srcc[40] )
          {
            do
            {
              v165 = (__int64)v164;
              v164 = (const __m128i *)((char *)v164 + 40);
              sub_2D22E80(v165);
            }
            while ( v161 != v164 );
          }
        }
      }
      v156 = *(_QWORD *)&v281[24];
      v166 = (__int64 *)(*(_QWORD *)&v281[16] + 352LL);
      *(_QWORD *)&v281[16] = v166;
      if ( v166 != *(__int64 **)&v281[24] )
      {
        while ( 1 )
        {
          while ( 1 )
          {
            v155 = *v166;
            if ( *v166 != -4096 )
              break;
            if ( v166[1] != -4096 )
              goto LABEL_191;
            v166 += 44;
            *(_QWORD *)&v281[16] = v166;
            if ( v166 == *(__int64 **)&v281[24] )
              goto LABEL_192;
          }
          if ( v155 != -8192 || v166[1] != -8192 )
            break;
          v166 += 44;
          *(_QWORD *)&v281[16] = v166;
          if ( v166 == *(__int64 **)&v281[24] )
            goto LABEL_192;
        }
LABEL_191:
        v156 = *(_QWORD *)&v281[16];
      }
LABEL_192:
      ;
    }
    while ( v157 != (_QWORD *)v156 );
    v154 = (__int64)v265;
    v167 = v266;
    v157 = &v265[44 * v267];
  }
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  if ( v167 )
  {
    *(_QWORD *)&v274[16] = v154;
    *(_QWORD *)&v274[24] = v157;
    *(_QWORD *)&v274[8] = v264;
    *(_QWORD *)v274 = &v264;
    sub_2D290A0((__int64)v274);
    v168 = *(_QWORD *)&v274[16];
    v223 = &v265[44 * v267];
    if ( v223 != *(_QWORD **)&v274[16] )
    {
      v169 = (_QWORD *)v8;
      v170 = v8 + 8;
      while ( 1 )
      {
        v228 = *(const __m128i **)(v168 + 16);
        v250 = (const __m128i *)((char *)v228 + 40 * *(unsigned int *)(v168 + 24));
        if ( v250 != v228 )
        {
          v254 = v170;
          while ( 1 )
          {
            if ( v228[1].m128i_i8[8] )
            {
              v171 = v228[1].m128i_i64[0];
              v234 = v228->m128i_i64[1];
            }
            else
            {
              v171 = qword_4F81350[1];
              v234 = qword_4F81350[0];
            }
            v237 = v171;
            v228 = (const __m128i *)((char *)v228 + 40);
            v172 = v228;
            *(__m128i *)v281 = _mm_loadu_si128((const __m128i *)((char *)v228 - 40));
            *(__m128i *)&v281[16] = _mm_loadu_si128((const __m128i *)((char *)v228 - 24));
            *(_QWORD *)&v281[32] = v228[-1].m128i_i64[1];
            v231 = sub_2D2C1F0(v169, (const __m128i *)v281);
            if ( v228 == v250 )
            {
              v170 = v254;
              break;
            }
            v173 = v228[1].m128i_i8[8];
            if ( !v173 )
              goto LABEL_229;
            while ( 2 )
            {
              v174 = v172->m128i_i64[1];
              v175 = v172[1].m128i_u64[0];
LABEL_204:
              *(__m128i *)v281 = _mm_loadu_si128(v172);
              *(__m128i *)&v281[16] = _mm_loadu_si128(v172 + 1);
              *(_QWORD *)&v281[32] = v172[2].m128i_i64[0];
              v176 = v169[2];
              if ( !v176 )
              {
                v177 = v254;
                goto LABEL_242;
              }
              v177 = v254;
              do
              {
                while ( 1 )
                {
                  if ( *(_QWORD *)v281 > *(_QWORD *)(v176 + 32) )
                  {
LABEL_209:
                    v176 = *(_QWORD *)(v176 + 24);
                    goto LABEL_210;
                  }
                  if ( *(_QWORD *)v281 == *(_QWORD *)(v176 + 32) )
                    break;
LABEL_207:
                  v177 = v176;
                  v176 = *(_QWORD *)(v176 + 16);
                  if ( !v176 )
                    goto LABEL_211;
                }
                v190 = *(_BYTE *)(v176 + 56);
                if ( v173 )
                {
                  if ( !v190 )
                    goto LABEL_209;
                  v191 = *(_QWORD *)(v176 + 40);
                  if ( v191 < *(_QWORD *)&v281[8]
                    || v191 == *(_QWORD *)&v281[8] && *(_QWORD *)(v176 + 48) < *(_QWORD *)&v281[16] )
                  {
                    goto LABEL_209;
                  }
                  if ( v191 > *(_QWORD *)&v281[8] || *(_QWORD *)&v281[16] < *(_QWORD *)(v176 + 48) )
                    goto LABEL_207;
                }
                else if ( v190 )
                {
                  goto LABEL_207;
                }
                if ( *(_QWORD *)(v176 + 64) >= *(_QWORD *)&v281[32] )
                  goto LABEL_207;
                v176 = *(_QWORD *)(v176 + 24);
LABEL_210:
                ;
              }
              while ( v176 );
LABEL_211:
              if ( v254 == v177 || *(_QWORD *)v281 < *(_QWORD *)(v177 + 32) )
                goto LABEL_242;
              if ( *(_QWORD *)v281 != *(_QWORD *)(v177 + 32) )
                goto LABEL_214;
              if ( !*(_BYTE *)(v177 + 56) )
              {
                if ( v173 || *(_QWORD *)&v281[32] >= *(_QWORD *)(v177 + 64) )
                  goto LABEL_214;
LABEL_242:
                srci = v174;
                *(_QWORD *)v271 = v281;
                v192 = sub_2D2C110(v169, v177, (const __m128i **)v271);
                v174 = srci;
                v177 = v192;
                goto LABEL_214;
              }
              if ( !v173 )
                goto LABEL_242;
              v193 = *(_QWORD *)(v177 + 40);
              if ( *(_QWORD *)&v281[8] < v193
                || *(_QWORD *)&v281[8] == v193 && *(_QWORD *)&v281[16] < *(_QWORD *)(v177 + 48) )
              {
                goto LABEL_242;
              }
              if ( *(_QWORD *)&v281[8] <= v193
                && *(_QWORD *)(v177 + 48) >= *(_QWORD *)&v281[16]
                && *(_QWORD *)&v281[32] < *(_QWORD *)(v177 + 64) )
              {
                goto LABEL_242;
              }
LABEL_214:
              v178 = *(_DWORD *)(v177 + 72);
              if ( !v178 )
              {
                *(_DWORD *)(v177 + 72) = -858993459 * ((__int64)(v169[7] - v169[6]) >> 3) + 1;
                v179 = (__m128i *)v169[7];
                if ( v179 == (__m128i *)v169[8] )
                {
                  v226 = v177;
                  srcj = v174;
                  sub_2D294F0(v169 + 6, v179, (const __m128i *)v281);
                  v177 = v226;
                  v174 = srcj;
                }
                else
                {
                  if ( v179 )
                  {
                    *v179 = _mm_loadu_si128((const __m128i *)v281);
                    v179[1] = _mm_loadu_si128((const __m128i *)&v281[16]);
                    v179[2].m128i_i64[0] = *(_QWORD *)&v281[32];
                    v179 = (__m128i *)v169[7];
                  }
                  v169[7] = (char *)v179 + 40;
                }
                v178 = *(_DWORD *)(v177 + 72);
              }
              if ( v237 < v175 || v175 + v174 < v237 + v234 )
                goto LABEL_227;
              v180 = *(_DWORD *)(a1 + 24);
              if ( !v180 )
              {
                ++*(_QWORD *)a1;
LABEL_271:
                sub_2D2EDE0(a1, 2 * v180);
                v196 = *(_DWORD *)(a1 + 24);
                if ( v196 )
                {
                  v197 = v196 - 1;
                  v198 = *(_QWORD *)(a1 + 8);
                  LODWORD(v199) = v197 & (37 * v178);
                  v195 = *(_DWORD *)(a1 + 16) + 1;
                  v186 = (_DWORD *)(v198 + 72LL * (unsigned int)v199);
                  v200 = *v186;
                  if ( v178 == *v186 )
                    goto LABEL_267;
                  v201 = 1;
                  v202 = 0;
                  while ( v200 != -1 )
                  {
                    if ( !v202 && v200 == -2 )
                      v202 = v186;
                    v199 = v197 & (unsigned int)(v199 + v201);
                    v186 = (_DWORD *)(v198 + 72 * v199);
                    v200 = *v186;
                    if ( v178 == *v186 )
                      goto LABEL_267;
                    ++v201;
                  }
LABEL_275:
                  if ( v202 )
                    v186 = v202;
                  goto LABEL_267;
                }
LABEL_342:
                ++*(_DWORD *)(a1 + 16);
                BUG();
              }
              v181 = *(_QWORD *)(a1 + 8);
              v182 = 1;
              v183 = (unsigned int)(37 * v178);
              LODWORD(v184) = (v180 - 1) & (37 * v178);
              v185 = (_DWORD *)(v181 + 72LL * (unsigned int)v184);
              v186 = 0;
              v187 = *v185;
              if ( *v185 == v178 )
              {
LABEL_224:
                v188 = (unsigned int)v185[4];
                v189 = v185 + 2;
                if ( (unsigned int)v185[5] < (unsigned __int64)(v188 + 1) )
                {
                  srch = v185;
                  sub_C8D5F0((__int64)(v185 + 2), v185 + 6, v188 + 1, 4u, (__int64)v185, v183);
                  v188 = (unsigned int)srch[4];
                }
                goto LABEL_226;
              }
              while ( v187 != -1 )
              {
                if ( v187 == -2 && !v186 )
                  v186 = v185;
                v184 = (v180 - 1) & ((_DWORD)v184 + v182);
                v185 = (_DWORD *)(v181 + 72 * v184);
                v187 = *v185;
                if ( v178 == *v185 )
                  goto LABEL_224;
                ++v182;
              }
              v194 = *(_DWORD *)(a1 + 16);
              if ( !v186 )
                v186 = v185;
              ++*(_QWORD *)a1;
              v195 = v194 + 1;
              if ( 4 * (v194 + 1) >= 3 * v180 )
                goto LABEL_271;
              if ( v180 - *(_DWORD *)(a1 + 20) - v195 <= v180 >> 3 )
              {
                srcd = 37 * v178;
                sub_2D2EDE0(a1, v180);
                v203 = *(_DWORD *)(a1 + 24);
                if ( v203 )
                {
                  v204 = v203 - 1;
                  v205 = *(_QWORD *)(a1 + 8);
                  v202 = 0;
                  v206 = 1;
                  v195 = *(_DWORD *)(a1 + 16) + 1;
                  LODWORD(v207) = v204 & srcd;
                  v186 = (_DWORD *)(v205 + 72LL * (v204 & (unsigned int)srcd));
                  v208 = *v186;
                  if ( v178 == *v186 )
                    goto LABEL_267;
                  while ( v208 != -1 )
                  {
                    if ( !v202 && v208 == -2 )
                      v202 = v186;
                    v207 = v204 & (unsigned int)(v207 + v206);
                    v186 = (_DWORD *)(v205 + 72 * v207);
                    v208 = *v186;
                    if ( v178 == *v186 )
                      goto LABEL_267;
                    ++v206;
                  }
                  goto LABEL_275;
                }
                goto LABEL_342;
              }
LABEL_267:
              *(_DWORD *)(a1 + 16) = v195;
              if ( *v186 != -1 )
                --*(_DWORD *)(a1 + 20);
              *v186 = v178;
              v189 = v186 + 2;
              *((_QWORD *)v186 + 1) = v186 + 6;
              *((_QWORD *)v186 + 2) = 0xC00000000LL;
              v188 = 0;
LABEL_226:
              *(_DWORD *)(*(_QWORD *)v189 + 4 * v188) = v231;
              ++v189[2];
LABEL_227:
              v172 = (const __m128i *)((char *)v172 + 40);
              if ( v172 != v250 )
              {
                v173 = v172[1].m128i_i8[8];
                if ( v173 )
                  continue;
LABEL_229:
                v174 = qword_4F81350[0];
                v175 = qword_4F81350[1];
                goto LABEL_204;
              }
              break;
            }
          }
        }
        v168 = *(_QWORD *)&v274[24];
        v209 = (_QWORD *)(*(_QWORD *)&v274[16] + 352LL);
        *(_QWORD *)&v274[16] = v209;
        if ( v209 == *(_QWORD **)&v274[24] )
          goto LABEL_291;
        while ( *v209 == -4096 )
        {
          if ( v209[1] != -4096 )
            goto LABEL_290;
LABEL_294:
          v209 += 44;
          *(_QWORD *)&v274[16] = v209;
          if ( v209 == *(_QWORD **)&v274[24] )
            goto LABEL_291;
        }
        if ( *v209 == -8192 && v209[1] == -8192 )
          goto LABEL_294;
LABEL_290:
        v168 = *(_QWORD *)&v274[16];
LABEL_291:
        if ( v223 == (_QWORD *)v168 )
        {
          v8 = (__int64)v169;
          break;
        }
      }
    }
  }
LABEL_24:
  v24 = (unsigned __int64)v275;
  *a6 = -858993459 * ((__int64)(*(_QWORD *)(v8 + 56) - *(_QWORD *)(v8 + 48)) >> 3) + 1;
  for ( n = v24 + 8LL * (unsigned int)v276; n != v24; v24 += 8LL )
  {
    v26 = *(_QWORD *)v24;
    v27 = *(_QWORD *)(*(_QWORD *)v24 + 48LL);
    v28 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v24 - 32LL * (*(_DWORD *)(*(_QWORD *)v24 + 4LL) & 0x7FFFFFF)) + 24LL);
    *(_QWORD *)v274 = v27;
    v252 = v28;
    if ( v27 )
    {
      srce = v26;
      sub_B96E90((__int64)v274, v27, 1);
      v26 = srce;
    }
    srcf = *(_QWORD *)(*(_QWORD *)(v26 + 32 * (2LL - (*(_DWORD *)(v26 + 4) & 0x7FFFFFF))) + 24LL);
    sub_AF4850((__int64)v281, v26);
    sub_2D2C3A0(
      v8,
      srcf,
      (__int64 *)v274,
      v252,
      v29,
      srcf,
      *(__int128 *)v281,
      *(__int128 *)&v281[16],
      *(__int64 *)&v281[32]);
    if ( *(_QWORD *)v274 )
      sub_B91220((__int64)v274, *(__int64 *)v274);
  }
  v30 = v278;
  v31 = &v278[(unsigned int)v279];
  if ( v31 != v278 )
  {
    v32 = v278;
    do
    {
      v33 = *v32;
      v34 = *(_QWORD *)(*v32 + 24);
      v253 = *(_QWORD *)(*v32 + 40);
      *(_QWORD *)v274 = v34;
      if ( v34 )
        sub_B96E90((__int64)v274, v34, 1);
      srcg = sub_B11F60(v33 + 80);
      sub_AF48C0(v281, v33);
      sub_2D2C3A0(
        v8,
        srcg,
        (__int64 *)v274,
        v253,
        v35,
        srcg,
        *(__int128 *)v281,
        *(__int128 *)&v281[16],
        *(__int64 *)&v281[32]);
      if ( *(_QWORD *)v274 )
        sub_B91220((__int64)v274, *(__int64 *)v274);
      ++v32;
    }
    while ( v31 != v32 );
    v30 = v278;
  }
  if ( v30 != (__int64 *)v280 )
    _libc_free((unsigned __int64)v30);
  if ( v275 != v277 )
    _libc_free((unsigned __int64)v275);
  v36 = v267;
  if ( v267 )
  {
    v37 = v265;
    v38 = &v265[44 * v267];
    while ( 1 )
    {
      while ( *v37 == -4096 )
      {
        if ( v37[1] != -4096 )
          goto LABEL_45;
        v37 += 44;
        if ( v38 == v37 )
        {
LABEL_51:
          v36 = v267;
          goto LABEL_52;
        }
      }
      if ( *v37 != -8192 || v37[1] != -8192 )
      {
LABEL_45:
        v39 = v37[2];
        if ( (_QWORD *)v39 != v37 + 4 )
          _libc_free(v39);
      }
      v37 += 44;
      if ( v38 == v37 )
        goto LABEL_51;
    }
  }
LABEL_52:
  sub_C7D6A0((__int64)v265, 352 * v36, 8);
  sub_C7D6A0(v261, 40LL * (unsigned int)v263, 8);
  return a1;
}
