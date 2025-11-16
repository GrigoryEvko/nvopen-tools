// Function: sub_251CD10
// Address: 0x251cd10
//
__int64 *__fastcall sub_251CD10(__int64 a1)
{
  __int64 v1; // r8
  __int64 v2; // r9
  __int64 *v3; // r13
  __int64 v4; // rax
  __int64 *v5; // rbx
  __int64 v6; // rax
  int v7; // edx
  __int64 v8; // rax
  _BYTE *v9; // r13
  _QWORD *v10; // r12
  _QWORD *v11; // r14
  int v12; // r10d
  int v13; // ecx
  __int64 *v14; // rdx
  __int64 v15; // rax
  unsigned __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  int v21; // eax
  __int64 v22; // rdx
  _QWORD *v23; // rax
  _QWORD *j; // rdx
  __int64 v25; // r15
  _QWORD *v26; // r10
  _QWORD *v27; // r11
  _QWORD *v28; // r12
  __int64 *v29; // rdi
  unsigned __int64 v30; // rax
  int v31; // esi
  __int64 *v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // r15
  __int64 v35; // r14
  unsigned __int64 v36; // rbx
  unsigned __int64 v37; // rdx
  unsigned __int64 v38; // rax
  _QWORD *v39; // rax
  __int64 v40; // r15
  __int64 *v41; // rdi
  _QWORD *k; // rdx
  __int64 *v43; // rax
  __int64 v44; // rdx
  int v45; // ecx
  _QWORD *v46; // r8
  __int64 v47; // r9
  int v48; // esi
  int v49; // ecx
  int v50; // esi
  unsigned int v51; // edx
  __int64 v52; // rax
  unsigned __int64 v53; // rdx
  int v54; // eax
  __int64 v55; // rdx
  _QWORD *v56; // rax
  _QWORD *n; // rdx
  __int64 v58; // rdx
  _QWORD *v59; // rax
  _QWORD *v60; // rdx
  void **v61; // rbx
  void **v62; // r12
  __int64 v63; // r13
  __int64 v64; // r8
  __int64 v65; // rax
  void *v66; // r9
  unsigned __int64 v67; // rdx
  __int64 v68; // r13
  __int64 v69; // r15
  __int64 v70; // rax
  __int64 v71; // r12
  unsigned __int64 v72; // rdx
  __int64 v73; // rbx
  void **v74; // r14
  _QWORD *v75; // rdi
  __int64 v76; // rcx
  int v77; // esi
  _QWORD *v78; // rax
  _QWORD *v79; // rdx
  void **v80; // rbx
  void **v81; // r12
  int v82; // r11d
  _QWORD *v83; // r10
  int v84; // eax
  _QWORD *v85; // rdi
  void *v86; // rcx
  int v87; // eax
  __int64 v88; // rcx
  int v89; // edx
  void *v90; // r13
  __int64 v91; // rax
  unsigned __int64 v92; // rdx
  void **v93; // rbx
  void **v94; // r12
  int v95; // r11d
  _QWORD *v96; // r10
  int v97; // eax
  _QWORD *v98; // rdi
  void *v99; // rcx
  int v100; // eax
  __int64 v101; // rcx
  int v102; // edx
  void *v103; // r13
  __int64 v104; // rax
  unsigned __int64 v105; // rdx
  int v106; // eax
  __int64 v107; // rdx
  _QWORD *v108; // rax
  _QWORD *mm; // rdx
  int v110; // edi
  int v111; // edx
  __int64 v112; // rax
  unsigned __int64 v113; // rdx
  unsigned __int64 v114; // rax
  unsigned __int64 v115; // rax
  __int64 v116; // rax
  _QWORD *v117; // rax
  __int64 v118; // r15
  _QWORD *m; // rdx
  __int64 *v120; // rax
  int v121; // ecx
  _QWORD *v122; // rdi
  __int64 v123; // r8
  int v124; // esi
  int v125; // esi
  unsigned int v126; // r13d
  __int64 v127; // rdx
  int v128; // r10d
  int v129; // ecx
  __int64 *v130; // rdx
  __int64 v131; // rax
  int v132; // edx
  __int64 v133; // rax
  unsigned __int64 v134; // rdx
  unsigned int v135; // ecx
  unsigned int v136; // eax
  _QWORD *v137; // rdi
  int v138; // ebx
  _QWORD *v139; // rax
  unsigned int v140; // ecx
  unsigned int v141; // eax
  _QWORD *v142; // rdi
  int v143; // ebx
  _QWORD *v144; // rax
  int v145; // r11d
  int v146; // eax
  __int64 v147; // rcx
  int v148; // r11d
  int v149; // eax
  __int64 v150; // rcx
  int v151; // eax
  __int64 v152; // rsi
  int v153; // r10d
  int v154; // r15d
  __int64 v155; // rdi
  __int64 v156; // rcx
  unsigned int v157; // edx
  unsigned int v158; // eax
  int v159; // r12d
  unsigned int v160; // eax
  _QWORD *v161; // rax
  _QWORD *kk; // rdx
  unsigned int v163; // ecx
  _QWORD *v164; // rdi
  __int64 v165; // r8
  unsigned int v166; // eax
  int v167; // r12d
  unsigned int v168; // eax
  _QWORD *v169; // rax
  _QWORD *jj; // rdx
  unsigned int v171; // ecx
  unsigned int v172; // eax
  int v173; // r12d
  _QWORD *v174; // rdi
  unsigned int v175; // eax
  _QWORD *v176; // rax
  __int64 v177; // rdx
  _QWORD *nn; // rdx
  _QWORD *v179; // rcx
  _QWORD *v180; // rdx
  _QWORD *v181; // rcx
  _QWORD *v182; // rdx
  _QWORD *v183; // r13
  unsigned int v184; // eax
  _QWORD *v185; // rax
  __int64 v186; // rdx
  _QWORD *ii; // rdx
  unsigned int v188; // eax
  _QWORD *v189; // rax
  __int64 v190; // rdx
  _QWORD *i; // rdx
  int v192; // eax
  __int64 v193; // rcx
  int v194; // r10d
  __int64 v195; // rdi
  int v196; // ecx
  int v197; // r15d
  __int64 v198; // rsi
  __int64 v199; // rax
  int v200; // r10d
  __int64 v201; // rax
  __int64 v202; // rsi
  int v203; // ebx
  __int64 v204; // rcx
  __int64 v205; // rax
  unsigned __int64 *v206; // rdx
  __int64 v207; // r12
  unsigned __int64 *v208; // rax
  __int64 *result; // rax
  char v210; // dl
  __int64 v211; // r14
  __int64 v212; // r8
  __int64 v213; // r9
  _QWORD *v214; // r14
  _QWORD *v215; // r13
  __int64 i1; // rax
  unsigned __int64 v217; // r15
  _QWORD *v218; // rax
  __int64 v219; // rbx
  __int64 (__fastcall *v220)(_QWORD, __int64); // rax
  __int64 *v221; // rax
  __int64 v222; // r13
  __int64 *v223; // r12
  __int64 v224; // rax
  __int64 v225; // rdx
  __int64 v226; // r8
  __int64 v227; // r9
  __m128i v228; // xmm1
  __m128i v229; // xmm2
  __m128i v230; // xmm3
  unsigned __int64 *v231; // r14
  __int64 v232; // r8
  unsigned __int64 *v233; // rbx
  unsigned __int64 v234; // rdi
  unsigned __int64 *v235; // rbx
  unsigned __int64 *v236; // r12
  unsigned __int64 v237; // rdi
  int v238; // r11d
  int v239; // r11d
  __int64 v240; // rax
  __int64 v241; // rax
  __int64 v242; // [rsp+0h] [rbp-5C0h]
  int v243; // [rsp+10h] [rbp-5B0h]
  __int64 v244; // [rsp+20h] [rbp-5A0h]
  unsigned int v245; // [rsp+28h] [rbp-598h]
  unsigned int v246; // [rsp+2Ch] [rbp-594h]
  _QWORD *v247; // [rsp+38h] [rbp-588h]
  _QWORD *v248; // [rsp+38h] [rbp-588h]
  _QWORD *v249; // [rsp+38h] [rbp-588h]
  _QWORD *v250; // [rsp+38h] [rbp-588h]
  _QWORD *v251; // [rsp+38h] [rbp-588h]
  int v252; // [rsp+38h] [rbp-588h]
  _QWORD *v253; // [rsp+38h] [rbp-588h]
  void **v255; // [rsp+50h] [rbp-570h]
  __int64 v256; // [rsp+60h] [rbp-560h]
  void *v257; // [rsp+60h] [rbp-560h]
  unsigned int v258; // [rsp+68h] [rbp-558h]
  unsigned __int64 v259; // [rsp+68h] [rbp-558h]
  __int64 v260; // [rsp+70h] [rbp-550h] BYREF
  _QWORD *v261; // [rsp+78h] [rbp-548h]
  __int64 v262; // [rsp+80h] [rbp-540h]
  __int64 v263; // [rsp+88h] [rbp-538h]
  void **v264; // [rsp+90h] [rbp-530h] BYREF
  __int64 v265; // [rsp+98h] [rbp-528h]
  __int64 v266; // [rsp+A0h] [rbp-520h] BYREF
  _QWORD *v267; // [rsp+A8h] [rbp-518h]
  __int64 v268; // [rsp+B0h] [rbp-510h]
  __int64 v269; // [rsp+B8h] [rbp-508h]
  _BYTE **v270; // [rsp+C0h] [rbp-500h] BYREF
  __int64 v271; // [rsp+C8h] [rbp-4F8h]
  _BYTE *v272[2]; // [rsp+D0h] [rbp-4F0h] BYREF
  __int64 v273; // [rsp+E0h] [rbp-4E0h] BYREF
  __int64 *v274; // [rsp+F0h] [rbp-4D0h]
  __int64 v275; // [rsp+F8h] [rbp-4C8h]
  __int64 v276; // [rsp+100h] [rbp-4C0h] BYREF
  __m128i v277; // [rsp+110h] [rbp-4B0h] BYREF
  void **v278; // [rsp+120h] [rbp-4A0h] BYREF
  __int64 v279; // [rsp+128h] [rbp-498h]
  _BYTE v280[256]; // [rsp+130h] [rbp-490h] BYREF
  __int64 *v281; // [rsp+230h] [rbp-390h] BYREF
  int v282; // [rsp+238h] [rbp-388h]
  char v283; // [rsp+23Ch] [rbp-384h]
  __int64 v284; // [rsp+240h] [rbp-380h] BYREF
  __m128i v285; // [rsp+248h] [rbp-378h] BYREF
  __int64 v286; // [rsp+258h] [rbp-368h]
  __m128i v287; // [rsp+260h] [rbp-360h] BYREF
  __m128i v288; // [rsp+270h] [rbp-350h]
  unsigned __int64 *v289; // [rsp+280h] [rbp-340h] BYREF
  __int64 v290; // [rsp+288h] [rbp-338h]
  _BYTE v291[324]; // [rsp+290h] [rbp-330h] BYREF
  int v292; // [rsp+3D4h] [rbp-1ECh]
  __int64 v293; // [rsp+3D8h] [rbp-1E8h]
  char *v294; // [rsp+3E0h] [rbp-1E0h] BYREF
  unsigned __int64 *v295; // [rsp+3E8h] [rbp-1D8h]
  __int64 v296; // [rsp+3F0h] [rbp-1D0h]
  __m128i v297; // [rsp+3F8h] [rbp-1C8h] BYREF
  __int64 v298; // [rsp+408h] [rbp-1B8h]
  __m128i v299; // [rsp+410h] [rbp-1B0h] BYREF
  __m128i v300; // [rsp+420h] [rbp-1A0h] BYREF
  unsigned __int64 *v301; // [rsp+430h] [rbp-190h] BYREF
  unsigned int v302; // [rsp+438h] [rbp-188h]
  _BYTE v303[324]; // [rsp+440h] [rbp-180h] BYREF
  int v304; // [rsp+584h] [rbp-3Ch]
  __int64 v305; // [rsp+588h] [rbp-38h]

  v242 = sub_C996C0("Attributor::runTillFixpoint", 27, 0, 0);
  v245 = qword_4FEF1C8;
  if ( *(_BYTE *)(a1 + 4388) )
    v245 = *(_DWORD *)(a1 + 4384);
  v260 = 0;
  v278 = (void **)v280;
  v279 = 0x2000000000LL;
  v264 = (void **)&v266;
  v270 = v272;
  v261 = 0;
  v3 = *(__int64 **)(a1 + 256);
  v4 = *(unsigned int *)(a1 + 264);
  v262 = 0;
  v263 = 0;
  v5 = &v3[v4];
  v265 = 0;
  v266 = 0;
  v267 = 0;
  v268 = 0;
  v269 = 0;
  v271 = 0;
  v244 = v4;
  if ( v5 == v3 )
  {
    v7 = 0;
  }
  else
  {
    do
    {
      v6 = *v3++;
      v294 = (char *)(v6 & 0xFFFFFFFFFFFFFFF8LL);
      sub_2519E30((__int64)&v260, (__int64 *)&v294);
    }
    while ( v5 != v3 );
    v7 = v271;
    v244 = *(unsigned int *)(a1 + 264);
  }
  v246 = 1;
  while ( 2 )
  {
    v8 = 0;
    v258 = 0;
    if ( v7 )
    {
      while ( 1 )
      {
        v9 = v270[v8];
        v10 = (_QWORD *)*((_QWORD *)v9 + 5);
        v11 = &v10[*((unsigned int *)v9 + 12)];
        if ( v10 != v11 )
          break;
LABEL_18:
        v21 = *((_DWORD *)v9 + 6);
        ++*((_QWORD *)v9 + 1);
        if ( v21 )
        {
          v140 = 4 * v21;
          v22 = *((unsigned int *)v9 + 8);
          if ( (unsigned int)(4 * v21) < 0x40 )
            v140 = 64;
          if ( (unsigned int)v22 <= v140 )
            goto LABEL_21;
          v141 = v21 - 1;
          if ( v141 )
          {
            _BitScanReverse(&v141, v141);
            v142 = (_QWORD *)*((_QWORD *)v9 + 2);
            v143 = 1 << (33 - (v141 ^ 0x1F));
            if ( v143 < 64 )
              v143 = 64;
            if ( (_DWORD)v22 == v143 )
            {
              *((_QWORD *)v9 + 3) = 0;
              v144 = &v142[v22];
              do
              {
                if ( v142 )
                  *v142 = -4;
                ++v142;
              }
              while ( v144 != v142 );
              goto LABEL_24;
            }
          }
          else
          {
            v142 = (_QWORD *)*((_QWORD *)v9 + 2);
            v143 = 64;
          }
          sub_C7D6A0((__int64)v142, 8 * v22, 8);
          v188 = sub_AF1560(4 * v143 / 3u + 1);
          *((_DWORD *)v9 + 8) = v188;
          if ( v188 )
          {
            v189 = (_QWORD *)sub_C7D670(8LL * v188, 8);
            v190 = *((unsigned int *)v9 + 8);
            *((_QWORD *)v9 + 3) = 0;
            *((_QWORD *)v9 + 2) = v189;
            for ( i = &v189[v190]; i != v189; ++v189 )
            {
              if ( v189 )
                *v189 = -4;
            }
            goto LABEL_24;
          }
          goto LABEL_323;
        }
        if ( *((_DWORD *)v9 + 7) )
        {
          v22 = *((unsigned int *)v9 + 8);
          if ( (unsigned int)v22 <= 0x40 )
          {
LABEL_21:
            v23 = (_QWORD *)*((_QWORD *)v9 + 2);
            for ( j = &v23[v22]; j != v23; ++v23 )
              *v23 = -4;
            goto LABEL_23;
          }
          sub_C7D6A0(*((_QWORD *)v9 + 2), 8 * v22, 8);
          *((_DWORD *)v9 + 8) = 0;
LABEL_323:
          *((_QWORD *)v9 + 2) = 0;
LABEL_23:
          *((_QWORD *)v9 + 3) = 0;
        }
LABEL_24:
        v8 = ++v258;
        *((_DWORD *)v9 + 12) = 0;
        if ( (unsigned int)v271 <= v258 )
          goto LABEL_25;
      }
      while ( 1 )
      {
        v16 = *v10 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*v10 & 4) != 0 )
          break;
        v17 = (*(__int64 (__fastcall **)(unsigned __int64))(*(_QWORD *)v16 + 40LL))(*v10 & 0xFFFFFFFFFFFFFFF8LL);
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v17 + 40LL))(v17);
        v18 = (*(__int64 (__fastcall **)(unsigned __int64))(*(_QWORD *)v16 + 40LL))(v16);
        if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v18 + 16LL))(v18) )
        {
          v19 = (unsigned int)v279;
          v20 = (unsigned int)v279 + 1LL;
          if ( v20 > HIDWORD(v279) )
          {
            sub_C8D5F0((__int64)&v278, v280, v20, 8u, v1, v2);
            v19 = (unsigned int)v279;
          }
          ++v10;
          v278[v19] = (void *)v16;
          LODWORD(v279) = v279 + 1;
          if ( v11 == v10 )
            goto LABEL_18;
        }
        else
        {
          if ( !(_DWORD)v269 )
          {
            ++v266;
            goto LABEL_237;
          }
          v1 = (unsigned int)(v269 - 1);
          v12 = 1;
          v2 = 0;
          v13 = v1 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v14 = &v267[v13];
          v15 = *v14;
          if ( v16 != *v14 )
          {
            while ( v15 != -4096 )
            {
              if ( v2 || v15 != -8192 )
                v14 = (__int64 *)v2;
              v2 = (unsigned int)(v12 + 1);
              v13 = v1 & (v12 + v13);
              v15 = v267[v13];
              if ( v16 == v15 )
                goto LABEL_12;
              ++v12;
              v2 = (__int64)v14;
              v14 = &v267[v13];
            }
            if ( !v2 )
              v2 = (__int64)v14;
            ++v266;
            v111 = v268 + 1;
            if ( 4 * ((int)v268 + 1) < (unsigned int)(3 * v269) )
            {
              if ( (int)v269 - HIDWORD(v268) - v111 <= (unsigned int)v269 >> 3 )
              {
                sub_2519C60((__int64)&v266, v269);
                if ( !(_DWORD)v269 )
                {
LABEL_515:
                  LODWORD(v268) = v268 + 1;
                  BUG();
                }
                v1 = 1;
                v154 = (v269 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
                v2 = (__int64)&v267[v154];
                v111 = v268 + 1;
                v155 = 0;
                v156 = *(_QWORD *)v2;
                if ( v16 != *(_QWORD *)v2 )
                {
                  while ( v156 != -4096 )
                  {
                    if ( !v155 && v156 == -8192 )
                      v155 = v2;
                    v154 = (v269 - 1) & (v1 + v154);
                    v2 = (__int64)&v267[v154];
                    v156 = *(_QWORD *)v2;
                    if ( v16 == *(_QWORD *)v2 )
                      goto LABEL_135;
                    v1 = (unsigned int)(v1 + 1);
                  }
                  if ( v155 )
                    v2 = v155;
                }
              }
              goto LABEL_135;
            }
LABEL_237:
            sub_2519C60((__int64)&v266, 2 * v269);
            if ( !(_DWORD)v269 )
              goto LABEL_515;
            v151 = (v269 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
            v2 = (__int64)&v267[v151];
            v152 = *(_QWORD *)v2;
            v111 = v268 + 1;
            if ( v16 != *(_QWORD *)v2 )
            {
              v153 = 1;
              v1 = 0;
              while ( v152 != -4096 )
              {
                if ( v152 == -8192 && !v1 )
                  v1 = v2;
                v151 = (v269 - 1) & (v153 + v151);
                v2 = (__int64)&v267[v151];
                v152 = *(_QWORD *)v2;
                if ( v16 == *(_QWORD *)v2 )
                  goto LABEL_135;
                ++v153;
              }
              if ( v1 )
                v2 = v1;
            }
LABEL_135:
            LODWORD(v268) = v111;
            if ( *(_QWORD *)v2 != -4096 )
              --HIDWORD(v268);
            *(_QWORD *)v2 = v16;
            v112 = (unsigned int)v271;
            v113 = (unsigned int)v271 + 1LL;
            if ( v113 > HIDWORD(v271) )
            {
              sub_C8D5F0((__int64)&v270, v272, v113, 8u, v1, v2);
              v112 = (unsigned int)v271;
            }
            v270[v112] = (_BYTE *)v16;
            LODWORD(v271) = v271 + 1;
          }
LABEL_12:
          if ( v11 == ++v10 )
            goto LABEL_18;
        }
      }
      if ( !(_DWORD)v263 )
      {
        ++v260;
        goto LABEL_330;
      }
      v1 = (unsigned int)(v263 - 1);
      v128 = 1;
      v2 = 0;
      v129 = v1 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v130 = &v261[v129];
      v131 = *v130;
      if ( v16 != *v130 )
      {
        while ( v131 != -4096 )
        {
          if ( v2 || v131 != -8192 )
            v130 = (__int64 *)v2;
          v2 = (unsigned int)(v128 + 1);
          v129 = v1 & (v128 + v129);
          v131 = v261[v129];
          if ( v16 == v131 )
            goto LABEL_12;
          ++v128;
          v2 = (__int64)v130;
          v130 = &v261[v129];
        }
        if ( !v2 )
          v2 = (__int64)v130;
        ++v260;
        v132 = v262 + 1;
        if ( 4 * ((int)v262 + 1) < (unsigned int)(3 * v263) )
        {
          if ( (int)v263 - HIDWORD(v262) - v132 <= (unsigned int)v263 >> 3 )
          {
            sub_2519C60((__int64)&v260, v263);
            if ( !(_DWORD)v263 )
            {
LABEL_513:
              LODWORD(v262) = v262 + 1;
              BUG();
            }
            v1 = (unsigned int)(v263 - 1);
            v196 = 1;
            v197 = v1 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
            v2 = (__int64)&v261[v197];
            v198 = *(_QWORD *)v2;
            v132 = v262 + 1;
            v199 = 0;
            if ( v16 != *(_QWORD *)v2 )
            {
              while ( v198 != -4096 )
              {
                if ( v198 == -8192 && !v199 )
                  v199 = v2;
                v197 = v1 & (v196 + v197);
                v2 = (__int64)&v261[v197];
                v198 = *(_QWORD *)v2;
                if ( v16 == *(_QWORD *)v2 )
                  goto LABEL_177;
                ++v196;
              }
              if ( v199 )
                v2 = v199;
            }
          }
          goto LABEL_177;
        }
LABEL_330:
        sub_2519C60((__int64)&v260, 2 * v263);
        if ( !(_DWORD)v263 )
          goto LABEL_513;
        v1 = (__int64)v261;
        v192 = (v263 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v2 = (__int64)&v261[v192];
        v132 = v262 + 1;
        v193 = *(_QWORD *)v2;
        if ( v16 != *(_QWORD *)v2 )
        {
          v194 = 1;
          v195 = 0;
          while ( v193 != -4096 )
          {
            if ( v193 == -8192 && !v195 )
              v195 = v2;
            v192 = (v263 - 1) & (v194 + v192);
            v2 = (__int64)&v261[v192];
            v193 = *(_QWORD *)v2;
            if ( v16 == *(_QWORD *)v2 )
              goto LABEL_177;
            ++v194;
          }
          if ( v195 )
            v2 = v195;
        }
LABEL_177:
        LODWORD(v262) = v132;
        if ( *(_QWORD *)v2 != -4096 )
          --HIDWORD(v262);
        *(_QWORD *)v2 = v16;
        v133 = (unsigned int)v265;
        v134 = (unsigned int)v265 + 1LL;
        if ( v134 > HIDWORD(v265) )
        {
          sub_C8D5F0((__int64)&v264, &v266, v134, 8u, v1, v2);
          v133 = (unsigned int)v265;
        }
        v264[v133] = (void *)v16;
        LODWORD(v265) = v265 + 1;
        goto LABEL_12;
      }
      goto LABEL_12;
    }
LABEL_25:
    v255 = &v278[(unsigned int)v279];
    if ( v255 == v278 )
      goto LABEL_63;
    v259 = (unsigned __int64)v278;
    do
    {
      v25 = *(_QWORD *)v259;
      v26 = *(_QWORD **)(*(_QWORD *)v259 + 40LL);
      v27 = &v26[*(unsigned int *)(*(_QWORD *)v259 + 48LL)];
      if ( v27 == v26 )
        goto LABEL_56;
      v256 = *(_QWORD *)v259;
      v28 = *(_QWORD **)(*(_QWORD *)v259 + 40LL);
      do
      {
        while ( 1 )
        {
          v34 = (unsigned int)v263;
          v35 = (__int64)v261;
          v36 = *v28 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !(_DWORD)v263 )
          {
            ++v260;
LABEL_33:
            v247 = v27;
            v37 = ((((((((unsigned int)(2 * v263 - 1) | ((unsigned __int64)(unsigned int)(2 * v263 - 1) >> 1)) >> 2)
                     | (unsigned int)(2 * v263 - 1)
                     | ((unsigned __int64)(unsigned int)(2 * v263 - 1) >> 1)) >> 4)
                   | (((unsigned int)(2 * v263 - 1) | ((unsigned __int64)(unsigned int)(2 * v263 - 1) >> 1)) >> 2)
                   | (unsigned int)(2 * v263 - 1)
                   | ((unsigned __int64)(unsigned int)(2 * v263 - 1) >> 1)) >> 8)
                 | (((((unsigned int)(2 * v263 - 1) | ((unsigned __int64)(unsigned int)(2 * v263 - 1) >> 1)) >> 2)
                   | (unsigned int)(2 * v263 - 1)
                   | ((unsigned __int64)(unsigned int)(2 * v263 - 1) >> 1)) >> 4)
                 | (((unsigned int)(2 * v263 - 1) | ((unsigned __int64)(unsigned int)(2 * v263 - 1) >> 1)) >> 2)
                 | (unsigned int)(2 * v263 - 1)
                 | ((unsigned __int64)(unsigned int)(2 * v263 - 1) >> 1)) >> 16;
            v38 = (v37
                 | (((((((unsigned int)(2 * v263 - 1) | ((unsigned __int64)(unsigned int)(2 * v263 - 1) >> 1)) >> 2)
                     | (unsigned int)(2 * v263 - 1)
                     | ((unsigned __int64)(unsigned int)(2 * v263 - 1) >> 1)) >> 4)
                   | (((unsigned int)(2 * v263 - 1) | ((unsigned __int64)(unsigned int)(2 * v263 - 1) >> 1)) >> 2)
                   | (unsigned int)(2 * v263 - 1)
                   | ((unsigned __int64)(unsigned int)(2 * v263 - 1) >> 1)) >> 8)
                 | (((((unsigned int)(2 * v263 - 1) | ((unsigned __int64)(unsigned int)(2 * v263 - 1) >> 1)) >> 2)
                   | (unsigned int)(2 * v263 - 1)
                   | ((unsigned __int64)(unsigned int)(2 * v263 - 1) >> 1)) >> 4)
                 | (((unsigned int)(2 * v263 - 1) | ((unsigned __int64)(unsigned int)(2 * v263 - 1) >> 1)) >> 2)
                 | (unsigned int)(2 * v263 - 1)
                 | ((unsigned __int64)(unsigned int)(2 * v263 - 1) >> 1))
                + 1;
            if ( (unsigned int)v38 < 0x40 )
              LODWORD(v38) = 64;
            LODWORD(v263) = v38;
            v39 = (_QWORD *)sub_C7D670(8LL * (unsigned int)v38, 8);
            v27 = v247;
            v261 = v39;
            if ( v35 )
            {
              v40 = 8 * v34;
              v262 = 0;
              v41 = (__int64 *)(v35 + v40);
              for ( k = &v39[(unsigned int)v263]; k != v39; ++v39 )
              {
                if ( v39 )
                  *v39 = -4096;
              }
              v43 = (__int64 *)v35;
              if ( (__int64 *)v35 != v41 )
              {
                do
                {
                  v44 = *v43;
                  if ( *v43 != -4096 && v44 != -8192 )
                  {
                    if ( !(_DWORD)v263 )
                      goto LABEL_514;
                    v45 = (v263 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
                    v46 = &v261[v45];
                    v47 = *v46;
                    if ( *v46 != v44 )
                    {
                      v252 = 1;
                      v183 = 0;
                      while ( v47 != -4096 )
                      {
                        if ( v183 || v47 != -8192 )
                          v46 = v183;
                        v45 = (v263 - 1) & (v252 + v45);
                        v47 = v261[v45];
                        if ( v44 == v47 )
                        {
                          v46 = &v261[v45];
                          goto LABEL_45;
                        }
                        ++v252;
                        v183 = v46;
                        v46 = &v261[v45];
                      }
                      if ( v183 )
                        v46 = v183;
                    }
LABEL_45:
                    *v46 = v44;
                    LODWORD(v262) = v262 + 1;
                  }
                  ++v43;
                }
                while ( v41 != v43 );
              }
              v248 = v27;
              sub_C7D6A0(v35, v40, 8);
              v39 = v261;
              v48 = v263;
              v27 = v248;
              v49 = v262 + 1;
            }
            else
            {
              v262 = 0;
              v179 = &v39[(unsigned int)v263];
              v48 = v263;
              if ( v39 != v179 )
              {
                v180 = v39;
                do
                {
                  if ( v180 )
                    *v180 = -4096;
                  ++v180;
                }
                while ( v179 != v180 );
              }
              v49 = 1;
            }
            if ( !v48 )
              goto LABEL_513;
            v50 = v48 - 1;
            v51 = v50 & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
            v29 = &v39[v51];
            v1 = *v29;
            if ( v36 != *v29 )
            {
              v200 = 1;
              v2 = 0;
              while ( v1 != -4096 )
              {
                if ( v2 || v1 != -8192 )
                  v29 = (__int64 *)v2;
                v2 = (unsigned int)(v200 + 1);
                v51 = v50 & (v200 + v51);
                v1 = v39[v51];
                if ( v36 == v1 )
                {
                  v29 = &v39[v51];
                  goto LABEL_50;
                }
                ++v200;
                v2 = (__int64)v29;
                v29 = &v39[v51];
              }
              if ( v2 )
                v29 = (__int64 *)v2;
            }
            goto LABEL_50;
          }
          v1 = 1;
          v29 = 0;
          v30 = (unsigned int)(v263 - 1);
          v31 = v30 & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
          v32 = &v261[v31];
          v33 = *v32;
          if ( v36 != *v32 )
            break;
LABEL_30:
          if ( v27 == ++v28 )
            goto LABEL_55;
        }
        while ( v33 != -4096 )
        {
          if ( v33 != -8192 || v29 )
            v32 = v29;
          v31 = v30 & (v1 + v31);
          v2 = (__int64)&v261[v31];
          v33 = *(_QWORD *)v2;
          if ( v36 == *(_QWORD *)v2 )
            goto LABEL_30;
          v1 = (unsigned int)(v1 + 1);
          v29 = v32;
          v32 = &v261[v31];
        }
        if ( !v29 )
          v29 = v32;
        ++v260;
        v49 = v262 + 1;
        if ( 4 * ((int)v262 + 1) >= (unsigned int)(3 * v263) )
          goto LABEL_33;
        if ( (int)v263 - HIDWORD(v262) - v49 <= (unsigned int)v263 >> 3 )
        {
          v249 = v27;
          v114 = (((v30 >> 1) | v30) >> 2) | (v30 >> 1) | v30;
          v115 = (((v114 >> 4) | v114) >> 8) | (v114 >> 4) | v114;
          v116 = ((v115 >> 16) | v115) + 1;
          if ( (unsigned int)v116 < 0x40 )
            LODWORD(v116) = 64;
          LODWORD(v263) = v116;
          v117 = (_QWORD *)sub_C7D670(8LL * (unsigned int)v116, 8);
          v27 = v249;
          v261 = v117;
          if ( v35 )
          {
            v118 = 8 * v34;
            v262 = 0;
            for ( m = &v117[(unsigned int)v263]; m != v117; ++v117 )
            {
              if ( v117 )
                *v117 = -4096;
            }
            v120 = (__int64 *)v35;
            do
            {
              v44 = *v120;
              if ( *v120 != -4096 && v44 != -8192 )
              {
                if ( !(_DWORD)v263 )
                {
LABEL_514:
                  MEMORY[0] = v44;
                  BUG();
                }
                v121 = (v263 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
                v122 = &v261[v121];
                v123 = *v122;
                if ( *v122 != v44 )
                {
                  v243 = 1;
                  v253 = 0;
                  while ( v123 != -4096 )
                  {
                    if ( v123 == -8192 )
                    {
                      if ( v253 )
                        v122 = v253;
                      v253 = v122;
                    }
                    v121 = (v263 - 1) & (v243 + v121);
                    v122 = &v261[v121];
                    v123 = *v122;
                    if ( v44 == *v122 )
                      goto LABEL_161;
                    ++v243;
                  }
                  if ( v253 )
                    v122 = v253;
                }
LABEL_161:
                *v122 = v44;
                LODWORD(v262) = v262 + 1;
              }
              ++v120;
            }
            while ( (__int64 *)(v35 + v118) != v120 );
            v250 = v27;
            sub_C7D6A0(v35, v118, 8);
            v117 = v261;
            v124 = v263;
            v27 = v250;
            v49 = v262 + 1;
          }
          else
          {
            v262 = 0;
            v181 = &v117[(unsigned int)v263];
            v124 = v263;
            if ( v117 != v181 )
            {
              v182 = v117;
              do
              {
                if ( v182 )
                  *v182 = -4096;
                ++v182;
              }
              while ( v181 != v182 );
            }
            v49 = 1;
          }
          if ( !v124 )
            goto LABEL_513;
          v125 = v124 - 1;
          v2 = 1;
          v1 = 0;
          v126 = v125 & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
          v29 = &v117[v126];
          v127 = *v29;
          if ( v36 != *v29 )
          {
            while ( v127 != -4096 )
            {
              if ( v127 == -8192 && !v1 )
                v1 = (__int64)v29;
              v126 = v125 & (v2 + v126);
              v29 = &v117[v126];
              v127 = *v29;
              if ( v36 == *v29 )
                goto LABEL_50;
              v2 = (unsigned int)(v2 + 1);
            }
            if ( v1 )
              v29 = (__int64 *)v1;
          }
        }
LABEL_50:
        LODWORD(v262) = v49;
        if ( *v29 != -4096 )
          --HIDWORD(v262);
        *v29 = v36;
        v52 = (unsigned int)v265;
        v53 = (unsigned int)v265 + 1LL;
        if ( v53 > HIDWORD(v265) )
        {
          v251 = v27;
          sub_C8D5F0((__int64)&v264, &v266, v53, 8u, v1, v2);
          v52 = (unsigned int)v265;
          v27 = v251;
        }
        ++v28;
        v264[v52] = (void *)v36;
        LODWORD(v265) = v265 + 1;
      }
      while ( v27 != v28 );
LABEL_55:
      v25 = v256;
LABEL_56:
      v54 = *(_DWORD *)(v25 + 24);
      ++*(_QWORD *)(v25 + 8);
      if ( v54 )
      {
        v135 = 4 * v54;
        v55 = *(unsigned int *)(v25 + 32);
        if ( (unsigned int)(4 * v54) < 0x40 )
          v135 = 64;
        if ( (unsigned int)v55 <= v135 )
        {
LABEL_59:
          v56 = *(_QWORD **)(v25 + 16);
          for ( n = &v56[v55]; n != v56; ++v56 )
            *v56 = -4;
          goto LABEL_61;
        }
        v136 = v54 - 1;
        if ( v136 )
        {
          _BitScanReverse(&v136, v136);
          v137 = *(_QWORD **)(v25 + 16);
          v138 = 1 << (33 - (v136 ^ 0x1F));
          if ( v138 < 64 )
            v138 = 64;
          if ( (_DWORD)v55 == v138 )
          {
            *(_QWORD *)(v25 + 24) = 0;
            v139 = &v137[v55];
            do
            {
              if ( v137 )
                *v137 = -4;
              ++v137;
            }
            while ( v139 != v137 );
            goto LABEL_62;
          }
        }
        else
        {
          v137 = *(_QWORD **)(v25 + 16);
          v138 = 64;
        }
        sub_C7D6A0((__int64)v137, 8 * v55, 8);
        v184 = sub_AF1560(4 * v138 / 3u + 1);
        *(_DWORD *)(v25 + 32) = v184;
        if ( !v184 )
          goto LABEL_314;
        v185 = (_QWORD *)sub_C7D670(8LL * v184, 8);
        v186 = *(unsigned int *)(v25 + 32);
        *(_QWORD *)(v25 + 24) = 0;
        *(_QWORD *)(v25 + 16) = v185;
        for ( ii = &v185[v186]; ii != v185; ++v185 )
        {
          if ( v185 )
            *v185 = -4;
        }
      }
      else if ( *(_DWORD *)(v25 + 28) )
      {
        v55 = *(unsigned int *)(v25 + 32);
        if ( (unsigned int)v55 <= 0x40 )
          goto LABEL_59;
        sub_C7D6A0(*(_QWORD *)(v25 + 16), 8 * v55, 8);
        *(_DWORD *)(v25 + 32) = 0;
LABEL_314:
        *(_QWORD *)(v25 + 16) = 0;
LABEL_61:
        *(_QWORD *)(v25 + 24) = 0;
      }
LABEL_62:
      v259 += 8LL;
      *(_DWORD *)(v25 + 48) = 0;
    }
    while ( v255 != (void **)v259 );
LABEL_63:
    ++v266;
    LODWORD(v279) = 0;
    if ( (_DWORD)v268 )
    {
      v163 = 4 * v268;
      v58 = (unsigned int)v269;
      if ( (unsigned int)(4 * v268) < 0x40 )
        v163 = 64;
      if ( v163 >= (unsigned int)v269 )
      {
LABEL_66:
        v59 = v267;
        v60 = &v267[v58];
        if ( v267 != v60 )
        {
          do
            *v59++ = -4096;
          while ( v60 != v59 );
        }
        goto LABEL_68;
      }
      v164 = v267;
      v165 = (unsigned int)v269;
      if ( (_DWORD)v268 == 1 )
      {
        v167 = 64;
      }
      else
      {
        _BitScanReverse(&v166, v268 - 1);
        v167 = 1 << (33 - (v166 ^ 0x1F));
        if ( v167 < 64 )
          v167 = 64;
        if ( (_DWORD)v269 == v167 )
        {
          v268 = 0;
          v1 = (__int64)&v267[v165];
          do
          {
            if ( v164 )
              *v164 = -4096;
            ++v164;
          }
          while ( (_QWORD *)v1 != v164 );
          goto LABEL_69;
        }
      }
      sub_C7D6A0((__int64)v267, v165 * 8, 8);
      v168 = sub_2507810(v167);
      LODWORD(v269) = v168;
      if ( !v168 )
        goto LABEL_385;
      v169 = (_QWORD *)sub_C7D670(8LL * v168, 8);
      v268 = 0;
      v267 = v169;
      for ( jj = &v169[(unsigned int)v269]; jj != v169; ++v169 )
      {
        if ( v169 )
          *v169 = -4096;
      }
    }
    else if ( HIDWORD(v268) )
    {
      v58 = (unsigned int)v269;
      if ( (unsigned int)v269 <= 0x40 )
        goto LABEL_66;
      sub_C7D6A0((__int64)v267, 8LL * (unsigned int)v269, 8);
      LODWORD(v269) = 0;
LABEL_385:
      v267 = 0;
LABEL_68:
      v268 = 0;
    }
LABEL_69:
    v61 = v264;
    LODWORD(v271) = 0;
    v62 = &v264[(unsigned int)v265];
    if ( v62 != v264 )
    {
      do
      {
LABEL_73:
        v294 = (char *)*v61;
        v63 = (*(__int64 (**)(void))(*(_QWORD *)v294 + 40LL))();
        if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v63 + 24LL))(v63)
          || (unsigned int)sub_251C580(a1, (__int64)v294) )
        {
          if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v63 + 16LL))(v63) )
            goto LABEL_72;
        }
        else
        {
          v65 = (unsigned int)v279;
          v66 = v294;
          v67 = (unsigned int)v279 + 1LL;
          if ( v67 > HIDWORD(v279) )
          {
            v257 = v294;
            sub_C8D5F0((__int64)&v278, v280, v67, 8u, v64, (__int64)v294);
            v65 = (unsigned int)v279;
            v66 = v257;
          }
          v278[v65] = v66;
          LODWORD(v279) = v279 + 1;
          if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v63 + 16LL))(v63) )
          {
LABEL_72:
            if ( v62 == ++v61 )
              break;
            goto LABEL_73;
          }
        }
        ++v61;
        sub_2519E30((__int64)&v266, (__int64 *)&v294);
      }
      while ( v62 != v61 );
    }
    v68 = 8 * v244 + *(_QWORD *)(a1 + 256);
    v69 = 8LL * *(unsigned int *)(a1 + 264) - 8 * v244;
    v70 = (unsigned int)v279;
    v71 = v69 >> 3;
    v72 = (v69 >> 3) + (unsigned int)v279;
    v73 = v69 >> 3;
    if ( v72 > HIDWORD(v279) )
    {
      sub_C8D5F0((__int64)&v278, v280, v72, 8u, v1, v2);
      v70 = (unsigned int)v279;
    }
    v74 = &v278[v70];
    if ( v69 > 0 )
    {
      do
      {
        v75 = (_QWORD *)v68;
        ++v74;
        v68 += 8;
        *(v74 - 1) = (void *)sub_2505D60(v75);
        --v73;
      }
      while ( v73 );
      LODWORD(v70) = v279;
    }
    ++v260;
    LODWORD(v279) = v70 + v71;
    v76 = (unsigned int)(v70 + v71);
    if ( (_DWORD)v262 )
    {
      v157 = 4 * v262;
      v77 = v263;
      if ( (unsigned int)(4 * v262) < 0x40 )
        v157 = 64;
      if ( v157 >= (unsigned int)v263 )
      {
LABEL_87:
        v78 = v261;
        v79 = &v261[v77];
        if ( v261 != v79 )
        {
          do
            *v78++ = -4096;
          while ( v79 != v78 );
          v76 = (unsigned int)v279;
        }
        v262 = 0;
        goto LABEL_91;
      }
      v1 = (__int64)v261;
      if ( (_DWORD)v262 == 1 )
      {
        v159 = 64;
      }
      else
      {
        _BitScanReverse(&v158, v262 - 1);
        v159 = 1 << (33 - (v158 ^ 0x1F));
        if ( v159 < 64 )
          v159 = 64;
        if ( v159 == (_DWORD)v263 )
        {
          v262 = 0;
          v2 = (__int64)&v261[(unsigned int)v263];
          do
          {
            if ( v1 )
              *(_QWORD *)v1 = -4096;
            v1 += 8;
          }
          while ( v2 != v1 );
          v76 = (unsigned int)v279;
          goto LABEL_91;
        }
      }
      sub_C7D6A0((__int64)v261, 8LL * (unsigned int)v263, 8);
      v160 = sub_2507810(v159);
      LODWORD(v263) = v160;
      if ( !v160 )
        goto LABEL_381;
      v161 = (_QWORD *)sub_C7D670(8LL * v160, 8);
      v262 = 0;
      v261 = v161;
      for ( kk = &v161[(unsigned int)v263]; kk != v161; ++v161 )
      {
        if ( v161 )
          *v161 = -4096;
      }
      v76 = (unsigned int)v279;
    }
    else
    {
      v2 = HIDWORD(v262);
      if ( HIDWORD(v262) )
      {
        v77 = v263;
        if ( (unsigned int)v263 <= 0x40 )
          goto LABEL_87;
        sub_C7D6A0((__int64)v261, 8LL * (unsigned int)v263, 8);
        LODWORD(v263) = 0;
LABEL_381:
        v261 = 0;
        v76 = (unsigned int)v279;
        v262 = 0;
      }
    }
LABEL_91:
    v80 = v278;
    LODWORD(v265) = 0;
    v81 = &v278[v76];
    if ( v278 != v81 )
    {
      while ( (_DWORD)v263 )
      {
        v2 = (unsigned int)(v263 - 1);
        v82 = 1;
        v83 = 0;
        v1 = (__int64)v261;
        v84 = v2 & (((unsigned int)*v80 >> 9) ^ ((unsigned int)*v80 >> 4));
        v85 = &v261[v84];
        v86 = (void *)*v85;
        if ( *v80 == (void *)*v85 )
        {
LABEL_94:
          if ( ++v80 == v81 )
            goto LABEL_104;
        }
        else
        {
          while ( v86 != (void *)-4096LL )
          {
            if ( v83 || v86 != (void *)-8192LL )
              v85 = v83;
            v84 = v2 & (v82 + v84);
            v86 = (void *)v261[v84];
            if ( *v80 == v86 )
              goto LABEL_94;
            ++v82;
            v83 = v85;
            v85 = &v261[v84];
          }
          if ( !v83 )
            v83 = v85;
          ++v260;
          v89 = v262 + 1;
          if ( 4 * ((int)v262 + 1) < (unsigned int)(3 * v263) )
          {
            if ( (int)v263 - HIDWORD(v262) - v89 > (unsigned int)v263 >> 3 )
              goto LABEL_99;
            sub_2519C60((__int64)&v260, v263);
            if ( !(_DWORD)v263 )
              goto LABEL_513;
            v2 = (__int64)v261;
            v1 = 0;
            v145 = 1;
            v146 = (v263 - 1) & (((unsigned int)*v80 >> 9) ^ ((unsigned int)*v80 >> 4));
            v83 = &v261[v146];
            v147 = *v83;
            v89 = v262 + 1;
            if ( (void *)*v83 == *v80 )
              goto LABEL_99;
            while ( v147 != -4096 )
            {
              if ( v147 == -8192 && !v1 )
                v1 = (__int64)v83;
              v146 = (v263 - 1) & (v145 + v146);
              v83 = &v261[v146];
              v147 = *v83;
              if ( *v80 == (void *)*v83 )
                goto LABEL_99;
              ++v145;
            }
            goto LABEL_218;
          }
LABEL_97:
          sub_2519C60((__int64)&v260, 2 * v263);
          if ( !(_DWORD)v263 )
            goto LABEL_513;
          v2 = (__int64)v261;
          v87 = (v263 - 1) & (((unsigned int)*v80 >> 9) ^ ((unsigned int)*v80 >> 4));
          v83 = &v261[v87];
          v88 = *v83;
          v89 = v262 + 1;
          if ( (void *)*v83 == *v80 )
            goto LABEL_99;
          v239 = 1;
          v1 = 0;
          while ( v88 != -4096 )
          {
            if ( !v1 && v88 == -8192 )
              v1 = (__int64)v83;
            v87 = (v263 - 1) & (v239 + v87);
            v83 = &v261[v87];
            v88 = *v83;
            if ( *v80 == (void *)*v83 )
              goto LABEL_99;
            ++v239;
          }
LABEL_218:
          if ( v1 )
            v83 = (_QWORD *)v1;
LABEL_99:
          LODWORD(v262) = v89;
          if ( *v83 != -4096 )
            --HIDWORD(v262);
          v90 = *v80;
          *v83 = *v80;
          v91 = (unsigned int)v265;
          v92 = (unsigned int)v265 + 1LL;
          if ( v92 > HIDWORD(v265) )
          {
            sub_C8D5F0((__int64)&v264, &v266, v92, 8u, v1, v2);
            v91 = (unsigned int)v265;
          }
          ++v80;
          v264[v91] = v90;
          LODWORD(v265) = v265 + 1;
          if ( v80 == v81 )
            goto LABEL_104;
        }
      }
      ++v260;
      goto LABEL_97;
    }
LABEL_104:
    v93 = *(void ***)(a1 + 4152);
    v94 = &v93[*(unsigned int *)(a1 + 4160)];
    if ( v93 != v94 )
    {
      while ( (_DWORD)v263 )
      {
        v2 = (unsigned int)(v263 - 1);
        v95 = 1;
        v96 = 0;
        v1 = (__int64)v261;
        v97 = v2 & (((unsigned int)*v93 >> 9) ^ ((unsigned int)*v93 >> 4));
        v98 = &v261[v97];
        v99 = (void *)*v98;
        if ( *v93 == (void *)*v98 )
        {
LABEL_107:
          if ( ++v93 == v94 )
            goto LABEL_117;
        }
        else
        {
          while ( v99 != (void *)-4096LL )
          {
            if ( v99 != (void *)-8192LL || v96 )
              v98 = v96;
            v97 = v2 & (v95 + v97);
            v99 = (void *)v261[v97];
            if ( *v93 == v99 )
              goto LABEL_107;
            ++v95;
            v96 = v98;
            v98 = &v261[v97];
          }
          if ( !v96 )
            v96 = v98;
          ++v260;
          v102 = v262 + 1;
          if ( 4 * ((int)v262 + 1) < (unsigned int)(3 * v263) )
          {
            if ( (int)v263 - HIDWORD(v262) - v102 > (unsigned int)v263 >> 3 )
              goto LABEL_112;
            sub_2519C60((__int64)&v260, v263);
            if ( !(_DWORD)v263 )
              goto LABEL_513;
            v2 = (__int64)v261;
            v1 = 0;
            v148 = 1;
            v149 = (v263 - 1) & (((unsigned int)*v93 >> 9) ^ ((unsigned int)*v93 >> 4));
            v96 = &v261[v149];
            v150 = *v96;
            v102 = v262 + 1;
            if ( *v93 == (void *)*v96 )
              goto LABEL_112;
            while ( v150 != -4096 )
            {
              if ( !v1 && v150 == -8192 )
                v1 = (__int64)v96;
              v149 = (v263 - 1) & (v148 + v149);
              v96 = &v261[v149];
              v150 = *v96;
              if ( *v93 == (void *)*v96 )
                goto LABEL_112;
              ++v148;
            }
            goto LABEL_233;
          }
LABEL_110:
          sub_2519C60((__int64)&v260, 2 * v263);
          if ( !(_DWORD)v263 )
            goto LABEL_513;
          v2 = (__int64)v261;
          v100 = (v263 - 1) & (((unsigned int)*v93 >> 9) ^ ((unsigned int)*v93 >> 4));
          v96 = &v261[v100];
          v101 = *v96;
          v102 = v262 + 1;
          if ( *v93 == (void *)*v96 )
            goto LABEL_112;
          v238 = 1;
          v1 = 0;
          while ( v101 != -4096 )
          {
            if ( !v1 && v101 == -8192 )
              v1 = (__int64)v96;
            v100 = (v263 - 1) & (v238 + v100);
            v96 = &v261[v100];
            v101 = *v96;
            if ( *v93 == (void *)*v96 )
              goto LABEL_112;
            ++v238;
          }
LABEL_233:
          if ( v1 )
            v96 = (_QWORD *)v1;
LABEL_112:
          LODWORD(v262) = v102;
          if ( *v96 != -4096 )
            --HIDWORD(v262);
          v103 = *v93;
          *v96 = *v93;
          v104 = (unsigned int)v265;
          v105 = (unsigned int)v265 + 1LL;
          if ( v105 > HIDWORD(v265) )
          {
            sub_C8D5F0((__int64)&v264, &v266, v105, 8u, v1, v2);
            v104 = (unsigned int)v265;
          }
          ++v93;
          v264[v104] = v103;
          LODWORD(v265) = v265 + 1;
          if ( v93 == v94 )
            goto LABEL_117;
        }
      }
      ++v260;
      goto LABEL_110;
    }
LABEL_117:
    ++*(_QWORD *)(a1 + 4120);
    v106 = *(_DWORD *)(a1 + 4136);
    if ( v106 )
    {
      v171 = 4 * v106;
      v107 = *(unsigned int *)(a1 + 4144);
      if ( (unsigned int)(4 * v106) < 0x40 )
        v171 = 64;
      if ( v171 >= (unsigned int)v107 )
      {
LABEL_120:
        v108 = *(_QWORD **)(a1 + 4128);
        for ( mm = &v108[v107]; mm != v108; ++v108 )
          *v108 = -4096;
        *(_QWORD *)(a1 + 4136) = 0;
        goto LABEL_123;
      }
      v172 = v106 - 1;
      if ( v172 )
      {
        _BitScanReverse(&v172, v172);
        v173 = 1 << (33 - (v172 ^ 0x1F));
        v174 = *(_QWORD **)(a1 + 4128);
        if ( v173 < 64 )
          v173 = 64;
        if ( (_DWORD)v107 == v173 )
        {
          *(_QWORD *)(a1 + 4136) = 0;
          v218 = &v174[v107];
          do
          {
            if ( v174 )
              *v174 = -4096;
            ++v174;
          }
          while ( v218 != v174 );
          goto LABEL_123;
        }
      }
      else
      {
        v173 = 64;
        v174 = *(_QWORD **)(a1 + 4128);
      }
      sub_C7D6A0((__int64)v174, 8 * v107, 8);
      v175 = sub_2507810(v173);
      *(_DWORD *)(a1 + 4144) = v175;
      if ( !v175 )
        goto LABEL_383;
      v176 = (_QWORD *)sub_C7D670(8LL * v175, 8);
      v177 = *(unsigned int *)(a1 + 4144);
      *(_QWORD *)(a1 + 4128) = v176;
      *(_QWORD *)(a1 + 4136) = 0;
      for ( nn = &v176[v177]; nn != v176; ++v176 )
      {
        if ( v176 )
          *v176 = -4096;
      }
    }
    else
    {
      v1 = *(unsigned int *)(a1 + 4140);
      if ( (_DWORD)v1 )
      {
        v107 = *(unsigned int *)(a1 + 4144);
        if ( (unsigned int)v107 <= 0x40 )
          goto LABEL_120;
        sub_C7D6A0(*(_QWORD *)(a1 + 4128), 8 * v107, 8);
        *(_DWORD *)(a1 + 4144) = 0;
LABEL_383:
        *(_QWORD *)(a1 + 4128) = 0;
        *(_QWORD *)(a1 + 4136) = 0;
      }
    }
LABEL_123:
    v110 = v265;
    *(_DWORD *)(a1 + 4160) = 0;
    if ( v110 )
    {
      if ( v246 < v245 )
      {
        ++v246;
        v7 = v271;
        v244 = *(unsigned int *)(a1 + 264);
        continue;
      }
      ++v246;
    }
    break;
  }
  if ( v246 > v245 )
  {
    v201 = *(_QWORD *)(a1 + 200);
    if ( *(_DWORD *)(v201 + 40) )
    {
      v219 = **(_QWORD **)(v201 + 32);
      v220 = *(__int64 (__fastcall **)(_QWORD, __int64))(a1 + 4392);
      if ( v220 )
      {
        v221 = (__int64 *)v220(*(_QWORD *)(a1 + 4400), v219);
        v222 = *v221;
        v223 = v221;
        v224 = sub_B2BE50(*v221);
        if ( sub_B6EA50(v224)
          || (v240 = sub_B2BE50(v222),
              v241 = sub_B6F970(v240),
              (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v241 + 48LL))(v241)) )
        {
          sub_B17770((__int64)&v294, *(_QWORD *)(a1 + 4408), (__int64)"FixedPoint", 10, v219);
          sub_B18290((__int64)&v294, "Attributor did not reach a fixpoint after ", 0x2Au);
          sub_B169E0((__int64 *)v272, "Iterations", 10, v245);
          v281 = &v284;
          sub_2506C40((__int64 *)&v281, v272[0], (__int64)&v272[0][(unsigned __int64)v272[1]]);
          v285.m128i_i64[1] = (__int64)&v287;
          sub_2506C40(&v285.m128i_i64[1], v274, (__int64)v274 + v275);
          v288 = _mm_loadu_si128(&v277);
          sub_B180C0((__int64)&v294, (unsigned __int64)&v281);
          if ( (__m128i *)v285.m128i_i64[1] != &v287 )
            j_j___libc_free_0(v285.m128i_u64[1]);
          if ( v281 != &v284 )
            j_j___libc_free_0((unsigned __int64)v281);
          sub_B18290((__int64)&v294, " iterations.", 0xCu);
          v228 = _mm_loadu_si128(&v297);
          v229 = _mm_loadu_si128(&v299);
          v230 = _mm_loadu_si128(&v300);
          v289 = (unsigned __int64 *)v291;
          v282 = (int)v295;
          v285 = v228;
          v283 = BYTE4(v295);
          v287 = v229;
          v284 = v296;
          v288 = v230;
          v281 = (__int64 *)&unk_49D9D40;
          v286 = v298;
          v290 = 0x400000000LL;
          if ( v302 )
            sub_2510D60((__int64)&v289, (__int64)&v301, v225, v302, v226, v227);
          v291[320] = v303[320];
          v292 = v304;
          v293 = v305;
          v281 = (__int64 *)&unk_49D9DB0;
          if ( v274 != &v276 )
            j_j___libc_free_0((unsigned __int64)v274);
          if ( (__int64 *)v272[0] != &v273 )
            j_j___libc_free_0((unsigned __int64)v272[0]);
          v231 = v301;
          v294 = (char *)&unk_49D9D40;
          v232 = 10LL * v302;
          v233 = &v301[v232];
          if ( v301 != &v301[v232] )
          {
            do
            {
              v233 -= 10;
              v234 = v233[4];
              if ( (unsigned __int64 *)v234 != v233 + 6 )
                j_j___libc_free_0(v234);
              if ( (unsigned __int64 *)*v233 != v233 + 2 )
                j_j___libc_free_0(*v233);
            }
            while ( v231 != v233 );
            v233 = v301;
          }
          if ( v233 != (unsigned __int64 *)v303 )
            _libc_free((unsigned __int64)v233);
          sub_1049740(v223, (__int64)&v281);
          v235 = v289;
          v281 = (__int64 *)&unk_49D9D40;
          v236 = &v289[10 * (unsigned int)v290];
          if ( v289 != v236 )
          {
            do
            {
              v236 -= 10;
              v237 = v236[4];
              if ( (unsigned __int64 *)v237 != v236 + 6 )
                j_j___libc_free_0(v237);
              if ( (unsigned __int64 *)*v236 != v236 + 2 )
                j_j___libc_free_0(*v236);
            }
            while ( v235 != v236 );
            v236 = v289;
          }
          if ( v236 != (unsigned __int64 *)v291 )
            _libc_free((unsigned __int64)v236);
        }
      }
    }
  }
  v202 = 1;
  v203 = 0;
  v204 = (__int64)&v278;
  v295 = &v297.m128i_u64[1];
  v205 = 0;
  v294 = 0;
  v296 = 32;
  v297.m128i_i32[0] = 0;
  v297.m128i_i8[4] = 1;
  if ( (_DWORD)v279 )
  {
    while ( 1 )
    {
      v206 = (unsigned __int64 *)v278;
      v207 = (__int64)v278[v205];
      if ( !(_BYTE)v202 )
        goto LABEL_368;
      v208 = v295;
      v204 = HIDWORD(v296);
      v206 = &v295[HIDWORD(v296)];
      if ( v295 != v206 )
      {
        while ( v207 != *v208 )
        {
          if ( v206 == ++v208 )
            goto LABEL_377;
        }
LABEL_357:
        v205 = (unsigned int)(v203 + 1);
        v203 = v205;
        if ( (unsigned int)v205 >= (unsigned int)v279 )
          goto LABEL_358;
        continue;
      }
LABEL_377:
      if ( HIDWORD(v296) < (unsigned int)v296 )
      {
        ++HIDWORD(v296);
        *v206 = v207;
        ++v294;
      }
      else
      {
LABEL_368:
        sub_C8CC70((__int64)&v294, v207, (__int64)v206, v204, v1, v2);
        v202 = v297.m128i_u8[4];
        if ( !v210 )
          goto LABEL_357;
      }
      v211 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v207 + 40LL))(v207, v202);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v211 + 24LL))(v211) )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v211 + 40LL))(v211);
      v214 = *(_QWORD **)(v207 + 40);
      v215 = &v214[*(unsigned int *)(v207 + 48)];
      for ( i1 = (unsigned int)v279; v215 != v214; LODWORD(v279) = v279 + 1 )
      {
        v217 = *v214 & 0xFFFFFFFFFFFFFFF8LL;
        if ( i1 + 1 > (unsigned __int64)HIDWORD(v279) )
        {
          sub_C8D5F0((__int64)&v278, v280, i1 + 1, 8u, v212, v213);
          i1 = (unsigned int)v279;
        }
        ++v214;
        v278[i1] = (void *)v217;
        i1 = (unsigned int)(v279 + 1);
      }
      sub_2510BF0(v207 + 8);
      v205 = (unsigned int)(v203 + 1);
      *(_DWORD *)(v207 + 48) = 0;
      v202 = v297.m128i_u8[4];
      v203 = v205;
      if ( (unsigned int)v205 >= (unsigned int)v279 )
      {
LABEL_358:
        if ( !(_BYTE)v202 )
          _libc_free((unsigned __int64)v295);
        break;
      }
    }
  }
  if ( v270 != v272 )
    _libc_free((unsigned __int64)v270);
  sub_C7D6A0((__int64)v267, 8LL * (unsigned int)v269, 8);
  if ( v264 != (void **)&v266 )
    _libc_free((unsigned __int64)v264);
  sub_C7D6A0((__int64)v261, 8LL * (unsigned int)v263, 8);
  if ( v278 != (void **)v280 )
    _libc_free((unsigned __int64)v278);
  result = (__int64 *)v242;
  if ( v242 )
    return sub_C9AF60(v242);
  return result;
}
