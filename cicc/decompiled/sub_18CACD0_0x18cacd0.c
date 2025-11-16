// Function: sub_18CACD0
// Address: 0x18cacd0
//
__int64 __fastcall sub_18CACD0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rax
  _QWORD *v23; // rbx
  _QWORD *v24; // r13
  char *v25; // rdi
  char *v26; // r8
  unsigned int v27; // eax
  __int64 v28; // rdx
  unsigned int v29; // r13d
  _QWORD *v30; // rbx
  _QWORD *v31; // r14
  __int64 v32; // rax
  unsigned __int64 *v33; // rax
  unsigned __int64 *v34; // r12
  unsigned __int8 v36; // al
  char v37; // cl
  _QWORD *i; // rcx
  _QWORD *v39; // rax
  __int64 v40; // rdi
  _QWORD *v41; // r15
  __int64 kk; // r15
  unsigned __int8 v43; // r13
  __int64 v44; // r13
  __int64 v45; // r12
  _QWORD *v46; // rax
  int v47; // r8d
  int v48; // r9d
  unsigned __int32 v49; // eax
  __int64 v50; // r12
  __int64 v51; // r13
  _QWORD *v52; // rax
  int v53; // r8d
  int v54; // r9d
  __int64 v55; // rdx
  unsigned __int8 v56; // al
  unsigned __int64 v57; // r12
  unsigned __int64 v58; // r12
  __int64 k; // r12
  char v60; // al
  _QWORD *v61; // rax
  bool v62; // zf
  __int64 v63; // rcx
  unsigned __int64 v64; // rsi
  __int64 v65; // rcx
  __int64 v66; // r8
  _QWORD *v67; // rdx
  unsigned __int64 v68; // r14
  char v69; // al
  unsigned __int64 v70; // r12
  __int64 m; // rdi
  int v72; // edi
  __int64 v73; // rax
  __int64 v74; // r14
  unsigned __int8 v75; // al
  __int64 v76; // rdi
  int v77; // edi
  __int64 v78; // rax
  __int64 v79; // r14
  unsigned __int8 v80; // al
  int v81; // eax
  __int64 v82; // rdx
  unsigned int v83; // eax
  char *v84; // r9
  unsigned int v85; // eax
  __int64 v86; // rdi
  int v87; // edi
  __int64 v88; // rax
  __int64 v89; // r12
  unsigned __int8 v90; // al
  int v91; // eax
  _QWORD *v92; // r14
  _QWORD *v93; // r13
  __int64 v94; // rbx
  _QWORD *v95; // r15
  __int64 v96; // r14
  unsigned __int8 v97; // al
  int v98; // ecx
  __int64 v99; // rcx
  _QWORD *v100; // r15
  __int64 n; // rdi
  int v102; // edi
  __int64 v103; // rax
  __int64 v104; // r14
  unsigned __int8 v105; // al
  __int64 j; // rdi
  int v107; // edi
  __int64 v108; // rax
  __int64 v109; // r12
  unsigned __int8 v110; // al
  char v111; // al
  __int64 v112; // r12
  __int64 v113; // r13
  _QWORD *v114; // rdi
  double v115; // xmm4_8
  double v116; // xmm5_8
  __m128 *v117; // r15
  __m128 *v118; // r13
  unsigned __int64 v119; // rax
  __int64 v120; // rax
  __int64 *v121; // rdx
  __int64 *v122; // rcx
  __int64 *v123; // r12
  __int64 v124; // rdi
  unsigned int v125; // esi
  int v126; // eax
  bool v127; // al
  char v128; // al
  __int64 v129; // rdi
  __int64 v130; // rsi
  _QWORD *v131; // rax
  __int64 v132; // rcx
  _QWORD *v133; // rdx
  _QWORD *v134; // rax
  __int64 v135; // rax
  int v136; // eax
  char *v137; // rcx
  __int64 v138; // r12
  _QWORD *v139; // rax
  _QWORD *v140; // rsi
  unsigned int v141; // eax
  __int64 v142; // rdi
  __int64 v143; // rdi
  int v144; // edi
  __int64 v145; // rax
  __int64 v146; // rbx
  unsigned __int8 v147; // al
  int v148; // eax
  __int64 v149; // rcx
  _QWORD *v150; // rax
  __int64 v151; // rsi
  unsigned __int64 v152; // r8
  __int64 v153; // rsi
  __int64 v154; // rax
  size_t v155; // rdx
  size_t v156; // r12
  _BYTE *v157; // r14
  _QWORD *v158; // rax
  __int64 *v159; // rax
  __int64 **v160; // rax
  __int64 v161; // rax
  __int64 *v162; // rax
  __int64 v163; // r12
  __int64 v164; // rdi
  _QWORD *v165; // rbx
  unsigned int v166; // eax
  char v167; // al
  _QWORD *v168; // r12
  __int64 v169; // rdi
  __int64 v170; // rdi
  __int64 ii; // rdi
  int v172; // edi
  __int64 v173; // rax
  __int64 v174; // rbx
  unsigned __int8 v175; // al
  __int64 v176; // rdx
  _QWORD *v177; // rax
  __int64 *v178; // rax
  __int64 v179; // rdx
  _QWORD *v180; // rax
  _QWORD *v181; // rax
  _QWORD *v182; // r14
  _QWORD *v183; // rax
  _QWORD *v184; // r14
  __int64 v185; // rdi
  _QWORD *v186; // r14
  __int64 *v187; // rax
  __int64 v188; // rdx
  _QWORD *v189; // rax
  _QWORD *v190; // rax
  double v191; // xmm4_8
  double v192; // xmm5_8
  __int64 v193; // r13
  _QWORD *v194; // rsi
  unsigned int v195; // edx
  _QWORD *v196; // rcx
  __int64 v197; // rsi
  unsigned __int8 *v198; // rsi
  char v199; // al
  __int64 *v200; // r14
  __int64 *v201; // rax
  __int64 *v202; // rax
  __int64 v203; // rax
  __int64 v204; // rax
  _QWORD *v205; // rax
  _QWORD *v206; // rax
  __int64 v207; // [rsp+8h] [rbp-1E8h]
  __int64 v208; // [rsp+10h] [rbp-1E0h]
  __int64 v209; // [rsp+18h] [rbp-1D8h]
  __int64 v210; // [rsp+18h] [rbp-1D8h]
  unsigned int v211; // [rsp+20h] [rbp-1D0h]
  char v212; // [rsp+27h] [rbp-1C9h]
  __int64 v213; // [rsp+28h] [rbp-1C8h]
  __int64 v214; // [rsp+28h] [rbp-1C8h]
  __int64 v215; // [rsp+28h] [rbp-1C8h]
  __int64 v216; // [rsp+30h] [rbp-1C0h]
  _QWORD *v217; // [rsp+38h] [rbp-1B8h]
  __int64 *v218; // [rsp+38h] [rbp-1B8h]
  _QWORD *v219; // [rsp+38h] [rbp-1B8h]
  _QWORD *v220; // [rsp+38h] [rbp-1B8h]
  __int64 *v221; // [rsp+38h] [rbp-1B8h]
  __int64 v222; // [rsp+38h] [rbp-1B8h]
  __int64 *v223; // [rsp+38h] [rbp-1B8h]
  _QWORD *jj; // [rsp+40h] [rbp-1B0h]
  char v226; // [rsp+50h] [rbp-1A0h]
  __int64 *v227; // [rsp+50h] [rbp-1A0h]
  _QWORD *v228; // [rsp+50h] [rbp-1A0h]
  _QWORD *v229; // [rsp+50h] [rbp-1A0h]
  _QWORD *v230; // [rsp+50h] [rbp-1A0h]
  _QWORD *v231; // [rsp+50h] [rbp-1A0h]
  _QWORD *v232; // [rsp+50h] [rbp-1A0h]
  int v233; // [rsp+58h] [rbp-198h]
  _QWORD *v234; // [rsp+58h] [rbp-198h]
  __int64 v235; // [rsp+58h] [rbp-198h]
  __int64 *v236; // [rsp+58h] [rbp-198h]
  _QWORD *v237; // [rsp+58h] [rbp-198h]
  __int64 v238; // [rsp+68h] [rbp-188h] BYREF
  __int64 v239; // [rsp+70h] [rbp-180h] BYREF
  _QWORD *v240; // [rsp+78h] [rbp-178h]
  void *v241; // [rsp+80h] [rbp-170h]
  unsigned int v242; // [rsp+88h] [rbp-168h]
  __m128i v243; // [rsp+90h] [rbp-160h] BYREF
  __m128i v244; // [rsp+A0h] [rbp-150h] BYREF
  __int64 v245; // [rsp+B0h] [rbp-140h]
  _OWORD v246[2]; // [rsp+C0h] [rbp-130h] BYREF
  __int64 v247; // [rsp+E0h] [rbp-110h]
  __m128 v248; // [rsp+F0h] [rbp-100h] BYREF
  __m128 v249; // [rsp+100h] [rbp-F0h] BYREF
  __int64 v250; // [rsp+110h] [rbp-E0h]
  __int64 v251; // [rsp+120h] [rbp-D0h] BYREF
  _BYTE *v252; // [rsp+128h] [rbp-C8h]
  void *v253; // [rsp+130h] [rbp-C0h]
  _BYTE v254[12]; // [rsp+138h] [rbp-B8h]
  _BYTE v255[40]; // [rsp+148h] [rbp-A8h] BYREF
  __int64 v256; // [rsp+170h] [rbp-80h] BYREF
  _BYTE *v257; // [rsp+178h] [rbp-78h]
  void *s; // [rsp+180h] [rbp-70h]
  _BYTE v259[12]; // [rsp+188h] [rbp-68h]
  _BYTE v260[88]; // [rsp+198h] [rbp-58h] BYREF

  v10 = *(__int64 **)(a1 + 8);
  *(_BYTE *)(a1 + 153) = 0;
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_406:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F96DB4 )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_406;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F96DB4);
  v14 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 160) = *(_QWORD *)(v13 + 160);
  v15 = *v14;
  v16 = v14[1];
  if ( v15 == v16 )
LABEL_403:
    BUG();
  while ( *(_UNKNOWN **)v15 != &unk_4F9E06C )
  {
    v15 += 16;
    if ( v16 == v15 )
      goto LABEL_403;
  }
  v17 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(*(_QWORD *)(v15 + 8), &unk_4F9E06C);
  v18 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 168) = v17 + 160;
  v19 = *v18;
  v20 = v18[1];
  if ( v19 == v20 )
LABEL_404:
    BUG();
  while ( *(_UNKNOWN **)v19 != &unk_4F96DB4 )
  {
    v19 += 16;
    if ( v20 == v19 )
      goto LABEL_404;
  }
  v21 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v19 + 8) + 104LL))(*(_QWORD *)(v19 + 8), &unk_4F96DB4);
  v239 = 0;
  v22 = *(_QWORD *)(v21 + 160);
  v240 = 0;
  v241 = 0;
  *(_QWORD *)(a1 + 176) = v22;
  v242 = 0;
  if ( (*(_BYTE *)(a2 + 18) & 8) != 0 )
  {
    v135 = sub_15E38F0(a2);
    v136 = sub_14DD7D0(v135);
    if ( v136 > 10 )
    {
      if ( v136 != 12 )
        goto LABEL_14;
    }
    else if ( v136 <= 6 )
    {
      goto LABEL_14;
    }
    sub_14DDFC0((__int64)&v256, a2);
    j___libc_free_0(v240);
    ++v256;
    ++v239;
    v240 = v257;
    v257 = 0;
    v241 = s;
    s = 0;
    v242 = *(_DWORD *)v259;
    *(_DWORD *)v259 = 0;
    j___libc_free_0(0);
  }
LABEL_14:
  v212 = 0;
  if ( !(*(_DWORD *)(*(_QWORD *)(a2 + 24) + 8LL) >> 8) )
    v212 = sub_15E3780(a2) ^ 1;
  v23 = *(_QWORD **)(a2 + 80);
  v251 = 0;
  v252 = v255;
  v253 = v255;
  *(_QWORD *)v254 = 4;
  *(_DWORD *)&v254[8] = 0;
  v256 = 0;
  v257 = v260;
  s = v260;
  *(_QWORD *)v259 = 4;
  *(_DWORD *)&v259[8] = 0;
  v216 = a2 + 72;
  if ( (_QWORD *)(a2 + 72) != v23 )
  {
    if ( !v23 )
      BUG();
    while ( 1 )
    {
      v24 = (_QWORD *)v23[3];
      if ( v24 != v23 + 2 )
        break;
      v23 = (_QWORD *)v23[1];
      if ( (_QWORD *)(a2 + 72) == v23 )
        goto LABEL_22;
      if ( !v23 )
        BUG();
    }
    while ( 2 )
    {
      if ( (_QWORD *)v216 != v23 )
      {
        for ( i = (_QWORD *)v24[1]; ; i = (_QWORD *)v23[3] )
        {
          v39 = v23 - 3;
          if ( !v23 )
            v39 = 0;
          if ( i != v39 + 5 )
            break;
          v23 = (_QWORD *)v23[1];
          if ( (_QWORD *)v216 == v23 )
            break;
          if ( !v23 )
            BUG();
        }
        v36 = *((_BYTE *)v24 - 8);
        jj = i;
        if ( v36 <= 0x17u )
          goto LABEL_50;
        if ( v36 != 78 )
        {
          if ( v36 != 29 )
          {
LABEL_47:
            v37 = v212;
            if ( v36 == 53 )
              v37 = 0;
            v212 = v37;
          }
          goto LABEL_50;
        }
        v40 = *(v24 - 6);
        if ( *(_BYTE *)(v40 + 16) )
          goto LABEL_50;
        v41 = v24 - 3;
        v233 = sub_1438F00(v40);
        switch ( v233 )
        {
          case 0:
            for ( j = v41[-3 * (*((_DWORD *)v24 - 1) & 0xFFFFFFF)];
                  ;
                  j = *(_QWORD *)(v109 - 24LL * (*(_DWORD *)(v109 + 20) & 0xFFFFFFF)) )
            {
              v108 = sub_1649C60(j);
              v107 = 23;
              v109 = v108;
              v110 = *(_BYTE *)(v108 + 16);
              if ( v110 > 0x17u )
              {
                if ( v110 == 78 )
                {
                  v107 = 21;
                  if ( !*(_BYTE *)(*(_QWORD *)(v109 - 24) + 16LL) )
                    v107 = sub_1438F00(*(_QWORD *)(v109 - 24));
                }
                else
                {
                  v107 = 2 * (v110 != 29) + 21;
                }
              }
              if ( !(unsigned __int8)sub_1439C90(v107) )
                break;
            }
            v56 = *(_BYTE *)(v109 + 16);
            if ( v56 <= 0x17u )
              goto LABEL_63;
            if ( v56 == 78 )
            {
              v57 = v109 | 4;
            }
            else
            {
              if ( v56 != 29 )
                goto LABEL_63;
              v57 = v109 & 0xFFFFFFFFFFFFFFFBLL;
            }
            v58 = v57 & 0xFFFFFFFFFFFFFFF8LL;
            if ( !v58 || *(_QWORD *)(v58 + 40) != v24[2] )
              goto LABEL_63;
            for ( k = *(_QWORD *)(v58 + 32); ; k = *(_QWORD *)(k + 8) )
            {
              if ( !k )
                BUG();
              v60 = *(_BYTE *)(k - 8);
              if ( v60 != 71 && (v60 != 56 || !(unsigned __int8)sub_15FA1F0(k - 24)) )
                break;
            }
            if ( v41 != (_QWORD *)(k - 24) )
              goto LABEL_63;
            *(_BYTE *)(a1 + 153) = 1;
            v61 = *(_QWORD **)(a1 + 304);
            if ( !v61 )
            {
              v236 = **(__int64 ***)(a1 + 248);
              v162 = (__int64 *)sub_1643330(v236);
              *(_QWORD *)&v246[0] = sub_1646BA0(v162, 0);
              v163 = sub_1644EA0(*(__int64 **)&v246[0], v246, 1, 0);
              v248.m128_u64[0] = 0;
              v248.m128_u64[0] = sub_1563AB0((__int64 *)&v248, v236, -1, 30);
              v61 = (_QWORD *)sub_1632080(
                                *(_QWORD *)(a1 + 248),
                                (__int64)"objc_retainAutoreleasedReturnValue",
                                34,
                                v163,
                                v248.m128_i64[0]);
              *(_QWORD *)(a1 + 304) = v61;
            }
            v62 = *(v24 - 6) == 0;
            v24[5] = *(_QWORD *)(*v61 + 24LL);
            if ( !v62 )
            {
              v63 = *(v24 - 5);
              v64 = *(v24 - 4) & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v64 = v63;
              if ( v63 )
                *(_QWORD *)(v63 + 16) = v64 | *(_QWORD *)(v63 + 16) & 3LL;
            }
            *(v24 - 6) = v61;
            v65 = v61[1];
            *(v24 - 5) = v65;
            if ( v65 )
              *(_QWORD *)(v65 + 16) = (unsigned __int64)(v24 - 5) | *(_QWORD *)(v65 + 16) & 3LL;
            *(v24 - 4) = *(v24 - 4) & 3LL | (unsigned __int64)(v61 + 1);
            v61[1] = v24 - 6;
LABEL_106:
            if ( !*(_QWORD *)(a1 + 336) )
              goto LABEL_63;
            v66 = v24[2];
            v67 = v24;
            if ( v24 == *(_QWORD **)(v66 + 48) )
              goto LABEL_214;
            while ( 1 )
            {
              v68 = *v67 & 0xFFFFFFFFFFFFFFF8LL;
              v67 = (_QWORD *)v68;
              if ( !v68 )
                BUG();
              v69 = *(_BYTE *)(v68 - 8);
              v70 = v68 - 24;
              if ( v69 != 71 )
              {
                if ( v69 != 56 )
                  break;
                v235 = v66;
                v128 = sub_15FA1F0(v68 - 24);
                v66 = v235;
                v67 = (_QWORD *)v68;
                if ( !v128 )
                  break;
              }
              if ( *(_QWORD *)(v66 + 48) == v68 )
              {
LABEL_214:
                v129 = sub_157F0B0(v66);
                if ( !v129 )
                  goto LABEL_63;
                v70 = sub_157EBA0(v129);
                break;
              }
            }
            for ( m = v41[-3 * (*((_DWORD *)v24 - 1) & 0xFFFFFFF)];
                  ;
                  m = *(_QWORD *)(v74 - 24LL * (*(_DWORD *)(v74 + 20) & 0xFFFFFFF)) )
            {
              v73 = sub_1649C60(m);
              v72 = 23;
              v74 = v73;
              v75 = *(_BYTE *)(v73 + 16);
              if ( v75 > 0x17u )
              {
                if ( v75 == 78 )
                {
                  v72 = 21;
                  if ( !*(_BYTE *)(*(_QWORD *)(v74 - 24) + 16LL) )
                    v72 = sub_1438F00(*(_QWORD *)(v74 - 24));
                }
                else
                {
                  v72 = 2 * (v75 != 29) + 21;
                }
              }
              if ( !(unsigned __int8)sub_1439C90(v72) )
                break;
            }
            if ( v74 == v70 )
            {
              *(_BYTE *)(a1 + 153) = 1;
              v154 = sub_161E970(*(_QWORD *)(a1 + 336));
              v156 = v155;
              v157 = (_BYTE *)v154;
              v158 = (_QWORD *)sub_16498A0((__int64)(v24 - 3));
              v159 = (__int64 *)sub_1643270(v158);
              v160 = (__int64 **)sub_16453E0(v159, 0);
              v161 = sub_15EE570(v160, v157, v156, byte_3F871B3, 0, 1, 0, 0);
              v249.m128_i16[0] = 257;
              sub_18CA960(v161, 0, 0, (__int64)&v248, (__int64)(v24 - 3), (__int64)&v239);
            }
            goto LABEL_63;
          case 1:
          case 2:
            goto LABEL_106;
          case 4:
            v86 = v41[-3 * (*((_DWORD *)v24 - 1) & 0xFFFFFFF)];
            while ( 2 )
            {
              v88 = sub_1649C60(v86);
              v87 = 23;
              v89 = v88;
              v90 = *(_BYTE *)(v88 + 16);
              if ( v90 <= 0x17u )
                break;
              if ( v90 == 78 )
              {
                v87 = 21;
                if ( !*(_BYTE *)(*(_QWORD *)(v89 - 24) + 16LL) )
                {
                  v91 = sub_1438F00(*(_QWORD *)(v89 - 24));
                  if ( !(unsigned __int8)sub_1439C90(v91) )
                  {
LABEL_150:
                    if ( *(_BYTE *)(v89 + 16) != 54 )
                      goto LABEL_50;
                    if ( sub_15F32D0(v89) )
                      goto LABEL_50;
                    v226 = *(_BYTE *)(v89 + 18) & 1;
                    if ( v226 )
                      goto LABEL_50;
                    if ( v24[2] != *(_QWORD *)(v89 + 40) )
                      goto LABEL_50;
                    v214 = a1 + 176;
                    v92 = *(_QWORD **)(a1 + 160);
                    sub_141EB40(&v243, (__int64 *)v89);
                    v207 = v243.m128i_i64[0];
                    v208 = sub_1649C60(v243.m128i_i64[0]);
                    v209 = *(_QWORD *)(v89 + 40) + 40LL;
                    if ( v209 == *(_QWORD *)(v89 + 32) )
                      goto LABEL_50;
                    v234 = v23;
                    v93 = 0;
                    v94 = *(_QWORD *)(v89 + 32);
                    v217 = v41;
                    v95 = v92;
                    while ( 1 )
                    {
                      if ( !v94 )
                        BUG();
                      v96 = v94 - 24;
                      if ( v217 == (_QWORD *)(v94 - 24) )
                      {
                        v226 = 1;
                      }
                      else
                      {
                        v97 = *(_BYTE *)(v94 - 8);
                        v98 = 23;
                        if ( v97 > 0x17u )
                        {
                          if ( v97 == 78 )
                          {
                            v164 = *(_QWORD *)(v94 - 48);
                            v98 = 21;
                            if ( !*(_BYTE *)(v164 + 16) )
                              v98 = sub_1438F00(v164);
                          }
                          else
                          {
                            v98 = 2 * (v97 != 29) + 21;
                          }
                        }
                        v211 = v98;
                        if ( !sub_1439C70(v98) )
                        {
                          if ( v93 )
                          {
                            if ( (unsigned __int8)sub_18DC540(v94 - 24, v89, v214, v211) )
                            {
LABEL_286:
                              v23 = v234;
                              goto LABEL_50;
                            }
                          }
                          else
                          {
                            a3 = (__m128)_mm_loadu_si128(&v244);
                            v243.m128i_i64[0] = v207;
                            a4 = _mm_loadu_si128(&v243);
                            v249 = a3;
                            v250 = v245;
                            v247 = v245;
                            v248 = (__m128)a4;
                            v246[0] = a4;
                            v246[1] = a3;
                            switch ( *(_BYTE *)(v94 - 8) )
                            {
                              case 0x1D:
                                v199 = sub_134F0E0(v95, v96 & 0xFFFFFFFFFFFFFFFBLL, (__int64)v246) & 2;
                                goto LABEL_349;
                              case 0x21:
                                v199 = sub_134D290((__int64)v95, v94 - 24, v246) & 2;
                                goto LABEL_349;
                              case 0x36:
                                v199 = sub_134D040((__int64)v95, v94 - 24, v246, v99) & 2;
                                goto LABEL_349;
                              case 0x37:
                                v199 = sub_134D0E0((__int64)v95, v94 - 24, v246, v99) & 2;
                                goto LABEL_349;
                              case 0x39:
                                v199 = sub_134D190((__int64)v95, v94 - 24, v246) & 2;
                                goto LABEL_349;
                              case 0x3A:
                                v199 = sub_134D2D0((__int64)v95, v94 - 24, v246) & 2;
                                goto LABEL_349;
                              case 0x3B:
                                v199 = sub_134D360((__int64)v95, v94 - 24, v246) & 2;
                                goto LABEL_349;
                              case 0x4A:
                                v199 = sub_134D250((__int64)v95, v94 - 24, v246) & 2;
                                goto LABEL_349;
                              case 0x4E:
                                v199 = sub_134F0E0(v95, v96 | 4, (__int64)v246) & 2;
                                goto LABEL_349;
                              case 0x52:
                                v199 = sub_134D1D0((__int64)v95, v94 - 24, v246) & 2;
LABEL_349:
                                if ( !v199 )
                                  goto LABEL_285;
                                if ( *(_BYTE *)(v94 - 8) != 55
                                  || sub_15F32D0(v94 - 24)
                                  || (*(_BYTE *)(v94 - 6) & 1) != 0
                                  || v208 != sub_1649C60(*(_QWORD *)(v94 - 48)) )
                                {
                                  goto LABEL_286;
                                }
                                v93 = (_QWORD *)(v94 - 24);
                                break;
                              default:
LABEL_285:
                                v94 = *(_QWORD *)(v94 + 8);
                                if ( v209 == v94 )
                                  goto LABEL_286;
                                continue;
                            }
                          }
                        }
                      }
                      v94 = *(_QWORD *)(v94 + 8);
                      if ( v209 == v94 )
                        break;
                      if ( v93 && v226 )
                      {
                        v23 = v234;
                        v100 = v217;
                        goto LABEL_168;
                      }
                    }
                    v23 = v234;
                    v100 = v217;
                    if ( !v93 || !v226 )
                      goto LABEL_50;
LABEL_168:
                    for ( n = *(v93 - 6); ; n = *(_QWORD *)(v104 - 24LL * (*(_DWORD *)(v104 + 20) & 0xFFFFFFF)) )
                    {
                      v103 = sub_1649C60(n);
                      v102 = 23;
                      v104 = v103;
                      v105 = *(_BYTE *)(v103 + 16);
                      if ( v105 > 0x17u )
                      {
                        if ( v105 == 78 )
                        {
                          v102 = 21;
                          if ( !*(_BYTE *)(*(_QWORD *)(v104 - 24) + 16LL) )
                            v102 = sub_1438F00(*(_QWORD *)(v104 - 24));
                        }
                        else
                        {
                          v102 = 2 * (v105 != 29) + 21;
                        }
                      }
                      if ( !(unsigned __int8)sub_1439C90(v102) )
                        break;
                    }
                    v237 = v93 + 3;
                    v219 = *(_QWORD **)(v93[5] + 48LL);
                    if ( v93 + 3 == v219 )
                    {
                      v232 = v93;
                    }
                    else
                    {
                      v231 = v23;
                      v165 = v93 + 3;
                      v210 = v89;
                      while ( 1 )
                      {
                        v168 = v165 - 3;
                        if ( *((_BYTE *)v165 - 8) == 78 )
                        {
                          v169 = *(v165 - 6);
                          if ( !*(_BYTE *)(v169 + 16) && !(unsigned int)sub_1438F00(v169) )
                          {
                            v237 = v165;
                            v23 = v231;
                            v232 = v168;
                            v89 = v210;
                            goto LABEL_298;
                          }
                        }
                        v166 = sub_14399D0((__int64)(v165 - 3));
                        v167 = sub_18DC4F0(v165 - 3, v104, v214, v166);
                        if ( v100 != v168 && v167 )
                        {
                          v23 = v231;
                          goto LABEL_50;
                        }
                        v165 = (_QWORD *)(*v165 & 0xFFFFFFFFFFFFFFF8LL);
                        if ( v219 == v165 )
                          break;
                        if ( !v165 )
                          BUG();
                      }
                      v237 = v165;
                      v89 = v210;
                      v23 = v231;
                      if ( !v219 )
                        BUG();
                      v232 = v237 - 3;
                    }
LABEL_298:
                    if ( *((_BYTE *)v237 - 8) != 78 )
                      goto LABEL_50;
                    v170 = *(v237 - 6);
                    if ( *(_BYTE *)(v170 + 16) || (unsigned int)sub_1438F00(v170) )
                      goto LABEL_50;
                    v220 = v23;
                    for ( ii = v232[-3 * (*((_DWORD *)v237 - 1) & 0xFFFFFFF)];
                          ;
                          ii = *(_QWORD *)(v174 - 24LL * (*(_DWORD *)(v174 + 20) & 0xFFFFFFF)) )
                    {
                      v173 = sub_1649C60(ii);
                      v172 = 23;
                      v174 = v173;
                      v175 = *(_BYTE *)(v173 + 16);
                      if ( v175 > 0x17u )
                      {
                        if ( v175 == 78 )
                        {
                          v172 = 21;
                          if ( !*(_BYTE *)(*(_QWORD *)(v174 - 24) + 16LL) )
                            v172 = sub_1438F00(*(_QWORD *)(v174 - 24));
                        }
                        else
                        {
                          v172 = 2 * (v175 != 29) + 21;
                        }
                      }
                      if ( !(unsigned __int8)sub_1439C90(v172) )
                        break;
                    }
                    v176 = v174;
                    v23 = v220;
                    if ( v104 != v176 )
                      goto LABEL_50;
                    *(_BYTE *)(a1 + 153) = 1;
                    v177 = (_QWORD *)sub_16498A0((__int64)v100);
                    v178 = (__int64 *)sub_1643330(v177);
                    v221 = (__int64 *)sub_1646BA0(v178, 0);
                    v179 = sub_1646BA0(v221, 0);
                    v180 = *(_QWORD **)(v89 - 24);
                    *((_QWORD *)&v246[0] + 1) = v104;
                    *(_QWORD *)&v246[0] = v180;
                    if ( v179 != *v180 )
                    {
                      v215 = v179;
                      v249.m128_i16[0] = 257;
                      v181 = sub_1648A60(56, 1u);
                      v182 = v181;
                      if ( v181 )
                        sub_15FD590((__int64)v181, *(__int64 *)&v246[0], v215, (__int64)&v248, (__int64)v93);
                      *(_QWORD *)&v246[0] = v182;
                      v104 = *((_QWORD *)&v246[0] + 1);
                    }
                    if ( v221 != *(__int64 **)v104 )
                    {
                      v249.m128_i16[0] = 257;
                      v183 = sub_1648A60(56, 1u);
                      v184 = v183;
                      if ( v183 )
                        sub_15FD590(
                          (__int64)v183,
                          *((__int64 *)&v246[0] + 1),
                          (__int64)v221,
                          (__int64)&v248,
                          (__int64)v93);
                      *((_QWORD *)&v246[0] + 1) = v184;
                    }
                    v185 = *(_QWORD *)(a1 + 296);
                    if ( !v185 )
                    {
                      v200 = **(__int64 ***)(a1 + 248);
                      v201 = (__int64 *)sub_1643330(v200);
                      v223 = (__int64 *)sub_1646BA0(v201, 0);
                      v248.m128_u64[0] = sub_1646BA0(v223, 0);
                      v248.m128_u64[1] = (unsigned __int64)v223;
                      v243.m128i_i64[0] = 0;
                      v238 = sub_1563AB0(v243.m128i_i64, v200, -1, 30);
                      v238 = sub_1563AB0(&v238, v200, 1, 22);
                      v202 = (__int64 *)sub_1643270(v200);
                      v203 = sub_1644EA0(v202, &v248, 2, 0);
                      v204 = sub_1632080(*(_QWORD *)(a1 + 248), (__int64)"objc_storeStrong", 16, v203, v238);
                      *(_QWORD *)(a1 + 296) = v204;
                      v185 = v204;
                    }
                    v249.m128_i16[0] = 257;
                    v186 = sub_18CA960(v185, (__int64 *)v246, 2, (__int64)&v248, (__int64)v93, (__int64)&v239);
                    v248.m128_u64[0] = v186[7];
                    v187 = (__int64 *)sub_16498A0((__int64)v186);
                    v248.m128_u64[0] = sub_1563AB0((__int64 *)&v248, v187, -1, 30);
                    v186[7] = v248.m128_u64[0];
                    v248.m128_u64[0] = v93[6];
                    if ( v248.m128_u64[0] )
                    {
                      sub_1623A60((__int64)&v248, v248.m128_i64[0], 2);
                      v188 = (__int64)(v186 + 6);
                      if ( v186 + 6 == (_QWORD *)&v248 )
                      {
                        if ( v248.m128_u64[0] )
                          sub_161E7C0((__int64)&v248, v248.m128_i64[0]);
                        goto LABEL_324;
                      }
                      v197 = v186[6];
                      if ( !v197 )
                      {
LABEL_343:
                        v198 = (unsigned __int8 *)v248.m128_u64[0];
                        v186[6] = v248.m128_u64[0];
                        if ( v198 )
                          sub_1623210((__int64)&v248, v198, v188);
LABEL_324:
                        v189 = *(_QWORD **)(a1 + 352);
                        if ( *(_QWORD **)(a1 + 360) != v189 )
                        {
LABEL_325:
                          sub_16CCBA0(a1 + 344, (__int64)v186);
                          goto LABEL_326;
                        }
                        v194 = &v189[*(unsigned int *)(a1 + 372)];
                        v195 = *(_DWORD *)(a1 + 372);
                        if ( v189 == v194 )
                          goto LABEL_346;
                        v196 = 0;
                        do
                        {
                          if ( v186 == (_QWORD *)*v189 )
                            goto LABEL_326;
                          if ( *v189 == -2 )
                            v196 = v189;
                          ++v189;
                        }
                        while ( v194 != v189 );
                        if ( !v196 )
                        {
LABEL_346:
                          if ( v195 >= *(_DWORD *)(a1 + 368) )
                            goto LABEL_325;
                          *(_DWORD *)(a1 + 372) = v195 + 1;
                          *v194 = v186;
                          ++*(_QWORD *)(a1 + 344);
                        }
                        else
                        {
                          *v196 = v186;
                          --*(_DWORD *)(a1 + 376);
                          ++*(_QWORD *)(a1 + 344);
                        }
LABEL_326:
                        if ( jj )
                        {
                          v190 = jj - 3;
                          if ( jj - 3 != v232 )
                            goto LABEL_328;
                          for ( jj = (_QWORD *)jj[1]; ; jj = (_QWORD *)v23[3] )
                          {
                            v206 = v23 - 3;
                            if ( !v23 )
                              v206 = 0;
                            if ( jj != v206 + 5 )
                              break;
                            v23 = (_QWORD *)v23[1];
                            if ( (_QWORD *)v216 == v23 )
                            {
                              v190 = v206 + 2;
                              goto LABEL_328;
                            }
                            if ( !v23 )
                              BUG();
                          }
                          if ( jj )
                          {
                            v190 = jj - 3;
LABEL_328:
                            if ( v93 == v190 )
                            {
                              for ( jj = (_QWORD *)jj[1]; ; jj = (_QWORD *)v23[3] )
                              {
                                v205 = v23 - 3;
                                if ( !v23 )
                                  v205 = 0;
                                if ( jj != v205 + 5 )
                                  break;
                                v23 = (_QWORD *)v23[1];
                                if ( (_QWORD *)v216 == v23 )
                                  break;
                                if ( !v23 )
                                  BUG();
                              }
                            }
                          }
                        }
                        sub_15F20C0(v93);
                        sub_15F20C0(v100);
                        v193 = v232[-3 * (*((_DWORD *)v237 - 1) & 0xFFFFFFF)];
                        if ( *(v237 - 2) )
                        {
                          sub_164D160(
                            (__int64)v232,
                            v232[-3 * (*((_DWORD *)v237 - 1) & 0xFFFFFFF)],
                            a3,
                            *(double *)a4.m128i_i64,
                            a5,
                            a6,
                            v191,
                            v192,
                            a9,
                            a10);
                          sub_15F20C0(v232);
                        }
                        else
                        {
                          sub_15F20C0(v232);
                          sub_1AEB370(v193, 0);
                        }
                        if ( !*(_QWORD *)(v89 + 8) )
                          sub_15F20C0((_QWORD *)v89);
                        goto LABEL_50;
                      }
                    }
                    else
                    {
                      v188 = (__int64)(v186 + 6);
                      if ( v186 + 6 == (_QWORD *)&v248 )
                        goto LABEL_324;
                      v197 = v186[6];
                      if ( !v197 )
                        goto LABEL_324;
                    }
                    v222 = v188;
                    sub_161E7C0(v188, v197);
                    v188 = v222;
                    goto LABEL_343;
                  }
LABEL_145:
                  v86 = *(_QWORD *)(v89 - 24LL * (*(_DWORD *)(v89 + 20) & 0xFFFFFFF));
                  continue;
                }
              }
              else
              {
                v87 = 2 * (v90 != 29) + 21;
              }
              break;
            }
            if ( !(unsigned __int8)sub_1439C90(v87) )
              goto LABEL_150;
            goto LABEL_145;
          case 5:
          case 6:
            v76 = v41[-3 * (*((_DWORD *)v24 - 1) & 0xFFFFFFF)];
            while ( 2 )
            {
              v78 = sub_1649C60(v76);
              v77 = 23;
              v79 = v78;
              v80 = *(_BYTE *)(v78 + 16);
              if ( v80 <= 0x17u )
                goto LABEL_121;
              if ( v80 != 78 )
              {
                v77 = 2 * (v80 != 29) + 21;
LABEL_121:
                if ( (unsigned __int8)sub_1439C90(v77) )
                  goto LABEL_122;
                break;
              }
              v77 = 21;
              if ( *(_BYTE *)(*(_QWORD *)(v79 - 24) + 16LL) )
                goto LABEL_121;
              v81 = sub_1438F00(*(_QWORD *)(v79 - 24));
              if ( (unsigned __int8)sub_1439C90(v81) )
              {
LABEL_122:
                v76 = *(_QWORD *)(v79 - 24LL * (*(_DWORD *)(v79 + 20) & 0xFFFFFFF));
                continue;
              }
              break;
            }
            v82 = v24[2];
            if ( v233 == 6 )
              sub_18DCE30(4, v79, v82, (_DWORD)v24 - 24, (unsigned int)&v251, (unsigned int)&v256, a1 + 176);
            else
              sub_18DCE30(3, v79, v82, (_DWORD)v24 - 24, (unsigned int)&v251, (unsigned int)&v256, a1 + 176);
            ++v256;
            if ( s == v257 )
              goto LABEL_134;
            v83 = 4 * (*(_DWORD *)&v259[4] - *(_DWORD *)&v259[8]);
            if ( v83 < 0x20 )
              v83 = 32;
            if ( *(_DWORD *)v259 > v83 )
            {
              sub_16CC920((__int64)&v256);
            }
            else
            {
              memset(s, -1, 8LL * *(unsigned int *)v259);
LABEL_134:
              *(_QWORD *)&v259[4] = 0;
            }
            v84 = (char *)v253;
            if ( *(_DWORD *)&v254[4] - *(_DWORD *)&v254[8] != 1 )
            {
              ++v251;
              if ( v253 != v252 )
              {
                v85 = 4 * (*(_DWORD *)&v254[4] - *(_DWORD *)&v254[8]);
                if ( v85 < 0x20 )
                  v85 = 32;
                if ( *(_DWORD *)v254 > v85 )
                {
                  sub_16CC920((__int64)&v251);
                  goto LABEL_63;
                }
                memset(v253, -1, 8LL * *(unsigned int *)v254);
              }
              *(_QWORD *)&v254[4] = 0;
              goto LABEL_63;
            }
            v137 = (char *)v253 + 8 * *(unsigned int *)&v254[4];
            if ( v253 != v252 )
              v137 = (char *)v253 + 8 * *(unsigned int *)v254;
            v138 = *(_QWORD *)v253;
            if ( v137 != v253 )
            {
              v139 = v253;
              while ( 1 )
              {
                v138 = *v139;
                v140 = v139;
                if ( *v139 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                if ( v137 == (char *)++v139 )
                {
                  v138 = v140[1];
                  break;
                }
              }
            }
            if ( v138 && *(_BYTE *)(v138 + 16) != 78 )
              v138 = 0;
            ++v251;
            if ( v253 == v252 )
              goto LABEL_250;
            if ( *(_DWORD *)v254 > 0x20u )
            {
              sub_16CC920((__int64)&v251);
            }
            else
            {
              v141 = 8 * *(_DWORD *)v254;
              if ( (unsigned int)(8 * *(_DWORD *)v254) >= 8 )
              {
                *(_QWORD *)v253 = -1;
                *(_QWORD *)&v84[v141 - 8] = -1;
                memset(
                  (void *)((unsigned __int64)(v84 + 8) & 0xFFFFFFFFFFFFFFF8LL),
                  0xFFu,
                  8LL * ((v141 + (_DWORD)v84 - (((_DWORD)v84 + 8) & 0xFFFFFFF8)) >> 3));
              }
              else if ( v141 )
              {
                *(_BYTE *)v253 = -1;
              }
LABEL_250:
              *(_QWORD *)&v254[4] = 0;
            }
            if ( !v138 )
              goto LABEL_63;
            if ( *(_BYTE *)(v138 + 16) != 78 )
              goto LABEL_63;
            v142 = *(_QWORD *)(v138 - 24);
            if ( *(_BYTE *)(v142 + 16) || (unsigned int)sub_1438F00(v142) )
              goto LABEL_63;
            v230 = v23;
            v143 = *(_QWORD *)(v138 - 24LL * (*(_DWORD *)(v138 + 20) & 0xFFFFFFF));
            while ( 2 )
            {
              v145 = sub_1649C60(v143);
              v144 = 23;
              v146 = v145;
              v147 = *(_BYTE *)(v145 + 16);
              if ( v147 <= 0x17u )
                goto LABEL_257;
              if ( v147 != 78 )
              {
                v144 = 2 * (v147 != 29) + 21;
LABEL_257:
                if ( !(unsigned __int8)sub_1439C90(v144) )
                  goto LABEL_263;
LABEL_258:
                v143 = *(_QWORD *)(v146 - 24LL * (*(_DWORD *)(v146 + 20) & 0xFFFFFFF));
                continue;
              }
              break;
            }
            v144 = 21;
            if ( *(_BYTE *)(*(_QWORD *)(v146 - 24) + 16LL) )
              goto LABEL_257;
            v148 = sub_1438F00(*(_QWORD *)(v146 - 24));
            if ( (unsigned __int8)sub_1439C90(v148) )
              goto LABEL_258;
LABEL_263:
            v149 = v146;
            v23 = v230;
            if ( v79 == v149 )
            {
              *(_BYTE *)(a1 + 153) = 1;
              v150 = (_QWORD *)sub_18C9A50((__int64 *)(a1 + 248), (unsigned int)(v233 == 6) + 7);
              v62 = *(_QWORD *)(v138 - 24) == 0;
              *(_QWORD *)(v138 + 64) = *(_QWORD *)(*v150 + 24LL);
              if ( !v62 )
              {
                v151 = *(_QWORD *)(v138 - 16);
                v152 = *(_QWORD *)(v138 - 8) & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v152 = v151;
                if ( v151 )
                  *(_QWORD *)(v151 + 16) = v152 | *(_QWORD *)(v151 + 16) & 3LL;
              }
              *(_QWORD *)(v138 - 24) = v150;
              v153 = v150[1];
              *(_QWORD *)(v138 - 16) = v153;
              if ( v153 )
                *(_QWORD *)(v153 + 16) = (v138 - 16) | *(_QWORD *)(v153 + 16) & 3LL;
              *(_QWORD *)(v138 - 8) = *(_QWORD *)(v138 - 8) & 3LL | (unsigned __int64)(v150 + 1);
              v150[1] = v138 - 24;
              v112 = v41[-3 * (*((_DWORD *)v24 - 1) & 0xFFFFFFF)];
              if ( *(v24 - 2) )
              {
LABEL_189:
                sub_164D160((__int64)v41, v112, a3, *(double *)a4.m128i_i64, a5, a6, v115, v116, a9, a10);
                sub_15F20C0(v41);
              }
              else
              {
                sub_15F20C0(v24 - 3);
                sub_1AEB370(v112, 0);
              }
              goto LABEL_50;
            }
LABEL_63:
            *(_QWORD *)&v246[0] = v24 - 3;
            *((_QWORD *)&v246[0] + 1) = a1;
            v213 = v41[-3 * (*((_DWORD *)v24 - 1) & 0xFFFFFFF)];
            for ( kk = v213; ; kk = *v121 )
            {
              while ( 2 )
              {
                sub_18C95F0((__int64 *)v246, *(_QWORD *)(kk + 8), *(_BYTE *)(kk + 16));
                v43 = *(_BYTE *)(kk + 16);
                if ( v43 > 0x17u )
                {
                  if ( v43 == 71 )
                  {
                    kk = *(_QWORD *)(kk - 24);
                    continue;
                  }
                  if ( v43 != 56 )
                    goto LABEL_195;
                }
                else
                {
                  if ( v43 != 5 )
                    goto LABEL_190;
                  if ( *(_WORD *)(kk + 18) != 32 )
                    goto LABEL_67;
                }
                break;
              }
              v120 = 3LL * (*(_DWORD *)(kk + 20) & 0xFFFFFFF);
              if ( (*(_BYTE *)(kk + 23) & 0x40) != 0 )
              {
                v121 = *(__int64 **)(kk - 8);
                v122 = &v121[v120];
              }
              else
              {
                v122 = (__int64 *)kk;
                v121 = (__int64 *)(kk - v120 * 8);
              }
              v123 = v121 + 3;
              if ( v122 != v121 + 3 )
                break;
LABEL_210:
              ;
            }
            while ( 1 )
            {
              v124 = *v123;
              if ( *(_BYTE *)(*v123 + 16) != 13 )
                break;
              v125 = *(_DWORD *)(v124 + 32);
              if ( v125 <= 0x40 )
              {
                v127 = *(_QWORD *)(v124 + 24) == 0;
              }
              else
              {
                v218 = v121;
                v227 = v122;
                v126 = sub_16A57B0(v124 + 24);
                v122 = v227;
                v121 = v218;
                v127 = v125 == v126;
              }
              if ( !v127 )
                break;
              v123 += 3;
              if ( v122 == v123 )
                goto LABEL_210;
            }
LABEL_190:
            if ( v43 == 1 )
              __asm { jmp     rax }
LABEL_195:
            if ( v43 == 77 )
            {
              v248.m128_u64[0] = (unsigned __int64)&v249;
              v248.m128_u64[1] = 0x100000000LL;
              sub_18CA560(kk, (__int64)&v248);
              v117 = (__m128 *)(v248.m128_u64[0] + 8LL * v248.m128_u32[2]);
              if ( (__m128 *)v248.m128_u64[0] != v117 )
              {
                v118 = (__m128 *)v248.m128_u64[0];
                do
                {
                  v119 = v118->m128_u64[0];
                  v118 = (__m128 *)((char *)v118 + 8);
                  sub_18C95F0((__int64 *)v246, *(_QWORD *)(v119 + 8), *(_BYTE *)(v119 + 16));
                }
                while ( v117 != v118 );
                v117 = (__m128 *)v248.m128_u64[0];
              }
              if ( v117 != &v249 )
                _libc_free((unsigned __int64)v117);
            }
LABEL_67:
            v44 = 0;
            v248.m128_u64[0] = (unsigned __int64)&v249;
            v248.m128_u64[1] = 0x200000000LL;
            v45 = *(_QWORD *)(v213 + 8);
            if ( v45 )
            {
              do
              {
                while ( 1 )
                {
                  v46 = sub_1648700(v45);
                  if ( *((_BYTE *)v46 + 16) == 71 )
                    break;
                  v45 = *(_QWORD *)(v45 + 8);
                  if ( !v45 )
                    goto LABEL_74;
                }
                if ( v248.m128_i32[3] <= (unsigned int)v44 )
                {
                  v229 = v46;
                  sub_16CD150((__int64)&v248, &v249, 0, 8, v47, v48);
                  v44 = v248.m128_u32[2];
                  v46 = v229;
                }
                *(_QWORD *)(v248.m128_u64[0] + 8 * v44) = v46;
                v44 = (unsigned int)++v248.m128_i32[2];
                v45 = *(_QWORD *)(v45 + 8);
              }
              while ( v45 );
LABEL_74:
              if ( (_DWORD)v44 )
              {
                v49 = v44;
                do
                {
                  v50 = *(_QWORD *)(v248.m128_u64[0] + 8LL * v49 - 8);
                  v248.m128_i32[2] = v49 - 1;
                  v51 = *(_QWORD *)(v50 + 8);
                  if ( v51 )
                  {
                    do
                    {
                      while ( 1 )
                      {
                        v52 = sub_1648700(v51);
                        if ( *((_BYTE *)v52 + 16) == 71 )
                          break;
                        v51 = *(_QWORD *)(v51 + 8);
                        if ( !v51 )
                          goto LABEL_83;
                      }
                      v55 = v248.m128_u32[2];
                      if ( v248.m128_i32[2] >= (unsigned __int32)v248.m128_i32[3] )
                      {
                        v228 = v52;
                        sub_16CD150((__int64)&v248, &v249, 0, 8, v53, v54);
                        v55 = v248.m128_u32[2];
                        v52 = v228;
                      }
                      *(_QWORD *)(v248.m128_u64[0] + 8 * v55) = v52;
                      ++v248.m128_i32[2];
                      v51 = *(_QWORD *)(v51 + 8);
                    }
                    while ( v51 );
LABEL_83:
                    v51 = *(_QWORD *)(v50 + 8);
                  }
                  sub_18C95F0((__int64 *)v246, v51, *(_BYTE *)(v50 + 16));
                  v49 = v248.m128_u32[2];
                }
                while ( v248.m128_i32[2] );
              }
              if ( (__m128 *)v248.m128_u64[0] != &v249 )
                _libc_free(v248.m128_u64[0]);
            }
LABEL_50:
            v24 = jj;
            continue;
          case 10:
          case 11:
            goto LABEL_63;
          case 14:
            v111 = *(_BYTE *)(v41[3 * (1LL - (*((_DWORD *)v24 - 1) & 0xFFFFFFF))] + 16LL);
            if ( v111 != 9 && v111 != 15 )
              goto LABEL_50;
            v112 = sub_1599A20((__int64 **)*(v24 - 3));
            *(_BYTE *)(a1 + 153) = 1;
            v113 = v41[-3 * (*((_DWORD *)v24 - 1) & 0xFFFFFFF)];
            v114 = sub_1648A60(64, 2u);
            if ( v114 )
              sub_15F9660((__int64)v114, v112, v113, (__int64)v41);
            goto LABEL_189;
          case 20:
            sub_15F20C0(v24 - 3);
            goto LABEL_50;
          case 23:
            v36 = *((_BYTE *)v24 - 8);
            goto LABEL_47;
          default:
            goto LABEL_50;
        }
      }
      break;
    }
  }
LABEL_22:
  v25 = *(char **)(a1 + 360);
  v26 = *(char **)(a1 + 352);
  if ( v212 )
  {
    v130 = (__int64)(v26 == v25 ? &v26[8 * *(unsigned int *)(a1 + 372)] : &v25[8 * *(unsigned int *)(a1 + 368)]);
    if ( (char *)v130 != v25 )
    {
      v131 = *(_QWORD **)(a1 + 360);
      while ( 1 )
      {
        v132 = *v131;
        v133 = v131;
        if ( *v131 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( (_QWORD *)v130 == ++v131 )
          goto LABEL_23;
      }
      if ( v131 != (_QWORD *)v130 )
      {
        do
        {
          *(_WORD *)(v132 + 18) = *(_WORD *)(v132 + 18) & 0xFFFC | 1;
          v134 = v133 + 1;
          if ( v133 + 1 == (_QWORD *)v130 )
            break;
          while ( 1 )
          {
            v132 = *v134;
            v133 = v134;
            if ( *v134 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( (_QWORD *)v130 == ++v134 )
              goto LABEL_227;
          }
        }
        while ( (_QWORD *)v130 != v134 );
LABEL_227:
        v25 = *(char **)(a1 + 360);
        v26 = *(char **)(a1 + 352);
      }
    }
  }
LABEL_23:
  ++*(_QWORD *)(a1 + 344);
  if ( v26 == v25 )
    goto LABEL_28;
  v27 = 4 * (*(_DWORD *)(a1 + 372) - *(_DWORD *)(a1 + 376));
  v28 = *(unsigned int *)(a1 + 368);
  if ( v27 < 0x20 )
    v27 = 32;
  if ( (unsigned int)v28 <= v27 )
  {
    memset(v25, -1, 8 * v28);
LABEL_28:
    *(_QWORD *)(a1 + 372) = 0;
    goto LABEL_29;
  }
  sub_16CC920(a1 + 344);
LABEL_29:
  v29 = *(unsigned __int8 *)(a1 + 153);
  if ( s != v257 )
    _libc_free((unsigned __int64)s);
  if ( v253 != v252 )
    _libc_free((unsigned __int64)v253);
  if ( v242 )
  {
    v30 = v240;
    v31 = &v240[2 * v242];
    do
    {
      if ( *v30 != -16 && *v30 != -8 )
      {
        v32 = v30[1];
        if ( (v32 & 4) != 0 )
        {
          v33 = (unsigned __int64 *)(v32 & 0xFFFFFFFFFFFFFFF8LL);
          v34 = v33;
          if ( v33 )
          {
            if ( (unsigned __int64 *)*v33 != v33 + 2 )
              _libc_free(*v33);
            j_j___libc_free_0(v34, 48);
          }
        }
      }
      v30 += 2;
    }
    while ( v31 != v30 );
  }
  j___libc_free_0(v240);
  return v29;
}
