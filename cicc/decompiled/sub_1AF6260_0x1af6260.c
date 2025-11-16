// Function: sub_1AF6260
// Address: 0x1af6260
//
__int64 __fastcall sub_1AF6260(
        __int64 a1,
        _QWORD *a2,
        char a3,
        __m128 a4,
        __m128i a5,
        __m128i a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 *v11; // rdx
  __int64 v12; // r15
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rbx
  unsigned __int64 v16; // r14
  unsigned int v17; // ebx
  __int64 v18; // rax
  unsigned __int64 v19; // rbx
  unsigned __int64 v20; // rdi
  int v21; // r15d
  unsigned __int64 v22; // r14
  unsigned int v23; // ebx
  __int64 v24; // rax
  unsigned int v25; // r14d
  __int64 v26; // r15
  unsigned __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r14
  __int64 v31; // rax
  __int64 v32; // rsi
  __int64 *v33; // rdx
  __int64 v35; // rdi
  __int64 v36; // r15
  __int64 v37; // rbx
  double v38; // xmm4_8
  double v39; // xmm5_8
  __int64 v40; // rcx
  _QWORD *v41; // rax
  _QWORD *v42; // rdx
  char v43; // cl
  __int64 v44; // r15
  __int64 v45; // r10
  __int64 v46; // rax
  char v47; // di
  unsigned int v48; // esi
  __int64 v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rcx
  __int64 v52; // rdx
  __int64 v53; // r13
  _QWORD *v54; // rdi
  __int64 v55; // rax
  __int64 v56; // r13
  unsigned __int64 v57; // rax
  char *v58; // rax
  _QWORD *j; // r15
  _QWORD *v60; // r12
  __int64 v61; // rbx
  __int64 v62; // r12
  __int64 v63; // r14
  __int64 v64; // r12
  __int64 v65; // r8
  __int64 v66; // rax
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // rsi
  __int64 v70; // rax
  int v71; // edi
  __int64 v72; // r9
  int v73; // edi
  unsigned int v74; // r8d
  __int64 *v75; // rax
  __int64 v76; // r11
  _QWORD *v77; // rcx
  unsigned int v78; // r8d
  __int64 *v79; // rax
  __int64 v80; // r11
  _QWORD *v81; // rax
  __int128 *v82; // rsi
  _QWORD *v83; // rax
  __int64 v84; // rdx
  _QWORD *v85; // rdi
  __int64 v86; // rax
  __int64 v87; // rdx
  unsigned int v88; // r15d
  unsigned __int64 v89; // r12
  __int64 v90; // rcx
  __int64 v91; // r8
  __int64 k; // r13
  __int64 v93; // r10
  __int64 v94; // rax
  unsigned int v95; // r11d
  __int64 v96; // rsi
  char v97; // di
  int v98; // r9d
  __int64 v99; // rdx
  __int64 v100; // rax
  __int64 v101; // rdx
  __int64 v102; // rdx
  __int64 v103; // rax
  int v104; // ecx
  __int64 *v105; // rax
  __int64 v106; // rsi
  __int64 v107; // rcx
  __int64 v108; // rdx
  __int64 v109; // r10
  __int64 m; // rbx
  unsigned int v111; // ecx
  __int64 v112; // r9
  unsigned int v113; // esi
  __int64 v114; // rax
  __int64 v115; // rdx
  _QWORD *v116; // rdi
  __int64 v117; // rax
  __int64 v118; // rax
  __int64 v119; // r8
  __int64 v120; // rdx
  __int64 v121; // rcx
  __int64 v122; // rax
  __int64 v123; // rdx
  __int64 v124; // rcx
  double v125; // xmm4_8
  double v126; // xmm5_8
  int v127; // edx
  __int64 v128; // r12
  __int64 v129; // r15
  unsigned int v130; // ecx
  _QWORD *v131; // rdx
  __int64 v132; // r8
  __int64 v133; // r14
  const char *v134; // rax
  __int64 v135; // rdx
  __int64 *v136; // r12
  __int64 *v137; // rbx
  _QWORD *v138; // rax
  __int64 v139; // rax
  _QWORD *v140; // r12
  __int64 v141; // rax
  _QWORD *v142; // r14
  __int64 *v143; // rax
  __int64 v144; // rdx
  __int64 *v145; // rax
  __int64 v146; // rdi
  unsigned __int64 v147; // rsi
  __int64 v148; // rsi
  __int64 v149; // rbx
  __int64 v150; // r14
  __int64 v151; // rdi
  unsigned __int64 v152; // rax
  __int64 v153; // rax
  _QWORD *v154; // rcx
  __int64 v155; // rax
  _QWORD *v156; // rdx
  __int64 v157; // rdi
  _QWORD *v158; // r12
  __int64 v159; // rax
  unsigned int v160; // ebx
  bool v161; // al
  __int64 v162; // rdx
  unsigned int v163; // ebx
  unsigned __int64 v164; // r12
  double v165; // xmm4_8
  double v166; // xmm5_8
  __int64 v167; // r12
  __int64 v168; // rdx
  double v169; // xmm4_8
  double v170; // xmm5_8
  __int64 v171; // r15
  int v172; // r8d
  int v173; // r9d
  __int64 v174; // r12
  __int64 v175; // rbx
  signed __int64 v176; // rbx
  __int64 *v177; // r12
  _QWORD *v178; // rax
  __int64 *v179; // rdx
  int v180; // ecx
  __int64 *v181; // rbx
  __int64 v182; // rax
  int v183; // ecx
  __int64 v184; // r13
  __int64 v185; // rsi
  int v186; // ecx
  unsigned int v187; // edx
  __int64 *v188; // rax
  __int64 v189; // rdi
  __int64 v190; // rdi
  unsigned int v191; // r15d
  __int64 v192; // rax
  __int64 v193; // rdx
  unsigned __int64 v194; // r13
  double v195; // xmm4_8
  double v196; // xmm5_8
  _QWORD *v197; // rax
  __int64 v198; // r9
  _QWORD *v199; // rbx
  __int64 v200; // rax
  _QWORD *v201; // r13
  __int64 v202; // rdx
  __int64 v203; // rax
  __int64 v204; // rax
  __int64 v205; // rax
  __int64 *v206; // r8
  __int64 v207; // rdx
  __int64 v208; // rcx
  __int64 v209; // rax
  int v210; // eax
  int v211; // r8d
  unsigned __int64 v212; // rax
  _QWORD *v213; // rbx
  _QWORD *v214; // r12
  __int64 v215; // rsi
  int v216; // edx
  int v217; // r9d
  int v218; // eax
  int v219; // r10d
  int v220; // eax
  int v221; // r10d
  _QWORD *v222; // rax
  _QWORD *v223; // rbx
  __int64 v224; // rsi
  __int128 *v225; // r13
  __int64 v226; // rax
  __int64 v227; // rsi
  unsigned __int8 *v228; // rsi
  __int64 v229; // [rsp+18h] [rbp-298h]
  __int64 v230; // [rsp+20h] [rbp-290h]
  unsigned __int8 v231; // [rsp+28h] [rbp-288h]
  __int64 v232; // [rsp+30h] [rbp-280h]
  _QWORD *v233; // [rsp+38h] [rbp-278h]
  _QWORD *v234; // [rsp+38h] [rbp-278h]
  __int64 v235; // [rsp+40h] [rbp-270h]
  __int64 v236; // [rsp+40h] [rbp-270h]
  __int64 v237; // [rsp+40h] [rbp-270h]
  __int64 v238; // [rsp+40h] [rbp-270h]
  unsigned __int8 v239; // [rsp+4Bh] [rbp-265h]
  unsigned __int64 v240; // [rsp+50h] [rbp-260h]
  __int64 v241; // [rsp+50h] [rbp-260h]
  int v242; // [rsp+58h] [rbp-258h]
  __int64 v243; // [rsp+58h] [rbp-258h]
  __int64 v244; // [rsp+58h] [rbp-258h]
  __int64 v245; // [rsp+58h] [rbp-258h]
  __int64 v246; // [rsp+58h] [rbp-258h]
  __int64 v247; // [rsp+58h] [rbp-258h]
  __int64 v248; // [rsp+60h] [rbp-250h]
  __int64 v249; // [rsp+60h] [rbp-250h]
  int v250; // [rsp+60h] [rbp-250h]
  _QWORD *v251; // [rsp+60h] [rbp-250h]
  __int64 v253; // [rsp+68h] [rbp-248h]
  __int64 v254; // [rsp+68h] [rbp-248h]
  __int64 *v255; // [rsp+68h] [rbp-248h]
  __int64 v256; // [rsp+70h] [rbp-240h]
  __int128 v258; // [rsp+80h] [rbp-230h] BYREF
  _QWORD v259[2]; // [rsp+90h] [rbp-220h] BYREF
  const char *v260; // [rsp+A0h] [rbp-210h] BYREF
  __int64 v261; // [rsp+A8h] [rbp-208h]
  _QWORD v262[2]; // [rsp+B0h] [rbp-200h] BYREF
  __int128 v263; // [rsp+C0h] [rbp-1F0h] BYREF
  __int64 v264; // [rsp+D0h] [rbp-1E0h] BYREF
  unsigned __int64 v265; // [rsp+D8h] [rbp-1D8h]
  __int64 v266; // [rsp+E0h] [rbp-1D0h]
  unsigned __int64 v267; // [rsp+E8h] [rbp-1C8h]
  __int64 v268; // [rsp+F0h] [rbp-1C0h]
  __int64 v269; // [rsp+F8h] [rbp-1B8h]
  __int64 v270; // [rsp+100h] [rbp-1B0h] BYREF
  _QWORD *v271; // [rsp+108h] [rbp-1A8h]
  __int64 v272; // [rsp+110h] [rbp-1A0h]
  __int64 v273; // [rsp+118h] [rbp-198h]
  __int64 v274; // [rsp+120h] [rbp-190h]
  _QWORD *v275; // [rsp+128h] [rbp-188h]
  __int64 v276; // [rsp+130h] [rbp-180h]
  __int64 v277; // [rsp+138h] [rbp-178h]
  __int64 v278; // [rsp+140h] [rbp-170h]
  int v279; // [rsp+148h] [rbp-168h]
  void *v280; // [rsp+150h] [rbp-160h] BYREF
  __int64 v281; // [rsp+158h] [rbp-158h] BYREF
  _BYTE *v282; // [rsp+160h] [rbp-150h] BYREF
  __int64 v283; // [rsp+168h] [rbp-148h]
  __int64 i; // [rsp+170h] [rbp-140h]
  _BYTE v285[168]; // [rsp+178h] [rbp-138h] BYREF
  char v286; // [rsp+220h] [rbp-90h] BYREF

  v11 = (__int64 *)a2[4];
  if ( a2[5] - (_QWORD)v11 == 8 )
    return 0;
  v12 = *v11;
  v256 = *v11;
  v248 = sub_13FCB50((__int64)a2);
  v14 = sub_157EBA0(v12);
  v15 = v14;
  if ( *(_BYTE *)(v14 + 16) != 26 )
    return 0;
  if ( (*(_DWORD *)(v14 + 20) & 0xFFFFFFF) == 1 )
    return 0;
  v242 = sub_15F4D60(v14);
  v16 = sub_157EBA0(v12);
  if ( !v242 )
    return 0;
  v240 = v15;
  v17 = 0;
  while ( 1 )
  {
    v18 = sub_15F4DF0(v16, v17);
    if ( !sub_1377F70((__int64)(a2 + 7), v18) )
      break;
    if ( v242 == ++v17 )
      return 0;
  }
  v19 = v240;
  if ( !v248 )
    return 0;
  v20 = sub_157EBA0(v248);
  if ( v20 )
  {
    v21 = sub_15F4D60(v20);
    v22 = sub_157EBA0(v248);
    if ( v21 )
    {
      v23 = 0;
      while ( 1 )
      {
        v24 = sub_15F4DF0(v22, v23);
        if ( !sub_1377F70((__int64)(a2 + 7), v24) )
          break;
        if ( v21 == ++v23 )
        {
          v19 = v240;
          goto LABEL_29;
        }
      }
      v19 = v240;
      if ( !a3 )
      {
        v25 = *(unsigned __int8 *)(a1 + 57);
        if ( !(_BYTE)v25 )
        {
          v26 = *(_QWORD *)a2[4];
          v27 = sub_157EBA0(v26);
          v249 = sub_15F4DF0(v27, 0);
          if ( sub_1377F70((__int64)(a2 + 7), v249) )
          {
            v212 = sub_157EBA0(v26);
            v249 = sub_15F4DF0(v212, 1u);
          }
          v28 = sub_157F280(v26);
          v243 = v29;
          v253 = v28;
          if ( v28 == v29 )
            return v25;
          while ( 1 )
          {
            v30 = *(_QWORD *)(v253 + 8);
            if ( !v30 )
              break;
            while ( v249 == sub_1648700(v30)[5] )
            {
              v30 = *(_QWORD *)(v30 + 8);
              if ( !v30 )
                goto LABEL_29;
            }
            v31 = *(_QWORD *)(v253 + 32);
            if ( !v31 )
              BUG();
            v253 = 0;
            if ( *(_BYTE *)(v31 - 8) == 77 )
              v253 = v31 - 24;
            if ( v243 == v253 )
              return 0;
          }
        }
      }
    }
  }
LABEL_29:
  v281 = (__int64)v285;
  v32 = *(_QWORD *)(a1 + 24);
  v282 = v285;
  v280 = 0;
  v283 = 32;
  LODWORD(i) = 0;
  sub_14D04F0((__int64)a2, v32, (__int64)&v280);
  v33 = *(__int64 **)(a1 + 16);
  LODWORD(v270) = 0;
  BYTE4(v270) = 0;
  v271 = 0;
  v272 = 0;
  v273 = 0;
  v274 = 0;
  LODWORD(v275) = 0;
  v276 = 0;
  v277 = 0;
  v278 = 0;
  v279 = 0;
  sub_14D0990((__int64)&v270, v256, v33, (__int64)&v280);
  if ( WORD1(v270) || (unsigned int)v271 > *(_DWORD *)a1 )
  {
    j___libc_free_0(v273);
    if ( (_BYTE *)v281 != v282 )
      _libc_free((unsigned __int64)v282);
    return 0;
  }
  j___libc_free_0(v273);
  if ( (_BYTE *)v281 != v282 )
    _libc_free((unsigned __int64)v282);
  v254 = sub_13FC520((__int64)a2);
  if ( !v254 )
    return 0;
  v25 = sub_13FC370((__int64)a2);
  if ( !(_BYTE)v25 )
    return 0;
  v35 = *(_QWORD *)(a1 + 40);
  if ( v35 )
    sub_1465DB0(v35, a2);
  v36 = *(_QWORD *)(v19 - 48);
  v230 = *(_QWORD *)(v19 - 24);
  v37 = v230;
  v232 = v36;
  if ( sub_1377F70((__int64)(a2 + 7), v230) )
  {
    v230 = v36;
    v232 = v37;
  }
  sub_1AA62D0(v232, 0, a4, *(double *)a5.m128i_i64, *(double *)a6.m128i_i64, a7, v38, v39, a10, a11);
  v40 = *(_QWORD *)(v256 + 48);
  v270 = 0;
  v241 = v256 + 40;
  v235 = v40;
  LODWORD(v273) = 128;
  v41 = (_QWORD *)sub_22077B0(0x2000);
  v272 = 0;
  v271 = v41;
  v281 = 2;
  v42 = &v41[8 * (unsigned __int64)(unsigned int)v273];
  v280 = &unk_49E6B50;
  v282 = 0;
  v283 = -8;
  for ( i = 0; v42 != v41; v41 += 8 )
  {
    if ( v41 )
    {
      v43 = v281;
      v41[2] = 0;
      v41[3] = -8;
      *v41 = &unk_49E6B50;
      v41[1] = v43 & 6;
      v41[4] = i;
    }
  }
  LOBYTE(v278) = 0;
  BYTE1(v279) = 1;
  v44 = v235;
  v233 = a2;
  while ( 1 )
  {
    if ( !v44 )
      BUG();
    if ( *(_BYTE *)(v44 - 8) != 77 )
      break;
    v45 = v44 - 24;
    v46 = 0x17FFFFFFE8LL;
    v47 = *(_BYTE *)(v44 - 1) & 0x40;
    v48 = *(_DWORD *)(v44 - 4) & 0xFFFFFFF;
    if ( v48 )
    {
      v49 = 24LL * *(unsigned int *)(v44 + 32) + 8;
      v50 = 0;
      do
      {
        v51 = v45 - 24LL * v48;
        if ( v47 )
          v51 = *(_QWORD *)(v44 - 32);
        if ( v254 == *(_QWORD *)(v51 + v49) )
        {
          v46 = 24 * v50;
          goto LABEL_57;
        }
        ++v50;
        v49 += 8;
      }
      while ( v48 != (_DWORD)v50 );
      v46 = 0x17FFFFFFE8LL;
    }
LABEL_57:
    if ( v47 )
      v52 = *(_QWORD *)(v44 - 32);
    else
      v52 = v45 - 24LL * v48;
    v53 = *(_QWORD *)(v52 + v46);
    v54 = sub_1AF5F00((__int64)&v270, v44 - 24);
    v55 = v54[2];
    if ( v55 != v53 )
    {
      if ( v55 != 0 && v55 != -8 && v55 != -16 )
        sub_1649B30(v54);
      v54[2] = v53;
      if ( v53 != 0 && v53 != -8 && v53 != -16 )
        sub_164C220((__int64)v54);
    }
    v44 = *(_QWORD *)(v44 + 8);
  }
  v236 = v44;
  v56 = (__int64)v233;
  v57 = sub_157EBA0(v254);
  v280 = 0;
  v281 = 1;
  v234 = (_QWORD *)v57;
  v58 = (char *)&v282;
  do
  {
    *(_QWORD *)v58 = -8;
    v58 += 24;
    *((_QWORD *)v58 - 2) = -8;
    *((_QWORD *)v58 - 1) = -8;
  }
  while ( v58 != &v286 );
  for ( j = (_QWORD *)(*(_QWORD *)(*(_QWORD *)(v254 + 40) & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL);
        (_QWORD *)(v254 + 40) != j;
        j = (_QWORD *)(*j & 0xFFFFFFFFFFFFFFF8LL) )
  {
    if ( !j )
      BUG();
    v60 = j - 3;
    if ( *((_BYTE *)j - 8) != 78 )
      break;
    v205 = *(j - 6);
    if ( *(_BYTE *)(v205 + 16) )
      break;
    if ( (*(_BYTE *)(v205 + 33) & 0x20) == 0 )
      break;
    if ( (unsigned int)(*(_DWORD *)(v205 + 36) - 35) > 3 )
      break;
    v206 = (__int64 *)sub_1601A30((__int64)(j - 3), 1);
    v207 = *((_DWORD *)j - 1) & 0xFFFFFFF;
    v208 = *(_QWORD *)(v60[3 * (1 - v207)] + 24LL);
    v209 = *(_QWORD *)(v60[3 * (2 - v207)] + 24LL);
    v260 = (const char *)v206;
    v261 = v208;
    v262[0] = v209;
    sub_1AF5D70((__int64)&v263, (__int64)&v280, (__int64 *)&v260);
  }
  if ( v236 == v241 )
    goto LABEL_98;
  v231 = v25;
  v61 = v236;
  do
  {
    v62 = v61;
    v61 = *(_QWORD *)(v61 + 8);
    v63 = v62 - 24;
    if ( sub_13FC1D0(v56, v62 - 24) )
    {
      if ( !(unsigned __int8)sub_15F2ED0(v62 - 24) && !(unsigned __int8)sub_15F3040(v62 - 24) )
      {
        v127 = *(unsigned __int8 *)(v62 - 8);
        if ( (unsigned int)(v127 - 25) > 9 )
        {
          if ( (_BYTE)v127 == 78 )
          {
            v226 = *(_QWORD *)(v62 - 48);
            if ( *(_BYTE *)(v226 + 16)
              || (*(_BYTE *)(v226 + 33) & 0x20) == 0
              || (unsigned int)(*(_DWORD *)(v226 + 36) - 35) > 3 )
            {
LABEL_158:
              sub_15F22F0((_QWORD *)(v62 - 24), (__int64)v234);
              continue;
            }
          }
          else if ( (_BYTE)v127 != 53 )
          {
            goto LABEL_158;
          }
        }
      }
    }
    v64 = sub_15F4880(v62 - 24);
    sub_1B75040(&v263, &v270, 3, 0, 0);
    sub_1B79630(&v263, v64);
    sub_1B75110(&v263);
    if ( *(_BYTE *)(v64 + 16) == 78 )
    {
      v118 = *(_QWORD *)(v64 - 24);
      if ( !*(_BYTE *)(v118 + 16)
        && (*(_BYTE *)(v118 + 33) & 0x20) != 0
        && (unsigned int)(*(_DWORD *)(v118 + 36) - 35) <= 3 )
      {
        v82 = &v263;
        v119 = sub_1601A30(v64, 1);
        v120 = *(_DWORD *)(v64 + 20) & 0xFFFFFFF;
        v121 = *(_QWORD *)(*(_QWORD *)(v64 + 24 * (1 - v120)) + 24LL);
        v122 = *(_QWORD *)(*(_QWORD *)(v64 + 24 * (2 - v120)) + 24LL);
        *(_QWORD *)&v263 = v119;
        *((_QWORD *)&v263 + 1) = v121;
        v264 = v122;
        if ( (unsigned __int8)sub_1AF57C0((__int64)&v280, (__int64 *)&v263, (__int64 **)&v260) )
          goto LABEL_153;
      }
    }
    v66 = sub_13E3350(v64, *(const __m128i **)(a1 + 48), 0, 1, v65);
    v67 = v66;
    if ( v66 )
    {
      if ( *(_BYTE *)(v66 + 16) <= 0x17u )
        goto LABEL_87;
      v68 = *(_QWORD *)(v66 + 40);
      v69 = *(_QWORD *)(v64 + 40);
      if ( v68 == v69 )
        goto LABEL_87;
      v70 = *(_QWORD *)(a1 + 8);
      v71 = *(_DWORD *)(v70 + 24);
      if ( !v71 )
        goto LABEL_87;
      v72 = *(_QWORD *)(v70 + 8);
      v73 = v71 - 1;
      v74 = v73 & (((unsigned int)v68 >> 9) ^ ((unsigned int)v68 >> 4));
      v75 = (__int64 *)(v72 + 16LL * v74);
      v76 = *v75;
      if ( v68 != *v75 )
      {
        v220 = 1;
        while ( v76 != -8 )
        {
          v221 = v220 + 1;
          v74 = v73 & (v220 + v74);
          v75 = (__int64 *)(v72 + 16LL * v74);
          v76 = *v75;
          if ( v68 == *v75 )
            goto LABEL_82;
          v220 = v221;
        }
LABEL_87:
        v82 = (__int128 *)v63;
        v244 = v67;
        v83 = sub_1AF5F00((__int64)&v270, v63);
        v84 = v244;
        v85 = v83;
        v86 = v83[2];
        if ( v244 != v86 )
        {
          LOBYTE(v82) = v86 != -8;
          if ( ((v86 != 0) & (unsigned __int8)v82) != 0 && v86 != -16 )
          {
            sub_1649B30(v85);
            v84 = v244;
          }
          v85[2] = v84;
          if ( v84 != -16 && v84 != -8 )
            sub_164C220((__int64)v85);
        }
        if ( !(unsigned __int8)sub_15F3040(v64) && !sub_15F3330(v64) )
        {
LABEL_153:
          sub_164BEC0(
            v64,
            (__int64)v82,
            v123,
            v124,
            a4,
            *(double *)a5.m128i_i64,
            *(double *)a6.m128i_i64,
            a7,
            v125,
            v126,
            a10,
            a11);
          continue;
        }
        goto LABEL_95;
      }
LABEL_82:
      v77 = (_QWORD *)v75[1];
      if ( !v77 )
        goto LABEL_87;
      v78 = v73 & (((unsigned int)v69 >> 9) ^ ((unsigned int)v69 >> 4));
      v79 = (__int64 *)(v72 + 16LL * v78);
      v80 = *v79;
      if ( v69 == *v79 )
      {
LABEL_84:
        v81 = (_QWORD *)v79[1];
        if ( v77 == v81 )
          goto LABEL_87;
        while ( v81 )
        {
          v81 = (_QWORD *)*v81;
          if ( v77 == v81 )
            goto LABEL_87;
        }
      }
      else
      {
        v218 = 1;
        while ( v80 != -8 )
        {
          v219 = v218 + 1;
          v78 = v73 & (v218 + v78);
          v79 = (__int64 *)(v72 + 16LL * v78);
          v80 = *v79;
          if ( v69 == *v79 )
            goto LABEL_84;
          v218 = v219;
        }
      }
    }
    v116 = sub_1AF5F00((__int64)&v270, v63);
    v117 = v116[2];
    if ( v64 != v117 )
    {
      if ( v117 != 0 && v117 != -8 && v117 != -16 )
        sub_1649B30(v116);
      v116[2] = v64;
      if ( v64 != -16 && v64 != -8 )
        sub_164C220((__int64)v116);
    }
LABEL_95:
    v260 = sub_1649960(v63);
    LOWORD(v264) = 261;
    v261 = v87;
    *(_QWORD *)&v263 = &v260;
    sub_164B780(v64, (__int64 *)&v263);
    sub_15F2120(v64, (__int64)v234);
    if ( *(_BYTE *)(v64 + 16) == 78 )
    {
      v204 = *(_QWORD *)(v64 - 24);
      if ( !*(_BYTE *)(v204 + 16) && (*(_BYTE *)(v204 + 33) & 0x20) != 0 && *(_DWORD *)(v204 + 36) == 4 )
        sub_14CE830(*(_QWORD *)(a1 + 24), v64);
    }
  }
  while ( v241 != v61 );
  v25 = v231;
LABEL_98:
  v88 = 0;
  v89 = sub_157EBA0(v256);
  v250 = sub_15F4D60(v89);
  if ( v250 )
  {
    v229 = v56;
    do
    {
      for ( k = *(_QWORD *)(sub_15F4DF0(v89, v88) + 48); ; k = *(_QWORD *)(k + 8) )
      {
        if ( !k )
          BUG();
        if ( *(_BYTE *)(k - 8) != 77 )
          break;
        v93 = k - 24;
        v94 = 0x17FFFFFFE8LL;
        v95 = *(_DWORD *)(k + 32);
        v96 = *(_DWORD *)(k - 4) & 0xFFFFFFF;
        v97 = *(_BYTE *)(k - 1) & 0x40;
        v98 = v96;
        if ( (_DWORD)v96 )
        {
          v91 = v93 - 24LL * (unsigned int)v96;
          v99 = 24LL * v95 + 8;
          v100 = 0;
          do
          {
            v90 = v93 - 24LL * (unsigned int)v96;
            if ( v97 )
              v90 = *(_QWORD *)(k - 32);
            if ( v256 == *(_QWORD *)(v90 + v99) )
            {
              v94 = 24 * v100;
              goto LABEL_110;
            }
            ++v100;
            v99 += 8;
          }
          while ( (_DWORD)v96 != (_DWORD)v100 );
          v94 = 0x17FFFFFFE8LL;
        }
LABEL_110:
        if ( v97 )
        {
          v101 = *(_QWORD *)(k - 32);
        }
        else
        {
          v90 = 24LL * (unsigned int)v96;
          v101 = v93 - v90;
        }
        v102 = *(_QWORD *)(v101 + v94);
        if ( (_DWORD)v96 == v95 )
        {
          v237 = v102;
          sub_15F55D0(k - 24, v96, v102, v90, v91, (unsigned int)v96);
          v102 = v237;
          v93 = k - 24;
          v98 = *(_DWORD *)(k - 4) & 0xFFFFFFF;
        }
        v103 = (v98 + 1) & 0xFFFFFFF;
        v104 = v103 | *(_DWORD *)(k - 4) & 0xF0000000;
        *(_DWORD *)(k - 4) = v104;
        if ( (v104 & 0x40000000) != 0 )
          v90 = *(_QWORD *)(k - 32);
        else
          v90 = v93 - 24 * v103;
        v105 = (__int64 *)(v90 + 24LL * (unsigned int)(v103 - 1));
        if ( *v105 )
        {
          v106 = v105[1];
          v90 = v105[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v90 = v106;
          if ( v106 )
          {
            v90 |= *(_QWORD *)(v106 + 16) & 3LL;
            *(_QWORD *)(v106 + 16) = v90;
          }
        }
        *v105 = v102;
        if ( v102 )
        {
          v107 = *(_QWORD *)(v102 + 8);
          v105[1] = v107;
          if ( v107 )
          {
            v91 = (__int64)(v105 + 1);
            *(_QWORD *)(v107 + 16) = (unsigned __int64)(v105 + 1) | *(_QWORD *)(v107 + 16) & 3LL;
          }
          v90 = (v102 + 8) | v105[2] & 3;
          v105[2] = v90;
          *(_QWORD *)(v102 + 8) = v105;
        }
        v108 = *(_DWORD *)(k - 4) & 0xFFFFFFF;
        if ( (*(_BYTE *)(k - 1) & 0x40) != 0 )
          v109 = *(_QWORD *)(k - 32);
        else
          v109 = v93 - 24 * v108;
        *(_QWORD *)(v109 + 8LL * (unsigned int)(v108 - 1) + 24LL * *(unsigned int *)(k + 32) + 8) = v254;
      }
      ++v88;
    }
    while ( v88 != v250 );
    v25 = (unsigned __int8)v25;
    v56 = v229;
  }
  sub_15F20C0(v234);
  v260 = (const char *)v262;
  v261 = 0x200000000LL;
  for ( m = *(_QWORD *)(v256 + 48); ; m = *(_QWORD *)(m + 8) )
  {
    if ( !m )
      BUG();
    if ( *(_BYTE *)(m - 8) != 77 )
      break;
    v111 = *(_DWORD *)(m - 4) & 0xFFFFFFF;
    if ( v111 )
    {
      v112 = m - 24;
      v113 = 0;
      v114 = 24LL * *(unsigned int *)(m + 32) + 8;
      while ( 1 )
      {
        v115 = m - 24 - 24LL * v111;
        if ( (*(_BYTE *)(m - 1) & 0x40) != 0 )
          v115 = *(_QWORD *)(m - 32);
        if ( v254 == *(_QWORD *)(v115 + v114) )
          break;
        ++v113;
        v114 += 8;
        if ( v111 == v113 )
        {
          v113 = -1;
          break;
        }
      }
    }
    else
    {
      v113 = -1;
      v112 = m - 24;
    }
    sub_15F5350(v112, v113, 1);
  }
  sub_1B3B830(&v263, &v260);
  if ( v241 == *(_QWORD *)(v256 + 48) )
    goto LABEL_208;
  v128 = *(_QWORD *)(v256 + 48);
  v239 = v25;
  while ( 2 )
  {
    if ( !v128 )
      BUG();
    if ( !*(_QWORD *)(v128 - 16) )
      goto LABEL_161;
    v129 = v128 - 24;
    if ( !(_DWORD)v273 )
      goto LABEL_206;
    v130 = (v273 - 1) & (((unsigned int)v129 >> 9) ^ ((unsigned int)v129 >> 4));
    v131 = &v271[8 * (unsigned __int64)v130];
    v132 = v131[3];
    if ( v129 != v132 )
    {
      v216 = 1;
      while ( v132 != -8 )
      {
        v217 = v216 + 1;
        v130 = (v273 - 1) & (v216 + v130);
        v131 = &v271[8 * (unsigned __int64)v130];
        v132 = v131[3];
        if ( v129 == v132 )
          goto LABEL_166;
        v216 = v217;
      }
      goto LABEL_206;
    }
LABEL_166:
    if ( v131 == &v271[8 * (unsigned __int64)(unsigned int)v273] )
    {
LABEL_206:
      v133 = 0;
      goto LABEL_172;
    }
    v258 = 6u;
    v133 = v131[7];
    v259[0] = v133;
    if ( v133 != 0 && v133 != -8 && v133 != -16 )
    {
      sub_1649AC0((unsigned __int64 *)&v258, v131[5] & 0xFFFFFFFFFFFFFFF8LL);
      v133 = v259[0];
      if ( v259[0] != -8 && v259[0] != 0 && v259[0] != -16 )
        sub_1649B30(&v258);
    }
LABEL_172:
    v134 = sub_1649960(v128 - 24);
    sub_1B3B8C0(&v263, *(_QWORD *)(v128 - 24), v134, v135);
    sub_1B3BE00(&v263, v256, v128 - 24);
    sub_1B3BE00(&v263, v254, v133);
    if ( !*(_QWORD *)(v128 - 16) )
      goto LABEL_181;
    v245 = v128;
    v136 = *(__int64 **)(v128 - 16);
    while ( 2 )
    {
      v137 = (__int64 *)v136[1];
      v138 = sub_1648700((__int64)v136);
      if ( *((_BYTE *)v138 + 16) == 77 )
      {
LABEL_178:
        sub_1B420C0(&v263, v136);
        goto LABEL_179;
      }
      v139 = v138[5];
      if ( v256 == v139 )
        goto LABEL_179;
      if ( v254 != v139 )
        goto LABEL_178;
      if ( !*v136 )
      {
        *v136 = v133;
        if ( !v133 )
          goto LABEL_179;
        goto LABEL_203;
      }
      v152 = v136[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v152 = v137;
      if ( v137 )
      {
        v137[2] = v137[2] & 3 | v152;
        *v136 = v133;
        if ( v133 )
          goto LABEL_203;
LABEL_174:
        v136 = v137;
        continue;
      }
      break;
    }
    *v136 = v133;
    if ( v133 )
    {
LABEL_203:
      v153 = *(_QWORD *)(v133 + 8);
      v136[1] = v153;
      if ( v153 )
        *(_QWORD *)(v153 + 16) = (unsigned __int64)(v136 + 1) | *(_QWORD *)(v153 + 16) & 3LL;
      v136[2] = (v133 + 8) | v136[2] & 3;
      *(_QWORD *)(v133 + 8) = v136;
LABEL_179:
      if ( !v137 )
        goto LABEL_180;
      goto LABEL_174;
    }
LABEL_180:
    v128 = v245;
LABEL_181:
    *(_QWORD *)&v258 = v259;
    *((_QWORD *)&v258 + 1) = 0x100000000LL;
    sub_1AEA1F0((__int64)&v258, v129);
    v251 = (_QWORD *)(v258 + 8LL * DWORD2(v258));
    if ( (_QWORD *)v258 != v251 )
    {
      v238 = v128;
      v140 = (_QWORD *)v258;
      v246 = v133;
      do
      {
        v149 = *v140;
        v150 = *(_QWORD *)(*v140 + 40LL);
        if ( v256 != v150 )
        {
          v151 = v246;
          if ( v254 != v150 )
          {
            if ( (unsigned __int8)sub_1B3BC80(&v263, *(_QWORD *)(*v140 + 40LL)) )
              v141 = sub_1B40B40(&v263);
            else
              v141 = sub_1599EF0(*(__int64 ***)(v238 - 24));
            v149 = *v140;
            v151 = v141;
          }
          v142 = sub_1624210(v151);
          v143 = (__int64 *)sub_16498A0(v129);
          v144 = sub_1628DA0(v143, (__int64)v142);
          v145 = (__int64 *)(v149 - 24LL * (*(_DWORD *)(v149 + 20) & 0xFFFFFFF));
          if ( *v145 )
          {
            v146 = v145[1];
            v147 = v145[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v147 = v146;
            if ( v146 )
              *(_QWORD *)(v146 + 16) = *(_QWORD *)(v146 + 16) & 3LL | v147;
          }
          *v145 = v144;
          if ( v144 )
          {
            v148 = *(_QWORD *)(v144 + 8);
            v145[1] = v148;
            if ( v148 )
              *(_QWORD *)(v148 + 16) = (unsigned __int64)(v145 + 1) | *(_QWORD *)(v148 + 16) & 3LL;
            v145[2] = (v144 + 8) | v145[2] & 3;
            *(_QWORD *)(v144 + 8) = v145;
          }
        }
        ++v140;
      }
      while ( v251 != v140 );
      v128 = v238;
      v251 = (_QWORD *)v258;
    }
    if ( v251 != v259 )
      _libc_free((unsigned __int64)v251);
LABEL_161:
    v128 = *(_QWORD *)(v128 + 8);
    if ( v241 != v128 )
      continue;
    break;
  }
  v25 = v239;
LABEL_208:
  sub_1B3B860(&v263);
  if ( (_DWORD)v261 )
    sub_1AF4F90(v256, (__int64)&v260);
  v154 = *(_QWORD **)(v56 + 32);
  LODWORD(v155) = 0;
  if ( v232 != *v154 )
  {
    do
    {
      v155 = (unsigned int)(v155 + 1);
      v156 = &v154[v155];
    }
    while ( v232 != *v156 );
    *v156 = *v154;
    **(_QWORD **)(v56 + 32) = v232;
  }
  v157 = *(_QWORD *)(a1 + 32);
  if ( v157 )
  {
    *(_QWORD *)&v263 = &v264;
    v264 = v254;
    v265 = v230 & 0xFFFFFFFFFFFFFFFBLL;
    v266 = v254;
    v268 = v254;
    v267 = v232 & 0xFFFFFFFFFFFFFFFBLL;
    v269 = v256 | 4;
    *((_QWORD *)&v263 + 1) = 0x300000003LL;
    sub_15DC140(v157, &v264, 3);
    if ( (__int64 *)v263 != &v264 )
      _libc_free(v263);
  }
  v158 = (_QWORD *)sub_157EBA0(v254);
  v159 = *(v158 - 9);
  if ( *(_BYTE *)(v159 + 16) == 13 )
  {
    v160 = *(_DWORD *)(v159 + 32);
    v161 = v160 <= 0x40 ? *(_QWORD *)(v159 + 24) == 0 : v160 == (unsigned int)sub_16A57B0(v159 + 24);
    if ( v158[-3 * v161 - 3] == v232 )
    {
      sub_157F2D0(v230, v254, 1);
      v222 = sub_1648A60(56, 1u);
      v223 = v222;
      if ( v222 )
        sub_15F8320((__int64)v222, v232, (__int64)v158);
      v224 = v158[6];
      v225 = (__int128 *)(v223 + 6);
      *(_QWORD *)&v263 = v224;
      if ( v224 )
      {
        sub_1623A60((__int64)&v263, v224, 2);
        if ( v225 == &v263 )
        {
          if ( (_QWORD)v263 )
            sub_161E7C0((__int64)&v263, v263);
          goto LABEL_333;
        }
        v227 = v223[6];
        if ( !v227 )
        {
LABEL_343:
          v228 = (unsigned __int8 *)v263;
          v223[6] = v263;
          if ( v228 )
            sub_1623210((__int64)&v263, v228, (__int64)(v223 + 6));
          goto LABEL_333;
        }
      }
      else if ( v225 == &v263 || (v227 = v223[6]) == 0 )
      {
LABEL_333:
        sub_15F20C0(v158);
        v198 = *(_QWORD *)(a1 + 32);
        if ( v198 )
        {
          sub_15D4360(*(_QWORD *)(a1 + 32), v254, v230);
          v198 = *(_QWORD *)(a1 + 32);
        }
        goto LABEL_255;
      }
      sub_161E7C0((__int64)(v223 + 6), v227);
      goto LABEL_343;
    }
  }
  LODWORD(v264) = (_DWORD)&loc_1010000;
  v162 = *(_QWORD *)(a1 + 32);
  *((_QWORD *)&v263 + 1) = *(_QWORD *)(a1 + 8);
  *(_QWORD *)&v263 = v162;
  v163 = 0;
  v164 = sub_157EBA0(v254);
  while ( v232 != sub_15F4DF0(v164, v163) )
    ++v163;
  v167 = sub_1AAC5F0(v164, v163, &v263, a4, a5, *(double *)a6.m128i_i64, a7, v165, v166, a10, a11);
  *(_QWORD *)&v258 = sub_1649960(v232);
  *((_QWORD *)&v258 + 1) = v168;
  *(_QWORD *)&v263 = &v258;
  *((_QWORD *)&v263 + 1) = ".lr.ph";
  LOWORD(v264) = 773;
  sub_164B780(v167, (__int64 *)&v263);
  v171 = *(_QWORD *)(v230 + 8);
  if ( v171 )
  {
    while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v171) + 16) - 25) > 9u )
    {
      v171 = *(_QWORD *)(v171 + 8);
      if ( !v171 )
        goto LABEL_254;
    }
    v174 = v171;
    v175 = 0;
    *(_QWORD *)&v263 = &v264;
    *((_QWORD *)&v263 + 1) = 0x400000000LL;
    while ( 1 )
    {
      v174 = *(_QWORD *)(v174 + 8);
      if ( !v174 )
        break;
      while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v174) + 16) - 25) <= 9u )
      {
        v174 = *(_QWORD *)(v174 + 8);
        ++v175;
        if ( !v174 )
          goto LABEL_229;
      }
    }
LABEL_229:
    v176 = v175 + 1;
    if ( v176 > 4 )
    {
      sub_16CD150((__int64)&v263, &v264, v176, 8, v172, v173);
      v177 = (__int64 *)(v263 + 8LL * DWORD2(v263));
    }
    else
    {
      v177 = &v264;
    }
    v178 = sub_1648700(v171);
LABEL_233:
    if ( v177 )
      *v177 = v178[5];
    v171 = *(_QWORD *)(v171 + 8);
    if ( v171 )
    {
      do
      {
        v178 = sub_1648700(v171);
        if ( (unsigned __int8)(*((_BYTE *)v178 + 16) - 25) <= 9u )
        {
          ++v177;
          goto LABEL_233;
        }
        v171 = *(_QWORD *)(v171 + 8);
      }
      while ( v171 );
      v179 = (__int64 *)v263;
      v180 = v176 + DWORD2(v263);
      v255 = (__int64 *)(v263 + 8LL * (unsigned int)(v176 + DWORD2(v263)));
    }
    else
    {
      v179 = (__int64 *)v263;
      v180 = DWORD2(v263) + v176;
      v255 = (__int64 *)(v263 + 8LL * (unsigned int)(DWORD2(v263) + v176));
    }
    DWORD2(v263) = v180;
    if ( v179 != v255 )
    {
      v247 = v56;
      v181 = v179;
      do
      {
        v182 = *(_QWORD *)(a1 + 8);
        v183 = *(_DWORD *)(v182 + 24);
        if ( v183 )
        {
          v184 = *v181;
          v185 = *(_QWORD *)(v182 + 8);
          v186 = v183 - 1;
          v187 = v186 & (((unsigned int)*v181 >> 9) ^ ((unsigned int)*v181 >> 4));
          v188 = (__int64 *)(v185 + 16LL * v187);
          v189 = *v188;
          if ( *v181 == *v188 )
          {
LABEL_243:
            v190 = v188[1];
            if ( v190 )
            {
              v191 = sub_1377F70(v190 + 56, v230);
              if ( !v191 && *(_BYTE *)(sub_157EBA0(v184) + 16) != 28 )
              {
                sub_13FCB50(v247);
                v192 = *(_QWORD *)(a1 + 8);
                v193 = *(_QWORD *)(a1 + 32);
                LODWORD(v259[0]) = (_DWORD)&loc_1010000;
                *(_QWORD *)&v258 = v193;
                *((_QWORD *)&v258 + 1) = v192;
                v194 = sub_157EBA0(v184);
                while ( v230 != sub_15F4DF0(v194, v191) )
                  ++v191;
                v197 = (_QWORD *)sub_1AAC5F0(
                                   v194,
                                   v191,
                                   &v258,
                                   a4,
                                   a5,
                                   *(double *)a6.m128i_i64,
                                   a7,
                                   v195,
                                   v196,
                                   a10,
                                   a11);
                sub_1580B80(v197, v230);
              }
            }
          }
          else
          {
            v210 = 1;
            while ( v189 != -8 )
            {
              v211 = v210 + 1;
              v187 = v186 & (v210 + v187);
              v188 = (__int64 *)(v185 + 16LL * v187);
              v189 = *v188;
              if ( v184 == *v188 )
                goto LABEL_243;
              v210 = v211;
            }
          }
        }
        ++v181;
      }
      while ( v255 != v181 );
      v25 = (unsigned __int8)v25;
      v255 = (__int64 *)v263;
    }
    if ( v255 != &v264 )
      _libc_free((unsigned __int64)v255);
  }
LABEL_254:
  v198 = *(_QWORD *)(a1 + 32);
LABEL_255:
  sub_1AA7EA0(v256, v198, *(_QWORD *)(a1 + 8), 0, 0, a4, a5, a6, a7, v169, v170, a10, a11);
  if ( v260 != (const char *)v262 )
    _libc_free((unsigned __int64)v260);
  if ( (v281 & 1) == 0 )
    j___libc_free_0(v282);
  if ( (_BYTE)v278 )
  {
    if ( (_DWORD)v277 )
    {
      v213 = v275;
      v214 = &v275[2 * (unsigned int)v277];
      do
      {
        if ( *v213 != -8 && *v213 != -4 )
        {
          v215 = v213[1];
          if ( v215 )
            sub_161E7C0((__int64)(v213 + 1), v215);
        }
        v213 += 2;
      }
      while ( v214 != v213 );
    }
    j___libc_free_0(v275);
  }
  if ( (_DWORD)v273 )
  {
    v199 = v271;
    *((_QWORD *)&v263 + 1) = 2;
    v264 = 0;
    v200 = -8;
    v201 = &v271[8 * (unsigned __int64)(unsigned int)v273];
    v265 = -8;
    *(_QWORD *)&v263 = &unk_49E6B50;
    v266 = 0;
    v281 = 2;
    v282 = 0;
    v283 = -16;
    v280 = &unk_49E6B50;
    i = 0;
    while ( 1 )
    {
      v202 = v199[3];
      if ( v200 != v202 )
      {
        v200 = v283;
        if ( v202 != v283 )
        {
          v203 = v199[7];
          if ( v203 != 0 && v203 != -8 && v203 != -16 )
          {
            sub_1649B30(v199 + 5);
            v202 = v199[3];
          }
          v200 = v202;
        }
      }
      *v199 = &unk_49EE2B0;
      if ( v200 != -8 && v200 != 0 && v200 != -16 )
        sub_1649B30(v199 + 1);
      v199 += 8;
      if ( v201 == v199 )
        break;
      v200 = v265;
    }
    v280 = &unk_49EE2B0;
    if ( v283 != 0 && v283 != -8 && v283 != -16 )
      sub_1649B30(&v281);
    *(_QWORD *)&v263 = &unk_49EE2B0;
    if ( v265 != 0 && v265 != -8 && v265 != -16 )
      sub_1649B30((_QWORD *)&v263 + 1);
  }
  j___libc_free_0(v271);
  return v25;
}
