// Function: sub_84EC30
// Address: 0x84ec30
//
void __fastcall sub_84EC30(
        __int64 a1,
        __int64 m128i_i64,
        __int64 a3,
        unsigned int a4,
        int a5,
        __m128i *a6,
        __m128i *a7,
        __int64 *a8,
        unsigned int a9,
        __int64 a10,
        _QWORD *a11,
        __int64 a12,
        _DWORD *a13,
        __m128i *a14,
        _DWORD *a15)
{
  bool v15; // r13
  unsigned __int8 v16; // bl
  _QWORD *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 *v21; // r9
  __int64 v22; // r15
  _QWORD *v23; // r14
  __int64 v24; // rdx
  _BYTE *v25; // rdx
  __int64 v26; // rax
  __int64 v27; // r13
  char v28; // dl
  __int64 v29; // rax
  __int64 v30; // rdx
  char i; // al
  unsigned int v32; // eax
  __int64 v33; // rax
  const __m128i *v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // rbx
  __int64 v38; // r8
  __int64 *v39; // r9
  __int64 v40; // r12
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 *v44; // r9
  __int64 v45; // rdx
  char nn; // al
  __int64 v47; // rdx
  __m128i *v48; // r13
  __int64 v49; // rax
  __int64 *v50; // r15
  __m128i *v51; // rbx
  __int64 v52; // rbx
  bool v53; // bl
  __int64 v54; // r8
  __int64 v55; // r9
  unsigned __int64 v56; // rsi
  __int64 v57; // rdx
  __int64 v58; // rdi
  int v59; // r13d
  char v60; // r15
  unsigned int v61; // r14d
  __m128i *v62; // r10
  __m128i *v63; // r14
  char v64; // bl
  __int64 v65; // rdx
  __int64 v66; // r8
  __int64 *v67; // r10
  __int64 v68; // rax
  __int64 v69; // rcx
  __int64 v70; // rdx
  __int64 v71; // r14
  __int64 *v72; // r10
  _BOOL4 v73; // r14d
  int v74; // ecx
  int v75; // eax
  int v76; // esi
  unsigned int v77; // r14d
  __int64 v78; // rdx
  __int64 v79; // rcx
  __int64 v80; // r8
  unsigned int v81; // esi
  __int64 v82; // rax
  unsigned __int64 v83; // rax
  __int64 v84; // rdx
  __int64 v85; // r14
  __int64 v86; // rax
  _QWORD *v87; // rbx
  __int64 v88; // rax
  __int64 v89; // r14
  __int64 v90; // rdi
  unsigned __int8 v91; // r13
  __int64 v92; // rdi
  __m128i *v93; // rbx
  int v94; // eax
  __int64 v95; // r13
  __int64 *v96; // rax
  __int64 *v97; // rcx
  __int64 *v98; // rdx
  __int64 v99; // r14
  __int64 *v100; // rbx
  __int64 v101; // r13
  char v102; // dl
  char v103; // al
  __int64 v104; // rcx
  __int64 v105; // rsi
  char v106; // dl
  __int64 *v107; // rax
  char v108; // al
  __int64 v109; // rsi
  __int64 v110; // rdi
  __int64 v111; // rdx
  __int64 v112; // rsi
  __int64 *v113; // rdi
  char *v114; // rcx
  char v115; // al
  __int64 v116; // rbx
  __int64 v117; // r15
  __int64 v118; // r14
  __int64 v119; // rax
  _QWORD *v120; // r13
  char v121; // dl
  __int64 m; // rdi
  __m128i *v123; // rax
  __int64 v124; // r14
  __int64 v125; // rax
  __int64 n; // r14
  __int64 v127; // rax
  __int64 v128; // r15
  __int64 v129; // rdi
  __int64 ii; // rsi
  _QWORD *v131; // rax
  _QWORD *v132; // rcx
  _QWORD *v133; // rdx
  __int64 *v134; // rax
  __int64 *v135; // rdx
  __int64 v136; // rdx
  __int64 v137; // rcx
  __int64 v138; // rax
  __int64 v139; // rdi
  __int64 *v140; // rcx
  __int64 *v141; // rdx
  __m128i *v142; // r10
  int v143; // ecx
  char v144; // si
  __int64 v145; // rdx
  __int64 v146; // rsi
  __int64 j; // rdx
  const __m128i *v148; // rdx
  _BOOL4 v149; // r9d
  __int64 v150; // rcx
  __int64 v151; // rax
  bool v152; // cl
  __int64 *v153; // rbx
  __int64 *v154; // rdi
  _QWORD *v155; // rcx
  __int64 v156; // r8
  _QWORD *v157; // r9
  const char *v158; // r13
  int v159; // ebx
  __int64 v160; // r14
  char *v161; // rax
  unsigned __int8 v162; // al
  __int64 v163; // rax
  __int64 v164; // rax
  __int64 v165; // rbx
  char v166; // al
  __int64 v167; // rax
  _QWORD *v168; // rdi
  __int64 v169; // rdx
  __int64 v170; // rax
  _QWORD *v171; // rdx
  __int64 v172; // r13
  _QWORD *v173; // r14
  __int64 v174; // r15
  __int64 *v175; // rbx
  __int64 v176; // rax
  __int64 v177; // r11
  __int64 v178; // rdx
  __int64 *v179; // rcx
  _QWORD *v180; // rax
  char mm; // al
  __int64 v182; // rax
  char v183; // al
  __int64 *v184; // rax
  __int64 *v185; // rdx
  __int64 v186; // rdx
  _QWORD *v187; // rax
  _QWORD *v188; // rcx
  _QWORD *v189; // rsi
  __int64 v190; // rax
  __int64 **v191; // r10
  __int64 v192; // rax
  __int64 v193; // rax
  __int64 v194; // rax
  __int64 v195; // rdi
  __m128i *v196; // r14
  __m128i *v197; // rbx
  __int64 v198; // rdx
  __int64 v199; // rcx
  __int64 v200; // r8
  __int64 v201; // r9
  __int64 v202; // rdx
  __int64 v203; // rcx
  __int64 v204; // r8
  __int64 v205; // r9
  __m128i *v206; // rdi
  __int64 v207; // rdx
  __int64 v208; // rcx
  __int64 v209; // r8
  __int64 v210; // r9
  __int64 v211; // rbx
  __int64 v212; // rdx
  __int64 v213; // rcx
  __int64 v214; // r8
  __int64 v215; // r9
  __int64 *v216; // rax
  __int64 *v217; // rax
  _BOOL4 v218; // eax
  __int64 v219; // r14
  char v220; // al
  __int64 v221; // rdx
  __int64 kk; // rbx
  __int64 v223; // rdi
  __int64 v224; // rdx
  __int8 v225; // bl
  __int64 v226; // rax
  _BYTE *v227; // rax
  __int64 v228; // r9
  __int64 v229; // rdx
  __int64 v230; // rax
  _BYTE *v231; // rax
  char v232; // al
  __int64 v233; // rax
  __int64 v234; // rsi
  __int64 *v235; // rax
  __int64 v236; // rsi
  __int64 v237; // rax
  __int64 v238; // rsi
  int v239; // eax
  _BYTE *v240; // [rsp-8h] [rbp-408h]
  unsigned int v241; // [rsp+0h] [rbp-400h]
  __m128i *v243; // [rsp+10h] [rbp-3F0h]
  int v245; // [rsp+1Ch] [rbp-3E4h]
  _QWORD *v246; // [rsp+20h] [rbp-3E0h]
  __int64 v247; // [rsp+30h] [rbp-3D0h]
  unsigned int v248; // [rsp+38h] [rbp-3C8h]
  int v249; // [rsp+3Ch] [rbp-3C4h]
  __int64 v250; // [rsp+40h] [rbp-3C0h]
  int v251; // [rsp+48h] [rbp-3B8h]
  unsigned int v252; // [rsp+4Ch] [rbp-3B4h]
  char v254; // [rsp+69h] [rbp-397h]
  unsigned __int8 v255; // [rsp+6Ah] [rbp-396h]
  unsigned __int8 v256; // [rsp+6Bh] [rbp-395h]
  unsigned int v257; // [rsp+6Ch] [rbp-394h]
  char *v258; // [rsp+70h] [rbp-390h]
  unsigned __int8 v259; // [rsp+78h] [rbp-388h]
  unsigned int v260; // [rsp+7Ch] [rbp-384h]
  int v261; // [rsp+80h] [rbp-380h]
  __int64 *v262; // [rsp+80h] [rbp-380h]
  char *v263; // [rsp+88h] [rbp-378h]
  __int64 *v264; // [rsp+90h] [rbp-370h]
  __int64 v265; // [rsp+90h] [rbp-370h]
  __int64 *v266; // [rsp+90h] [rbp-370h]
  char v267; // [rsp+98h] [rbp-368h]
  __int64 v268; // [rsp+98h] [rbp-368h]
  __int64 v269; // [rsp+A0h] [rbp-360h]
  __m128i *v270; // [rsp+A0h] [rbp-360h]
  unsigned int v271; // [rsp+A8h] [rbp-358h]
  __int64 jj; // [rsp+A8h] [rbp-358h]
  __int64 v273; // [rsp+A8h] [rbp-358h]
  __int64 *v274; // [rsp+A8h] [rbp-358h]
  __int64 *v275; // [rsp+A8h] [rbp-358h]
  __int64 v276; // [rsp+B0h] [rbp-350h]
  __int64 v277; // [rsp+B0h] [rbp-350h]
  unsigned int v278; // [rsp+B8h] [rbp-348h]
  char *v279; // [rsp+B8h] [rbp-348h]
  __int64 *v280; // [rsp+B8h] [rbp-348h]
  __int64 v281; // [rsp+B8h] [rbp-348h]
  _QWORD *v282; // [rsp+C0h] [rbp-340h]
  __int64 *k; // [rsp+C0h] [rbp-340h]
  bool v284; // [rsp+C0h] [rbp-340h]
  _QWORD *v285; // [rsp+C0h] [rbp-340h]
  __m128i *v286; // [rsp+C0h] [rbp-340h]
  __int64 v287; // [rsp+C0h] [rbp-340h]
  unsigned __int8 v288; // [rsp+C8h] [rbp-338h]
  __m128i *v289; // [rsp+C8h] [rbp-338h]
  __int64 *v290; // [rsp+C8h] [rbp-338h]
  _BYTE *v291; // [rsp+C8h] [rbp-338h]
  __int64 v292; // [rsp+C8h] [rbp-338h]
  __int64 v293; // [rsp+C8h] [rbp-338h]
  _QWORD *v294; // [rsp+C8h] [rbp-338h]
  __int64 v295; // [rsp+C8h] [rbp-338h]
  int v296; // [rsp+D4h] [rbp-32Ch] BYREF
  __int64 v297; // [rsp+D8h] [rbp-328h] BYREF
  int v298; // [rsp+E0h] [rbp-320h] BYREF
  int v299; // [rsp+E4h] [rbp-31Ch] BYREF
  __int64 v300; // [rsp+E8h] [rbp-318h] BYREF
  __m128i *v301; // [rsp+F0h] [rbp-310h] BYREF
  __int64 v302; // [rsp+F8h] [rbp-308h] BYREF
  _QWORD *v303; // [rsp+100h] [rbp-300h] BYREF
  _QWORD *v304; // [rsp+108h] [rbp-2F8h] BYREF
  __m128i v305[22]; // [rsp+110h] [rbp-2F0h] BYREF
  __m128i v306; // [rsp+270h] [rbp-190h] BYREF
  __int64 v307; // [rsp+288h] [rbp-178h]

  v15 = 0;
  v16 = a1;
  v257 = m128i_i64;
  v248 = a3;
  v288 = a1;
  HIDWORD(v297) = 0;
  v17 = (_QWORD *)sub_82BD70(a1, m128i_i64, a3);
  v22 = v17[128];
  v23 = v17;
  if ( v22 )
  {
    v19 = v17[126];
    v15 = (*(_BYTE *)(v19 + 8 * (5 * v22 - 5)) & 2) != 0;
  }
  if ( v22 == v17[127] )
  {
    a1 = (__int64)v17;
    sub_8332F0((__int64)v17, m128i_i64, v18, v19, v20, v21);
  }
  v24 = v23[126] + 40 * v22;
  if ( v24 )
  {
    *(_BYTE *)v24 &= 0xFCu;
    *(_QWORD *)(v24 + 8) = 0;
    *(_QWORD *)(v24 + 16) = 0;
    *(_QWORD *)(v24 + 24) = 0;
    *(_QWORD *)(v24 + 32) = 0;
    v24 = v23[126] + 40 * v22;
  }
  v23[128] = v22 + 1;
  *(_BYTE *)v24 = *(_BYTE *)v24 & 0xFD | (2 * v15);
  *a15 = 0;
  if ( a13 )
    *a13 = 0;
  if ( dword_4F04C44 == -1 )
  {
    v25 = qword_4F04C68;
    v26 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( (*(_BYTE *)(v26 + 6) & 6) == 0 && *(_BYTE *)(v26 + 4) != 12 )
    {
      if ( *(_BYTE *)(qword_4D03C50 + 16LL) > 3u || word_4D04898 )
      {
        v27 = a6->m128i_i64[0];
        if ( !(_DWORD)m128i_i64 )
          goto LABEL_30;
LABEL_15:
        if ( !a6[1].m128i_i8[0] )
          goto LABEL_33;
        goto LABEL_16;
      }
      goto LABEL_57;
    }
  }
  a1 = (__int64)a6;
  if ( sub_82ED00((__int64)a6, m128i_i64) )
    goto LABEL_44;
  if ( (_DWORD)m128i_i64 )
  {
    if ( *(_BYTE *)(qword_4D03C50 + 16LL) > 3u || (v25 = (_BYTE *)word_4D04898, word_4D04898) )
    {
      v27 = a6->m128i_i64[0];
      goto LABEL_15;
    }
    goto LABEL_57;
  }
  a1 = (__int64)a7;
  if ( sub_82ED00((__int64)a7, m128i_i64) )
  {
LABEL_44:
    a1 = v16;
    sub_831920(v16, m128i_i64, a6, a7, (__m128i *)a12, a8, a9, a11);
    *a15 = 1;
    v25 = v240;
    goto LABEL_45;
  }
  if ( *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u && !word_4D04898 )
  {
LABEL_57:
    if ( !*a15 )
      goto LABEL_46;
LABEL_45:
    m128i_i64 = a12;
    *(_QWORD *)(a12 + 68) = *a8;
    goto LABEL_46;
  }
  v27 = a6->m128i_i64[0];
LABEL_30:
  v32 = 0;
  if ( a7[1].m128i_i8[0] != 5 )
    v32 = a4;
  a4 = v32;
  if ( !a6[1].m128i_i8[0] )
    goto LABEL_33;
LABEL_16:
  v28 = *(_BYTE *)(v27 + 140);
  if ( v28 == 12 )
  {
    v29 = v27;
    do
    {
      v29 = *(_QWORD *)(v29 + 160);
      v28 = *(_BYTE *)(v29 + 140);
    }
    while ( v28 == 12 );
  }
  if ( v28 )
  {
    if ( (_DWORD)m128i_i64 )
    {
      v255 = v16;
      v246 = 0;
      v249 = 0;
      goto LABEL_63;
    }
    if ( a7[1].m128i_i8[0] )
    {
      v30 = a7->m128i_i64[0];
      for ( i = *(_BYTE *)(a7->m128i_i64[0] + 140); i == 12; i = *(_BYTE *)(v30 + 140) )
        v30 = *(_QWORD *)(v30 + 160);
      if ( i )
      {
        v249 = 0;
        v255 = v16;
        v246 = 0;
LABEL_63:
        v241 = v288;
        while ( 1 )
        {
          a1 = v27;
          m128i_i64 = (__int64)dword_4F07508;
          v300 = 0;
          v301 = 0;
          v243 = (__m128i *)v27;
          *(_QWORD *)dword_4F07508 = *a8;
          v252 = sub_8D3A70(v27);
          if ( !v252 )
          {
            if ( !dword_4D047E4 )
              break;
            a1 = v27;
            if ( !(unsigned int)sub_8D2870(v27) )
              break;
          }
          v298 = 0;
          v299 = 0;
          v247 = 0;
          v53 = (a6[1].m128i_i8[2] & 2) != 0;
          v50 = (__int64 *)sub_6E3060(a6);
          if ( !v257 )
            goto LABEL_306;
LABEL_83:
          v301 = 0;
          v250 = 0;
          v261 = 0;
          if ( dword_4F04C44 == -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0 )
          {
            v261 = dword_4D047C8;
            if ( dword_4D047C8 )
            {
              if ( unk_4F04C48 == -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) == 0 )
              {
                v261 = 0;
              }
              else
              {
                v261 = 1;
                if ( a9 )
                {
                  v138 = sub_878D80(a9, a10, qword_4F04C68[0]);
                  v261 = v138 == 0;
                  if ( v138 )
                  {
                    v139 = *(_QWORD *)(v138 + 32);
                    if ( v139 )
                    {
                      if ( (*(_BYTE *)(v138 + 40) & 2) != 0 )
                      {
                        if ( !v50 )
LABEL_596:
                          BUG();
                        v140 = 0;
                        while ( 1 )
                        {
                          v141 = (__int64 *)*v50;
                          *v50 = (__int64)v140;
                          if ( !v141 )
                            break;
                          v140 = v50;
                          v50 = v141;
                        }
                        v142 = a6;
                        v63 = a7;
                        v247 = (__int64)v140;
                        v143 = 9;
                      }
                      else
                      {
                        v142 = a7;
                        v63 = a6;
                        v143 = 0;
                      }
                      v144 = *(_BYTE *)(v139 + 80);
                      v145 = v139;
                      if ( v144 == 16 )
                      {
                        v145 = **(_QWORD **)(v139 + 88);
                        v144 = *(_BYTE *)(v145 + 80);
                      }
                      if ( v144 == 24 )
                        v145 = *(_QWORD *)(v145 + 88);
                      if ( (*(_BYTE *)(v145 + 81) & 0x10) != 0 )
                      {
                        v146 = *(_QWORD *)(v145 + 88);
                        if ( *(_BYTE *)(v145 + 80) == 20 )
                          v146 = *(_QWORD *)(v146 + 176);
                        for ( j = *(_QWORD *)(v146 + 152); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
                          ;
                        v148 = *(const __m128i **)(*(_QWORD *)(j + 168) + 40LL);
                        if ( v148 )
                        {
                          v63[1].m128i_i8[2] &= ~2u;
                          v148 = v63;
                          v149 = 1;
                        }
                        else
                        {
                          v247 = (__int64)v50;
                          v149 = 0;
                        }
                      }
                      else
                      {
                        v247 = (__int64)v50;
                        v148 = 0;
                        v149 = 0;
                      }
                      v286 = v142;
                      v295 = v138;
                      sub_8360D0(
                        v139,
                        0,
                        0,
                        v247,
                        0,
                        v149,
                        v148,
                        0,
                        0,
                        1u,
                        0,
                        0,
                        0,
                        1,
                        1u,
                        0,
                        v143,
                        (__int64 *)&v301,
                        (__int64)&v300,
                        &v298,
                        &v299);
                      v62 = v286;
                      v63[1].m128i_i8[2] = (2 * v53) | v63[1].m128i_i8[2] & 0xFD;
                      v183 = *(_BYTE *)(v295 + 40);
                      if ( (v183 & 1) == 0 )
                      {
                        v245 = 0;
                        goto LABEL_98;
                      }
                      if ( (v183 & 2) != 0 )
                      {
                        if ( v50 )
                        {
                          v184 = 0;
                          while ( 1 )
                          {
                            v185 = (__int64 *)*v50;
                            *v50 = (__int64)v184;
                            v184 = v50;
                            if ( !v185 )
                              break;
                            v50 = v185;
                          }
                        }
                        v186 = (__int64)v301;
                        if ( v301 )
                        {
                          v187 = (_QWORD *)v301[7].m128i_i64[1];
                          v301[9].m128i_i8[1] |= 3u;
                          if ( v187 )
                          {
                            v188 = 0;
                            while ( 1 )
                            {
                              v189 = (_QWORD *)*v187;
                              *v187 = v188;
                              v188 = v187;
                              if ( !v189 )
                                break;
                              v187 = v189;
                            }
                          }
                          *(_QWORD *)(v186 + 120) = v187;
                          v62 = v63;
                          v63 = v286;
                          v245 = 0;
                          goto LABEL_98;
                        }
                        v62 = v63;
                        v63 = v286;
                        goto LABEL_484;
                      }
                      if ( v301 )
                      {
                        v301[9].m128i_i8[1] |= 1u;
                      }
                      else
                      {
LABEL_484:
                        v64 = 0;
                        if ( a13 )
                          goto LABEL_397;
                      }
                      m128i_i64 = (__int64)a8;
                      a1 = (__int64)&v301;
                      sub_82D8D0((__int64 *)&v301, (__int64)a8, &v297, &v296, v54, v55);
                      v245 = 0;
                      goto LABEL_103;
                    }
                  }
                  v250 = (__int64)v301;
                }
              }
            }
          }
          v56 = v252;
          v245 = 0;
          v271 = 0;
          v256 = v255;
          v251 = 0;
          v254 = 2 * v53;
          v264 = v50;
          if ( !v252 )
          {
LABEL_85:
            v57 = v248;
            if ( v248 )
              goto LABEL_86;
LABEL_168:
            v77 = v256;
            goto LABEL_131;
          }
          while ( 1 )
          {
            v91 = v256;
            if ( !(unsigned int)sub_8D23B0(v243) )
              goto LABEL_162;
LABEL_184:
            if ( (unsigned int)sub_8D3A70(v243) )
            {
              sub_8AD220(v243, 0);
              v56 = (unsigned __int64)v243;
              if ( v243[8].m128i_i8[12] == 12 )
                goto LABEL_163;
LABEL_186:
              v56 = (unsigned __int64)v243;
              goto LABEL_164;
            }
LABEL_162:
            v56 = (unsigned __int64)v243;
            if ( v243[8].m128i_i8[12] != 12 )
              goto LABEL_186;
            do
LABEL_163:
              v56 = *(_QWORD *)(v56 + 160);
            while ( *(_BYTE *)(v56 + 140) == 12 );
LABEL_164:
            v92 = sub_7D3790(v91, (const char *)v56);
            if ( !v92 )
              goto LABEL_85;
            if ( v271 )
            {
              v93 = a7;
              v94 = 9;
            }
            else
            {
              v93 = a6;
              v94 = 0;
            }
            v56 = 0;
            v93[1].m128i_i8[2] &= ~2u;
            sub_8360D0(
              v92,
              0,
              0,
              v247,
              0,
              1,
              v93,
              0,
              0,
              1u,
              0,
              v261,
              0,
              1,
              1u,
              0,
              v94,
              (__int64 *)&v301,
              (__int64)&v300,
              &v298,
              &v299);
            v57 = v248;
            v93[1].m128i_i8[2] = v254 | v93[1].m128i_i8[2] & 0xFD;
            if ( !v248 )
              goto LABEL_168;
LABEL_86:
            if ( v271 || !a4 )
            {
LABEL_88:
              v58 = dword_4D041C0;
              if ( !dword_4D041C0 )
                goto LABEL_240;
              goto LABEL_89;
            }
            v56 = v256;
            v259 = v256;
            if ( v257 )
            {
              switch ( v256 )
              {
                case 5u:
                  LOBYTE(v76) = 65;
                  v259 = 5;
                  v279 = "A;P";
                  goto LABEL_219;
                case 6u:
                  LOBYTE(v76) = 65;
                  v259 = 6;
                  v279 = "A";
                  goto LABEL_219;
                case 7u:
                  LOBYTE(v76) = 79;
                  v259 = 7;
                  v279 = "O;F";
                  goto LABEL_219;
                case 0xDu:
                  LOBYTE(v76) = 73;
                  v259 = 13;
                  v279 = "I";
                  goto LABEL_219;
                case 0xEu:
                  v57 = (__int64)"b";
                  v259 = 14;
                  v271 = 0;
                  v76 = dword_4D0439C == 0 ? 98 : 66;
                  v161 = "B";
                  if ( !dword_4D0439C )
                    v161 = "b";
                  v279 = v161;
                  goto LABEL_219;
                case 0x25u:
                  v158 = "La;O";
                  if ( dword_4F077C4 == 2 && unk_4F07778 >= 201703 )
                    v158 = "Ln;O";
                  goto LABEL_378;
                case 0x26u:
                  v158 = "Ln;O";
                  goto LABEL_378;
                default:
                  goto LABEL_129;
              }
            }
            switch ( v256 )
            {
              case 5u:
                LOBYTE(v76) = 65;
                v271 = 0;
                v259 = 5;
                v279 = "AA;OD;DO";
                break;
              case 6u:
                LOBYTE(v76) = 65;
                v271 = 0;
                v259 = 6;
                v279 = "AA;OD;=OO";
                break;
              case 7u:
              case 8u:
                LOBYTE(v76) = 65;
                v271 = 0;
                v279 = "AA";
                break;
              case 9u:
              case 0xAu:
              case 0xBu:
              case 0xCu:
              case 0x1Au:
              case 0x1Bu:
                LOBYTE(v76) = 73;
                v271 = 0;
                v279 = "II";
                break;
              case 0x10u:
              case 0x11u:
              case 0x20u:
              case 0x21u:
              case 0x22u:
                goto LABEL_217;
              case 0x12u:
              case 0x13u:
                v158 = "LaA;OD";
                goto LABEL_378;
              case 0x14u:
              case 0x15u:
                v158 = "LaA";
                goto LABEL_378;
              case 0x16u:
              case 0x17u:
              case 0x18u:
              case 0x19u:
              case 0x1Cu:
              case 0x1Du:
                v158 = "LiI";
                goto LABEL_378;
              case 0x1Eu:
              case 0x1Fu:
                LOBYTE(v76) = 65;
                v271 = 0;
                v279 = "AA;=PP;NN;=MM;=EE;=HH";
                break;
              case 0x23u:
              case 0x24u:
                LOBYTE(v76) = 98;
                v279 = "bb";
                v271 = dword_4D0439C;
                if ( dword_4D0439C )
                {
                  LOBYTE(v76) = 66;
                  v271 = 0;
                  v279 = "BB";
                }
                break;
              case 0x25u:
                v158 = "Lai;Oi";
                if ( dword_4F077C4 == 2 && unk_4F07778 >= 201703 )
                  v158 = "Lni;Oi";
                goto LABEL_378;
              case 0x26u:
                v158 = "Lni;Oi";
LABEL_378:
                v159 = HIDWORD(qword_4D0495C);
                v160 = v264[3];
                if ( *(_BYTE *)(v160 + 25) != 1 || sub_6ED0A0(v160 + 8) )
                {
                  v271 = sub_8E31E0(*(_QWORD *)(v160 + 8));
                  if ( !v271 )
                    goto LABEL_88;
                }
                LOBYTE(v76) = v158[1];
                v271 = v159 == 0;
                v279 = (char *)(v158 + 1);
                break;
              case 0x28u:
                LOBYTE(v76) = 61;
                v271 = 0;
                v259 = 40;
                v279 = "=OM";
                break;
              case 0x2Bu:
                LOBYTE(v76) = 79;
                v271 = 0;
                v259 = 43;
                v279 = "OD;DO;hD";
                break;
              case 0x2Cu:
                LOBYTE(v76) = 65;
                v271 = 0;
                v259 = 44;
                v279 = "AA;=PP;=MM;=SS";
                break;
              case 0x2Du:
              case 0x2Eu:
                LOBYTE(v76) = 65;
                v279 = "AA;=PP";
                v271 = dword_4D047E4;
                if ( dword_4D047E4 )
                {
                  v271 = 0;
                  v279 = "AA;=PP;=EE";
                }
                break;
              default:
                goto LABEL_129;
            }
LABEL_219:
            if ( !(_BYTE)v76 )
              goto LABEL_235;
LABEL_220:
            if ( (_BYTE)v76 == 59 )
              goto LABEL_235;
            v114 = v279;
            v57 = 0;
            v54 = dword_4D047E4;
            v115 = v76;
            do
            {
              if ( v115 == 78 )
              {
                if ( !((unsigned int)qword_4F077B4 | unk_4F0773C) )
                  v57 = 1;
LABEL_224:
                v115 = *++v114;
                if ( !v115 )
                  break;
                continue;
              }
              if ( v115 > 78 )
              {
                if ( v115 == 104 )
                  v57 = 1;
                goto LABEL_224;
              }
              if ( v115 == 69 )
              {
                if ( !dword_4D047E4 )
                  v57 = 1;
                goto LABEL_224;
              }
              if ( v115 == 72 )
                v57 = 1;
              v115 = *++v114;
              if ( !v115 )
                break;
            }
            while ( v115 != 59 );
            if ( (_DWORD)v57 )
              goto LABEL_233;
            while ( 1 )
            {
LABEL_235:
              if ( (_BYTE)v76 != 61 )
              {
                sub_84DCB0(v259, v279, v271, v264, &v301, 0);
                v258 = v279;
                goto LABEL_237;
              }
              v258 = v279 + 1;
              if ( v264 )
              {
                v276 = 0;
                v292 = 0;
                v263 = v279 + 1;
                for ( k = v264; ; k = (__int64 *)v125 )
                {
                  v116 = k[3];
                  v269 = *(_QWORD *)(v116 + 8);
                  v267 = *v263;
                  if ( *v263 != 79 || v279[2] != 77 )
                  {
                    v117 = *(_QWORD *)(v116 + 8);
                    if ( (unsigned int)sub_8E31E0(v269) )
                    {
                      if ( *(_BYTE *)(v269 + 140) == 12 )
                      {
                        do
                          v117 = *(_QWORD *)(v117 + 160);
                        while ( *(_BYTE *)(v117 + 140) == 12 );
                      }
                      else
                      {
                        v117 = v269;
                      }
                      v118 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v117 + 96LL) + 40LL);
                      if ( v118 )
                      {
                        v119 = *(_QWORD *)(v118 + 8);
                        if ( v119 )
                        {
                          v120 = *(_QWORD **)(*(_QWORD *)(*(_QWORD *)v117 + 96LL) + 40LL);
                          v260 = 0;
                          do
                          {
                            v121 = *(_BYTE *)(v119 + 80);
                            if ( v121 == 16 )
                            {
                              v119 = **(_QWORD **)(v119 + 88);
                              v121 = *(_BYTE *)(v119 + 80);
                            }
                            if ( v121 == 24 )
                              v119 = *(_QWORD *)(v119 + 88);
                            for ( m = *(_QWORD *)(*(_QWORD *)(v119 + 88) + 152LL);
                                  *(_BYTE *)(m + 140) == 12;
                                  m = *(_QWORD *)(m + 160) )
                            {
                              ;
                            }
                            v123 = sub_73D790(m);
                            v124 = sub_6EEB30((__int64)v123, 0);
                            if ( (unsigned int)sub_827590(v124, v267) )
                            {
                              v306.m128i_i64[0] = v124;
                              if ( !(unsigned int)sub_829DD0(v124, v117, v120, v292, v276) )
                              {
                                if ( v292 | v276 && (unsigned int)sub_8D2E30(v306.m128i_i64[0]) )
                                  sub_829EB0(v306.m128i_i64, v117, v120, v292, v276);
                                sub_84DCB0(v259, v258, v271, v264, &v301, (const __m128i *)v306.m128i_i64[0]);
                                v260 = 1;
                              }
                            }
                            v120 = (_QWORD *)*v120;
                            if ( !v120 )
                              break;
                            v119 = v120[1];
                          }
                          while ( v119 );
                          v55 = v260;
                          if ( !v260 )
                            v117 = v292;
                          v292 = v117;
                        }
                      }
                      if ( v267 == 67 )
                      {
                        v306.m128i_i64[0] = v269;
                        if ( k == v264 )
                        {
                          v127 = *k;
                          if ( !*k )
                            BUG();
                          if ( *(_BYTE *)(v127 + 8) == 3 )
                            v127 = sub_6BBB10(k);
                        }
                        else
                        {
                          v127 = (__int64)v264;
                        }
                        v128 = *(_QWORD *)(*(_QWORD *)(v127 + 24) + 8LL);
                        if ( (unsigned int)sub_8D3A70(v128) && sub_8D5CE0(v128, v269) )
                        {
                          v306.m128i_i64[0] = (__int64)sub_73CA70((const __m128i *)v306.m128i_i64[0], v128);
                          v129 = v306.m128i_i64[0];
                        }
                        else
                        {
                          v129 = v306.m128i_i64[0];
                        }
                        if ( !(unsigned int)sub_829DD0(v129, 0, 0, v292, v276) )
                          goto LABEL_284;
                      }
                    }
                    else
                    {
                      for ( n = sub_6EEB30(v269, v116 + 8); *(_BYTE *)(n + 140) == 12; n = *(_QWORD *)(n + 160) )
                        ;
                      if ( (unsigned int)sub_827590(n, v267) )
                      {
                        v306.m128i_i64[0] = n;
                        if ( !(unsigned int)sub_829DD0(n, 0, 0, v292, v276) )
                        {
                          if ( v292 | v276 && (unsigned int)sub_8D2E30(v306.m128i_i64[0]) )
                            sub_829EB0(v306.m128i_i64, 0, 0, v292, v276);
LABEL_284:
                          v276 = v306.m128i_i64[0];
                          sub_84DCB0(v259, v258, v271, v264, &v301, (const __m128i *)v306.m128i_i64[0]);
                        }
                      }
                    }
                  }
                  v125 = *k;
                  if ( !*k )
                    break;
                  if ( *(_BYTE *)(v125 + 8) == 3 )
                  {
                    v125 = sub_6BBB10(k);
                    if ( !v125 )
                      break;
                  }
                  ++v263;
                }
              }
LABEL_237:
              v54 = v257;
              if ( !v257 )
                break;
              v114 = v258 + 1;
              v56 = (unsigned __int64)(v258 + 2);
              v279 = v258 + 2;
              if ( v258[1] != 59 )
                goto LABEL_239;
LABEL_234:
              LOBYTE(v76) = v114[1];
              if ( (_BYTE)v76 )
                goto LABEL_220;
            }
            v114 = v258 + 2;
            v115 = v258[2];
LABEL_233:
            v56 = (unsigned __int64)(v114 + 1);
            v279 = v114 + 1;
            if ( v115 == 59 )
              goto LABEL_234;
LABEL_239:
            v271 = 0;
            v58 = dword_4D041C0;
            if ( !dword_4D041C0 )
            {
LABEL_240:
              v50 = v264;
              v64 = v271 & 1;
              goto LABEL_241;
            }
LABEL_89:
            v59 = v251;
            v60 = v256;
            v61 = v252;
LABEL_90:
            v57 = sub_82BD70(v58, v56, v57);
            if ( (*(_BYTE *)(*(_QWORD *)(v57 + 1008) + 8 * (5LL * *(_QWORD *)(v57 + 1024) - 5)) & 2) != 0 )
              goto LABEL_240;
            if ( v271 )
              break;
            if ( v59 )
            {
              v251 = v59;
              v256 = v60;
              v99 = (__int64)v301;
              if ( v301 != (__m128i *)v250 )
              {
                v100 = (__int64 *)&v301;
                if ( v250 )
                {
LABEL_190:
                  v101 = v250;
                  do
                  {
                    v102 = *(_BYTE *)(v99 + 32);
                    v103 = *(_BYTE *)(v101 + 32);
                    v104 = *(_QWORD *)(v101 + 8);
                    v105 = *(_QWORD *)(v99 + 8);
                    LOBYTE(v54) = v102 == v103;
                    if ( v104 != 0 && v102 == v103 && v105 )
                    {
                      v106 = *(_BYTE *)(v105 + 80);
                      if ( v106 == 16 )
                      {
                        v107 = *(__int64 **)(v105 + 88);
                        v105 = *v107;
                        v106 = *(_BYTE *)(*v107 + 80);
                      }
                      if ( v106 == 24 )
                      {
                        v105 = *(_QWORD *)(v105 + 88);
                        v106 = *(_BYTE *)(v105 + 80);
                      }
                      v108 = *(_BYTE *)(v104 + 80);
                      if ( v108 == 16 )
                      {
                        v104 = **(_QWORD **)(v104 + 88);
                        v108 = *(_BYTE *)(v104 + 80);
                      }
                      if ( v108 == 24 )
                      {
                        v104 = *(_QWORD *)(v104 + 88);
                        v108 = *(_BYTE *)(v104 + 80);
                      }
                      v109 = *(_QWORD *)(v105 + 88);
                      if ( v106 == 20 )
                        v109 = *(_QWORD *)(v109 + 176);
                      v110 = *(_QWORD *)(v109 + 152);
                      v111 = *(_QWORD *)(v104 + 88);
                      if ( v108 == 20 )
                        v111 = *(_QWORD *)(v111 + 176);
                      v112 = *(_QWORD *)(v111 + 152);
                      if ( v110 == v112 || (unsigned int)sub_8D97D0(v110, v112, 0, v104, v54) )
                      {
LABEL_209:
                        *v100 = *(_QWORD *)v99;
                        v113 = *(__int64 **)(v99 + 40);
                        *(_QWORD *)v99 = 0;
                        sub_725130(v113);
                        sub_82D8A0(*(_QWORD **)(v99 + 120));
                        *(_QWORD *)v99 = qword_4D03C68;
                        qword_4D03C68 = (_QWORD *)v99;
                        goto LABEL_210;
                      }
                    }
                    else if ( v102 == v103 )
                    {
                      goto LABEL_209;
                    }
                    v101 = *(_QWORD *)v101;
                  }
                  while ( v101 );
                }
                while ( 1 )
                {
                  *(_BYTE *)(v99 + 145) |= 1u;
                  v100 = (__int64 *)v99;
LABEL_210:
                  v99 = *v100;
                  if ( *v100 == v250 )
                    break;
                  if ( v250 )
                    goto LABEL_190;
                }
              }
              goto LABEL_178;
            }
            if ( v60 == 30 )
            {
              v251 = 0;
              v256 = 30;
              goto LABEL_178;
            }
            if ( v60 != 34 )
            {
              if ( v60 == 31 )
              {
                v60 = 30;
                v250 = (__int64)v301;
                v75 = 30;
                if ( v61 )
                  goto LABEL_161;
                v55 = v248;
                if ( v248 )
                {
                  v54 = a4;
                  if ( a4 )
                  {
                    v251 = 1;
                    LOBYTE(v76) = 65;
                    v256 = 30;
                    v279 = "AA;=PP;NN;=MM;=EE;=HH";
                    v259 = 30;
                    if ( v257 )
                      goto LABEL_129;
                    v252 = 0;
                    v271 = 0;
                    goto LABEL_219;
                  }
LABEL_123:
                  v56 = dword_4D041C0;
                  v61 = 0;
                  v59 = 1;
                  if ( dword_4D041C0 )
                    goto LABEL_90;
LABEL_97:
                  v50 = v264;
                  v62 = a7;
                  v63 = a6;
LABEL_98:
                  v64 = 0;
                  if ( !a13 )
                  {
                    m128i_i64 = (__int64)a8;
                    a1 = (__int64)&v301;
                    sub_82D8D0((__int64 *)&v301, (__int64)a8, &v297, &v296, v54, v55);
                    v48 = v301;
                    goto LABEL_104;
                  }
LABEL_99:
                  if ( v301 || v245 )
                    goto LABEL_101;
LABEL_397:
                  v245 = 0;
                  *a13 = 1;
                  goto LABEL_101;
                }
LABEL_130:
                v251 = 1;
                v256 = v60;
                v252 = 0;
                v77 = v75;
LABEL_131:
                v302 = 0;
                v303 = 0;
                v304 = 0;
                if ( !v257 )
                {
                  if ( (unsigned int)sub_8D3A70(a7->m128i_i64[0]) )
                  {
                    v95 = a7->m128i_i64[0];
                    if ( (unsigned int)sub_8D23B0(a7->m128i_i64[0]) )
                    {
                      if ( (unsigned int)sub_8D3A70(v95) )
                        sub_8AD220(v95, 0);
                    }
                  }
                }
                sub_87A720(v77, &v306, a8);
                v81 = 128;
                if ( dword_4F077BC )
                {
                  if ( qword_4F077A8 <= 0x76BFu || (v81 = 2097280, (v261 & 1) == 0) )
                    v81 = 128;
                }
                v82 = sub_7D5DD0(&v306, v81, v78, v79, v80);
                v291 = (_BYTE *)v82;
                if ( v82 )
                {
                  v83 = *(unsigned __int8 *)(v82 + 80);
                  if ( (unsigned __int8)v83 <= 0x14u && (v84 = 1182720, _bittest64(&v84, v83)) )
                  {
                    v85 = v307;
                    if ( (unsigned int)sub_8287B0(v291) )
                      goto LABEL_139;
                  }
                  else
                  {
                    v291 = 0;
                    v85 = 0;
                  }
                  v90 = (__int64)a6;
                  if ( a6[1].m128i_i8[0] == 3 )
                  {
LABEL_157:
                    sub_82C920(v90, &v302, &v304, &v303);
                    goto LABEL_158;
                  }
                }
                else
                {
                  v90 = (__int64)a6;
                  v85 = 0;
                  if ( a6[1].m128i_i8[0] == 3 )
                    goto LABEL_157;
                }
                sub_7D38C0(a6->m128i_i64[0], &v302);
LABEL_158:
                if ( !v257 )
                {
                  if ( a7[1].m128i_i8[0] == 3 )
                    sub_82C920((__int64)a7, &v302, &v304, &v303);
                  else
                    sub_7D38C0(a7->m128i_i64[0], &v302);
                }
LABEL_139:
                v56 = (unsigned __int64)&v306;
                v86 = sub_7D4C80(v85, &v306, &v302, &v304, &v303, 0);
                v282 = (_QWORD *)v86;
                if ( v86 )
                {
                  v87 = (_QWORD *)v86;
                  while ( 1 )
                  {
                    v89 = v87[1];
                    if ( dword_4F04C44 != -1
                      || (v88 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v88 + 6) & 6) != 0)
                      || *(_BYTE *)(v88 + 4) == 12 )
                    {
                      if ( sub_82EEC0(v87[1]) )
                        break;
                      if ( dword_4F04C44 != -1 )
                        goto LABEL_150;
                      v88 = qword_4F04C68[0] + 776LL * dword_4F04C64;
                      if ( (*(_BYTE *)(v88 + 6) & 6) != 0 )
                        goto LABEL_150;
                    }
                    if ( *(_BYTE *)(v88 + 4) == 12 )
                    {
LABEL_150:
                      if ( (_DWORD)qword_4F077B4 && !dword_4D047A8 && (*(_BYTE *)(qword_4D03C50 + 20LL) & 0x20) != 0 )
                        break;
                    }
                    v56 = 0;
                    sub_8360D0(
                      v89,
                      0,
                      0,
                      (__int64)v264,
                      0,
                      0,
                      0,
                      0,
                      0,
                      1u,
                      v89 != (_QWORD)v291 || v282 != v87,
                      v261,
                      0,
                      0,
                      1u,
                      0,
                      v271 != 0 ? 9 : 0,
                      (__int64 *)&v301,
                      (__int64)&v300,
                      &v298,
                      &v299);
                    v87 = (_QWORD *)*v87;
                    if ( !v87 )
                      goto LABEL_152;
                  }
                  v245 = 1;
                }
LABEL_152:
                sub_878490(v282);
                goto LABEL_86;
              }
              if ( ((v60 - 16) & 0xEE) != 0 )
                goto LABEL_97;
              v60 = 34;
              v250 = (__int64)v301;
              v75 = 34;
              if ( !v61 )
              {
                v57 = v248;
                if ( !v248 )
                  goto LABEL_130;
                if ( !a4 )
                {
                  v60 = 34;
                  goto LABEL_123;
                }
                if ( v257 )
LABEL_129:
                  sub_721090();
                v252 = 0;
                v251 = 1;
                v259 = 34;
                v256 = 34;
LABEL_217:
                v279 = "AA;=PP;=EE;NN";
                LOBYTE(v76) = 65;
                v271 = HIDWORD(qword_4D0495C);
                if ( HIDWORD(qword_4D0495C) )
                {
                  v271 = 0;
                  v279 = "AA;=PP;=MM;=EE;NN";
                }
                goto LABEL_219;
              }
LABEL_161:
              v251 = 1;
              v91 = v75;
              v256 = v60;
              v252 = v61;
              if ( (unsigned int)sub_8D23B0(v243) )
                goto LABEL_184;
              goto LABEL_162;
            }
            v256 = 34;
            v251 = 0;
LABEL_178:
            v96 = v264;
            if ( !v264 )
              goto LABEL_596;
            v97 = 0;
            while ( 1 )
            {
              v98 = (__int64 *)*v96;
              *v96 = (__int64)v97;
              if ( !v98 )
                break;
              v97 = v96;
              v96 = v98;
            }
            v264 = v96;
            v247 = (__int64)v97;
            v243 = (__m128i *)a7->m128i_i64[0];
            v271 = 1;
            v252 = sub_8D3A70(a7->m128i_i64[0]);
            v56 = v252;
            v250 = (__int64)v301;
            if ( !v252 )
              goto LABEL_85;
          }
          v50 = v264;
          for ( ii = (__int64)v301; ii != v250; ii = *(_QWORD *)ii )
          {
            v131 = *(_QWORD **)(ii + 120);
            if ( v131 )
            {
              v132 = 0;
              while ( 1 )
              {
                v133 = (_QWORD *)*v131;
                *v131 = v132;
                v132 = v131;
                if ( !v133 )
                  break;
                v131 = v133;
              }
            }
            *(_BYTE *)(ii + 145) |= 3u;
            *(_QWORD *)(ii + 120) = v131;
          }
          if ( v264 )
          {
            v134 = 0;
            while ( 1 )
            {
              v135 = (__int64 *)*v50;
              *v50 = (__int64)v134;
              v134 = v50;
              if ( !v135 )
                break;
              v50 = v135;
            }
          }
          v64 = 1;
LABEL_241:
          v62 = a7;
          v63 = a6;
          if ( a13 )
            goto LABEL_99;
LABEL_101:
          m128i_i64 = (__int64)a8;
          a1 = (__int64)&v301;
          v289 = v62;
          sub_82D8D0((__int64 *)&v301, (__int64)a8, &v297, &v296, v54, v55);
          if ( dword_4D04964 || (v67 = (__int64 *)v289, !v64) )
          {
LABEL_103:
            v48 = v301;
            goto LABEL_104;
          }
          v48 = v301;
          if ( v296 )
          {
            if ( v301 )
            {
              v150 = v301->m128i_i64[0];
              if ( v301->m128i_i64[0] )
              {
                if ( !*(_QWORD *)v150 && (v301[9].m128i_i8[1] & 2) != 0 )
                {
                  v151 = v301->m128i_i64[1];
                  if ( v151 )
                  {
                    if ( (*(_BYTE *)(v151 + 81) & 0x10) != 0 && v301[7].m128i_i64[1] )
                    {
                      v65 = *(_QWORD *)(v150 + 8);
                      v152 = v301[2].m128i_i8[0] == *(_BYTE *)(v150 + 32);
                      LOBYTE(a1) = v65 != 0;
                      m128i_i64 = v152;
                      LOBYTE(v150) = v65 != 0 && v152;
                      if ( (_BYTE)v150 )
                      {
                        if ( *(_BYTE *)(v151 + 80) == 16 )
                          v151 = **(_QWORD **)(v151 + 88);
                        if ( *(_BYTE *)(v151 + 80) == 24 )
                          v151 = *(_QWORD *)(v151 + 88);
                        if ( *(_BYTE *)(v65 + 80) == 16 )
                          v65 = **(_QWORD **)(v65 + 88);
                        if ( *(_BYTE *)(v65 + 80) == 24 )
                          v65 = *(_QWORD *)(v65 + 88);
                        v236 = *(_QWORD *)(v151 + 88);
                        if ( *(_BYTE *)(v151 + 80) == 20 )
                          v236 = *(_QWORD *)(v236 + 176);
                        a1 = *(_QWORD *)(v236 + 152);
                        v237 = *(_QWORD *)(v65 + 88);
                        if ( *(_BYTE *)(v65 + 80) == 20 )
                          v237 = *(_QWORD *)(v237 + 176);
                        v238 = *(_QWORD *)(v237 + 152);
                        if ( a1 != v238 )
                        {
                          v239 = sub_8D97D0(a1, v238, 0, v150, v66);
                          v67 = (__int64 *)v289;
                          LOBYTE(v150) = v239 != 0;
                        }
                        m128i_i64 = (unsigned __int8)v150;
                      }
                      if ( (_DWORD)m128i_i64
                        && ((a1 = v63->m128i_i64[0], m128i_i64 = *v67, v63->m128i_i64[0] == *v67)
                         || (unsigned int)sub_8D97D0(a1, m128i_i64, 32, v150, v66)) )
                      {
                        v153 = (__int64 *)v301;
LABEL_373:
                        m128i_i64 = 3133;
                        if ( sub_6E53E0(5, 0xC3Du, a8) )
                        {
                          m128i_i64 = (__int64)a8;
                          sub_685490(0xC3Du, (FILE *)a8, v153[1]);
                        }
                        v154 = (__int64 *)v153[5];
                        v296 = 0;
                        v301 = (__m128i *)v301->m128i_i64[0];
                        *v153 = 0;
                        sub_725130(v154);
                        a1 = v153[15];
                        sub_82D8A0((_QWORD *)a1);
                        *v153 = (__int64)qword_4D03C68;
                        v48 = v301;
                        qword_4D03C68 = v153;
                        if ( v301[2].m128i_i8[0] )
                        {
                          m128i_i64 = (__int64)a8;
                          a1 = (__int64)v301;
                          sub_82B390((__int64)v301, (__int64)a8, (__int64 *)v65, v155, v156, v157);
                          goto LABEL_103;
                        }
                      }
                      else
                      {
                        v153 = (__int64 *)v301;
                        v48 = v301;
                        if ( dword_4F077BC )
                        {
                          if ( !(_DWORD)qword_4F077B4 )
                          {
                            if ( qword_4F077A8 )
                            {
                              v65 = v301->m128i_i64[1];
                              if ( *(_BYTE *)(v65 + 80) == 20 )
                              {
                                v233 = *(_QWORD *)(v301->m128i_i64[0] + 8);
                                if ( *(_BYTE *)(v233 + 80) == 20 )
                                {
                                  v234 = *(_QWORD *)(v65 + 88);
                                  if ( *(_QWORD *)(v234 + 88) && (*(_BYTE *)(v234 + 160) & 1) == 0 )
                                    v65 = *(_QWORD *)(v234 + 88);
                                  m128i_i64 = *(_QWORD *)(v233 + 88);
                                  if ( *(_QWORD *)(m128i_i64 + 88) && (*(_BYTE *)(m128i_i64 + 160) & 1) == 0 )
                                    v233 = *(_QWORD *)(m128i_i64 + 88);
                                  if ( v65 == v233 )
                                    goto LABEL_373;
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
LABEL_104:
          v68 = sub_82BD70(a1, m128i_i64, v65);
          if ( *(_QWORD *)(v68 + 1024) && (**(_BYTE **)(v68 + 1008) & 1) != 0 )
            goto LABEL_69;
          if ( !v50 )
            goto LABEL_76;
          if ( v245 )
          {
            m128i_i64 = v257;
            sub_831920(v255, v257, a6, a7, (__m128i *)a12, a8, a9, a11);
            if ( sub_827E90(v255, v257, v136, v137) )
            {
              m128i_i64 = a9;
              sub_878E70(0, a9, a10, 0, 0);
            }
            v249 = 0;
            *a15 = 1;
            if ( v48 )
              goto LABEL_70;
            goto LABEL_72;
          }
          v249 = v297;
          if ( (_DWORD)v297 )
          {
            *a15 = 1;
            sub_6E6260((_QWORD *)a12);
            if ( v48 )
            {
LABEL_356:
              v249 = 1;
              goto LABEL_70;
            }
LABEL_313:
            sub_6E5970((__int64)v50);
            v249 = 1;
            goto LABEL_72;
          }
          if ( !v48 )
          {
            if ( a5 )
              goto LABEL_72;
            m128i_i64 = dword_4F077BC;
            if ( dword_4F077BC
              && (dword_4F04C44 != -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0) )
            {
              if ( v255 == 15 || (m128i_i64 = v257, v162 = sub_6E9B70(v255, v257), sub_730030(v162)) )
              {
                *a15 = 1;
                v249 = sub_6E5430();
                if ( v249 )
                {
                  sub_684B10(0x15Du, a8, qword_4F064C0[v255]);
                  sub_831920(v255, v257, a6, a7, (__m128i *)a12, a8, a9, a11);
                  m128i_i64 = a9;
                  sub_878E70(0, a9, a10, 0, 0);
                  v249 = 0;
                }
                goto LABEL_72;
              }
            }
            *a15 = 1;
            if ( (unsigned int)sub_6E5430() )
            {
              v246 = sub_67D9E0(0x15Du, a8, qword_4F064C0[v255]);
              sub_82B170((__int64)v50, v255, v246);
              m128i_i64 = (__int64)v246;
              sub_87CA90(v300, v246);
            }
            sub_6E6260((_QWORD *)a12);
            goto LABEL_313;
          }
          v249 = v296;
          if ( v296 )
          {
            *a15 = 1;
            if ( (unsigned int)sub_6E5430() )
            {
              m128i_i64 = 0;
              v246 = sub_67D9E0(0x15Eu, a8, qword_4F064C0[v255]);
              sub_82E650(v48->m128i_i64, 0, (__int64)v50, v255, v246);
            }
            sub_6E6260((_QWORD *)a12);
            goto LABEL_356;
          }
          v70 = v48[9].m128i_u8[1];
          v71 = v48->m128i_i64[1];
          if ( (v70 & 1) == 0 )
          {
            v278 = 0;
            v72 = (__int64 *)v48[7].m128i_i64[1];
            goto LABEL_113;
          }
          v70 &= 2u;
          if ( v71 )
          {
            v69 = *(unsigned __int8 *)(v71 + 80);
            v163 = v48->m128i_i64[1];
            if ( (_BYTE)v69 == 16 )
            {
              v163 = **(_QWORD **)(v71 + 88);
              v69 = *(unsigned __int8 *)(v163 + 80);
            }
            if ( (_BYTE)v69 == 24 )
            {
              v163 = *(_QWORD *)(v163 + 88);
              v69 = *(unsigned __int8 *)(v163 + 80);
            }
            v164 = *(_QWORD *)(v163 + 88);
            if ( (_BYTE)v69 == 20 )
              v164 = *(_QWORD *)(v164 + 176);
            v255 = *(_BYTE *)(v164 + 176);
            if ( !(_BYTE)v70 )
            {
              v278 = 1;
              v72 = (__int64 *)v48[7].m128i_i64[1];
              goto LABEL_407;
            }
LABEL_564:
            v70 = 0;
            while ( 1 )
            {
              v235 = (__int64 *)*v50;
              *v50 = v70;
              v70 = (__int64)v50;
              if ( !v235 )
                break;
              v50 = v235;
            }
            v72 = (__int64 *)v48[7].m128i_i64[1];
            if ( v72 )
            {
              while ( 1 )
              {
                v70 = *v72;
                *v72 = (__int64)v235;
                v235 = v72;
                if ( !v70 )
                  break;
                v72 = (__int64 *)v70;
              }
            }
            v48[7].m128i_i64[1] = (__int64)v72;
            v278 = 1;
LABEL_113:
            if ( !v71 )
              goto LABEL_114;
LABEL_407:
            v165 = v71;
            *a15 = 1;
            v166 = *(_BYTE *)(v71 + 80);
            if ( v166 == 16 )
            {
              v165 = **(_QWORD **)(v71 + 88);
              v166 = *(_BYTE *)(v165 + 80);
            }
            if ( v166 == 24 )
              v165 = *(_QWORD *)(v165 + 88);
            v167 = *(_QWORD *)(*(_QWORD *)(v165 + 88) + 152LL);
            v293 = v167;
            if ( *(_BYTE *)(v167 + 140) == 12 )
            {
              do
                v167 = *(_QWORD *)(v167 + 160);
              while ( *(_BYTE *)(v167 + 140) == 12 );
              v293 = v167;
            }
            v284 = (*(_BYTE *)(v165 + 81) & 0x10) != 0;
            if ( (*(_BYTE *)(v165 + 81) & 0x10) != 0 )
              v284 = *(_QWORD *)(*(_QWORD *)(v293 + 168) + 40LL) != 0;
            v277 = v48[1].m128i_i64[1];
            if ( dword_4D047C8 )
            {
              v275 = v72;
              v218 = sub_827E90(a1, m128i_i64, v70, v69);
              v72 = v275;
              if ( v218 )
              {
                sub_878E70(v48[1].m128i_i64[0], a9, a10, v48[9].m128i_i8[1] & 1, (v48[9].m128i_i8[1] & 2) != 0);
                v72 = v275;
              }
            }
            if ( v255 == 15 && *(_BYTE *)(v165 + 80) == 10 )
            {
              v193 = *(_QWORD *)(v165 + 88);
              if ( (*(_BYTE *)(v193 + 206) & 0x10) == 0 && (*(_DWORD *)(v193 + 192) & 0x40400) == 0x40000 )
              {
                v280 = v72;
                sub_82F430(v165, v277, 0, a8, 1, 0, 0, 0, 0, v306.m128i_i32, 1);
                LODWORD(v304) = 0;
                if ( !v284 )
                {
                  sub_8316D0(0, v293);
                  BUG();
                }
                v194 = v50[3];
                v195 = (__int64)v280;
                LODWORD(v304) = 1;
                v281 = v194;
                v196 = (__m128i *)(v194 + 8);
                sub_82F0D0(v195, (_DWORD *)(v194 + 76));
                v287 = *v50;
                if ( *v50 && *(_BYTE *)(v287 + 8) == 3 )
                  v287 = sub_6BBB10(v50);
                v197 = *(__m128i **)(v165 + 64);
                sub_8316D0(v196, v293);
                if ( *(_BYTE *)(v281 + 25) != 1 || sub_6ED0A0((__int64)v196) )
                {
                  if ( *(_BYTE *)(v281 + 25) == 2 || sub_6ED0A0((__int64)v196) )
                  {
                    sub_6FF940(v196, v293, v198, v199, v200, v201);
                    sub_6F9270(v196, v293, v202, v203, v204, v205);
                  }
                }
                else
                {
                  sub_6ECF90((__int64)v196, 0);
                }
                if ( *(_BYTE *)(v287 + 8) )
                {
                  sub_839D30(v287, v197, 0, 1, 0, 0, 1, 0, 0, (__int64)&v306, 0, 0);
                  v206 = &v306;
                }
                else
                {
                  v206 = (__m128i *)(*(_QWORD *)(v287 + 24) + 8LL);
                }
                sub_847710(v206, v197, 0xA7u, (FILE *)a8);
                v211 = sub_6F6F40(v206, 0, v207, v208, v209, v210);
                v216 = (__int64 *)sub_6F6F40(v196, 0, v212, v213, v214, v215);
                v216[2] = v211;
                v217 = (__int64 *)sub_73DC30(0x49u, *v216, (__int64)v216);
                *((_BYTE *)v217 + 58) |= 1u;
                if ( dword_4D04810 )
                  *((_BYTE *)v217 + 60) |= 2u;
                m128i_i64 = a12;
                sub_6E7150(v217, a12);
                if ( sub_6E9250(a8) )
                  sub_6E6840(a12);
                goto LABEL_70;
              }
            }
            LODWORD(v304) = 0;
            if ( v284 )
            {
              v190 = v50[3];
              v274 = v72;
              LODWORD(v304) = 1;
              v285 = (_QWORD *)(v190 + 8);
              sub_82F0D0((__int64)v72, (_DWORD *)(v190 + 76));
              v177 = *v50;
              v191 = (__int64 **)v274;
              if ( *v50 )
              {
                if ( *(_BYTE *)(v177 + 8) != 3 || (v192 = sub_6BBB10(v50), v191 = (__int64 **)v274, (v177 = v192) != 0) )
                {
                  v72 = *v191;
                  v168 = (_QWORD *)v177;
                  v169 = **(_QWORD **)(v293 + 168);
                  goto LABEL_420;
                }
              }
            }
            else
            {
              v168 = v50;
              v285 = 0;
              v169 = **(_QWORD **)(v293 + 168);
LABEL_420:
              v265 = v169;
              v262 = v72;
              v170 = sub_84A490((__int64)v168, (__int64)v72, v169, *(_QWORD *)(v165 + 88));
              v171 = (_QWORD *)v265;
              v270 = v48;
              v268 = v71;
              v172 = v170;
              v266 = v50;
              v173 = v171;
              v174 = v165;
              v175 = v262;
              for ( jj = v170; ; jj = v176 )
              {
                if ( v173 )
                  v173 = (_QWORD *)*v173;
                if ( !*v168 )
                  break;
                if ( *(_BYTE *)(*v168 + 8LL) == 3 )
                {
                  v182 = sub_6BBB10(v168);
                  v175 = (__int64 *)*v175;
                  v168 = (_QWORD *)v182;
                  if ( !v182 )
                    break;
                }
                else
                {
                  v175 = (__int64 *)*v175;
                  v168 = (_QWORD *)*v168;
                }
                v176 = sub_84A490((__int64)v168, (__int64)v175, (__int64)v173, *(_QWORD *)(v174 + 88));
                if ( v172 )
                  *(_QWORD *)(jj + 16) = v176;
                else
                  v172 = v176;
              }
              v177 = v172;
              v71 = v268;
              v48 = v270;
              v50 = v266;
            }
            if ( (_DWORD)v304 )
            {
              v273 = v177;
              sub_831410(v293, (__int64)v285);
              v177 = v273;
            }
            v294 = (_QWORD *)v177;
            sub_8310F0(v71, v277, 0, a8, (int *)&v304, (__int64)v285, 0, 1u, v305);
            m128i_i64 = (__int64)v285;
            sub_7022F0(
              v305,
              v285,
              v294,
              v278,
              0,
              0,
              0,
              1,
              (__int64 *)&dword_4F077C8,
              a8,
              &dword_4F077C8,
              a12,
              (int *)&v297 + 1,
              v306.m128i_i64);
            v178 = v306.m128i_i64[0];
            if ( v306.m128i_i64[0] )
            {
              if ( (v48[9].m128i_i8[1] & 2) != 0 )
              {
                m128i_i64 = (*(_BYTE *)(v306.m128i_i64[0] + 60) & 2) != 0;
                *(_BYTE *)(v306.m128i_i64[0] + 60) = *(_BYTE *)(v306.m128i_i64[0] + 60) & 0xFC
                                                   | m128i_i64
                                                   | (2 * (*(_BYTE *)(v306.m128i_i64[0] + 60) & 1));
              }
              if ( *(_BYTE *)(a12 + 16) != 1 || v178 != *(_QWORD *)(a12 + 144) || (v48[9].m128i_i8[1] & 1) != 0 )
              {
                if ( v257 )
                {
                  v179 = a8;
                  v180 = (__int64 *)((char *)&a6[4].m128i_i64[1] + 4);
                }
                else
                {
                  v179 = (__int64 *)((char *)a6[4].m128i_i64 + 4);
                  v180 = (__int64 *)((char *)&a7[4].m128i_i64[1] + 4);
                  if ( a11 )
                    v180 = a11;
                }
                *(_QWORD *)(v178 + 36) = *v179;
                *(_QWORD *)(v178 + 44) = *v180;
                goto LABEL_439;
              }
            }
            else
            {
LABEL_439:
              if ( (v48[9].m128i_i8[1] & 1) != 0 )
              {
                if ( a14 )
                {
                  m128i_i64 = (__int64)a14;
                  *a14 = _mm_loadu_si128(v48);
                  a14[1] = _mm_loadu_si128(v48 + 1);
                  a14[2] = _mm_loadu_si128(v48 + 2);
                  a14[3] = _mm_loadu_si128(v48 + 3);
                  a14[4] = _mm_loadu_si128(v48 + 4);
                  a14[5] = _mm_loadu_si128(v48 + 5);
                  a14[6] = _mm_loadu_si128(v48 + 6);
                  a14[7] = _mm_loadu_si128(v48 + 7);
                  a14[8] = _mm_loadu_si128(v48 + 8);
                  a14[9].m128i_i64[0] = v48[9].m128i_i64[0];
                }
                else
                {
                  v219 = v48->m128i_i64[1];
                  v220 = *(_BYTE *)(v219 + 80);
                  if ( v220 == 16 )
                  {
                    v219 = **(_QWORD **)(v219 + 88);
                    v220 = *(_BYTE *)(v219 + 80);
                  }
                  if ( v220 == 24 )
                  {
                    v219 = *(_QWORD *)(v219 + 88);
                    v220 = *(_BYTE *)(v219 + 80);
                  }
                  v221 = *(_QWORD *)(v219 + 88);
                  if ( v220 == 20 )
                    v221 = *(_QWORD *)(v221 + 176);
                  for ( kk = *(_QWORD *)(v221 + 152); *(_BYTE *)(kk + 140) == 12; kk = *(_QWORD *)(kk + 160) )
                    ;
                  v223 = *(_QWORD *)(kk + 160);
                  if ( (unsigned int)sub_8D29A0(v223)
                    || (m128i_i64 = v255, (unsigned __int8)(v255 - 30) > 3u) && (unsigned __int8)(v255 - 16) > 1u )
                  {
                    v225 = v48[9].m128i_i8[1];
                    v226 = sub_82BD70(v223, m128i_i64, v224);
                    v227 = (_BYTE *)(*(_QWORD *)(v226 + 1008) + 40 * (*(_QWORD *)(v226 + 1024) - 1LL));
                    *v227 |= 2u;
                    m128i_i64 = v306.m128i_i64[0];
                    sub_69BA40(v241, (__m128i *)v306.m128i_i64[0], a9, (const __m128i *)a12, (v225 & 2) != 0, v228);
                    v230 = sub_82BD70(v241, m128i_i64, v229);
                    v231 = (_BYTE *)(*(_QWORD *)(v230 + 1008) + 40 * (*(_QWORD *)(v230 + 1024) - 1LL));
                    *v231 &= ~2u;
                  }
                  else
                  {
                    if ( (unsigned int)sub_8D2930(*(_QWORD *)(kk + 160)) )
                    {
                      v232 = (_DWORD)qword_4F077B4 == 0 ? 7 : 5;
                    }
                    else
                    {
                      sub_6E6840(a12);
                      v232 = 8;
                    }
                    m128i_i64 = 3021;
                    sub_6E5CD0(v232, 0xBCDu, (FILE *)a8, v219);
                  }
                }
              }
            }
          }
          else
          {
            v255 = v48[9].m128i_u8[0];
            if ( (_BYTE)v70 )
              goto LABEL_564;
            v72 = (__int64 *)v48[7].m128i_i64[1];
LABEL_114:
            if ( !a14 )
            {
              if ( (unsigned __int8)(v255 - 35) <= 1u )
              {
                v74 = 0;
                v73 = 1;
              }
              else
              {
                v73 = v255 == 44;
                v74 = v73;
              }
              m128i_i64 = (__int64)v48;
              v290 = v72;
              sub_845950(a6, (__int64)v48, 1, v74, (__int64)v72);
              if ( !v257 )
              {
                m128i_i64 = (__int64)v48;
                sub_845950(a7, (__int64)v48, 2, v73, *v290);
                v249 = 0;
              }
            }
          }
          do
          {
LABEL_70:
            v51 = v48;
            v48 = (__m128i *)v48->m128i_i64[0];
            sub_725130((__int64 *)v51[2].m128i_i64[1]);
            sub_82D8A0((_QWORD *)v51[7].m128i_i64[1]);
            v51->m128i_i64[0] = (__int64)qword_4D03C68;
            qword_4D03C68 = v51->m128i_i64;
          }
          while ( v48 );
LABEL_71:
          if ( v249 )
            goto LABEL_313;
LABEL_72:
          if ( !v257 && a7[1].m128i_i8[0] == 5 )
            *v50 = 0;
          a1 = (__int64)v50;
          sub_6E1990(v50);
LABEL_76:
          if ( !v246 )
            goto LABEL_36;
          v52 = sub_82BD70(a1, m128i_i64, v25);
          if ( !(unsigned int)sub_6E5430() )
            goto LABEL_37;
          if ( *(__int64 *)(v52 + 1024) > 1 )
            goto LABEL_37;
          v25 = *(_BYTE **)(v52 + 1008);
          if ( (*v25 & 1) != 0 )
            goto LABEL_37;
          *v25 |= 1u;
          v27 = a6->m128i_i64[0];
        }
        if ( v257 )
          goto LABEL_67;
        if ( !(unsigned int)sub_8D3A70(a7->m128i_i64[0]) )
        {
          a1 = dword_4D047E4;
          if ( !dword_4D047E4 || (a1 = a7->m128i_i64[0], !(unsigned int)sub_8D2870(a7->m128i_i64[0])) )
          {
            if ( a7[1].m128i_i8[0] )
            {
              v47 = a7->m128i_i64[0];
              for ( mm = *(_BYTE *)(a7->m128i_i64[0] + 140); mm == 12; mm = *(_BYTE *)(v47 + 140) )
                v47 = *(_QWORD *)(v47 + 160);
              if ( mm )
              {
LABEL_67:
                v48 = v301;
                v49 = sub_82BD70(a1, dword_4F07508, v47);
                if ( !*(_QWORD *)(v49 + 1024) )
                  goto LABEL_76;
                v50 = 0;
                if ( (**(_BYTE **)(v49 + 1008) & 1) == 0 )
                  goto LABEL_76;
LABEL_69:
                if ( v48 )
                  goto LABEL_70;
                goto LABEL_71;
              }
            }
          }
        }
        v298 = 0;
        v299 = 0;
        v53 = (a6[1].m128i_i8[2] & 2) != 0;
        v50 = (__int64 *)sub_6E3060(a6);
LABEL_306:
        if ( a7[1].m128i_i8[0] == 5 )
          v247 = a7[9].m128i_i64[0];
        else
          v247 = sub_6E3060(a7);
        *v50 = v247;
        goto LABEL_83;
      }
    }
  }
LABEL_33:
  a1 = v16;
  if ( !sub_7D3880(v16) )
    goto LABEL_34;
  if ( v248 && a6[1].m128i_i8[0] )
  {
    v45 = a6->m128i_i64[0];
    for ( nn = *(_BYTE *)(a6->m128i_i64[0] + 140); nn == 12; nn = *(_BYTE *)(v45 + 140) )
      v45 = *(_QWORD *)(v45 + 160);
    if ( nn )
    {
      a1 = v27;
      if ( !(unsigned int)sub_8D3A70(v27) )
      {
LABEL_34:
        if ( a5 )
          goto LABEL_36;
      }
    }
  }
  *a15 = 1;
  sub_6E6260((_QWORD *)a12);
  a1 = (__int64)a6;
  sub_6E6450((__int64)a6);
  if ( (_DWORD)m128i_i64 )
  {
LABEL_36:
    v246 = 0;
  }
  else
  {
    a1 = (__int64)a7;
    sub_6E6450((__int64)a7);
    v246 = 0;
  }
LABEL_37:
  if ( *a15 )
  {
    if ( HIDWORD(v297) || (m128i_i64 = a12, sub_6E26D0(2, a12), a1 = (unsigned int)*a15, (_DWORD)a1) )
      *(_QWORD *)(a12 + 68) = *a8;
  }
  if ( v246 )
  {
    v33 = sub_82BD70(a1, m128i_i64, v25);
    v34 = (const __m128i *)(*(_QWORD *)(v33 + 1008) + 8 * (5LL * *(_QWORD *)(v33 + 1024) - 5));
    if ( v34[1].m128i_i64[0] )
    {
      m128i_i64 = (__int64)v34[1].m128i_i64;
      sub_67E370((__int64)v246, v34 + 1);
    }
    a1 = (__int64)v246;
    sub_685910((__int64)v246, (FILE *)m128i_i64);
  }
LABEL_46:
  v37 = sub_82BD70(a1, m128i_i64, v25);
  v40 = *(_QWORD *)(*(_QWORD *)(v37 + 1008) + 8 * (5LL * *(_QWORD *)(v37 + 1024) - 5) + 32);
  if ( v40 )
  {
    sub_823A00(*(_QWORD *)v40, 16LL * (unsigned int)(*(_DWORD *)(v40 + 8) + 1), v35, v36, v38, v39);
    sub_823A00(v40, 16, v41, v42, v43, v44);
  }
  --*(_QWORD *)(v37 + 1024);
}
