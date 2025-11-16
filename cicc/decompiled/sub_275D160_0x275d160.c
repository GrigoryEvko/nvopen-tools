// Function: sub_275D160
// Address: 0x275d160
//
__int64 __fastcall sub_275D160(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        const __m128i *a4,
        unsigned __int8 *a5,
        _DWORD *a6,
        unsigned int *a7,
        char a8,
        _DWORD *a9,
        char a10)
{
  unsigned int v10; // r13d
  __int64 v12; // rbx
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 *v16; // rax
  unsigned int v17; // eax
  unsigned __int8 *v18; // r13
  __int8 v19; // r15
  __int64 v20; // r14
  unsigned __int8 *v21; // r14
  __int64 v22; // rcx
  int v23; // eax
  bool v24; // al
  int v25; // r14d
  char v26; // al
  char v27; // al
  __int64 v28; // r14
  unsigned __int8 *v29; // r12
  __int64 v30; // r12
  char v31; // al
  char v32; // al
  __m128i v33; // xmm2
  __m128i v34; // xmm3
  bool v35; // r14
  __int64 v36; // rdi
  __int64 v37; // rsi
  __int64 v38; // rax
  int v39; // ecx
  __int64 v40; // r8
  int v41; // ecx
  unsigned int v42; // edx
  __int64 *v43; // rax
  __int64 v44; // r10
  __int64 v45; // rdi
  unsigned int v46; // edx
  __int64 *v47; // rax
  __int64 v48; // r10
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // rcx
  __int64 *v52; // rax
  __int64 v53; // r14
  char v54; // al
  __int64 v55; // rsi
  bool v56; // al
  __int64 v57; // r14
  char v58; // al
  __int64 v59; // rax
  _QWORD *v60; // rdi
  __m128i v61; // xmm6
  __m128i v62; // xmm4
  __int64 v63; // rax
  _QWORD *v64; // rdi
  __m128i v65; // xmm4
  __m128i v66; // xmm6
  __int64 v67; // rcx
  _QWORD *v68; // rax
  __int64 v69; // rax
  _QWORD *v70; // rax
  int v71; // eax
  __int8 v72; // al
  _QWORD *v73; // rax
  _QWORD *v74; // rdx
  __int64 v75; // rdi
  __int64 v76; // rsi
  __int64 v77; // rax
  int v78; // ecx
  __int64 v79; // r8
  int v80; // ecx
  unsigned int v81; // edx
  __int64 *v82; // rax
  __int64 v83; // r10
  __int64 v84; // rdi
  unsigned int v85; // edx
  __int64 *v86; // rax
  __int64 v87; // r10
  int v88; // eax
  int v89; // r13d
  bool v90; // zf
  unsigned int v91; // eax
  __int64 *v92; // rdx
  __int64 v93; // rdx
  __m128i v94; // xmm6
  __m128i v95; // xmm4
  __m128i v96; // xmm5
  __int64 v97; // rax
  __int64 v98; // rax
  __int64 v99; // r8
  __int64 v100; // r9
  __int64 v101; // rdx
  __int64 v102; // rcx
  __int64 v103; // r12
  unsigned __int64 v104; // rsi
  unsigned __int64 v105; // rcx
  __int64 *v106; // r15
  unsigned __int8 *v107; // r14
  __int64 *v108; // r13
  __int64 v109; // rsi
  char v110; // r13
  char v111; // al
  __int64 v112; // rsi
  __int64 v113; // rdi
  __int64 v114; // rax
  int v115; // ecx
  __int64 v116; // r8
  int v117; // ecx
  unsigned int v118; // edx
  __int64 *v119; // rax
  __int64 v120; // r11
  __int64 v121; // rdi
  unsigned int v122; // edx
  __int64 *v123; // rax
  __int64 v124; // r11
  unsigned int v125; // eax
  unsigned int v126; // edx
  __int64 v127; // r8
  __int64 v128; // rsi
  unsigned int v129; // edi
  unsigned int v130; // r12d
  __int64 *v131; // rax
  __int64 v132; // r9
  unsigned int v133; // r15d
  __int64 v134; // r8
  unsigned int v135; // r12d
  __int64 *v136; // rax
  __int64 v137; // r9
  unsigned int v138; // eax
  __int64 v139; // rdx
  unsigned __int8 **v140; // rcx
  __int64 v141; // r9
  __int64 *v142; // rax
  unsigned __int64 v143; // r14
  __int64 v144; // rax
  __int64 *v145; // r13
  __int64 v146; // rdx
  char v147; // al
  __int64 *v148; // rax
  char v149; // al
  __int64 v150; // rax
  __m128i v151; // xmm7
  _QWORD *v152; // rdi
  __m128i v153; // xmm0
  __m128i v154; // xmm4
  __int64 v155; // rcx
  __int64 v156; // rcx
  int v157; // edx
  unsigned __int8 **v158; // rax
  __int64 *v159; // rdx
  __int64 **v160; // rax
  __int64 **v161; // r14
  __int64 *v162; // rcx
  __int64 **v163; // r13
  __int64 *v164; // rcx
  __int64 v165; // rax
  __int64 *v166; // rsi
  unsigned __int8 *v167; // rax
  __int64 v168; // r8
  __int64 v169; // rdi
  __int64 v170; // r11
  unsigned int v171; // r10d
  __int64 v172; // rdx
  __int64 v173; // rcx
  __int64 v174; // r11
  unsigned int v175; // r8d
  __int64 v176; // rcx
  __int64 v177; // r8
  __int64 v178; // rsi
  char v179; // al
  __int8 v180; // dl
  _QWORD *v181; // rax
  __int64 v182; // rcx
  __int64 v183; // r8
  __int64 v184; // rsi
  _QWORD *v185; // rax
  __int64 **v186; // rax
  int v187; // eax
  int v188; // r9d
  __int64 v189; // rax
  int v190; // eax
  int v191; // r9d
  __int64 v192; // rax
  int v193; // eax
  int v194; // r9d
  __int64 v195; // rax
  int v196; // eax
  int v197; // r9d
  __int64 v198; // rax
  int v199; // eax
  int v200; // r9d
  __int64 v201; // rax
  int v202; // eax
  int v203; // r9d
  __int64 v204; // rax
  unsigned __int8 *v205; // rdi
  int v206; // r15d
  __int64 v207; // rdx
  __int64 v208; // r13
  __int64 v209; // rbx
  _QWORD *v210; // rax
  __int64 v211; // rdx
  unsigned int v212; // eax
  __int64 v213; // rsi
  __int64 v214; // rdx
  unsigned int v215; // eax
  __int64 v216; // rbx
  __int64 v217; // rdx
  __int64 v218; // rax
  __int64 *v219; // r15
  __int64 v220; // rax
  __int64 *v221; // rbx
  unsigned __int64 v222; // rax
  int v223; // eax
  int v224; // r10d
  __int64 v225; // [rsp-8h] [rbp-508h]
  unsigned __int8 v226; // [rsp+0h] [rbp-500h]
  __int64 v227; // [rsp+0h] [rbp-500h]
  unsigned __int8 v228; // [rsp+10h] [rbp-4F0h]
  __int64 v229; // [rsp+20h] [rbp-4E0h]
  bool v230; // [rsp+20h] [rbp-4E0h]
  unsigned int v231; // [rsp+20h] [rbp-4E0h]
  __int64 v232; // [rsp+30h] [rbp-4D0h]
  __int64 v233; // [rsp+38h] [rbp-4C8h]
  __int64 v235; // [rsp+48h] [rbp-4B8h]
  unsigned __int64 v236; // [rsp+50h] [rbp-4B0h]
  char v240; // [rsp+78h] [rbp-488h]
  unsigned int v241; // [rsp+78h] [rbp-488h]
  __int64 v242; // [rsp+78h] [rbp-488h]
  __int64 v243; // [rsp+98h] [rbp-468h] BYREF
  __int64 v244; // [rsp+A0h] [rbp-460h] BYREF
  __int64 v245; // [rsp+A8h] [rbp-458h]
  __m128i v246; // [rsp+B0h] [rbp-450h] BYREF
  __m128i v247; // [rsp+C0h] [rbp-440h] BYREF
  __m128i v248; // [rsp+D0h] [rbp-430h] BYREF
  __m128i v249; // [rsp+E0h] [rbp-420h] BYREF
  __m128i v250; // [rsp+F0h] [rbp-410h] BYREF
  __m128i v251; // [rsp+100h] [rbp-400h] BYREF
  __int64 v252; // [rsp+110h] [rbp-3F0h]
  unsigned __int8 *v253; // [rsp+120h] [rbp-3E0h] BYREF
  __int64 v254[2]; // [rsp+128h] [rbp-3D8h] BYREF
  __int64 v255; // [rsp+138h] [rbp-3C8h]
  unsigned __int8 *v256; // [rsp+140h] [rbp-3C0h]
  __int64 v257; // [rsp+148h] [rbp-3B8h]
  _BYTE v258[16]; // [rsp+150h] [rbp-3B0h] BYREF
  __int64 v259; // [rsp+160h] [rbp-3A0h] BYREF
  __int64 *v260; // [rsp+168h] [rbp-398h]
  __int64 v261; // [rsp+170h] [rbp-390h]
  int v262; // [rsp+178h] [rbp-388h]
  char v263; // [rsp+17Ch] [rbp-384h]
  __int64 v264; // [rsp+180h] [rbp-380h] BYREF
  __m128i v265; // [rsp+200h] [rbp-300h] BYREF
  __m128i v266; // [rsp+210h] [rbp-2F0h]
  __m128i v267; // [rsp+220h] [rbp-2E0h] BYREF
  char v268; // [rsp+230h] [rbp-2D0h]
  unsigned __int8 *v269; // [rsp+2A0h] [rbp-260h] BYREF
  __int64 v270; // [rsp+2A8h] [rbp-258h] BYREF
  _BYTE v271[256]; // [rsp+2B0h] [rbp-250h] BYREF
  __m128i v272; // [rsp+3B0h] [rbp-150h] BYREF
  __m128i v273; // [rsp+3C0h] [rbp-140h] BYREF
  __m128i v274; // [rsp+3D0h] [rbp-130h] BYREF
  __int64 v275; // [rsp+3E0h] [rbp-120h]

  if ( !*a6 )
    goto LABEL_3;
  v10 = *a7;
  if ( !*a7 )
    goto LABEL_3;
  v12 = a1;
  v236 = *(_QWORD *)(a2 + 72);
  if ( !(_BYTE)qword_4FFA388 )
    goto LABEL_6;
  v68 = (_QWORD *)(a2 - 64);
  if ( *(_BYTE *)a2 == 26 )
    v68 = (_QWORD *)(a2 - 32);
  if ( a3 == *v68 )
  {
    v252 = 0;
    v249 = 0;
    v240 = sub_B46420(v236) ^ 1;
    v69 = *(_QWORD *)(a1 + 784);
    v250 = 0;
    v15 = *(_QWORD *)(v69 + 128);
    v251 = 0;
    if ( a3 == v15 )
    {
LABEL_104:
      if ( v240 )
      {
        v70 = (_QWORD *)(a2 - 64);
        if ( *(_BYTE *)a2 == 26 )
          v70 = (_QWORD *)(a2 - 32);
        if ( *v70 != v15 )
        {
          sub_AC2B30(a2 - 32, v15);
          if ( *(_BYTE *)v15 == 27 )
            v71 = *(_DWORD *)(v15 + 80);
          else
            v71 = *(_DWORD *)(v15 + 72);
          *(_DWORD *)(a2 + 84) = v71;
        }
      }
      goto LABEL_3;
    }
  }
  else
  {
LABEL_6:
    v14 = *(_QWORD *)(a1 + 784);
    v252 = 0;
    v249 = 0;
    v250 = 0;
    v251 = 0;
    if ( a3 == *(_QWORD *)(v14 + 128) )
      goto LABEL_3;
    v240 = 0;
  }
  v232 = a3;
  v15 = a3;
  while ( 1 )
  {
    v16 = &qword_4FFA4C0;
    if ( *(_QWORD *)(a2 + 64) == *(_QWORD *)(v15 + 64) )
      v16 = &qword_4FFA5A0;
    v17 = *((_DWORD *)v16 + 34);
    if ( v17 >= v10 )
      goto LABEL_3;
    *a7 = v10 - v17;
    if ( *(_BYTE *)v15 == 28 )
    {
      v244 = v15;
      LOBYTE(v245) = 1;
      return v244;
    }
    v18 = *(unsigned __int8 **)(v15 + 72);
    if ( (unsigned __int8)sub_CF7590(a5, &v265) )
    {
      v19 = v265.m128i_i8[0];
      if ( v265.m128i_i8[0] )
      {
        v269 = a5;
        LOBYTE(v270) = 1;
        sub_275ABE0((__int64)&v272, v12 + 1432, (__int64 *)&v269, &v270);
        v20 = v273.m128i_i64[0];
        if ( v274.m128i_i8[0] )
        {
          v72 = sub_D13FA0((__int64)a5, 0, 0);
          *(_BYTE *)(v20 + 8) = v72;
          v19 = v72;
        }
        else
        {
          v19 = *(_BYTE *)(v273.m128i_i64[0] + 8);
        }
      }
      v21 = *(unsigned __int8 **)(v15 + 72);
      if ( (unsigned __int8)(*v21 - 34) <= 0x33u )
      {
        v22 = 0x8000000000041LL;
        if ( _bittest64(&v22, (unsigned int)*v21 - 34) )
        {
          if ( sub_B49EA0(*(_QWORD *)(v15 + 72)) )
            goto LABEL_67;
        }
      }
      if ( (unsigned __int8)sub_B46790(v21, 0) == 1 && !v19 )
        goto LABEL_67;
      goto LABEL_22;
    }
    v21 = *(unsigned __int8 **)(v15 + 72);
    v23 = *v21;
    if ( (unsigned __int8)(v23 - 34) <= 0x33u )
    {
      v51 = 0x8000000000041LL;
      if ( !_bittest64(&v51, (unsigned int)(v23 - 34)) )
      {
LABEL_23:
        if ( (_BYTE)v23 == 64 )
          goto LABEL_67;
        goto LABEL_24;
      }
      if ( sub_B49EA0(*(_QWORD *)(v15 + 72)) )
        goto LABEL_67;
LABEL_22:
      LOBYTE(v23) = *v21;
      goto LABEL_23;
    }
LABEL_24:
    if ( sub_2753FC0((__int64)v21) )
      goto LABEL_67;
    if ( !a5
      || !(unsigned __int8)sub_CF7590(a5, &v265)
      || v265.m128i_i8[0]
      && ((v269 = a5,
           LOBYTE(v270) = 1,
           sub_275ABE0((__int64)&v272, v12 + 1432, (__int64 *)&v269, &v270),
           v53 = v273.m128i_i64[0],
           v274.m128i_i8[0])
        ? (v54 = sub_D13FA0((__int64)a5, 0, 0), *(_BYTE *)(v53 + 8) = v54)
        : (v54 = *(_BYTE *)(v273.m128i_i64[0] + 8)),
          v54) )
    {
      v55 = *(_QWORD *)(v236 + 40);
      if ( v55 != *((_QWORD *)v18 + 5) )
      {
        v56 = *(_DWORD *)(v12 + 1516) != *(_DWORD *)(v12 + 1520);
        goto LABEL_77;
      }
      if ( !*(_BYTE *)(v12 + 1524) )
      {
        v56 = sub_C8CA60(v12 + 1496, v55) != 0;
LABEL_77:
        if ( v56 )
          goto LABEL_3;
        if ( !(unsigned __int8)sub_B46790(v18, 0) )
          goto LABEL_29;
        goto LABEL_79;
      }
      v73 = *(_QWORD **)(v12 + 1504);
      v74 = &v73[*(unsigned int *)(v12 + 1516)];
      if ( v73 != v74 )
      {
        while ( v55 != *v73 )
        {
          if ( v74 == ++v73 )
            goto LABEL_28;
        }
LABEL_3:
        LOBYTE(v245) = 0;
        return v244;
      }
    }
LABEL_28:
    if ( !(unsigned __int8)sub_B46790(v18, 0) )
      goto LABEL_29;
LABEL_79:
    if ( !(unsigned __int8)sub_CF7590(a5, &v265) )
      goto LABEL_3;
    if ( v265.m128i_i8[0] )
    {
      v269 = a5;
      LOBYTE(v270) = 1;
      sub_275ABE0((__int64)&v272, v12 + 1432, (__int64 *)&v269, &v270);
      v57 = v273.m128i_i64[0];
      if ( v274.m128i_i8[0] )
      {
        v58 = sub_D13FA0((__int64)a5, 0, 0);
        *(_BYTE *)(v57 + 8) = v58;
      }
      else
      {
        v58 = *(_BYTE *)(v273.m128i_i64[0] + 8);
      }
      if ( v58 )
        goto LABEL_3;
    }
LABEL_29:
    v24 = sub_B46500(v18);
    v25 = *v18;
    if ( v24 )
    {
      switch ( (_BYTE)v25 )
      {
        case '=':
        case '>':
          v26 = byte_3F70480[8 * ((*((_WORD *)v18 + 1) >> 7) & 7) + 2];
          break;
        case 'B':
          v26 = byte_3F70480[8 * ((*((_WORD *)v18 + 1) >> 1) & 7) + 2];
          break;
        case 'A':
          if ( byte_3F70480[8 * ((*((_WORD *)v18 + 1) >> 2) & 7) + 2]
            || byte_3F70480[8 * ((*((_WORD *)v18 + 1) >> 5) & 7) + 2] )
          {
            goto LABEL_3;
          }
          if ( sub_2753FC0((__int64)v18) )
            goto LABEL_37;
          goto LABEL_205;
        default:
          BUG();
      }
      if ( v26 )
        goto LABEL_3;
LABEL_33:
      if ( sub_2753FC0((__int64)v18) )
        goto LABEL_37;
      if ( (_BYTE)v25 == 62 )
      {
        v27 = byte_3F70480[8 * ((*((_WORD *)v18 + 1) >> 7) & 7) + 2];
        goto LABEL_36;
      }
LABEL_205:
      if ( !(unsigned __int8)sub_B46420((__int64)v18) )
        goto LABEL_37;
      v143 = (unsigned int)(v25 - 34);
      if ( (unsigned __int8)v143 > 0x33u || (v144 = 0x8000000000041LL, !_bittest64(&v144, v143)) )
      {
LABEL_92:
        v60 = *(_QWORD **)(v12 + 104);
        LOBYTE(v275) = 1;
        v61 = _mm_loadu_si128(a4 + 1);
        v272 = _mm_loadu_si128(a4);
        v62 = _mm_loadu_si128(a4 + 2);
        v273 = v61;
        v274 = v62;
        v27 = sub_CF63E0(v60, v18, &v272, v12 + 112) & 1;
LABEL_36:
        if ( v27 )
          goto LABEL_3;
        goto LABEL_37;
      }
      goto LABEL_91;
    }
    if ( (_BYTE)v25 != 85 )
      goto LABEL_33;
    v59 = *((_QWORD *)v18 - 4);
    if ( v59 && !*(_BYTE *)v59 && *(_QWORD *)(v59 + 24) == *((_QWORD *)v18 + 10) && (*(_BYTE *)(v59 + 33) & 0x20) != 0
      || sub_2753FC0((__int64)v18)
      || !(unsigned __int8)sub_B46420((__int64)v18) )
    {
      goto LABEL_37;
    }
LABEL_91:
    if ( !sub_B49EA0((__int64)v18) )
      goto LABEL_92;
LABEL_37:
    v28 = *(_QWORD *)(v15 + 16);
    if ( !v28 )
      goto LABEL_48;
    v229 = v15;
    do
    {
      while ( 1 )
      {
        v29 = *(unsigned __int8 **)(v28 + 24);
        if ( (unsigned int)*v29 - 26 <= 1 && !sub_1041420(*(_QWORD *)(v12 + 784), v232, *(_QWORD *)(v28 + 24)) )
        {
          v30 = *((_QWORD *)v29 + 9);
          if ( !sub_2753FC0(v30) )
          {
            if ( *(_BYTE *)v30 == 62 )
            {
              v31 = byte_3F70480[8 * ((*(_WORD *)(v30 + 2) >> 7) & 7) + 2];
              goto LABEL_45;
            }
            v228 = *(_BYTE *)v30;
            if ( (unsigned __int8)sub_B46420(v30) )
            {
              if ( (unsigned __int8)(v228 - 34) > 0x33u )
                break;
              v63 = 0x8000000000041LL;
              if ( !_bittest64(&v63, (unsigned int)v228 - 34) || !sub_B49EA0(v30) )
                break;
            }
          }
        }
        v28 = *(_QWORD *)(v28 + 8);
        if ( !v28 )
          goto LABEL_47;
      }
      v64 = *(_QWORD **)(v12 + 104);
      LOBYTE(v275) = 1;
      v65 = _mm_loadu_si128(a4 + 1);
      v272 = _mm_loadu_si128(a4);
      v66 = _mm_loadu_si128(a4 + 2);
      v273 = v65;
      v274 = v66;
      v31 = sub_CF63E0(v64, (unsigned __int8 *)v30, &v272, v12 + 112) & 1;
LABEL_45:
      if ( v31 )
        goto LABEL_3;
      v28 = *(_QWORD *)(v28 + 8);
    }
    while ( v28 );
LABEL_47:
    v15 = v229;
LABEL_48:
    v32 = sub_B46490((__int64)v18);
    if ( v32 )
    {
      if ( (unsigned __int8)(*v18 - 34) <= 0x33u && (v67 = 0x8000000000041LL, _bittest64(&v67, (unsigned int)*v18 - 34)) )
      {
        sub_D67230(&v272, v18, *(__int64 **)(v12 + 808));
        v32 = v275;
      }
      else
      {
        sub_D66840(&v272, v18);
        v32 = v275;
      }
    }
    LOBYTE(v275) = v32;
    v33 = _mm_loadu_si128(&v273);
    v34 = _mm_loadu_si128(&v274);
    v249 = _mm_loadu_si128(&v272);
    v250 = v33;
    v252 = v275;
    v251 = v34;
    if ( !v32 )
      goto LABEL_67;
    v35 = sub_27543D0(v18);
    if ( !v35 )
      goto LABEL_67;
    v36 = *((_QWORD *)v18 + 5);
    v37 = *(_QWORD *)(v236 + 40);
    if ( v36 == v37 )
      goto LABEL_61;
    v38 = *(_QWORD *)(v12 + 824);
    v39 = *(_DWORD *)(v38 + 24);
    v40 = *(_QWORD *)(v38 + 8);
    if ( !v39 )
      goto LABEL_60;
    v41 = v39 - 1;
    v42 = v41 & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
    v43 = (__int64 *)(v40 + 16LL * v42);
    v44 = *v43;
    if ( v36 != *v43 )
    {
      v187 = 1;
      while ( v44 != -4096 )
      {
        v188 = v187 + 1;
        v189 = v41 & (v42 + v187);
        v42 = v189;
        v43 = (__int64 *)(v40 + 16 * v189);
        v44 = *v43;
        if ( v36 == *v43 )
          goto LABEL_56;
        v187 = v188;
      }
LABEL_60:
      if ( sub_2753D10(v12, (unsigned __int8 *)v249.m128i_i64[0]) )
        goto LABEL_61;
      goto LABEL_67;
    }
LABEL_56:
    v45 = v43[1];
    if ( *(_BYTE *)(v12 + 832) == 1 || !v45 )
      goto LABEL_60;
    v46 = v41 & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
    v47 = (__int64 *)(v40 + 16LL * v46);
    v48 = *v47;
    if ( v37 != *v47 )
    {
      v190 = 1;
      while ( v48 != -4096 )
      {
        v191 = v190 + 1;
        v192 = v41 & (v46 + v190);
        v46 = v192;
        v47 = (__int64 *)(v40 + 16 * v192);
        v48 = *v47;
        if ( v37 == *v47 )
          goto LABEL_59;
        v190 = v191;
      }
      goto LABEL_60;
    }
LABEL_59:
    if ( v45 != v47[1] )
      goto LABEL_60;
LABEL_61:
    if ( !a8 )
      break;
    if ( sub_2757210(v12, (unsigned __int8 **)&v249, (__int64)v18, v236) )
      goto LABEL_63;
LABEL_67:
    v240 = 0;
LABEL_68:
    v52 = (__int64 *)(v15 - 64);
    if ( *(_BYTE *)v15 == 26 )
      v52 = (__int64 *)(v15 - 32);
    v15 = *v52;
    if ( *(_QWORD *)(*(_QWORD *)(v12 + 784) + 128LL) == *v52 )
      goto LABEL_104;
    v10 = *a7;
  }
  v269 = 0;
  v272.m128i_i64[0] = 0;
  v75 = *((_QWORD *)v18 + 5);
  v76 = *(_QWORD *)(v236 + 40);
  if ( v75 == v76 )
    goto LABEL_128;
  v77 = *(_QWORD *)(v12 + 824);
  v78 = *(_DWORD *)(v77 + 24);
  v79 = *(_QWORD *)(v77 + 8);
  if ( v78 )
  {
    v80 = v78 - 1;
    v81 = v80 & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
    v82 = (__int64 *)(v79 + 16LL * v81);
    v83 = *v82;
    if ( v75 == *v82 )
    {
LABEL_124:
      v84 = v82[1];
      if ( *(_BYTE *)(v12 + 832) != 1 && v84 )
      {
        v85 = v80 & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4));
        v86 = (__int64 *)(v79 + 16LL * v85);
        v87 = *v86;
        if ( v76 == *v86 )
        {
LABEL_127:
          if ( v84 == v86[1] )
            goto LABEL_128;
        }
        else
        {
          v199 = 1;
          while ( v87 != -4096 )
          {
            v200 = v199 + 1;
            v201 = v80 & (v85 + v199);
            v85 = v201;
            v86 = (__int64 *)(v79 + 16 * v201);
            v87 = *v86;
            if ( v76 == *v86 )
              goto LABEL_127;
            v199 = v200;
          }
        }
      }
    }
    else
    {
      v202 = 1;
      while ( v83 != -4096 )
      {
        v203 = v202 + 1;
        v204 = v80 & (v81 + v202);
        v81 = v204;
        v82 = (__int64 *)(v79 + 16 * v204);
        v83 = *v82;
        if ( v75 == *v82 )
          goto LABEL_124;
        v202 = v203;
      }
    }
  }
  if ( sub_2753D10(v12, (unsigned __int8 *)v249.m128i_i64[0]) )
  {
LABEL_128:
    v88 = sub_27567E0((__int64 *)v12, v236, (__int64)v18, (__int64)a4, (__int64)&v249, (__int64 *)&v269, v272.m128i_i64);
    v89 = v88;
    v49 = v225;
    if ( v240 )
    {
      v90 = v88 == 5;
      v91 = v88 - 5;
      v240 = v90;
      goto LABEL_130;
    }
    v91 = v88 - 5;
  }
  else
  {
    if ( !v240 )
      goto LABEL_68;
    v240 = 0;
    v91 = 1;
    v89 = 6;
LABEL_130:
    v92 = (__int64 *)(a2 - 64);
    if ( *(_BYTE *)a2 == 26 )
      v92 = (__int64 *)(a2 - 32);
    v93 = *v92;
    if ( (!v93 || v93 != v15) && (v89 == 1 || v89 == 4) )
    {
      v231 = v91;
      sub_AC2B30(a2 - 32, v15);
      v91 = v231;
      if ( *(_BYTE *)v15 == 27 )
        v157 = *(_DWORD *)(v15 + 80);
      else
        v157 = *(_DWORD *)(v15 + 72);
      *(_DWORD *)(a2 + 84) = v157;
    }
  }
  if ( v91 <= 1 )
    goto LABEL_68;
  if ( v89 == 4 )
  {
    if ( *a9 <= 1u )
    {
      --*a7;
      goto LABEL_68;
    }
    v230 = v35;
    --*a9;
  }
  else
  {
LABEL_63:
    v230 = v35;
  }
  v94 = _mm_loadu_si128(&v251);
  v95 = _mm_loadu_si128(&v249);
  v260 = &v264;
  v248 = v94;
  v96 = _mm_loadu_si128(&v250);
  v97 = *(_QWORD *)(a2 + 72);
  v261 = 0x100000010LL;
  v262 = 0;
  v264 = v97;
  v98 = *(_QWORD *)(v15 + 72);
  v263 = 1;
  v235 = v98;
  v269 = v271;
  v270 = 0x2000000000LL;
  v259 = 1;
  v272.m128i_i64[0] = 0;
  v272.m128i_i64[1] = (__int64)&v274;
  v273.m128i_i64[0] = 32;
  v273.m128i_i32[2] = 0;
  v273.m128i_i8[12] = 1;
  v246 = v95;
  v247 = v96;
  sub_2754040(v15, (__int64)&v269, &v272, 0x100000010uLL, v49, v50);
  v101 = (unsigned int)v270;
  if ( !(_DWORD)v270 )
    goto LABEL_184;
  v241 = 0;
  v102 = 0;
  v233 = v15;
  while ( 2 )
  {
    v103 = *(_QWORD *)&v269[8 * v102];
    v104 = (unsigned int)*a6;
    if ( v104 < v101 - v102 )
    {
LABEL_234:
      LOBYTE(v245) = 0;
      goto LABEL_186;
    }
    v105 = (unsigned __int64)a6;
    v106 = v260;
    *a6 = v104 - 1;
    if ( *(_BYTE *)v103 == 28 )
    {
      if ( v263 )
        v145 = &v106[HIDWORD(v261)];
      else
        v145 = &v106[(unsigned int)v261];
      if ( v145 != v106 )
      {
        while ( 1 )
        {
          v146 = *v106;
          if ( (unsigned __int64)*v106 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v145 == ++v106 )
            goto LABEL_214;
        }
        if ( v106 != v145 )
        {
          while ( 1 )
          {
            sub_B196A0(*(_QWORD *)(v12 + 792), *(_QWORD *)(v146 + 40), *(_QWORD *)(v103 + 64));
            if ( v147 )
              break;
            v148 = v106 + 1;
            if ( v145 == v106 + 1 )
              goto LABEL_214;
            v146 = *v148;
            for ( ++v106; (unsigned __int64)*v148 >= 0xFFFFFFFFFFFFFFFELL; v106 = v148 )
            {
              if ( v145 == ++v148 )
                goto LABEL_214;
              v146 = *v148;
            }
            if ( v145 == v106 )
              goto LABEL_214;
          }
          if ( v145 != v106 )
            goto LABEL_182;
        }
      }
      goto LABEL_214;
    }
    v107 = *(unsigned __int8 **)(v103 + 72);
    if ( v263 )
      v108 = &v106[HIDWORD(v261)];
    else
      v108 = &v106[(unsigned int)v261];
    if ( v108 != v106 )
    {
      while ( 1 )
      {
        v109 = *v106;
        if ( (unsigned __int64)*v106 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v108 == ++v106 )
          goto LABEL_149;
      }
      while ( 1 )
      {
        if ( v108 == v106 )
          goto LABEL_149;
        if ( (unsigned __int8)sub_B19DB0(*(_QWORD *)(v12 + 792), v109, (__int64)v107) )
          break;
        v142 = v106 + 1;
        if ( v108 == v106 + 1 )
          goto LABEL_149;
        while ( 1 )
        {
          v109 = *v142;
          v106 = v142;
          if ( (unsigned __int64)*v142 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v108 == ++v142 )
            goto LABEL_149;
        }
      }
      if ( v108 != v106 )
        goto LABEL_182;
    }
LABEL_149:
    if ( sub_2757210(v12, (unsigned __int8 **)&v246, v235, (unsigned __int64)v107) )
      goto LABEL_182;
    if ( sub_2753FC0(*(_QWORD *)(v103 + 72)) )
      goto LABEL_214;
    if ( (unsigned __int8)sub_B46790(v107, 0) )
    {
      if ( !(unsigned __int8)sub_CF7590(a5, &v244) )
        goto LABEL_234;
      if ( (_BYTE)v244 )
      {
        v253 = a5;
        LOBYTE(v254[0]) = 1;
        sub_275ABE0((__int64)&v265, v12 + 1432, (__int64 *)&v253, v254);
        if ( v267.m128i_i8[0] )
        {
          v227 = v266.m128i_i64[0];
          v149 = sub_D13FA0((__int64)a5, 0, 0);
          *(_BYTE *)(v227 + 8) = v149;
        }
        else
        {
          v149 = *(_BYTE *)(v266.m128i_i64[0] + 8);
        }
        if ( v149 )
          goto LABEL_234;
      }
    }
    v110 = a10 & (v107 == (unsigned __int8 *)v236);
    if ( v110 )
      v110 = a5 == sub_98ACB0((unsigned __int8 *)v246.m128i_i64[0], 6u);
    if ( !sub_2753FC0((__int64)v107) )
    {
      if ( *v107 == 62 )
      {
        v111 = byte_3F70480[8 * ((*((_WORD *)v107 + 1) >> 7) & 7) + 2];
        goto LABEL_157;
      }
      v226 = *v107;
      if ( (unsigned __int8)sub_B46420((__int64)v107) )
      {
        if ( (unsigned __int8)(v226 - 34) > 0x33u
          || (v150 = 0x8000000000041LL, !_bittest64(&v150, (unsigned int)v226 - 34))
          || !sub_B49EA0((__int64)v107) )
        {
          v151 = _mm_loadu_si128(&v246);
          v152 = *(_QWORD **)(v12 + 104);
          v153 = _mm_loadu_si128(&v247);
          v154 = _mm_loadu_si128(&v248);
          v268 = 1;
          v265 = v151;
          v266 = v153;
          v267 = v154;
          v111 = sub_CF63E0(v152, v107, &v265, v12 + 112) & 1;
LABEL_157:
          if ( v111 == 1 && !v110 )
            goto LABEL_234;
        }
      }
    }
    if ( v103 == v233 )
    {
      if ( !sub_2753D10(v12, (unsigned __int8 *)v246.m128i_i64[0]) )
        goto LABEL_234;
      goto LABEL_182;
    }
    if ( v103 == a2 || *(_BYTE *)v103 != 27 )
      goto LABEL_182;
    if ( !(unsigned __int8)sub_B46490((__int64)v107) )
      goto LABEL_214;
    if ( (unsigned __int8)(*v107 - 34) > 0x33u )
      goto LABEL_164;
    v155 = 0x8000000000041LL;
    if ( !_bittest64(&v155, (unsigned int)*v107 - 34) )
      goto LABEL_164;
    if ( sub_B49EA0((__int64)v107) || !(unsigned __int8)sub_B46490((__int64)v107) )
    {
LABEL_214:
      sub_2754040(v103, (__int64)&v269, &v272, v105, v99, v100);
      goto LABEL_182;
    }
    if ( (unsigned __int8)(*v107 - 34) > 0x33u
      || (v156 = 0x8000000000041LL, !_bittest64(&v156, (unsigned int)*v107 - 34)) )
    {
LABEL_164:
      sub_D66840(&v265, v107);
      goto LABEL_165;
    }
    sub_D67230(&v265, v107, *(__int64 **)(v12 + 808));
LABEL_165:
    if ( !v268 )
      goto LABEL_214;
    v112 = *((_QWORD *)v107 + 5);
    v113 = *(_QWORD *)(v235 + 40);
    if ( v113 == v112 )
      goto LABEL_174;
    v114 = *(_QWORD *)(v12 + 824);
    v115 = *(_DWORD *)(v114 + 24);
    v116 = *(_QWORD *)(v114 + 8);
    if ( !v115 )
      goto LABEL_173;
    v117 = v115 - 1;
    v118 = v117 & (((unsigned int)v113 >> 9) ^ ((unsigned int)v113 >> 4));
    v119 = (__int64 *)(v116 + 16LL * v118);
    v120 = *v119;
    if ( v113 != *v119 )
    {
      v196 = 1;
      while ( v120 != -4096 )
      {
        v197 = v196 + 1;
        v198 = v117 & (v118 + v196);
        v118 = v198;
        v119 = (__int64 *)(v116 + 16 * v198);
        v120 = *v119;
        if ( v113 == *v119 )
          goto LABEL_169;
        v196 = v197;
      }
LABEL_173:
      if ( sub_2753D10(v12, (unsigned __int8 *)v246.m128i_i64[0]) )
        goto LABEL_174;
      goto LABEL_214;
    }
LABEL_169:
    v121 = v119[1];
    if ( *(_BYTE *)(v12 + 832) == 1 || !v121 )
      goto LABEL_173;
    v122 = v117 & (((unsigned int)v112 >> 9) ^ ((unsigned int)v112 >> 4));
    v123 = (__int64 *)(v116 + 16LL * v122);
    v124 = *v123;
    if ( *v123 != v112 )
    {
      v193 = 1;
      while ( v124 != -4096 )
      {
        v194 = v193 + 1;
        v195 = v117 & (v122 + v193);
        v122 = v195;
        v123 = (__int64 *)(v116 + 16 * v195);
        v124 = *v123;
        if ( v112 == *v123 )
          goto LABEL_172;
        v193 = v194;
      }
      goto LABEL_173;
    }
LABEL_172:
    if ( v121 != v123[1] )
      goto LABEL_173;
LABEL_174:
    v125 = sub_27567E0(
             (__int64 *)v12,
             (unsigned __int64)v107,
             v235,
             (__int64)&v265,
             (__int64)&v246,
             &v244,
             (__int64 *)&v253);
    v105 = v125;
    if ( v125 != 1 )
      goto LABEL_214;
    v126 = *(_DWORD *)(v12 + 1680);
    v127 = *((_QWORD *)v107 + 5);
    v128 = *(_QWORD *)(v12 + 1664);
    if ( !v126 )
      goto LABEL_234;
    v129 = v126 - 1;
    v130 = (v126 - 1) & (((unsigned int)v127 >> 9) ^ ((unsigned int)v127 >> 4));
    v131 = (__int64 *)(v128 + 16LL * v130);
    v132 = *v131;
    if ( v127 == *v131 )
    {
LABEL_177:
      v133 = *((_DWORD *)v131 + 2);
      v134 = *(_QWORD *)(v233 + 64);
    }
    else
    {
      v223 = 1;
      while ( v132 != -4096 )
      {
        v224 = v223 + 1;
        v130 = v129 & (v223 + v130);
        v131 = (__int64 *)(v128 + 16LL * v130);
        v132 = *v131;
        if ( v127 == *v131 )
          goto LABEL_177;
        v223 = v224;
      }
      v133 = *(_DWORD *)(v128 + 16LL * v126 + 8);
      v134 = *(_QWORD *)(v233 + 64);
    }
    v135 = v129 & (((unsigned int)v134 >> 9) ^ ((unsigned int)v134 >> 4));
    v136 = (__int64 *)(v128 + 16LL * v135);
    v137 = *v136;
    if ( *v136 == v134 )
    {
LABEL_179:
      v138 = *((_DWORD *)v136 + 2);
    }
    else
    {
      while ( v137 != -4096 )
      {
        v135 = v129 & (v105 + v135);
        v136 = (__int64 *)(v128 + 16LL * v135);
        v137 = *v136;
        if ( v134 == *v136 )
          goto LABEL_179;
        LODWORD(v105) = v105 + 1;
      }
      v138 = *(_DWORD *)(v128 + 16LL * v126 + 8);
    }
    if ( v133 >= v138 )
      goto LABEL_234;
    if ( !(unsigned __int8)sub_275C7B0(v12, a5) )
    {
      if ( v263 )
      {
        v158 = (unsigned __int8 **)v260;
        v139 = HIDWORD(v261);
        v140 = (unsigned __int8 **)&v260[HIDWORD(v261)];
        if ( v260 != (__int64 *)v140 )
        {
          while ( v107 != *v158 )
          {
            if ( v140 == ++v158 )
              goto LABEL_262;
          }
          goto LABEL_182;
        }
LABEL_262:
        if ( HIDWORD(v261) < (unsigned int)v261 )
        {
          ++HIDWORD(v261);
          *v140 = v107;
          ++v259;
          goto LABEL_182;
        }
      }
      sub_C8CC70((__int64)&v259, (__int64)v107, v139, (__int64)v140, v99, v100);
    }
LABEL_182:
    ++v241;
    v101 = (unsigned int)v270;
    v102 = v241;
    if ( (unsigned int)v270 > v241 )
      continue;
    break;
  }
  v15 = v233;
LABEL_184:
  if ( (unsigned __int8)sub_275C7B0(v12, a5) )
  {
LABEL_185:
    v244 = v15;
    LOBYTE(v245) = 1;
    goto LABEL_186;
  }
  v159 = (__int64 *)&v267;
  v265.m128i_i64[0] = 0;
  v265.m128i_i64[1] = (__int64)&v267;
  v160 = (__int64 **)v260;
  v266.m128i_i64[0] = 16;
  v266.m128i_i32[2] = 0;
  v266.m128i_i8[12] = 1;
  if ( v263 )
    v161 = (__int64 **)&v260[HIDWORD(v261)];
  else
    v161 = (__int64 **)&v260[(unsigned int)v261];
  if ( v260 == (__int64 *)v161 )
    goto LABEL_271;
  while ( 1 )
  {
    v162 = *v160;
    v163 = v160;
    if ( (unsigned __int64)*v160 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v161 == ++v160 )
      goto LABEL_271;
  }
  if ( v160 == v161 )
  {
LABEL_271:
    v164 = &v159[v266.m128i_u32[1]];
    goto LABEL_272;
  }
  v183 = v230;
  while ( 2 )
  {
    v184 = v162[5];
    if ( !(_BYTE)v183 )
    {
LABEL_321:
      sub_C8CC70((__int64)&v265, v184, (__int64)v159, (__int64)v162, v183, v141);
      v183 = v266.m128i_u8[12];
      v159 = (__int64 *)v265.m128i_i64[1];
      goto LABEL_314;
    }
    v159 = (__int64 *)v265.m128i_i64[1];
    v162 = (__int64 *)(v265.m128i_i64[1] + 8LL * v266.m128i_u32[1]);
    if ( (__int64 *)v265.m128i_i64[1] == v162 )
    {
LABEL_322:
      if ( v266.m128i_i32[1] < (unsigned __int32)v266.m128i_i32[0] )
      {
        ++v266.m128i_i32[1];
        *v162 = v184;
        v159 = (__int64 *)v265.m128i_i64[1];
        ++v265.m128i_i64[0];
        v183 = v266.m128i_u8[12];
        goto LABEL_314;
      }
      goto LABEL_321;
    }
    v185 = (_QWORD *)v265.m128i_i64[1];
    while ( v184 != *v185 )
    {
      if ( v162 == ++v185 )
        goto LABEL_322;
    }
LABEL_314:
    v186 = v163 + 1;
    if ( v163 + 1 != v161 )
    {
      while ( 1 )
      {
        v162 = *v186;
        v163 = v186;
        if ( (unsigned __int64)*v186 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v161 == ++v186 )
          goto LABEL_317;
      }
      if ( v186 != v161 )
        continue;
    }
    break;
  }
LABEL_317:
  if ( (_BYTE)v183 )
    goto LABEL_271;
  v164 = &v159[v266.m128i_u32[0]];
LABEL_272:
  v165 = *v159;
  if ( v159 != v164 )
  {
    while ( 1 )
    {
      v165 = *v159;
      v166 = v159;
      if ( (unsigned __int64)*v159 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v164 == ++v159 )
      {
        v165 = v166[1];
        break;
      }
    }
  }
  v243 = v165;
  sub_27574D0(&v253, v265.m128i_i64, 1);
  v167 = v253;
  if ( v256 == v253 )
  {
    v169 = *(_QWORD *)(v12 + 800);
    v168 = v243;
  }
  else
  {
    v168 = v243;
    v169 = *(_QWORD *)(v12 + 800);
    do
    {
      v170 = *(_QWORD *)v167;
      if ( !v168 )
        break;
      v171 = *(_DWORD *)(v169 + 56);
      v172 = 0;
      v173 = (unsigned int)(*(_DWORD *)(v168 + 44) + 1);
      if ( (unsigned int)v173 < v171 )
        v172 = *(_QWORD *)(*(_QWORD *)(v169 + 48) + 8 * v173);
      if ( v170 )
      {
        v174 = (unsigned int)(*(_DWORD *)(v170 + 44) + 1);
        v175 = v174;
      }
      else
      {
        v174 = 0;
        v175 = 0;
      }
      v176 = 0;
      if ( v171 > v175 )
        v176 = *(_QWORD *)(*(_QWORD *)(v169 + 48) + 8 * v174);
      while ( v172 != v176 )
      {
        if ( *(_DWORD *)(v172 + 16) < *(_DWORD *)(v176 + 16) )
        {
          v177 = v172;
          v172 = v176;
          v176 = v177;
        }
        v172 = *(_QWORD *)(v172 + 8);
      }
      v168 = *(_QWORD *)v176;
      v243 = *(_QWORD *)v176;
      do
        v167 += 8;
      while ( v167 != (unsigned __int8 *)v254[0] && *(_QWORD *)v167 >= 0xFFFFFFFFFFFFFFFELL );
    }
    while ( v256 != v167 );
  }
  sub_B19AA0(v169, v168, *(_QWORD *)(v15 + 64));
  v178 = v243;
  if ( !v179 )
  {
    if ( !*(_BYTE *)(v12 + 1736) )
    {
      LOBYTE(v245) = 0;
      v180 = v266.m128i_i8[12];
      goto LABEL_296;
    }
    v243 = 0;
    v178 = 0;
  }
  v180 = v266.m128i_i8[12];
  if ( !v266.m128i_i8[12] )
  {
    if ( !sub_C8CA60((__int64)&v265, v178) )
    {
      v178 = v243;
      goto LABEL_352;
    }
    v180 = v266.m128i_i8[12];
LABEL_305:
    v244 = v15;
    LOBYTE(v245) = 1;
    goto LABEL_296;
  }
  v181 = (_QWORD *)v265.m128i_i64[1];
  v182 = v265.m128i_i64[1] + 8LL * v266.m128i_u32[1];
  if ( v265.m128i_i64[1] != v182 )
  {
    while ( v178 != *v181 )
    {
      if ( (_QWORD *)v182 == ++v181 )
        goto LABEL_352;
    }
    goto LABEL_305;
  }
LABEL_352:
  v253 = 0;
  v254[0] = 0;
  v254[1] = 0;
  v255 = 0;
  v256 = v258;
  v257 = 0;
  if ( v178 )
  {
    sub_275CEE0((__int64)&v253, &v243);
  }
  else
  {
    v218 = *(_QWORD *)(v12 + 800);
    v219 = *(__int64 **)v218;
    v220 = *(unsigned int *)(v218 + 8);
    if ( v219 != &v219[v220] )
    {
      v242 = v12;
      v221 = &v219[v220];
      do
      {
        v244 = *v219;
        v222 = *(_QWORD *)(v244 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v222 == v244 + 48 )
          goto LABEL_411;
        if ( !v222 )
          BUG();
        if ( (unsigned int)*(unsigned __int8 *)(v222 - 24) - 30 > 0xA )
LABEL_411:
          BUG();
        if ( *(_BYTE *)(v222 - 24) != 36 )
          sub_275CEE0((__int64)&v253, &v244);
        ++v219;
      }
      while ( v221 != v219 );
      v12 = v242;
    }
  }
  v205 = v256;
  if ( !(_DWORD)v257 )
  {
LABEL_363:
    if ( v205 != v258 )
      _libc_free((unsigned __int64)v205);
    sub_C7D6A0(v254[0], 8LL * (unsigned int)v255, 8);
    if ( !v266.m128i_i8[12] )
      _libc_free(v265.m128i_u64[1]);
    goto LABEL_185;
  }
  v206 = 0;
  v207 = 0;
  v208 = v12;
  while ( 2 )
  {
    v209 = *(_QWORD *)&v205[8 * v207];
    if ( !v266.m128i_i8[12] )
    {
      if ( !sub_C8CA60((__int64)&v265, v209) )
        goto LABEL_368;
LABEL_373:
      v212 = v257;
      v205 = v256;
LABEL_362:
      v207 = (unsigned int)(v206 + 1);
      v206 = v207;
      if ( (unsigned int)v207 >= v212 )
        goto LABEL_363;
      continue;
    }
    break;
  }
  v210 = (_QWORD *)v265.m128i_i64[1];
  v211 = v265.m128i_i64[1] + 8LL * v266.m128i_u32[1];
  if ( v265.m128i_i64[1] != v211 )
  {
    while ( v209 != *v210 )
    {
      if ( (_QWORD *)v211 == ++v210 )
        goto LABEL_368;
    }
    v212 = v257;
    goto LABEL_362;
  }
LABEL_368:
  if ( v209 == *(_QWORD *)(v15 + 64) )
    goto LABEL_385;
  v213 = *(_QWORD *)(v208 + 792);
  if ( v209 )
  {
    v214 = (unsigned int)(*(_DWORD *)(v209 + 44) + 1);
    v215 = *(_DWORD *)(v209 + 44) + 1;
  }
  else
  {
    v214 = 0;
    v215 = 0;
  }
  if ( v215 >= *(_DWORD *)(v213 + 32) || !*(_QWORD *)(*(_QWORD *)(v213 + 24) + 8 * v214) )
    goto LABEL_373;
  v216 = *(_QWORD *)(v209 + 16);
  if ( v216 )
  {
    while ( 1 )
    {
      v217 = *(_QWORD *)(v216 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v217 - 30) <= 0xAu )
        break;
      v216 = *(_QWORD *)(v216 + 8);
      if ( !v216 )
        goto LABEL_381;
    }
LABEL_379:
    v244 = *(_QWORD *)(v217 + 40);
    sub_275CEE0((__int64)&v253, &v244);
    while ( 1 )
    {
      v216 = *(_QWORD *)(v216 + 8);
      if ( !v216 )
        break;
      v217 = *(_QWORD *)(v216 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v217 - 30) <= 0xAu )
        goto LABEL_379;
    }
  }
LABEL_381:
  v212 = v257;
  if ( (unsigned int)qword_4FFA468 > (unsigned int)v257 )
  {
    v205 = v256;
    goto LABEL_362;
  }
LABEL_385:
  LOBYTE(v245) = 0;
  if ( v256 != v258 )
    _libc_free((unsigned __int64)v256);
  sub_C7D6A0(v254[0], 8LL * (unsigned int)v255, 8);
  v180 = v266.m128i_i8[12];
LABEL_296:
  if ( !v180 )
    _libc_free(v265.m128i_u64[1]);
LABEL_186:
  if ( !v273.m128i_i8[12] )
    _libc_free(v272.m128i_u64[1]);
  if ( v269 != v271 )
    _libc_free((unsigned __int64)v269);
  if ( !v263 )
    _libc_free((unsigned __int64)v260);
  return v244;
}
