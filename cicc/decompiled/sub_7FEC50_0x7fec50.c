// Function: sub_7FEC50
// Address: 0x7fec50
//
__int64 __fastcall sub_7FEC50(
        __int64 a1,
        __m128i *a2,
        __int64 *a3,
        __int64 a4,
        int a5,
        int a6,
        __m128i *a7,
        _BOOL4 *a8,
        __m128i *a9)
{
  unsigned __int8 v11; // al
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // r9
  __int64 v15; // r12
  _QWORD *v16; // rdx
  unsigned __int8 *v17; // r15
  _QWORD *v18; // rax
  int v19; // eax
  char *v20; // r13
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rax
  char v24; // al
  _BYTE *v25; // rax
  __m128i *v26; // rdi
  __int64 v27; // rdi
  int v28; // r8d
  _QWORD *v29; // rax
  __m128i *v30; // rax
  __int64 v31; // rcx
  __int64 v32; // r9
  const __m128i *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 *v38; // rdi
  int *v39; // rsi
  __int64 v40; // r10
  __int64 v41; // rdi
  _QWORD *v42; // rdi
  __int64 result; // rax
  _BYTE *v44; // r8
  __int64 v45; // r10
  __int64 j; // rsi
  __m128i *v47; // rax
  __int64 v48; // r10
  char v49; // al
  __int64 v50; // rcx
  __m128i *v51; // rax
  __int64 v52; // rax
  __int64 v53; // rdi
  _QWORD *v54; // rax
  __int64 v55; // rdi
  const __m128i *v56; // rax
  const __m128i *v57; // rax
  __int64 v58; // rax
  _BYTE *v59; // rax
  __int64 v60; // rax
  _QWORD *v61; // rax
  _QWORD *v62; // r8
  _QWORD *v63; // r8
  _QWORD *v64; // rcx
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rsi
  __int64 v70; // rbx
  _QWORD *v71; // rax
  __int64 v72; // rsi
  __m128i *v73; // rax
  __m128i *v74; // rdi
  __m128i *v75; // rbx
  _BYTE *v76; // rax
  __int64 v77; // rdi
  _QWORD *v78; // rdi
  char v79; // al
  __int64 v80; // rdi
  __int64 v81; // rax
  __int64 v82; // r11
  _QWORD *v83; // rax
  const __m128i *v84; // rsi
  __int64 v85; // rdi
  _QWORD *v86; // r10
  __m128i *v87; // rax
  __m128i *v88; // rax
  _BYTE *v89; // rax
  __m128i *v90; // rdi
  __m128i *v91; // rax
  int v92; // eax
  __int64 v93; // r10
  __int64 v94; // r8
  __int64 *v95; // rax
  __int64 v96; // r8
  _QWORD *v97; // rdi
  __m128i *v98; // rax
  unsigned int v99; // r9d
  __int64 k; // rax
  __int64 *v101; // r10
  char v102; // dl
  __m128i *v103; // rax
  __m128i *v104; // r10
  __int64 v105; // rdi
  __m128i *v106; // rax
  _BOOL4 v107; // eax
  __m128i *v108; // r8
  __int64 v109; // rsi
  int v110; // eax
  __int64 v111; // rdx
  __int64 v112; // rcx
  __int64 v113; // r9
  __int64 v114; // r8
  _QWORD *v115; // rax
  __int64 v116; // rsi
  _BYTE *v117; // rax
  __int64 v118; // rax
  __m128i *v119; // rax
  _QWORD *v120; // rax
  int v121; // eax
  __int16 v122; // ax
  __m128i *v123; // r8
  __int64 v124; // rdx
  __int64 v125; // rax
  __int64 v126; // rsi
  __m128i *v127; // rax
  const __m128i *v128; // rdi
  __int64 v129; // rax
  __int64 v130; // rdx
  __int64 v131; // rcx
  __int64 v132; // r8
  __m128i *v133; // rax
  __int64 v134; // r11
  _BYTE *v135; // rax
  __int64 v136; // rcx
  __int64 v137; // r8
  __int64 v138; // r9
  int v139; // eax
  _QWORD *v140; // rax
  _BYTE *v141; // rax
  __m128i *v142; // rax
  __m128i *v143; // rax
  __int64 v144; // rdx
  _QWORD *v145; // rdi
  __int64 v146; // rax
  _BYTE *v147; // rax
  __m128i *v148; // r8
  __int64 v149; // rax
  __int64 v150; // rsi
  __int64 v151; // rax
  _BYTE *v152; // rax
  __int64 v153; // rax
  _QWORD *v154; // rax
  __int64 v155; // rsi
  int v156; // eax
  _BYTE *v157; // rax
  __int64 v158; // rdi
  _QWORD *v159; // rax
  __m128i *v160; // rsi
  __m128i *v161; // rax
  _QWORD *v162; // rax
  const __m128i *v163; // rbx
  _QWORD *v164; // r12
  _BYTE *v165; // rax
  void *v166; // rax
  __m128i *v167; // rax
  __m128i *v168; // rax
  __m128i *v169; // rax
  __m128i *v170; // rax
  __m128i *v171; // rax
  __int64 v172; // rcx
  __int64 v173; // r11
  __int64 v174; // r10
  __int64 v175; // rdi
  _BYTE *v176; // rsi
  const __m128i *v177; // rsi
  _QWORD *v178; // r9
  __int64 v179; // rax
  _BYTE *v180; // rax
  __int64 v181; // rdx
  __int64 v182; // rax
  _BYTE *v183; // rax
  __int64 *v184; // rax
  __int64 v185; // rax
  __int64 v186; // [rsp-10h] [rbp-310h]
  __int64 v187; // [rsp+0h] [rbp-300h]
  _QWORD *v188; // [rsp+8h] [rbp-2F8h]
  __int64 v189; // [rsp+8h] [rbp-2F8h]
  _QWORD *v191; // [rsp+10h] [rbp-2F0h]
  __int64 v192; // [rsp+10h] [rbp-2F0h]
  __int64 v193; // [rsp+18h] [rbp-2E8h]
  _QWORD *v194; // [rsp+18h] [rbp-2E8h]
  _QWORD *v195; // [rsp+18h] [rbp-2E8h]
  __int64 v196; // [rsp+18h] [rbp-2E8h]
  __int64 v197; // [rsp+18h] [rbp-2E8h]
  __m128i *v198; // [rsp+18h] [rbp-2E8h]
  __m128i *v199; // [rsp+18h] [rbp-2E8h]
  int v200; // [rsp+24h] [rbp-2DCh]
  __int64 i; // [rsp+28h] [rbp-2D8h]
  _BOOL4 v203; // [rsp+28h] [rbp-2D8h]
  __int64 *v204; // [rsp+28h] [rbp-2D8h]
  char v205; // [rsp+28h] [rbp-2D8h]
  __int64 v206; // [rsp+28h] [rbp-2D8h]
  __m128i *v207; // [rsp+28h] [rbp-2D8h]
  __m128i *v210; // [rsp+38h] [rbp-2C8h]
  __int64 *v211; // [rsp+38h] [rbp-2C8h]
  __int64 v212; // [rsp+38h] [rbp-2C8h]
  __int64 v213; // [rsp+38h] [rbp-2C8h]
  __int64 v214; // [rsp+38h] [rbp-2C8h]
  __int64 *v215; // [rsp+38h] [rbp-2C8h]
  _QWORD *v216; // [rsp+38h] [rbp-2C8h]
  __m128i *v217; // [rsp+38h] [rbp-2C8h]
  __int64 v218; // [rsp+38h] [rbp-2C8h]
  _BOOL4 v219; // [rsp+40h] [rbp-2C0h]
  unsigned int v220; // [rsp+40h] [rbp-2C0h]
  __int64 v221; // [rsp+40h] [rbp-2C0h]
  __int64 v222; // [rsp+40h] [rbp-2C0h]
  __int64 v223; // [rsp+40h] [rbp-2C0h]
  __int64 v224; // [rsp+40h] [rbp-2C0h]
  __int64 v225; // [rsp+40h] [rbp-2C0h]
  __int64 v226; // [rsp+40h] [rbp-2C0h]
  __int64 v227; // [rsp+40h] [rbp-2C0h]
  __int64 *v228; // [rsp+40h] [rbp-2C0h]
  _QWORD *v229; // [rsp+40h] [rbp-2C0h]
  __int64 v230; // [rsp+40h] [rbp-2C0h]
  __int64 v231; // [rsp+40h] [rbp-2C0h]
  __int64 v232; // [rsp+40h] [rbp-2C0h]
  __int64 v233; // [rsp+40h] [rbp-2C0h]
  __m128i *v234; // [rsp+40h] [rbp-2C0h]
  __int64 v235; // [rsp+40h] [rbp-2C0h]
  _QWORD *v236; // [rsp+40h] [rbp-2C0h]
  __int64 v237; // [rsp+40h] [rbp-2C0h]
  unsigned __int8 v238; // [rsp+49h] [rbp-2B7h]
  __int16 v239; // [rsp+4Ah] [rbp-2B6h]
  int v240; // [rsp+4Ch] [rbp-2B4h]
  _QWORD **v241; // [rsp+50h] [rbp-2B0h]
  const __m128i *v242; // [rsp+50h] [rbp-2B0h]
  _QWORD *v243; // [rsp+50h] [rbp-2B0h]
  __int64 v244; // [rsp+50h] [rbp-2B0h]
  __int64 v245; // [rsp+58h] [rbp-2A8h]
  const __m128i *v246; // [rsp+58h] [rbp-2A8h]
  __int64 v247; // [rsp+58h] [rbp-2A8h]
  unsigned int v248; // [rsp+60h] [rbp-2A0h]
  __int64 v249; // [rsp+60h] [rbp-2A0h]
  _QWORD *v250; // [rsp+60h] [rbp-2A0h]
  __int64 v251; // [rsp+60h] [rbp-2A0h]
  __int64 v252; // [rsp+60h] [rbp-2A0h]
  __m128i *v253; // [rsp+60h] [rbp-2A0h]
  __int64 *v254; // [rsp+60h] [rbp-2A0h]
  __int64 v255; // [rsp+60h] [rbp-2A0h]
  __int64 v256; // [rsp+60h] [rbp-2A0h]
  __m128i *v257; // [rsp+60h] [rbp-2A0h]
  __int64 v258; // [rsp+60h] [rbp-2A0h]
  __int64 v259; // [rsp+60h] [rbp-2A0h]
  __int64 v260; // [rsp+60h] [rbp-2A0h]
  __int64 v261; // [rsp+60h] [rbp-2A0h]
  __int64 v262; // [rsp+60h] [rbp-2A0h]
  _BYTE *v263; // [rsp+60h] [rbp-2A0h]
  __int64 v264; // [rsp+60h] [rbp-2A0h]
  int v265; // [rsp+68h] [rbp-298h]
  __int16 v266; // [rsp+6Ch] [rbp-294h]
  bool v267; // [rsp+6Fh] [rbp-291h]
  __m128i *v268; // [rsp+70h] [rbp-290h]
  __int64 v269; // [rsp+70h] [rbp-290h]
  _BYTE *v270; // [rsp+70h] [rbp-290h]
  _BYTE *v271; // [rsp+70h] [rbp-290h]
  _BOOL4 v272; // [rsp+80h] [rbp-280h]
  __m128i *v273; // [rsp+80h] [rbp-280h]
  _QWORD *v274; // [rsp+80h] [rbp-280h]
  __m128i *v275; // [rsp+80h] [rbp-280h]
  _QWORD *v276; // [rsp+80h] [rbp-280h]
  _QWORD *v277; // [rsp+80h] [rbp-280h]
  const __m128i *v278; // [rsp+80h] [rbp-280h]
  __int64 v279; // [rsp+80h] [rbp-280h]
  __int64 v280; // [rsp+80h] [rbp-280h]
  __int64 v281; // [rsp+80h] [rbp-280h]
  const __m128i *v282; // [rsp+80h] [rbp-280h]
  __int64 v283; // [rsp+80h] [rbp-280h]
  __m128i *v284; // [rsp+80h] [rbp-280h]
  __int64 *v285; // [rsp+80h] [rbp-280h]
  _BYTE *v286; // [rsp+80h] [rbp-280h]
  __int64 v287; // [rsp+80h] [rbp-280h]
  _QWORD *v288; // [rsp+80h] [rbp-280h]
  __m128i *v289; // [rsp+80h] [rbp-280h]
  _QWORD *v290; // [rsp+80h] [rbp-280h]
  __m128i *v291; // [rsp+80h] [rbp-280h]
  __int64 v292; // [rsp+80h] [rbp-280h]
  __m128i *v293; // [rsp+80h] [rbp-280h]
  _BOOL4 v294; // [rsp+94h] [rbp-26Ch] BYREF
  __int64 v295; // [rsp+98h] [rbp-268h] BYREF
  __int64 v296; // [rsp+A0h] [rbp-260h] BYREF
  __int64 *v297; // [rsp+A8h] [rbp-258h] BYREF
  __m128i v298; // [rsp+B0h] [rbp-250h] BYREF
  __m128i v299; // [rsp+C0h] [rbp-240h]
  __m128i v300[2]; // [rsp+D0h] [rbp-230h] BYREF
  _BYTE v301[96]; // [rsp+F0h] [rbp-210h] BYREF
  char v302[96]; // [rsp+150h] [rbp-1B0h] BYREF
  char v303[96]; // [rsp+1B0h] [rbp-150h] BYREF
  __m128i v304[15]; // [rsp+210h] [rbp-F0h] BYREF

  v241 = (_QWORD **)qword_4D03F68;
  v11 = *(_BYTE *)(a1 + 48);
  v295 = 0;
  v238 = v11;
  v296 = 0;
  v245 = 0;
  v265 = dword_4D03F38[0];
  v239 = dword_4D03F38[1];
  v240 = dword_4F07508[0];
  v266 = dword_4F07508[1];
  if ( a3 )
  {
    v12 = *a3;
    v245 = *a3;
    if ( *a3 )
    {
      if ( *(_DWORD *)(v12 + 40) )
      {
        *(_QWORD *)dword_4F07508 = *(_QWORD *)(v12 + 40);
        *(_QWORD *)dword_4D03F38 = *(_QWORD *)dword_4F07508;
      }
    }
  }
  if ( a9 )
    a9->m128i_i64[0] = 0;
  v13 = 0;
  sub_802F60(a1, 0);
  v15 = *(_QWORD *)(a1 + 8);
  v267 = v15 != 0 && *(_QWORD *)(a1 + 112) == 0;
  if ( !v267 )
  {
    if ( !a2[1].m128i_i8[0] )
      v267 = *(_BYTE *)(a2->m128i_i64[1] + 136) <= 2u;
    v248 = 1;
    if ( v245 )
      v248 = *(_BYTE *)(v245 + 8) > 1u;
    if ( (*(_BYTE *)(a1 + 51) & 2) != 0 )
    {
      *(_QWORD *)(a1 + 88) = a2;
      v15 = 0;
      v200 = 0;
    }
    else
    {
      v200 = 0;
      v15 = 0;
    }
    v219 = 1;
    v272 = 0;
LABEL_15:
    v16 = *(_QWORD **)(a1 + 24);
    v17 = 0;
    if ( !v16 )
      goto LABEL_29;
    goto LABEL_16;
  }
  v13 = *(unsigned int *)(v15 + 64);
  if ( (_DWORD)v13 )
  {
    *(_QWORD *)dword_4F07508 = *(_QWORD *)(v15 + 64);
    *(_QWORD *)dword_4D03F38 = *(_QWORD *)dword_4F07508;
  }
  *(_BYTE *)(v15 + 173) |= 8u;
  if ( a2[1].m128i_i8[0] || (v219 = 0, *(_BYTE *)(a2->m128i_i64[1] + 136) > 2u) )
  {
    v267 = 0;
    v219 = a8 == 0;
  }
  v24 = *(_BYTE *)(v15 + 173);
  if ( v24 >= 0 && ((*(_BYTE *)(v15 + 172) & 0x20) == 0 || !dword_4D0481C)
    || *(_BYTE *)(v15 + 136)
    || *(_QWORD *)(v15 + 240) )
  {
    if ( v219 )
    {
      v200 = 0;
      v272 = 0;
      if ( *(_BYTE *)(v15 + 177) != 4 )
        goto LABEL_48;
LABEL_59:
      v13 = qword_4D03F68[1];
      v17 = (unsigned __int8 *)sub_72F9B0(v15, v13);
      v298 = _mm_loadu_si128(a7);
      v299 = _mm_loadu_si128(a7 + 1);
      if ( v17 )
        goto LABEL_60;
      goto LABEL_64;
    }
    v200 = 0;
    v272 = *(_BYTE *)(a1 + 48) == 2;
  }
  else
  {
    v200 = 1;
    v219 = 1;
    v272 = 0;
  }
  if ( *(_BYTE *)(v15 + 177) == 4 )
    goto LABEL_59;
LABEL_48:
  if ( (v24 & 0x40) != 0 )
  {
    v17 = (unsigned __int8 *)unk_4D03F40;
    if ( unk_4D03F40 )
    {
      while ( *((_QWORD *)v17 + 1) != v15 )
      {
        v17 = *(unsigned __int8 **)v17;
        if ( !v17 )
          goto LABEL_290;
      }
      v298 = _mm_loadu_si128(a7);
      v299 = _mm_loadu_si128(a7 + 1);
    }
    else
    {
LABEL_290:
      v122 = *(_WORD *)(v15 + 172);
      v298 = _mm_loadu_si128(a7);
      v299 = _mm_loadu_si128(a7 + 1);
      if ( (v122 & 0x4020) != 0x4020 )
      {
        v248 = 1;
        v272 = 0;
        goto LABEL_15;
      }
      v17 = 0;
    }
    goto LABEL_54;
  }
  v298 = _mm_loadu_si128(a7);
  v299 = _mm_loadu_si128(a7 + 1);
LABEL_64:
  v248 = 1;
  if ( (*(_WORD *)(v15 + 172) & 0x4020) != 0x4020 )
    goto LABEL_15;
  v17 = 0;
LABEL_60:
  if ( !v272 )
  {
LABEL_54:
    v13 = (__int64)&v298;
    sub_7F8CF0(v15, v298.m128i_i32, a7, &v295, &v296);
    v272 = 0;
  }
  v248 = 1;
  v16 = *(_QWORD **)(a1 + 24);
  if ( v16 )
  {
LABEL_16:
    if ( (_QWORD *)qword_4F06BC0 != v16 )
    {
      if ( *(_BYTE *)v16 == 3 )
      {
        v13 = 0;
        sub_7E18E0((__int64)v302, 0, (__int64)v16);
        v241 = (_QWORD **)qword_4D03F68;
        sub_7E1AA0();
      }
      else if ( *(_BYTE *)qword_4F06BC0 == 4 && *(_QWORD **)(qword_4F06BC0 + 32LL) == v16
             || dword_4D03EB8[0] && !*(_BYTE *)v16 )
      {
        v241 = sub_7E71B0(v16);
      }
      else
      {
        v241 = (_QWORD **)qword_4D03F68;
        if ( !*((_BYTE *)qword_4D03F68 + 26) )
LABEL_363:
          sub_721090();
        v18 = qword_4D03F68;
        do
        {
          if ( (_QWORD *)v18[2] == v16 )
            break;
          v18 = (_QWORD *)*v18;
          if ( !v18 )
            break;
        }
        while ( *((_BYTE *)v18 + 26) );
      }
    }
  }
  if ( v17 )
  {
    v17 = (unsigned __int8 *)*((_QWORD *)v17 + 4);
    if ( v17 )
    {
      sub_7E18E0((__int64)v303, 0, (__int64)v17);
      v13 = (__int64)a7;
      sub_7E9190((__int64)v17, (__int64)a7);
      sub_733650((__int64)v17);
      if ( dword_4D03F8C )
      {
        if ( v295 )
        {
          v13 = 20;
          sub_732E60(v17, 0x14u, *(_QWORD **)(v295 + 80));
        }
      }
    }
  }
LABEL_29:
  v19 = *(unsigned __int8 *)(a1 + 48);
  v20 = *(char **)(a1 + 40);
  if ( (_BYTE)v19 == 5 && a2[1].m128i_i8[1] )
  {
    v20 = 0;
    v268 = a7;
    goto LABEL_120;
  }
  v21 = dword_4D03EB8[0];
  if ( v20 )
  {
    if ( dword_4D03EB8[0] && (*(v20 - 8) & 1) != 0 )
    {
      v269 = qword_4F06BC0;
      qword_4F06BC0 = *(_QWORD *)(qword_4F04C50 + 88LL);
      sub_733780(0, 0, 0, *v20, 0);
      sub_733650((__int64)v20);
      v13 = 0;
      v20 = (char *)qword_4F06BC0;
      qword_4F06BC0 = v269;
      sub_7E18E0((__int64)v301, 0, (__int64)v20);
      if ( !dword_4D03F8C )
      {
        v268 = a7;
LABEL_69:
        if ( !dword_4D03EB8[0] )
          goto LABEL_75;
        goto LABEL_70;
      }
      v270 = sub_726B30(11);
      sub_7E6810((__int64)v270, (__int64)a7, 1);
      sub_7E1740((__int64)v270, (__int64)&v298);
      v25 = v270;
    }
    else
    {
      v13 = 0;
      sub_7E18E0((__int64)v301, 0, *(_QWORD *)(a1 + 40));
      if ( !dword_4D03F8C )
      {
        v268 = a7;
        if ( !dword_4D03EB8[0] )
        {
LABEL_35:
          v23 = *(_QWORD *)(a1 + 40);
          *(_QWORD *)(a1 + 40) = v20;
          v13 = (__int64)v268;
          v193 = v23;
          sub_7E9190((__int64)v20, (__int64)v268);
          *(_QWORD *)(a1 + 40) = v193;
LABEL_36:
          LOBYTE(v19) = *(_BYTE *)(a1 + 48);
          goto LABEL_37;
        }
LABEL_70:
        v19 = *(unsigned __int8 *)(a1 + 48);
        goto LABEL_71;
      }
      v271 = sub_726B30(11);
      sub_7E6810((__int64)v271, (__int64)a7, 1);
      sub_7E1740((__int64)v271, (__int64)&v298);
      sub_733650((__int64)v20);
      v25 = v271;
    }
    v13 = 20;
    v268 = &v298;
    sub_732E60((unsigned __int8 *)v20, 0x14u, *((_QWORD **)v25 + 10));
    goto LABEL_69;
  }
  v22 = (__int64)a7;
  v268 = a7;
  if ( dword_4D03EB8[0] )
  {
LABEL_71:
    v21 = (unsigned int)(v19 - 3);
    if ( (unsigned __int8)(v19 - 3) <= 1u )
    {
      v26 = *(__m128i **)(a1 + 56);
      if ( (v26[-1].m128i_i8[8] & 1) == 0 )
        goto LABEL_75;
    }
    else
    {
      if ( (_BYTE)v19 != 7 )
      {
        if ( (_BYTE)v19 == 5 )
        {
          v128 = *(const __m128i **)(a1 + 64);
          if ( !v128 || (v128[-1].m128i_i8[8] & 1) == 0 )
          {
            if ( !v20 )
            {
LABEL_120:
              v44 = sub_7F98A0((__int64)a2, 1);
              v45 = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 152LL);
              for ( i = *(_QWORD *)(a1 + 56); *(_BYTE *)(v45 + 140) == 12; v45 = *(_QWORD *)(v45 + 160) )
                ;
              v274 = v44;
              if ( !*(_QWORD *)(*(_QWORD *)(v45 + 168) + 40LL) )
                BUG();
              v221 = v45;
              for ( j = sub_8D71D0(v45); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
                ;
              v47 = (__m128i *)sub_73E130(v274, j);
              v48 = v221;
              v275 = v47;
              v49 = *(_BYTE *)(a1 + 72);
              if ( (v49 & 1) != 0 )
              {
                v109 = (__int64)a2;
                v127 = sub_7F9500((__int64)a3, (__int64)a2, 1);
                v48 = v221;
                v114 = (__int64)v127;
              }
              else
              {
                if ( (v49 & 0x10) == 0 )
                {
LABEL_127:
                  v222 = 0;
                  v50 = 0;
                  goto LABEL_128;
                }
                v108 = *(__m128i **)(a1 + 64);
                v109 = 1;
                v214 = v221;
                v228 = (__int64 *)v108;
                *(_QWORD *)(a1 + 64) = v108[1].m128i_i64[0];
                v108[1].m128i_i64[0] = 0;
                sub_7EE560(v108, (__m128i *)1);
                v110 = sub_8D3410(*v228);
                v114 = (__int64)v228;
                v48 = v214;
                if ( !v110 )
                {
LABEL_279:
                  if ( (*(_BYTE *)(v114 + 25) & 1) != 0 )
                  {
                    v232 = v48;
                    v141 = sub_73E1B0(v114, v109);
                    v48 = v232;
                    v114 = (__int64)v141;
                  }
                  v197 = v48;
                  v229 = (_QWORD *)v114;
                  v215 = sub_7E5340(i);
                  v115 = (_QWORD *)sub_8D46C0(v215[1]);
                  v116 = sub_72D2E0(v115);
                  v117 = sub_73E130(v229, v116);
                  v48 = v197;
                  v222 = (__int64)v117;
                  v50 = *v215;
LABEL_128:
                  if ( (*(_BYTE *)(a2->m128i_i64[1] + 156) & 1) != 0 )
                  {
                    v196 = v48;
                    v213 = v50;
                    v106 = (__m128i *)sub_7E5340(i);
                    v50 = v213;
                    v48 = v196;
                    a9 = v106;
                    if ( !v106 )
                    {
                      v107 = sub_7F7E80(i);
                      v50 = v213;
                      v48 = v196;
                      if ( v107 )
                      {
LABEL_275:
                        v248 = 0;
                        v40 = 0;
                        v272 = 0;
                        goto LABEL_90;
                      }
                    }
                  }
                  if ( a2[1].m128i_i8[1] )
                  {
                    v203 = sub_7F9D00(a1);
                    v51 = sub_7FDF40(*(_QWORD *)(a1 + 56), 1, 0);
                    v249 = sub_7FE9A0((__int64)v51, *(__m128i **)(a1 + 64), v222 != 0);
                    v52 = *(_QWORD *)(a1 + 16);
                    v210 = (__m128i *)v52;
                    if ( v52 )
                      v210 = sub_7FDF40(v52, 1, 0);
                    v53 = *(_QWORD *)(a1 + 40);
                    if ( v53 )
                      sub_733650(v53);
                    if ( v222 )
                    {
                      sub_72BA30(byte_4F06A51[0]);
                      v204 = sub_7F9160((__int64)a2);
                      v54 = sub_7F5ED0(v275->m128i_i64[0]);
                      v55 = qword_4F18B28;
                      v194 = v54;
                      if ( !qword_4F18B28 )
                      {
                        v192 = sub_7E1C10();
                        v189 = sub_7E1C10();
                        if ( HIDWORD(qword_4F0688C) )
                          v158 = sub_7E1C10();
                        else
                          v158 = sub_72CBE0();
                        v159 = sub_7F5250(v158, v189, v192);
                        qword_4F18B28 = sub_72D2E0(v159);
                        v55 = qword_4F18B28;
                      }
                      v56 = (const __m128i *)sub_724DC0();
                      v304[0].m128i_i64[0] = (__int64)v56;
                      if ( v249 )
                      {
                        v191 = sub_731330(v249);
                      }
                      else
                      {
                        v155 = (__int64)v56;
                        sub_72BB40(v55, v56);
                        v191 = sub_73A720((const __m128i *)v304[0].m128i_i64[0], v155);
                      }
                      sub_724E30((__int64)v304);
                      v187 = sub_7F9D60();
                      v57 = (const __m128i *)sub_724DC0();
                      v304[0].m128i_i64[0] = (__int64)v57;
                      if ( v210 )
                      {
                        v188 = sub_731330((__int64)v210);
                        sub_724E30((__int64)v304);
                        v58 = sub_7E1C10();
                        v59 = sub_73E110(v222, v58);
                        v275[1].m128i_i64[0] = (__int64)v59;
                        *((_QWORD *)v59 + 2) = v204;
                        v204[2] = (__int64)v194;
                        v194[2] = v191;
                        v191[2] = v188;
                        v60 = sub_72CBE0();
                        v61 = sub_7F89D0("__cxa_vec_cctor", &qword_4F18B60, v60, v275);
                        v62 = v61;
                        if ( v249 )
                        {
                          v276 = v61;
                          sub_8255D0(v249);
                          v62 = v276;
                        }
                        v277 = v62;
                        sub_8255D0(v210);
                        v63 = v277;
                      }
                      else
                      {
                        v150 = (__int64)v57;
                        sub_72BB40(v187, v57);
                        v216 = sub_73A720((const __m128i *)v304[0].m128i_i64[0], v150);
                        sub_724E30((__int64)v304);
                        v151 = sub_7E1C10();
                        v152 = sub_73E110(v222, v151);
                        v275[1].m128i_i64[0] = (__int64)v152;
                        *((_QWORD *)v152 + 2) = v204;
                        v204[2] = (__int64)v194;
                        v194[2] = v191;
                        v191[2] = v216;
                        v153 = sub_72CBE0();
                        v154 = sub_7F89D0("__cxa_vec_cctor", &qword_4F18B60, v153, v275);
                        v63 = v154;
                        if ( v249 )
                        {
                          v290 = v154;
                          sub_8255D0(v249);
                          v63 = v290;
                        }
                      }
                    }
                    else
                    {
                      v142 = (__m128i *)sub_7F9160((__int64)a2);
                      v63 = sub_7FC1E0(v275, v275->m128i_i64[0], v142, v249, 0, 0, 0, v203);
                    }
                    sub_7E6A50(v63, v268->m128i_i32);
                    a9 = 0;
                    v40 = 0;
                    v248 = 0;
                    v272 = 0;
                    goto LABEL_90;
                  }
                  sub_7F1A60(*(__m128i **)(a1 + 64), v48, i, v50, (*(_BYTE *)(a1 + 72) & 8) != 0, 0, 0, v268->m128i_i32);
                  v212 = *(_QWORD *)(a1 + 56);
                  if ( v245 )
                  {
                    v205 = *(_BYTE *)(v245 + 8);
                    if ( sub_7F9D00(a1) )
                    {
                      v198 = (__m128i *)sub_7E8090(v275, 0);
                      sub_7FB7C0(*(_QWORD *)(*(_QWORD *)(v212 + 40) + 32LL), v248, v275->m128i_i64, 0, 0, 0, v268);
                      v275 = v198;
                    }
                    v101 = 0;
                    v102 = 1;
                    if ( *(_BYTE *)(v245 + 8) <= 1u && *(_BYTE *)(a1 + 48) == 5 )
                    {
                      sub_7FCD20(*(_QWORD **)(a1 + 80), *(_QWORD *)(v245 + 16), (__int64)a2, a4, (__int64)v268, &v297);
                      v101 = v297;
                      v102 = 2;
                    }
                    if ( v205 == 3 )
                    {
                      if ( (*(_BYTE *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 205LL) & 0x1C) == 0x10 )
                      {
                        v259 = *(_QWORD *)(*(_QWORD *)(qword_4F04C50 + 40LL) + 112LL);
                        v275[1].m128i_i64[0] = *(_QWORD *)(a1 + 64);
                        v234 = (__m128i *)sub_73B8B0(v275, 0);
                        v162 = sub_73E830(v259);
                        v234[1].m128i_i64[0] = (__int64)v162;
                        if ( v275[1].m128i_i64[0] )
                        {
                          v206 = v15;
                          v199 = a2;
                          v163 = (const __m128i *)v275[1].m128i_i64[0];
                          do
                          {
                            v164 = v162;
                            v162 = sub_73B8B0(v163, 0);
                            v164[2] = v162;
                            v163 = (const __m128i *)v163[1].m128i_i64[0];
                          }
                          while ( v163 );
                          v15 = v206;
                          a2 = v199;
                        }
                        v165 = sub_73E830(v259);
                        v166 = sub_7F0830(v165);
                        sub_7F8BA0((__int64)v166, 0, v268->m128i_i32, 0, (__int64)v300, (__int64)v304);
                        v167 = sub_7FDF40(*(_QWORD *)(a1 + 56), 2, 0);
                        sub_7F88F0((__int64)v167, v234, 0, v300);
                        v168 = sub_7FDF40(*(_QWORD *)(a1 + 56), 1, 0);
                        sub_7F88F0((__int64)v168, v275, 0, v304);
                        goto LABEL_268;
                      }
                      v102 = (*(_BYTE *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 205LL) >> 2) & 7;
                      if ( v102 == 2 )
                      {
                        v184 = sub_73E830(*(_QWORD *)(*(_QWORD *)(qword_4F04C50 + 40LL) + 112LL));
                        v102 = 2;
                        v101 = v184;
                      }
                    }
                    v254 = v101;
                    v103 = sub_7FDF40(v212, v102, 0);
                    v104 = (__m128i *)v254;
                    *(_QWORD *)(a1 + 56) = v103;
                    v105 = (__int64)v103;
                    if ( v254 )
                      v275[1].m128i_i64[0] = (__int64)v254;
                    else
                      v104 = v275;
                  }
                  else if ( sub_7F9D00(a1) )
                  {
                    v207 = (__m128i *)sub_7E8090(v275, 0);
                    sub_7FB7C0(*(_QWORD *)(*(_QWORD *)(v212 + 40) + 32LL), v248, v275->m128i_i64, 0, 0, 0, v268);
                    v169 = sub_7FDF40(v212, 1, 0);
                    v104 = v207;
                    *(_QWORD *)(a1 + 56) = v169;
                    v105 = (__int64)v169;
                    v275 = v207;
                  }
                  else
                  {
                    v143 = sub_7FDF40(v212, 1, 0);
                    v104 = v275;
                    *(_QWORD *)(a1 + 56) = v143;
                    v105 = (__int64)v143;
                  }
                  if ( v222 )
                    v104[1].m128i_i64[0] = v222;
                  else
                    v222 = (__int64)v104;
                  *(_QWORD *)(v222 + 16) = *(_QWORD *)(a1 + 64);
                  sub_7F88F0(v105, v275, 0, v268);
LABEL_268:
                  sub_7F6D70(v212, (_BYTE *)a2->m128i_i64[1]);
                  a9 = 0;
                  v40 = 0;
                  v248 = 0;
                  v272 = 0;
                  goto LABEL_90;
                }
                v157 = sub_7E2230(v228, 1, v111, v112, (__int64)v228, v113);
                v48 = v214;
                v114 = (__int64)v157;
              }
              if ( !v114 )
                goto LABEL_127;
              goto LABEL_279;
            }
            goto LABEL_35;
          }
          v13 = 10;
          *(_QWORD *)(a1 + 64) = sub_73F6B0(v128, 10);
        }
        else
        {
          if ( (_BYTE)v19 != 6 )
            goto LABEL_75;
          v85 = *(_QWORD *)(a1 + 56);
          if ( (*(_BYTE *)(v85 - 8) & 1) == 0 )
          {
            if ( v20 )
              goto LABEL_35;
LABEL_200:
            v86 = 0;
            if ( dword_4F077C4 == 2 )
              v86 = v241[5];
            v294 = 0;
            if ( *(char *)(a1 + 50) < 0 )
            {
              v288 = v86;
              sub_7FBA90(a1, (__int64)a2, v248, v268);
              v86 = v288;
            }
            if ( v15 )
            {
              v252 = (__int64)v86;
              sub_8032D0(*(_QWORD *)(a1 + 56), (_DWORD)a2, 0, (_DWORD)a3, a6, (_DWORD)v268, (__int64)&v294, a5);
              v40 = v252;
              v272 = v294;
              if ( v294 )
              {
LABEL_206:
                a9 = *(__m128i **)(a1 + 56);
                if ( *(char *)(v15 + 173) < 0 && (v283 = v40, v121 = sub_7E76A0((__int64)a9), v40 = v283, v121) )
                {
                  sub_7FACF0(v15, (__int64)a9);
                  v40 = v283;
                  v248 = 0;
                  v272 = 0;
                }
                else
                {
                  v248 = v200 | v219;
                  if ( v200 | v219 )
                  {
                    if ( v200 )
                    {
                      v258 = v40;
                      sub_7E1740(v295, (__int64)&v298);
                      v40 = v258;
                    }
                    v225 = v40;
                    v253 = sub_7F9430((__int64)a2, 1, 1);
                    v87 = sub_73C570(*(const __m128i **)(v15 + 120), 1);
                    v88 = sub_7E9300((__int64)v87, 1);
                    v88[11].m128i_i8[1] = 1;
                    v279 = (__int64)v88;
                    sub_7296C0(v304);
                    *(_QWORD *)(v279 + 184) = sub_740190((__int64)a9, 0, 0x40u);
                    sub_729730(v304[0].m128i_i32[0]);
                    v89 = sub_731250(v279);
                    sub_7E6A80(v253, 0x56u, (__int64)v89, v298.m128i_i32, (__int64)v253, (__int64)&v298);
                    v40 = v225;
                    *(_BYTE *)(v15 + 177) = 0;
                    *(_QWORD *)(v15 + 184) = 0;
                    v248 = 0;
                    v272 = 0;
                  }
                  else
                  {
                    v272 = 1;
                    if ( (a9[10].m128i_i8[10] & 0x40) != 0 )
                      *(_BYTE *)(a1 + 50) |= 0x80u;
                  }
                }
                goto LABEL_90;
              }
LABEL_237:
              v248 = 0;
              a9 = 0;
              goto LABEL_90;
            }
            v134 = *(_QWORD *)(a1 + 112);
            if ( v134 )
            {
              v256 = (__int64)v86;
              v286 = sub_731250(*(_QWORD *)(v134 + 8));
              v135 = sub_731250(*(_QWORD *)(*(_QWORD *)(a1 + 112) + 8LL));
              v287 = sub_698020(v135, 86, (__int64)v286, v136, v137, v138);
              sub_7E25D0(v287, v268->m128i_i32);
              sub_8032D0(*(_QWORD *)(a1 + 56), (_DWORD)a2, 0, (_DWORD)a3, a6, (_DWORD)v268, (__int64)&v294, a5);
              v82 = v287;
              v40 = v256;
              if ( !v294 )
                goto LABEL_195;
            }
            else
            {
              v227 = (__int64)v86;
              sub_8032D0(*(_QWORD *)(a1 + 56), (_DWORD)a2, 0, (_DWORD)a3, a6, (_DWORD)v268, (__int64)&v294, a5);
              v82 = 0;
              v40 = v227;
              v272 = v294;
              if ( !v294 )
                goto LABEL_237;
            }
LABEL_193:
            if ( !a9 )
            {
              v260 = v40;
              v292 = v82;
              v170 = sub_73C570(*(const __m128i **)(*(_QWORD *)(*(_QWORD *)(a1 + 112) + 8LL) + 120LL), 1);
              v171 = sub_7E9300((__int64)v170, 1);
              v172 = *(_QWORD *)(a1 + 56);
              v173 = v292;
              v174 = v260;
              v175 = (__int64)v171;
              if ( (v171[5].m128i_i8[9] & 2) != 0 )
              {
                v218 = v260;
                v237 = v292;
                v264 = *(_QWORD *)(a1 + 56);
                v293 = v171;
                v185 = sub_72F070((__int64)v171);
                v175 = (__int64)v293;
                v172 = v264;
                v173 = v237;
                v174 = v218;
                v176 = (_BYTE *)v185;
              }
              else
              {
                v176 = (_BYTE *)v171[2].m128i_i64[1];
              }
              v235 = v174;
              v261 = v173;
              sub_7333B0(v175, v176, 1, v172, 0);
              v177 = (const __m128i *)sub_731250(v175);
              sub_730620(*(_QWORD *)(*(_QWORD *)(v261 + 72) + 16LL), v177);
              v40 = v235;
              v248 = 0;
              v272 = 0;
              goto LABEL_90;
            }
            a9->m128i_i64[0] = *(_QWORD *)(a1 + 56);
LABEL_195:
            v251 = v82;
            if ( v82 )
            {
              v224 = v40;
              v304[0].m128i_i64[0] = (__int64)sub_724DC0();
              v278 = (const __m128i *)v304[0].m128i_i64[0];
              v83 = sub_72BA30(5u);
              sub_72BB40((__int64)v83, v278);
              v84 = (const __m128i *)sub_73A720((const __m128i *)v304[0].m128i_i64[0], (__int64)v278);
              sub_730620(v251, v84);
              sub_724E30((__int64)v304);
              a9 = 0;
              v40 = v224;
            }
            else
            {
              a9 = 0;
            }
            v248 = 0;
            v272 = 0;
            goto LABEL_90;
          }
          v13 = 0;
          *(_QWORD *)(a1 + 56) = sub_740190(v85, 0, 0xAu);
        }
LABEL_75:
        if ( !v20 )
          goto LABEL_36;
        goto LABEL_35;
      }
      v26 = *(__m128i **)(a1 + 56);
      if ( (v26[-1].m128i_i8[8] & 1) == 0 )
      {
        if ( !v20 )
        {
LABEL_78:
          sub_7EE560(v26, 0);
          v273 = *(__m128i **)(a1 + 56);
LABEL_79:
          v27 = v273->m128i_i64[0];
          if ( v248 || !(unsigned int)sub_8D3A70(v27) )
            goto LABEL_371;
          for ( k = v27; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
            ;
          if ( (*(_BYTE *)(k + 179) & 1) != 0 )
          {
            v272 = 0;
            v40 = 0;
            a9 = 0;
          }
          else
          {
LABEL_371:
            if ( (unsigned int)sub_8D2FB0(v27) )
            {
              v140 = (_QWORD *)sub_8D46C0(v27);
              v27 = sub_72D2E0(v140);
            }
            if ( (v273[1].m128i_i8[9] & 1) != 0 || (v139 = sub_8D3410(v27), v28 = 73, v139) )
              v28 = 86;
            if ( a2[1].m128i_i8[1] )
            {
              v29 = (_QWORD *)a2[2].m128i_i64[0];
              a2[2].m128i_i64[0] = *v29;
              *v29 = 0;
              *v29 = qword_4D03F78;
              qword_4D03F78 = v29;
              a2[1].m128i_i8[1] = 0;
            }
            v220 = v28;
            v30 = sub_7F9430((__int64)a2, 1, 1);
            v33 = (const __m128i *)sub_698020(v30, v220, (__int64)v273, v31, v220, v32);
            v38 = (__int64 *)v33;
            if ( !v248 && v33[3].m128i_i8[9] == 10 )
            {
              v282 = v33;
              sub_7FA680(v33, v248, v34, v35, v36, v37);
              v38 = (__int64 *)v282;
            }
            v39 = (int *)v268;
LABEL_88:
            sub_7E69E0(v38, v39);
LABEL_89:
            v248 = 0;
            v40 = 0;
            a9 = 0;
            v272 = 0;
          }
          goto LABEL_90;
        }
        goto LABEL_35;
      }
    }
    v13 = 10;
    v195 = sub_73B8B0(v26, 10);
    sub_7E99E0((__int64)v26);
    *(_QWORD *)(a1 + 56) = v195;
    if ( !v20 )
      goto LABEL_36;
    goto LABEL_35;
  }
LABEL_37:
  switch ( (char)v19 )
  {
    case 0:
      goto LABEL_89;
    case 1:
      if ( v15 )
        goto LABEL_89;
      v280 = sub_7F9140((__int64)a2);
      v92 = sub_8D3B80(v280);
      v93 = v280;
      if ( v92 || (v156 = sub_7E1F90(v280), v93 = v280, v156) )
      {
        if ( !a2[1].m128i_i8[1] )
        {
          v94 = 0;
LABEL_223:
          v226 = v94;
          v281 = v93;
          v95 = (__int64 *)sub_7F98A0((__int64)a2, 0);
          sub_7FB7C0(v281, v248, v95, 0, v226, 0, v268);
          v40 = 0;
          a9 = 0;
          v248 = 0;
          v272 = 0;
          break;
        }
LABEL_339:
        v94 = a2[2].m128i_i64[1];
        v93 = a2[3].m128i_i64[0];
        goto LABEL_223;
      }
      if ( a2[1].m128i_i8[1] )
        goto LABEL_339;
LABEL_246:
      v98 = sub_7F9430((__int64)a2, 1, 1);
      v99 = 0;
      if ( a3 )
        v99 = a3[1] != 0;
      sub_7FBC20(a1, 0, v98->m128i_i64, v248, v268, v99, (__int64)a2);
      v40 = 0;
      v248 = 1;
      a9 = 0;
      v272 = 0;
      break;
    case 2:
      goto LABEL_181;
    case 3:
      v96 = *(_QWORD *)(a1 + 56);
      if ( dword_4F077C4 != 2 )
      {
        if ( unk_4F07778 > 199900 || (v22 = dword_4F077C0) != 0 )
        {
          v97 = *(_QWORD **)(a1 + 56);
          if ( (a5 & 1) != 0 )
            sub_7D9EC0(v97, (const __m128i *)v13, v21, v22, v96, v14);
          else
            sub_7D9DD0(v97, v13, v21, v22, v96, v14);
        }
        goto LABEL_246;
      }
      if ( (*(_BYTE *)(a1 + 51) & 2) != 0 )
      {
        v144 = *(_QWORD *)(v96 + 72);
        v145 = *(_QWORD **)(v144 + 16);
        if ( *(_BYTE *)(v96 + 56) == 103 )
        {
          v178 = (_QWORD *)v145[2];
          v145[2] = 0;
          v217 = (__m128i *)v96;
          v262 = v144;
          v236 = v178;
          sub_7304E0((__int64)v145);
          v179 = sub_72CBE0();
          v180 = sub_73E130(v145, v179);
          v181 = v262;
          v263 = v180;
          *(_QWORD *)(v181 + 16) = v180;
          sub_7304E0((__int64)v236);
          v182 = sub_72CBE0();
          v183 = sub_73E130(v236, v182);
          v148 = v217;
          *((_QWORD *)v263 + 2) = v183;
        }
        else
        {
          v233 = *(_QWORD *)(v96 + 72);
          v257 = *(__m128i **)(a1 + 56);
          sub_7304E0((__int64)v145);
          v146 = sub_72CBE0();
          v147 = sub_73E130(v145, v146);
          v148 = v257;
          *(_QWORD *)(v233 + 16) = v147;
        }
        v289 = v148;
        v148->m128i_i64[0] = sub_72CBE0();
        sub_7304E0((__int64)v289);
        sub_7EE560(v289, 0);
        v39 = (int *)v268;
        v38 = (__int64 *)v289;
        goto LABEL_88;
      }
      if ( (a5 & 1) != 0 )
      {
        v284 = *(__m128i **)(a1 + 56);
        if ( v20 )
        {
          sub_7EE560(v284, 0);
          sub_7E7010(v284);
        }
        else
        {
          sub_7F2600((__int64)v284, 0);
        }
        v123 = v284;
      }
      else
      {
        v291 = *(__m128i **)(a1 + 56);
        sub_7EE560(v291, 0);
        v123 = v291;
      }
      v285 = (__int64 *)v123;
      v304[0].m128i_i64[0] = (__int64)sub_724DC0();
      if ( dword_4D03EB8[0] | v219
        || !(unsigned int)sub_8D2E30(*v285)
        || (v160 = (__m128i *)v304[0].m128i_i64[0], !(unsigned int)sub_717520(v285, v304[0].m128i_i64[0], 1)) )
      {
        sub_724E30((__int64)v304);
        goto LABEL_246;
      }
      v161 = (__m128i *)sub_724E50(v304[0].m128i_i64, v160);
      v161[-1].m128i_i8[8] &= ~8u;
      a9 = v161;
      sub_7EB800((__int64)v161, v160);
      v40 = 0;
      v248 = 0;
      v272 = 1;
      break;
    case 4:
      v90 = *(__m128i **)(a1 + 56);
      if ( v90[1].m128i_i8[8] == 17 )
      {
        v124 = *(_QWORD *)(a1 + 8);
        v125 = unk_4D03F50;
        v126 = unk_4D03F48;
        unk_4D03F48 = a2;
        unk_4D03F50 = v124;
        v255 = v125;
        sub_7EE3B0((__int64)v90);
        unk_4D03F50 = v255;
        unk_4D03F48 = v126;
      }
      else
      {
        sub_7F1E10(v90, a2, 0, 0);
      }
      sub_7E6A50(*(_QWORD **)(a1 + 56), v268->m128i_i32);
      if ( v245 )
      {
        v91 = *(__m128i **)(a1 + 16);
        a9 = v91;
        if ( !v91 )
        {
          v248 = 0;
          v272 = 0;
          goto LABEL_103;
        }
        if ( (((v91[12].m128i_i8[13] & 0x1C) - 8) & 0xF4) != 0
          || (*(_BYTE *)(*(_QWORD *)(v91[2].m128i_i64[1] + 32) + 176LL) & 0x10) == 0
          || *(_BYTE *)(v245 + 8) > 1u )
        {
          v248 = 0;
          v40 = 0;
          a9 = 0;
          v272 = 0;
          goto LABEL_91;
        }
        sub_7FCD20(*(_QWORD **)(a1 + 80), *(_QWORD *)(v245 + 16), (__int64)a2, a4, (__int64)v268, (__int64 **)v304);
        a9 = 0;
        v40 = 0;
        v248 = 0;
        v272 = 0;
      }
      else
      {
        a9 = 0;
        v40 = 0;
        v248 = 0;
        v272 = 0;
      }
      break;
    case 5:
      goto LABEL_120;
    case 6:
      goto LABEL_200;
    case 7:
      v26 = *(__m128i **)(a1 + 56);
      if ( v26 )
        goto LABEL_78;
      v273 = sub_7F9500((__int64)a3, (__int64)a2, 0);
      if ( (unsigned int)sub_8D3410(v273->m128i_i64[0]) )
        v273 = sub_7F9500((__int64)a3, (__int64)a2, 1);
      goto LABEL_79;
    case 8:
      if ( *(char *)(a1 + 50) < 0 )
      {
        v13 = (__int64)a2;
        sub_7FBA90(a1, (__int64)a2, v248, v268);
      }
      if ( (*(_BYTE *)(a1 + 72) & 1) == 0 )
      {
        sub_7EB190(*(_QWORD *)(a1 + 56), (__m128i *)v13);
        goto LABEL_246;
      }
      sub_7F9B80((__int64)v304);
      v80 = *(_QWORD *)(a1 + 56);
      v81 = **(_QWORD **)(a1 + 64);
      v294 = 0;
      v304[0].m128i_i64[1] = v81;
      v250 = v241[5];
      sub_8032D0(v80, (_DWORD)a2, 0, (unsigned int)v304, a6, (_DWORD)v268, (__int64)&v294, a5);
      v40 = (__int64)v250;
      v272 = v294;
      if ( !v294 )
        goto LABEL_237;
      if ( !v15 )
      {
        v82 = 0;
        goto LABEL_193;
      }
      goto LABEL_206;
    case 9:
      a9 = *(__m128i **)(a1 + 56);
      if ( !a9 )
        goto LABEL_275;
LABEL_181:
      if ( dword_4F077C4 == 2 )
      {
        sub_7EB190(*(_QWORD *)(a1 + 56), (__m128i *)v13);
      }
      else if ( unk_4F07778 > 199900 || dword_4F077C0 )
      {
        sub_7D8CF0(*(__m128i **)(a1 + 56));
      }
      v79 = *(_BYTE *)(a1 + 50);
      if ( v272 )
      {
        if ( v79 >= 0 )
        {
          v40 = 0;
          v248 = 0;
          a9 = *(__m128i **)(a1 + 56);
          break;
        }
      }
      else if ( v79 >= 0 )
      {
        goto LABEL_246;
      }
      sub_7FBA90(a1, (__int64)a2, v248, v268);
      goto LABEL_246;
    default:
      goto LABEL_363;
  }
LABEL_90:
  if ( !*(_QWORD *)(a1 + 16) )
    goto LABEL_103;
LABEL_91:
  if ( !*(_QWORD *)(a1 + 24) || (*(_BYTE *)(a2->m128i_i64[1] + 156) & 1) != 0 )
    goto LABEL_103;
  if ( !v267 || (*(_BYTE *)(a1 + 50) & 1) != 0 )
  {
    *(_BYTE *)(*(_QWORD *)(a1 + 80) + 128LL) = 1;
    if ( ((*(_BYTE *)(a1 + 48) - 6) & 0xFD) == 0 )
    {
      v41 = (__int64)v241[5];
      if ( v41 != v40 )
        sub_7F7260(v41, v40, v300, v304);
    }
    if ( ((a6 == 0) & *(_BYTE *)(a1 + 50)) != 0 && !v20 )
    {
      *(_BYTE *)(*(_QWORD *)(a1 + 80) + 132LL) = 1;
      if ( *(_QWORD *)(a1 + 24) )
        goto LABEL_105;
      v42 = *(_QWORD **)(a1 + 80);
      if ( !v42 )
        goto LABEL_105;
      goto LABEL_102;
    }
    sub_7F72E0(a1, a2, v245 == 0, (__int64)v241, v268->m128i_i32);
    goto LABEL_100;
  }
  if ( ((*(_BYTE *)(a1 + 48) - 6) & 0xFD) == 0 && dword_4D03EB8[0] && dword_4F077C4 == 2 )
  {
    v64 = qword_4D03F68;
    v65 = qword_4D03F68[5];
    if ( v65 != v40 && v65 )
    {
      do
      {
        if ( (*(_BYTE *)(v65 + 50) & 1) == 0 )
          break;
        *(_BYTE *)(*(_QWORD *)(v65 + 80) + 132LL) = 1;
        v65 = *(_QWORD *)(v65 + 32);
        if ( !v65 )
          break;
      }
      while ( v65 != v40 );
    }
    v64[5] = v65;
    sub_7E18B0();
  }
  v66 = sub_7F9140((__int64)a2);
  if ( a2[1].m128i_i8[1] || (unsigned int)sub_8D3410(v66) )
  {
    v242 = (const __m128i *)sub_724D50(6);
    v246 = (const __m128i *)sub_724D50(6);
    v67 = sub_7E1C10();
    sub_72BB40(v67, v242);
    v68 = sub_7E1C10();
    v211 = sub_7F7930(v68, 0, 0, (__int64)v300, (int *)&v297, (__int64)v304, 0);
    v223 = v211[4];
    sub_7FE6E0(*(_QWORD *)(a1 + 16), (__int64)a2, 1, 0, v300);
    *(_BYTE *)(a2->m128i_i64[1] + 170) |= 4u;
    sub_7FB010((__int64)v211, (unsigned int)v297, (__int64)v304);
    sub_72D3B0(v223, (__int64)v246, 1);
    v69 = v186;
  }
  else
  {
    if ( a2[1].m128i_i8[0] )
    {
      v242 = (const __m128i *)sub_724D50(6);
      v246 = (const __m128i *)sub_724D50(6);
LABEL_286:
      v118 = sub_7E1C10();
      sub_72BB40(v118, v242);
      v119 = sub_7FDF40(*(_QWORD *)(a1 + 16), 1, 0);
      sub_72D3B0((__int64)v119, (__int64)v246, 1);
      v69 = 0;
      v243 = sub_7F98A0((__int64)a2, 0);
      goto LABEL_172;
    }
    v231 = a2[2].m128i_i64[0];
    v242 = (const __m128i *)sub_724D50(6);
    v246 = (const __m128i *)sub_724D50(6);
    if ( v231 )
      goto LABEL_286;
    sub_72D510(a2->m128i_i64[1], (__int64)v242, 1);
    v129 = sub_7E1C10();
    sub_70FEE0((__int64)v242, v129, v130, v131, v132);
    v133 = sub_7FDF40(*(_QWORD *)(a1 + 16), 1, 0);
    v69 = (__int64)v246;
    sub_72D3B0((__int64)v133, (__int64)v246, 1);
  }
  v243 = sub_73A720(v242, v69);
LABEL_172:
  v70 = sub_7F7F20();
  v71 = sub_73A720(v246, v69);
  v72 = v70;
  v73 = (__m128i *)sub_73E110((__int64)v71, v70);
  v74 = (__m128i *)qword_4F18AE0;
  v75 = v73;
  if ( !qword_4F18AE0 )
  {
    v149 = sub_7E1C10();
    v72 = 0;
    v74 = sub_7E2190("__dso_handle", 0, v149, 1);
    qword_4F18AE0 = (__int64)v74;
    v74[10].m128i_i8[8] = v74[10].m128i_i8[8] & 0xF8 | 1;
  }
  v76 = sub_73E230((__int64)v74, v72);
  v77 = qword_4F18B00;
  v75[1].m128i_i64[0] = (__int64)v243;
  v243[2] = v76;
  if ( v77 )
  {
    v78 = sub_7F88E0(v77, v75);
  }
  else
  {
    v230 = sub_7E1C10();
    v244 = sub_7E1C10();
    v247 = sub_7F7F20();
    v120 = sub_72BA30(5u);
    v78 = sub_7F8AB0("__cxa_atexit", &qword_4F18B00, (__int64)v120, v247, v244, v230, 0, 0, 0, 0, v75);
  }
  sub_7E6A50(v78, v268->m128i_i32);
  sub_733B20((_QWORD *)a1);
LABEL_100:
  if ( !*(_QWORD *)(a1 + 24) )
  {
    v42 = *(_QWORD **)(a1 + 80);
    if ( v42 )
    {
LABEL_102:
      sub_7F93F0(v42);
      *(_QWORD *)(a1 + 80) = 0;
    }
  }
LABEL_103:
  if ( v20 )
  {
    sub_7E7530((__int64)v20, (__int64)v268);
    sub_7E1AA0();
  }
LABEL_105:
  if ( v17 )
  {
    sub_7E7530((__int64)v17, (__int64)v268);
    sub_7E1AA0();
  }
  if ( v296 )
    sub_7F6E70(v296, a7->m128i_i32);
  if ( !v15 )
  {
LABEL_115:
    sub_7259F0(a1, 0);
    v272 = 0;
    dword_4F07508[0] = v240;
    LOWORD(dword_4F07508[1]) = v266;
    result = (__int64)dword_4D03F38;
    dword_4D03F38[0] = v265;
    LOWORD(dword_4D03F38[1]) = v239;
    if ( !a8 )
      return result;
    goto LABEL_116;
  }
  if ( !v272 )
  {
    if ( *(_BYTE *)(a1 + 48) == 1 )
    {
      *(_BYTE *)(v15 + 177) = 3;
    }
    else if ( (!v267 || (*(_BYTE *)(v15 + 89) & 1) != 0 || !dword_4F0696C || dword_4F077C4 != 2 || v200)
           && (*(char *)(a1 + 50) >= 0 || (unsigned int)sub_8D4070(*(_QWORD *)(v15 + 120)) | v248) )
    {
      *(_BYTE *)(v15 + 177) = 0;
    }
    else
    {
      *(_BYTE *)(v15 + 177) = 3;
      sub_7EC360(v15, (__m128i *)(v15 + 177), (__int64 *)(v15 + 184));
    }
    goto LABEL_115;
  }
  sub_7F5C00((__int64)a9);
  if ( v267 )
  {
    if ( (a9[-1].m128i_i8[8] & 1) == 0 )
    {
      v304[0].m128i_i32[0] = 0;
      sub_7296C0(v304);
      a9 = sub_7401F0((__int64)a9);
      sub_729730(v304[0].m128i_i32[0]);
    }
    *(_BYTE *)(v15 + 177) = 1;
    *(_QWORD *)(v15 + 184) = a9;
    goto LABEL_115;
  }
  sub_7259F0(a1, 2u);
  *(_QWORD *)(a1 + 56) = a9;
  dword_4F07508[0] = v240;
  LOWORD(dword_4F07508[1]) = v266;
  dword_4D03F38[0] = v265;
  LOWORD(dword_4D03F38[1]) = v239;
  if ( a8 )
  {
LABEL_116:
    result = (__int64)a8;
    *a8 = v272;
    return result;
  }
  result = v238 & 0xFB;
  if ( (v238 & 0xFB) == 2 || v238 == 8 )
    return sub_7FCA60(v15, a1);
  return result;
}
