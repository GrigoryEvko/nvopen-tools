// Function: sub_1864060
// Address: 0x1864060
//
__int64 __fastcall sub_1864060(
        __int64 a1,
        _QWORD *a2,
        __int64 (__fastcall *a3)(_QWORD, _QWORD),
        __int64 a4,
        __m128 a5,
        __m128i a6,
        __m128i a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 v12; // r15
  const char *v15; // rax
  unsigned __int64 v16; // rdx
  __int64 v17; // rsi
  unsigned int v18; // r14d
  unsigned __int8 v19; // dl
  unsigned int v20; // eax
  char v21; // dl
  char v22; // cl
  unsigned __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 *v25; // r8
  int v26; // r9d
  double v27; // xmm4_8
  double v28; // xmm5_8
  unsigned __int64 v29; // rax
  __int64 v30; // r13
  _QWORD *v31; // rax
  unsigned __int8 v32; // r12
  unsigned __int64 v33; // rax
  double v34; // xmm4_8
  double v35; // xmm5_8
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // rdi
  __int64 v42; // r13
  __int64 v43; // rbx
  _QWORD *v44; // rax
  char v45; // al
  __int64 k; // rbx
  char v47; // al
  __int64 i; // rbx
  __int64 v49; // rax
  __int64 v50; // rcx
  __int64 *v51; // r8
  int v52; // r9d
  double v53; // xmm4_8
  double v54; // xmm5_8
  __int64 *v55; // rax
  __int64 v56; // rsi
  __int64 v57; // rdi
  int v58; // r12d
  _BOOL4 v59; // ebx
  _QWORD *v60; // rdi
  unsigned __int8 v61; // al
  double v62; // xmm4_8
  double v63; // xmm5_8
  __int64 j; // rbx
  _QWORD *v65; // r13
  unsigned __int8 v66; // al
  _QWORD *v67; // r12
  unsigned __int8 v68; // al
  const char **v69; // rdi
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // rbx
  _QWORD *v73; // r14
  const char *v74; // rax
  __int64 v75; // rdx
  _QWORD *v76; // rax
  double v77; // xmm4_8
  double v78; // xmm5_8
  __int64 v79; // r12
  __int64 v80; // r14
  _QWORD *v81; // rdi
  __int64 v82; // rbx
  unsigned int v83; // edx
  __int64 v84; // r14
  _QWORD *v85; // rax
  int v86; // r8d
  int v87; // r9d
  _QWORD *v88; // rbx
  __int64 v89; // rdx
  char v90; // bl
  _QWORD *v91; // rax
  __int64 v92; // r13
  __int64 v93; // rdx
  double v94; // xmm4_8
  double v95; // xmm5_8
  _QWORD *v96; // rax
  __int64 *v97; // rax
  const char *v98; // rax
  __int64 v99; // rbx
  __int64 v100; // rax
  __int64 v101; // rdx
  __int16 v102; // bx
  _QWORD *v103; // r14
  __int64 v104; // rcx
  __int64 v105; // rbx
  __int64 v106; // rdx
  __int64 v107; // rcx
  unsigned int v108; // ebx
  __int64 v109; // rax
  __int64 v110; // rbx
  _QWORD *v111; // rax
  unsigned __int64 v112; // r13
  __int64 v113; // rcx
  __int64 v114; // r10
  _QWORD *v115; // rdi
  __int64 *v116; // rax
  __int64 v117; // rax
  __int64 v118; // rcx
  __int64 *v119; // rdx
  unsigned int v120; // ebx
  int v121; // ebx
  __int64 v122; // r12
  __int16 v123; // bx
  _QWORD *v124; // rdi
  __int64 v125; // rdi
  _QWORD *v126; // rax
  __int64 v127; // r13
  const char *v128; // rax
  __int64 *v129; // rdx
  __int16 v130; // bx
  _QWORD *v131; // r12
  __int64 v132; // rdx
  _QWORD *v133; // rax
  __int64 v134; // rbx
  double v135; // xmm4_8
  double v136; // xmm5_8
  __int64 *v137; // rcx
  unsigned __int8 v138; // bl
  _QWORD *v139; // rax
  __int64 v140; // rax
  __int64 v141; // rax
  __int64 *v142; // r15
  __int64 v143; // rax
  __int64 v144; // rax
  __int64 v145; // rsi
  __int64 v146; // r12
  __int64 v147; // rsi
  __int64 v148; // r12
  __int64 v149; // r12
  __int64 v150; // r12
  __int64 v151; // rdi
  __int64 v152; // r13
  __int64 v153; // rax
  __int64 v154; // rbx
  __int64 v155; // rdi
  __int64 v156; // r13
  __int64 v157; // rdi
  __int64 v158; // r13
  __int64 v159; // rdi
  __int64 v160; // r13
  int v161; // eax
  __int64 v162; // rax
  __int64 v163; // rbx
  __int64 v164; // rbx
  __int64 v165; // rdi
  unsigned __int64 v166; // r14
  int v167; // eax
  __int64 v168; // rax
  __int64 v169; // rbx
  __int64 v170; // rbx
  __int64 v171; // rdi
  unsigned __int64 v172; // r14
  __int64 v173; // rax
  __int64 v174; // rax
  __int64 v175; // rax
  __int64 v176; // rax
  __int64 v177; // rax
  __int64 v178; // rax
  __int64 v179; // rax
  __int64 v180; // rax
  __int64 v181; // rax
  __int64 v182; // r12
  unsigned __int64 v183; // r13
  __int64 v184; // rax
  __int64 v185; // rax
  __int64 v186; // r14
  __int64 v187; // rax
  __int64 v188; // rax
  __int64 v189; // rax
  __int64 v190; // rax
  __int64 v191; // rax
  __int64 v192; // r12
  unsigned __int64 v193; // r13
  __int64 v194; // rax
  __int64 v195; // rax
  __int64 v196; // r14
  __int64 v197; // rax
  _QWORD *v198; // rax
  int v199; // eax
  _QWORD *v200; // rax
  int v201; // eax
  __int64 v202; // rax
  int v203; // eax
  int v204; // eax
  __int64 *v205; // r13
  __int64 *v206; // rbx
  __int64 v207; // rsi
  _QWORD *v208; // rax
  __int64 v209; // rdx
  unsigned __int64 v210; // rsi
  __int64 v211; // rdx
  __int64 v212; // rdx
  unsigned __int64 v213; // rcx
  __int64 v214; // rdx
  __int64 v215; // rdx
  unsigned __int64 v216; // rcx
  __int64 v217; // rdx
  signed __int64 v218; // rax
  __int64 v219; // rbx
  __int64 v220; // rdi
  unsigned __int64 v221; // r14
  _QWORD *v222; // rax
  __int64 v223; // rax
  __int64 v224; // rax
  __int64 v225; // rax
  __int64 v226; // rax
  __int64 v227; // rax
  __int64 v228; // rax
  __int64 v229; // r12
  unsigned __int64 v230; // r13
  __int64 v231; // rax
  __int64 v232; // rax
  __int64 v233; // r14
  __int64 v234; // rax
  __int64 v235; // r13
  __int64 v236; // rbx
  __int64 v237; // r14
  int v238; // r8d
  int v239; // r9d
  const char *v240; // r12
  __int64 *v241; // rbx
  __int64 *v242; // r14
  __int64 v243; // r15
  __int64 v244; // r12
  __int64 v245; // rax
  __int64 v246; // rax
  int v247; // eax
  __int64 v248; // rax
  __int64 v249; // rbx
  __int64 v250; // rbx
  __int64 v251; // rdi
  unsigned __int64 v252; // r14
  __int64 v253; // rax
  __int64 v254; // rax
  __int64 v255; // rax
  __int64 v256; // rax
  __int64 v257; // rax
  __int64 v258; // rax
  __int64 v259; // r12
  unsigned __int64 v260; // r13
  __int64 v261; // rax
  __int64 v262; // rax
  __int64 v263; // r14
  __int64 v264; // rax
  _QWORD *v265; // rax
  int v266; // eax
  __int64 v267; // [rsp+10h] [rbp-1B0h]
  __int64 v268; // [rsp+10h] [rbp-1B0h]
  __int64 v269; // [rsp+10h] [rbp-1B0h]
  __int64 v270; // [rsp+10h] [rbp-1B0h]
  unsigned __int64 v271; // [rsp+18h] [rbp-1A8h]
  unsigned __int64 v272; // [rsp+18h] [rbp-1A8h]
  unsigned __int64 v273; // [rsp+18h] [rbp-1A8h]
  unsigned __int64 v274; // [rsp+18h] [rbp-1A8h]
  unsigned __int64 v275; // [rsp+18h] [rbp-1A8h]
  unsigned __int64 v276; // [rsp+18h] [rbp-1A8h]
  unsigned __int64 v277; // [rsp+18h] [rbp-1A8h]
  unsigned __int64 v278; // [rsp+18h] [rbp-1A8h]
  __int64 v279; // [rsp+20h] [rbp-1A0h]
  __int64 v280; // [rsp+20h] [rbp-1A0h]
  __int64 v281; // [rsp+20h] [rbp-1A0h]
  __int64 v282; // [rsp+20h] [rbp-1A0h]
  __int64 v283; // [rsp+20h] [rbp-1A0h]
  __int64 v284; // [rsp+20h] [rbp-1A0h]
  __int64 v285; // [rsp+20h] [rbp-1A0h]
  __int64 v286; // [rsp+20h] [rbp-1A0h]
  __int64 v287; // [rsp+20h] [rbp-1A0h]
  __int64 v288; // [rsp+20h] [rbp-1A0h]
  __int64 v289; // [rsp+20h] [rbp-1A0h]
  __int64 v290; // [rsp+20h] [rbp-1A0h]
  unsigned __int64 v291; // [rsp+28h] [rbp-198h]
  unsigned __int64 v292; // [rsp+28h] [rbp-198h]
  __int64 v293; // [rsp+28h] [rbp-198h]
  unsigned __int64 v294; // [rsp+28h] [rbp-198h]
  __int64 v295; // [rsp+28h] [rbp-198h]
  __int64 v296; // [rsp+28h] [rbp-198h]
  unsigned __int64 v297; // [rsp+28h] [rbp-198h]
  __int64 v298; // [rsp+28h] [rbp-198h]
  char *v299; // [rsp+30h] [rbp-190h]
  const char *v300; // [rsp+38h] [rbp-188h]
  __int64 v301; // [rsp+40h] [rbp-180h]
  char v302; // [rsp+4Fh] [rbp-171h]
  char *v303; // [rsp+50h] [rbp-170h]
  __int64 v304; // [rsp+58h] [rbp-168h]
  __int64 v305; // [rsp+58h] [rbp-168h]
  __int64 v306; // [rsp+58h] [rbp-168h]
  __int64 (__fastcall *v307)(__int64, __int64); // [rsp+60h] [rbp-160h]
  __int64 v308; // [rsp+60h] [rbp-160h]
  __int64 *v309; // [rsp+60h] [rbp-160h]
  __int64 v310; // [rsp+60h] [rbp-160h]
  __int64 v311; // [rsp+68h] [rbp-158h]
  __int64 v312; // [rsp+68h] [rbp-158h]
  __int64 v313; // [rsp+68h] [rbp-158h]
  char v314; // [rsp+68h] [rbp-158h]
  char v315; // [rsp+68h] [rbp-158h]
  char v316; // [rsp+68h] [rbp-158h]
  __int64 v317; // [rsp+68h] [rbp-158h]
  _QWORD *v319; // [rsp+68h] [rbp-158h]
  __int64 v320; // [rsp+68h] [rbp-158h]
  char *v321; // [rsp+70h] [rbp-150h]
  unsigned __int8 v322; // [rsp+70h] [rbp-150h]
  unsigned __int8 v323; // [rsp+70h] [rbp-150h]
  bool v324; // [rsp+78h] [rbp-148h]
  unsigned __int8 v325; // [rsp+78h] [rbp-148h]
  char *v326; // [rsp+78h] [rbp-148h]
  __int64 (__fastcall *v327)(_QWORD, _QWORD); // [rsp+80h] [rbp-140h]
  bool v328; // [rsp+80h] [rbp-140h]
  __int64 v329; // [rsp+80h] [rbp-140h]
  __int64 v330; // [rsp+80h] [rbp-140h]
  __int64 v331; // [rsp+88h] [rbp-138h]
  int v332; // [rsp+88h] [rbp-138h]
  __int64 v333; // [rsp+88h] [rbp-138h]
  __int64 v334; // [rsp+88h] [rbp-138h]
  _BYTE *v335; // [rsp+90h] [rbp-130h]
  unsigned int v336; // [rsp+90h] [rbp-130h]
  bool v337; // [rsp+90h] [rbp-130h]
  __int64 *v338; // [rsp+90h] [rbp-130h]
  __int64 v340; // [rsp+98h] [rbp-128h]
  unsigned int v341; // [rsp+98h] [rbp-128h]
  unsigned int v342; // [rsp+98h] [rbp-128h]
  __int64 *v343; // [rsp+98h] [rbp-128h]
  __int64 *v344; // [rsp+A0h] [rbp-120h] BYREF
  __int64 v345; // [rsp+A8h] [rbp-118h] BYREF
  char v346; // [rsp+B0h] [rbp-110h] BYREF
  unsigned __int8 v347; // [rsp+B1h] [rbp-10Fh]
  int v348; // [rsp+B4h] [rbp-10Ch]
  __int64 v349; // [rsp+B8h] [rbp-108h]
  __int64 v350; // [rsp+C0h] [rbp-100h]
  char v351; // [rsp+C8h] [rbp-F8h]
  int v352; // [rsp+CCh] [rbp-F4h]
  const char *v353; // [rsp+D0h] [rbp-F0h] BYREF
  __int64 *v354; // [rsp+D8h] [rbp-E8h]
  __int64 v355; // [rsp+E0h] [rbp-E0h]
  __int64 *v356; // [rsp+E8h] [rbp-D8h]
  char *v357; // [rsp+F0h] [rbp-D0h] BYREF
  __int64 v358; // [rsp+F8h] [rbp-C8h]
  _BYTE v359[32]; // [rsp+100h] [rbp-C0h] BYREF
  const char **v360; // [rsp+120h] [rbp-A0h] BYREF
  __int64 v361; // [rsp+128h] [rbp-98h]
  _QWORD v362[18]; // [rsp+130h] [rbp-90h] BYREF

  v12 = a1;
  v15 = sub_1649960(a1);
  if ( v16 > 4 && *(_DWORD *)v15 == 1836477548 && v15[4] == 46 )
    return 0;
  sub_1ACF5D0(&v346);
  v17 = (__int64)&v346;
  v18 = sub_1ACF600(a1, &v346);
  if ( (_BYTE)v18 )
    return 0;
  v19 = *(_BYTE *)(a1 + 32);
  v20 = (v19 & 0xF) - 7;
  if ( !v346 )
  {
    v21 = v19 >> 6;
    if ( v21 != 2 )
    {
      if ( v20 > 1 )
      {
        v22 = 1;
        if ( v21 == 1 )
          return v18;
      }
      else
      {
        v22 = 2;
      }
      v18 = 1;
      *(_BYTE *)(a1 + 32) = (v22 << 6) | *(_BYTE *)(a1 + 32) & 0x3F;
    }
  }
  if ( v20 <= 1 && *(_BYTE *)(a1 + 16) == 3 && (*(_BYTE *)(a1 + 80) & 1) == 0 && !sub_15E4F60(a1) )
  {
    v335 = (_BYTE *)sub_1632FA0(*(_QWORD *)(a1 + 40));
    if ( v351 )
      goto LABEL_25;
    v331 = v350;
    if ( !v350 )
      goto LABEL_25;
    v29 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 24) + 8LL);
    if ( (unsigned __int8)v29 > 0x10u )
      goto LABEL_25;
    v23 = 100990;
    if ( !_bittest64((const __int64 *)&v23, v29) )
      goto LABEL_25;
    if ( *(_DWORD *)(*(_QWORD *)a1 + 8LL) >> 8 )
      goto LABEL_25;
    v324 = (*(_BYTE *)(a1 + 80) & 2) != 0;
    if ( (*(_BYTE *)(a1 + 80) & 2) != 0 )
      goto LABEL_25;
    if ( *(_QWORD *)(a1 + 8) )
    {
      v327 = a3;
      v30 = *(_QWORD *)(a1 + 8);
      do
      {
        v31 = sub_1648700(v30);
        v23 = *((unsigned __int8 *)v31 + 16);
        if ( (unsigned __int8)v23 <= 0x17u )
        {
          if ( (_BYTE)v23 != 5 )
            goto LABEL_25;
          for ( i = v31[1]; i; i = *(_QWORD *)(i + 8) )
          {
            if ( *((_BYTE *)sub_1648700(i) + 16) <= 0x17u )
              goto LABEL_25;
          }
        }
        v30 = *(_QWORD *)(v30 + 8);
      }
      while ( v30 );
      a3 = v327;
    }
    v17 = 27;
    v302 = sub_1560180(v331 + 112, 27);
    if ( !v302 )
      goto LABEL_25;
    v311 = v350;
    v304 = a4;
    v301 = sub_1632FA0(*(_QWORD *)(a1 + 40));
    v357 = v359;
    v360 = (const char **)v362;
    v358 = 0x400000000LL;
    v361 = 0x400000000LL;
    v307 = a3;
    for ( j = *(_QWORD *)(a1 + 8); j; j = *(_QWORD *)(j + 8) )
    {
      v65 = sub_1648700(j);
      v66 = *((_BYTE *)v65 + 16);
      if ( v66 <= 0x17u )
      {
        if ( v66 != 5 || *((_WORD *)v65 + 9) != 47 )
          goto LABEL_99;
LABEL_95:
        while ( 1 )
        {
          v65 = (_QWORD *)v65[1];
          if ( !v65 )
            break;
          v67 = sub_1648700((__int64)v65);
          v68 = *((_BYTE *)v67 + 16);
          if ( v68 <= 0x17u )
            goto LABEL_99;
          if ( v68 == 54 )
          {
            if ( (unsigned int)v358 >= HIDWORD(v358) )
            {
              v17 = (__int64)v359;
              sub_16CD150((__int64)&v357, v359, 0, 8, (int)v25, v26);
            }
            *(_QWORD *)&v357[8 * (unsigned int)v358] = v67;
            LODWORD(v358) = v358 + 1;
          }
          else
          {
            if ( v68 != 55 )
              goto LABEL_99;
            if ( (unsigned int)v361 >= HIDWORD(v361) )
            {
              v17 = (__int64)v362;
              sub_16CD150((__int64)&v360, v362, 0, 8, (int)v25, v26);
            }
            v360[(unsigned int)v361] = (const char *)v67;
            LODWORD(v361) = v361 + 1;
          }
        }
      }
      else
      {
        switch ( v66 )
        {
          case 'G':
            goto LABEL_95;
          case '6':
            if ( (unsigned int)v358 >= HIDWORD(v358) )
            {
              v17 = (__int64)v359;
              sub_16CD150((__int64)&v357, v359, 0, 8, (int)v25, v26);
            }
            *(_QWORD *)&v357[8 * (unsigned int)v358] = v65;
            LODWORD(v358) = v358 + 1;
            break;
          case '7':
            if ( (unsigned int)v361 >= HIDWORD(v361) )
            {
              v17 = (__int64)v362;
              sub_16CD150((__int64)&v360, v362, 0, 8, (int)v25, v26);
            }
            v360[(unsigned int)v361] = (const char *)v65;
            LODWORD(v361) = v361 + 1;
            break;
          default:
            goto LABEL_99;
        }
      }
    }
    v17 = v311;
    v141 = v307(v304, v311);
    v23 = (unsigned int)v358;
    v300 = (const char *)v141;
    if ( (unsigned int)v358 * (unsigned __int64)(unsigned int)v361 > 0x64 )
    {
LABEL_99:
      v69 = v360;
      goto LABEL_100;
    }
    v323 = v18;
    v303 = v357;
    v299 = &v357[8 * (unsigned int)v358];
    while ( 1 )
    {
      v142 = (__int64 *)v360;
      v69 = v360;
      if ( v299 == v303 )
      {
        v18 = v323;
        v12 = a1;
        v324 = v302;
LABEL_100:
        if ( v69 != v362 )
          _libc_free((unsigned __int64)v69);
        if ( v357 != v359 )
          _libc_free((unsigned __int64)v357);
        if ( v324 )
        {
          v70 = sub_1632FA0(*(_QWORD *)(v12 + 40));
          v71 = *(_QWORD *)(v350 + 80);
          if ( !v71 )
LABEL_468:
            JUMPOUT(0x41B6A1);
          v72 = *(_QWORD *)(v71 + 24);
          if ( v72 )
            v72 -= 24;
          v73 = *(_QWORD **)(v12 + 24);
          v341 = *(_DWORD *)(v70 + 4);
          v74 = sub_1649960(v12);
          LOWORD(v362[0]) = 261;
          v357 = (char *)v74;
          v358 = v75;
          v360 = (const char **)&v357;
          v76 = sub_1648A60(64, 1u);
          v79 = (__int64)v76;
          if ( v76 )
            sub_15F8BC0((__int64)v76, v73, v341, 0, (__int64)&v360, v72);
          v80 = *(_QWORD *)(v12 - 24);
          if ( *(_BYTE *)(v80 + 16) != 9 )
          {
            v81 = sub_1648A60(64, 2u);
            if ( v81 )
              sub_15F9660((__int64)v81, v80, v79, v72);
          }
          v82 = *(_QWORD *)(v12 + 8);
          v83 = 0;
          v357 = v359;
          v84 = v82;
          v358 = 0x400000000LL;
          while ( v84 )
          {
            v342 = v83;
            v85 = sub_1648700(v84);
            v83 = v342;
            v88 = v85;
            if ( *((_BYTE *)v85 + 16) == 5 )
            {
              if ( HIDWORD(v358) <= v342 )
                sub_16CD150((__int64)&v357, v359, 0, 8, v86, v87);
              *(_QWORD *)&v357[8 * (unsigned int)v358] = v88;
              LODWORD(v358) = v358 + 1;
              v83 = v358;
            }
            v84 = *(_QWORD *)(v84 + 8);
          }
          v320 = v79;
          v310 = v12;
          v360 = (const char **)v362;
          v343 = (__int64 *)v357;
          v361 = 0x400000000LL;
          v338 = (__int64 *)&v357[8 * v83];
          while ( v338 != v343 )
          {
            v235 = *v343;
            v236 = 0;
            LODWORD(v361) = 0;
            v237 = *(_QWORD *)(v235 + 8);
            if ( v237 )
            {
              do
              {
                v240 = (const char *)sub_1648700(v237);
                if ( HIDWORD(v361) <= (unsigned int)v236 )
                {
                  sub_16CD150((__int64)&v360, v362, 0, 8, v238, v239);
                  v236 = (unsigned int)v361;
                }
                v360[v236] = v240;
                v236 = (unsigned int)(v361 + 1);
                LODWORD(v361) = v361 + 1;
                v237 = *(_QWORD *)(v237 + 8);
              }
              while ( v237 );
              v241 = (__int64 *)&v360[v236];
              if ( v360 != (const char **)v241 )
              {
                v242 = (__int64 *)v360;
                do
                {
                  v243 = *v242++;
                  v244 = sub_1596970(v235);
                  sub_15F2120(v244, v243);
                  sub_1648780(v243, v235, v244);
                }
                while ( v241 != v242 );
              }
            }
            sub_159D850(v235);
            ++v343;
          }
          if ( v360 != v362 )
            _libc_free((unsigned __int64)v360);
          if ( v357 != v359 )
            _libc_free((unsigned __int64)v357);
          sub_164D160(v310, v320, a5, *(double *)a6.m128i_i64, *(double *)a7.m128i_i64, a8, v77, v78, a11, a12);
          sub_15E55B0(v310);
          return v324;
        }
LABEL_25:
        v32 = v347;
        if ( v347 )
        {
          if ( v348 <= 1 )
          {
            *(_BYTE *)(v12 + 80) |= 1u;
            v17 = *(_QWORD *)(v12 - 24);
            sub_185FD30(
              v12,
              v17,
              v335,
              (__int64)a2,
              a5,
              *(double *)a6.m128i_i64,
              *(double *)a7.m128i_i64,
              a8,
              v27,
              v28,
              a11,
              a12);
            if ( !*(_QWORD *)(v12 + 8) )
              goto LABEL_58;
          }
          v33 = *(unsigned __int8 *)(**(_QWORD **)(v12 - 24) + 8LL);
          if ( (unsigned __int8)v33 > 0x10u || (v37 = 100990, !_bittest64(&v37, v33)) )
          {
            v17 = sub_1632FA0(*(_QWORD *)(v12 + 40));
            if ( sub_185E850(v12, v17, a5, a6, a7, a8, v34, v35, a11, a12) )
              return 1;
          }
          if ( v348 != 2 || !v349 || *(_DWORD *)(*(_QWORD *)v12 + 8LL) >> 8 == 3 )
            return v18;
          if ( *(_BYTE *)(v349 + 16) > 0x10u || *(_BYTE *)(*(_QWORD *)(v12 - 24) + 16LL) != 9 )
          {
            v332 = v352;
            v38 = sub_1649C60(v349);
            v41 = *(_QWORD *)(v12 - 24);
            v42 = v38;
            if ( *(_BYTE *)(*(_QWORD *)v41 + 8LL) != 15 )
              goto LABEL_42;
            v328 = sub_1593BB0(v41, v17, v39, v40);
            if ( !v328 )
              goto LABEL_42;
            v49 = **(_QWORD **)(v12 - 24);
            if ( *(_BYTE *)(v49 + 8) == 16 )
              v49 = **(_QWORD **)(v49 + 16);
            if ( sub_15E4690(0, *(_DWORD *)(v49 + 8) >> 8) )
              goto LABEL_42;
            if ( *(_BYTE *)(v42 + 16) > 0x10u )
            {
              v91 = (_QWORD *)sub_140B220(v42, a2);
              v92 = (__int64)v91;
              if ( v91 )
              {
                v93 = sub_140B2D0(v91);
                if ( v93 )
                {
                  if ( (unsigned __int8)sub_18612A0(
                                          v12,
                                          v92,
                                          v93,
                                          v332,
                                          v335,
                                          (__int64)a2,
                                          a5,
                                          a6,
                                          *(double *)a7.m128i_i64,
                                          a8,
                                          v94,
                                          v95,
                                          a11,
                                          a12) )
                    return 1;
                }
              }
              goto LABEL_42;
            }
            v55 = *(__int64 **)(v12 - 24);
            v56 = *v55;
            if ( *v55 != *(_QWORD *)v42 )
              v42 = sub_15A4AD0((__int64 ***)v42, v56);
            v325 = v32;
            v57 = *(_QWORD *)(v12 + 8);
            v58 = 0;
            v59 = v328;
            while ( 2 )
            {
              if ( !v57 )
              {
                v89 = v59;
                v90 = v58;
                v32 = v325;
                if ( (_BYTE)v89 )
                {
                  if ( (*(_BYTE *)(v12 + 32) & 0xF) == 8 || !sub_185B2A0(v12, v56, v89, v50, v51, v52) )
                  {
                    sub_185FD30(
                      v12,
                      0,
                      v335,
                      (__int64)a2,
                      a5,
                      *(double *)a6.m128i_i64,
                      *(double *)a7.m128i_i64,
                      a8,
                      v53,
                      v54,
                      a11,
                      a12);
                    if ( !*(_QWORD *)(v12 + 8) )
                      goto LABEL_130;
                    return 1;
                  }
                  v90 |= sub_185B9F0(v12, a2);
                  if ( !*(_QWORD *)(v12 + 8) )
                  {
LABEL_130:
                    sub_15E55B0(v12);
                    return 1;
                  }
                }
                if ( v90 )
                  return 1;
LABEL_42:
                v340 = v349;
                if ( *(_BYTE *)(v349 + 16) > 0x10u )
                  return v18;
                if ( v352 )
                  return v18;
                v43 = *(_QWORD *)(v12 + 24);
                v44 = (_QWORD *)sub_16498A0(v12);
                if ( v43 == sub_1643320(v44) )
                  return v18;
                v45 = *(_BYTE *)(v43 + 8);
                if ( (unsigned __int8)(v45 - 1) <= 5u || (unsigned __int8)(v45 - 15) <= 1u )
                  return v18;
                for ( k = *(_QWORD *)(v12 + 8); k; k = *(_QWORD *)(k + 8) )
                {
                  if ( (unsigned __int8)(*((_BYTE *)sub_1648700(k) + 16) - 54) > 1u )
                    return v18;
                }
                v96 = (_QWORD *)sub_16498A0(v12);
                v329 = sub_1643320(v96);
                v97 = (__int64 *)sub_16498A0(v12);
                v334 = sub_159C540(v97);
                v98 = sub_1649960(v12);
                v99 = *(unsigned __int8 *)(v12 + 33);
                LOWORD(v362[0]) = 773;
                v357 = (char *)v98;
                v361 = (__int64)".b";
                v100 = *(_QWORD *)v12;
                v358 = v101;
                v102 = ((unsigned __int8)v99 >> 2) & 7;
                v360 = (const char **)&v357;
                v336 = *(_DWORD *)(v100 + 8) >> 8;
                v103 = sub_1648A60(88, 1u);
                if ( v103 )
                  sub_15E5070((__int64)v103, v329, 0, 7, v334, (__int64)&v360, v102, v336, 0);
                sub_15E6480((__int64)v103, v12);
                sub_1631BE0(*(_QWORD *)(v12 + 40) + 8LL, (__int64)v103);
                v104 = *(_QWORD *)(v12 + 56);
                v103[8] = v12 + 56;
                v104 &= 0xFFFFFFFFFFFFFFF8LL;
                v103[7] = v104 | v103[7] & 7LL;
                *(_QWORD *)(v104 + 8) = v103 + 7;
                v105 = *(_QWORD *)(v12 - 24);
                v330 = v105;
                *(_QWORD *)(v12 + 56) = *(_QWORD *)(v12 + 56) & 7LL | (unsigned __int64)(v103 + 7);
                v357 = v359;
                v358 = 0x100000000LL;
                sub_1626700(v12, (__int64)&v357);
                if ( *(_BYTE *)(v340 + 16) == 13 )
                {
                  v337 = sub_1593BB0(v105, (__int64)&v357, v106, v107);
                  if ( v337 )
                  {
                    v108 = *(_DWORD *)(v340 + 32);
                    if ( v108 <= 0x40 )
                      v337 = *(_QWORD *)(v340 + 24) == 1;
                    else
                      v337 = v108 - 1 == (unsigned int)sub_16A57B0(v340 + 24);
                  }
                  v109 = *(_QWORD *)(v12 - 24);
                  if ( *(_BYTE *)(v109 + 16) == 13 )
                  {
                    v110 = *(_QWORD *)(v109 + 24);
                    if ( *(_DWORD *)(v109 + 32) > 0x40u )
                      v110 = **(_QWORD **)(v109 + 24);
                    v111 = *(_QWORD **)(v340 + 24);
                    if ( *(_DWORD *)(v340 + 32) > 0x40u )
                      v111 = (_QWORD *)*v111;
                    v326 = (char *)v111 - v110;
                    v112 = (unsigned __int64)v357;
                    v321 = &v357[8 * (unsigned int)v358];
                    while ( v321 != (char *)v112 )
                    {
                      v113 = *(unsigned int *)(*(_QWORD *)v112 + 8LL);
                      v114 = *(_QWORD *)(*(_QWORD *)v112 - 8 * v113);
                      v115 = *(_QWORD **)(*(_QWORD *)v112 + 8 * (1 - v113));
                      v360 = (const char **)v362;
                      v305 = v114;
                      v362[2] = v326;
                      v362[3] = 30;
                      v362[4] = 16;
                      v362[5] = v110;
                      v362[6] = 34;
                      v361 = 0xC00000007LL;
                      v362[0] = 6;
                      v362[1] = 16;
                      v312 = sub_15C46E0(v115, (__int64)&v360, 1);
                      v116 = (__int64 *)sub_16498A0((__int64)v103);
                      v117 = sub_15C5570(v116, v305, v312, 0, 1);
                      sub_1626A90((__int64)v103, v117);
                      if ( v360 != v362 )
                        _libc_free((unsigned __int64)v360);
                      v112 += 8LL;
                    }
LABEL_160:
                    v322 = v32;
                    while ( 2 )
                    {
                      v125 = *(_QWORD *)(v12 + 8);
                      if ( !v125 )
                      {
                        sub_164B7C0((__int64)v103, v12);
                        sub_15E55B0(v12);
                        if ( v357 != v359 )
                          _libc_free((unsigned __int64)v357);
                        return v322;
                      }
                      v126 = sub_1648700(v125);
                      v127 = (__int64)v126;
                      if ( *((_BYTE *)v126 + 16) != 55 )
                      {
                        v128 = sub_1649960((__int64)v126);
                        LOWORD(v362[0]) = 773;
                        v353 = v128;
                        v354 = v129;
                        v360 = &v353;
                        v361 = (__int64)".b";
                        v316 = *(_BYTE *)(v127 + 56);
                        v130 = (*(_WORD *)(v127 + 18) >> 7) & 7;
                        v131 = sub_1648A60(64, 1u);
                        if ( v131 )
                          sub_15F8F80(
                            (__int64)v131,
                            *(_QWORD *)(*v103 + 24LL),
                            (__int64)v103,
                            (__int64)&v360,
                            0,
                            0,
                            v130,
                            v316,
                            v127);
                        if ( v337 )
                        {
                          v132 = *(_QWORD *)v127;
                          LOWORD(v362[0]) = 257;
                          v317 = v132;
                          v133 = sub_1648A60(56, 1u);
                          v134 = (__int64)v133;
                          if ( v133 )
                            sub_15FC690((__int64)v133, (__int64)v131, v317, (__int64)&v360, v127);
                        }
                        else
                        {
                          LOWORD(v362[0]) = 257;
                          v208 = sub_1648A60(56, 3u);
                          v134 = (__int64)v208;
                          if ( v208 )
                          {
                            v319 = v208 - 9;
                            sub_15F1EA0((__int64)v208, *(_QWORD *)v340, 55, (__int64)(v208 - 9), 3, v127);
                            if ( *(_QWORD *)(v134 - 72) )
                            {
                              v209 = *(_QWORD *)(v134 - 64);
                              v210 = *(_QWORD *)(v134 - 56) & 0xFFFFFFFFFFFFFFFCLL;
                              *(_QWORD *)v210 = v209;
                              if ( v209 )
                                *(_QWORD *)(v209 + 16) = v210 | *(_QWORD *)(v209 + 16) & 3LL;
                            }
                            *(_QWORD *)(v134 - 72) = v131;
                            if ( v131 )
                            {
                              v211 = v131[1];
                              *(_QWORD *)(v134 - 64) = v211;
                              if ( v211 )
                                *(_QWORD *)(v211 + 16) = (v134 - 64) | *(_QWORD *)(v211 + 16) & 3LL;
                              *(_QWORD *)(v134 - 56) = (unsigned __int64)(v131 + 1) | *(_QWORD *)(v134 - 56) & 3LL;
                              v131[1] = v319;
                            }
                            if ( *(_QWORD *)(v134 - 48) )
                            {
                              v212 = *(_QWORD *)(v134 - 40);
                              v213 = *(_QWORD *)(v134 - 32) & 0xFFFFFFFFFFFFFFFCLL;
                              *(_QWORD *)v213 = v212;
                              if ( v212 )
                                *(_QWORD *)(v212 + 16) = v213 | *(_QWORD *)(v212 + 16) & 3LL;
                            }
                            *(_QWORD *)(v134 - 48) = v340;
                            v214 = *(_QWORD *)(v340 + 8);
                            *(_QWORD *)(v134 - 40) = v214;
                            if ( v214 )
                              *(_QWORD *)(v214 + 16) = (v134 - 40) | *(_QWORD *)(v214 + 16) & 3LL;
                            *(_QWORD *)(v134 - 32) = *(_QWORD *)(v134 - 32) & 3LL | (v340 + 8);
                            *(_QWORD *)(v340 + 8) = v134 - 48;
                            if ( *(_QWORD *)(v134 - 24) )
                            {
                              v215 = *(_QWORD *)(v134 - 16);
                              v216 = *(_QWORD *)(v134 - 8) & 0xFFFFFFFFFFFFFFFCLL;
                              *(_QWORD *)v216 = v215;
                              if ( v215 )
                                *(_QWORD *)(v215 + 16) = v216 | *(_QWORD *)(v215 + 16) & 3LL;
                            }
                            *(_QWORD *)(v134 - 24) = v330;
                            if ( v330 )
                            {
                              v217 = *(_QWORD *)(v330 + 8);
                              *(_QWORD *)(v134 - 16) = v217;
                              if ( v217 )
                                *(_QWORD *)(v217 + 16) = (v134 - 16) | *(_QWORD *)(v217 + 16) & 3LL;
                              *(_QWORD *)(v134 - 8) = (v330 + 8) | *(_QWORD *)(v134 - 8) & 3LL;
                              *(_QWORD *)(v330 + 8) = v134 - 24;
                            }
                            sub_164B780(v134, (__int64 *)&v360);
                          }
                        }
                        sub_164B7C0(v134, v127);
                        sub_164D160(
                          v127,
                          v134,
                          a5,
                          *(double *)a6.m128i_i64,
                          *(double *)a7.m128i_i64,
                          a8,
                          v135,
                          v136,
                          a11,
                          a12);
LABEL_169:
                        sub_15F20C0((_QWORD *)v127);
                        continue;
                      }
                      break;
                    }
                    v118 = *(v126 - 6);
                    if ( v118 && v340 == v118 )
                    {
                      v138 = v322;
                    }
                    else
                    {
                      if ( v330 != v118 )
                      {
                        if ( *(_BYTE *)(v118 + 16) == 54 )
                        {
                          v313 = *(v126 - 6);
                          v353 = sub_1649960(v313);
                          v354 = v119;
                          v360 = &v353;
                          LOWORD(v362[0]) = 773;
                          v361 = (__int64)".b";
                          v120 = *(unsigned __int16 *)(v313 + 18);
                          v308 = v313;
                          v314 = *(_BYTE *)(v313 + 56);
                          v121 = (v120 >> 7) & 7;
                          v122 = (__int64)sub_1648A60(64, 1u);
                          if ( v122 )
                            sub_15F8F80(
                              v122,
                              *(_QWORD *)(*v103 + 24LL),
                              (__int64)v103,
                              (__int64)&v360,
                              0,
                              0,
                              v121,
                              v314,
                              v308);
                        }
                        else
                        {
                          if ( (*(_BYTE *)(v118 + 23) & 0x40) != 0 )
                            v137 = *(__int64 **)(v118 - 8);
                          else
                            v137 = (__int64 *)(v118 - 24LL * (*(_DWORD *)(v118 + 20) & 0xFFFFFFF));
                          v122 = *v137;
                        }
                        goto LABEL_167;
                      }
                      v138 = 0;
                    }
                    v139 = (_QWORD *)sub_16498A0(v12);
                    v140 = sub_1643320(v139);
                    v122 = sub_159C470(v140, v138, 0);
LABEL_167:
                    v315 = *(_BYTE *)(v127 + 56);
                    v123 = (*(_WORD *)(v127 + 18) >> 7) & 7;
                    v124 = sub_1648A60(64, 2u);
                    if ( v124 )
                      sub_15F9480((__int64)v124, v122, (__int64)v103, 0, 0, v123, v315, v127);
                    goto LABEL_169;
                  }
                }
                else
                {
                  v337 = 0;
                }
                v205 = (__int64 *)v357;
                v206 = (__int64 *)&v357[8 * (unsigned int)v358];
                while ( v206 != v205 )
                {
                  v207 = *v205++;
                  sub_1626A90((__int64)v103, v207);
                }
                goto LABEL_160;
              }
              v333 = *(_QWORD *)(v57 + 8);
              v60 = sub_1648700(v57);
              v61 = *((_BYTE *)v60 + 16);
              if ( v61 > 0x17u )
              {
                if ( v61 == 54 )
                {
                  v56 = v42;
                  v58 |= sub_185D7C0((__int64)v60, v42);
                  if ( v60[1] )
                    goto LABEL_80;
                  sub_15F20C0(v60);
                  v58 = v328;
                }
                else if ( v61 != 55 )
                {
                  v59 = 0;
                }
              }
              else
              {
LABEL_80:
                v59 = 0;
              }
              v57 = v333;
              continue;
            }
          }
          sub_15E5440(v12, v349);
          sub_185FD30(
            v12,
            *(_QWORD *)(v12 - 24),
            v335,
            (__int64)a2,
            a5,
            *(double *)a6.m128i_i64,
            *(double *)a7.m128i_i64,
            a8,
            v62,
            v63,
            a11,
            a12);
          if ( *(_QWORD *)(v12 + 8) )
            return 1;
        }
        else
        {
          if ( (*(_BYTE *)(v12 + 32) & 0xF) != 8 && sub_185B2A0(v12, v17, v23, v24, v25, v26) )
            v47 = sub_185B9F0(v12, a2);
          else
            v47 = sub_185FD30(
                    v12,
                    *(_QWORD *)(v12 - 24),
                    v335,
                    (__int64)a2,
                    a5,
                    *(double *)a6.m128i_i64,
                    *(double *)a7.m128i_i64,
                    a8,
                    v27,
                    v28,
                    a11,
                    a12);
          if ( *(_QWORD *)(v12 + 8) )
          {
            if ( !v47 )
              return v18;
            return 1;
          }
        }
LABEL_58:
        v18 = 1;
        sub_15E55B0(v12);
        return v18;
      }
      v344 = *(__int64 **)v303;
      v143 = *v344;
      v354 = (__int64 *)&v344;
      v23 = (unsigned __int64)&v345;
      v345 = v143;
      v356 = &v345;
      v144 = (unsigned int)v361;
      v309 = (__int64 *)&v360[v144];
      v306 = (v144 * 8) >> 5;
      v353 = v300;
      v355 = v301;
      while ( v306 )
      {
        v150 = **(_QWORD **)(*v142 - 48);
        if ( sub_15CCEE0((__int64)v353, *v142, *v354) )
        {
          v151 = v355;
          v152 = 1;
          v17 = *v356;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v17 + 8) )
            {
              case 1:
                v153 = 16;
                goto LABEL_196;
              case 2:
                v153 = 32;
                goto LABEL_196;
              case 3:
              case 9:
                v153 = 64;
                goto LABEL_196;
              case 4:
                v153 = 80;
                goto LABEL_196;
              case 5:
              case 6:
                v153 = 128;
                goto LABEL_196;
              case 7:
                v17 = 0;
                v204 = sub_15A9520(v355, 0);
                v151 = v355;
                v153 = (unsigned int)(8 * v204);
                goto LABEL_196;
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v202 = *(_QWORD *)(v17 + 32);
                v17 = *(_QWORD *)(v17 + 24);
                v152 *= v202;
                continue;
              case 0xB:
                v153 = *(_DWORD *)(v17 + 8) >> 8;
                goto LABEL_196;
              case 0xD:
                v222 = (_QWORD *)sub_15A9930(v355, v17);
                v151 = v355;
                v153 = 8LL * *v222;
                goto LABEL_196;
              case 0xE:
                v285 = v355;
                v219 = 1;
                v296 = *(_QWORD *)(v17 + 32);
                v17 = *(_QWORD *)(v17 + 24);
                v220 = v355;
                v221 = (unsigned int)sub_15A9FE0(v355, v17);
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v17 + 8) )
                  {
                    case 1:
                      v223 = 16;
                      goto LABEL_363;
                    case 2:
                      v223 = 32;
                      goto LABEL_363;
                    case 3:
                    case 9:
                      v223 = 64;
                      goto LABEL_363;
                    case 4:
                      v223 = 80;
                      goto LABEL_363;
                    case 5:
                    case 6:
                      v223 = 128;
                      goto LABEL_363;
                    case 7:
                      v17 = 0;
                      v223 = 8 * (unsigned int)sub_15A9520(v285, 0);
                      goto LABEL_363;
                    case 8:
                    case 0xA:
                    case 0xC:
                    case 0x10:
                      v225 = *(_QWORD *)(v17 + 32);
                      v17 = *(_QWORD *)(v17 + 24);
                      v219 *= v225;
                      continue;
                    case 0xB:
                      v223 = *(_DWORD *)(v17 + 8) >> 8;
                      goto LABEL_363;
                    case 0xD:
                      v223 = 8LL * *(_QWORD *)sub_15A9930(v285, v17);
                      goto LABEL_363;
                    case 0xE:
                      v269 = v285;
                      v286 = *(_QWORD *)(v17 + 32);
                      v17 = *(_QWORD *)(v17 + 24);
                      v275 = (unsigned int)sub_15A9FE0(v220, v17);
                      v224 = sub_127FA20(v269, v17);
                      v24 = v286 * v275;
                      v223 = 8 * v286 * v275 * ((v275 + ((unsigned __int64)(v224 + 7) >> 3) - 1) / v275);
                      goto LABEL_363;
                    case 0xF:
                      v17 = *(_DWORD *)(v17 + 8) >> 8;
                      v223 = 8 * (unsigned int)sub_15A9520(v285, v17);
LABEL_363:
                      v151 = v355;
                      v23 = (v221 + ((unsigned __int64)(v219 * v223 + 7) >> 3) - 1) % v221;
                      v153 = 8 * v296 * v221 * ((v221 + ((unsigned __int64)(v219 * v223 + 7) >> 3) - 1) / v221);
                      break;
                  }
                  goto LABEL_196;
                }
              case 0xF:
                v17 = *(_DWORD *)(v17 + 8) >> 8;
                v203 = sub_15A9520(v355, v17);
                v151 = v355;
                v153 = (unsigned int)(8 * v203);
LABEL_196:
                v154 = 1;
                v291 = (unsigned __int64)(v152 * v153 + 7) >> 3;
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v150 + 8) )
                  {
                    case 1:
                      v227 = 16;
                      goto LABEL_376;
                    case 2:
                      v227 = 32;
                      goto LABEL_376;
                    case 3:
                    case 9:
                      v227 = 64;
                      goto LABEL_376;
                    case 4:
                      v227 = 80;
                      goto LABEL_376;
                    case 5:
                    case 6:
                      v227 = 128;
                      goto LABEL_376;
                    case 7:
                      v17 = 0;
                      v227 = 8 * (unsigned int)sub_15A9520(v151, 0);
                      goto LABEL_376;
                    case 0xB:
                      v227 = *(_DWORD *)(v150 + 8) >> 8;
                      goto LABEL_376;
                    case 0xD:
                      v17 = v150;
                      v227 = 8LL * *(_QWORD *)sub_15A9930(v151, v150);
                      goto LABEL_376;
                    case 0xE:
                      v17 = *(_QWORD *)(v150 + 24);
                      v228 = *(_QWORD *)(v150 + 32);
                      v229 = 1;
                      v287 = v228;
                      v230 = (unsigned int)sub_15A9FE0(v151, v17);
                      while ( 2 )
                      {
                        switch ( *(_BYTE *)(v17 + 8) )
                        {
                          case 1:
                            v232 = 16;
                            goto LABEL_383;
                          case 2:
                            v232 = 32;
                            goto LABEL_383;
                          case 3:
                          case 9:
                            v232 = 64;
                            goto LABEL_383;
                          case 4:
                            v232 = 80;
                            goto LABEL_383;
                          case 5:
                          case 6:
                            v232 = 128;
                            goto LABEL_383;
                          case 7:
                            v17 = 0;
                            v232 = 8 * (unsigned int)sub_15A9520(v151, 0);
                            goto LABEL_383;
                          case 0xB:
                            v232 = *(_DWORD *)(v17 + 8) >> 8;
                            goto LABEL_383;
                          case 0xD:
                            v232 = 8LL * *(_QWORD *)sub_15A9930(v151, v17);
                            goto LABEL_383;
                          case 0xE:
                            v233 = *(_QWORD *)(v17 + 32);
                            v17 = *(_QWORD *)(v17 + 24);
                            v276 = (unsigned int)sub_15A9FE0(v151, v17);
                            v234 = sub_127FA20(v151, v17);
                            v24 = v233 * v276;
                            v232 = 8 * v233 * v276 * ((v276 + ((unsigned __int64)(v234 + 7) >> 3) - 1) / v276);
                            goto LABEL_383;
                          case 0xF:
                            v17 = *(_DWORD *)(v17 + 8) >> 8;
                            v232 = 8 * (unsigned int)sub_15A9520(v151, v17);
LABEL_383:
                            v23 = (v230 + ((unsigned __int64)(v229 * v232 + 7) >> 3) - 1) % v230;
                            v227 = 8 * v287 * v230 * ((v230 + ((unsigned __int64)(v229 * v232 + 7) >> 3) - 1) / v230);
                            goto LABEL_376;
                          case 0x10:
                            v231 = *(_QWORD *)(v17 + 32);
                            v17 = *(_QWORD *)(v17 + 24);
                            v229 *= v231;
                            continue;
                          default:
                            goto LABEL_468;
                        }
                      }
                    case 0xF:
                      v17 = *(_DWORD *)(v150 + 8) >> 8;
                      v227 = 8 * (unsigned int)sub_15A9520(v151, v17);
LABEL_376:
                      if ( v291 > (unsigned __int64)(v154 * v227 + 7) >> 3 )
                        goto LABEL_187;
                      goto LABEL_239;
                    case 0x10:
                      v226 = *(_QWORD *)(v150 + 32);
                      v150 = *(_QWORD *)(v150 + 24);
                      v154 *= v226;
                      continue;
                    default:
                      goto LABEL_468;
                  }
                }
            }
          }
        }
LABEL_187:
        v145 = v142[1];
        v146 = **(_QWORD **)(v145 - 48);
        if ( sub_15CCEE0((__int64)v353, v145, *v354) )
        {
          v159 = v355;
          v160 = 1;
          v17 = *v356;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v17 + 8) )
            {
              case 1:
                v168 = 16;
                goto LABEL_211;
              case 2:
                v168 = 32;
                goto LABEL_211;
              case 3:
              case 9:
                v168 = 64;
                goto LABEL_211;
              case 4:
                v168 = 80;
                goto LABEL_211;
              case 5:
              case 6:
                v168 = 128;
                goto LABEL_211;
              case 7:
                v17 = 0;
                v199 = sub_15A9520(v355, 0);
                v159 = v355;
                v168 = (unsigned int)(8 * v199);
                goto LABEL_211;
              case 0xB:
                v168 = *(_DWORD *)(v17 + 8) >> 8;
                goto LABEL_211;
              case 0xD:
                v198 = (_QWORD *)sub_15A9930(v355, v17);
                v159 = v355;
                v168 = 8LL * *v198;
                goto LABEL_211;
              case 0xE:
                v280 = v355;
                v170 = 1;
                v295 = *(_QWORD *)(v17 + 32);
                v17 = *(_QWORD *)(v17 + 24);
                v171 = v355;
                v172 = (unsigned int)sub_15A9FE0(v355, v17);
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v17 + 8) )
                  {
                    case 1:
                      v176 = 16;
                      goto LABEL_220;
                    case 2:
                      v176 = 32;
                      goto LABEL_220;
                    case 3:
                    case 9:
                      v176 = 64;
                      goto LABEL_220;
                    case 4:
                      v176 = 80;
                      goto LABEL_220;
                    case 5:
                    case 6:
                      v176 = 128;
                      goto LABEL_220;
                    case 7:
                      v17 = 0;
                      v176 = 8 * (unsigned int)sub_15A9520(v280, 0);
                      goto LABEL_220;
                    case 0xB:
                      v176 = *(_DWORD *)(v17 + 8) >> 8;
                      goto LABEL_220;
                    case 0xD:
                      v176 = 8LL * *(_QWORD *)sub_15A9930(v280, v17);
                      goto LABEL_220;
                    case 0xE:
                      v268 = v280;
                      v282 = *(_QWORD *)(v17 + 32);
                      v17 = *(_QWORD *)(v17 + 24);
                      v272 = (unsigned int)sub_15A9FE0(v171, v17);
                      v177 = sub_127FA20(v268, v17);
                      v24 = v282 * v272;
                      v176 = 8 * v282 * v272 * ((v272 + ((unsigned __int64)(v177 + 7) >> 3) - 1) / v272);
                      goto LABEL_220;
                    case 0xF:
                      v17 = *(_DWORD *)(v17 + 8) >> 8;
                      v176 = 8 * (unsigned int)sub_15A9520(v280, v17);
LABEL_220:
                      v159 = v355;
                      v23 = (v172 + ((unsigned __int64)(v170 * v176 + 7) >> 3) - 1) % v172;
                      v168 = 8 * v295 * v172 * ((v172 + ((unsigned __int64)(v170 * v176 + 7) >> 3) - 1) / v172);
                      goto LABEL_211;
                    case 0x10:
                      v175 = *(_QWORD *)(v17 + 32);
                      v17 = *(_QWORD *)(v17 + 24);
                      v170 *= v175;
                      continue;
                    default:
                      goto LABEL_468;
                  }
                }
              case 0xF:
                v17 = *(_DWORD *)(v17 + 8) >> 8;
                v167 = sub_15A9520(v355, v17);
                v159 = v355;
                v168 = (unsigned int)(8 * v167);
LABEL_211:
                v169 = 1;
                v294 = (unsigned __int64)(v160 * v168 + 7) >> 3;
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v146 + 8) )
                  {
                    case 1:
                      v180 = 16;
                      goto LABEL_237;
                    case 2:
                      v180 = 32;
                      goto LABEL_237;
                    case 3:
                    case 9:
                      v180 = 64;
                      goto LABEL_237;
                    case 4:
                      v180 = 80;
                      goto LABEL_237;
                    case 5:
                    case 6:
                      v180 = 128;
                      goto LABEL_237;
                    case 7:
                      v17 = 0;
                      v180 = 8 * (unsigned int)sub_15A9520(v159, 0);
                      goto LABEL_237;
                    case 0xB:
                      v180 = *(_DWORD *)(v146 + 8) >> 8;
                      goto LABEL_237;
                    case 0xD:
                      v17 = v146;
                      v180 = 8LL * *(_QWORD *)sub_15A9930(v159, v146);
                      goto LABEL_237;
                    case 0xE:
                      v17 = *(_QWORD *)(v146 + 24);
                      v181 = *(_QWORD *)(v146 + 32);
                      v182 = 1;
                      v283 = v181;
                      v183 = (unsigned int)sub_15A9FE0(v159, v17);
                      while ( 2 )
                      {
                        switch ( *(_BYTE *)(v17 + 8) )
                        {
                          case 1:
                            v185 = 16;
                            goto LABEL_253;
                          case 2:
                            v185 = 32;
                            goto LABEL_253;
                          case 3:
                          case 9:
                            v185 = 64;
                            goto LABEL_253;
                          case 4:
                            v185 = 80;
                            goto LABEL_253;
                          case 5:
                          case 6:
                            v185 = 128;
                            goto LABEL_253;
                          case 7:
                            v17 = 0;
                            v185 = 8 * (unsigned int)sub_15A9520(v159, 0);
                            goto LABEL_253;
                          case 0xB:
                            v185 = *(_DWORD *)(v17 + 8) >> 8;
                            goto LABEL_253;
                          case 0xD:
                            v185 = 8LL * *(_QWORD *)sub_15A9930(v159, v17);
                            goto LABEL_253;
                          case 0xE:
                            v186 = *(_QWORD *)(v17 + 32);
                            v17 = *(_QWORD *)(v17 + 24);
                            v273 = (unsigned int)sub_15A9FE0(v159, v17);
                            v187 = sub_127FA20(v159, v17);
                            v24 = v186 * v273;
                            v185 = 8 * v186 * v273 * ((v273 + ((unsigned __int64)(v187 + 7) >> 3) - 1) / v273);
                            goto LABEL_253;
                          case 0xF:
                            v17 = *(_DWORD *)(v17 + 8) >> 8;
                            v185 = 8 * (unsigned int)sub_15A9520(v159, v17);
LABEL_253:
                            v23 = (v183 + ((unsigned __int64)(v182 * v185 + 7) >> 3) - 1) % v183;
                            v180 = 8 * v283 * v183 * ((v183 + ((unsigned __int64)(v182 * v185 + 7) >> 3) - 1) / v183);
                            goto LABEL_237;
                          case 0x10:
                            v184 = *(_QWORD *)(v17 + 32);
                            v17 = *(_QWORD *)(v17 + 24);
                            v182 *= v184;
                            continue;
                          default:
                            goto LABEL_468;
                        }
                      }
                    case 0xF:
                      v17 = *(_DWORD *)(v146 + 8) >> 8;
                      v180 = 8 * (unsigned int)sub_15A9520(v159, v17);
LABEL_237:
                      if ( v294 > (unsigned __int64)(v169 * v180 + 7) >> 3 )
                        goto LABEL_188;
                      ++v142;
                      break;
                    case 0x10:
                      v179 = *(_QWORD *)(v146 + 32);
                      v146 = *(_QWORD *)(v146 + 24);
                      v169 *= v179;
                      continue;
                    default:
                      goto LABEL_468;
                  }
                  goto LABEL_239;
                }
              case 0x10:
                v188 = *(_QWORD *)(v17 + 32);
                v17 = *(_QWORD *)(v17 + 24);
                v160 *= v188;
                continue;
              default:
                goto LABEL_468;
            }
          }
        }
LABEL_188:
        v147 = v142[2];
        v148 = **(_QWORD **)(v147 - 48);
        if ( sub_15CCEE0((__int64)v353, v147, *v354) )
        {
          v157 = v355;
          v158 = 1;
          v17 = *v356;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v17 + 8) )
            {
              case 1:
                v248 = 16;
                goto LABEL_418;
              case 2:
                v248 = 32;
                goto LABEL_418;
              case 3:
              case 9:
                v248 = 64;
                goto LABEL_418;
              case 4:
                v248 = 80;
                goto LABEL_418;
              case 5:
              case 6:
                v248 = 128;
                goto LABEL_418;
              case 7:
                v17 = 0;
                v266 = sub_15A9520(v355, 0);
                v157 = v355;
                v248 = (unsigned int)(8 * v266);
                goto LABEL_418;
              case 0xB:
                v248 = *(_DWORD *)(v17 + 8) >> 8;
                goto LABEL_418;
              case 0xD:
                v265 = (_QWORD *)sub_15A9930(v355, v17);
                v157 = v355;
                v248 = 8LL * *v265;
                goto LABEL_418;
              case 0xE:
                v288 = v355;
                v250 = 1;
                v298 = *(_QWORD *)(v17 + 32);
                v17 = *(_QWORD *)(v17 + 24);
                v251 = v355;
                v252 = (unsigned int)sub_15A9FE0(v355, v17);
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v17 + 8) )
                  {
                    case 1:
                      v255 = 16;
                      goto LABEL_425;
                    case 2:
                      v255 = 32;
                      goto LABEL_425;
                    case 3:
                    case 9:
                      v255 = 64;
                      goto LABEL_425;
                    case 4:
                      v255 = 80;
                      goto LABEL_425;
                    case 5:
                    case 6:
                      v255 = 128;
                      goto LABEL_425;
                    case 7:
                      v17 = 0;
                      v255 = 8 * (unsigned int)sub_15A9520(v288, 0);
                      goto LABEL_425;
                    case 0xB:
                      v255 = *(_DWORD *)(v17 + 8) >> 8;
                      goto LABEL_425;
                    case 0xD:
                      v255 = 8LL * *(_QWORD *)sub_15A9930(v288, v17);
                      goto LABEL_425;
                    case 0xE:
                      v270 = v288;
                      v289 = *(_QWORD *)(v17 + 32);
                      v17 = *(_QWORD *)(v17 + 24);
                      v277 = (unsigned int)sub_15A9FE0(v251, v17);
                      v256 = sub_127FA20(v270, v17);
                      v24 = v289 * v277;
                      v255 = 8 * v289 * v277 * ((v277 + ((unsigned __int64)(v256 + 7) >> 3) - 1) / v277);
                      goto LABEL_425;
                    case 0xF:
                      v17 = *(_DWORD *)(v17 + 8) >> 8;
                      v255 = 8 * (unsigned int)sub_15A9520(v288, v17);
LABEL_425:
                      v157 = v355;
                      v23 = (v252 + ((unsigned __int64)(v250 * v255 + 7) >> 3) - 1) % v252;
                      v248 = 8 * v298 * v252 * ((v252 + ((unsigned __int64)(v250 * v255 + 7) >> 3) - 1) / v252);
                      goto LABEL_418;
                    case 0x10:
                      v254 = *(_QWORD *)(v17 + 32);
                      v17 = *(_QWORD *)(v17 + 24);
                      v250 *= v254;
                      continue;
                    default:
                      goto LABEL_468;
                  }
                }
              case 0xF:
                v17 = *(_DWORD *)(v17 + 8) >> 8;
                v247 = sub_15A9520(v355, v17);
                v157 = v355;
                v248 = (unsigned int)(8 * v247);
LABEL_418:
                v249 = 1;
                v297 = (unsigned __int64)(v158 * v248 + 7) >> 3;
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v148 + 8) )
                  {
                    case 1:
                      v257 = 16;
                      goto LABEL_436;
                    case 2:
                      v257 = 32;
                      goto LABEL_436;
                    case 3:
                    case 9:
                      v257 = 64;
                      goto LABEL_436;
                    case 4:
                      v257 = 80;
                      goto LABEL_436;
                    case 5:
                    case 6:
                      v257 = 128;
                      goto LABEL_436;
                    case 7:
                      v17 = 0;
                      v257 = 8 * (unsigned int)sub_15A9520(v157, 0);
                      goto LABEL_436;
                    case 0xB:
                      v257 = *(_DWORD *)(v148 + 8) >> 8;
                      goto LABEL_436;
                    case 0xD:
                      v17 = v148;
                      v257 = 8LL * *(_QWORD *)sub_15A9930(v157, v148);
                      goto LABEL_436;
                    case 0xE:
                      v17 = *(_QWORD *)(v148 + 24);
                      v258 = *(_QWORD *)(v148 + 32);
                      v259 = 1;
                      v290 = v258;
                      v260 = (unsigned int)sub_15A9FE0(v157, v17);
                      while ( 2 )
                      {
                        switch ( *(_BYTE *)(v17 + 8) )
                        {
                          case 1:
                            v262 = 16;
                            goto LABEL_444;
                          case 2:
                            v262 = 32;
                            goto LABEL_444;
                          case 3:
                          case 9:
                            v262 = 64;
                            goto LABEL_444;
                          case 4:
                            v262 = 80;
                            goto LABEL_444;
                          case 5:
                          case 6:
                            v262 = 128;
                            goto LABEL_444;
                          case 7:
                            v17 = 0;
                            v262 = 8 * (unsigned int)sub_15A9520(v157, 0);
                            goto LABEL_444;
                          case 0xB:
                            v262 = *(_DWORD *)(v17 + 8) >> 8;
                            goto LABEL_444;
                          case 0xD:
                            v262 = 8LL * *(_QWORD *)sub_15A9930(v157, v17);
                            goto LABEL_444;
                          case 0xE:
                            v263 = *(_QWORD *)(v17 + 32);
                            v17 = *(_QWORD *)(v17 + 24);
                            v278 = (unsigned int)sub_15A9FE0(v157, v17);
                            v264 = sub_127FA20(v157, v17);
                            v24 = v263 * v278;
                            v262 = 8 * v263 * v278 * ((v278 + ((unsigned __int64)(v264 + 7) >> 3) - 1) / v278);
                            goto LABEL_444;
                          case 0xF:
                            v17 = *(_DWORD *)(v17 + 8) >> 8;
                            v262 = 8 * (unsigned int)sub_15A9520(v157, v17);
LABEL_444:
                            v23 = (v260 + ((unsigned __int64)(v259 * v262 + 7) >> 3) - 1) % v260;
                            v257 = 8 * v290 * v260 * ((v260 + ((unsigned __int64)(v259 * v262 + 7) >> 3) - 1) / v260);
                            goto LABEL_436;
                          case 0x10:
                            v261 = *(_QWORD *)(v17 + 32);
                            v17 = *(_QWORD *)(v17 + 24);
                            v259 *= v261;
                            continue;
                          default:
                            goto LABEL_468;
                        }
                      }
                    case 0xF:
                      v17 = *(_DWORD *)(v148 + 8) >> 8;
                      v257 = 8 * (unsigned int)sub_15A9520(v157, v17);
LABEL_436:
                      if ( v297 > (unsigned __int64)(v249 * v257 + 7) >> 3 )
                        goto LABEL_189;
                      v142 += 2;
                      break;
                    case 0x10:
                      v253 = *(_QWORD *)(v148 + 32);
                      v148 = *(_QWORD *)(v148 + 24);
                      v249 *= v253;
                      continue;
                    default:
                      goto LABEL_468;
                  }
                  goto LABEL_239;
                }
              case 0x10:
                v246 = *(_QWORD *)(v17 + 32);
                v17 = *(_QWORD *)(v17 + 24);
                v158 *= v246;
                continue;
              default:
                goto LABEL_468;
            }
          }
        }
LABEL_189:
        v17 = v142[3];
        v149 = **(_QWORD **)(v17 - 48);
        if ( sub_15CCEE0((__int64)v353, v17, *v354) )
        {
          v155 = v355;
          v156 = 1;
          v17 = *v356;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v17 + 8) )
            {
              case 1:
                v162 = 16;
                goto LABEL_206;
              case 2:
                v162 = 32;
                goto LABEL_206;
              case 3:
              case 9:
                v162 = 64;
                goto LABEL_206;
              case 4:
                v162 = 80;
                goto LABEL_206;
              case 5:
              case 6:
                v162 = 128;
                goto LABEL_206;
              case 7:
                v17 = 0;
                v201 = sub_15A9520(v355, 0);
                v155 = v355;
                v162 = (unsigned int)(8 * v201);
                goto LABEL_206;
              case 0xB:
                v162 = *(_DWORD *)(v17 + 8) >> 8;
                goto LABEL_206;
              case 0xD:
                v200 = (_QWORD *)sub_15A9930(v355, v17);
                v155 = v355;
                v162 = 8LL * *v200;
                goto LABEL_206;
              case 0xE:
                v279 = v355;
                v164 = 1;
                v293 = *(_QWORD *)(v17 + 32);
                v17 = *(_QWORD *)(v17 + 24);
                v165 = v355;
                v166 = (unsigned int)sub_15A9FE0(v355, v17);
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v17 + 8) )
                  {
                    case 1:
                      v173 = 16;
                      goto LABEL_216;
                    case 2:
                      v173 = 32;
                      goto LABEL_216;
                    case 3:
                    case 9:
                      v173 = 64;
                      goto LABEL_216;
                    case 4:
                      v173 = 80;
                      goto LABEL_216;
                    case 5:
                    case 6:
                      v173 = 128;
                      goto LABEL_216;
                    case 7:
                      v17 = 0;
                      v173 = 8 * (unsigned int)sub_15A9520(v279, 0);
                      goto LABEL_216;
                    case 0xB:
                      v173 = *(_DWORD *)(v17 + 8) >> 8;
                      goto LABEL_216;
                    case 0xD:
                      v173 = 8LL * *(_QWORD *)sub_15A9930(v279, v17);
                      goto LABEL_216;
                    case 0xE:
                      v267 = v279;
                      v281 = *(_QWORD *)(v17 + 32);
                      v17 = *(_QWORD *)(v17 + 24);
                      v271 = (unsigned int)sub_15A9FE0(v165, v17);
                      v174 = sub_127FA20(v267, v17);
                      v24 = v281 * v271;
                      v173 = 8 * v281 * v271 * ((v271 + ((unsigned __int64)(v174 + 7) >> 3) - 1) / v271);
                      goto LABEL_216;
                    case 0xF:
                      v17 = *(_DWORD *)(v17 + 8) >> 8;
                      v173 = 8 * (unsigned int)sub_15A9520(v279, v17);
LABEL_216:
                      v155 = v355;
                      v23 = (v166 + ((unsigned __int64)(v164 * v173 + 7) >> 3) - 1) % v166;
                      v162 = 8 * v293 * v166 * ((v166 + ((unsigned __int64)(v164 * v173 + 7) >> 3) - 1) / v166);
                      goto LABEL_206;
                    case 0x10:
                      v178 = *(_QWORD *)(v17 + 32);
                      v17 = *(_QWORD *)(v17 + 24);
                      v164 *= v178;
                      continue;
                    default:
                      goto LABEL_468;
                  }
                }
              case 0xF:
                v17 = *(_DWORD *)(v17 + 8) >> 8;
                v161 = sub_15A9520(v355, v17);
                v155 = v355;
                v162 = (unsigned int)(8 * v161);
LABEL_206:
                v163 = 1;
                v292 = (unsigned __int64)(v156 * v162 + 7) >> 3;
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v149 + 8) )
                  {
                    case 1:
                      v190 = 16;
                      goto LABEL_270;
                    case 2:
                      v190 = 32;
                      goto LABEL_270;
                    case 3:
                    case 9:
                      v190 = 64;
                      goto LABEL_270;
                    case 4:
                      v190 = 80;
                      goto LABEL_270;
                    case 5:
                    case 6:
                      v190 = 128;
                      goto LABEL_270;
                    case 7:
                      v17 = 0;
                      v190 = 8 * (unsigned int)sub_15A9520(v155, 0);
                      goto LABEL_270;
                    case 0xB:
                      v190 = *(_DWORD *)(v149 + 8) >> 8;
                      goto LABEL_270;
                    case 0xD:
                      v17 = v149;
                      v190 = 8LL * *(_QWORD *)sub_15A9930(v155, v149);
                      goto LABEL_270;
                    case 0xE:
                      v17 = *(_QWORD *)(v149 + 24);
                      v191 = *(_QWORD *)(v149 + 32);
                      v192 = 1;
                      v284 = v191;
                      v193 = (unsigned int)sub_15A9FE0(v155, v17);
                      while ( 2 )
                      {
                        switch ( *(_BYTE *)(v17 + 8) )
                        {
                          case 1:
                            v195 = 16;
                            goto LABEL_284;
                          case 2:
                            v195 = 32;
                            goto LABEL_284;
                          case 3:
                          case 9:
                            v195 = 64;
                            goto LABEL_284;
                          case 4:
                            v195 = 80;
                            goto LABEL_284;
                          case 5:
                          case 6:
                            v195 = 128;
                            goto LABEL_284;
                          case 7:
                            v17 = 0;
                            v195 = 8 * (unsigned int)sub_15A9520(v155, 0);
                            goto LABEL_284;
                          case 0xB:
                            v195 = *(_DWORD *)(v17 + 8) >> 8;
                            goto LABEL_284;
                          case 0xD:
                            v195 = 8LL * *(_QWORD *)sub_15A9930(v155, v17);
                            goto LABEL_284;
                          case 0xE:
                            v196 = *(_QWORD *)(v17 + 32);
                            v17 = *(_QWORD *)(v17 + 24);
                            v274 = (unsigned int)sub_15A9FE0(v155, v17);
                            v197 = sub_127FA20(v155, v17);
                            v24 = v196 * v274;
                            v195 = 8 * v196 * v274 * ((v274 + ((unsigned __int64)(v197 + 7) >> 3) - 1) / v274);
                            goto LABEL_284;
                          case 0xF:
                            v17 = *(_DWORD *)(v17 + 8) >> 8;
                            v195 = 8 * (unsigned int)sub_15A9520(v155, v17);
LABEL_284:
                            v23 = (v193 + ((unsigned __int64)(v192 * v195 + 7) >> 3) - 1) % v193;
                            v190 = 8 * v284 * v193 * ((v193 + ((unsigned __int64)(v192 * v195 + 7) >> 3) - 1) / v193);
                            goto LABEL_270;
                          case 0x10:
                            v194 = *(_QWORD *)(v17 + 32);
                            v17 = *(_QWORD *)(v17 + 24);
                            v192 *= v194;
                            continue;
                          default:
                            goto LABEL_468;
                        }
                      }
                    case 0xF:
                      v17 = *(_DWORD *)(v149 + 8) >> 8;
                      v190 = 8 * (unsigned int)sub_15A9520(v155, v17);
LABEL_270:
                      if ( v292 > (unsigned __int64)(v163 * v190 + 7) >> 3 )
                        goto LABEL_190;
                      v142 += 3;
                      break;
                    case 0x10:
                      v189 = *(_QWORD *)(v149 + 32);
                      v149 = *(_QWORD *)(v149 + 24);
                      v163 *= v189;
                      continue;
                    default:
                      goto LABEL_468;
                  }
                  goto LABEL_239;
                }
              case 0x10:
                v245 = *(_QWORD *)(v17 + 32);
                v17 = *(_QWORD *)(v17 + 24);
                v156 *= v245;
                continue;
              default:
                goto LABEL_468;
            }
          }
        }
LABEL_190:
        --v306;
        v142 += 4;
      }
      v218 = (char *)v309 - (char *)v142;
      if ( (char *)v309 - (char *)v142 == 16 )
        goto LABEL_357;
      if ( v218 == 24 )
        break;
      if ( v218 != 8 )
        goto LABEL_240;
LABEL_352:
      v17 = *v142;
      if ( !sub_185CCC0((__int64)&v353, *v142) )
      {
LABEL_240:
        v18 = v323;
        v12 = a1;
        v69 = v360;
        goto LABEL_100;
      }
LABEL_239:
      v303 += 8;
      if ( v309 == v142 )
        goto LABEL_240;
    }
    v17 = *v142;
    if ( sub_185CCC0((__int64)&v353, *v142) )
      goto LABEL_239;
    ++v142;
LABEL_357:
    v17 = *v142;
    if ( sub_185CCC0((__int64)&v353, *v142) )
      goto LABEL_239;
    ++v142;
    goto LABEL_352;
  }
  return v18;
}
