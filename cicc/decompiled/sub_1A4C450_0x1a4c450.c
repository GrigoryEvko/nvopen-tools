// Function: sub_1A4C450
// Address: 0x1a4c450
//
__int64 __fastcall sub_1A4C450(
        __int64 a1,
        __int64 a2,
        __m128i a3,
        __m128i a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 *v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  double v26; // xmm4_8
  double v27; // xmm5_8
  _QWORD *v28; // r15
  bool v29; // zf
  _QWORD *v30; // r14
  _QWORD *v31; // r9
  __int64 *v32; // r13
  __int64 v33; // r12
  _QWORD *v34; // rbx
  __int64 *v35; // r15
  _QWORD *v36; // rdi
  __int64 v37; // rax
  __int64 v38; // rcx
  unsigned __int64 v39; // rdx
  __int64 v40; // rdx
  char v41; // al
  __int64 v42; // r10
  unsigned __int64 v43; // r12
  __int64 v44; // r13
  _QWORD *v45; // rbx
  __int64 *v46; // rbx
  __int64 v47; // r10
  int v48; // r12d
  __int64 v49; // rdx
  __int64 v50; // rax
  char v51; // al
  char v52; // al
  __int64 v53; // rax
  unsigned __int64 v54; // rbx
  unsigned __int64 v55; // r15
  __int64 v56; // rax
  _QWORD *v57; // r13
  __int64 v58; // r14
  __int64 v59; // rbx
  __int64 v60; // rdx
  __int64 v61; // r12
  __int64 v62; // rsi
  __int64 v63; // r13
  unsigned int v64; // eax
  __int64 v65; // rcx
  unsigned __int64 v66; // r8
  __int64 v67; // rax
  unsigned int v68; // esi
  int v69; // eax
  __int64 v70; // rax
  unsigned int v71; // eax
  __int64 v72; // rdx
  __int64 v73; // rsi
  unsigned __int64 v74; // r9
  _QWORD *v75; // rax
  __int64 v76; // rax
  _QWORD *v77; // r15
  __int64 *v78; // rdx
  __int64 v79; // rax
  __int64 v80; // rdx
  __int64 v81; // r12
  __int64 v82; // rax
  _QWORD *v83; // rax
  __int64 v84; // rax
  int v85; // r12d
  unsigned int v86; // edx
  unsigned __int64 v87; // r13
  unsigned __int64 v88; // rbx
  __int64 v89; // rax
  __int64 v90; // r15
  __int64 v91; // rax
  __int64 v92; // rax
  char v93; // al
  __int64 v94; // rdx
  __int64 v95; // rcx
  __int64 v96; // r8
  int v97; // r9d
  int v98; // r15d
  __int64 i; // rdx
  int v100; // r8d
  int v101; // r9d
  const char *v102; // rax
  __int64 v103; // rcx
  const char *v104; // rdi
  unsigned int v105; // r15d
  __int64 v106; // rsi
  __int64 v107; // rax
  __int64 v108; // rsi
  __int64 v109; // r15
  __int64 *v110; // rdx
  __int64 v111; // rsi
  unsigned __int64 v112; // rcx
  __int64 v113; // rsi
  char v114; // al
  __int64 v115; // r12
  __int64 v116; // rbx
  unsigned __int64 v117; // r13
  __int64 v118; // rbx
  __int64 v119; // r12
  signed __int64 v120; // rcx
  __int64 v121; // r13
  __int64 v122; // rdx
  __int64 v123; // r12
  _QWORD *v124; // rax
  __int64 v125; // r13
  _QWORD *v126; // rcx
  __int64 v127; // rax
  unsigned int v128; // ebx
  __int64 *v129; // rax
  __int64 v130; // rax
  __int64 v131; // rcx
  __int64 *v132; // r11
  __int64 v133; // rax
  double v134; // xmm4_8
  double v135; // xmm5_8
  __int64 v136; // rax
  __int64 v137; // rdi
  const char *v138; // rax
  double v139; // xmm4_8
  double v140; // xmm5_8
  __int64 v141; // rax
  unsigned int v142; // r13d
  _QWORD *v143; // rax
  _QWORD *v144; // rax
  __int64 v145; // rbx
  _QWORD *v146; // rax
  _QWORD *v147; // rax
  _QWORD *v148; // rcx
  __int64 v149; // rax
  __int64 *v150; // rax
  __int64 v151; // rax
  __int64 v152; // rcx
  __int64 *v153; // r11
  __int64 v154; // rax
  __int64 v155; // r12
  __int64 v156; // rdx
  _QWORD *v157; // rax
  _QWORD *v158; // rbx
  __int64 *v159; // rax
  __int64 *v160; // rax
  __int64 v161; // rax
  __int64 v162; // rax
  unsigned __int8 *v163; // rsi
  __int64 v164; // r12
  _QWORD *v165; // rbx
  __int64 *v166; // rbx
  __int64 v167; // rax
  double v168; // xmm4_8
  double v169; // xmm5_8
  int v170; // r14d
  unsigned __int64 v171; // r13
  unsigned __int64 v172; // rbx
  __int64 v173; // r15
  __int64 v174; // rsi
  __int64 v175; // r9
  __int64 v176; // rax
  unsigned int v177; // eax
  __int64 v178; // r9
  __int64 v179; // rcx
  unsigned __int64 v180; // r8
  __int64 v181; // rax
  __int64 v182; // rcx
  unsigned int v183; // edx
  __int64 v184; // rax
  char v185; // al
  unsigned int v186; // esi
  int v187; // eax
  __int64 v188; // rax
  unsigned int v189; // eax
  __int64 v190; // r9
  __int64 v191; // rsi
  __int64 v192; // rdi
  _QWORD *v193; // rax
  unsigned __int64 v194; // rax
  int v195; // eax
  __int64 v196; // rax
  __int64 *v197; // r15
  __int64 v198; // rax
  __int64 v199; // rcx
  __int64 v200; // rsi
  unsigned __int8 *v201; // rsi
  __int64 **v202; // rdx
  __int64 v203; // rax
  __int64 v204; // rax
  __int64 v205; // rdx
  int v206; // edi
  __int64 *v207; // rsi
  __int64 v208; // rsi
  __int64 v209; // rax
  __int64 v210; // rsi
  __int64 v211; // rdx
  unsigned __int8 *v212; // rsi
  __int64 v213; // rax
  __int64 v214; // rax
  unsigned int v215; // esi
  int v216; // eax
  unsigned __int64 v217; // rax
  _QWORD *v218; // rax
  __int64 *v219; // rbx
  __int64 v220; // rax
  __int64 v221; // rcx
  __int64 v222; // rsi
  unsigned __int8 *v223; // rsi
  __int64 v224; // rax
  unsigned int v225; // esi
  int v226; // eax
  __int64 v227; // rax
  __int64 v228; // rdi
  unsigned __int64 v229; // rax
  _QWORD *v230; // rax
  __int64 *v231; // rbx
  __int64 v232; // rax
  __int64 v233; // rcx
  __int64 v234; // rsi
  unsigned __int8 *v235; // rsi
  __int64 v236; // rsi
  __int64 *v237; // rbx
  __int64 v238; // rax
  __int64 v239; // rcx
  __int64 v240; // rsi
  unsigned __int8 *v241; // rsi
  __int64 v242; // r13
  _BYTE *v243; // rax
  __int64 v244; // [rsp+0h] [rbp-230h]
  __int64 v245; // [rsp+0h] [rbp-230h]
  __int64 *v247; // [rsp+10h] [rbp-220h]
  __int64 v248; // [rsp+10h] [rbp-220h]
  __int64 v249; // [rsp+10h] [rbp-220h]
  __int64 v250; // [rsp+10h] [rbp-220h]
  __int64 v251; // [rsp+10h] [rbp-220h]
  __int64 v252; // [rsp+18h] [rbp-218h]
  char v253; // [rsp+20h] [rbp-210h]
  __int64 v254; // [rsp+20h] [rbp-210h]
  __int64 v255; // [rsp+20h] [rbp-210h]
  _QWORD *v256; // [rsp+28h] [rbp-208h]
  char v257; // [rsp+30h] [rbp-200h]
  __int64 v258; // [rsp+30h] [rbp-200h]
  __int64 v259; // [rsp+30h] [rbp-200h]
  __int64 v260; // [rsp+30h] [rbp-200h]
  __int64 v261; // [rsp+30h] [rbp-200h]
  __int64 v262; // [rsp+38h] [rbp-1F8h]
  char v263; // [rsp+40h] [rbp-1F0h]
  __int64 v264; // [rsp+40h] [rbp-1F0h]
  unsigned __int64 v265; // [rsp+40h] [rbp-1F0h]
  __int64 v266; // [rsp+40h] [rbp-1F0h]
  unsigned __int64 v267; // [rsp+40h] [rbp-1F0h]
  __int64 v268; // [rsp+40h] [rbp-1F0h]
  _QWORD *v269; // [rsp+50h] [rbp-1E0h]
  __int64 v270; // [rsp+50h] [rbp-1E0h]
  _QWORD *v271; // [rsp+50h] [rbp-1E0h]
  __int64 v272; // [rsp+58h] [rbp-1D8h]
  __int64 v273; // [rsp+58h] [rbp-1D8h]
  _QWORD *v274; // [rsp+58h] [rbp-1D8h]
  _QWORD *v275; // [rsp+58h] [rbp-1D8h]
  unsigned __int64 v276; // [rsp+58h] [rbp-1D8h]
  __int64 v277; // [rsp+58h] [rbp-1D8h]
  unsigned __int64 v278; // [rsp+58h] [rbp-1D8h]
  __int64 v279; // [rsp+60h] [rbp-1D0h]
  unsigned __int64 v280; // [rsp+60h] [rbp-1D0h]
  __int64 v281; // [rsp+60h] [rbp-1D0h]
  unsigned __int64 v282; // [rsp+60h] [rbp-1D0h]
  unsigned __int64 v283; // [rsp+60h] [rbp-1D0h]
  __int64 v284; // [rsp+60h] [rbp-1D0h]
  unsigned __int64 v285; // [rsp+60h] [rbp-1D0h]
  unsigned int v286; // [rsp+60h] [rbp-1D0h]
  __int64 v287; // [rsp+60h] [rbp-1D0h]
  __int64 v288; // [rsp+60h] [rbp-1D0h]
  __int64 v289; // [rsp+60h] [rbp-1D0h]
  __int64 v290; // [rsp+68h] [rbp-1C8h]
  __int64 v291; // [rsp+68h] [rbp-1C8h]
  __int64 v292; // [rsp+68h] [rbp-1C8h]
  int v293; // [rsp+68h] [rbp-1C8h]
  __int64 v294; // [rsp+68h] [rbp-1C8h]
  int v295; // [rsp+68h] [rbp-1C8h]
  __int64 v296; // [rsp+68h] [rbp-1C8h]
  __int64 v297; // [rsp+68h] [rbp-1C8h]
  __int64 v298; // [rsp+68h] [rbp-1C8h]
  __int64 v299; // [rsp+68h] [rbp-1C8h]
  __int64 v300; // [rsp+68h] [rbp-1C8h]
  __int64 *v301; // [rsp+68h] [rbp-1C8h]
  _QWORD *v302; // [rsp+70h] [rbp-1C0h]
  int v303; // [rsp+70h] [rbp-1C0h]
  __int64 v304; // [rsp+70h] [rbp-1C0h]
  int v305; // [rsp+70h] [rbp-1C0h]
  __int64 v306; // [rsp+70h] [rbp-1C0h]
  __int64 v307; // [rsp+78h] [rbp-1B8h]
  __int64 v308; // [rsp+78h] [rbp-1B8h]
  __int64 v309; // [rsp+78h] [rbp-1B8h]
  __int64 v310; // [rsp+78h] [rbp-1B8h]
  __int64 v311; // [rsp+78h] [rbp-1B8h]
  __int64 v312; // [rsp+78h] [rbp-1B8h]
  int v313; // [rsp+78h] [rbp-1B8h]
  _QWORD *v314; // [rsp+80h] [rbp-1B0h]
  char v315; // [rsp+80h] [rbp-1B0h]
  char v316; // [rsp+80h] [rbp-1B0h]
  __int64 v317; // [rsp+80h] [rbp-1B0h]
  __int64 v318; // [rsp+80h] [rbp-1B0h]
  char v319; // [rsp+88h] [rbp-1A8h]
  __int64 *v320; // [rsp+88h] [rbp-1A8h]
  __int64 *v321; // [rsp+88h] [rbp-1A8h]
  bool v322; // [rsp+88h] [rbp-1A8h]
  __int64 *v323; // [rsp+88h] [rbp-1A8h]
  unsigned __int8 v325; // [rsp+98h] [rbp-198h]
  _QWORD *v326; // [rsp+98h] [rbp-198h]
  __int64 v327; // [rsp+A8h] [rbp-188h] BYREF
  __int64 v328; // [rsp+B0h] [rbp-180h] BYREF
  unsigned int v329; // [rsp+B8h] [rbp-178h]
  __int64 v330[2]; // [rsp+C0h] [rbp-170h] BYREF
  __int16 v331; // [rsp+D0h] [rbp-160h]
  __int64 v332; // [rsp+E0h] [rbp-150h] BYREF
  __int64 v333; // [rsp+E8h] [rbp-148h]
  __int16 v334; // [rsp+F0h] [rbp-140h] BYREF
  const char *v335; // [rsp+100h] [rbp-130h] BYREF
  __int64 v336; // [rsp+108h] [rbp-128h]
  __int64 *v337; // [rsp+110h] [rbp-120h] BYREF
  __int64 v338; // [rsp+118h] [rbp-118h]
  __int64 v339; // [rsp+120h] [rbp-110h]
  __int64 *v340; // [rsp+128h] [rbp-108h]
  __int64 v341; // [rsp+130h] [rbp-100h]
  __int64 v342; // [rsp+138h] [rbp-F8h]
  _BYTE *v343; // [rsp+150h] [rbp-E0h]
  __int64 v344; // [rsp+158h] [rbp-D8h]
  _BYTE v345[128]; // [rsp+160h] [rbp-D0h] BYREF
  _QWORD *v346; // [rsp+1E0h] [rbp-50h]
  __int64 v347; // [rsp+1E8h] [rbp-48h]
  __int64 v348; // [rsp+1F0h] [rbp-40h]

  v325 = sub_1636880(a1, a2);
  if ( v325 )
    return 0;
  v253 = byte_4FB41E0;
  if ( !byte_4FB41E0 )
  {
    v11 = *(__int64 **)(a1 + 8);
    v12 = *v11;
    v13 = v11[1];
    if ( v12 == v13 )
LABEL_377:
      BUG();
    while ( *(_UNKNOWN **)v12 != &unk_4F9E06C )
    {
      v12 += 16;
      if ( v13 == v12 )
        goto LABEL_377;
    }
    v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(
            *(_QWORD *)(v12 + 8),
            &unk_4F9E06C);
    v15 = *(__int64 **)(a1 + 8);
    *(_QWORD *)(a1 + 168) = v14 + 160;
    v16 = *v15;
    v17 = v15[1];
    if ( v16 == v17 )
LABEL_382:
      BUG();
    while ( *(_UNKNOWN **)v16 != &unk_4F9A488 )
    {
      v16 += 16;
      if ( v17 == v16 )
        goto LABEL_382;
    }
    v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v16 + 8) + 104LL))(
            *(_QWORD *)(v16 + 8),
            &unk_4F9A488);
    v19 = *(__int64 **)(a1 + 8);
    *(_QWORD *)(a1 + 176) = *(_QWORD *)(v18 + 160);
    v20 = *v19;
    v21 = v19[1];
    if ( v20 == v21 )
LABEL_381:
      BUG();
    while ( *(_UNKNOWN **)v20 != &unk_4F9920C )
    {
      v20 += 16;
      if ( v21 == v20 )
        goto LABEL_381;
    }
    v22 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v20 + 8) + 104LL))(
            *(_QWORD *)(v20 + 8),
            &unk_4F9920C);
    v23 = *(__int64 **)(a1 + 8);
    *(_QWORD *)(a1 + 184) = v22 + 160;
    v24 = *v23;
    v25 = v23[1];
    if ( v24 == v25 )
LABEL_380:
      BUG();
    while ( *(_UNKNOWN **)v24 != &unk_4F9B6E8 )
    {
      v24 += 16;
      if ( v25 == v24 )
        goto LABEL_380;
    }
    *(_QWORD *)(a1 + 192) = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v24 + 8) + 104LL))(
                              *(_QWORD *)(v24 + 8),
                              &unk_4F9B6E8)
                          + 360;
    v252 = a2 + 72;
    v256 = *(_QWORD **)(a2 + 80);
    if ( v256 == (_QWORD *)(a2 + 72) )
    {
LABEL_78:
      v325 = v253 | sub_1A4B060(a1, a3, a4, a5, a6, v26, v27, a9, a10);
      if ( byte_4FB4100 )
      {
        v58 = *(_QWORD *)(a2 + 80);
        if ( v252 != v58 )
        {
          while ( 1 )
          {
            if ( !v58 )
              BUG();
            v59 = *(_QWORD *)(v58 + 24);
            if ( v59 != v58 + 16 )
              break;
LABEL_86:
            v58 = *(_QWORD *)(v58 + 8);
            if ( v252 == v58 )
              return v325;
          }
          while ( 1 )
          {
            v60 = v59 - 24;
            if ( !v59 )
              v60 = 0;
            v61 = v60;
            if ( (unsigned __int8)sub_1AE9990(v60, 0) )
              break;
            v59 = *(_QWORD *)(v59 + 8);
            if ( v58 + 16 == v59 )
              goto LABEL_86;
          }
          v333 = 0;
          v332 = (__int64)&v334;
          LOBYTE(v334) = 0;
          LODWORD(v339) = 1;
          v335 = (const char *)&unk_49EFBE0;
          v338 = 0;
          v337 = 0;
          v336 = 0;
          v340 = &v332;
          v242 = sub_16E7EE0((__int64)&v335, "Dead instruction detected!\n", 0x1Bu);
          sub_155C2B0(v61, v242, 0);
          v243 = *(_BYTE **)(v242 + 24);
          if ( *(_BYTE **)(v242 + 16) == v243 )
          {
            sub_16E7EE0(v242, "\n", 1u);
          }
          else
          {
            *v243 = 10;
            ++*(_QWORD *)(v242 + 24);
          }
LABEL_375:
          BUG();
        }
      }
      return v325;
    }
    while ( 1 )
    {
      if ( !v256 )
        BUG();
      v326 = (_QWORD *)v256[3];
      if ( v326 != v256 + 2 )
        break;
LABEL_77:
      v256 = (_QWORD *)v256[1];
      if ( (_QWORD *)v252 == v256 )
        goto LABEL_78;
    }
LABEL_23:
    v28 = v326;
    v29 = *((_BYTE *)v326 - 8) == 56;
    v326 = (_QWORD *)v326[1];
    if ( !v29 )
      goto LABEL_76;
    if ( *(_BYTE *)(*(v28 - 3) + 8LL) == 16 )
      goto LABEL_76;
    v30 = v28 - 3;
    v263 = sub_15FA290((__int64)(v28 - 3));
    if ( v263 )
      goto LABEL_76;
    v307 = sub_15A9650(*(_QWORD *)(a1 + 160), *(v28 - 3));
    if ( (*((_BYTE *)v28 - 1) & 0x40) != 0 )
      v31 = (_QWORD *)*(v28 - 4);
    else
      v31 = &v30[-3 * (*((_DWORD *)v28 - 1) & 0xFFFFFFF)];
    v32 = v31 + 3;
    v33 = sub_16348C0((__int64)(v28 - 3)) | 4;
    v34 = &v30[3 * (1LL - (*((_DWORD *)v28 - 1) & 0xFFFFFFF))];
    if ( v30 != v34 )
    {
      v302 = v28;
      v35 = v32;
      v257 = 0;
      while ( 1 )
      {
        while ( 1 )
        {
          v42 = v33;
          v314 = v34;
          v43 = v33 & 0xFFFFFFFFFFFFFFF8LL;
          v34 += 3;
          v44 = v43;
          LODWORD(v42) = (v42 >> 2) & 1;
          v319 = v42;
          if ( (_DWORD)v42 )
          {
            v36 = (_QWORD *)*(v34 - 3);
            if ( v307 != *v36 )
            {
              v335 = "idxprom";
              LOWORD(v337) = 259;
              v37 = sub_15FE0A0(v36, v307, 1, (__int64)&v335, (__int64)v30);
              if ( *(v34 - 3) )
              {
                v38 = *(v34 - 2);
                v39 = *(v34 - 1) & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v39 = v38;
                if ( v38 )
                  *(_QWORD *)(v38 + 16) = *(_QWORD *)(v38 + 16) & 3LL | v39;
              }
              *(v34 - 3) = v37;
              if ( v37 )
              {
                v40 = *(_QWORD *)(v37 + 8);
                *(v34 - 2) = v40;
                if ( v40 )
                  *(_QWORD *)(v40 + 16) = (unsigned __int64)(v34 - 2) | *(_QWORD *)(v40 + 16) & 3LL;
                *(v34 - 1) = (v37 + 8) | *(v34 - 1) & 3LL;
                *(_QWORD *)(v37 + 8) = v314;
              }
              v257 = v319;
            }
            if ( v43 )
              break;
          }
          v44 = sub_1643D30(v43, *v35);
          v41 = *(_BYTE *)(v44 + 8);
          if ( ((v41 - 14) & 0xFD) == 0 )
            goto LABEL_41;
LABEL_44:
          v33 = 0;
          if ( v41 == 13 )
            v33 = v44;
          v35 += 3;
          if ( v30 == v34 )
          {
LABEL_47:
            v28 = v302;
            goto LABEL_48;
          }
        }
        v41 = *(_BYTE *)(v43 + 8);
        if ( ((v41 - 14) & 0xFD) != 0 )
          goto LABEL_44;
LABEL_41:
        v35 += 3;
        v33 = *(_QWORD *)(v44 + 24) | 4LL;
        if ( v30 == v34 )
          goto LABEL_47;
      }
    }
    v257 = 0;
LABEL_48:
    if ( (*((_BYTE *)v28 - 1) & 0x40) != 0 )
      v45 = (_QWORD *)*(v28 - 4);
    else
      v45 = &v30[-3 * (*((_DWORD *)v28 - 1) & 0xFFFFFFF)];
    v46 = v45 + 3;
    v47 = sub_16348C0((__int64)v30) | 4;
    v303 = *((_DWORD *)v28 - 1) & 0xFFFFFFF;
    if ( v303 == 1 )
      goto LABEL_75;
    v320 = v46;
    v48 = 2;
    v262 = 0;
    v269 = v28;
    while ( 1 )
    {
      v53 = (unsigned int)(v48 - 1);
      v54 = v47 & 0xFFFFFFFFFFFFFFF8LL;
      v55 = v47 & 0xFFFFFFFFFFFFFFF8LL;
      v315 = (v47 >> 2) & 1;
      if ( v315 )
      {
        v272 = *(_QWORD *)(a1 + 168);
        v49 = v30[3 * (v53 - (*((_DWORD *)v269 - 1) & 0xFFFFFFF))];
        v335 = (const char *)&v337;
        v336 = 0x800000000LL;
        v343 = v345;
        v279 = v49;
        v344 = 0x1000000000LL;
        v346 = v30;
        v50 = sub_15F2050((__int64)v30);
        v347 = sub_1632FA0(v50);
        v348 = v272;
        v51 = sub_15FA300((__int64)v30);
        sub_1A49080((__int64)&v332, (__int64)&v335, v279, 0, 0, v51);
        if ( (unsigned int)v333 > 0x40 )
        {
          v308 = *(_QWORD *)v332;
          j_j___libc_free_0_0(v332);
        }
        else
        {
          v308 = v332 << (64 - (unsigned __int8)v333) >> (64 - (unsigned __int8)v333);
        }
        if ( v343 != v345 )
          _libc_free((unsigned __int64)v343);
        if ( v335 != (const char *)&v337 )
          _libc_free((unsigned __int64)v335);
        if ( v308 )
        {
          v62 = v54;
          v63 = *(_QWORD *)(a1 + 160);
          if ( !v54 )
            v62 = sub_1643D30(0, *v320);
          v64 = sub_15A9FE0(v63, v62);
          v65 = 1;
          v66 = v64;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v62 + 8) )
            {
              case 0:
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v70 = *(_QWORD *)(v62 + 32);
                v62 = *(_QWORD *)(v62 + 24);
                v65 *= v70;
                continue;
              case 1:
                v67 = 16;
                goto LABEL_94;
              case 2:
                v67 = 32;
                goto LABEL_94;
              case 3:
              case 9:
                v67 = 64;
                goto LABEL_94;
              case 4:
                v67 = 80;
                goto LABEL_94;
              case 5:
              case 6:
                v67 = 128;
                goto LABEL_94;
              case 7:
                v280 = v66;
                v68 = 0;
                v290 = v65;
                goto LABEL_100;
              case 0xB:
                v67 = *(_DWORD *)(v62 + 8) >> 8;
                goto LABEL_94;
              case 0xD:
                v282 = v66;
                v292 = v65;
                v75 = (_QWORD *)sub_15A9930(v63, v62);
                v65 = v292;
                v66 = v282;
                v67 = 8LL * *v75;
                goto LABEL_94;
              case 0xE:
                v264 = v66;
                v273 = v65;
                v281 = *(_QWORD *)(v62 + 24);
                v291 = *(_QWORD *)(v62 + 32);
                v71 = sub_15A9FE0(v63, v281);
                v66 = v264;
                v72 = 1;
                v73 = v281;
                v65 = v273;
                v74 = v71;
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v73 + 8) )
                  {
                    case 0:
                    case 8:
                    case 0xA:
                    case 0xC:
                    case 0x10:
                      v214 = *(_QWORD *)(v73 + 32);
                      v73 = *(_QWORD *)(v73 + 24);
                      v72 *= v214;
                      continue;
                    case 1:
                      v76 = 16;
                      goto LABEL_108;
                    case 2:
                      v76 = 32;
                      goto LABEL_108;
                    case 3:
                    case 9:
                      v76 = 64;
                      goto LABEL_108;
                    case 4:
                      v76 = 80;
                      goto LABEL_108;
                    case 5:
                    case 6:
                      v76 = 128;
                      goto LABEL_108;
                    case 7:
                      v249 = v264;
                      v215 = 0;
                      v266 = v273;
                      v276 = v74;
                      v287 = v72;
                      goto LABEL_323;
                    case 0xB:
                      v76 = *(_DWORD *)(v73 + 8) >> 8;
                      goto LABEL_108;
                    case 0xD:
                      v251 = v264;
                      v268 = v273;
                      v278 = v74;
                      v289 = v72;
                      v218 = (_QWORD *)sub_15A9930(v63, v73);
                      v72 = v289;
                      v74 = v278;
                      v65 = v268;
                      v66 = v251;
                      v76 = 8LL * *v218;
                      goto LABEL_108;
                    case 0xE:
                      v245 = v264;
                      v250 = v273;
                      v267 = v74;
                      v277 = v72;
                      v288 = *(_QWORD *)(v73 + 32);
                      v217 = sub_12BE0A0(v63, *(_QWORD *)(v73 + 24));
                      v72 = v277;
                      v74 = v267;
                      v65 = v250;
                      v66 = v245;
                      v76 = 8 * v288 * v217;
                      goto LABEL_108;
                    case 0xF:
                      v249 = v264;
                      v266 = v273;
                      v276 = v74;
                      v215 = *(_DWORD *)(v73 + 8) >> 8;
                      v287 = v72;
LABEL_323:
                      v216 = sub_15A9520(v63, v215);
                      v72 = v287;
                      v74 = v276;
                      v65 = v266;
                      v66 = v249;
                      v76 = (unsigned int)(8 * v216);
LABEL_108:
                      v67 = 8 * v291 * v74 * ((v74 + ((unsigned __int64)(v76 * v72 + 7) >> 3) - 1) / v74);
                      break;
                  }
                  goto LABEL_94;
                }
              case 0xF:
                v280 = v66;
                v290 = v65;
                v68 = *(_DWORD *)(v62 + 8) >> 8;
LABEL_100:
                v69 = sub_15A9520(v63, v68);
                v65 = v290;
                v66 = v280;
                v67 = (unsigned int)(8 * v69);
LABEL_94:
                v262 += v66 * v308 * ((v66 + ((unsigned __int64)(v67 * v65 + 7) >> 3) - 1) / v66);
                v263 = v315;
                break;
            }
            break;
          }
        }
        if ( v54 )
        {
          v52 = *(_BYTE *)(v55 + 8);
          if ( ((v52 - 14) & 0xFD) == 0 )
            goto LABEL_61;
          goto LABEL_71;
        }
      }
      else if ( *(_BYTE *)(a1 + 200) )
      {
        v56 = v30[3 * (v53 - (*((_DWORD *)v269 - 1) & 0xFFFFFFF))];
        v57 = *(_QWORD **)(v56 + 24);
        if ( *(_DWORD *)(v56 + 32) > 0x40u )
          v57 = (_QWORD *)*v57;
        if ( v57 )
        {
          v263 = *(_BYTE *)(a1 + 200);
          v262 += *(_QWORD *)(sub_15A9930(*(_QWORD *)(a1 + 160), v47 & 0xFFFFFFFFFFFFFFF8LL)
                            + 8LL * (unsigned int)v57
                            + 16);
        }
      }
      v55 = sub_1643D30(v54, *v320);
      v52 = *(_BYTE *)(v55 + 8);
      if ( ((v52 - 14) & 0xFD) == 0 )
      {
LABEL_61:
        v47 = *(_QWORD *)(v55 + 24) | 4LL;
        goto LABEL_62;
      }
LABEL_71:
      v47 = 0;
      if ( v52 == 13 )
        v47 = v55;
LABEL_62:
      v320 += 3;
      if ( v303 == v48 )
      {
        v77 = v269;
        if ( v263 )
        {
          v78 = *(__int64 **)(a1 + 8);
          v79 = *v78;
          v80 = v78[1];
          if ( v79 == v80 )
            goto LABEL_375;
          while ( *(_UNKNOWN **)v79 != &unk_4F9D3C0 )
          {
            v79 += 16;
            if ( v80 == v79 )
              goto LABEL_375;
          }
          v81 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v79 + 8) + 104LL))(
                  *(_QWORD *)(v79 + 8),
                  &unk_4F9D3C0);
          v82 = sub_15F2060((__int64)v30);
          v247 = (__int64 *)sub_14A4050(v81, v82);
          if ( *(_BYTE *)(a1 + 200) || sub_14A2A90(v247, v269[5], 0, v262, 1u, 0) )
          {
            if ( (*((_BYTE *)v269 - 1) & 0x40) != 0 )
              v83 = (_QWORD *)*(v269 - 4);
            else
              v83 = &v30[-3 * (*((_DWORD *)v269 - 1) & 0xFFFFFFF)];
            v321 = v83 + 3;
            v84 = sub_16348C0((__int64)v30) | 4;
            v293 = *((_DWORD *)v269 - 1) & 0xFFFFFFF;
            if ( v293 != 1 )
            {
              v274 = v269;
              v85 = 1;
              while ( 2 )
              {
                v86 = v85++;
                v87 = v84 & 0xFFFFFFFFFFFFFFF8LL;
                v88 = v84 & 0xFFFFFFFFFFFFFFF8LL;
                if ( (v84 & 4) == 0 )
                  goto LABEL_172;
                v270 = v86;
                v89 = v30[3 * (v86 - (unsigned __int64)(*((_DWORD *)v274 - 1) & 0xFFFFFFF))];
                v335 = (const char *)&v337;
                v90 = *(_QWORD *)(a1 + 168);
                v343 = v345;
                v336 = 0x800000000LL;
                v344 = 0x1000000000LL;
                v309 = v89;
                v346 = v30;
                v91 = sub_15F2050((__int64)v30);
                v92 = sub_1632FA0(v91);
                v348 = v90;
                v347 = v92;
                v93 = sub_15FA300((__int64)v30);
                sub_1A49080((__int64)&v332, (__int64)&v335, v309, 0, 0, v93);
                v98 = v333;
                if ( (unsigned int)v333 <= 0x40 )
                {
                  if ( v332 )
                    goto LABEL_123;
                  v258 = 0;
                  v109 = 0;
LABEL_132:
                  if ( v343 != v345 )
                    _libc_free((unsigned __int64)v343);
                  if ( v335 != (const char *)&v337 )
                    _libc_free((unsigned __int64)v335);
                  if ( v109 )
                  {
                    v110 = &v30[3 * (v270 - (*((_DWORD *)v274 - 1) & 0xFFFFFFF))];
                    if ( *v110 )
                    {
                      v111 = v110[1];
                      v112 = v110[2] & 0xFFFFFFFFFFFFFFFCLL;
                      *(_QWORD *)v112 = v111;
                      if ( v111 )
                        *(_QWORD *)(v111 + 16) = *(_QWORD *)(v111 + 16) & 3LL | v112;
                    }
                    *v110 = v109;
                    v113 = *(_QWORD *)(v109 + 8);
                    v110[1] = v113;
                    if ( v113 )
                      *(_QWORD *)(v113 + 16) = (unsigned __int64)(v110 + 1) | *(_QWORD *)(v113 + 16) & 3LL;
                    v110[2] = v110[2] & 3 | (v109 + 8);
                    *(_QWORD *)(v109 + 8) = v110;
                    sub_1AEB370(v258, 0);
                    sub_1AEB370(v309, 0);
                  }
                  if ( v87 )
                  {
                    v114 = *(_BYTE *)(v88 + 8);
                    if ( ((v114 - 14) & 0xFD) != 0 )
                      goto LABEL_173;
                    goto LABEL_145;
                  }
LABEL_172:
                  v88 = sub_1643D30(v87, *v321);
                  v114 = *(_BYTE *)(v88 + 8);
                  if ( ((v114 - 14) & 0xFD) != 0 )
                  {
LABEL_173:
                    v29 = v114 == 13;
                    v84 = 0;
                    if ( v29 )
                      v84 = v88;
                    goto LABEL_146;
                  }
LABEL_145:
                  v84 = *(_QWORD *)(v88 + 24) | 4LL;
LABEL_146:
                  v321 += 3;
                  if ( v293 == v85 )
                  {
                    v77 = v274;
                    goto LABEL_148;
                  }
                  continue;
                }
                break;
              }
              if ( v98 - (unsigned int)sub_16A57B0((__int64)&v332) <= 0x40 )
              {
                v137 = v332;
                if ( !*(_QWORD *)v332 )
                {
                  v258 = 0;
                  v109 = 0;
LABEL_179:
                  j_j___libc_free_0_0(v137);
                  goto LABEL_132;
                }
              }
LABEL_123:
              sub_1A48EA0((__int64)&v335, v336 - 1, v94, v95, v96, v97);
              v102 = v335;
              v103 = (unsigned int)v336;
              v104 = &v335[8 * (unsigned int)v336];
              if ( v335 == v104 )
              {
                v108 = 0xFFFFFFFFLL;
                if ( (_DWORD)v336 )
                {
                  v105 = 0;
LABEL_130:
                  LODWORD(v336) = v105;
                }
              }
              else
              {
                v105 = 0;
                do
                {
                  if ( *(_QWORD *)v102 )
                  {
                    v103 = (__int64)v335;
                    v106 = v105++;
                    *(_QWORD *)&v335[8 * v106] = *(_QWORD *)v102;
                  }
                  v102 += 8;
                }
                while ( v104 != v102 );
                v107 = (unsigned int)v336;
                i = v105;
                if ( (unsigned int)v336 > (unsigned __int64)v105 )
                {
                  v108 = v105 - 1;
                  goto LABEL_130;
                }
                if ( (unsigned int)v336 >= (unsigned __int64)v105 )
                {
                  v108 = (unsigned int)(v336 - 1);
                }
                else
                {
                  if ( v105 > HIDWORD(v336) )
                  {
                    sub_16CD150((__int64)&v335, &v337, v105, 8, v100, v101);
                    v107 = (unsigned int)v336;
                    i = v105;
                  }
                  v103 = (__int64)v335;
                  v138 = &v335[8 * v107];
                  for ( i = (__int64)&v335[8 * i]; (const char *)i != v138; v138 += 8 )
                  {
                    if ( v138 )
                      *(_QWORD *)v138 = 0;
                  }
                  LODWORD(v336) = v105;
                  v108 = v105 - 1;
                }
              }
              v109 = sub_1A48A90((__int64)&v335, v108, i, v103);
              v258 = *(_QWORD *)&v335[8 * (unsigned int)v336 - 8];
              if ( (unsigned int)v333 <= 0x40 )
                goto LABEL_132;
              v137 = v332;
              if ( !v332 )
                goto LABEL_132;
              goto LABEL_179;
            }
LABEL_148:
            v322 = sub_15FA300((__int64)v30);
            sub_15FA2E0((__int64)v30, 0);
            v316 = *(_BYTE *)(a1 + 200);
            if ( !v316 )
            {
              v253 = v263;
              if ( v262 )
              {
                v317 = sub_15F4880((__int64)v30);
                sub_15F2120(v317, (__int64)v30);
                v115 = v77[5];
                v116 = *(_QWORD *)(a1 + 160);
                v310 = 1;
                v117 = (unsigned int)sub_15A9FE0(v116, v115);
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v115 + 8) )
                  {
                    case 1:
                      v118 = 16;
                      goto LABEL_153;
                    case 2:
                      v118 = 32;
                      goto LABEL_153;
                    case 3:
                    case 9:
                      v118 = 64;
                      goto LABEL_153;
                    case 4:
                      v118 = 80;
                      goto LABEL_153;
                    case 5:
                    case 6:
                      v118 = 128;
                      goto LABEL_153;
                    case 7:
                      v118 = 8 * (unsigned int)sub_15A9520(v116, 0);
                      goto LABEL_153;
                    case 0xB:
                      v118 = *(_DWORD *)(v115 + 8) >> 8;
                      goto LABEL_153;
                    case 0xD:
                      v118 = 8LL * *(_QWORD *)sub_15A9930(v116, v115);
                      goto LABEL_153;
                    case 0xE:
                      v304 = *(_QWORD *)(v115 + 32);
                      v118 = 8 * sub_12BE0A0(v116, *(_QWORD *)(v115 + 24)) * v304;
                      goto LABEL_153;
                    case 0xF:
                      v118 = 8 * (unsigned int)sub_15A9520(v116, *(_DWORD *)(v115 + 8) >> 8);
LABEL_153:
                      v119 = sub_15A9650(*(_QWORD *)(a1 + 160), *(v77 - 3));
                      v120 = v117 * ((v117 + ((unsigned __int64)(v310 * v118 + 7) >> 3) - 1) / v117);
                      v121 = v262 / v120;
                      if ( v262 % v120 )
                      {
                        v141 = *(_QWORD *)v77[-3 * (*((_DWORD *)v77 - 1) & 0xFFFFFFF) - 3];
                        if ( *(_BYTE *)(v141 + 8) == 16 )
                          v141 = **(_QWORD **)(v141 + 16);
                        v142 = *(_DWORD *)(v141 + 8);
                        v143 = (_QWORD *)sub_16498A0((__int64)v30);
                        v312 = sub_16471D0(v143, v142 >> 8);
                        LOWORD(v337) = 257;
                        v144 = sub_1648A60(56, 1u);
                        v145 = (__int64)v144;
                        if ( v144 )
                          sub_15FD590((__int64)v144, v317, v312, (__int64)&v335, (__int64)v30);
                        v335 = "uglygep";
                        LOWORD(v337) = 259;
                        v332 = sub_15A0680(v119, v262, 1u);
                        v146 = (_QWORD *)sub_16498A0((__int64)v30);
                        v318 = sub_1643330(v146);
                        if ( !v318 )
                        {
                          v203 = *(_QWORD *)v145;
                          if ( *(_BYTE *)(*(_QWORD *)v145 + 8LL) == 16 )
                            v203 = **(_QWORD **)(v203 + 16);
                          v318 = *(_QWORD *)(v203 + 24);
                        }
                        v147 = sub_1648A60(72, 2u);
                        v125 = (__int64)v147;
                        if ( v147 )
                        {
                          v148 = v147 - 6;
                          v149 = *(_QWORD *)v145;
                          if ( *(_BYTE *)(*(_QWORD *)v145 + 8LL) == 16 )
                            v149 = **(_QWORD **)(v149 + 16);
                          v294 = (__int64)v148;
                          v305 = *(_DWORD *)(v149 + 8) >> 8;
                          v150 = (__int64 *)sub_15F9F50(v318, (__int64)&v332, 1);
                          v151 = sub_1646BA0(v150, v305);
                          v152 = v294;
                          v153 = (__int64 *)v151;
                          v154 = *(_QWORD *)v145;
                          if ( *(_BYTE *)(*(_QWORD *)v145 + 8LL) == 16
                            || (v154 = *(_QWORD *)v332, *(_BYTE *)(*(_QWORD *)v332 + 8LL) == 16) )
                          {
                            v160 = sub_16463B0(v153, *(_QWORD *)(v154 + 32));
                            v152 = v294;
                            v153 = v160;
                          }
                          sub_15F1EA0(v125, (__int64)v153, 32, v152, 2, (__int64)v30);
                          *(_QWORD *)(v125 + 56) = v318;
                          *(_QWORD *)(v125 + 64) = sub_15F9F50(v318, (__int64)&v332, 1);
                          sub_15F9CE0(v125, v145, &v332, 1, (__int64)&v335);
                        }
                        sub_15F4370(v125, (__int64)v30, 0, 0);
                        sub_15FA2E0(v125, v322);
                        v155 = *(v77 - 3);
                        if ( v312 != v155 )
                        {
                          v332 = (__int64)sub_1649960((__int64)v30);
                          LOWORD(v337) = 261;
                          v333 = v156;
                          v335 = (const char *)&v332;
                          v157 = sub_1648A60(56, 1u);
                          v158 = v157;
                          if ( v157 )
                            sub_15FD590((__int64)v157, v125, v155, (__int64)&v335, (__int64)v30);
                          v125 = (__int64)v158;
                        }
                      }
                      else
                      {
                        v332 = (__int64)sub_1649960((__int64)v30);
                        v333 = v122;
                        LOWORD(v337) = 261;
                        v335 = (const char *)&v332;
                        v330[0] = sub_15A0680(v119, v121, 1u);
                        v123 = v77[5];
                        if ( !v123 )
                        {
                          v204 = *(_QWORD *)v317;
                          if ( *(_BYTE *)(*(_QWORD *)v317 + 8LL) == 16 )
                            v204 = **(_QWORD **)(v204 + 16);
                          v123 = *(_QWORD *)(v204 + 24);
                        }
                        v124 = sub_1648A60(72, 2u);
                        v125 = (__int64)v124;
                        if ( v124 )
                        {
                          v126 = v124 - 6;
                          v127 = *(_QWORD *)v317;
                          if ( *(_BYTE *)(*(_QWORD *)v317 + 8LL) == 16 )
                            v127 = **(_QWORD **)(v127 + 16);
                          v128 = *(_DWORD *)(v127 + 8);
                          v311 = (__int64)v126;
                          v129 = (__int64 *)sub_15F9F50(v123, (__int64)v330, 1);
                          v130 = sub_1646BA0(v129, v128 >> 8);
                          v131 = v311;
                          v132 = (__int64 *)v130;
                          v133 = *(_QWORD *)v317;
                          if ( *(_BYTE *)(*(_QWORD *)v317 + 8LL) == 16
                            || (v133 = *(_QWORD *)v330[0], *(_BYTE *)(*(_QWORD *)v330[0] + 8LL) == 16) )
                          {
                            v159 = sub_16463B0(v132, *(_QWORD *)(v133 + 32));
                            v131 = v311;
                            v132 = v159;
                          }
                          sub_15F1EA0(v125, (__int64)v132, 32, v131, 2, (__int64)v30);
                          *(_QWORD *)(v125 + 56) = v123;
                          *(_QWORD *)(v125 + 64) = sub_15F9F50(v123, (__int64)v330, 1);
                          sub_15F9CE0(v125, v317, v330, 1, (__int64)&v335);
                        }
                        sub_15F4370(v125, (__int64)v30, 0, 0);
                        sub_15FA2E0(v125, v322);
                      }
                      sub_164D160((__int64)v30, v125, (__m128)a3, *(double *)a4.m128i_i64, a5, a6, v134, v135, a9, a10);
                      sub_15F20C0(v30);
                      v253 = v263;
                      goto LABEL_76;
                    case 0x10:
                      v136 = v310 * *(_QWORD *)(v115 + 32);
                      v115 = *(_QWORD *)(v115 + 24);
                      v310 = v136;
                      continue;
                    default:
                      goto LABEL_377;
                  }
                }
              }
              goto LABEL_76;
            }
            v253 = sub_14A2D50((__int64)v247);
            if ( v253 )
            {
              sub_1A49A50(
                (__int64 *)a1,
                (__int64)v30,
                v262,
                (__m128)a3,
                *(double *)a4.m128i_i64,
                a5,
                a6,
                v139,
                v140,
                a9,
                a10);
              goto LABEL_76;
            }
            v161 = sub_16498A0((__int64)v30);
            v337 = 0;
            v335 = 0;
            v338 = v161;
            v339 = 0;
            LODWORD(v340) = 0;
            v341 = 0;
            v342 = 0;
            v162 = v77[2];
            v337 = v77;
            v336 = v162;
            v163 = (unsigned __int8 *)v77[3];
            v332 = (__int64)v163;
            if ( v163 )
            {
              sub_1623A60((__int64)&v332, (__int64)v163, 2);
              if ( v335 )
                sub_161E7C0((__int64)&v335, (__int64)v335);
              v335 = (const char *)v332;
              if ( v332 )
                sub_1623210((__int64)&v332, (unsigned __int8 *)v332, (__int64)&v335);
            }
            v306 = sub_15A9650(*(_QWORD *)(a1 + 160), *(v77 - 3));
            v331 = 257;
            v164 = v77[-3 * (*((_DWORD *)v77 - 1) & 0xFFFFFFF) - 3];
            if ( v306 != *(_QWORD *)v164 )
            {
              if ( *(_BYTE *)(v164 + 16) > 0x10u )
              {
                v236 = v77[-3 * (*((_DWORD *)v77 - 1) & 0xFFFFFFF) - 3];
                v334 = 257;
                v164 = sub_15FDBD0(45, v236, v306, (__int64)&v332, 0);
                if ( v336 )
                {
                  v237 = v337;
                  sub_157E9D0(v336 + 40, v164);
                  v238 = *(_QWORD *)(v164 + 24);
                  v239 = *v237;
                  *(_QWORD *)(v164 + 32) = v237;
                  v239 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)(v164 + 24) = v239 | v238 & 7;
                  *(_QWORD *)(v239 + 8) = v164 + 24;
                  *v237 = *v237 & 7 | (v164 + 24);
                }
                sub_164B780(v164, v330);
                if ( v335 )
                {
                  v328 = (__int64)v335;
                  sub_1623A60((__int64)&v328, (__int64)v335, 2);
                  v240 = *(_QWORD *)(v164 + 48);
                  if ( v240 )
                    sub_161E7C0(v164 + 48, v240);
                  v241 = (unsigned __int8 *)v328;
                  *(_QWORD *)(v164 + 48) = v328;
                  if ( v241 )
                    sub_1623210((__int64)&v328, v241, v164 + 48);
                }
              }
              else
              {
                v164 = sub_15A46C0(
                         45,
                         (__int64 ***)v77[-3 * (*((_DWORD *)v77 - 1) & 0xFFFFFFF) - 3],
                         (__int64 **)v306,
                         0);
              }
            }
            if ( (*((_BYTE *)v77 - 1) & 0x40) != 0 )
              v165 = (_QWORD *)*(v77 - 4);
            else
              v165 = &v30[-3 * (*((_DWORD *)v77 - 1) & 0xFFFFFFF)];
            v166 = v165 + 3;
            v167 = sub_16348C0((__int64)v30) | 4;
            v313 = *((_DWORD *)v77 - 1) & 0xFFFFFFF;
            if ( v313 == 1 )
            {
LABEL_290:
              if ( v262 )
              {
                v331 = 257;
                v213 = sub_15A0680(v306, v262, 0);
                if ( *(_BYTE *)(v164 + 16) > 0x10u || *(_BYTE *)(v213 + 16) > 0x10u )
                {
                  v334 = 257;
                  v164 = sub_15FB440(11, (__int64 *)v164, v213, (__int64)&v332, 0);
                  if ( v336 )
                  {
                    v219 = v337;
                    sub_157E9D0(v336 + 40, v164);
                    v220 = *(_QWORD *)(v164 + 24);
                    v221 = *v219;
                    *(_QWORD *)(v164 + 32) = v219;
                    v221 &= 0xFFFFFFFFFFFFFFF8LL;
                    *(_QWORD *)(v164 + 24) = v221 | v220 & 7;
                    *(_QWORD *)(v221 + 8) = v164 + 24;
                    *v219 = *v219 & 7 | (v164 + 24);
                  }
                  sub_164B780(v164, v330);
                  if ( v335 )
                  {
                    v328 = (__int64)v335;
                    sub_1623A60((__int64)&v328, (__int64)v335, 2);
                    v222 = *(_QWORD *)(v164 + 48);
                    if ( v222 )
                      sub_161E7C0(v164 + 48, v222);
                    v223 = (unsigned __int8 *)v328;
                    *(_QWORD *)(v164 + 48) = v328;
                    if ( v223 )
                      sub_1623210((__int64)&v328, v223, v164 + 48);
                  }
                }
                else
                {
                  v164 = sub_15A2B30((__int64 *)v164, v213, 0, 0, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
                }
              }
              v331 = 257;
              v202 = (__int64 **)*(v77 - 3);
              if ( v202 != *(__int64 ***)v164 )
              {
                if ( *(_BYTE *)(v164 + 16) > 0x10u )
                {
                  v334 = 257;
                  v164 = sub_15FDBD0(46, v164, (__int64)v202, (__int64)&v332, 0);
                  if ( v336 )
                  {
                    v231 = v337;
                    sub_157E9D0(v336 + 40, v164);
                    v232 = *(_QWORD *)(v164 + 24);
                    v233 = *v231;
                    *(_QWORD *)(v164 + 32) = v231;
                    v233 &= 0xFFFFFFFFFFFFFFF8LL;
                    *(_QWORD *)(v164 + 24) = v233 | v232 & 7;
                    *(_QWORD *)(v233 + 8) = v164 + 24;
                    *v231 = *v231 & 7 | (v164 + 24);
                  }
                  sub_164B780(v164, v330);
                  if ( v335 )
                  {
                    v328 = (__int64)v335;
                    sub_1623A60((__int64)&v328, (__int64)v335, 2);
                    v234 = *(_QWORD *)(v164 + 48);
                    if ( v234 )
                      sub_161E7C0(v164 + 48, v234);
                    v235 = (unsigned __int8 *)v328;
                    *(_QWORD *)(v164 + 48) = v328;
                    if ( v235 )
                      sub_1623210((__int64)&v328, v235, v164 + 48);
                  }
                }
                else
                {
                  v164 = sub_15A46C0(46, (__int64 ***)v164, v202, 0);
                }
              }
              sub_164D160((__int64)v30, v164, (__m128)a3, *(double *)a4.m128i_i64, a5, a6, v168, v169, a9, a10);
              sub_15F20C0(v30);
              if ( v335 )
                sub_161E7C0((__int64)&v335, (__int64)v335);
              v253 = v316;
              goto LABEL_76;
            }
            v271 = v77;
            v323 = v166;
            v275 = v30;
            v170 = 2;
            while ( 2 )
            {
              v171 = v167 & 0xFFFFFFFFFFFFFFF8LL;
              v172 = v167 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (v167 & 4) == 0 )
                goto LABEL_272;
              v173 = v275[3 * ((unsigned int)(v170 - 1) - (unsigned __int64)(*((_DWORD *)v271 - 1) & 0xFFFFFFF))];
              if ( *(_BYTE *)(v173 + 16) == 13 )
              {
                if ( *(_DWORD *)(v173 + 32) <= 0x40u )
                {
                  if ( *(_QWORD *)(v173 + 24) )
                    goto LABEL_233;
                }
                else
                {
                  v295 = *(_DWORD *)(v173 + 32);
                  if ( v295 != (unsigned int)sub_16A57B0(v173 + 24) )
                    goto LABEL_233;
                }
              }
              else
              {
LABEL_233:
                v174 = v171;
                v175 = *(_QWORD *)(a1 + 160);
                if ( !v171 )
                {
                  v296 = *(_QWORD *)(a1 + 160);
                  v176 = sub_1643D30(0, *v323);
                  v175 = v296;
                  v174 = v176;
                }
                v297 = v175;
                v177 = sub_15A9FE0(v175, v174);
                v178 = v297;
                v179 = 1;
                v180 = v177;
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v174 + 8) )
                  {
                    case 1:
                      v181 = 16;
                      goto LABEL_238;
                    case 2:
                      v181 = 32;
                      goto LABEL_238;
                    case 3:
                    case 9:
                      v181 = 64;
                      goto LABEL_238;
                    case 4:
                      v181 = 80;
                      goto LABEL_238;
                    case 5:
                    case 6:
                      v181 = 128;
                      goto LABEL_238;
                    case 7:
                      v283 = v180;
                      v186 = 0;
                      v298 = v179;
                      goto LABEL_262;
                    case 0xB:
                      v181 = *(_DWORD *)(v174 + 8) >> 8;
                      goto LABEL_238;
                    case 0xD:
                      v192 = v297;
                      v285 = v180;
                      v300 = v179;
                      v193 = (_QWORD *)sub_15A9930(v192, v174);
                      v179 = v300;
                      v180 = v285;
                      v181 = 8LL * *v193;
                      goto LABEL_238;
                    case 0xE:
                      v244 = v180;
                      v248 = v179;
                      v254 = *(_QWORD *)(v174 + 24);
                      v259 = v297;
                      v284 = *(_QWORD *)(v174 + 32);
                      v189 = sub_15A9FE0(v297, v254);
                      v190 = v297;
                      v299 = 1;
                      v180 = v244;
                      v191 = v254;
                      v265 = v189;
                      v179 = v248;
                      while ( 2 )
                      {
                        switch ( *(_BYTE *)(v191 + 8) )
                        {
                          case 1:
                            v224 = 16;
                            goto LABEL_339;
                          case 2:
                            v224 = 32;
                            goto LABEL_339;
                          case 3:
                          case 9:
                            v224 = 64;
                            goto LABEL_339;
                          case 4:
                            v224 = 80;
                            goto LABEL_339;
                          case 5:
                          case 6:
                            v224 = 128;
                            goto LABEL_339;
                          case 7:
                            v255 = v244;
                            v225 = 0;
                            v260 = v248;
                            goto LABEL_346;
                          case 0xB:
                            v224 = *(_DWORD *)(v191 + 8) >> 8;
                            goto LABEL_339;
                          case 0xD:
                            v230 = (_QWORD *)sub_15A9930(v259, v191);
                            v179 = v248;
                            v180 = v244;
                            v224 = 8LL * *v230;
                            goto LABEL_339;
                          case 0xE:
                            v228 = v259;
                            v261 = *(_QWORD *)(v191 + 32);
                            v229 = sub_12BE0A0(v228, *(_QWORD *)(v191 + 24));
                            v179 = v248;
                            v180 = v244;
                            v224 = 8 * v261 * v229;
                            goto LABEL_339;
                          case 0xF:
                            v255 = v244;
                            v260 = v248;
                            v225 = *(_DWORD *)(v191 + 8) >> 8;
LABEL_346:
                            v226 = sub_15A9520(v190, v225);
                            v179 = v260;
                            v180 = v255;
                            v224 = (unsigned int)(8 * v226);
LABEL_339:
                            v181 = 8 * v265 * v284 * ((v265 + ((unsigned __int64)(v299 * v224 + 7) >> 3) - 1) / v265);
                            goto LABEL_238;
                          case 0x10:
                            v227 = v299 * *(_QWORD *)(v191 + 32);
                            v191 = *(_QWORD *)(v191 + 24);
                            v299 = v227;
                            continue;
                          default:
                            goto LABEL_377;
                        }
                      }
                    case 0xF:
                      v283 = v180;
                      v298 = v179;
                      v186 = *(_DWORD *)(v174 + 8) >> 8;
LABEL_262:
                      v187 = sub_15A9520(v178, v186);
                      v179 = v298;
                      v180 = v283;
                      v181 = (unsigned int)(8 * v187);
LABEL_238:
                      v182 = v181 * v179;
                      v183 = *(_DWORD *)(v306 + 8) >> 8;
                      v329 = v183;
                      if ( v183 <= 0x40 )
                      {
                        v328 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v183)
                             & (v180 * ((v180 + ((unsigned __int64)(v182 + 7) >> 3) - 1) / v180));
LABEL_240:
                        if ( v328 == 1 )
                          goto LABEL_246;
                        if ( !v328 || (v328 & (v328 - 1)) != 0 )
                          goto LABEL_243;
                        _BitScanReverse64(&v194, v328);
                        v331 = 257;
                        v195 = v183 + (v194 ^ 0x3F) - 64;
                        goto LABEL_269;
                      }
                      sub_16A4EF0((__int64)&v328, v180 * ((v180 + ((unsigned __int64)(v182 + 7) >> 3) - 1) / v180), 0);
                      v183 = v329;
                      if ( v329 <= 0x40 )
                        goto LABEL_240;
                      v286 = v329;
                      if ( v286 - (unsigned int)sub_16A57B0((__int64)&v328) <= 0x40 && *(_QWORD *)v328 == 1 )
                        goto LABEL_246;
                      if ( (unsigned int)sub_16A5940((__int64)&v328) != 1 )
                      {
LABEL_243:
                        v331 = 257;
                        v184 = sub_15A1070(v306, (__int64)&v328);
                        if ( *(_BYTE *)(v173 + 16) <= 0x10u && *(_BYTE *)(v184 + 16) <= 0x10u )
                        {
                          v173 = sub_15A2C20(
                                   (__int64 *)v173,
                                   v184,
                                   0,
                                   0,
                                   *(double *)a3.m128i_i64,
                                   *(double *)a4.m128i_i64,
                                   a5);
                          goto LABEL_246;
                        }
                        v207 = (__int64 *)v173;
                        v206 = 15;
                        v334 = 257;
                        v205 = v184;
                        goto LABEL_307;
                      }
                      v331 = 257;
                      v195 = sub_16A57B0((__int64)&v328);
                      v183 = v286;
LABEL_269:
                      v196 = sub_15A0680(v306, v183 - 1 - v195, 0);
                      if ( *(_BYTE *)(v173 + 16) <= 0x10u && *(_BYTE *)(v196 + 16) <= 0x10u )
                      {
                        v173 = sub_15A2D50(
                                 (__int64 *)v173,
                                 v196,
                                 0,
                                 0,
                                 *(double *)a3.m128i_i64,
                                 *(double *)a4.m128i_i64,
                                 a5);
                        goto LABEL_246;
                      }
                      v205 = v196;
                      v206 = 23;
                      v334 = 257;
                      v207 = (__int64 *)v173;
LABEL_307:
                      v173 = sub_15FB440(v206, v207, v205, (__int64)&v332, 0);
                      if ( v336 )
                      {
                        v301 = v337;
                        sub_157E9D0(v336 + 40, v173);
                        v208 = *v301;
                        v209 = *(_QWORD *)(v173 + 24) & 7LL;
                        *(_QWORD *)(v173 + 32) = v301;
                        v208 &= 0xFFFFFFFFFFFFFFF8LL;
                        *(_QWORD *)(v173 + 24) = v208 | v209;
                        *(_QWORD *)(v208 + 8) = v173 + 24;
                        *v301 = *v301 & 7 | (v173 + 24);
                      }
                      sub_164B780(v173, v330);
                      if ( v335 )
                      {
                        v327 = (__int64)v335;
                        sub_1623A60((__int64)&v327, (__int64)v335, 2);
                        v210 = *(_QWORD *)(v173 + 48);
                        v211 = v173 + 48;
                        if ( v210 )
                        {
                          sub_161E7C0(v173 + 48, v210);
                          v211 = v173 + 48;
                        }
                        v212 = (unsigned __int8 *)v327;
                        *(_QWORD *)(v173 + 48) = v327;
                        if ( v212 )
                          sub_1623210((__int64)&v327, v212, v211);
                      }
LABEL_246:
                      v331 = 257;
                      if ( *(_BYTE *)(v164 + 16) > 0x10u || *(_BYTE *)(v173 + 16) > 0x10u )
                      {
                        v334 = 257;
                        v164 = sub_15FB440(11, (__int64 *)v164, v173, (__int64)&v332, 0);
                        if ( v336 )
                        {
                          v197 = v337;
                          sub_157E9D0(v336 + 40, v164);
                          v198 = *(_QWORD *)(v164 + 24);
                          v199 = *v197;
                          *(_QWORD *)(v164 + 32) = v197;
                          v199 &= 0xFFFFFFFFFFFFFFF8LL;
                          *(_QWORD *)(v164 + 24) = v199 | v198 & 7;
                          *(_QWORD *)(v199 + 8) = v164 + 24;
                          *v197 = *v197 & 7 | (v164 + 24);
                        }
                        sub_164B780(v164, v330);
                        if ( v335 )
                        {
                          v327 = (__int64)v335;
                          sub_1623A60((__int64)&v327, (__int64)v335, 2);
                          v200 = *(_QWORD *)(v164 + 48);
                          if ( v200 )
                            sub_161E7C0(v164 + 48, v200);
                          v201 = (unsigned __int8 *)v327;
                          *(_QWORD *)(v164 + 48) = v327;
                          if ( v201 )
                            sub_1623210((__int64)&v327, v201, v164 + 48);
                        }
                      }
                      else
                      {
                        v164 = sub_15A2B30(
                                 (__int64 *)v164,
                                 v173,
                                 0,
                                 0,
                                 *(double *)a3.m128i_i64,
                                 *(double *)a4.m128i_i64,
                                 a5);
                      }
                      if ( v329 > 0x40 && v328 )
                        j_j___libc_free_0_0(v328);
                      break;
                    case 0x10:
                      v188 = *(_QWORD *)(v174 + 32);
                      v174 = *(_QWORD *)(v174 + 24);
                      v179 *= v188;
                      continue;
                    default:
                      goto LABEL_377;
                  }
                  break;
                }
              }
              if ( v171 )
              {
                v185 = *(_BYTE *)(v172 + 8);
                if ( ((v185 - 14) & 0xFD) == 0 )
                  goto LABEL_254;
LABEL_273:
                v29 = v185 == 13;
                v167 = 0;
                if ( v29 )
                  v167 = v172;
              }
              else
              {
LABEL_272:
                v172 = sub_1643D30(v171, *v323);
                v185 = *(_BYTE *)(v172 + 8);
                if ( ((v185 - 14) & 0xFD) != 0 )
                  goto LABEL_273;
LABEL_254:
                v167 = *(_QWORD *)(v172 + 24) | 4LL;
              }
              v323 += 3;
              if ( v313 == v170 )
              {
                v30 = v275;
                v77 = v271;
                goto LABEL_290;
              }
              ++v170;
              continue;
            }
          }
        }
LABEL_75:
        v253 |= v257;
LABEL_76:
        if ( v256 + 2 == v326 )
          goto LABEL_77;
        goto LABEL_23;
      }
      ++v48;
    }
  }
  return v325;
}
