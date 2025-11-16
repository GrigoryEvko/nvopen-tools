// Function: sub_1C47810
// Address: 0x1c47810
//
_BOOL8 __fastcall sub_1C47810(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // r15
  __int64 v3; // rdx
  __int64 v4; // rdi
  __int64 v5; // rbx
  __int64 v6; // rdi
  __int64 v7; // rbx
  __int64 v8; // rdi
  _QWORD *v9; // rbx
  unsigned __int64 v10; // rsi
  _QWORD *v11; // rax
  __int64 v12; // r8
  __int64 v13; // rcx
  __int64 v14; // rdx
  _QWORD *v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rcx
  __int64 v18; // rdx
  _QWORD *v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rcx
  __int64 v22; // rdx
  _QWORD *v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rcx
  __int64 v26; // rdx
  _QWORD *v27; // r13
  __int64 v28; // rsi
  bool v29; // zf
  unsigned __int64 v30; // r12
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rcx
  char v34; // al
  __int64 v35; // rsi
  __int64 v36; // rcx
  __int64 v37; // rax
  __int64 v38; // rdi
  char v39; // al
  __int64 v40; // rsi
  __int64 v41; // rcx
  __int64 v42; // rax
  __int64 v43; // rdi
  char v44; // al
  __int64 v45; // rsi
  __int64 v46; // rcx
  __int64 v47; // rax
  __int64 v48; // rdi
  unsigned __int64 v49; // rax
  __int64 v50; // rbx
  unsigned int v51; // r12d
  __int64 v52; // rsi
  _QWORD *v53; // rax
  _QWORD *v54; // rcx
  _QWORD *v55; // rax
  char v56; // r15
  __int64 v57; // rsi
  _QWORD *v58; // rdx
  _QWORD *v59; // rax
  __int64 v60; // rsi
  _QWORD *v61; // rdx
  _QWORD *v62; // rax
  char v63; // r15
  __int64 v64; // rsi
  _QWORD *v65; // rdx
  int v66; // r15d
  __int64 v67; // rbx
  unsigned __int64 v68; // r12
  __int64 v69; // rax
  _QWORD *v70; // rax
  __int64 v71; // r12
  __int64 v72; // rsi
  __int64 v73; // rcx
  _QWORD *v74; // rax
  __int64 v75; // rsi
  __int64 v76; // rdi
  __int64 v77; // rcx
  _QWORD *v78; // rax
  __int64 v79; // r12
  __int64 v80; // rsi
  __int64 v81; // rcx
  _QWORD *v82; // rax
  __int64 v83; // rsi
  __int64 v84; // rdi
  __int64 v85; // rcx
  __int64 v86; // rsi
  __int64 v87; // rcx
  __int64 v88; // rax
  __int64 v89; // rdi
  _QWORD *v90; // rbx
  int v91; // r13d
  __int64 v92; // rax
  __int64 v93; // rdi
  __int64 v94; // rdi
  __int64 v95; // rdi
  __int64 v96; // rdi
  __int64 v97; // rdi
  __int64 v98; // rdi
  __int64 v99; // rdi
  __int64 v100; // rdi
  __int64 v101; // rdi
  __int64 v102; // rdi
  __int64 v103; // rdi
  __int64 v104; // rdi
  __int64 v105; // rdi
  _QWORD *v107; // rax
  __int64 v108; // rsi
  _QWORD *v109; // rdx
  char v110; // bl
  _QWORD *v111; // rax
  __int64 v112; // rsi
  _QWORD *v113; // rdx
  _QWORD *v114; // rax
  __int64 v115; // rsi
  _QWORD *v116; // rdx
  char v117; // bl
  _QWORD *v118; // rax
  __int64 v119; // rsi
  _QWORD *v120; // rdx
  _QWORD *v121; // rax
  __int64 v122; // r8
  __int64 v123; // rcx
  __int64 v124; // rdx
  _QWORD *v125; // rax
  __int64 v126; // rsi
  __int64 v127; // rdi
  __int64 v128; // rcx
  _QWORD *v129; // rax
  __int64 v130; // rsi
  __int64 v131; // rdi
  __int64 v132; // rcx
  _QWORD *v133; // rax
  __int64 v134; // rsi
  __int64 v135; // rcx
  __int64 v136; // rdx
  _QWORD *v137; // rax
  __int64 v138; // rsi
  __int64 v139; // rcx
  __int64 v140; // rdx
  __int64 v141; // rsi
  _QWORD *v142; // rdi
  _QWORD *v143; // rax
  __int64 v144; // rcx
  __int64 v145; // rdx
  _QWORD *v146; // rax
  char v147; // r13
  unsigned __int64 v148; // rdx
  _QWORD *v149; // rsi
  __int64 v150; // rdi
  __int64 v151; // rcx
  __int64 v152; // rax
  char v153; // r12
  __int64 v154; // r15
  __int64 v155; // rdi
  __int64 v156; // rax
  _QWORD *v157; // rax
  __int64 v158; // rsi
  __int64 v159; // rcx
  __int64 v160; // rdx
  _QWORD *v161; // rax
  char v162; // r13
  unsigned __int64 v163; // rdx
  _QWORD *v164; // rsi
  __int64 v165; // rdi
  __int64 v166; // rcx
  __int64 v167; // rax
  char v168; // bl
  _QWORD *v169; // r15
  unsigned __int64 v170; // rcx
  unsigned __int64 v171; // r14
  __int64 v172; // rdi
  __int64 v173; // rax
  _QWORD *v174; // r13
  _QWORD *v175; // r12
  __int64 v176; // rax
  bool v177; // r14
  _QWORD *v178; // rax
  unsigned __int64 v179; // rdx
  _QWORD *v180; // r8
  _QWORD *v181; // rsi
  _QWORD *v182; // rcx
  _QWORD *v183; // rax
  __int64 v184; // r8
  __int64 v185; // rsi
  __int64 v186; // rcx
  __int64 v187; // rax
  __int64 v188; // rax
  const char *v189; // rcx
  size_t v190; // rdx
  size_t v191; // r12
  __int64 v192; // rbx
  _BYTE *v193; // rax
  unsigned __int64 *v194; // rax
  _QWORD *v195; // rax
  __int64 v196; // rsi
  __int64 v197; // rcx
  __int64 v198; // rdx
  _QWORD *v199; // rax
  __int64 v200; // rsi
  __int64 v201; // rcx
  __int64 v202; // rdx
  __int64 v203; // r8
  _QWORD *v204; // rcx
  __int64 v205; // rsi
  _QWORD *v206; // rax
  unsigned __int64 v207; // rdx
  _QWORD *v208; // r9
  __int64 v209; // rdi
  __int64 v210; // rcx
  __int64 v211; // rsi
  _QWORD *v212; // rax
  __int64 v213; // rcx
  __int64 v214; // rdx
  __int64 v215; // r8
  __int64 v216; // rax
  __int64 v217; // rcx
  __int64 v218; // r8
  _QWORD *v219; // rax
  __int64 v220; // rcx
  __int64 v221; // rdx
  __int64 v222; // rsi
  _QWORD *v223; // rax
  __int64 v224; // rdi
  __int64 v225; // rcx
  __int64 v226; // rax
  _QWORD *v227; // rax
  __int64 v228; // rsi
  __int64 v229; // rcx
  __int64 v230; // rdx
  _QWORD *v231; // rax
  __int64 v232; // rsi
  __int64 v233; // rcx
  __int64 v234; // rdx
  __int64 v235; // rdi
  __m128i *v236; // rax
  __m128i si128; // xmm0
  __int64 v238; // r13
  void *v239; // rax
  _QWORD *v240; // rax
  __int64 v241; // rsi
  __int64 v242; // rdi
  __int64 v243; // rcx
  __int64 v244; // r12
  void *v245; // rax
  _QWORD *v246; // rdx
  __int64 v247; // rax
  __int64 v248; // rdi
  __int64 v249; // rsi
  _QWORD *v250; // r13
  __int64 v251; // r12
  void *v252; // rax
  _QWORD *v253; // rax
  __int64 v254; // rsi
  __int64 v255; // rdi
  __int64 v256; // rcx
  __int64 v257; // r12
  void *v258; // rax
  _QWORD *v259; // rax
  __int64 v260; // rsi
  __int64 v261; // rdi
  __int64 v262; // rcx
  _QWORD *v263; // rbx
  __int64 v264; // r12
  void *v265; // rax
  char *v266; // rdi
  char *v267; // rax
  size_t v268; // rdx
  char *v269; // rsi
  size_t v270; // rdx
  char *v271; // rsi
  __int64 v272; // rdi
  _BYTE *v273; // rax
  __int64 v274; // rdi
  _BYTE *v275; // rax
  __int64 v276; // rax
  __int64 v277; // rax
  unsigned __int64 *v278; // rax
  unsigned __int64 *v279; // rdi
  _QWORD *v281; // [rsp+8h] [rbp-108h]
  _QWORD *v282; // [rsp+10h] [rbp-100h]
  bool v283; // [rsp+1Dh] [rbp-F3h]
  char v284; // [rsp+1Eh] [rbp-F2h]
  char v285; // [rsp+1Fh] [rbp-F1h]
  char v286; // [rsp+20h] [rbp-F0h]
  _QWORD *v287; // [rsp+20h] [rbp-F0h]
  char v288; // [rsp+28h] [rbp-E8h]
  int v289; // [rsp+30h] [rbp-E0h]
  _QWORD *v290; // [rsp+30h] [rbp-E0h]
  _QWORD *v291; // [rsp+30h] [rbp-E0h]
  char v292; // [rsp+38h] [rbp-D8h]
  _QWORD *v293; // [rsp+38h] [rbp-D8h]
  _QWORD *v294; // [rsp+38h] [rbp-D8h]
  unsigned __int64 v295; // [rsp+40h] [rbp-D0h]
  _QWORD *v296; // [rsp+40h] [rbp-D0h]
  _QWORD *v297; // [rsp+40h] [rbp-D0h]
  _QWORD *src; // [rsp+48h] [rbp-C8h]
  _QWORD *srca; // [rsp+48h] [rbp-C8h]
  void *srcb; // [rsp+48h] [rbp-C8h]
  const char *srcd; // [rsp+48h] [rbp-C8h]
  char *srcc; // [rsp+48h] [rbp-C8h]
  const char *srce; // [rsp+48h] [rbp-C8h]
  _QWORD *v304; // [rsp+50h] [rbp-C0h]
  unsigned int v305; // [rsp+50h] [rbp-C0h]
  _QWORD *v306; // [rsp+50h] [rbp-C0h]
  _QWORD *v307; // [rsp+58h] [rbp-B8h]
  _QWORD *v308; // [rsp+60h] [rbp-B0h]
  unsigned __int64 v309; // [rsp+68h] [rbp-A8h]
  __int64 v310; // [rsp+68h] [rbp-A8h]
  _QWORD *v311; // [rsp+68h] [rbp-A8h]
  _QWORD *v312; // [rsp+68h] [rbp-A8h]
  _QWORD *v313; // [rsp+70h] [rbp-A0h]
  _QWORD *v314; // [rsp+78h] [rbp-98h]
  __int64 v315; // [rsp+80h] [rbp-90h]
  __int64 v316; // [rsp+88h] [rbp-88h]
  __int64 v317; // [rsp+90h] [rbp-80h]
  __int64 v318; // [rsp+98h] [rbp-78h]
  unsigned __int64 v319; // [rsp+A0h] [rbp-70h] BYREF
  _QWORD *v320; // [rsp+A8h] [rbp-68h] BYREF
  unsigned __int64 v321; // [rsp+B0h] [rbp-60h] BYREF
  unsigned __int64 v322; // [rsp+B8h] [rbp-58h] BYREF
  unsigned __int64 *v323; // [rsp+C0h] [rbp-50h] BYREF
  size_t v324; // [rsp+C8h] [rbp-48h]
  _QWORD v325[8]; // [rsp+D0h] [rbp-40h] BYREF

  v2 = a1;
  v283 = 0;
  a1[1] = sub_1649960(a2);
  v314 = a1 + 39;
  v313 = a1 + 45;
  v308 = a1 + 51;
  v307 = a1 + 57;
  v282 = (_QWORD *)(a2 + 72);
  v316 = (__int64)(a1 + 58);
  v315 = (__int64)(a1 + 52);
  v317 = (__int64)(a1 + 46);
  v318 = (__int64)(a1 + 40);
  a1[2] = v3;
  v281 = a1 + 70;
LABEL_2:
  sub_1C46330(v2, a2, 1);
  sub_1C46620(v2, a2);
  sub_1C46330(v2, a2, 0);
  sub_1C45C70(v2[41]);
  v4 = v2[47];
  v2[41] = 0;
  v2[44] = 0;
  v2[42] = v318;
  v2[43] = v318;
  sub_1C45C70(v4);
  v5 = v2[53];
  v2[47] = 0;
  v2[50] = 0;
  v2[48] = v317;
  v2[49] = v317;
  while ( v5 )
  {
    sub_1C45C70(*(_QWORD *)(v5 + 24));
    v6 = v5;
    v5 = *(_QWORD *)(v5 + 16);
    j_j___libc_free_0(v6, 48);
  }
  v7 = v2[59];
  v2[53] = 0;
  v2[56] = 0;
  v2[54] = v315;
  v2[55] = v315;
  while ( v7 )
  {
    sub_1C45C70(*(_QWORD *)(v7 + 24));
    v8 = v7;
    v7 = *(_QWORD *)(v7 + 16);
    j_j___libc_free_0(v8, 48);
  }
  v2[59] = 0;
  v2[62] = 0;
  v2[60] = v316;
  v2[61] = v316;
  v9 = *(_QWORD **)(a2 + 80);
  if ( v9 == v282 )
    goto LABEL_188;
  do
  {
    v10 = (unsigned __int64)(v9 - 3);
    v11 = (_QWORD *)v2[41];
    v12 = v318;
    if ( !v9 )
      v10 = 0;
    v322 = v10;
    if ( !v11 )
      goto LABEL_16;
    do
    {
      while ( 1 )
      {
        v13 = v11[2];
        v14 = v11[3];
        if ( v11[4] >= v10 )
          break;
        v11 = (_QWORD *)v11[3];
        if ( !v14 )
          goto LABEL_14;
      }
      v12 = (__int64)v11;
      v11 = (_QWORD *)v11[2];
    }
    while ( v13 );
LABEL_14:
    if ( v12 == v318 || *(_QWORD *)(v12 + 32) > v10 )
    {
LABEL_16:
      v323 = &v322;
      v12 = sub_1C46280(v314, (_QWORD *)v12, &v323);
    }
    *(_BYTE *)(v12 + 40) = 0;
    v15 = (_QWORD *)v2[47];
    if ( !v15 )
    {
      v16 = v317;
LABEL_24:
      v323 = &v322;
      v16 = sub_1C46280(v313, (_QWORD *)v16, &v323);
      goto LABEL_25;
    }
    v16 = v317;
    do
    {
      while ( 1 )
      {
        v17 = v15[2];
        v18 = v15[3];
        if ( v15[4] >= v322 )
          break;
        v15 = (_QWORD *)v15[3];
        if ( !v18 )
          goto LABEL_22;
      }
      v16 = (__int64)v15;
      v15 = (_QWORD *)v15[2];
    }
    while ( v17 );
LABEL_22:
    if ( v16 == v317 || *(_QWORD *)(v16 + 32) > v322 )
      goto LABEL_24;
LABEL_25:
    *(_BYTE *)(v16 + 40) = 0;
    v19 = (_QWORD *)v2[53];
    if ( !v19 )
    {
      v20 = v315;
LABEL_32:
      v323 = &v322;
      v20 = sub_1C46280(v308, (_QWORD *)v20, &v323);
      goto LABEL_33;
    }
    v20 = v315;
    do
    {
      while ( 1 )
      {
        v21 = v19[2];
        v22 = v19[3];
        if ( v19[4] >= v322 )
          break;
        v19 = (_QWORD *)v19[3];
        if ( !v22 )
          goto LABEL_30;
      }
      v20 = (__int64)v19;
      v19 = (_QWORD *)v19[2];
    }
    while ( v21 );
LABEL_30:
    if ( v20 == v315 || *(_QWORD *)(v20 + 32) > v322 )
      goto LABEL_32;
LABEL_33:
    *(_BYTE *)(v20 + 40) = 0;
    v23 = (_QWORD *)v2[59];
    if ( !v23 )
    {
      v24 = v316;
LABEL_40:
      v323 = &v322;
      v24 = sub_1C46280(v307, (_QWORD *)v24, &v323);
      goto LABEL_41;
    }
    v24 = v316;
    do
    {
      while ( 1 )
      {
        v25 = v23[2];
        v26 = v23[3];
        if ( v23[4] >= v322 )
          break;
        v23 = (_QWORD *)v23[3];
        if ( !v26 )
          goto LABEL_38;
      }
      v24 = (__int64)v23;
      v23 = (_QWORD *)v23[2];
    }
    while ( v25 );
LABEL_38:
    if ( v24 == v316 || *(_QWORD *)(v24 + 32) > v322 )
      goto LABEL_40;
LABEL_41:
    *(_BYTE *)(v24 + 40) = 0;
    v9 = (_QWORD *)v9[1];
  }
  while ( v9 != v282 );
  v27 = v2;
  v304 = *(_QWORD **)(a2 + 80);
  while ( v304 != v282 )
  {
    v286 = 0;
    src = v282;
    do
    {
      v28 = v27[41];
      v29 = (*src & 0xFFFFFFFFFFFFFFF8LL) == 0;
      v30 = (*src & 0xFFFFFFFFFFFFFFF8LL) - 24;
      v295 = *src & 0xFFFFFFFFFFFFFFF8LL;
      src = (_QWORD *)v295;
      if ( v29 )
        v30 = 0;
      v321 = v30;
      if ( !v28 )
      {
        v28 = v318;
LABEL_55:
        v323 = &v321;
        v28 = sub_1C46280(v314, (_QWORD *)v28, &v323);
        goto LABEL_56;
      }
      v31 = v318;
      while ( 1 )
      {
        v32 = *(_QWORD *)(v28 + 16);
        v33 = *(_QWORD *)(v28 + 24);
        if ( *(_QWORD *)(v28 + 32) < v30 )
        {
          v28 = v31;
          v32 = v33;
        }
        if ( !v32 )
          break;
        v31 = v28;
        v28 = v32;
      }
      if ( v28 == v318 || *(_QWORD *)(v28 + 32) > v30 )
        goto LABEL_55;
LABEL_56:
      v34 = *(_BYTE *)(v28 + 40);
      v35 = v27[47];
      v292 = v34;
      if ( !v35 )
      {
        v35 = v317;
LABEL_64:
        v323 = &v321;
        v35 = sub_1C46280(v313, (_QWORD *)v35, &v323);
        goto LABEL_65;
      }
      v36 = v317;
      while ( 1 )
      {
        v37 = *(_QWORD *)(v35 + 16);
        v38 = *(_QWORD *)(v35 + 24);
        if ( *(_QWORD *)(v35 + 32) < v321 )
        {
          v35 = v36;
          v37 = v38;
        }
        if ( !v37 )
          break;
        v36 = v35;
        v35 = v37;
      }
      if ( v35 == v317 || *(_QWORD *)(v35 + 32) > v321 )
        goto LABEL_64;
LABEL_65:
      v39 = *(_BYTE *)(v35 + 40);
      v40 = v27[53];
      v288 = v39;
      if ( !v40 )
      {
        v40 = v315;
LABEL_73:
        v323 = &v321;
        v40 = sub_1C46280(v308, (_QWORD *)v40, &v323);
        goto LABEL_74;
      }
      v41 = v315;
      while ( 1 )
      {
        v42 = *(_QWORD *)(v40 + 16);
        v43 = *(_QWORD *)(v40 + 24);
        if ( *(_QWORD *)(v40 + 32) < v321 )
        {
          v40 = v41;
          v42 = v43;
        }
        if ( !v42 )
          break;
        v41 = v40;
        v40 = v42;
      }
      if ( v40 == v315 || *(_QWORD *)(v40 + 32) > v321 )
        goto LABEL_73;
LABEL_74:
      v44 = *(_BYTE *)(v40 + 40);
      v45 = v27[59];
      v285 = v44;
      if ( !v45 )
      {
        v45 = v316;
LABEL_82:
        v323 = &v321;
        v45 = sub_1C46280(v307, (_QWORD *)v45, &v323);
        goto LABEL_83;
      }
      v46 = v316;
      while ( 1 )
      {
        v47 = *(_QWORD *)(v45 + 16);
        v48 = *(_QWORD *)(v45 + 24);
        if ( *(_QWORD *)(v45 + 32) < v321 )
        {
          v45 = v46;
          v47 = v48;
        }
        if ( !v47 )
          break;
        v46 = v45;
        v45 = v47;
      }
      if ( v45 == v316 || *(_QWORD *)(v45 + 32) > v321 )
        goto LABEL_82;
LABEL_83:
      v284 = *(_BYTE *)(v45 + 40);
      v49 = sub_157EBA0(v30);
      v50 = v49;
      if ( v49 )
      {
        v289 = sub_15F4D60(v49);
        if ( v289 )
        {
          v51 = 0;
          while ( 1 )
          {
            v52 = v318;
            v322 = sub_15F4DF0(v50, v51);
            v53 = (_QWORD *)v27[41];
            if ( !v53 )
              goto LABEL_94;
            while ( 1 )
            {
              v54 = (_QWORD *)v53[3];
              if ( v53[4] >= v322 )
              {
                v54 = (_QWORD *)v53[2];
                v52 = (__int64)v53;
              }
              if ( !v54 )
                break;
              v53 = v54;
            }
            if ( v52 == v318 || *(_QWORD *)(v52 + 32) > v322 )
            {
LABEL_94:
              v323 = &v322;
              v52 = sub_1C46280(v314, (_QWORD *)v52, &v323);
            }
            v55 = (_QWORD *)v27[53];
            v56 = *(_BYTE *)(v52 + 40);
            if ( !v55 )
              break;
            v57 = v315;
            while ( 1 )
            {
              v58 = (_QWORD *)v55[3];
              if ( v55[4] >= v321 )
              {
                v58 = (_QWORD *)v55[2];
                v57 = (__int64)v55;
              }
              if ( !v58 )
                break;
              v55 = v58;
            }
            if ( v57 == v315 || *(_QWORD *)(v57 + 32) > v321 )
              goto LABEL_103;
LABEL_104:
            *(_BYTE *)(v57 + 40) |= v56;
            v59 = (_QWORD *)v27[47];
            if ( !v59 )
            {
              v60 = v317;
LABEL_112:
              v323 = &v322;
              v60 = sub_1C46280(v313, (_QWORD *)v60, &v323);
              goto LABEL_113;
            }
            v60 = v317;
            while ( 1 )
            {
              v61 = (_QWORD *)v59[3];
              if ( v59[4] >= v322 )
              {
                v61 = (_QWORD *)v59[2];
                v60 = (__int64)v59;
              }
              if ( !v61 )
                break;
              v59 = v61;
            }
            if ( v60 == v317 || *(_QWORD *)(v60 + 32) > v322 )
              goto LABEL_112;
LABEL_113:
            v62 = (_QWORD *)v27[59];
            v63 = *(_BYTE *)(v60 + 40);
            if ( v62 )
            {
              v64 = v316;
              while ( 1 )
              {
                v65 = (_QWORD *)v62[3];
                if ( v62[4] >= v321 )
                {
                  v65 = (_QWORD *)v62[2];
                  v64 = (__int64)v62;
                }
                if ( !v65 )
                  break;
                v62 = v65;
              }
              if ( v64 != v316 && *(_QWORD *)(v64 + 32) <= v321 )
                goto LABEL_122;
            }
            else
            {
              v64 = v316;
            }
            v323 = &v321;
            v64 = sub_1C46280(v307, (_QWORD *)v64, &v323);
LABEL_122:
            *(_BYTE *)(v64 + 40) |= v63;
            if ( v289 == ++v51 )
              goto LABEL_123;
          }
          v57 = v315;
LABEL_103:
          v323 = &v321;
          v57 = sub_1C46280(v308, (_QWORD *)v57, &v323);
          goto LABEL_104;
        }
      }
LABEL_123:
      v66 = 0;
      v67 = *(_QWORD *)(v321 + 48);
      v68 = v321 + 40;
      if ( v67 != v321 + 40 )
      {
        do
        {
          if ( !v67 )
            BUG();
          if ( *(_BYTE *)(v67 - 8) == 78 )
          {
            v69 = *(_QWORD *)(v67 - 48);
            if ( !*(_BYTE *)(v69 + 16) && (*(_BYTE *)(v69 + 33) & 0x20) != 0 )
              v66 -= !sub_1C301F0(*(_DWORD *)(v69 + 36)) - 1;
          }
          v67 = *(_QWORD *)(v67 + 8);
        }
        while ( v68 != v67 );
        if ( v66 )
        {
          v70 = (_QWORD *)v27[5];
          if ( !v70 )
          {
            v71 = (__int64)(v27 + 4);
            goto LABEL_138;
          }
          v71 = (__int64)(v27 + 4);
          do
          {
            while ( 1 )
            {
              v72 = v70[2];
              v73 = v70[3];
              if ( v70[4] >= v321 )
                break;
              v70 = (_QWORD *)v70[3];
              if ( !v73 )
                goto LABEL_136;
            }
            v71 = (__int64)v70;
            v70 = (_QWORD *)v70[2];
          }
          while ( v72 );
LABEL_136:
          if ( v27 + 4 == (_QWORD *)v71 || *(_QWORD *)(v71 + 32) > v321 )
          {
LABEL_138:
            v323 = &v321;
            v71 = sub_1C46280(v27 + 3, (_QWORD *)v71, &v323);
          }
          v74 = (_QWORD *)v27[41];
          if ( !v74 )
          {
            v75 = v318;
            goto LABEL_146;
          }
          v75 = v318;
          do
          {
            while ( 1 )
            {
              v76 = v74[2];
              v77 = v74[3];
              if ( v74[4] >= v321 )
                break;
              v74 = (_QWORD *)v74[3];
              if ( !v77 )
                goto LABEL_144;
            }
            v75 = (__int64)v74;
            v74 = (_QWORD *)v74[2];
          }
          while ( v76 );
LABEL_144:
          if ( v75 == v318 || *(_QWORD *)(v75 + 32) > v321 )
          {
LABEL_146:
            v323 = &v321;
            v75 = sub_1C46280(v314, (_QWORD *)v75, &v323);
          }
          *(_BYTE *)(v75 + 40) = *(_BYTE *)(v71 + 40);
          v78 = (_QWORD *)v27[11];
          if ( !v78 )
          {
            v79 = (__int64)(v27 + 10);
            goto LABEL_154;
          }
          v79 = (__int64)(v27 + 10);
          do
          {
            while ( 1 )
            {
              v80 = v78[2];
              v81 = v78[3];
              if ( v78[4] >= v321 )
                break;
              v78 = (_QWORD *)v78[3];
              if ( !v81 )
                goto LABEL_152;
            }
            v79 = (__int64)v78;
            v78 = (_QWORD *)v78[2];
          }
          while ( v80 );
LABEL_152:
          if ( v27 + 10 == (_QWORD *)v79 || *(_QWORD *)(v79 + 32) > v321 )
          {
LABEL_154:
            v323 = &v321;
            v79 = sub_1C46280(v27 + 9, (_QWORD *)v79, &v323);
          }
          v82 = (_QWORD *)v27[47];
          if ( v82 )
          {
            v83 = v317;
            do
            {
              while ( 1 )
              {
                v84 = v82[2];
                v85 = v82[3];
                if ( v82[4] >= v321 )
                  break;
                v82 = (_QWORD *)v82[3];
                if ( !v85 )
                  goto LABEL_160;
              }
              v83 = (__int64)v82;
              v82 = (_QWORD *)v82[2];
            }
            while ( v84 );
LABEL_160:
            if ( v83 == v317 || *(_QWORD *)(v83 + 32) > v321 )
            {
LABEL_162:
              v323 = &v321;
              v83 = sub_1C46280(v313, (_QWORD *)v83, &v323);
            }
            *(_BYTE *)(v83 + 40) = *(_BYTE *)(v79 + 40);
            v86 = v27[41];
            if ( !v86 )
              goto LABEL_231;
            goto LABEL_164;
          }
          v83 = v317;
          goto LABEL_162;
        }
      }
      v107 = (_QWORD *)v27[53];
      if ( !v107 )
      {
        v108 = v315;
LABEL_200:
        v323 = &v321;
        v108 = sub_1C46280(v308, (_QWORD *)v108, &v323);
        goto LABEL_201;
      }
      v108 = v315;
      while ( 1 )
      {
        v109 = (_QWORD *)v107[3];
        if ( v107[4] >= v321 )
        {
          v109 = (_QWORD *)v107[2];
          v108 = (__int64)v107;
        }
        if ( !v109 )
          break;
        v107 = v109;
      }
      if ( v108 == v315 || *(_QWORD *)(v108 + 32) > v321 )
        goto LABEL_200;
LABEL_201:
      v110 = *(_BYTE *)(v108 + 40);
      if ( v110 )
      {
        v111 = (_QWORD *)v27[41];
        if ( !v111 )
          goto LABEL_269;
        goto LABEL_203;
      }
      v133 = (_QWORD *)v27[5];
      if ( !v133 )
      {
        v134 = (__int64)(v27 + 4);
LABEL_267:
        v323 = &v321;
        v134 = sub_1C46280(v27 + 3, (_QWORD *)v134, &v323);
        goto LABEL_268;
      }
      v134 = (__int64)(v27 + 4);
      do
      {
        while ( 1 )
        {
          v135 = v133[2];
          v136 = v133[3];
          if ( v133[4] >= v321 )
            break;
          v133 = (_QWORD *)v133[3];
          if ( !v136 )
            goto LABEL_265;
        }
        v134 = (__int64)v133;
        v133 = (_QWORD *)v133[2];
      }
      while ( v135 );
LABEL_265:
      if ( (_QWORD *)v134 == v27 + 4 || *(_QWORD *)(v134 + 32) > v321 )
        goto LABEL_267;
LABEL_268:
      v111 = (_QWORD *)v27[41];
      v110 = *(_BYTE *)(v134 + 40);
      if ( !v111 )
      {
LABEL_269:
        v112 = v318;
LABEL_210:
        v323 = &v321;
        v112 = sub_1C46280(v314, (_QWORD *)v112, &v323);
        goto LABEL_211;
      }
LABEL_203:
      v112 = v318;
      while ( 1 )
      {
        v113 = (_QWORD *)v111[3];
        if ( v111[4] >= v321 )
        {
          v113 = (_QWORD *)v111[2];
          v112 = (__int64)v111;
        }
        if ( !v113 )
          break;
        v111 = v113;
      }
      if ( v112 == v318 || *(_QWORD *)(v112 + 32) > v321 )
        goto LABEL_210;
LABEL_211:
      *(_BYTE *)(v112 + 40) = v110;
      v114 = (_QWORD *)v27[59];
      if ( !v114 )
      {
        v115 = v316;
LABEL_219:
        v323 = &v321;
        v115 = sub_1C46280(v307, (_QWORD *)v115, &v323);
        goto LABEL_220;
      }
      v115 = v316;
      while ( 1 )
      {
        v116 = (_QWORD *)v114[3];
        if ( v114[4] >= v321 )
        {
          v116 = (_QWORD *)v114[2];
          v115 = (__int64)v114;
        }
        if ( !v116 )
          break;
        v114 = v116;
      }
      if ( v316 == v115 || *(_QWORD *)(v115 + 32) > v321 )
        goto LABEL_219;
LABEL_220:
      v117 = *(_BYTE *)(v115 + 40);
      if ( v117 )
      {
        v118 = (_QWORD *)v27[47];
        if ( !v118 )
          goto LABEL_279;
        goto LABEL_222;
      }
      v137 = (_QWORD *)v27[11];
      if ( !v137 )
      {
        v138 = (__int64)(v27 + 10);
LABEL_277:
        v323 = &v321;
        v138 = sub_1C46280(v27 + 9, (_QWORD *)v138, &v323);
        goto LABEL_278;
      }
      v138 = (__int64)(v27 + 10);
      do
      {
        while ( 1 )
        {
          v139 = v137[2];
          v140 = v137[3];
          if ( v137[4] >= v321 )
            break;
          v137 = (_QWORD *)v137[3];
          if ( !v140 )
            goto LABEL_275;
        }
        v138 = (__int64)v137;
        v137 = (_QWORD *)v137[2];
      }
      while ( v139 );
LABEL_275:
      if ( (_QWORD *)v138 == v27 + 10 || *(_QWORD *)(v138 + 32) > v321 )
        goto LABEL_277;
LABEL_278:
      v118 = (_QWORD *)v27[47];
      v117 = *(_BYTE *)(v138 + 40);
      if ( !v118 )
      {
LABEL_279:
        v119 = v317;
LABEL_229:
        v323 = &v321;
        v119 = sub_1C46280(v313, (_QWORD *)v119, &v323);
        goto LABEL_230;
      }
LABEL_222:
      v119 = v317;
      while ( 1 )
      {
        v120 = (_QWORD *)v118[3];
        if ( v118[4] >= v321 )
        {
          v120 = (_QWORD *)v118[2];
          v119 = (__int64)v118;
        }
        if ( !v120 )
          break;
        v118 = v120;
      }
      if ( v119 == v317 || *(_QWORD *)(v119 + 32) > v321 )
        goto LABEL_229;
LABEL_230:
      *(_BYTE *)(v119 + 40) = v117;
      v86 = v27[41];
      if ( !v86 )
      {
LABEL_231:
        v86 = v318;
LABEL_171:
        v323 = &v321;
        v86 = sub_1C46280(v314, (_QWORD *)v86, &v323);
        goto LABEL_172;
      }
LABEL_164:
      v87 = v318;
      while ( 1 )
      {
        v88 = *(_QWORD *)(v86 + 16);
        v89 = *(_QWORD *)(v86 + 24);
        if ( *(_QWORD *)(v86 + 32) < v321 )
        {
          v86 = v87;
          v88 = v89;
        }
        if ( !v88 )
          break;
        v87 = v86;
        v86 = v88;
      }
      if ( v86 == v318 || *(_QWORD *)(v86 + 32) > v321 )
        goto LABEL_171;
LABEL_172:
      if ( v292 != *(_BYTE *)(v86 + 40) )
        goto LABEL_173;
      v121 = (_QWORD *)v27[47];
      if ( !v121 )
      {
        v122 = v317;
LABEL_239:
        v323 = &v321;
        v122 = sub_1C46280(v313, (_QWORD *)v122, &v323);
        goto LABEL_240;
      }
      v122 = v317;
      do
      {
        while ( 1 )
        {
          v123 = v121[2];
          v124 = v121[3];
          if ( v121[4] >= v321 )
            break;
          v121 = (_QWORD *)v121[3];
          if ( !v124 )
            goto LABEL_237;
        }
        v122 = (__int64)v121;
        v121 = (_QWORD *)v121[2];
      }
      while ( v123 );
LABEL_237:
      if ( v122 == v317 || *(_QWORD *)(v122 + 32) > v321 )
        goto LABEL_239;
LABEL_240:
      if ( v288 != *(_BYTE *)(v122 + 40) )
        goto LABEL_173;
      v125 = (_QWORD *)v27[53];
      if ( !v125 )
      {
        v126 = v315;
LABEL_248:
        v323 = &v321;
        v126 = sub_1C46280(v308, (_QWORD *)v126, &v323);
        goto LABEL_249;
      }
      v126 = v315;
      do
      {
        while ( 1 )
        {
          v127 = v125[2];
          v128 = v125[3];
          if ( v125[4] >= v321 )
            break;
          v125 = (_QWORD *)v125[3];
          if ( !v128 )
            goto LABEL_246;
        }
        v126 = (__int64)v125;
        v125 = (_QWORD *)v125[2];
      }
      while ( v127 );
LABEL_246:
      if ( v315 == v126 || *(_QWORD *)(v126 + 32) > v321 )
        goto LABEL_248;
LABEL_249:
      if ( v285 != *(_BYTE *)(v126 + 40) )
        goto LABEL_173;
      v129 = (_QWORD *)v27[59];
      if ( !v129 )
      {
        v130 = v316;
LABEL_257:
        v323 = &v321;
        v130 = sub_1C46280(v307, (_QWORD *)v130, &v323);
        goto LABEL_258;
      }
      v130 = v316;
      do
      {
        while ( 1 )
        {
          v131 = v129[2];
          v132 = v129[3];
          if ( v129[4] >= v321 )
            break;
          v129 = (_QWORD *)v129[3];
          if ( !v132 )
            goto LABEL_255;
        }
        v130 = (__int64)v129;
        v129 = (_QWORD *)v129[2];
      }
      while ( v131 );
LABEL_255:
      if ( v316 == v130 || *(_QWORD *)(v130 + 32) > v321 )
        goto LABEL_257;
LABEL_258:
      if ( v284 != *(_BYTE *)(v130 + 40) )
LABEL_173:
        v286 = 1;
    }
    while ( (_QWORD *)v295 != v304 );
    v304 = *(_QWORD **)(a2 + 80);
    if ( !v286 )
    {
      v2 = v27;
      if ( v282 == *(_QWORD **)(a2 + 80) )
        goto LABEL_188;
      v287 = v27 + 16;
      while ( 2 )
      {
        if ( !v304 )
          BUG();
        v90 = (_QWORD *)v304[3];
        v91 = 0;
        if ( v90 == v304 + 2 )
          goto LABEL_187;
        do
        {
          if ( !v90 )
            BUG();
          if ( *((_BYTE *)v90 - 8) == 78 )
          {
            v92 = *(v90 - 6);
            if ( !*(_BYTE *)(v92 + 16) && (*(_BYTE *)(v92 + 33) & 0x20) != 0 )
              v91 -= !sub_1C301F0(*(_DWORD *)(v92 + 36)) - 1;
          }
          v90 = (_QWORD *)v90[1];
        }
        while ( v304 + 2 != v90 );
        if ( !v91 )
        {
LABEL_187:
          v304 = (_QWORD *)v304[1];
          if ( v304 == v282 )
            goto LABEL_188;
          continue;
        }
        break;
      }
      v141 = (__int64)v287;
      v142 = v304 - 3;
      v143 = (_QWORD *)v2[17];
      v321 = (unsigned __int64)(v304 - 3);
      if ( !v143 )
        goto LABEL_303;
      do
      {
        while ( 1 )
        {
          v144 = v143[2];
          v145 = v143[3];
          if ( v143[4] >= (unsigned __int64)v142 )
            break;
          v143 = (_QWORD *)v143[3];
          if ( !v145 )
            goto LABEL_301;
        }
        v141 = (__int64)v143;
        v143 = (_QWORD *)v143[2];
      }
      while ( v144 );
LABEL_301:
      if ( v287 == (_QWORD *)v141 || *(_QWORD *)(v141 + 32) > (unsigned __int64)v142 )
      {
LABEL_303:
        v323 = &v321;
        v141 = sub_1C46280(v2 + 15, (_QWORD *)v141, &v323);
      }
      v146 = (_QWORD *)v2[23];
      v147 = *(_BYTE *)(v141 + 40);
      if ( !v146 )
      {
        v149 = v2 + 22;
        goto LABEL_311;
      }
      v148 = v321;
      v149 = v2 + 22;
      do
      {
        while ( 1 )
        {
          v150 = v146[2];
          v151 = v146[3];
          if ( v146[4] >= v321 )
            break;
          v146 = (_QWORD *)v146[3];
          if ( !v151 )
            goto LABEL_309;
        }
        v149 = v146;
        v146 = (_QWORD *)v146[2];
      }
      while ( v150 );
LABEL_309:
      if ( v2 + 22 == v149 || v149[4] > v321 )
      {
LABEL_311:
        v323 = &v321;
        v152 = sub_1C46280(v2 + 21, v149, &v323);
        v148 = v321;
        v149 = (_QWORD *)v152;
      }
      v153 = *((_BYTE *)v149 + 40);
      v309 = v148 + 40;
      if ( *(_QWORD *)(v148 + 48) == v148 + 40 )
        goto LABEL_323;
      srca = v2;
      v154 = *(_QWORD *)(v148 + 48);
      while ( 2 )
      {
        if ( !v154 )
        {
          v322 = 0;
          BUG();
        }
        v155 = v154 - 24;
        v322 = v154 - 24;
        if ( *(_BYTE *)(v154 - 8) == 78
          && (v156 = *(_QWORD *)(v154 - 48), !*(_BYTE *)(v156 + 16))
          && (*(_BYTE *)(v156 + 33) & 0x20) != 0 )
        {
          if ( !sub_1C301F0(*(_DWORD *)(v156 + 36)) )
          {
            v155 = v322;
            goto LABEL_320;
          }
          v195 = (_QWORD *)srca[65];
          if ( !v195 )
          {
            v196 = (__int64)(srca + 64);
            goto LABEL_397;
          }
          v196 = (__int64)(srca + 64);
          do
          {
            while ( 1 )
            {
              v197 = v195[2];
              v198 = v195[3];
              if ( v195[4] >= v322 )
                break;
              v195 = (_QWORD *)v195[3];
              if ( !v198 )
                goto LABEL_395;
            }
            v196 = (__int64)v195;
            v195 = (_QWORD *)v195[2];
          }
          while ( v197 );
LABEL_395:
          if ( srca + 64 == (_QWORD *)v196 || *(_QWORD *)(v196 + 32) > v322 )
          {
LABEL_397:
            v323 = &v322;
            v196 = sub_1C47760(srca + 63, (_QWORD *)v196, &v323);
          }
          *(_BYTE *)(v196 + 40) = v147;
          v199 = (_QWORD *)srca[71];
          if ( !v199 )
          {
            v200 = (__int64)v281;
            goto LABEL_405;
          }
          v200 = (__int64)v281;
          do
          {
            while ( 1 )
            {
              v201 = v199[2];
              v202 = v199[3];
              if ( v199[4] >= v322 )
                break;
              v199 = (_QWORD *)v199[3];
              if ( !v202 )
                goto LABEL_403;
            }
            v200 = (__int64)v199;
            v199 = (_QWORD *)v199[2];
          }
          while ( v201 );
LABEL_403:
          if ( (_QWORD *)v200 == v281 || *(_QWORD *)(v200 + 32) > v322 )
          {
LABEL_405:
            v323 = &v322;
            v200 = sub_1C47760(srca + 69, (_QWORD *)v200, &v323);
          }
          *(_BYTE *)(v200 + 40) = v153;
          v147 = 0;
          v153 = 0;
        }
        else
        {
LABEL_320:
          LOBYTE(v320) = 0;
          LOBYTE(v323) = 0;
          sub_1C45690(v155, &v320, &v323);
          v147 |= (unsigned __int8)v320;
          v153 |= (unsigned __int8)v323;
        }
        v154 = *(_QWORD *)(v154 + 8);
        if ( v309 == v154 )
        {
          v2 = srca;
LABEL_323:
          v157 = (_QWORD *)v2[53];
          if ( !v157 )
          {
            v158 = v315;
            goto LABEL_330;
          }
          v158 = v315;
          do
          {
            while ( 1 )
            {
              v159 = v157[2];
              v160 = v157[3];
              if ( v157[4] >= v321 )
                break;
              v157 = (_QWORD *)v157[3];
              if ( !v160 )
                goto LABEL_328;
            }
            v158 = (__int64)v157;
            v157 = (_QWORD *)v157[2];
          }
          while ( v159 );
LABEL_328:
          if ( v315 == v158 || *(_QWORD *)(v158 + 32) > v321 )
          {
LABEL_330:
            v323 = &v321;
            v158 = sub_1C46280(v308, (_QWORD *)v158, &v323);
          }
          v161 = (_QWORD *)v2[59];
          v162 = *(_BYTE *)(v158 + 40);
          if ( !v161 )
          {
            v164 = (_QWORD *)v316;
            goto LABEL_338;
          }
          v163 = v321;
          v164 = (_QWORD *)v316;
          do
          {
            while ( 1 )
            {
              v165 = v161[2];
              v166 = v161[3];
              if ( v161[4] >= v321 )
                break;
              v161 = (_QWORD *)v161[3];
              if ( !v166 )
                goto LABEL_336;
            }
            v164 = v161;
            v161 = (_QWORD *)v161[2];
          }
          while ( v165 );
LABEL_336:
          if ( (_QWORD *)v316 == v164 || v164[4] > v321 )
          {
LABEL_338:
            v323 = &v321;
            v167 = sub_1C46280(v307, v164, &v323);
            v163 = v321;
            v164 = (_QWORD *)v167;
          }
          v310 = *(_QWORD *)(v163 + 48);
          v293 = v2 + 82;
          if ( v163 + 40 == v310 )
            goto LABEL_350;
          v290 = v90;
          v168 = *((_BYTE *)v164 + 40);
          v296 = v2;
          v169 = (_QWORD *)(v163 + 40);
          while ( 2 )
          {
            v170 = *v169 & 0xFFFFFFFFFFFFFFF8LL;
            v171 = v170;
            v169 = (_QWORD *)v170;
            if ( !v170 )
            {
              v322 = 0;
              BUG();
            }
            v172 = v170 - 24;
            v322 = v170 - 24;
            if ( *(_BYTE *)(v170 - 8) == 78
              && (v173 = *(_QWORD *)(v170 - 48), !*(_BYTE *)(v173 + 16))
              && (*(_BYTE *)(v173 + 33) & 0x20) != 0 )
            {
              if ( !sub_1C301F0(*(_DWORD *)(v173 + 36)) )
              {
                v172 = v322;
                goto LABEL_347;
              }
              v227 = (_QWORD *)v296[77];
              if ( !v227 )
              {
                v228 = (__int64)(v296 + 76);
                goto LABEL_481;
              }
              v228 = (__int64)(v296 + 76);
              do
              {
                while ( 1 )
                {
                  v229 = v227[2];
                  v230 = v227[3];
                  if ( v227[4] >= v322 )
                    break;
                  v227 = (_QWORD *)v227[3];
                  if ( !v230 )
                    goto LABEL_479;
                }
                v228 = (__int64)v227;
                v227 = (_QWORD *)v227[2];
              }
              while ( v229 );
LABEL_479:
              if ( (_QWORD *)v228 == v296 + 76 || *(_QWORD *)(v228 + 32) > v322 )
              {
LABEL_481:
                v323 = &v322;
                v228 = sub_1C47760(v296 + 75, (_QWORD *)v228, &v323);
              }
              *(_BYTE *)(v228 + 40) = v162;
              v231 = (_QWORD *)v296[83];
              if ( !v231 )
              {
                v232 = (__int64)v293;
                goto LABEL_489;
              }
              v232 = (__int64)v293;
              do
              {
                while ( 1 )
                {
                  v233 = v231[2];
                  v234 = v231[3];
                  if ( v231[4] >= v322 )
                    break;
                  v231 = (_QWORD *)v231[3];
                  if ( !v234 )
                    goto LABEL_487;
                }
                v232 = (__int64)v231;
                v231 = (_QWORD *)v231[2];
              }
              while ( v233 );
LABEL_487:
              if ( (_QWORD *)v232 == v293 || *(_QWORD *)(v232 + 32) > v322 )
              {
LABEL_489:
                v323 = &v322;
                v232 = sub_1C47760(v296 + 81, (_QWORD *)v232, &v323);
              }
              *(_BYTE *)(v232 + 40) = v168;
              v162 = 0;
              v168 = 0;
            }
            else
            {
LABEL_347:
              LOBYTE(v320) = 0;
              LOBYTE(v323) = 0;
              sub_1C45690(v172, &v320, &v323);
              v162 |= (unsigned __int8)v320;
              v168 |= (unsigned __int8)v323;
            }
            if ( v310 != v171 )
              continue;
            break;
          }
          v90 = v290;
          v2 = v296;
LABEL_350:
          v174 = (_QWORD *)v304[3];
          if ( v174 == v90 )
            goto LABEL_187;
          v175 = v2 + 64;
          srcb = v2 + 82;
          while ( 2 )
          {
            if ( !v174 )
            {
              v319 = 0;
              BUG();
            }
            v319 = (unsigned __int64)(v174 - 3);
            if ( *((_BYTE *)v174 - 8) != 78
              || (v176 = *(v174 - 6), *(_BYTE *)(v176 + 16))
              || (*(_BYTE *)(v176 + 33) & 0x20) == 0
              || !(v177 = sub_1C301F0(*(_DWORD *)(v176 + 36)))
              || *(_BYTE *)(v319 + 16) == 78
              && (v226 = *(_QWORD *)(v319 - 24), !*(_BYTE *)(v226 + 16))
              && (*(_BYTE *)(v226 + 33) & 0x20) != 0
              && (unsigned int)(*(_DWORD *)(v226 + 36) - 3734) <= 2
              && (unsigned __int8)sub_1648D00(v319, 1) )
            {
LABEL_352:
              v174 = (_QWORD *)v174[1];
              if ( v174 == v90 )
                goto LABEL_187;
              continue;
            }
            break;
          }
          v178 = (_QWORD *)v2[65];
          v179 = v319;
          v180 = v2 + 64;
          v181 = v178;
          if ( v178 )
          {
            while ( 1 )
            {
              v182 = (_QWORD *)v181[3];
              if ( v181[4] >= v319 )
              {
                v182 = (_QWORD *)v181[2];
                v180 = v181;
              }
              if ( !v182 )
                break;
              v181 = v182;
            }
            if ( v180 != v175 && v180[4] <= v319 )
            {
              v291 = v281;
              if ( !*((_BYTE *)v180 + 40) )
                goto LABEL_368;
LABEL_410:
              v203 = (__int64)(v2 + 64);
              while ( 1 )
              {
                v204 = (_QWORD *)v178[3];
                if ( v178[4] >= v179 )
                {
                  v204 = (_QWORD *)v178[2];
                  v203 = (__int64)v178;
                }
                if ( !v204 )
                  break;
                v178 = v204;
              }
              if ( v175 == (_QWORD *)v203 || *(_QWORD *)(v203 + 32) > v179 )
                goto LABEL_417;
              goto LABEL_418;
            }
          }
          v323 = &v319;
          v291 = v281;
          if ( *(_BYTE *)(sub_1C47760(v2 + 63, v180, &v323) + 40) )
          {
LABEL_408:
            v178 = (_QWORD *)v2[65];
            if ( v178 )
            {
              v179 = v319;
              goto LABEL_410;
            }
            v203 = (__int64)(v2 + 64);
LABEL_417:
            v323 = &v319;
            v203 = sub_1C47760(v2 + 63, (_QWORD *)v203, &v323);
LABEL_418:
            if ( !*(_BYTE *)(v203 + 40) )
              goto LABEL_439;
            v205 = v2[71];
            v206 = (_QWORD *)v205;
            if ( v205 )
            {
              v207 = v319;
              v208 = v281;
              do
              {
                while ( 1 )
                {
                  v209 = v206[2];
                  v210 = v206[3];
                  if ( v206[4] >= v319 )
                    break;
                  v206 = (_QWORD *)v206[3];
                  if ( !v210 )
                    goto LABEL_424;
                }
                v208 = v206;
                v206 = (_QWORD *)v206[2];
              }
              while ( v209 );
LABEL_424:
              if ( v208 != v281 && v208[4] <= v319 )
              {
                if ( !*((_BYTE *)v208 + 40) )
                  goto LABEL_427;
LABEL_441:
                v215 = (__int64)v281;
                while ( 1 )
                {
                  v216 = *(_QWORD *)(v205 + 16);
                  v217 = *(_QWORD *)(v205 + 24);
                  if ( *(_QWORD *)(v205 + 32) < v207 )
                  {
                    v205 = v215;
                    v216 = v217;
                  }
                  if ( !v216 )
                    break;
                  v215 = v205;
                  v205 = v216;
                }
                if ( (_QWORD *)v205 == v281 || *(_QWORD *)(v205 + 32) > v207 )
                  goto LABEL_448;
                goto LABEL_449;
              }
            }
            else
            {
              v208 = v281;
            }
            v323 = &v319;
            if ( *(_BYTE *)(sub_1C47760(v2 + 69, v208, &v323) + 40) )
            {
LABEL_439:
              v205 = v2[71];
              if ( v205 )
              {
                v207 = v319;
                goto LABEL_441;
              }
              v205 = (__int64)v281;
LABEL_448:
              v323 = &v319;
              v205 = sub_1C47760(v2 + 69, (_QWORD *)v205, &v323);
LABEL_449:
              if ( !*(_BYTE *)(v205 + 40) )
                goto LABEL_352;
              v218 = (__int64)(v2 + 76);
              v219 = (_QWORD *)v2[77];
              v294 = v2 + 76;
              if ( !v219 )
              {
                v218 = (__int64)(v2 + 76);
                goto LABEL_457;
              }
              do
              {
                while ( 1 )
                {
                  v220 = v219[2];
                  v221 = v219[3];
                  if ( v219[4] >= v319 )
                    break;
                  v219 = (_QWORD *)v219[3];
                  if ( !v221 )
                    goto LABEL_455;
                }
                v218 = (__int64)v219;
                v219 = (_QWORD *)v219[2];
              }
              while ( v220 );
LABEL_455:
              if ( (_QWORD *)v218 == v2 + 76 || *(_QWORD *)(v218 + 32) > v319 )
              {
LABEL_457:
                v323 = &v319;
                v218 = sub_1C47760(v2 + 75, (_QWORD *)v218, &v323);
              }
              if ( *(_BYTE *)(v218 + 40) )
                goto LABEL_352;
              v222 = (__int64)(v2 + 82);
              v223 = (_QWORD *)v2[83];
              v297 = v2 + 82;
              if ( v223 )
              {
                do
                {
                  while ( 1 )
                  {
                    v224 = v223[2];
                    v225 = v223[3];
                    if ( v223[4] >= v319 )
                      break;
                    v223 = (_QWORD *)v223[3];
                    if ( !v225 )
                      goto LABEL_464;
                  }
                  v222 = (__int64)v223;
                  v223 = (_QWORD *)v223[2];
                }
                while ( v224 );
LABEL_464:
                if ( (void *)v222 == srcb || *(_QWORD *)(v222 + 32) > v319 )
                {
LABEL_466:
                  v323 = &v319;
                  v222 = sub_1C47760(v2 + 81, (_QWORD *)v222, &v323);
                }
                if ( !*(_BYTE *)(v222 + 40) )
                {
                  v311 = v2 + 64;
LABEL_378:
                  v320 = (_QWORD *)v319;
                  v321 = v319;
                  if ( *(_QWORD *)(v319 + 48) || *(__int16 *)(v319 + 18) < 0 )
                  {
                    v187 = sub_1625940(v319, "dbg", 3u);
                    if ( v187 )
                    {
                      v305 = *(_DWORD *)(v187 + 4);
                      v188 = *(_QWORD *)(v187 - 8LL * *(unsigned int *)(v187 + 8));
                      if ( *(_BYTE *)v188 == 15 || (v188 = *(_QWORD *)(v188 - 8LL * *(unsigned int *)(v188 + 8))) != 0 )
                      {
                        v189 = *(const char **)(v188 - 8LL * *(unsigned int *)(v188 + 8));
                        if ( v189 )
                        {
                          v189 = (const char *)sub_161E970(*(_QWORD *)(v188 - 8LL * *(unsigned int *)(v188 + 8)));
                          v191 = v190;
                        }
                        else
                        {
                          v191 = 0;
                        }
                      }
                      else
                      {
                        v191 = 0;
                        v189 = byte_3F871B3;
                      }
                      v192 = *v2;
                      v193 = *(_BYTE **)(*v2 + 24LL);
                      if ( *(_BYTE **)(*v2 + 16LL) == v193 )
                      {
                        srcd = v189;
                        v276 = sub_16E7EE0(*v2, "[", 1u);
                        v189 = srcd;
                        v192 = v276;
                      }
                      else
                      {
                        *v193 = 91;
                        ++*(_QWORD *)(v192 + 24);
                      }
                      if ( v189 )
                      {
                        v322 = v191;
                        v323 = v325;
                        if ( v191 > 0xF )
                        {
                          srce = v189;
                          v278 = (unsigned __int64 *)sub_22409D0(&v323, &v322, 0);
                          v189 = srce;
                          v323 = v278;
                          v279 = v278;
                          v325[0] = v322;
                        }
                        else
                        {
                          if ( v191 == 1 )
                          {
                            LOBYTE(v325[0]) = *v189;
                            v194 = v325;
                            goto LABEL_585;
                          }
                          if ( !v191 )
                          {
                            v194 = v325;
                            goto LABEL_585;
                          }
                          v279 = v325;
                        }
                        memcpy(v279, v189, v191);
                        v191 = v322;
                        v194 = v323;
LABEL_585:
                        v324 = v191;
                        *((_BYTE *)v194 + v191) = 0;
                        v270 = v324;
                        v271 = (char *)v323;
                      }
                      else
                      {
                        v324 = 0;
                        v270 = 0;
                        v323 = v325;
                        v271 = (char *)v325;
                        LOBYTE(v325[0]) = 0;
                      }
                      sub_16E7EE0(v192, v271, v270);
                      if ( v323 != v325 )
                        j_j___libc_free_0(v323, v325[0] + 1LL);
                      v272 = *v2;
                      v273 = *(_BYTE **)(*v2 + 24LL);
                      if ( *(_BYTE **)(*v2 + 16LL) == v273 )
                      {
                        v272 = sub_16E7EE0(v272, ":", 1u);
                      }
                      else
                      {
                        *v273 = 58;
                        ++*(_QWORD *)(v272 + 24);
                      }
                      v274 = sub_16E7A90(v272, v305);
                      v275 = *(_BYTE **)(v274 + 24);
                      if ( *(_BYTE **)(v274 + 16) == v275 )
                      {
                        sub_16E7EE0(v274, "]", 1u);
                      }
                      else
                      {
                        *v275 = 93;
                        ++*(_QWORD *)(v274 + 24);
                      }
                    }
                  }
                  v235 = *v2;
                  v236 = *(__m128i **)(*v2 + 24LL);
                  if ( *(_QWORD *)(*v2 + 16LL) - (_QWORD)v236 <= 0x14u )
                  {
                    sub_16E7EE0(v235, " Removed dead synch: ", 0x15u);
                  }
                  else
                  {
                    si128 = _mm_load_si128((const __m128i *)&xmmword_42D1060);
                    v236[1].m128i_i32[0] = 979919726;
                    v236[1].m128i_i8[4] = 32;
                    *v236 = si128;
                    *(_QWORD *)(v235 + 24) += 21LL;
                  }
                  v238 = *v2;
                  v239 = *(void **)(*v2 + 24LL);
                  if ( *(_QWORD *)(*v2 + 16LL) - (_QWORD)v239 <= 0xBu )
                  {
                    v238 = sub_16E7EE0(*v2, "Read above: ", 0xCu);
                  }
                  else
                  {
                    qmemcpy(v239, "Read above: ", 12);
                    *(_QWORD *)(v238 + 24) += 12LL;
                  }
                  v240 = (_QWORD *)v2[65];
                  if ( !v240 )
                  {
                    v241 = (__int64)v311;
LABEL_551:
                    v312 = v2 + 63;
                    v323 = &v321;
                    v241 = sub_1C47760(v2 + 63, (_QWORD *)v241, &v323);
                    goto LABEL_509;
                  }
                  v241 = (__int64)v311;
                  do
                  {
                    while ( 1 )
                    {
                      v242 = v240[2];
                      v243 = v240[3];
                      if ( v240[4] >= v321 )
                        break;
                      v240 = (_QWORD *)v240[3];
                      if ( !v243 )
                        goto LABEL_506;
                    }
                    v241 = (__int64)v240;
                    v240 = (_QWORD *)v240[2];
                  }
                  while ( v242 );
LABEL_506:
                  if ( v311 == (_QWORD *)v241 || *(_QWORD *)(v241 + 32) > v321 )
                    goto LABEL_551;
                  v312 = v2 + 63;
LABEL_509:
                  v244 = sub_16E7AB0(v238, *(unsigned __int8 *)(v241 + 40));
                  v245 = *(void **)(v244 + 24);
                  if ( *(_QWORD *)(v244 + 16) - (_QWORD)v245 <= 0xEu )
                  {
                    v244 = sub_16E7EE0(v244, ", Write above: ", 0xFu);
                  }
                  else
                  {
                    qmemcpy(v245, ", Write above: ", 15);
                    *(_QWORD *)(v244 + 24) += 15LL;
                  }
                  v246 = (_QWORD *)v2[71];
                  if ( !v246 )
                    goto LABEL_519;
                  v247 = (__int64)v281;
                  do
                  {
                    while ( 1 )
                    {
                      v248 = v246[2];
                      v249 = v246[3];
                      if ( v246[4] >= v321 )
                        break;
                      v246 = (_QWORD *)v246[3];
                      if ( !v249 )
                        goto LABEL_516;
                    }
                    v247 = (__int64)v246;
                    v246 = (_QWORD *)v246[2];
                  }
                  while ( v248 );
LABEL_516:
                  if ( (_QWORD *)v247 == v281 )
                  {
LABEL_519:
                    v250 = v2 + 69;
                    v323 = &v321;
                    v247 = sub_1C47760(v2 + 69, v291, &v323);
                  }
                  else
                  {
                    v250 = v2 + 69;
                    if ( *(_QWORD *)(v247 + 32) > v321 )
                    {
                      v291 = (_QWORD *)v247;
                      goto LABEL_519;
                    }
                  }
                  v251 = sub_16E7AB0(v244, *(unsigned __int8 *)(v247 + 40));
                  v252 = *(void **)(v251 + 24);
                  if ( *(_QWORD *)(v251 + 16) - (_QWORD)v252 <= 0xDu )
                  {
                    v251 = sub_16E7EE0(v251, ", Read below: ", 0xEu);
                  }
                  else
                  {
                    qmemcpy(v252, ", Read below: ", 14);
                    *(_QWORD *)(v251 + 24) += 14LL;
                  }
                  v253 = (_QWORD *)v2[77];
                  if ( v253 )
                  {
                    v254 = (__int64)v294;
                    do
                    {
                      while ( 1 )
                      {
                        v255 = v253[2];
                        v256 = v253[3];
                        if ( v253[4] >= v321 )
                          break;
                        v253 = (_QWORD *)v253[3];
                        if ( !v256 )
                          goto LABEL_527;
                      }
                      v254 = (__int64)v253;
                      v253 = (_QWORD *)v253[2];
                    }
                    while ( v255 );
LABEL_527:
                    if ( v294 != (_QWORD *)v254 && *(_QWORD *)(v254 + 32) <= v321 )
                    {
                      v306 = v2 + 75;
                      goto LABEL_530;
                    }
                  }
                  else
                  {
                    v254 = (__int64)v294;
                  }
                  v306 = v2 + 75;
                  v323 = &v321;
                  v254 = sub_1C47760(v2 + 75, (_QWORD *)v254, &v323);
LABEL_530:
                  v257 = sub_16E7AB0(v251, *(unsigned __int8 *)(v254 + 40));
                  v258 = *(void **)(v257 + 24);
                  if ( *(_QWORD *)(v257 + 16) - (_QWORD)v258 <= 0xEu )
                  {
                    v257 = sub_16E7EE0(v257, ", Write below: ", 0xFu);
                  }
                  else
                  {
                    qmemcpy(v258, ", Write below: ", 15);
                    *(_QWORD *)(v257 + 24) += 15LL;
                  }
                  v259 = (_QWORD *)v2[83];
                  if ( !v259 )
                  {
                    v260 = (__int64)v297;
                    goto LABEL_539;
                  }
                  v260 = (__int64)v297;
                  do
                  {
                    while ( 1 )
                    {
                      v261 = v259[2];
                      v262 = v259[3];
                      if ( v259[4] >= v321 )
                        break;
                      v259 = (_QWORD *)v259[3];
                      if ( !v262 )
                        goto LABEL_537;
                    }
                    v260 = (__int64)v259;
                    v259 = (_QWORD *)v259[2];
                  }
                  while ( v261 );
LABEL_537:
                  if ( v297 == (_QWORD *)v260 || (v263 = v2 + 81, *(_QWORD *)(v260 + 32) > v321) )
                  {
LABEL_539:
                    v263 = v2 + 81;
                    v323 = &v321;
                    v260 = sub_1C47760(v2 + 81, (_QWORD *)v260, &v323);
                  }
                  v264 = sub_16E7AB0(v257, *(unsigned __int8 *)(v260 + 40));
                  v265 = *(void **)(v264 + 24);
                  if ( *(_QWORD *)(v264 + 16) - (_QWORD)v265 <= 0xCu )
                  {
                    v277 = sub_16E7EE0(v264, " in function ", 0xDu);
                    v266 = *(char **)(v277 + 24);
                    v264 = v277;
                  }
                  else
                  {
                    qmemcpy(v265, " in function ", 13);
                    v266 = (char *)(*(_QWORD *)(v264 + 24) + 13LL);
                    *(_QWORD *)(v264 + 24) = v266;
                  }
                  v267 = *(char **)(v264 + 16);
                  v268 = v2[2];
                  v269 = (char *)v2[1];
                  if ( v268 > v267 - v266 )
                  {
                    v264 = sub_16E7EE0(v264, v269, v268);
                    v267 = *(char **)(v264 + 16);
                    v266 = *(char **)(v264 + 24);
                  }
                  else if ( v268 )
                  {
                    srcc = (char *)v2[2];
                    memcpy(v266, v269, v268);
                    v267 = *(char **)(v264 + 16);
                    v266 = &srcc[*(_QWORD *)(v264 + 24)];
                    *(_QWORD *)(v264 + 24) = v266;
                  }
                  if ( v266 == v267 )
                  {
                    sub_16E7EE0(v264, "\n", 1u);
                  }
                  else
                  {
                    *v266 = 10;
                    ++*(_QWORD *)(v264 + 24);
                  }
                  sub_1C45B10(v312, (unsigned __int64 *)&v320);
                  sub_1C45B10(v250, (unsigned __int64 *)&v320);
                  sub_1C45B10(v306, (unsigned __int64 *)&v320);
                  sub_1C45B10(v263, (unsigned __int64 *)&v320);
                  sub_15F20C0(v320);
                  v283 = v177;
                  goto LABEL_2;
                }
                goto LABEL_352;
              }
              v222 = (__int64)(v2 + 82);
              goto LABEL_466;
            }
LABEL_427:
            v211 = (__int64)(v2 + 82);
            v212 = (_QWORD *)v2[83];
            v297 = v2 + 82;
            if ( v212 )
            {
              do
              {
                while ( 1 )
                {
                  v213 = v212[2];
                  v214 = v212[3];
                  if ( v212[4] >= v319 )
                    break;
                  v212 = (_QWORD *)v212[3];
                  if ( !v214 )
                    goto LABEL_432;
                }
                v211 = (__int64)v212;
                v212 = (_QWORD *)v212[2];
              }
              while ( v213 );
LABEL_432:
              if ( srcb == (void *)v211 || *(_QWORD *)(v211 + 32) > v319 )
              {
LABEL_434:
                v323 = &v319;
                v211 = sub_1C47760(v2 + 81, (_QWORD *)v211, &v323);
              }
              if ( !*(_BYTE *)(v211 + 40) )
              {
                v311 = v2 + 64;
                v294 = v2 + 76;
                goto LABEL_378;
              }
              goto LABEL_352;
            }
            v211 = (__int64)(v2 + 82);
            goto LABEL_434;
          }
LABEL_368:
          v183 = (_QWORD *)v2[71];
          if ( v183 )
          {
            v184 = (__int64)v281;
            do
            {
              while ( 1 )
              {
                v185 = v183[2];
                v186 = v183[3];
                if ( v183[4] >= v319 )
                  break;
                v183 = (_QWORD *)v183[3];
                if ( !v186 )
                  goto LABEL_373;
              }
              v184 = (__int64)v183;
              v183 = (_QWORD *)v183[2];
            }
            while ( v185 );
LABEL_373:
            if ( (_QWORD *)v184 == v281 || *(_QWORD *)(v184 + 32) > v319 )
            {
LABEL_375:
              v323 = &v319;
              v184 = sub_1C47760(v2 + 69, (_QWORD *)v184, &v323);
            }
            if ( !*(_BYTE *)(v184 + 40) )
            {
              v311 = v2 + 64;
              v297 = v2 + 82;
              v294 = v2 + 76;
              goto LABEL_378;
            }
            goto LABEL_408;
          }
          v184 = (__int64)v281;
          goto LABEL_375;
        }
        continue;
      }
    }
  }
  v2 = v27;
LABEL_188:
  sub_1C45C70(v2[5]);
  v93 = v2[11];
  v2[5] = 0;
  v2[6] = v2 + 4;
  v2[7] = v2 + 4;
  v2[8] = 0;
  sub_1C45C70(v93);
  v2[11] = 0;
  v94 = v2[17];
  v2[12] = v2 + 10;
  v2[13] = v2 + 10;
  v2[14] = 0;
  sub_1C45C70(v94);
  v2[17] = 0;
  v95 = v2[23];
  v2[18] = v2 + 16;
  v2[19] = v2 + 16;
  v2[20] = 0;
  sub_1C45C70(v95);
  v2[23] = 0;
  v96 = v2[29];
  v2[24] = v2 + 22;
  v2[25] = v2 + 22;
  v2[26] = 0;
  sub_1C45C70(v96);
  v2[29] = 0;
  v97 = v2[35];
  v2[30] = v2 + 28;
  v2[31] = v2 + 28;
  v2[32] = 0;
  sub_1C45C70(v97);
  v2[35] = 0;
  v98 = v2[41];
  v2[36] = v2 + 34;
  v2[37] = v2 + 34;
  v2[38] = 0;
  sub_1C45C70(v98);
  v99 = v2[47];
  v2[41] = 0;
  v2[44] = 0;
  v2[42] = v318;
  v2[43] = v318;
  sub_1C45C70(v99);
  v100 = v2[53];
  v2[47] = 0;
  v2[50] = 0;
  v2[48] = v317;
  v2[49] = v317;
  sub_1C45C70(v100);
  v2[53] = 0;
  v2[56] = 0;
  v101 = v2[59];
  v2[54] = v315;
  v2[55] = v315;
  sub_1C45C70(v101);
  v2[59] = 0;
  v2[62] = 0;
  v102 = v2[65];
  v2[60] = v316;
  v2[61] = v316;
  sub_1C45940(v102);
  v2[65] = 0;
  v103 = v2[71];
  v2[66] = v2 + 64;
  v2[67] = v2 + 64;
  v2[68] = 0;
  sub_1C45940(v103);
  v2[71] = 0;
  v104 = v2[77];
  v2[72] = v2 + 70;
  v2[73] = v2 + 70;
  v2[74] = 0;
  sub_1C45940(v104);
  v2[77] = 0;
  v105 = v2[83];
  v2[78] = v2 + 76;
  v2[79] = v2 + 76;
  v2[80] = 0;
  sub_1C45940(v105);
  v2[83] = 0;
  v2[84] = v2 + 82;
  v2[85] = v2 + 82;
  v2[86] = 0;
  return v283;
}
