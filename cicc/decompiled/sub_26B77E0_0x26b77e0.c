// Function: sub_26B77E0
// Address: 0x26b77e0
//
__int64 __fastcall sub_26B77E0(
        __int64 *a1,
        __int64 **a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        char a9)
{
  __int64 **v9; // r10
  __int64 v13; // rcx
  void (__fastcall *v14)(__int64 **, __int64, __int64); // rax
  __int64 **v15; // rsi
  void (__fastcall *v16)(__int64 **, __int64, __int64); // rax
  void (__fastcall *v17)(_QWORD **, __int64, __int64); // rax
  void (__fastcall *v18)(__int64 **, __int64, __int64); // rax
  void (__fastcall *v19)(_QWORD *, __int64, __int64); // rax
  __int64 v20; // rax
  void (__fastcall *v21)(_QWORD, _QWORD, _QWORD); // rax
  __int64 *v22; // rbx
  __int64 v23; // r13
  __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // r15
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 **v30; // rax
  __int64 **v31; // r15
  __int64 **i; // r13
  __int64 v33; // r13
  __int64 v34; // r12
  __int64 v35; // rax
  __int64 v36; // r15
  unsigned __int8 v37; // r14
  __int64 v38; // rsi
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r15
  char v42; // r12
  __int64 v43; // rax
  __int64 v44; // rax
  unsigned __int64 v45; // rdx
  __int64 v46; // r9
  __int64 v47; // r14
  __int64 v48; // r8
  __int64 v49; // r9
  char v50; // al
  __int64 v51; // rdx
  __int64 v52; // rcx
  int v53; // r9d
  int v54; // esi
  unsigned int k; // eax
  __int64 v56; // r8
  unsigned int v57; // eax
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 *v60; // r15
  __int64 *v61; // r14
  __int64 v62; // rdi
  __int64 v63; // r8
  __int64 v64; // r9
  _QWORD *v65; // rsi
  __int64 v66; // rdx
  __int64 v67; // rcx
  __int64 v68; // r8
  __int64 v69; // r9
  __int64 v70; // rdi
  __int64 v71; // rdi
  __int64 v72; // r8
  __int64 v73; // r9
  __int64 v74; // r13
  char v75; // al
  unsigned __int64 v76; // r13
  __int64 *v77; // r14
  __int64 v78; // r12
  __int64 v79; // r14
  __int64 v80; // rdx
  __int64 v81; // rcx
  __int64 v82; // r8
  __int64 v83; // r9
  _QWORD *v84; // r12
  _QWORD *v85; // r15
  void (__fastcall *v86)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v87; // rax
  __int64 v88; // rax
  __int64 v89; // rax
  __int64 *v90; // r13
  __int64 *v91; // rbx
  __int64 v92; // r15
  __int64 v93; // rsi
  __int64 *v94; // rax
  __int64 *v95; // r14
  __int64 v96; // r13
  unsigned __int64 v97; // r12
  unsigned __int64 v98; // r15
  __int64 *v99; // r14
  __int64 v100; // r12
  __int64 v101; // rbx
  __int64 v102; // rdx
  __int64 v103; // r8
  __int64 v104; // r9
  __int64 v105; // rax
  __int64 v106; // rcx
  __int64 v107; // rcx
  _QWORD *v108; // rax
  __int64 v109; // rdx
  __int64 v110; // rdi
  __int64 v111; // rdx
  _QWORD *v112; // rdx
  unsigned __int64 v113; // rdx
  __int64 v114; // rcx
  __int64 v115; // r13
  __int64 v116; // rbx
  __int64 v117; // r12
  int v118; // esi
  unsigned __int8 *v119; // r15
  int v120; // eax
  __int64 v121; // r13
  __int64 v122; // rdi
  int v123; // r10d
  unsigned int m; // eax
  __int64 v125; // r8
  unsigned int v126; // eax
  __int64 v127; // rcx
  __int64 v128; // r13
  __int64 v129; // r12
  int v130; // eax
  unsigned __int64 v131; // rax
  __int64 v132; // r12
  unsigned __int64 v133; // rax
  unsigned __int8 *v134; // r15
  int v135; // eax
  unsigned __int64 v136; // rax
  __int64 v137; // rcx
  __int64 v138; // rax
  int v139; // eax
  __int64 *v140; // rax
  __int64 *v141; // rax
  __int64 *v142; // rax
  unsigned int v143; // r12d
  __int64 v144; // rax
  __int64 v145; // rdx
  __int64 v146; // rbx
  int v147; // ebx
  __int64 v148; // rax
  __int64 v149; // rdx
  unsigned __int8 *v150; // r13
  __int64 v151; // rbx
  unsigned __int8 *v152; // r12
  __int64 *v153; // rax
  unsigned __int8 *v154; // rdx
  __int64 v155; // rax
  __int64 *v156; // rdx
  __int64 *v157; // rbx
  __int64 v159; // rdx
  __int64 v160; // rcx
  __int64 v161; // r9
  __int64 v162; // r8
  int v163; // r11d
  __int64 *v164; // r10
  __int64 v165; // rdi
  __int64 *v166; // rdx
  __int64 v167; // rcx
  int v168; // eax
  __int64 v169; // rax
  unsigned __int64 v170; // rdx
  __int64 v171; // rax
  __int64 *v172; // rbx
  __int64 *v173; // r12
  __int64 v174; // rax
  __int64 *v175; // rdi
  __int64 v176; // r8
  __int64 v177; // rdx
  __int64 *v178; // rcx
  __int64 v179; // r9
  int v180; // eax
  int v181; // edi
  char v182; // r13
  __int64 *jj; // r15
  __int64 v184; // r12
  __int64 v185; // r13
  __int64 v186; // rax
  unsigned __int64 *v187; // rcx
  unsigned __int64 v188; // rdx
  __int64 v189; // rdx
  __int64 v190; // rcx
  unsigned __int8 *v191; // rdx
  __int64 v192; // rdx
  __int64 v193; // r12
  __int64 v194; // rbx
  __int64 v195; // r14
  unsigned __int64 v196; // r13
  __int64 v197; // rsi
  char *v198; // r14
  char *v199; // r13
  __int64 v200; // rsi
  unsigned __int64 v201; // r13
  unsigned __int64 v202; // r14
  unsigned __int64 v203; // rdi
  char *v204; // r14
  char *v205; // r13
  __int64 v206; // rsi
  char *v207; // r14
  char *v208; // r13
  __int64 v209; // rsi
  char *v210; // r14
  char *v211; // r13
  __int64 v212; // rsi
  int v213; // r11d
  __int64 v214; // rdx
  __int64 v215; // rcx
  int v216; // edi
  __int64 v217; // rdx
  __int64 v218; // r9
  unsigned int v219; // ecx
  __int64 v220; // rdi
  __int64 *v221; // rcx
  unsigned int v222; // r12d
  unsigned int v223; // r12d
  __int64 *v224; // rax
  __int64 v225; // r12
  __int64 *v226; // rax
  unsigned __int64 v227; // [rsp+0h] [rbp-1820h]
  unsigned __int64 v228; // [rsp+8h] [rbp-1818h]
  __int64 j; // [rsp+18h] [rbp-1808h]
  __int64 *v232; // [rsp+48h] [rbp-17D8h]
  __int64 *v233; // [rsp+48h] [rbp-17D8h]
  __int64 v234; // [rsp+50h] [rbp-17D0h]
  __int64 *v235; // [rsp+50h] [rbp-17D0h]
  unsigned __int8 v236; // [rsp+58h] [rbp-17C8h]
  unsigned int v237; // [rsp+58h] [rbp-17C8h]
  __int64 v238; // [rsp+58h] [rbp-17C8h]
  __int64 *v239; // [rsp+58h] [rbp-17C8h]
  __int64 v240; // [rsp+60h] [rbp-17C0h]
  __int64 v241; // [rsp+60h] [rbp-17C0h]
  __int64 v242; // [rsp+68h] [rbp-17B8h]
  __int64 v243; // [rsp+68h] [rbp-17B8h]
  __int64 v244; // [rsp+68h] [rbp-17B8h]
  __int64 n; // [rsp+68h] [rbp-17B8h]
  __int64 ii; // [rsp+68h] [rbp-17B8h]
  char v247[8]; // [rsp+78h] [rbp-17A8h] BYREF
  __m128i **v248; // [rsp+80h] [rbp-17A0h] BYREF
  __int64 v249; // [rsp+88h] [rbp-1798h]
  _BYTE v250[16]; // [rsp+90h] [rbp-1790h] BYREF
  _QWORD v251[2]; // [rsp+A0h] [rbp-1780h] BYREF
  void (__fastcall *v252)(_QWORD, _QWORD, _QWORD); // [rsp+B0h] [rbp-1770h]
  __int64 v253; // [rsp+B8h] [rbp-1768h]
  _QWORD *v254; // [rsp+C0h] [rbp-1760h]
  __int64 *v255; // [rsp+E0h] [rbp-1740h] BYREF
  unsigned __int64 v256; // [rsp+E8h] [rbp-1738h]
  __int64 v257; // [rsp+F0h] [rbp-1730h] BYREF
  __int64 v258; // [rsp+F8h] [rbp-1728h]
  char v259; // [rsp+100h] [rbp-1720h] BYREF
  _QWORD *v260; // [rsp+200h] [rbp-1620h] BYREF
  __int64 v261; // [rsp+208h] [rbp-1618h]
  void (__fastcall *v262)(char *, _QWORD **, __int64); // [rsp+210h] [rbp-1610h] BYREF
  __int64 v263; // [rsp+218h] [rbp-1608h]
  _BYTE *v264; // [rsp+220h] [rbp-1600h] BYREF
  __int64 v265; // [rsp+228h] [rbp-15F8h]
  _BYTE v266[480]; // [rsp+230h] [rbp-15F0h] BYREF
  __int64 v267; // [rsp+410h] [rbp-1410h]
  __int64 v268; // [rsp+418h] [rbp-1408h]
  __int64 v269; // [rsp+420h] [rbp-1400h]
  __int64 v270; // [rsp+428h] [rbp-13F8h]
  char v271; // [rsp+430h] [rbp-13F0h]
  __int64 v272; // [rsp+438h] [rbp-13E8h]
  char *v273; // [rsp+440h] [rbp-13E0h]
  __int64 v274; // [rsp+448h] [rbp-13D8h]
  int v275; // [rsp+450h] [rbp-13D0h]
  char v276; // [rsp+454h] [rbp-13CCh]
  char v277; // [rsp+458h] [rbp-13C8h] BYREF
  __int16 v278; // [rsp+498h] [rbp-1388h]
  _QWORD *v279; // [rsp+4A0h] [rbp-1380h]
  _QWORD *v280; // [rsp+4A8h] [rbp-1378h]
  __int64 v281; // [rsp+4B0h] [rbp-1370h]
  char *v282; // [rsp+4C0h] [rbp-1360h] BYREF
  __int64 *v283; // [rsp+4C8h] [rbp-1358h]
  __int64 v284; // [rsp+4D0h] [rbp-1350h]
  char v285[16]; // [rsp+4D8h] [rbp-1348h] BYREF
  void (__fastcall *v286)(_QWORD, _QWORD, _QWORD); // [rsp+4E8h] [rbp-1338h]
  __int64 v287; // [rsp+4F0h] [rbp-1330h]
  char v288[16]; // [rsp+4F8h] [rbp-1328h] BYREF
  __int64 v289; // [rsp+508h] [rbp-1318h]
  __int64 v290; // [rsp+510h] [rbp-1310h]
  char v291[16]; // [rsp+518h] [rbp-1308h] BYREF
  void (__fastcall *v292)(_QWORD, _QWORD, _QWORD); // [rsp+528h] [rbp-12F8h]
  __int64 v293; // [rsp+530h] [rbp-12F0h]
  char v294[16]; // [rsp+538h] [rbp-12E8h] BYREF
  void (__fastcall *v295)(__int64 **, __int64 **, __int64); // [rsp+548h] [rbp-12D8h]
  __int64 v296; // [rsp+550h] [rbp-12D0h]
  __int64 v297; // [rsp+558h] [rbp-12C8h]
  char *v298; // [rsp+560h] [rbp-12C0h]
  __int64 v299; // [rsp+568h] [rbp-12B8h]
  int v300; // [rsp+570h] [rbp-12B0h]
  char v301; // [rsp+574h] [rbp-12ACh]
  char v302; // [rsp+578h] [rbp-12A8h] BYREF
  __int64 v303; // [rsp+678h] [rbp-11A8h]
  char *v304; // [rsp+680h] [rbp-11A0h]
  __int64 v305; // [rsp+688h] [rbp-1198h]
  int v306; // [rsp+690h] [rbp-1190h]
  char v307; // [rsp+694h] [rbp-118Ch]
  char v308; // [rsp+698h] [rbp-1188h] BYREF
  __int64 v309; // [rsp+798h] [rbp-1088h]
  __int64 v310; // [rsp+7A0h] [rbp-1080h]
  __int64 v311; // [rsp+7A8h] [rbp-1078h]
  int v312; // [rsp+7B0h] [rbp-1070h]
  __int64 v313; // [rsp+7B8h] [rbp-1068h]
  __int64 v314; // [rsp+7C0h] [rbp-1060h]
  __int64 v315; // [rsp+7C8h] [rbp-1058h]
  int v316; // [rsp+7D0h] [rbp-1050h]
  int v317; // [rsp+7D8h] [rbp-1048h]
  __int64 *v318; // [rsp+7E0h] [rbp-1040h] BYREF
  unsigned __int64 v319; // [rsp+7E8h] [rbp-1038h]
  void (__fastcall *v320)(__int64 **, __int64 **, __int64); // [rsp+7F0h] [rbp-1030h] BYREF
  __int64 v321; // [rsp+7F8h] [rbp-1028h]
  char *v322; // [rsp+818h] [rbp-1008h]
  int v323; // [rsp+820h] [rbp-1000h]
  char v324; // [rsp+828h] [rbp-FF8h] BYREF
  char *v325; // [rsp+848h] [rbp-FD8h]
  int v326; // [rsp+850h] [rbp-FD0h]
  char v327; // [rsp+858h] [rbp-FC8h] BYREF
  char *v328; // [rsp+878h] [rbp-FA8h]
  char v329; // [rsp+888h] [rbp-F98h] BYREF
  char *v330; // [rsp+8A8h] [rbp-F78h]
  char v331; // [rsp+8B8h] [rbp-F68h] BYREF
  char *v332; // [rsp+8D8h] [rbp-F48h]
  int v333; // [rsp+8E0h] [rbp-F40h]
  char v334; // [rsp+8E8h] [rbp-F38h] BYREF
  __int64 v335; // [rsp+910h] [rbp-F10h]
  unsigned int v336; // [rsp+920h] [rbp-F00h]
  unsigned __int64 v337; // [rsp+928h] [rbp-EF8h]
  unsigned int v338; // [rsp+930h] [rbp-EF0h]
  char *v339; // [rsp+938h] [rbp-EE8h] BYREF
  int v340; // [rsp+940h] [rbp-EE0h]
  char v341; // [rsp+948h] [rbp-ED8h] BYREF
  __int64 v342; // [rsp+978h] [rbp-EA8h]
  unsigned int v343; // [rsp+988h] [rbp-E98h]

  v9 = a2;
  v13 = *a1;
  v14 = *(void (__fastcall **)(__int64 **, __int64, __int64))(a4 + 16);
  v320 = 0;
  if ( v14 )
  {
    v242 = v13;
    v14(&v318, a4, 2);
    v9 = a2;
    v13 = v242;
    v321 = *(_QWORD *)(a4 + 24);
    v320 = *(void (__fastcall **)(__int64 **, __int64 **, __int64))(a4 + 16);
  }
  v15 = v9;
  sub_2A63AD0(v247, v9, &v318, v13);
  if ( v320 )
  {
    v15 = &v318;
    v320(&v318, &v318, 3);
  }
  v320 = 0;
  v16 = *(void (__fastcall **)(__int64 **, __int64, __int64))(a6 + 16);
  if ( v16 )
  {
    v15 = (__int64 **)a6;
    v16(&v318, a6, 2);
    v321 = *(_QWORD *)(a6 + 24);
    v320 = *(void (__fastcall **)(__int64 **, __int64 **, __int64))(a6 + 16);
  }
  v262 = 0;
  v17 = *(void (__fastcall **)(_QWORD **, __int64, __int64))(a5 + 16);
  if ( v17 )
  {
    v15 = (__int64 **)a5;
    v17(&v260, a5, 2);
    v263 = *(_QWORD *)(a5 + 24);
    v262 = *(void (__fastcall **)(char *, _QWORD **, __int64))(a5 + 16);
  }
  v257 = 0;
  v18 = *(void (__fastcall **)(__int64 **, __int64, __int64))(a4 + 16);
  if ( v18 )
  {
    v15 = (__int64 **)a4;
    v18(&v255, a4, 2);
    v258 = *(_QWORD *)(a4 + 24);
    v257 = *(_QWORD *)(a4 + 16);
  }
  v252 = 0;
  v19 = *(void (__fastcall **)(_QWORD *, __int64, __int64))(a8 + 16);
  if ( v19 )
  {
    v15 = (__int64 **)a8;
    v19(v251, a8, 2);
    v20 = *(_QWORD *)(a8 + 24);
    v286 = 0;
    v282 = v247;
    v253 = v20;
    v21 = *(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(a8 + 16);
    v283 = a1;
    v252 = v21;
    v284 = a3;
    if ( v21 )
    {
      v15 = (__int64 **)v251;
      v21(v285, v251, 2);
      v287 = v253;
      v286 = v252;
    }
  }
  else
  {
    v286 = 0;
    v282 = v247;
    v283 = a1;
    v284 = a3;
  }
  v289 = 0;
  if ( v257 )
  {
    v15 = &v255;
    ((void (__fastcall *)(char *, __int64 **, __int64))v257)(v288, &v255, 2);
    v290 = v258;
    v289 = v257;
  }
  v292 = 0;
  if ( v262 )
  {
    v15 = &v260;
    v262(v291, &v260, 2);
    v293 = v263;
    v292 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v262;
  }
  v295 = 0;
  if ( v320 )
  {
    v15 = &v318;
    v320((__int64 **)v294, &v318, 2);
    v296 = v321;
    v295 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v320;
  }
  v297 = 0;
  v298 = &v302;
  v304 = &v308;
  v299 = 32;
  v300 = 0;
  v301 = 1;
  v303 = 0;
  v305 = 32;
  v306 = 0;
  v307 = 1;
  v309 = 0;
  v310 = 0;
  v311 = 0;
  v312 = 0;
  v313 = 0;
  v314 = 0;
  v315 = 0;
  v316 = 0;
  v317 = 0;
  if ( v252 )
  {
    v15 = (__int64 **)v251;
    v252(v251, v251, 3);
  }
  if ( v257 )
  {
    v15 = &v255;
    ((void (__fastcall *)(__int64 **, __int64 **, __int64))v257)(&v255, &v255, 3);
  }
  if ( v262 )
  {
    v15 = &v260;
    v262((char *)&v260, &v260, 3);
  }
  if ( v320 )
  {
    v15 = &v318;
    v320(&v318, &v318, 3);
  }
  v22 = (__int64 *)a1[4];
  v232 = a1 + 3;
  if ( v22 != a1 + 3 )
  {
    v243 = a6;
    while ( 1 )
    {
      v23 = 0;
      if ( v22 )
        v23 = (__int64)(v22 - 7);
      v24 = v23;
      if ( sub_B2FC80(v23) )
        goto LABEL_44;
      if ( !*(_QWORD *)(a7 + 16)
        || (v15 = (__int64 **)v23,
            v24 = a7,
            v26 = (*(__int64 (__fastcall **)(__int64, __int64))(a7 + 24))(a7, v23),
            !*(_QWORD *)(v243 + 16)) )
      {
        sub_4263D6(v24, v15, v25);
      }
      v27 = (*(__int64 (__fastcall **)(__int64, __int64))(v243 + 24))(v243, v23);
      sub_2A6E4B0(v247, v23, v26, v27);
      if ( (unsigned __int8)sub_310F860(v23) )
        sub_2A73720(v247, v23);
      if ( (unsigned __int8)sub_310F810(v23) )
      {
        v15 = (__int64 **)v23;
        sub_2A64130(v247, v23);
        v22 = (__int64 *)v22[1];
        if ( v22 == v232 )
          break;
      }
      else
      {
        v15 = *(__int64 ***)(v23 + 80);
        if ( v15 )
          v15 -= 3;
        sub_2A63F40(v247, v15);
        if ( (*(_BYTE *)(v23 + 2) & 1) != 0 )
        {
          sub_B2C6D0(v23, (__int64)v15, v28, v29);
          v30 = *(__int64 ***)(v23 + 96);
          v31 = &v30[5 * *(_QWORD *)(v23 + 104)];
          if ( (*(_BYTE *)(v23 + 2) & 1) != 0 )
          {
            sub_B2C6D0(v23, (__int64)v15, 5LL * *(_QWORD *)(v23 + 104), v127);
            v30 = *(__int64 ***)(v23 + 96);
          }
        }
        else
        {
          v30 = *(__int64 ***)(v23 + 96);
          v31 = &v30[5 * *(_QWORD *)(v23 + 104)];
        }
        for ( i = v30; v31 != i; i += 5 )
        {
          v15 = i;
          sub_2A6B770(v247, v15);
        }
LABEL_44:
        v22 = (__int64 *)v22[1];
        if ( v22 == v232 )
          break;
      }
    }
  }
  v33 = a1[2];
  for ( j = (__int64)(a1 + 1); v33 != j; v33 = *(_QWORD *)(v33 + 8) )
  {
    while ( 1 )
    {
      v34 = v33 - 56;
      if ( !v33 )
        v34 = 0;
      sub_AD0030(v34);
      if ( (unsigned __int8)sub_310F8B0(v34) )
        break;
      v33 = *(_QWORD *)(v33 + 8);
      if ( v33 == j )
        goto LABEL_52;
    }
    sub_2A69B70(v247, v34);
  }
LABEL_52:
  sub_2A72BA0(v247, a1);
  if ( a9 )
  {
    v143 = 0;
    do
    {
      if ( (unsigned int)qword_4FF6028 <= v143 )
        break;
      ++v143;
    }
    while ( (unsigned __int8)sub_317BE50(&v282) );
  }
  v234 = a1[4];
  if ( (__int64 *)v234 != v232 )
  {
    v236 = 0;
    while ( 1 )
    {
      v35 = 0;
      if ( v234 )
        v35 = v234 - 56;
      v244 = v35;
      v36 = v35;
      v37 = sub_B2FC80(v35);
      if ( !v37 )
        break;
LABEL_55:
      v234 = *(_QWORD *)(v234 + 8);
      if ( v232 == (__int64 *)v234 )
        goto LABEL_141;
    }
    v318 = (__int64 *)&v320;
    v319 = 0x20000000000LL;
    v38 = *(_QWORD *)(v36 + 80);
    if ( v38 )
      v38 -= 24;
    if ( (unsigned __int8)sub_2A64220(v247, v38) )
    {
      if ( (*(_BYTE *)(v244 + 2) & 1) != 0 )
      {
        sub_B2C6D0(v244, v38, v39, v40);
        v128 = *(_QWORD *)(v244 + 96);
        v129 = v128 + 40LL * *(_QWORD *)(v244 + 104);
        if ( (*(_BYTE *)(v244 + 2) & 1) != 0 )
        {
          sub_B2C6D0(v244, v38, v214, v215);
          v128 = *(_QWORD *)(v244 + 96);
        }
      }
      else
      {
        v128 = *(_QWORD *)(v244 + 96);
        v129 = v128 + 40LL * *(_QWORD *)(v244 + 104);
      }
      if ( v129 != v128 )
      {
        do
        {
          if ( *(_QWORD *)(v128 + 16) && (unsigned __int8)sub_2A66C70(v247, v128) )
            v37 |= *(_BYTE *)(*(_QWORD *)(v128 + 8) + 8LL) == 14;
          v128 += 40;
        }
        while ( v128 != v129 );
        v236 |= v37;
        if ( v37 )
        {
          v260 = *(_QWORD **)(v244 + 120);
          v130 = sub_A746F0(&v260);
          if ( v130 == 255 )
          {
            v131 = (unsigned __int64)v260;
          }
          else
          {
            v223 = v130 | ((v130 & 3) << 6) | (16 * (v130 & 3));
            v224 = (__int64 *)sub_B2BE50(v244);
            v225 = sub_A77AB0(v224, v223);
            v226 = (__int64 *)sub_B2BE50(v244);
            v131 = sub_A7B440((__int64 *)&v260, v226, -1, v225);
          }
          v132 = *(_QWORD *)(v244 + 16);
          for ( *(_QWORD *)(v244 + 120) = v131; v132; v132 = *(_QWORD *)(v132 + 8) )
          {
            v134 = *(unsigned __int8 **)(v132 + 24);
            v135 = *v134;
            if ( (unsigned __int8)v135 > 0x1Cu )
            {
              v136 = (unsigned int)(v135 - 34);
              if ( (unsigned __int8)v136 <= 0x33u )
              {
                v137 = 0x8000000000041LL;
                if ( _bittest64(&v137, v136) )
                {
                  v138 = *((_QWORD *)v134 - 4);
                  if ( v138 )
                  {
                    if ( !*(_BYTE *)v138 && *((_QWORD *)v134 + 10) == *(_QWORD *)(v138 + 24) && v138 == v244 )
                    {
                      v260 = (_QWORD *)*((_QWORD *)v134 + 9);
                      v139 = sub_A746F0(&v260);
                      if ( v139 == 255 )
                      {
                        v133 = (unsigned __int64)v260;
                      }
                      else
                      {
                        v237 = v139 | ((v139 & 3) << 6) | (16 * (v139 & 3));
                        v140 = (__int64 *)sub_B2BE50(v244);
                        v238 = sub_A77AB0(v140, v237);
                        v141 = (__int64 *)sub_B2BE50(v244);
                        v133 = sub_A7B440((__int64 *)&v260, v141, -1, v238);
                      }
                      *((_QWORD *)v134 + 9) = v133;
                    }
                  }
                }
              }
            }
          }
          v236 = v37;
        }
      }
    }
    v255 = 0;
    v256 = (unsigned __int64)&v259;
    v257 = 32;
    LODWORD(v258) = 0;
    BYTE4(v258) = 1;
    v41 = *(_QWORD *)(v244 + 80);
    v240 = v244 + 72;
    if ( v41 != v244 + 72 )
    {
      v42 = v236;
      do
      {
        while ( 1 )
        {
          v46 = v41 - 24;
          if ( !v41 )
            v46 = 0;
          v47 = v46;
          if ( !(unsigned __int8)sub_2A64220(v247, v46) )
            break;
          v50 = sub_2A66DA0(v247, v47, &v255, &unk_4FF6069, &unk_4FF6068);
          v41 = *(_QWORD *)(v41 + 8);
          v42 |= v50;
          if ( v41 == v240 )
            goto LABEL_75;
        }
        v42 = 1;
        v43 = *(_QWORD *)(v244 + 80);
        if ( v43 )
          v43 -= 24;
        if ( v43 != v47 )
        {
          v44 = (unsigned int)v319;
          v45 = (unsigned int)v319 + 1LL;
          if ( v45 > HIDWORD(v319) )
          {
            sub_C8D5F0((__int64)&v318, &v320, v45, 8u, v48, v49);
            v44 = (unsigned int)v319;
          }
          v42 = 1;
          v318[v44] = v47;
          LODWORD(v319) = v319 + 1;
        }
        v41 = *(_QWORD *)(v41 + 8);
      }
      while ( v41 != v240 );
LABEL_75:
      v236 = v42;
    }
    v51 = *(unsigned int *)(a3 + 88);
    v52 = *(_QWORD *)(a3 + 72);
    if ( (_DWORD)v51 )
    {
      v53 = 1;
      v54 = v51 - 1;
      v228 = (unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32;
      for ( k = (v51 - 1)
              & (((0xBF58476D1CE4E5B9LL * (v228 | ((unsigned int)v244 >> 9) ^ ((unsigned int)v244 >> 4))) >> 31)
               ^ (484763065 * (v228 | ((unsigned int)v244 >> 9) ^ ((unsigned int)v244 >> 4)))); ; k = v54 & v57 )
      {
        v56 = v52 + 24LL * k;
        if ( *(_UNKNOWN **)v56 == &unk_4F81450 && v244 == *(_QWORD *)(v56 + 8) )
          break;
        if ( *(_QWORD *)v56 == -4096 && *(_QWORD *)(v56 + 8) == -4096 )
          goto LABEL_84;
        v57 = v53 + k;
        ++v53;
      }
      v122 = v52 + 24 * v51;
      if ( v122 != v56 )
      {
        v58 = *(_QWORD *)(*(_QWORD *)(v56 + 16) + 24LL);
        if ( v58 )
          v58 += 8;
        goto LABEL_190;
      }
    }
    else
    {
LABEL_84:
      v56 = v52 + 24LL * (unsigned int)v51;
      if ( !(_DWORD)v51 )
      {
        v58 = 0;
        goto LABEL_86;
      }
      v54 = v51 - 1;
    }
    v122 = v56;
    v58 = 0;
LABEL_190:
    v123 = 1;
    v227 = (unsigned __int64)(((unsigned int)&unk_4F8FBC8 >> 9) ^ ((unsigned int)&unk_4F8FBC8 >> 4)) << 32;
    for ( m = v54
            & (((0xBF58476D1CE4E5B9LL * (v227 | ((unsigned int)v244 >> 9) ^ ((unsigned int)v244 >> 4))) >> 31)
             ^ (484763065 * (v227 | ((unsigned int)v244 >> 9) ^ ((unsigned int)v244 >> 4)))); ; m = v54 & v126 )
    {
      v125 = v52 + 24LL * m;
      if ( *(_UNKNOWN **)v125 == &unk_4F8FBC8 && v244 == *(_QWORD *)(v125 + 8) )
        break;
      if ( *(_QWORD *)v125 == -4096 && *(_QWORD *)(v125 + 8) == -4096 )
        goto LABEL_86;
      v126 = v123 + m;
      ++v123;
    }
    if ( v125 != v122 )
    {
      v59 = *(_QWORD *)(*(_QWORD *)(v125 + 16) + 24LL);
      if ( v59 )
        v59 += 8;
      goto LABEL_87;
    }
LABEL_86:
    v59 = 0;
LABEL_87:
    v270 = v59;
    v60 = v318;
    v273 = &v277;
    v260 = &v262;
    v61 = &v318[(unsigned int)v319];
    v269 = v58;
    v261 = 0x1000000000LL;
    v267 = 0;
    v268 = 0;
    v271 = 1;
    v272 = 0;
    v274 = 8;
    v275 = 0;
    v276 = 1;
    v278 = 0;
    v279 = 0;
    v280 = 0;
    v281 = 0;
    if ( v61 != v318 )
    {
      do
      {
        v62 = sub_AA5030(*v60, 1);
        if ( v62 )
          v62 -= 24;
        ++v60;
        sub_F55BE0(v62, 0, (__int64)&v260, 0, v63, v64);
      }
      while ( v61 != v60 );
    }
    v65 = *(_QWORD **)(v244 + 80);
    if ( v65 )
      v65 -= 3;
    if ( !(unsigned __int8)sub_2A64220(v247, v65) )
    {
      v70 = *(_QWORD *)(v244 + 80);
      if ( v70 )
        v70 -= 24;
      v71 = sub_AA5030(v70, 1);
      if ( v71 )
        v71 -= 24;
      v65 = 0;
      sub_F55BE0(v71, 0, (__int64)&v260, 0, v72, v73);
    }
    v251[0] = 0;
    if ( v240 == *(_QWORD *)(v244 + 80) )
    {
      v76 = (unsigned __int64)v318;
      v77 = &v318[(unsigned int)v319];
      if ( v77 != v318 )
        goto LABEL_107;
    }
    else
    {
      v74 = *(_QWORD *)(v244 + 80);
      do
      {
        v65 = (_QWORD *)(v74 - 24);
        if ( !v74 )
          v65 = 0;
        v75 = sub_2A64290(v247, v65, &v260, v251);
        v74 = *(_QWORD *)(v74 + 8);
        v236 |= v75;
      }
      while ( v240 != v74 );
      v76 = (unsigned __int64)v318;
      v77 = &v318[(unsigned int)v319];
      if ( v77 != v318 )
      {
        do
        {
LABEL_107:
          while ( 1 )
          {
            v65 = *(_QWORD **)v76;
            if ( (*(_WORD *)(*(_QWORD *)v76 + 2LL) & 0x7FFF) == 0 )
              break;
            v76 += 8LL;
            if ( v77 == (__int64 *)v76 )
              goto LABEL_109;
          }
          v76 += 8LL;
          sub_FFBF00((__int64)&v260, v65);
        }
        while ( v77 != (__int64 *)v76 );
      }
LABEL_109:
      for ( n = *(_QWORD *)(v244 + 80); v240 != n; n = *(_QWORD *)(n + 8) )
      {
        if ( !n )
          BUG();
        v78 = *(_QWORD *)(n + 32);
        while ( n + 24 != v78 )
        {
          v79 = v78;
          v78 = *(_QWORD *)(v78 + 8);
          v65 = (_QWORD *)(v79 - 24);
          if ( sub_2A63F50(v247, v79 - 24) )
          {
            if ( *(_BYTE *)(v79 - 24) == 85 )
            {
              v88 = *(_QWORD *)(v79 - 56);
              if ( v88 )
              {
                if ( !*(_BYTE *)v88 )
                {
                  v67 = *(_QWORD *)(v79 + 56);
                  if ( *(_QWORD *)(v88 + 24) == v67
                    && (*(_BYTE *)(v88 + 33) & 0x20) != 0
                    && *(_DWORD *)(v88 + 36) == 336 )
                  {
                    v65 = *(_QWORD **)(v79 - 32LL * (*(_DWORD *)(v79 - 20) & 0x7FFFFFF) - 24);
                    sub_BD84D0(v79 - 24, (__int64)v65);
                    sub_B43D60((_QWORD *)(v79 - 24));
                  }
                }
              }
            }
          }
        }
      }
    }
    sub_FFCE90((__int64)&v260, (__int64)v65, v66, v67, v68, v69);
    sub_FFD870((__int64)&v260, (__int64)v65, v80, v81, v82, v83);
    sub_FFBC40((__int64)&v260, (__int64)v65);
    v84 = v280;
    v85 = v279;
    if ( v280 != v279 )
    {
      do
      {
        v86 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v85[7];
        *v85 = &unk_49E5048;
        if ( v86 )
          v86(v85 + 5, v85 + 5, 3);
        *v85 = &unk_49DB368;
        v87 = v85[3];
        if ( v87 != -4096 && v87 != 0 && v87 != -8192 )
          sub_BD60C0(v85 + 1);
        v85 += 9;
      }
      while ( v84 != v85 );
      v85 = v279;
    }
    if ( v85 )
      j_j___libc_free_0((unsigned __int64)v85);
    if ( !v276 )
      _libc_free((unsigned __int64)v273);
    if ( v260 != &v262 )
      _libc_free((unsigned __int64)v260);
    if ( !BYTE4(v258) )
      _libc_free(v256);
    if ( v318 != (__int64 *)&v320 )
      _libc_free((unsigned __int64)v318);
    goto LABEL_55;
  }
  v236 = 0;
LABEL_141:
  v255 = &v257;
  v256 = 0x800000000LL;
  sub_2A654F0(v247);
  sub_2A65120(v247);
  v89 = sub_2A654E0(v247);
  v90 = *(__int64 **)(v89 + 32);
  v91 = &v90[6 * *(unsigned int *)(v89 + 40)];
  while ( v90 != v91 )
  {
    while ( 1 )
    {
      if ( (unsigned __int8)sub_2A62D90(v90 + 1) || *((_BYTE *)v90 + 8) <= 1u )
      {
        v92 = *v90;
        if ( (unsigned __int8)sub_2A641A0(v247, *v90) )
        {
          if ( !(unsigned __int8)sub_2A640C0(v247, v92) )
            break;
        }
      }
      v90 += 6;
      if ( v90 == v91 )
        goto LABEL_149;
    }
    v90 += 6;
    sub_26B72C0(v92, (__int64)&v255);
  }
LABEL_149:
  v93 = (__int64)v247;
  sub_2A65550(&v318, v247);
  v94 = (__int64 *)v319;
  if ( BYTE4(v321) )
  {
    v95 = (__int64 *)(v319 + 8LL * HIDWORD(v320));
    if ( (__int64 *)v319 == v95 )
      goto LABEL_154;
    v96 = *(_QWORD *)v319;
    v97 = v319;
    if ( *(_QWORD *)v319 >= 0xFFFFFFFFFFFFFFFELL )
      goto LABEL_152;
  }
  else
  {
    v95 = (__int64 *)(v319 + 8LL * (unsigned int)v320);
    if ( (__int64 *)v319 == v95 )
      goto LABEL_234;
    while ( 1 )
    {
      v96 = *v94;
      v97 = (unsigned __int64)v94;
      if ( (unsigned __int64)*v94 < 0xFFFFFFFFFFFFFFFELL )
        break;
LABEL_152:
      if ( v95 == ++v94 )
        goto LABEL_153;
    }
  }
  if ( v94 == v95 )
  {
LABEL_153:
    if ( BYTE4(v321) )
      goto LABEL_154;
    goto LABEL_234;
  }
  do
  {
    v93 = v96;
    if ( (unsigned __int8)sub_2A65580(v247, v96, **(_QWORD **)(*(_QWORD *)(v96 + 24) + 16LL)) )
    {
      v93 = v96;
      if ( (unsigned __int8)sub_2A641A0(v247, v96) )
      {
        v93 = v96;
        if ( !(unsigned __int8)sub_2A640C0(v247, v96) )
        {
          v93 = (__int64)&v255;
          sub_26B72C0(v96, (__int64)&v255);
        }
      }
    }
    v142 = (__int64 *)(v97 + 8);
    if ( (__int64 *)(v97 + 8) == v95 )
      break;
    while ( 1 )
    {
      v96 = *v142;
      v97 = (unsigned __int64)v142;
      if ( (unsigned __int64)*v142 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v95 == ++v142 )
        goto LABEL_233;
    }
  }
  while ( v142 != v95 );
LABEL_233:
  if ( !BYTE4(v321) )
LABEL_234:
    _libc_free(v319);
LABEL_154:
  v98 = (unsigned __int64)v255;
  v260 = 0;
  v264 = v266;
  v265 = 0x800000000LL;
  v261 = 0;
  v262 = 0;
  v99 = &v255[(unsigned int)v256];
  v263 = 0;
  if ( v99 != v255 )
  {
    while ( 1 )
    {
      v100 = *(_QWORD *)v98;
      v101 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v98 + 40LL) + 72LL);
      v102 = sub_ACADE0(**(__int64 ****)(*(_QWORD *)(v101 + 24) + 16LL));
      v105 = v100 - 32LL * (*(_DWORD *)(v100 + 4) & 0x7FFFFFF);
      if ( *(_QWORD *)v105 )
      {
        v106 = *(_QWORD *)(v105 + 8);
        **(_QWORD **)(v105 + 16) = v106;
        if ( v106 )
          *(_QWORD *)(v106 + 16) = *(_QWORD *)(v105 + 16);
      }
      *(_QWORD *)v105 = v102;
      if ( v102 )
      {
        v107 = *(_QWORD *)(v102 + 16);
        *(_QWORD *)(v105 + 8) = v107;
        if ( v107 )
          *(_QWORD *)(v107 + 16) = v105 + 8;
        *(_QWORD *)(v105 + 16) = v102 + 16;
        *(_QWORD *)(v102 + 16) = v105;
      }
      if ( (_DWORD)v262 )
        break;
      v108 = v264;
      v109 = 8LL * (unsigned int)v265;
      v93 = (__int64)&v264[v109];
      v110 = v109 >> 3;
      v111 = v109 >> 5;
      if ( !v111 )
        goto LABEL_277;
      v112 = &v264[32 * v111];
      do
      {
        if ( v101 == *v108 )
          goto LABEL_170;
        if ( v101 == v108[1] )
        {
          ++v108;
          goto LABEL_170;
        }
        if ( v101 == v108[2] )
        {
          v108 += 2;
          goto LABEL_170;
        }
        if ( v101 == v108[3] )
        {
          v108 += 3;
LABEL_170:
          if ( (_QWORD *)v93 == v108 )
            goto LABEL_281;
          goto LABEL_171;
        }
        v108 += 4;
      }
      while ( v112 != v108 );
      v110 = (v93 - (__int64)v108) >> 3;
LABEL_277:
      if ( v110 == 2 )
        goto LABEL_306;
      if ( v110 != 3 )
      {
        if ( v110 == 1 )
          goto LABEL_280;
        goto LABEL_281;
      }
      if ( v101 == *v108 )
        goto LABEL_170;
      ++v108;
LABEL_306:
      if ( v101 == *v108 )
        goto LABEL_170;
      ++v108;
LABEL_280:
      if ( v101 == *v108 )
        goto LABEL_170;
LABEL_281:
      if ( (unsigned __int64)(unsigned int)v265 + 1 > HIDWORD(v265) )
      {
        sub_C8D5F0((__int64)&v264, v266, (unsigned int)v265 + 1LL, 8u, v103, v104);
        v93 = (__int64)&v264[8 * (unsigned int)v265];
      }
      *(_QWORD *)v93 = v101;
      v171 = (unsigned int)(v265 + 1);
      LODWORD(v265) = v171;
      if ( (unsigned int)v171 > 8 )
      {
        v172 = (__int64 *)v264;
        v173 = (__int64 *)&v264[8 * v171];
        while ( 1 )
        {
          v93 = (unsigned int)v263;
          if ( !(_DWORD)v263 )
            break;
          LODWORD(v174) = (v263 - 1) & (((unsigned int)*v172 >> 9) ^ ((unsigned int)*v172 >> 4));
          v175 = (__int64 *)(v261 + 8LL * (unsigned int)v174);
          v176 = *v175;
          if ( *v172 != *v175 )
          {
            v213 = 1;
            v178 = 0;
            while ( v176 != -4096 )
            {
              if ( !v178 && v176 == -8192 )
                v178 = v175;
              v174 = ((_DWORD)v263 - 1) & (unsigned int)(v174 + v213);
              v175 = (__int64 *)(v261 + 8 * v174);
              v176 = *v175;
              if ( *v172 == *v175 )
                goto LABEL_286;
              ++v213;
            }
            if ( !v178 )
              v178 = v175;
            v260 = (_QWORD *)((char *)v260 + 1);
            v180 = (_DWORD)v262 + 1;
            if ( 4 * ((int)v262 + 1) < (unsigned int)(3 * v263) )
            {
              if ( (int)v263 - HIDWORD(v262) - v180 <= (unsigned int)v263 >> 3 )
              {
                sub_A35F10((__int64)&v260, v263);
                if ( !(_DWORD)v263 )
                  goto LABEL_453;
                v93 = 0;
                v216 = 1;
                LODWORD(v217) = (v263 - 1) & (((unsigned int)*v172 >> 9) ^ ((unsigned int)*v172 >> 4));
                v178 = (__int64 *)(v261 + 8LL * (unsigned int)v217);
                v218 = *v178;
                v180 = (_DWORD)v262 + 1;
                if ( *v172 != *v178 )
                {
                  while ( v218 != -4096 )
                  {
                    if ( !v93 && v218 == -8192 )
                      v93 = (__int64)v178;
                    v217 = ((_DWORD)v263 - 1) & (unsigned int)(v217 + v216);
                    v178 = (__int64 *)(v261 + 8 * v217);
                    v218 = *v178;
                    if ( *v172 == *v178 )
                      goto LABEL_395;
                    ++v216;
                  }
LABEL_293:
                  if ( v93 )
                    v178 = (__int64 *)v93;
                }
              }
LABEL_395:
              LODWORD(v262) = v180;
              if ( *v178 != -4096 )
                --HIDWORD(v262);
              *v178 = *v172;
              goto LABEL_286;
            }
LABEL_289:
            v93 = (unsigned int)(2 * v263);
            sub_A35F10((__int64)&v260, v93);
            if ( !(_DWORD)v263 )
              goto LABEL_453;
            LODWORD(v177) = (v263 - 1) & (((unsigned int)*v172 >> 9) ^ ((unsigned int)*v172 >> 4));
            v178 = (__int64 *)(v261 + 8LL * (unsigned int)v177);
            v179 = *v178;
            v180 = (_DWORD)v262 + 1;
            if ( *v172 != *v178 )
            {
              v181 = 1;
              v93 = 0;
              while ( v179 != -4096 )
              {
                if ( !v93 && v179 == -8192 )
                  v93 = (__int64)v178;
                v177 = ((_DWORD)v263 - 1) & (unsigned int)(v177 + v181);
                v178 = (__int64 *)(v261 + 8 * v177);
                v179 = *v178;
                if ( *v172 == *v178 )
                  goto LABEL_395;
                ++v181;
              }
              goto LABEL_293;
            }
            goto LABEL_395;
          }
LABEL_286:
          if ( v173 == ++v172 )
            goto LABEL_171;
        }
        v260 = (_QWORD *)((char *)v260 + 1);
        goto LABEL_289;
      }
LABEL_171:
      v98 += 8LL;
      if ( v99 == (__int64 *)v98 )
        goto LABEL_172;
    }
    v93 = (unsigned int)v263;
    if ( (_DWORD)v263 )
    {
      v161 = (unsigned int)(v263 - 1);
      v162 = v261;
      v163 = 1;
      v164 = 0;
      LODWORD(v165) = v161 & (((unsigned int)v101 >> 9) ^ ((unsigned int)v101 >> 4));
      v166 = (__int64 *)(v261 + 8LL * (unsigned int)v165);
      v167 = *v166;
      if ( v101 == *v166 )
        goto LABEL_171;
      while ( v167 != -4096 )
      {
        if ( v167 == -8192 && !v164 )
          v164 = v166;
        v165 = (unsigned int)v161 & ((_DWORD)v165 + v163);
        v166 = (__int64 *)(v261 + 8 * v165);
        v167 = *v166;
        if ( v101 == *v166 )
          goto LABEL_171;
        ++v163;
      }
      if ( v164 )
        v166 = v164;
      v168 = (_DWORD)v262 + 1;
      v260 = (_QWORD *)((char *)v260 + 1);
      if ( 4 * ((int)v262 + 1) < (unsigned int)(3 * v263) )
      {
        if ( (int)v263 - HIDWORD(v262) - v168 <= (unsigned int)v263 >> 3 )
        {
          sub_A35F10((__int64)&v260, v263);
          if ( !(_DWORD)v263 )
          {
LABEL_453:
            LODWORD(v262) = (_DWORD)v262 + 1;
            BUG();
          }
          v161 = v261;
          v162 = 1;
          v221 = 0;
          v222 = (v263 - 1) & (((unsigned int)v101 >> 9) ^ ((unsigned int)v101 >> 4));
          v166 = (__int64 *)(v261 + 8LL * v222);
          v93 = *v166;
          v168 = (_DWORD)v262 + 1;
          if ( v101 != *v166 )
          {
            while ( v93 != -4096 )
            {
              if ( v93 == -8192 && !v221 )
                v221 = v166;
              v222 = (v263 - 1) & (v162 + v222);
              v166 = (__int64 *)(v261 + 8LL * v222);
              v93 = *v166;
              if ( v101 == *v166 )
                goto LABEL_271;
              v162 = (unsigned int)(v162 + 1);
            }
            if ( v221 )
              v166 = v221;
          }
        }
        goto LABEL_271;
      }
    }
    else
    {
      v260 = (_QWORD *)((char *)v260 + 1);
    }
    v93 = (unsigned int)(2 * v263);
    sub_A35F10((__int64)&v260, v93);
    if ( !(_DWORD)v263 )
      goto LABEL_453;
    v161 = (unsigned int)(v263 - 1);
    v219 = v161 & (((unsigned int)v101 >> 9) ^ ((unsigned int)v101 >> 4));
    v166 = (__int64 *)(v261 + 8LL * v219);
    v220 = *v166;
    v168 = (_DWORD)v262 + 1;
    if ( v101 != *v166 )
    {
      v162 = 1;
      v93 = 0;
      while ( v220 != -4096 )
      {
        if ( !v93 && v220 == -8192 )
          v93 = (__int64)v166;
        v219 = v161 & (v162 + v219);
        v166 = (__int64 *)(v261 + 8LL * v219);
        v220 = *v166;
        if ( v101 == *v166 )
          goto LABEL_271;
        v162 = (unsigned int)(v162 + 1);
      }
      if ( v93 )
        v166 = (__int64 *)v93;
    }
LABEL_271:
    LODWORD(v262) = v168;
    if ( *v166 != -4096 )
      --HIDWORD(v262);
    *v166 = v101;
    v169 = (unsigned int)v265;
    v170 = (unsigned int)v265 + 1LL;
    if ( v170 > HIDWORD(v265) )
    {
      v93 = (__int64)v266;
      sub_C8D5F0((__int64)&v264, v266, v170, 8u, v162, v161);
      v169 = (unsigned int)v265;
    }
    *(_QWORD *)&v264[8 * v169] = v101;
    LODWORD(v265) = v265 + 1;
    goto LABEL_171;
  }
LABEL_172:
  sub_A753E0((__int64)v251);
  v113 = (unsigned int)v265;
  v114 = (__int64)&v264[8 * (unsigned int)v265];
  v233 = (__int64 *)v114;
  if ( (_BYTE *)v114 != v264 )
  {
    v235 = (__int64 *)v264;
    do
    {
      v115 = *v235;
      if ( (*(_BYTE *)(*v235 + 2) & 1) != 0 )
      {
        sub_B2C6D0(*v235, v93, v113, v114);
        v116 = *(_QWORD *)(v115 + 96);
        v117 = v116 + 40LL * *(_QWORD *)(v115 + 104);
        if ( (*(_BYTE *)(v115 + 2) & 1) != 0 )
        {
          sub_B2C6D0(v115, v93, v159, v160);
          v116 = *(_QWORD *)(v115 + 96);
        }
      }
      else
      {
        v116 = *(_QWORD *)(v115 + 96);
        v117 = v116 + 40LL * *(_QWORD *)(v115 + 104);
      }
      while ( v116 != v117 )
      {
        v118 = *(_DWORD *)(v116 + 32);
        v116 += 40;
        sub_B2D580(v115, v118, 52);
      }
      v93 = (__int64)v251;
      sub_B2D550(v115, (__int64)v251);
      for ( ii = *(_QWORD *)(v115 + 16); ii; ii = *(_QWORD *)(ii + 8) )
      {
        v119 = *(unsigned __int8 **)(ii + 24);
        v120 = *v119;
        if ( (unsigned __int8)v120 > 0x1Cu )
        {
          v113 = (unsigned int)(v120 - 34);
          if ( (unsigned __int8)(v120 - 34) <= 0x33u )
          {
            v114 = 0x8000000000041LL;
            if ( _bittest64(&v114, v113) )
            {
              if ( v120 == 40 )
              {
                v121 = -32 - 32LL * (unsigned int)sub_B491D0(*(_QWORD *)(ii + 24));
              }
              else
              {
                v121 = -32;
                if ( v120 != 85 )
                {
                  if ( v120 != 34 )
                    goto LABEL_454;
                  v121 = -96;
                }
              }
              if ( (v119[7] & 0x80u) != 0 )
              {
                v144 = sub_BD2BC0((__int64)v119);
                v146 = v144 + v145;
                if ( (v119[7] & 0x80u) == 0 )
                {
                  if ( (unsigned int)(v146 >> 4) )
LABEL_454:
                    BUG();
                }
                else if ( (unsigned int)((v146 - sub_BD2BC0((__int64)v119)) >> 4) )
                {
                  if ( (v119[7] & 0x80u) == 0 )
                    goto LABEL_454;
                  v147 = *(_DWORD *)(sub_BD2BC0((__int64)v119) + 8);
                  if ( (v119[7] & 0x80u) == 0 )
                    BUG();
                  v148 = sub_BD2BC0((__int64)v119);
                  v121 -= 32LL * (unsigned int)(*(_DWORD *)(v148 + v149 - 4) - v147);
                }
              }
              v150 = &v119[v121];
              v151 = *((_DWORD *)v119 + 1) & 0x7FFFFFF;
              v152 = &v119[-32 * v151];
              if ( v152 != v150 )
              {
                while ( 1 )
                {
                  v153 = (__int64 *)sub_BD5C60((__int64)v119);
                  v154 = v152;
                  v152 += 32;
                  *((_QWORD *)v119 + 9) = sub_A7B980(
                                            (__int64 *)v119 + 9,
                                            v153,
                                            (unsigned int)((v154 - &v119[-32 * v151]) >> 5) + 1,
                                            52);
                  if ( v150 == v152 )
                    break;
                  v151 = *((_DWORD *)v119 + 1) & 0x7FFFFFF;
                }
              }
              v93 = sub_BD5C60((__int64)v119);
              *((_QWORD *)v119 + 9) = sub_A7A440((__int64 *)v119 + 9, (__int64 *)v93, 0, (__int64)v251);
            }
          }
        }
      }
      ++v235;
    }
    while ( v233 != v235 );
  }
  v155 = sub_2A65540(v247);
  v156 = *(__int64 **)(v155 + 8);
  v157 = &v156[6 * *(unsigned int *)(v155 + 24)];
  if ( *(_DWORD *)(v155 + 16) && v156 != v157 )
  {
    while ( *v156 == -8192 || *v156 == -4096 )
    {
      v156 += 6;
      if ( v157 == v156 )
        goto LABEL_254;
    }
    if ( v156 != v157 )
    {
      v182 = v236;
      while ( 1 )
      {
        for ( jj = v156 + 6; v157 != jj; jj += 6 )
        {
          if ( *jj != -8192 && *jj != -4096 )
            break;
        }
        v184 = *v156;
        if ( (unsigned __int8)sub_2A62E90(v156 + 1) )
        {
          if ( v157 == jj )
            goto LABEL_330;
        }
        else
        {
          v185 = *(_QWORD *)(v184 + 16);
          while ( v185 )
          {
            v186 = v185;
            v185 = *(_QWORD *)(v185 + 8);
            sub_B43D60(*(_QWORD **)(v186 + 24));
          }
          v248 = (__m128i **)v250;
          v249 = 0x100000000LL;
          sub_B92230(v184, (__int64)&v248);
          if ( (_DWORD)v249 == 1 )
          {
            sub_AE0470((__int64)&v318, a1, 1, 0);
            v191 = (unsigned __int8 *)sub_F57DF0((__int64)&v318, *(char **)(v184 - 32), *(_QWORD *)(v184 + 24));
            if ( v191 )
              sub_BA6610(*v248, 1u, v191);
            v192 = v343;
            if ( v343 )
            {
              v241 = v184;
              v239 = v157;
              v193 = v342;
              v194 = v342 + 56LL * v343;
              do
              {
                if ( *(_QWORD *)v193 != -4096 && *(_QWORD *)v193 != -8192 )
                {
                  v195 = *(_QWORD *)(v193 + 8);
                  v196 = v195 + 8LL * *(unsigned int *)(v193 + 16);
                  if ( v195 != v196 )
                  {
                    do
                    {
                      v197 = *(_QWORD *)(v196 - 8);
                      v196 -= 8LL;
                      if ( v197 )
                        sub_B91220(v196, v197);
                    }
                    while ( v195 != v196 );
                    v196 = *(_QWORD *)(v193 + 8);
                  }
                  if ( v196 != v193 + 24 )
                    _libc_free(v196);
                }
                v193 += 56;
              }
              while ( v194 != v193 );
              v184 = v241;
              v157 = v239;
              v192 = v343;
            }
            sub_C7D6A0(v342, 56 * v192, 8);
            v198 = &v339[8 * v340];
            if ( v339 != v198 )
            {
              v199 = v339;
              do
              {
                v200 = *((_QWORD *)v198 - 1);
                v198 -= 8;
                if ( v200 )
                  sub_B91220((__int64)v198, v200);
              }
              while ( v199 != v198 );
              v198 = v339;
            }
            if ( v198 != &v341 )
              _libc_free((unsigned __int64)v198);
            v201 = v337;
            v202 = v337 + 56LL * v338;
            if ( v337 != v202 )
            {
              do
              {
                v202 -= 56LL;
                v203 = *(_QWORD *)(v202 + 40);
                if ( v203 != v202 + 56 )
                  _libc_free(v203);
                sub_C7D6A0(*(_QWORD *)(v202 + 16), 8LL * *(unsigned int *)(v202 + 32), 8);
              }
              while ( v201 != v202 );
              v202 = v337;
            }
            if ( (char **)v202 != &v339 )
              _libc_free(v202);
            sub_C7D6A0(v335, 16LL * v336, 8);
            v204 = &v332[8 * v333];
            if ( v332 != v204 )
            {
              v205 = v332;
              do
              {
                v206 = *((_QWORD *)v204 - 1);
                v204 -= 8;
                if ( v206 )
                  sub_B91220((__int64)v204, v206);
              }
              while ( v205 != v204 );
              v204 = v332;
            }
            if ( v204 != &v334 )
              _libc_free((unsigned __int64)v204);
            if ( v330 != &v331 )
              _libc_free((unsigned __int64)v330);
            if ( v328 != &v329 )
              _libc_free((unsigned __int64)v328);
            v207 = &v325[8 * v326];
            if ( v325 != v207 )
            {
              v208 = v325;
              do
              {
                v209 = *((_QWORD *)v207 - 1);
                v207 -= 8;
                if ( v209 )
                  sub_B91220((__int64)v207, v209);
              }
              while ( v208 != v207 );
              v207 = v325;
            }
            if ( v207 != &v327 )
              _libc_free((unsigned __int64)v207);
            v210 = &v322[8 * v323];
            if ( v322 != v210 )
            {
              v211 = v322;
              do
              {
                v212 = *((_QWORD *)v210 - 1);
                v210 -= 8;
                if ( v212 )
                  sub_B91220((__int64)v210, v212);
              }
              while ( v211 != v210 );
              v210 = v322;
            }
            if ( v210 != &v324 )
              _libc_free((unsigned __int64)v210);
          }
          sub_BA85F0(j, v184);
          v187 = *(unsigned __int64 **)(v184 + 64);
          v188 = *(_QWORD *)(v184 + 56) & 0xFFFFFFFFFFFFFFF8LL;
          *v187 = v188 | *v187 & 7;
          *(_QWORD *)(v188 + 8) = v187;
          *(_QWORD *)(v184 + 56) &= 7uLL;
          *(_QWORD *)(v184 + 64) = 0;
          sub_B30220(v184);
          *(_DWORD *)(v184 + 4) = *(_DWORD *)(v184 + 4) & 0xF8000000 | 1;
          sub_B2F9E0(v184, v184, v189, v190);
          sub_BD2DD0(v184);
          if ( v248 != (__m128i **)v250 )
            _libc_free((unsigned __int64)v248);
          v182 = 1;
          if ( v157 == jj )
          {
LABEL_330:
            v236 = v182;
            break;
          }
        }
        v156 = jj;
      }
    }
  }
LABEL_254:
  sub_26B7540(v254);
  if ( v264 != v266 )
    _libc_free((unsigned __int64)v264);
  sub_C7D6A0(v261, 8LL * (unsigned int)v263, 8);
  if ( v255 != &v257 )
    _libc_free((unsigned __int64)v255);
  sub_3176310(&v282);
  sub_2A665E0(v247);
  return v236;
}
