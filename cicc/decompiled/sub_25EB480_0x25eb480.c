// Function: sub_25EB480
// Address: 0x25eb480
//
__int64 __fastcall sub_25EB480(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 i; // rax
  unsigned int v13; // eax
  _QWORD *v14; // rbx
  char v15; // al
  _QWORD *v16; // rax
  _QWORD *v17; // r12
  _QWORD *j; // r14
  __int64 v19; // r13
  __int64 v20; // r9
  _QWORD *v21; // rax
  _QWORD *v22; // r13
  _QWORD *v23; // rbx
  __int64 v24; // r14
  __int64 v25; // rcx
  __int64 v26; // rsi
  __int64 v27; // r8
  __int64 v28; // r9
  char v29; // al
  __int64 v30; // rdx
  _QWORD *v31; // rax
  __int64 **v32; // rax
  _QWORD *v33; // rbx
  _QWORD *v34; // rax
  __int64 v35; // rax
  _QWORD *v36; // rbx
  __int64 v37; // rdi
  _QWORD *v38; // rax
  _QWORD *v39; // r15
  _QWORD *v40; // rdx
  _QWORD *v41; // rcx
  _QWORD *v42; // r12
  _QWORD *v43; // r14
  _BYTE *v44; // rsi
  char v45; // r15
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  char v49; // al
  char v50; // al
  __int64 v51; // rax
  char v52; // si
  char v53; // r13
  _QWORD *v54; // r13
  __int64 v55; // r12
  char v56; // al
  _QWORD *v57; // r14
  _QWORD *v58; // r12
  unsigned __int8 *v59; // r13
  __int64 v60; // rbx
  __int64 v61; // rsi
  unsigned __int64 v62; // r9
  __int64 v63; // rbx
  __int64 v64; // r15
  bool v65; // r10
  __int64 v66; // rdx
  __int64 v67; // rax
  __int64 v68; // r11
  __int64 v69; // rcx
  _QWORD *v70; // r8
  __int64 v71; // rax
  __int64 v72; // rcx
  bool v73; // r10
  char v74; // al
  __int64 (__fastcall *v75)(__int64, __int64); // rsi
  char v76; // bl
  __int64 v77; // rdi
  __int64 (__fastcall *v78)(__int64, __int64); // rsi
  __int64 v79; // rdi
  _QWORD *v80; // rbx
  __int64 v81; // r12
  _BYTE *v82; // rax
  _BYTE *v83; // r14
  _QWORD *v84; // rcx
  _QWORD *v85; // r14
  _QWORD *v86; // rbx
  unsigned __int8 *v87; // rdi
  char v88; // al
  _QWORD *v89; // rdx
  unsigned __int64 v90; // rax
  _BYTE *v91; // r14
  _QWORD *v92; // r13
  __int64 v93; // rax
  __int64 v94; // rbx
  _BYTE *v95; // rax
  __int64 v96; // rbx
  __int64 v97; // r12
  __int64 v98; // rsi
  __int64 v99; // r13
  __int64 v100; // rbx
  unsigned __int64 v101; // rax
  _BYTE *v102; // rsi
  void **v103; // rdi
  __int64 v104; // rdi
  __int64 v105; // rsi
  char v106; // r12
  _QWORD *v107; // rbx
  __int64 v108; // rdi
  char v109; // al
  __int64 v110; // rax
  bool v111; // al
  char v112; // al
  char v113; // dl
  __int64 v114; // rax
  char v115; // r14
  void **v116; // r13
  unsigned __int64 *v117; // rbx
  unsigned int v118; // ecx
  __int64 *v119; // rdx
  __int64 v120; // r9
  unsigned __int64 v121; // rax
  __int64 *v122; // r15
  int v123; // ecx
  __int64 *v124; // r12
  __int64 v125; // rbx
  __int64 *v126; // r14
  unsigned __int64 v127; // rax
  __int64 *v128; // rbx
  __int64 v129; // r12
  unsigned int v130; // r14d
  unsigned int v131; // r8d
  __int64 v132; // rdi
  __int64 *v133; // rdx
  unsigned int v134; // r10d
  int v135; // r11d
  __int64 *v136; // rax
  __int64 v137; // r9
  unsigned __int64 v138; // r13
  int v139; // r11d
  __int64 *v140; // r9
  unsigned int v141; // edx
  __int64 *v142; // rax
  __int64 v143; // r10
  unsigned __int64 v144; // rcx
  unsigned int v145; // esi
  __int64 v146; // rdi
  int v147; // ecx
  int v148; // eax
  __int64 v149; // r14
  __int64 v150; // r13
  unsigned __int8 *v151; // r15
  int v152; // eax
  unsigned __int64 v153; // rax
  __int64 v154; // rsi
  bool v155; // zf
  __int64 v156; // r9
  __int64 v157; // rax
  __int64 v158; // rdx
  unsigned __int64 v159; // r8
  __int64 *v160; // r12
  unsigned __int64 v161; // rbx
  __int64 *v162; // r14
  unsigned __int64 v163; // rax
  __int64 *v164; // rbx
  __int64 v165; // r12
  unsigned int v166; // r14d
  unsigned int v167; // r8d
  __int64 v168; // rdi
  __int64 *v169; // rdx
  unsigned int v170; // r10d
  int v171; // r11d
  __int64 *v172; // rax
  __int64 v173; // r9
  unsigned __int64 v174; // r13
  __int64 *v175; // r9
  int v176; // r11d
  unsigned int v177; // edx
  __int64 *v178; // rax
  __int64 v179; // r10
  __int64 v180; // rcx
  unsigned int v181; // esi
  __int64 v182; // rdi
  int v183; // ecx
  int v184; // eax
  unsigned int v185; // ebx
  __int64 *v186; // r13
  __int64 v187; // r12
  __int64 v188; // r14
  __int64 *v189; // r14
  __int64 *v190; // rdi
  int v191; // r11d
  unsigned int v192; // edx
  __int64 *v193; // rax
  __int64 v194; // r8
  __int64 v195; // rax
  int v196; // esi
  __int64 *v197; // r13
  int v198; // edx
  __int64 v199; // rax
  __int64 v200; // rax
  _QWORD *v201; // rbx
  _BYTE *v202; // r12
  unsigned __int64 v203; // rdi
  __int64 *v204; // rax
  int v205; // esi
  int v206; // edx
  __int64 v207; // rdx
  __int64 *v208; // rsi
  __int64 *k; // rdi
  __int64 v210; // rax
  __int64 v211; // rdx
  __int64 v212; // rcx
  __int64 v213; // rcx
  __int64 v214; // rax
  int v215; // r10d
  __int64 v216; // rdx
  __int16 v217; // ax
  __int64 v218; // rcx
  __int64 v219; // rbx
  __int64 v220; // r13
  __int64 v221; // r12
  __int64 v222; // rax
  unsigned __int64 v223; // rax
  __int64 v224; // rcx
  __int64 v225; // rdi
  unsigned __int64 v226; // rcx
  __int64 **v227; // rsi
  int v228; // ecx
  unsigned int v229; // edx
  __int64 **v230; // rax
  __int64 *v231; // rdi
  __int64 v232; // rdx
  __int16 v233; // ax
  __int64 v234; // rcx
  int v235; // edx
  int v236; // esi
  __int64 v237; // rdx
  __int64 v238; // rdx
  unsigned __int64 v239; // r8
  int v240; // esi
  __int64 *v241; // rdx
  int v242; // eax
  __int64 v243; // rax
  __int64 v244; // rax
  __int64 v245; // rax
  unsigned int v246; // r15d
  _BYTE *v247; // rdi
  __int64 v248; // rax
  signed __int64 v249; // rdx
  char v250; // r10
  __int64 v251; // r11
  __int64 v252; // rbx
  char *v253; // rsi
  char *v254; // rax
  char *v255; // rdx
  __int64 v256; // rax
  char *v257; // rax
  int v258; // ebx
  void *v259; // r13
  void *v260; // r12
  __int64 v261; // rdx
  __int64 v262; // rcx
  int v264; // eax
  int v265; // r8d
  size_t v266; // rbx
  char *v267; // r15
  _QWORD *v268; // rax
  __int64 *v269; // rax
  __int64 **v270; // rax
  __int64 v271; // rax
  __int64 v272; // rbx
  __int64 *v273; // rax
  __int64 *v274; // rdx
  unsigned __int8 *v275; // rax
  __int64 v276; // r11
  char v277; // r10
  unsigned __int8 *v278; // r15
  char v279; // r10
  __int64 v280; // r11
  __int64 v281; // r14
  __int64 v282; // rbx
  __int64 v283; // rax
  unsigned __int8 *v284; // rdx
  __int64 v285; // r12
  __int64 v286; // rax
  __int64 v287; // rbx
  __int64 v288; // rdx
  __int64 v289; // rdx
  unsigned int v290; // r13d
  __int64 *v291; // rax
  __int64 *v292; // r13
  __int64 v293; // r15
  _QWORD *v294; // rax
  __int64 v295; // r15
  __int64 *v296; // r13
  __int64 v297; // r12
  __int64 *v298; // rbx
  __int64 v299; // rdx
  unsigned int v300; // esi
  unsigned __int64 v301; // rax
  unsigned int v302; // [rsp+8h] [rbp-318h]
  __int64 v303; // [rsp+8h] [rbp-318h]
  unsigned __int8 *v304; // [rsp+10h] [rbp-310h]
  int v305; // [rsp+18h] [rbp-308h]
  __int64 v306; // [rsp+18h] [rbp-308h]
  _QWORD *v307; // [rsp+18h] [rbp-308h]
  int v308; // [rsp+20h] [rbp-300h]
  char v309; // [rsp+20h] [rbp-300h]
  __int64 v310; // [rsp+20h] [rbp-300h]
  char v312; // [rsp+30h] [rbp-2F0h]
  char v313; // [rsp+30h] [rbp-2F0h]
  unsigned __int8 v314; // [rsp+36h] [rbp-2EAh]
  unsigned __int8 v315; // [rsp+37h] [rbp-2E9h]
  __int64 v316; // [rsp+38h] [rbp-2E8h]
  __int64 v317; // [rsp+40h] [rbp-2E0h]
  __int64 v318; // [rsp+40h] [rbp-2E0h]
  char v319; // [rsp+40h] [rbp-2E0h]
  __int64 v320; // [rsp+40h] [rbp-2E0h]
  __int64 v321; // [rsp+50h] [rbp-2D0h]
  __int64 *v322; // [rsp+50h] [rbp-2D0h]
  __int64 *v323; // [rsp+50h] [rbp-2D0h]
  __int64 v324; // [rsp+50h] [rbp-2D0h]
  __int64 v325; // [rsp+50h] [rbp-2D0h]
  __int64 v326; // [rsp+50h] [rbp-2D0h]
  __int64 *v327; // [rsp+60h] [rbp-2C0h]
  __int64 v328; // [rsp+68h] [rbp-2B8h]
  bool v329; // [rsp+68h] [rbp-2B8h]
  char v330; // [rsp+68h] [rbp-2B8h]
  int v331; // [rsp+68h] [rbp-2B8h]
  __int64 v332; // [rsp+70h] [rbp-2B0h]
  char v333; // [rsp+70h] [rbp-2B0h]
  _QWORD *v334; // [rsp+70h] [rbp-2B0h]
  __int64 v335; // [rsp+70h] [rbp-2B0h]
  bool v336; // [rsp+80h] [rbp-2A0h]
  unsigned __int8 v337; // [rsp+80h] [rbp-2A0h]
  __int64 v338; // [rsp+80h] [rbp-2A0h]
  __int64 v339; // [rsp+80h] [rbp-2A0h]
  signed __int64 v340; // [rsp+80h] [rbp-2A0h]
  _QWORD *v341; // [rsp+80h] [rbp-2A0h]
  char v342; // [rsp+80h] [rbp-2A0h]
  unsigned __int8 *v343; // [rsp+80h] [rbp-2A0h]
  char v344; // [rsp+90h] [rbp-290h]
  __int64 *v345; // [rsp+90h] [rbp-290h]
  __int64 *v346; // [rsp+90h] [rbp-290h]
  _QWORD *v347; // [rsp+90h] [rbp-290h]
  char v348; // [rsp+98h] [rbp-288h]
  __int64 (__fastcall *v349)(__int64, __int64); // [rsp+98h] [rbp-288h]
  __int64 *v350; // [rsp+98h] [rbp-288h]
  __int64 *v351; // [rsp+98h] [rbp-288h]
  char v352; // [rsp+98h] [rbp-288h]
  char v353; // [rsp+98h] [rbp-288h]
  __int64 v354; // [rsp+98h] [rbp-288h]
  _QWORD *v355; // [rsp+A0h] [rbp-280h]
  __int64 (__fastcall *v356)(__int64, __int64); // [rsp+A8h] [rbp-278h]
  __int64 v357; // [rsp+A8h] [rbp-278h]
  _QWORD *v358; // [rsp+A8h] [rbp-278h]
  __int64 v359; // [rsp+A8h] [rbp-278h]
  __int64 v360; // [rsp+B0h] [rbp-270h]
  _QWORD *v361; // [rsp+B0h] [rbp-270h]
  __int64 v362; // [rsp+B0h] [rbp-270h]
  __int64 *v363; // [rsp+B0h] [rbp-270h]
  __int64 v364; // [rsp+B0h] [rbp-270h]
  __int64 v365; // [rsp+B0h] [rbp-270h]
  _QWORD *v366; // [rsp+C0h] [rbp-260h]
  char v368; // [rsp+D0h] [rbp-250h]
  _QWORD *v369; // [rsp+D0h] [rbp-250h]
  __int64 v370; // [rsp+E0h] [rbp-240h]
  _QWORD *v371; // [rsp+E8h] [rbp-238h]
  char v372; // [rsp+E8h] [rbp-238h]
  char v373; // [rsp+E8h] [rbp-238h]
  _QWORD *v374; // [rsp+E8h] [rbp-238h]
  __int64 v375; // [rsp+E8h] [rbp-238h]
  __int64 v376; // [rsp+F8h] [rbp-228h] BYREF
  __int64 v377; // [rsp+100h] [rbp-220h] BYREF
  __int64 v378; // [rsp+108h] [rbp-218h] BYREF
  __int64 v379; // [rsp+110h] [rbp-210h] BYREF
  __int64 v380; // [rsp+118h] [rbp-208h] BYREF
  __int64 v381; // [rsp+120h] [rbp-200h] BYREF
  __int64 *v382; // [rsp+128h] [rbp-1F8h] BYREF
  __int64 *v383; // [rsp+130h] [rbp-1F0h] BYREF
  __int64 v384; // [rsp+138h] [rbp-1E8h] BYREF
  __int64 (__fastcall *v385)(__int64, __int64); // [rsp+140h] [rbp-1E0h] BYREF
  __int64 *v386; // [rsp+148h] [rbp-1D8h]
  __int64 v387; // [rsp+150h] [rbp-1D0h] BYREF
  __int64 v388; // [rsp+158h] [rbp-1C8h]
  __int64 v389; // [rsp+160h] [rbp-1C0h]
  unsigned int v390; // [rsp+168h] [rbp-1B8h]
  unsigned __int64 v391; // [rsp+170h] [rbp-1B0h] BYREF
  _BYTE *v392; // [rsp+178h] [rbp-1A8h]
  _BYTE *v393; // [rsp+180h] [rbp-1A0h]
  unsigned int v394; // [rsp+188h] [rbp-198h]
  __int16 v395; // [rsp+190h] [rbp-190h]
  __int64 v396; // [rsp+1A0h] [rbp-180h] BYREF
  void *s; // [rsp+1A8h] [rbp-178h]
  _BYTE v398[12]; // [rsp+1B0h] [rbp-170h]
  unsigned __int8 v399; // [rsp+1BCh] [rbp-164h]
  char v400; // [rsp+1C0h] [rbp-160h] BYREF
  void *src[12]; // [rsp+200h] [rbp-120h] BYREF
  __int64 *v402[13]; // [rsp+260h] [rbp-C0h] BYREF
  int v403; // [rsp+2C8h] [rbp-58h]
  __int16 v404; // [rsp+2CCh] [rbp-54h]
  char v405; // [rsp+2CEh] [rbp-52h]
  __int64 v406; // [rsp+2D0h] [rbp-50h]
  __int64 v407; // [rsp+2D8h] [rbp-48h]
  void *v408; // [rsp+2E0h] [rbp-40h] BYREF
  void *v409; // [rsp+2E8h] [rbp-38h] BYREF
  char v410; // [rsp+2F0h] [rbp-30h] BYREF

  v5 = (__int64)&unk_4F82418;
  v327 = a3 + 39;
  v6 = sub_BC0510(a4, &unk_4F82418, (__int64)a3);
  v315 = 0;
  v10 = 1;
  v384 = 0;
  v11 = *(_QWORD *)(v6 + 8);
  *(_QWORD *)v398 = 8;
  v316 = v11;
  v376 = v11;
  v377 = v11;
  v378 = v11;
  v379 = v11;
  v385 = (__int64 (__fastcall *)(__int64, __int64))sub_25DC220;
  v386 = &v377;
  *(_DWORD *)&v398[8] = 0;
  v399 = 1;
  s = &v400;
  v366 = a3 + 1;
  for ( i = 0; ; i = v396 )
  {
    v396 = i + 1;
    if ( (_BYTE)v10 )
      goto LABEL_7;
    v13 = 4 * (*(_DWORD *)&v398[4] - *(_DWORD *)&v398[8]);
    if ( v13 < 0x20 )
      v13 = 32;
    if ( *(_DWORD *)v398 <= v13 )
    {
      v5 = 0xFFFFFFFFLL;
      memset(s, -1, 8LL * *(unsigned int *)v398);
LABEL_7:
      *(_QWORD *)&v398[4] = 0;
      goto LABEL_8;
    }
    sub_C8C990((__int64)&v396, v5);
LABEL_8:
    v14 = (_QWORD *)a3[2];
    if ( v14 != v366 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          if ( !v14 )
LABEL_556:
            BUG();
          v5 = *(v14 - 1);
          if ( v5 )
          {
            v15 = *(_BYTE *)(v14 - 3) & 0xF;
            v10 = (v15 + 9) & 0xF;
            if ( (unsigned __int8)v10 > 1u && ((v15 + 15) & 0xFu) > 2 )
              break;
            if ( *(v14 - 5) )
              break;
          }
LABEL_19:
          v14 = (_QWORD *)v14[1];
          if ( v14 == v366 )
            goto LABEL_20;
        }
        if ( v399 )
        {
          v16 = s;
          v7 = *(unsigned int *)&v398[4];
          v10 = (__int64)s + 8 * *(unsigned int *)&v398[4];
          if ( s != (void *)v10 )
          {
            while ( v5 != *v16 )
            {
              if ( (_QWORD *)v10 == ++v16 )
                goto LABEL_180;
            }
            goto LABEL_19;
          }
LABEL_180:
          if ( *(_DWORD *)&v398[4] >= *(_DWORD *)v398 )
            goto LABEL_178;
          v7 = (unsigned int)++*(_DWORD *)&v398[4];
          *(_QWORD *)v10 = v5;
          v14 = (_QWORD *)v14[1];
          ++v396;
          if ( v14 == v366 )
            break;
        }
        else
        {
LABEL_178:
          sub_C8CC70((__int64)&v396, v5, v10, v7, v8, v9);
          v14 = (_QWORD *)v14[1];
          if ( v14 == v366 )
            break;
        }
      }
    }
LABEL_20:
    v17 = (_QWORD *)a3[4];
    v371 = a3 + 3;
    for ( j = a3 + 3; j != v17; ++v396 )
    {
      while ( 1 )
      {
        if ( !v17 )
          goto LABEL_556;
        v19 = *(v17 - 1);
        if ( v19 )
        {
          if ( !(unsigned __int8)sub_B2E360((__int64)(v17 - 7), v5, v10, v7, v8) )
            break;
        }
LABEL_22:
        v17 = (_QWORD *)v17[1];
        if ( j == v17 )
          goto LABEL_32;
      }
      if ( !v399 )
        goto LABEL_391;
      v21 = s;
      v7 = *(unsigned int *)&v398[4];
      v10 = (__int64)s + 8 * *(unsigned int *)&v398[4];
      if ( s != (void *)v10 )
      {
        while ( v19 != *v21 )
        {
          if ( (_QWORD *)v10 == ++v21 )
            goto LABEL_30;
        }
        goto LABEL_22;
      }
LABEL_30:
      if ( *(_DWORD *)&v398[4] >= *(_DWORD *)v398 )
      {
LABEL_391:
        v5 = v19;
        sub_C8CC70((__int64)&v396, v19, v10, v7, v8, v20);
        goto LABEL_22;
      }
      v7 = (unsigned int)++*(_DWORD *)&v398[4];
      *(_QWORD *)v10 = v19;
      v17 = (_QWORD *)v17[1];
    }
LABEL_32:
    v22 = (_QWORD *)a3[6];
    v23 = a3 + 5;
    if ( a3 + 5 != v22 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v24 = (__int64)(v22 - 6);
          if ( !v22 )
            v24 = 0;
          v26 = sub_B326A0(v24);
          if ( v26 )
          {
            v29 = *(_BYTE *)(v24 + 32) & 0xF;
            v30 = (v29 + 9) & 0xF;
            if ( (unsigned __int8)v30 > 1u && ((v29 + 15) & 0xFu) > 2 )
              break;
            if ( *(_QWORD *)(v24 + 16) )
              break;
          }
LABEL_44:
          v22 = (_QWORD *)v22[1];
          if ( v23 == v22 )
            goto LABEL_45;
        }
        if ( v399 )
        {
          v31 = s;
          v25 = *(unsigned int *)&v398[4];
          v30 = (__int64)s + 8 * *(unsigned int *)&v398[4];
          if ( s != (void *)v30 )
          {
            while ( v26 != *v31 )
            {
              if ( (_QWORD *)v30 == ++v31 )
                goto LABEL_191;
            }
            goto LABEL_44;
          }
LABEL_191:
          if ( *(_DWORD *)&v398[4] >= *(_DWORD *)v398 )
            goto LABEL_183;
          ++*(_DWORD *)&v398[4];
          *(_QWORD *)v30 = v26;
          v22 = (_QWORD *)v22[1];
          ++v396;
          if ( v23 == v22 )
            break;
        }
        else
        {
LABEL_183:
          sub_C8CC70((__int64)&v396, v26, v30, v25, v27, v28);
          v22 = (_QWORD *)v22[1];
          if ( v23 == v22 )
            break;
        }
      }
    }
LABEL_45:
    v402[0] = 0;
    v402[1] = (__int64 *)1;
    v356 = v385;
    v360 = (__int64)v386;
    v32 = &v402[2];
    do
    {
      *v32 = (__int64 *)-4096LL;
      v32 += 2;
    }
    while ( v32 != (__int64 **)&v410 );
    v391 = 0;
    v392 = 0;
    v393 = 0;
    v33 = (_QWORD *)a3[4];
    if ( v371 != v33 )
    {
      do
      {
        v34 = v33;
        v33 = (_QWORD *)v33[1];
        if ( (*((_BYTE *)v34 - 23) & 0x20) == 0 )
        {
          v35 = sub_BC1CD0(v378, &unk_4F89C30, (__int64)(v34 - 7));
          if ( (unsigned __int8)sub_DFA9E0(v35 + 8) )
          {
            v36 = (_QWORD *)a3[4];
LABEL_52:
            if ( v371 != v36 )
            {
              do
              {
                while ( 1 )
                {
                  v38 = v36;
                  v36 = (_QWORD *)v36[1];
                  v39 = (_QWORD *)v38[3];
                  v40 = v38 - 7;
                  if ( v39 != v38 + 2 )
                  {
                    v41 = v38 + 2;
                    do
                    {
                      if ( !v39 )
                        BUG();
                      v42 = (_QWORD *)v39[4];
                      v43 = v39 + 3;
                      if ( v42 != v39 + 3 )
                      {
                        while ( 1 )
                        {
                          if ( !v42 )
LABEL_557:
                            BUG();
                          if ( *((_BYTE *)v42 - 24) != 85 )
                            goto LABEL_64;
                          v37 = *(v42 - 7);
                          if ( *(_BYTE *)v37 == 25 )
                            goto LABEL_64;
                          if ( *(_BYTE *)v37 || *(_QWORD *)(v37 + 24) != v42[7] )
                            goto LABEL_57;
                          if ( *(_DWORD *)(v37 + 36) )
                          {
LABEL_64:
                            v42 = (_QWORD *)v42[1];
                            if ( v43 == v42 )
                              break;
                          }
                          else
                          {
                            v355 = v41;
                            v370 = (__int64)v40;
                            if ( (*(_BYTE *)(v37 + 32) & 0xFu) - 7 > 1 )
                              goto LABEL_57;
                            if ( !(unsigned __int8)sub_25E26A0(v37, (__int64)v402) )
                              goto LABEL_57;
                            v110 = sub_BC1CD0(v316, &unk_4F8D9A8, v370);
                            v111 = sub_25DC260((__int64)(v42 - 3), (__int64 *)(v110 + 8));
                            v40 = (_QWORD *)v370;
                            v41 = v355;
                            if ( !v111 )
                              goto LABEL_57;
                            v42 = (_QWORD *)v42[1];
                            if ( v43 == v42 )
                              break;
                          }
                        }
                      }
                      v39 = (_QWORD *)v39[1];
                    }
                    while ( v41 != v39 );
                  }
                  src[0] = v40;
                  v44 = v392;
                  if ( v392 == v393 )
                    break;
                  if ( v392 )
                  {
                    *(_QWORD *)v392 = v40;
                    v44 = v392;
                  }
                  v392 = v44 + 8;
                  if ( v371 == v36 )
                    goto LABEL_70;
                }
                sub_24147A0((__int64)&v391, v392, src);
LABEL_57:
                ;
              }
              while ( v371 != v36 );
LABEL_70:
              v36 = (_QWORD *)a3[4];
              goto LABEL_71;
            }
            goto LABEL_487;
          }
        }
      }
      while ( v371 != v33 );
      v36 = (_QWORD *)a3[4];
      if ( (_BYTE)qword_4FF1748 )
        goto LABEL_52;
LABEL_71:
      if ( v371 != v36 )
      {
        v45 = 0;
        while ( 1 )
        {
          while ( 1 )
          {
            v54 = v36;
            v36 = (_QWORD *)v36[1];
            v55 = (__int64)(v54 - 7);
            if ( !(unsigned __int8)sub_B2D610((__int64)(v54 - 7), 20) )
              break;
LABEL_82:
            if ( v371 == v36 )
              goto LABEL_87;
          }
          if ( (*((_BYTE *)v54 - 49) & 0x10) == 0
            && !sub_B2FC80((__int64)(v54 - 7))
            && (*(_BYTE *)(v54 - 3) & 0xFu) - 7 > 1 )
          {
            v112 = *((_BYTE *)v54 - 23) & 0xFC;
            *((_BYTE *)v54 - 24) = *(_BYTE *)(v54 - 3) & 0xC0 | 7;
            *((_BYTE *)v54 - 23) = v112 | 0x40;
          }
          v56 = sub_25DD4B0(
                  (__int64)(v54 - 7),
                  (__int64)&v396,
                  (void (__fastcall *)(__int64, __int64))sub_25DC5A0,
                  (__int64)&v379);
          if ( !v56 )
            break;
          v45 = v56;
          if ( v371 == v36 )
            goto LABEL_87;
        }
        if ( !sub_B2FC80((__int64)(v54 - 7)) )
        {
          v115 = sub_F62E00((__int64)(v54 - 7), 0, 0, v46, v47, v48);
          if ( v115 )
          {
            memset(src, 0, sizeof(src));
            LODWORD(src[2]) = 2;
            src[1] = &src[4];
            BYTE4(src[3]) = 1;
            src[7] = &src[10];
            LODWORD(src[8]) = 2;
            BYTE4(src[9]) = 1;
            sub_BBE020(v316, (__int64)(v54 - 7), (__int64)src, 0);
            if ( !BYTE4(src[9]) )
              _libc_free((unsigned __int64)src[7]);
            if ( !BYTE4(src[3]) )
              _libc_free((unsigned __int64)src[1]);
            v45 = v115;
          }
        }
        v45 |= sub_25E94B0((__int64)(v54 - 7), (__int64)sub_25DC240, &v378, v356, v360, v48, (int)sub_25DC200, &v376);
        if ( (*(_BYTE *)(v54 - 3) & 0xFu) - 7 > 1 )
          goto LABEL_82;
        src[0] = *((void **)v54 + 8);
        v49 = sub_A74390((__int64 *)src, 83, 0);
        if ( v49 )
        {
          v353 = v49;
          if ( !(unsigned __int8)sub_B2DDD0((__int64)(v54 - 7), 0, 0, 1, 0, 0, 0)
            && !sub_25DC970((__int64)(v54 - 7))
            && !(*(_DWORD *)(*(v54 - 4) + 8LL) >> 8) )
          {
            sub_25DCA60((__int64)(v54 - 7), 83);
            v45 = v353;
          }
        }
        src[0] = *((void **)v54 + 8);
        v50 = sub_A74390((__int64 *)src, 84, 0);
        if ( v50 )
        {
          v352 = v50;
          if ( !(unsigned __int8)sub_B2DDD0((__int64)(v54 - 7), 0, 0, 1, 0, 0, 0) && !sub_25DC970((__int64)(v54 - 7)) )
          {
            v214 = *(v54 - 5);
            if ( v214 )
            {
              while ( **(_BYTE **)(v214 + 24) != 34 )
              {
                v214 = *(_QWORD *)(v214 + 8);
                if ( !v214 )
                  goto LABEL_398;
              }
            }
            else
            {
LABEL_398:
              sub_25DF290((__int64)(v54 - 7));
              v45 = v352;
            }
          }
          goto LABEL_82;
        }
        v348 = sub_25E26A0((__int64)(v54 - 7), (__int64)v402);
        if ( !v348 )
          goto LABEL_80;
        v51 = sub_BC1CD0(v378, &unk_4F89C30, (__int64)(v54 - 7));
        if ( !(_BYTE)qword_4FF1748 )
        {
          if ( (unsigned __int8)sub_DFA9E0(v51 + 8) && *(v54 - 5) )
          {
            v338 = (__int64)(v54 - 7);
            v334 = v54;
            v347 = v36;
            v219 = *(v54 - 5);
            while ( 2 )
            {
              v220 = *(_QWORD *)(v219 + 24);
              if ( *(_BYTE *)v220 == 4 )
                goto LABEL_431;
              v221 = *(_QWORD *)(*(_QWORD *)(v220 + 40) + 72LL);
              v222 = sub_BC1CD0(v316, &unk_4F8D9A8, v221);
              if ( !sub_25DC260(v220, (__int64 *)(v222 + 8)) )
                goto LABEL_493;
              v223 = v391;
              v224 = (__int64)&v392[-v391] >> 5;
              v225 = (__int64)&v392[-v391] >> 3;
              if ( v224 > 0 )
              {
                v226 = v391 + 32 * v224;
                while ( v221 != *(_QWORD *)v223 )
                {
                  if ( v221 == *(_QWORD *)(v223 + 8) )
                  {
                    v223 += 8LL;
                    goto LABEL_430;
                  }
                  if ( v221 == *(_QWORD *)(v223 + 16) )
                  {
                    v223 += 16LL;
                    goto LABEL_430;
                  }
                  if ( v221 == *(_QWORD *)(v223 + 24) )
                  {
                    v223 += 24LL;
                    goto LABEL_430;
                  }
                  v223 += 32LL;
                  if ( v226 == v223 )
                  {
                    v225 = (__int64)&v392[-v223] >> 3;
                    goto LABEL_490;
                  }
                }
                goto LABEL_430;
              }
LABEL_490:
              if ( v225 == 2 )
                goto LABEL_496;
              if ( v225 != 3 )
              {
                if ( v225 != 1 )
                  goto LABEL_493;
                goto LABEL_498;
              }
              if ( v221 != *(_QWORD *)v223 )
              {
                v223 += 8LL;
LABEL_496:
                if ( v221 != *(_QWORD *)v223 )
                {
                  v223 += 8LL;
LABEL_498:
                  if ( v221 != *(_QWORD *)v223 )
                  {
LABEL_493:
                    v36 = v347;
                    v55 = v338;
                    v54 = v334;
                    goto LABEL_80;
                  }
                }
              }
LABEL_430:
              if ( v392 == (_BYTE *)v223 )
                goto LABEL_493;
LABEL_431:
              v219 = *(_QWORD *)(v219 + 8);
              if ( !v219 )
              {
                v36 = v347;
                v55 = v338;
                v54 = v334;
                goto LABEL_433;
              }
              continue;
            }
          }
          goto LABEL_80;
        }
LABEL_433:
        if ( ((__int64)v402[1] & 1) != 0 )
        {
          v227 = &v402[2];
          v228 = 7;
LABEL_435:
          v229 = v228 & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
          v230 = &v227[2 * v229];
          v231 = *v230;
          if ( (__int64 *)v55 == *v230 )
          {
LABEL_436:
            *v230 = (__int64 *)-8192LL;
            ++HIDWORD(v402[1]);
            LODWORD(v402[1]) = (2 * (LODWORD(v402[1]) >> 1) - 2) | (__int64)v402[1] & 1;
          }
          else
          {
            v264 = 1;
            while ( v231 != (__int64 *)-4096LL )
            {
              v265 = v264 + 1;
              v229 = v228 & (v264 + v229);
              v230 = &v227[2 * v229];
              v231 = *v230;
              if ( (__int64 *)v55 == *v230 )
                goto LABEL_436;
              v264 = v265;
            }
          }
        }
        else
        {
          v227 = (__int64 **)v402[2];
          if ( LODWORD(v402[3]) )
          {
            v228 = LODWORD(v402[3]) - 1;
            goto LABEL_435;
          }
        }
        v232 = *(v54 - 5);
        v233 = *((_WORD *)v54 - 27) & 0xC00F;
        LOBYTE(v233) = v233 | 0x90;
        for ( *((_WORD *)v54 - 27) = v233; v232; v232 = *(_QWORD *)(v232 + 8) )
        {
          v234 = *(_QWORD *)(v232 + 24);
          if ( *(_BYTE *)v234 != 4 )
            *(_WORD *)(v234 + 2) = *(_WORD *)(v234 + 2) & 0xF003 | 0x24;
        }
        v45 = v348;
LABEL_80:
        v52 = sub_25E26A0(v55, (__int64)v402);
        if ( v52 )
        {
          v216 = *(v54 - 5);
          v217 = *((_WORD *)v54 - 27) & 0xC00F;
          LOBYTE(v217) = v217 | 0x80;
          for ( *((_WORD *)v54 - 27) = v217; v216; v216 = *(_QWORD *)(v216 + 8) )
          {
            v218 = *(_QWORD *)(v216 + 24);
            if ( *(_BYTE *)v218 != 4 )
              *(_WORD *)(v218 + 2) = *(_WORD *)(v218 + 2) & 0xF003 | 0x20;
          }
          v45 = v52;
        }
        src[0] = *((void **)v54 + 8);
        v53 = sub_A74390((__int64 *)src, 21, 0);
        if ( v53 && !(unsigned __int8)sub_B2DDD0(v55, 0, 0, 1, 0, 0, 0) )
        {
          v45 = v53;
          sub_25DCA60(v55, 21);
        }
        goto LABEL_82;
      }
    }
LABEL_487:
    v45 = 0;
LABEL_87:
    if ( v391 )
      j_j___libc_free_0(v391);
    if ( ((__int64)v402[1] & 1) == 0 )
      sub_C7D6A0((__int64)v402[2], 16LL * LODWORD(v402[3]), 8);
    v402[0] = &v384;
    v402[1] = v327;
    v402[2] = (__int64 *)&v385;
    v368 = 0;
    v344 = v45 | sub_29C00F0(a3, sub_25E1F50, v402);
    if ( (_QWORD *)a3[2] != v366 )
    {
      v357 = (__int64)v386;
      v57 = (_QWORD *)a3[2];
      v349 = v385;
      do
      {
        v58 = v57;
        v57 = (_QWORD *)v57[1];
        v59 = (unsigned __int8 *)(v58 - 7);
        if ( (*((_BYTE *)v58 - 49) & 0x10) == 0
          && !sub_B2FC80((__int64)(v58 - 7))
          && (*(_BYTE *)(v58 - 3) & 0xFu) - 7 > 1 )
        {
          v113 = *((_BYTE *)v58 - 23) & 0xFC;
          *((_BYTE *)v58 - 24) = *(_BYTE *)(v58 - 3) & 0xC0 | 7;
          *((_BYTE *)v58 - 23) = v113 | 0x40;
        }
        if ( !sub_B2FC80((__int64)(v58 - 7)) )
        {
          v60 = *(v58 - 11);
          if ( v60 )
          {
            v61 = sub_97B670((_BYTE *)*(v58 - 11), (__int64)v327, 0);
            if ( v60 != v61 )
              sub_B30160((__int64)(v58 - 7), v61);
          }
        }
        v372 = sub_25DD4B0((__int64)(v58 - 7), (__int64)&v396, 0, 0);
        if ( v372 )
        {
          v368 = v372;
        }
        else
        {
          if ( !sub_B2FC80((__int64)(v58 - 7))
            && (v58[3] & 1) != 0
            && (*(_BYTE *)(v58 - 3) & 0xFu) - 7 <= 1
            && *((_BYTE *)v58 - 24) >> 6 == 2 )
          {
            v63 = *(v58 - 5);
            if ( v63 )
            {
              while ( 1 )
              {
                v64 = *(_QWORD *)(v63 + 24);
                if ( *(_BYTE *)v64 != 85 )
                  v64 = 0;
                if ( (unsigned __int8)sub_25DC1A0(v64) )
                {
                  v65 = sub_25DC9D0(v64);
                  if ( v65 )
                  {
                    v66 = *(_DWORD *)(v64 + 4) & 0x7FFFFFF;
                    v67 = 32 * (2 - v66);
                    v68 = *(_QWORD *)(v64 + v67);
                    if ( *(_BYTE *)v68 == 17 )
                    {
                      v69 = *(v58 - 11);
                      if ( *(_BYTE *)v69 == 15 )
                      {
                        v70 = *(_QWORD **)(v68 + 24);
                        if ( *(_DWORD *)(v68 + 32) > 0x40u )
                          v70 = (_QWORD *)*v70;
                        v332 = *(_QWORD *)(v64 + v67);
                        v336 = v65;
                        v71 = *(_QWORD *)(v64 - 32 * v66);
                        if ( *(_BYTE *)v71 != 60 )
                          BUG();
                        v305 = (int)v70;
                        v308 = (int)v70;
                        v321 = *(v58 - 11);
                        v317 = *(_QWORD *)(*(_QWORD *)(v71 + 72) + 32LL);
                        v328 = *(_QWORD *)(*(_QWORD *)(v69 + 8) + 32LL);
                        v62 = sub_AC52A0(v69);
                        v72 = v321;
                        v73 = v336;
                        if ( v317 == v328
                          && (v308 != 0) + (v308 - (unsigned int)(v308 != 0)) / (unsigned int)v62 == v317 )
                        {
                          v324 = v332;
                          v339 = v72;
                          v329 = v73;
                          v244 = sub_B43CB0(v64);
                          v245 = sub_BC1CD0(v378, &unk_4F89C30, v244);
                          v246 = sub_DFE700(v245 + 8);
                          if ( v246 )
                            break;
                        }
                      }
                    }
                  }
                }
                v63 = *(_QWORD *)(v63 + 8);
                if ( !v63 )
                  goto LABEL_115;
              }
              v335 = v339;
              if ( sub_B2FC80((__int64)(v58 - 7)) || (v247 = (_BYTE *)*(v58 - 11), *v247 != 15) )
              {
                v372 = 0;
              }
              else
              {
                v248 = sub_AC52D0((__int64)v247);
                v250 = v329;
                memset(src, 0, 24);
                v251 = v324;
                v252 = v248;
                if ( v249 < 0 )
                  sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
                v253 = 0;
                if ( v249 )
                {
                  v340 = v249;
                  v254 = (char *)sub_22077B0(v249);
                  v250 = v329;
                  v251 = v324;
                  src[0] = v254;
                  v255 = v254;
                  v253 = &v254[v340];
                  v256 = 0;
                  src[2] = v253;
                  do
                  {
                    v255[v256] = *(_BYTE *)(v252 + v256);
                    ++v256;
                  }
                  while ( v256 != v340 );
                }
                v257 = v253;
                src[1] = v253;
                v258 = 0;
                while ( 1 )
                {
                  LOBYTE(v402[0]) = 0;
                  if ( v253 == v257 )
                  {
                    v325 = v251;
                    v330 = v250;
                    sub_C8FB10((__int64)src, v253, (char *)v402);
                    v251 = v325;
                    v250 = v330;
                  }
                  else
                  {
                    if ( v253 )
                    {
                      *v253 = 0;
                      v253 = (char *)src[1];
                    }
                    src[1] = v253 + 1;
                  }
                  if ( v246 <= ++v258 )
                    break;
                  v253 = (char *)src[1];
                  v257 = (char *)src[2];
                }
                v309 = v250;
                v266 = v305 + v246;
                v318 = v251;
                v331 = v305 + v246;
                v267 = (char *)src[0];
                v326 = v266;
                v268 = (_QWORD *)sub_BD5C60((__int64)v59);
                v269 = (__int64 *)sub_BCD140(v268, 8u);
                v270 = (__int64 **)sub_BCD420(v269, v266);
                v271 = sub_AC9630(v267, v266, v270);
                v272 = *(v58 - 2);
                v306 = v271;
                v312 = *(_BYTE *)(v58 - 3) & 0xF;
                v341 = *(_QWORD **)(v271 + 8);
                v273 = (__int64 *)sub_BD5D20(v271);
                BYTE4(v391) = 0;
                v402[0] = v273;
                LOWORD(v402[4]) = 261;
                v402[1] = v274;
                v275 = (unsigned __int8 *)sub_BD2C40(88, unk_3F0FAE8);
                v276 = v318;
                v277 = v309;
                v278 = v275;
                if ( v275 )
                {
                  v310 = v318;
                  v319 = v277;
                  sub_B30000((__int64)v275, v272, v341, 1, v312, v306, (__int64)v402, 0, 0, v391, 0);
                  v276 = v310;
                  v277 = v319;
                }
                v320 = v276;
                v342 = v277;
                sub_B32030((__int64)v278, (__int64)v59);
                sub_BD6B90(v278, v59);
                v279 = v342;
                v280 = v320;
                if ( src[0] )
                {
                  j_j___libc_free_0((unsigned __int64)src[0]);
                  v280 = v320;
                  v279 = v342;
                }
                if ( v278 )
                {
                  v375 = *(v58 - 5);
                  if ( v375 )
                  {
                    v343 = (unsigned __int8 *)(v58 - 7);
                    v313 = v279;
                    v304 = v278;
                    v307 = v57;
                    v281 = v280;
                    do
                    {
                      v282 = *(_QWORD *)(v375 + 24);
                      if ( *(_BYTE *)v282 != 85 )
                        v282 = 0;
                      if ( (unsigned __int8)sub_25DC1A0(v282) )
                      {
                        if ( sub_25DC9D0(v282) )
                        {
                          v283 = *(_DWORD *)(v282 + 4) & 0x7FFFFFF;
                          v284 = *(unsigned __int8 **)(v282 + 32 * (1 - v283));
                          if ( v343 == v284 )
                          {
                            if ( v284 )
                            {
                              v285 = *(_QWORD *)(v282 - 32 * v283);
                              if ( *(_BYTE *)v285 == 60 )
                              {
                                v290 = (v331 - (unsigned int)(v331 != 0)) / (unsigned int)sub_AC52A0(v335) + (v331 != 0);
                                v291 = (__int64 *)sub_BD5C60(v285);
                                v405 = 7;
                                v402[9] = v291;
                                v402[10] = (__int64 *)&v408;
                                v402[11] = (__int64 *)&v409;
                                v402[0] = (__int64 *)&v402[2];
                                v402[1] = (__int64 *)0x200000000LL;
                                v408 = &unk_49DA100;
                                v404 = 512;
                                v409 = &unk_49DA0B0;
                                v402[12] = 0;
                                v403 = 0;
                                v406 = 0;
                                v407 = 0;
                                memset(&v402[6], 0, 18);
                                sub_D5F1F0((__int64)v402, v285);
                                v395 = 257;
                                v292 = sub_BCD420(**(__int64 ***)(*(_QWORD *)(v285 + 72) + 16LL), v290);
                                v293 = sub_AA4E30((__int64)v402[6]);
                                v314 = sub_AE5260(v293, (__int64)v292);
                                v302 = *(_DWORD *)(v293 + 4);
                                LOWORD(src[4]) = 257;
                                v294 = sub_BD2C40(80, unk_3F10A14);
                                v295 = (__int64)v294;
                                if ( v294 )
                                  sub_B4CCA0((__int64)v294, v292, v302, 0, v314, (__int64)src, 0, 0);
                                (*(void (__fastcall **)(__int64 *, __int64, unsigned __int64 *, __int64 *, __int64 *))(*v402[11] + 16))(
                                  v402[11],
                                  v295,
                                  &v391,
                                  v402[7],
                                  v402[8]);
                                if ( v402[0] != &v402[0][2 * LODWORD(v402[1])] )
                                {
                                  v303 = v285;
                                  v296 = v402[0];
                                  v297 = v282;
                                  v298 = &v402[0][2 * LODWORD(v402[1])];
                                  do
                                  {
                                    v299 = v296[1];
                                    v300 = *(_DWORD *)v296;
                                    v296 += 2;
                                    sub_B99FD0(v295, v300, v299);
                                  }
                                  while ( v298 != v296 );
                                  v282 = v297;
                                  v285 = v303;
                                }
                                sub_BD6B90((unsigned __int8 *)v295, (unsigned __int8 *)v285);
                                _BitScanReverse64(&v301, 1LL << *(_WORD *)(v285 + 2));
                                *(_WORD *)(v295 + 2) = (63 - (v301 ^ 0x3F)) | *(_WORD *)(v295 + 2) & 0xFFC0;
                                sub_BD84D0(v285, v295);
                                sub_B43D60((_QWORD *)v285);
                                nullsub_61();
                                v408 = &unk_49DA100;
                                nullsub_63();
                                if ( (__int64 **)v402[0] != &v402[2] )
                                  _libc_free((unsigned __int64)v402[0]);
                              }
                              v286 = sub_AD64C0(*(_QWORD *)(v281 + 8), v326, 0);
                              v287 = 32 * (2LL - (*(_DWORD *)(v282 + 4) & 0x7FFFFFF)) + v282;
                              if ( *(_QWORD *)v287 )
                              {
                                v288 = *(_QWORD *)(v287 + 8);
                                **(_QWORD **)(v287 + 16) = v288;
                                if ( v288 )
                                  *(_QWORD *)(v288 + 16) = *(_QWORD *)(v287 + 16);
                              }
                              *(_QWORD *)v287 = v286;
                              if ( v286 )
                              {
                                v289 = *(_QWORD *)(v286 + 16);
                                *(_QWORD *)(v287 + 8) = v289;
                                if ( v289 )
                                  *(_QWORD *)(v289 + 16) = v287 + 8;
                                *(_QWORD *)(v287 + 16) = v286 + 16;
                                *(_QWORD *)(v286 + 16) = v287;
                              }
                            }
                          }
                        }
                      }
                      v375 = *(_QWORD *)(v375 + 8);
                    }
                    while ( v375 );
                    v57 = v307;
                    v59 = v343;
                    v279 = v313;
                    v278 = v304;
                  }
                  v372 = v279;
                  sub_BD84D0((__int64)v59, (__int64)v278);
                }
              }
            }
          }
LABEL_115:
          v368 |= sub_25E94B0((__int64)v59, (__int64)sub_25DC240, &v378, v349, v357, v62, (int)sub_25DC200, &v376)
                | v372;
        }
      }
      while ( v57 != v366 );
      v344 |= v368;
    }
    v74 = sub_25DFDC0((__int64)a3, (__int64)&v396);
    v75 = v385;
    v337 = v344 | v74;
    v76 = v344 | v74;
    v77 = sub_25DCB60((__int64)a3, v385, (__int64)v386, 0x58u);
    if ( v77 )
      v337 = sub_25DC2E0(v77, (__int64)v75) | v76;
    v78 = v385;
    v79 = sub_25DCB60((__int64)a3, v385, (__int64)v386, 0x59u);
    if ( v79 )
      v337 |= sub_25DC2E0(v79, (__int64)v78);
    v80 = (_QWORD *)a3[8];
    v369 = a3 + 7;
    if ( a3 + 7 == v80 )
    {
      v387 = 0;
      v104 = 0;
      v105 = 0;
      v388 = 0;
      v389 = 0;
      v390 = 0;
    }
    else
    {
      v373 = 0;
      do
      {
        v81 = (__int64)(v80 - 7);
        if ( !v80 )
          v81 = 0;
        if ( !(unsigned __int8)sub_B2F6B0(v81) )
        {
          v82 = sub_B30850(v81);
          v83 = v82;
          if ( v82 )
          {
            if ( !(unsigned __int8)sub_B2F6B0((__int64)v82) )
            {
              v84 = (_QWORD *)*((_QWORD *)v83 + 10);
              if ( (_BYTE *)v84[1] == v83 + 72 )
              {
                v85 = v84 + 3;
                if ( v84 + 3 == (_QWORD *)v84[4] )
                  goto LABEL_137;
                v361 = v80;
                v86 = (_QWORD *)v84[4];
                while ( 1 )
                {
                  v87 = (unsigned __int8 *)(v86 - 3);
                  v358 = v84;
                  if ( !v86 )
                    v87 = 0;
                  v88 = sub_B46970(v87);
                  v84 = v358;
                  if ( v88 )
                    break;
                  v86 = (_QWORD *)v86[1];
                  if ( v85 == v86 )
                  {
                    v80 = v361;
                    goto LABEL_137;
                  }
                }
                v89 = v86;
                v80 = v361;
                if ( v85 == v89 )
                {
LABEL_137:
                  v90 = v84[3] & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v85 == (_QWORD *)v90 || !v90 || (unsigned int)*(unsigned __int8 *)(v90 - 24) - 30 > 0xA )
                    goto LABEL_557;
                  if ( *(_BYTE *)(v90 - 24) == 30 )
                  {
                    if ( (*(_DWORD *)(v90 - 20) & 0x7FFFFFF) == 0
                      || (v91 = *(_BYTE **)(v90 - 32LL * (*(_DWORD *)(v90 - 20) & 0x7FFFFFF) - 24)) == 0 )
                    {
LABEL_201:
                      BUG();
                    }
                    if ( !*v91 && *(_QWORD *)(v81 + 16) )
                    {
                      if ( sub_B2FC80(*(_QWORD *)(v90 - 32LL * (*(_DWORD *)(v90 - 20) & 0x7FFFFFF) - 24))
                        && (v114 = *(_QWORD *)(v81 + 16)) != 0 )
                      {
                        while ( **(_BYTE **)(v114 + 24) != 1 )
                        {
                          v114 = *(_QWORD *)(v114 + 8);
                          if ( !v114 )
                            goto LABEL_146;
                        }
                      }
                      else
                      {
LABEL_146:
                        sub_BD84D0(v81, (__int64)v91);
                        v373 = 1;
                      }
                    }
                  }
                }
              }
            }
          }
        }
        v80 = (_QWORD *)v80[1];
      }
      while ( a3 + 7 != v80 );
      v387 = 0;
      v337 |= v373;
      v388 = 0;
      v92 = (_QWORD *)a3[8];
      v389 = 0;
      v390 = 0;
      if ( v92 != v369 )
      {
        v333 = 0;
        v374 = v92;
        while ( 1 )
        {
          v93 = (__int64)(v374 - 7);
          if ( !v374 )
            v93 = 0;
          v362 = v93;
          v94 = v93;
          if ( (unsigned __int8)sub_B2F6B0(v93) )
            goto LABEL_169;
          v95 = sub_B30850(v94);
          v96 = (__int64)v95;
          if ( !v95 || (unsigned __int8)sub_B2F6B0((__int64)v95) )
            goto LABEL_169;
          v97 = v96 + 72;
          v98 = sub_BC1CD0(v378, &unk_4F89C30, v96) + 8;
          src[0] = &src[2];
          src[1] = (void *)0x600000000LL;
          v99 = *(_QWORD *)(v96 + 80);
          v359 = v98;
          if ( v96 + 72 == v99 )
          {
            v103 = &src[2];
          }
          else
          {
            v100 = v98;
            do
            {
              if ( !v99 )
                goto LABEL_556;
              v101 = *(_QWORD *)(v99 + 24) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v101 != v99 + 24 )
              {
                if ( !v101 )
                  goto LABEL_201;
                if ( *(_BYTE *)(v101 - 24) == 30 )
                {
                  v102 = 0;
                  if ( (*(_DWORD *)(v101 - 20) & 0x7FFFFFF) != 0 )
                    v102 = *(_BYTE **)(v101 - 32LL * (*(_DWORD *)(v101 - 20) & 0x7FFFFFF) - 24);
                  if ( !(unsigned __int8)sub_25DCD90(v100, v102, (__int64)src) )
                    break;
                }
              }
              v99 = *(_QWORD *)(v99 + 8);
            }
            while ( v97 != v99 );
            v103 = (void **)src[0];
          }
          if ( v97 == v99 )
            break;
LABEL_167:
          if ( v103 != &src[2] )
            _libc_free((unsigned __int64)v103);
LABEL_169:
          v374 = (_QWORD *)v374[1];
          if ( v374 == v369 )
          {
            v337 |= v333;
            v104 = v388;
            v105 = 16LL * v390;
            goto LABEL_171;
          }
        }
        v116 = &v103[LODWORD(src[1])];
        if ( v116 == v103 )
          goto LABEL_248;
        v117 = (unsigned __int64 *)v103;
        while ( 1 )
        {
          v121 = *v117;
          v391 = *v117;
          if ( !v390 )
            break;
          v118 = (v390 - 1) & (((unsigned int)v121 >> 9) ^ ((unsigned int)v121 >> 4));
          v119 = (__int64 *)(v388 + 16LL * v118);
          v120 = *v119;
          if ( v121 == *v119 )
          {
LABEL_217:
            if ( v116 == (void **)++v117 )
              goto LABEL_224;
          }
          else
          {
            v215 = 1;
            v122 = 0;
            while ( v120 != -4096 )
            {
              if ( v122 || v120 != -8192 )
                v119 = v122;
              v118 = (v390 - 1) & (v215 + v118);
              v120 = *(_QWORD *)(v388 + 16LL * v118);
              if ( v121 == v120 )
                goto LABEL_217;
              ++v215;
              v122 = v119;
              v119 = (__int64 *)(v388 + 16LL * v118);
            }
            if ( !v122 )
              v122 = v119;
            ++v387;
            v123 = v389 + 1;
            v402[0] = v122;
            if ( 4 * ((int)v389 + 1) < 3 * v390 )
            {
              if ( v390 - HIDWORD(v389) - v123 <= v390 >> 3 )
              {
                sub_9DDA50((__int64)&v387, v390);
                sub_25E0C90((__int64)&v387, (__int64 *)&v391, v402);
                v121 = v391;
                v122 = v402[0];
                v123 = v389 + 1;
              }
              goto LABEL_221;
            }
LABEL_220:
            sub_9DDA50((__int64)&v387, 2 * v390);
            sub_25E0C90((__int64)&v387, (__int64 *)&v391, v402);
            v121 = v391;
            v122 = v402[0];
            v123 = v389 + 1;
LABEL_221:
            LODWORD(v389) = v123;
            if ( *v122 != -4096 )
              --HIDWORD(v389);
            *v122 = v121;
            ++v117;
            v122[1] = 0;
            v122[1] = sub_DFE460(v359);
            if ( v116 == (void **)v117 )
            {
LABEL_224:
              v124 = (__int64 *)src[0];
              v125 = 8LL * LODWORD(src[1]);
              v345 = (__int64 *)((char *)src[0] + v125);
              v126 = (__int64 *)((char *)src[0] + v125);
              if ( src[0] == (char *)src[0] + v125 )
                goto LABEL_248;
              _BitScanReverse64(&v127, v125 >> 3);
              sub_25E3800((__int64 *)src[0], (char *)src[0] + v125, 2LL * (int)(63 - (v127 ^ 0x3F)), (__int64)&v387);
              if ( (unsigned __int64)v125 > 0x80 )
              {
                sub_25E6960(v124, v124 + 16, (__int64)&v387);
                if ( v126 == v124 + 16 )
                  goto LABEL_248;
                v350 = v124 + 16;
                while ( 1 )
                {
                  v128 = v350;
                  v129 = *v350;
                  v130 = ((unsigned int)*v350 >> 9) ^ ((unsigned int)*v350 >> 4);
                  while ( 1 )
                  {
                    v144 = *(v128 - 1);
                    v145 = v390;
                    v322 = v128;
                    v383 = (__int64 *)v129;
                    v391 = v144;
                    if ( v390 )
                    {
                      v131 = v390 - 1;
                      v132 = v388;
                      v133 = 0;
                      v134 = (v390 - 1) & v130;
                      v135 = 1;
                      v136 = (__int64 *)(v388 + 16LL * v134);
                      v137 = *v136;
                      if ( v129 == *v136 )
                      {
LABEL_230:
                        v138 = v136[1];
                        goto LABEL_231;
                      }
                      while ( v137 != -4096 )
                      {
                        if ( !v133 && v137 == -8192 )
                          v133 = v136;
                        v134 = v131 & (v135 + v134);
                        v136 = (__int64 *)(v388 + 16LL * v134);
                        v137 = *v136;
                        if ( v129 == *v136 )
                          goto LABEL_230;
                        ++v135;
                      }
                      if ( !v133 )
                        v133 = v136;
                      ++v387;
                      v147 = v389 + 1;
                      v402[0] = v133;
                      if ( 4 * ((int)v389 + 1) < 3 * v390 )
                      {
                        v146 = v129;
                        if ( v390 - HIDWORD(v389) - v147 > v390 >> 3 )
                          goto LABEL_238;
                        goto LABEL_237;
                      }
                    }
                    else
                    {
                      ++v387;
                      v402[0] = 0;
                    }
                    v145 = 2 * v390;
LABEL_237:
                    sub_9DDA50((__int64)&v387, v145);
                    sub_25E0C90((__int64)&v387, (__int64 *)&v383, v402);
                    v146 = (__int64)v383;
                    v133 = v402[0];
                    v147 = v389 + 1;
LABEL_238:
                    LODWORD(v389) = v147;
                    if ( *v133 != -4096 )
                      --HIDWORD(v389);
                    *v133 = v146;
                    v133[1] = 0;
                    v145 = v390;
                    if ( !v390 )
                    {
                      ++v387;
                      v138 = 0;
                      v402[0] = 0;
                      goto LABEL_242;
                    }
                    v132 = v388;
                    v144 = v391;
                    v131 = v390 - 1;
                    v138 = 0;
LABEL_231:
                    v139 = 1;
                    v140 = 0;
                    v141 = v131 & (((unsigned int)v144 >> 9) ^ ((unsigned int)v144 >> 4));
                    v142 = (__int64 *)(v132 + 16LL * v141);
                    v143 = *v142;
                    if ( *v142 != v144 )
                      break;
LABEL_232:
                    --v128;
                    if ( v138 <= v142[1] )
                      goto LABEL_247;
LABEL_233:
                    v128[1] = *v128;
                  }
                  while ( v143 != -4096 )
                  {
                    if ( !v140 && v143 == -8192 )
                      v140 = v142;
                    v141 = v131 & (v139 + v141);
                    v142 = (__int64 *)(v132 + 16LL * v141);
                    v143 = *v142;
                    if ( v144 == *v142 )
                      goto LABEL_232;
                    ++v139;
                  }
                  if ( !v140 )
                    v140 = v142;
                  ++v387;
                  v148 = v389 + 1;
                  v402[0] = v140;
                  if ( 4 * ((int)v389 + 1) >= 3 * v145 )
                  {
LABEL_242:
                    v145 *= 2;
                    goto LABEL_243;
                  }
                  if ( v145 - (v148 + HIDWORD(v389)) <= v145 >> 3 )
                  {
LABEL_243:
                    sub_9DDA50((__int64)&v387, v145);
                    sub_25E0C90((__int64)&v387, (__int64 *)&v391, v402);
                    v144 = v391;
                    v140 = v402[0];
                    v148 = v389 + 1;
                  }
                  LODWORD(v389) = v148;
                  if ( *v140 != -4096 )
                    --HIDWORD(v389);
                  *v140 = v144;
                  --v128;
                  v140[1] = 0;
                  if ( v138 )
                    goto LABEL_233;
LABEL_247:
                  ++v350;
                  *v322 = v129;
                  if ( v345 == v350 )
                    goto LABEL_248;
                }
              }
              sub_25E6960(v124, v345, (__int64)&v387);
LABEL_248:
              v149 = v362;
              v391 = 0;
              v392 = 0;
              v402[0] = (__int64 *)&v402[2];
              v402[1] = (__int64 *)0x600000000LL;
              v393 = 0;
              v394 = 0;
              v150 = *(_QWORD *)(v362 + 16);
              if ( !v150 )
                goto LABEL_302;
              while ( 2 )
              {
                while ( 1 )
                {
                  v151 = *(unsigned __int8 **)(v150 + 24);
                  v152 = *v151;
                  if ( (unsigned __int8)v152 > 0x1Cu )
                  {
                    v153 = (unsigned int)(v152 - 34);
                    if ( (unsigned __int8)v153 <= 0x33u )
                    {
                      v154 = 0x8000000000041LL;
                      if ( _bittest64(&v154, v153) )
                      {
                        if ( v149 == *((_QWORD *)v151 - 4) )
                          break;
                      }
                    }
                  }
                  v150 = *(_QWORD *)(v150 + 8);
                  if ( !v150 )
                    goto LABEL_259;
                }
                v381 = sub_B43CB0(*(_QWORD *)(v150 + 24));
                if ( (unsigned __int8)sub_25E0C90((__int64)&v387, &v381, &v382) )
                {
LABEL_256:
                  v155 = (unsigned __int8)sub_25E0D50((__int64)&v391, &v381, &v382) == 0;
                  v157 = (__int64)v382;
                  if ( !v155 )
                  {
                    v158 = *((unsigned int *)v382 + 4);
                    v159 = v158 + 1;
                    if ( v158 + 1 <= (unsigned __int64)*((unsigned int *)v382 + 5) )
                      goto LABEL_258;
LABEL_449:
                    v364 = v157;
                    sub_C8D5F0(v157 + 8, (const void *)(v157 + 24), v159, 8u, v159, v156);
                    v157 = v364;
                    v158 = *(unsigned int *)(v364 + 16);
LABEL_258:
                    *(_QWORD *)(*(_QWORD *)(v157 + 8) + 8 * v158) = v151;
                    ++*(_DWORD *)(v157 + 16);
                    v150 = *(_QWORD *)(v150 + 8);
                    if ( v150 )
                      continue;
LABEL_259:
                    v160 = v402[0];
                    v161 = LODWORD(v402[1]);
                    v346 = &v402[0][v161];
                    v162 = &v402[0][v161];
                    if ( v402[0] != &v402[0][v161] )
                    {
                      _BitScanReverse64(&v163, (__int64)(v161 * 8) >> 3);
                      sub_25E4D10(v402[0], (char *)&v402[0][v161], 2LL * (int)(63 - (v163 ^ 0x3F)), (__int64)&v387);
                      if ( v161 > 16 )
                      {
                        sub_25E6120(v160, v160 + 16, (__int64)&v387);
                        if ( v162 == v160 + 16 )
                          goto LABEL_283;
                        v363 = v160 + 16;
                        while ( 1 )
                        {
                          v164 = v363;
                          v165 = *v363;
                          v166 = ((unsigned int)*v363 >> 9) ^ ((unsigned int)*v363 >> 4);
                          while ( 1 )
                          {
                            v180 = *(v164 - 1);
                            v181 = v390;
                            v323 = v164;
                            v381 = v165;
                            v382 = (__int64 *)v180;
                            if ( v390 )
                            {
                              v167 = v390 - 1;
                              v168 = v388;
                              v169 = 0;
                              v170 = (v390 - 1) & v166;
                              v171 = 1;
                              v172 = (__int64 *)(v388 + 16LL * v170);
                              v173 = *v172;
                              if ( v165 == *v172 )
                              {
LABEL_265:
                                v174 = v172[1];
                                goto LABEL_266;
                              }
                              while ( v173 != -4096 )
                              {
                                if ( !v169 && v173 == -8192 )
                                  v169 = v172;
                                v170 = v167 & (v171 + v170);
                                v172 = (__int64 *)(v388 + 16LL * v170);
                                v173 = *v172;
                                if ( v165 == *v172 )
                                  goto LABEL_265;
                                ++v171;
                              }
                              if ( !v169 )
                                v169 = v172;
                              ++v387;
                              v183 = v389 + 1;
                              v383 = v169;
                              if ( 4 * ((int)v389 + 1) < 3 * v390 )
                              {
                                v182 = v165;
                                if ( v390 - HIDWORD(v389) - v183 > v390 >> 3 )
                                  goto LABEL_273;
                                goto LABEL_272;
                              }
                            }
                            else
                            {
                              ++v387;
                              v383 = 0;
                            }
                            v181 = 2 * v390;
LABEL_272:
                            sub_9DDA50((__int64)&v387, v181);
                            sub_25E0C90((__int64)&v387, &v381, &v383);
                            v182 = v381;
                            v169 = v383;
                            v183 = v389 + 1;
LABEL_273:
                            LODWORD(v389) = v183;
                            if ( *v169 != -4096 )
                              --HIDWORD(v389);
                            *v169 = v182;
                            v169[1] = 0;
                            v181 = v390;
                            if ( !v390 )
                            {
                              ++v387;
                              v174 = 0;
                              v383 = 0;
                              goto LABEL_277;
                            }
                            v168 = v388;
                            v180 = (__int64)v382;
                            v167 = v390 - 1;
                            v174 = 0;
LABEL_266:
                            v175 = 0;
                            v176 = 1;
                            v177 = v167 & (((unsigned int)v180 >> 9) ^ ((unsigned int)v180 >> 4));
                            v178 = (__int64 *)(v168 + 16LL * v177);
                            v179 = *v178;
                            if ( v180 != *v178 )
                              break;
LABEL_267:
                            --v164;
                            if ( v174 <= v178[1] )
                              goto LABEL_282;
LABEL_268:
                            v164[1] = *v164;
                          }
                          while ( v179 != -4096 )
                          {
                            if ( !v175 && v179 == -8192 )
                              v175 = v178;
                            v177 = v167 & (v176 + v177);
                            v178 = (__int64 *)(v168 + 16LL * v177);
                            v179 = *v178;
                            if ( v180 == *v178 )
                              goto LABEL_267;
                            ++v176;
                          }
                          if ( !v175 )
                            v175 = v178;
                          ++v387;
                          v184 = v389 + 1;
                          v383 = v175;
                          if ( 4 * ((int)v389 + 1) >= 3 * v181 )
                          {
LABEL_277:
                            v181 *= 2;
                            goto LABEL_278;
                          }
                          if ( v181 - (v184 + HIDWORD(v389)) <= v181 >> 3 )
                          {
LABEL_278:
                            sub_9DDA50((__int64)&v387, v181);
                            sub_25E0C90((__int64)&v387, (__int64 *)&v382, &v383);
                            v180 = (__int64)v382;
                            v175 = v383;
                            v184 = v389 + 1;
                          }
                          LODWORD(v389) = v184;
                          if ( *v175 != -4096 )
                            --HIDWORD(v389);
                          *v175 = v180;
                          --v164;
                          v175[1] = 0;
                          if ( v174 )
                            goto LABEL_268;
LABEL_282:
                          ++v363;
                          *v323 = v165;
                          if ( v346 == v363 )
                            goto LABEL_283;
                        }
                      }
                      sub_25E6120(v160, v346, (__int64)&v387);
LABEL_283:
                      v351 = &v402[0][LODWORD(v402[1])];
                      if ( v402[0] != v351 )
                      {
                        v185 = 0;
                        v186 = v402[0];
                        while ( 1 )
                        {
                          v380 = *v186;
                          v381 = *((_QWORD *)src[0] + v185);
                          v187 = *sub_9DDC30((__int64)&v387, &v380);
                          v188 = *sub_9DDC30((__int64)&v387, &v381);
                          if ( (unsigned __int8)sub_DFE490(v359) )
                          {
                            if ( v187 == v188 )
                            {
                              ++v185;
                            }
                            else if ( v188 != (v188 & v187) )
                            {
                              if ( v187 != (v188 & v187) )
                                goto LABEL_301;
                              v189 = v186;
                              while ( 2 )
                              {
                                while ( 2 )
                                {
                                  v195 = v185 + 1;
                                  v185 = v195;
                                  if ( (_DWORD)v195 == LODWORD(src[1]) )
                                  {
LABEL_300:
                                    v186 = v189;
                                    goto LABEL_301;
                                  }
                                  v196 = v390;
                                  v197 = (__int64 *)((char *)src[0] + 8 * v195);
                                  if ( !v390 )
                                  {
                                    ++v387;
                                    v383 = 0;
                                    goto LABEL_295;
                                  }
                                  v190 = 0;
                                  v191 = 1;
                                  v192 = (v390 - 1) & (((unsigned int)*v197 >> 9) ^ ((unsigned int)*v197 >> 4));
                                  v193 = (__int64 *)(v388 + 16LL * v192);
                                  v194 = *v193;
                                  if ( *v193 == *v197 )
                                  {
LABEL_291:
                                    if ( v187 != (v187 & v193[1]) )
                                      goto LABEL_300;
                                    continue;
                                  }
                                  break;
                                }
                                while ( v194 != -4096 )
                                {
                                  if ( v194 == -8192 && !v190 )
                                    v190 = v193;
                                  v192 = (v390 - 1) & (v191 + v192);
                                  v193 = (__int64 *)(v388 + 16LL * v192);
                                  v194 = *v193;
                                  if ( *v197 == *v193 )
                                    goto LABEL_291;
                                  ++v191;
                                }
                                if ( !v190 )
                                  v190 = v193;
                                ++v387;
                                v198 = v389 + 1;
                                v383 = v190;
                                if ( 4 * ((int)v389 + 1) < 3 * v390 )
                                {
                                  if ( v390 - HIDWORD(v389) - v198 <= v390 >> 3 )
                                  {
LABEL_296:
                                    sub_9DDA50((__int64)&v387, v196);
                                    sub_25E0C90((__int64)&v387, v197, &v383);
                                    v190 = v383;
                                    v198 = v389 + 1;
                                  }
                                  LODWORD(v389) = v198;
                                  if ( *v190 != -4096 )
                                    --HIDWORD(v389);
                                  v199 = *v197;
                                  v190[1] = 0;
                                  *v190 = v199;
                                  if ( v187 )
                                    goto LABEL_300;
                                  continue;
                                }
                                break;
                              }
LABEL_295:
                              v196 = 2 * v390;
                              goto LABEL_296;
                            }
LABEL_349:
                            v155 = (unsigned __int8)sub_25E0D50((__int64)&v391, &v380, &v382) == 0;
                            v204 = v382;
                            if ( !v155 )
                            {
                              v208 = (__int64 *)v382[1];
                              for ( k = &v208[*((unsigned int *)v382 + 4)]; k != v208; ++v208 )
                              {
                                v210 = *v208;
                                v211 = v381;
                                if ( *(_QWORD *)(*v208 - 32) )
                                {
                                  v212 = *(_QWORD *)(v210 - 24);
                                  **(_QWORD **)(v210 - 16) = v212;
                                  if ( v212 )
                                    *(_QWORD *)(v212 + 16) = *(_QWORD *)(v210 - 16);
                                }
                                *(_QWORD *)(v210 - 32) = v211;
                                if ( v211 )
                                {
                                  v213 = *(_QWORD *)(v211 + 16);
                                  *(_QWORD *)(v210 - 24) = v213;
                                  if ( v213 )
                                    *(_QWORD *)(v213 + 16) = v210 - 24;
                                  *(_QWORD *)(v210 - 16) = v211 + 16;
                                  *(_QWORD *)(v211 + 16) = v210 - 32;
                                }
                              }
                              goto LABEL_355;
                            }
                            v205 = v394;
                            v383 = v382;
                            ++v391;
                            v206 = (_DWORD)v393 + 1;
                            if ( 4 * ((int)v393 + 1) >= 3 * v394 )
                            {
                              v205 = 2 * v394;
LABEL_458:
                              sub_25E71A0((__int64)&v391, v205);
                              sub_25E0D50((__int64)&v391, &v380, &v383);
                              v206 = (_DWORD)v393 + 1;
                              v204 = v383;
                              goto LABEL_352;
                            }
                            if ( v394 - HIDWORD(v393) - v206 <= v394 >> 3 )
                              goto LABEL_458;
LABEL_352:
                            LODWORD(v393) = v206;
                            if ( *v204 != -4096 )
                              --HIDWORD(v393);
                            v207 = v380;
                            v204[2] = 0x600000000LL;
                            *v204 = v207;
                            v204[1] = (__int64)(v204 + 3);
LABEL_355:
                            v333 = 1;
                            if ( v351 == ++v186 )
                              break;
                          }
                          else
                          {
                            if ( (_BYTE)qword_4FF1828 == 1 && !v185 && v188 == (v188 & v187) )
                              goto LABEL_349;
LABEL_301:
                            if ( v351 == ++v186 )
                              break;
                          }
                        }
                      }
                    }
LABEL_302:
                    v200 = v394;
                    if ( v394 )
                    {
                      v201 = v392;
                      v202 = &v392[72 * v394];
                      do
                      {
                        if ( *v201 != -4096 && *v201 != -8192 )
                        {
                          v203 = v201[1];
                          if ( (_QWORD *)v203 != v201 + 3 )
                            _libc_free(v203);
                        }
                        v201 += 9;
                      }
                      while ( v202 != (_BYTE *)v201 );
                      v200 = v394;
                    }
                    sub_C7D6A0((__int64)v392, 72 * v200, 8);
                    if ( (__int64 **)v402[0] != &v402[2] )
                      _libc_free((unsigned __int64)v402[0]);
                    v103 = (void **)src[0];
                    goto LABEL_167;
                  }
                  ++v391;
                  v383 = v382;
                  v235 = (_DWORD)v393 + 1;
                  v236 = v394;
                  if ( 4 * ((int)v393 + 1) >= 3 * v394 )
                  {
                    v236 = 2 * v394;
                  }
                  else if ( v394 - HIDWORD(v393) - v235 > v394 >> 3 )
                  {
                    goto LABEL_444;
                  }
                  sub_25E71A0((__int64)&v391, v236);
                  sub_25E0D50((__int64)&v391, &v381, &v383);
                  v235 = (_DWORD)v393 + 1;
                  v157 = (__int64)v383;
LABEL_444:
                  LODWORD(v393) = v235;
                  if ( *(_QWORD *)v157 != -4096 )
                    --HIDWORD(v393);
                  v237 = v381;
                  *(_QWORD *)(v157 + 16) = 0x600000000LL;
                  *(_QWORD *)v157 = v237;
                  *(_QWORD *)(v157 + 8) = v157 + 24;
                  v238 = LODWORD(v402[1]);
                  v156 = v381;
                  v239 = LODWORD(v402[1]) + 1LL;
                  if ( v239 > HIDWORD(v402[1]) )
                  {
                    v354 = v381;
                    v365 = v157;
                    sub_C8D5F0((__int64)v402, &v402[2], LODWORD(v402[1]) + 1LL, 8u, v239, v381);
                    v238 = LODWORD(v402[1]);
                    v156 = v354;
                    v157 = v365;
                  }
                  v402[0][v238] = v156;
                  ++LODWORD(v402[1]);
                  v158 = *(unsigned int *)(v157 + 16);
                  v159 = v158 + 1;
                  if ( v158 + 1 > (unsigned __int64)*(unsigned int *)(v157 + 20) )
                    goto LABEL_449;
                  goto LABEL_258;
                }
                break;
              }
              v240 = v390;
              v241 = v382;
              ++v387;
              v242 = v389 + 1;
              v383 = v382;
              if ( 4 * ((int)v389 + 1) >= 3 * v390 )
              {
                v240 = 2 * v390;
              }
              else if ( v390 - HIDWORD(v389) - v242 > v390 >> 3 )
              {
LABEL_452:
                LODWORD(v389) = v242;
                if ( *v241 != -4096 )
                  --HIDWORD(v389);
                v243 = v381;
                v241[1] = 0;
                *v241 = v243;
                v241[1] = sub_DFE460(v359);
                goto LABEL_256;
              }
              sub_9DDA50((__int64)&v387, v240);
              sub_25E0C90((__int64)&v387, &v381, &v383);
              v241 = v383;
              v242 = v389 + 1;
              goto LABEL_452;
            }
          }
        }
        ++v387;
        v402[0] = 0;
        goto LABEL_220;
      }
      v104 = 0;
      v105 = 0;
    }
LABEL_171:
    v106 = 0;
    sub_C7D6A0(v104, v105, 8);
    v107 = (_QWORD *)a3[8];
    if ( v107 != v369 )
    {
      do
      {
        v108 = (__int64)(v107 - 7);
        v107 = (_QWORD *)v107[1];
        v109 = sub_25DD4B0(v108, (__int64)&v396, 0, 0);
        if ( v109 )
          v106 = v109;
      }
      while ( v107 != v369 );
      v337 |= v106;
    }
    v5 = v337;
    v10 = v399;
    if ( !v337 )
      break;
    v315 = v337;
  }
  if ( !v399 )
    _libc_free((unsigned __int64)s);
  v259 = (void *)(a1 + 32);
  v260 = (void *)(a1 + 80);
  if ( v315 )
  {
    memset(v402, 0, 0x60u);
    BYTE4(v402[9]) = 1;
    v402[1] = (__int64 *)&v402[4];
    LODWORD(v402[2]) = 2;
    BYTE4(v402[3]) = 1;
    v402[7] = (__int64 *)&v402[10];
    LODWORD(v402[8]) = 2;
    sub_AE6EC0((__int64)v402, (__int64)&unk_4F82418);
    if ( HIDWORD(v402[8]) != LODWORD(v402[9])
      || !(unsigned __int8)sub_B19060((__int64)v402, (__int64)&qword_4F82400, v261, v262) )
    {
      sub_AE6EC0((__int64)v402, (__int64)&unk_4F82408);
    }
    sub_C8CF70(a1, v259, 2, (__int64)&v402[4], (__int64)v402);
    sub_C8CF70(a1 + 48, v260, 2, (__int64)&v402[10], (__int64)&v402[6]);
    if ( !BYTE4(v402[9]) )
      _libc_free((unsigned __int64)v402[7]);
    if ( !BYTE4(v402[3]) )
      _libc_free((unsigned __int64)v402[1]);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v259;
    *(_QWORD *)(a1 + 56) = v260;
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 16) = 2;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    sub_AE6EC0(a1, (__int64)&qword_4F82400);
  }
  return a1;
}
