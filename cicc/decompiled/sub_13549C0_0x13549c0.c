// Function: sub_13549C0
// Address: 0x13549c0
//
__int64 __fastcall sub_13549C0(_QWORD *a1, __int64 *a2, _QWORD *a3)
{
  __int64 *v4; // rdi
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v7; // rdx
  unsigned __int64 *v8; // rax
  char *v9; // rbx
  char *v10; // r12
  __int64 *v11; // r12
  __int64 *v12; // r15
  __int64 *v13; // r13
  __int64 v14; // rax
  void *v15; // rdx
  __int64 v16; // r12
  __int64 v17; // rax
  size_t v18; // rdx
  _WORD *v19; // rdi
  const char *v20; // rsi
  unsigned __int64 v21; // rax
  __int64 v22; // rax
  void *v23; // rdx
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 *i; // r12
  __int64 v27; // r13
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rax
  __int64 *v30; // rbx
  char *v31; // r14
  char v32; // al
  __int64 v33; // r9
  char *v34; // rdx
  unsigned __int64 v35; // rax
  char *v36; // rcx
  unsigned __int8 v37; // al
  unsigned __int64 v38; // rax
  __int64 v39; // rcx
  __int64 v40; // rcx
  __int64 v41; // rax
  int v42; // eax
  int v43; // eax
  unsigned int v44; // eax
  __int64 v45; // rsi
  __int64 v46; // r10
  unsigned __int64 v47; // r9
  _QWORD *v48; // rax
  __int64 *v49; // rcx
  __int64 v50; // r13
  __int64 *v51; // r15
  __int64 v52; // r14
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  unsigned __int8 v56; // al
  __int64 *v57; // rbx
  __int64 *v58; // rcx
  __int64 *v59; // r14
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // r8
  unsigned __int8 v63; // al
  char **v64; // r15
  unsigned __int64 v65; // r14
  char *v66; // r12
  __int64 v67; // r8
  unsigned __int64 v68; // rax
  char v69; // al
  unsigned __int64 v70; // rax
  unsigned __int8 v71; // dl
  char *v72; // rsi
  __int64 v73; // rdx
  __int64 v74; // rax
  _WORD *v75; // rdx
  __int64 v76; // rdi
  _QWORD *v77; // rdx
  _QWORD *v78; // rdx
  __int64 v79; // rax
  __int64 v80; // rsi
  __int64 v81; // rdx
  __int64 v82; // rax
  _DWORD *v83; // rdx
  __int64 v84; // r12
  _BYTE *v85; // rax
  __int64 *v86; // r13
  __int64 *v87; // r12
  __int64 v88; // r9
  unsigned __int64 v89; // rax
  unsigned __int8 v90; // dl
  __int64 v91; // rsi
  unsigned __int64 v92; // rax
  unsigned __int8 v93; // dl
  __int64 v94; // rax
  _WORD *v95; // rdx
  __int64 v96; // rdi
  _QWORD *v97; // rdx
  _QWORD *v98; // rdx
  __int64 v99; // rsi
  __int64 v100; // rdx
  __int64 v101; // rax
  _DWORD *v102; // rdx
  __int64 v103; // r12
  _BYTE *v104; // rax
  __int64 v105; // rax
  _WORD *v106; // rdx
  __int64 v107; // rdi
  __int64 v108; // rdx
  __m128i v109; // xmm0
  _QWORD *v110; // rdx
  __int64 v111; // rsi
  __int64 v112; // rdx
  __int64 v113; // rax
  _DWORD *v114; // rdx
  __int64 v115; // r12
  _BYTE *v116; // rax
  __int64 v117; // rax
  _WORD *v118; // rdx
  __int64 v119; // rdi
  char *v120; // rsi
  __int64 v121; // rdx
  __m128i si128; // xmm0
  _QWORD *v123; // rdx
  __int64 v124; // rsi
  __int64 v125; // rdx
  __int64 v126; // rax
  _DWORD *v127; // rdx
  __int64 v128; // r12
  _BYTE *v129; // rax
  __int64 v130; // rax
  _WORD *v131; // rdx
  __int64 v132; // rdi
  __int64 v133; // rdx
  __m128i v134; // xmm0
  _QWORD *v135; // rdx
  __int64 v136; // rsi
  __int64 v137; // rdx
  __int64 v138; // rax
  _DWORD *v139; // rdx
  __int64 v140; // r12
  _BYTE *v141; // rax
  __int64 v142; // rax
  _WORD *v143; // rdx
  __int64 v144; // rdi
  _DWORD *v145; // rdx
  _QWORD *v146; // rdx
  __int64 v147; // rsi
  __int64 v148; // rdx
  __int64 v149; // rax
  _DWORD *v150; // rdx
  __int64 v151; // r12
  _BYTE *v152; // rax
  __int64 v153; // rax
  _WORD *v154; // rdx
  __int64 v155; // rdi
  void *v156; // rdx
  _QWORD *v157; // rdx
  __int64 v158; // rsi
  __int64 v159; // rdx
  __int64 v160; // rax
  _DWORD *v161; // rdx
  __int64 v162; // r12
  _BYTE *v163; // rax
  __int64 v164; // rax
  _WORD *v165; // rdx
  __int64 v166; // rdi
  _QWORD *v167; // rdx
  _QWORD *v168; // rdx
  __int64 v169; // rsi
  __int64 v170; // rdx
  __int64 v171; // rax
  _DWORD *v172; // rdx
  __int64 v173; // r12
  _BYTE *v174; // rax
  __int64 v175; // rdx
  __int64 v176; // r13
  __int64 v177; // rax
  unsigned int v178; // eax
  __int64 v179; // rsi
  __int64 v180; // r8
  unsigned __int64 v181; // r9
  __int64 v182; // r14
  __int64 v183; // rax
  __int64 v184; // rax
  __int64 v185; // rax
  __int64 v186; // rax
  __int64 v187; // rax
  __int64 v188; // rax
  __int64 v189; // rax
  __int64 v190; // rax
  __int64 v191; // rax
  __int64 v192; // rax
  __int64 v193; // rax
  __int64 v194; // rax
  __int64 v195; // rax
  __int64 v196; // rax
  __int64 v197; // rax
  __int64 v199; // rax
  __int64 v200; // rax
  __int64 v201; // rax
  __int64 v202; // rbx
  __int64 v203; // rsi
  unsigned __int64 v204; // r13
  unsigned __int64 v205; // rax
  __int64 v206; // rsi
  int v207; // eax
  __int64 v208; // rax
  unsigned int v209; // eax
  __int64 v210; // rsi
  _QWORD *v211; // rax
  __int64 v212; // rsi
  int v213; // eax
  unsigned int v214; // eax
  __int64 v215; // rsi
  _QWORD *v216; // rax
  __int64 v217; // rax
  __int64 *v218; // r14
  unsigned __int8 v219; // al
  __int64 v220; // rax
  __int64 *v221; // r14
  __int64 *v222; // rcx
  __int64 *v223; // rax
  __int64 v224; // rax
  unsigned __int64 v225; // rdx
  char **v226; // rcx
  char *v227; // rcx
  char v228; // si
  __int64 v229; // rcx
  unsigned __int64 v230; // r14
  __int64 v231; // rdx
  unsigned __int64 v232; // r14
  __int64 v233; // rax
  __int64 v234; // rax
  __int64 v235; // rax
  __int64 v236; // rax
  __int64 v237; // rax
  __int64 v238; // rax
  __int64 v239; // rsi
  int v240; // eax
  _QWORD *v241; // rax
  __int64 v242; // rax
  __int64 v243; // rax
  __int64 v244; // rsi
  int v245; // eax
  __int64 v246; // rax
  _QWORD *v247; // rax
  __int64 v248; // [rsp+8h] [rbp-308h]
  __int64 v249; // [rsp+10h] [rbp-300h]
  __int64 v250; // [rsp+10h] [rbp-300h]
  __int64 v251; // [rsp+18h] [rbp-2F8h]
  __int64 v252; // [rsp+18h] [rbp-2F8h]
  __int64 v253; // [rsp+18h] [rbp-2F8h]
  __int64 v254; // [rsp+20h] [rbp-2F0h]
  __int64 v255; // [rsp+20h] [rbp-2F0h]
  __int64 v256; // [rsp+20h] [rbp-2F0h]
  unsigned __int64 v257; // [rsp+20h] [rbp-2F0h]
  __int64 v258; // [rsp+20h] [rbp-2F0h]
  __int64 v259; // [rsp+28h] [rbp-2E8h]
  __int64 v260; // [rsp+28h] [rbp-2E8h]
  __int64 v261; // [rsp+28h] [rbp-2E8h]
  __int64 v262; // [rsp+28h] [rbp-2E8h]
  __int64 v263; // [rsp+28h] [rbp-2E8h]
  __int64 v264; // [rsp+28h] [rbp-2E8h]
  unsigned __int64 v265; // [rsp+30h] [rbp-2E0h]
  __int64 v266; // [rsp+30h] [rbp-2E0h]
  unsigned __int64 v267; // [rsp+30h] [rbp-2E0h]
  unsigned __int64 v268; // [rsp+30h] [rbp-2E0h]
  unsigned __int64 v269; // [rsp+38h] [rbp-2D8h]
  __int64 v270; // [rsp+38h] [rbp-2D8h]
  unsigned __int64 v271; // [rsp+38h] [rbp-2D8h]
  __int64 v272; // [rsp+38h] [rbp-2D8h]
  __int64 v273; // [rsp+38h] [rbp-2D8h]
  __int64 v274; // [rsp+38h] [rbp-2D8h]
  __int64 v275; // [rsp+48h] [rbp-2C8h]
  __int64 v276; // [rsp+48h] [rbp-2C8h]
  __int64 v277; // [rsp+48h] [rbp-2C8h]
  __int64 v278; // [rsp+48h] [rbp-2C8h]
  __int64 v279; // [rsp+48h] [rbp-2C8h]
  unsigned __int64 v280; // [rsp+48h] [rbp-2C8h]
  __int64 v281; // [rsp+48h] [rbp-2C8h]
  __int64 v282; // [rsp+48h] [rbp-2C8h]
  __int64 v283; // [rsp+50h] [rbp-2C0h]
  _QWORD *v284; // [rsp+50h] [rbp-2C0h]
  unsigned __int64 v285; // [rsp+50h] [rbp-2C0h]
  __int64 v286; // [rsp+50h] [rbp-2C0h]
  unsigned __int64 v287; // [rsp+50h] [rbp-2C0h]
  unsigned __int64 v288; // [rsp+50h] [rbp-2C0h]
  __int64 v289; // [rsp+58h] [rbp-2B8h]
  __int64 v290; // [rsp+60h] [rbp-2B0h]
  __int64 *v291; // [rsp+60h] [rbp-2B0h]
  __int64 v292; // [rsp+60h] [rbp-2B0h]
  __int64 v293; // [rsp+60h] [rbp-2B0h]
  __int64 v294; // [rsp+60h] [rbp-2B0h]
  __int64 v295; // [rsp+60h] [rbp-2B0h]
  __int64 v296; // [rsp+60h] [rbp-2B0h]
  __int64 v297; // [rsp+60h] [rbp-2B0h]
  __int64 *v298; // [rsp+68h] [rbp-2A8h]
  __int64 *v299; // [rsp+68h] [rbp-2A8h]
  _QWORD *v300; // [rsp+68h] [rbp-2A8h]
  __int64 v302; // [rsp+78h] [rbp-298h]
  __int64 v303; // [rsp+78h] [rbp-298h]
  __int64 v304; // [rsp+78h] [rbp-298h]
  __int64 v305; // [rsp+78h] [rbp-298h]
  __int64 v306; // [rsp+78h] [rbp-298h]
  __int64 *v307; // [rsp+78h] [rbp-298h]
  char **v308; // [rsp+78h] [rbp-298h]
  __int64 v309; // [rsp+78h] [rbp-298h]
  size_t v310; // [rsp+78h] [rbp-298h]
  __int64 *v311; // [rsp+78h] [rbp-298h]
  __int64 v312; // [rsp+78h] [rbp-298h]
  __int64 *v313; // [rsp+80h] [rbp-290h]
  __int64 *v314; // [rsp+80h] [rbp-290h]
  __int64 v316; // [rsp+98h] [rbp-278h] BYREF
  char *v317[6]; // [rsp+A0h] [rbp-270h] BYREF
  char *v318; // [rsp+D0h] [rbp-240h] BYREF
  unsigned __int64 v319; // [rsp+D8h] [rbp-238h]
  __int64 v320; // [rsp+E0h] [rbp-230h]
  __int64 v321; // [rsp+E8h] [rbp-228h]
  __int64 v322; // [rsp+F0h] [rbp-220h]
  __int64 v323; // [rsp+100h] [rbp-210h] BYREF
  __int64 v324; // [rsp+108h] [rbp-208h]
  __int64 v325; // [rsp+110h] [rbp-200h]
  __int64 v326; // [rsp+118h] [rbp-1F8h]
  __int64 *v327; // [rsp+120h] [rbp-1F0h]
  __int64 *v328; // [rsp+128h] [rbp-1E8h]
  __int64 v329; // [rsp+130h] [rbp-1E0h]
  __int64 v330; // [rsp+140h] [rbp-1D0h] BYREF
  __int64 v331; // [rsp+148h] [rbp-1C8h]
  __int64 v332; // [rsp+150h] [rbp-1C0h]
  __int64 v333; // [rsp+158h] [rbp-1B8h]
  __int64 *v334; // [rsp+160h] [rbp-1B0h]
  __int64 *v335; // [rsp+168h] [rbp-1A8h]
  __int64 v336; // [rsp+170h] [rbp-1A0h]
  __int64 v337; // [rsp+180h] [rbp-190h] BYREF
  __int64 v338; // [rsp+188h] [rbp-188h]
  __int64 v339; // [rsp+190h] [rbp-180h]
  __int64 v340; // [rsp+198h] [rbp-178h]
  __int64 *v341; // [rsp+1A0h] [rbp-170h]
  __int64 *v342; // [rsp+1A8h] [rbp-168h]
  __int64 v343; // [rsp+1B0h] [rbp-160h]
  __int64 v344; // [rsp+1C0h] [rbp-150h] BYREF
  __int64 v345; // [rsp+1C8h] [rbp-148h]
  __int64 v346; // [rsp+1D0h] [rbp-140h] BYREF
  _BYTE *v347; // [rsp+250h] [rbp-C0h] BYREF
  __int64 v348; // [rsp+258h] [rbp-B8h]
  _BYTE v349[176]; // [rsp+260h] [rbp-B0h] BYREF

  v4 = (__int64 *)a2[5];
  v313 = a2;
  v5 = sub_1632FA0(v4);
  ++*a1;
  v7 = (__int64)&v347;
  v323 = 0;
  v324 = 0;
  v325 = 0;
  v326 = 0;
  v327 = 0;
  v328 = 0;
  v329 = 0;
  v344 = 0;
  v345 = 1;
  v289 = v5;
  v8 = (unsigned __int64 *)&v346;
  do
    *v8++ = -4;
  while ( v8 != (unsigned __int64 *)&v347 );
  v330 = 0;
  v347 = v349;
  v348 = 0x1000000000LL;
  v331 = 0;
  v332 = 0;
  v333 = 0;
  v334 = 0;
  v335 = 0;
  v336 = 0;
  v337 = 0;
  v338 = 0;
  v339 = 0;
  v340 = 0;
  v341 = 0;
  v342 = 0;
  v343 = 0;
  if ( (*((_BYTE *)a2 + 18) & 1) != 0 )
  {
    v4 = a2;
    sub_15E08E0(a2);
    v9 = (char *)a2[11];
    v10 = &v9[40 * a2[12]];
    if ( (*((_BYTE *)a2 + 18) & 1) != 0 )
    {
      v4 = a2;
      sub_15E08E0(a2);
      v9 = (char *)a2[11];
    }
  }
  else
  {
    v9 = (char *)a2[11];
    v10 = &v9[40 * a2[12]];
  }
  while ( v10 != v9 )
  {
    while ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) != 15 )
    {
      v9 += 40;
      if ( v10 == v9 )
        goto LABEL_10;
    }
    v318 = v9;
    v4 = &v323;
    a2 = (__int64 *)&v318;
    v9 += 40;
    sub_13540B0((__int64)&v323, &v318);
  }
LABEL_10:
  v11 = (__int64 *)v313[10];
  v12 = v313 + 9;
  if ( v313 + 9 != v11 )
  {
    if ( !v11 )
      BUG();
    while ( 1 )
    {
      v13 = (__int64 *)v11[3];
      if ( v13 != v11 + 2 )
        break;
      v11 = (__int64 *)v11[1];
      if ( v12 == v11 )
        goto LABEL_16;
      if ( !v11 )
        BUG();
    }
    while ( 1 )
    {
      if ( v11 == v12 )
        goto LABEL_16;
      if ( !v13 )
LABEL_499:
        BUG();
      v218 = v13 - 3;
      if ( *(_BYTE *)(*(v13 - 3) + 8) == 15 )
      {
        v4 = &v323;
        a2 = (__int64 *)&v318;
        v318 = (char *)(v13 - 3);
        sub_13540B0((__int64)&v323, &v318);
      }
      v219 = *((_BYTE *)v13 - 8);
      if ( byte_4F96F40 )
      {
        if ( v219 != 54
          || (v4 = &v330,
              a2 = (__int64 *)&v318,
              v318 = (char *)(v13 - 3),
              sub_13540B0((__int64)&v330, &v318),
              v219 = *((_BYTE *)v13 - 8),
              byte_4F96F40) )
        {
          if ( v219 == 55 )
          {
            v4 = &v337;
            a2 = (__int64 *)&v318;
            v318 = (char *)(v13 - 3);
            sub_13540B0((__int64)&v337, &v318);
            v219 = *((_BYTE *)v13 - 8);
          }
        }
      }
      if ( v219 <= 0x17u )
      {
LABEL_408:
        v316 = 0;
LABEL_409:
        v220 = 24LL * (*((_DWORD *)v13 - 1) & 0xFFFFFFF);
        if ( (*((_BYTE *)v13 - 1) & 0x40) != 0 )
        {
          v221 = (__int64 *)*(v13 - 4);
          v222 = &v221[(unsigned __int64)v220 / 8];
        }
        else
        {
          v222 = v13 - 3;
          v221 = &v218[v220 / 0xFFFFFFFFFFFFFFF8LL];
        }
        while ( v221 != v222 )
        {
          if ( *(_BYTE *)(*(_QWORD *)*v221 + 8LL) == 15 && *(_BYTE *)(*v221 + 16) != 15 )
          {
            v4 = &v323;
            a2 = (__int64 *)&v318;
            v311 = v222;
            v318 = (char *)*v221;
            sub_13540B0((__int64)&v323, &v318);
            v222 = v311;
          }
          v221 += 3;
        }
        goto LABEL_417;
      }
      if ( v219 == 78 )
      {
        v224 = (unsigned __int64)v218 | 4;
        v225 = (unsigned __int64)v218 & 0xFFFFFFFFFFFFFFF8LL;
      }
      else
      {
        if ( v219 != 29 )
          goto LABEL_408;
        v224 = (unsigned __int64)v218 & 0xFFFFFFFFFFFFFFFBLL;
        v225 = (unsigned __int64)v218 & 0xFFFFFFFFFFFFFFF8LL;
      }
      v316 = v224;
      if ( !v225 )
        goto LABEL_409;
      v226 = (char **)(v225 - 72);
      LODWORD(v224) = (v224 >> 2) & 1;
      if ( (_DWORD)v224 )
        v226 = (char **)(v225 - 24);
      v227 = *v226;
      v317[0] = v227;
      v228 = v227[16];
      if ( v228 && *(_BYTE *)(*(_QWORD *)v227 + 8LL) == 15 && v228 != 15 )
      {
        sub_13540B0((__int64)&v323, v317);
        v225 = v316 & 0xFFFFFFFFFFFFFFF8LL;
        v224 = (v316 >> 2) & 1;
      }
      v229 = 24LL * (*(_DWORD *)(v225 + 20) & 0xFFFFFFF);
      if ( (*(_BYTE *)(v225 + 23) & 0x40) == 0 )
        break;
      v230 = *(_QWORD *)(v225 - 8);
      v231 = v230 + v229 - 24;
      if ( !(_BYTE)v224 )
        goto LABEL_435;
LABEL_436:
      while ( v231 != v230 )
      {
        if ( *(_BYTE *)(**(_QWORD **)v230 + 8LL) == 15 && *(_BYTE *)(*(_QWORD *)v230 + 16LL) != 15 )
        {
          v312 = v231;
          v318 = *(char **)v230;
          sub_13540B0((__int64)&v323, &v318);
          v231 = v312;
        }
        v230 += 24LL;
      }
      a2 = &v316;
      v4 = &v344;
      sub_13546D0((__int64)&v344, &v316);
LABEL_417:
      v13 = (__int64 *)v13[1];
      v7 = 0;
      while ( 1 )
      {
        v223 = v11 - 3;
        if ( !v11 )
          v223 = 0;
        if ( v13 != v223 + 5 )
          break;
        v11 = (__int64 *)v11[1];
        if ( v12 == v11 )
          break;
        if ( !v11 )
          BUG();
        v13 = (__int64 *)v11[3];
      }
    }
    v232 = v225;
    v231 = v225 - 24;
    v230 = v232 - v229;
    if ( (_BYTE)v224 )
      goto LABEL_436;
LABEL_435:
    v231 = v230 + v229 - 72;
    goto LABEL_436;
  }
LABEL_16:
  if ( byte_4F97AA0
    || byte_4F979C0
    || byte_4F978E0
    || byte_4F97800
    || byte_4F97720
    || byte_4F97640
    || byte_4F97480
    || byte_4F97560
    || byte_4F973A0 )
  {
    v14 = sub_16E8CB0(v4, a2, v7);
    v15 = *(void **)(v14 + 24);
    v16 = v14;
    if ( *(_QWORD *)(v14 + 16) - (_QWORD)v15 <= 9u )
    {
      v16 = sub_16E7EE0(v14, "Function: ", 10);
    }
    else
    {
      qmemcpy(v15, "Function: ", 10);
      *(_QWORD *)(v14 + 24) += 10LL;
    }
    v17 = sub_1649960(v313);
    v19 = *(_WORD **)(v16 + 24);
    v20 = (const char *)v17;
    v21 = *(_QWORD *)(v16 + 16) - (_QWORD)v19;
    if ( v18 > v21 )
    {
      v233 = sub_16E7EE0(v16, v20);
      v19 = *(_WORD **)(v233 + 24);
      v16 = v233;
      v21 = *(_QWORD *)(v233 + 16) - (_QWORD)v19;
    }
    else if ( v18 )
    {
      v310 = v18;
      memcpy(v19, v20, v18);
      v19 = (_WORD *)(v310 + *(_QWORD *)(v16 + 24));
      v205 = *(_QWORD *)(v16 + 16) - (_QWORD)v19;
      *(_QWORD *)(v16 + 24) = v19;
      if ( v205 > 1 )
        goto LABEL_23;
      goto LABEL_358;
    }
    if ( v21 > 1 )
    {
LABEL_23:
      *v19 = 8250;
      *(_QWORD *)(v16 + 24) += 2LL;
LABEL_24:
      v22 = sub_16E7A90(v16, v328 - v327);
      v23 = *(void **)(v22 + 24);
      v24 = v22;
      if ( *(_QWORD *)(v22 + 16) - (_QWORD)v23 <= 0xAu )
      {
        v24 = sub_16E7EE0(v22, " pointers, ", 11);
      }
      else
      {
        qmemcpy(v23, " pointers, ", 11);
        *(_QWORD *)(v22 + 24) += 11LL;
      }
      v25 = sub_16E7A90(v24, (unsigned int)v348);
      v7 = *(_QWORD *)(v25 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v25 + 16) - v7) <= 0xB )
      {
        sub_16E7EE0(v25, " call sites\n", 12);
      }
      else
      {
        qmemcpy((void *)v7, " call sites\n", 12);
        *(_QWORD *)(v25 + 24) += 12LL;
      }
      goto LABEL_28;
    }
LABEL_358:
    v16 = sub_16E7EE0(v16, ": ", 2);
    goto LABEL_24;
  }
LABEL_28:
  v298 = v328;
  for ( i = v327; v298 != i; ++i )
  {
    v27 = *(_QWORD *)(*(_QWORD *)*i + 24LL);
    v28 = *(unsigned __int8 *)(v27 + 8);
    if ( (unsigned __int8)v28 > 0xFu || (v7 = 35454, !_bittest64(&v7, v28)) )
    {
LABEL_30:
      v7 = (unsigned int)(v28 - 13);
      if ( (unsigned int)v7 > 1 && (_DWORD)v28 != 16 || !(unsigned __int8)sub_16435F0(v27, 0) )
      {
        v29 = -1;
        goto LABEL_34;
      }
      LODWORD(v28) = *(unsigned __int8 *)(v27 + 8);
    }
    v182 = 1;
    while ( 2 )
    {
      switch ( (char)v28 )
      {
        case 0:
        case 8:
        case 10:
        case 12:
          v28 = *(_QWORD *)(v27 + 32);
          v27 = *(_QWORD *)(v27 + 24);
          v182 *= v28;
          LODWORD(v28) = *(unsigned __int8 *)(v27 + 8);
          continue;
        case 1:
          v199 = 16;
          break;
        case 2:
          v199 = 32;
          break;
        case 3:
        case 9:
          v199 = 64;
          break;
        case 4:
          v199 = 80;
          break;
        case 5:
        case 6:
          v199 = 128;
          break;
        case 7:
          v199 = 8 * (unsigned int)sub_15A9520(v289, 0);
          break;
        case 11:
          v199 = *(_DWORD *)(v27 + 8) >> 8;
          break;
        case 13:
          v199 = 8LL * *(_QWORD *)sub_15A9930(v289, v27);
          break;
        case 14:
          v202 = 1;
          v309 = *(_QWORD *)(v27 + 32);
          v203 = *(_QWORD *)(v27 + 24);
          v204 = (unsigned int)sub_15A9FE0(v289, v203);
          while ( 2 )
          {
            switch ( *(_BYTE *)(v203 + 8) )
            {
              case 0:
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v237 = *(_QWORD *)(v203 + 32);
                v203 = *(_QWORD *)(v203 + 24);
                v202 *= v237;
                continue;
              case 1:
                v234 = 16;
                break;
              case 2:
                v234 = 32;
                break;
              case 3:
              case 9:
                v234 = 64;
                break;
              case 4:
                v234 = 80;
                break;
              case 5:
              case 6:
                v234 = 128;
                break;
              case 7:
                v234 = 8 * (unsigned int)sub_15A9520(v289, 0);
                break;
              case 0xB:
                v234 = *(_DWORD *)(v203 + 8) >> 8;
                break;
              case 0xD:
                v234 = 8LL * *(_QWORD *)sub_15A9930(v289, v203);
                break;
              case 0xE:
                v297 = *(_QWORD *)(v203 + 32);
                v282 = *(_QWORD *)(v203 + 24);
                v288 = (unsigned int)sub_15A9FE0(v289, v282);
                v234 = 8 * v297 * v288 * ((v288 + ((unsigned __int64)(sub_127FA20(v289, v282) + 7) >> 3) - 1) / v288);
                break;
              case 0xF:
                v234 = 8 * (unsigned int)sub_15A9520(v289, *(_DWORD *)(v203 + 8) >> 8);
                break;
            }
            break;
          }
          v7 = v204 * v309;
          v199 = 8 * v204 * v309 * ((v204 + ((unsigned __int64)(v202 * v234 + 7) >> 3) - 1) / v204);
          break;
        case 15:
          v199 = 8 * (unsigned int)sub_15A9520(v289, *(_DWORD *)(v27 + 8) >> 8);
          break;
        default:
          goto LABEL_30;
      }
      break;
    }
    v29 = (unsigned __int64)(v199 * v182 + 7) >> 3;
LABEL_34:
    v30 = v327;
    v31 = (char *)v29;
    if ( v327 != i )
    {
      while ( 1 )
      {
        v34 = (char *)*v30;
        v33 = *(_QWORD *)(*(_QWORD *)*v30 + 24LL);
        v38 = *(unsigned __int8 *)(v33 + 8);
        if ( (unsigned __int8)v38 <= 0xFu )
        {
          v39 = 35454;
          if ( _bittest64(&v39, v38) )
            goto LABEL_50;
        }
        if ( (unsigned int)(v38 - 13) <= 1 || (_DWORD)v38 == 16 )
        {
          v302 = *(_QWORD *)(*(_QWORD *)*v30 + 24LL);
          v32 = sub_16435F0(v302, 0);
          v33 = v302;
          if ( v32 )
          {
            LOBYTE(v38) = *(_BYTE *)(v302 + 8);
LABEL_50:
            v40 = 1;
            while ( 2 )
            {
              switch ( (char)v38 )
              {
                case 1:
                  v41 = 16;
                  goto LABEL_53;
                case 2:
                  v41 = 32;
                  goto LABEL_53;
                case 3:
                case 9:
                  v41 = 64;
                  goto LABEL_53;
                case 4:
                  v41 = 80;
                  goto LABEL_53;
                case 5:
                case 6:
                  v41 = 128;
                  goto LABEL_53;
                case 7:
                  v303 = v40;
                  v42 = sub_15A9520(v289, 0);
                  v40 = v303;
                  v41 = (unsigned int)(8 * v42);
                  goto LABEL_53;
                case 11:
                  v41 = *(_DWORD *)(v33 + 8) >> 8;
                  goto LABEL_53;
                case 13:
                  v306 = v40;
                  v48 = (_QWORD *)sub_15A9930(v289, v33);
                  v40 = v306;
                  v41 = 8LL * *v48;
                  goto LABEL_53;
                case 14:
                  v283 = v40;
                  v290 = *(_QWORD *)(v33 + 24);
                  v305 = *(_QWORD *)(v33 + 32);
                  v44 = sub_15A9FE0(v289, v290);
                  v45 = v290;
                  v40 = v283;
                  v46 = 1;
                  v47 = v44;
                  while ( 2 )
                  {
                    switch ( *(_BYTE *)(v45 + 8) )
                    {
                      case 1:
                        v201 = 16;
                        goto LABEL_345;
                      case 2:
                        v201 = 32;
                        goto LABEL_345;
                      case 3:
                      case 9:
                        v201 = 64;
                        goto LABEL_345;
                      case 4:
                        v201 = 80;
                        goto LABEL_345;
                      case 5:
                      case 6:
                        v201 = 128;
                        goto LABEL_345;
                      case 7:
                        v279 = v283;
                        v212 = 0;
                        v285 = v47;
                        v294 = v46;
                        goto LABEL_369;
                      case 0xB:
                        v201 = *(_DWORD *)(v45 + 8) >> 8;
                        goto LABEL_345;
                      case 0xD:
                        v281 = v283;
                        v287 = v47;
                        v296 = v46;
                        v216 = (_QWORD *)sub_15A9930(v289, v45);
                        v46 = v296;
                        v47 = v287;
                        v40 = v281;
                        v201 = 8LL * *v216;
                        goto LABEL_345;
                      case 0xE:
                        v255 = v283;
                        v260 = v47;
                        v266 = v46;
                        v272 = *(_QWORD *)(v45 + 24);
                        v286 = *(_QWORD *)(v45 + 32);
                        v214 = sub_15A9FE0(v289, v272);
                        v40 = v255;
                        v295 = 1;
                        v47 = v260;
                        v215 = v272;
                        v280 = v214;
                        v46 = v266;
                        while ( 2 )
                        {
                          switch ( *(_BYTE *)(v215 + 8) )
                          {
                            case 1:
                              v236 = 16;
                              goto LABEL_464;
                            case 2:
                              v236 = 32;
                              goto LABEL_464;
                            case 3:
                            case 9:
                              v236 = 64;
                              goto LABEL_464;
                            case 4:
                              v236 = 80;
                              goto LABEL_464;
                            case 5:
                            case 6:
                              v236 = 128;
                              goto LABEL_464;
                            case 7:
                              v261 = v255;
                              v239 = 0;
                              v267 = v47;
                              v273 = v46;
                              goto LABEL_478;
                            case 0xB:
                              v236 = *(_DWORD *)(v215 + 8) >> 8;
                              goto LABEL_464;
                            case 0xD:
                              v241 = (_QWORD *)sub_15A9930(v289, v215);
                              v46 = v266;
                              v47 = v260;
                              v40 = v255;
                              v236 = 8LL * *v241;
                              goto LABEL_464;
                            case 0xE:
                              v249 = v255;
                              v252 = v260;
                              v256 = v266;
                              v262 = *(_QWORD *)(v215 + 24);
                              v274 = *(_QWORD *)(v215 + 32);
                              v268 = (unsigned int)sub_15A9FE0(v289, v262);
                              v242 = sub_127FA20(v289, v262);
                              v46 = v256;
                              v47 = v252;
                              v40 = v249;
                              v236 = 8 * v274 * v268 * ((v268 + ((unsigned __int64)(v242 + 7) >> 3) - 1) / v268);
                              goto LABEL_464;
                            case 0xF:
                              v261 = v255;
                              v267 = v47;
                              v273 = v46;
                              v239 = *(_DWORD *)(v215 + 8) >> 8;
LABEL_478:
                              v240 = sub_15A9520(v289, v239);
                              v46 = v273;
                              v47 = v267;
                              v40 = v261;
                              v236 = (unsigned int)(8 * v240);
LABEL_464:
                              v201 = 8 * v280 * v286 * ((v280 + ((unsigned __int64)(v295 * v236 + 7) >> 3) - 1) / v280);
                              goto LABEL_345;
                            case 0x10:
                              v238 = v295 * *(_QWORD *)(v215 + 32);
                              v215 = *(_QWORD *)(v215 + 24);
                              v295 = v238;
                              continue;
                            default:
                              goto LABEL_499;
                          }
                        }
                      case 0xF:
                        v279 = v283;
                        v285 = v47;
                        v294 = v46;
                        v212 = *(_DWORD *)(v45 + 8) >> 8;
LABEL_369:
                        v213 = sub_15A9520(v289, v212);
                        v46 = v294;
                        v47 = v285;
                        v40 = v279;
                        v201 = (unsigned int)(8 * v213);
LABEL_345:
                        v41 = 8 * v305 * v47 * ((v47 + ((unsigned __int64)(v201 * v46 + 7) >> 3) - 1) / v47);
                        goto LABEL_53;
                      case 0x10:
                        v217 = *(_QWORD *)(v45 + 32);
                        v45 = *(_QWORD *)(v45 + 24);
                        v46 *= v217;
                        continue;
                      default:
                        goto LABEL_499;
                    }
                  }
                case 15:
                  v304 = v40;
                  v43 = sub_15A9520(v289, *(_DWORD *)(v33 + 8) >> 8);
                  v40 = v304;
                  v41 = (unsigned int)(8 * v43);
LABEL_53:
                  v34 = (char *)*v30;
                  v35 = (unsigned __int64)(v41 * v40 + 7) >> 3;
                  goto LABEL_40;
                case 16:
                  v38 = *(_QWORD *)(v33 + 32);
                  v33 = *(_QWORD *)(v33 + 24);
                  v40 *= v38;
                  LOBYTE(v38) = *(_BYTE *)(v33 + 8);
                  continue;
                default:
                  goto LABEL_499;
              }
            }
          }
          v34 = (char *)*v30;
          v35 = -1;
        }
        else
        {
          v35 = -1;
        }
LABEL_40:
        v36 = (char *)*i;
        v318 = v34;
        v319 = v35;
        v320 = 0;
        v321 = 0;
        v322 = 0;
        v317[0] = v36;
        v317[1] = v31;
        memset(&v317[2], 0, 24);
        v37 = sub_134CB50((__int64)a3, (__int64)v317, (__int64)&v318);
        if ( v37 == 2 )
        {
          if ( byte_4F97800 || byte_4F97AA0 )
            sub_1352080(2u, *i, *v30);
          ++a1[3];
          goto LABEL_47;
        }
        if ( v37 <= 2u )
          break;
        if ( v37 == 3 )
        {
          if ( byte_4F97720 || byte_4F97AA0 )
            sub_1352080(3u, *i, *v30);
          ++v30;
          ++a1[4];
          if ( i == v30 )
            goto LABEL_70;
        }
        else
        {
LABEL_47:
          if ( i == ++v30 )
            goto LABEL_70;
        }
      }
      if ( v37 )
      {
        if ( byte_4F978E0 || byte_4F97AA0 )
          sub_1352080(1u, *i, *v30);
        ++a1[2];
      }
      else
      {
        if ( byte_4F979C0 || byte_4F97AA0 )
          sub_1352080(0, *i, *v30);
        ++a1[1];
      }
      goto LABEL_47;
    }
LABEL_70:
    ;
  }
  if ( !byte_4F96F40 )
    goto LABEL_107;
  v49 = v341;
  v291 = v335;
  v307 = v342;
  if ( v335 == v334 )
    goto LABEL_90;
  v299 = v334;
  do
  {
    v50 = *v299;
    if ( v307 == v49 )
      goto LABEL_89;
    v51 = v49;
    do
    {
      while ( 1 )
      {
        v52 = *v51;
        sub_141EDF0(&v318, *v51, v7, v49, v6);
        sub_141EB40(v317, v50, v53, v54, v55);
        v56 = sub_134CB50((__int64)a3, (__int64)v317, (__int64)&v318);
        if ( v56 == 2 )
        {
          if ( byte_4F97AA0 || byte_4F97800 )
            sub_1351E00(2, v50, v52);
          ++a1[3];
          goto LABEL_80;
        }
        if ( v56 > 2u )
        {
          if ( v56 == 3 )
          {
            if ( byte_4F97AA0 || byte_4F97720 )
              sub_1351E00(3, v50, v52);
            ++a1[4];
          }
          goto LABEL_80;
        }
        if ( !v56 )
          break;
        if ( byte_4F97AA0 || byte_4F978E0 )
          sub_1351E00(1, v50, v52);
        ++a1[2];
LABEL_80:
        if ( v307 == ++v51 )
          goto LABEL_88;
      }
      if ( byte_4F97AA0 || byte_4F979C0 )
        sub_1351E00(0, v50, v52);
      ++v51;
      ++a1[1];
    }
    while ( v307 != v51 );
LABEL_88:
    v49 = v341;
    v307 = v342;
LABEL_89:
    ++v299;
  }
  while ( v291 != v299 );
LABEL_90:
  if ( v307 != v49 )
  {
    v57 = v49 + 1;
    if ( v49 + 1 != v307 )
    {
      while ( 1 )
      {
        v58 = v341;
        if ( v341 != v57 )
          break;
LABEL_106:
        if ( ++v57 == v307 )
          goto LABEL_107;
      }
      v59 = v341;
      while ( 2 )
      {
        while ( 1 )
        {
          sub_141EDF0(&v318, *v59, v7, v58, v6);
          sub_141EDF0(v317, *v57, v60, v61, v62);
          v63 = sub_134CB50((__int64)a3, (__int64)v317, (__int64)&v318);
          if ( v63 == 2 )
            break;
          if ( v63 > 2u )
          {
            if ( v63 == 3 )
            {
              if ( byte_4F97AA0 || byte_4F97720 )
                sub_1351E00(3, *v57, *v59);
              ++a1[4];
            }
            goto LABEL_98;
          }
          if ( v63 )
          {
            if ( byte_4F97AA0 || byte_4F978E0 )
              sub_1351E00(1, *v57, *v59);
            ++a1[2];
            goto LABEL_98;
          }
          if ( byte_4F97AA0 || byte_4F979C0 )
            sub_1351E00(0, *v57, *v59);
          ++v59;
          ++a1[1];
          if ( v57 == v59 )
            goto LABEL_106;
        }
        if ( byte_4F97AA0 || byte_4F97800 )
          sub_1351E00(2, *v57, *v59);
        ++a1[3];
LABEL_98:
        if ( v57 == ++v59 )
          goto LABEL_106;
        continue;
      }
    }
  }
LABEL_107:
  v284 = &v347[8 * (unsigned int)v348];
  if ( v284 != (_QWORD *)v347 )
  {
    v300 = v347;
    do
    {
      v64 = (char **)v327;
      v65 = *v300 & 0xFFFFFFFFFFFFFFF8LL;
      v308 = (char **)v328;
      if ( v328 != v327 )
      {
        while ( 1 )
        {
          v66 = *v64;
          v67 = *(_QWORD *)(*(_QWORD *)*v64 + 24LL);
          v68 = *(unsigned __int8 *)(v67 + 8);
          if ( (unsigned __int8)v68 <= 0xFu )
          {
            v175 = 35454;
            if ( _bittest64(&v175, v68) )
              goto LABEL_258;
          }
          if ( (unsigned int)(v68 - 13) <= 1 || (_DWORD)v68 == 16 )
          {
            v292 = *(_QWORD *)(*(_QWORD *)*v64 + 24LL);
            v69 = sub_16435F0(v292, 0);
            v67 = v292;
            if ( v69 )
            {
              LOBYTE(v68) = *(_BYTE *)(v292 + 8);
LABEL_258:
              v176 = 1;
              while ( 2 )
              {
                switch ( (char)v68 )
                {
                  case 1:
                    v177 = 16;
                    goto LABEL_261;
                  case 2:
                    v177 = 32;
                    goto LABEL_261;
                  case 3:
                  case 9:
                    v177 = 64;
                    goto LABEL_261;
                  case 4:
                    v177 = 80;
                    goto LABEL_261;
                  case 5:
                  case 6:
                    v177 = 128;
                    goto LABEL_261;
                  case 7:
                    v177 = 8 * (unsigned int)sub_15A9520(v289, 0);
                    goto LABEL_261;
                  case 11:
                    v177 = *(_DWORD *)(v67 + 8) >> 8;
                    goto LABEL_261;
                  case 13:
                    v177 = 8LL * *(_QWORD *)sub_15A9930(v289, v67);
                    goto LABEL_261;
                  case 14:
                    v275 = *(_QWORD *)(v67 + 24);
                    v293 = *(_QWORD *)(v67 + 32);
                    v178 = sub_15A9FE0(v289, v275);
                    v179 = v275;
                    v180 = 1;
                    v181 = v178;
                    while ( 2 )
                    {
                      switch ( *(_BYTE *)(v179 + 8) )
                      {
                        case 1:
                          v200 = 16;
                          goto LABEL_343;
                        case 2:
                          v200 = 32;
                          goto LABEL_343;
                        case 3:
                        case 9:
                          v200 = 64;
                          goto LABEL_343;
                        case 4:
                          v200 = 80;
                          goto LABEL_343;
                        case 5:
                        case 6:
                          v200 = 128;
                          goto LABEL_343;
                        case 7:
                          v269 = v181;
                          v206 = 0;
                          v276 = v180;
                          goto LABEL_360;
                        case 0xB:
                          v200 = *(_DWORD *)(v179 + 8) >> 8;
                          goto LABEL_343;
                        case 0xD:
                          v271 = v181;
                          v278 = v180;
                          v211 = (_QWORD *)sub_15A9930(v289, v179);
                          v180 = v278;
                          v181 = v271;
                          v200 = 8LL * *v211;
                          goto LABEL_343;
                        case 0xE:
                          v251 = v181;
                          v254 = v180;
                          v270 = *(_QWORD *)(v179 + 32);
                          v259 = *(_QWORD *)(v179 + 24);
                          v209 = sub_15A9FE0(v289, v259);
                          v181 = v251;
                          v277 = 1;
                          v210 = v259;
                          v180 = v254;
                          v265 = v209;
                          while ( 2 )
                          {
                            switch ( *(_BYTE *)(v210 + 8) )
                            {
                              case 1:
                                v235 = 16;
                                goto LABEL_462;
                              case 2:
                                v235 = 32;
                                goto LABEL_462;
                              case 3:
                              case 9:
                                v235 = 64;
                                goto LABEL_462;
                              case 4:
                                v235 = 80;
                                goto LABEL_462;
                              case 5:
                              case 6:
                                v235 = 128;
                                goto LABEL_462;
                              case 7:
                                v258 = v251;
                                v244 = 0;
                                v264 = v180;
                                goto LABEL_484;
                              case 0xB:
                                v235 = *(_DWORD *)(v210 + 8) >> 8;
                                goto LABEL_462;
                              case 0xD:
                                v247 = (_QWORD *)sub_15A9930(v289, v210);
                                v180 = v254;
                                v181 = v251;
                                v235 = 8LL * *v247;
                                goto LABEL_462;
                              case 0xE:
                                v248 = v251;
                                v250 = v254;
                                v263 = *(_QWORD *)(v210 + 32);
                                v253 = *(_QWORD *)(v210 + 24);
                                v257 = (unsigned int)sub_15A9FE0(v289, v253);
                                v243 = sub_127FA20(v289, v253);
                                v180 = v250;
                                v181 = v248;
                                v235 = 8 * v263 * v257 * ((v257 + ((unsigned __int64)(v243 + 7) >> 3) - 1) / v257);
                                goto LABEL_462;
                              case 0xF:
                                v258 = v251;
                                v264 = v180;
                                v244 = *(_DWORD *)(v210 + 8) >> 8;
LABEL_484:
                                v245 = sub_15A9520(v289, v244);
                                v180 = v264;
                                v181 = v258;
                                v235 = (unsigned int)(8 * v245);
LABEL_462:
                                v200 = 8
                                     * v265
                                     * v270
                                     * ((v265 + ((unsigned __int64)(v277 * v235 + 7) >> 3) - 1)
                                      / v265);
                                goto LABEL_343;
                              case 0x10:
                                v246 = v277 * *(_QWORD *)(v210 + 32);
                                v210 = *(_QWORD *)(v210 + 24);
                                v277 = v246;
                                continue;
                              default:
                                goto LABEL_499;
                            }
                          }
                        case 0xF:
                          v269 = v181;
                          v276 = v180;
                          v206 = *(_DWORD *)(v179 + 8) >> 8;
LABEL_360:
                          v207 = sub_15A9520(v289, v206);
                          v180 = v276;
                          v181 = v269;
                          v200 = (unsigned int)(8 * v207);
LABEL_343:
                          v177 = 8 * v181 * v293 * ((v181 + ((unsigned __int64)(v200 * v180 + 7) >> 3) - 1) / v181);
                          goto LABEL_261;
                        case 0x10:
                          v208 = *(_QWORD *)(v179 + 32);
                          v179 = *(_QWORD *)(v179 + 24);
                          v180 *= v208;
                          continue;
                        default:
                          goto LABEL_499;
                      }
                    }
                  case 15:
                    v177 = 8 * (unsigned int)sub_15A9520(v289, *(_DWORD *)(v67 + 8) >> 8);
LABEL_261:
                    v70 = (unsigned __int64)(v177 * v176 + 7) >> 3;
                    goto LABEL_115;
                  case 16:
                    v68 = *(_QWORD *)(v67 + 32);
                    v67 = *(_QWORD *)(v67 + 24);
                    v176 *= v68;
                    LOBYTE(v68) = *(_BYTE *)(v67 + 8);
                    continue;
                  default:
                    goto LABEL_499;
                }
              }
            }
          }
          v70 = -1;
LABEL_115:
          v71 = *(_BYTE *)(v65 + 16);
          v72 = 0;
          if ( v71 > 0x17u )
          {
            if ( v71 == 78 )
            {
              v72 = (char *)(v65 | 4);
            }
            else if ( v71 == 29 )
            {
              v72 = (char *)v65;
            }
          }
          v318 = v66;
          v319 = v70;
          v320 = 0;
          v321 = 0;
          v322 = 0;
          switch ( (unsigned __int8)sub_134F0E0(a3, (__int64)v72, (__int64)&v318) )
          {
            case 0u:
              if ( byte_4F972C0 || byte_4F97AA0 )
              {
                v142 = sub_16E8CB0(a3, v72, v73);
                v143 = *(_WORD **)(v142 + 24);
                v144 = v142;
                if ( *(_QWORD *)(v142 + 16) - (_QWORD)v143 <= 1u )
                {
                  v72 = "  ";
                  v191 = sub_16E7EE0(v142, "  ", 2);
                  v145 = *(_DWORD **)(v191 + 24);
                  v144 = v191;
                }
                else
                {
                  *v143 = 8224;
                  v145 = (_DWORD *)(*(_QWORD *)(v142 + 24) + 2LL);
                  *(_QWORD *)(v142 + 24) = v145;
                }
                if ( *(_QWORD *)(v144 + 16) - (_QWORD)v145 <= 3u )
                {
                  v72 = "Must";
                  v185 = sub_16E7EE0(v144, "Must", 4);
                  v146 = *(_QWORD **)(v185 + 24);
                  v144 = v185;
                }
                else
                {
                  *v145 = 1953723725;
                  v146 = (_QWORD *)(*(_QWORD *)(v144 + 24) + 4LL);
                  *(_QWORD *)(v144 + 24) = v146;
                }
                if ( *(_QWORD *)(v144 + 16) - (_QWORD)v146 <= 7u )
                {
                  v72 = ":  Ptr: ";
                  sub_16E7EE0(v144, ":  Ptr: ", 8);
                }
                else
                {
                  *v146 = 0x203A72745020203ALL;
                  *(_QWORD *)(v144 + 24) += 8LL;
                }
                v147 = sub_16E8CB0(v144, v72, v146);
                sub_15537D0(v66, v147, 1);
                v149 = sub_16E8CB0(v66, v147, v148);
                v150 = *(_DWORD **)(v149 + 24);
                v151 = v149;
                if ( *(_QWORD *)(v149 + 16) - (_QWORD)v150 <= 3u )
                {
                  v151 = sub_16E7EE0(v149, "\t<->", 4);
                }
                else
                {
                  *v150 = 1043151881;
                  *(_QWORD *)(v149 + 24) += 4LL;
                }
                sub_155C2B0(v65, v151, 0);
                v152 = *(_BYTE **)(v151 + 24);
                if ( (unsigned __int64)v152 >= *(_QWORD *)(v151 + 16) )
                {
                  sub_16E7DE0(v151, 10);
                }
                else
                {
                  *(_QWORD *)(v151 + 24) = v152 + 1;
                  *v152 = 10;
                }
              }
              ++a1[9];
              goto LABEL_140;
            case 1u:
              if ( byte_4F971E0 || byte_4F97AA0 )
              {
                v117 = sub_16E8CB0(a3, v72, v73);
                v118 = *(_WORD **)(v117 + 24);
                v119 = v117;
                if ( *(_QWORD *)(v117 + 16) - (_QWORD)v118 <= 1u )
                {
                  v120 = "  ";
                  v195 = sub_16E7EE0(v117, "  ", 2);
                  v121 = *(_QWORD *)(v195 + 24);
                  v119 = v195;
                }
                else
                {
                  v120 = (char *)8224;
                  *v118 = 8224;
                  v121 = *(_QWORD *)(v117 + 24) + 2LL;
                  *(_QWORD *)(v117 + 24) = v121;
                }
                if ( (unsigned __int64)(*(_QWORD *)(v119 + 16) - v121) <= 0x13 )
                {
                  v120 = "Just Ref (MustAlias)";
                  v194 = sub_16E7EE0(v119, "Just Ref (MustAlias)", 20);
                  v123 = *(_QWORD **)(v194 + 24);
                  v119 = v194;
                }
                else
                {
                  si128 = _mm_load_si128((const __m128i *)&xmmword_42888B0);
                  *(_DWORD *)(v121 + 16) = 695427433;
                  *(__m128i *)v121 = si128;
                  v123 = (_QWORD *)(*(_QWORD *)(v119 + 24) + 20LL);
                  *(_QWORD *)(v119 + 24) = v123;
                }
                if ( *(_QWORD *)(v119 + 16) - (_QWORD)v123 <= 7u )
                {
                  v120 = ":  Ptr: ";
                  sub_16E7EE0(v119, ":  Ptr: ", 8);
                }
                else
                {
                  *v123 = 0x203A72745020203ALL;
                  *(_QWORD *)(v119 + 24) += 8LL;
                }
                v124 = sub_16E8CB0(v119, v120, v123);
                sub_15537D0(v66, v124, 1);
                v126 = sub_16E8CB0(v66, v124, v125);
                v127 = *(_DWORD **)(v126 + 24);
                v128 = v126;
                if ( *(_QWORD *)(v126 + 16) - (_QWORD)v127 <= 3u )
                {
                  v128 = sub_16E7EE0(v126, "\t<->", 4);
                }
                else
                {
                  *v127 = 1043151881;
                  *(_QWORD *)(v126 + 24) += 4LL;
                }
                sub_155C2B0(v65, v128, 0);
                v129 = *(_BYTE **)(v128 + 24);
                if ( (unsigned __int64)v129 >= *(_QWORD *)(v128 + 16) )
                {
                  sub_16E7DE0(v128, 10);
                }
                else
                {
                  *(_QWORD *)(v128 + 24) = v129 + 1;
                  *v129 = 10;
                }
              }
              ++a1[10];
              goto LABEL_140;
            case 2u:
              if ( byte_4F97100 || byte_4F97AA0 )
              {
                v130 = sub_16E8CB0(a3, v72, v73);
                v131 = *(_WORD **)(v130 + 24);
                v132 = v130;
                if ( *(_QWORD *)(v130 + 16) - (_QWORD)v131 <= 1u )
                {
                  v72 = "  ";
                  v193 = sub_16E7EE0(v130, "  ", 2);
                  v133 = *(_QWORD *)(v193 + 24);
                  v132 = v193;
                }
                else
                {
                  *v131 = 8224;
                  v133 = *(_QWORD *)(v130 + 24) + 2LL;
                  *(_QWORD *)(v130 + 24) = v133;
                }
                if ( (unsigned __int64)(*(_QWORD *)(v132 + 16) - v133) <= 0x13 )
                {
                  v72 = "Just Mod (MustAlias)";
                  v192 = sub_16E7EE0(v132, "Just Mod (MustAlias)", 20);
                  v135 = *(_QWORD **)(v192 + 24);
                  v132 = v192;
                }
                else
                {
                  v134 = _mm_load_si128((const __m128i *)&xmmword_42888A0);
                  *(_DWORD *)(v133 + 16) = 695427433;
                  *(__m128i *)v133 = v134;
                  v135 = (_QWORD *)(*(_QWORD *)(v132 + 24) + 20LL);
                  *(_QWORD *)(v132 + 24) = v135;
                }
                if ( *(_QWORD *)(v132 + 16) - (_QWORD)v135 <= 7u )
                {
                  v72 = ":  Ptr: ";
                  sub_16E7EE0(v132, ":  Ptr: ", 8);
                }
                else
                {
                  *v135 = 0x203A72745020203ALL;
                  *(_QWORD *)(v132 + 24) += 8LL;
                }
                v136 = sub_16E8CB0(v132, v72, v135);
                sub_15537D0(v66, v136, 1);
                v138 = sub_16E8CB0(v66, v136, v137);
                v139 = *(_DWORD **)(v138 + 24);
                v140 = v138;
                if ( *(_QWORD *)(v138 + 16) - (_QWORD)v139 <= 3u )
                {
                  v140 = sub_16E7EE0(v138, "\t<->", 4);
                }
                else
                {
                  *v139 = 1043151881;
                  *(_QWORD *)(v138 + 24) += 4LL;
                }
                sub_155C2B0(v65, v140, 0);
                v141 = *(_BYTE **)(v140 + 24);
                if ( (unsigned __int64)v141 >= *(_QWORD *)(v140 + 16) )
                {
                  sub_16E7DE0(v140, 10);
                }
                else
                {
                  *(_QWORD *)(v140 + 24) = v141 + 1;
                  *v141 = 10;
                }
              }
              ++a1[11];
              goto LABEL_140;
            case 3u:
              if ( byte_4F97020 || byte_4F97AA0 )
              {
                v105 = sub_16E8CB0(a3, v72, v73);
                v106 = *(_WORD **)(v105 + 24);
                v107 = v105;
                if ( *(_QWORD *)(v105 + 16) - (_QWORD)v106 <= 1u )
                {
                  v72 = "  ";
                  v197 = sub_16E7EE0(v105, "  ", 2);
                  v108 = *(_QWORD *)(v197 + 24);
                  v107 = v197;
                }
                else
                {
                  *v106 = 8224;
                  v108 = *(_QWORD *)(v105 + 24) + 2LL;
                  *(_QWORD *)(v105 + 24) = v108;
                }
                if ( (unsigned __int64)(*(_QWORD *)(v107 + 16) - v108) <= 0x16 )
                {
                  v72 = "Both ModRef (MustAlias)";
                  v196 = sub_16E7EE0(v107, "Both ModRef (MustAlias)", 23);
                  v110 = *(_QWORD **)(v196 + 24);
                  v107 = v196;
                }
                else
                {
                  v109 = _mm_load_si128((const __m128i *)&xmmword_42888C0);
                  *(_DWORD *)(v108 + 16) = 1768702324;
                  *(_WORD *)(v108 + 20) = 29537;
                  *(_BYTE *)(v108 + 22) = 41;
                  *(__m128i *)v108 = v109;
                  v110 = (_QWORD *)(*(_QWORD *)(v107 + 24) + 23LL);
                  *(_QWORD *)(v107 + 24) = v110;
                }
                if ( *(_QWORD *)(v107 + 16) - (_QWORD)v110 <= 7u )
                {
                  v72 = ":  Ptr: ";
                  sub_16E7EE0(v107, ":  Ptr: ", 8);
                }
                else
                {
                  *v110 = 0x203A72745020203ALL;
                  *(_QWORD *)(v107 + 24) += 8LL;
                }
                v111 = sub_16E8CB0(v107, v72, v110);
                sub_15537D0(v66, v111, 1);
                v113 = sub_16E8CB0(v66, v111, v112);
                v114 = *(_DWORD **)(v113 + 24);
                v115 = v113;
                if ( *(_QWORD *)(v113 + 16) - (_QWORD)v114 <= 3u )
                {
                  v115 = sub_16E7EE0(v113, "\t<->", 4);
                }
                else
                {
                  *v114 = 1043151881;
                  *(_QWORD *)(v113 + 24) += 4LL;
                }
                sub_155C2B0(v65, v115, 0);
                v116 = *(_BYTE **)(v115 + 24);
                if ( (unsigned __int64)v116 >= *(_QWORD *)(v115 + 16) )
                {
                  sub_16E7DE0(v115, 10);
                }
                else
                {
                  *(_QWORD *)(v115 + 24) = v116 + 1;
                  *v116 = 10;
                }
              }
              ++a1[12];
              goto LABEL_140;
            case 4u:
              if ( byte_4F97640 || byte_4F97AA0 )
              {
                v164 = sub_16E8CB0(a3, v72, v73);
                v165 = *(_WORD **)(v164 + 24);
                v166 = v164;
                if ( *(_QWORD *)(v164 + 16) - (_QWORD)v165 <= 1u )
                {
                  v72 = "  ";
                  v187 = sub_16E7EE0(v164, "  ", 2);
                  v167 = *(_QWORD **)(v187 + 24);
                  v166 = v187;
                }
                else
                {
                  *v165 = 8224;
                  v167 = (_QWORD *)(*(_QWORD *)(v164 + 24) + 2LL);
                  *(_QWORD *)(v164 + 24) = v167;
                }
                if ( *(_QWORD *)(v166 + 16) - (_QWORD)v167 <= 7u )
                {
                  v72 = "NoModRef";
                  v186 = sub_16E7EE0(v166, "NoModRef", 8);
                  v168 = *(_QWORD **)(v186 + 24);
                  v166 = v186;
                }
                else
                {
                  *v167 = 0x666552646F4D6F4ELL;
                  v168 = (_QWORD *)(*(_QWORD *)(v166 + 24) + 8LL);
                  *(_QWORD *)(v166 + 24) = v168;
                }
                if ( *(_QWORD *)(v166 + 16) - (_QWORD)v168 <= 7u )
                {
                  v72 = ":  Ptr: ";
                  sub_16E7EE0(v166, ":  Ptr: ", 8);
                }
                else
                {
                  *v168 = 0x203A72745020203ALL;
                  *(_QWORD *)(v166 + 24) += 8LL;
                }
                v169 = sub_16E8CB0(v166, v72, v168);
                sub_15537D0(v66, v169, 1);
                v171 = sub_16E8CB0(v66, v169, v170);
                v172 = *(_DWORD **)(v171 + 24);
                v173 = v171;
                if ( *(_QWORD *)(v171 + 16) - (_QWORD)v172 <= 3u )
                {
                  v173 = sub_16E7EE0(v171, "\t<->", 4);
                }
                else
                {
                  *v172 = 1043151881;
                  *(_QWORD *)(v171 + 24) += 4LL;
                }
                sub_155C2B0(v65, v173, 0);
                v174 = *(_BYTE **)(v173 + 24);
                if ( (unsigned __int64)v174 >= *(_QWORD *)(v173 + 16) )
                {
                  sub_16E7DE0(v173, 10);
                }
                else
                {
                  *(_QWORD *)(v173 + 24) = v174 + 1;
                  *v174 = 10;
                }
              }
              ++a1[5];
              goto LABEL_140;
            case 5u:
              if ( byte_4F97560 || byte_4F97AA0 )
              {
                v94 = sub_16E8CB0(a3, v72, v73);
                v95 = *(_WORD **)(v94 + 24);
                v96 = v94;
                if ( *(_QWORD *)(v94 + 16) - (_QWORD)v95 <= 1u )
                {
                  v72 = "  ";
                  v190 = sub_16E7EE0(v94, "  ", 2);
                  v97 = *(_QWORD **)(v190 + 24);
                  v96 = v190;
                }
                else
                {
                  *v95 = 8224;
                  v97 = (_QWORD *)(*(_QWORD *)(v94 + 24) + 2LL);
                  *(_QWORD *)(v94 + 24) = v97;
                }
                if ( *(_QWORD *)(v96 + 16) - (_QWORD)v97 <= 7u )
                {
                  v72 = "Just Ref";
                  v189 = sub_16E7EE0(v96, "Just Ref", 8);
                  v98 = *(_QWORD **)(v189 + 24);
                  v96 = v189;
                }
                else
                {
                  *v97 = 0x666552207473754ALL;
                  v98 = (_QWORD *)(*(_QWORD *)(v96 + 24) + 8LL);
                  *(_QWORD *)(v96 + 24) = v98;
                }
                if ( *(_QWORD *)(v96 + 16) - (_QWORD)v98 <= 7u )
                {
                  v72 = ":  Ptr: ";
                  sub_16E7EE0(v96, ":  Ptr: ", 8);
                }
                else
                {
                  *v98 = 0x203A72745020203ALL;
                  *(_QWORD *)(v96 + 24) += 8LL;
                }
                v99 = sub_16E8CB0(v96, v72, v98);
                sub_15537D0(v66, v99, 1);
                v101 = sub_16E8CB0(v66, v99, v100);
                v102 = *(_DWORD **)(v101 + 24);
                v103 = v101;
                if ( *(_QWORD *)(v101 + 16) - (_QWORD)v102 <= 3u )
                {
                  v103 = sub_16E7EE0(v101, "\t<->", 4);
                }
                else
                {
                  *v102 = 1043151881;
                  *(_QWORD *)(v101 + 24) += 4LL;
                }
                sub_155C2B0(v65, v103, 0);
                v104 = *(_BYTE **)(v103 + 24);
                if ( (unsigned __int64)v104 >= *(_QWORD *)(v103 + 16) )
                {
                  sub_16E7DE0(v103, 10);
                }
                else
                {
                  *(_QWORD *)(v103 + 24) = v104 + 1;
                  *v104 = 10;
                }
              }
              ++a1[7];
              goto LABEL_140;
            case 6u:
              if ( !byte_4F97480 && !byte_4F97AA0 )
                goto LABEL_139;
              v74 = sub_16E8CB0(a3, v72, v73);
              v75 = *(_WORD **)(v74 + 24);
              v76 = v74;
              if ( *(_QWORD *)(v74 + 16) - (_QWORD)v75 <= 1u )
              {
                v72 = "  ";
                v188 = sub_16E7EE0(v74, "  ", 2);
                v77 = *(_QWORD **)(v188 + 24);
                v76 = v188;
              }
              else
              {
                *v75 = 8224;
                v77 = (_QWORD *)(*(_QWORD *)(v74 + 24) + 2LL);
                *(_QWORD *)(v74 + 24) = v77;
              }
              if ( *(_QWORD *)(v76 + 16) - (_QWORD)v77 <= 7u )
              {
                v72 = "Just Mod";
                v76 = sub_16E7EE0(v76, "Just Mod", 8);
                v78 = *(_QWORD **)(v76 + 24);
                if ( *(_QWORD *)(v76 + 16) - (_QWORD)v78 <= 7u )
                {
LABEL_295:
                  v72 = ":  Ptr: ";
                  sub_16E7EE0(v76, ":  Ptr: ", 8);
                  goto LABEL_135;
                }
              }
              else
              {
                *v77 = 0x646F4D207473754ALL;
                v78 = (_QWORD *)(*(_QWORD *)(v76 + 24) + 8LL);
                v79 = *(_QWORD *)(v76 + 16);
                *(_QWORD *)(v76 + 24) = v78;
                if ( (unsigned __int64)(v79 - (_QWORD)v78) <= 7 )
                  goto LABEL_295;
              }
              *v78 = 0x203A72745020203ALL;
              *(_QWORD *)(v76 + 24) += 8LL;
LABEL_135:
              v80 = sub_16E8CB0(v76, v72, v78);
              sub_15537D0(v66, v80, 1);
              v82 = sub_16E8CB0(v66, v80, v81);
              v83 = *(_DWORD **)(v82 + 24);
              v84 = v82;
              if ( *(_QWORD *)(v82 + 16) - (_QWORD)v83 <= 3u )
              {
                v84 = sub_16E7EE0(v82, "\t<->", 4);
              }
              else
              {
                *v83 = 1043151881;
                *(_QWORD *)(v82 + 24) += 4LL;
              }
              sub_155C2B0(v65, v84, 0);
              v85 = *(_BYTE **)(v84 + 24);
              if ( (unsigned __int64)v85 >= *(_QWORD *)(v84 + 16) )
              {
                sub_16E7DE0(v84, 10);
              }
              else
              {
                *(_QWORD *)(v84 + 24) = v85 + 1;
                *v85 = 10;
              }
LABEL_139:
              ++a1[6];
LABEL_140:
              if ( v308 == ++v64 )
                goto LABEL_141;
              break;
            case 7u:
              if ( byte_4F973A0 || byte_4F97AA0 )
              {
                v153 = sub_16E8CB0(a3, v72, v73);
                v154 = *(_WORD **)(v153 + 24);
                v155 = v153;
                if ( *(_QWORD *)(v153 + 16) - (_QWORD)v154 <= 1u )
                {
                  v72 = "  ";
                  v184 = sub_16E7EE0(v153, "  ", 2);
                  v156 = *(void **)(v184 + 24);
                  v155 = v184;
                }
                else
                {
                  *v154 = 8224;
                  v156 = (void *)(*(_QWORD *)(v153 + 24) + 2LL);
                  *(_QWORD *)(v153 + 24) = v156;
                }
                if ( *(_QWORD *)(v155 + 16) - (_QWORD)v156 <= 0xAu )
                {
                  v72 = "Both ModRef";
                  v183 = sub_16E7EE0(v155, "Both ModRef", 11);
                  v157 = *(_QWORD **)(v183 + 24);
                  v155 = v183;
                }
                else
                {
                  qmemcpy(v156, "Both ModRef", 11);
                  v157 = (_QWORD *)(*(_QWORD *)(v155 + 24) + 11LL);
                  *(_QWORD *)(v155 + 24) = v157;
                }
                if ( *(_QWORD *)(v155 + 16) - (_QWORD)v157 <= 7u )
                {
                  v72 = ":  Ptr: ";
                  sub_16E7EE0(v155, ":  Ptr: ", 8);
                }
                else
                {
                  *v157 = 0x203A72745020203ALL;
                  *(_QWORD *)(v155 + 24) += 8LL;
                }
                v158 = sub_16E8CB0(v155, v72, v157);
                sub_15537D0(v66, v158, 1);
                v160 = sub_16E8CB0(v66, v158, v159);
                v161 = *(_DWORD **)(v160 + 24);
                v162 = v160;
                if ( *(_QWORD *)(v160 + 16) - (_QWORD)v161 <= 3u )
                {
                  v162 = sub_16E7EE0(v160, "\t<->", 4);
                }
                else
                {
                  *v161 = 1043151881;
                  *(_QWORD *)(v160 + 24) += 4LL;
                }
                sub_155C2B0(v65, v162, 0);
                v163 = *(_BYTE **)(v162 + 24);
                if ( (unsigned __int64)v163 >= *(_QWORD *)(v162 + 16) )
                {
                  sub_16E7DE0(v162, 10);
                }
                else
                {
                  *(_QWORD *)(v162 + 24) = v163 + 1;
                  *v163 = 10;
                }
              }
              ++a1[8];
              goto LABEL_140;
            default:
              goto LABEL_140;
          }
        }
      }
LABEL_141:
      ++v300;
    }
    while ( v284 != v300 );
    v86 = (__int64 *)v347;
    if ( &v347[8 * (unsigned int)v348] != v347 )
    {
      v314 = (__int64 *)&v347[8 * (unsigned int)v348];
      v87 = (__int64 *)v347;
      while ( 1 )
      {
        for ( ; v86 != v314; ++v86 )
        {
          if ( v87 != v86 )
          {
            v88 = 0;
            v89 = *v86 & 0xFFFFFFFFFFFFFFF8LL;
            v90 = *(_BYTE *)(v89 + 16);
            if ( v90 > 0x17u )
            {
              if ( v90 == 78 )
              {
                v88 = v89 | 4;
              }
              else if ( v90 == 29 )
              {
                v88 = *v86 & 0xFFFFFFFFFFFFFFF8LL;
              }
            }
            v91 = 0;
            v92 = *v87 & 0xFFFFFFFFFFFFFFF8LL;
            v93 = *(_BYTE *)(v92 + 16);
            if ( v93 > 0x17u )
            {
              if ( v93 == 78 )
              {
                v91 = v92 | 4;
              }
              else if ( v93 == 29 )
              {
                v91 = *v87 & 0xFFFFFFFFFFFFFFF8LL;
              }
            }
            switch ( (unsigned __int8)sub_134F530(a3, v91, v88) )
            {
              case 0u:
                sub_13523B0("Must", (unsigned __int8)byte_4F972C0, *v87, *v86);
                ++a1[9];
                break;
              case 1u:
                sub_13523B0("Just Ref (MustAlias)", (unsigned __int8)byte_4F971E0, *v87, *v86);
                ++a1[10];
                break;
              case 2u:
                sub_13523B0("Just Mod (MustAlias)", (unsigned __int8)byte_4F97100, *v87, *v86);
                ++a1[11];
                break;
              case 3u:
                sub_13523B0("Both ModRef (MustAlias)", (unsigned __int8)byte_4F97020, *v87, *v86);
                ++a1[12];
                break;
              case 4u:
                sub_13523B0("NoModRef", (unsigned __int8)byte_4F97640, *v87, *v86);
                ++a1[5];
                break;
              case 5u:
                sub_13523B0("Just Ref", (unsigned __int8)byte_4F97560, *v87, *v86);
                ++a1[7];
                break;
              case 6u:
                sub_13523B0("Just Mod", (unsigned __int8)byte_4F97480, *v87, *v86);
                ++a1[6];
                break;
              case 7u:
                sub_13523B0("Both ModRef", (unsigned __int8)byte_4F973A0, *v87, *v86);
                ++a1[8];
                break;
              default:
                continue;
            }
          }
        }
        if ( ++v87 == v314 )
          break;
        v86 = (__int64 *)v347;
      }
    }
  }
  if ( v341 )
    j_j___libc_free_0(v341, v343 - (_QWORD)v341);
  j___libc_free_0(v338);
  if ( v334 )
    j_j___libc_free_0(v334, v336 - (_QWORD)v334);
  j___libc_free_0(v331);
  if ( v347 != v349 )
    _libc_free((unsigned __int64)v347);
  if ( (v345 & 1) == 0 )
    j___libc_free_0(v346);
  if ( v327 )
    j_j___libc_free_0(v327, v329 - (_QWORD)v327);
  return j___libc_free_0(v324);
}
