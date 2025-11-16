// Function: sub_21174B0
// Address: 0x21174b0
//
__int64 __fastcall sub_21174B0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // rax
  __int64 v11; // rsi
  unsigned __int64 *v12; // rax
  __int64 v13; // rbx
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  double v18; // xmm4_8
  double v19; // xmm5_8
  __int64 v20; // r12
  int v21; // ecx
  _QWORD *v22; // r9
  __int64 *v23; // rdx
  __int64 v24; // r8
  __int64 v25; // rdi
  int v26; // r8d
  int v27; // r9d
  unsigned __int64 v28; // r12
  char v29; // al
  __int64 v30; // rax
  unsigned int v31; // r12d
  unsigned int v33; // ecx
  unsigned int v34; // esi
  unsigned int v35; // r8d
  __int64 v36; // r9
  unsigned int v37; // edi
  int v38; // r8d
  unsigned int v39; // ecx
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 i; // rbx
  __int64 v43; // r15
  __int64 *v44; // r13
  _QWORD *v45; // r12
  __int64 v46; // rdx
  unsigned __int64 v47; // rcx
  __int64 v48; // rdx
  __int64 v49; // rax
  _QWORD *v50; // r12
  __int64 **v51; // r12
  __int64 *v52; // rax
  __int64 v53; // r14
  __int64 v54; // r12
  __int64 v55; // rdx
  _QWORD *v56; // rax
  double v57; // xmm4_8
  double v58; // xmm5_8
  __int64 v59; // rbx
  __int64 v60; // rdx
  unsigned __int64 v61; // rsi
  __int64 v62; // rdx
  __int64 v63; // rdx
  unsigned __int64 v64; // rcx
  __int64 v65; // rdx
  unsigned __int64 v66; // r14
  __int64 v67; // rdx
  unsigned __int64 v68; // rcx
  __int64 v69; // rdx
  __int64 *v70; // r14
  __int64 v71; // rbx
  __int64 v72; // rdi
  __int64 v73; // r12
  _QWORD *v74; // rax
  __int64 v75; // r15
  __int64 v76; // r12
  int v77; // r8d
  int v78; // r9d
  __int64 *v79; // rax
  unsigned int v80; // eax
  __int64 v81; // rcx
  __int64 v82; // r15
  __int64 v83; // r13
  __int64 v84; // rbx
  __int64 v85; // rdx
  __int64 v86; // rax
  __int64 *v87; // rdi
  __int64 v88; // r12
  __int64 v89; // r15
  _QWORD *v90; // rbx
  __int64 v91; // r13
  __int64 *v92; // r14
  _QWORD *v93; // rax
  __int64 v94; // rbx
  __int64 v95; // rax
  double v96; // xmm4_8
  double v97; // xmm5_8
  _QWORD *v98; // r12
  __int64 v99; // r14
  unsigned __int64 v100; // rdi
  __int64 v101; // rsi
  __int64 *v102; // rax
  __int64 *v103; // rdi
  __int64 *v104; // rcx
  unsigned __int64 v105; // rbx
  __int64 v106; // r15
  __int64 v107; // rdi
  __int64 v108; // r12
  __int64 v109; // r13
  unsigned int v110; // eax
  unsigned int v111; // r14d
  __int64 v112; // r15
  _QWORD *v113; // r13
  __int64 v114; // r14
  __int64 v115; // rax
  __int64 v116; // r12
  __int64 v117; // rbx
  _QWORD *v118; // rax
  unsigned __int8 *v119; // rsi
  _BYTE *v120; // rdx
  __int64 v121; // rsi
  _BYTE *v122; // rax
  __int64 v123; // rsi
  _BYTE *v124; // r13
  __int64 v125; // r14
  __int64 v126; // r12
  __int64 *v127; // rbx
  __int64 v128; // rax
  __int64 v129; // rcx
  __int64 v130; // rsi
  unsigned __int8 *v131; // rsi
  __int64 **v132; // rdx
  __int64 v133; // rsi
  __int64 v134; // r14
  __int64 *v135; // r13
  unsigned __int64 *v136; // rbx
  __int64 v137; // rax
  unsigned __int64 v138; // rcx
  unsigned __int8 **v139; // r8
  int v140; // r9d
  __int64 v141; // rsi
  unsigned __int8 *v142; // rsi
  signed __int64 v143; // rbx
  __int64 v144; // r14
  __int64 v145; // rax
  unsigned __int8 *v146; // r15
  double v147; // xmm4_8
  double v148; // xmm5_8
  unsigned int v149; // eax
  __int64 v150; // rcx
  __int64 v151; // r14
  int v152; // eax
  __int64 v153; // rbx
  __int64 v154; // rax
  unsigned __int8 *v155; // rsi
  __int64 v156; // r10
  _QWORD *v157; // rbx
  double v158; // xmm4_8
  double v159; // xmm5_8
  unsigned __int64 v160; // rbx
  _QWORD *v161; // rax
  unsigned __int8 *v162; // rsi
  __int64 v163; // r13
  __int64 v164; // r12
  __int64 **v165; // rdx
  _QWORD *v166; // rax
  _QWORD *v167; // r15
  unsigned __int64 *v168; // r12
  __int64 v169; // rax
  unsigned __int64 v170; // rcx
  __int64 v171; // rsi
  unsigned __int8 *v172; // rsi
  __int64 v173; // rdx
  __int64 v174; // rax
  _BYTE *v175; // rdx
  __int64 v176; // rsi
  __int64 v177; // r13
  __int64 v178; // r15
  _QWORD *v179; // r12
  unsigned __int64 *v180; // r13
  __int64 v181; // rax
  unsigned __int64 v182; // rcx
  __int64 v183; // rsi
  unsigned __int8 *v184; // rsi
  _BYTE *v185; // r12
  __int64 v186; // rcx
  __int64 v187; // rax
  unsigned __int64 v188; // r12
  _QWORD *v189; // rax
  unsigned __int8 *v190; // rsi
  __int64 v191; // rsi
  _BYTE *v192; // rax
  __int64 v193; // rsi
  _BYTE *v194; // r12
  __int64 v195; // r15
  __int64 v196; // rax
  unsigned __int8 *v197; // rax
  __int64 v198; // rdx
  _QWORD *v199; // rax
  _QWORD *v200; // r13
  unsigned __int64 *v201; // r15
  __int64 v202; // rax
  unsigned __int64 v203; // rcx
  __int64 v204; // rsi
  unsigned __int8 *v205; // rsi
  __int64 v206; // rsi
  __int64 v207; // rax
  __int64 v208; // rdx
  __int64 v209; // rax
  __int64 v210; // r13
  _QWORD *v211; // r12
  unsigned __int64 *v212; // r13
  __int64 v213; // rax
  unsigned __int64 v214; // rcx
  __int64 v215; // rsi
  unsigned __int8 *v216; // rsi
  __int64 v217; // rdx
  __int64 **v218; // rax
  _BYTE *v219; // r12
  __int64 v220; // rbx
  _QWORD *v221; // rax
  __int64 v222; // rax
  unsigned __int8 *v223; // rax
  __int64 v224; // r9
  __int64 v225; // r14
  __int64 v226; // r15
  _QWORD *v227; // rax
  __int64 v228; // rdi
  __int64 v229; // r12
  __int64 v230; // rax
  __int64 v231; // r15
  __int64 j; // rbx
  __int64 v233; // r14
  __int64 *v234; // rdx
  __int64 *v235; // r14
  __int64 *v236; // rax
  __int64 v237; // rdi
  __int64 *v238; // r15
  __int64 v239; // rsi
  __int64 *v240; // rax
  _QWORD *v241; // rax
  __int64 v242; // r12
  unsigned __int64 *v243; // r13
  __int64 v244; // rax
  unsigned __int64 v245; // rcx
  __int64 v246; // rsi
  unsigned __int8 *v247; // rsi
  _QWORD *v248; // rax
  __int64 v249; // r10
  __int64 v250; // r14
  __int64 *v251; // rbx
  __int64 v252; // rcx
  __int64 v253; // rax
  __int64 v254; // rsi
  __int64 v255; // rbx
  unsigned __int8 *v256; // rsi
  __int64 *v257; // rbx
  __int64 v258; // rax
  __int64 v259; // rcx
  __int64 v260; // rsi
  unsigned __int8 *v261; // rsi
  unsigned __int64 v262; // rax
  __int64 v263; // r12
  __int64 v264; // r13
  _QWORD *v265; // rax
  __int64 v266; // r9
  __int64 v267; // rbx
  __int64 v268; // r10
  __int64 *v269; // rax
  __int64 v270; // r12
  __int64 v271; // rax
  __int64 v272; // rbx
  __int64 v273; // r12
  __int64 v274; // rax
  char v275; // al
  _QWORD *v276; // r14
  _QWORD *v277; // rax
  __int64 v278; // r15
  _QWORD *v279; // rax
  __int64 v280; // r14
  __int64 *v281; // rbx
  __int64 v282; // r15
  __int64 v283; // r14
  _QWORD *v284; // rax
  __int64 v285; // rdi
  int v286; // eax
  __int64 *v287; // r10
  int v288; // edi
  _QWORD *v289; // r10
  unsigned int v290; // ecx
  __int64 *v291; // rsi
  _QWORD *v292; // rdi
  unsigned __int64 *v293; // r13
  __int64 v294; // rax
  unsigned __int64 v295; // rcx
  __int64 v296; // rsi
  unsigned __int8 *v297; // rsi
  __int64 *v298; // rbx
  __int64 v299; // rax
  __int64 v300; // rcx
  __int64 v301; // rsi
  unsigned __int8 *v302; // rsi
  int v303; // edi
  _QWORD *v304; // r10
  unsigned int v305; // ecx
  int v306; // r11d
  __int64 v307; // rax
  __int64 v308; // [rsp+28h] [rbp-548h]
  __int64 v309; // [rsp+48h] [rbp-528h]
  __int64 v310; // [rsp+48h] [rbp-528h]
  __int64 v311; // [rsp+50h] [rbp-520h]
  _QWORD *v312; // [rsp+58h] [rbp-518h]
  __int64 v313; // [rsp+58h] [rbp-518h]
  __int64 v314; // [rsp+58h] [rbp-518h]
  __int64 *v315; // [rsp+58h] [rbp-518h]
  __int64 v316; // [rsp+58h] [rbp-518h]
  __int64 v317; // [rsp+58h] [rbp-518h]
  __int64 v318; // [rsp+58h] [rbp-518h]
  __int64 v319; // [rsp+58h] [rbp-518h]
  unsigned __int64 v322; // [rsp+70h] [rbp-500h]
  __int64 v323; // [rsp+70h] [rbp-500h]
  unsigned __int64 v324; // [rsp+78h] [rbp-4F8h]
  __int64 *v325; // [rsp+80h] [rbp-4F0h]
  __int64 v326; // [rsp+80h] [rbp-4F0h]
  __int64 v327; // [rsp+80h] [rbp-4F0h]
  _QWORD *v328; // [rsp+88h] [rbp-4E8h]
  __int64 v329; // [rsp+88h] [rbp-4E8h]
  __int64 v330; // [rsp+88h] [rbp-4E8h]
  __int64 v331; // [rsp+88h] [rbp-4E8h]
  __int64 v332; // [rsp+88h] [rbp-4E8h]
  __int64 v333; // [rsp+88h] [rbp-4E8h]
  __int64 *v334; // [rsp+88h] [rbp-4E8h]
  _QWORD *v335; // [rsp+90h] [rbp-4E0h]
  __int64 v336; // [rsp+90h] [rbp-4E0h]
  _QWORD *v337; // [rsp+90h] [rbp-4E0h]
  unsigned __int64 v338; // [rsp+90h] [rbp-4E0h]
  _QWORD *v339; // [rsp+90h] [rbp-4E0h]
  __int64 v340; // [rsp+90h] [rbp-4E0h]
  __int64 v341; // [rsp+90h] [rbp-4E0h]
  __int64 *v342; // [rsp+98h] [rbp-4D8h]
  _QWORD *v343; // [rsp+98h] [rbp-4D8h]
  unsigned int v344; // [rsp+98h] [rbp-4D8h]
  int v345; // [rsp+A4h] [rbp-4CCh] BYREF
  unsigned __int8 *v346; // [rsp+A8h] [rbp-4C8h] BYREF
  _BYTE *v347[2]; // [rsp+B0h] [rbp-4C0h] BYREF
  char v348; // [rsp+C0h] [rbp-4B0h]
  char v349; // [rsp+C1h] [rbp-4AFh]
  unsigned __int8 *v350[2]; // [rsp+D0h] [rbp-4A0h] BYREF
  __int16 v351; // [rsp+E0h] [rbp-490h]
  unsigned __int8 *v352; // [rsp+F0h] [rbp-480h] BYREF
  __int64 v353; // [rsp+F8h] [rbp-478h]
  __int64 *v354; // [rsp+100h] [rbp-470h]
  _QWORD *v355; // [rsp+108h] [rbp-468h]
  __int64 v356; // [rsp+110h] [rbp-460h]
  int v357; // [rsp+118h] [rbp-458h]
  __int64 v358; // [rsp+120h] [rbp-450h]
  __int64 v359; // [rsp+128h] [rbp-448h]
  __int64 *v360; // [rsp+140h] [rbp-430h] BYREF
  __int64 v361; // [rsp+148h] [rbp-428h]
  _BYTE v362[128]; // [rsp+150h] [rbp-420h] BYREF
  _BYTE *v363; // [rsp+1D0h] [rbp-3A0h] BYREF
  __int64 v364; // [rsp+1D8h] [rbp-398h]
  _BYTE v365[128]; // [rsp+1E0h] [rbp-390h] BYREF
  char *v366; // [rsp+260h] [rbp-310h] BYREF
  __int64 v367; // [rsp+268h] [rbp-308h]
  _QWORD v368[3]; // [rsp+270h] [rbp-300h] BYREF
  int v369; // [rsp+288h] [rbp-2E8h]
  __int64 v370; // [rsp+290h] [rbp-2E0h]
  __int64 v371; // [rsp+298h] [rbp-2D8h]
  __int64 v372; // [rsp+2F0h] [rbp-280h] BYREF
  __int64 v373; // [rsp+2F8h] [rbp-278h]
  _QWORD *v374; // [rsp+300h] [rbp-270h] BYREF
  unsigned int v375; // [rsp+308h] [rbp-268h]
  _BYTE *v376; // [rsp+380h] [rbp-1F0h] BYREF
  __int64 v377; // [rsp+388h] [rbp-1E8h]
  _BYTE v378[128]; // [rsp+390h] [rbp-1E0h] BYREF
  __int64 v379; // [rsp+410h] [rbp-160h] BYREF
  __int64 v380; // [rsp+418h] [rbp-158h]
  __int64 *v381; // [rsp+420h] [rbp-150h] BYREF
  __int64 v382; // [rsp+428h] [rbp-148h]
  __int64 v383; // [rsp+430h] [rbp-140h]
  __int64 v384; // [rsp+438h] [rbp-138h] BYREF
  __int64 v385; // [rsp+440h] [rbp-130h]
  __int64 v386; // [rsp+448h] [rbp-128h]

  v11 = (__int64)v365;
  v363 = v365;
  v372 = 0;
  v373 = 1;
  v360 = (__int64 *)v362;
  v361 = 0x1000000000LL;
  v364 = 0x1000000000LL;
  v12 = (unsigned __int64 *)&v374;
  do
    *v12++ = -8;
  while ( v12 != (unsigned __int64 *)&v376 );
  v376 = v378;
  v377 = 0x1000000000LL;
  v13 = *(_QWORD *)(a2 + 80);
  v14 = a2 + 72;
  if ( v13 == a2 + 72 )
  {
    v31 = 0;
    goto LABEL_23;
  }
  do
  {
    while ( 1 )
    {
      v25 = v13 - 24;
      if ( !v13 )
        v25 = 0;
      v28 = sub_157EBA0(v25);
      v29 = *(_BYTE *)(v28 + 16);
      if ( v29 != 29 )
        break;
      v15 = *(_QWORD *)(v28 - 72);
      if ( !*(_BYTE *)(v15 + 16) && *(_DWORD *)(v15 + 36) == 40 )
      {
        v11 = 1;
        v341 = *(_QWORD *)(v28 - 48);
        v292 = sub_1648A60(56, 1u);
        if ( v292 )
        {
          v11 = v341;
          sub_15F8320((__int64)v292, v341, v28);
        }
        sub_15F20C0((_QWORD *)v28);
        goto LABEL_12;
      }
      v16 = (unsigned int)v364;
      if ( (unsigned int)v364 >= HIDWORD(v364) )
      {
        sub_16CD150((__int64)&v363, v365, 0, 8, v26, v27);
        v16 = (unsigned int)v364;
      }
      *(_QWORD *)&v363[8 * v16] = v28;
      LODWORD(v364) = v364 + 1;
      v17 = sub_157F7B0(*(_QWORD *)(v28 - 24));
      v20 = v17;
      if ( (v373 & 1) != 0 )
      {
        v21 = 15;
        v22 = &v374;
      }
      else
      {
        v33 = v375;
        v22 = v374;
        if ( !v375 )
        {
          v34 = v373;
          ++v372;
          v23 = 0;
          v35 = ((unsigned int)v373 >> 1) + 1;
          goto LABEL_33;
        }
        v21 = v375 - 1;
      }
      v11 = v21 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v23 = &v22[v11];
      v24 = *v23;
      if ( v17 != *v23 )
      {
        v286 = 1;
        v287 = 0;
        while ( v24 != -8 )
        {
          if ( !v287 && v24 == -16 )
            v287 = v23;
          v306 = v286 + 1;
          v307 = v21 & (unsigned int)(v11 + v286);
          v23 = &v22[v307];
          v11 = (unsigned int)v307;
          v24 = *v23;
          if ( v20 == *v23 )
            goto LABEL_12;
          v286 = v306;
        }
        v34 = v373;
        if ( v287 )
          v23 = v287;
        ++v372;
        v35 = ((unsigned int)v373 >> 1) + 1;
        if ( (v373 & 1) != 0 )
        {
          LODWORD(v36) = 4 * v35;
          v33 = 16;
          if ( 4 * v35 >= 0x30 )
            goto LABEL_403;
          goto LABEL_34;
        }
        v33 = v375;
LABEL_33:
        LODWORD(v36) = 4 * v35;
        if ( 4 * v35 >= 3 * v33 )
        {
LABEL_403:
          sub_21170F0((__int64)&v372, 2 * v33);
          if ( (v373 & 1) != 0 )
          {
            v288 = 15;
            v289 = &v374;
          }
          else
          {
            v289 = v374;
            if ( !v375 )
            {
LABEL_466:
              LODWORD(v373) = (2 * ((unsigned int)v373 >> 1) + 2) | v373 & 1;
              BUG();
            }
            v288 = v375 - 1;
          }
          v34 = v373;
          v290 = v288 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
          v23 = &v289[v290];
          v36 = *v23;
          if ( v20 == *v23 )
            goto LABEL_35;
          v38 = 1;
          v291 = 0;
          while ( v36 != -8 )
          {
            if ( v36 == -16 && !v291 )
              v291 = v23;
            v290 = v288 & (v38 + v290);
            v23 = &v289[v290];
            v36 = *v23;
            if ( v20 == *v23 )
              goto LABEL_410;
            ++v38;
          }
          goto LABEL_408;
        }
LABEL_34:
        v37 = v33 - HIDWORD(v373) - v35;
        v38 = v33 >> 3;
        if ( v37 > v33 >> 3 )
          goto LABEL_35;
        sub_21170F0((__int64)&v372, v33);
        if ( (v373 & 1) != 0 )
        {
          v303 = 15;
          v304 = &v374;
        }
        else
        {
          v304 = v374;
          if ( !v375 )
            goto LABEL_466;
          v303 = v375 - 1;
        }
        v34 = v373;
        v305 = v303 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        v23 = &v304[v305];
        v36 = *v23;
        if ( v20 == *v23 )
          goto LABEL_35;
        v38 = 1;
        v291 = 0;
        while ( v36 != -8 )
        {
          if ( v36 == -16 && !v291 )
            v291 = v23;
          v305 = v303 & (v38 + v305);
          v23 = &v304[v305];
          v36 = *v23;
          if ( v20 == *v23 )
            goto LABEL_410;
          ++v38;
        }
LABEL_408:
        if ( v291 )
          v23 = v291;
LABEL_410:
        v34 = v373;
LABEL_35:
        v39 = v34;
        v11 = v34 & 1;
        LODWORD(v373) = (2 * (v39 >> 1) + 2) | v11;
        if ( *v23 != -8 )
          --HIDWORD(v373);
        *v23 = v20;
        v40 = (unsigned int)v377;
        if ( (unsigned int)v377 >= HIDWORD(v377) )
        {
          v11 = (__int64)v378;
          sub_16CD150((__int64)&v376, v378, 0, 8, v38, v36);
          v40 = (unsigned int)v377;
        }
        *(_QWORD *)&v376[8 * v40] = v20;
        LODWORD(v377) = v377 + 1;
      }
LABEL_12:
      v13 = *(_QWORD *)(v13 + 8);
      if ( v14 == v13 )
        goto LABEL_20;
    }
    if ( v29 != 25 )
      goto LABEL_12;
    v30 = (unsigned int)v361;
    if ( (unsigned int)v361 >= HIDWORD(v361) )
    {
      v11 = (__int64)v362;
      sub_16CD150((__int64)&v360, v362, 0, 8, v26, v27);
      v30 = (unsigned int)v361;
    }
    v360[v30] = v28;
    LODWORD(v361) = v361 + 1;
    v13 = *(_QWORD *)(v13 + 8);
  }
  while ( v14 != v13 );
LABEL_20:
  v311 = v13;
  v31 = 0;
  if ( !(_DWORD)v364 )
    goto LABEL_21;
  v41 = *(_QWORD *)(a2 + 80);
  if ( !v41 )
    BUG();
  for ( i = *(_QWORD *)(v41 + 24); ; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    v43 = i - 24;
    if ( *(_BYTE *)(i - 8) != 53 || !(unsigned __int8)sub_15F8F00(i - 24) )
      break;
  }
  if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
  {
    sub_15E08E0(a2, v11);
    v44 = *(__int64 **)(a2 + 88);
    v342 = &v44[5 * *(_QWORD *)(a2 + 96)];
    if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
    {
      sub_15E08E0(a2, v11);
      v44 = *(__int64 **)(a2 + 88);
    }
  }
  else
  {
    v44 = *(__int64 **)(a2 + 88);
    v342 = &v44[5 * *(_QWORD *)(a2 + 96)];
  }
  for ( ; v342 != v44; v44 += 5 )
  {
    if ( !(unsigned __int8)sub_1649A90((__int64)v44) )
    {
      v51 = (__int64 **)*v44;
      v52 = (__int64 *)sub_15E0530(a2);
      v53 = sub_159C4F0(v52);
      v54 = sub_1599EF0(v51);
      v366 = (char *)sub_1649960((__int64)v44);
      LOWORD(v381) = 773;
      v379 = (__int64)&v366;
      v367 = v55;
      v380 = (__int64)".tmp";
      v56 = sub_1648A60(56, 3u);
      v59 = (__int64)v56;
      if ( v56 )
      {
        v328 = v56 - 9;
        v335 = v56;
        sub_15F1EA0((__int64)v56, *v44, 55, (__int64)(v56 - 9), 3, v43);
        if ( *(_QWORD *)(v59 - 72) )
        {
          v60 = *(_QWORD *)(v59 - 64);
          v61 = *(_QWORD *)(v59 - 56) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v61 = v60;
          if ( v60 )
            *(_QWORD *)(v60 + 16) = v61 | *(_QWORD *)(v60 + 16) & 3LL;
        }
        *(_QWORD *)(v59 - 72) = v53;
        if ( v53 )
        {
          v62 = *(_QWORD *)(v53 + 8);
          *(_QWORD *)(v59 - 64) = v62;
          if ( v62 )
            *(_QWORD *)(v62 + 16) = (v59 - 64) | *(_QWORD *)(v62 + 16) & 3LL;
          *(_QWORD *)(v59 - 56) = (v53 + 8) | *(_QWORD *)(v59 - 56) & 3LL;
          *(_QWORD *)(v53 + 8) = v328;
        }
        if ( *(_QWORD *)(v59 - 48) )
        {
          v63 = *(_QWORD *)(v59 - 40);
          v64 = *(_QWORD *)(v59 - 32) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v64 = v63;
          if ( v63 )
            *(_QWORD *)(v63 + 16) = v64 | *(_QWORD *)(v63 + 16) & 3LL;
        }
        *(_QWORD *)(v59 - 48) = v44;
        v65 = v44[1];
        *(_QWORD *)(v59 - 40) = v65;
        if ( v65 )
          *(_QWORD *)(v65 + 16) = (v59 - 40) | *(_QWORD *)(v65 + 16) & 3LL;
        v66 = (unsigned __int64)(v44 + 1);
        *(_QWORD *)(v59 - 32) = (unsigned __int64)(v44 + 1) | *(_QWORD *)(v59 - 32) & 3LL;
        v44[1] = v59 - 48;
        if ( *(_QWORD *)(v59 - 24) )
        {
          v67 = *(_QWORD *)(v59 - 16);
          v68 = *(_QWORD *)(v59 - 8) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v68 = v67;
          if ( v67 )
            *(_QWORD *)(v67 + 16) = v68 | *(_QWORD *)(v67 + 16) & 3LL;
        }
        *(_QWORD *)(v59 - 24) = v54;
        if ( v54 )
        {
          v69 = *(_QWORD *)(v54 + 8);
          *(_QWORD *)(v59 - 16) = v69;
          if ( v69 )
            *(_QWORD *)(v69 + 16) = (v59 - 16) | *(_QWORD *)(v69 + 16) & 3LL;
          *(_QWORD *)(v59 - 8) = (v54 + 8) | *(_QWORD *)(v59 - 8) & 3LL;
          *(_QWORD *)(v54 + 8) = v59 - 24;
        }
        sub_164B780(v59, &v379);
      }
      else
      {
        v335 = 0;
        v66 = (unsigned __int64)(v44 + 1);
      }
      sub_164D160((__int64)v44, v59, a3, *(double *)a4.m128i_i64, a5, a6, v57, v58, a9, a10);
      if ( (*(_BYTE *)(v59 + 23) & 0x40) != 0 )
        v45 = *(_QWORD **)(v59 - 8);
      else
        v45 = &v335[-3 * (*(_DWORD *)(v59 + 20) & 0xFFFFFFF)];
      if ( v45[3] )
      {
        v46 = v45[4];
        v47 = v45[5] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v47 = v46;
        if ( v46 )
          *(_QWORD *)(v46 + 16) = v47 | *(_QWORD *)(v46 + 16) & 3LL;
      }
      v45[3] = v44;
      v48 = v44[1];
      v45[4] = v48;
      if ( v48 )
        *(_QWORD *)(v48 + 16) = (unsigned __int64)(v45 + 4) | *(_QWORD *)(v48 + 16) & 3LL;
      v49 = v45[5];
      v50 = v45 + 3;
      v50[2] = v49 & 3 | v66;
      v44[1] = (__int64)v50;
    }
  }
  v70 = &v379;
  v324 = (unsigned __int64)v363;
  v309 = *(_QWORD *)(a2 + 80);
  v343 = &v363[8 * (unsigned int)v364];
  if ( v311 == v309 )
    goto LABEL_147;
  while ( 2 )
  {
    if ( !v309 )
      BUG();
    v71 = v309 - 24;
    v336 = *(_QWORD *)(v309 + 24);
    if ( v336 == v309 + 16 )
      goto LABEL_146;
    while ( 2 )
    {
      if ( !v336 )
        BUG();
      v72 = *(_QWORD *)(v336 - 16);
      v73 = v336 - 24;
      if ( !v72 )
        goto LABEL_87;
      if ( !*(_QWORD *)(v72 + 8) )
      {
        v74 = sub_1648700(v72);
        if ( v71 == v74[5] && *((_BYTE *)v74 + 16) != 77 )
          goto LABEL_87;
      }
      if ( *(_BYTE *)(v336 - 8) == 53 && (unsigned __int8)sub_15F8F00(v73) )
        goto LABEL_87;
      v75 = 0;
      v366 = (char *)v368;
      v367 = 0x1000000000LL;
      if ( !*(_QWORD *)(v336 - 16) )
      {
        v79 = &v384;
        v384 = v71;
        v380 = (__int64)&v384;
        v381 = &v384;
        v382 = 0x100000020LL;
        LODWORD(v383) = 0;
        v379 = 1;
LABEL_316:
        if ( (_QWORD *)v324 == v343 )
          goto LABEL_127;
        v87 = &v384;
LABEL_117:
        v330 = v73;
        v88 = (__int64)v70;
        v89 = v71;
        v90 = (_QWORD *)v324;
        while ( 1 )
        {
          v91 = *(_QWORD *)(*v90 - 24LL);
          if ( v89 != v91 )
            break;
LABEL_130:
          if ( v343 == ++v90 )
          {
            v71 = v89;
            v70 = (__int64 *)v88;
            goto LABEL_125;
          }
        }
        if ( v79 == v87 )
        {
          v92 = &v79[HIDWORD(v382)];
          if ( v92 == v79 )
          {
            v234 = v79;
          }
          else
          {
            do
            {
              if ( v91 == *v79 )
                break;
              ++v79;
            }
            while ( v92 != v79 );
            v234 = v92;
          }
        }
        else
        {
          v92 = &v87[(unsigned int)v382];
          v79 = sub_16CC9F0(v88, *(_QWORD *)(*v90 - 24LL));
          if ( v91 == *v79 )
          {
            if ( v381 == (__int64 *)v380 )
              v234 = &v381[HIDWORD(v382)];
            else
              v234 = &v381[(unsigned int)v382];
          }
          else
          {
            if ( v381 != (__int64 *)v380 )
            {
              v79 = &v381[(unsigned int)v382];
LABEL_123:
              if ( v79 != v92 )
              {
                v70 = (__int64 *)v88;
                v71 = v89;
                sub_1AC3CB0(v330, 1u, 0, a3, a4, a5, a6, v18, v19, a9, a10);
                v87 = v381;
                v79 = (__int64 *)v380;
                goto LABEL_125;
              }
              v87 = v381;
              v79 = (__int64 *)v380;
              goto LABEL_130;
            }
            v79 = &v381[HIDWORD(v382)];
            v234 = v79;
          }
        }
        while ( v234 != v79 && (unsigned __int64)*v79 >= 0xFFFFFFFFFFFFFFFELL )
          ++v79;
        goto LABEL_123;
      }
      v76 = *(_QWORD *)(v336 - 16);
      do
      {
        while ( 1 )
        {
          v93 = sub_1648700(v76);
          if ( v71 != v93[5] || *((_BYTE *)v93 + 16) == 77 )
            break;
          v76 = *(_QWORD *)(v76 + 8);
          if ( !v76 )
            goto LABEL_101;
        }
        if ( HIDWORD(v367) <= (unsigned int)v75 )
        {
          v312 = v93;
          sub_16CD150((__int64)&v366, v368, 0, 8, v77, v78);
          v75 = (unsigned int)v367;
          v93 = v312;
        }
        *(_QWORD *)&v366[8 * v75] = v93;
        v75 = (unsigned int)(v367 + 1);
        LODWORD(v367) = v367 + 1;
        v76 = *(_QWORD *)(v76 + 8);
      }
      while ( v76 );
LABEL_101:
      v79 = &v384;
      v384 = v71;
      v73 = v336 - 24;
      v380 = (__int64)&v384;
      v381 = &v384;
      v382 = 0x100000020LL;
      LODWORD(v383) = 0;
      v379 = 1;
      if ( !(_DWORD)v75 )
        goto LABEL_316;
      v329 = v71;
      v80 = v75;
      while ( 2 )
      {
        while ( 2 )
        {
          v81 = v80--;
          v82 = *(_QWORD *)&v366[8 * v81 - 8];
          LODWORD(v367) = v80;
          if ( *(_BYTE *)(v82 + 16) != 77 )
          {
            sub_2116AE0(*(_QWORD *)(v82 + 40), (__int64)v70);
            v80 = v367;
            goto LABEL_104;
          }
          if ( (*(_DWORD *)(v82 + 20) & 0xFFFFFFF) == 0 )
          {
LABEL_104:
            if ( !v80 )
              goto LABEL_116;
            continue;
          }
          break;
        }
        v83 = 0;
        v84 = 8LL * (*(_DWORD *)(v82 + 20) & 0xFFFFFFF);
        do
        {
          while ( 1 )
          {
            v85 = (*(_BYTE *)(v82 + 23) & 0x40) != 0
                ? *(_QWORD *)(v82 - 8)
                : v82 - 24LL * (*(_DWORD *)(v82 + 20) & 0xFFFFFFF);
            v86 = *(_QWORD *)(v85 + 3 * v83);
            if ( v86 )
            {
              if ( v86 == v73 )
                break;
            }
            v83 += 8;
            if ( v83 == v84 )
              goto LABEL_115;
          }
          v83 += 8;
          sub_2116AE0(*(_QWORD *)(v83 + v85 + 24LL * *(unsigned int *)(v82 + 56)), (__int64)v70);
        }
        while ( v83 != v84 );
LABEL_115:
        v80 = v367;
        if ( (_DWORD)v367 )
          continue;
        break;
      }
LABEL_116:
      v71 = v329;
      v87 = v381;
      v79 = (__int64 *)v380;
      if ( (_QWORD *)v324 != v343 )
        goto LABEL_117;
LABEL_125:
      if ( v79 != v87 )
        _libc_free((unsigned __int64)v87);
LABEL_127:
      if ( v366 != (char *)v368 )
        _libc_free((unsigned __int64)v366);
LABEL_87:
      v336 = *(_QWORD *)(v336 + 8);
      if ( v309 + 16 != v336 )
        continue;
      break;
    }
LABEL_146:
    v309 = *(_QWORD *)(v309 + 8);
    if ( v311 != v309 )
      continue;
    break;
  }
LABEL_147:
  if ( (_QWORD *)v324 == v343 )
    goto LABEL_167;
  v337 = (_QWORD *)v324;
  while ( 2 )
  {
    v94 = *(_QWORD *)(*v337 - 24LL);
    v95 = sub_157F7B0(v94);
    v379 = 0;
    v382 = 8;
    v98 = (_QWORD *)v95;
    LODWORD(v383) = 0;
    v380 = (__int64)&v384;
    v381 = &v384;
    v99 = *(_QWORD *)(v94 + 48);
    while ( 2 )
    {
      if ( !v99 )
        BUG();
      v100 = (unsigned __int64)v381;
      v101 = v99 - 24;
      v102 = (__int64 *)v380;
      if ( *(_BYTE *)(v99 - 8) == 77 )
      {
        if ( (__int64 *)v380 != v381 )
          goto LABEL_150;
        v103 = (__int64 *)(v380 + 8LL * HIDWORD(v382));
        if ( v103 != (__int64 *)v380 )
        {
          v104 = 0;
          while ( v101 != *v102 )
          {
            if ( *v102 == -2 )
              v104 = v102;
            if ( v103 == ++v102 )
            {
              if ( !v104 )
                goto LABEL_313;
              *v104 = v101;
              LODWORD(v383) = v383 - 1;
              ++v379;
              goto LABEL_151;
            }
          }
          goto LABEL_151;
        }
LABEL_313:
        if ( HIDWORD(v382) < (unsigned int)v382 )
        {
          ++HIDWORD(v382);
          *v103 = v101;
          ++v379;
        }
        else
        {
LABEL_150:
          sub_16CCBA0((__int64)&v379, v101);
        }
LABEL_151:
        v99 = *(_QWORD *)(v99 + 8);
        continue;
      }
      break;
    }
    if ( HIDWORD(v382) == (_DWORD)v383 )
    {
      if ( (__int64 *)v380 != v381 )
        goto LABEL_165;
    }
    else
    {
      v235 = &v381[HIDWORD(v382)];
      if ( (__int64 *)v380 != v381 )
        v235 = &v381[(unsigned int)v382];
      v236 = v381;
      if ( v235 != v381 )
      {
        while ( 1 )
        {
          v237 = *v236;
          v238 = v236;
          if ( (unsigned __int64)*v236 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v235 == ++v236 )
            goto LABEL_323;
        }
        if ( v235 != v236 )
        {
          do
          {
            sub_1AC3A00(v237, 0, a3, *(double *)a4.m128i_i64, a5, a6, v96, v97, a9, a10);
            v240 = v238 + 1;
            if ( v238 + 1 == v235 )
              break;
            v237 = *v240;
            for ( ++v238; (unsigned __int64)*v240 >= 0xFFFFFFFFFFFFFFFELL; v238 = v240 )
            {
              if ( v235 == ++v240 )
                goto LABEL_323;
              v237 = *v240;
            }
          }
          while ( v238 != v235 );
        }
      }
LABEL_323:
      v239 = *(_QWORD *)(v94 + 48);
      if ( v239 )
        v239 -= 24;
      sub_15F22F0(v98, v239);
      v100 = (unsigned __int64)v381;
      if ( v381 != (__int64 *)v380 )
LABEL_165:
        _libc_free(v100);
    }
    if ( v343 != ++v337 )
      continue;
    break;
  }
LABEL_167:
  v105 = (unsigned __int64)v376;
  v106 = *(_QWORD *)(a2 + 80);
  v107 = *(_QWORD *)(a2 + 40);
  v108 = 8LL * (unsigned int)v377;
  if ( !v106 )
  {
    v10 = sub_1632FA0(v107);
    sub_15AAE50(v10, *(_QWORD *)(a1 + 176));
    v379 = (__int64)"fn_context";
    LOWORD(v381) = 259;
    BUG();
  }
  v308 = v106 - 24;
  v109 = sub_1632FA0(v107);
  v110 = sub_15AAE50(v109, *(_QWORD *)(a1 + 176));
  v111 = *(_DWORD *)(v109 + 4);
  v379 = (__int64)"fn_context";
  LOWORD(v381) = 259;
  v112 = *(_QWORD *)(v106 + 24);
  v344 = v110;
  if ( v112 )
    v112 -= 24;
  v113 = sub_1648A60(64, 1u);
  if ( v113 )
    sub_15F8A50((__int64)v113, *(_QWORD **)(a1 + 176), v111, 0, v344, (__int64)&v379, v112);
  v325 = (__int64 *)v105;
  *(_QWORD *)(a1 + 256) = v113;
  v322 = v105 + v108;
  if ( v105 != v105 + v108 )
  {
    while ( 2 )
    {
      v114 = *v325;
      v331 = *v325;
      v115 = sub_157EE30(*(_QWORD *)(*v325 + 40));
      v116 = *(_QWORD *)(v114 + 40);
      v117 = v115;
      v118 = (_QWORD *)sub_157E9C0(v116);
      v353 = v116;
      v352 = 0;
      v355 = v118;
      v356 = 0;
      v357 = 0;
      v358 = 0;
      v359 = 0;
      v354 = (__int64 *)v117;
      if ( v117 != v116 + 40 )
      {
        if ( !v117 )
          BUG();
        v119 = *(unsigned __int8 **)(v117 + 24);
        v379 = (__int64)v119;
        if ( v119 )
        {
          sub_1623A60((__int64)&v379, (__int64)v119, 2);
          if ( v352 )
            sub_161E7C0((__int64)&v352, (__int64)v352);
          v352 = (unsigned __int8 *)v379;
          if ( v379 )
            sub_1623210((__int64)&v379, (unsigned __int8 *)v379, (__int64)&v352);
        }
      }
      v120 = *(_BYTE **)(a1 + 256);
      v121 = *(_QWORD *)(a1 + 176);
      v379 = (__int64)"__data";
      LOWORD(v381) = 259;
      v122 = (_BYTE *)sub_18174F0((__int64)&v352, v121, v120, 0, 2u, &v379);
      v123 = *(_QWORD *)(a1 + 160);
      v124 = v122;
      v379 = (__int64)"exception_gep";
      LOWORD(v381) = 259;
      v125 = sub_18174F0((__int64)&v352, v123, v122, 0, 0, &v379);
      LOWORD(v381) = 259;
      v379 = (__int64)"exn_val";
      v126 = (__int64)sub_1648A60(64, 1u);
      if ( v126 )
        sub_15F9210(v126, *(_QWORD *)(*(_QWORD *)v125 + 24LL), v125, 0, 1u, 0);
      if ( v353 )
      {
        v127 = v354;
        sub_157E9D0(v353 + 40, v126);
        v128 = *(_QWORD *)(v126 + 24);
        v129 = *v127;
        *(_QWORD *)(v126 + 32) = v127;
        v129 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v126 + 24) = v129 | v128 & 7;
        *(_QWORD *)(v129 + 8) = v126 + 24;
        *v127 = *v127 & 7 | (v126 + 24);
      }
      sub_164B780(v126, &v379);
      if ( v352 )
      {
        v366 = (char *)v352;
        sub_1623A60((__int64)&v366, (__int64)v352, 2);
        v130 = *(_QWORD *)(v126 + 48);
        if ( v130 )
          sub_161E7C0(v126 + 48, v130);
        v131 = (unsigned __int8 *)v366;
        *(_QWORD *)(v126 + 48) = v366;
        if ( v131 )
          sub_1623210((__int64)&v366, v131, v126 + 48);
      }
      LOWORD(v368[0]) = 257;
      v132 = (__int64 **)sub_16471D0(v355, 0);
      if ( v132 != *(__int64 ***)v126 )
      {
        if ( *(_BYTE *)(v126 + 16) > 0x10u )
        {
          LOWORD(v381) = 257;
          v126 = sub_15FDBD0(46, v126, (__int64)v132, (__int64)&v379, 0);
          if ( v353 )
          {
            v257 = v354;
            sub_157E9D0(v353 + 40, v126);
            v258 = *(_QWORD *)(v126 + 24);
            v259 = *v257;
            *(_QWORD *)(v126 + 32) = v257;
            v259 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v126 + 24) = v259 | v258 & 7;
            *(_QWORD *)(v259 + 8) = v126 + 24;
            *v257 = *v257 & 7 | (v126 + 24);
          }
          sub_164B780(v126, (__int64 *)&v366);
          if ( v352 )
          {
            v350[0] = v352;
            sub_1623A60((__int64)v350, (__int64)v352, 2);
            v260 = *(_QWORD *)(v126 + 48);
            if ( v260 )
              sub_161E7C0(v126 + 48, v260);
            v261 = v350[0];
            *(unsigned __int8 **)(v126 + 48) = v350[0];
            if ( v261 )
              sub_1623210((__int64)v350, v261, v126 + 48);
          }
        }
        else
        {
          v126 = sub_15A46C0(46, (__int64 ***)v126, v132, 0);
        }
      }
      v379 = (__int64)"exn_selector_gep";
      v133 = *(_QWORD *)(a1 + 160);
      LOWORD(v381) = 259;
      v134 = sub_18174F0((__int64)&v352, v133, v124, 0, 1u, &v379);
      LOWORD(v381) = 259;
      v379 = (__int64)"exn_selector_val";
      v135 = sub_1648A60(64, 1u);
      if ( v135 )
        sub_15F9210((__int64)v135, *(_QWORD *)(*(_QWORD *)v134 + 24LL), v134, 0, 1u, 0);
      if ( v353 )
      {
        v136 = (unsigned __int64 *)v354;
        sub_157E9D0(v353 + 40, (__int64)v135);
        v137 = v135[3];
        v138 = *v136;
        v135[4] = (__int64)v136;
        v138 &= 0xFFFFFFFFFFFFFFF8LL;
        v135[3] = v138 | v137 & 7;
        *(_QWORD *)(v138 + 8) = v135 + 3;
        *v136 = *v136 & 7 | (unsigned __int64)(v135 + 3);
      }
      sub_164B780((__int64)v135, &v379);
      if ( v352 )
      {
        v366 = (char *)v352;
        sub_1623A60((__int64)&v366, (__int64)v352, 2);
        v141 = v135[6];
        v139 = (unsigned __int8 **)&v366;
        if ( v141 )
        {
          sub_161E7C0((__int64)(v135 + 6), v141);
          v139 = (unsigned __int8 **)&v366;
        }
        v142 = (unsigned __int8 *)v366;
        v135[6] = (__int64)v366;
        if ( v142 )
          sub_1623210((__int64)&v366, v142, (__int64)(v135 + 6));
      }
      v143 = 0;
      v144 = *(_QWORD *)(v331 + 8);
      v379 = (__int64)&v381;
      v380 = 0x800000000LL;
      v145 = v144;
      if ( !v144 )
      {
        LODWORD(v380) = 0;
        if ( *(_QWORD *)(v331 + 8) )
        {
LABEL_217:
          v153 = sub_1599EF0(*(__int64 ***)v331);
          v310 = v135[4];
          v313 = v135[5];
          v154 = sub_157E9C0(v313);
          v366 = 0;
          v368[1] = v154;
          v367 = v313;
          v368[2] = 0;
          v369 = 0;
          v370 = 0;
          v371 = 0;
          v368[0] = v310;
          if ( v310 != v313 + 40 )
          {
            if ( !v310 )
              BUG();
            v155 = *(unsigned __int8 **)(v310 + 24);
            v350[0] = v155;
            if ( v155 )
            {
              sub_1623A60((__int64)v350, (__int64)v155, 2);
              if ( v366 )
                sub_161E7C0((__int64)&v366, (__int64)v366);
              v366 = (char *)v350[0];
              if ( v350[0] )
                sub_1623210((__int64)v350, v350[0], (__int64)&v366);
            }
          }
          v349 = 1;
          v347[0] = "lpad.val";
          v348 = 3;
          v345 = 0;
          if ( *(_BYTE *)(v153 + 16) > 0x10u || *(_BYTE *)(v126 + 16) > 0x10u )
          {
            v351 = 257;
            v248 = sub_1648A60(88, 2u);
            v249 = (__int64)v248;
            if ( v248 )
            {
              v316 = (__int64)v248;
              v250 = (__int64)v248;
              sub_15F1EA0((__int64)v248, *(_QWORD *)v153, 63, (__int64)(v248 - 6), 2, 0);
              *(_QWORD *)(v316 + 56) = v316 + 72;
              *(_QWORD *)(v316 + 64) = 0x400000000LL;
              sub_15FAD90(v316, v153, v126, &v345, 1, (__int64)v350);
              v249 = v316;
            }
            else
            {
              v250 = 0;
            }
            if ( v367 )
            {
              v251 = (__int64 *)v368[0];
              v317 = v249;
              sub_157E9D0(v367 + 40, v249);
              v249 = v317;
              v252 = *v251;
              v253 = *(_QWORD *)(v317 + 24);
              *(_QWORD *)(v317 + 32) = v251;
              v252 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v317 + 24) = v252 | v253 & 7;
              *(_QWORD *)(v252 + 8) = v317 + 24;
              *v251 = *v251 & 7 | (v317 + 24);
            }
            v318 = v249;
            sub_164B780(v250, (__int64 *)v347);
            v156 = v318;
            if ( v366 )
            {
              v346 = (unsigned __int8 *)v366;
              sub_1623A60((__int64)&v346, (__int64)v366, 2);
              v156 = v318;
              v254 = *(_QWORD *)(v318 + 48);
              v255 = v318 + 48;
              if ( v254 )
              {
                sub_161E7C0(v318 + 48, v254);
                v156 = v318;
              }
              v256 = v346;
              *(_QWORD *)(v156 + 48) = v346;
              if ( v256 )
              {
                v319 = v156;
                sub_1623210((__int64)&v346, v256, v255);
                v156 = v319;
              }
            }
          }
          else
          {
            v156 = sub_15A3A20((__int64 *)v153, (__int64 *)v126, &v345, 1, 0);
          }
          v349 = 1;
          v347[0] = "lpad.val";
          v348 = 3;
          v345 = 1;
          if ( *(_BYTE *)(v156 + 16) > 0x10u || *((_BYTE *)v135 + 16) > 0x10u )
          {
            v315 = (__int64 *)v156;
            v351 = 257;
            v241 = sub_1648A60(88, 2u);
            v157 = v241;
            if ( v241 )
            {
              v242 = (__int64)v241;
              sub_15F1EA0((__int64)v241, *v315, 63, (__int64)(v241 - 6), 2, 0);
              v157[7] = v157 + 9;
              v157[8] = 0x400000000LL;
              sub_15FAD90((__int64)v157, (__int64)v315, (__int64)v135, &v345, 1, (__int64)v350);
            }
            else
            {
              v242 = 0;
            }
            if ( v367 )
            {
              v243 = (unsigned __int64 *)v368[0];
              sub_157E9D0(v367 + 40, (__int64)v157);
              v244 = v157[3];
              v245 = *v243;
              v157[4] = v243;
              v245 &= 0xFFFFFFFFFFFFFFF8LL;
              v157[3] = v245 | v244 & 7;
              *(_QWORD *)(v245 + 8) = v157 + 3;
              *v243 = *v243 & 7 | (unsigned __int64)(v157 + 3);
            }
            sub_164B780(v242, (__int64 *)v347);
            if ( v366 )
            {
              v346 = (unsigned __int8 *)v366;
              sub_1623A60((__int64)&v346, (__int64)v366, 2);
              v246 = v157[6];
              if ( v246 )
                sub_161E7C0((__int64)(v157 + 6), v246);
              v247 = v346;
              v157[6] = v346;
              if ( v247 )
                sub_1623210((__int64)&v346, v247, (__int64)(v157 + 6));
            }
          }
          else
          {
            v157 = (_QWORD *)sub_15A3A20((__int64 *)v156, v135, &v345, 1, 0);
          }
          sub_164D160(v331, (__int64)v157, a3, *(double *)a4.m128i_i64, a5, a6, v158, v159, a9, a10);
          if ( v366 )
            sub_161E7C0((__int64)&v366, (__int64)v366);
          goto LABEL_232;
        }
LABEL_234:
        if ( v352 )
          sub_161E7C0((__int64)&v352, (__int64)v352);
        if ( (__int64 *)v322 == ++v325 )
          goto LABEL_237;
        continue;
      }
      break;
    }
    do
    {
      v145 = *(_QWORD *)(v145 + 8);
      ++v143;
    }
    while ( v145 );
    v146 = (unsigned __int8 *)&v381;
    if ( v143 > 8 )
    {
      sub_16CD150((__int64)&v379, &v381, v143, 8, (int)v139, v140);
      v146 = (unsigned __int8 *)(v379 + 8LL * (unsigned int)v380);
    }
    do
    {
      v146 += 8;
      *((_QWORD *)v146 - 1) = sub_1648700(v144);
      v144 = *(_QWORD *)(v144 + 8);
    }
    while ( v144 );
    LODWORD(v380) = v380 + v143;
    v149 = v380;
    if ( !(_DWORD)v380 )
    {
LABEL_216:
      if ( *(_QWORD *)(v331 + 8) )
        goto LABEL_217;
LABEL_232:
      if ( (__int64 **)v379 != &v381 )
        _libc_free(v379);
      goto LABEL_234;
    }
    while ( 2 )
    {
      while ( 1 )
      {
        v150 = v149--;
        v151 = *(_QWORD *)(v379 + 8 * v150 - 8);
        LODWORD(v380) = v149;
        if ( *(_BYTE *)(v151 + 16) == 86 && *(_DWORD *)(v151 + 64) == 1 )
          break;
        if ( !v149 )
          goto LABEL_216;
      }
      v152 = **(_DWORD **)(v151 + 56);
      if ( v152 )
      {
        if ( v152 == 1 )
          sub_164D160(v151, (__int64)v135, a3, *(double *)a4.m128i_i64, a5, a6, v147, v148, a9, a10);
        if ( *(_QWORD *)(v151 + 8) )
        {
LABEL_215:
          v149 = v380;
          if ( !(_DWORD)v380 )
            goto LABEL_216;
          continue;
        }
      }
      else
      {
        sub_164D160(v151, v126, a3, *(double *)a4.m128i_i64, a5, a6, v147, v148, a9, a10);
        if ( *(_QWORD *)(v151 + 8) )
          goto LABEL_215;
      }
      break;
    }
    sub_15F20C0((_QWORD *)v151);
    goto LABEL_215;
  }
LABEL_237:
  v160 = sub_157EBA0(v308);
  v161 = (_QWORD *)sub_16498A0(v160);
  v379 = 0;
  v382 = (__int64)v161;
  v383 = 0;
  LODWORD(v384) = 0;
  v385 = 0;
  v386 = 0;
  v380 = *(_QWORD *)(v160 + 40);
  v381 = (__int64 *)(v160 + 24);
  v162 = *(unsigned __int8 **)(v160 + 48);
  v366 = (char *)v162;
  if ( v162 )
  {
    sub_1623A60((__int64)&v366, (__int64)v162, 2);
    if ( v379 )
      sub_161E7C0((__int64)&v379, v379);
    v379 = (__int64)v366;
    if ( v366 )
      sub_1623210((__int64)&v366, (unsigned __int8 *)v366, (__int64)&v379);
  }
  v163 = sub_15E38F0(a2);
  v366 = "pers_fn_gep";
  LOWORD(v368[0]) = 259;
  v164 = sub_18174F0((__int64)&v379, *(_QWORD *)(a1 + 176), *(_BYTE **)(a1 + 256), 0, 3u, (__int64 *)&v366);
  LOWORD(v354) = 257;
  v165 = (__int64 **)sub_16471D0((_QWORD *)v382, 0);
  if ( v165 != *(__int64 ***)v163 )
  {
    if ( *(_BYTE *)(v163 + 16) > 0x10u )
    {
      LOWORD(v368[0]) = 257;
      v163 = sub_15FDBD0(47, v163, (__int64)v165, (__int64)&v366, 0);
      if ( v380 )
      {
        v298 = v381;
        sub_157E9D0(v380 + 40, v163);
        v299 = *(_QWORD *)(v163 + 24);
        v300 = *v298;
        *(_QWORD *)(v163 + 32) = v298;
        v300 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v163 + 24) = v300 | v299 & 7;
        *(_QWORD *)(v300 + 8) = v163 + 24;
        *v298 = *v298 & 7 | (v163 + 24);
      }
      sub_164B780(v163, (__int64 *)&v352);
      if ( v379 )
      {
        v350[0] = (unsigned __int8 *)v379;
        sub_1623A60((__int64)v350, v379, 2);
        v301 = *(_QWORD *)(v163 + 48);
        if ( v301 )
          sub_161E7C0(v163 + 48, v301);
        v302 = v350[0];
        *(unsigned __int8 **)(v163 + 48) = v350[0];
        if ( v302 )
          sub_1623210((__int64)v350, v302, v163 + 48);
      }
    }
    else
    {
      v163 = sub_15A46C0(47, (__int64 ***)v163, v165, 0);
    }
  }
  LOWORD(v368[0]) = 257;
  v166 = sub_1648A60(64, 2u);
  v167 = v166;
  if ( v166 )
    sub_15F9650((__int64)v166, v163, v164, 1u, 0);
  if ( v380 )
  {
    v168 = (unsigned __int64 *)v381;
    sub_157E9D0(v380 + 40, (__int64)v167);
    v169 = v167[3];
    v170 = *v168;
    v167[4] = v168;
    v170 &= 0xFFFFFFFFFFFFFFF8LL;
    v167[3] = v170 | v169 & 7;
    *(_QWORD *)(v170 + 8) = v167 + 3;
    *v168 = *v168 & 7 | (unsigned __int64)(v167 + 3);
  }
  sub_164B780((__int64)v167, (__int64 *)&v366);
  if ( v379 )
  {
    v350[0] = (unsigned __int8 *)v379;
    sub_1623A60((__int64)v350, v379, 2);
    v171 = v167[6];
    if ( v171 )
      sub_161E7C0((__int64)(v167 + 6), v171);
    v172 = v350[0];
    v167[6] = v350[0];
    if ( v172 )
      sub_1623210((__int64)v350, v172, (__int64)(v167 + 6));
  }
  v173 = *(_QWORD *)(a1 + 232);
  v366 = "lsda_addr";
  LOWORD(v368[0]) = 259;
  v174 = sub_1285290(&v379, *(_QWORD *)(*(_QWORD *)v173 + 24LL), v173, 0, 0, (__int64)&v366, 0);
  v175 = *(_BYTE **)(a1 + 256);
  v176 = *(_QWORD *)(a1 + 176);
  v177 = v174;
  v366 = "lsda_gep";
  LOWORD(v368[0]) = 259;
  v178 = sub_18174F0((__int64)&v379, v176, v175, 0, 4u, (__int64 *)&v366);
  LOWORD(v368[0]) = 257;
  v179 = sub_1648A60(64, 2u);
  if ( v179 )
    sub_15F9650((__int64)v179, v177, v178, 1u, 0);
  if ( v380 )
  {
    v180 = (unsigned __int64 *)v381;
    sub_157E9D0(v380 + 40, (__int64)v179);
    v181 = v179[3];
    v182 = *v180;
    v179[4] = v180;
    v182 &= 0xFFFFFFFFFFFFFFF8LL;
    v179[3] = v182 | v181 & 7;
    *(_QWORD *)(v182 + 8) = v179 + 3;
    *v180 = *v180 & 7 | (unsigned __int64)(v179 + 3);
  }
  sub_164B780((__int64)v179, (__int64 *)&v366);
  if ( v379 )
  {
    v352 = (unsigned __int8 *)v379;
    sub_1623A60((__int64)&v352, v379, 2);
    v183 = v179[6];
    if ( v183 )
      sub_161E7C0((__int64)(v179 + 6), v183);
    v184 = v352;
    v179[6] = v352;
    if ( v184 )
      sub_1623210((__int64)&v352, v184, (__int64)(v179 + 6));
    v185 = *(_BYTE **)(a1 + 256);
    if ( v379 )
      sub_161E7C0((__int64)&v379, v379);
  }
  else
  {
    v185 = *(_BYTE **)(a1 + 256);
  }
  v347[0] = v185;
  v186 = *(_QWORD *)(a2 + 80);
  v187 = v186 - 24;
  if ( !v186 )
    v187 = 0;
  v314 = v187;
  v188 = sub_157EBA0(v187);
  v189 = (_QWORD *)sub_16498A0(v188);
  v379 = 0;
  v382 = (__int64)v189;
  v383 = 0;
  LODWORD(v384) = 0;
  v385 = 0;
  v386 = 0;
  v380 = *(_QWORD *)(v188 + 40);
  v381 = (__int64 *)(v188 + 24);
  v190 = *(unsigned __int8 **)(v188 + 48);
  v366 = (char *)v190;
  if ( v190 )
  {
    sub_1623A60((__int64)&v366, (__int64)v190, 2);
    if ( v379 )
      sub_161E7C0((__int64)&v379, v379);
    v379 = (__int64)v366;
    if ( v366 )
      sub_1623210((__int64)&v366, (unsigned __int8 *)v366, (__int64)&v379);
  }
  v191 = *(_QWORD *)(a1 + 176);
  v366 = "jbuf_gep";
  LOWORD(v368[0]) = 259;
  v192 = (_BYTE *)sub_18174F0((__int64)&v379, v191, v347[0], 0, 5u, (__int64 *)&v366);
  v193 = *(_QWORD *)(a1 + 168);
  v194 = v192;
  v366 = "jbuf_fp_gep";
  LOWORD(v368[0]) = 259;
  v195 = sub_18174F0((__int64)&v379, v193, v192, 0, 0, (__int64 *)&v366);
  LOWORD(v368[0]) = 259;
  v366 = "fp";
  v196 = sub_1643350((_QWORD *)v382);
  v197 = (unsigned __int8 *)sub_159C470(v196, 0, 0);
  v198 = *(_QWORD *)(a1 + 208);
  v352 = v197;
  v332 = sub_1285290(&v379, *(_QWORD *)(*(_QWORD *)v198 + 24LL), v198, (int)&v352, 1, (__int64)&v366, 0);
  LOWORD(v368[0]) = 257;
  v199 = sub_1648A60(64, 2u);
  v200 = v199;
  if ( v199 )
    sub_15F9650((__int64)v199, v332, v195, 1u, 0);
  if ( v380 )
  {
    v201 = (unsigned __int64 *)v381;
    sub_157E9D0(v380 + 40, (__int64)v200);
    v202 = v200[3];
    v203 = *v201;
    v200[4] = v201;
    v203 &= 0xFFFFFFFFFFFFFFF8LL;
    v200[3] = v203 | v202 & 7;
    *(_QWORD *)(v203 + 8) = v200 + 3;
    *v201 = *v201 & 7 | (unsigned __int64)(v200 + 3);
  }
  sub_164B780((__int64)v200, (__int64 *)&v366);
  if ( v379 )
  {
    v352 = (unsigned __int8 *)v379;
    sub_1623A60((__int64)&v352, v379, 2);
    v204 = v200[6];
    if ( v204 )
      sub_161E7C0((__int64)(v200 + 6), v204);
    v205 = v352;
    v200[6] = v352;
    if ( v205 )
      sub_1623210((__int64)&v352, v205, (__int64)(v200 + 6));
  }
  v206 = *(_QWORD *)(a1 + 168);
  v366 = "jbuf_sp_gep";
  LOWORD(v368[0]) = 259;
  v207 = sub_18174F0((__int64)&v379, v206, v194, 0, 2u, (__int64 *)&v366);
  v208 = *(_QWORD *)(a1 + 216);
  v366 = "sp";
  LOWORD(v368[0]) = 259;
  v333 = v207;
  v209 = sub_1285290(&v379, *(_QWORD *)(*(_QWORD *)v208 + 24LL), v208, 0, 0, (__int64)&v366, 0);
  LOWORD(v368[0]) = 257;
  v210 = v209;
  v211 = sub_1648A60(64, 2u);
  if ( v211 )
    sub_15F9650((__int64)v211, v210, v333, 1u, 0);
  if ( v380 )
  {
    v212 = (unsigned __int64 *)v381;
    sub_157E9D0(v380 + 40, (__int64)v211);
    v213 = v211[3];
    v214 = *v212;
    v211[4] = v212;
    v214 &= 0xFFFFFFFFFFFFFFF8LL;
    v211[3] = v214 | v213 & 7;
    *(_QWORD *)(v214 + 8) = v211 + 3;
    *v212 = *v212 & 7 | (unsigned __int64)(v211 + 3);
  }
  sub_164B780((__int64)v211, (__int64 *)&v366);
  if ( v379 )
  {
    v352 = (unsigned __int8 *)v379;
    sub_1623A60((__int64)&v352, v379, 2);
    v215 = v211[6];
    if ( v215 )
      sub_161E7C0((__int64)(v211 + 6), v215);
    v216 = v352;
    v211[6] = v352;
    if ( v216 )
      sub_1623210((__int64)&v352, v216, (__int64)(v211 + 6));
  }
  v217 = *(_QWORD *)(a1 + 200);
  LOWORD(v368[0]) = 257;
  sub_1285290(&v379, *(_QWORD *)(*(_QWORD *)v217 + 24LL), v217, 0, 0, (__int64)&v366, 0);
  LOWORD(v354) = 257;
  v218 = (__int64 **)sub_16471D0((_QWORD *)v382, 0);
  v219 = v347[0];
  if ( v218 != *(__int64 ***)v347[0] )
  {
    if ( v347[0][16] > 0x10u )
    {
      LOWORD(v368[0]) = 257;
      v219 = (_BYTE *)sub_15FDBD0(47, (__int64)v347[0], (__int64)v218, (__int64)&v366, 0);
      if ( v380 )
      {
        v293 = (unsigned __int64 *)v381;
        sub_157E9D0(v380 + 40, (__int64)v219);
        v294 = *((_QWORD *)v219 + 3);
        v295 = *v293;
        *((_QWORD *)v219 + 4) = v293;
        v295 &= 0xFFFFFFFFFFFFFFF8LL;
        *((_QWORD *)v219 + 3) = v295 | v294 & 7;
        *(_QWORD *)(v295 + 8) = v219 + 24;
        *v293 = *v293 & 7 | (unsigned __int64)(v219 + 24);
      }
      sub_164B780((__int64)v219, (__int64 *)&v352);
      if ( v379 )
      {
        v350[0] = (unsigned __int8 *)v379;
        sub_1623A60((__int64)v350, v379, 2);
        v296 = *((_QWORD *)v219 + 6);
        if ( v296 )
          sub_161E7C0((__int64)(v219 + 48), v296);
        v297 = v350[0];
        *((unsigned __int8 **)v219 + 6) = v350[0];
        if ( v297 )
          sub_1623210((__int64)v350, v297, (__int64)(v219 + 48));
      }
    }
    else
    {
      v219 = (_BYTE *)sub_15A46C0(47, (__int64 ***)v347[0], v218, 0);
    }
  }
  v350[0] = v219;
  LOWORD(v368[0]) = 257;
  sub_1285290(
    &v379,
    *(_QWORD *)(**(_QWORD **)(a1 + 248) + 24LL),
    *(_QWORD *)(a1 + 248),
    (int)v350,
    1,
    (__int64)&v366,
    0);
  if ( (_DWORD)v364 )
  {
    v220 = 1;
    v323 = (unsigned int)(v364 - 1) + 2LL;
    do
    {
      sub_2116330(a1, *(_QWORD *)&v363[8 * v220 - 8], v220);
      v221 = (_QWORD *)sub_15E0530(a2);
      v222 = sub_1643350(v221);
      v223 = (unsigned __int8 *)sub_159C470(v222, v220, 0);
      v224 = *(_QWORD *)&v363[8 * v220 - 8];
      v352 = v223;
      LOWORD(v368[0]) = 257;
      v225 = *(_QWORD *)(a1 + 240);
      v326 = v224;
      v226 = *(_QWORD *)(*(_QWORD *)v225 + 24LL);
      v227 = sub_1648AB0(72, 2u, 0);
      v228 = (__int64)v227;
      if ( v227 )
      {
        sub_15F1EA0((__int64)v227, **(_QWORD **)(v226 + 16), 54, (__int64)(v227 - 6), 2, v326);
        *(_QWORD *)(v228 + 56) = 0;
        sub_15F5B40(v228, v226, v225, (__int64 *)&v352, 1, (__int64)&v366, 0, 0);
      }
      ++v220;
    }
    while ( v323 != v220 );
  }
  v229 = *(_QWORD *)(a2 + 80);
  v230 = v229;
  if ( v311 != v229 )
  {
    if ( !v229 )
      goto LABEL_311;
LABEL_300:
    if ( !v230 || v230 != v229 )
    {
      v231 = *(_QWORD *)(v229 + 24);
      for ( j = v229 + 16; j != v231; v231 = *(_QWORD *)(v231 + 8) )
      {
        while ( 1 )
        {
          v233 = v231 - 24;
          if ( !v231 )
            v233 = 0;
          if ( sub_15F3330(v233) )
            break;
          v231 = *(_QWORD *)(v231 + 8);
          if ( j == v231 )
            goto LABEL_309;
        }
        sub_2116330(a1, v233, -1);
      }
    }
LABEL_309:
    while ( 1 )
    {
      v229 = *(_QWORD *)(v229 + 8);
      if ( v311 == v229 )
        break;
      v230 = *(_QWORD *)(a2 + 80);
      if ( v229 )
        goto LABEL_300;
LABEL_311:
      if ( v230 )
        BUG();
    }
  }
  v262 = sub_157EBA0(v314);
  LOWORD(v368[0]) = 257;
  v338 = v262;
  v263 = *(_QWORD *)(a1 + 184);
  v264 = *(_QWORD *)(*(_QWORD *)v263 + 24LL);
  v265 = sub_1648AB0(72, 2u, 0);
  v266 = v338;
  v267 = (__int64)v265;
  if ( v265 )
  {
    v339 = v265;
    sub_15F1EA0((__int64)v265, **(_QWORD **)(v264 + 16), 54, (__int64)(v265 - 6), 2, v266);
    *(_QWORD *)(v267 + 56) = 0;
    sub_15F5B40(v267, v264, v263, (__int64 *)v347, 1, (__int64)&v366, 0, 0);
    v268 = (__int64)v339;
  }
  else
  {
    v268 = 0;
  }
  v366 = *(char **)(v267 + 56);
  v269 = (__int64 *)sub_16498A0(v268);
  v366 = (char *)sub_1563AB0((__int64 *)&v366, v269, -1, 30);
  *(_QWORD *)(v267 + 56) = v366;
  v270 = *(_QWORD *)(a2 + 80);
  if ( v311 != v270 )
  {
    v271 = *(_QWORD *)(a2 + 80);
    if ( !v270 )
      goto LABEL_386;
    while ( 1 )
    {
      if ( !v271 || v271 != v270 )
      {
        v272 = v270 + 16;
        if ( v270 + 16 != *(_QWORD *)(v270 + 24) )
          break;
      }
LABEL_384:
      v270 = *(_QWORD *)(v270 + 8);
      if ( v311 == v270 )
        goto LABEL_388;
      while ( 1 )
      {
        v271 = *(_QWORD *)(a2 + 80);
        if ( v270 )
          break;
LABEL_386:
        if ( v271 )
          BUG();
        v270 = *(_QWORD *)(v270 + 8);
        if ( v311 == v270 )
          goto LABEL_388;
      }
    }
    v327 = v270;
    v273 = *(_QWORD *)(v270 + 24);
    while ( 1 )
    {
      if ( !v273 )
        BUG();
      v275 = *(_BYTE *)(v273 - 8);
      if ( v275 == 78 )
      {
        v274 = *(_QWORD *)(v273 - 48);
        if ( *(_BYTE *)(v274 + 16) )
          v274 = 0;
        if ( *(_QWORD *)(a1 + 224) != v274 )
        {
LABEL_374:
          v273 = *(_QWORD *)(v273 + 8);
          if ( v272 == v273 )
            goto LABEL_383;
          continue;
        }
      }
      else if ( v275 != 53 )
      {
        goto LABEL_374;
      }
      v366 = "sp";
      v276 = *(_QWORD **)(a1 + 216);
      LOWORD(v368[0]) = 259;
      v277 = sub_1648A60(72, 1u);
      v278 = (__int64)v277;
      if ( v277 )
        sub_15F5ED0((__int64)v277, v276, (__int64)&v366, 0);
      sub_15F2180(v278, v273 - 24);
      v279 = sub_1648A60(64, 2u);
      v280 = (__int64)v279;
      if ( v279 )
        sub_15F9650((__int64)v279, v278, v333, 1u, 0);
      sub_15F2180(v280, v278);
      v273 = *(_QWORD *)(v273 + 8);
      if ( v272 == v273 )
      {
LABEL_383:
        v270 = v327;
        goto LABEL_384;
      }
    }
  }
LABEL_388:
  v281 = v360;
  v334 = &v360[(unsigned int)v361];
  if ( v334 != v360 )
  {
    do
    {
      v340 = *v281;
      v282 = *(_QWORD *)(a1 + 192);
      LOWORD(v368[0]) = 257;
      v283 = *(_QWORD *)(*(_QWORD *)v282 + 24LL);
      v284 = sub_1648AB0(72, 2u, 0);
      v285 = (__int64)v284;
      if ( v284 )
      {
        sub_15F1EA0((__int64)v284, **(_QWORD **)(v283 + 16), 54, (__int64)(v284 - 6), 2, v340);
        *(_QWORD *)(v285 + 56) = 0;
        sub_15F5B40(v285, v283, v282, (__int64 *)v347, 1, (__int64)&v366, 0, 0);
      }
      ++v281;
    }
    while ( v334 != v281 );
  }
  if ( v379 )
    sub_161E7C0((__int64)&v379, v379);
  v31 = 1;
LABEL_21:
  if ( v376 != v378 )
    _libc_free((unsigned __int64)v376);
LABEL_23:
  if ( (v373 & 1) == 0 )
    j___libc_free_0(v374);
  if ( v363 != v365 )
    _libc_free((unsigned __int64)v363);
  if ( v360 != (__int64 *)v362 )
    _libc_free((unsigned __int64)v360);
  return v31;
}
