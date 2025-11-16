// Function: sub_394CF20
// Address: 0x394cf20
//
__int64 __fastcall sub_394CF20(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned __int64 v10; // rdi
  _QWORD *i; // r15
  __int64 v12; // rbx
  _QWORD *v13; // rbx
  int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // rdi
  int v17; // ecx
  unsigned int v18; // esi
  int *v19; // rax
  int v20; // r8d
  unsigned int v21; // r12d
  _QWORD *v22; // rax
  __int64 **v23; // rax
  double v24; // xmm4_8
  double v25; // xmm5_8
  __int64 *v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r11
  char v29; // dl
  unsigned __int8 v30; // r12
  int v31; // r13d
  unsigned int j; // esi
  unsigned __int64 v33; // rcx
  __int64 v34; // r10
  unsigned int v35; // esi
  __int64 v36; // rsi
  __int64 v37; // rax
  int v39; // eax
  char v40; // dl
  _QWORD *v41; // rdi
  __int64 v42; // rcx
  __int64 v43; // rax
  unsigned __int8 *v44; // rsi
  __int64 *v45; // rax
  __int64 v46; // rcx
  __int64 v47; // rax
  __int64 **v48; // rdx
  __int64 v49; // rax
  __int64 ***v50; // rax
  __int64 **v51; // rdx
  __int64 **v52; // rdx
  _QWORD *v53; // rax
  __int64 v54; // rcx
  __int64 v55; // rax
  __int64 v56; // rdi
  __int64 v57; // rcx
  __int64 v58; // rsi
  __int64 v59; // rdx
  unsigned __int8 *v60; // rsi
  __int64 v61; // r13
  __int64 v62; // rax
  unsigned __int8 *v63; // rsi
  __int64 v64; // r13
  __int64 v65; // rsi
  __int64 v66; // rax
  __int64 v67; // rdx
  unsigned __int8 **v68; // r8
  __int64 v69; // r9
  unsigned __int8 *v70; // rsi
  __int64 v71; // rsi
  int v72; // eax
  __int64 v73; // rax
  int v74; // edx
  __int64 v75; // rdx
  __int64 *v76; // rax
  __int64 v77; // rcx
  unsigned __int64 v78; // rsi
  __int64 v79; // rcx
  __int64 v80; // rdx
  __int64 v81; // rax
  __int64 v82; // rcx
  __int64 v83; // rdx
  __int64 v84; // rcx
  int v85; // eax
  __int64 v86; // rax
  int v87; // edx
  __int64 v88; // rdx
  __int64 *v89; // rax
  __int64 v90; // rsi
  unsigned __int64 v91; // rdi
  __int64 v92; // rsi
  __int64 v93; // rdx
  __int64 v94; // r13
  __int64 v95; // rsi
  __int64 v96; // r13
  __int64 v97; // rsi
  __int64 v98; // rax
  __int64 v99; // rdx
  unsigned __int8 **v100; // r8
  __int64 v101; // r9
  unsigned __int8 *v102; // rsi
  __int64 v103; // rsi
  int v104; // eax
  __int64 v105; // rax
  __int64 v106; // rdx
  __int64 *v107; // rax
  __int64 v108; // rcx
  unsigned __int64 v109; // rsi
  __int64 v110; // rcx
  __int64 v111; // rdx
  __int64 v112; // rax
  __int64 v113; // rcx
  __int64 v114; // rdx
  __int64 v115; // rcx
  int v116; // eax
  __int64 v117; // rax
  int v118; // edx
  __int64 v119; // rdx
  __int64 *v120; // rax
  __int64 v121; // rsi
  unsigned __int64 v122; // rdi
  __int64 v123; // rsi
  __int64 v124; // rdx
  __int64 v125; // r13
  unsigned __int8 *v126; // rsi
  __int64 v127; // r9
  __int64 v128; // r11
  bool v129; // r13
  __int64 v130; // rax
  __int64 v131; // rcx
  __int64 *v132; // r11
  __int64 v133; // r9
  unsigned __int64 v134; // rsi
  __int64 v135; // rax
  __int64 v136; // r11
  __int64 v137; // rdx
  unsigned __int8 v138; // al
  __int64 v139; // rax
  __int64 v140; // rax
  unsigned __int8 *v141; // rsi
  _QWORD *v142; // r13
  __int64 v143; // rax
  _QWORD *v144; // rax
  __int64 v145; // r9
  __int64 v146; // rdx
  __int64 *v147; // r13
  __int64 v148; // rcx
  __int64 v149; // rax
  __int64 v150; // r9
  __int64 v151; // rsi
  unsigned __int8 *v152; // rsi
  char v153; // r13
  unsigned int v154; // esi
  unsigned __int64 v155; // r9
  int v156; // r10d
  int v157; // r13d
  unsigned int v158; // ecx
  unsigned __int64 v159; // rax
  __int64 v160; // r11
  _QWORD *v161; // r12
  _QWORD *v162; // rax
  _QWORD *v163; // rbx
  __int64 v164; // r13
  __int64 v165; // rdi
  int v166; // eax
  int v167; // r10d
  __int64 v168; // r13
  __int64 v169; // rax
  unsigned __int8 *v170; // rsi
  __int64 **v171; // rdx
  __int64 v172; // rax
  _QWORD *v173; // r13
  __int64 v174; // r9
  __int64 ***v175; // rax
  __int64 **v176; // rdx
  __int64 **v177; // rdx
  _QWORD *v178; // rax
  __int64 v179; // r11
  _QWORD **v180; // rax
  __int64 *v181; // rax
  __int64 v182; // rax
  __int64 v183; // r11
  __int64 v184; // r9
  unsigned __int64 v185; // rsi
  __int64 v186; // rax
  __int64 v187; // rsi
  __int64 v188; // rdx
  unsigned __int8 *v189; // rsi
  __int64 v190; // rsi
  __int64 v191; // rax
  __int64 v192; // rsi
  __int64 v193; // rdx
  unsigned __int8 *v194; // rsi
  __int64 v195; // rsi
  __int64 v196; // rax
  __int64 v197; // rsi
  __int64 v198; // rdx
  unsigned __int8 *v199; // rsi
  __int64 v200; // r11
  __int64 *v201; // r13
  __int64 v202; // rcx
  __int64 v203; // rax
  __int64 v204; // rsi
  __int64 v205; // r13
  unsigned __int8 *v206; // rsi
  bool v207; // al
  __int64 v208; // r11
  __int64 *v209; // r13
  __int64 v210; // rcx
  __int64 v211; // rax
  __int64 v212; // rsi
  __int64 v213; // r13
  unsigned __int8 *v214; // rsi
  __int64 v215; // rsi
  __int64 v216; // rax
  __int64 v217; // rsi
  __int64 v218; // rdx
  unsigned __int8 *v219; // rsi
  __int64 v220; // rsi
  __int64 v221; // rax
  __int64 v222; // rsi
  __int64 v223; // rdx
  unsigned __int8 *v224; // rsi
  unsigned int v225; // esi
  int v226; // eax
  __int64 v227; // rsi
  __int64 v228; // rax
  __int64 v229; // rsi
  __int64 v230; // rdx
  unsigned __int8 *v231; // rsi
  __int64 v232; // rsi
  __int64 v233; // rax
  __int64 v234; // rsi
  __int64 v235; // rdx
  unsigned __int8 *v236; // rsi
  unsigned __int64 v237; // rsi
  int v238; // ecx
  unsigned int v239; // edx
  __int64 v240; // r9
  unsigned int v241; // edx
  unsigned __int64 v242; // rsi
  __int64 v243; // rax
  __int64 v244; // rsi
  __int64 v245; // rdx
  unsigned __int8 *v246; // rsi
  __int64 *v247; // r13
  __int64 v248; // rcx
  __int64 v249; // rax
  __int64 v250; // r13
  __int64 v251; // rsi
  unsigned __int8 *v252; // rsi
  __int64 *v253; // r13
  __int64 v254; // rcx
  __int64 v255; // rax
  __int64 v256; // r13
  __int64 v257; // rsi
  unsigned __int8 *v258; // rsi
  __int64 v259; // r9
  __int64 v260; // rax
  __int64 v261; // rsi
  __int64 v262; // rsi
  __int64 v263; // rdx
  unsigned __int8 *v264; // rsi
  unsigned int v265; // ecx
  int v266; // edx
  int v267; // esi
  unsigned int k; // r13d
  unsigned __int64 v269; // rcx
  __int64 v270; // rdi
  unsigned int v271; // r13d
  __int64 v272; // [rsp+0h] [rbp-250h]
  int v273; // [rsp+8h] [rbp-248h]
  __int64 v274; // [rsp+8h] [rbp-248h]
  _QWORD *v275; // [rsp+10h] [rbp-240h]
  __int64 v276; // [rsp+18h] [rbp-238h]
  __int64 *v277; // [rsp+18h] [rbp-238h]
  int v278; // [rsp+18h] [rbp-238h]
  __int64 v279; // [rsp+18h] [rbp-238h]
  __int64 v280; // [rsp+18h] [rbp-238h]
  __int64 v281; // [rsp+18h] [rbp-238h]
  int v282; // [rsp+18h] [rbp-238h]
  __int64 v283; // [rsp+18h] [rbp-238h]
  __int64 v284; // [rsp+18h] [rbp-238h]
  __int64 v285; // [rsp+20h] [rbp-230h]
  __int64 *v286; // [rsp+20h] [rbp-230h]
  __int64 v287; // [rsp+30h] [rbp-220h]
  __int64 v288; // [rsp+30h] [rbp-220h]
  __int64 v289; // [rsp+30h] [rbp-220h]
  _QWORD *v290; // [rsp+30h] [rbp-220h]
  unsigned __int64 *v291; // [rsp+30h] [rbp-220h]
  __int64 *v292; // [rsp+30h] [rbp-220h]
  __int64 v293; // [rsp+30h] [rbp-220h]
  __int64 v294; // [rsp+30h] [rbp-220h]
  int v295; // [rsp+38h] [rbp-218h]
  _QWORD *v296; // [rsp+38h] [rbp-218h]
  __int64 *v297; // [rsp+38h] [rbp-218h]
  __int64 v298; // [rsp+38h] [rbp-218h]
  _QWORD *v299; // [rsp+38h] [rbp-218h]
  __int64 v300; // [rsp+38h] [rbp-218h]
  __int64 v301; // [rsp+38h] [rbp-218h]
  __int64 v302; // [rsp+38h] [rbp-218h]
  __int64 v303; // [rsp+38h] [rbp-218h]
  __int64 v304; // [rsp+38h] [rbp-218h]
  __int64 *v305; // [rsp+38h] [rbp-218h]
  __int64 *v306; // [rsp+38h] [rbp-218h]
  __int64 v307; // [rsp+38h] [rbp-218h]
  __int64 v308; // [rsp+40h] [rbp-210h]
  __int64 *v309; // [rsp+40h] [rbp-210h]
  unsigned __int8 *v310; // [rsp+40h] [rbp-210h]
  _QWORD *v311; // [rsp+40h] [rbp-210h]
  __int64 v312; // [rsp+40h] [rbp-210h]
  __int64 v313; // [rsp+48h] [rbp-208h]
  __int64 v315; // [rsp+58h] [rbp-1F8h]
  __int64 v316; // [rsp+58h] [rbp-1F8h]
  __int64 v317; // [rsp+58h] [rbp-1F8h]
  _QWORD *v318; // [rsp+58h] [rbp-1F8h]
  __int64 v319; // [rsp+58h] [rbp-1F8h]
  __int64 v320; // [rsp+58h] [rbp-1F8h]
  __int64 v321; // [rsp+58h] [rbp-1F8h]
  __int64 *v322; // [rsp+58h] [rbp-1F8h]
  __int64 *v323; // [rsp+58h] [rbp-1F8h]
  __int64 *v324; // [rsp+58h] [rbp-1F8h]
  __int64 *v325; // [rsp+58h] [rbp-1F8h]
  __int64 v326; // [rsp+58h] [rbp-1F8h]
  __int64 v327; // [rsp+60h] [rbp-1F0h]
  __int64 v328; // [rsp+60h] [rbp-1F0h]
  __int64 v329; // [rsp+60h] [rbp-1F0h]
  __int64 v330; // [rsp+60h] [rbp-1F0h]
  __int64 v331; // [rsp+60h] [rbp-1F0h]
  __int64 *v332; // [rsp+60h] [rbp-1F0h]
  __int64 *v333; // [rsp+60h] [rbp-1F0h]
  unsigned __int64 *v334; // [rsp+60h] [rbp-1F0h]
  __int64 v335; // [rsp+60h] [rbp-1F0h]
  __int64 v336; // [rsp+60h] [rbp-1F0h]
  __int64 v337; // [rsp+60h] [rbp-1F0h]
  __int64 v338; // [rsp+60h] [rbp-1F0h]
  unsigned __int8 v339; // [rsp+6Fh] [rbp-1E1h]
  __int64 v340; // [rsp+70h] [rbp-1E0h]
  __int64 v341; // [rsp+78h] [rbp-1D8h]
  unsigned __int8 *v342; // [rsp+88h] [rbp-1C8h] BYREF
  unsigned __int8 *v343; // [rsp+90h] [rbp-1C0h] BYREF
  __int64 v344; // [rsp+98h] [rbp-1B8h]
  __int64 v345; // [rsp+A0h] [rbp-1B0h]
  __int64 v346[2]; // [rsp+B0h] [rbp-1A0h] BYREF
  __int16 v347; // [rsp+C0h] [rbp-190h]
  unsigned __int8 *v348[2]; // [rsp+D0h] [rbp-180h] BYREF
  __int16 v349; // [rsp+E0h] [rbp-170h]
  __int64 v350; // [rsp+F0h] [rbp-160h] BYREF
  unsigned __int64 v351; // [rsp+F8h] [rbp-158h]
  __int64 v352; // [rsp+100h] [rbp-150h]
  unsigned int v353; // [rsp+108h] [rbp-148h]
  char v354[8]; // [rsp+110h] [rbp-140h] BYREF
  _QWORD *v355; // [rsp+118h] [rbp-138h]
  __int64 **v356; // [rsp+120h] [rbp-130h]
  _QWORD *v357; // [rsp+128h] [rbp-128h]
  __int64 v358; // [rsp+130h] [rbp-120h] BYREF
  _BYTE *v359; // [rsp+138h] [rbp-118h]
  _BYTE *v360; // [rsp+140h] [rbp-110h]
  __int64 v361; // [rsp+148h] [rbp-108h]
  int v362; // [rsp+150h] [rbp-100h]
  _BYTE v363[40]; // [rsp+158h] [rbp-F8h] BYREF
  __int64 v364; // [rsp+180h] [rbp-D0h] BYREF
  _BYTE *v365; // [rsp+188h] [rbp-C8h]
  _BYTE *v366; // [rsp+190h] [rbp-C0h]
  __int64 v367; // [rsp+198h] [rbp-B8h]
  int v368; // [rsp+1A0h] [rbp-B0h]
  _BYTE v369[40]; // [rsp+1A8h] [rbp-A8h] BYREF
  unsigned __int8 *v370; // [rsp+1D0h] [rbp-80h] BYREF
  _QWORD *v371; // [rsp+1D8h] [rbp-78h]
  unsigned __int64 *v372; // [rsp+1E0h] [rbp-70h]
  __int64 v373; // [rsp+1E8h] [rbp-68h]
  __int64 v374; // [rsp+1F0h] [rbp-60h]
  int v375; // [rsp+1F8h] [rbp-58h]
  __int64 v376; // [rsp+200h] [rbp-50h]
  __int64 v377; // [rsp+208h] [rbp-48h]

  v10 = *(_QWORD *)(a1 + 48);
  v350 = 0;
  v351 = 0;
  v352 = 0;
  v353 = 0;
  if ( !v10 )
  {
    v339 = 0;
    goto LABEL_33;
  }
  v339 = 0;
  for ( i = (_QWORD *)(v10 - 24); ; i = v13 )
  {
    v12 = i[4];
    if ( v12 == i[5] + 40LL || !v12 )
      v13 = 0;
    else
      v13 = (_QWORD *)(v12 - 24);
    v356 = 0;
    v357 = 0;
    v14 = *((unsigned __int8 *)i + 16);
    v354[0] = 0;
    if ( (unsigned int)(v14 - 24) <= 0x12 )
    {
      if ( (unsigned int)(v14 - 24) <= 0x10 )
        goto LABEL_4;
    }
    else if ( (unsigned int)(v14 - 44) > 1 )
    {
      goto LABEL_4;
    }
    v355 = i;
    if ( *(_BYTE *)(*i + 8LL) == 11 )
    {
      v15 = *(unsigned int *)(a2 + 24);
      if ( (_DWORD)v15 )
      {
        v16 = *(_QWORD *)(a2 + 8);
        v17 = *(_DWORD *)(*i + 8LL) >> 8;
        v18 = (v15 - 1) & (37 * v17);
        v19 = (int *)(v16 + 8LL * v18);
        v20 = *v19;
        if ( v17 != *v19 )
        {
          v166 = 1;
          while ( v20 != -1 )
          {
            v167 = v166 + 1;
            v18 = (v15 - 1) & (v166 + v18);
            v19 = (int *)(v16 + 8LL * v18);
            v20 = *v19;
            if ( v17 == *v19 )
              goto LABEL_14;
            v166 = v167;
          }
          goto LABEL_4;
        }
LABEL_14:
        if ( v19 != (int *)(v16 + 8 * v15) )
          break;
      }
    }
LABEL_4:
    if ( !v13 )
      goto LABEL_32;
LABEL_5:
    ;
  }
  v21 = v19[1];
  v22 = (_QWORD *)sub_16498A0((__int64)i);
  v23 = (__int64 **)sub_1644900(v22, v21);
  v354[0] = 1;
  v356 = v23;
  v357 = (_QWORD *)i[5];
  if ( (*((_BYTE *)v355 + 23) & 0x40) != 0 )
    v26 = (__int64 *)*(v355 - 1);
  else
    v26 = &v355[-3 * (*((_DWORD *)v355 + 5) & 0xFFFFFFF)];
  v27 = *v26;
  v28 = v26[3];
  v29 = *((_BYTE *)v355 + 16);
  v341 = v27;
  v340 = v28;
  v30 = v29 == 45 || v29 == 42;
  if ( v353 )
  {
    v31 = 1;
    for ( j = (v353 - 1) & (v30 ^ v28 ^ v27); ; j = (v353 - 1) & v35 )
    {
      v33 = v351 + 40LL * j;
      v34 = *(_QWORD *)(v33 + 8);
      if ( *(_BYTE *)v33 == v30 && v341 == v34 && v28 == *(_QWORD *)(v33 + 16) )
        break;
      if ( !*(_BYTE *)v33 && !v34 && !*(_QWORD *)(v33 + 16) )
        goto LABEL_36;
      v35 = v31 + j;
      ++v31;
    }
    if ( v33 != v351 + 40LL * v353 )
    {
      v36 = *(_QWORD *)(v33 + 32);
      v37 = *(_QWORD *)(v33 + 24);
      goto LABEL_28;
    }
  }
LABEL_36:
  v358 = 0;
  v359 = v363;
  v360 = v363;
  v361 = 4;
  v362 = 0;
  v295 = sub_394C0B0((__int64)v354, v341, (__int64)&v358);
  if ( v295 == 2 )
  {
    if ( v359 != v360 )
      _libc_free((unsigned __int64)v360);
    goto LABEL_4;
  }
  v364 = 0;
  v365 = v369;
  v366 = v369;
  v367 = 4;
  v368 = 0;
  v39 = sub_394C0B0((__int64)v354, v340, (__int64)&v364);
  v273 = v39;
  if ( v39 == 2 )
    goto LABEL_194;
  if ( !(v39 | v295) )
  {
    v168 = (__int64)v355;
    v169 = sub_16498A0((__int64)v355);
    v370 = 0;
    v373 = v169;
    v374 = 0;
    v375 = 0;
    v376 = 0;
    v377 = 0;
    v371 = *(_QWORD **)(v168 + 40);
    v372 = (unsigned __int64 *)(v168 + 24);
    v170 = *(unsigned __int8 **)(v168 + 48);
    v348[0] = v170;
    if ( v170 )
    {
      sub_1623A60((__int64)v348, (__int64)v170, 2);
      if ( v370 )
        sub_161E7C0((__int64)&v370, (__int64)v370);
      v370 = v348[0];
      if ( v348[0] )
        sub_1623210((__int64)v348, v348[0], (__int64)&v370);
    }
    v347 = 257;
    v171 = *(__int64 ***)v341;
    if ( v356 == *(__int64 ***)v341 )
    {
      v173 = (_QWORD *)v341;
    }
    else if ( *(_BYTE *)(v341 + 16) > 0x10u )
    {
      v349 = 257;
      v173 = (_QWORD *)sub_15FDBD0(36, v341, (__int64)v356, (__int64)v348, 0);
      if ( v371 )
      {
        v334 = v372;
        sub_157E9D0((__int64)(v371 + 5), (__int64)v173);
        v242 = *v334;
        v243 = v173[3] & 7LL;
        v173[4] = v334;
        v242 &= 0xFFFFFFFFFFFFFFF8LL;
        v173[3] = v242 | v243;
        *(_QWORD *)(v242 + 8) = v173 + 3;
        *v334 = *v334 & 7 | (unsigned __int64)(v173 + 3);
      }
      sub_164B780((__int64)v173, v346);
      if ( v370 )
      {
        v343 = v370;
        sub_1623A60((__int64)&v343, (__int64)v370, 2);
        v244 = v173[6];
        v245 = (__int64)(v173 + 6);
        if ( v244 )
        {
          sub_161E7C0((__int64)(v173 + 6), v244);
          v245 = (__int64)(v173 + 6);
        }
        v246 = v343;
        v173[6] = v343;
        if ( v246 )
          sub_1623210((__int64)&v343, v246, v245);
      }
      v171 = v356;
    }
    else
    {
      v172 = sub_15A46C0(36, (__int64 ***)v341, v356, 0);
      v171 = v356;
      v173 = (_QWORD *)v172;
    }
    v347 = 257;
    if ( *(__int64 ***)v340 == v171 )
    {
      v174 = v340;
    }
    else if ( *(_BYTE *)(v340 + 16) > 0x10u )
    {
      v349 = 257;
      v259 = sub_15FDBD0(36, v340, (__int64)v171, (__int64)v348, 0);
      if ( v371 )
      {
        v335 = v259;
        v325 = (__int64 *)v372;
        sub_157E9D0((__int64)(v371 + 5), v259);
        v259 = v335;
        v260 = *(_QWORD *)(v335 + 24);
        v261 = *v325;
        *(_QWORD *)(v335 + 32) = v325;
        v261 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v335 + 24) = v261 | v260 & 7;
        *(_QWORD *)(v261 + 8) = v335 + 24;
        *v325 = *v325 & 7 | (v335 + 24);
      }
      v336 = v259;
      sub_164B780(v259, v346);
      v174 = v336;
      if ( v370 )
      {
        v343 = v370;
        sub_1623A60((__int64)&v343, (__int64)v370, 2);
        v174 = v336;
        v262 = *(_QWORD *)(v336 + 48);
        v263 = v336 + 48;
        if ( v262 )
        {
          v326 = v336;
          v337 = v336 + 48;
          sub_161E7C0(v337, v262);
          v174 = v326;
          v263 = v337;
        }
        v264 = v343;
        *(_QWORD *)(v174 + 48) = v343;
        if ( v264 )
        {
          v338 = v174;
          sub_1623210((__int64)&v343, v264, v263);
          v174 = v338;
        }
      }
    }
    else
    {
      v174 = sub_15A46C0(36, (__int64 ***)v340, v171, 0);
    }
    v331 = v174;
    v349 = 257;
    v315 = sub_394C7A0((__int64 *)&v370, (__int64)v173, v174, (__int64 *)v348, 0, *(double *)a3.m128_u64, a4, a5);
    v349 = 257;
    v327 = (__int64)sub_1B0E450((__int64 *)&v370, (__int64)v173, v331, (__int64 *)v348, *(double *)a3.m128_u64, a4, a5);
    v347 = 257;
    v175 = (__int64 ***)v355;
    v176 = (__int64 **)*v355;
    if ( *v355 != *(_QWORD *)v315 )
    {
      if ( *(_BYTE *)(v315 + 16) > 0x10u )
      {
        v349 = 257;
        v315 = sub_15FDBD0(37, v315, (__int64)v176, (__int64)v348, 0);
        if ( v371 )
        {
          v253 = (__int64 *)v372;
          sub_157E9D0((__int64)(v371 + 5), v315);
          v254 = *v253;
          v255 = *(_QWORD *)(v315 + 24);
          *(_QWORD *)(v315 + 32) = v253;
          v254 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v315 + 24) = v254 | v255 & 7;
          *(_QWORD *)(v254 + 8) = v315 + 24;
          *v253 = *v253 & 7 | (v315 + 24);
        }
        sub_164B780(v315, v346);
        if ( v370 )
        {
          v343 = v370;
          sub_1623A60((__int64)&v343, (__int64)v370, 2);
          v256 = v315 + 48;
          v257 = *(_QWORD *)(v315 + 48);
          if ( v257 )
            sub_161E7C0(v256, v257);
          v258 = v343;
          *(_QWORD *)(v315 + 48) = v343;
          if ( v258 )
            sub_1623210((__int64)&v343, v258, v256);
        }
        v175 = (__int64 ***)v355;
      }
      else
      {
        v315 = sub_15A46C0(37, (__int64 ***)v315, v176, 0);
        v175 = (__int64 ***)v355;
      }
    }
    v347 = 257;
    v177 = *v175;
    if ( *v175 != *(__int64 ***)v327 )
    {
      if ( *(_BYTE *)(v327 + 16) > 0x10u )
      {
        v349 = 257;
        v327 = sub_15FDBD0(37, v327, (__int64)v177, (__int64)v348, 0);
        if ( v371 )
        {
          v247 = (__int64 *)v372;
          sub_157E9D0((__int64)(v371 + 5), v327);
          v248 = *v247;
          v249 = *(_QWORD *)(v327 + 24);
          *(_QWORD *)(v327 + 32) = v247;
          v248 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v327 + 24) = v248 | v249 & 7;
          *(_QWORD *)(v248 + 8) = v327 + 24;
          *v247 = *v247 & 7 | (v327 + 24);
        }
        sub_164B780(v327, v346);
        if ( !v370 )
          goto LABEL_191;
        v343 = v370;
        sub_1623A60((__int64)&v343, (__int64)v370, 2);
        v250 = v327 + 48;
        v251 = *(_QWORD *)(v327 + 48);
        if ( v251 )
          sub_161E7C0(v250, v251);
        v252 = v343;
        *(_QWORD *)(v327 + 48) = v343;
        if ( v252 )
          sub_1623210((__int64)&v343, v252, v250);
      }
      else
      {
        v327 = sub_15A46C0(37, (__int64 ***)v327, v177, 0);
      }
    }
    goto LABEL_189;
  }
  v40 = *(_BYTE *)(v340 + 16);
  if ( v40 != 13 && (v40 != 71 || *(_QWORD *)(v340 + 40) != v355[5] || *(_BYTE *)(*(_QWORD *)(v340 - 24) + 16LL) != 13) )
  {
    LOWORD(v372) = 257;
    v276 = sub_157FBF0(v357, v355 + 3, (__int64)&v370);
    v41 = (_QWORD *)((v357[5] & 0xFFFFFFFFFFFFFFF8LL) - 24);
    if ( (v357[5] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      v41 = 0;
    sub_15F20C0(v41);
    v42 = v357[7];
    LOWORD(v372) = 257;
    v316 = v42;
    v328 = sub_15E0530(v42);
    v275 = (_QWORD *)sub_22077B0(0x40u);
    if ( v275 )
      sub_157FB60(v275, v328, (__int64)&v370, v316, v276);
    v329 = v275[6];
    v43 = sub_157E9C0((__int64)v275);
    v370 = 0;
    v373 = v43;
    v374 = 0;
    v371 = v275;
    v375 = 0;
    v376 = 0;
    v377 = 0;
    v372 = (unsigned __int64 *)v329;
    if ( (_QWORD *)v329 != v275 + 5 )
    {
      if ( !v329 )
        BUG();
      v44 = *(unsigned __int8 **)(v329 + 24);
      v348[0] = v44;
      if ( v44 )
      {
        sub_1623A60((__int64)v348, (__int64)v44, 2);
        if ( v370 )
          sub_161E7C0((__int64)&v370, (__int64)v370);
        v370 = v348[0];
        if ( v348[0] )
          sub_1623210((__int64)v348, v348[0], (__int64)&v370);
      }
    }
    if ( (*((_BYTE *)v355 + 23) & 0x40) != 0 )
      v45 = (__int64 *)*(v355 - 1);
    else
      v45 = &v355[-3 * (*((_DWORD *)v355 + 5) & 0xFFFFFFF)];
    v46 = *v45;
    v47 = v45[3];
    v347 = 257;
    v48 = *(__int64 ***)v47;
    v317 = v46;
    v285 = v47;
    if ( v356 != *(__int64 ***)v47 )
    {
      if ( *(_BYTE *)(v47 + 16) > 0x10u )
      {
        v349 = 257;
        v285 = sub_15FDBD0(36, v47, (__int64)v356, (__int64)v348, 0);
        if ( v371 )
        {
          v333 = (__int64 *)v372;
          sub_157E9D0((__int64)(v371 + 5), v285);
          v220 = *v333;
          v221 = *(_QWORD *)(v285 + 24);
          *(_QWORD *)(v285 + 32) = v333;
          v220 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v285 + 24) = v220 | v221 & 7;
          *(_QWORD *)(v220 + 8) = v285 + 24;
          *v333 = *v333 & 7 | (v285 + 24);
        }
        sub_164B780(v285, v346);
        if ( v370 )
        {
          v343 = v370;
          sub_1623A60((__int64)&v343, (__int64)v370, 2);
          v222 = *(_QWORD *)(v285 + 48);
          v223 = v285 + 48;
          if ( v222 )
          {
            sub_161E7C0(v285 + 48, v222);
            v223 = v285 + 48;
          }
          v224 = v343;
          *(_QWORD *)(v285 + 48) = v343;
          if ( v224 )
            sub_1623210((__int64)&v343, v224, v223);
        }
        v48 = v356;
      }
      else
      {
        v49 = sub_15A46C0(36, (__int64 ***)v47, v356, 0);
        v48 = v356;
        v285 = v49;
      }
    }
    v347 = 257;
    if ( *(__int64 ***)v317 != v48 )
    {
      if ( *(_BYTE *)(v317 + 16) > 0x10u )
      {
        v349 = 257;
        v317 = sub_15FDBD0(36, v317, (__int64)v48, (__int64)v348, 0);
        if ( v371 )
        {
          v332 = (__int64 *)v372;
          sub_157E9D0((__int64)(v371 + 5), v317);
          v215 = *v332;
          v216 = *(_QWORD *)(v317 + 24);
          *(_QWORD *)(v317 + 32) = v332;
          v215 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v317 + 24) = v215 | v216 & 7;
          *(_QWORD *)(v215 + 8) = v317 + 24;
          *v332 = *v332 & 7 | (v317 + 24);
        }
        sub_164B780(v317, v346);
        if ( v370 )
        {
          v343 = v370;
          sub_1623A60((__int64)&v343, (__int64)v370, 2);
          v217 = *(_QWORD *)(v317 + 48);
          v218 = v317 + 48;
          if ( v217 )
          {
            sub_161E7C0(v317 + 48, v217);
            v218 = v317 + 48;
          }
          v219 = v343;
          *(_QWORD *)(v317 + 48) = v343;
          if ( v219 )
            sub_1623210((__int64)&v343, v219, v218);
        }
      }
      else
      {
        v317 = sub_15A46C0(36, (__int64 ***)v317, v48, 0);
      }
    }
    v347 = 257;
    if ( *(_BYTE *)(v317 + 16) > 0x10u || *(_BYTE *)(v285 + 16) > 0x10u )
    {
      v349 = 257;
      v330 = sub_15FB440(17, (__int64 *)v317, v285, (__int64)v348, 0);
      if ( v371 )
      {
        v292 = (__int64 *)v372;
        sub_157E9D0((__int64)(v371 + 5), v330);
        v190 = *v292;
        v191 = *(_QWORD *)(v330 + 24);
        *(_QWORD *)(v330 + 32) = v292;
        v190 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v330 + 24) = v190 | v191 & 7;
        *(_QWORD *)(v190 + 8) = v330 + 24;
        *v292 = *v292 & 7 | (v330 + 24);
      }
      sub_164B780(v330, v346);
      if ( v370 )
      {
        v343 = v370;
        sub_1623A60((__int64)&v343, (__int64)v370, 2);
        v192 = *(_QWORD *)(v330 + 48);
        v193 = v330 + 48;
        if ( v192 )
        {
          sub_161E7C0(v330 + 48, v192);
          v193 = v330 + 48;
        }
        v194 = v343;
        *(_QWORD *)(v330 + 48) = v343;
        if ( v194 )
          sub_1623210((__int64)&v343, v194, v193);
      }
    }
    else
    {
      v330 = sub_15A2C70((__int64 *)v317, v285, 0, *(double *)a3.m128_u64, a4, a5);
    }
    v347 = 257;
    if ( *(_BYTE *)(v317 + 16) > 0x10u
      || *(_BYTE *)(v285 + 16) > 0x10u
      || (v287 = sub_15A2A30((__int64 *)0x14, (__int64 *)v317, v285, 0, 0, *(double *)a3.m128_u64, a4, a5)) == 0 )
    {
      v349 = 257;
      v287 = sub_15FB440(20, (__int64 *)v317, v285, (__int64)v348, 0);
      if ( v371 )
      {
        v322 = (__int64 *)v372;
        sub_157E9D0((__int64)(v371 + 5), v287);
        v195 = *v322;
        v196 = *(_QWORD *)(v287 + 24);
        *(_QWORD *)(v287 + 32) = v322;
        v195 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v287 + 24) = v195 | v196 & 7;
        *(_QWORD *)(v195 + 8) = v287 + 24;
        *v322 = *v322 & 7 | (v287 + 24);
      }
      sub_164B780(v287, v346);
      if ( v370 )
      {
        v343 = v370;
        sub_1623A60((__int64)&v343, (__int64)v370, 2);
        v197 = *(_QWORD *)(v287 + 48);
        v198 = v287 + 48;
        if ( v197 )
        {
          sub_161E7C0(v287 + 48, v197);
          v198 = v287 + 48;
        }
        v199 = v343;
        *(_QWORD *)(v287 + 48) = v343;
        if ( v199 )
          sub_1623210((__int64)&v343, v199, v198);
      }
    }
    v50 = (__int64 ***)v355;
    v347 = 257;
    v51 = (__int64 **)*v355;
    if ( *v355 != *(_QWORD *)v330 )
    {
      if ( *(_BYTE *)(v330 + 16) > 0x10u )
      {
        v349 = 257;
        v330 = sub_15FDBD0(37, v330, (__int64)v51, (__int64)v348, 0);
        if ( v371 )
        {
          v324 = (__int64 *)v372;
          sub_157E9D0((__int64)(v371 + 5), v330);
          v232 = *v324;
          v233 = *(_QWORD *)(v330 + 24);
          *(_QWORD *)(v330 + 32) = v324;
          v232 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v330 + 24) = v232 | v233 & 7;
          *(_QWORD *)(v232 + 8) = v330 + 24;
          *v324 = *v324 & 7 | (v330 + 24);
        }
        sub_164B780(v330, v346);
        if ( v370 )
        {
          v343 = v370;
          sub_1623A60((__int64)&v343, (__int64)v370, 2);
          v234 = *(_QWORD *)(v330 + 48);
          v235 = v330 + 48;
          if ( v234 )
          {
            sub_161E7C0(v330 + 48, v234);
            v235 = v330 + 48;
          }
          v236 = v343;
          *(_QWORD *)(v330 + 48) = v343;
          if ( v236 )
            sub_1623210((__int64)&v343, v236, v235);
        }
        v50 = (__int64 ***)v355;
      }
      else
      {
        v330 = sub_15A46C0(37, (__int64 ***)v330, v51, 0);
        v50 = (__int64 ***)v355;
      }
    }
    v347 = 257;
    v52 = *v50;
    if ( *v50 != *(__int64 ***)v287 )
    {
      if ( *(_BYTE *)(v287 + 16) > 0x10u )
      {
        v349 = 257;
        v287 = sub_15FDBD0(37, v287, (__int64)v52, (__int64)v348, 0);
        if ( v371 )
        {
          v323 = (__int64 *)v372;
          sub_157E9D0((__int64)(v371 + 5), v287);
          v227 = *v323;
          v228 = *(_QWORD *)(v287 + 24);
          *(_QWORD *)(v287 + 32) = v323;
          v227 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v287 + 24) = v227 | v228 & 7;
          *(_QWORD *)(v227 + 8) = v287 + 24;
          *v323 = *v323 & 7 | (v287 + 24);
        }
        sub_164B780(v287, v346);
        if ( v370 )
        {
          v343 = v370;
          sub_1623A60((__int64)&v343, (__int64)v370, 2);
          v229 = *(_QWORD *)(v287 + 48);
          v230 = v287 + 48;
          if ( v229 )
          {
            sub_161E7C0(v287 + 48, v229);
            v230 = v287 + 48;
          }
          v231 = v343;
          *(_QWORD *)(v287 + 48) = v343;
          if ( v231 )
            sub_1623210((__int64)&v343, v231, v230);
        }
      }
      else
      {
        v287 = sub_15A46C0(37, (__int64 ***)v287, v52, 0);
      }
    }
    v349 = 257;
    v53 = sub_1648A60(56, 1u);
    v54 = (__int64)v53;
    if ( v53 )
    {
      v318 = v53;
      sub_15F8320((__int64)v53, v276, 0);
      v54 = (__int64)v318;
    }
    if ( v371 )
    {
      v319 = v54;
      v286 = (__int64 *)v372;
      sub_157E9D0((__int64)(v371 + 5), v54);
      v54 = v319;
      v55 = *(_QWORD *)(v319 + 24);
      v56 = *v286;
      *(_QWORD *)(v319 + 32) = v286;
      v56 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v319 + 24) = v56 | v55 & 7;
      *(_QWORD *)(v56 + 8) = v319 + 24;
      *v286 = *v286 & 7 | (v319 + 24);
    }
    v320 = v54;
    sub_164B780(v54, (__int64 *)v348);
    if ( v370 )
    {
      v346[0] = (__int64)v370;
      sub_1623A60((__int64)v346, (__int64)v370, 2);
      v57 = v320;
      v58 = *(_QWORD *)(v320 + 48);
      v59 = v320 + 48;
      if ( v58 )
      {
        v272 = v320;
        v321 = v320 + 48;
        sub_161E7C0(v321, v58);
        v57 = v272;
        v59 = v321;
      }
      v60 = (unsigned __int8 *)v346[0];
      *(_QWORD *)(v57 + 48) = v346[0];
      if ( v60 )
        sub_1623210((__int64)v346, v60, v59);
      if ( v370 )
        sub_161E7C0((__int64)&v370, (__int64)v370);
    }
    sub_394C8F0((__int64 *)&v343, (__int64)v354, v276, *(double *)a3.m128_u64, a4, a5);
    v61 = *(_QWORD *)(v276 + 48);
    v62 = sub_157E9C0(v276);
    v370 = 0;
    v373 = v62;
    v374 = 0;
    v371 = (_QWORD *)v276;
    v375 = 0;
    v376 = 0;
    v377 = 0;
    v372 = (unsigned __int64 *)v61;
    if ( v61 != v276 + 40 )
    {
      if ( !v61 )
        BUG();
      v63 = *(unsigned __int8 **)(v61 + 24);
      v348[0] = v63;
      if ( v63 )
      {
        sub_1623A60((__int64)v348, (__int64)v63, 2);
        if ( v370 )
          sub_161E7C0((__int64)&v370, (__int64)v370);
        v370 = v348[0];
        if ( v348[0] )
          sub_1623210((__int64)v348, v348[0], (__int64)&v370);
      }
    }
    v347 = 257;
    v308 = *v355;
    v349 = 257;
    v315 = sub_1648B60(64);
    if ( v315 )
    {
      v64 = v315;
      sub_15F1EA0(v315, v308, 53, 0, 0, 0);
      *(_DWORD *)(v315 + 56) = 2;
      sub_164B780(v315, (__int64 *)v348);
      sub_1648880(v315, *(_DWORD *)(v315 + 56), 1);
    }
    else
    {
      v64 = 0;
    }
    if ( v371 )
    {
      v309 = (__int64 *)v372;
      sub_157E9D0((__int64)(v371 + 5), v315);
      v65 = *v309;
      v66 = *(_QWORD *)(v315 + 24);
      *(_QWORD *)(v315 + 32) = v309;
      v65 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v315 + 24) = v65 | v66 & 7;
      *(_QWORD *)(v65 + 8) = v315 + 24;
      *v309 = *v309 & 7 | (v315 + 24);
    }
    sub_164B780(v64, v346);
    v70 = v370;
    if ( v370 )
    {
      v342 = v370;
      sub_1623A60((__int64)&v342, (__int64)v370, 2);
      v68 = &v342;
      v71 = *(_QWORD *)(v315 + 48);
      v67 = v315 + 48;
      if ( v71 )
      {
        sub_161E7C0(v315 + 48, v71);
        v68 = &v342;
        v67 = v315 + 48;
      }
      v70 = v342;
      *(_QWORD *)(v315 + 48) = v342;
      if ( v70 )
        sub_1623210((__int64)&v342, v70, v67);
    }
    v72 = *(_DWORD *)(v315 + 20) & 0xFFFFFFF;
    if ( v72 == *(_DWORD *)(v315 + 56) )
    {
      sub_15F55D0(v315, (__int64)v70, v67, v315, (__int64)v68, v69);
      v72 = *(_DWORD *)(v315 + 20) & 0xFFFFFFF;
    }
    v73 = (v72 + 1) & 0xFFFFFFF;
    v74 = v73 | *(_DWORD *)(v315 + 20) & 0xF0000000;
    *(_DWORD *)(v315 + 20) = v74;
    if ( (v74 & 0x40000000) != 0 )
      v75 = *(_QWORD *)(v315 - 8);
    else
      v75 = v64 - 24 * v73;
    v76 = (__int64 *)(v75 + 24LL * (unsigned int)(v73 - 1));
    if ( *v76 )
    {
      v77 = v76[1];
      v78 = v76[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v78 = v77;
      if ( v77 )
        *(_QWORD *)(v77 + 16) = v78 | *(_QWORD *)(v77 + 16) & 3LL;
    }
    *v76 = v330;
    if ( v330 )
    {
      v79 = *(_QWORD *)(v330 + 8);
      v76[1] = v79;
      if ( v79 )
        *(_QWORD *)(v79 + 16) = (unsigned __int64)(v76 + 1) | *(_QWORD *)(v79 + 16) & 3LL;
      v76[2] = (v330 + 8) | v76[2] & 3;
      *(_QWORD *)(v330 + 8) = v76;
    }
    v80 = *(_DWORD *)(v315 + 20) & 0xFFFFFFF;
    v81 = (unsigned int)(v80 - 1);
    if ( (*(_BYTE *)(v315 + 23) & 0x40) != 0 )
      v82 = *(_QWORD *)(v315 - 8);
    else
      v82 = v64 - 24 * v80;
    v83 = 3LL * *(unsigned int *)(v315 + 56);
    *(_QWORD *)(v82 + 8 * v81 + 24LL * *(unsigned int *)(v315 + 56) + 8) = v275;
    v84 = v344;
    v310 = v343;
    v85 = *(_DWORD *)(v315 + 20) & 0xFFFFFFF;
    if ( v85 == *(_DWORD *)(v315 + 56) )
    {
      v284 = v344;
      sub_15F55D0(v315, v315, v83, v344, (__int64)v68, v69);
      v84 = v284;
      v85 = *(_DWORD *)(v315 + 20) & 0xFFFFFFF;
    }
    v86 = (v85 + 1) & 0xFFFFFFF;
    v87 = v86 | *(_DWORD *)(v315 + 20) & 0xF0000000;
    *(_DWORD *)(v315 + 20) = v87;
    if ( (v87 & 0x40000000) != 0 )
      v88 = *(_QWORD *)(v315 - 8);
    else
      v88 = v64 - 24 * v86;
    v89 = (__int64 *)(v88 + 24LL * (unsigned int)(v86 - 1));
    if ( *v89 )
    {
      v90 = v89[1];
      v91 = v89[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v91 = v90;
      if ( v90 )
        *(_QWORD *)(v90 + 16) = v91 | *(_QWORD *)(v90 + 16) & 3LL;
    }
    *v89 = v84;
    if ( v84 )
    {
      v92 = *(_QWORD *)(v84 + 8);
      v89[1] = v92;
      if ( v92 )
        *(_QWORD *)(v92 + 16) = (unsigned __int64)(v89 + 1) | *(_QWORD *)(v92 + 16) & 3LL;
      v89[2] = (v84 + 8) | v89[2] & 3;
      *(_QWORD *)(v84 + 8) = v89;
    }
    v93 = *(_DWORD *)(v315 + 20) & 0xFFFFFFF;
    if ( (*(_BYTE *)(v315 + 23) & 0x40) != 0 )
      v94 = *(_QWORD *)(v315 - 8);
    else
      v94 = v64 - 24 * v93;
    *(_QWORD *)(v94 + 8LL * (unsigned int)(v93 - 1) + 24LL * *(unsigned int *)(v315 + 56) + 8) = v310;
    v347 = 257;
    v95 = *v355;
    v349 = 257;
    v327 = sub_1648B60(64);
    if ( v327 )
    {
      v96 = v327;
      sub_15F1EA0(v327, v95, 53, 0, 0, 0);
      *(_DWORD *)(v327 + 56) = 2;
      sub_164B780(v327, (__int64 *)v348);
      sub_1648880(v327, *(_DWORD *)(v327 + 56), 1);
    }
    else
    {
      v96 = 0;
    }
    if ( v371 )
    {
      v277 = (__int64 *)v372;
      sub_157E9D0((__int64)(v371 + 5), v327);
      v97 = *v277;
      v98 = *(_QWORD *)(v327 + 24);
      *(_QWORD *)(v327 + 32) = v277;
      v97 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v327 + 24) = v97 | v98 & 7;
      *(_QWORD *)(v97 + 8) = v327 + 24;
      *v277 = *v277 & 7 | (v327 + 24);
    }
    sub_164B780(v96, v346);
    v102 = v370;
    if ( v370 )
    {
      v342 = v370;
      sub_1623A60((__int64)&v342, (__int64)v370, 2);
      v100 = &v342;
      v103 = *(_QWORD *)(v327 + 48);
      v99 = v327 + 48;
      if ( v103 )
      {
        sub_161E7C0(v327 + 48, v103);
        v100 = &v342;
        v99 = v327 + 48;
      }
      v102 = v342;
      *(_QWORD *)(v327 + 48) = v342;
      if ( v102 )
        sub_1623210((__int64)&v342, v102, v99);
    }
    v104 = *(_DWORD *)(v327 + 20) & 0xFFFFFFF;
    if ( v104 == *(_DWORD *)(v327 + 56) )
    {
      sub_15F55D0(v327, (__int64)v102, v99, v327, (__int64)v100, v101);
      v104 = *(_DWORD *)(v327 + 20) & 0xFFFFFFF;
    }
    v105 = (v104 + 1) & 0xFFFFFFF;
    v278 = *(_DWORD *)(v327 + 20);
    *(_DWORD *)(v327 + 20) = v105 | v278 & 0xF0000000;
    if ( v105 & 0x40000000 | v278 & 0x40000000 )
      v106 = *(_QWORD *)(v327 - 8);
    else
      v106 = v96 - 24 * v105;
    v107 = (__int64 *)(v106 + 24LL * (unsigned int)(v105 - 1));
    if ( *v107 )
    {
      v108 = v107[1];
      v109 = v107[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v109 = v108;
      if ( v108 )
        *(_QWORD *)(v108 + 16) = v109 | *(_QWORD *)(v108 + 16) & 3LL;
    }
    *v107 = v287;
    if ( v287 )
    {
      v110 = *(_QWORD *)(v287 + 8);
      v107[1] = v110;
      if ( v110 )
        *(_QWORD *)(v110 + 16) = (unsigned __int64)(v107 + 1) | *(_QWORD *)(v110 + 16) & 3LL;
      v107[2] = (v287 + 8) | v107[2] & 3;
      *(_QWORD *)(v287 + 8) = v107;
    }
    v111 = *(_DWORD *)(v327 + 20) & 0xFFFFFFF;
    v112 = (unsigned int)(v111 - 1);
    if ( (*(_BYTE *)(v327 + 23) & 0x40) != 0 )
      v113 = *(_QWORD *)(v327 - 8);
    else
      v113 = v96 - 24 * v111;
    v114 = 3LL * *(unsigned int *)(v327 + 56);
    *(_QWORD *)(v113 + 8 * v112 + 24LL * *(unsigned int *)(v327 + 56) + 8) = v275;
    v115 = v345;
    v116 = *(_DWORD *)(v327 + 20) & 0xFFFFFFF;
    if ( v116 == *(_DWORD *)(v327 + 56) )
    {
      v283 = v345;
      sub_15F55D0(v327, v327, v114, v345, (__int64)v100, v101);
      v115 = v283;
      v116 = *(_DWORD *)(v327 + 20) & 0xFFFFFFF;
    }
    v117 = (v116 + 1) & 0xFFFFFFF;
    v118 = v117 | *(_DWORD *)(v327 + 20) & 0xF0000000;
    *(_DWORD *)(v327 + 20) = v118;
    if ( (v118 & 0x40000000) != 0 )
      v119 = *(_QWORD *)(v327 - 8);
    else
      v119 = v96 - 24 * v117;
    v120 = (__int64 *)(v119 + 24LL * (unsigned int)(v117 - 1));
    if ( *v120 )
    {
      v121 = v120[1];
      v122 = v120[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v122 = v121;
      if ( v121 )
        *(_QWORD *)(v121 + 16) = v122 | *(_QWORD *)(v121 + 16) & 3LL;
    }
    *v120 = v115;
    if ( v115 )
    {
      v123 = *(_QWORD *)(v115 + 8);
      v120[1] = v123;
      if ( v123 )
        *(_QWORD *)(v123 + 16) = (unsigned __int64)(v120 + 1) | *(_QWORD *)(v123 + 16) & 3LL;
      v120[2] = (v115 + 8) | v120[2] & 3;
      *(_QWORD *)(v115 + 8) = v120;
    }
    v124 = *(_DWORD *)(v327 + 20) & 0xFFFFFFF;
    if ( (*(_BYTE *)(v327 + 23) & 0x40) != 0 )
      v125 = *(_QWORD *)(v327 - 8);
    else
      v125 = v96 - 24 * v124;
    *(_QWORD *)(v125 + 8LL * (unsigned int)(v124 - 1) + 24LL * *(unsigned int *)(v327 + 56) + 8) = v310;
    v126 = v370;
    if ( v370 )
      sub_161E7C0((__int64)&v370, (__int64)v370);
    v127 = v340;
    if ( !v273 )
      v127 = 0;
    if ( v295 )
    {
      v128 = v341;
      v129 = v341 != 0 && v273 != 0;
    }
    else
    {
      v129 = 0;
      v128 = 0;
    }
    v279 = v127;
    v288 = v128;
    v296 = v357;
    v130 = sub_157E9C0((__int64)v357);
    v370 = 0;
    v373 = v130;
    v132 = (__int64 *)v288;
    v371 = v296;
    v133 = v279;
    v374 = 0;
    v375 = 0;
    v376 = 0;
    v377 = 0;
    v372 = v296 + 5;
    if ( !v129 )
    {
      if ( !v288 )
        v132 = (__int64 *)v279;
      goto LABEL_171;
    }
    v347 = 257;
    if ( *(_BYTE *)(v279 + 16) <= 0x10u )
    {
      v207 = sub_1593BB0(v279, (__int64)v126, (__int64)(v296 + 5), v131);
      v132 = (__int64 *)v288;
      if ( v207 )
      {
LABEL_171:
        v297 = v132;
        v134 = 0xFFFFFFFFFFFFFFFFLL >> (64 - BYTE1(*((_DWORD *)v356 + 2)));
        v347 = 257;
        v135 = sub_15A0680(*v132, ~v134, 0);
        v136 = (__int64)v297;
        v137 = v135;
        v138 = *(_BYTE *)(v135 + 16);
        if ( v138 > 0x10u )
          goto LABEL_281;
        if ( v138 == 13 )
        {
          v225 = *(_DWORD *)(v137 + 32);
          if ( v225 <= 0x40 )
          {
            if ( *(_QWORD *)(v137 + 24) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v225) )
              goto LABEL_175;
          }
          else
          {
            v282 = *(_DWORD *)(v137 + 32);
            v294 = (__int64)v297;
            v307 = v137;
            v226 = sub_16A58F0(v137 + 24);
            v137 = v307;
            v136 = v294;
            if ( v282 == v226 )
              goto LABEL_175;
          }
        }
        if ( *(_BYTE *)(v136 + 16) > 0x10u )
        {
LABEL_281:
          v349 = 257;
          v200 = sub_15FB440(26, (__int64 *)v136, v137, (__int64)v348, 0);
          if ( v371 )
          {
            v201 = (__int64 *)v372;
            v301 = v200;
            sub_157E9D0((__int64)(v371 + 5), v200);
            v200 = v301;
            v202 = *v201;
            v203 = *(_QWORD *)(v301 + 24);
            *(_QWORD *)(v301 + 32) = v201;
            v202 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v301 + 24) = v202 | v203 & 7;
            *(_QWORD *)(v202 + 8) = v301 + 24;
            *v201 = *v201 & 7 | (v301 + 24);
          }
          v302 = v200;
          sub_164B780(v200, v346);
          v136 = v302;
          if ( v370 )
          {
            v342 = v370;
            sub_1623A60((__int64)&v342, (__int64)v370, 2);
            v136 = v302;
            v204 = *(_QWORD *)(v302 + 48);
            v205 = v302 + 48;
            if ( v204 )
            {
              sub_161E7C0(v302 + 48, v204);
              v136 = v302;
            }
            v206 = v342;
            *(_QWORD *)(v136 + 48) = v342;
            if ( v206 )
            {
              v303 = v136;
              sub_1623210((__int64)&v342, v206, v205);
              v136 = v303;
            }
          }
        }
        else
        {
          v136 = sub_15A2CF0((__int64 *)v136, v137, *(double *)a3.m128_u64, a4, a5);
        }
LABEL_175:
        v298 = v136;
        v139 = sub_15A0930(*v355, 0);
        v347 = 257;
        if ( *(_BYTE *)(v298 + 16) > 0x10u || *(_BYTE *)(v139 + 16) > 0x10u )
        {
          v289 = v139;
          v349 = 257;
          v178 = sub_1648A60(56, 2u);
          v179 = v298;
          v142 = v178;
          if ( v178 )
          {
            v300 = (__int64)v178;
            v180 = *(_QWORD ***)v179;
            if ( *(_BYTE *)(*(_QWORD *)v179 + 8LL) == 16 )
            {
              v274 = v289;
              v280 = v179;
              v290 = v180[4];
              v181 = (__int64 *)sub_1643320(*v180);
              v182 = (__int64)sub_16463B0(v181, (unsigned int)v290);
              v183 = v280;
              v184 = v274;
            }
            else
            {
              v281 = v289;
              v293 = v179;
              v182 = sub_1643320(*v180);
              v184 = v281;
              v183 = v293;
            }
            sub_15FEC10((__int64)v142, v182, 51, 32, v183, v184, (__int64)v348, 0);
          }
          else
          {
            v300 = 0;
          }
          if ( v371 )
          {
            v291 = v372;
            sub_157E9D0((__int64)(v371 + 5), (__int64)v142);
            v185 = *v291;
            v186 = v142[3] & 7LL;
            v142[4] = v291;
            v185 &= 0xFFFFFFFFFFFFFFF8LL;
            v142[3] = v185 | v186;
            *(_QWORD *)(v185 + 8) = v142 + 3;
            *v291 = *v291 & 7 | (unsigned __int64)(v142 + 3);
          }
          sub_164B780(v300, v346);
          if ( !v370 )
            goto LABEL_180;
          v342 = v370;
          sub_1623A60((__int64)&v342, (__int64)v370, 2);
          v187 = v142[6];
          v188 = (__int64)(v142 + 6);
          if ( v187 )
          {
            sub_161E7C0((__int64)(v142 + 6), v187);
            v188 = (__int64)(v142 + 6);
          }
          v189 = v342;
          v142[6] = v342;
          if ( v189 )
            sub_1623210((__int64)&v342, v189, v188);
          v141 = v370;
        }
        else
        {
          v140 = sub_15A37B0(0x20u, (_QWORD *)v298, (_QWORD *)v139, 0);
          v141 = v370;
          v142 = (_QWORD *)v140;
        }
        if ( v141 )
          sub_161E7C0((__int64)&v370, (__int64)v141);
LABEL_180:
        v299 = v357;
        v143 = sub_157E9C0((__int64)v357);
        v370 = 0;
        v371 = v299;
        v373 = v143;
        v374 = 0;
        v375 = 0;
        v376 = 0;
        v377 = 0;
        v372 = v299 + 5;
        v349 = 257;
        v144 = sub_1648A60(56, 3u);
        v145 = (__int64)v144;
        if ( v144 )
        {
          v146 = (__int64)v310;
          v311 = v144;
          sub_15F83E0((__int64)v144, (__int64)v275, v146, (__int64)v142, 0);
          v145 = (__int64)v311;
        }
        if ( v371 )
        {
          v147 = (__int64 *)v372;
          v312 = v145;
          sub_157E9D0((__int64)(v371 + 5), v145);
          v145 = v312;
          v148 = *v147;
          v149 = *(_QWORD *)(v312 + 24);
          *(_QWORD *)(v312 + 32) = v147;
          v148 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v312 + 24) = v148 | v149 & 7;
          *(_QWORD *)(v148 + 8) = v312 + 24;
          *v147 = *v147 & 7 | (v312 + 24);
        }
        v313 = v145;
        sub_164B780(v145, (__int64 *)v348);
        if ( !v370 )
          goto LABEL_191;
        v346[0] = (__int64)v370;
        sub_1623A60((__int64)v346, (__int64)v370, 2);
        v150 = v313;
        v151 = *(_QWORD *)(v313 + 48);
        if ( v151 )
        {
          sub_161E7C0(v313 + 48, v151);
          v150 = v313;
        }
        v152 = (unsigned __int8 *)v346[0];
        *(_QWORD *)(v150 + 48) = v346[0];
        if ( v152 )
          sub_1623210((__int64)v346, v152, v313 + 48);
LABEL_189:
        if ( v370 )
          sub_161E7C0((__int64)&v370, (__int64)v370);
LABEL_191:
        v153 = 1;
        goto LABEL_195;
      }
      v133 = v279;
      if ( *(_BYTE *)(v288 + 16) <= 0x10u )
      {
        v132 = (__int64 *)sub_15A2D10((__int64 *)v288, v279, *(double *)a3.m128_u64, a4, a5);
        goto LABEL_171;
      }
    }
    v349 = 257;
    v208 = sub_15FB440(27, v132, v133, (__int64)v348, 0);
    if ( v371 )
    {
      v209 = (__int64 *)v372;
      v304 = v208;
      sub_157E9D0((__int64)(v371 + 5), v208);
      v208 = v304;
      v210 = *v209;
      v211 = *(_QWORD *)(v304 + 24);
      *(_QWORD *)(v304 + 32) = v209;
      v210 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v304 + 24) = v210 | v211 & 7;
      *(_QWORD *)(v210 + 8) = v304 + 24;
      *v209 = *v209 & 7 | (v304 + 24);
    }
    v305 = (__int64 *)v208;
    sub_164B780(v208, v346);
    v132 = v305;
    if ( v370 )
    {
      v342 = v370;
      sub_1623A60((__int64)&v342, (__int64)v370, 2);
      v132 = v305;
      v212 = v305[6];
      v213 = (__int64)(v305 + 6);
      if ( v212 )
      {
        sub_161E7C0((__int64)(v305 + 6), v212);
        v132 = v305;
      }
      v214 = v342;
      v132[6] = (__int64)v342;
      if ( v214 )
      {
        v306 = v132;
        sub_1623210((__int64)&v342, v214, v213);
        v132 = v306;
      }
    }
    goto LABEL_171;
  }
LABEL_194:
  v153 = 0;
LABEL_195:
  if ( v366 != v365 )
    _libc_free((unsigned __int64)v366);
  if ( v360 != v359 )
    _libc_free((unsigned __int64)v360);
  if ( !v153 )
    goto LABEL_4;
  v154 = v353;
  if ( !v353 )
  {
    ++v350;
    goto LABEL_339;
  }
  v155 = 0;
  v156 = 1;
  v157 = v340 ^ v341 ^ v30;
  v158 = (v353 - 1) & v157;
  while ( 2 )
  {
    v159 = v351 + 40LL * v158;
    v160 = *(_QWORD *)(v159 + 8);
    if ( *(_BYTE *)v159 == v30 && v341 == v160 && v340 == *(_QWORD *)(v159 + 16) )
    {
      v36 = *(_QWORD *)(v159 + 32);
      v37 = *(_QWORD *)(v159 + 24);
      goto LABEL_222;
    }
    if ( *(_BYTE *)v159 )
    {
      if ( !v160 && !(*(_QWORD *)(v159 + 16) | v155) )
        v155 = v351 + 40LL * v158;
      goto LABEL_377;
    }
    if ( v160 || *(_QWORD *)(v159 + 16) )
    {
LABEL_377:
      v265 = v156 + v158;
      ++v156;
      v158 = (v353 - 1) & v265;
      continue;
    }
    break;
  }
  v154 = v353;
  if ( v155 )
    v159 = v155;
  ++v350;
  v266 = v352 + 1;
  if ( 4 * ((int)v352 + 1) < 3 * v353 )
  {
    if ( v353 - HIDWORD(v352) - v266 > v353 >> 3 )
      goto LABEL_384;
    sub_394BE70((__int64)&v350, v353);
    if ( v353 )
    {
      v159 = 0;
      v267 = 1;
      for ( k = (v353 - 1) & v157; ; k = (v353 - 1) & v271 )
      {
        v269 = v351 + 40LL * k;
        v270 = *(_QWORD *)(v269 + 8);
        if ( *(_BYTE *)v269 == v30 && v341 == v270 && v340 == *(_QWORD *)(v269 + 16) )
        {
          v266 = v352 + 1;
          v159 = v351 + 40LL * k;
          goto LABEL_384;
        }
        if ( *(_BYTE *)v269 )
        {
          if ( !v270 && !(*(_QWORD *)(v269 + 16) | v159) )
            v159 = v351 + 40LL * k;
        }
        else if ( !v270 && !*(_QWORD *)(v269 + 16) )
        {
          if ( !v159 )
            v159 = v351 + 40LL * k;
          v266 = v352 + 1;
          goto LABEL_384;
        }
        v271 = v267 + k;
        ++v267;
      }
    }
LABEL_425:
    LODWORD(v352) = v352 + 1;
    BUG();
  }
LABEL_339:
  sub_394BE70((__int64)&v350, 2 * v154);
  if ( !v353 )
    goto LABEL_425;
  v237 = 0;
  v238 = 1;
  v239 = (v353 - 1) & (v30 ^ v340 ^ v341);
  while ( 2 )
  {
    v159 = v351 + 40LL * v239;
    v240 = *(_QWORD *)(v159 + 8);
    if ( *(_BYTE *)v159 == v30 && v341 == v240 && v340 == *(_QWORD *)(v159 + 16) )
    {
      v266 = v352 + 1;
      goto LABEL_384;
    }
    if ( *(_BYTE *)v159 )
    {
      if ( !v240 && !(*(_QWORD *)(v159 + 16) | v237) )
        v237 = v351 + 40LL * v239;
      goto LABEL_344;
    }
    if ( v240 || *(_QWORD *)(v159 + 16) )
    {
LABEL_344:
      v241 = v238 + v239;
      ++v238;
      v239 = (v353 - 1) & v241;
      continue;
    }
    break;
  }
  if ( v237 )
    v159 = v237;
  v266 = v352 + 1;
LABEL_384:
  LODWORD(v352) = v266;
  if ( *(_BYTE *)v159 || *(_QWORD *)(v159 + 8) || *(_QWORD *)(v159 + 16) )
    --HIDWORD(v352);
  v36 = v327;
  *(_BYTE *)v159 = v30;
  *(_QWORD *)(v159 + 8) = v341;
  *(_QWORD *)(v159 + 32) = v327;
  *(_QWORD *)(v159 + 16) = v340;
  *(_QWORD *)(v159 + 24) = v315;
  v37 = v315;
LABEL_222:
  v29 = *((_BYTE *)v355 + 16);
LABEL_28:
  if ( (unsigned __int8)(v29 - 41) <= 1u )
    v36 = v37;
  if ( !v36 )
    goto LABEL_4;
  sub_164D160((__int64)i, v36, a3, a4, a5, a6, v24, v25, a9, a10);
  sub_15F20C0(i);
  v339 = 1;
  if ( v13 )
    goto LABEL_5;
LABEL_32:
  v10 = v351;
  if ( (_DWORD)v352 )
  {
    v161 = (_QWORD *)(v351 + 40LL * v353);
    if ( v161 != (_QWORD *)v351 )
    {
      v162 = (_QWORD *)v351;
      while ( 1 )
      {
        v163 = v162;
        if ( v162[1] || v162[2] )
          break;
        v162 += 5;
        if ( v161 == v162 )
          goto LABEL_33;
      }
      if ( v161 != v162 )
      {
        do
        {
          v164 = v163[4];
          v165 = v163[3];
          v163 += 5;
          sub_1AEB370(v165, 0);
          sub_1AEB370(v164, 0);
          if ( v163 == v161 )
            break;
          while ( !v163[1] && !v163[2] )
          {
            v163 += 5;
            if ( v161 == v163 )
              goto LABEL_218;
          }
        }
        while ( v163 != v161 );
LABEL_218:
        v10 = v351;
      }
    }
  }
LABEL_33:
  j___libc_free_0(v10);
  return v339;
}
