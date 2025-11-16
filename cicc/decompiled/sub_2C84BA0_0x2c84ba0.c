// Function: sub_2C84BA0
// Address: 0x2c84ba0
//
_BOOL8 __fastcall sub_2C84BA0(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // r15
  __int64 v3; // rdx
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rbx
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  _QWORD *v10; // rbx
  unsigned __int64 v11; // rdx
  _QWORD *v12; // rax
  __int64 v13; // r8
  __int64 v14; // rsi
  __int64 v15; // rcx
  _QWORD *v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // rdx
  _QWORD *v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rcx
  __int64 v23; // rdx
  _QWORD *v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rcx
  __int64 v27; // rdx
  _QWORD *v28; // r12
  _QWORD *v29; // r15
  unsigned __int64 *v30; // r14
  unsigned __int64 v31; // rbx
  __int64 v32; // rsi
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rcx
  char v36; // al
  __int64 v37; // rsi
  __int64 v38; // rcx
  __int64 v39; // rax
  __int64 v40; // rdi
  char v41; // al
  __int64 v42; // rsi
  __int64 v43; // rcx
  __int64 v44; // rax
  __int64 v45; // rdi
  char v46; // al
  __int64 v47; // rsi
  __int64 v48; // rcx
  __int64 v49; // rax
  __int64 v50; // rdi
  __int64 v51; // rbx
  __int64 v52; // rax
  unsigned __int64 v53; // rcx
  _QWORD *v54; // rax
  __int64 v55; // rsi
  _QWORD *v56; // rdx
  _QWORD *v57; // rax
  char v58; // r15
  __int64 v59; // rsi
  _QWORD *v60; // rdx
  _QWORD *v61; // rax
  __int64 v62; // rsi
  _QWORD *v63; // rdx
  _QWORD *v64; // rax
  char v65; // r15
  __int64 v66; // rsi
  _QWORD *v67; // rdx
  _QWORD *v68; // r14
  int v69; // r12d
  __int64 v70; // rbx
  __int64 v71; // rdi
  bool v72; // al
  int v73; // ebx
  _QWORD *v74; // rax
  __int64 v75; // rsi
  __int64 v76; // rdi
  __int64 v77; // rcx
  _QWORD *v78; // rax
  char v79; // bl
  __int64 v80; // rsi
  __int64 v81; // rdi
  __int64 v82; // rcx
  _QWORD *v83; // rax
  __int64 v84; // rsi
  __int64 v85; // rdi
  __int64 v86; // rcx
  _QWORD *v87; // rax
  char v88; // bl
  __int64 v89; // rsi
  __int64 v90; // rdi
  __int64 v91; // rcx
  __int64 v92; // rsi
  __int64 v93; // rcx
  __int64 v94; // rax
  __int64 v95; // rdi
  unsigned __int64 v96; // rbx
  unsigned __int64 v97; // rdi
  unsigned __int64 v98; // rbx
  unsigned __int64 v99; // rdi
  unsigned __int64 v100; // rdi
  unsigned __int64 v101; // rbx
  unsigned __int64 v102; // rdi
  _QWORD *v103; // rbx
  unsigned __int64 v104; // rdx
  _QWORD *v105; // rax
  __int64 v106; // r8
  __int64 v107; // rsi
  __int64 v108; // rcx
  _QWORD *v109; // rax
  __int64 v110; // rsi
  __int64 v111; // rcx
  __int64 v112; // rdx
  _QWORD *v113; // rax
  __int64 v114; // rsi
  __int64 v115; // rcx
  __int64 v116; // rdx
  _QWORD *v117; // rax
  __int64 v118; // rsi
  __int64 v119; // rcx
  __int64 v120; // rdx
  _QWORD *v121; // rbx
  __int64 v122; // rsi
  bool v123; // zf
  unsigned __int64 v124; // r13
  __int64 v125; // rdx
  __int64 v126; // rax
  __int64 v127; // rcx
  char v128; // al
  __int64 v129; // rsi
  __int64 v130; // rcx
  __int64 v131; // rax
  __int64 v132; // rdi
  char v133; // al
  __int64 v134; // rsi
  __int64 v135; // rcx
  __int64 v136; // rax
  __int64 v137; // rdi
  char v138; // al
  __int64 v139; // rsi
  __int64 v140; // rcx
  __int64 v141; // rax
  __int64 v142; // rdi
  _QWORD *v143; // r13
  unsigned __int64 v144; // rax
  unsigned int v145; // r13d
  __int64 v146; // rsi
  _QWORD *v147; // rax
  _QWORD *v148; // rcx
  _QWORD *v149; // rax
  char v150; // r14
  __int64 v151; // rsi
  _QWORD *v152; // rdx
  _QWORD *v153; // rax
  __int64 v154; // rsi
  _QWORD *v155; // rdx
  _QWORD *v156; // rax
  char v157; // r14
  __int64 v158; // rsi
  _QWORD *v159; // rdx
  __int64 v160; // r13
  unsigned __int64 v161; // r14
  int v162; // ebx
  __int64 v163; // rdi
  bool v164; // al
  int v165; // edx
  _QWORD *v166; // rax
  __int64 v167; // rsi
  __int64 v168; // rdi
  __int64 v169; // rcx
  _QWORD *v170; // rax
  char v171; // r13
  __int64 v172; // rsi
  __int64 v173; // rdi
  __int64 v174; // rcx
  _QWORD *v175; // rax
  __int64 v176; // rsi
  __int64 v177; // rdi
  __int64 v178; // rcx
  _QWORD *v179; // rax
  char v180; // r13
  __int64 v181; // rsi
  __int64 v182; // rdi
  __int64 v183; // rcx
  __int64 v184; // rsi
  __int64 v185; // rcx
  __int64 v186; // rax
  __int64 v187; // rdi
  _QWORD *v188; // rbx
  int v189; // r12d
  __int64 v190; // rdi
  bool v191; // al
  unsigned __int64 v192; // rdi
  unsigned __int64 v193; // rdi
  unsigned __int64 v194; // rdi
  unsigned __int64 v195; // rdi
  unsigned __int64 v196; // rdi
  unsigned __int64 v197; // rdi
  unsigned __int64 v198; // rdi
  unsigned __int64 v199; // rdi
  unsigned __int64 v200; // rdi
  unsigned __int64 v201; // rdi
  unsigned __int64 v202; // rdi
  unsigned __int64 v203; // rdi
  unsigned __int64 v204; // rdi
  _QWORD *v206; // rax
  __int64 v207; // rsi
  _QWORD *v208; // rdx
  char v209; // bl
  _QWORD *v210; // rax
  __int64 v211; // rsi
  _QWORD *v212; // rdx
  _QWORD *v213; // rax
  __int64 v214; // rsi
  _QWORD *v215; // rdx
  _QWORD *v216; // rax
  _QWORD *v217; // rdx
  _QWORD *v218; // rax
  __int64 v219; // rsi
  __int64 v220; // rcx
  __int64 v221; // rdx
  _QWORD *v222; // rax
  __int64 v223; // rsi
  __int64 v224; // rdi
  __int64 v225; // rcx
  _QWORD *v226; // rax
  __int64 v227; // rsi
  __int64 v228; // rdi
  __int64 v229; // rcx
  _QWORD *v230; // rax
  __int64 v231; // rsi
  __int64 v232; // rcx
  __int64 v233; // rdx
  _QWORD *v234; // rax
  __int64 v235; // rsi
  __int64 v236; // rcx
  __int64 v237; // rdx
  _QWORD *v238; // rax
  __int64 v239; // rsi
  _QWORD *v240; // rdx
  char v241; // r13
  _QWORD *v242; // rax
  __int64 v243; // rsi
  _QWORD *v244; // rdx
  _QWORD *v245; // rax
  __int64 v246; // rsi
  _QWORD *v247; // rdx
  _QWORD *v248; // rax
  _QWORD *v249; // rdx
  _QWORD *v250; // rax
  __int64 v251; // rsi
  __int64 v252; // rcx
  __int64 v253; // rdx
  _QWORD *v254; // rax
  __int64 v255; // rsi
  __int64 v256; // rdi
  __int64 v257; // rcx
  _QWORD *v258; // rax
  __int64 v259; // rsi
  __int64 v260; // rdi
  __int64 v261; // rcx
  _QWORD *v262; // rax
  __int64 v263; // rsi
  __int64 v264; // rcx
  __int64 v265; // rdx
  _QWORD *v266; // rax
  __int64 v267; // rsi
  __int64 v268; // rcx
  __int64 v269; // rdx
  __int64 v270; // rsi
  char *v271; // rdx
  _QWORD *v272; // rax
  __int64 v273; // rdi
  __int64 v274; // rcx
  _QWORD *v275; // rax
  unsigned __int64 v276; // rdx
  __int64 v277; // rsi
  __int64 v278; // rdi
  __int64 v279; // rcx
  __int64 v280; // rax
  __int64 v281; // rbx
  char v282; // r13
  __int64 v283; // rdi
  _QWORD *v284; // rax
  __int64 v285; // r8
  __int64 v286; // rcx
  __int64 v287; // rdx
  _QWORD *v288; // rax
  __int64 v289; // r8
  __int64 v290; // rcx
  __int64 v291; // rdx
  char v292; // al
  _QWORD *v293; // rax
  __int64 v294; // rsi
  __int64 v295; // rdi
  __int64 v296; // rcx
  _QWORD *v297; // rax
  unsigned __int64 v298; // rdx
  __int64 v299; // rsi
  __int64 v300; // rdi
  __int64 v301; // rcx
  __int64 v302; // rax
  _QWORD *v303; // r12
  unsigned __int64 v304; // rbx
  __int64 v305; // rdi
  _QWORD *v306; // rax
  __int64 v307; // r8
  __int64 v308; // rcx
  __int64 v309; // rdx
  _QWORD *v310; // rax
  __int64 v311; // r8
  __int64 v312; // rcx
  __int64 v313; // rdx
  char v314; // al
  _QWORD *v315; // r13
  __int64 v316; // rdi
  bool v317; // r12
  _QWORD *v318; // rax
  _QWORD *v319; // rdx
  unsigned __int64 v320; // rsi
  __int64 v321; // r9
  __int64 v322; // rdi
  __int64 v323; // rcx
  _QWORD *v324; // rax
  __int64 v325; // r9
  __int64 v326; // rcx
  __int64 v327; // rdx
  bool v328; // r13
  __int64 v329; // rax
  unsigned __int8 v330; // dl
  unsigned int v331; // r14d
  _BYTE **v332; // rax
  _BYTE *v333; // rax
  unsigned __int8 v334; // dl
  unsigned __int8 v335; // dl
  const char *v336; // r9
  size_t v337; // rdx
  size_t v338; // r12
  __int64 v339; // r8
  _BYTE *v340; // rax
  unsigned __int64 *v341; // rax
  size_t v342; // rdx
  unsigned __int8 *v343; // rsi
  __int64 v344; // rdi
  _BYTE *v345; // rax
  __int64 v346; // rdi
  _BYTE *v347; // rax
  __int64 v348; // rdi
  __m128i *v349; // rax
  __m128i si128; // xmm0
  __int64 v351; // r12
  void *v352; // rax
  _QWORD *v353; // rax
  __int64 v354; // rsi
  __int64 v355; // rdi
  __int64 v356; // rcx
  __int64 v357; // r12
  void *v358; // rax
  _QWORD *v359; // rax
  __int64 v360; // rsi
  __int64 v361; // rdi
  __int64 v362; // rcx
  _QWORD *v363; // r14
  __int64 v364; // r12
  void *v365; // rax
  _QWORD *v366; // rax
  __int64 v367; // rsi
  __int64 v368; // rdi
  __int64 v369; // rcx
  __int64 v370; // r12
  void *v371; // rax
  _QWORD *v372; // rdx
  __int64 v373; // rax
  __int64 v374; // rdi
  __int64 v375; // rsi
  _QWORD *v376; // rbx
  __int64 v377; // r12
  void *v378; // rax
  _BYTE *v379; // rdi
  _BYTE *v380; // rax
  size_t v381; // rdx
  unsigned __int8 *v382; // rsi
  __int64 v383; // r9
  _QWORD *v384; // rax
  unsigned __int64 v385; // rsi
  __int64 v386; // r9
  _QWORD *v387; // rdx
  __int64 v388; // r9
  _QWORD *v389; // rdx
  _QWORD *v390; // rax
  __int64 v391; // rdi
  __int64 v392; // rcx
  __int64 v393; // rsi
  _QWORD *v394; // rax
  __int64 v395; // rcx
  __int64 v396; // rdx
  __int64 v397; // r9
  _QWORD *v398; // rax
  __int64 v399; // rcx
  __int64 v400; // rdx
  __int64 v401; // rsi
  _QWORD *v402; // rax
  __int64 v403; // rdi
  __int64 v404; // rcx
  __int64 v405; // rax
  __int64 v406; // rax
  unsigned __int64 *v407; // rax
  unsigned __int64 *v408; // rdi
  __int64 v409; // rax
  __int64 v410; // [rsp+0h] [rbp-150h]
  __int64 v412; // [rsp+10h] [rbp-140h]
  bool v413; // [rsp+1Dh] [rbp-133h]
  char v414; // [rsp+1Eh] [rbp-132h]
  char v415; // [rsp+1Fh] [rbp-131h]
  _QWORD *v416; // [rsp+20h] [rbp-130h]
  char v417; // [rsp+28h] [rbp-128h]
  char v418; // [rsp+28h] [rbp-128h]
  _QWORD *v419; // [rsp+28h] [rbp-128h]
  char src; // [rsp+30h] [rbp-120h]
  char srca; // [rsp+30h] [rbp-120h]
  _QWORD *srcb; // [rsp+30h] [rbp-120h]
  const char *srcc; // [rsp+30h] [rbp-120h]
  _QWORD *v424; // [rsp+38h] [rbp-118h]
  char v425; // [rsp+40h] [rbp-110h]
  int v426; // [rsp+40h] [rbp-110h]
  _QWORD *v427; // [rsp+40h] [rbp-110h]
  _QWORD *v428; // [rsp+40h] [rbp-110h]
  __int64 v429; // [rsp+40h] [rbp-110h]
  __int64 v430; // [rsp+40h] [rbp-110h]
  const char *v431; // [rsp+40h] [rbp-110h]
  char v432; // [rsp+48h] [rbp-108h]
  char v433; // [rsp+48h] [rbp-108h]
  char v434; // [rsp+48h] [rbp-108h]
  char v435; // [rsp+48h] [rbp-108h]
  __int64 v436; // [rsp+48h] [rbp-108h]
  unsigned __int64 v437; // [rsp+50h] [rbp-100h]
  unsigned __int64 v438; // [rsp+50h] [rbp-100h]
  unsigned __int64 v439; // [rsp+50h] [rbp-100h]
  __int64 v440; // [rsp+50h] [rbp-100h]
  _QWORD *v441; // [rsp+50h] [rbp-100h]
  __int64 v442; // [rsp+50h] [rbp-100h]
  char v443; // [rsp+58h] [rbp-F8h]
  _QWORD *v444; // [rsp+58h] [rbp-F8h]
  char v445; // [rsp+58h] [rbp-F8h]
  char v446; // [rsp+58h] [rbp-F8h]
  _QWORD *v447; // [rsp+58h] [rbp-F8h]
  _QWORD *v448; // [rsp+58h] [rbp-F8h]
  _QWORD *v449; // [rsp+60h] [rbp-F0h]
  _QWORD *v450; // [rsp+68h] [rbp-E8h]
  _QWORD *v451; // [rsp+70h] [rbp-E0h]
  _QWORD *v452; // [rsp+78h] [rbp-D8h]
  _QWORD *v453; // [rsp+80h] [rbp-D0h]
  _QWORD *v454; // [rsp+88h] [rbp-C8h]
  _QWORD *v455; // [rsp+90h] [rbp-C0h]
  unsigned __int64 *v456; // [rsp+90h] [rbp-C0h]
  __int64 v457; // [rsp+90h] [rbp-C0h]
  _QWORD *v458; // [rsp+90h] [rbp-C0h]
  _QWORD *v459; // [rsp+90h] [rbp-C0h]
  _QWORD *v460; // [rsp+90h] [rbp-C0h]
  _QWORD *v461; // [rsp+98h] [rbp-B8h]
  _QWORD *v462; // [rsp+A0h] [rbp-B0h]
  __int64 v463; // [rsp+A8h] [rbp-A8h]
  __int64 v464; // [rsp+B0h] [rbp-A0h]
  __int64 v465; // [rsp+B8h] [rbp-98h]
  __int64 v466; // [rsp+C8h] [rbp-88h]
  __int64 v467; // [rsp+D0h] [rbp-80h]
  __int64 v468; // [rsp+D8h] [rbp-78h]
  unsigned __int64 v469; // [rsp+E0h] [rbp-70h] BYREF
  _QWORD *v470; // [rsp+E8h] [rbp-68h] BYREF
  unsigned __int64 v471; // [rsp+F0h] [rbp-60h] BYREF
  size_t v472; // [rsp+F8h] [rbp-58h] BYREF
  unsigned __int8 *v473; // [rsp+100h] [rbp-50h] BYREF
  size_t v474; // [rsp+108h] [rbp-48h]
  _QWORD v475[8]; // [rsp+110h] [rbp-40h] BYREF

  v2 = a1;
  v413 = 0;
  a1[1] = sub_BD5D20(a2);
  v462 = a1 + 39;
  v452 = a1 + 45;
  v451 = a1 + 51;
  v449 = a1 + 57;
  v461 = a1 + 15;
  v450 = a1 + 21;
  v454 = a1 + 27;
  v453 = a1 + 33;
  v424 = (_QWORD *)(a2 + 72);
  v466 = (__int64)(a1 + 34);
  v467 = (__int64)(a1 + 28);
  v465 = (__int64)(a1 + 22);
  v412 = (__int64)(a1 + 16);
  v463 = (__int64)(a1 + 58);
  v464 = (__int64)(a1 + 52);
  v468 = (__int64)(a1 + 46);
  v410 = (__int64)(a1 + 40);
  a1[2] = v3;
LABEL_2:
  sub_2C84640(v2, a2, 1);
  v4 = v2[17];
  while ( v4 )
  {
    sub_2C84080(*(_QWORD *)(v4 + 24));
    v5 = v4;
    v4 = *(_QWORD *)(v4 + 16);
    j_j___libc_free_0(v5);
  }
  v6 = v2[23];
  v2[17] = 0;
  v2[20] = 0;
  v2[18] = v412;
  v2[19] = v412;
  sub_2C84080(v6);
  v7 = v2[29];
  v2[23] = 0;
  v2[26] = 0;
  v2[24] = v465;
  v2[25] = v465;
  while ( v7 )
  {
    sub_2C84080(*(_QWORD *)(v7 + 24));
    v8 = v7;
    v7 = *(_QWORD *)(v7 + 16);
    j_j___libc_free_0(v8);
  }
  v9 = v2[35];
  v2[29] = 0;
  v2[32] = 0;
  v2[30] = v467;
  v2[31] = v467;
  sub_2C84080(v9);
  v2[35] = 0;
  v2[38] = 0;
  v2[36] = v466;
  v2[37] = v466;
  v10 = *(_QWORD **)(a2 + 80);
  if ( v10 != v424 )
  {
    while ( 1 )
    {
      v11 = (unsigned __int64)(v10 - 3);
      v12 = (_QWORD *)v2[17];
      v13 = v412;
      if ( !v10 )
        v11 = 0;
      v472 = v11;
      if ( !v12 )
        goto LABEL_16;
      do
      {
        while ( 1 )
        {
          v14 = v12[2];
          v15 = v12[3];
          if ( v12[4] >= v11 )
            break;
          v12 = (_QWORD *)v12[3];
          if ( !v15 )
            goto LABEL_14;
        }
        v13 = (__int64)v12;
        v12 = (_QWORD *)v12[2];
      }
      while ( v14 );
LABEL_14:
      if ( v13 == v412 || *(_QWORD *)(v13 + 32) > v11 )
      {
LABEL_16:
        v473 = (unsigned __int8 *)&v472;
        v13 = sub_2C84590(v461, v13, (unsigned __int64 **)&v473);
      }
      *(_BYTE *)(v13 + 40) = 0;
      v16 = (_QWORD *)v2[23];
      if ( !v16 )
        break;
      v17 = v465;
      do
      {
        while ( 1 )
        {
          v18 = v16[2];
          v19 = v16[3];
          if ( v16[4] >= v472 )
            break;
          v16 = (_QWORD *)v16[3];
          if ( !v19 )
            goto LABEL_22;
        }
        v17 = (__int64)v16;
        v16 = (_QWORD *)v16[2];
      }
      while ( v18 );
LABEL_22:
      if ( v17 == v465 || *(_QWORD *)(v17 + 32) > v472 )
        goto LABEL_24;
LABEL_25:
      *(_BYTE *)(v17 + 40) = 0;
      v20 = (_QWORD *)v2[29];
      if ( !v20 )
      {
        v21 = v467;
LABEL_32:
        v473 = (unsigned __int8 *)&v472;
        v21 = sub_2C84590(v454, v21, (unsigned __int64 **)&v473);
        goto LABEL_33;
      }
      v21 = v467;
      do
      {
        while ( 1 )
        {
          v22 = v20[2];
          v23 = v20[3];
          if ( v20[4] >= v472 )
            break;
          v20 = (_QWORD *)v20[3];
          if ( !v23 )
            goto LABEL_30;
        }
        v21 = (__int64)v20;
        v20 = (_QWORD *)v20[2];
      }
      while ( v22 );
LABEL_30:
      if ( v21 == v467 || *(_QWORD *)(v21 + 32) > v472 )
        goto LABEL_32;
LABEL_33:
      *(_BYTE *)(v21 + 40) = 0;
      v24 = (_QWORD *)v2[35];
      if ( v24 )
      {
        v25 = v466;
        do
        {
          while ( 1 )
          {
            v26 = v24[2];
            v27 = v24[3];
            if ( v24[4] >= v472 )
              break;
            v24 = (_QWORD *)v24[3];
            if ( !v27 )
              goto LABEL_38;
          }
          v25 = (__int64)v24;
          v24 = (_QWORD *)v24[2];
        }
        while ( v26 );
LABEL_38:
        if ( v25 != v466 && *(_QWORD *)(v25 + 32) <= v472 )
          goto LABEL_41;
      }
      else
      {
        v25 = v466;
      }
      v473 = (unsigned __int8 *)&v472;
      v25 = sub_2C84590(v453, v25, (unsigned __int64 **)&v473);
LABEL_41:
      *(_BYTE *)(v25 + 40) = 0;
      v10 = (_QWORD *)v10[1];
      if ( v10 == v424 )
      {
        v28 = v2;
        while ( 1 )
        {
          v29 = *(_QWORD **)(a2 + 80);
          if ( v29 == v424 )
            goto LABEL_177;
          v425 = 0;
          v30 = &v471;
          do
          {
            v31 = (unsigned __int64)(v29 - 3);
            v32 = v28[17];
            if ( !v29 )
              v31 = 0;
            v471 = v31;
            if ( !v32 )
            {
              v32 = v412;
LABEL_55:
              v473 = (unsigned __int8 *)v30;
              v32 = sub_2C84590(v461, v32, (unsigned __int64 **)&v473);
              goto LABEL_56;
            }
            v33 = v412;
            while ( 1 )
            {
              v34 = *(_QWORD *)(v32 + 16);
              v35 = *(_QWORD *)(v32 + 24);
              if ( *(_QWORD *)(v32 + 32) < v31 )
              {
                v32 = v33;
                v34 = v35;
              }
              if ( !v34 )
                break;
              v33 = v32;
              v32 = v34;
            }
            if ( v32 == v412 || *(_QWORD *)(v32 + 32) > v31 )
              goto LABEL_55;
LABEL_56:
            v36 = *(_BYTE *)(v32 + 40);
            v37 = v28[23];
            v443 = v36;
            if ( !v37 )
            {
              v37 = v465;
LABEL_64:
              v473 = (unsigned __int8 *)v30;
              v37 = sub_2C84590(v450, v37, (unsigned __int64 **)&v473);
              goto LABEL_65;
            }
            v38 = v465;
            while ( 1 )
            {
              v39 = *(_QWORD *)(v37 + 16);
              v40 = *(_QWORD *)(v37 + 24);
              if ( *(_QWORD *)(v37 + 32) < v471 )
              {
                v37 = v38;
                v39 = v40;
              }
              if ( !v39 )
                break;
              v38 = v37;
              v37 = v39;
            }
            if ( v37 == v465 || *(_QWORD *)(v37 + 32) > v471 )
              goto LABEL_64;
LABEL_65:
            v41 = *(_BYTE *)(v37 + 40);
            v42 = v28[29];
            v432 = v41;
            if ( !v42 )
            {
              v42 = v467;
LABEL_73:
              v473 = (unsigned __int8 *)v30;
              v42 = sub_2C84590(v454, v42, (unsigned __int64 **)&v473);
              goto LABEL_74;
            }
            v43 = v467;
            while ( 1 )
            {
              v44 = *(_QWORD *)(v42 + 16);
              v45 = *(_QWORD *)(v42 + 24);
              if ( *(_QWORD *)(v42 + 32) < v471 )
              {
                v42 = v43;
                v44 = v45;
              }
              if ( !v44 )
                break;
              v43 = v42;
              v42 = v44;
            }
            if ( v42 == v467 || *(_QWORD *)(v42 + 32) > v471 )
              goto LABEL_73;
LABEL_74:
            v46 = *(_BYTE *)(v42 + 40);
            v47 = v28[35];
            src = v46;
            if ( !v47 )
            {
              v47 = v466;
LABEL_82:
              v473 = (unsigned __int8 *)v30;
              v47 = sub_2C84590(v453, v47, (unsigned __int64 **)&v473);
              goto LABEL_83;
            }
            v48 = v466;
            while ( 1 )
            {
              v49 = *(_QWORD *)(v47 + 16);
              v50 = *(_QWORD *)(v47 + 24);
              if ( *(_QWORD *)(v47 + 32) < v471 )
              {
                v47 = v48;
                v49 = v50;
              }
              if ( !v49 )
                break;
              v48 = v47;
              v47 = v49;
            }
            if ( v47 == v466 || *(_QWORD *)(v47 + 32) > v471 )
              goto LABEL_82;
LABEL_83:
            v51 = *(_QWORD *)(v31 + 16);
            v417 = *(_BYTE *)(v47 + 40);
            if ( v51 )
            {
              while ( 1 )
              {
                v52 = *(_QWORD *)(v51 + 24);
                if ( (unsigned __int8)(*(_BYTE *)v52 - 30) <= 0xAu )
                  break;
                v51 = *(_QWORD *)(v51 + 8);
                if ( !v51 )
                  goto LABEL_126;
              }
              v455 = v29;
              while ( 1 )
              {
                v53 = *(_QWORD *)(v52 + 40);
                v54 = (_QWORD *)v28[29];
                v55 = v467;
                v472 = v53;
                if ( !v54 )
                  goto LABEL_94;
                while ( 1 )
                {
                  v56 = (_QWORD *)v54[3];
                  if ( v54[4] >= v53 )
                  {
                    v56 = (_QWORD *)v54[2];
                    v55 = (__int64)v54;
                  }
                  if ( !v56 )
                    break;
                  v54 = v56;
                }
                if ( v467 == v55 || *(_QWORD *)(v55 + 32) > v53 )
                {
LABEL_94:
                  v473 = (unsigned __int8 *)&v472;
                  v55 = sub_2C84590(v454, v55, (unsigned __int64 **)&v473);
                }
                v57 = (_QWORD *)v28[17];
                v58 = *(_BYTE *)(v55 + 40);
                if ( !v57 )
                  break;
                v59 = v412;
                while ( 1 )
                {
                  v60 = (_QWORD *)v57[3];
                  if ( v57[4] >= v471 )
                  {
                    v60 = (_QWORD *)v57[2];
                    v59 = (__int64)v57;
                  }
                  if ( !v60 )
                    break;
                  v57 = v60;
                }
                if ( v59 == v412 || *(_QWORD *)(v59 + 32) > v471 )
                  goto LABEL_103;
LABEL_104:
                *(_BYTE *)(v59 + 40) |= v58;
                v61 = (_QWORD *)v28[35];
                if ( !v61 )
                {
                  v62 = v466;
LABEL_112:
                  v473 = (unsigned __int8 *)&v472;
                  v62 = sub_2C84590(v453, v62, (unsigned __int64 **)&v473);
                  goto LABEL_113;
                }
                v62 = v466;
                while ( 1 )
                {
                  v63 = (_QWORD *)v61[3];
                  if ( v61[4] >= v472 )
                  {
                    v63 = (_QWORD *)v61[2];
                    v62 = (__int64)v61;
                  }
                  if ( !v63 )
                    break;
                  v61 = v63;
                }
                if ( v62 == v466 || *(_QWORD *)(v62 + 32) > v472 )
                  goto LABEL_112;
LABEL_113:
                v64 = (_QWORD *)v28[23];
                v65 = *(_BYTE *)(v62 + 40);
                if ( !v64 )
                {
                  v66 = v465;
LABEL_121:
                  v473 = (unsigned __int8 *)v30;
                  v66 = sub_2C84590(v450, v66, (unsigned __int64 **)&v473);
                  goto LABEL_122;
                }
                v66 = v465;
                while ( 1 )
                {
                  v67 = (_QWORD *)v64[3];
                  if ( v64[4] >= v471 )
                  {
                    v67 = (_QWORD *)v64[2];
                    v66 = (__int64)v64;
                  }
                  if ( !v67 )
                    break;
                  v64 = v67;
                }
                if ( v66 == v465 || *(_QWORD *)(v66 + 32) > v471 )
                  goto LABEL_121;
LABEL_122:
                *(_BYTE *)(v66 + 40) |= v65;
                v51 = *(_QWORD *)(v51 + 8);
                if ( !v51 )
                {
LABEL_125:
                  v29 = v455;
                  goto LABEL_126;
                }
                while ( 1 )
                {
                  v52 = *(_QWORD *)(v51 + 24);
                  if ( (unsigned __int8)(*(_BYTE *)v52 - 30) <= 0xAu )
                    break;
                  v51 = *(_QWORD *)(v51 + 8);
                  if ( !v51 )
                    goto LABEL_125;
                }
              }
              v59 = v412;
LABEL_103:
              v473 = (unsigned __int8 *)v30;
              v59 = sub_2C84590(v461, v59, (unsigned __int64 **)&v473);
              goto LABEL_104;
            }
LABEL_126:
            v437 = v471 + 48;
            if ( *(_QWORD *)(v471 + 56) != v471 + 48 )
            {
              v456 = v30;
              v68 = v28;
              v69 = 0;
              v70 = *(_QWORD *)(v471 + 56);
              do
              {
                v71 = v70 - 24;
                if ( !v70 )
                  v71 = 0;
                v72 = sub_2C83D20(v71);
                v70 = *(_QWORD *)(v70 + 8);
                v69 -= !v72 - 1;
              }
              while ( v437 != v70 );
              v73 = v69;
              v28 = v68;
              v30 = v456;
              if ( v73 )
              {
                v74 = (_QWORD *)v28[5];
                if ( !v74 )
                {
                  v75 = (__int64)(v28 + 4);
                  goto LABEL_139;
                }
                v75 = (__int64)(v28 + 4);
                do
                {
                  while ( 1 )
                  {
                    v76 = v74[2];
                    v77 = v74[3];
                    if ( v74[4] >= v471 )
                      break;
                    v74 = (_QWORD *)v74[3];
                    if ( !v77 )
                      goto LABEL_137;
                  }
                  v75 = (__int64)v74;
                  v74 = (_QWORD *)v74[2];
                }
                while ( v76 );
LABEL_137:
                if ( v28 + 4 == (_QWORD *)v75 || *(_QWORD *)(v75 + 32) > v471 )
                {
LABEL_139:
                  v473 = (unsigned __int8 *)v456;
                  v75 = sub_2C84590(v28 + 3, v75, (unsigned __int64 **)&v473);
                }
                v78 = (_QWORD *)v28[29];
                v79 = *(_BYTE *)(v75 + 40);
                if ( !v78 )
                {
                  v80 = v467;
                  goto LABEL_147;
                }
                v80 = v467;
                do
                {
                  while ( 1 )
                  {
                    v81 = v78[2];
                    v82 = v78[3];
                    if ( v78[4] >= v471 )
                      break;
                    v78 = (_QWORD *)v78[3];
                    if ( !v82 )
                      goto LABEL_145;
                  }
                  v80 = (__int64)v78;
                  v78 = (_QWORD *)v78[2];
                }
                while ( v81 );
LABEL_145:
                if ( v80 == v467 || *(_QWORD *)(v80 + 32) > v471 )
                {
LABEL_147:
                  v473 = (unsigned __int8 *)v456;
                  v80 = sub_2C84590(v454, v80, (unsigned __int64 **)&v473);
                }
                *(_BYTE *)(v80 + 40) = v79;
                v83 = (_QWORD *)v28[11];
                if ( !v83 )
                {
                  v84 = (__int64)(v28 + 10);
                  goto LABEL_155;
                }
                v84 = (__int64)(v28 + 10);
                do
                {
                  while ( 1 )
                  {
                    v85 = v83[2];
                    v86 = v83[3];
                    if ( v83[4] >= v471 )
                      break;
                    v83 = (_QWORD *)v83[3];
                    if ( !v86 )
                      goto LABEL_153;
                  }
                  v84 = (__int64)v83;
                  v83 = (_QWORD *)v83[2];
                }
                while ( v85 );
LABEL_153:
                if ( v28 + 10 == (_QWORD *)v84 || *(_QWORD *)(v84 + 32) > v471 )
                {
LABEL_155:
                  v473 = (unsigned __int8 *)v456;
                  v84 = sub_2C84590(v28 + 9, v84, (unsigned __int64 **)&v473);
                }
                v87 = (_QWORD *)v28[35];
                v88 = *(_BYTE *)(v84 + 40);
                if ( v87 )
                {
                  v89 = v466;
                  do
                  {
                    while ( 1 )
                    {
                      v90 = v87[2];
                      v91 = v87[3];
                      if ( v87[4] >= v471 )
                        break;
                      v87 = (_QWORD *)v87[3];
                      if ( !v91 )
                        goto LABEL_161;
                    }
                    v89 = (__int64)v87;
                    v87 = (_QWORD *)v87[2];
                  }
                  while ( v90 );
LABEL_161:
                  if ( v89 == v466 || *(_QWORD *)(v89 + 32) > v471 )
                  {
LABEL_163:
                    v473 = (unsigned __int8 *)v30;
                    v89 = sub_2C84590(v453, v89, (unsigned __int64 **)&v473);
                  }
                  *(_BYTE *)(v89 + 40) = v88;
                  v92 = v28[17];
                  if ( !v92 )
                    goto LABEL_407;
                  goto LABEL_165;
                }
LABEL_445:
                v89 = v466;
                goto LABEL_163;
              }
            }
            v206 = (_QWORD *)v28[17];
            if ( !v206 )
            {
              v207 = v412;
LABEL_377:
              v473 = (unsigned __int8 *)v30;
              v207 = sub_2C84590(v461, v207, (unsigned __int64 **)&v473);
              goto LABEL_378;
            }
            v207 = v412;
            while ( 1 )
            {
              v208 = (_QWORD *)v206[3];
              if ( v206[4] >= v471 )
              {
                v208 = (_QWORD *)v206[2];
                v207 = (__int64)v206;
              }
              if ( !v208 )
                break;
              v206 = v208;
            }
            if ( v207 == v412 || *(_QWORD *)(v207 + 32) > v471 )
              goto LABEL_377;
LABEL_378:
            v209 = *(_BYTE *)(v207 + 40);
            if ( v209 )
            {
              v210 = (_QWORD *)v28[29];
              if ( !v210 )
                goto LABEL_455;
              goto LABEL_380;
            }
            v234 = (_QWORD *)v28[5];
            if ( !v234 )
            {
              v235 = (__int64)(v28 + 4);
LABEL_453:
              v473 = (unsigned __int8 *)v30;
              v235 = sub_2C84590(v28 + 3, v235, (unsigned __int64 **)&v473);
              goto LABEL_454;
            }
            v235 = (__int64)(v28 + 4);
            do
            {
              while ( 1 )
              {
                v236 = v234[2];
                v237 = v234[3];
                if ( v234[4] >= v471 )
                  break;
                v234 = (_QWORD *)v234[3];
                if ( !v237 )
                  goto LABEL_451;
              }
              v235 = (__int64)v234;
              v234 = (_QWORD *)v234[2];
            }
            while ( v236 );
LABEL_451:
            if ( (_QWORD *)v235 == v28 + 4 || *(_QWORD *)(v235 + 32) > v471 )
              goto LABEL_453;
LABEL_454:
            v210 = (_QWORD *)v28[29];
            v209 = *(_BYTE *)(v235 + 40);
            if ( !v210 )
            {
LABEL_455:
              v211 = v467;
LABEL_387:
              v473 = (unsigned __int8 *)v30;
              v211 = sub_2C84590(v454, v211, (unsigned __int64 **)&v473);
              goto LABEL_388;
            }
LABEL_380:
            v211 = v467;
            while ( 1 )
            {
              v212 = (_QWORD *)v210[3];
              if ( v210[4] >= v471 )
              {
                v212 = (_QWORD *)v210[2];
                v211 = (__int64)v210;
              }
              if ( !v212 )
                break;
              v210 = v212;
            }
            if ( v211 == v467 || *(_QWORD *)(v211 + 32) > v471 )
              goto LABEL_387;
LABEL_388:
            *(_BYTE *)(v211 + 40) = v209;
            v213 = (_QWORD *)v28[23];
            if ( !v213 )
            {
              v214 = v465;
LABEL_396:
              v473 = (unsigned __int8 *)v30;
              v214 = sub_2C84590(v450, v214, (unsigned __int64 **)&v473);
              goto LABEL_397;
            }
            v214 = v465;
            while ( 1 )
            {
              v215 = (_QWORD *)v213[3];
              if ( v213[4] >= v471 )
              {
                v215 = (_QWORD *)v213[2];
                v214 = (__int64)v213;
              }
              if ( !v215 )
                break;
              v213 = v215;
            }
            if ( v214 == v465 || *(_QWORD *)(v214 + 32) > v471 )
              goto LABEL_396;
LABEL_397:
            v88 = *(_BYTE *)(v214 + 40);
            if ( v88 )
            {
              v216 = (_QWORD *)v28[35];
              if ( !v216 )
                goto LABEL_445;
              goto LABEL_399;
            }
            v230 = (_QWORD *)v28[11];
            if ( !v230 )
            {
              v231 = (__int64)(v28 + 10);
LABEL_443:
              v473 = (unsigned __int8 *)v30;
              v231 = sub_2C84590(v28 + 9, v231, (unsigned __int64 **)&v473);
              goto LABEL_444;
            }
            v231 = (__int64)(v28 + 10);
            do
            {
              while ( 1 )
              {
                v232 = v230[2];
                v233 = v230[3];
                if ( v230[4] >= v471 )
                  break;
                v230 = (_QWORD *)v230[3];
                if ( !v233 )
                  goto LABEL_441;
              }
              v231 = (__int64)v230;
              v230 = (_QWORD *)v230[2];
            }
            while ( v232 );
LABEL_441:
            if ( (_QWORD *)v231 == v28 + 10 || *(_QWORD *)(v231 + 32) > v471 )
              goto LABEL_443;
LABEL_444:
            v216 = (_QWORD *)v28[35];
            v88 = *(_BYTE *)(v231 + 40);
            if ( !v216 )
              goto LABEL_445;
LABEL_399:
            v89 = v466;
            while ( 1 )
            {
              v217 = (_QWORD *)v216[3];
              if ( v216[4] >= v471 )
              {
                v217 = (_QWORD *)v216[2];
                v89 = (__int64)v216;
              }
              if ( !v217 )
                break;
              v216 = v217;
            }
            if ( v89 == v466 || *(_QWORD *)(v89 + 32) > v471 )
              goto LABEL_163;
            *(_BYTE *)(v89 + 40) = v88;
            v92 = v28[17];
            if ( !v92 )
            {
LABEL_407:
              v92 = v412;
LABEL_172:
              v473 = (unsigned __int8 *)v30;
              v92 = sub_2C84590(v461, v92, (unsigned __int64 **)&v473);
              goto LABEL_173;
            }
LABEL_165:
            v93 = v412;
            while ( 1 )
            {
              v94 = *(_QWORD *)(v92 + 16);
              v95 = *(_QWORD *)(v92 + 24);
              if ( *(_QWORD *)(v92 + 32) < v471 )
              {
                v92 = v93;
                v94 = v95;
              }
              if ( !v94 )
                break;
              v93 = v92;
              v92 = v94;
            }
            if ( v92 == v412 || *(_QWORD *)(v92 + 32) > v471 )
              goto LABEL_172;
LABEL_173:
            if ( v443 != *(_BYTE *)(v92 + 40) )
              goto LABEL_174;
            v218 = (_QWORD *)v28[23];
            if ( !v218 )
            {
              v219 = v465;
LABEL_415:
              v473 = (unsigned __int8 *)v30;
              v219 = sub_2C84590(v450, v219, (unsigned __int64 **)&v473);
              goto LABEL_416;
            }
            v219 = v465;
            do
            {
              while ( 1 )
              {
                v220 = v218[2];
                v221 = v218[3];
                if ( v218[4] >= v471 )
                  break;
                v218 = (_QWORD *)v218[3];
                if ( !v221 )
                  goto LABEL_413;
              }
              v219 = (__int64)v218;
              v218 = (_QWORD *)v218[2];
            }
            while ( v220 );
LABEL_413:
            if ( v219 == v465 || *(_QWORD *)(v219 + 32) > v471 )
              goto LABEL_415;
LABEL_416:
            if ( v432 != *(_BYTE *)(v219 + 40) )
              goto LABEL_174;
            v222 = (_QWORD *)v28[29];
            if ( !v222 )
            {
              v223 = v467;
LABEL_424:
              v473 = (unsigned __int8 *)v30;
              v223 = sub_2C84590(v454, v223, (unsigned __int64 **)&v473);
              goto LABEL_425;
            }
            v223 = v467;
            do
            {
              while ( 1 )
              {
                v224 = v222[2];
                v225 = v222[3];
                if ( v222[4] >= v471 )
                  break;
                v222 = (_QWORD *)v222[3];
                if ( !v225 )
                  goto LABEL_422;
              }
              v223 = (__int64)v222;
              v222 = (_QWORD *)v222[2];
            }
            while ( v224 );
LABEL_422:
            if ( v223 == v467 || *(_QWORD *)(v223 + 32) > v471 )
              goto LABEL_424;
LABEL_425:
            if ( src != *(_BYTE *)(v223 + 40) )
              goto LABEL_174;
            v226 = (_QWORD *)v28[35];
            if ( !v226 )
            {
              v227 = v466;
LABEL_433:
              v473 = (unsigned __int8 *)v30;
              v227 = sub_2C84590(v453, v227, (unsigned __int64 **)&v473);
              goto LABEL_434;
            }
            v227 = v466;
            do
            {
              while ( 1 )
              {
                v228 = v226[2];
                v229 = v226[3];
                if ( v226[4] >= v471 )
                  break;
                v226 = (_QWORD *)v226[3];
                if ( !v229 )
                  goto LABEL_431;
              }
              v227 = (__int64)v226;
              v226 = (_QWORD *)v226[2];
            }
            while ( v228 );
LABEL_431:
            if ( v227 == v466 || *(_QWORD *)(v227 + 32) > v471 )
              goto LABEL_433;
LABEL_434:
            if ( v417 != *(_BYTE *)(v227 + 40) )
LABEL_174:
              v425 = 1;
            v29 = (_QWORD *)v29[1];
          }
          while ( v29 != v424 );
          if ( !v425 )
          {
LABEL_177:
            v2 = v28;
            goto LABEL_178;
          }
        }
      }
    }
    v17 = v465;
LABEL_24:
    v473 = (unsigned __int8 *)&v472;
    v17 = sub_2C84590(v450, v17, (unsigned __int64 **)&v473);
    goto LABEL_25;
  }
LABEL_178:
  sub_2C84640(v2, a2, 0);
  v96 = v2[41];
  while ( v96 )
  {
    sub_2C84080(*(_QWORD *)(v96 + 24));
    v97 = v96;
    v96 = *(_QWORD *)(v96 + 16);
    j_j___libc_free_0(v97);
  }
  v98 = v2[47];
  v2[41] = 0;
  v2[44] = 0;
  v2[42] = v410;
  v2[43] = v410;
  while ( v98 )
  {
    sub_2C84080(*(_QWORD *)(v98 + 24));
    v99 = v98;
    v98 = *(_QWORD *)(v98 + 16);
    j_j___libc_free_0(v99);
  }
  v100 = v2[53];
  v2[47] = 0;
  v2[50] = 0;
  v2[48] = v468;
  v2[49] = v468;
  sub_2C84080(v100);
  v101 = v2[59];
  v2[53] = 0;
  v2[56] = 0;
  v2[54] = v464;
  v2[55] = v464;
  while ( v101 )
  {
    sub_2C84080(*(_QWORD *)(v101 + 24));
    v102 = v101;
    v101 = *(_QWORD *)(v101 + 16);
    j_j___libc_free_0(v102);
  }
  v2[59] = 0;
  v2[62] = 0;
  v2[60] = v463;
  v2[61] = v463;
  v103 = *(_QWORD **)(a2 + 80);
  if ( v103 == v424 )
    goto LABEL_363;
  do
  {
    v104 = (unsigned __int64)(v103 - 3);
    v105 = (_QWORD *)v2[41];
    v106 = v410;
    if ( !v103 )
      v104 = 0;
    v472 = v104;
    if ( !v105 )
      goto LABEL_194;
    do
    {
      while ( 1 )
      {
        v107 = v105[2];
        v108 = v105[3];
        if ( v105[4] >= v104 )
          break;
        v105 = (_QWORD *)v105[3];
        if ( !v108 )
          goto LABEL_192;
      }
      v106 = (__int64)v105;
      v105 = (_QWORD *)v105[2];
    }
    while ( v107 );
LABEL_192:
    if ( v410 == v106 || *(_QWORD *)(v106 + 32) > v104 )
    {
LABEL_194:
      v473 = (unsigned __int8 *)&v472;
      v106 = sub_2C84590(v462, v106, (unsigned __int64 **)&v473);
    }
    *(_BYTE *)(v106 + 40) = 0;
    v109 = (_QWORD *)v2[47];
    if ( !v109 )
    {
      v110 = v468;
LABEL_202:
      v473 = (unsigned __int8 *)&v472;
      v110 = sub_2C84590(v452, v110, (unsigned __int64 **)&v473);
      goto LABEL_203;
    }
    v110 = v468;
    do
    {
      while ( 1 )
      {
        v111 = v109[2];
        v112 = v109[3];
        if ( v109[4] >= v472 )
          break;
        v109 = (_QWORD *)v109[3];
        if ( !v112 )
          goto LABEL_200;
      }
      v110 = (__int64)v109;
      v109 = (_QWORD *)v109[2];
    }
    while ( v111 );
LABEL_200:
    if ( v468 == v110 || *(_QWORD *)(v110 + 32) > v472 )
      goto LABEL_202;
LABEL_203:
    *(_BYTE *)(v110 + 40) = 0;
    v113 = (_QWORD *)v2[53];
    if ( !v113 )
    {
      v114 = v464;
LABEL_210:
      v473 = (unsigned __int8 *)&v472;
      v114 = sub_2C84590(v451, v114, (unsigned __int64 **)&v473);
      goto LABEL_211;
    }
    v114 = v464;
    do
    {
      while ( 1 )
      {
        v115 = v113[2];
        v116 = v113[3];
        if ( v113[4] >= v472 )
          break;
        v113 = (_QWORD *)v113[3];
        if ( !v116 )
          goto LABEL_208;
      }
      v114 = (__int64)v113;
      v113 = (_QWORD *)v113[2];
    }
    while ( v115 );
LABEL_208:
    if ( v114 == v464 || *(_QWORD *)(v114 + 32) > v472 )
      goto LABEL_210;
LABEL_211:
    *(_BYTE *)(v114 + 40) = 0;
    v117 = (_QWORD *)v2[59];
    if ( !v117 )
    {
      v118 = v463;
LABEL_218:
      v473 = (unsigned __int8 *)&v472;
      v118 = sub_2C84590(v449, v118, (unsigned __int64 **)&v473);
      goto LABEL_219;
    }
    v118 = v463;
    do
    {
      while ( 1 )
      {
        v119 = v117[2];
        v120 = v117[3];
        if ( v117[4] >= v472 )
          break;
        v117 = (_QWORD *)v117[3];
        if ( !v120 )
          goto LABEL_216;
      }
      v118 = (__int64)v117;
      v117 = (_QWORD *)v117[2];
    }
    while ( v119 );
LABEL_216:
    if ( v118 == v463 || *(_QWORD *)(v118 + 32) > v472 )
      goto LABEL_218;
LABEL_219:
    *(_BYTE *)(v118 + 40) = 0;
    v103 = (_QWORD *)v103[1];
  }
  while ( v103 != v424 );
  v121 = v2;
  while ( 1 )
  {
    v416 = *(_QWORD **)(a2 + 80);
    if ( v416 == v424 )
      break;
    v444 = v424;
    v418 = 0;
    do
    {
      v122 = v121[41];
      v123 = (*v444 & 0xFFFFFFFFFFFFFFF8LL) == 0;
      v124 = (*v444 & 0xFFFFFFFFFFFFFFF8LL) - 24;
      v438 = *v444 & 0xFFFFFFFFFFFFFFF8LL;
      v444 = (_QWORD *)v438;
      if ( v123 )
        v124 = 0;
      v471 = v124;
      if ( !v122 )
      {
        v122 = v410;
LABEL_233:
        v473 = (unsigned __int8 *)&v471;
        v122 = sub_2C84590(v462, v122, (unsigned __int64 **)&v473);
        goto LABEL_234;
      }
      v125 = v410;
      while ( 1 )
      {
        v126 = *(_QWORD *)(v122 + 16);
        v127 = *(_QWORD *)(v122 + 24);
        if ( *(_QWORD *)(v122 + 32) < v124 )
        {
          v122 = v125;
          v126 = v127;
        }
        if ( !v126 )
          break;
        v125 = v122;
        v122 = v126;
      }
      if ( v410 == v122 || *(_QWORD *)(v122 + 32) > v124 )
        goto LABEL_233;
LABEL_234:
      v128 = *(_BYTE *)(v122 + 40);
      v129 = v121[47];
      v433 = v128;
      if ( !v129 )
      {
        v129 = v468;
LABEL_242:
        v473 = (unsigned __int8 *)&v471;
        v129 = sub_2C84590(v452, v129, (unsigned __int64 **)&v473);
        goto LABEL_243;
      }
      v130 = v468;
      while ( 1 )
      {
        v131 = *(_QWORD *)(v129 + 16);
        v132 = *(_QWORD *)(v129 + 24);
        if ( *(_QWORD *)(v129 + 32) < v471 )
        {
          v129 = v130;
          v131 = v132;
        }
        if ( !v131 )
          break;
        v130 = v129;
        v129 = v131;
      }
      if ( v129 == v468 || *(_QWORD *)(v129 + 32) > v471 )
        goto LABEL_242;
LABEL_243:
      v133 = *(_BYTE *)(v129 + 40);
      v134 = v121[53];
      srca = v133;
      if ( !v134 )
      {
        v134 = v464;
LABEL_251:
        v473 = (unsigned __int8 *)&v471;
        v134 = sub_2C84590(v451, v134, (unsigned __int64 **)&v473);
        goto LABEL_252;
      }
      v135 = v464;
      while ( 1 )
      {
        v136 = *(_QWORD *)(v134 + 16);
        v137 = *(_QWORD *)(v134 + 24);
        if ( *(_QWORD *)(v134 + 32) < v471 )
        {
          v134 = v135;
          v136 = v137;
        }
        if ( !v136 )
          break;
        v135 = v134;
        v134 = v136;
      }
      if ( v134 == v464 || *(_QWORD *)(v134 + 32) > v471 )
        goto LABEL_251;
LABEL_252:
      v138 = *(_BYTE *)(v134 + 40);
      v139 = v121[59];
      v415 = v138;
      if ( !v139 )
      {
        v139 = v463;
LABEL_260:
        v473 = (unsigned __int8 *)&v471;
        v139 = sub_2C84590(v449, v139, (unsigned __int64 **)&v473);
        goto LABEL_261;
      }
      v140 = v463;
      while ( 1 )
      {
        v141 = *(_QWORD *)(v139 + 16);
        v142 = *(_QWORD *)(v139 + 24);
        if ( *(_QWORD *)(v139 + 32) < v471 )
        {
          v139 = v140;
          v141 = v142;
        }
        if ( !v141 )
          break;
        v140 = v139;
        v139 = v141;
      }
      if ( v139 == v463 || *(_QWORD *)(v139 + 32) > v471 )
        goto LABEL_260;
LABEL_261:
      v143 = (_QWORD *)(v124 + 48);
      v414 = *(_BYTE *)(v139 + 40);
      v144 = *v143 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (_QWORD *)v144 != v143 )
      {
        if ( !v144 )
          BUG();
        v457 = v144 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v144 - 24) - 30 <= 0xA )
        {
          v426 = sub_B46E30(v457);
          if ( v426 )
          {
            v145 = 0;
            while ( 1 )
            {
              v146 = v410;
              v472 = sub_B46EC0(v457, v145);
              v147 = (_QWORD *)v121[41];
              if ( !v147 )
                goto LABEL_274;
              while ( 1 )
              {
                v148 = (_QWORD *)v147[3];
                if ( v147[4] >= v472 )
                {
                  v148 = (_QWORD *)v147[2];
                  v146 = (__int64)v147;
                }
                if ( !v148 )
                  break;
                v147 = v148;
              }
              if ( v410 == v146 || *(_QWORD *)(v146 + 32) > v472 )
              {
LABEL_274:
                v473 = (unsigned __int8 *)&v472;
                v146 = sub_2C84590(v462, v146, (unsigned __int64 **)&v473);
              }
              v149 = (_QWORD *)v121[53];
              v150 = *(_BYTE *)(v146 + 40);
              if ( !v149 )
                break;
              v151 = v464;
              while ( 1 )
              {
                v152 = (_QWORD *)v149[3];
                if ( v149[4] >= v471 )
                {
                  v152 = (_QWORD *)v149[2];
                  v151 = (__int64)v149;
                }
                if ( !v152 )
                  break;
                v149 = v152;
              }
              if ( v151 == v464 || *(_QWORD *)(v151 + 32) > v471 )
                goto LABEL_283;
LABEL_284:
              *(_BYTE *)(v151 + 40) |= v150;
              v153 = (_QWORD *)v121[47];
              if ( !v153 )
              {
                v154 = v468;
LABEL_292:
                v473 = (unsigned __int8 *)&v472;
                v154 = sub_2C84590(v452, v154, (unsigned __int64 **)&v473);
                goto LABEL_293;
              }
              v154 = v468;
              while ( 1 )
              {
                v155 = (_QWORD *)v153[3];
                if ( v153[4] >= v472 )
                {
                  v155 = (_QWORD *)v153[2];
                  v154 = (__int64)v153;
                }
                if ( !v155 )
                  break;
                v153 = v155;
              }
              if ( v468 == v154 || *(_QWORD *)(v154 + 32) > v472 )
                goto LABEL_292;
LABEL_293:
              v156 = (_QWORD *)v121[59];
              v157 = *(_BYTE *)(v154 + 40);
              if ( v156 )
              {
                v158 = v463;
                while ( 1 )
                {
                  v159 = (_QWORD *)v156[3];
                  if ( v156[4] >= v471 )
                  {
                    v159 = (_QWORD *)v156[2];
                    v158 = (__int64)v156;
                  }
                  if ( !v159 )
                    break;
                  v156 = v159;
                }
                if ( v158 != v463 && *(_QWORD *)(v158 + 32) <= v471 )
                  goto LABEL_302;
              }
              else
              {
                v158 = v463;
              }
              v473 = (unsigned __int8 *)&v471;
              v158 = sub_2C84590(v449, v158, (unsigned __int64 **)&v473);
LABEL_302:
              *(_BYTE *)(v158 + 40) |= v157;
              if ( v426 == ++v145 )
                goto LABEL_303;
            }
            v151 = v464;
LABEL_283:
            v473 = (unsigned __int8 *)&v471;
            v151 = sub_2C84590(v451, v151, (unsigned __int64 **)&v473);
            goto LABEL_284;
          }
        }
      }
LABEL_303:
      v160 = *(_QWORD *)(v471 + 56);
      v161 = v471 + 48;
      if ( v160 != v471 + 48 )
      {
        v458 = v121;
        v162 = 0;
        do
        {
          v163 = v160 - 24;
          if ( !v160 )
            v163 = 0;
          v164 = sub_2C83D20(v163);
          v160 = *(_QWORD *)(v160 + 8);
          v162 -= !v164 - 1;
        }
        while ( v161 != v160 );
        v165 = v162;
        v121 = v458;
        if ( v165 )
        {
          v166 = (_QWORD *)v458[5];
          if ( !v166 )
          {
            v167 = (__int64)(v458 + 4);
            goto LABEL_316;
          }
          v167 = (__int64)(v458 + 4);
          do
          {
            while ( 1 )
            {
              v168 = v166[2];
              v169 = v166[3];
              if ( v166[4] >= v471 )
                break;
              v166 = (_QWORD *)v166[3];
              if ( !v169 )
                goto LABEL_314;
            }
            v167 = (__int64)v166;
            v166 = (_QWORD *)v166[2];
          }
          while ( v168 );
LABEL_314:
          if ( v458 + 4 == (_QWORD *)v167 || *(_QWORD *)(v167 + 32) > v471 )
          {
LABEL_316:
            v473 = (unsigned __int8 *)&v471;
            v167 = sub_2C84590(v458 + 3, v167, (unsigned __int64 **)&v473);
          }
          v170 = (_QWORD *)v458[41];
          v171 = *(_BYTE *)(v167 + 40);
          if ( !v170 )
          {
            v172 = v410;
            goto LABEL_324;
          }
          v172 = v410;
          do
          {
            while ( 1 )
            {
              v173 = v170[2];
              v174 = v170[3];
              if ( v170[4] >= v471 )
                break;
              v170 = (_QWORD *)v170[3];
              if ( !v174 )
                goto LABEL_322;
            }
            v172 = (__int64)v170;
            v170 = (_QWORD *)v170[2];
          }
          while ( v173 );
LABEL_322:
          if ( v410 == v172 || *(_QWORD *)(v172 + 32) > v471 )
          {
LABEL_324:
            v473 = (unsigned __int8 *)&v471;
            v172 = sub_2C84590(v462, v172, (unsigned __int64 **)&v473);
          }
          *(_BYTE *)(v172 + 40) = v171;
          v175 = (_QWORD *)v458[11];
          if ( !v175 )
          {
            v176 = (__int64)(v458 + 10);
            goto LABEL_332;
          }
          v176 = (__int64)(v458 + 10);
          do
          {
            while ( 1 )
            {
              v177 = v175[2];
              v178 = v175[3];
              if ( v175[4] >= v471 )
                break;
              v175 = (_QWORD *)v175[3];
              if ( !v178 )
                goto LABEL_330;
            }
            v176 = (__int64)v175;
            v175 = (_QWORD *)v175[2];
          }
          while ( v177 );
LABEL_330:
          if ( v458 + 10 == (_QWORD *)v176 || *(_QWORD *)(v176 + 32) > v471 )
          {
LABEL_332:
            v473 = (unsigned __int8 *)&v471;
            v176 = sub_2C84590(v458 + 9, v176, (unsigned __int64 **)&v473);
          }
          v179 = (_QWORD *)v458[47];
          v180 = *(_BYTE *)(v176 + 40);
          if ( v179 )
          {
            v181 = v468;
            do
            {
              while ( 1 )
              {
                v182 = v179[2];
                v183 = v179[3];
                if ( v179[4] >= v471 )
                  break;
                v179 = (_QWORD *)v179[3];
                if ( !v183 )
                  goto LABEL_338;
              }
              v181 = (__int64)v179;
              v179 = (_QWORD *)v179[2];
            }
            while ( v182 );
LABEL_338:
            if ( v468 == v181 || *(_QWORD *)(v181 + 32) > v471 )
            {
LABEL_340:
              v473 = (unsigned __int8 *)&v471;
              v181 = sub_2C84590(v452, v181, (unsigned __int64 **)&v473);
            }
            *(_BYTE *)(v181 + 40) = v180;
            v184 = v121[41];
            if ( !v184 )
              goto LABEL_507;
            goto LABEL_342;
          }
LABEL_546:
          v181 = v468;
          goto LABEL_340;
        }
      }
      v238 = (_QWORD *)v121[53];
      if ( !v238 )
      {
        v239 = v464;
LABEL_477:
        v473 = (unsigned __int8 *)&v471;
        v239 = sub_2C84590(v451, v239, (unsigned __int64 **)&v473);
        goto LABEL_478;
      }
      v239 = v464;
      while ( 1 )
      {
        v240 = (_QWORD *)v238[3];
        if ( v238[4] >= v471 )
        {
          v240 = (_QWORD *)v238[2];
          v239 = (__int64)v238;
        }
        if ( !v240 )
          break;
        v238 = v240;
      }
      if ( v239 == v464 || *(_QWORD *)(v239 + 32) > v471 )
        goto LABEL_477;
LABEL_478:
      v241 = *(_BYTE *)(v239 + 40);
      if ( v241 )
      {
        v242 = (_QWORD *)v121[41];
        if ( !v242 )
          goto LABEL_556;
        goto LABEL_480;
      }
      v266 = (_QWORD *)v121[5];
      if ( !v266 )
      {
        v267 = (__int64)(v121 + 4);
LABEL_554:
        v473 = (unsigned __int8 *)&v471;
        v267 = sub_2C84590(v121 + 3, v267, (unsigned __int64 **)&v473);
        goto LABEL_555;
      }
      v267 = (__int64)(v121 + 4);
      do
      {
        while ( 1 )
        {
          v268 = v266[2];
          v269 = v266[3];
          if ( v266[4] >= v471 )
            break;
          v266 = (_QWORD *)v266[3];
          if ( !v269 )
            goto LABEL_552;
        }
        v267 = (__int64)v266;
        v266 = (_QWORD *)v266[2];
      }
      while ( v268 );
LABEL_552:
      if ( (_QWORD *)v267 == v121 + 4 || *(_QWORD *)(v267 + 32) > v471 )
        goto LABEL_554;
LABEL_555:
      v242 = (_QWORD *)v121[41];
      v241 = *(_BYTE *)(v267 + 40);
      if ( !v242 )
      {
LABEL_556:
        v243 = v410;
LABEL_487:
        v473 = (unsigned __int8 *)&v471;
        v243 = sub_2C84590(v462, v243, (unsigned __int64 **)&v473);
        goto LABEL_488;
      }
LABEL_480:
      v243 = v410;
      while ( 1 )
      {
        v244 = (_QWORD *)v242[3];
        if ( v242[4] >= v471 )
        {
          v244 = (_QWORD *)v242[2];
          v243 = (__int64)v242;
        }
        if ( !v244 )
          break;
        v242 = v244;
      }
      if ( v410 == v243 || *(_QWORD *)(v243 + 32) > v471 )
        goto LABEL_487;
LABEL_488:
      *(_BYTE *)(v243 + 40) = v241;
      v245 = (_QWORD *)v121[59];
      if ( !v245 )
      {
        v246 = v463;
LABEL_496:
        v473 = (unsigned __int8 *)&v471;
        v246 = sub_2C84590(v449, v246, (unsigned __int64 **)&v473);
        goto LABEL_497;
      }
      v246 = v463;
      while ( 1 )
      {
        v247 = (_QWORD *)v245[3];
        if ( v245[4] >= v471 )
        {
          v247 = (_QWORD *)v245[2];
          v246 = (__int64)v245;
        }
        if ( !v247 )
          break;
        v245 = v247;
      }
      if ( v246 == v463 || *(_QWORD *)(v246 + 32) > v471 )
        goto LABEL_496;
LABEL_497:
      v180 = *(_BYTE *)(v246 + 40);
      if ( v180 )
      {
        v248 = (_QWORD *)v121[47];
        if ( !v248 )
          goto LABEL_546;
        goto LABEL_499;
      }
      v262 = (_QWORD *)v121[11];
      if ( !v262 )
      {
        v263 = (__int64)(v121 + 10);
LABEL_544:
        v473 = (unsigned __int8 *)&v471;
        v263 = sub_2C84590(v121 + 9, v263, (unsigned __int64 **)&v473);
        goto LABEL_545;
      }
      v263 = (__int64)(v121 + 10);
      do
      {
        while ( 1 )
        {
          v264 = v262[2];
          v265 = v262[3];
          if ( v262[4] >= v471 )
            break;
          v262 = (_QWORD *)v262[3];
          if ( !v265 )
            goto LABEL_542;
        }
        v263 = (__int64)v262;
        v262 = (_QWORD *)v262[2];
      }
      while ( v264 );
LABEL_542:
      if ( (_QWORD *)v263 == v121 + 10 || *(_QWORD *)(v263 + 32) > v471 )
        goto LABEL_544;
LABEL_545:
      v248 = (_QWORD *)v121[47];
      v180 = *(_BYTE *)(v263 + 40);
      if ( !v248 )
        goto LABEL_546;
LABEL_499:
      v181 = v468;
      while ( 1 )
      {
        v249 = (_QWORD *)v248[3];
        if ( v248[4] >= v471 )
        {
          v249 = (_QWORD *)v248[2];
          v181 = (__int64)v248;
        }
        if ( !v249 )
          break;
        v248 = v249;
      }
      if ( v468 == v181 || *(_QWORD *)(v181 + 32) > v471 )
        goto LABEL_340;
      *(_BYTE *)(v181 + 40) = v180;
      v184 = v121[41];
      if ( !v184 )
      {
LABEL_507:
        v184 = v410;
LABEL_349:
        v473 = (unsigned __int8 *)&v471;
        v184 = sub_2C84590(v462, v184, (unsigned __int64 **)&v473);
        goto LABEL_350;
      }
LABEL_342:
      v185 = v410;
      while ( 1 )
      {
        v186 = *(_QWORD *)(v184 + 16);
        v187 = *(_QWORD *)(v184 + 24);
        if ( *(_QWORD *)(v184 + 32) < v471 )
        {
          v184 = v185;
          v186 = v187;
        }
        if ( !v186 )
          break;
        v185 = v184;
        v184 = v186;
      }
      if ( v410 == v184 || *(_QWORD *)(v184 + 32) > v471 )
        goto LABEL_349;
LABEL_350:
      if ( v433 != *(_BYTE *)(v184 + 40) )
        goto LABEL_351;
      v250 = (_QWORD *)v121[47];
      if ( !v250 )
      {
        v251 = v468;
LABEL_515:
        v473 = (unsigned __int8 *)&v471;
        v251 = sub_2C84590(v452, v251, (unsigned __int64 **)&v473);
        goto LABEL_516;
      }
      v251 = v468;
      do
      {
        while ( 1 )
        {
          v252 = v250[2];
          v253 = v250[3];
          if ( v250[4] >= v471 )
            break;
          v250 = (_QWORD *)v250[3];
          if ( !v253 )
            goto LABEL_513;
        }
        v251 = (__int64)v250;
        v250 = (_QWORD *)v250[2];
      }
      while ( v252 );
LABEL_513:
      if ( v468 == v251 || *(_QWORD *)(v251 + 32) > v471 )
        goto LABEL_515;
LABEL_516:
      if ( srca != *(_BYTE *)(v251 + 40) )
        goto LABEL_351;
      v254 = (_QWORD *)v121[53];
      if ( !v254 )
      {
        v255 = v464;
LABEL_524:
        v473 = (unsigned __int8 *)&v471;
        v255 = sub_2C84590(v451, v255, (unsigned __int64 **)&v473);
        goto LABEL_525;
      }
      v255 = v464;
      do
      {
        while ( 1 )
        {
          v256 = v254[2];
          v257 = v254[3];
          if ( v254[4] >= v471 )
            break;
          v254 = (_QWORD *)v254[3];
          if ( !v257 )
            goto LABEL_522;
        }
        v255 = (__int64)v254;
        v254 = (_QWORD *)v254[2];
      }
      while ( v256 );
LABEL_522:
      if ( v255 == v464 || *(_QWORD *)(v255 + 32) > v471 )
        goto LABEL_524;
LABEL_525:
      if ( v415 != *(_BYTE *)(v255 + 40) )
        goto LABEL_351;
      v258 = (_QWORD *)v121[59];
      if ( !v258 )
      {
        v259 = v463;
LABEL_533:
        v473 = (unsigned __int8 *)&v471;
        v259 = sub_2C84590(v449, v259, (unsigned __int64 **)&v473);
        goto LABEL_534;
      }
      v259 = v463;
      do
      {
        while ( 1 )
        {
          v260 = v258[2];
          v261 = v258[3];
          if ( v258[4] >= v471 )
            break;
          v258 = (_QWORD *)v258[3];
          if ( !v261 )
            goto LABEL_531;
        }
        v259 = (__int64)v258;
        v258 = (_QWORD *)v258[2];
      }
      while ( v260 );
LABEL_531:
      if ( v259 == v463 || *(_QWORD *)(v259 + 32) > v471 )
        goto LABEL_533;
LABEL_534:
      if ( v414 != *(_BYTE *)(v259 + 40) )
LABEL_351:
        v418 = 1;
    }
    while ( v416 != (_QWORD *)v438 );
    if ( !v418 )
    {
      v2 = v121;
      srcb = *(_QWORD **)(a2 + 80);
      if ( srcb == v424 )
        goto LABEL_363;
      v459 = v121 + 70;
      while ( 2 )
      {
        if ( !srcb )
          BUG();
        v188 = (_QWORD *)srcb[4];
        v189 = 0;
        if ( v188 == srcb + 3 )
          goto LABEL_362;
        do
        {
          v190 = (__int64)(v188 - 3);
          if ( !v188 )
            v190 = 0;
          v191 = sub_2C83D20(v190);
          v188 = (_QWORD *)v188[1];
          v189 -= !v191 - 1;
        }
        while ( srcb + 3 != v188 );
        if ( !v189 )
        {
LABEL_362:
          srcb = (_QWORD *)srcb[1];
          if ( srcb == v424 )
            goto LABEL_363;
          continue;
        }
        break;
      }
      v270 = v412;
      v271 = (char *)(srcb - 3);
      v272 = (_QWORD *)v2[17];
      v471 = (unsigned __int64)(srcb - 3);
      if ( !v272 )
        goto LABEL_574;
      do
      {
        while ( 1 )
        {
          v273 = v272[2];
          v274 = v272[3];
          if ( v272[4] >= (unsigned __int64)v271 )
            break;
          v272 = (_QWORD *)v272[3];
          if ( !v274 )
            goto LABEL_572;
        }
        v270 = (__int64)v272;
        v272 = (_QWORD *)v272[2];
      }
      while ( v273 );
LABEL_572:
      if ( v270 == v412 || *(_QWORD *)(v270 + 32) > (unsigned __int64)v271 )
      {
LABEL_574:
        v473 = (unsigned __int8 *)&v471;
        v270 = sub_2C84590(v461, v270, (unsigned __int64 **)&v473);
      }
      v434 = *(_BYTE *)(v270 + 40);
      v275 = (_QWORD *)v2[23];
      if ( !v275 )
      {
        v277 = v465;
        goto LABEL_582;
      }
      v276 = v471;
      v277 = v465;
      do
      {
        while ( 1 )
        {
          v278 = v275[2];
          v279 = v275[3];
          if ( v275[4] >= v471 )
            break;
          v275 = (_QWORD *)v275[3];
          if ( !v279 )
            goto LABEL_580;
        }
        v277 = (__int64)v275;
        v275 = (_QWORD *)v275[2];
      }
      while ( v278 );
LABEL_580:
      if ( v277 == v465 || *(_QWORD *)(v277 + 32) > v471 )
      {
LABEL_582:
        v473 = (unsigned __int8 *)&v471;
        v280 = sub_2C84590(v450, v277, (unsigned __int64 **)&v473);
        v276 = v471;
        v277 = v280;
      }
      v445 = *(_BYTE *)(v277 + 40);
      v439 = v276 + 48;
      if ( *(_QWORD *)(v276 + 56) == v276 + 48 )
        goto LABEL_607;
      v427 = v188;
      v281 = *(_QWORD *)(v276 + 56);
      v282 = v434;
      while ( 1 )
      {
LABEL_586:
        v283 = v281 - 24;
        if ( !v281 )
          v283 = 0;
        v472 = v283;
        if ( sub_2C83D20(v283) )
          break;
        LOBYTE(v470) = 0;
        LOBYTE(v473) = 0;
        sub_2C83AE0(v472, &v470, &v473);
        v281 = *(_QWORD *)(v281 + 8);
        v445 |= (unsigned __int8)v473;
        v282 |= (unsigned __int8)v470;
        if ( v439 == v281 )
          goto LABEL_606;
      }
      v284 = (_QWORD *)v2[65];
      if ( !v284 )
      {
        v285 = (__int64)(v2 + 64);
        goto LABEL_596;
      }
      v285 = (__int64)(v2 + 64);
      do
      {
        while ( 1 )
        {
          v286 = v284[2];
          v287 = v284[3];
          if ( v284[4] >= v472 )
            break;
          v284 = (_QWORD *)v284[3];
          if ( !v287 )
            goto LABEL_594;
        }
        v285 = (__int64)v284;
        v284 = (_QWORD *)v284[2];
      }
      while ( v286 );
LABEL_594:
      if ( v2 + 64 == (_QWORD *)v285 || *(_QWORD *)(v285 + 32) > v472 )
      {
LABEL_596:
        v473 = (unsigned __int8 *)&v472;
        v285 = sub_2C84AF0(v2 + 63, v285, (unsigned __int64 **)&v473);
      }
      *(_BYTE *)(v285 + 40) = v282;
      v288 = (_QWORD *)v2[71];
      if ( v288 )
      {
        v289 = (__int64)v459;
        do
        {
          while ( 1 )
          {
            v290 = v288[2];
            v291 = v288[3];
            if ( v288[4] >= v472 )
              break;
            v288 = (_QWORD *)v288[3];
            if ( !v291 )
              goto LABEL_602;
          }
          v289 = (__int64)v288;
          v288 = (_QWORD *)v288[2];
        }
        while ( v290 );
LABEL_602:
        if ( (_QWORD *)v289 == v459 || *(_QWORD *)(v289 + 32) > v472 )
        {
LABEL_604:
          v473 = (unsigned __int8 *)&v472;
          v289 = sub_2C84AF0(v2 + 69, v289, (unsigned __int64 **)&v473);
        }
        v292 = v445;
        v282 = 0;
        v445 = 0;
        *(_BYTE *)(v289 + 40) = v292;
        v281 = *(_QWORD *)(v281 + 8);
        if ( v439 == v281 )
        {
LABEL_606:
          v188 = v427;
LABEL_607:
          v293 = (_QWORD *)v2[53];
          if ( !v293 )
          {
            v294 = v464;
            goto LABEL_614;
          }
          v294 = v464;
          do
          {
            while ( 1 )
            {
              v295 = v293[2];
              v296 = v293[3];
              if ( v293[4] >= v471 )
                break;
              v293 = (_QWORD *)v293[3];
              if ( !v296 )
                goto LABEL_612;
            }
            v294 = (__int64)v293;
            v293 = (_QWORD *)v293[2];
          }
          while ( v295 );
LABEL_612:
          if ( v294 == v464 || *(_QWORD *)(v294 + 32) > v471 )
          {
LABEL_614:
            v473 = (unsigned __int8 *)&v471;
            v294 = sub_2C84590(v451, v294, (unsigned __int64 **)&v473);
          }
          v435 = *(_BYTE *)(v294 + 40);
          v297 = (_QWORD *)v2[59];
          if ( !v297 )
          {
            v299 = v463;
            goto LABEL_622;
          }
          v298 = v471;
          v299 = v463;
          do
          {
            while ( 1 )
            {
              v300 = v297[2];
              v301 = v297[3];
              if ( v297[4] >= v471 )
                break;
              v297 = (_QWORD *)v297[3];
              if ( !v301 )
                goto LABEL_620;
            }
            v299 = (__int64)v297;
            v297 = (_QWORD *)v297[2];
          }
          while ( v300 );
LABEL_620:
          if ( v299 == v463 || *(_QWORD *)(v299 + 32) > v471 )
          {
LABEL_622:
            v473 = (unsigned __int8 *)&v471;
            v302 = sub_2C84590(v449, v299, (unsigned __int64 **)&v473);
            v298 = v471;
            v299 = v302;
          }
          v446 = *(_BYTE *)(v299 + 40);
          v440 = *(_QWORD *)(v298 + 56);
          if ( v298 + 48 == v440 )
            goto LABEL_647;
          v303 = (_QWORD *)(v298 + 48);
          v419 = v188;
          while ( 1 )
          {
LABEL_626:
            v304 = *v303 & 0xFFFFFFFFFFFFFFF8LL;
            v305 = v304 - 24;
            v303 = (_QWORD *)v304;
            if ( !v304 )
              v305 = 0;
            v472 = v305;
            if ( sub_2C83D20(v305) )
              break;
            LOBYTE(v470) = 0;
            LOBYTE(v473) = 0;
            sub_2C83AE0(v472, &v470, &v473);
            v435 |= (unsigned __int8)v470;
            v446 |= (unsigned __int8)v473;
            if ( v440 == v304 )
              goto LABEL_646;
          }
          v306 = (_QWORD *)v2[77];
          if ( !v306 )
          {
            v307 = (__int64)(v2 + 76);
            goto LABEL_636;
          }
          v307 = (__int64)(v2 + 76);
          do
          {
            while ( 1 )
            {
              v308 = v306[2];
              v309 = v306[3];
              if ( v306[4] >= v472 )
                break;
              v306 = (_QWORD *)v306[3];
              if ( !v309 )
                goto LABEL_634;
            }
            v307 = (__int64)v306;
            v306 = (_QWORD *)v306[2];
          }
          while ( v308 );
LABEL_634:
          if ( (_QWORD *)v307 == v2 + 76 || *(_QWORD *)(v307 + 32) > v472 )
          {
LABEL_636:
            v473 = (unsigned __int8 *)&v472;
            v307 = sub_2C84AF0(v2 + 75, v307, (unsigned __int64 **)&v473);
          }
          *(_BYTE *)(v307 + 40) = v435;
          v310 = (_QWORD *)v2[83];
          if ( !v310 )
          {
            v311 = (__int64)(v2 + 82);
            goto LABEL_644;
          }
          v311 = (__int64)(v2 + 82);
          do
          {
            while ( 1 )
            {
              v312 = v310[2];
              v313 = v310[3];
              if ( v310[4] >= v472 )
                break;
              v310 = (_QWORD *)v310[3];
              if ( !v313 )
                goto LABEL_642;
            }
            v311 = (__int64)v310;
            v310 = (_QWORD *)v310[2];
          }
          while ( v312 );
LABEL_642:
          if ( (_QWORD *)v311 == v2 + 82 || *(_QWORD *)(v311 + 32) > v472 )
          {
LABEL_644:
            v473 = (unsigned __int8 *)&v472;
            v311 = sub_2C84AF0(v2 + 81, v311, (unsigned __int64 **)&v473);
          }
          v314 = v446;
          v435 = 0;
          v446 = 0;
          *(_BYTE *)(v311 + 40) = v314;
          if ( v440 != v304 )
            goto LABEL_626;
LABEL_646:
          v188 = v419;
LABEL_647:
          if ( (_QWORD *)srcb[4] == v188 )
            goto LABEL_362;
          v315 = (_QWORD *)srcb[4];
          v447 = v2 + 64;
          v428 = v2 + 82;
          while ( 1 )
          {
LABEL_649:
            v316 = (__int64)(v315 - 3);
            if ( !v315 )
              v316 = 0;
            v469 = v316;
            v317 = sub_2C83D20(v316);
            if ( !v317 )
              goto LABEL_758;
            if ( *(_BYTE *)v469 != 85 )
              break;
            v405 = *(_QWORD *)(v469 - 32);
            if ( !v405
              || *(_BYTE *)v405
              || *(_QWORD *)(v405 + 24) != *(_QWORD *)(v469 + 80)
              || (*(_BYTE *)(v405 + 33) & 0x20) == 0
              || (unsigned int)(*(_DWORD *)(v405 + 36) - 8260) > 2
              || !(unsigned __int8)sub_BD3660(v469, 1) )
            {
              break;
            }
            v315 = (_QWORD *)v315[1];
            if ( v315 == v188 )
              goto LABEL_362;
          }
          v318 = (_QWORD *)v2[65];
          v319 = v318;
          if ( v318 )
          {
            v320 = v469;
            v321 = (__int64)(v2 + 64);
            do
            {
              while ( 1 )
              {
                v322 = v319[2];
                v323 = v319[3];
                if ( v319[4] >= v469 )
                  break;
                v319 = (_QWORD *)v319[3];
                if ( !v323 )
                  goto LABEL_658;
              }
              v321 = (__int64)v319;
              v319 = (_QWORD *)v319[2];
            }
            while ( v322 );
LABEL_658:
            if ( v447 != (_QWORD *)v321 && *(_QWORD *)(v321 + 32) <= v469 )
            {
              if ( !*(_BYTE *)(v321 + 40) )
                goto LABEL_661;
LABEL_764:
              v388 = (__int64)(v2 + 64);
              while ( 1 )
              {
                v389 = (_QWORD *)v318[3];
                if ( v318[4] >= v320 )
                {
                  v389 = (_QWORD *)v318[2];
                  v388 = (__int64)v318;
                }
                if ( !v389 )
                  break;
                v318 = v389;
              }
              if ( v447 == (_QWORD *)v388 || *(_QWORD *)(v388 + 32) > v320 )
                goto LABEL_771;
              goto LABEL_772;
            }
          }
          else
          {
            v321 = (__int64)(v2 + 64);
          }
          v473 = (unsigned __int8 *)&v469;
          if ( *(_BYTE *)(sub_2C84AF0(v2 + 63, v321, (unsigned __int64 **)&v473) + 40) )
          {
LABEL_762:
            v318 = (_QWORD *)v2[65];
            if ( v318 )
            {
              v320 = v469;
              goto LABEL_764;
            }
            v388 = (__int64)(v2 + 64);
LABEL_771:
            v473 = (unsigned __int8 *)&v469;
            v388 = sub_2C84AF0(v2 + 63, v388, (unsigned __int64 **)&v473);
LABEL_772:
            if ( *(_BYTE *)(v388 + 40) )
            {
              v390 = (_QWORD *)v2[71];
              if ( v390 )
              {
                v385 = v469;
                v383 = (__int64)v459;
                do
                {
                  while ( 1 )
                  {
                    v391 = v390[2];
                    v392 = v390[3];
                    if ( v390[4] >= v469 )
                      break;
                    v390 = (_QWORD *)v390[3];
                    if ( !v392 )
                      goto LABEL_778;
                  }
                  v383 = (__int64)v390;
                  v390 = (_QWORD *)v390[2];
                }
                while ( v391 );
LABEL_778:
                if ( (_QWORD *)v383 != v459 && *(_QWORD *)(v383 + 32) <= v469 )
                {
                  v384 = (_QWORD *)v2[71];
                  if ( !*(_BYTE *)(v383 + 40) )
                  {
LABEL_781:
                    v393 = (__int64)(v2 + 82);
                    v394 = (_QWORD *)v2[83];
                    v436 = (__int64)(v2 + 82);
                    if ( !v394 )
                    {
                      v393 = (__int64)(v2 + 82);
                      goto LABEL_788;
                    }
                    do
                    {
                      while ( 1 )
                      {
                        v395 = v394[2];
                        v396 = v394[3];
                        if ( v394[4] >= v469 )
                          break;
                        v394 = (_QWORD *)v394[3];
                        if ( !v396 )
                          goto LABEL_786;
                      }
                      v393 = (__int64)v394;
                      v394 = (_QWORD *)v394[2];
                    }
                    while ( v395 );
LABEL_786:
                    if ( v428 == (_QWORD *)v393 || *(_QWORD *)(v393 + 32) > v469 )
                    {
LABEL_788:
                      v473 = (unsigned __int8 *)&v469;
                      v393 = sub_2C84AF0(v2 + 81, v393, (unsigned __int64 **)&v473);
                    }
                    if ( !*(_BYTE *)(v393 + 40) )
                    {
                      v328 = v317;
                      v441 = v2 + 76;
                      goto LABEL_671;
                    }
LABEL_758:
                    v315 = (_QWORD *)v315[1];
                    if ( v315 == v188 )
                      goto LABEL_362;
                    goto LABEL_649;
                  }
LABEL_749:
                  v386 = (__int64)v459;
                  while ( 1 )
                  {
                    v387 = (_QWORD *)v384[3];
                    if ( v384[4] >= v385 )
                    {
                      v387 = (_QWORD *)v384[2];
                      v386 = (__int64)v384;
                    }
                    if ( !v387 )
                      break;
                    v384 = v387;
                  }
                  if ( (_QWORD *)v386 == v459 || *(_QWORD *)(v386 + 32) > v385 )
                    goto LABEL_756;
                  goto LABEL_757;
                }
              }
              else
              {
                v383 = (__int64)v459;
              }
              v473 = (unsigned __int8 *)&v469;
              if ( !*(_BYTE *)(sub_2C84AF0(v2 + 69, v383, (unsigned __int64 **)&v473) + 40) )
                goto LABEL_781;
            }
            v384 = (_QWORD *)v2[71];
            if ( v384 )
            {
              v385 = v469;
              goto LABEL_749;
            }
            v386 = (__int64)v459;
LABEL_756:
            v473 = (unsigned __int8 *)&v469;
            v386 = sub_2C84AF0(v2 + 69, v386, (unsigned __int64 **)&v473);
LABEL_757:
            if ( !*(_BYTE *)(v386 + 40) )
              goto LABEL_758;
            v441 = v2 + 76;
            v397 = (__int64)(v2 + 76);
            v398 = (_QWORD *)v2[77];
            if ( !v398 )
            {
              v397 = (__int64)(v2 + 76);
              goto LABEL_798;
            }
            do
            {
              while ( 1 )
              {
                v399 = v398[2];
                v400 = v398[3];
                if ( v398[4] >= v469 )
                  break;
                v398 = (_QWORD *)v398[3];
                if ( !v400 )
                  goto LABEL_796;
              }
              v397 = (__int64)v398;
              v398 = (_QWORD *)v398[2];
            }
            while ( v399 );
LABEL_796:
            if ( (_QWORD *)v397 == v441 || *(_QWORD *)(v397 + 32) > v469 )
            {
LABEL_798:
              v473 = (unsigned __int8 *)&v469;
              v397 = sub_2C84AF0(v2 + 75, v397, (unsigned __int64 **)&v473);
            }
            if ( *(_BYTE *)(v397 + 40) )
              goto LABEL_758;
            v401 = (__int64)(v2 + 82);
            v402 = (_QWORD *)v2[83];
            v436 = (__int64)(v2 + 82);
            if ( v402 )
            {
              do
              {
                while ( 1 )
                {
                  v403 = v402[2];
                  v404 = v402[3];
                  if ( v402[4] >= v469 )
                    break;
                  v402 = (_QWORD *)v402[3];
                  if ( !v404 )
                    goto LABEL_805;
                }
                v401 = (__int64)v402;
                v402 = (_QWORD *)v402[2];
              }
              while ( v403 );
LABEL_805:
              if ( (_QWORD *)v401 == v428 || *(_QWORD *)(v401 + 32) > v469 )
              {
LABEL_807:
                v473 = (unsigned __int8 *)&v469;
                v401 = sub_2C84AF0(v2 + 81, v401, (unsigned __int64 **)&v473);
              }
              if ( !*(_BYTE *)(v401 + 40) )
              {
                v328 = v317;
LABEL_671:
                v470 = (_QWORD *)v469;
                v471 = v469;
                if ( !*(_QWORD *)(v469 + 48) && (*(_BYTE *)(v469 + 7) & 0x20) == 0 )
                  goto LABEL_694;
                v329 = sub_B91F50(v469, "dbg", 3u);
                if ( !v329 )
                  goto LABEL_694;
                v330 = *(_BYTE *)(v329 - 16);
                v331 = *(_DWORD *)(v329 + 4);
                if ( (v330 & 2) != 0 )
                  v332 = *(_BYTE ***)(v329 - 32);
                else
                  v332 = (_BYTE **)(v329 - 16 - 8LL * ((v330 >> 2) & 0xF));
                v333 = *v332;
                if ( *v333 == 16 )
                  goto LABEL_678;
                v334 = *(v333 - 16);
                if ( (v334 & 2) != 0 )
                {
                  v333 = (_BYTE *)**((_QWORD **)v333 - 4);
                  if ( v333 )
                    goto LABEL_678;
LABEL_873:
                  v338 = 0;
                  v336 = byte_3F871B3;
LABEL_681:
                  v339 = *v2;
                  v340 = *(_BYTE **)(*v2 + 32LL);
                  if ( *(_BYTE **)(*v2 + 24LL) == v340 )
                  {
                    v431 = v336;
                    v409 = sub_CB6200(*v2, (unsigned __int8 *)"[", 1u);
                    v336 = v431;
                    v339 = v409;
                  }
                  else
                  {
                    *v340 = 91;
                    ++*(_QWORD *)(v339 + 32);
                  }
                  if ( !v336 )
                  {
                    v474 = 0;
                    v342 = 0;
                    v473 = (unsigned __int8 *)v475;
                    v343 = (unsigned __int8 *)v475;
                    LOBYTE(v475[0]) = 0;
                    goto LABEL_688;
                  }
                  v472 = v338;
                  v473 = (unsigned __int8 *)v475;
                  if ( v338 > 0xF )
                  {
                    srcc = v336;
                    v429 = v339;
                    v407 = (unsigned __int64 *)sub_22409D0((__int64)&v473, &v472, 0);
                    v339 = v429;
                    v336 = srcc;
                    v473 = (unsigned __int8 *)v407;
                    v408 = v407;
                    v475[0] = v472;
                  }
                  else
                  {
                    if ( v338 == 1 )
                    {
                      LOBYTE(v475[0]) = *v336;
                      v341 = v475;
                      goto LABEL_687;
                    }
                    if ( !v338 )
                    {
                      v341 = v475;
LABEL_687:
                      v474 = v338;
                      *((_BYTE *)v341 + v338) = 0;
                      v342 = v474;
                      v343 = v473;
LABEL_688:
                      sub_CB6200(v339, v343, v342);
                      if ( v473 != (unsigned __int8 *)v475 )
                        j_j___libc_free_0((unsigned __int64)v473);
                      v344 = *v2;
                      v345 = *(_BYTE **)(*v2 + 32LL);
                      if ( *(_BYTE **)(*v2 + 24LL) == v345 )
                      {
                        v344 = sub_CB6200(v344, (unsigned __int8 *)":", 1u);
                      }
                      else
                      {
                        *v345 = 58;
                        ++*(_QWORD *)(v344 + 32);
                      }
                      v346 = sub_CB59D0(v344, v331);
                      v347 = *(_BYTE **)(v346 + 32);
                      if ( *(_BYTE **)(v346 + 24) == v347 )
                      {
                        sub_CB6200(v346, (unsigned __int8 *)"]", 1u);
                      }
                      else
                      {
                        *v347 = 93;
                        ++*(_QWORD *)(v346 + 32);
                      }
LABEL_694:
                      v348 = *v2;
                      v349 = *(__m128i **)(*v2 + 32LL);
                      if ( *(_QWORD *)(*v2 + 24LL) - (_QWORD)v349 <= 0x14u )
                      {
                        sub_CB6200(v348, " Removed dead synch: ", 0x15u);
                      }
                      else
                      {
                        si128 = _mm_load_si128((const __m128i *)&xmmword_42D1060);
                        v349[1].m128i_i32[0] = 979919726;
                        v349[1].m128i_i8[4] = 32;
                        *v349 = si128;
                        *(_QWORD *)(v348 + 32) += 21LL;
                      }
                      v351 = *v2;
                      v352 = *(void **)(*v2 + 32LL);
                      if ( *(_QWORD *)(*v2 + 24LL) - (_QWORD)v352 <= 0xBu )
                      {
                        v351 = sub_CB6200(*v2, "Read above: ", 0xCu);
                      }
                      else
                      {
                        qmemcpy(v352, "Read above: ", 12);
                        *(_QWORD *)(v351 + 32) += 12LL;
                      }
                      v353 = (_QWORD *)v2[65];
                      if ( v353 )
                      {
                        v354 = (__int64)(v2 + 64);
                        do
                        {
                          while ( 1 )
                          {
                            v355 = v353[2];
                            v356 = v353[3];
                            if ( v353[4] >= v471 )
                              break;
                            v353 = (_QWORD *)v353[3];
                            if ( !v356 )
                              goto LABEL_703;
                          }
                          v354 = (__int64)v353;
                          v353 = (_QWORD *)v353[2];
                        }
                        while ( v355 );
LABEL_703:
                        if ( v447 != (_QWORD *)v354 && *(_QWORD *)(v354 + 32) <= v471 )
                        {
                          v448 = v2 + 63;
                          goto LABEL_706;
                        }
                      }
                      else
                      {
                        v354 = (__int64)(v2 + 64);
                      }
                      v448 = v2 + 63;
                      v473 = (unsigned __int8 *)&v471;
                      v354 = sub_2C84AF0(v2 + 63, v354, (unsigned __int64 **)&v473);
LABEL_706:
                      v357 = sub_CB59F0(v351, *(unsigned __int8 *)(v354 + 40));
                      v358 = *(void **)(v357 + 32);
                      if ( *(_QWORD *)(v357 + 24) - (_QWORD)v358 <= 0xEu )
                      {
                        v357 = sub_CB6200(v357, ", Write above: ", 0xFu);
                      }
                      else
                      {
                        qmemcpy(v358, ", Write above: ", 15);
                        *(_QWORD *)(v357 + 32) += 15LL;
                      }
                      v359 = (_QWORD *)v2[71];
                      if ( !v359 )
                      {
                        v360 = (__int64)v459;
                        goto LABEL_715;
                      }
                      v360 = (__int64)v459;
                      do
                      {
                        while ( 1 )
                        {
                          v361 = v359[2];
                          v362 = v359[3];
                          if ( v359[4] >= v471 )
                            break;
                          v359 = (_QWORD *)v359[3];
                          if ( !v362 )
                            goto LABEL_713;
                        }
                        v360 = (__int64)v359;
                        v359 = (_QWORD *)v359[2];
                      }
                      while ( v361 );
LABEL_713:
                      if ( (_QWORD *)v360 == v459 || (v363 = v2 + 69, *(_QWORD *)(v360 + 32) > v471) )
                      {
LABEL_715:
                        v363 = v2 + 69;
                        v473 = (unsigned __int8 *)&v471;
                        v360 = sub_2C84AF0(v2 + 69, v360, (unsigned __int64 **)&v473);
                      }
                      v364 = sub_CB59F0(v357, *(unsigned __int8 *)(v360 + 40));
                      v365 = *(void **)(v364 + 32);
                      if ( *(_QWORD *)(v364 + 24) - (_QWORD)v365 <= 0xDu )
                      {
                        v364 = sub_CB6200(v364, ", Read below: ", 0xEu);
                      }
                      else
                      {
                        qmemcpy(v365, ", Read below: ", 14);
                        *(_QWORD *)(v364 + 32) += 14LL;
                      }
                      v366 = (_QWORD *)v2[77];
                      if ( !v366 )
                      {
                        v367 = (__int64)v441;
LABEL_845:
                        v460 = v2 + 75;
                        v473 = (unsigned __int8 *)&v471;
                        v367 = sub_2C84AF0(v2 + 75, v367, (unsigned __int64 **)&v473);
                        goto LABEL_726;
                      }
                      v367 = (__int64)v441;
                      do
                      {
                        while ( 1 )
                        {
                          v368 = v366[2];
                          v369 = v366[3];
                          if ( v366[4] >= v471 )
                            break;
                          v366 = (_QWORD *)v366[3];
                          if ( !v369 )
                            goto LABEL_723;
                        }
                        v367 = (__int64)v366;
                        v366 = (_QWORD *)v366[2];
                      }
                      while ( v368 );
LABEL_723:
                      if ( v441 == (_QWORD *)v367 || *(_QWORD *)(v367 + 32) > v471 )
                        goto LABEL_845;
                      v460 = v2 + 75;
LABEL_726:
                      v370 = sub_CB59F0(v364, *(unsigned __int8 *)(v367 + 40));
                      v371 = *(void **)(v370 + 32);
                      if ( *(_QWORD *)(v370 + 24) - (_QWORD)v371 <= 0xEu )
                      {
                        v370 = sub_CB6200(v370, ", Write below: ", 0xFu);
                      }
                      else
                      {
                        qmemcpy(v371, ", Write below: ", 15);
                        *(_QWORD *)(v370 + 32) += 15LL;
                      }
                      v372 = (_QWORD *)v2[83];
                      if ( !v372 )
                        goto LABEL_736;
                      v373 = v436;
                      do
                      {
                        while ( 1 )
                        {
                          v374 = v372[2];
                          v375 = v372[3];
                          if ( v372[4] >= v471 )
                            break;
                          v372 = (_QWORD *)v372[3];
                          if ( !v375 )
                            goto LABEL_733;
                        }
                        v373 = (__int64)v372;
                        v372 = (_QWORD *)v372[2];
                      }
                      while ( v374 );
LABEL_733:
                      if ( v373 == v436 )
                      {
LABEL_736:
                        v376 = v2 + 81;
                        v473 = (unsigned __int8 *)&v471;
                        v373 = sub_2C84AF0(v2 + 81, v436, (unsigned __int64 **)&v473);
                      }
                      else
                      {
                        v376 = v2 + 81;
                        if ( *(_QWORD *)(v373 + 32) > v471 )
                        {
                          v436 = v373;
                          goto LABEL_736;
                        }
                      }
                      v377 = sub_CB59F0(v370, *(unsigned __int8 *)(v373 + 40));
                      v378 = *(void **)(v377 + 32);
                      if ( *(_QWORD *)(v377 + 24) - (_QWORD)v378 <= 0xCu )
                      {
                        v406 = sub_CB6200(v377, (unsigned __int8 *)" in function ", 0xDu);
                        v379 = *(_BYTE **)(v406 + 32);
                        v377 = v406;
                      }
                      else
                      {
                        qmemcpy(v378, " in function ", 13);
                        v379 = (_BYTE *)(*(_QWORD *)(v377 + 32) + 13LL);
                        *(_QWORD *)(v377 + 32) = v379;
                      }
                      v380 = *(_BYTE **)(v377 + 24);
                      v381 = v2[2];
                      v382 = (unsigned __int8 *)v2[1];
                      if ( v381 > v380 - v379 )
                      {
                        v377 = sub_CB6200(v377, v382, v381);
                        v380 = *(_BYTE **)(v377 + 24);
                        v379 = *(_BYTE **)(v377 + 32);
                      }
                      else if ( v381 )
                      {
                        v442 = v2[2];
                        memcpy(v379, v382, v381);
                        v380 = *(_BYTE **)(v377 + 24);
                        v379 = (_BYTE *)(v442 + *(_QWORD *)(v377 + 32));
                        *(_QWORD *)(v377 + 32) = v379;
                      }
                      if ( v379 == v380 )
                      {
                        sub_CB6200(v377, (unsigned __int8 *)"\n", 1u);
                      }
                      else
                      {
                        *v379 = 10;
                        ++*(_QWORD *)(v377 + 32);
                      }
                      sub_2C83F20(v448, (unsigned __int64 *)&v470);
                      sub_2C83F20(v363, (unsigned __int64 *)&v470);
                      sub_2C83F20(v460, (unsigned __int64 *)&v470);
                      sub_2C83F20(v376, (unsigned __int64 *)&v470);
                      sub_B43D60(v470);
                      v413 = v328;
                      goto LABEL_2;
                    }
                    v408 = v475;
                  }
                  v430 = v339;
                  memcpy(v408, v336, v338);
                  v338 = v472;
                  v341 = (unsigned __int64 *)v473;
                  v339 = v430;
                  goto LABEL_687;
                }
                v333 = *(_BYTE **)&v333[-8 * ((v334 >> 2) & 0xF) - 16];
                if ( !v333 )
                  goto LABEL_873;
LABEL_678:
                v335 = *(v333 - 16);
                if ( (v335 & 2) != 0 )
                {
                  v336 = (const char *)**((_QWORD **)v333 - 4);
                  if ( v336 )
                  {
LABEL_680:
                    v336 = (const char *)sub_B91420((__int64)v336);
                    v338 = v337;
                    goto LABEL_681;
                  }
                }
                else
                {
                  v336 = *(const char **)&v333[-8 * ((v335 >> 2) & 0xF) - 16];
                  if ( v336 )
                    goto LABEL_680;
                }
                v338 = 0;
                goto LABEL_681;
              }
              goto LABEL_758;
            }
            v401 = (__int64)(v2 + 82);
            goto LABEL_807;
          }
LABEL_661:
          v324 = (_QWORD *)v2[71];
          if ( v324 )
          {
            v325 = (__int64)v459;
            do
            {
              while ( 1 )
              {
                v326 = v324[2];
                v327 = v324[3];
                if ( v324[4] >= v469 )
                  break;
                v324 = (_QWORD *)v324[3];
                if ( !v327 )
                  goto LABEL_666;
              }
              v325 = (__int64)v324;
              v324 = (_QWORD *)v324[2];
            }
            while ( v326 );
LABEL_666:
            if ( (_QWORD *)v325 == v459 || *(_QWORD *)(v325 + 32) > v469 )
            {
LABEL_668:
              v473 = (unsigned __int8 *)&v469;
              v325 = sub_2C84AF0(v2 + 69, v325, (unsigned __int64 **)&v473);
            }
            if ( !*(_BYTE *)(v325 + 40) )
            {
              v328 = v317;
              v436 = (__int64)(v2 + 82);
              v441 = v2 + 76;
              goto LABEL_671;
            }
            goto LABEL_762;
          }
          v325 = (__int64)v459;
          goto LABEL_668;
        }
        goto LABEL_586;
      }
      v289 = (__int64)v459;
      goto LABEL_604;
    }
  }
  v2 = v121;
LABEL_363:
  sub_2C84080(v2[5]);
  v192 = v2[11];
  v2[5] = 0;
  v2[6] = v2 + 4;
  v2[7] = v2 + 4;
  v2[8] = 0;
  sub_2C84080(v192);
  v2[11] = 0;
  v193 = v2[17];
  v2[12] = v2 + 10;
  v2[13] = v2 + 10;
  v2[14] = 0;
  sub_2C84080(v193);
  v2[17] = 0;
  v2[20] = 0;
  v194 = v2[23];
  v2[18] = v412;
  v2[19] = v412;
  sub_2C84080(v194);
  v2[23] = 0;
  v2[26] = 0;
  v195 = v2[29];
  v2[24] = v465;
  v2[25] = v465;
  sub_2C84080(v195);
  v196 = v2[35];
  v2[29] = 0;
  v2[32] = 0;
  v2[30] = v467;
  v2[31] = v467;
  sub_2C84080(v196);
  v2[35] = 0;
  v2[38] = 0;
  v197 = v2[41];
  v2[36] = v466;
  v2[37] = v466;
  sub_2C84080(v197);
  v2[41] = 0;
  v2[44] = 0;
  v198 = v2[47];
  v2[42] = v410;
  v2[43] = v410;
  sub_2C84080(v198);
  v199 = v2[53];
  v2[47] = 0;
  v2[50] = 0;
  v2[48] = v468;
  v2[49] = v468;
  sub_2C84080(v199);
  v2[53] = 0;
  v2[56] = 0;
  v200 = v2[59];
  v2[54] = v464;
  v2[55] = v464;
  sub_2C84080(v200);
  v2[59] = 0;
  v2[62] = 0;
  v201 = v2[65];
  v2[60] = v463;
  v2[61] = v463;
  sub_2C83D50(v201);
  v2[65] = 0;
  v202 = v2[71];
  v2[66] = v2 + 64;
  v2[67] = v2 + 64;
  v2[68] = 0;
  sub_2C83D50(v202);
  v2[71] = 0;
  v203 = v2[77];
  v2[72] = v2 + 70;
  v2[73] = v2 + 70;
  v2[74] = 0;
  sub_2C83D50(v203);
  v2[77] = 0;
  v204 = v2[83];
  v2[78] = v2 + 76;
  v2[79] = v2 + 76;
  v2[80] = 0;
  sub_2C83D50(v204);
  v2[83] = 0;
  v2[84] = v2 + 82;
  v2[85] = v2 + 82;
  v2[86] = 0;
  return v413;
}
