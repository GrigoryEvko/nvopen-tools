// Function: sub_1608300
// Address: 0x1608300
//
void __fastcall sub_1608300(__int64 a1)
{
  int v2; // esi
  _QWORD *v3; // rax
  _QWORD *v4; // rdx
  __int64 v5; // r12
  _QWORD *v6; // rcx
  __int64 *v7; // r12
  __int64 *i; // rbx
  __int64 v9; // rdi
  __m128i v10; // rax
  __int64 v11; // rcx
  __m128i v12; // rax
  __int64 v13; // rcx
  __m128i v14; // rax
  __int64 v15; // rcx
  __m128i v16; // rax
  __int64 v17; // rcx
  __m128i v18; // rax
  __int64 v19; // rcx
  __m128i v20; // rax
  __int64 v21; // rcx
  __m128i v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rax
  __int64 v25; // r12
  __int64 v26; // rdx
  _QWORD *v27; // rax
  __int64 v28; // rax
  __int64 v29; // r12
  unsigned __int64 v30; // rdx
  __m128i v31; // xmm3
  _QWORD *v32; // rax
  __int64 v33; // rax
  __int64 v34; // r12
  unsigned __int64 v35; // rdx
  __m128i v36; // xmm5
  _QWORD *v37; // rax
  __int64 v38; // rax
  __int64 v39; // r12
  unsigned __int64 v40; // rdx
  __m128i v41; // xmm7
  _QWORD *v42; // rax
  __m128i v43; // rax
  __int64 v44; // rcx
  __int64 v45; // rax
  __int64 v46; // r12
  __int64 v47; // rdx
  __m128i v48; // xmm1
  _QWORD *v49; // rax
  __m128i v50; // rax
  __m128i v51; // rax
  __m128i v52; // rax
  __int64 v53; // rcx
  __int64 v54; // rax
  __int64 v55; // r12
  __int64 v56; // rdx
  __m128i v57; // xmm3
  _QWORD *v58; // rax
  __int64 v59; // rax
  __int64 v60; // r12
  unsigned __int64 v61; // rdx
  __m128i v62; // xmm5
  _QWORD *v63; // rax
  __int64 v64; // rax
  __int64 v65; // r12
  unsigned __int64 v66; // rdx
  __m128i v67; // xmm7
  _QWORD *v68; // rax
  __int64 v69; // rax
  __int64 v70; // r12
  unsigned __int64 v71; // rdx
  __m128i v72; // xmm1
  _QWORD *v73; // rax
  __int64 v74; // rax
  __int64 v75; // r12
  unsigned __int64 v76; // rdx
  __m128i v77; // xmm3
  _QWORD *v78; // rax
  __int64 v79; // rax
  __int64 v80; // r12
  unsigned __int64 v81; // rdx
  __m128i v82; // xmm5
  _QWORD *v83; // rax
  __int64 v84; // rax
  __int64 v85; // r12
  unsigned __int64 v86; // rdx
  __m128i v87; // xmm7
  _QWORD *v88; // rax
  __m128i v89; // rax
  __int64 v90; // rcx
  __m128i v91; // rax
  __int64 v92; // rax
  __int64 v93; // r12
  __int64 v94; // rdx
  __m128i v95; // xmm1
  _QWORD *v96; // rax
  __m128i v97; // rax
  __int64 v98; // rcx
  __int64 v99; // rax
  __int64 v100; // r12
  __int64 v101; // rdx
  __m128i v102; // xmm3
  _QWORD *v103; // rax
  __m128i v104; // rax
  __int64 v105; // rcx
  __int64 k; // rsi
  __int64 v107; // rcx
  __m128i v108; // rax
  __int64 v109; // rdx
  __int64 v110; // rdx
  __int64 *v111; // rbx
  __int64 *m; // r12
  __int64 v113; // rdi
  __int64 v114; // rax
  __int64 v115; // r12
  unsigned __int64 v116; // rdx
  __m128i v117; // xmm5
  _QWORD *v118; // rax
  __int64 v119; // r13
  __int64 v120; // rax
  __int64 v121; // r12
  unsigned __int64 v122; // rdx
  __m128i v123; // xmm7
  _QWORD *v124; // rax
  __int64 v125; // r13
  __int64 v126; // rax
  __int64 v127; // r12
  unsigned __int64 v128; // rdx
  __m128i v129; // xmm1
  __int64 **v130; // rax
  __int64 *v131; // r13
  __int64 v132; // rdi
  __int64 v133; // rax
  __int64 v134; // r12
  unsigned __int64 v135; // rdx
  __m128i v136; // xmm3
  __int64 *v137; // rax
  __int64 v138; // r13
  __int64 v139; // rax
  __int64 v140; // r12
  __int64 v141; // rdx
  __m128i v142; // xmm5
  _QWORD *v143; // rax
  __int64 v144; // r13
  __m128i v145; // rax
  __int64 v146; // rcx
  __m128i v147; // rax
  __m128i v148; // rax
  __int64 v149; // rcx
  __int64 v150; // rax
  __int64 v151; // r12
  unsigned __int64 v152; // rdx
  __m128i v153; // xmm7
  __int64 *v154; // rax
  __int64 v155; // r13
  __int64 v156; // rax
  __int64 v157; // r12
  unsigned __int64 v158; // rdx
  __m128i v159; // xmm1
  __int64 *v160; // rax
  __int64 v161; // r13
  __int64 v162; // rax
  __int64 v163; // r12
  unsigned __int64 v164; // rdx
  __m128i v165; // xmm3
  __int64 *v166; // rax
  __int64 v167; // r13
  __int64 v168; // rax
  __int64 v169; // r12
  unsigned __int64 v170; // rdx
  __m128i v171; // xmm5
  __int64 *v172; // rax
  __int64 v173; // r13
  __int64 v174; // rax
  __int64 v175; // r12
  unsigned __int64 v176; // rdx
  __m128i v177; // xmm7
  __int64 *v178; // rax
  __int64 v179; // r13
  __m128i v180; // rax
  __int64 v181; // rcx
  __m128i v182; // rax
  __int64 v183; // rcx
  __int64 v184; // rax
  __int64 v185; // r12
  unsigned __int64 v186; // rdx
  __m128i v187; // xmm1
  __int64 *v188; // rax
  __int64 v189; // r13
  __int64 v190; // rax
  __int64 v191; // r12
  unsigned __int64 v192; // rdx
  __m128i v193; // xmm3
  __int64 *v194; // rax
  __int64 v195; // r13
  __int64 v196; // rax
  __int64 v197; // r12
  unsigned __int64 v198; // rdx
  __m128i v199; // xmm5
  __int64 *v200; // rax
  __int64 v201; // r13
  __int64 v202; // rax
  __int64 v203; // r12
  unsigned __int64 v204; // rdx
  __m128i v205; // xmm7
  __int64 *v206; // rax
  __int64 v207; // r13
  __int64 v208; // rax
  __int64 v209; // r12
  unsigned __int64 v210; // rdx
  __m128i v211; // xmm1
  __int64 *v212; // rax
  __int64 v213; // r13
  __int64 v214; // rax
  __int64 v215; // r12
  unsigned __int64 v216; // rdx
  __m128i v217; // xmm3
  __int64 *v218; // rax
  __int64 v219; // r13
  __int64 v220; // rax
  __int64 v221; // r12
  unsigned __int64 v222; // rdx
  __m128i v223; // xmm5
  __int64 *v224; // rax
  __int64 v225; // r13
  __m128i v226; // rax
  __int64 v227; // rcx
  __int64 v228; // rax
  __int64 v229; // r12
  unsigned __int64 v230; // rdx
  __m128i v231; // xmm7
  __int64 *v232; // rax
  __int64 v233; // r13
  __int64 v234; // rax
  __int64 v235; // r12
  unsigned __int64 v236; // rdx
  __m128i v237; // xmm1
  __int64 *v238; // rax
  __int64 v239; // r13
  __int64 v240; // rax
  __int64 v241; // r12
  unsigned __int64 v242; // rdx
  __m128i v243; // xmm3
  __int64 *v244; // rax
  __int64 v245; // r13
  __int64 v246; // rax
  __int64 v247; // r12
  unsigned __int64 v248; // rdx
  __m128i v249; // xmm5
  __int64 *v250; // rax
  __int64 v251; // r13
  __int64 v252; // rax
  __int64 v253; // r12
  unsigned __int64 v254; // rdx
  __m128i v255; // xmm7
  __int64 *v256; // rax
  __int64 v257; // r13
  __int64 v258; // rax
  __int64 v259; // r12
  unsigned __int64 v260; // rdx
  __m128i v261; // xmm1
  __int64 *v262; // rax
  __int64 v263; // r13
  __int64 v264; // rax
  __int64 v265; // r12
  unsigned __int64 v266; // rdx
  __m128i v267; // xmm3
  __int64 *v268; // rax
  __int64 v269; // r13
  __m128i v270; // rax
  __int64 v271; // rcx
  __int64 v272; // rbx
  _QWORD **n; // rax
  __m128i v274; // rax
  __m128i v275; // rax
  __int64 v276; // rsi
  __int64 v277; // rcx
  __m128i v278; // rax
  __int64 v279; // rcx
  __m128i v280; // rax
  __int64 v281; // rcx
  __m128i v282; // rax
  __int64 v283; // rcx
  __m128i v284; // rax
  __int64 v285; // rcx
  _QWORD *v286; // rbx
  _QWORD *v287; // r12
  int v288; // ecx
  _QWORD *v289; // rsi
  __int64 *v290; // rax
  __int64 v291; // rdx
  __int64 *v292; // r15
  __int64 *v293; // rbx
  __int64 v294; // r12
  __int64 v295; // rax
  __int64 *v296; // rax
  __int64 v297; // rdx
  __int64 v298; // rax
  __int64 v299; // rbx
  __int64 v300; // r12
  unsigned __int64 *v301; // r15
  unsigned __int64 v302; // rdi
  __int64 v303; // rdx
  __int64 v304; // rbx
  __int64 v305; // r12
  __int64 v306; // rdx
  __int64 v307; // rbx
  __int64 v308; // r12
  __int64 v309; // rdx
  __int64 v310; // rbx
  __int64 v311; // r12
  __int64 v312; // r8
  int v313; // edx
  __int64 v314; // rcx
  __int64 v315; // rsi
  __int64 v316; // rax
  __int64 v317; // rdx
  _QWORD *v318; // rax
  _QWORD *ii; // rdx
  __m128i *v320; // r15
  __m128i *v321; // r12
  __int64 v322; // r13
  __int64 v323; // rdx
  __int64 v324; // rax
  _QWORD *v325; // rbx
  _QWORD *v326; // r15
  _QWORD *v327; // rdi
  __int64 v328; // rsi
  __int64 v329; // rax
  unsigned __int64 v330; // r8
  __int64 v331; // r15
  __int64 v332; // rbx
  unsigned __int64 v333; // rdi
  __int64 v334; // rax
  unsigned __int64 v335; // r8
  __int64 v336; // r15
  __int64 v337; // rbx
  unsigned __int64 v338; // rdi
  __int64 v339; // rdx
  __int64 v340; // r12
  __int64 v341; // rbx
  __int64 v342; // r13
  unsigned __int64 v343; // r15
  __int64 v344; // rdx
  __int64 v345; // r12
  __int64 v346; // rbx
  __int64 v347; // r13
  unsigned __int64 v348; // r15
  __int64 v349; // rax
  unsigned __int64 v350; // r8
  __int64 v351; // r15
  __int64 v352; // rbx
  unsigned __int64 v353; // rdi
  __int64 v354; // r15
  __int64 v355; // rax
  unsigned __int64 v356; // r8
  __int64 v357; // r15
  __int64 v358; // rbx
  unsigned __int64 v359; // rdi
  __int64 v360; // rdi
  __int64 v361; // rdi
  __int64 v362; // rdi
  void (*v363)(void); // rax
  unsigned __int64 v364; // rdi
  _QWORD *v365; // rbx
  __int64 v366; // r13
  __int64 v367; // r12
  __int64 v368; // r15
  _QWORD *v369; // r15
  __int64 v370; // r12
  __int64 v371; // rcx
  int v372; // eax
  unsigned int v373; // ecx
  unsigned int v374; // eax
  int v375; // r15d
  unsigned int v376; // eax
  __int64 v377; // rcx
  _QWORD *v378; // rax
  __int64 jj; // rbx
  __int64 v380; // r12
  _QWORD *v381; // r12
  __int64 v382; // r13
  __int64 v383; // rbx
  __int64 v384; // r15
  _QWORD *v385; // r12
  __int64 v386; // r15
  __int64 v387; // rbx
  __int64 v388; // r13
  __int64 v389; // r13
  _QWORD *v390; // r15
  __int64 v391; // rbx
  __int64 v392; // r12
  _QWORD **v393; // r13
  __int64 v394; // r15
  __int64 v395; // rbx
  _QWORD **v396; // r13
  __int64 v397; // r15
  __int64 v398; // rbx
  _QWORD **v399; // r12
  __int64 v400; // r13
  __int64 v401; // rbx
  __int64 *v402; // r12
  __int64 v403; // r15
  __int64 v404; // rbx
  __int64 v405; // r13
  __int64 *v406; // r12
  __int64 v407; // r15
  __int64 v408; // rbx
  __int64 v409; // r13
  __int64 *v410; // r12
  __int64 v411; // r15
  __int64 v412; // rbx
  __int64 v413; // r13
  __int64 *v414; // r12
  __int64 v415; // r13
  __int64 v416; // rbx
  __int64 v417; // r15
  __int64 *v418; // r12
  __int64 v419; // r13
  __int64 v420; // rbx
  __int64 v421; // r15
  __int64 *v422; // r12
  __int64 v423; // r13
  __int64 v424; // rbx
  __int64 v425; // r15
  __int64 v426; // rcx
  __int64 *v427; // rax
  __int64 v428; // rcx
  _QWORD *v429; // rax
  __int64 j; // rbx
  __int64 v431; // rbx
  _QWORD *v432; // r13
  __int64 v433; // r12
  __int64 v434; // rbx
  _QWORD *v435; // r13
  __int64 v436; // r12
  __int64 v437; // rbx
  _QWORD *v438; // r13
  __int64 v439; // r12
  __int64 v440; // rbx
  _QWORD *v441; // r13
  __int64 v442; // r12
  __int64 v443; // rbx
  _QWORD *v444; // r13
  __int64 v445; // r12
  __int64 v446; // rbx
  _QWORD *v447; // r13
  __int64 v448; // r12
  __int64 v449; // rbx
  _QWORD *v450; // r13
  __int64 v451; // r12
  __int64 v452; // rbx
  _QWORD *v453; // r13
  __int64 v454; // r12
  __int64 v455; // rbx
  _QWORD *v456; // r13
  __int64 v457; // r12
  __int64 v458; // rbx
  _QWORD *v459; // r13
  __int64 v460; // r12
  __int64 v461; // rbx
  _QWORD *v462; // r13
  __int64 v463; // r12
  __int64 v464; // rbx
  _QWORD *v465; // r13
  __int64 v466; // r12
  __int64 v467; // rbx
  _QWORD *v468; // r13
  __int64 v469; // r12
  __int64 v470; // rbx
  _QWORD *v471; // r13
  __int64 v472; // r12
  __int64 v473; // rbx
  _QWORD *v474; // r13
  __int64 v475; // r12
  __int64 v476; // rbx
  _QWORD *v477; // r13
  __int64 v478; // r12
  _QWORD *v479; // r13
  _QWORD *v480; // rdi
  _QWORD *v481; // rdi
  unsigned __int64 v482; // [rsp+B0h] [rbp-F0h]
  unsigned __int64 v483; // [rsp+B8h] [rbp-E8h]
  unsigned __int64 v484; // [rsp+C0h] [rbp-E0h]
  unsigned __int64 v485; // [rsp+C8h] [rbp-D8h]
  unsigned __int64 v486; // [rsp+D0h] [rbp-D0h]
  unsigned __int64 v487; // [rsp+D8h] [rbp-C8h]
  unsigned __int64 v488; // [rsp+E0h] [rbp-C0h]
  unsigned __int64 v489; // [rsp+E8h] [rbp-B8h]
  __m128i v490; // [rsp+100h] [rbp-A0h] BYREF
  __m128i v491; // [rsp+110h] [rbp-90h]
  unsigned __int128 v492; // [rsp+120h] [rbp-80h] BYREF
  __m128i v493; // [rsp+130h] [rbp-70h] BYREF

  v2 = *(_DWORD *)(a1 + 28);
  while ( v2 != *(_DWORD *)(a1 + 32) )
  {
    v3 = *(_QWORD **)(a1 + 16);
    v4 = &v3[v2];
    if ( v3 != *(_QWORD **)(a1 + 8) )
      v4 = &v3[*(unsigned int *)(a1 + 24)];
    v5 = *v3;
    if ( v3 != v4 )
    {
      while ( 1 )
      {
        v5 = *v3;
        v6 = v3;
        if ( *v3 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v4 == ++v3 )
        {
          v5 = v6[1];
          break;
        }
      }
    }
    if ( v5 )
    {
      sub_1633490(v5);
      j_j___libc_free_0(v5, 736);
      v2 = *(_DWORD *)(a1 + 28);
    }
  }
  v7 = *(__int64 **)(a1 + 1504);
  for ( i = *(__int64 **)(a1 + 1496); v7 != i; ++i )
  {
    v9 = *i;
    sub_1623E60(v9);
  }
  v10.m128i_i64[0] = *(_QWORD *)(a1 + 504);
  v11 = *(_QWORD *)(a1 + 496);
  if ( *(_DWORD *)(a1 + 512) )
  {
    v10.m128i_i64[1] = v10.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 520);
    v493 = v10;
    *(_QWORD *)&v492 = a1 + 496;
    *((_QWORD *)&v492 + 1) = v11;
    sub_1607CA0((__int64)&v492);
    v476 = v493.m128i_i64[1];
    v477 = (_QWORD *)v493.m128i_i64[0];
    v478 = *(_QWORD *)(a1 + 504) + 8LL * *(unsigned int *)(a1 + 520);
    if ( v493.m128i_i64[0] != v478 )
    {
      do
      {
        sub_1623E60(*v477);
        do
          ++v477;
        while ( v477 != (_QWORD *)v476 && (*v477 == -16 || *v477 == -8) );
      }
      while ( v477 != (_QWORD *)v478 );
    }
  }
  v12.m128i_i64[0] = *(_QWORD *)(a1 + 536);
  v13 = *(_QWORD *)(a1 + 528);
  if ( *(_DWORD *)(a1 + 544) )
  {
    v12.m128i_i64[1] = v12.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 552);
    v493 = v12;
    *(_QWORD *)&v492 = a1 + 528;
    *((_QWORD *)&v492 + 1) = v13;
    sub_1607CD0((__int64)&v492);
    v473 = v493.m128i_i64[1];
    v474 = (_QWORD *)v493.m128i_i64[0];
    v475 = *(_QWORD *)(a1 + 536) + 8LL * *(unsigned int *)(a1 + 552);
    if ( v475 != v493.m128i_i64[0] )
    {
      do
      {
        sub_1623E60(*v474);
        do
          ++v474;
        while ( v474 != (_QWORD *)v473 && (*v474 == -16 || *v474 == -8) );
      }
      while ( v474 != (_QWORD *)v475 );
    }
  }
  v14.m128i_i64[0] = *(_QWORD *)(a1 + 568);
  v15 = *(_QWORD *)(a1 + 560);
  if ( *(_DWORD *)(a1 + 576) )
  {
    v14.m128i_i64[1] = v14.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 584);
    v493 = v14;
    *(_QWORD *)&v492 = a1 + 560;
    *((_QWORD *)&v492 + 1) = v15;
    sub_1607D00((__int64)&v492);
    v470 = v493.m128i_i64[1];
    v471 = (_QWORD *)v493.m128i_i64[0];
    v472 = *(_QWORD *)(a1 + 568) + 8LL * *(unsigned int *)(a1 + 584);
    if ( v493.m128i_i64[0] != v472 )
    {
      do
      {
        sub_1623E60(*v471);
        do
          ++v471;
        while ( v471 != (_QWORD *)v470 && (*v471 == -16 || *v471 == -8) );
      }
      while ( v471 != (_QWORD *)v472 );
    }
  }
  v16.m128i_i64[0] = *(_QWORD *)(a1 + 600);
  v17 = *(_QWORD *)(a1 + 592);
  if ( *(_DWORD *)(a1 + 608) )
  {
    v16.m128i_i64[1] = v16.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 616);
    v493 = v16;
    *(_QWORD *)&v492 = a1 + 592;
    *((_QWORD *)&v492 + 1) = v17;
    sub_1607D30((__int64)&v492);
    v467 = v493.m128i_i64[1];
    v468 = (_QWORD *)v493.m128i_i64[0];
    v469 = *(_QWORD *)(a1 + 600) + 8LL * *(unsigned int *)(a1 + 616);
    if ( v493.m128i_i64[0] != v469 )
    {
      do
      {
        sub_1623E60(*v468);
        do
          ++v468;
        while ( v468 != (_QWORD *)v467 && (*v468 == -16 || *v468 == -8) );
      }
      while ( v468 != (_QWORD *)v469 );
    }
  }
  v18.m128i_i64[0] = *(_QWORD *)(a1 + 632);
  v19 = *(_QWORD *)(a1 + 624);
  if ( *(_DWORD *)(a1 + 640) )
  {
    v18.m128i_i64[1] = v18.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 648);
    v493 = v18;
    *(_QWORD *)&v492 = a1 + 624;
    *((_QWORD *)&v492 + 1) = v19;
    sub_1607D60((__int64)&v492);
    v464 = v493.m128i_i64[1];
    v465 = (_QWORD *)v493.m128i_i64[0];
    v466 = *(_QWORD *)(a1 + 632) + 8LL * *(unsigned int *)(a1 + 648);
    if ( v493.m128i_i64[0] != v466 )
    {
      do
      {
        sub_1623E60(*v465);
        do
          ++v465;
        while ( v465 != (_QWORD *)v464 && (*v465 == -8 || *v465 == -16) );
      }
      while ( v465 != (_QWORD *)v466 );
    }
  }
  v20.m128i_i64[0] = *(_QWORD *)(a1 + 664);
  v21 = *(_QWORD *)(a1 + 656);
  if ( *(_DWORD *)(a1 + 672) )
  {
    v20.m128i_i64[1] = v20.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 680);
    v493 = v20;
    *(_QWORD *)&v492 = a1 + 656;
    *((_QWORD *)&v492 + 1) = v21;
    sub_1607D90((__int64)&v492);
    v461 = v493.m128i_i64[1];
    v462 = (_QWORD *)v493.m128i_i64[0];
    v463 = *(_QWORD *)(a1 + 664) + 8LL * *(unsigned int *)(a1 + 680);
    if ( v463 != v493.m128i_i64[0] )
    {
      do
      {
        sub_1623E60(*v462);
        do
          ++v462;
        while ( v462 != (_QWORD *)v461 && (*v462 == -8 || *v462 == -16) );
      }
      while ( v462 != (_QWORD *)v463 );
    }
  }
  v22.m128i_i64[0] = *(_QWORD *)(a1 + 696);
  v23 = *(_QWORD *)(a1 + 688);
  if ( *(_DWORD *)(a1 + 704) )
  {
    v22.m128i_i64[1] = v22.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 712);
    v493 = v22;
    *(_QWORD *)&v492 = a1 + 688;
    *((_QWORD *)&v492 + 1) = v23;
    sub_1607DC0((__int64)&v492);
    v458 = v493.m128i_i64[1];
    v459 = (_QWORD *)v493.m128i_i64[0];
    v460 = *(_QWORD *)(a1 + 696) + 8LL * *(unsigned int *)(a1 + 712);
    if ( v493.m128i_i64[0] != v460 )
    {
      do
      {
        sub_1623E60(*v459);
        do
          ++v459;
        while ( v459 != (_QWORD *)v458 && (*v459 == -8 || *v459 == -16) );
      }
      while ( v459 != (_QWORD *)v460 );
    }
  }
  v24 = *(_QWORD *)(a1 + 728);
  v25 = v24 + 8LL * *(unsigned int *)(a1 + 744);
  v26 = *(_QWORD *)(a1 + 720);
  if ( *(_DWORD *)(a1 + 736) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 720);
    v493.m128i_i64[0] = v24;
    v493.m128i_i64[1] = v25;
    *(_QWORD *)&v492 = a1 + 720;
    sub_1607DF0((__int64)&v492);
    v25 = *(_QWORD *)(a1 + 728) + 8LL * *(unsigned int *)(a1 + 744);
  }
  else
  {
    *(_QWORD *)&v492 = a1 + 720;
    *((_QWORD *)&v492 + 1) = v26;
    v493.m128i_i64[0] = v25;
    v493.m128i_i64[1] = v25;
  }
  v27 = (_QWORD *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = _mm_loadu_si128(&v493);
  if ( v493.m128i_i64[0] != v25 )
  {
    do
    {
      sub_1623E60(*v27);
      v491.m128i_i64[0] += 8;
      sub_1607DF0((__int64)&v490);
      v27 = (_QWORD *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v25 );
  }
  v28 = *(_QWORD *)(a1 + 760);
  v482 = a1 + 752;
  v29 = v28 + 8LL * *(unsigned int *)(a1 + 776);
  v30 = *(_QWORD *)(a1 + 752);
  if ( *(_DWORD *)(a1 + 768) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 752);
    v493.m128i_i64[0] = v28;
    v493.m128i_i64[1] = v29;
    *(_QWORD *)&v492 = a1 + 752;
    sub_1607E20((__int64)&v492);
    v29 = *(_QWORD *)(a1 + 760) + 8LL * *(unsigned int *)(a1 + 776);
  }
  else
  {
    v493.m128i_i64[0] = v28 + 8LL * *(unsigned int *)(a1 + 776);
    v492 = __PAIR128__(v30, v482);
    v493.m128i_i64[1] = v29;
  }
  v31 = _mm_loadu_si128(&v493);
  v32 = (_QWORD *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v31;
  if ( v493.m128i_i64[0] != v29 )
  {
    do
    {
      sub_1623E60(*v32);
      v491.m128i_i64[0] += 8;
      sub_1607E20((__int64)&v490);
      v32 = (_QWORD *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v29 );
  }
  v33 = *(_QWORD *)(a1 + 792);
  v483 = a1 + 784;
  v34 = v33 + 8LL * *(unsigned int *)(a1 + 808);
  v35 = *(_QWORD *)(a1 + 784);
  if ( *(_DWORD *)(a1 + 800) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 784);
    v493.m128i_i64[0] = v33;
    v493.m128i_i64[1] = v34;
    *(_QWORD *)&v492 = a1 + 784;
    sub_1607E50((__int64)&v492);
    v34 = *(_QWORD *)(a1 + 792) + 8LL * *(unsigned int *)(a1 + 808);
  }
  else
  {
    v493.m128i_i64[0] = v33 + 8LL * *(unsigned int *)(a1 + 808);
    v492 = __PAIR128__(v35, v483);
    v493.m128i_i64[1] = v34;
  }
  v36 = _mm_loadu_si128(&v493);
  v37 = (_QWORD *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v36;
  if ( v493.m128i_i64[0] != v34 )
  {
    do
    {
      sub_1623E60(*v37);
      v491.m128i_i64[0] += 8;
      sub_1607E50((__int64)&v490);
      v37 = (_QWORD *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v34 );
  }
  v38 = *(_QWORD *)(a1 + 824);
  v484 = a1 + 816;
  v39 = v38 + 8LL * *(unsigned int *)(a1 + 840);
  v40 = *(_QWORD *)(a1 + 816);
  if ( *(_DWORD *)(a1 + 832) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 816);
    v493.m128i_i64[0] = v38;
    v493.m128i_i64[1] = v39;
    *(_QWORD *)&v492 = a1 + 816;
    sub_1607E80((__int64)&v492);
    v39 = *(_QWORD *)(a1 + 824) + 8LL * *(unsigned int *)(a1 + 840);
  }
  else
  {
    v493.m128i_i64[0] = v38 + 8LL * *(unsigned int *)(a1 + 840);
    v492 = __PAIR128__(v40, v484);
    v493.m128i_i64[1] = v39;
  }
  v41 = _mm_loadu_si128(&v493);
  v42 = (_QWORD *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v41;
  if ( v493.m128i_i64[0] != v39 )
  {
    do
    {
      sub_1623E60(*v42);
      v491.m128i_i64[0] += 8;
      sub_1607E80((__int64)&v490);
      v42 = (_QWORD *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v39 );
  }
  v43.m128i_i64[0] = *(_QWORD *)(a1 + 856);
  v44 = *(_QWORD *)(a1 + 848);
  if ( *(_DWORD *)(a1 + 864) )
  {
    v43.m128i_i64[1] = v43.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 872);
    v493 = v43;
    *(_QWORD *)&v492 = a1 + 848;
    *((_QWORD *)&v492 + 1) = v44;
    sub_1607EB0((__int64)&v492);
    v455 = v493.m128i_i64[1];
    v456 = (_QWORD *)v493.m128i_i64[0];
    v457 = *(_QWORD *)(a1 + 856) + 8LL * *(unsigned int *)(a1 + 872);
    if ( v457 != v493.m128i_i64[0] )
    {
      do
      {
        sub_1623E60(*v456);
        do
          ++v456;
        while ( v456 != (_QWORD *)v455 && (*v456 == -16 || *v456 == -8) );
      }
      while ( v456 != (_QWORD *)v457 );
    }
  }
  v45 = *(_QWORD *)(a1 + 888);
  v46 = v45 + 8LL * *(unsigned int *)(a1 + 904);
  v47 = *(_QWORD *)(a1 + 880);
  if ( *(_DWORD *)(a1 + 896) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 880);
    v493.m128i_i64[0] = v45;
    v493.m128i_i64[1] = v46;
    *(_QWORD *)&v492 = a1 + 880;
    sub_1607EE0((__int64)&v492);
    v46 = *(_QWORD *)(a1 + 888) + 8LL * *(unsigned int *)(a1 + 904);
  }
  else
  {
    *(_QWORD *)&v492 = a1 + 880;
    *((_QWORD *)&v492 + 1) = v47;
    v493.m128i_i64[0] = v46;
    v493.m128i_i64[1] = v46;
  }
  v48 = _mm_loadu_si128(&v493);
  v49 = (_QWORD *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v48;
  if ( v493.m128i_i64[0] != v46 )
  {
    do
    {
      sub_1623E60(*v49);
      v491.m128i_i64[0] += 8;
      sub_1607EE0((__int64)&v490);
      v49 = (_QWORD *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v46 );
  }
  v50.m128i_i64[0] = *(_QWORD *)(a1 + 920);
  v50.m128i_i64[1] = v50.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 936);
  if ( *(_DWORD *)(a1 + 928) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 912);
    *(_QWORD *)&v492 = a1 + 912;
    v493 = v50;
    sub_1607F10((__int64)&v492);
    v452 = v493.m128i_i64[1];
    v453 = (_QWORD *)v493.m128i_i64[0];
    v454 = *(_QWORD *)(a1 + 920) + 8LL * *(unsigned int *)(a1 + 936);
    if ( v493.m128i_i64[0] != v454 )
    {
      do
      {
        sub_1623E60(*v453);
        do
          ++v453;
        while ( v453 != (_QWORD *)v452 && (*v453 == -8 || *v453 == -16) );
      }
      while ( v453 != (_QWORD *)v454 );
    }
  }
  v51.m128i_i64[0] = *(_QWORD *)(a1 + 952);
  v51.m128i_i64[1] = v51.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 968);
  if ( *(_DWORD *)(a1 + 960) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 944);
    *(_QWORD *)&v492 = a1 + 944;
    v493 = v51;
    sub_1607F40((__int64)&v492);
    v449 = v493.m128i_i64[1];
    v450 = (_QWORD *)v493.m128i_i64[0];
    v451 = *(_QWORD *)(a1 + 952) + 8LL * *(unsigned int *)(a1 + 968);
    if ( v493.m128i_i64[0] != v451 )
    {
      do
      {
        sub_1623E60(*v450);
        do
          ++v450;
        while ( v450 != (_QWORD *)v449 && (*v450 == -16 || *v450 == -8) );
      }
      while ( v450 != (_QWORD *)v451 );
    }
  }
  v52.m128i_i64[0] = *(_QWORD *)(a1 + 984);
  v53 = *(_QWORD *)(a1 + 976);
  if ( *(_DWORD *)(a1 + 992) )
  {
    v52.m128i_i64[1] = v52.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 1000);
    v493 = v52;
    *(_QWORD *)&v492 = a1 + 976;
    *((_QWORD *)&v492 + 1) = v53;
    sub_1607F70((__int64)&v492);
    v446 = v493.m128i_i64[1];
    v447 = (_QWORD *)v493.m128i_i64[0];
    v448 = *(_QWORD *)(a1 + 984) + 8LL * *(unsigned int *)(a1 + 1000);
    if ( v448 != v493.m128i_i64[0] )
    {
      do
      {
        sub_1623E60(*v447);
        do
          ++v447;
        while ( v447 != (_QWORD *)v446 && (*v447 == -8 || *v447 == -16) );
      }
      while ( v447 != (_QWORD *)v448 );
    }
  }
  v54 = *(_QWORD *)(a1 + 1016);
  v55 = v54 + 8LL * *(unsigned int *)(a1 + 1032);
  v56 = *(_QWORD *)(a1 + 1008);
  if ( *(_DWORD *)(a1 + 1024) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 1008);
    v493.m128i_i64[0] = v54;
    v493.m128i_i64[1] = v55;
    *(_QWORD *)&v492 = a1 + 1008;
    sub_1607FA0((__int64)&v492);
    v55 = *(_QWORD *)(a1 + 1016) + 8LL * *(unsigned int *)(a1 + 1032);
  }
  else
  {
    *(_QWORD *)&v492 = a1 + 1008;
    *((_QWORD *)&v492 + 1) = v56;
    v493.m128i_i64[0] = v55;
    v493.m128i_i64[1] = v55;
  }
  v57 = _mm_loadu_si128(&v493);
  v58 = (_QWORD *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v57;
  if ( v493.m128i_i64[0] != v55 )
  {
    do
    {
      sub_1623E60(*v58);
      v491.m128i_i64[0] += 8;
      sub_1607FA0((__int64)&v490);
      v58 = (_QWORD *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v55 );
  }
  v59 = *(_QWORD *)(a1 + 1048);
  v485 = a1 + 1040;
  v60 = v59 + 8LL * *(unsigned int *)(a1 + 1064);
  v61 = *(_QWORD *)(a1 + 1040);
  if ( *(_DWORD *)(a1 + 1056) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 1040);
    v493.m128i_i64[0] = v59;
    v493.m128i_i64[1] = v60;
    *(_QWORD *)&v492 = a1 + 1040;
    sub_1607FD0((__int64)&v492);
    v60 = *(_QWORD *)(a1 + 1048) + 8LL * *(unsigned int *)(a1 + 1064);
  }
  else
  {
    v493.m128i_i64[0] = v59 + 8LL * *(unsigned int *)(a1 + 1064);
    v492 = __PAIR128__(v61, v485);
    v493.m128i_i64[1] = v60;
  }
  v62 = _mm_loadu_si128(&v493);
  v63 = (_QWORD *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v62;
  if ( v493.m128i_i64[0] != v60 )
  {
    do
    {
      sub_1623E60(*v63);
      v491.m128i_i64[0] += 8;
      sub_1607FD0((__int64)&v490);
      v63 = (_QWORD *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v60 );
  }
  v64 = *(_QWORD *)(a1 + 1080);
  v486 = a1 + 1072;
  v65 = v64 + 8LL * *(unsigned int *)(a1 + 1096);
  v66 = *(_QWORD *)(a1 + 1072);
  if ( *(_DWORD *)(a1 + 1088) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 1072);
    v493.m128i_i64[0] = v64;
    v493.m128i_i64[1] = v65;
    *(_QWORD *)&v492 = a1 + 1072;
    sub_1608000((__int64)&v492);
    v65 = *(_QWORD *)(a1 + 1080) + 8LL * *(unsigned int *)(a1 + 1096);
  }
  else
  {
    v493.m128i_i64[0] = v64 + 8LL * *(unsigned int *)(a1 + 1096);
    v492 = __PAIR128__(v66, v486);
    v493.m128i_i64[1] = v65;
  }
  v67 = _mm_loadu_si128(&v493);
  v68 = (_QWORD *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v67;
  if ( v493.m128i_i64[0] != v65 )
  {
    do
    {
      sub_1623E60(*v68);
      v491.m128i_i64[0] += 8;
      sub_1608000((__int64)&v490);
      v68 = (_QWORD *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v65 );
  }
  v69 = *(_QWORD *)(a1 + 1112);
  v487 = a1 + 1104;
  v70 = v69 + 8LL * *(unsigned int *)(a1 + 1128);
  v71 = *(_QWORD *)(a1 + 1104);
  if ( *(_DWORD *)(a1 + 1120) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 1104);
    v493.m128i_i64[0] = v69;
    v493.m128i_i64[1] = v70;
    *(_QWORD *)&v492 = a1 + 1104;
    sub_1608030((__int64)&v492);
    v70 = *(_QWORD *)(a1 + 1112) + 8LL * *(unsigned int *)(a1 + 1128);
  }
  else
  {
    v493.m128i_i64[0] = v69 + 8LL * *(unsigned int *)(a1 + 1128);
    v492 = __PAIR128__(v71, v487);
    v493.m128i_i64[1] = v70;
  }
  v72 = _mm_loadu_si128(&v493);
  v73 = (_QWORD *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v72;
  if ( v493.m128i_i64[0] != v70 )
  {
    do
    {
      sub_1623E60(*v73);
      v491.m128i_i64[0] += 8;
      sub_1608030((__int64)&v490);
      v73 = (_QWORD *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v70 );
  }
  v74 = *(_QWORD *)(a1 + 1144);
  v488 = a1 + 1136;
  v75 = v74 + 8LL * *(unsigned int *)(a1 + 1160);
  v76 = *(_QWORD *)(a1 + 1136);
  if ( *(_DWORD *)(a1 + 1152) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 1136);
    v493.m128i_i64[0] = v74;
    v493.m128i_i64[1] = v75;
    *(_QWORD *)&v492 = a1 + 1136;
    sub_1608060((__int64)&v492);
    v75 = *(_QWORD *)(a1 + 1144) + 8LL * *(unsigned int *)(a1 + 1160);
  }
  else
  {
    v493.m128i_i64[0] = v74 + 8LL * *(unsigned int *)(a1 + 1160);
    v492 = __PAIR128__(v76, v488);
    v493.m128i_i64[1] = v75;
  }
  v77 = _mm_loadu_si128(&v493);
  v78 = (_QWORD *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v77;
  if ( v493.m128i_i64[0] != v75 )
  {
    do
    {
      sub_1623E60(*v78);
      v491.m128i_i64[0] += 8;
      sub_1608060((__int64)&v490);
      v78 = (_QWORD *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v75 );
  }
  v79 = *(_QWORD *)(a1 + 1176);
  v489 = a1 + 1168;
  v80 = v79 + 8LL * *(unsigned int *)(a1 + 1192);
  v81 = *(_QWORD *)(a1 + 1168);
  if ( *(_DWORD *)(a1 + 1184) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 1168);
    v493.m128i_i64[0] = v79;
    v493.m128i_i64[1] = v80;
    *(_QWORD *)&v492 = a1 + 1168;
    sub_1608090((__int64)&v492);
    v80 = *(_QWORD *)(a1 + 1176) + 8LL * *(unsigned int *)(a1 + 1192);
  }
  else
  {
    v493.m128i_i64[0] = v79 + 8LL * *(unsigned int *)(a1 + 1192);
    v492 = __PAIR128__(v81, v489);
    v493.m128i_i64[1] = v80;
  }
  v82 = _mm_loadu_si128(&v493);
  v83 = (_QWORD *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v82;
  if ( v493.m128i_i64[0] != v80 )
  {
    do
    {
      sub_1623E60(*v83);
      v491.m128i_i64[0] += 8;
      sub_1608090((__int64)&v490);
      v83 = (_QWORD *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v80 );
  }
  v84 = *(_QWORD *)(a1 + 1208);
  v85 = v84 + 8LL * *(unsigned int *)(a1 + 1224);
  v86 = *(_QWORD *)(a1 + 1200);
  if ( *(_DWORD *)(a1 + 1216) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 1200);
    v493.m128i_i64[0] = v84;
    v493.m128i_i64[1] = v85;
    *(_QWORD *)&v492 = a1 + 1200;
    sub_16080C0((__int64)&v492);
    v85 = *(_QWORD *)(a1 + 1208) + 8LL * *(unsigned int *)(a1 + 1224);
  }
  else
  {
    v493.m128i_i64[0] = v84 + 8LL * *(unsigned int *)(a1 + 1224);
    v492 = __PAIR128__(v86, a1 + 1200);
    v493.m128i_i64[1] = v85;
  }
  v87 = _mm_loadu_si128(&v493);
  v88 = (_QWORD *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v87;
  if ( v493.m128i_i64[0] != v85 )
  {
    do
    {
      sub_1623E60(*v88);
      v491.m128i_i64[0] += 8;
      sub_16080C0((__int64)&v490);
      v88 = (_QWORD *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v85 );
  }
  v89.m128i_i64[0] = *(_QWORD *)(a1 + 1240);
  v90 = *(_QWORD *)(a1 + 1232);
  if ( *(_DWORD *)(a1 + 1248) )
  {
    v89.m128i_i64[1] = v89.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 1256);
    v493 = v89;
    *(_QWORD *)&v492 = a1 + 1232;
    *((_QWORD *)&v492 + 1) = v90;
    sub_16080F0((__int64)&v492);
    v443 = v493.m128i_i64[1];
    v444 = (_QWORD *)v493.m128i_i64[0];
    v445 = *(_QWORD *)(a1 + 1240) + 8LL * *(unsigned int *)(a1 + 1256);
    if ( v445 != v493.m128i_i64[0] )
    {
      do
      {
        sub_1623E60(*v444);
        do
          ++v444;
        while ( v444 != (_QWORD *)v443 && (*v444 == -16 || *v444 == -8) );
      }
      while ( v444 != (_QWORD *)v445 );
    }
  }
  v91.m128i_i64[0] = *(_QWORD *)(a1 + 1272);
  v91.m128i_i64[1] = v91.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 1288);
  if ( *(_DWORD *)(a1 + 1280) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 1264);
    *(_QWORD *)&v492 = a1 + 1264;
    v493 = v91;
    sub_1608120((__int64)&v492);
    v440 = v493.m128i_i64[1];
    v441 = (_QWORD *)v493.m128i_i64[0];
    v442 = *(_QWORD *)(a1 + 1272) + 8LL * *(unsigned int *)(a1 + 1288);
    if ( v493.m128i_i64[0] != v442 )
    {
      do
      {
        sub_1623E60(*v441);
        do
          ++v441;
        while ( v441 != (_QWORD *)v440 && (*v441 == -8 || *v441 == -16) );
      }
      while ( v441 != (_QWORD *)v442 );
    }
  }
  v92 = *(_QWORD *)(a1 + 1304);
  v93 = v92 + 8LL * *(unsigned int *)(a1 + 1320);
  v94 = *(_QWORD *)(a1 + 1296);
  if ( *(_DWORD *)(a1 + 1312) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 1296);
    v493.m128i_i64[0] = v92;
    v493.m128i_i64[1] = v93;
    *(_QWORD *)&v492 = a1 + 1296;
    sub_1608150((__int64)&v492);
    v93 = *(_QWORD *)(a1 + 1304) + 8LL * *(unsigned int *)(a1 + 1320);
  }
  else
  {
    *(_QWORD *)&v492 = a1 + 1296;
    *((_QWORD *)&v492 + 1) = v94;
    v493.m128i_i64[0] = v93;
    v493.m128i_i64[1] = v93;
  }
  v95 = _mm_loadu_si128(&v493);
  v96 = (_QWORD *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v95;
  if ( v493.m128i_i64[0] != v93 )
  {
    do
    {
      sub_1623E60(*v96);
      v491.m128i_i64[0] += 8;
      sub_1608150((__int64)&v490);
      v96 = (_QWORD *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v93 );
  }
  v97.m128i_i64[0] = *(_QWORD *)(a1 + 1336);
  v98 = *(_QWORD *)(a1 + 1328);
  if ( *(_DWORD *)(a1 + 1344) )
  {
    v97.m128i_i64[1] = v97.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 1352);
    v493 = v97;
    *(_QWORD *)&v492 = a1 + 1328;
    *((_QWORD *)&v492 + 1) = v98;
    sub_1608180((__int64)&v492);
    v437 = v493.m128i_i64[1];
    v438 = (_QWORD *)v493.m128i_i64[0];
    v439 = *(_QWORD *)(a1 + 1336) + 8LL * *(unsigned int *)(a1 + 1352);
    if ( v439 != v493.m128i_i64[0] )
    {
      do
      {
        sub_1623E60(*v438);
        do
          ++v438;
        while ( v438 != (_QWORD *)v437 && (*v438 == -8 || *v438 == -16) );
      }
      while ( v438 != (_QWORD *)v439 );
    }
  }
  v99 = *(_QWORD *)(a1 + 1368);
  v100 = v99 + 8LL * *(unsigned int *)(a1 + 1384);
  v101 = *(_QWORD *)(a1 + 1360);
  if ( *(_DWORD *)(a1 + 1376) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 1360);
    v493.m128i_i64[0] = v99;
    v493.m128i_i64[1] = v100;
    *(_QWORD *)&v492 = a1 + 1360;
    sub_16081B0((__int64)&v492);
    v100 = *(_QWORD *)(a1 + 1368) + 8LL * *(unsigned int *)(a1 + 1384);
  }
  else
  {
    *(_QWORD *)&v492 = a1 + 1360;
    *((_QWORD *)&v492 + 1) = v101;
    v493.m128i_i64[0] = v100;
    v493.m128i_i64[1] = v100;
  }
  v102 = _mm_loadu_si128(&v493);
  v103 = (_QWORD *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v102;
  if ( v493.m128i_i64[0] != v100 )
  {
    do
    {
      sub_1623E60(*v103);
      v491.m128i_i64[0] += 8;
      sub_16081B0((__int64)&v490);
      v103 = (_QWORD *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v100 );
  }
  v104.m128i_i64[0] = *(_QWORD *)(a1 + 1400);
  v105 = *(_QWORD *)(a1 + 1392);
  if ( *(_DWORD *)(a1 + 1408) )
  {
    v104.m128i_i64[1] = v104.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 1416);
    v493 = v104;
    *(_QWORD *)&v492 = a1 + 1392;
    *((_QWORD *)&v492 + 1) = v105;
    sub_16081E0((__int64)&v492);
    v434 = v493.m128i_i64[1];
    v435 = (_QWORD *)v493.m128i_i64[0];
    v436 = *(_QWORD *)(a1 + 1400) + 8LL * *(unsigned int *)(a1 + 1416);
    if ( v436 != v493.m128i_i64[0] )
    {
      do
      {
        sub_1623E60(*v435);
        do
          ++v435;
        while ( v435 != (_QWORD *)v434 && (*v435 == -8 || *v435 == -16) );
      }
      while ( v435 != (_QWORD *)v436 );
    }
  }
  v108.m128i_i64[0] = *(_QWORD *)(a1 + 1432);
  k = a1 + 1424;
  v107 = *(_QWORD *)(a1 + 1424);
  v108.m128i_i64[1] = v108.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 1448);
  if ( *(_DWORD *)(a1 + 1440) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 1424);
    *(_QWORD *)&v492 = a1 + 1424;
    v493 = v108;
    sub_1608210((__int64)&v492);
    v107 = *(unsigned int *)(a1 + 1448);
    v431 = v493.m128i_i64[1];
    v432 = (_QWORD *)v493.m128i_i64[0];
    v433 = *(_QWORD *)(a1 + 1432) + 8 * v107;
    if ( v493.m128i_i64[0] != v433 )
    {
      do
      {
        sub_1623E60(*v432);
        do
          ++v432;
        while ( v432 != (_QWORD *)v431 && (*v432 == -8 || *v432 == -16) );
      }
      while ( v432 != (_QWORD *)v433 );
    }
  }
  v109 = *(_QWORD *)(a1 + 408);
  if ( *(_DWORD *)(a1 + 416) )
  {
    v493.m128i_i64[1] = v109 + 16LL * *(unsigned int *)(a1 + 424);
    *(_QWORD *)&v492 = a1 + 400;
    v428 = *(_QWORD *)(a1 + 400);
    v493.m128i_i64[0] = v109;
    *((_QWORD *)&v492 + 1) = v428;
    sub_16070A0((__int64)&v492);
    v429 = (_QWORD *)v493.m128i_i64[0];
    for ( j = *(_QWORD *)(a1 + 408) + 16LL * *(unsigned int *)(a1 + 424);
          j != v493.m128i_i64[0];
          v429 = (_QWORD *)v493.m128i_i64[0] )
    {
      while ( 1 )
      {
        k = 0;
        sub_161EF50(v429[1] + 8LL, 0);
        v107 = v493.m128i_i64[1];
        v429 = (_QWORD *)(v493.m128i_i64[0] + 16);
        v493.m128i_i64[0] = (__int64)v429;
        if ( (_QWORD *)v493.m128i_i64[1] != v429 )
          break;
LABEL_525:
        if ( (_QWORD *)j == v429 )
          goto LABEL_86;
      }
      while ( *v429 == -8 || *v429 == -16 )
      {
        v429 += 2;
        v493.m128i_i64[0] = (__int64)v429;
        if ( (_QWORD *)v493.m128i_i64[1] == v429 )
          goto LABEL_525;
      }
    }
  }
LABEL_86:
  v110 = *(_QWORD *)(a1 + 440);
  if ( *(_DWORD *)(a1 + 448) )
  {
    v426 = *(_QWORD *)(a1 + 432);
    v493.m128i_i64[1] = v110 + 16LL * *(unsigned int *)(a1 + 456);
    v493.m128i_i64[0] = v110;
    *(_QWORD *)&v492 = a1 + 432;
    *((_QWORD *)&v492 + 1) = v426;
    sub_16070D0((__int64)&v492);
    v427 = (__int64 *)v493.m128i_i64[0];
    for ( k = *(_QWORD *)(a1 + 440) + 16LL * *(unsigned int *)(a1 + 456);
          k != v493.m128i_i64[0];
          v427 = (__int64 *)v493.m128i_i64[0] )
    {
      while ( 1 )
      {
        v110 = v427[1];
        v427 += 2;
        *(_QWORD *)(v110 + 24) = 0;
        v107 = v493.m128i_i64[1];
        v493.m128i_i64[0] = (__int64)v427;
        if ( v427 != (__int64 *)v493.m128i_i64[1] )
          break;
LABEL_516:
        if ( (__int64 *)k == v427 )
          goto LABEL_87;
      }
      while ( 1 )
      {
        v110 = *v427;
        if ( *v427 != -4 && v110 != -8 )
          break;
        v427 += 2;
        v493.m128i_i64[0] = (__int64)v427;
        if ( (__int64 *)v493.m128i_i64[1] == v427 )
          goto LABEL_516;
      }
    }
  }
LABEL_87:
  v111 = *(__int64 **)(a1 + 1504);
  for ( m = *(__int64 **)(a1 + 1496); v111 != m; ++m )
  {
    v113 = *m;
    sub_1623F10(v113, k, v110, v107);
  }
  v114 = *(_QWORD *)(a1 + 504);
  v115 = v114 + 8LL * *(unsigned int *)(a1 + 520);
  v116 = *(_QWORD *)(a1 + 496);
  if ( *(_DWORD *)(a1 + 512) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 496);
    v493.m128i_i64[0] = v114;
    v493.m128i_i64[1] = v115;
    *(_QWORD *)&v492 = a1 + 496;
    sub_1607CA0((__int64)&v492);
    v115 = *(_QWORD *)(a1 + 504) + 8LL * *(unsigned int *)(a1 + 520);
  }
  else
  {
    v493.m128i_i64[0] = v114 + 8LL * *(unsigned int *)(a1 + 520);
    v492 = __PAIR128__(v116, a1 + 496);
    v493.m128i_i64[1] = v115;
  }
  v117 = _mm_loadu_si128(&v493);
  v118 = (_QWORD *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v117;
  if ( v493.m128i_i64[0] != v115 )
  {
    do
    {
      v119 = *v118;
      if ( *v118 )
      {
        sub_1623E60(*v118);
        sub_1604260((__int64 *)(v119 + 16));
        sub_161E9C0(v119);
      }
      v491.m128i_i64[0] += 8;
      sub_1607CA0((__int64)&v490);
      v118 = (_QWORD *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v115 );
  }
  v120 = *(_QWORD *)(a1 + 536);
  v121 = v120 + 8LL * *(unsigned int *)(a1 + 552);
  v122 = *(_QWORD *)(a1 + 528);
  if ( *(_DWORD *)(a1 + 544) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 528);
    v493.m128i_i64[0] = v120;
    v493.m128i_i64[1] = v121;
    *(_QWORD *)&v492 = a1 + 528;
    sub_1607CD0((__int64)&v492);
    v121 = *(_QWORD *)(a1 + 536) + 8LL * *(unsigned int *)(a1 + 552);
  }
  else
  {
    v493.m128i_i64[0] = v120 + 8LL * *(unsigned int *)(a1 + 552);
    v492 = __PAIR128__(v122, a1 + 528);
    v493.m128i_i64[1] = v121;
  }
  v123 = _mm_loadu_si128(&v493);
  v124 = (_QWORD *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v123;
  if ( v493.m128i_i64[0] != v121 )
  {
    do
    {
      v125 = *v124;
      if ( *v124 )
      {
        sub_1623E60(*v124);
        sub_1604260((__int64 *)(v125 + 16));
        sub_161E9C0(v125);
      }
      v491.m128i_i64[0] += 8;
      sub_1607CD0((__int64)&v490);
      v124 = (_QWORD *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v121 );
  }
  v126 = *(_QWORD *)(a1 + 568);
  v127 = v126 + 8LL * *(unsigned int *)(a1 + 584);
  v128 = *(_QWORD *)(a1 + 560);
  if ( *(_DWORD *)(a1 + 576) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 560);
    v493.m128i_i64[0] = v126;
    v493.m128i_i64[1] = v127;
    *(_QWORD *)&v492 = a1 + 560;
    sub_1607D00((__int64)&v492);
    v127 = *(_QWORD *)(a1 + 568) + 8LL * *(unsigned int *)(a1 + 584);
  }
  else
  {
    v493.m128i_i64[0] = v126 + 8LL * *(unsigned int *)(a1 + 584);
    v492 = __PAIR128__(v128, a1 + 560);
    v493.m128i_i64[1] = v127;
  }
  v129 = _mm_loadu_si128(&v493);
  v130 = (__int64 **)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v129;
  if ( v493.m128i_i64[0] != v127 )
  {
    do
    {
      v131 = *v130;
      if ( *v130 )
      {
        v132 = v131[3];
        if ( v132 )
          j_j___libc_free_0(v132, v131[5] - v132);
        sub_1604260(v131 + 2);
        sub_161E9C0(v131);
      }
      v491.m128i_i64[0] += 8;
      sub_1607D00((__int64)&v490);
      v130 = (__int64 **)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v127 );
  }
  v133 = *(_QWORD *)(a1 + 600);
  v134 = v133 + 8LL * *(unsigned int *)(a1 + 616);
  v135 = *(_QWORD *)(a1 + 592);
  if ( *(_DWORD *)(a1 + 608) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 592);
    v493.m128i_i64[0] = v133;
    v493.m128i_i64[1] = v134;
    *(_QWORD *)&v492 = a1 + 592;
    sub_1607D30((__int64)&v492);
    v134 = *(_QWORD *)(a1 + 600) + 8LL * *(unsigned int *)(a1 + 616);
  }
  else
  {
    v493.m128i_i64[0] = v133 + 8LL * *(unsigned int *)(a1 + 616);
    v492 = __PAIR128__(v135, a1 + 592);
    v493.m128i_i64[1] = v134;
  }
  v136 = _mm_loadu_si128(&v493);
  v137 = (__int64 *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v136;
  if ( v493.m128i_i64[0] != v134 )
  {
    do
    {
      v138 = *v137;
      if ( *v137 )
      {
        sub_1604260((__int64 *)(v138 + 16));
        sub_161E9C0(v138);
      }
      v491.m128i_i64[0] += 8;
      sub_1607D30((__int64)&v490);
      v137 = (__int64 *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v134 );
  }
  v139 = *(_QWORD *)(a1 + 632);
  v140 = v139 + 8LL * *(unsigned int *)(a1 + 648);
  v141 = *(_QWORD *)(a1 + 624);
  if ( *(_DWORD *)(a1 + 640) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 624);
    v493.m128i_i64[0] = v139;
    v493.m128i_i64[1] = v140;
    *(_QWORD *)&v492 = a1 + 624;
    sub_1607D60((__int64)&v492);
    v140 = *(_QWORD *)(a1 + 632) + 8LL * *(unsigned int *)(a1 + 648);
  }
  else
  {
    *(_QWORD *)&v492 = a1 + 624;
    *((_QWORD *)&v492 + 1) = v141;
    v493.m128i_i64[0] = v140;
    v493.m128i_i64[1] = v140;
  }
  v142 = _mm_loadu_si128(&v493);
  v143 = (_QWORD *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v142;
  if ( v493.m128i_i64[0] != v140 )
  {
    do
    {
      v144 = *v143;
      if ( *v143 )
      {
        sub_1623E60(*v143);
        sub_1604260((__int64 *)(v144 + 16));
        sub_161E9C0(v144);
      }
      v491.m128i_i64[0] += 8;
      sub_1607D60((__int64)&v490);
      v143 = (_QWORD *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v140 );
  }
  v145.m128i_i64[0] = *(_QWORD *)(a1 + 664);
  v146 = *(_QWORD *)(a1 + 656);
  if ( *(_DWORD *)(a1 + 672) )
  {
    v145.m128i_i64[1] = v145.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 680);
    v493 = v145;
    *(_QWORD *)&v492 = a1 + 656;
    *((_QWORD *)&v492 + 1) = v146;
    sub_1607D90((__int64)&v492);
    v423 = v493.m128i_i64[1];
    v422 = (__int64 *)v493.m128i_i64[0];
    v424 = *(_QWORD *)(a1 + 664) + 8LL * *(unsigned int *)(a1 + 680);
    if ( v493.m128i_i64[0] != v424 )
    {
      do
      {
        v425 = *v422;
        if ( *v422 )
        {
          sub_1604260((__int64 *)(v425 + 16));
          sub_161E9C0(v425);
        }
        do
          ++v422;
        while ( v422 != (__int64 *)v423 && (*v422 == -8 || *v422 == -16) );
      }
      while ( v422 != (__int64 *)v424 );
    }
  }
  v147.m128i_i64[0] = *(_QWORD *)(a1 + 696);
  v147.m128i_i64[1] = v147.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 712);
  if ( *(_DWORD *)(a1 + 704) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 688);
    v493 = v147;
    *(_QWORD *)&v492 = a1 + 688;
    sub_1607DC0((__int64)&v492);
    v419 = v493.m128i_i64[1];
    v418 = (__int64 *)v493.m128i_i64[0];
    v420 = *(_QWORD *)(a1 + 696) + 8LL * *(unsigned int *)(a1 + 712);
    if ( v420 != v493.m128i_i64[0] )
    {
      do
      {
        v421 = *v418;
        if ( *v418 )
        {
          sub_1604260((__int64 *)(v421 + 16));
          sub_161E9C0(v421);
        }
        do
          ++v418;
        while ( v418 != (__int64 *)v419 && (*v418 == -16 || *v418 == -8) );
      }
      while ( v418 != (__int64 *)v420 );
    }
  }
  v148.m128i_i64[0] = *(_QWORD *)(a1 + 728);
  v149 = *(_QWORD *)(a1 + 720);
  if ( *(_DWORD *)(a1 + 736) )
  {
    v148.m128i_i64[1] = v148.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 744);
    v493 = v148;
    *(_QWORD *)&v492 = a1 + 720;
    *((_QWORD *)&v492 + 1) = v149;
    sub_1607DF0((__int64)&v492);
    v415 = v493.m128i_i64[1];
    v414 = (__int64 *)v493.m128i_i64[0];
    v416 = *(_QWORD *)(a1 + 728) + 8LL * *(unsigned int *)(a1 + 744);
    if ( v493.m128i_i64[0] != v416 )
    {
      do
      {
        v417 = *v414;
        if ( *v414 )
        {
          sub_1604260((__int64 *)(v417 + 16));
          sub_161E9C0(v417);
        }
        do
          ++v414;
        while ( v414 != (__int64 *)v415 && (*v414 == -8 || *v414 == -16) );
      }
      while ( v414 != (__int64 *)v416 );
    }
  }
  v150 = *(_QWORD *)(a1 + 760);
  v151 = v150 + 8LL * *(unsigned int *)(a1 + 776);
  v152 = *(_QWORD *)(a1 + 752);
  if ( *(_DWORD *)(a1 + 768) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 752);
    v493.m128i_i64[0] = v150;
    v493.m128i_i64[1] = v151;
    *(_QWORD *)&v492 = a1 + 752;
    sub_1607E20((__int64)&v492);
    v151 = *(_QWORD *)(a1 + 760) + 8LL * *(unsigned int *)(a1 + 776);
  }
  else
  {
    v493.m128i_i64[0] = v150 + 8LL * *(unsigned int *)(a1 + 776);
    v492 = __PAIR128__(v152, v482);
    v493.m128i_i64[1] = v151;
  }
  v153 = _mm_loadu_si128(&v493);
  v154 = (__int64 *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v153;
  if ( v493.m128i_i64[0] != v151 )
  {
    do
    {
      v155 = *v154;
      if ( *v154 )
      {
        sub_1604260((__int64 *)(v155 + 16));
        sub_161E9C0(v155);
      }
      v491.m128i_i64[0] += 8;
      sub_1607E20((__int64)&v490);
      v154 = (__int64 *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v151 );
  }
  v156 = *(_QWORD *)(a1 + 792);
  v157 = v156 + 8LL * *(unsigned int *)(a1 + 808);
  v158 = *(_QWORD *)(a1 + 784);
  if ( *(_DWORD *)(a1 + 800) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 784);
    v493.m128i_i64[0] = v156;
    v493.m128i_i64[1] = v157;
    *(_QWORD *)&v492 = a1 + 784;
    sub_1607E50((__int64)&v492);
    v157 = *(_QWORD *)(a1 + 792) + 8LL * *(unsigned int *)(a1 + 808);
  }
  else
  {
    v493.m128i_i64[0] = v156 + 8LL * *(unsigned int *)(a1 + 808);
    v492 = __PAIR128__(v158, v483);
    v493.m128i_i64[1] = v157;
  }
  v159 = _mm_loadu_si128(&v493);
  v160 = (__int64 *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v159;
  if ( v493.m128i_i64[0] != v157 )
  {
    do
    {
      v161 = *v160;
      if ( *v160 )
      {
        sub_1604260((__int64 *)(v161 + 16));
        sub_161E9C0(v161);
      }
      v491.m128i_i64[0] += 8;
      sub_1607E50((__int64)&v490);
      v160 = (__int64 *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v157 );
  }
  v162 = *(_QWORD *)(a1 + 824);
  v163 = v162 + 8LL * *(unsigned int *)(a1 + 840);
  v164 = *(_QWORD *)(a1 + 816);
  if ( *(_DWORD *)(a1 + 832) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 816);
    v493.m128i_i64[0] = v162;
    v493.m128i_i64[1] = v163;
    *(_QWORD *)&v492 = a1 + 816;
    sub_1607E80((__int64)&v492);
    v163 = *(_QWORD *)(a1 + 824) + 8LL * *(unsigned int *)(a1 + 840);
  }
  else
  {
    v493.m128i_i64[0] = v162 + 8LL * *(unsigned int *)(a1 + 840);
    v492 = __PAIR128__(v164, v484);
    v493.m128i_i64[1] = v163;
  }
  v165 = _mm_loadu_si128(&v493);
  v166 = (__int64 *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v165;
  if ( v493.m128i_i64[0] != v163 )
  {
    do
    {
      v167 = *v166;
      if ( *v166 )
      {
        sub_1604260((__int64 *)(v167 + 16));
        sub_161E9C0(v167);
      }
      v491.m128i_i64[0] += 8;
      sub_1607E80((__int64)&v490);
      v166 = (__int64 *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v163 );
  }
  v168 = *(_QWORD *)(a1 + 856);
  v169 = v168 + 8LL * *(unsigned int *)(a1 + 872);
  v170 = *(_QWORD *)(a1 + 848);
  if ( *(_DWORD *)(a1 + 864) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 848);
    v493.m128i_i64[0] = v168;
    v493.m128i_i64[1] = v169;
    *(_QWORD *)&v492 = a1 + 848;
    sub_1607EB0((__int64)&v492);
    v169 = *(_QWORD *)(a1 + 856) + 8LL * *(unsigned int *)(a1 + 872);
  }
  else
  {
    v493.m128i_i64[0] = v168 + 8LL * *(unsigned int *)(a1 + 872);
    v492 = __PAIR128__(v170, a1 + 848);
    v493.m128i_i64[1] = v169;
  }
  v171 = _mm_loadu_si128(&v493);
  v172 = (__int64 *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v171;
  if ( v493.m128i_i64[0] != v169 )
  {
    do
    {
      v173 = *v172;
      if ( *v172 )
      {
        sub_1604260((__int64 *)(v173 + 16));
        sub_161E9C0(v173);
      }
      v491.m128i_i64[0] += 8;
      sub_1607EB0((__int64)&v490);
      v172 = (__int64 *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v169 );
  }
  v174 = *(_QWORD *)(a1 + 888);
  v175 = v174 + 8LL * *(unsigned int *)(a1 + 904);
  v176 = *(_QWORD *)(a1 + 880);
  if ( *(_DWORD *)(a1 + 896) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 880);
    v493.m128i_i64[0] = v174;
    v493.m128i_i64[1] = v175;
    *(_QWORD *)&v492 = a1 + 880;
    sub_1607EE0((__int64)&v492);
    v175 = *(_QWORD *)(a1 + 888) + 8LL * *(unsigned int *)(a1 + 904);
  }
  else
  {
    v493.m128i_i64[0] = v174 + 8LL * *(unsigned int *)(a1 + 904);
    v492 = __PAIR128__(v176, a1 + 880);
    v493.m128i_i64[1] = v175;
  }
  v177 = _mm_loadu_si128(&v493);
  v178 = (__int64 *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v177;
  if ( v493.m128i_i64[0] != v175 )
  {
    do
    {
      v179 = *v178;
      if ( *v178 )
      {
        sub_1604260((__int64 *)(v179 + 16));
        sub_161E9C0(v179);
      }
      v491.m128i_i64[0] += 8;
      sub_1607EE0((__int64)&v490);
      v178 = (__int64 *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v175 );
  }
  v180.m128i_i64[0] = *(_QWORD *)(a1 + 920);
  v181 = *(_QWORD *)(a1 + 912);
  if ( *(_DWORD *)(a1 + 928) )
  {
    v180.m128i_i64[1] = v180.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 936);
    v493 = v180;
    *(_QWORD *)&v492 = a1 + 912;
    *((_QWORD *)&v492 + 1) = v181;
    sub_1607F10((__int64)&v492);
    v411 = v493.m128i_i64[1];
    v410 = (__int64 *)v493.m128i_i64[0];
    v412 = *(_QWORD *)(a1 + 920) + 8LL * *(unsigned int *)(a1 + 936);
    if ( v412 != v493.m128i_i64[0] )
    {
      do
      {
        v413 = *v410;
        if ( *v410 )
        {
          sub_1604260((__int64 *)(v413 + 16));
          sub_161E9C0(v413);
        }
        do
          ++v410;
        while ( v410 != (__int64 *)v411 && (*v410 == -8 || *v410 == -16) );
      }
      while ( v410 != (__int64 *)v412 );
    }
  }
  v182.m128i_i64[0] = *(_QWORD *)(a1 + 952);
  v183 = *(_QWORD *)(a1 + 944);
  if ( *(_DWORD *)(a1 + 960) )
  {
    v182.m128i_i64[1] = v182.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 968);
    v493 = v182;
    *(_QWORD *)&v492 = a1 + 944;
    *((_QWORD *)&v492 + 1) = v183;
    sub_1607F40((__int64)&v492);
    v407 = v493.m128i_i64[1];
    v406 = (__int64 *)v493.m128i_i64[0];
    v408 = *(_QWORD *)(a1 + 952) + 8LL * *(unsigned int *)(a1 + 968);
    if ( v493.m128i_i64[0] != v408 )
    {
      do
      {
        v409 = *v406;
        if ( *v406 )
        {
          sub_1604260((__int64 *)(v409 + 16));
          sub_161E9C0(v409);
        }
        do
          ++v406;
        while ( v406 != (__int64 *)v407 && (*v406 == -16 || *v406 == -8) );
      }
      while ( v406 != (__int64 *)v408 );
    }
  }
  v184 = *(_QWORD *)(a1 + 984);
  v185 = v184 + 8LL * *(unsigned int *)(a1 + 1000);
  v186 = *(_QWORD *)(a1 + 976);
  if ( *(_DWORD *)(a1 + 992) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 976);
    v493.m128i_i64[0] = v184;
    v493.m128i_i64[1] = v185;
    *(_QWORD *)&v492 = a1 + 976;
    sub_1607F70((__int64)&v492);
    v185 = *(_QWORD *)(a1 + 984) + 8LL * *(unsigned int *)(a1 + 1000);
  }
  else
  {
    v493.m128i_i64[0] = v184 + 8LL * *(unsigned int *)(a1 + 1000);
    v492 = __PAIR128__(v186, a1 + 976);
    v493.m128i_i64[1] = v185;
  }
  v187 = _mm_loadu_si128(&v493);
  v188 = (__int64 *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v187;
  if ( v493.m128i_i64[0] != v185 )
  {
    do
    {
      v189 = *v188;
      if ( *v188 )
      {
        sub_1604260((__int64 *)(v189 + 16));
        sub_161E9C0(v189);
      }
      v491.m128i_i64[0] += 8;
      sub_1607F70((__int64)&v490);
      v188 = (__int64 *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v185 );
  }
  v190 = *(_QWORD *)(a1 + 1016);
  v191 = v190 + 8LL * *(unsigned int *)(a1 + 1032);
  v192 = *(_QWORD *)(a1 + 1008);
  if ( *(_DWORD *)(a1 + 1024) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 1008);
    v493.m128i_i64[0] = v190;
    v493.m128i_i64[1] = v191;
    *(_QWORD *)&v492 = a1 + 1008;
    sub_1607FA0((__int64)&v492);
    v191 = *(_QWORD *)(a1 + 1016) + 8LL * *(unsigned int *)(a1 + 1032);
  }
  else
  {
    v493.m128i_i64[0] = v190 + 8LL * *(unsigned int *)(a1 + 1032);
    v492 = __PAIR128__(v192, a1 + 1008);
    v493.m128i_i64[1] = v191;
  }
  v193 = _mm_loadu_si128(&v493);
  v194 = (__int64 *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v193;
  if ( v493.m128i_i64[0] != v191 )
  {
    do
    {
      v195 = *v194;
      if ( *v194 )
      {
        sub_1604260((__int64 *)(v195 + 16));
        sub_161E9C0(v195);
      }
      v491.m128i_i64[0] += 8;
      sub_1607FA0((__int64)&v490);
      v194 = (__int64 *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v191 );
  }
  v196 = *(_QWORD *)(a1 + 1048);
  v197 = v196 + 8LL * *(unsigned int *)(a1 + 1064);
  v198 = *(_QWORD *)(a1 + 1040);
  if ( *(_DWORD *)(a1 + 1056) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 1040);
    v493.m128i_i64[0] = v196;
    v493.m128i_i64[1] = v197;
    *(_QWORD *)&v492 = a1 + 1040;
    sub_1607FD0((__int64)&v492);
    v197 = *(_QWORD *)(a1 + 1048) + 8LL * *(unsigned int *)(a1 + 1064);
  }
  else
  {
    v493.m128i_i64[0] = v196 + 8LL * *(unsigned int *)(a1 + 1064);
    v492 = __PAIR128__(v198, v485);
    v493.m128i_i64[1] = v197;
  }
  v199 = _mm_loadu_si128(&v493);
  v200 = (__int64 *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v199;
  if ( v493.m128i_i64[0] != v197 )
  {
    do
    {
      v201 = *v200;
      if ( *v200 )
      {
        sub_1604260((__int64 *)(v201 + 16));
        sub_161E9C0(v201);
      }
      v491.m128i_i64[0] += 8;
      sub_1607FD0((__int64)&v490);
      v200 = (__int64 *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v197 );
  }
  v202 = *(_QWORD *)(a1 + 1080);
  v203 = v202 + 8LL * *(unsigned int *)(a1 + 1096);
  v204 = *(_QWORD *)(a1 + 1072);
  if ( *(_DWORD *)(a1 + 1088) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 1072);
    v493.m128i_i64[0] = v202;
    v493.m128i_i64[1] = v203;
    *(_QWORD *)&v492 = a1 + 1072;
    sub_1608000((__int64)&v492);
    v203 = *(_QWORD *)(a1 + 1080) + 8LL * *(unsigned int *)(a1 + 1096);
  }
  else
  {
    v493.m128i_i64[0] = v202 + 8LL * *(unsigned int *)(a1 + 1096);
    v492 = __PAIR128__(v204, v486);
    v493.m128i_i64[1] = v203;
  }
  v205 = _mm_loadu_si128(&v493);
  v206 = (__int64 *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v205;
  if ( v493.m128i_i64[0] != v203 )
  {
    do
    {
      v207 = *v206;
      if ( *v206 )
      {
        sub_1604260((__int64 *)(v207 + 16));
        sub_161E9C0(v207);
      }
      v491.m128i_i64[0] += 8;
      sub_1608000((__int64)&v490);
      v206 = (__int64 *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v203 );
  }
  v208 = *(_QWORD *)(a1 + 1112);
  v209 = v208 + 8LL * *(unsigned int *)(a1 + 1128);
  v210 = *(_QWORD *)(a1 + 1104);
  if ( *(_DWORD *)(a1 + 1120) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 1104);
    v493.m128i_i64[0] = v208;
    v493.m128i_i64[1] = v209;
    *(_QWORD *)&v492 = a1 + 1104;
    sub_1608030((__int64)&v492);
    v209 = *(_QWORD *)(a1 + 1112) + 8LL * *(unsigned int *)(a1 + 1128);
  }
  else
  {
    v493.m128i_i64[0] = v208 + 8LL * *(unsigned int *)(a1 + 1128);
    v492 = __PAIR128__(v210, v487);
    v493.m128i_i64[1] = v209;
  }
  v211 = _mm_loadu_si128(&v493);
  v212 = (__int64 *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v211;
  if ( v493.m128i_i64[0] != v209 )
  {
    do
    {
      v213 = *v212;
      if ( *v212 )
      {
        sub_1604260((__int64 *)(v213 + 16));
        sub_161E9C0(v213);
      }
      v491.m128i_i64[0] += 8;
      sub_1608030((__int64)&v490);
      v212 = (__int64 *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v209 );
  }
  v214 = *(_QWORD *)(a1 + 1144);
  v215 = v214 + 8LL * *(unsigned int *)(a1 + 1160);
  v216 = *(_QWORD *)(a1 + 1136);
  if ( *(_DWORD *)(a1 + 1152) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 1136);
    v493.m128i_i64[0] = v214;
    v493.m128i_i64[1] = v215;
    *(_QWORD *)&v492 = a1 + 1136;
    sub_1608060((__int64)&v492);
    v215 = *(_QWORD *)(a1 + 1144) + 8LL * *(unsigned int *)(a1 + 1160);
  }
  else
  {
    v493.m128i_i64[0] = v214 + 8LL * *(unsigned int *)(a1 + 1160);
    v492 = __PAIR128__(v216, v488);
    v493.m128i_i64[1] = v215;
  }
  v217 = _mm_loadu_si128(&v493);
  v218 = (__int64 *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v217;
  if ( v493.m128i_i64[0] != v215 )
  {
    do
    {
      v219 = *v218;
      if ( *v218 )
      {
        sub_1604260((__int64 *)(v219 + 16));
        sub_161E9C0(v219);
      }
      v491.m128i_i64[0] += 8;
      sub_1608060((__int64)&v490);
      v218 = (__int64 *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v215 );
  }
  v220 = *(_QWORD *)(a1 + 1176);
  v221 = v220 + 8LL * *(unsigned int *)(a1 + 1192);
  v222 = *(_QWORD *)(a1 + 1168);
  if ( *(_DWORD *)(a1 + 1184) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 1168);
    v493.m128i_i64[0] = v220;
    v493.m128i_i64[1] = v221;
    *(_QWORD *)&v492 = a1 + 1168;
    sub_1608090((__int64)&v492);
    v221 = *(_QWORD *)(a1 + 1176) + 8LL * *(unsigned int *)(a1 + 1192);
  }
  else
  {
    v493.m128i_i64[0] = v220 + 8LL * *(unsigned int *)(a1 + 1192);
    v492 = __PAIR128__(v222, v489);
    v493.m128i_i64[1] = v221;
  }
  v223 = _mm_loadu_si128(&v493);
  v224 = (__int64 *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v223;
  if ( v493.m128i_i64[0] != v221 )
  {
    do
    {
      v225 = *v224;
      if ( *v224 )
      {
        sub_1604260((__int64 *)(v225 + 16));
        sub_161E9C0(v225);
      }
      v491.m128i_i64[0] += 8;
      sub_1608090((__int64)&v490);
      v224 = (__int64 *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v221 );
  }
  v226.m128i_i64[0] = *(_QWORD *)(a1 + 1208);
  v227 = *(_QWORD *)(a1 + 1200);
  if ( *(_DWORD *)(a1 + 1216) )
  {
    v226.m128i_i64[1] = v226.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 1224);
    v493 = v226;
    *(_QWORD *)&v492 = a1 + 1200;
    *((_QWORD *)&v492 + 1) = v227;
    sub_16080C0((__int64)&v492);
    v403 = v493.m128i_i64[1];
    v402 = (__int64 *)v493.m128i_i64[0];
    v404 = *(_QWORD *)(a1 + 1208) + 8LL * *(unsigned int *)(a1 + 1224);
    if ( v404 != v493.m128i_i64[0] )
    {
      do
      {
        v405 = *v402;
        if ( *v402 )
        {
          sub_1604260((__int64 *)(v405 + 16));
          sub_161E9C0(v405);
        }
        do
          ++v402;
        while ( v402 != (__int64 *)v403 && (*v402 == -8 || *v402 == -16) );
      }
      while ( v402 != (__int64 *)v404 );
    }
  }
  v228 = *(_QWORD *)(a1 + 1240);
  v229 = v228 + 8LL * *(unsigned int *)(a1 + 1256);
  v230 = *(_QWORD *)(a1 + 1232);
  if ( *(_DWORD *)(a1 + 1248) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 1232);
    v493.m128i_i64[0] = v228;
    v493.m128i_i64[1] = v229;
    *(_QWORD *)&v492 = a1 + 1232;
    sub_16080F0((__int64)&v492);
    v229 = *(_QWORD *)(a1 + 1240) + 8LL * *(unsigned int *)(a1 + 1256);
  }
  else
  {
    v493.m128i_i64[0] = v228 + 8LL * *(unsigned int *)(a1 + 1256);
    v492 = __PAIR128__(v230, a1 + 1232);
    v493.m128i_i64[1] = v229;
  }
  v231 = _mm_loadu_si128(&v493);
  v232 = (__int64 *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v231;
  if ( v493.m128i_i64[0] != v229 )
  {
    do
    {
      v233 = *v232;
      if ( *v232 )
      {
        sub_1604260((__int64 *)(v233 + 16));
        sub_161E9C0(v233);
      }
      v491.m128i_i64[0] += 8;
      sub_16080F0((__int64)&v490);
      v232 = (__int64 *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v229 );
  }
  v234 = *(_QWORD *)(a1 + 1272);
  v235 = v234 + 8LL * *(unsigned int *)(a1 + 1288);
  v236 = *(_QWORD *)(a1 + 1264);
  if ( *(_DWORD *)(a1 + 1280) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 1264);
    v493.m128i_i64[0] = v234;
    v493.m128i_i64[1] = v235;
    *(_QWORD *)&v492 = a1 + 1264;
    sub_1608120((__int64)&v492);
    v235 = *(_QWORD *)(a1 + 1272) + 8LL * *(unsigned int *)(a1 + 1288);
  }
  else
  {
    v493.m128i_i64[0] = v234 + 8LL * *(unsigned int *)(a1 + 1288);
    v492 = __PAIR128__(v236, a1 + 1264);
    v493.m128i_i64[1] = v235;
  }
  v237 = _mm_loadu_si128(&v493);
  v238 = (__int64 *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v237;
  if ( v493.m128i_i64[0] != v235 )
  {
    do
    {
      v239 = *v238;
      if ( *v238 )
      {
        sub_1604260((__int64 *)(v239 + 16));
        sub_161E9C0(v239);
      }
      v491.m128i_i64[0] += 8;
      sub_1608120((__int64)&v490);
      v238 = (__int64 *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v235 );
  }
  v240 = *(_QWORD *)(a1 + 1304);
  v241 = v240 + 8LL * *(unsigned int *)(a1 + 1320);
  v242 = *(_QWORD *)(a1 + 1296);
  if ( *(_DWORD *)(a1 + 1312) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 1296);
    v493.m128i_i64[0] = v240;
    v493.m128i_i64[1] = v241;
    *(_QWORD *)&v492 = a1 + 1296;
    sub_1608150((__int64)&v492);
    v241 = *(_QWORD *)(a1 + 1304) + 8LL * *(unsigned int *)(a1 + 1320);
  }
  else
  {
    v493.m128i_i64[0] = v240 + 8LL * *(unsigned int *)(a1 + 1320);
    v492 = __PAIR128__(v242, a1 + 1296);
    v493.m128i_i64[1] = v241;
  }
  v243 = _mm_loadu_si128(&v493);
  v244 = (__int64 *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v243;
  if ( v493.m128i_i64[0] != v241 )
  {
    do
    {
      v245 = *v244;
      if ( *v244 )
      {
        sub_1604260((__int64 *)(v245 + 16));
        sub_161E9C0(v245);
      }
      v491.m128i_i64[0] += 8;
      sub_1608150((__int64)&v490);
      v244 = (__int64 *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v241 );
  }
  v246 = *(_QWORD *)(a1 + 1336);
  v247 = v246 + 8LL * *(unsigned int *)(a1 + 1352);
  v248 = *(_QWORD *)(a1 + 1328);
  if ( *(_DWORD *)(a1 + 1344) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 1328);
    v493.m128i_i64[0] = v246;
    v493.m128i_i64[1] = v247;
    *(_QWORD *)&v492 = a1 + 1328;
    sub_1608180((__int64)&v492);
    v247 = *(_QWORD *)(a1 + 1336) + 8LL * *(unsigned int *)(a1 + 1352);
  }
  else
  {
    v493.m128i_i64[0] = v246 + 8LL * *(unsigned int *)(a1 + 1352);
    v492 = __PAIR128__(v248, a1 + 1328);
    v493.m128i_i64[1] = v247;
  }
  v249 = _mm_loadu_si128(&v493);
  v250 = (__int64 *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v249;
  if ( v493.m128i_i64[0] != v247 )
  {
    do
    {
      v251 = *v250;
      if ( *v250 )
      {
        sub_1604260((__int64 *)(v251 + 16));
        sub_161E9C0(v251);
      }
      v491.m128i_i64[0] += 8;
      sub_1608180((__int64)&v490);
      v250 = (__int64 *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v247 );
  }
  v252 = *(_QWORD *)(a1 + 1368);
  v253 = v252 + 8LL * *(unsigned int *)(a1 + 1384);
  v254 = *(_QWORD *)(a1 + 1360);
  if ( *(_DWORD *)(a1 + 1376) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 1360);
    v493.m128i_i64[0] = v252;
    v493.m128i_i64[1] = v253;
    *(_QWORD *)&v492 = a1 + 1360;
    sub_16081B0((__int64)&v492);
    v253 = *(_QWORD *)(a1 + 1368) + 8LL * *(unsigned int *)(a1 + 1384);
  }
  else
  {
    v493.m128i_i64[0] = v252 + 8LL * *(unsigned int *)(a1 + 1384);
    v492 = __PAIR128__(v254, a1 + 1360);
    v493.m128i_i64[1] = v253;
  }
  v255 = _mm_loadu_si128(&v493);
  v256 = (__int64 *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v255;
  if ( v493.m128i_i64[0] != v253 )
  {
    do
    {
      v257 = *v256;
      if ( *v256 )
      {
        sub_1604260((__int64 *)(v257 + 16));
        sub_161E9C0(v257);
      }
      v491.m128i_i64[0] += 8;
      sub_16081B0((__int64)&v490);
      v256 = (__int64 *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v253 );
  }
  v258 = *(_QWORD *)(a1 + 1400);
  v259 = v258 + 8LL * *(unsigned int *)(a1 + 1416);
  v260 = *(_QWORD *)(a1 + 1392);
  if ( *(_DWORD *)(a1 + 1408) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 1392);
    v493.m128i_i64[0] = v258;
    v493.m128i_i64[1] = v259;
    *(_QWORD *)&v492 = a1 + 1392;
    sub_16081E0((__int64)&v492);
    v259 = *(_QWORD *)(a1 + 1400) + 8LL * *(unsigned int *)(a1 + 1416);
  }
  else
  {
    v493.m128i_i64[0] = v258 + 8LL * *(unsigned int *)(a1 + 1416);
    v492 = __PAIR128__(v260, a1 + 1392);
    v493.m128i_i64[1] = v259;
  }
  v261 = _mm_loadu_si128(&v493);
  v262 = (__int64 *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v261;
  if ( v493.m128i_i64[0] != v259 )
  {
    do
    {
      v263 = *v262;
      if ( *v262 )
      {
        sub_1604260((__int64 *)(v263 + 16));
        sub_161E9C0(v263);
      }
      v491.m128i_i64[0] += 8;
      sub_16081E0((__int64)&v490);
      v262 = (__int64 *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v259 );
  }
  v264 = *(_QWORD *)(a1 + 1432);
  v265 = v264 + 8LL * *(unsigned int *)(a1 + 1448);
  v266 = *(_QWORD *)(a1 + 1424);
  if ( *(_DWORD *)(a1 + 1440) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 1424);
    v493.m128i_i64[0] = v264;
    v493.m128i_i64[1] = v265;
    *(_QWORD *)&v492 = a1 + 1424;
    sub_1608210((__int64)&v492);
    v265 = *(_QWORD *)(a1 + 1432) + 8LL * *(unsigned int *)(a1 + 1448);
  }
  else
  {
    v493.m128i_i64[0] = v264 + 8LL * *(unsigned int *)(a1 + 1448);
    v492 = __PAIR128__(v266, a1 + 1424);
    v493.m128i_i64[1] = v265;
  }
  v267 = _mm_loadu_si128(&v493);
  v268 = (__int64 *)v493.m128i_i64[0];
  v490 = _mm_loadu_si128((const __m128i *)&v492);
  v491 = v267;
  if ( v493.m128i_i64[0] != v265 )
  {
    do
    {
      v269 = *v268;
      if ( *v268 )
      {
        sub_1604260((__int64 *)(v269 + 16));
        sub_161E9C0(v269);
      }
      v491.m128i_i64[0] += 8;
      sub_1608210((__int64)&v490);
      v268 = (__int64 *)v491.m128i_i64[0];
    }
    while ( v491.m128i_i64[0] != v265 );
  }
  v270.m128i_i64[0] = *(_QWORD *)(a1 + 1784);
  v271 = *(_QWORD *)(a1 + 1776);
  if ( *(_DWORD *)(a1 + 1792) )
  {
    v270.m128i_i64[1] = v270.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 1800);
    v493 = v270;
    *(_QWORD *)&v492 = a1 + 1776;
    *((_QWORD *)&v492 + 1) = v271;
    sub_1608240((__int64)&v492);
    v400 = v493.m128i_i64[1];
    v399 = (_QWORD **)v493.m128i_i64[0];
    v401 = *(_QWORD *)(a1 + 1784) + 8LL * *(unsigned int *)(a1 + 1800);
    if ( v401 != v493.m128i_i64[0] )
    {
      do
      {
        sub_16041F0(*v399);
        do
          ++v399;
        while ( v399 != (_QWORD **)v400 && (*v399 == (_QWORD *)-8LL || *v399 == (_QWORD *)-16LL) );
      }
      while ( v399 != (_QWORD **)v401 );
    }
  }
  sub_1607100(&v492, (__int64 *)(a1 + 1552));
  v272 = *(_QWORD *)(a1 + 1560) + 8LL * *(unsigned int *)(a1 + 1576);
  for ( n = (_QWORD **)v493.m128i_i64[0]; v272 != v493.m128i_i64[0]; n = (_QWORD **)v493.m128i_i64[0] )
  {
    sub_16041F0(*n);
    v493.m128i_i64[0] += 8;
    sub_1608270((__int64)&v492);
  }
  v274.m128i_i64[0] = *(_QWORD *)(a1 + 1592);
  v274.m128i_i64[1] = v274.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 1608);
  if ( *(_DWORD *)(a1 + 1600) )
  {
    *((_QWORD *)&v492 + 1) = *(_QWORD *)(a1 + 1584);
    v493 = v274;
    *(_QWORD *)&v492 = a1 + 1584;
    sub_16082A0((__int64)&v492);
    v397 = v493.m128i_i64[1];
    v396 = (_QWORD **)v493.m128i_i64[0];
    v398 = *(_QWORD *)(a1 + 1592) + 8LL * *(unsigned int *)(a1 + 1608);
    if ( v398 != v493.m128i_i64[0] )
    {
      do
      {
        sub_16041F0(*v396);
        do
          ++v396;
        while ( v396 != (_QWORD **)v397 && (*v396 == (_QWORD *)-8LL || *v396 == (_QWORD *)-16LL) );
      }
      while ( v396 != (_QWORD **)v398 );
    }
  }
  v275.m128i_i64[0] = *(_QWORD *)(a1 + 1624);
  v276 = a1 + 1616;
  v277 = *(_QWORD *)(a1 + 1616);
  if ( *(_DWORD *)(a1 + 1632) )
  {
    v275.m128i_i64[1] = v275.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 1640);
    v493 = v275;
    *(_QWORD *)&v492 = a1 + 1616;
    *((_QWORD *)&v492 + 1) = v277;
    sub_16082D0((__int64)&v492);
    v394 = v493.m128i_i64[1];
    v393 = (_QWORD **)v493.m128i_i64[0];
    v395 = *(_QWORD *)(a1 + 1624) + 8LL * *(unsigned int *)(a1 + 1640);
    if ( v395 != v493.m128i_i64[0] )
    {
      do
      {
        sub_16041F0(*v393);
        do
          ++v393;
        while ( v393 != (_QWORD **)v394 && (*v393 == (_QWORD *)-16LL || *v393 == (_QWORD *)-8LL) );
      }
      while ( v393 != (_QWORD **)v395 );
    }
  }
  v278.m128i_i64[0] = *(_QWORD *)(a1 + 1784);
  v279 = *(_QWORD *)(a1 + 1776);
  if ( *(_DWORD *)(a1 + 1792) )
  {
    v276 = a1 + 1776;
    v278.m128i_i64[1] = v278.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 1800);
    v493 = v278;
    *(_QWORD *)&v492 = a1 + 1776;
    *((_QWORD *)&v492 + 1) = v279;
    sub_1608240((__int64)&v492);
    v389 = v493.m128i_i64[1];
    v390 = (_QWORD *)v493.m128i_i64[0];
    v391 = *(_QWORD *)(a1 + 1784) + 8LL * *(unsigned int *)(a1 + 1800);
    if ( v493.m128i_i64[0] != v391 )
    {
      do
      {
        v392 = *v390;
        if ( *v390 )
        {
          sub_164BE60(*v390);
          sub_1648B90(v392);
        }
        do
          ++v390;
        while ( v390 != (_QWORD *)v389 && (*v390 == -8 || *v390 == -16) );
      }
      while ( v390 != (_QWORD *)v391 );
    }
  }
  v280.m128i_i64[0] = *(_QWORD *)(a1 + 1560);
  v281 = *(_QWORD *)(a1 + 1552);
  if ( *(_DWORD *)(a1 + 1568) )
  {
    v276 = a1 + 1552;
    v280.m128i_i64[1] = v280.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 1576);
    v493 = v280;
    *(_QWORD *)&v492 = a1 + 1552;
    *((_QWORD *)&v492 + 1) = v281;
    sub_1608270((__int64)&v492);
    v386 = v493.m128i_i64[1];
    v385 = (_QWORD *)v493.m128i_i64[0];
    v387 = *(_QWORD *)(a1 + 1560) + 8LL * *(unsigned int *)(a1 + 1576);
    if ( v387 != v493.m128i_i64[0] )
    {
      do
      {
        v388 = *v385;
        if ( *v385 )
        {
          sub_164BE60(*v385);
          sub_1648B90(v388);
        }
        do
          ++v385;
        while ( v385 != (_QWORD *)v386 && (*v385 == -8 || *v385 == -16) );
      }
      while ( v385 != (_QWORD *)v387 );
    }
  }
  v282.m128i_i64[0] = *(_QWORD *)(a1 + 1592);
  v283 = *(_QWORD *)(a1 + 1584);
  if ( *(_DWORD *)(a1 + 1600) )
  {
    v276 = a1 + 1584;
    v282.m128i_i64[1] = v282.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 1608);
    v493 = v282;
    *(_QWORD *)&v492 = a1 + 1584;
    *((_QWORD *)&v492 + 1) = v283;
    sub_16082A0((__int64)&v492);
    v382 = v493.m128i_i64[1];
    v381 = (_QWORD *)v493.m128i_i64[0];
    v383 = *(_QWORD *)(a1 + 1592) + 8LL * *(unsigned int *)(a1 + 1608);
    if ( v383 != v493.m128i_i64[0] )
    {
      do
      {
        v384 = *v381;
        if ( *v381 )
        {
          sub_164BE60(*v381);
          sub_1648B90(v384);
        }
        do
          ++v381;
        while ( v381 != (_QWORD *)v382 && (*v381 == -16 || *v381 == -8) );
      }
      while ( v381 != (_QWORD *)v383 );
    }
  }
  v284.m128i_i64[0] = *(_QWORD *)(a1 + 1624);
  v285 = *(_QWORD *)(a1 + 1616);
  if ( *(_DWORD *)(a1 + 1632) )
  {
    v276 = a1 + 1616;
    v284.m128i_i64[1] = v284.m128i_i64[0] + 8LL * *(unsigned int *)(a1 + 1640);
    v493 = v284;
    *(_QWORD *)&v492 = a1 + 1616;
    *((_QWORD *)&v492 + 1) = v285;
    sub_16082D0((__int64)&v492);
    v366 = v493.m128i_i64[1];
    v365 = (_QWORD *)v493.m128i_i64[0];
    v367 = *(_QWORD *)(a1 + 1624) + 8LL * *(unsigned int *)(a1 + 1640);
    if ( v493.m128i_i64[0] != v367 )
    {
      do
      {
        v368 = *v365;
        if ( *v365 )
        {
          sub_164BE60(*v365);
          sub_1648B90(v368);
        }
        do
          ++v365;
        while ( v365 != (_QWORD *)v366 && (*v365 == -8 || *v365 == -16) );
      }
      while ( v365 != (_QWORD *)v367 );
    }
  }
  v286 = *(_QWORD **)(a1 + 1816);
  v287 = &v286[*(unsigned int *)(a1 + 1832)];
  if ( *(_DWORD *)(a1 + 1824) && v286 != v287 )
  {
    while ( *v286 == -8 || *v286 == -16 )
    {
      if ( ++v286 == v287 )
        goto LABEL_251;
    }
LABEL_660:
    if ( v286 != v287 )
    {
      v479 = (_QWORD *)*v286;
      if ( *v286 )
      {
        v480 = (_QWORD *)v479[7];
        if ( v480 != v479 + 9 )
          j_j___libc_free_0(v480, v479[9] + 1LL);
        v481 = (_QWORD *)v479[3];
        if ( v481 != v479 + 5 )
          j_j___libc_free_0(v481, v479[5] + 1LL);
        sub_164BE60(v479);
        v276 = 104;
        j_j___libc_free_0(v479, 104);
      }
      while ( ++v286 != v287 )
      {
        if ( *v286 != -8 && *v286 != -16 )
          goto LABEL_660;
      }
    }
  }
LABEL_251:
  sub_1605A70(a1 + 1520);
  sub_1605C90(a1 + 1648);
  sub_1605EB0(a1 + 1680);
  sub_1606190(a1 + 136);
  sub_1606860(a1 + 168, v276);
  v288 = *(_DWORD *)(a1 + 1720);
  if ( v288 )
  {
    v289 = *(_QWORD **)(a1 + 1712);
    if ( *v289 && *v289 != -8 )
    {
      v292 = *(__int64 **)(a1 + 1712);
    }
    else
    {
      v290 = v289 + 1;
      do
      {
        do
        {
          v291 = *v290;
          v292 = v290++;
        }
        while ( v291 == -8 );
      }
      while ( !v291 );
    }
    v293 = &v289[v288];
    while ( v293 != v292 )
    {
      while ( 1 )
      {
        v294 = *(_QWORD *)(*v292 + 8);
        if ( v294 )
        {
          sub_16042C0(*(_QWORD *)(*v292 + 8));
          sub_1648B90(v294);
        }
        v295 = v292[1];
        if ( v295 != -8 )
        {
          if ( v295 )
            break;
        }
        v296 = v292 + 2;
        do
        {
          do
          {
            v297 = *v296;
            v292 = v296++;
          }
          while ( !v297 );
        }
        while ( v297 == -8 );
        if ( v293 == v292 )
          goto LABEL_267;
      }
      ++v292;
    }
  }
LABEL_267:
  if ( *(_DWORD *)(a1 + 1724) )
  {
    v298 = *(unsigned int *)(a1 + 1720);
    v299 = 0;
    v300 = 8 * v298;
    if ( (_DWORD)v298 )
    {
      do
      {
        v301 = (unsigned __int64 *)(v299 + *(_QWORD *)(a1 + 1712));
        v302 = *v301;
        if ( *v301 != -8 && v302 )
          _libc_free(v302);
        v299 += 8;
        *v301 = 0;
      }
      while ( v300 != v299 );
    }
    *(_QWORD *)(a1 + 1724) = 0;
  }
  sub_16BDD10(&v492, *(_QWORD *)(a1 + 208));
  v303 = *(unsigned int *)(a1 + 216);
  v490.m128i_i64[0] = v492;
  sub_16BDD10(&v492, *(_QWORD *)(a1 + 208) + 8 * v303);
  v304 = v492;
  while ( 1 )
  {
    v305 = v490.m128i_i64[0];
    if ( v490.m128i_i64[0] == v304 )
      break;
    sub_16BDD40(&v490);
    if ( v305 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)(v305 - 8) + 8LL))(v305 - 8);
  }
  sub_16BDD10(&v492, *(_QWORD *)(a1 + 232));
  v306 = *(unsigned int *)(a1 + 240);
  v490.m128i_i64[0] = v492;
  sub_16BDD10(&v492, *(_QWORD *)(a1 + 232) + 8 * v306);
  v307 = v492;
  while ( 1 )
  {
    v308 = v490.m128i_i64[0];
    if ( v490.m128i_i64[0] == v307 )
      break;
    sub_16BDD40(&v490);
    if ( v308 )
      j___libc_free_0(v308);
  }
  sub_16BDD10(&v492, *(_QWORD *)(a1 + 256));
  v309 = *(unsigned int *)(a1 + 264);
  v490.m128i_i64[0] = v492;
  sub_16BDD10(&v492, *(_QWORD *)(a1 + 256) + 8 * v309);
  v310 = v492;
  while ( 1 )
  {
    v311 = v490.m128i_i64[0];
    if ( v490.m128i_i64[0] == v310 )
      break;
    sub_16BDD40(&v490);
    if ( v311 )
      j___libc_free_0(v311);
  }
  v312 = *(unsigned int *)(a1 + 448);
  *(_QWORD *)&v492 = &v493;
  *((_QWORD *)&v492 + 1) = 0x800000000LL;
  v313 = v312;
  if ( (unsigned int)v312 > 8 )
  {
    sub_16CD150(&v492, &v493, v312, 8);
    v313 = *(_DWORD *)(a1 + 448);
  }
  v314 = *(_QWORD *)(a1 + 440);
  v315 = *(_QWORD *)(a1 + 432);
  v316 = v314 + 16LL * *(unsigned int *)(a1 + 456);
  if ( !v313 )
  {
    ++*(_QWORD *)(a1 + 432);
LABEL_290:
    if ( !*(_DWORD *)(a1 + 452) )
      goto LABEL_295;
    v317 = *(unsigned int *)(a1 + 456);
    if ( (unsigned int)v317 <= 0x40 )
      goto LABEL_292;
    j___libc_free_0(*(_QWORD *)(a1 + 440));
    *(_DWORD *)(a1 + 456) = 0;
LABEL_703:
    *(_QWORD *)(a1 + 440) = 0;
LABEL_294:
    *(_QWORD *)(a1 + 448) = 0;
    goto LABEL_295;
  }
  v490.m128i_i64[1] = *(_QWORD *)(a1 + 432);
  v491.m128i_i64[0] = v314;
  v490.m128i_i64[0] = a1 + 432;
  v491.m128i_i64[1] = v316;
  sub_16070D0((__int64)&v490);
  v369 = (_QWORD *)v491.m128i_i64[0];
  v370 = *(_QWORD *)(a1 + 440) + 16LL * *(unsigned int *)(a1 + 456);
  if ( v370 != v491.m128i_i64[0] )
  {
    v371 = DWORD2(v492);
    do
    {
      while ( 1 )
      {
        if ( HIDWORD(v492) <= (unsigned int)v371 )
        {
          v315 = (__int64)&v493;
          sub_16CD150(&v492, &v493, 0, 8);
          v371 = DWORD2(v492);
        }
        *(_QWORD *)(v492 + 8 * v371) = v369[1];
        v371 = (unsigned int)++DWORD2(v492);
        v369 = (_QWORD *)(v491.m128i_i64[0] + 16);
        v491.m128i_i64[0] = (__int64)v369;
        if ( (_QWORD *)v491.m128i_i64[1] != v369 )
          break;
LABEL_391:
        if ( (_QWORD *)v370 == v369 )
          goto LABEL_392;
      }
      while ( *v369 == -4 || *v369 == -8 )
      {
        v369 += 2;
        v491.m128i_i64[0] = (__int64)v369;
        if ( (_QWORD *)v491.m128i_i64[1] == v369 )
          goto LABEL_391;
      }
      v369 = (_QWORD *)v491.m128i_i64[0];
    }
    while ( v370 != v491.m128i_i64[0] );
  }
LABEL_392:
  v372 = *(_DWORD *)(a1 + 448);
  ++*(_QWORD *)(a1 + 432);
  if ( !v372 )
    goto LABEL_290;
  v373 = 4 * v372;
  v315 = 64;
  v317 = *(unsigned int *)(a1 + 456);
  if ( (unsigned int)(4 * v372) < 0x40 )
    v373 = 64;
  if ( v373 >= (unsigned int)v317 )
  {
LABEL_292:
    v318 = *(_QWORD **)(a1 + 440);
    for ( ii = &v318[2 * v317]; ii != v318; v318 += 2 )
      *v318 = -4;
    goto LABEL_294;
  }
  v374 = v372 - 1;
  if ( v374 )
  {
    _BitScanReverse(&v374, v374);
    v375 = 1 << (33 - (v374 ^ 0x1F));
    if ( v375 < 64 )
      v375 = 64;
    if ( (_DWORD)v317 == v375 )
      goto LABEL_402;
  }
  else
  {
    v375 = 64;
  }
  j___libc_free_0(*(_QWORD *)(a1 + 440));
  v376 = sub_1603F30(v375);
  *(_DWORD *)(a1 + 456) = v376;
  if ( !v376 )
    goto LABEL_703;
  *(_QWORD *)(a1 + 440) = sub_22077B0(16LL * v376);
LABEL_402:
  sub_1607C60(a1 + 432);
LABEL_295:
  v320 = (__m128i *)v492;
  v321 = (__m128i *)(v492 + 8LL * DWORD2(v492));
  if ( (__m128i *)v492 != v321 )
  {
    do
    {
      v322 = v320->m128i_i64[0];
      if ( v320->m128i_i64[0] )
      {
        sub_161E830(v320->m128i_i64[0], v315);
        v315 = 32;
        j_j___libc_free_0(v322, 32);
      }
      v320 = (__m128i *)((char *)v320 + 8);
    }
    while ( v321 != v320 );
    v321 = (__m128i *)v492;
  }
  if ( v321 != &v493 )
    _libc_free((unsigned __int64)v321);
  v323 = *(_QWORD *)(a1 + 408);
  if ( *(_DWORD *)(a1 + 416) )
  {
    v493.m128i_i64[1] = v323 + 16LL * *(unsigned int *)(a1 + 424);
    *(_QWORD *)&v492 = a1 + 400;
    v377 = *(_QWORD *)(a1 + 400);
    v493.m128i_i64[0] = v323;
    *((_QWORD *)&v492 + 1) = v377;
    sub_16070A0((__int64)&v492);
    v378 = (_QWORD *)v493.m128i_i64[0];
    for ( jj = *(_QWORD *)(a1 + 408) + 16LL * *(unsigned int *)(a1 + 424);
          jj != v493.m128i_i64[0];
          v378 = (_QWORD *)v493.m128i_i64[0] )
    {
      while ( 1 )
      {
        v380 = v378[1];
        if ( v380 )
        {
          if ( (*(_BYTE *)(v380 + 32) & 1) == 0 )
            j___libc_free_0(*(_QWORD *)(v380 + 40));
          j_j___libc_free_0(v380, 144);
        }
        v378 = (_QWORD *)(v493.m128i_i64[0] + 16);
        v493.m128i_i64[0] = (__int64)v378;
        if ( v378 != (_QWORD *)v493.m128i_i64[1] )
          break;
LABEL_414:
        if ( (_QWORD *)jj == v378 )
          goto LABEL_303;
      }
      while ( *v378 == -8 || *v378 == -16 )
      {
        v378 += 2;
        v493.m128i_i64[0] = (__int64)v378;
        if ( (_QWORD *)v493.m128i_i64[1] == v378 )
          goto LABEL_414;
      }
    }
  }
LABEL_303:
  v324 = *(unsigned int *)(a1 + 2952);
  if ( (_DWORD)v324 )
  {
    v325 = *(_QWORD **)(a1 + 2936);
    v326 = &v325[5 * v324];
    do
    {
      if ( *v325 != -16 && *v325 != -8 )
      {
        v327 = (_QWORD *)v325[1];
        if ( v327 != v325 + 3 )
          j_j___libc_free_0(v327, v325[3] + 1LL);
      }
      v325 += 5;
    }
    while ( v326 != v325 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 2936));
  v328 = *(unsigned int *)(a1 + 2908);
  if ( (_DWORD)v328 )
  {
    v329 = *(unsigned int *)(a1 + 2904);
    v330 = *(_QWORD *)(a1 + 2896);
    if ( (_DWORD)v329 )
    {
      v331 = 8 * v329;
      v332 = 0;
      do
      {
        v333 = *(_QWORD *)(v330 + v332);
        if ( v333 && v333 != -8 )
        {
          _libc_free(v333);
          v330 = *(_QWORD *)(a1 + 2896);
        }
        v332 += 8;
      }
      while ( v332 != v331 );
    }
  }
  else
  {
    v330 = *(_QWORD *)(a1 + 2896);
  }
  _libc_free(v330);
  sub_1605A00(a1 + 2864);
  j___libc_free_0(*(_QWORD *)(a1 + 2840));
  if ( *(_DWORD *)(a1 + 2812) )
  {
    v334 = *(unsigned int *)(a1 + 2808);
    v335 = *(_QWORD *)(a1 + 2800);
    if ( (_DWORD)v334 )
    {
      v336 = 8 * v334;
      v337 = 0;
      do
      {
        v338 = *(_QWORD *)(v335 + v337);
        if ( v338 && v338 != -8 )
        {
          _libc_free(v338);
          v335 = *(_QWORD *)(a1 + 2800);
        }
        v337 += 8;
      }
      while ( v337 != v336 );
    }
  }
  else
  {
    v335 = *(_QWORD *)(a1 + 2800);
  }
  _libc_free(v335);
  j___libc_free_0(*(_QWORD *)(a1 + 2776));
  v339 = *(unsigned int *)(a1 + 2760);
  if ( (_DWORD)v339 )
  {
    v340 = *(_QWORD *)(a1 + 2744);
    v341 = v340 + 40 * v339;
    do
    {
      if ( *(_QWORD *)v340 != -8 && *(_QWORD *)v340 != -16 )
      {
        v342 = *(_QWORD *)(v340 + 8);
        v343 = v342 + 16LL * *(unsigned int *)(v340 + 16);
        if ( v342 != v343 )
        {
          do
          {
            v328 = *(_QWORD *)(v343 - 8);
            v343 -= 16LL;
            if ( v328 )
              sub_161E7C0(v343 + 8);
          }
          while ( v342 != v343 );
          v343 = *(_QWORD *)(v340 + 8);
        }
        if ( v343 != v340 + 24 )
          _libc_free(v343);
      }
      v340 += 40;
    }
    while ( v341 != v340 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 2744));
  v344 = *(unsigned int *)(a1 + 2728);
  if ( (_DWORD)v344 )
  {
    v345 = *(_QWORD *)(a1 + 2712);
    v346 = v345 + 56 * v344;
    do
    {
      if ( *(_QWORD *)v345 != -16 && *(_QWORD *)v345 != -8 )
      {
        v347 = *(_QWORD *)(v345 + 8);
        v348 = v347 + 16LL * *(unsigned int *)(v345 + 16);
        if ( v347 != v348 )
        {
          do
          {
            v328 = *(_QWORD *)(v348 - 8);
            v348 -= 16LL;
            if ( v328 )
              sub_161E7C0(v348 + 8);
          }
          while ( v347 != v348 );
          v348 = *(_QWORD *)(v345 + 8);
        }
        if ( v348 != v345 + 24 )
          _libc_free(v348);
      }
      v345 += 56;
    }
    while ( v346 != v345 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 2712));
  sub_1605A00(a1 + 2672);
  j___libc_free_0(*(_QWORD *)(a1 + 2648));
  j___libc_free_0(*(_QWORD *)(a1 + 2616));
  j___libc_free_0(*(_QWORD *)(a1 + 2584));
  j___libc_free_0(*(_QWORD *)(a1 + 2552));
  j___libc_free_0(*(_QWORD *)(a1 + 2520));
  if ( *(_DWORD *)(a1 + 2484) )
  {
    v349 = *(unsigned int *)(a1 + 2480);
    v350 = *(_QWORD *)(a1 + 2472);
    if ( (_DWORD)v349 )
    {
      v351 = 8 * v349;
      v352 = 0;
      do
      {
        v353 = *(_QWORD *)(v350 + v352);
        if ( v353 && v353 != -8 )
        {
          _libc_free(v353);
          v350 = *(_QWORD *)(a1 + 2472);
        }
        v352 += 8;
      }
      while ( v351 != v352 );
    }
  }
  else
  {
    v350 = *(_QWORD *)(a1 + 2472);
  }
  _libc_free(v350);
  j___libc_free_0(*(_QWORD *)(a1 + 2448));
  j___libc_free_0(*(_QWORD *)(a1 + 2416));
  j___libc_free_0(*(_QWORD *)(a1 + 2384));
  sub_1605960(a1 + 2272);
  v354 = *(_QWORD *)(a1 + 1856);
  if ( v354 )
  {
    sub_164BE60(*(_QWORD *)(a1 + 1856));
    sub_1648B90(v354);
  }
  j___libc_free_0(*(_QWORD *)(a1 + 1816));
  j___libc_free_0(*(_QWORD *)(a1 + 1784));
  j___libc_free_0(*(_QWORD *)(a1 + 1752));
  if ( *(_DWORD *)(a1 + 1724) )
  {
    v355 = *(unsigned int *)(a1 + 1720);
    v356 = *(_QWORD *)(a1 + 1712);
    if ( (_DWORD)v355 )
    {
      v357 = 8 * v355;
      v358 = 0;
      do
      {
        v359 = *(_QWORD *)(v356 + v358);
        if ( v359 && v359 != -8 )
        {
          _libc_free(v359);
          v356 = *(_QWORD *)(a1 + 1712);
        }
        v358 += 8;
      }
      while ( v357 != v358 );
    }
  }
  else
  {
    v356 = *(_QWORD *)(a1 + 1712);
  }
  _libc_free(v356);
  sub_1607030(a1 + 1680);
  j___libc_free_0(*(_QWORD *)(a1 + 1688));
  sub_1606FC0(a1 + 1648);
  j___libc_free_0(*(_QWORD *)(a1 + 1656));
  j___libc_free_0(*(_QWORD *)(a1 + 1624));
  j___libc_free_0(*(_QWORD *)(a1 + 1592));
  j___libc_free_0(*(_QWORD *)(a1 + 1560));
  sub_1606F50(a1 + 1520);
  j___libc_free_0(*(_QWORD *)(a1 + 1528));
  v360 = *(_QWORD *)(a1 + 1496);
  if ( v360 )
  {
    v328 = *(_QWORD *)(a1 + 1512) - v360;
    j_j___libc_free_0(v360, v328);
  }
  if ( *(_BYTE *)(a1 + 1488) )
    j___libc_free_0(*(_QWORD *)(a1 + 1464));
  j___libc_free_0(*(_QWORD *)(a1 + 1432));
  j___libc_free_0(*(_QWORD *)(a1 + 1400));
  j___libc_free_0(*(_QWORD *)(a1 + 1368));
  j___libc_free_0(*(_QWORD *)(a1 + 1336));
  j___libc_free_0(*(_QWORD *)(a1 + 1304));
  j___libc_free_0(*(_QWORD *)(a1 + 1272));
  j___libc_free_0(*(_QWORD *)(a1 + 1240));
  j___libc_free_0(*(_QWORD *)(a1 + 1208));
  j___libc_free_0(*(_QWORD *)(a1 + 1176));
  j___libc_free_0(*(_QWORD *)(a1 + 1144));
  j___libc_free_0(*(_QWORD *)(a1 + 1112));
  j___libc_free_0(*(_QWORD *)(a1 + 1080));
  j___libc_free_0(*(_QWORD *)(a1 + 1048));
  j___libc_free_0(*(_QWORD *)(a1 + 1016));
  j___libc_free_0(*(_QWORD *)(a1 + 984));
  j___libc_free_0(*(_QWORD *)(a1 + 952));
  j___libc_free_0(*(_QWORD *)(a1 + 920));
  j___libc_free_0(*(_QWORD *)(a1 + 888));
  j___libc_free_0(*(_QWORD *)(a1 + 856));
  j___libc_free_0(*(_QWORD *)(a1 + 824));
  j___libc_free_0(*(_QWORD *)(a1 + 792));
  j___libc_free_0(*(_QWORD *)(a1 + 760));
  j___libc_free_0(*(_QWORD *)(a1 + 728));
  j___libc_free_0(*(_QWORD *)(a1 + 696));
  j___libc_free_0(*(_QWORD *)(a1 + 664));
  j___libc_free_0(*(_QWORD *)(a1 + 632));
  j___libc_free_0(*(_QWORD *)(a1 + 600));
  j___libc_free_0(*(_QWORD *)(a1 + 568));
  j___libc_free_0(*(_QWORD *)(a1 + 536));
  j___libc_free_0(*(_QWORD *)(a1 + 504));
  j___libc_free_0(*(_QWORD *)(a1 + 472));
  j___libc_free_0(*(_QWORD *)(a1 + 440));
  j___libc_free_0(*(_QWORD *)(a1 + 408));
  _libc_free(*(_QWORD *)(a1 + 272));
  sub_1605960(a1 + 296);
  *(_QWORD *)(a1 + 248) = &unk_49ED4A0;
  sub_16BD9D0(a1 + 248);
  *(_QWORD *)(a1 + 224) = &unk_49ED440;
  sub_16BD9D0(a1 + 224);
  *(_QWORD *)(a1 + 200) = &unk_49ED3E0;
  sub_16BD9D0(a1 + 200);
  sub_1606530(a1 + 168, v328);
  j___libc_free_0(*(_QWORD *)(a1 + 176));
  sub_16060D0(a1 + 136);
  j___libc_free_0(*(_QWORD *)(a1 + 144));
  v361 = *(_QWORD *)(a1 + 112);
  if ( v361 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v361 + 8LL))(v361);
  v362 = *(_QWORD *)(a1 + 88);
  if ( v362 )
  {
    v363 = *(void (**)(void))(*(_QWORD *)v362 + 8LL);
    if ( (char *)v363 == (char *)sub_1602560 )
      j_j___libc_free_0(v362, 24);
    else
      v363();
  }
  v364 = *(_QWORD *)(a1 + 16);
  if ( v364 != *(_QWORD *)(a1 + 8) )
    _libc_free(v364);
}
