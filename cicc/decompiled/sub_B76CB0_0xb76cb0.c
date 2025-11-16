// Function: sub_B76CB0
// Address: 0xb76cb0
//
__int64 __fastcall sub_B76CB0(__int64 a1)
{
  __int64 v1; // r14
  int v2; // esi
  _QWORD *v3; // rax
  _QWORD *v4; // rdx
  __int64 v5; // r12
  _QWORD *v6; // rcx
  __int64 *v7; // r12
  __int64 *i; // rbx
  __int64 v9; // rdi
  __int64 v10; // rbx
  __int64 v11; // rdx
  _QWORD *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rbx
  unsigned __int64 v15; // rdx
  __m128i v16; // xmm3
  _QWORD *v17; // rax
  __int64 v18; // rax
  __int64 v19; // rbx
  unsigned __int64 v20; // rdx
  __m128i v21; // xmm5
  _QWORD *v22; // rax
  __int64 v23; // rax
  __int64 v24; // rbx
  unsigned __int64 v25; // rdx
  __m128i v26; // xmm7
  _QWORD *v27; // rax
  __int64 v28; // rax
  __int64 v29; // rbx
  unsigned __int64 v30; // rdx
  __m128i v31; // xmm1
  _QWORD *v32; // rax
  __int64 v33; // rax
  __int64 v34; // rbx
  unsigned __int64 v35; // rdx
  __m128i v36; // xmm3
  _QWORD *v37; // rax
  __int64 v38; // rax
  __int64 v39; // rbx
  unsigned __int64 v40; // rdx
  __m128i v41; // xmm5
  _QWORD *v42; // rax
  __int64 v43; // rax
  __int64 v44; // rbx
  unsigned __int64 v45; // rdx
  __m128i v46; // xmm7
  _QWORD *v47; // rax
  __int64 v48; // rax
  __int64 v49; // rbx
  unsigned __int64 v50; // rdx
  __m128i v51; // xmm1
  _QWORD *v52; // rax
  __int64 v53; // rax
  __int64 v54; // rbx
  unsigned __int64 v55; // rdx
  __m128i v56; // xmm3
  _QWORD *v57; // rax
  __int64 v58; // rax
  __int64 v59; // rbx
  unsigned __int64 v60; // rdx
  __m128i v61; // xmm5
  _QWORD *v62; // rax
  __int64 v63; // rax
  __int64 v64; // rbx
  unsigned __int64 v65; // rdx
  __m128i v66; // xmm7
  _QWORD *v67; // rax
  __int64 v68; // rax
  __int64 v69; // rbx
  unsigned __int64 v70; // rdx
  __m128i v71; // xmm1
  _QWORD *v72; // rax
  __int64 v73; // rax
  __int64 v74; // rbx
  unsigned __int64 v75; // rdx
  __m128i v76; // xmm3
  _QWORD *v77; // rax
  __int64 v78; // rax
  __int64 v79; // rbx
  unsigned __int64 v80; // rdx
  __m128i v81; // xmm5
  _QWORD *v82; // rax
  __int64 v83; // rax
  __int64 v84; // rbx
  unsigned __int64 v85; // rdx
  __m128i v86; // xmm7
  _QWORD *v87; // rax
  __int64 v88; // rax
  __int64 v89; // rbx
  unsigned __int64 v90; // rdx
  __m128i v91; // xmm1
  _QWORD *v92; // rax
  __int64 v93; // rax
  __int64 v94; // rbx
  unsigned __int64 v95; // rdx
  __m128i v96; // xmm3
  _QWORD *v97; // rax
  __int64 v98; // rax
  __int64 v99; // rbx
  unsigned __int64 v100; // rdx
  __m128i v101; // xmm5
  _QWORD *v102; // rax
  __int64 v103; // rax
  __int64 v104; // rbx
  unsigned __int64 v105; // rdx
  __m128i v106; // xmm7
  _QWORD *v107; // rax
  __int64 v108; // rax
  __int64 v109; // rbx
  unsigned __int64 v110; // rdx
  __m128i v111; // xmm1
  _QWORD *v112; // rax
  __int64 v113; // rax
  __int64 v114; // rbx
  unsigned __int64 v115; // rdx
  __m128i v116; // xmm3
  _QWORD *v117; // rax
  __int64 v118; // rax
  __int64 v119; // rbx
  unsigned __int64 v120; // rdx
  __m128i v121; // xmm5
  _QWORD *v122; // rax
  __int64 v123; // rax
  __int64 v124; // rbx
  unsigned __int64 v125; // rdx
  __m128i v126; // xmm7
  _QWORD *v127; // rax
  __int64 v128; // rax
  __int64 v129; // rbx
  unsigned __int64 v130; // rdx
  __m128i v131; // xmm1
  _QWORD *v132; // rax
  __int64 v133; // rax
  __int64 v134; // rbx
  unsigned __int64 v135; // rdx
  __m128i v136; // xmm3
  _QWORD *v137; // rax
  __int64 v138; // rax
  __int64 v139; // rbx
  unsigned __int64 v140; // rdx
  __m128i v141; // xmm5
  _QWORD *v142; // rax
  __int64 v143; // rax
  __int64 v144; // rbx
  unsigned __int64 v145; // rdx
  __m128i v146; // xmm7
  _QWORD *v147; // rax
  __int64 v148; // rax
  __int64 v149; // rbx
  unsigned __int64 v150; // rdx
  __m128i v151; // xmm1
  _QWORD *v152; // rax
  __int64 v153; // rax
  __int64 v154; // rbx
  unsigned __int64 v155; // rdx
  __m128i v156; // xmm3
  _QWORD *v157; // rax
  __int64 v158; // rdx
  __int64 v159; // rdx
  __int64 *v160; // rbx
  __int64 *v161; // r12
  __int64 *v162; // rbx
  __int64 *m; // r12
  __int64 v164; // rdi
  __int64 *v165; // r12
  __int64 *v166; // r15
  __int64 v167; // r13
  unsigned int v168; // ebx
  __int64 v169; // rax
  __int64 *v170; // r14
  __int64 v171; // rdi
  __int64 v172; // rax
  __int64 v173; // rbx
  unsigned __int64 v174; // rdx
  __m128i v175; // xmm5
  _QWORD *v176; // rax
  __int64 v177; // r12
  __int64 v178; // rax
  __int64 v179; // rbx
  unsigned __int64 v180; // rdx
  __m128i v181; // xmm7
  _QWORD *v182; // rax
  __int64 v183; // r12
  __int64 v184; // rax
  __int64 v185; // rbx
  unsigned __int64 v186; // rdx
  __m128i v187; // xmm1
  __int64 **v188; // rax
  __int64 *v189; // r12
  __int64 v190; // rdi
  __int64 v191; // rax
  __int64 v192; // rbx
  unsigned __int64 v193; // rdx
  __m128i v194; // xmm3
  __int64 *v195; // rax
  __int64 v196; // r12
  __int64 v197; // rax
  __int64 v198; // rbx
  unsigned __int64 v199; // rdx
  __m128i v200; // xmm5
  _QWORD *v201; // rax
  __int64 v202; // r12
  __int64 v203; // rax
  __int64 v204; // rbx
  unsigned __int64 v205; // rdx
  __m128i v206; // xmm7
  __int64 *v207; // rax
  __int64 v208; // r12
  __int64 v209; // rax
  __int64 v210; // rbx
  unsigned __int64 v211; // rdx
  __m128i v212; // xmm1
  __int64 *v213; // rax
  __int64 v214; // r12
  __int64 v215; // rax
  __int64 v216; // rbx
  unsigned __int64 v217; // rdx
  __m128i v218; // xmm3
  __int64 *v219; // rax
  __int64 v220; // r12
  __int64 v221; // rax
  __int64 v222; // rbx
  unsigned __int64 v223; // rdx
  __m128i v224; // xmm5
  __int64 *v225; // rax
  __int64 v226; // r12
  __int64 v227; // rax
  __int64 v228; // rbx
  unsigned __int64 v229; // rdx
  __m128i v230; // xmm7
  __int64 *v231; // rax
  __int64 v232; // r12
  __int64 v233; // rax
  __int64 v234; // rbx
  unsigned __int64 v235; // rdx
  __m128i v236; // xmm1
  __int64 *v237; // rax
  __int64 v238; // r12
  __int64 v239; // rax
  __int64 v240; // rbx
  unsigned __int64 v241; // rdx
  __m128i v242; // xmm3
  __int64 *v243; // rax
  __int64 v244; // r12
  __int64 v245; // rax
  __int64 v246; // rbx
  unsigned __int64 v247; // rdx
  __m128i v248; // xmm5
  __int64 *v249; // rax
  __int64 v250; // r12
  __int64 v251; // rax
  __int64 v252; // rbx
  unsigned __int64 v253; // rdx
  __m128i v254; // xmm7
  __int64 *v255; // rax
  __int64 v256; // r12
  __int64 v257; // rax
  __int64 v258; // rbx
  unsigned __int64 v259; // rdx
  __m128i v260; // xmm1
  __int64 *v261; // rax
  __int64 v262; // r12
  __int64 v263; // rax
  __int64 v264; // rbx
  unsigned __int64 v265; // rdx
  __m128i v266; // xmm3
  __int64 *v267; // rax
  __int64 v268; // r12
  __int64 v269; // rax
  __int64 v270; // rbx
  unsigned __int64 v271; // rdx
  __m128i v272; // xmm5
  __int64 *v273; // rax
  __int64 v274; // r12
  __int64 v275; // rax
  __int64 v276; // rbx
  unsigned __int64 v277; // rdx
  __m128i v278; // xmm7
  __int64 *v279; // rax
  __int64 v280; // r12
  __int64 v281; // rax
  __int64 v282; // rbx
  unsigned __int64 v283; // rdx
  __m128i v284; // xmm1
  __int64 *v285; // rax
  __int64 v286; // r12
  __int64 v287; // rax
  __int64 v288; // rbx
  unsigned __int64 v289; // rdx
  __m128i v290; // xmm3
  __int64 *v291; // rax
  __int64 v292; // r12
  __int64 v293; // rax
  __int64 v294; // rbx
  unsigned __int64 v295; // rdx
  __m128i v296; // xmm5
  __int64 *v297; // rax
  __int64 v298; // r12
  __int64 v299; // rax
  __int64 v300; // rbx
  unsigned __int64 v301; // rdx
  __m128i v302; // xmm7
  __int64 *v303; // rax
  __int64 v304; // r12
  __int64 v305; // rax
  __int64 v306; // rbx
  unsigned __int64 v307; // rdx
  __m128i v308; // xmm1
  __int64 *v309; // rax
  __int64 v310; // r12
  __int64 v311; // rax
  __int64 v312; // rbx
  unsigned __int64 v313; // rdx
  __m128i v314; // xmm3
  __int64 *v315; // rax
  __int64 v316; // r12
  __int64 v317; // rax
  __int64 v318; // rbx
  unsigned __int64 v319; // rdx
  __m128i v320; // xmm5
  __int64 *v321; // rax
  __int64 v322; // r12
  __int64 v323; // rax
  __int64 v324; // rbx
  unsigned __int64 v325; // rdx
  __m128i v326; // xmm7
  __int64 *v327; // rax
  __int64 v328; // r12
  __int64 v329; // rax
  __int64 v330; // rbx
  unsigned __int64 v331; // rdx
  __m128i v332; // xmm1
  __int64 *v333; // rax
  __int64 v334; // r12
  __int64 v335; // rax
  __int64 v336; // rbx
  unsigned __int64 v337; // rdx
  __m128i v338; // xmm3
  __int64 *v339; // rax
  __int64 v340; // r12
  __int64 v341; // rax
  __int64 v342; // rbx
  unsigned __int64 v343; // rdx
  __m128i v344; // xmm5
  __int64 *v345; // rax
  __int64 v346; // r12
  __int64 v347; // rax
  __int64 v348; // rbx
  unsigned __int64 v349; // rdx
  __m128i v350; // xmm7
  __int64 *v351; // rax
  __int64 v352; // r12
  __int64 v353; // rax
  __int64 v354; // rbx
  __int64 v355; // rdx
  __m128i v356; // xmm1
  __int64 *v357; // rax
  __int64 v358; // rsi
  __int64 v359; // rbx
  __int64 *n; // rax
  __int64 v361; // rax
  __int64 v362; // rbx
  unsigned __int64 v363; // rdx
  __m128i v364; // xmm3
  __int64 *v365; // rax
  __int64 v366; // rax
  __int64 v367; // rbx
  unsigned __int64 v368; // rdx
  __m128i v369; // xmm5
  __int64 *v370; // rax
  __int64 v371; // rcx
  __m128i v372; // rax
  __int64 v373; // rcx
  __m128i v374; // rax
  __m128i v375; // rax
  __int64 v376; // rcx
  __m128i v377; // rax
  __int64 v378; // rcx
  __int64 v379; // rdx
  __int64 v380; // rcx
  __int64 v381; // r8
  __int64 v382; // rdx
  __int64 v383; // rcx
  __int64 v384; // r8
  __int64 v385; // rdx
  __int64 v386; // rsi
  __int64 v387; // rbx
  __int64 v388; // r13
  __int64 v389; // r8
  int v390; // edx
  __int64 v391; // rcx
  __m128i *v392; // rdi
  __int64 v393; // r13
  _QWORD *v394; // rbx
  __int64 v395; // r15
  __int64 v396; // rdx
  __int64 v397; // rax
  _QWORD *v398; // rbx
  _QWORD *v399; // r12
  __int64 v400; // rsi
  __int64 v401; // rax
  __int64 v402; // r8
  __int64 v403; // r12
  __int64 v404; // rbx
  _QWORD *v405; // rdi
  __int64 v406; // rsi
  __int64 v407; // rax
  _QWORD *v408; // rbx
  _QWORD *v409; // r12
  _QWORD *v410; // rdi
  __int64 v411; // rsi
  __int64 v412; // rdx
  __int64 v413; // r13
  __int64 v414; // rbx
  __int64 v415; // r15
  __int64 v416; // r12
  __int64 v417; // rsi
  __int64 v418; // rsi
  __int64 v419; // rax
  __int64 v420; // r8
  __int64 v421; // r12
  __int64 v422; // rbx
  _QWORD *v423; // rdi
  __int64 v424; // rsi
  __int64 v425; // rsi
  __int64 v426; // r12
  __int64 v427; // rsi
  __int64 v428; // rax
  __int64 v429; // rdi
  __int64 v430; // rbx
  __int64 v431; // r12
  _QWORD *v432; // r13
  __int64 v433; // r8
  __int64 v434; // r15
  __int64 v435; // r9
  __int64 v436; // r10
  __int64 v437; // rdi
  __int64 v438; // rdi
  __int64 v439; // rax
  __int64 v440; // rsi
  __int64 v441; // rsi
  __int64 v442; // rdx
  __int64 v443; // rcx
  __int64 v444; // r8
  __int64 v445; // rsi
  __int64 v446; // rdx
  __int64 v447; // rcx
  __int64 v448; // r8
  __int64 v449; // rdi
  __int64 v450; // rdi
  void (*v451)(void); // rax
  __int64 v452; // r12
  __int64 v453; // rdi
  __int64 v454; // rsi
  __int64 result; // rax
  __int64 *v456; // rbx
  __int64 v457; // r13
  __int64 v458; // r12
  __int64 *v459; // r12
  __int64 v460; // r13
  __int64 v461; // rbx
  __int64 *v462; // r12
  __int64 v463; // r15
  __int64 v464; // rbx
  __int64 v465; // rcx
  __int64 v466; // rax
  __int64 jj; // rbx
  __int64 v468; // r12
  __int64 v469; // rdx
  __int64 v470; // rax
  __int64 ii; // rbx
  __int64 v472; // r15
  __int64 v473; // rax
  unsigned __int64 v474; // rdx
  __int64 v475; // rcx
  __int64 v476; // rax
  __int64 j; // rbx
  __int64 v478; // r13
  __int64 *v479; // r15
  __int64 v480; // rbx
  __int64 v481; // rax
  __int64 k; // rbx
  __int64 v483; // rsi
  __int64 v484; // r13
  __int64 v485; // rdi
  __int64 v486; // [rsp+8h] [rbp-1A8h]
  __int64 v487; // [rsp+10h] [rbp-1A0h]
  __int64 v488; // [rsp+18h] [rbp-198h]
  __int64 v489; // [rsp+20h] [rbp-190h]
  __int64 v490; // [rsp+28h] [rbp-188h]
  __int64 v491; // [rsp+30h] [rbp-180h]
  __int64 v492; // [rsp+38h] [rbp-178h]
  __int64 v493; // [rsp+40h] [rbp-170h]
  __int64 v494; // [rsp+48h] [rbp-168h]
  __int64 v495; // [rsp+50h] [rbp-160h]
  __int64 v496; // [rsp+58h] [rbp-158h]
  __int64 v497; // [rsp+60h] [rbp-150h]
  __int64 v498; // [rsp+68h] [rbp-148h]
  __int64 v499; // [rsp+70h] [rbp-140h]
  __int64 v500; // [rsp+78h] [rbp-138h]
  __int64 v501; // [rsp+80h] [rbp-130h]
  __int64 v502; // [rsp+88h] [rbp-128h]
  __int64 v503; // [rsp+90h] [rbp-120h]
  __int64 v504; // [rsp+98h] [rbp-118h]
  __int64 v505; // [rsp+A0h] [rbp-110h]
  __int64 v506; // [rsp+A0h] [rbp-110h]
  __int64 v507; // [rsp+A8h] [rbp-108h]
  __int64 v508; // [rsp+B0h] [rbp-100h]
  __int64 v509; // [rsp+B8h] [rbp-F8h]
  __int64 v510; // [rsp+C0h] [rbp-F0h]
  __int64 v511; // [rsp+C8h] [rbp-E8h]
  __int64 v512; // [rsp+D0h] [rbp-E0h]
  __int64 v513; // [rsp+D8h] [rbp-D8h]
  __int64 v514; // [rsp+E0h] [rbp-D0h]
  __int64 v515; // [rsp+E8h] [rbp-C8h]
  __int64 v516; // [rsp+F0h] [rbp-C0h]
  __int64 v517; // [rsp+F0h] [rbp-C0h]
  __int64 v518; // [rsp+F8h] [rbp-B8h]
  __int64 v519; // [rsp+F8h] [rbp-B8h]
  __int64 v520; // [rsp+100h] [rbp-B0h]
  __int64 v521; // [rsp+100h] [rbp-B0h]
  __int64 v522; // [rsp+100h] [rbp-B0h]
  __int64 v523; // [rsp+108h] [rbp-A8h]
  __int64 v524; // [rsp+108h] [rbp-A8h]
  __int64 v525; // [rsp+108h] [rbp-A8h]
  __m128i v526; // [rsp+110h] [rbp-A0h] BYREF
  __m128i v527; // [rsp+120h] [rbp-90h]
  unsigned __int128 v528; // [rsp+130h] [rbp-80h] BYREF
  __m128i v529; // [rsp+140h] [rbp-70h] BYREF

  v1 = a1;
  v2 = *(_DWORD *)(a1 + 20);
  while ( *(_DWORD *)(a1 + 24) != v2 )
  {
    v3 = *(_QWORD **)(a1 + 8);
    if ( *(_BYTE *)(a1 + 28) )
      v4 = &v3[v2];
    else
      v4 = &v3[*(unsigned int *)(a1 + 16)];
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
      sub_BA9C10(v5);
      j_j___libc_free_0(v5, 880);
      v2 = *(_DWORD *)(a1 + 20);
    }
  }
  v7 = *(__int64 **)(a1 + 1672);
  for ( i = *(__int64 **)(a1 + 1664); v7 != i; ++i )
  {
    v9 = *i;
    sub_B972A0(v9);
  }
  v487 = v1 + 664;
  v10 = *(_QWORD *)(v1 + 672) + 8LL * *(unsigned int *)(v1 + 688);
  v11 = *(_QWORD *)(v1 + 664);
  if ( *(_DWORD *)(v1 + 680) )
  {
    v529.m128i_i64[0] = *(_QWORD *)(v1 + 672);
    *((_QWORD *)&v528 + 1) = v11;
    v529.m128i_i64[1] = v10;
    *(_QWORD *)&v528 = v1 + 664;
    sub_B76430((__int64)&v528);
    v10 = *(_QWORD *)(v1 + 672) + 8LL * *(unsigned int *)(v1 + 688);
  }
  else
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 664);
    v529.m128i_i64[0] = v10;
    *(_QWORD *)&v528 = v1 + 664;
    v529.m128i_i64[1] = v10;
  }
  v12 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = _mm_loadu_si128(&v529);
  if ( v529.m128i_i64[0] != v10 )
  {
    do
    {
      sub_B972A0(*v12);
      v527.m128i_i64[0] += 8;
      sub_B76430((__int64)&v526);
      v12 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v10 );
  }
  v13 = *(_QWORD *)(v1 + 704);
  v488 = v1 + 696;
  v14 = v13 + 8LL * *(unsigned int *)(v1 + 720);
  v15 = *(_QWORD *)(v1 + 696);
  if ( *(_DWORD *)(v1 + 712) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 696);
    v529.m128i_i64[0] = v13;
    v529.m128i_i64[1] = v14;
    *(_QWORD *)&v528 = v1 + 696;
    sub_B76470((__int64)&v528);
    v14 = *(_QWORD *)(v1 + 704) + 8LL * *(unsigned int *)(v1 + 720);
  }
  else
  {
    v529.m128i_i64[0] = v13 + 8LL * *(unsigned int *)(v1 + 720);
    v528 = __PAIR128__(v15, v488);
    v529.m128i_i64[1] = v14;
  }
  v16 = _mm_loadu_si128(&v529);
  v17 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v16;
  if ( v529.m128i_i64[0] != v14 )
  {
    do
    {
      sub_B972A0(*v17);
      v527.m128i_i64[0] += 8;
      sub_B76470((__int64)&v526);
      v17 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v14 );
  }
  v18 = *(_QWORD *)(v1 + 736);
  v489 = v1 + 728;
  v19 = v18 + 8LL * *(unsigned int *)(v1 + 752);
  v20 = *(_QWORD *)(v1 + 728);
  if ( *(_DWORD *)(v1 + 744) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 728);
    v529.m128i_i64[0] = v18;
    v529.m128i_i64[1] = v19;
    *(_QWORD *)&v528 = v1 + 728;
    sub_B764B0((__int64)&v528);
    v19 = *(_QWORD *)(v1 + 736) + 8LL * *(unsigned int *)(v1 + 752);
  }
  else
  {
    v529.m128i_i64[0] = v18 + 8LL * *(unsigned int *)(v1 + 752);
    v528 = __PAIR128__(v20, v489);
    v529.m128i_i64[1] = v19;
  }
  v21 = _mm_loadu_si128(&v529);
  v22 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v21;
  if ( v529.m128i_i64[0] != v19 )
  {
    do
    {
      sub_B972A0(*v22);
      v527.m128i_i64[0] += 8;
      sub_B764B0((__int64)&v526);
      v22 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v19 );
  }
  v23 = *(_QWORD *)(v1 + 768);
  v490 = v1 + 760;
  v24 = v23 + 8LL * *(unsigned int *)(v1 + 784);
  v25 = *(_QWORD *)(v1 + 760);
  if ( *(_DWORD *)(v1 + 776) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 760);
    v529.m128i_i64[0] = v23;
    v529.m128i_i64[1] = v24;
    *(_QWORD *)&v528 = v1 + 760;
    sub_B764F0((__int64)&v528);
    v24 = *(_QWORD *)(v1 + 768) + 8LL * *(unsigned int *)(v1 + 784);
  }
  else
  {
    v529.m128i_i64[0] = v23 + 8LL * *(unsigned int *)(v1 + 784);
    v528 = __PAIR128__(v25, v490);
    v529.m128i_i64[1] = v24;
  }
  v26 = _mm_loadu_si128(&v529);
  v27 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v26;
  if ( v529.m128i_i64[0] != v24 )
  {
    do
    {
      sub_B972A0(*v27);
      v527.m128i_i64[0] += 8;
      sub_B764F0((__int64)&v526);
      v27 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v24 );
  }
  v28 = *(_QWORD *)(v1 + 800);
  v491 = v1 + 792;
  v29 = v28 + 8LL * *(unsigned int *)(v1 + 816);
  v30 = *(_QWORD *)(v1 + 792);
  if ( *(_DWORD *)(v1 + 808) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 792);
    v529.m128i_i64[0] = v28;
    v529.m128i_i64[1] = v29;
    *(_QWORD *)&v528 = v1 + 792;
    sub_B76530((__int64)&v528);
    v29 = *(_QWORD *)(v1 + 800) + 8LL * *(unsigned int *)(v1 + 816);
  }
  else
  {
    v529.m128i_i64[0] = v28 + 8LL * *(unsigned int *)(v1 + 816);
    v528 = __PAIR128__(v30, v491);
    v529.m128i_i64[1] = v29;
  }
  v31 = _mm_loadu_si128(&v529);
  v32 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v31;
  if ( v529.m128i_i64[0] != v29 )
  {
    do
    {
      sub_B972A0(*v32);
      v527.m128i_i64[0] += 8;
      sub_B76530((__int64)&v526);
      v32 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v29 );
  }
  v33 = *(_QWORD *)(v1 + 832);
  v492 = v1 + 824;
  v34 = v33 + 8LL * *(unsigned int *)(v1 + 848);
  v35 = *(_QWORD *)(v1 + 824);
  if ( *(_DWORD *)(v1 + 840) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 824);
    v529.m128i_i64[0] = v33;
    v529.m128i_i64[1] = v34;
    *(_QWORD *)&v528 = v1 + 824;
    sub_B76570((__int64)&v528);
    v34 = *(_QWORD *)(v1 + 832) + 8LL * *(unsigned int *)(v1 + 848);
  }
  else
  {
    v529.m128i_i64[0] = v33 + 8LL * *(unsigned int *)(v1 + 848);
    v528 = __PAIR128__(v35, v492);
    v529.m128i_i64[1] = v34;
  }
  v36 = _mm_loadu_si128(&v529);
  v37 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v36;
  if ( v529.m128i_i64[0] != v34 )
  {
    do
    {
      sub_B972A0(*v37);
      v527.m128i_i64[0] += 8;
      sub_B76570((__int64)&v526);
      v37 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v34 );
  }
  v38 = *(_QWORD *)(v1 + 864);
  v493 = v1 + 856;
  v39 = v38 + 8LL * *(unsigned int *)(v1 + 880);
  v40 = *(_QWORD *)(v1 + 856);
  if ( *(_DWORD *)(v1 + 872) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 856);
    v529.m128i_i64[0] = v38;
    v529.m128i_i64[1] = v39;
    *(_QWORD *)&v528 = v1 + 856;
    sub_B765B0((__int64)&v528);
    v39 = *(_QWORD *)(v1 + 864) + 8LL * *(unsigned int *)(v1 + 880);
  }
  else
  {
    v529.m128i_i64[0] = v38 + 8LL * *(unsigned int *)(v1 + 880);
    v528 = __PAIR128__(v40, v493);
    v529.m128i_i64[1] = v39;
  }
  v41 = _mm_loadu_si128(&v529);
  v42 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v41;
  if ( v529.m128i_i64[0] != v39 )
  {
    do
    {
      sub_B972A0(*v42);
      v527.m128i_i64[0] += 8;
      sub_B765B0((__int64)&v526);
      v42 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v39 );
  }
  v43 = *(_QWORD *)(v1 + 896);
  v494 = v1 + 888;
  v44 = v43 + 8LL * *(unsigned int *)(v1 + 912);
  v45 = *(_QWORD *)(v1 + 888);
  if ( *(_DWORD *)(v1 + 904) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 888);
    v529.m128i_i64[0] = v43;
    v529.m128i_i64[1] = v44;
    *(_QWORD *)&v528 = v1 + 888;
    sub_B765F0((__int64)&v528);
    v44 = *(_QWORD *)(v1 + 896) + 8LL * *(unsigned int *)(v1 + 912);
  }
  else
  {
    v529.m128i_i64[0] = v43 + 8LL * *(unsigned int *)(v1 + 912);
    v528 = __PAIR128__(v45, v494);
    v529.m128i_i64[1] = v44;
  }
  v46 = _mm_loadu_si128(&v529);
  v47 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v46;
  if ( v529.m128i_i64[0] != v44 )
  {
    do
    {
      sub_B972A0(*v47);
      v527.m128i_i64[0] += 8;
      sub_B765F0((__int64)&v526);
      v47 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v44 );
  }
  v48 = *(_QWORD *)(v1 + 928);
  v495 = v1 + 920;
  v49 = v48 + 8LL * *(unsigned int *)(v1 + 944);
  v50 = *(_QWORD *)(v1 + 920);
  if ( *(_DWORD *)(v1 + 936) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 920);
    v529.m128i_i64[0] = v48;
    v529.m128i_i64[1] = v49;
    *(_QWORD *)&v528 = v1 + 920;
    sub_B76630((__int64)&v528);
    v49 = *(_QWORD *)(v1 + 928) + 8LL * *(unsigned int *)(v1 + 944);
  }
  else
  {
    v529.m128i_i64[0] = v48 + 8LL * *(unsigned int *)(v1 + 944);
    v528 = __PAIR128__(v50, v495);
    v529.m128i_i64[1] = v49;
  }
  v51 = _mm_loadu_si128(&v529);
  v52 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v51;
  if ( v529.m128i_i64[0] != v49 )
  {
    do
    {
      sub_B972A0(*v52);
      v527.m128i_i64[0] += 8;
      sub_B76630((__int64)&v526);
      v52 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v49 );
  }
  v53 = *(_QWORD *)(v1 + 960);
  v496 = v1 + 952;
  v54 = v53 + 8LL * *(unsigned int *)(v1 + 976);
  v55 = *(_QWORD *)(v1 + 952);
  if ( *(_DWORD *)(v1 + 968) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 952);
    v529.m128i_i64[0] = v53;
    v529.m128i_i64[1] = v54;
    *(_QWORD *)&v528 = v1 + 952;
    sub_B76670((__int64)&v528);
    v54 = *(_QWORD *)(v1 + 960) + 8LL * *(unsigned int *)(v1 + 976);
  }
  else
  {
    v529.m128i_i64[0] = v53 + 8LL * *(unsigned int *)(v1 + 976);
    v528 = __PAIR128__(v55, v496);
    v529.m128i_i64[1] = v54;
  }
  v56 = _mm_loadu_si128(&v529);
  v57 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v56;
  if ( v529.m128i_i64[0] != v54 )
  {
    do
    {
      sub_B972A0(*v57);
      v527.m128i_i64[0] += 8;
      sub_B76670((__int64)&v526);
      v57 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v54 );
  }
  v58 = *(_QWORD *)(v1 + 992);
  v497 = v1 + 984;
  v59 = v58 + 8LL * *(unsigned int *)(v1 + 1008);
  v60 = *(_QWORD *)(v1 + 984);
  if ( *(_DWORD *)(v1 + 1000) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 984);
    v529.m128i_i64[0] = v58;
    v529.m128i_i64[1] = v59;
    *(_QWORD *)&v528 = v1 + 984;
    sub_B766B0((__int64)&v528);
    v59 = *(_QWORD *)(v1 + 992) + 8LL * *(unsigned int *)(v1 + 1008);
  }
  else
  {
    v529.m128i_i64[0] = v58 + 8LL * *(unsigned int *)(v1 + 1008);
    v528 = __PAIR128__(v60, v497);
    v529.m128i_i64[1] = v59;
  }
  v61 = _mm_loadu_si128(&v529);
  v62 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v61;
  if ( v529.m128i_i64[0] != v59 )
  {
    do
    {
      sub_B972A0(*v62);
      v527.m128i_i64[0] += 8;
      sub_B766B0((__int64)&v526);
      v62 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v59 );
  }
  v63 = *(_QWORD *)(v1 + 1024);
  v498 = v1 + 1016;
  v64 = v63 + 8LL * *(unsigned int *)(v1 + 1040);
  v65 = *(_QWORD *)(v1 + 1016);
  if ( *(_DWORD *)(v1 + 1032) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1016);
    v529.m128i_i64[0] = v63;
    v529.m128i_i64[1] = v64;
    *(_QWORD *)&v528 = v1 + 1016;
    sub_B766F0((__int64)&v528);
    v64 = *(_QWORD *)(v1 + 1024) + 8LL * *(unsigned int *)(v1 + 1040);
  }
  else
  {
    v529.m128i_i64[0] = v63 + 8LL * *(unsigned int *)(v1 + 1040);
    v528 = __PAIR128__(v65, v498);
    v529.m128i_i64[1] = v64;
  }
  v66 = _mm_loadu_si128(&v529);
  v67 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v66;
  if ( v529.m128i_i64[0] != v64 )
  {
    do
    {
      sub_B972A0(*v67);
      v527.m128i_i64[0] += 8;
      sub_B766F0((__int64)&v526);
      v67 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v64 );
  }
  v68 = *(_QWORD *)(v1 + 1056);
  v499 = v1 + 1048;
  v69 = v68 + 8LL * *(unsigned int *)(v1 + 1072);
  v70 = *(_QWORD *)(v1 + 1048);
  if ( *(_DWORD *)(v1 + 1064) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1048);
    v529.m128i_i64[0] = v68;
    v529.m128i_i64[1] = v69;
    *(_QWORD *)&v528 = v1 + 1048;
    sub_B76730((__int64)&v528);
    v69 = *(_QWORD *)(v1 + 1056) + 8LL * *(unsigned int *)(v1 + 1072);
  }
  else
  {
    v529.m128i_i64[0] = v68 + 8LL * *(unsigned int *)(v1 + 1072);
    v528 = __PAIR128__(v70, v499);
    v529.m128i_i64[1] = v69;
  }
  v71 = _mm_loadu_si128(&v529);
  v72 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v71;
  if ( v529.m128i_i64[0] != v69 )
  {
    do
    {
      sub_B972A0(*v72);
      v527.m128i_i64[0] += 8;
      sub_B76730((__int64)&v526);
      v72 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v69 );
  }
  v73 = *(_QWORD *)(v1 + 1088);
  v500 = v1 + 1080;
  v74 = v73 + 8LL * *(unsigned int *)(v1 + 1104);
  v75 = *(_QWORD *)(v1 + 1080);
  if ( *(_DWORD *)(v1 + 1096) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1080);
    v529.m128i_i64[0] = v73;
    v529.m128i_i64[1] = v74;
    *(_QWORD *)&v528 = v1 + 1080;
    sub_B76770((__int64)&v528);
    v74 = *(_QWORD *)(v1 + 1088) + 8LL * *(unsigned int *)(v1 + 1104);
  }
  else
  {
    v529.m128i_i64[0] = v73 + 8LL * *(unsigned int *)(v1 + 1104);
    v528 = __PAIR128__(v75, v500);
    v529.m128i_i64[1] = v74;
  }
  v76 = _mm_loadu_si128(&v529);
  v77 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v76;
  if ( v529.m128i_i64[0] != v74 )
  {
    do
    {
      sub_B972A0(*v77);
      v527.m128i_i64[0] += 8;
      sub_B76770((__int64)&v526);
      v77 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v74 );
  }
  v78 = *(_QWORD *)(v1 + 1120);
  v501 = v1 + 1112;
  v79 = v78 + 8LL * *(unsigned int *)(v1 + 1136);
  v80 = *(_QWORD *)(v1 + 1112);
  if ( *(_DWORD *)(v1 + 1128) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1112);
    v529.m128i_i64[0] = v78;
    v529.m128i_i64[1] = v79;
    *(_QWORD *)&v528 = v1 + 1112;
    sub_B767B0((__int64)&v528);
    v79 = *(_QWORD *)(v1 + 1120) + 8LL * *(unsigned int *)(v1 + 1136);
  }
  else
  {
    v529.m128i_i64[0] = v78 + 8LL * *(unsigned int *)(v1 + 1136);
    v528 = __PAIR128__(v80, v501);
    v529.m128i_i64[1] = v79;
  }
  v81 = _mm_loadu_si128(&v529);
  v82 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v81;
  if ( v529.m128i_i64[0] != v79 )
  {
    do
    {
      sub_B972A0(*v82);
      v527.m128i_i64[0] += 8;
      sub_B767B0((__int64)&v526);
      v82 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v79 );
  }
  v83 = *(_QWORD *)(v1 + 1152);
  v502 = v1 + 1144;
  v84 = v83 + 8LL * *(unsigned int *)(v1 + 1168);
  v85 = *(_QWORD *)(v1 + 1144);
  if ( *(_DWORD *)(v1 + 1160) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1144);
    v529.m128i_i64[0] = v83;
    v529.m128i_i64[1] = v84;
    *(_QWORD *)&v528 = v1 + 1144;
    sub_B767F0((__int64)&v528);
    v84 = *(_QWORD *)(v1 + 1152) + 8LL * *(unsigned int *)(v1 + 1168);
  }
  else
  {
    v529.m128i_i64[0] = v83 + 8LL * *(unsigned int *)(v1 + 1168);
    v528 = __PAIR128__(v85, v502);
    v529.m128i_i64[1] = v84;
  }
  v86 = _mm_loadu_si128(&v529);
  v87 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v86;
  if ( v529.m128i_i64[0] != v84 )
  {
    do
    {
      sub_B972A0(*v87);
      v527.m128i_i64[0] += 8;
      sub_B767F0((__int64)&v526);
      v87 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v84 );
  }
  v88 = *(_QWORD *)(v1 + 1184);
  v503 = v1 + 1176;
  v89 = v88 + 8LL * *(unsigned int *)(v1 + 1200);
  v90 = *(_QWORD *)(v1 + 1176);
  if ( *(_DWORD *)(v1 + 1192) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1176);
    v529.m128i_i64[0] = v88;
    v529.m128i_i64[1] = v89;
    *(_QWORD *)&v528 = v1 + 1176;
    sub_B76830((__int64)&v528);
    v89 = *(_QWORD *)(v1 + 1184) + 8LL * *(unsigned int *)(v1 + 1200);
  }
  else
  {
    v529.m128i_i64[0] = v88 + 8LL * *(unsigned int *)(v1 + 1200);
    v528 = __PAIR128__(v90, v503);
    v529.m128i_i64[1] = v89;
  }
  v91 = _mm_loadu_si128(&v529);
  v92 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v91;
  if ( v529.m128i_i64[0] != v89 )
  {
    do
    {
      sub_B972A0(*v92);
      v527.m128i_i64[0] += 8;
      sub_B76830((__int64)&v526);
      v92 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v89 );
  }
  v93 = *(_QWORD *)(v1 + 1216);
  v504 = v1 + 1208;
  v94 = v93 + 8LL * *(unsigned int *)(v1 + 1232);
  v95 = *(_QWORD *)(v1 + 1208);
  if ( *(_DWORD *)(v1 + 1224) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1208);
    v529.m128i_i64[0] = v93;
    v529.m128i_i64[1] = v94;
    *(_QWORD *)&v528 = v1 + 1208;
    sub_B76870((__int64)&v528);
    v94 = *(_QWORD *)(v1 + 1216) + 8LL * *(unsigned int *)(v1 + 1232);
  }
  else
  {
    v529.m128i_i64[0] = v93 + 8LL * *(unsigned int *)(v1 + 1232);
    v528 = __PAIR128__(v95, v504);
    v529.m128i_i64[1] = v94;
  }
  v96 = _mm_loadu_si128(&v529);
  v97 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v96;
  if ( v529.m128i_i64[0] != v94 )
  {
    do
    {
      sub_B972A0(*v97);
      v527.m128i_i64[0] += 8;
      sub_B76870((__int64)&v526);
      v97 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v94 );
  }
  v98 = *(_QWORD *)(v1 + 1248);
  v505 = v1 + 1240;
  v99 = v98 + 8LL * *(unsigned int *)(v1 + 1264);
  v100 = *(_QWORD *)(v1 + 1240);
  if ( *(_DWORD *)(v1 + 1256) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1240);
    v529.m128i_i64[0] = v98;
    v529.m128i_i64[1] = v99;
    *(_QWORD *)&v528 = v1 + 1240;
    sub_B768B0((__int64)&v528);
    v99 = *(_QWORD *)(v1 + 1248) + 8LL * *(unsigned int *)(v1 + 1264);
  }
  else
  {
    v529.m128i_i64[0] = v98 + 8LL * *(unsigned int *)(v1 + 1264);
    v528 = __PAIR128__(v100, v505);
    v529.m128i_i64[1] = v99;
  }
  v101 = _mm_loadu_si128(&v529);
  v102 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v101;
  if ( v529.m128i_i64[0] != v99 )
  {
    do
    {
      sub_B972A0(*v102);
      v527.m128i_i64[0] += 8;
      sub_B768B0((__int64)&v526);
      v102 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v99 );
  }
  v103 = *(_QWORD *)(v1 + 1280);
  v507 = v1 + 1272;
  v104 = v103 + 8LL * *(unsigned int *)(v1 + 1296);
  v105 = *(_QWORD *)(v1 + 1272);
  if ( *(_DWORD *)(v1 + 1288) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1272);
    v529.m128i_i64[0] = v103;
    v529.m128i_i64[1] = v104;
    *(_QWORD *)&v528 = v1 + 1272;
    sub_B768F0((__int64)&v528);
    v104 = *(_QWORD *)(v1 + 1280) + 8LL * *(unsigned int *)(v1 + 1296);
  }
  else
  {
    v529.m128i_i64[0] = v103 + 8LL * *(unsigned int *)(v1 + 1296);
    v528 = __PAIR128__(v105, v507);
    v529.m128i_i64[1] = v104;
  }
  v106 = _mm_loadu_si128(&v529);
  v107 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v106;
  if ( v529.m128i_i64[0] != v104 )
  {
    do
    {
      sub_B972A0(*v107);
      v527.m128i_i64[0] += 8;
      sub_B768F0((__int64)&v526);
      v107 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v104 );
  }
  v108 = *(_QWORD *)(v1 + 1312);
  v508 = v1 + 1304;
  v109 = v108 + 8LL * *(unsigned int *)(v1 + 1328);
  v110 = *(_QWORD *)(v1 + 1304);
  if ( *(_DWORD *)(v1 + 1320) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1304);
    v529.m128i_i64[0] = v108;
    v529.m128i_i64[1] = v109;
    *(_QWORD *)&v528 = v1 + 1304;
    sub_B76930((__int64)&v528);
    v109 = *(_QWORD *)(v1 + 1312) + 8LL * *(unsigned int *)(v1 + 1328);
  }
  else
  {
    v529.m128i_i64[0] = v108 + 8LL * *(unsigned int *)(v1 + 1328);
    v528 = __PAIR128__(v110, v508);
    v529.m128i_i64[1] = v109;
  }
  v111 = _mm_loadu_si128(&v529);
  v112 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v111;
  if ( v529.m128i_i64[0] != v109 )
  {
    do
    {
      sub_B972A0(*v112);
      v527.m128i_i64[0] += 8;
      sub_B76930((__int64)&v526);
      v112 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v109 );
  }
  v113 = *(_QWORD *)(v1 + 1344);
  v509 = v1 + 1336;
  v114 = v113 + 8LL * *(unsigned int *)(v1 + 1360);
  v115 = *(_QWORD *)(v1 + 1336);
  if ( *(_DWORD *)(v1 + 1352) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1336);
    v529.m128i_i64[0] = v113;
    v529.m128i_i64[1] = v114;
    *(_QWORD *)&v528 = v1 + 1336;
    sub_B76970((__int64)&v528);
    v114 = *(_QWORD *)(v1 + 1344) + 8LL * *(unsigned int *)(v1 + 1360);
  }
  else
  {
    v529.m128i_i64[0] = v113 + 8LL * *(unsigned int *)(v1 + 1360);
    v528 = __PAIR128__(v115, v509);
    v529.m128i_i64[1] = v114;
  }
  v116 = _mm_loadu_si128(&v529);
  v117 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v116;
  if ( v529.m128i_i64[0] != v114 )
  {
    do
    {
      sub_B972A0(*v117);
      v527.m128i_i64[0] += 8;
      sub_B76970((__int64)&v526);
      v117 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v114 );
  }
  v118 = *(_QWORD *)(v1 + 1376);
  v510 = v1 + 1368;
  v119 = v118 + 8LL * *(unsigned int *)(v1 + 1392);
  v120 = *(_QWORD *)(v1 + 1368);
  if ( *(_DWORD *)(v1 + 1384) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1368);
    v529.m128i_i64[0] = v118;
    v529.m128i_i64[1] = v119;
    *(_QWORD *)&v528 = v1 + 1368;
    sub_B769B0((__int64)&v528);
    v119 = *(_QWORD *)(v1 + 1376) + 8LL * *(unsigned int *)(v1 + 1392);
  }
  else
  {
    v529.m128i_i64[0] = v118 + 8LL * *(unsigned int *)(v1 + 1392);
    v528 = __PAIR128__(v120, v510);
    v529.m128i_i64[1] = v119;
  }
  v121 = _mm_loadu_si128(&v529);
  v122 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v121;
  if ( v529.m128i_i64[0] != v119 )
  {
    do
    {
      sub_B972A0(*v122);
      v527.m128i_i64[0] += 8;
      sub_B769B0((__int64)&v526);
      v122 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v119 );
  }
  v123 = *(_QWORD *)(v1 + 1408);
  v511 = v1 + 1400;
  v124 = v123 + 8LL * *(unsigned int *)(v1 + 1424);
  v125 = *(_QWORD *)(v1 + 1400);
  if ( *(_DWORD *)(v1 + 1416) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1400);
    v529.m128i_i64[0] = v123;
    v529.m128i_i64[1] = v124;
    *(_QWORD *)&v528 = v1 + 1400;
    sub_B769F0((__int64)&v528);
    v124 = *(_QWORD *)(v1 + 1408) + 8LL * *(unsigned int *)(v1 + 1424);
  }
  else
  {
    v529.m128i_i64[0] = v123 + 8LL * *(unsigned int *)(v1 + 1424);
    v528 = __PAIR128__(v125, v511);
    v529.m128i_i64[1] = v124;
  }
  v126 = _mm_loadu_si128(&v529);
  v127 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v126;
  if ( v529.m128i_i64[0] != v124 )
  {
    do
    {
      sub_B972A0(*v127);
      v527.m128i_i64[0] += 8;
      sub_B769F0((__int64)&v526);
      v127 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v124 );
  }
  v128 = *(_QWORD *)(v1 + 1440);
  v512 = v1 + 1432;
  v129 = v128 + 8LL * *(unsigned int *)(v1 + 1456);
  v130 = *(_QWORD *)(v1 + 1432);
  if ( *(_DWORD *)(v1 + 1448) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1432);
    v529.m128i_i64[0] = v128;
    v529.m128i_i64[1] = v129;
    *(_QWORD *)&v528 = v1 + 1432;
    sub_B76A30((__int64)&v528);
    v129 = *(_QWORD *)(v1 + 1440) + 8LL * *(unsigned int *)(v1 + 1456);
  }
  else
  {
    v529.m128i_i64[0] = v128 + 8LL * *(unsigned int *)(v1 + 1456);
    v528 = __PAIR128__(v130, v512);
    v529.m128i_i64[1] = v129;
  }
  v131 = _mm_loadu_si128(&v529);
  v132 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v131;
  if ( v529.m128i_i64[0] != v129 )
  {
    do
    {
      sub_B972A0(*v132);
      v527.m128i_i64[0] += 8;
      sub_B76A30((__int64)&v526);
      v132 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v129 );
  }
  v133 = *(_QWORD *)(v1 + 1472);
  v513 = v1 + 1464;
  v134 = v133 + 8LL * *(unsigned int *)(v1 + 1488);
  v135 = *(_QWORD *)(v1 + 1464);
  if ( *(_DWORD *)(v1 + 1480) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1464);
    v529.m128i_i64[0] = v133;
    v529.m128i_i64[1] = v134;
    *(_QWORD *)&v528 = v1 + 1464;
    sub_B76A70((__int64)&v528);
    v134 = *(_QWORD *)(v1 + 1472) + 8LL * *(unsigned int *)(v1 + 1488);
  }
  else
  {
    v529.m128i_i64[0] = v133 + 8LL * *(unsigned int *)(v1 + 1488);
    v528 = __PAIR128__(v135, v513);
    v529.m128i_i64[1] = v134;
  }
  v136 = _mm_loadu_si128(&v529);
  v137 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v136;
  if ( v529.m128i_i64[0] != v134 )
  {
    do
    {
      sub_B972A0(*v137);
      v527.m128i_i64[0] += 8;
      sub_B76A70((__int64)&v526);
      v137 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v134 );
  }
  v138 = *(_QWORD *)(v1 + 1504);
  v514 = v1 + 1496;
  v139 = v138 + 8LL * *(unsigned int *)(v1 + 1520);
  v140 = *(_QWORD *)(v1 + 1496);
  if ( *(_DWORD *)(v1 + 1512) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1496);
    v529.m128i_i64[0] = v138;
    v529.m128i_i64[1] = v139;
    *(_QWORD *)&v528 = v1 + 1496;
    sub_B76AB0((__int64)&v528);
    v139 = *(_QWORD *)(v1 + 1504) + 8LL * *(unsigned int *)(v1 + 1520);
  }
  else
  {
    v529.m128i_i64[0] = v138 + 8LL * *(unsigned int *)(v1 + 1520);
    v528 = __PAIR128__(v140, v514);
    v529.m128i_i64[1] = v139;
  }
  v141 = _mm_loadu_si128(&v529);
  v142 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v141;
  if ( v529.m128i_i64[0] != v139 )
  {
    do
    {
      sub_B972A0(*v142);
      v527.m128i_i64[0] += 8;
      sub_B76AB0((__int64)&v526);
      v142 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v139 );
  }
  v143 = *(_QWORD *)(v1 + 1536);
  v515 = v1 + 1528;
  v144 = v143 + 8LL * *(unsigned int *)(v1 + 1552);
  v145 = *(_QWORD *)(v1 + 1528);
  if ( *(_DWORD *)(v1 + 1544) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1528);
    v529.m128i_i64[0] = v143;
    v529.m128i_i64[1] = v144;
    *(_QWORD *)&v528 = v1 + 1528;
    sub_B76AF0((__int64)&v528);
    v144 = *(_QWORD *)(v1 + 1536) + 8LL * *(unsigned int *)(v1 + 1552);
  }
  else
  {
    v529.m128i_i64[0] = v143 + 8LL * *(unsigned int *)(v1 + 1552);
    v528 = __PAIR128__(v145, v515);
    v529.m128i_i64[1] = v144;
  }
  v146 = _mm_loadu_si128(&v529);
  v147 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v146;
  if ( v529.m128i_i64[0] != v144 )
  {
    do
    {
      sub_B972A0(*v147);
      v527.m128i_i64[0] += 8;
      sub_B76AF0((__int64)&v526);
      v147 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v144 );
  }
  v148 = *(_QWORD *)(v1 + 1568);
  v516 = v1 + 1560;
  v149 = v148 + 8LL * *(unsigned int *)(v1 + 1584);
  v150 = *(_QWORD *)(v1 + 1560);
  if ( *(_DWORD *)(v1 + 1576) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1560);
    v529.m128i_i64[0] = v148;
    v529.m128i_i64[1] = v149;
    *(_QWORD *)&v528 = v1 + 1560;
    sub_B76B30((__int64)&v528);
    v149 = *(_QWORD *)(v1 + 1568) + 8LL * *(unsigned int *)(v1 + 1584);
  }
  else
  {
    v529.m128i_i64[0] = v148 + 8LL * *(unsigned int *)(v1 + 1584);
    v528 = __PAIR128__(v150, v516);
    v529.m128i_i64[1] = v149;
  }
  v151 = _mm_loadu_si128(&v529);
  v152 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v151;
  if ( v529.m128i_i64[0] != v149 )
  {
    do
    {
      sub_B972A0(*v152);
      v527.m128i_i64[0] += 8;
      sub_B76B30((__int64)&v526);
      v152 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v149 );
  }
  v153 = *(_QWORD *)(v1 + 1600);
  v518 = v1 + 1592;
  v154 = v153 + 8LL * *(unsigned int *)(v1 + 1616);
  v155 = *(_QWORD *)(v1 + 1592);
  if ( *(_DWORD *)(v1 + 1608) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1592);
    v529.m128i_i64[0] = v153;
    v529.m128i_i64[1] = v154;
    *(_QWORD *)&v528 = v1 + 1592;
    sub_B76B70((__int64)&v528);
    v154 = *(_QWORD *)(v1 + 1600) + 8LL * *(unsigned int *)(v1 + 1616);
  }
  else
  {
    v529.m128i_i64[0] = v153 + 8LL * *(unsigned int *)(v1 + 1616);
    v528 = __PAIR128__(v155, v518);
    v529.m128i_i64[1] = v154;
  }
  v156 = _mm_loadu_si128(&v529);
  v157 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v156;
  if ( v529.m128i_i64[0] != v154 )
  {
    do
    {
      sub_B972A0(*v157);
      v527.m128i_i64[0] += 8;
      sub_B76B70((__int64)&v526);
      v157 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v154 );
  }
  v158 = *(_QWORD *)(v1 + 576);
  if ( *(_DWORD *)(v1 + 584) )
  {
    v529.m128i_i64[1] = v158 + 16LL * *(unsigned int *)(v1 + 592);
    *(_QWORD *)&v528 = v1 + 568;
    v475 = *(_QWORD *)(v1 + 568);
    v529.m128i_i64[0] = v158;
    *((_QWORD *)&v528 + 1) = v475;
    sub_B74F30((__int64)&v528);
    v476 = v529.m128i_i64[0];
    for ( j = *(_QWORD *)(v1 + 576) + 16LL * *(unsigned int *)(v1 + 592); j != v529.m128i_i64[0]; v476 = v529.m128i_i64[0] )
    {
      sub_B92F50(*(_QWORD *)(v476 + 8) + 8LL, 0);
      v529.m128i_i64[0] += 16;
      sub_B74F30((__int64)&v528);
    }
  }
  v159 = *(_QWORD *)(v1 + 608);
  v520 = v1 + 600;
  if ( *(_DWORD *)(v1 + 616) )
  {
    v529.m128i_i64[1] = v159 + 16LL * *(unsigned int *)(v1 + 624);
    v529.m128i_i64[0] = v159;
    *(_QWORD *)&v528 = v1 + 600;
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 600);
    sub_B74F70((__int64)&v528);
    v481 = v529.m128i_i64[0];
    for ( k = *(_QWORD *)(v1 + 608) + 16LL * *(unsigned int *)(v1 + 624); k != v529.m128i_i64[0]; v481 = v529.m128i_i64[0] )
    {
      *(_QWORD *)(*(_QWORD *)(v481 + 8) + 24LL) = 0;
      v529.m128i_i64[0] = v481 + 16;
      sub_B74F70((__int64)&v528);
    }
  }
  v160 = *(__int64 **)(v1 + 640);
  v161 = &v160[*(unsigned int *)(v1 + 656)];
  if ( *(_DWORD *)(v1 + 648) && v160 != v161 )
  {
    while ( *v160 == -4096 || *v160 == -8192 )
    {
      if ( ++v160 == v161 )
        goto LABEL_136;
    }
LABEL_533:
    if ( v160 != v161 )
    {
      v484 = *v160;
      sub_AF5100(*v160, 0);
      if ( v484 )
      {
        sub_AF50C0(v484);
        v485 = *(_QWORD *)(v484 + 136);
        if ( v485 != v484 + 152 )
          _libc_free(v485, 0);
        sub_B73450(v484 + 24);
        j_j___libc_free_0(v484, 184);
      }
      while ( ++v160 != v161 )
      {
        if ( *v160 != -8192 && *v160 != -4096 )
          goto LABEL_533;
      }
    }
  }
LABEL_136:
  sub_B74FB0(v1 + 632);
  v162 = *(__int64 **)(v1 + 1672);
  for ( m = *(__int64 **)(v1 + 1664); v162 != m; ++m )
  {
    v164 = *m;
    sub_B97380(v164);
  }
  v165 = *(__int64 **)(v1 + 1688);
  if ( *(__int64 **)(v1 + 1696) != v165 )
  {
    v486 = v1;
    v166 = *(__int64 **)(v1 + 1696);
    do
    {
      v167 = *v165;
      v168 = 0;
      if ( *(_DWORD *)(*v165 + 16) )
      {
        do
        {
          v169 = 32LL * v168;
          v170 = (__int64 *)(v167 + v169 + 24);
          if ( *(_DWORD *)(v167 + v169 + 48) > 0x40u )
          {
            v171 = *(_QWORD *)(v167 + v169 + 40);
            if ( v171 )
              j_j___libc_free_0_0(v171);
          }
          ++v168;
          sub_969240(v170);
        }
        while ( *(_DWORD *)(v167 + 16) != v168 );
      }
      ++v165;
    }
    while ( v166 != v165 );
    v1 = v486;
  }
  v172 = *(_QWORD *)(v1 + 672);
  v173 = v172 + 8LL * *(unsigned int *)(v1 + 688);
  v174 = *(_QWORD *)(v1 + 664);
  if ( *(_DWORD *)(v1 + 680) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 664);
    v529.m128i_i64[0] = v172;
    v529.m128i_i64[1] = v173;
    *(_QWORD *)&v528 = v487;
    sub_B76430((__int64)&v528);
    v173 = *(_QWORD *)(v1 + 672) + 8LL * *(unsigned int *)(v1 + 688);
  }
  else
  {
    v529.m128i_i64[0] = v172 + 8LL * *(unsigned int *)(v1 + 688);
    v528 = __PAIR128__(v174, v487);
    v529.m128i_i64[1] = v173;
  }
  v175 = _mm_loadu_si128(&v529);
  v176 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v175;
  if ( v529.m128i_i64[0] != v173 )
  {
    do
    {
      v177 = *v176;
      if ( *v176 )
      {
        sub_B972A0(*v176);
        sub_B706B0((__int64 *)(v177 + 8));
        sub_B914E0(v177);
      }
      v527.m128i_i64[0] += 8;
      sub_B76430((__int64)&v526);
      v176 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v173 );
  }
  v178 = *(_QWORD *)(v1 + 704);
  v179 = v178 + 8LL * *(unsigned int *)(v1 + 720);
  v180 = *(_QWORD *)(v1 + 696);
  if ( *(_DWORD *)(v1 + 712) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 696);
    v529.m128i_i64[0] = v178;
    v529.m128i_i64[1] = v179;
    *(_QWORD *)&v528 = v488;
    sub_B76470((__int64)&v528);
    v179 = *(_QWORD *)(v1 + 704) + 8LL * *(unsigned int *)(v1 + 720);
  }
  else
  {
    v529.m128i_i64[0] = v178 + 8LL * *(unsigned int *)(v1 + 720);
    v528 = __PAIR128__(v180, v488);
    v529.m128i_i64[1] = v179;
  }
  v181 = _mm_loadu_si128(&v529);
  v182 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v181;
  if ( v529.m128i_i64[0] != v179 )
  {
    do
    {
      v183 = *v182;
      if ( *v182 )
      {
        sub_B972A0(*v182);
        sub_B706B0((__int64 *)(v183 + 8));
        sub_B914E0(v183);
      }
      v527.m128i_i64[0] += 8;
      sub_B76470((__int64)&v526);
      v182 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v179 );
  }
  v184 = *(_QWORD *)(v1 + 736);
  v185 = v184 + 8LL * *(unsigned int *)(v1 + 752);
  v186 = *(_QWORD *)(v1 + 728);
  if ( *(_DWORD *)(v1 + 744) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 728);
    v529.m128i_i64[0] = v184;
    v529.m128i_i64[1] = v185;
    *(_QWORD *)&v528 = v489;
    sub_B764B0((__int64)&v528);
    v185 = *(_QWORD *)(v1 + 736) + 8LL * *(unsigned int *)(v1 + 752);
  }
  else
  {
    v529.m128i_i64[0] = v184 + 8LL * *(unsigned int *)(v1 + 752);
    v528 = __PAIR128__(v186, v489);
    v529.m128i_i64[1] = v185;
  }
  v187 = _mm_loadu_si128(&v529);
  v188 = (__int64 **)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v187;
  if ( v529.m128i_i64[0] != v185 )
  {
    do
    {
      v189 = *v188;
      if ( *v188 )
      {
        v190 = v189[2];
        if ( v190 )
          j_j___libc_free_0(v190, v189[4] - v190);
        sub_B706B0(v189 + 1);
        sub_B914E0(v189);
      }
      v527.m128i_i64[0] += 8;
      sub_B764B0((__int64)&v526);
      v188 = (__int64 **)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v185 );
  }
  v191 = *(_QWORD *)(v1 + 768);
  v192 = v191 + 8LL * *(unsigned int *)(v1 + 784);
  v193 = *(_QWORD *)(v1 + 760);
  if ( *(_DWORD *)(v1 + 776) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 760);
    v529.m128i_i64[0] = v191;
    v529.m128i_i64[1] = v192;
    *(_QWORD *)&v528 = v490;
    sub_B764F0((__int64)&v528);
    v192 = *(_QWORD *)(v1 + 768) + 8LL * *(unsigned int *)(v1 + 784);
  }
  else
  {
    v529.m128i_i64[0] = v191 + 8LL * *(unsigned int *)(v1 + 784);
    v528 = __PAIR128__(v193, v490);
    v529.m128i_i64[1] = v192;
  }
  v194 = _mm_loadu_si128(&v529);
  v195 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v194;
  if ( v529.m128i_i64[0] != v192 )
  {
    do
    {
      v196 = *v195;
      if ( *v195 )
      {
        sub_B706B0((__int64 *)(v196 + 8));
        sub_B914E0(v196);
      }
      v527.m128i_i64[0] += 8;
      sub_B764F0((__int64)&v526);
      v195 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v192 );
  }
  v197 = *(_QWORD *)(v1 + 800);
  v198 = v197 + 8LL * *(unsigned int *)(v1 + 816);
  v199 = *(_QWORD *)(v1 + 792);
  if ( *(_DWORD *)(v1 + 808) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 792);
    v529.m128i_i64[0] = v197;
    v529.m128i_i64[1] = v198;
    *(_QWORD *)&v528 = v491;
    sub_B76530((__int64)&v528);
    v198 = *(_QWORD *)(v1 + 800) + 8LL * *(unsigned int *)(v1 + 816);
  }
  else
  {
    v529.m128i_i64[0] = v197 + 8LL * *(unsigned int *)(v1 + 816);
    v528 = __PAIR128__(v199, v491);
    v529.m128i_i64[1] = v198;
  }
  v200 = _mm_loadu_si128(&v529);
  v201 = (_QWORD *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v200;
  if ( v529.m128i_i64[0] != v198 )
  {
    do
    {
      v202 = *v201;
      if ( *v201 )
      {
        sub_B972A0(*v201);
        sub_B706B0((__int64 *)(v202 + 8));
        sub_B914E0(v202);
      }
      v527.m128i_i64[0] += 8;
      sub_B76530((__int64)&v526);
      v201 = (_QWORD *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v198 );
  }
  v203 = *(_QWORD *)(v1 + 832);
  v204 = v203 + 8LL * *(unsigned int *)(v1 + 848);
  v205 = *(_QWORD *)(v1 + 824);
  if ( *(_DWORD *)(v1 + 840) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 824);
    v529.m128i_i64[0] = v203;
    v529.m128i_i64[1] = v204;
    *(_QWORD *)&v528 = v492;
    sub_B76570((__int64)&v528);
    v204 = *(_QWORD *)(v1 + 832) + 8LL * *(unsigned int *)(v1 + 848);
  }
  else
  {
    v529.m128i_i64[0] = v203 + 8LL * *(unsigned int *)(v1 + 848);
    v528 = __PAIR128__(v205, v492);
    v529.m128i_i64[1] = v204;
  }
  v206 = _mm_loadu_si128(&v529);
  v207 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v206;
  if ( v529.m128i_i64[0] != v204 )
  {
    do
    {
      v208 = *v207;
      if ( *v207 )
      {
        sub_B706B0((__int64 *)(v208 + 8));
        sub_B914E0(v208);
      }
      v527.m128i_i64[0] += 8;
      sub_B76570((__int64)&v526);
      v207 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v204 );
  }
  v209 = *(_QWORD *)(v1 + 864);
  v210 = v209 + 8LL * *(unsigned int *)(v1 + 880);
  v211 = *(_QWORD *)(v1 + 856);
  if ( *(_DWORD *)(v1 + 872) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 856);
    v529.m128i_i64[0] = v209;
    v529.m128i_i64[1] = v210;
    *(_QWORD *)&v528 = v493;
    sub_B765B0((__int64)&v528);
    v210 = *(_QWORD *)(v1 + 864) + 8LL * *(unsigned int *)(v1 + 880);
  }
  else
  {
    v529.m128i_i64[0] = v209 + 8LL * *(unsigned int *)(v1 + 880);
    v528 = __PAIR128__(v211, v493);
    v529.m128i_i64[1] = v210;
  }
  v212 = _mm_loadu_si128(&v529);
  v213 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v212;
  if ( v529.m128i_i64[0] != v210 )
  {
    do
    {
      v214 = *v213;
      if ( *v213 )
      {
        sub_969240((__int64 *)(v214 + 16));
        sub_B706B0((__int64 *)(v214 + 8));
        sub_B914E0(v214);
      }
      v527.m128i_i64[0] += 8;
      sub_B765B0((__int64)&v526);
      v213 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v210 );
  }
  v215 = *(_QWORD *)(v1 + 896);
  v216 = v215 + 8LL * *(unsigned int *)(v1 + 912);
  v217 = *(_QWORD *)(v1 + 888);
  if ( *(_DWORD *)(v1 + 904) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 888);
    v529.m128i_i64[0] = v215;
    v529.m128i_i64[1] = v216;
    *(_QWORD *)&v528 = v494;
    sub_B765F0((__int64)&v528);
    v216 = *(_QWORD *)(v1 + 896) + 8LL * *(unsigned int *)(v1 + 912);
  }
  else
  {
    v529.m128i_i64[0] = v215 + 8LL * *(unsigned int *)(v1 + 912);
    v528 = __PAIR128__(v217, v494);
    v529.m128i_i64[1] = v216;
  }
  v218 = _mm_loadu_si128(&v529);
  v219 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v218;
  if ( v529.m128i_i64[0] != v216 )
  {
    do
    {
      v220 = *v219;
      if ( *v219 )
      {
        sub_B706B0((__int64 *)(v220 + 8));
        sub_B914E0(v220);
      }
      v527.m128i_i64[0] += 8;
      sub_B765F0((__int64)&v526);
      v219 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v216 );
  }
  v221 = *(_QWORD *)(v1 + 928);
  v222 = v221 + 8LL * *(unsigned int *)(v1 + 944);
  v223 = *(_QWORD *)(v1 + 920);
  if ( *(_DWORD *)(v1 + 936) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 920);
    v529.m128i_i64[0] = v221;
    v529.m128i_i64[1] = v222;
    *(_QWORD *)&v528 = v495;
    sub_B76630((__int64)&v528);
    v222 = *(_QWORD *)(v1 + 928) + 8LL * *(unsigned int *)(v1 + 944);
  }
  else
  {
    v529.m128i_i64[0] = v221 + 8LL * *(unsigned int *)(v1 + 944);
    v528 = __PAIR128__(v223, v495);
    v529.m128i_i64[1] = v222;
  }
  v224 = _mm_loadu_si128(&v529);
  v225 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v224;
  if ( v529.m128i_i64[0] != v222 )
  {
    do
    {
      v226 = *v225;
      if ( *v225 )
      {
        sub_B706B0((__int64 *)(v226 + 8));
        sub_B914E0(v226);
      }
      v527.m128i_i64[0] += 8;
      sub_B76630((__int64)&v526);
      v225 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v222 );
  }
  v227 = *(_QWORD *)(v1 + 960);
  v228 = v227 + 8LL * *(unsigned int *)(v1 + 976);
  v229 = *(_QWORD *)(v1 + 952);
  if ( *(_DWORD *)(v1 + 968) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 952);
    v529.m128i_i64[0] = v227;
    v529.m128i_i64[1] = v228;
    *(_QWORD *)&v528 = v496;
    sub_B76670((__int64)&v528);
    v228 = *(_QWORD *)(v1 + 960) + 8LL * *(unsigned int *)(v1 + 976);
  }
  else
  {
    v529.m128i_i64[0] = v227 + 8LL * *(unsigned int *)(v1 + 976);
    v528 = __PAIR128__(v229, v496);
    v529.m128i_i64[1] = v228;
  }
  v230 = _mm_loadu_si128(&v529);
  v231 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v230;
  if ( v529.m128i_i64[0] != v228 )
  {
    do
    {
      v232 = *v231;
      if ( *v231 )
      {
        sub_B706B0((__int64 *)(v232 + 8));
        sub_B914E0(v232);
      }
      v527.m128i_i64[0] += 8;
      sub_B76670((__int64)&v526);
      v231 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v228 );
  }
  v233 = *(_QWORD *)(v1 + 992);
  v234 = v233 + 8LL * *(unsigned int *)(v1 + 1008);
  v235 = *(_QWORD *)(v1 + 984);
  if ( *(_DWORD *)(v1 + 1000) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 984);
    v529.m128i_i64[0] = v233;
    v529.m128i_i64[1] = v234;
    *(_QWORD *)&v528 = v497;
    sub_B766B0((__int64)&v528);
    v234 = *(_QWORD *)(v1 + 992) + 8LL * *(unsigned int *)(v1 + 1008);
  }
  else
  {
    v529.m128i_i64[0] = v233 + 8LL * *(unsigned int *)(v1 + 1008);
    v528 = __PAIR128__(v235, v497);
    v529.m128i_i64[1] = v234;
  }
  v236 = _mm_loadu_si128(&v529);
  v237 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v236;
  if ( v529.m128i_i64[0] != v234 )
  {
    do
    {
      v238 = *v237;
      if ( *v237 )
      {
        sub_B706B0((__int64 *)(v238 + 8));
        sub_B914E0(v238);
      }
      v527.m128i_i64[0] += 8;
      sub_B766B0((__int64)&v526);
      v237 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v234 );
  }
  v239 = *(_QWORD *)(v1 + 1024);
  v240 = v239 + 8LL * *(unsigned int *)(v1 + 1040);
  v241 = *(_QWORD *)(v1 + 1016);
  if ( *(_DWORD *)(v1 + 1032) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1016);
    v529.m128i_i64[0] = v239;
    v529.m128i_i64[1] = v240;
    *(_QWORD *)&v528 = v498;
    sub_B766F0((__int64)&v528);
    v240 = *(_QWORD *)(v1 + 1024) + 8LL * *(unsigned int *)(v1 + 1040);
  }
  else
  {
    v529.m128i_i64[0] = v239 + 8LL * *(unsigned int *)(v1 + 1040);
    v528 = __PAIR128__(v241, v498);
    v529.m128i_i64[1] = v240;
  }
  v242 = _mm_loadu_si128(&v529);
  v243 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v242;
  if ( v529.m128i_i64[0] != v240 )
  {
    do
    {
      v244 = *v243;
      if ( *v243 )
      {
        sub_B706B0((__int64 *)(v244 + 8));
        sub_B914E0(v244);
      }
      v527.m128i_i64[0] += 8;
      sub_B766F0((__int64)&v526);
      v243 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v240 );
  }
  v245 = *(_QWORD *)(v1 + 1056);
  v246 = v245 + 8LL * *(unsigned int *)(v1 + 1072);
  v247 = *(_QWORD *)(v1 + 1048);
  if ( *(_DWORD *)(v1 + 1064) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1048);
    v529.m128i_i64[0] = v245;
    v529.m128i_i64[1] = v246;
    *(_QWORD *)&v528 = v499;
    sub_B76730((__int64)&v528);
    v246 = *(_QWORD *)(v1 + 1056) + 8LL * *(unsigned int *)(v1 + 1072);
  }
  else
  {
    v529.m128i_i64[0] = v245 + 8LL * *(unsigned int *)(v1 + 1072);
    v528 = __PAIR128__(v247, v499);
    v529.m128i_i64[1] = v246;
  }
  v248 = _mm_loadu_si128(&v529);
  v249 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v248;
  if ( v529.m128i_i64[0] != v246 )
  {
    do
    {
      v250 = *v249;
      if ( *v249 )
      {
        sub_B706B0((__int64 *)(v250 + 8));
        sub_B914E0(v250);
      }
      v527.m128i_i64[0] += 8;
      sub_B76730((__int64)&v526);
      v249 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v246 );
  }
  v251 = *(_QWORD *)(v1 + 1088);
  v252 = v251 + 8LL * *(unsigned int *)(v1 + 1104);
  v253 = *(_QWORD *)(v1 + 1080);
  if ( *(_DWORD *)(v1 + 1096) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1080);
    v529.m128i_i64[0] = v251;
    v529.m128i_i64[1] = v252;
    *(_QWORD *)&v528 = v500;
    sub_B76770((__int64)&v528);
    v252 = *(_QWORD *)(v1 + 1088) + 8LL * *(unsigned int *)(v1 + 1104);
  }
  else
  {
    v529.m128i_i64[0] = v251 + 8LL * *(unsigned int *)(v1 + 1104);
    v528 = __PAIR128__(v253, v500);
    v529.m128i_i64[1] = v252;
  }
  v254 = _mm_loadu_si128(&v529);
  v255 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v254;
  if ( v529.m128i_i64[0] != v252 )
  {
    do
    {
      v256 = *v255;
      if ( *v255 )
      {
        sub_B706B0((__int64 *)(v256 + 8));
        sub_B914E0(v256);
      }
      v527.m128i_i64[0] += 8;
      sub_B76770((__int64)&v526);
      v255 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v252 );
  }
  v257 = *(_QWORD *)(v1 + 1120);
  v258 = v257 + 8LL * *(unsigned int *)(v1 + 1136);
  v259 = *(_QWORD *)(v1 + 1112);
  if ( *(_DWORD *)(v1 + 1128) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1112);
    v529.m128i_i64[0] = v257;
    v529.m128i_i64[1] = v258;
    *(_QWORD *)&v528 = v501;
    sub_B767B0((__int64)&v528);
    v258 = *(_QWORD *)(v1 + 1120) + 8LL * *(unsigned int *)(v1 + 1136);
  }
  else
  {
    v529.m128i_i64[0] = v257 + 8LL * *(unsigned int *)(v1 + 1136);
    v528 = __PAIR128__(v259, v501);
    v529.m128i_i64[1] = v258;
  }
  v260 = _mm_loadu_si128(&v529);
  v261 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v260;
  if ( v529.m128i_i64[0] != v258 )
  {
    do
    {
      v262 = *v261;
      if ( *v261 )
      {
        sub_B706B0((__int64 *)(v262 + 8));
        sub_B914E0(v262);
      }
      v527.m128i_i64[0] += 8;
      sub_B767B0((__int64)&v526);
      v261 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v258 );
  }
  v263 = *(_QWORD *)(v1 + 1152);
  v264 = v263 + 8LL * *(unsigned int *)(v1 + 1168);
  v265 = *(_QWORD *)(v1 + 1144);
  if ( *(_DWORD *)(v1 + 1160) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1144);
    v529.m128i_i64[0] = v263;
    v529.m128i_i64[1] = v264;
    *(_QWORD *)&v528 = v502;
    sub_B767F0((__int64)&v528);
    v264 = *(_QWORD *)(v1 + 1152) + 8LL * *(unsigned int *)(v1 + 1168);
  }
  else
  {
    v529.m128i_i64[0] = v263 + 8LL * *(unsigned int *)(v1 + 1168);
    v528 = __PAIR128__(v265, v502);
    v529.m128i_i64[1] = v264;
  }
  v266 = _mm_loadu_si128(&v529);
  v267 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v266;
  if ( v529.m128i_i64[0] != v264 )
  {
    do
    {
      v268 = *v267;
      if ( *v267 )
      {
        sub_B706B0((__int64 *)(v268 + 8));
        sub_B914E0(v268);
      }
      v527.m128i_i64[0] += 8;
      sub_B767F0((__int64)&v526);
      v267 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v264 );
  }
  v269 = *(_QWORD *)(v1 + 1184);
  v270 = v269 + 8LL * *(unsigned int *)(v1 + 1200);
  v271 = *(_QWORD *)(v1 + 1176);
  if ( *(_DWORD *)(v1 + 1192) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1176);
    v529.m128i_i64[0] = v269;
    v529.m128i_i64[1] = v270;
    *(_QWORD *)&v528 = v503;
    sub_B76830((__int64)&v528);
    v270 = *(_QWORD *)(v1 + 1184) + 8LL * *(unsigned int *)(v1 + 1200);
  }
  else
  {
    v529.m128i_i64[0] = v269 + 8LL * *(unsigned int *)(v1 + 1200);
    v528 = __PAIR128__(v271, v503);
    v529.m128i_i64[1] = v270;
  }
  v272 = _mm_loadu_si128(&v529);
  v273 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v272;
  if ( v529.m128i_i64[0] != v270 )
  {
    do
    {
      v274 = *v273;
      if ( *v273 )
      {
        sub_B706B0((__int64 *)(v274 + 8));
        sub_B914E0(v274);
      }
      v527.m128i_i64[0] += 8;
      sub_B76830((__int64)&v526);
      v273 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v270 );
  }
  v275 = *(_QWORD *)(v1 + 1216);
  v276 = v275 + 8LL * *(unsigned int *)(v1 + 1232);
  v277 = *(_QWORD *)(v1 + 1208);
  if ( *(_DWORD *)(v1 + 1224) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1208);
    v529.m128i_i64[0] = v275;
    v529.m128i_i64[1] = v276;
    *(_QWORD *)&v528 = v504;
    sub_B76870((__int64)&v528);
    v276 = *(_QWORD *)(v1 + 1216) + 8LL * *(unsigned int *)(v1 + 1232);
  }
  else
  {
    v529.m128i_i64[0] = v275 + 8LL * *(unsigned int *)(v1 + 1232);
    v528 = __PAIR128__(v277, v504);
    v529.m128i_i64[1] = v276;
  }
  v278 = _mm_loadu_si128(&v529);
  v279 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v278;
  if ( v529.m128i_i64[0] != v276 )
  {
    do
    {
      v280 = *v279;
      if ( *v279 )
      {
        sub_B706B0((__int64 *)(v280 + 8));
        sub_B914E0(v280);
      }
      v527.m128i_i64[0] += 8;
      sub_B76870((__int64)&v526);
      v279 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v276 );
  }
  v281 = *(_QWORD *)(v1 + 1248);
  v282 = v281 + 8LL * *(unsigned int *)(v1 + 1264);
  v283 = *(_QWORD *)(v1 + 1240);
  if ( *(_DWORD *)(v1 + 1256) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1240);
    v529.m128i_i64[0] = v281;
    v529.m128i_i64[1] = v282;
    *(_QWORD *)&v528 = v505;
    sub_B768B0((__int64)&v528);
    v282 = *(_QWORD *)(v1 + 1248) + 8LL * *(unsigned int *)(v1 + 1264);
  }
  else
  {
    v529.m128i_i64[0] = v281 + 8LL * *(unsigned int *)(v1 + 1264);
    v528 = __PAIR128__(v283, v505);
    v529.m128i_i64[1] = v282;
  }
  v284 = _mm_loadu_si128(&v529);
  v285 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v284;
  if ( v529.m128i_i64[0] != v282 )
  {
    do
    {
      v286 = *v285;
      if ( *v285 )
      {
        sub_B706B0((__int64 *)(v286 + 8));
        sub_B914E0(v286);
      }
      v527.m128i_i64[0] += 8;
      sub_B768B0((__int64)&v526);
      v285 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v282 );
  }
  v287 = *(_QWORD *)(v1 + 1280);
  v288 = v287 + 8LL * *(unsigned int *)(v1 + 1296);
  v289 = *(_QWORD *)(v1 + 1272);
  if ( *(_DWORD *)(v1 + 1288) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1272);
    v529.m128i_i64[0] = v287;
    v529.m128i_i64[1] = v288;
    *(_QWORD *)&v528 = v507;
    sub_B768F0((__int64)&v528);
    v288 = *(_QWORD *)(v1 + 1280) + 8LL * *(unsigned int *)(v1 + 1296);
  }
  else
  {
    v529.m128i_i64[0] = v287 + 8LL * *(unsigned int *)(v1 + 1296);
    v528 = __PAIR128__(v289, v507);
    v529.m128i_i64[1] = v288;
  }
  v290 = _mm_loadu_si128(&v529);
  v291 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v290;
  if ( v529.m128i_i64[0] != v288 )
  {
    do
    {
      v292 = *v291;
      if ( *v291 )
      {
        sub_B706B0((__int64 *)(v292 + 8));
        sub_B914E0(v292);
      }
      v527.m128i_i64[0] += 8;
      sub_B768F0((__int64)&v526);
      v291 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v288 );
  }
  v293 = *(_QWORD *)(v1 + 1312);
  v294 = v293 + 8LL * *(unsigned int *)(v1 + 1328);
  v295 = *(_QWORD *)(v1 + 1304);
  if ( *(_DWORD *)(v1 + 1320) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1304);
    v529.m128i_i64[0] = v293;
    v529.m128i_i64[1] = v294;
    *(_QWORD *)&v528 = v508;
    sub_B76930((__int64)&v528);
    v294 = *(_QWORD *)(v1 + 1312) + 8LL * *(unsigned int *)(v1 + 1328);
  }
  else
  {
    v529.m128i_i64[0] = v293 + 8LL * *(unsigned int *)(v1 + 1328);
    v528 = __PAIR128__(v295, v508);
    v529.m128i_i64[1] = v294;
  }
  v296 = _mm_loadu_si128(&v529);
  v297 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v296;
  if ( v529.m128i_i64[0] != v294 )
  {
    do
    {
      v298 = *v297;
      if ( *v297 )
      {
        sub_B706B0((__int64 *)(v298 + 8));
        sub_B914E0(v298);
      }
      v527.m128i_i64[0] += 8;
      sub_B76930((__int64)&v526);
      v297 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v294 );
  }
  v299 = *(_QWORD *)(v1 + 1344);
  v300 = v299 + 8LL * *(unsigned int *)(v1 + 1360);
  v301 = *(_QWORD *)(v1 + 1336);
  if ( *(_DWORD *)(v1 + 1352) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1336);
    v529.m128i_i64[0] = v299;
    v529.m128i_i64[1] = v300;
    *(_QWORD *)&v528 = v509;
    sub_B76970((__int64)&v528);
    v300 = *(_QWORD *)(v1 + 1344) + 8LL * *(unsigned int *)(v1 + 1360);
  }
  else
  {
    v529.m128i_i64[0] = v299 + 8LL * *(unsigned int *)(v1 + 1360);
    v528 = __PAIR128__(v301, v509);
    v529.m128i_i64[1] = v300;
  }
  v302 = _mm_loadu_si128(&v529);
  v303 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v302;
  if ( v529.m128i_i64[0] != v300 )
  {
    do
    {
      v304 = *v303;
      if ( *v303 )
      {
        sub_B706B0((__int64 *)(v304 + 8));
        sub_B914E0(v304);
      }
      v527.m128i_i64[0] += 8;
      sub_B76970((__int64)&v526);
      v303 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v300 );
  }
  v305 = *(_QWORD *)(v1 + 1376);
  v306 = v305 + 8LL * *(unsigned int *)(v1 + 1392);
  v307 = *(_QWORD *)(v1 + 1368);
  if ( *(_DWORD *)(v1 + 1384) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1368);
    v529.m128i_i64[0] = v305;
    v529.m128i_i64[1] = v306;
    *(_QWORD *)&v528 = v510;
    sub_B769B0((__int64)&v528);
    v306 = *(_QWORD *)(v1 + 1376) + 8LL * *(unsigned int *)(v1 + 1392);
  }
  else
  {
    v529.m128i_i64[0] = v305 + 8LL * *(unsigned int *)(v1 + 1392);
    v528 = __PAIR128__(v307, v510);
    v529.m128i_i64[1] = v306;
  }
  v308 = _mm_loadu_si128(&v529);
  v309 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v308;
  if ( v529.m128i_i64[0] != v306 )
  {
    do
    {
      v310 = *v309;
      if ( *v309 )
      {
        sub_B706B0((__int64 *)(v310 + 8));
        sub_B914E0(v310);
      }
      v527.m128i_i64[0] += 8;
      sub_B769B0((__int64)&v526);
      v309 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v306 );
  }
  v311 = *(_QWORD *)(v1 + 1408);
  v312 = v311 + 8LL * *(unsigned int *)(v1 + 1424);
  v313 = *(_QWORD *)(v1 + 1400);
  if ( *(_DWORD *)(v1 + 1416) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1400);
    v529.m128i_i64[0] = v311;
    v529.m128i_i64[1] = v312;
    *(_QWORD *)&v528 = v511;
    sub_B769F0((__int64)&v528);
    v312 = *(_QWORD *)(v1 + 1408) + 8LL * *(unsigned int *)(v1 + 1424);
  }
  else
  {
    v529.m128i_i64[0] = v311 + 8LL * *(unsigned int *)(v1 + 1424);
    v528 = __PAIR128__(v313, v511);
    v529.m128i_i64[1] = v312;
  }
  v314 = _mm_loadu_si128(&v529);
  v315 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v314;
  if ( v529.m128i_i64[0] != v312 )
  {
    do
    {
      v316 = *v315;
      if ( *v315 )
      {
        sub_B706B0((__int64 *)(v316 + 8));
        sub_B914E0(v316);
      }
      v527.m128i_i64[0] += 8;
      sub_B769F0((__int64)&v526);
      v315 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v312 );
  }
  v317 = *(_QWORD *)(v1 + 1440);
  v318 = v317 + 8LL * *(unsigned int *)(v1 + 1456);
  v319 = *(_QWORD *)(v1 + 1432);
  if ( *(_DWORD *)(v1 + 1448) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1432);
    v529.m128i_i64[0] = v317;
    v529.m128i_i64[1] = v318;
    *(_QWORD *)&v528 = v512;
    sub_B76A30((__int64)&v528);
    v318 = *(_QWORD *)(v1 + 1440) + 8LL * *(unsigned int *)(v1 + 1456);
  }
  else
  {
    v529.m128i_i64[0] = v317 + 8LL * *(unsigned int *)(v1 + 1456);
    v528 = __PAIR128__(v319, v512);
    v529.m128i_i64[1] = v318;
  }
  v320 = _mm_loadu_si128(&v529);
  v321 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v320;
  if ( v529.m128i_i64[0] != v318 )
  {
    do
    {
      v322 = *v321;
      if ( *v321 )
      {
        sub_B706B0((__int64 *)(v322 + 8));
        sub_B914E0(v322);
      }
      v527.m128i_i64[0] += 8;
      sub_B76A30((__int64)&v526);
      v321 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v318 );
  }
  v323 = *(_QWORD *)(v1 + 1472);
  v324 = v323 + 8LL * *(unsigned int *)(v1 + 1488);
  v325 = *(_QWORD *)(v1 + 1464);
  if ( *(_DWORD *)(v1 + 1480) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1464);
    v529.m128i_i64[0] = v323;
    v529.m128i_i64[1] = v324;
    *(_QWORD *)&v528 = v513;
    sub_B76A70((__int64)&v528);
    v324 = *(_QWORD *)(v1 + 1472) + 8LL * *(unsigned int *)(v1 + 1488);
  }
  else
  {
    v529.m128i_i64[0] = v323 + 8LL * *(unsigned int *)(v1 + 1488);
    v528 = __PAIR128__(v325, v513);
    v529.m128i_i64[1] = v324;
  }
  v326 = _mm_loadu_si128(&v529);
  v327 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v326;
  if ( v529.m128i_i64[0] != v324 )
  {
    do
    {
      v328 = *v327;
      if ( *v327 )
      {
        sub_B706B0((__int64 *)(v328 + 8));
        sub_B914E0(v328);
      }
      v527.m128i_i64[0] += 8;
      sub_B76A70((__int64)&v526);
      v327 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v324 );
  }
  v329 = *(_QWORD *)(v1 + 1504);
  v330 = v329 + 8LL * *(unsigned int *)(v1 + 1520);
  v331 = *(_QWORD *)(v1 + 1496);
  if ( *(_DWORD *)(v1 + 1512) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1496);
    v529.m128i_i64[0] = v329;
    v529.m128i_i64[1] = v330;
    *(_QWORD *)&v528 = v514;
    sub_B76AB0((__int64)&v528);
    v330 = *(_QWORD *)(v1 + 1504) + 8LL * *(unsigned int *)(v1 + 1520);
  }
  else
  {
    v529.m128i_i64[0] = v329 + 8LL * *(unsigned int *)(v1 + 1520);
    v528 = __PAIR128__(v331, v514);
    v529.m128i_i64[1] = v330;
  }
  v332 = _mm_loadu_si128(&v529);
  v333 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v332;
  if ( v529.m128i_i64[0] != v330 )
  {
    do
    {
      v334 = *v333;
      if ( *v333 )
      {
        sub_B706B0((__int64 *)(v334 + 8));
        sub_B914E0(v334);
      }
      v527.m128i_i64[0] += 8;
      sub_B76AB0((__int64)&v526);
      v333 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v330 );
  }
  v335 = *(_QWORD *)(v1 + 1536);
  v336 = v335 + 8LL * *(unsigned int *)(v1 + 1552);
  v337 = *(_QWORD *)(v1 + 1528);
  if ( *(_DWORD *)(v1 + 1544) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1528);
    v529.m128i_i64[0] = v335;
    v529.m128i_i64[1] = v336;
    *(_QWORD *)&v528 = v515;
    sub_B76AF0((__int64)&v528);
    v336 = *(_QWORD *)(v1 + 1536) + 8LL * *(unsigned int *)(v1 + 1552);
  }
  else
  {
    v529.m128i_i64[0] = v335 + 8LL * *(unsigned int *)(v1 + 1552);
    v528 = __PAIR128__(v337, v515);
    v529.m128i_i64[1] = v336;
  }
  v338 = _mm_loadu_si128(&v529);
  v339 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v338;
  if ( v529.m128i_i64[0] != v336 )
  {
    do
    {
      v340 = *v339;
      if ( *v339 )
      {
        sub_B706B0((__int64 *)(v340 + 8));
        sub_B914E0(v340);
      }
      v527.m128i_i64[0] += 8;
      sub_B76AF0((__int64)&v526);
      v339 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v336 );
  }
  v341 = *(_QWORD *)(v1 + 1568);
  v342 = v341 + 8LL * *(unsigned int *)(v1 + 1584);
  v343 = *(_QWORD *)(v1 + 1560);
  if ( *(_DWORD *)(v1 + 1576) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1560);
    v529.m128i_i64[0] = v341;
    v529.m128i_i64[1] = v342;
    *(_QWORD *)&v528 = v516;
    sub_B76B30((__int64)&v528);
    v342 = *(_QWORD *)(v1 + 1568) + 8LL * *(unsigned int *)(v1 + 1584);
  }
  else
  {
    v529.m128i_i64[0] = v341 + 8LL * *(unsigned int *)(v1 + 1584);
    v528 = __PAIR128__(v343, v516);
    v529.m128i_i64[1] = v342;
  }
  v344 = _mm_loadu_si128(&v529);
  v345 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v344;
  if ( v529.m128i_i64[0] != v342 )
  {
    do
    {
      v346 = *v345;
      if ( *v345 )
      {
        sub_B706B0((__int64 *)(v346 + 8));
        sub_B914E0(v346);
      }
      v527.m128i_i64[0] += 8;
      sub_B76B30((__int64)&v526);
      v345 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v342 );
  }
  v347 = *(_QWORD *)(v1 + 1600);
  v348 = v347 + 8LL * *(unsigned int *)(v1 + 1616);
  v349 = *(_QWORD *)(v1 + 1592);
  if ( *(_DWORD *)(v1 + 1608) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1592);
    v529.m128i_i64[0] = v347;
    v529.m128i_i64[1] = v348;
    *(_QWORD *)&v528 = v518;
    sub_B76B70((__int64)&v528);
    v348 = *(_QWORD *)(v1 + 1600) + 8LL * *(unsigned int *)(v1 + 1616);
  }
  else
  {
    v529.m128i_i64[0] = v347 + 8LL * *(unsigned int *)(v1 + 1616);
    v528 = __PAIR128__(v349, v518);
    v529.m128i_i64[1] = v348;
  }
  v350 = _mm_loadu_si128(&v529);
  v351 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v350;
  if ( v529.m128i_i64[0] != v348 )
  {
    do
    {
      v352 = *v351;
      if ( *v351 )
      {
        sub_B706B0((__int64 *)(v352 + 8));
        sub_B914E0(v352);
      }
      v527.m128i_i64[0] += 8;
      sub_B76B70((__int64)&v526);
      v351 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v348 );
  }
  v353 = *(_QWORD *)(v1 + 2128);
  v354 = v353 + 8LL * *(unsigned int *)(v1 + 2144);
  v355 = *(_QWORD *)(v1 + 2120);
  if ( *(_DWORD *)(v1 + 2136) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 2120);
    v529.m128i_i64[0] = v353;
    v529.m128i_i64[1] = v354;
    *(_QWORD *)&v528 = v1 + 2120;
    sub_B76BB0((__int64)&v528);
    v354 = *(_QWORD *)(v1 + 2128) + 8LL * *(unsigned int *)(v1 + 2144);
  }
  else
  {
    *(_QWORD *)&v528 = v1 + 2120;
    *((_QWORD *)&v528 + 1) = v355;
    v529.m128i_i64[0] = v354;
    v529.m128i_i64[1] = v354;
  }
  v356 = _mm_loadu_si128(&v529);
  v357 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v356;
  if ( v529.m128i_i64[0] != v354 )
  {
    do
    {
      sub_B70650(*v357);
      v527.m128i_i64[0] += 8;
      sub_B76BB0((__int64)&v526);
      v357 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v354 );
  }
  v358 = v1 + 1744;
  sub_B75170(&v528, (__int64 *)(v1 + 1744));
  v359 = *(_QWORD *)(v1 + 1752) + 8LL * *(unsigned int *)(v1 + 1768);
  for ( n = (__int64 *)v529.m128i_i64[0]; v359 != v529.m128i_i64[0]; n = (__int64 *)v529.m128i_i64[0] )
  {
    sub_B70650(*n);
    v529.m128i_i64[0] += 8;
    sub_B76BF0((__int64)&v528);
  }
  v361 = *(_QWORD *)(v1 + 1784);
  v517 = v1 + 1776;
  v362 = v361 + 8LL * *(unsigned int *)(v1 + 1800);
  v363 = *(_QWORD *)(v1 + 1776);
  if ( *(_DWORD *)(v1 + 1792) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1776);
    v529.m128i_i64[0] = v361;
    v529.m128i_i64[1] = v362;
    *(_QWORD *)&v528 = v1 + 1776;
    sub_B76C30((__int64)&v528);
    v362 = *(_QWORD *)(v1 + 1784) + 8LL * *(unsigned int *)(v1 + 1800);
  }
  else
  {
    v529.m128i_i64[0] = v361 + 8LL * *(unsigned int *)(v1 + 1800);
    v528 = __PAIR128__(v363, v517);
    v529.m128i_i64[1] = v362;
  }
  v364 = _mm_loadu_si128(&v529);
  v365 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v364;
  if ( v529.m128i_i64[0] != v362 )
  {
    do
    {
      sub_B70650(*v365);
      v527.m128i_i64[0] += 8;
      sub_B76C30((__int64)&v526);
      v365 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v362 );
  }
  v366 = *(_QWORD *)(v1 + 1816);
  v519 = v1 + 1808;
  v367 = v366 + 8LL * *(unsigned int *)(v1 + 1832);
  v368 = *(_QWORD *)(v1 + 1808);
  if ( *(_DWORD *)(v1 + 1824) )
  {
    *((_QWORD *)&v528 + 1) = *(_QWORD *)(v1 + 1808);
    v529.m128i_i64[0] = v366;
    v529.m128i_i64[1] = v367;
    *(_QWORD *)&v528 = v1 + 1808;
    sub_B76C70((__int64)&v528);
    v367 = *(_QWORD *)(v1 + 1816) + 8LL * *(unsigned int *)(v1 + 1832);
  }
  else
  {
    v529.m128i_i64[0] = v366 + 8LL * *(unsigned int *)(v1 + 1832);
    v528 = __PAIR128__(v368, v519);
    v529.m128i_i64[1] = v367;
  }
  v369 = _mm_loadu_si128(&v529);
  v370 = (__int64 *)v529.m128i_i64[0];
  v526 = _mm_loadu_si128((const __m128i *)&v528);
  v527 = v369;
  if ( v529.m128i_i64[0] != v367 )
  {
    do
    {
      sub_B70650(*v370);
      v527.m128i_i64[0] += 8;
      sub_B76C70((__int64)&v526);
      v370 = (__int64 *)v527.m128i_i64[0];
    }
    while ( v527.m128i_i64[0] != v367 );
  }
  v372.m128i_i64[0] = *(_QWORD *)(v1 + 2128);
  v371 = *(_QWORD *)(v1 + 2120);
  v372.m128i_i64[1] = v372.m128i_i64[0] + 8LL * *(unsigned int *)(v1 + 2144);
  if ( *(_DWORD *)(v1 + 2136) )
  {
    *(_QWORD *)&v528 = v1 + 2120;
    v529 = v372;
    *((_QWORD *)&v528 + 1) = v371;
    sub_B76BB0((__int64)&v528);
    v478 = v529.m128i_i64[1];
    v479 = (__int64 *)v529.m128i_i64[0];
    v480 = *(_QWORD *)(v1 + 2128) + 8LL * *(unsigned int *)(v1 + 2144);
    if ( v529.m128i_i64[0] != v480 )
    {
      do
      {
        sub_AC70E0(*v479, v358);
        do
          ++v479;
        while ( v479 != (__int64 *)v478 && (*v479 == -4096 || *v479 == -8192) );
      }
      while ( v479 != (__int64 *)v480 );
    }
  }
  v374.m128i_i64[0] = *(_QWORD *)(v1 + 1752);
  v373 = *(_QWORD *)(v1 + 1744);
  v374.m128i_i64[1] = v374.m128i_i64[0] + 8LL * *(unsigned int *)(v1 + 1768);
  if ( *(_DWORD *)(v1 + 1760) )
  {
    *(_QWORD *)&v528 = v1 + 1744;
    v529 = v374;
    *((_QWORD *)&v528 + 1) = v373;
    sub_B76BF0((__int64)&v528);
    v463 = v529.m128i_i64[1];
    v462 = (__int64 *)v529.m128i_i64[0];
    v464 = *(_QWORD *)(v1 + 1752) + 8LL * *(unsigned int *)(v1 + 1768);
    if ( v529.m128i_i64[0] != v464 )
    {
      do
      {
        sub_AC70E0(*v462, v358);
        do
          ++v462;
        while ( v462 != (__int64 *)v463 && (*v462 == -4096 || *v462 == -8192) );
      }
      while ( v462 != (__int64 *)v464 );
    }
  }
  v375.m128i_i64[0] = *(_QWORD *)(v1 + 1784);
  v376 = *(_QWORD *)(v1 + 1776);
  if ( *(_DWORD *)(v1 + 1792) )
  {
    v358 = v1 + 1776;
    v375.m128i_i64[1] = v375.m128i_i64[0] + 8LL * *(unsigned int *)(v1 + 1800);
    v529 = v375;
    *(_QWORD *)&v528 = v1 + 1776;
    *((_QWORD *)&v528 + 1) = v376;
    sub_B76C30((__int64)&v528);
    v460 = v529.m128i_i64[1];
    v459 = (__int64 *)v529.m128i_i64[0];
    v461 = *(_QWORD *)(v1 + 1784) + 8LL * *(unsigned int *)(v1 + 1800);
    if ( v529.m128i_i64[0] != v461 )
    {
      do
      {
        sub_AC70E0(*v459, v517);
        do
          ++v459;
        while ( v459 != (__int64 *)v460 && (*v459 == -8192 || *v459 == -4096) );
      }
      while ( v459 != (__int64 *)v461 );
    }
  }
  v377.m128i_i64[0] = *(_QWORD *)(v1 + 1816);
  v378 = *(_QWORD *)(v1 + 1808);
  if ( *(_DWORD *)(v1 + 1824) )
  {
    v358 = v1 + 1808;
    v377.m128i_i64[1] = v377.m128i_i64[0] + 8LL * *(unsigned int *)(v1 + 1832);
    v529 = v377;
    *(_QWORD *)&v528 = v1 + 1808;
    *((_QWORD *)&v528 + 1) = v378;
    sub_B76C70((__int64)&v528);
    v457 = v529.m128i_i64[1];
    v456 = (__int64 *)v529.m128i_i64[0];
    v458 = *(_QWORD *)(v1 + 1816) + 8LL * *(unsigned int *)(v1 + 1832);
    if ( v529.m128i_i64[0] != v458 )
    {
      do
      {
        sub_AC70E0(*v456, v519);
        do
          ++v456;
        while ( v456 != (__int64 *)v457 && (*v456 == -8192 || *v456 == -4096) );
      }
      while ( v456 != (__int64 *)v458 );
    }
  }
  sub_B70720(v1 + 2152);
  sub_B72480(v1 + 1712);
  sub_B726D0(v1 + 1840);
  sub_B72920(v1 + 1872);
  sub_B72B70(v1 + 1904);
  sub_B72DC0(v1 + 1936);
  sub_B73010(v1 + 208);
  sub_B73010(v1 + 240);
  sub_B73550(v1 + 272);
  sub_B739A0(v1 + 304);
  sub_B74080(v1 + 336, v358, v379, v380, v381);
  sub_B75DF0(v1 + 368, v358, v382, v383, v384);
  if ( *(_DWORD *)(v1 + 1980) )
    sub_B71FD0(v1 + 1968);
  sub_C65AC0(&v528, *(_QWORD *)(v1 + 432));
  v385 = *(unsigned int *)(v1 + 440);
  v526.m128i_i64[0] = v528;
  v386 = *(_QWORD *)(v1 + 432) + 8 * v385;
  sub_C65AC0(&v528, v386);
  v387 = v528;
  while ( 1 )
  {
    v388 = v526.m128i_i64[0];
    if ( v387 == v526.m128i_i64[0] )
      break;
    sub_C65AF0(&v526);
    if ( v388 )
    {
      v386 = 24LL * *(unsigned int *)(v388 + 56);
      sub_C7D6A0(*(_QWORD *)(v388 + 40), v386, 8);
      j___libc_free_0(v388);
    }
  }
  v389 = *(unsigned int *)(v1 + 616);
  *(_QWORD *)&v528 = &v529;
  *((_QWORD *)&v528 + 1) = 0x800000000LL;
  v390 = v389;
  if ( (unsigned int)v389 > 8 )
  {
    v386 = (__int64)&v529;
    sub_C8D5F0(&v528, &v529, v389, 8);
    v390 = *(_DWORD *)(v1 + 616);
  }
  v391 = *(_QWORD *)(v1 + 608);
  if ( v390 )
  {
    v386 = v520;
    v469 = *(_QWORD *)(v1 + 600);
    v527.m128i_i64[1] = v391 + 16LL * *(unsigned int *)(v1 + 624);
    v527.m128i_i64[0] = v391;
    v526.m128i_i64[0] = v520;
    v526.m128i_i64[1] = v469;
    sub_B74F70((__int64)&v526);
    v470 = v527.m128i_i64[0];
    for ( ii = *(_QWORD *)(v1 + 608) + 16LL * *(unsigned int *)(v1 + 624); ii != v527.m128i_i64[0]; v470 = v527.m128i_i64[0] )
    {
      v472 = *(_QWORD *)(v470 + 8);
      v473 = DWORD2(v528);
      v474 = DWORD2(v528) + 1LL;
      if ( v474 > HIDWORD(v528) )
      {
        v386 = (__int64)&v529;
        sub_C8D5F0(&v528, &v529, v474, 8);
        v473 = DWORD2(v528);
      }
      *(_QWORD *)(v528 + 8 * v473) = v472;
      ++DWORD2(v528);
      v527.m128i_i64[0] += 16;
      sub_B74F70((__int64)&v526);
    }
  }
  sub_B73280(v520);
  v392 = (__m128i *)v528;
  v393 = v528 + 8LL * DWORD2(v528);
  if ( v393 != (_QWORD)v528 )
  {
    v394 = (_QWORD *)v528;
    do
    {
      v395 = *v394;
      if ( *v394 )
      {
        sub_B91290(*v394);
        v386 = 32;
        j_j___libc_free_0(v395, 32);
      }
      ++v394;
    }
    while ( (_QWORD *)v393 != v394 );
    v392 = (__m128i *)v528;
  }
  if ( v392 != &v529 )
    _libc_free(v392, v386);
  v396 = *(_QWORD *)(v1 + 576);
  if ( *(_DWORD *)(v1 + 584) )
  {
    v529.m128i_i64[1] = v396 + 16LL * *(unsigned int *)(v1 + 592);
    *(_QWORD *)&v528 = v1 + 568;
    v465 = *(_QWORD *)(v1 + 568);
    v529.m128i_i64[0] = v396;
    *((_QWORD *)&v528 + 1) = v465;
    sub_B74F30((__int64)&v528);
    v466 = v529.m128i_i64[0];
    for ( jj = *(_QWORD *)(v1 + 576) + 16LL * *(unsigned int *)(v1 + 592); jj != v529.m128i_i64[0]; v466 = v529.m128i_i64[0] )
    {
      v468 = *(_QWORD *)(v466 + 8);
      if ( v468 )
      {
        sub_B73450(v468 + 24);
        j_j___libc_free_0(v468, 144);
      }
      v529.m128i_i64[0] += 16;
      sub_B74F30((__int64)&v528);
    }
  }
  sub_2240A30(v1 + 3624);
  sub_2240A30(v1 + 3592);
  if ( (*(_BYTE *)(v1 + 3520) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(v1 + 3528), 16LL * *(unsigned int *)(v1 + 3536), 8);
  v397 = *(unsigned int *)(v1 + 3488);
  if ( (_DWORD)v397 )
  {
    v398 = *(_QWORD **)(v1 + 3472);
    v399 = &v398[5 * v397];
    do
    {
      if ( *v398 != -8192 && *v398 != -4096 )
        sub_2240A30(v398 + 1);
      v398 += 5;
    }
    while ( v399 != v398 );
    v397 = *(unsigned int *)(v1 + 3488);
  }
  v400 = 40 * v397;
  sub_C7D6A0(*(_QWORD *)(v1 + 3472), 40 * v397, 8);
  if ( *(_DWORD *)(v1 + 3452) )
  {
    v401 = *(unsigned int *)(v1 + 3448);
    v402 = *(_QWORD *)(v1 + 3440);
    if ( (_DWORD)v401 )
    {
      v403 = 8 * v401;
      v404 = 0;
      do
      {
        v405 = *(_QWORD **)(v402 + v404);
        if ( v405 != (_QWORD *)-8LL && v405 )
        {
          v400 = *v405 + 17LL;
          sub_C7D6A0(v405, v400, 8);
          v402 = *(_QWORD *)(v1 + 3440);
        }
        v404 += 8;
      }
      while ( v403 != v404 );
    }
  }
  else
  {
    v402 = *(_QWORD *)(v1 + 3440);
  }
  _libc_free(v402, v400);
  sub_B72400((__int64 *)(v1 + 3416), v400);
  sub_C7D6A0(*(_QWORD *)(v1 + 3392), 24LL * *(unsigned int *)(v1 + 3408), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 3360), 16LL * *(unsigned int *)(v1 + 3376), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 3328), 24LL * *(unsigned int *)(v1 + 3344), 8);
  v406 = 24LL * *(unsigned int *)(v1 + 3312);
  sub_C7D6A0(*(_QWORD *)(v1 + 3296), v406, 8);
  v407 = *(unsigned int *)(v1 + 3280);
  if ( (_DWORD)v407 )
  {
    v408 = *(_QWORD **)(v1 + 3264);
    v409 = &v408[4 * v407];
    do
    {
      if ( *v408 != -4096 && *v408 != -8192 )
      {
        v410 = (_QWORD *)v408[1];
        if ( v410 != v408 + 3 )
          _libc_free(v410, v406);
      }
      v408 += 4;
    }
    while ( v409 != v408 );
    LODWORD(v407) = *(_DWORD *)(v1 + 3280);
  }
  v411 = 32LL * (unsigned int)v407;
  sub_C7D6A0(*(_QWORD *)(v1 + 3264), v411, 8);
  v412 = *(unsigned int *)(v1 + 3248);
  if ( (_DWORD)v412 )
  {
    v413 = *(_QWORD *)(v1 + 3232);
    v414 = v413 + 40 * v412;
    do
    {
      if ( *(_QWORD *)v413 != -4096 && *(_QWORD *)v413 != -8192 )
      {
        v415 = *(_QWORD *)(v413 + 8);
        v416 = v415 + 16LL * *(unsigned int *)(v413 + 16);
        if ( v415 != v416 )
        {
          do
          {
            v411 = *(_QWORD *)(v416 - 8);
            v416 -= 16;
            if ( v411 )
              sub_B91220(v416 + 8);
          }
          while ( v415 != v416 );
          v416 = *(_QWORD *)(v413 + 8);
        }
        if ( v416 != v413 + 24 )
          _libc_free(v416, v411);
      }
      v413 += 40;
    }
    while ( v414 != v413 );
    v412 = *(unsigned int *)(v1 + 3248);
  }
  v417 = 40 * v412;
  sub_C7D6A0(*(_QWORD *)(v1 + 3232), 40 * v412, 8);
  sub_B72400((__int64 *)(v1 + 3200), v417);
  sub_C7D6A0(*(_QWORD *)(v1 + 3176), 16LL * *(unsigned int *)(v1 + 3192), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 3144), 24LL * *(unsigned int *)(v1 + 3160), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 3112), 16LL * *(unsigned int *)(v1 + 3128), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 3072), 24LL * *(unsigned int *)(v1 + 3088), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 3040), 24LL * *(unsigned int *)(v1 + 3056), 8);
  v418 = 8LL * *(unsigned int *)(v1 + 3024);
  sub_C7D6A0(*(_QWORD *)(v1 + 3008), v418, 8);
  if ( *(_DWORD *)(v1 + 2980) )
  {
    v419 = *(unsigned int *)(v1 + 2976);
    v420 = *(_QWORD *)(v1 + 2968);
    if ( (_DWORD)v419 )
    {
      v421 = 8 * v419;
      v422 = 0;
      do
      {
        v423 = *(_QWORD **)(v420 + v422);
        if ( v423 && v423 != (_QWORD *)-8LL )
        {
          v418 = *v423 + 17LL;
          sub_C7D6A0(v423, v418, 8);
          v420 = *(_QWORD *)(v1 + 2968);
        }
        v422 += 8;
      }
      while ( v421 != v422 );
    }
  }
  else
  {
    v420 = *(_QWORD *)(v1 + 2968);
  }
  _libc_free(v420, v418);
  sub_C7D6A0(*(_QWORD *)(v1 + 2944), 8LL * *(unsigned int *)(v1 + 2960), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 2912), 8LL * *(unsigned int *)(v1 + 2928), 8);
  v424 = 16LL * *(unsigned int *)(v1 + 2896);
  sub_C7D6A0(*(_QWORD *)(v1 + 2880), v424, 8);
  sub_B74CF0(v1 + 2776);
  sub_B72320(v1 + 2776, v424);
  v425 = 16LL * *(unsigned int *)(v1 + 2768);
  sub_C7D6A0(*(_QWORD *)(v1 + 2752), v425, 8);
  sub_B72320(v1 + 2640, v425);
  v426 = *(_QWORD *)(v1 + 2632);
  if ( v426 )
  {
    sub_BD7260(*(_QWORD *)(v1 + 2632));
    sub_BD2DD0(v426);
  }
  sub_C7D6A0(*(_QWORD *)(v1 + 2160), 8LL * *(unsigned int *)(v1 + 2176), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 2128), 8LL * *(unsigned int *)(v1 + 2144), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 2096), 8LL * *(unsigned int *)(v1 + 2112), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 2064), 16LL * *(unsigned int *)(v1 + 2080), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 2032), 16LL * *(unsigned int *)(v1 + 2048), 8);
  v427 = 24LL * *(unsigned int *)(v1 + 2016);
  sub_C7D6A0(*(_QWORD *)(v1 + 2000), v427, 8);
  if ( *(_DWORD *)(v1 + 1980) )
  {
    v428 = *(unsigned int *)(v1 + 1976);
    v429 = *(_QWORD *)(v1 + 1968);
    if ( (_DWORD)v428 )
    {
      v430 = 8 * v428;
      v431 = 0;
      do
      {
        v432 = *(_QWORD **)(v429 + v431);
        if ( v432 != (_QWORD *)-8LL && v432 )
        {
          v433 = v432[1];
          v434 = *v432 + 17LL;
          if ( v433 )
          {
            v435 = *(_QWORD *)(v433 + 32);
            if ( v435 )
            {
              v436 = *(_QWORD *)(v435 + 32);
              if ( v436 )
              {
                v506 = *(_QWORD *)(v433 + 32);
                v521 = v432[1];
                v523 = *(_QWORD *)(v435 + 32);
                sub_AC5B80((__int64 *)(v436 + 32));
                sub_BD7260(v523);
                sub_BD2DD0(v523);
                v435 = v506;
                v433 = v521;
              }
              v522 = v433;
              v524 = v435;
              sub_BD7260(v435);
              sub_BD2DD0(v524);
              v433 = v522;
            }
            v525 = v433;
            sub_BD7260(v433);
            sub_BD2DD0(v525);
          }
          v427 = v434;
          sub_C7D6A0(v432, v434, 8);
          v429 = *(_QWORD *)(v1 + 1968);
        }
        v431 += 8;
      }
      while ( v431 != v430 );
    }
  }
  else
  {
    v429 = *(_QWORD *)(v1 + 1968);
  }
  _libc_free(v429, v427);
  sub_B74C80(v1 + 1936);
  sub_C7D6A0(*(_QWORD *)(v1 + 1944), 16LL * *(unsigned int *)(v1 + 1960), 8);
  sub_B74C10(v1 + 1904);
  sub_C7D6A0(*(_QWORD *)(v1 + 1912), 16LL * *(unsigned int *)(v1 + 1928), 8);
  sub_B74BA0(v1 + 1872);
  sub_C7D6A0(*(_QWORD *)(v1 + 1880), 16LL * *(unsigned int *)(v1 + 1896), 8);
  sub_B74B30(v1 + 1840);
  sub_C7D6A0(*(_QWORD *)(v1 + 1848), 16LL * *(unsigned int *)(v1 + 1864), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 1816), 8LL * *(unsigned int *)(v1 + 1832), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 1784), 8LL * *(unsigned int *)(v1 + 1800), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 1752), 8LL * *(unsigned int *)(v1 + 1768), 8);
  sub_B74AC0(v1 + 1712);
  sub_C7D6A0(*(_QWORD *)(v1 + 1720), 16LL * *(unsigned int *)(v1 + 1736), 8);
  v437 = *(_QWORD *)(v1 + 1688);
  if ( v437 )
    j_j___libc_free_0(v437, *(_QWORD *)(v1 + 1704) - v437);
  v438 = *(_QWORD *)(v1 + 1664);
  if ( v438 )
    j_j___libc_free_0(v438, *(_QWORD *)(v1 + 1680) - v438);
  if ( *(_BYTE *)(v1 + 1656) )
  {
    v483 = *(unsigned int *)(v1 + 1648);
    *(_BYTE *)(v1 + 1656) = 0;
    sub_C7D6A0(*(_QWORD *)(v1 + 1632), 16 * v483, 8);
  }
  sub_C7D6A0(*(_QWORD *)(v1 + 1600), 8LL * *(unsigned int *)(v1 + 1616), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 1568), 8LL * *(unsigned int *)(v1 + 1584), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 1536), 8LL * *(unsigned int *)(v1 + 1552), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 1504), 8LL * *(unsigned int *)(v1 + 1520), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 1472), 8LL * *(unsigned int *)(v1 + 1488), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 1440), 8LL * *(unsigned int *)(v1 + 1456), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 1408), 8LL * *(unsigned int *)(v1 + 1424), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 1376), 8LL * *(unsigned int *)(v1 + 1392), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 1344), 8LL * *(unsigned int *)(v1 + 1360), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 1312), 8LL * *(unsigned int *)(v1 + 1328), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 1280), 8LL * *(unsigned int *)(v1 + 1296), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 1248), 8LL * *(unsigned int *)(v1 + 1264), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 1216), 8LL * *(unsigned int *)(v1 + 1232), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 1184), 8LL * *(unsigned int *)(v1 + 1200), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 1152), 8LL * *(unsigned int *)(v1 + 1168), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 1120), 8LL * *(unsigned int *)(v1 + 1136), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 1088), 8LL * *(unsigned int *)(v1 + 1104), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 1056), 8LL * *(unsigned int *)(v1 + 1072), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 1024), 8LL * *(unsigned int *)(v1 + 1040), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 992), 8LL * *(unsigned int *)(v1 + 1008), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 960), 8LL * *(unsigned int *)(v1 + 976), 8);
  v439 = *(unsigned int *)(v1 + 944);
  v440 = 0;
  if ( (_DWORD)v439 )
    v440 = 8 * v439;
  sub_C7D6A0(*(_QWORD *)(v1 + 928), v440, 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 896), 8LL * *(unsigned int *)(v1 + 912), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 864), 8LL * *(unsigned int *)(v1 + 880), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 832), 8LL * *(unsigned int *)(v1 + 848), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 800), 8LL * *(unsigned int *)(v1 + 816), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 768), 8LL * *(unsigned int *)(v1 + 784), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 736), 8LL * *(unsigned int *)(v1 + 752), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 704), 8LL * *(unsigned int *)(v1 + 720), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 672), 8LL * *(unsigned int *)(v1 + 688), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 640), 8LL * *(unsigned int *)(v1 + 656), 8);
  sub_C7D6A0(*(_QWORD *)(v1 + 608), 16LL * *(unsigned int *)(v1 + 624), 8);
  v441 = 16LL * *(unsigned int *)(v1 + 592);
  sub_C7D6A0(*(_QWORD *)(v1 + 576), v441, 8);
  sub_B72320(v1 + 472, v441);
  _libc_free(*(_QWORD *)(v1 + 448), v441);
  sub_C65770(v1 + 432);
  sub_C65770(v1 + 416);
  sub_C65770(v1 + 400);
  sub_B74660(v1 + 368, v441, v442, v443, v444);
  v445 = 40LL * *(unsigned int *)(v1 + 392);
  sub_C7D6A0(*(_QWORD *)(v1 + 376), v445, 8);
  sub_B73DB0(v1 + 336, v445, v446, v447, v448);
  sub_C7D6A0(*(_QWORD *)(v1 + 344), 32LL * *(unsigned int *)(v1 + 360), 8);
  sub_B738D0(v1 + 304);
  sub_C7D6A0(*(_QWORD *)(v1 + 312), 32LL * *(unsigned int *)(v1 + 328), 8);
  sub_B73490(v1 + 272);
  sub_C7D6A0(*(_QWORD *)(v1 + 280), 24LL * *(unsigned int *)(v1 + 296), 8);
  sub_B72290(v1 + 240);
  sub_B72290(v1 + 208);
  sub_C7D6A0(*(_QWORD *)(v1 + 184), 16LL * *(unsigned int *)(v1 + 200), 8);
  v449 = *(_QWORD *)(v1 + 152);
  if ( v449 )
    j_j___libc_free_0(v449, 8);
  v450 = *(_QWORD *)(v1 + 104);
  if ( v450 )
  {
    v451 = *(void (**)(void))(*(_QWORD *)v450 + 8LL);
    if ( (char *)v451 == (char *)sub_B70640 )
      j_j___libc_free_0(v450, 32);
    else
      v451();
  }
  v452 = *(_QWORD *)(v1 + 96);
  if ( v452 )
  {
    if ( *(_BYTE *)(v452 + 64) )
    {
      *(_BYTE *)(v452 + 64) = 0;
      sub_2240A30(v452 + 32);
      v453 = *(_QWORD *)(v452 + 24);
      if ( !v453 )
        goto LABEL_441;
    }
    else
    {
      v453 = *(_QWORD *)(v452 + 24);
      if ( !v453 )
      {
LABEL_441:
        if ( *(_BYTE *)(v452 + 16) )
        {
          *(_BYTE *)(v452 + 16) = 0;
          sub_C88FF0((void *)v452);
        }
        j_j___libc_free_0(v452, 72);
        goto LABEL_444;
      }
    }
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v453 + 8LL))(v453);
    goto LABEL_441;
  }
LABEL_444:
  v454 = 16LL * *(unsigned int *)(v1 + 88);
  result = sub_C7D6A0(*(_QWORD *)(v1 + 72), v454, 8);
  if ( !*(_BYTE *)(v1 + 28) )
    return _libc_free(*(_QWORD *)(v1 + 8), v454);
  return result;
}
