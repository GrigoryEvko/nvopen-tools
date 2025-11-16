// Function: sub_1834E90
// Address: 0x1834e90
//
__int64 __fastcall sub_1834E90(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128 a5,
        __m128i a6,
        __m128i a7,
        __m128i a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 v12; // rax
  unsigned __int64 *v13; // r12
  unsigned __int64 *v14; // rax
  int v15; // r8d
  _BYTE *v16; // r15
  __int64 v17; // rax
  unsigned __int64 v18; // r14
  const void *v19; // r9
  int v20; // ebx
  signed __int64 v21; // r13
  __int64 v22; // rdx
  int v23; // ecx
  _QWORD *v24; // rdx
  _QWORD *v25; // rax
  __int64 v26; // r12
  __m128 v27; // rax
  char v28; // r13
  char v29; // r13
  __int64 v30; // rax
  _QWORD *v31; // r15
  unsigned __int64 v32; // rbx
  __int64 v33; // r13
  __int64 v34; // r12
  __int64 *v35; // rax
  __int64 v36; // rax
  unsigned __int64 **v37; // rsi
  __int64 v38; // rcx
  __int64 v39; // rdi
  unsigned __int64 v40; // rax
  int v41; // r8d
  int v42; // r9d
  unsigned __int8 v43; // dl
  unsigned __int64 v44; // r15
  unsigned __int64 *v45; // rdx
  int v46; // r8d
  int v47; // r9d
  __int64 *v48; // rdi
  int *v49; // rax
  int *v50; // rsi
  __int64 v51; // rcx
  __int64 v52; // rdx
  int *v53; // r12
  _BYTE *v54; // rax
  _BYTE *v55; // rsi
  unsigned __int64 v56; // r13
  __int64 v57; // rax
  char *v58; // rdi
  signed __int64 v59; // rbx
  size_t v60; // r14
  unsigned __int64 *v61; // rdx
  char *v62; // r9
  unsigned __int64 *v63; // rsi
  char *v64; // r10
  char *v65; // rax
  char *v66; // rcx
  _QWORD *v67; // rdx
  signed __int64 v68; // rax
  char *v69; // rax
  unsigned __int64 *v70; // rax
  __int64 *v71; // r14
  __int64 *v72; // rbx
  unsigned __int64 v73; // rdx
  _QWORD *v74; // rax
  __int64 v75; // rbx
  int v76; // r8d
  int v77; // r9d
  __int64 v78; // rax
  __int64 v79; // rax
  unsigned __int64 v80; // r15
  signed __int64 v81; // rdi
  signed __int64 v82; // rbx
  char *v83; // rax
  _BYTE *v84; // r14
  char *v85; // r13
  __int64 v86; // r12
  __int64 v87; // r13
  _QWORD *v88; // rax
  __int64 v89; // rdi
  __int64 v90; // rax
  char *v91; // rsi
  unsigned __int64 *v92; // rax
  __int64 *v93; // r14
  unsigned __int64 v94; // rdx
  signed __int64 v95; // r13
  _QWORD *v96; // rax
  __int64 v97; // rbx
  __int64 v98; // rax
  __int64 *v99; // rax
  __int64 *v100; // rax
  int v101; // r8d
  __int64 *v102; // r11
  __int64 *v103; // rax
  __int64 v104; // rdx
  __int64 *v105; // rax
  _QWORD *v106; // rax
  unsigned __int64 v107; // rbx
  __int64 v108; // rax
  __int64 v109; // rbx
  __int64 v110; // rax
  __int64 v111; // rax
  __int64 v112; // r14
  _QWORD *v113; // rax
  __int64 v114; // rax
  __int64 v115; // rbx
  unsigned __int64 **v116; // rdx
  char v117; // al
  __int64 v118; // r12
  _QWORD *v119; // r15
  __int64 v120; // rax
  __int64 *v121; // rax
  __int64 v122; // rax
  __int64 v123; // rcx
  __int64 *v124; // r11
  __int64 v125; // rax
  unsigned __int64 *v126; // rax
  unsigned __int64 v127; // rdx
  _QWORD *v128; // rax
  unsigned __int64 v129; // r12
  __int64 v130; // rax
  __int64 v131; // rax
  _QWORD *v132; // rax
  __int64 v133; // rax
  void *v134; // rax
  char *v135; // rdx
  __int64 *v136; // rax
  __int64 *v137; // rbx
  int v138; // r13d
  __int64 v139; // r14
  __int64 v140; // rax
  __int64 v141; // rdx
  __int64 v142; // r15
  int v143; // r15d
  __int64 v144; // rax
  __int64 v145; // rdx
  __int64 v146; // rax
  unsigned __int64 v147; // r15
  __int64 v148; // rax
  __int64 v149; // r15
  __int64 v150; // rax
  char v151; // al
  __int64 v152; // rax
  __int64 v153; // rdx
  __int64 v154; // r15
  int v155; // r15d
  __int64 v156; // rax
  __int64 v157; // rdx
  __int64 v158; // r15
  __int64 v159; // rax
  __int64 v160; // rdx
  __int64 v161; // rbx
  __int64 v162; // rax
  __int64 v163; // r15
  __int64 v164; // rax
  char **v165; // rax
  __int64 v166; // rdx
  char *v167; // rsi
  __int64 v168; // rax
  __int64 v169; // rdi
  __int64 *v170; // rbx
  __int64 *v171; // r12
  __int64 v172; // r14
  unsigned __int64 v173; // rdx
  int v174; // esi
  __int64 v175; // rax
  unsigned int v176; // r12d
  __int64 v177; // rax
  __int64 *v178; // rcx
  unsigned __int64 v179; // r14
  unsigned __int64 v180; // r12
  __int64 v181; // rbx
  __int64 *v182; // rax
  __int64 v183; // rsi
  unsigned __int64 **v184; // rbx
  unsigned __int64 *v185; // rsi
  double v186; // xmm4_8
  double v187; // xmm5_8
  __m128i *v188; // rbx
  __m128i *v189; // r12
  __int64 v190; // rdi
  __int64 *v191; // rbx
  unsigned __int64 *v192; // r13
  __int64 *v193; // r12
  unsigned __int64 v194; // rdx
  unsigned __int64 v195; // rcx
  unsigned __int64 *v196; // r13
  unsigned __int64 *v197; // rax
  unsigned __int64 *v198; // r15
  double v199; // xmm4_8
  double v200; // xmm5_8
  __int64 v201; // rdi
  __int64 *v203; // rax
  signed __int64 v204; // rsi
  __int64 v205; // rdi
  __int64 v206; // rax
  double v207; // xmm4_8
  double v208; // xmm5_8
  __int64 v209; // rax
  __int64 v210; // rdx
  __int64 v211; // rbx
  __int64 v212; // rax
  __int64 v213; // r15
  __int64 v214; // rax
  char **v215; // rax
  __int64 v216; // rdx
  char *v217; // rsi
  __int64 v218; // rax
  __int64 v219; // rdi
  __int64 v220; // rsi
  unsigned __int8 *v221; // rsi
  unsigned __int64 v222; // rsi
  int v223; // edx
  __int64 v224; // rax
  __int64 v225; // r14
  __int64 *v226; // r10
  __int64 v227; // r11
  __int64 v228; // rdx
  __int64 v229; // rax
  __int64 v230; // rcx
  int v231; // r12d
  __int64 v232; // rax
  __int64 v233; // rcx
  unsigned __int64 v234; // rbx
  int v235; // eax
  unsigned __int64 v236; // rax
  char *v237; // rsi
  __int64 v238; // rbx
  int v239; // r8d
  int v240; // r9d
  __int64 v241; // rax
  int *v242; // rax
  int *v243; // rsi
  __int64 v244; // rcx
  __int64 v245; // rdx
  void **v246; // r13
  __int64 v247; // r12
  _QWORD *v248; // rax
  _QWORD *v249; // rbx
  unsigned __int64 v250; // rax
  _QWORD *v251; // r15
  _QWORD *v252; // r14
  _QWORD *v253; // r15
  void **v254; // rax
  _QWORD *v255; // r13
  __int64 v256; // r14
  __int64 v257; // rax
  __int64 *v258; // rax
  _BYTE *v259; // r8
  unsigned __int64 v260; // r15
  size_t v261; // r14
  char *v262; // rcx
  _BYTE *v263; // rax
  unsigned __int64 v264; // r14
  __int64 v265; // rax
  char *v266; // rdi
  size_t v267; // r15
  unsigned __int64 *v268; // rdx
  char *v269; // r9
  unsigned __int64 *v270; // rsi
  char *v271; // rcx
  char *v272; // rax
  char *v273; // r10
  _QWORD *v274; // rdx
  signed __int64 v275; // rax
  char *v276; // rax
  unsigned __int64 *v277; // rax
  __int64 v278; // r14
  char *v279; // rsi
  __int64 v280; // rax
  __int64 **v281; // rax
  unsigned __int64 *v282; // rax
  int v283; // r8d
  int v284; // r9d
  __int64 v285; // rax
  _BYTE *v286; // r14
  unsigned __int64 v287; // rdx
  signed __int64 v288; // rbx
  unsigned __int64 v289; // rax
  bool v290; // cf
  unsigned __int64 v291; // rax
  char *v292; // rbx
  char *v293; // rcx
  int *v294; // rax
  __int64 v295; // rcx
  __int64 v296; // rdx
  _QWORD *v297; // rax
  __int64 v298; // r12
  char *v299; // rsi
  __int64 v300; // r13
  __int64 v301; // rax
  unsigned int v302; // edx
  __int64 *v303; // rax
  void *v304; // rbx
  size_t v305; // r15
  __int64 v306; // r13
  __int64 k; // r14
  const void *v308; // rdi
  __int64 v309; // rdx
  char *v310; // rsi
  signed __int64 v311; // rax
  unsigned __int64 v312; // rcx
  __int8 *v313; // r8
  size_t v314; // r9
  void **v315; // rax
  __m128i *v316; // rax
  char *v317; // rax
  unsigned __int64 v318; // rax
  unsigned __int64 *v319; // rax
  void **v320; // rdi
  __int64 v321; // rax
  void *v322; // rdi
  _QWORD *v323; // rax
  __int64 v324; // rax
  __int64 v325; // r13
  unsigned int v326; // r12d
  unsigned int v327; // eax
  unsigned int v328; // ebx
  _QWORD *v329; // rax
  _QWORD *v330; // r14
  _QWORD *v331; // rax
  __int64 v332; // rax
  double v333; // xmm4_8
  double v334; // xmm5_8
  __int64 v335; // r15
  _QWORD *v336; // rax
  __int64 v337; // rax
  void *v338; // rax
  char *v339; // rdx
  char v340; // al
  unsigned __int64 **v341; // rdx
  _QWORD *v342; // rax
  __int64 v343; // r12
  __int64 v344; // rax
  __int64 *v345; // rax
  __int64 *v346; // rdi
  __int64 v347; // rax
  char *v348; // rdx
  char v349; // al
  unsigned __int64 **v350; // rdx
  _QWORD *v351; // rdi
  __int64 j; // rbx
  _QWORD *v353; // rax
  __int64 m; // rdi
  _QWORD *v355; // r14
  double v356; // xmm4_8
  double v357; // xmm5_8
  unsigned __int64 *v358; // rax
  unsigned __int64 v359; // rdx
  double v360; // xmm4_8
  double v361; // xmm5_8
  __int64 v362; // rbx
  char *v363; // rax
  const void *v364; // r9
  const void *v365; // rsi
  __int64 *v367; // [rsp+10h] [rbp-360h]
  unsigned __int64 **v368; // [rsp+30h] [rbp-340h]
  __int64 v369; // [rsp+48h] [rbp-328h]
  __int64 *v370; // [rsp+60h] [rbp-310h]
  __int64 *v371; // [rsp+68h] [rbp-308h]
  int v372; // [rsp+68h] [rbp-308h]
  __int8 *v373; // [rsp+68h] [rbp-308h]
  int v374; // [rsp+68h] [rbp-308h]
  unsigned __int64 *v375; // [rsp+70h] [rbp-300h]
  _QWORD *v377; // [rsp+80h] [rbp-2F0h]
  _DWORD *v378; // [rsp+80h] [rbp-2F0h]
  char *v379; // [rsp+80h] [rbp-2F0h]
  char n; // [rsp+88h] [rbp-2E8h]
  char *nb; // [rsp+88h] [rbp-2E8h]
  size_t na; // [rsp+88h] [rbp-2E8h]
  const void *nc; // [rsp+88h] [rbp-2E8h]
  int v384; // [rsp+90h] [rbp-2E0h]
  char *v385; // [rsp+90h] [rbp-2E0h]
  unsigned __int64 *v386; // [rsp+90h] [rbp-2E0h]
  void *v387; // [rsp+90h] [rbp-2E0h]
  char *v388; // [rsp+90h] [rbp-2E0h]
  char *v389; // [rsp+90h] [rbp-2E0h]
  int *v391; // [rsp+A0h] [rbp-2D0h]
  __int64 v392; // [rsp+A0h] [rbp-2D0h]
  __int64 v393; // [rsp+A0h] [rbp-2D0h]
  signed __int64 v394; // [rsp+A0h] [rbp-2D0h]
  char *v395; // [rsp+A8h] [rbp-2C8h]
  __int64 v396; // [rsp+B0h] [rbp-2C0h]
  int *v397; // [rsp+B0h] [rbp-2C0h]
  __int64 v398; // [rsp+B0h] [rbp-2C0h]
  __int64 v399; // [rsp+B0h] [rbp-2C0h]
  unsigned __int64 *v401; // [rsp+C0h] [rbp-2B0h]
  __int64 v402; // [rsp+C0h] [rbp-2B0h]
  __int64 v403; // [rsp+C0h] [rbp-2B0h]
  int v404; // [rsp+C0h] [rbp-2B0h]
  __int64 *v405; // [rsp+C0h] [rbp-2B0h]
  __int64 v406; // [rsp+C0h] [rbp-2B0h]
  __int64 v407; // [rsp+C8h] [rbp-2A8h]
  unsigned __int64 *v408; // [rsp+D0h] [rbp-2A0h]
  int v409; // [rsp+D8h] [rbp-298h]
  __int64 v410; // [rsp+D8h] [rbp-298h]
  __int64 v411; // [rsp+D8h] [rbp-298h]
  _QWORD *v412; // [rsp+D8h] [rbp-298h]
  __int64 v413; // [rsp+D8h] [rbp-298h]
  char *v414; // [rsp+D8h] [rbp-298h]
  __int64 v415; // [rsp+D8h] [rbp-298h]
  __int64 *v416; // [rsp+E0h] [rbp-290h]
  __int64 v417; // [rsp+E0h] [rbp-290h]
  unsigned __int64 v418; // [rsp+E0h] [rbp-290h]
  __int64 v419; // [rsp+E0h] [rbp-290h]
  unsigned __int64 *v420; // [rsp+E0h] [rbp-290h]
  char *v421; // [rsp+E0h] [rbp-290h]
  __int64 v422; // [rsp+E0h] [rbp-290h]
  __int64 v423; // [rsp+E0h] [rbp-290h]
  char *v424; // [rsp+E8h] [rbp-288h]
  __int64 v425; // [rsp+E8h] [rbp-288h]
  int v426; // [rsp+E8h] [rbp-288h]
  int *v427; // [rsp+F0h] [rbp-280h]
  int *v428; // [rsp+F0h] [rbp-280h]
  void *v429; // [rsp+F0h] [rbp-280h]
  char *v430; // [rsp+F0h] [rbp-280h]
  char *i; // [rsp+F0h] [rbp-280h]
  _BYTE *v432; // [rsp+F0h] [rbp-280h]
  char *v433; // [rsp+F0h] [rbp-280h]
  __int64 v434; // [rsp+F8h] [rbp-278h]
  unsigned __int64 *v435; // [rsp+F8h] [rbp-278h]
  __int64 v436; // [rsp+F8h] [rbp-278h]
  __int64 *v437; // [rsp+F8h] [rbp-278h]
  __int64 v438; // [rsp+F8h] [rbp-278h]
  __int64 v439; // [rsp+F8h] [rbp-278h]
  __int64 v440; // [rsp+F8h] [rbp-278h]
  __int64 v441; // [rsp+F8h] [rbp-278h]
  unsigned __int64 v442; // [rsp+F8h] [rbp-278h]
  signed __int64 v443; // [rsp+F8h] [rbp-278h]
  __int64 v444; // [rsp+100h] [rbp-270h] BYREF
  __int64 v445; // [rsp+108h] [rbp-268h] BYREF
  void *v446; // [rsp+110h] [rbp-260h] BYREF
  void *dest; // [rsp+118h] [rbp-258h]
  char *v448; // [rsp+120h] [rbp-250h]
  void *v449[2]; // [rsp+130h] [rbp-240h] BYREF
  _BYTE *v450; // [rsp+140h] [rbp-230h]
  void *src; // [rsp+150h] [rbp-220h] BYREF
  char *v452; // [rsp+158h] [rbp-218h]
  char *v453; // [rsp+160h] [rbp-210h]
  __m128i v454; // [rsp+170h] [rbp-200h] BYREF
  char *v455; // [rsp+180h] [rbp-1F0h] BYREF
  char *v456; // [rsp+188h] [rbp-1E8h]
  __m128 *v457; // [rsp+190h] [rbp-1E0h] BYREF
  unsigned __int64 v458; // [rsp+198h] [rbp-1D8h]
  char *v459; // [rsp+1A0h] [rbp-1D0h] BYREF
  char *v460; // [rsp+1A8h] [rbp-1C8h]
  __int64 v461; // [rsp+1B0h] [rbp-1C0h] BYREF
  int v462; // [rsp+1B8h] [rbp-1B8h] BYREF
  int *v463; // [rsp+1C0h] [rbp-1B0h]
  int *v464; // [rsp+1C8h] [rbp-1A8h]
  int *v465; // [rsp+1D0h] [rbp-1A0h]
  __int64 v466; // [rsp+1D8h] [rbp-198h]
  __int64 v467; // [rsp+1E0h] [rbp-190h] BYREF
  int v468; // [rsp+1E8h] [rbp-188h] BYREF
  unsigned __int64 *v469; // [rsp+1F0h] [rbp-180h]
  int *v470; // [rsp+1F8h] [rbp-178h]
  int *v471; // [rsp+200h] [rbp-170h]
  __int64 v472; // [rsp+208h] [rbp-168h]
  __m128 v473; // [rsp+210h] [rbp-160h] BYREF
  __m128i v474; // [rsp+220h] [rbp-150h] BYREF
  _BYTE *v475; // [rsp+260h] [rbp-110h] BYREF
  __int64 v476; // [rsp+268h] [rbp-108h]
  _BYTE v477[64]; // [rsp+270h] [rbp-100h] BYREF
  __m128 *v478; // [rsp+2B0h] [rbp-C0h] BYREF
  __int64 v479; // [rsp+2B8h] [rbp-B8h]
  _WORD v480[88]; // [rsp+2C0h] [rbp-B0h] BYREF

  v12 = *(_QWORD *)(a1 + 112);
  v396 = *(_QWORD *)(a1 + 24);
  v464 = &v462;
  v465 = &v462;
  v470 = &v468;
  v471 = &v468;
  v475 = v477;
  v446 = 0;
  dest = 0;
  v448 = 0;
  v462 = 0;
  v463 = 0;
  v466 = 0;
  v468 = 0;
  v469 = 0;
  v472 = 0;
  v476 = 0x800000000LL;
  v444 = v12;
  if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
  {
    sub_15E08E0(a1, a2);
    v13 = *(unsigned __int64 **)(a1 + 88);
    if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
      sub_15E08E0(a1, a2);
    v14 = *(unsigned __int64 **)(a1 + 88);
  }
  else
  {
    v13 = *(unsigned __int64 **)(a1 + 88);
    v14 = v13;
  }
  v409 = 0;
  v401 = &v14[5 * *(_QWORD *)(a1 + 96)];
  while ( v13 != v401 )
  {
    if ( sub_1833FA0(a3, (__int64)v13) )
    {
      v16 = dest;
      v17 = *(_QWORD *)(*v13 + 24);
      v18 = *(unsigned int *)(v17 + 12);
      v19 = *(const void **)(v17 + 16);
      v434 = v17;
      v20 = *(_DWORD *)(v17 + 12);
      v21 = 8 * v18;
      if ( !(8 * v18) )
      {
LABEL_8:
        v22 = (unsigned int)v476;
        v23 = v476;
        if ( v18 > HIDWORD(v476) - (unsigned __int64)(unsigned int)v476 )
        {
          sub_16CD150((__int64)&v475, v477, v18 + (unsigned int)v476, 8, v15, (int)v19);
          v22 = (unsigned int)v476;
          v23 = v476;
        }
        v24 = &v475[8 * v22];
        if ( v18 )
        {
          v25 = &v24[v18];
          do
          {
            if ( v24 )
              *v24 = 0;
            ++v24;
          }
          while ( v24 != v25 );
          v23 = v476;
        }
        LODWORD(v476) = v23 + v20;
        goto LABEL_17;
      }
      v424 = v448;
      if ( v21 <= (unsigned __int64)(v448 - (_BYTE *)dest) )
      {
        memmove(dest, v19, 8 * v18);
        dest = (char *)dest + v21;
        v18 = *(unsigned int *)(v434 + 12);
        v20 = *(_DWORD *)(v434 + 12);
        goto LABEL_8;
      }
      v286 = v446;
      v287 = v21 >> 3;
      v288 = (_BYTE *)dest - (_BYTE *)v446;
      v289 = ((_BYTE *)dest - (_BYTE *)v446) >> 3;
      if ( v21 >> 3 > 0xFFFFFFFFFFFFFFFLL - v289 )
        sub_4262D8((__int64)"vector::_M_range_insert");
      if ( v287 < v289 )
        v287 = ((_BYTE *)dest - (_BYTE *)v446) >> 3;
      v290 = __CFADD__(v287, v289);
      v291 = v287 + v289;
      if ( v290 )
      {
        v362 = 0x7FFFFFFFFFFFFFF8LL;
      }
      else
      {
        if ( !v291 )
        {
          v421 = (char *)(v21 + v288);
          if ( dest != v446 )
          {
            v429 = (void *)v19;
            memmove(0, v446, (_BYTE *)dest - (_BYTE *)v446);
            v322 = (void *)v288;
            v292 = 0;
            memcpy(v322, v429, v21);
            v293 = 0;
            goto LABEL_438;
          }
          v292 = 0;
          memcpy((void *)((_BYTE *)dest - (_BYTE *)v446), v19, v21);
          v293 = 0;
LABEL_384:
          if ( !v286 )
          {
LABEL_385:
            v448 = v292;
            v446 = v293;
            dest = v421;
            v18 = *(unsigned int *)(v434 + 12);
            v20 = *(_DWORD *)(v434 + 12);
            goto LABEL_8;
          }
LABEL_438:
          v430 = v293;
          j_j___libc_free_0(v286, v424 - v286);
          v293 = v430;
          goto LABEL_385;
        }
        if ( v291 > 0xFFFFFFFFFFFFFFFLL )
          v291 = 0xFFFFFFFFFFFFFFFLL;
        v362 = 8 * v291;
      }
      v387 = (void *)v19;
      v363 = (char *)sub_22077B0(v362);
      v286 = v446;
      v364 = v387;
      v292 = &v363[v362];
      v424 = v448;
      v394 = v21 + v16 - (_BYTE *)v446;
      v432 = dest;
      v395 = (char *)((_BYTE *)dest - v16);
      v421 = (char *)dest + v21 - (_QWORD)v446 + (_QWORD)v363;
      if ( v16 == v446 )
      {
        v365 = v387;
        v389 = v363;
        memcpy(&v363[v16 - (_BYTE *)v446], v365, v21);
        v293 = v389;
        if ( v16 == v432 )
          goto LABEL_384;
      }
      else
      {
        v388 = v363;
        v379 = &v363[v16 - (_BYTE *)v446];
        nc = v364;
        memmove(v363, v446, v16 - (_BYTE *)v446);
        memcpy(v379, nc, v21);
        v293 = v388;
        if ( v16 == v432 )
          goto LABEL_438;
      }
      v433 = v293;
      memcpy(&v293[v394], v16, (size_t)v395);
      v293 = v433;
      goto LABEL_384;
    }
    if ( !sub_1833FA0(a2, (__int64)v13) )
    {
      v236 = *v13;
      v237 = (char *)dest;
      v478 = (__m128 *)*v13;
      if ( dest == v448 )
      {
        sub_1278040((__int64)&v446, dest, &v478);
      }
      else
      {
        if ( dest )
        {
          *(_QWORD *)dest = v236;
          v237 = (char *)dest;
        }
        dest = v237 + 8;
      }
      v238 = sub_1560230(&v444, v409);
      v241 = (unsigned int)v476;
      if ( (unsigned int)v476 >= HIDWORD(v476) )
      {
        sub_16CD150((__int64)&v475, v477, 0, 8, v239, v240);
        v241 = (unsigned int)v476;
      }
      *(_QWORD *)&v475[8 * v241] = v238;
      LODWORD(v476) = v476 + 1;
      goto LABEL_17;
    }
    v205 = v13[1];
    if ( !v205 )
    {
      v206 = sub_1599EF0((__int64 **)*v13);
      sub_164D160(
        (__int64)v13,
        v206,
        a5,
        *(double *)a6.m128i_i64,
        *(double *)a7.m128i_i64,
        *(double *)a8.m128i_i64,
        v207,
        v208,
        a11,
        a12);
      goto LABEL_17;
    }
    v242 = v463;
    v473.m128_u64[0] = (unsigned __int64)v13;
    if ( v463 )
    {
      v243 = &v462;
      do
      {
        while ( 1 )
        {
          v244 = *((_QWORD *)v242 + 2);
          v245 = *((_QWORD *)v242 + 3);
          if ( *((_QWORD *)v242 + 4) >= (unsigned __int64)v13 )
            break;
          v242 = (int *)*((_QWORD *)v242 + 3);
          if ( !v245 )
            goto LABEL_292;
        }
        v243 = v242;
        v242 = (int *)*((_QWORD *)v242 + 2);
      }
      while ( v244 );
LABEL_292:
      v427 = v243;
      if ( v243 != &v462 && *((_QWORD *)v243 + 4) <= (unsigned __int64)v13 )
        goto LABEL_295;
    }
    else
    {
      v427 = &v462;
    }
    v478 = &v473;
    v427 = (int *)sub_18347A0(&v461, v427, (unsigned __int64 **)&v478);
    v205 = v13[1];
    if ( !v205 )
      goto LABEL_354;
LABEL_295:
    v420 = v13;
    v246 = v449;
    v247 = v205;
    do
    {
      v248 = sub_1648700(v247);
      v249 = v248;
      if ( *((_BYTE *)v248 + 16) == 54 )
        v250 = *v248;
      else
        v250 = v248[7];
      v442 = v250;
      v449[0] = 0;
      v48 = (__int64 *)v246;
      v449[1] = 0;
      v450 = 0;
      sub_9C9810((__int64)v246, (*((_DWORD *)v249 + 5) & 0xFFFFFFFu) - 1);
      if ( (*((_BYTE *)v249 + 23) & 0x40) != 0 )
      {
        v251 = (_QWORD *)*(v249 - 1);
        v252 = &v251[3 * (*((_DWORD *)v249 + 5) & 0xFFFFFFF)];
      }
      else
      {
        v252 = v249;
        v251 = &v249[-3 * (*((_DWORD *)v249 + 5) & 0xFFFFFFF)];
      }
      v253 = v251 + 3;
      v55 = v449[1];
      if ( v253 != v252 )
      {
        v254 = v246;
        v255 = v252;
        v256 = (__int64)v254;
        while ( 1 )
        {
          v52 = *(unsigned int *)(*v253 + 32LL);
          v258 = *(__int64 **)(*v253 + 24LL);
          if ( (unsigned int)v52 <= 0x40 )
          {
            v257 = (__int64)((_QWORD)v258 << (64 - (unsigned __int8)v52)) >> (64 - (unsigned __int8)v52);
            v478 = (__m128 *)v257;
            if ( v55 != v450 )
              goto LABEL_303;
LABEL_308:
            v48 = (__int64 *)v256;
            v253 += 3;
            sub_A235E0(v256, v55, &v478);
            v55 = v449[1];
            if ( v253 == v255 )
              goto LABEL_309;
          }
          else
          {
            v257 = *v258;
            v478 = (__m128 *)v257;
            if ( v55 == v450 )
              goto LABEL_308;
LABEL_303:
            if ( v55 )
            {
              *(_QWORD *)v55 = v257;
              v55 = v449[1];
            }
            v55 += 8;
            v253 += 3;
            v449[1] = v55;
            if ( v253 == v255 )
            {
LABEL_309:
              v246 = (void **)v256;
              break;
            }
          }
        }
      }
      v259 = v449[0];
      v260 = v55 - (char *)v449[0];
      if ( (_BYTE *)(v55 - (char *)v449[0]) != (_BYTE *)8 )
      {
        v455 = 0;
        v454 = (__m128i)v442;
        v456 = 0;
        if ( !v260 )
        {
          v261 = 0;
          v262 = 0;
          goto LABEL_313;
        }
        if ( v260 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_522:
          sub_4261EA(v48, v55, v52);
LABEL_374:
        v285 = sub_22077B0(v260);
        v55 = v449[1];
        v259 = v449[0];
        v262 = (char *)v285;
        v261 = (char *)v449[1] - (char *)v449[0];
LABEL_313:
        v454.m128i_i64[1] = (__int64)v262;
        v455 = v262;
        v456 = &v262[v260];
        if ( v259 != v55 )
          v262 = (char *)memmove(v262, v259, v261);
        goto LABEL_315;
      }
      if ( *(_QWORD *)v449[0] || v449[0] == v55 )
      {
        v260 = 8;
        v455 = 0;
        v454 = (__m128i)v442;
        v456 = 0;
        goto LABEL_374;
      }
      v261 = 0;
      v262 = 0;
      v449[1] = v449[0];
      v454 = (__m128i)v442;
      v456 = 0;
LABEL_315:
      v455 = &v262[v261];
      sub_18341E0((__int64)(v427 + 10), (unsigned __int64 *)&v454);
      v48 = (__int64 *)v454.m128i_i64[1];
      if ( v454.m128i_i64[1] )
        j_j___libc_free_0(v454.m128i_i64[1], &v456[-v454.m128i_i64[1]]);
      if ( *((_BYTE *)v249 + 16) != 54 )
      {
        v48 = (__int64 *)v249[1];
        v249 = sub_1648700((__int64)v48);
      }
      v55 = v449[0];
      v458 = 0;
      v459 = 0;
      v457 = (__m128 *)v420;
      v263 = v449[1];
      v460 = 0;
      v264 = (char *)v449[1] - (char *)v449[0];
      if ( v449[1] == v449[0] )
      {
        v443 = 0;
        v267 = 0;
        v266 = 0;
      }
      else
      {
        if ( v264 > 0x7FFFFFFFFFFFFFF8LL )
          goto LABEL_522;
        v265 = sub_22077B0((char *)v449[1] - (char *)v449[0]);
        v55 = v449[0];
        v266 = (char *)v265;
        v263 = v449[1];
        v443 = (char *)v449[1] - (char *)v449[0];
        v267 = (char *)v449[1] - (char *)v449[0];
      }
      v458 = (unsigned __int64)v266;
      v459 = v266;
      v460 = &v266[v264];
      if ( v55 != v263 )
        v266 = (char *)memmove(v266, v55, v267);
      v268 = v469;
      v269 = &v266[v267];
      v459 = &v266[v267];
      if ( !v469 )
      {
        v270 = (unsigned __int64 *)&v468;
        goto LABEL_347;
      }
      v270 = (unsigned __int64 *)&v468;
      do
      {
        if ( v268[4] < (unsigned __int64)v457 )
          goto LABEL_335;
        if ( (__m128 *)v268[4] == v457 )
        {
          v271 = (char *)v268[6];
          v272 = (char *)v268[5];
          if ( v271 - v272 > v443 )
            v271 = &v272[v267];
          v273 = v266;
          if ( v272 != v271 )
          {
            while ( *(_QWORD *)v272 >= *(_QWORD *)v273 )
            {
              if ( *(_QWORD *)v272 > *(_QWORD *)v273 )
                goto LABEL_366;
              v272 += 8;
              v273 += 8;
              if ( v271 == v272 )
                goto LABEL_365;
            }
LABEL_335:
            v268 = (unsigned __int64 *)v268[3];
            continue;
          }
LABEL_365:
          if ( v269 != v273 )
            goto LABEL_335;
        }
LABEL_366:
        v270 = v268;
        v268 = (unsigned __int64 *)v268[2];
      }
      while ( v268 );
      if ( v270 == (unsigned __int64 *)&v468 || (unsigned __int64)v457 < v270[4] )
        goto LABEL_347;
      if ( v457 == (__m128 *)v270[4] )
      {
        v274 = (_QWORD *)v270[5];
        v275 = v270[6] - (_QWORD)v274;
        if ( v275 < v443 )
          v269 = &v266[v275];
        if ( v266 == v269 )
        {
LABEL_386:
          if ( (_QWORD *)v270[6] != v274 )
            goto LABEL_347;
        }
        else
        {
          v276 = v266;
          while ( *(_QWORD *)v276 >= *v274 )
          {
            if ( *(_QWORD *)v276 > *v274 )
              goto LABEL_348;
            v276 += 8;
            ++v274;
            if ( v269 == v276 )
              goto LABEL_386;
          }
LABEL_347:
          v478 = (__m128 *)&v457;
          v277 = sub_1834D00(&v467, v270, (unsigned __int64 **)&v478);
          v266 = (char *)v458;
          v270 = v277;
          v264 = (unsigned __int64)&v460[-v458];
        }
      }
LABEL_348:
      v270[8] = (unsigned __int64)v249;
      if ( v266 )
        j_j___libc_free_0(v266, v264);
      if ( v449[0] )
        j_j___libc_free_0(v449[0], v450 - (char *)v449[0]);
      v247 = *(_QWORD *)(v247 + 8);
    }
    while ( v247 );
    v13 = v420;
LABEL_354:
    v278 = *((_QWORD *)v427 + 8);
    if ( v427 + 12 != (int *)v278 )
    {
      while ( 1 )
      {
        v281 = (__int64 **)*v13;
        if ( *(_BYTE *)(*v13 + 8) == 16 )
          v281 = (__int64 **)*v281[2];
        v282 = (unsigned __int64 *)sub_15FA110(
                                     (__int64)v281[3],
                                     *(_QWORD *)(v278 + 40),
                                     (__int64)(*(_QWORD *)(v278 + 48) - *(_QWORD *)(v278 + 40)) >> 3);
        v279 = (char *)dest;
        v478 = (__m128 *)v282;
        if ( dest != v448 )
          break;
        sub_1278040((__int64)&v446, dest, &v478);
        v280 = (unsigned int)v476;
        if ( (unsigned int)v476 >= HIDWORD(v476) )
          goto LABEL_364;
LABEL_359:
        *(_QWORD *)&v475[8 * v280] = 0;
        LODWORD(v476) = v476 + 1;
        v278 = sub_220EF30(v278);
        if ( v427 + 12 == (int *)v278 )
          goto LABEL_17;
      }
      if ( dest )
      {
        *(_QWORD *)dest = v282;
        v279 = (char *)dest;
      }
      v280 = (unsigned int)v476;
      dest = v279 + 8;
      if ( (unsigned int)v476 < HIDWORD(v476) )
        goto LABEL_359;
LABEL_364:
      sub_16CD150((__int64)&v475, v477, 0, 8, v283, v284);
      v280 = (unsigned int)v476;
      goto LABEL_359;
    }
LABEL_17:
    ++v409;
    v13 += 5;
  }
  v26 = sub_1644EA0(
          **(__int64 ***)(v396 + 16),
          v446,
          ((_BYTE *)dest - (_BYTE *)v446) >> 3,
          *(_DWORD *)(v396 + 8) >> 8 != 0);
  v27.m128_u64[0] = (unsigned __int64)sub_1649960(a1);
  v28 = *(_BYTE *)(a1 + 32);
  v478 = &v473;
  v473 = v27;
  v29 = v28 & 0xF;
  v480[0] = 261;
  v369 = sub_1648B60(120);
  if ( v369 )
    sub_15E2490(v369, v26, v29, (__int64)&v478, 0);
  sub_15E4330(v369, a1);
  v30 = sub_1626D20(a1);
  sub_1627150(v369, v30);
  sub_1627150(a1, 0);
  v31 = v475;
  v32 = (unsigned int)v476;
  v33 = sub_1560240(&v444);
  v34 = sub_1560250(&v444);
  v35 = (__int64 *)sub_15E0530(a1);
  v36 = sub_155FDB0(v35, v34, v33, v31, v32);
  LODWORD(v476) = 0;
  *(_QWORD *)(v369 + 112) = v36;
  sub_1631B60(*(_QWORD *)(a1 + 40) + 24LL, v369);
  v37 = (unsigned __int64 **)a1;
  v38 = *(_QWORD *)(a1 + 56);
  *(_QWORD *)(v369 + 64) = a1 + 56;
  v38 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(v369 + 56) = v38 | *(_QWORD *)(v369 + 56) & 7LL;
  *(_QWORD *)(v38 + 8) = v369 + 56;
  *(_QWORD *)(a1 + 56) = *(_QWORD *)(a1 + 56) & 7LL | (v369 + 56);
  sub_164B7C0(v369, a1);
  v39 = *(_QWORD *)(a1 + 8);
  v478 = (__m128 *)v480;
  v479 = 0x1000000000LL;
  if ( v39 )
  {
LABEL_21:
    v40 = (unsigned __int64)sub_1648700(v39);
    v43 = *(_BYTE *)(v40 + 16);
    if ( v43 <= 0x17u )
    {
      v368 = 0;
      v44 = 0;
      goto LABEL_24;
    }
    if ( v43 == 78 )
    {
      v232 = v40 | 4;
      v368 = (unsigned __int64 **)v232;
    }
    else
    {
      v368 = 0;
      v44 = 0;
      if ( v43 != 29 )
        goto LABEL_24;
      v232 = v40 & 0xFFFFFFFFFFFFFFFBLL;
      v368 = (unsigned __int64 **)v232;
    }
    v44 = v232 & 0xFFFFFFFFFFFFFFF8LL;
    v233 = *(_QWORD *)((v232 & 0xFFFFFFFFFFFFFFF8LL) + 56);
    v234 = (v232 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((v232 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
    v235 = (v232 >> 2) & 1;
    v416 = (__int64 *)v234;
    n = v235;
    if ( v235 )
    {
      v445 = v233;
      goto LABEL_25;
    }
LABEL_24:
    n = 0;
    v445 = *(_QWORD *)(v44 + 56);
    v416 = (__int64 *)(v44 - 24LL * (*(_DWORD *)(v44 + 20) & 0xFFFFFFF));
LABEL_25:
    if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
    {
      sub_15E08E0(a1, (__int64)v37);
      v435 = *(unsigned __int64 **)(a1 + 88);
      if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
        sub_15E08E0(a1, (__int64)v37);
      v45 = *(unsigned __int64 **)(a1 + 88);
    }
    else
    {
      v435 = *(unsigned __int64 **)(a1 + 88);
      v45 = v435;
    }
    v384 = 0;
    v375 = &v45[5 * *(_QWORD *)(a1 + 96)];
    if ( v435 != v375 )
    {
      v410 = v44;
      do
      {
        if ( sub_1833FA0(a2, (__int64)v435) || sub_1833FA0(a3, (__int64)v435) )
        {
          if ( !sub_1833FA0(a3, (__int64)v435) )
          {
            if ( !v435[1] )
              goto LABEL_29;
            v457 = (__m128 *)v435;
            v48 = (__int64 *)v435;
            v49 = v463;
            v50 = &v462;
            if ( !v463 )
              goto LABEL_41;
            do
            {
              while ( 1 )
              {
                v51 = *((_QWORD *)v49 + 2);
                v52 = *((_QWORD *)v49 + 3);
                if ( *((_QWORD *)v49 + 4) >= (unsigned __int64)v435 )
                  break;
                v49 = (int *)*((_QWORD *)v49 + 3);
                if ( !v52 )
                  goto LABEL_39;
              }
              v50 = v49;
              v49 = (int *)*((_QWORD *)v49 + 2);
            }
            while ( v51 );
LABEL_39:
            if ( v50 == &v462 || *((_QWORD *)v50 + 4) > (unsigned __int64)v435 )
            {
LABEL_41:
              v48 = &v461;
              v473.m128_u64[0] = (unsigned __int64)&v457;
              v50 = (int *)sub_18347A0(&v461, v50, (unsigned __int64 **)&v473);
            }
            src = 0;
            v452 = 0;
            v453 = 0;
            v53 = (int *)*((_QWORD *)v50 + 8);
            v397 = v50 + 12;
            if ( v50 + 12 == v53 )
              goto LABEL_29;
            while ( 2 )
            {
              v425 = *v416;
              v473 = (__m128)(unsigned __int64)v435;
              v54 = (_BYTE *)*((_QWORD *)v53 + 6);
              v55 = (_BYTE *)*((_QWORD *)v53 + 5);
              v474 = 0u;
              v56 = v54 - v55;
              if ( v54 == v55 )
              {
                v59 = 0;
                v60 = 0;
                v58 = 0;
              }
              else
              {
                if ( v56 > 0x7FFFFFFFFFFFFFF8LL )
                  goto LABEL_522;
                v57 = sub_22077B0(v56);
                v55 = (_BYTE *)*((_QWORD *)v53 + 5);
                v58 = (char *)v57;
                v54 = (_BYTE *)*((_QWORD *)v53 + 6);
                v59 = v54 - v55;
                v60 = v54 - v55;
              }
              v473.m128_u64[1] = (unsigned __int64)v58;
              v474.m128i_i64[0] = (__int64)v58;
              v474.m128i_i64[1] = (__int64)&v58[v56];
              if ( v54 != v55 )
                v58 = (char *)memmove(v58, v55, v60);
              v61 = v469;
              v62 = &v58[v60];
              v474.m128i_i64[0] = (__int64)&v58[v60];
              if ( !v469 )
              {
                v63 = (unsigned __int64 *)&v468;
                goto LABEL_71;
              }
              v63 = (unsigned __int64 *)&v468;
LABEL_50:
              if ( v61[4] >= v473.m128_u64[0] )
              {
                if ( v61[4] != v473.m128_u64[0] )
                  goto LABEL_85;
                v64 = (char *)v61[6];
                v65 = (char *)v61[5];
                if ( v64 - v65 > v59 )
                  v64 = &v65[v60];
                v66 = v58;
                if ( v65 == v64 )
                {
LABEL_84:
                  if ( v66 == v62 )
                  {
LABEL_85:
                    v63 = v61;
                    v61 = (unsigned __int64 *)v61[2];
LABEL_60:
                    if ( !v61 )
                    {
                      if ( v63 == (unsigned __int64 *)&v468 || v473.m128_u64[0] < v63[4] )
                        goto LABEL_71;
                      if ( v473.m128_u64[0] == v63[4] )
                      {
                        v67 = (_QWORD *)v63[5];
                        v68 = v63[6] - (_QWORD)v67;
                        if ( v68 < v59 )
                          v62 = &v58[v68];
                        if ( v58 == v62 )
                        {
LABEL_236:
                          if ( (_QWORD *)v63[6] != v67 )
                            goto LABEL_71;
                        }
                        else
                        {
                          v69 = v58;
                          while ( *(_QWORD *)v69 >= *v67 )
                          {
                            if ( *(_QWORD *)v69 > *v67 )
                              goto LABEL_72;
                            v69 += 8;
                            ++v67;
                            if ( v62 == v69 )
                              goto LABEL_236;
                          }
LABEL_71:
                          v457 = &v473;
                          v70 = sub_1834D00(&v467, v63, (unsigned __int64 **)&v457);
                          v58 = (char *)v473.m128_u64[1];
                          v63 = v70;
                          v56 = v474.m128i_i64[1] - v473.m128_u64[1];
                        }
                      }
LABEL_72:
                      v402 = v63[8];
                      if ( v58 )
                        j_j___libc_free_0(v58, v56);
                      v71 = (__int64 *)*((_QWORD *)v53 + 6);
                      v72 = (__int64 *)*((_QWORD *)v53 + 5);
                      if ( v71 != v72 )
                      {
                        v80 = (char *)v71 - (char *)v72;
                        if ( (unsigned __int64)((char *)v71 - (char *)v72) > 0x7FFFFFFFFFFFFFF8LL )
                          sub_4262D8((__int64)"vector::reserve");
                        if ( v453 - (_BYTE *)src >= v80 )
                          goto LABEL_92;
                        v81 = (char *)v71 - (char *)v72;
                        v82 = v452 - (_BYTE *)src;
                        v83 = (char *)sub_22077B0(v81);
                        v84 = src;
                        v85 = v83;
                        if ( v452 - (_BYTE *)src > 0 )
                        {
                          memmove(v83, src, v452 - (_BYTE *)src);
                          v204 = v453 - v84;
                        }
                        else
                        {
                          if ( !src )
                            goto LABEL_91;
                          v204 = v453 - (_BYTE *)src;
                        }
                        j_j___libc_free_0(v84, v204);
LABEL_91:
                        src = v85;
                        v452 = &v85[v82];
                        v453 = &v85[v80];
                        v72 = (__int64 *)*((_QWORD *)v53 + 5);
                        v71 = (__int64 *)*((_QWORD *)v53 + 6);
                        if ( v71 != v72 )
                        {
LABEL_92:
                          v391 = v53;
                          v86 = *(_QWORD *)v425;
                          do
                          {
                            while ( 1 )
                            {
                              v87 = *v72;
                              if ( *(_BYTE *)(v86 + 8) == 13 )
                              {
                                v106 = (_QWORD *)sub_15E0530(a1);
                                v89 = sub_1643350(v106);
                              }
                              else
                              {
                                v88 = (_QWORD *)sub_15E0530(a1);
                                v89 = sub_1643360(v88);
                              }
                              v90 = sub_15A0680(v89, v87, 0);
                              v91 = v452;
                              v473.m128_u64[0] = v90;
                              if ( v452 == v453 )
                              {
                                sub_12879C0((__int64)&src, v452, &v473);
                              }
                              else
                              {
                                if ( v452 )
                                {
                                  *(_QWORD *)v452 = v90;
                                  v91 = v452;
                                }
                                v452 = v91 + 8;
                              }
                              if ( *(_BYTE *)(v86 + 8) != 15 )
                                break;
                              ++v72;
                              v86 = *(_QWORD *)(v86 + 24);
                              if ( v71 == v72 )
                                goto LABEL_102;
                            }
                            ++v72;
                            v86 = sub_1643D80(v86, v87);
                          }
                          while ( v71 != v72 );
LABEL_102:
                          v53 = v391;
                        }
                        v92 = (unsigned __int64 *)sub_1649960(v425);
                        v93 = (__int64 *)src;
                        v474.m128i_i16[0] = 773;
                        v457 = (__m128 *)v92;
                        v458 = v94;
                        v473.m128_u64[0] = (unsigned __int64)&v457;
                        v473.m128_u64[1] = (unsigned __int64)".idx";
                        v95 = (v452 - (_BYTE *)src) >> 3;
                        v392 = *((_QWORD *)v53 + 4);
                        if ( !v392 )
                        {
                          v111 = *(_QWORD *)v425;
                          if ( *(_BYTE *)(*(_QWORD *)v425 + 8LL) == 16 )
                            v111 = **(_QWORD **)(v111 + 16);
                          v392 = *(_QWORD *)(v111 + 24);
                        }
                        v371 = (__int64 *)v452;
                        v96 = sub_1648A60(72, (int)v95 + 1);
                        v97 = (__int64)v96;
                        if ( v96 )
                        {
                          v377 = &v96[-3 * (unsigned int)(v95 + 1)];
                          v98 = *(_QWORD *)v425;
                          if ( *(_BYTE *)(*(_QWORD *)v425 + 8LL) == 16 )
                            v98 = **(_QWORD **)(v98 + 16);
                          v370 = v371;
                          v372 = *(_DWORD *)(v98 + 8) >> 8;
                          v99 = (__int64 *)sub_15F9F50(v392, (__int64)v93, v95);
                          v100 = (__int64 *)sub_1646BA0(v99, v372);
                          v101 = v95 + 1;
                          v102 = v100;
                          if ( *(_BYTE *)(*(_QWORD *)v425 + 8LL) == 16 )
                          {
                            v203 = sub_16463B0(v100, *(_QWORD *)(*(_QWORD *)v425 + 32LL));
                            v101 = v95 + 1;
                            v102 = v203;
                          }
                          else if ( v93 != v370 )
                          {
                            v103 = v93;
                            while ( 1 )
                            {
                              v104 = *(_QWORD *)*v103;
                              if ( *(_BYTE *)(v104 + 8) == 16 )
                                break;
                              if ( v370 == ++v103 )
                                goto LABEL_113;
                            }
                            v105 = sub_16463B0(v102, *(_QWORD *)(v104 + 32));
                            v101 = v95 + 1;
                            v102 = v105;
                          }
LABEL_113:
                          sub_15F1EA0(v97, (__int64)v102, 32, (__int64)v377, v101, v410);
                          *(_QWORD *)(v97 + 56) = v392;
                          *(_QWORD *)(v97 + 64) = sub_15F9F50(v392, (__int64)v93, v95);
                          sub_15F9CE0(v97, v425, v93, v95, (__int64)&v473);
                        }
                        v425 = v97;
                        if ( src != v452 )
                          v452 = (char *)src;
                      }
                      v457 = (__m128 *)sub_1649960(v425);
                      v458 = v73;
                      v474.m128i_i16[0] = 773;
                      v473.m128_u64[0] = (unsigned __int64)&v457;
                      v473.m128_u64[1] = (unsigned __int64)".val";
                      v74 = sub_1648A60(64, 1u);
                      v75 = (__int64)v74;
                      if ( v74 )
                        sub_15F90E0((__int64)v74, v425, (__int64)&v473, v410);
                      sub_15F8F50(v75, 1 << (*(unsigned __int16 *)(v402 + 18) >> 1) >> 1);
                      v473 = 0u;
                      v474.m128i_i64[0] = 0;
                      sub_14A8180(v402, (__int64 *)&v473, 0);
                      sub_1626170(v75, (__int64 *)&v473);
                      v78 = (unsigned int)v479;
                      if ( (unsigned int)v479 >= HIDWORD(v479) )
                      {
                        sub_16CD150((__int64)&v478, v480, 0, 8, v76, v77);
                        v78 = (unsigned int)v479;
                      }
                      v478->m128_u64[v78] = v75;
                      v79 = (unsigned int)v476;
                      LODWORD(v479) = v479 + 1;
                      if ( (unsigned int)v476 >= HIDWORD(v476) )
                      {
                        sub_16CD150((__int64)&v475, v477, 0, 8, v76, v77);
                        v79 = (unsigned int)v476;
                      }
                      v48 = (__int64 *)v53;
                      *(_QWORD *)&v475[8 * v79] = 0;
                      LODWORD(v476) = v476 + 1;
                      v53 = (int *)sub_220EF30(v53);
                      if ( v397 == v53 )
                      {
                        if ( src )
                          j_j___libc_free_0(src, v453 - (_BYTE *)src);
                        goto LABEL_29;
                      }
                      continue;
                    }
                    goto LABEL_50;
                  }
                }
                else
                {
                  while ( *(_QWORD *)v65 >= *(_QWORD *)v66 )
                  {
                    if ( *(_QWORD *)v65 > *(_QWORD *)v66 )
                      goto LABEL_85;
                    v65 += 8;
                    v66 += 8;
                    if ( v64 == v65 )
                      goto LABEL_84;
                  }
                }
              }
              break;
            }
            v61 = (unsigned __int64 *)v61[3];
            goto LABEL_60;
          }
          a5 = 0;
          v112 = *(_QWORD *)(*v435 + 24);
          *(_OWORD *)v449 = 0;
          v113 = (_QWORD *)sub_15E0530(a1);
          v114 = sub_1643350(v113);
          v449[0] = (void *)sub_159C470(v114, 0, 0);
          v393 = *(unsigned int *)(v112 + 12);
          v115 = 0;
          if ( *(_DWORD *)(v112 + 12) )
          {
            do
            {
              v132 = (_QWORD *)sub_15E0530(a1);
              v133 = sub_1643350(v132);
              v134 = (void *)sub_159C470(v133, v115, 0);
              LOWORD(v459) = 265;
              v449[1] = v134;
              LODWORD(v457) = v115;
              src = (void *)sub_1649960(*v416);
              LOWORD(v455) = 773;
              v452 = v135;
              v454.m128i_i64[0] = (__int64)&src;
              v454.m128i_i64[1] = (__int64)".";
              v117 = (char)v459;
              if ( (_BYTE)v459 )
              {
                if ( (_BYTE)v459 == 1 )
                {
                  a6 = _mm_load_si128(&v454);
                  v473 = (__m128)a6;
                  v474.m128i_i64[0] = (__int64)v455;
                }
                else
                {
                  v116 = (unsigned __int64 **)v457;
                  if ( BYTE1(v459) != 1 )
                  {
                    v116 = (unsigned __int64 **)&v457;
                    v117 = 2;
                  }
                  v473.m128_u64[1] = (unsigned __int64)v116;
                  v473.m128_u64[0] = (unsigned __int64)&v454;
                  v474.m128i_i8[0] = 2;
                  v474.m128i_i8[1] = v117;
                }
              }
              else
              {
                v474.m128i_i16[0] = 256;
              }
              v118 = *v416;
              v119 = sub_1648A60(72, 3u);
              if ( v119 )
              {
                v120 = *(_QWORD *)v118;
                if ( *(_BYTE *)(*(_QWORD *)v118 + 8LL) == 16 )
                  v120 = **(_QWORD **)(v120 + 16);
                v426 = *(_DWORD *)(v120 + 8) >> 8;
                v121 = (__int64 *)sub_15F9F50(v112, (__int64)v449, 2);
                v122 = sub_1646BA0(v121, v426);
                v123 = (__int64)(v119 - 9);
                v124 = (__int64 *)v122;
                v125 = *(_QWORD *)v118;
                if ( *(_BYTE *)(*(_QWORD *)v118 + 8LL) == 16
                  || (v125 = *(_QWORD *)v449[0], *(_BYTE *)(*(_QWORD *)v449[0] + 8LL) == 16)
                  || (v125 = *(_QWORD *)v449[1], *(_BYTE *)(*(_QWORD *)v449[1] + 8LL) == 16) )
                {
                  v136 = sub_16463B0(v124, *(_QWORD *)(v125 + 32));
                  v123 = (__int64)(v119 - 9);
                  v124 = v136;
                }
                sub_15F1EA0((__int64)v119, (__int64)v124, 32, v123, 3, v410);
                v119[7] = v112;
                v119[8] = sub_15F9F50(v112, (__int64)v449, 2);
                sub_15F9CE0((__int64)v119, v118, (__int64 *)v449, 2, (__int64)&v473);
              }
              v126 = (unsigned __int64 *)sub_1649960((__int64)v119);
              v474.m128i_i16[0] = 773;
              v457 = (__m128 *)v126;
              v458 = v127;
              v473.m128_u64[0] = (unsigned __int64)&v457;
              v473.m128_u64[1] = (unsigned __int64)".val";
              v128 = sub_1648A60(64, 1u);
              v129 = (unsigned __int64)v128;
              if ( v128 )
                sub_15F90E0((__int64)v128, (__int64)v119, (__int64)&v473, v410);
              v130 = (unsigned int)v479;
              if ( (unsigned int)v479 >= HIDWORD(v479) )
              {
                sub_16CD150((__int64)&v478, v480, 0, 8, v41, v42);
                v130 = (unsigned int)v479;
              }
              v478->m128_u64[v130] = v129;
              v131 = (unsigned int)v476;
              LODWORD(v479) = v479 + 1;
              if ( (unsigned int)v476 >= HIDWORD(v476) )
              {
                sub_16CD150((__int64)&v475, v477, 0, 8, v41, v42);
                v131 = (unsigned int)v476;
              }
              ++v115;
              *(_QWORD *)&v475[8 * v131] = 0;
              LODWORD(v476) = v476 + 1;
            }
            while ( v393 != v115 );
          }
        }
        else
        {
          v107 = *v416;
          v108 = (unsigned int)v479;
          if ( (unsigned int)v479 >= HIDWORD(v479) )
          {
            sub_16CD150((__int64)&v478, v480, 0, 8, v46, v47);
            v108 = (unsigned int)v479;
          }
          v478->m128_u64[v108] = v107;
          LODWORD(v479) = v479 + 1;
          v109 = sub_1560230(&v445, v384);
          v110 = (unsigned int)v476;
          if ( (unsigned int)v476 >= HIDWORD(v476) )
          {
            sub_16CD150((__int64)&v475, v477, 0, 8, v41, v42);
            v110 = (unsigned int)v476;
          }
          *(_QWORD *)&v475[8 * v110] = v109;
          LODWORD(v476) = v476 + 1;
        }
LABEL_29:
        v435 += 5;
        v416 += 3;
        ++v384;
      }
      while ( v435 != v375 );
      v44 = v410;
    }
    v137 = v416;
    v138 = v384;
    v139 = v44;
    while ( 1 )
    {
      v151 = *(_BYTE *)(v139 + 23);
      if ( n )
        break;
      if ( v151 >= 0 )
        goto LABEL_235;
      v152 = sub_1648A40(v139);
      v154 = v152 + v153;
      if ( *(char *)(v139 + 23) >= 0 )
      {
        if ( (unsigned int)(v154 >> 4) )
LABEL_529:
          BUG();
LABEL_235:
        v146 = -72;
        goto LABEL_158;
      }
      if ( !(unsigned int)((v154 - sub_1648A40(v139)) >> 4) )
        goto LABEL_235;
      if ( *(char *)(v139 + 23) >= 0 )
        goto LABEL_529;
      v155 = *(_DWORD *)(sub_1648A40(v139) + 8);
      if ( *(char *)(v139 + 23) >= 0 )
LABEL_527:
        BUG();
      v156 = sub_1648A40(v139);
      if ( v137 == (__int64 *)(v139 + -72 - 24LL * (unsigned int)(*(_DWORD *)(v156 + v157 - 4) - v155)) )
      {
LABEL_171:
        v158 = v139;
        v473.m128_u64[0] = (unsigned __int64)&v474;
        v473.m128_u64[1] = 0x100000000LL;
        if ( n )
        {
          if ( *(char *)(v139 + 23) < 0 )
          {
            v159 = sub_1648A40(v139);
            v161 = v159 + v160;
            if ( *(char *)(v139 + 23) >= 0 )
              v162 = v161 >> 4;
            else
              LODWORD(v162) = (v161 - sub_1648A40(v139)) >> 4;
            if ( (_DWORD)v162 )
            {
              v436 = 16LL * (unsigned int)v162;
              v163 = 0;
              do
              {
                v164 = 0;
                if ( *(char *)(v139 + 23) < 0 )
                  v164 = sub_1648A40(v139);
                v165 = (char **)(v163 + v164);
                v163 += 16;
                v166 = *((unsigned int *)v165 + 2);
                v167 = *v165;
                v168 = *((unsigned int *)v165 + 3);
                v169 = 3 * v166;
                LODWORD(v166) = *(_DWORD *)(v139 + 20);
                v459 = v167;
                v169 *= 8;
                v457 = (__m128 *)(v139 + v169 - 24 * (v166 & 0xFFFFFFF));
                v458 = 0xAAAAAAAAAAAAAAABLL * ((24 * v168 - v169) >> 3);
                sub_1740580((__int64)&v473, (__int64)&v457);
              }
              while ( v436 != v163 );
              v158 = v139;
            }
          }
        }
        else if ( *(char *)(v139 + 23) < 0 )
        {
          v209 = sub_1648A40(v139);
          v211 = v209 + v210;
          if ( *(char *)(v139 + 23) >= 0 )
            v212 = v211 >> 4;
          else
            LODWORD(v212) = (v211 - sub_1648A40(v139)) >> 4;
          if ( (_DWORD)v212 )
          {
            v213 = 0;
            v441 = 16LL * (unsigned int)v212;
            do
            {
              v214 = 0;
              if ( *(char *)(v139 + 23) < 0 )
                v214 = sub_1648A40(v139);
              v215 = (char **)(v213 + v214);
              v213 += 16;
              v216 = *((unsigned int *)v215 + 2);
              v217 = *v215;
              v218 = *((unsigned int *)v215 + 3);
              v219 = 3 * v216;
              LODWORD(v216) = *(_DWORD *)(v139 + 20);
              v459 = v217;
              v219 *= 8;
              v457 = (__m128 *)(v139 + v219 - 24 * (v216 & 0xFFFFFFF));
              v458 = 0xAAAAAAAAAAAAAAABLL * ((24 * v218 - v219) >> 3);
              sub_1740580((__int64)&v473, (__int64)&v457);
            }
            while ( v441 != v213 );
            v158 = v139;
          }
        }
        v170 = (__int64 *)v473.m128_u64[0];
        v437 = (__int64 *)v478;
        v171 = (__int64 *)(v473.m128_u64[0] + 56LL * v473.m128_u32[2]);
        if ( *(_BYTE *)(v158 + 16) != 29 )
        {
          LOWORD(v459) = 257;
          v413 = *(_QWORD *)(*(_QWORD *)v369 + 24LL);
          if ( (__int64 *)v473.m128_u64[0] == v171 )
          {
            v231 = v479 + 1;
            v406 = v473.m128_u32[2];
            v419 = (unsigned int)v479;
            v323 = sub_1648AB0(72, (int)v479 + 1, 16 * v473.m128_i32[2]);
            v230 = v419;
            v225 = (__int64)v323;
            if ( v323 )
            {
              v226 = v170;
              v227 = v406;
              goto LABEL_267;
            }
          }
          else
          {
            v222 = v473.m128_u64[0];
            v223 = 0;
            do
            {
              v224 = *(_QWORD *)(v222 + 40) - *(_QWORD *)(v222 + 32);
              v222 += 56LL;
              v223 += v224 >> 3;
            }
            while ( (__int64 *)v222 != v171 );
            v399 = v473.m128_u32[2];
            v419 = (unsigned int)v479;
            v404 = v479 + 1;
            v225 = (__int64)sub_1648AB0(72, v223 + (int)v479 + 1, 16 * v473.m128_i32[2]);
            if ( v225 )
            {
              v226 = v170;
              v227 = v399;
              LODWORD(v228) = 0;
              do
              {
                v229 = v170[5] - v170[4];
                v170 += 7;
                v228 = (unsigned int)(v229 >> 3) + (unsigned int)v228;
              }
              while ( v171 != v170 );
              v230 = v228 + v419;
              v231 = v228 + v404;
LABEL_267:
              v405 = v226;
              v407 = v227;
              sub_15F1EA0(v225, **(_QWORD **)(v413 + 16), 54, v225 - 24 * v230 - 24, v231, v158);
              *(_QWORD *)(v225 + 56) = 0;
              sub_15F5B40(v225, v413, v369, v437, v419, (__int64)&v457, v405, v407);
            }
          }
          *(_WORD *)(v225 + 18) = *(_WORD *)(v158 + 18) & 3 | *(_WORD *)(v225 + 18) & 0xFFFC;
          v179 = v225 | 4;
          goto LABEL_188;
        }
        LOWORD(v459) = 257;
        v411 = *(_QWORD *)(v158 - 24);
        v417 = *(_QWORD *)(v158 - 48);
        v172 = *(_QWORD *)(*(_QWORD *)v369 + 24LL);
        if ( (__int64 *)v473.m128_u64[0] == v171 )
        {
          v174 = 0;
        }
        else
        {
          v173 = v473.m128_u64[0];
          v174 = 0;
          do
          {
            v175 = *(_QWORD *)(v173 + 40) - *(_QWORD *)(v173 + 32);
            v173 += 56LL;
            v174 += v175 >> 3;
          }
          while ( (__int64 *)v173 != v171 );
        }
        v176 = v174 + v479 + 3;
        v398 = v473.m128_u32[2];
        v403 = (unsigned int)v479;
        v177 = (__int64)sub_1648AB0(72, v176, 16 * v473.m128_i32[2]);
        if ( v177 )
        {
          v178 = v437;
          v438 = v177;
          v367 = v178;
          sub_15F1EA0(v177, **(_QWORD **)(v172 + 16), 5, v177 - 24LL * v176, v176, v158);
          *(_QWORD *)(v438 + 56) = 0;
          sub_15F6500(v438, v172, v369, v417, v411, (__int64)&v457, v367, v403, v170, v398);
          v177 = v438;
        }
        v179 = v177 & 0xFFFFFFFFFFFFFFFBLL;
LABEL_188:
        v180 = v179 & 0xFFFFFFFFFFFFFFF8LL;
        v412 = v475;
        v418 = (unsigned int)v476;
        *(_WORD *)((v179 & 0xFFFFFFFFFFFFFFF8LL) + 18) = *(_WORD *)((v179 & 0xFFFFFFFFFFFFFFF8LL) + 18) & 0x8000
                                                       | *(_WORD *)((v179 & 0xFFFFFFFFFFFFFFF8LL) + 18) & 3
                                                       | (4 * ((*(_WORD *)(v158 + 18) >> 2) & 0xDFFF));
        v439 = sub_1560240(&v445);
        v181 = sub_1560250(&v445);
        v182 = (__int64 *)sub_15E0530(a1);
        v183 = v181;
        v184 = (unsigned __int64 **)((v179 & 0xFFFFFFFFFFFFFFF8LL) + 48);
        *(_QWORD *)((v179 & 0xFFFFFFFFFFFFFFF8LL) + 56) = sub_155FDB0(v182, v183, v439, v412, v418);
        v185 = *(unsigned __int64 **)(v158 + 48);
        v457 = (__m128 *)v185;
        if ( !v185 )
        {
          if ( v184 == (unsigned __int64 **)&v457 )
            goto LABEL_192;
          v220 = *(_QWORD *)(v180 + 48);
          if ( !v220 )
            goto LABEL_192;
LABEL_255:
          sub_161E7C0(v180 + 48, v220);
LABEL_256:
          v221 = (unsigned __int8 *)v457;
          *(_QWORD *)(v180 + 48) = v457;
          if ( !v221 )
            goto LABEL_192;
          sub_1623210((__int64)&v457, v221, v180 + 48);
          v37 = (unsigned __int64 **)&v457;
          if ( !(unsigned __int8)sub_1625980(v158, &v457) )
            goto LABEL_193;
LABEL_258:
          v37 = (unsigned __int64 **)v457;
          sub_15F3B70(v179 & 0xFFFFFFFFFFFFFFF8LL, (int)v457);
          goto LABEL_193;
        }
        sub_1623A60((__int64)&v457, (__int64)v185, 2);
        if ( v184 != (unsigned __int64 **)&v457 )
        {
          v220 = *(_QWORD *)(v180 + 48);
          if ( v220 )
            goto LABEL_255;
          goto LABEL_256;
        }
        if ( v457 )
          sub_161E7C0((__int64)&v457, (__int64)v457);
LABEL_192:
        v37 = (unsigned __int64 **)&v457;
        if ( (unsigned __int8)sub_1625980(v158, &v457) )
          goto LABEL_258;
LABEL_193:
        LODWORD(v479) = 0;
        LODWORD(v476) = 0;
        if ( *(_BYTE *)(a4 + 16) )
        {
          v37 = v368;
          (*(void (__fastcall **)(_QWORD, unsigned __int64 **, unsigned __int64))a4)(*(_QWORD *)(a4 + 8), v368, v179);
        }
        if ( *(_QWORD *)(v158 + 8) )
        {
          sub_164D160(
            v158,
            v179 & 0xFFFFFFFFFFFFFFF8LL,
            a5,
            *(double *)a6.m128i_i64,
            *(double *)a7.m128i_i64,
            *(double *)a8.m128i_i64,
            v186,
            v187,
            a11,
            a12);
          v37 = (unsigned __int64 **)v158;
          sub_164B7C0(v179 & 0xFFFFFFFFFFFFFFF8LL, v158);
        }
        sub_15F20C0((_QWORD *)v158);
        v188 = (__m128i *)v473.m128_u64[0];
        v189 = (__m128i *)(v473.m128_u64[0] + 56LL * v473.m128_u32[2]);
        if ( (__m128i *)v473.m128_u64[0] != v189 )
        {
          do
          {
            v190 = v189[-2].m128i_i64[1];
            v189 = (__m128i *)((char *)v189 - 56);
            if ( v190 )
            {
              v37 = (unsigned __int64 **)(v189[3].m128i_i64[0] - v190);
              j_j___libc_free_0(v190, v37);
            }
            if ( (__m128i *)v189->m128i_i64[0] != &v189[1] )
            {
              v37 = (unsigned __int64 **)(v189[1].m128i_i64[0] + 1);
              j_j___libc_free_0(v189->m128i_i64[0], v37);
            }
          }
          while ( v188 != v189 );
          v189 = (__m128i *)v473.m128_u64[0];
        }
        if ( v189 != &v474 )
          _libc_free((unsigned __int64)v189);
        v39 = *(_QWORD *)(a1 + 8);
        if ( !v39 )
          goto LABEL_207;
        goto LABEL_21;
      }
LABEL_159:
      v147 = *v137;
      v148 = (unsigned int)v479;
      if ( (unsigned int)v479 >= HIDWORD(v479) )
      {
        sub_16CD150((__int64)&v478, v480, 0, 8, v41, v42);
        v148 = (unsigned int)v479;
      }
      v478->m128_u64[v148] = v147;
      LODWORD(v479) = v479 + 1;
      v149 = sub_1560230(&v445, v138);
      v150 = (unsigned int)v476;
      if ( (unsigned int)v476 >= HIDWORD(v476) )
      {
        sub_16CD150((__int64)&v475, v477, 0, 8, v41, v42);
        v150 = (unsigned int)v476;
      }
      v137 += 3;
      ++v138;
      *(_QWORD *)&v475[8 * v150] = v149;
      LODWORD(v476) = v476 + 1;
    }
    if ( v151 < 0 )
    {
      v140 = sub_1648A40(v139);
      v142 = v140 + v141;
      if ( *(char *)(v139 + 23) >= 0 )
      {
        if ( (unsigned int)(v142 >> 4) )
          goto LABEL_529;
      }
      else if ( (unsigned int)((v142 - sub_1648A40(v139)) >> 4) )
      {
        if ( *(char *)(v139 + 23) >= 0 )
          goto LABEL_529;
        v143 = *(_DWORD *)(sub_1648A40(v139) + 8);
        if ( *(char *)(v139 + 23) >= 0 )
          goto LABEL_527;
        v144 = sub_1648A40(v139);
        v146 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v144 + v145 - 4) - v143);
        goto LABEL_158;
      }
    }
    v146 = -24;
LABEL_158:
    if ( v137 == (__int64 *)(v139 + v146) )
      goto LABEL_171;
    goto LABEL_159;
  }
LABEL_207:
  v378 = (_DWORD *)sub_1632FA0(*(_QWORD *)(a1 + 40));
  v191 = (__int64 *)(a1 + 72);
  if ( v191 != (__int64 *)(*v191 & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v192 = *(unsigned __int64 **)(v369 + 80);
    v193 = *(__int64 **)(a1 + 80);
    if ( v191 != (__int64 *)v192 )
    {
      if ( (__int64 *)(v369 + 72) != v191 )
      {
        v37 = (unsigned __int64 **)(a1 + 72);
        sub_15809C0(v369 + 72, (__int64)v191, *(_QWORD *)(a1 + 80), (__int64)v191);
      }
      if ( v191 != (__int64 *)v192 && v191 != v193 )
      {
        v194 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)((*v193 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v191;
        *(_QWORD *)(a1 + 72) = *(_QWORD *)(a1 + 72) & 7LL | *v193 & 0xFFFFFFFFFFFFFFF8LL;
        v195 = *v192;
        *(_QWORD *)(v194 + 8) = v192;
        v195 &= 0xFFFFFFFFFFFFFFF8LL;
        *v193 = v195 | *v193 & 7;
        *(_QWORD *)(v195 + 8) = v193;
        *v192 = v194 | *v192 & 7;
      }
    }
  }
  if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
  {
    sub_15E08E0(a1, (__int64)v37);
    v196 = *(unsigned __int64 **)(a1 + 88);
    if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
      sub_15E08E0(a1, (__int64)v37);
    v197 = *(unsigned __int64 **)(a1 + 88);
  }
  else
  {
    v196 = *(unsigned __int64 **)(a1 + 88);
    v197 = v196;
  }
  v408 = &v197[5 * *(_QWORD *)(a1 + 96)];
  if ( (*(_BYTE *)(v369 + 18) & 1) != 0 )
    sub_15E08E0(v369, (__int64)v37);
  v440 = *(_QWORD *)(v369 + 88);
  if ( v408 != v196 )
  {
    v198 = v196;
    while ( 1 )
    {
      if ( !sub_1833FA0(a2, (__int64)v198) && !sub_1833FA0(a3, (__int64)v198) )
      {
        sub_164D160(
          (__int64)v198,
          v440,
          a5,
          *(double *)a6.m128i_i64,
          *(double *)a7.m128i_i64,
          *(double *)a8.m128i_i64,
          v199,
          v200,
          a11,
          a12);
        sub_164B7C0(v440, (__int64)v198);
        v440 += 40;
        goto LABEL_224;
      }
      if ( sub_1833FA0(a3, (__int64)v198) )
      {
        v324 = *(_QWORD *)(v369 + 80);
        if ( !v324 )
          BUG();
        v325 = *(_QWORD *)(v324 + 24);
        v326 = v378[1];
        if ( v325 )
          v325 -= 24;
        v423 = *(_QWORD *)(*v198 + 24);
        v327 = sub_15E0370((__int64)v198);
        v474.m128i_i16[0] = 257;
        v328 = v327;
        v329 = sub_1648A60(64, 1u);
        v330 = v329;
        if ( v329 )
          sub_15F8A50((__int64)v329, (_QWORD *)v423, v326, 0, v328, (__int64)&v473, v325);
        a5 = 0;
        *(_OWORD *)v449 = 0;
        v331 = (_QWORD *)sub_15E0530(a1);
        v332 = sub_1643350(v331);
        v449[0] = (void *)sub_159C470(v332, 0, 0);
        if ( *(_DWORD *)(v423 + 12) )
        {
          v415 = *(unsigned int *)(v423 + 12);
          v386 = v198;
          v335 = 0;
          for ( i = (char *)v440; ; i += 40 )
          {
            v336 = (_QWORD *)sub_15E0530(a1);
            v337 = sub_1643350(v336);
            v338 = (void *)sub_159C470(v337, v335, 0);
            LODWORD(v457) = v335;
            v449[1] = v338;
            LOWORD(v459) = 265;
            src = (void *)sub_1649960((__int64)v330);
            v452 = v339;
            v454.m128i_i64[0] = (__int64)&src;
            LOWORD(v455) = 773;
            v454.m128i_i64[1] = (__int64)".";
            v340 = (char)v459;
            if ( (_BYTE)v459 )
            {
              if ( (_BYTE)v459 == 1 )
              {
                a7 = _mm_load_si128(&v454);
                v473 = (__m128)a7;
                v474.m128i_i64[0] = (__int64)v455;
              }
              else
              {
                v341 = (unsigned __int64 **)v457;
                if ( BYTE1(v459) != 1 )
                {
                  v341 = (unsigned __int64 **)&v457;
                  v340 = 2;
                }
                v473.m128_u64[1] = (unsigned __int64)v341;
                v474.m128i_i8[0] = 2;
                v473.m128_u64[0] = (unsigned __int64)&v454;
                v474.m128i_i8[1] = v340;
              }
            }
            else
            {
              v474.m128i_i16[0] = 256;
            }
            v342 = sub_1648A60(72, 3u);
            v343 = (__int64)v342;
            if ( v342 )
            {
              na = (size_t)(v342 - 9);
              v344 = *v330;
              if ( *(_BYTE *)(*v330 + 8LL) == 16 )
                v344 = **(_QWORD **)(v344 + 16);
              v374 = *(_DWORD *)(v344 + 8) >> 8;
              v345 = (__int64 *)sub_15F9F50(v423, (__int64)v449, 2);
              v346 = (__int64 *)sub_1646BA0(v345, v374);
              v347 = *v330;
              if ( *(_BYTE *)(*v330 + 8LL) == 16
                || (v347 = *(_QWORD *)v449[0], *(_BYTE *)(*(_QWORD *)v449[0] + 8LL) == 16)
                || (v347 = *(_QWORD *)v449[1], *(_BYTE *)(*(_QWORD *)v449[1] + 8LL) == 16) )
              {
                v346 = sub_16463B0(v346, *(_QWORD *)(v347 + 32));
              }
              sub_15F1EA0(v343, (__int64)v346, 32, na, 3, v325);
              *(_QWORD *)(v343 + 56) = v423;
              *(_QWORD *)(v343 + 64) = sub_15F9F50(v423, (__int64)v449, 2);
              sub_15F9CE0(v343, (__int64)v330, (__int64 *)v449, 2, (__int64)&v473);
            }
            LODWORD(v457) = v335;
            LOWORD(v459) = 265;
            src = (void *)sub_1649960((__int64)v386);
            v452 = v348;
            v454.m128i_i64[0] = (__int64)&src;
            LOWORD(v455) = 773;
            v454.m128i_i64[1] = (__int64)".";
            v349 = (char)v459;
            if ( (_BYTE)v459 )
            {
              if ( (_BYTE)v459 == 1 )
              {
                a8 = _mm_load_si128(&v454);
                v473 = (__m128)a8;
                v474.m128i_i64[0] = (__int64)v455;
              }
              else
              {
                v350 = (unsigned __int64 **)v457;
                if ( BYTE1(v459) != 1 )
                {
                  v350 = (unsigned __int64 **)&v457;
                  v349 = 2;
                }
                v473.m128_u64[1] = (unsigned __int64)v350;
                v474.m128i_i8[0] = 2;
                v473.m128_u64[0] = (unsigned __int64)&v454;
                v474.m128i_i8[1] = v349;
              }
            }
            else
            {
              v474.m128i_i16[0] = 256;
            }
            sub_164B780((__int64)i, (__int64 *)&v473);
            v351 = sub_1648A60(64, 2u);
            if ( v351 )
              sub_15F9660((__int64)v351, (__int64)i, v343, v325);
            if ( v415 == ++v335 )
              break;
          }
          v198 = v386;
          v440 += 40 * v415;
        }
        sub_164D160(
          (__int64)v198,
          (__int64)v330,
          (__m128)0LL,
          *(double *)a6.m128i_i64,
          *(double *)a7.m128i_i64,
          *(double *)a8.m128i_i64,
          v333,
          v334,
          a11,
          a12);
        sub_164B7C0((__int64)v330, (__int64)v198);
        for ( j = v330[1]; j; j = *(_QWORD *)(j + 8) )
        {
          v353 = sub_1648700(j);
          if ( *((_BYTE *)v353 + 16) == 78 )
            *((_WORD *)v353 + 9) &= 0xFFFCu;
        }
        goto LABEL_224;
      }
      v201 = v198[1];
      if ( v201 )
        break;
LABEL_224:
      v198 += 5;
      if ( v198 == v408 )
        goto LABEL_225;
    }
    v294 = v463;
    v457 = (__m128 *)v198;
    if ( v463 )
    {
      v428 = &v462;
      do
      {
        while ( 1 )
        {
          v295 = *((_QWORD *)v294 + 2);
          v296 = *((_QWORD *)v294 + 3);
          if ( *((_QWORD *)v294 + 4) >= (unsigned __int64)v198 )
            break;
          v294 = (int *)*((_QWORD *)v294 + 3);
          if ( !v296 )
            goto LABEL_393;
        }
        v428 = v294;
        v294 = (int *)*((_QWORD *)v294 + 2);
      }
      while ( v295 );
LABEL_393:
      if ( v428 != &v462 && *((_QWORD *)v428 + 4) <= (unsigned __int64)v198 )
      {
LABEL_395:
        v422 = (__int64)v198;
        while ( 1 )
        {
          while ( 1 )
          {
            v297 = sub_1648700(v201);
            v298 = (__int64)v297;
            if ( *((_BYTE *)v297 + 16) != 54 )
              break;
            v457 = (__m128 *)sub_1649960(v422);
            v458 = v359;
            v473.m128_u64[0] = (unsigned __int64)&v457;
            v474.m128i_i16[0] = 773;
            v473.m128_u64[1] = (unsigned __int64)".val";
            sub_164B780(v440, (__int64 *)&v473);
            sub_164D160(
              v298,
              v440,
              a5,
              *(double *)a6.m128i_i64,
              *(double *)a7.m128i_i64,
              *(double *)a8.m128i_i64,
              v360,
              v361,
              a11,
              a12);
            sub_15F20C0((_QWORD *)v298);
            v358 = (unsigned __int64 *)v422;
            v201 = *(_QWORD *)(v422 + 8);
            if ( !v201 )
              goto LABEL_500;
          }
          src = 0;
          v452 = 0;
          v453 = 0;
          sub_9C9810((__int64)&src, (*((_DWORD *)v297 + 5) & 0xFFFFFFFu) - 1);
          v299 = v452;
          v300 = v298 + 24 * (1LL - (*(_DWORD *)(v298 + 20) & 0xFFFFFFF));
          if ( v298 != v300 )
          {
            while ( 1 )
            {
              v302 = *(_DWORD *)(*(_QWORD *)v300 + 32LL);
              v303 = *(__int64 **)(*(_QWORD *)v300 + 24LL);
              if ( v302 <= 0x40 )
              {
                v301 = (__int64)((_QWORD)v303 << (64 - (unsigned __int8)v302)) >> (64 - (unsigned __int8)v302);
                v473.m128_u64[0] = v301;
                if ( v299 != v453 )
                  goto LABEL_400;
LABEL_405:
                v300 += 24;
                sub_A235E0((__int64)&src, v299, &v473);
                v299 = v452;
                if ( v300 == v298 )
                  break;
              }
              else
              {
                v301 = *v303;
                v473.m128_u64[0] = v301;
                if ( v299 == v453 )
                  goto LABEL_405;
LABEL_400:
                if ( v299 )
                {
                  *(_QWORD *)v299 = v301;
                  v299 = v452;
                }
                v299 += 8;
                v300 += 24;
                v452 = v299;
                if ( v300 == v298 )
                  break;
              }
            }
          }
          v304 = src;
          v305 = v299 - (_BYTE *)src;
          if ( v299 - (_BYTE *)src == 8 && !*(_QWORD *)src && src != v299 )
          {
            v452 = (char *)src;
            v305 = 0;
          }
          v306 = v440;
          for ( k = *((_QWORD *)v428 + 8); ; k = sub_220EF30(k) )
          {
            v308 = *(const void **)(k + 40);
            if ( v305 == *(_QWORD *)(k + 48) - (_QWORD)v308 && (!v305 || !memcmp(v308, v304, v305)) )
              break;
            v306 += 40;
          }
          v310 = (char *)sub_1649960(v422);
          if ( v310 )
          {
            v454.m128i_i64[0] = (__int64)&v455;
            sub_1832480(v454.m128i_i64, v310, (__int64)&v310[v309]);
          }
          else
          {
            LOBYTE(v455) = 0;
            v454 = (__m128i)(unsigned __int64)&v455;
          }
          v311 = (v452 - (_BYTE *)src) >> 3;
          if ( (_DWORD)v311 )
            break;
LABEL_492:
          if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v454.m128i_i64[1]) <= 3 )
            sub_4262D8((__int64)"basic_string::append");
          sub_2241490(&v454, ".val", 4);
          v474.m128i_i16[0] = 260;
          v473.m128_u64[0] = (unsigned __int64)&v454;
          sub_164B780(v306, (__int64 *)&v473);
          for ( m = *(_QWORD *)(v298 + 8); m; m = *(_QWORD *)(v298 + 8) )
          {
            v355 = sub_1648700(m);
            sub_164D160(
              (__int64)v355,
              v306,
              a5,
              *(double *)a6.m128i_i64,
              *(double *)a7.m128i_i64,
              *(double *)a8.m128i_i64,
              v356,
              v357,
              a11,
              a12);
            sub_15F20C0(v355);
          }
          sub_15F20C0((_QWORD *)v298);
          if ( (char **)v454.m128i_i64[0] != &v455 )
            j_j___libc_free_0(v454.m128i_i64[0], v455 + 1);
          if ( src )
            j_j___libc_free_0(src, v453 - (_BYTE *)src);
          v358 = (unsigned __int64 *)v422;
          v201 = *(_QWORD *)(v422 + 8);
          if ( !v201 )
          {
LABEL_500:
            v198 = v358;
            goto LABEL_436;
          }
        }
        v414 = 0;
        v385 = (char *)(8LL * (unsigned int)(v311 - 1));
        v312 = *(_QWORD *)src;
        if ( !*(_QWORD *)src )
        {
LABEL_416:
          v474.m128i_i8[4] = 48;
          v313 = &v474.m128i_i8[4];
          v457 = (__m128 *)&v459;
LABEL_417:
          v314 = 1;
          LOBYTE(v459) = *v313;
          v315 = (void **)&v459;
          goto LABEL_418;
        }
        while ( 1 )
        {
          v313 = &v474.m128i_i8[5];
          do
          {
            *--v313 = v312 % 0xA + 48;
            v318 = v312;
            v312 /= 0xAu;
          }
          while ( v318 > 9 );
          v457 = (__m128 *)&v459;
          v314 = &v474.m128i_u8[5] - (unsigned __int8 *)v313;
          v449[0] = (void *)(&v474.m128i_u8[5] - (unsigned __int8 *)v313);
          if ( (unsigned __int64)(&v474.m128i_u8[5] - (unsigned __int8 *)v313) > 0xF )
          {
            v373 = v313;
            nb = (char *)(&v474.m128i_u8[5] - (unsigned __int8 *)v313);
            v319 = (unsigned __int64 *)sub_22409D0(&v457, v449, 0);
            v314 = (size_t)nb;
            v313 = v373;
            v457 = (__m128 *)v319;
            v320 = (void **)v319;
            v459 = (char *)v449[0];
          }
          else
          {
            if ( v314 == 1 )
              goto LABEL_417;
            if ( !v314 )
            {
              v315 = (void **)&v459;
              goto LABEL_418;
            }
            v320 = (void **)&v459;
          }
          memcpy(v320, v313, v314);
          v314 = (size_t)v449[0];
          v315 = (void **)v457;
LABEL_418:
          v458 = v314;
          *((_BYTE *)v315 + v314) = 0;
          v316 = (__m128i *)sub_2241130(&v457, 0, 0, ".", 1);
          v473.m128_u64[0] = (unsigned __int64)&v474;
          if ( (__m128i *)v316->m128i_i64[0] == &v316[1] )
          {
            v474 = _mm_loadu_si128(v316 + 1);
          }
          else
          {
            v473.m128_u64[0] = v316->m128i_i64[0];
            v474.m128i_i64[0] = v316[1].m128i_i64[0];
          }
          v473.m128_u64[1] = v316->m128i_u64[1];
          v316->m128i_i64[0] = (__int64)v316[1].m128i_i64;
          v316->m128i_i64[1] = 0;
          v316[1].m128i_i8[0] = 0;
          sub_2241490(&v454, (const char *)v473.m128_u64[0], v473.m128_u64[1]);
          if ( (__m128i *)v473.m128_u64[0] != &v474 )
            j_j___libc_free_0(v473.m128_u64[0], v474.m128i_i64[0] + 1);
          if ( v457 != (__m128 *)&v459 )
            j_j___libc_free_0(v457, v459 + 1);
          v317 = v414;
          if ( v414 == v385 )
            goto LABEL_492;
          v414 += 8;
          v312 = *(_QWORD *)&v317[(_QWORD)src + 8];
          if ( !v312 )
            goto LABEL_416;
        }
      }
    }
    else
    {
      v428 = &v462;
    }
    v473.m128_u64[0] = (unsigned __int64)&v457;
    v321 = sub_18347A0(&v461, v428, (unsigned __int64 **)&v473);
    v201 = v198[1];
    v428 = (int *)v321;
    if ( !v201 )
    {
LABEL_436:
      v440 += 40LL * *((_QWORD *)v428 + 10);
      goto LABEL_224;
    }
    goto LABEL_395;
  }
LABEL_225:
  if ( v478 != (__m128 *)v480 )
    _libc_free((unsigned __int64)v478);
  if ( v475 != v477 )
    _libc_free((unsigned __int64)v475);
  sub_1832BC0(v469);
  sub_1832B30(v463);
  if ( v446 )
    j_j___libc_free_0(v446, v448 - (_BYTE *)v446);
  return v369;
}
