// Function: sub_18C04B0
// Address: 0x18c04b0
//
__int64 __fastcall sub_18C04B0(
        __int64 a1,
        __m128 a2,
        __m128 si128,
        __m128i a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v9; // r12
  char *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // rbx
  char *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v19; // r13
  char *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  double v23; // xmm4_8
  double v24; // xmm5_8
  _QWORD *v25; // r12
  char *v26; // rax
  char *v27; // r13
  char *v28; // rcx
  __int64 v29; // r15
  void *v30; // r14
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  char *v33; // rcx
  __int8 *v34; // r8
  size_t v35; // rbx
  void *v36; // rax
  __m128i *v37; // rdx
  char *v38; // r11
  char *v39; // rbx
  __m128i *v40; // r8
  char *v41; // r14
  char *v42; // r15
  void *v43; // r13
  unsigned __int64 v44; // r12
  size_t v45; // rdx
  signed __int64 v46; // rax
  char *v47; // r11
  char *v48; // r9
  signed __int64 v49; // rcx
  unsigned __int64 v50; // r14
  __int64 v51; // r15
  unsigned __int64 v52; // rbx
  size_t v53; // rdx
  signed __int64 v54; // rax
  char *v55; // rax
  __int64 v56; // rsi
  __int64 v57; // rcx
  __int64 v58; // rbx
  __int64 *v59; // r13
  __int64 v60; // r12
  char *v61; // rsi
  __int64 *v62; // rcx
  __int64 *v63; // r8
  char *v64; // rbx
  __int64 v65; // rdi
  char *v66; // r9
  char *v67; // rax
  __int64 *v68; // rdx
  _QWORD *v69; // rdx
  __int64 v70; // rax
  __int64 *v71; // rax
  int v72; // eax
  __int64 v73; // rax
  double v74; // xmm4_8
  double v75; // xmm5_8
  _BYTE *v76; // rax
  double v77; // xmm4_8
  double v78; // xmm5_8
  __m128i *v79; // rax
  __m128i *v80; // rdi
  _QWORD **v82; // rbx
  __int64 *v83; // rax
  __int64 v84; // r9
  __int64 v85; // r10
  __int64 v86; // rax
  __int64 v87; // rbx
  __int64 v88; // r12
  __int64 v89; // r9
  void *v90; // r15
  __int64 v91; // rax
  __int64 v92; // r15
  double v93; // xmm4_8
  double v94; // xmm5_8
  __int64 v95; // rbx
  _QWORD *v96; // rax
  _QWORD *v97; // r12
  __int64 v98; // rax
  __int64 v99; // rax
  __int64 v100; // r13
  int v101; // edi
  unsigned int v102; // r14d
  __int64 *v103; // rdx
  unsigned int v104; // r9d
  __int64 *v105; // rax
  __int64 v106; // r8
  __int64 **v107; // r15
  __int64 **v108; // r13
  _QWORD *v109; // rdi
  double v110; // xmm4_8
  double v111; // xmm5_8
  __int64 v112; // rdx
  _BYTE **v113; // rax
  __int64 v114; // r15
  _BYTE *v115; // r12
  __int64 v116; // r10
  unsigned int v117; // r11d
  __int64 *v118; // rbx
  _BYTE *v119; // rdi
  __int64 *i; // r12
  __int64 v121; // r13
  char *v122; // rax
  __int64 v123; // rdx
  char *v124; // r13
  char *v125; // rdi
  double v126; // xmm4_8
  double v127; // xmm5_8
  __int64 *v128; // rbx
  __int64 *v129; // r12
  __int64 *v130; // rsi
  _QWORD *v131; // rbx
  __int64 v132; // r12
  __int64 *v133; // rbx
  __int64 *v134; // r12
  __int64 v135; // rdi
  __int64 v136; // rdi
  __int64 v137; // rdi
  __int64 v138; // rdi
  __int64 *v139; // rax
  __int64 *v140; // rbx
  int v141; // esi
  unsigned int v142; // eax
  __int64 *v143; // r15
  _BYTE *v144; // r11
  __int64 v145; // r12
  char *v146; // r15
  __int64 v147; // rax
  __int64 v148; // rdi
  __int64 v149; // rax
  const char *v150; // rax
  __int64 v151; // rdx
  __m128 *v152; // rsi
  __int64 *v153; // rdx
  int v154; // ecx
  int v155; // eax
  __int64 v156; // rax
  __int64 v157; // rax
  __int64 v158; // rsi
  _QWORD *v159; // r13
  __int64 v160; // r9
  __int64 v161; // r12
  unsigned __int64 v162; // rdx
  const char *v163; // r14
  size_t v164; // rbx
  void *v165; // rax
  __m128i *v166; // rdx
  __int64 v167; // r15
  const void *v168; // rsi
  __int64 *v169; // r14
  __int64 *v170; // rbx
  void *v171; // r13
  unsigned __int64 v172; // r12
  size_t v173; // rdx
  signed __int64 v174; // rax
  __int64 *v175; // rbx
  void *v176; // rdx
  __int8 *v177; // r8
  size_t v178; // r14
  __m128i *v179; // rax
  __int64 v180; // r15
  _QWORD *v181; // r8
  __int64 v182; // r14
  const void *v183; // r10
  void *v184; // r12
  unsigned __int64 v185; // rbx
  size_t v186; // rdx
  signed __int64 v187; // rax
  void *v188; // rcx
  unsigned __int64 v189; // r15
  size_t v190; // rdx
  signed __int64 v191; // rax
  unsigned __int64 *v192; // rsi
  unsigned __int64 v193; // rcx
  unsigned __int64 *v194; // rdi
  unsigned __int64 *v195; // rdx
  unsigned __int64 v196; // r8
  __m128 *v197; // rax
  __int64 *v198; // rsi
  void *v199; // r14
  __int64 v200; // rax
  __int64 *v201; // rdx
  __int64 *v202; // r13
  __int64 *v203; // r9
  __int64 v204; // rdi
  __int64 v205; // rdi
  __int64 *v206; // rdi
  __m128i *v207; // r15
  int *v208; // r14
  size_t v209; // rdx
  size_t v210; // r13
  __int64 *v211; // r14
  __int64 *v212; // r13
  __int64 v213; // rdi
  __int64 v214; // r13
  _BYTE *v215; // rsi
  __int64 v216; // r14
  _QWORD *v217; // rdi
  _QWORD *v218; // r12
  __int64 v219; // r8
  _QWORD *v220; // r15
  __int64 v221; // rbx
  __int64 v222; // r12
  __int64 v223; // rdi
  __int64 v224; // r8
  _QWORD *v225; // r15
  __int64 v226; // rbx
  __int64 v227; // r12
  __int64 v228; // rdi
  __int64 v229; // rdi
  __int64 v230; // rdi
  _QWORD *v231; // r12
  __int64 v232; // r8
  _QWORD *v233; // r15
  __int64 v234; // rbx
  __int64 v235; // r12
  __int64 v236; // rdi
  __int64 v237; // r8
  _QWORD *v238; // r15
  __int64 v239; // rbx
  __int64 v240; // r12
  __int64 v241; // rdi
  __int64 v242; // rdi
  __int64 v243; // rdi
  int *v244; // rbx
  size_t v245; // rdx
  size_t v246; // r14
  int v247; // esi
  __int64 v248; // rax
  int v249; // r9d
  unsigned int v250; // edx
  __int64 *v251; // rbx
  __int64 v252; // r8
  __int64 v253; // rax
  unsigned __int64 v254; // r14
  __int64 v255; // rax
  unsigned __int64 v256; // r13
  __int64 v257; // rax
  __int64 v258; // rax
  _BYTE **v259; // rax
  __int64 v260; // rax
  unsigned int v261; // esi
  _QWORD *v262; // rax
  __int64 v263; // r12
  void *v264; // r14
  unsigned int v265; // r9d
  _QWORD *v266; // rbx
  __int64 v267; // rcx
  unsigned __int64 v268; // rax
  void **v269; // rdx
  __int64 v270; // rbx
  void **v271; // r15
  void *v272; // rax
  __int64 v273; // r13
  void *v274; // r15
  __int64 v275; // rdi
  _QWORD *v276; // rbx
  __int64 v277; // rcx
  unsigned __int64 v278; // rax
  void **v279; // rcx
  void **v280; // rbx
  _BYTE *v281; // rsi
  void *v282; // rax
  __int64 v283; // rax
  int v284; // edx
  _QWORD *v285; // rax
  int v286; // eax
  __int64 *v287; // r12
  unsigned int v288; // edx
  _QWORD *v289; // rax
  __int64 v290; // r8
  unsigned __int64 v291; // rdx
  void **v292; // r13
  __int64 v293; // rax
  _QWORD *v294; // rcx
  __int64 v295; // r14
  _QWORD *v296; // r15
  char *v297; // r10
  char *v298; // rsi
  _QWORD *v299; // r11
  signed __int64 v300; // rdi
  char *v301; // r8
  char *v302; // rax
  char *v303; // rdx
  _QWORD *v304; // rax
  signed __int64 v305; // rdx
  __int64 *v306; // r12
  unsigned int v307; // edx
  _QWORD *v308; // rax
  __int64 v309; // rbx
  unsigned __int64 v310; // rdx
  void **v311; // r13
  __int64 v312; // rax
  _QWORD *v313; // rcx
  __int64 v314; // r11
  _QWORD *v315; // r15
  char *v316; // r10
  char *v317; // rsi
  _QWORD *v318; // r14
  signed __int64 v319; // rdi
  char *v320; // r8
  char *v321; // rax
  char *v322; // rdx
  _QWORD *v323; // rax
  signed __int64 v324; // rdx
  _BYTE *v325; // rsi
  __int64 v326; // rax
  _QWORD *v327; // r13
  _QWORD *v328; // rbx
  __int64 v329; // r12
  unsigned __int64 *v330; // r12
  int v331; // edi
  _QWORD *v332; // rcx
  int v333; // edx
  __int64 v334; // rdx
  int v335; // r9d
  unsigned int v336; // esi
  __int64 v337; // rbx
  _QWORD *v338; // rcx
  int v339; // edi
  unsigned int v340; // ecx
  __int64 v341; // r8
  int v342; // edi
  _QWORD *v343; // rsi
  int v344; // ecx
  _QWORD *v345; // rdx
  unsigned int v346; // r14d
  __int64 v347; // rsi
  unsigned int v348; // edx
  __int64 v349; // r10
  int v350; // esi
  _QWORD *v351; // rcx
  int v352; // edi
  _QWORD *v353; // rcx
  int v354; // r9d
  __int64 v355; // rdx
  unsigned int v356; // ecx
  _QWORD *v357; // rdx
  __int64 v358; // rbx
  int v359; // r11d
  unsigned int v360; // edx
  __int64 v361; // rbx
  int v362; // esi
  _QWORD *v363; // rcx
  int v364; // edx
  _QWORD *v365; // rax
  int v366; // eax
  int v367; // ecx
  _QWORD *v368; // rdx
  unsigned int v369; // r13d
  __int64 v370; // rsi
  unsigned int v371; // edx
  __int64 v372; // r8
  int v373; // esi
  _QWORD *v374; // rcx
  int v375; // edx
  __int64 v376; // r9
  double v377; // xmm4_8
  double v378; // xmm5_8
  __m128i *v379; // rdi
  __int64 *v380; // rcx
  int v381; // edi
  size_t v382; // r13
  size_t v383; // rdx
  int v384; // eax
  unsigned int v385; // edi
  __m128i *v386; // rdi
  __m128i *v387; // rax
  int v388; // edx
  unsigned int v389; // esi
  __int64 v390; // r10
  __int64 *v391; // r14
  __m128i *v392; // rsi
  __int64 v393; // rax
  __int64 v394; // r15
  _QWORD *v395; // rax
  _QWORD *v396; // rax
  int v397; // r8d
  __int64 *v398; // rdi
  unsigned int v399; // esi
  int v400; // r8d
  __int64 v401; // r10
  __int64 *v402; // [rsp+0h] [rbp-3C0h]
  __int64 *v403; // [rsp+0h] [rbp-3C0h]
  __int64 *v404; // [rsp+0h] [rbp-3C0h]
  char *v405; // [rsp+8h] [rbp-3B8h]
  __m128i *v406; // [rsp+10h] [rbp-3B0h]
  __int64 v407; // [rsp+10h] [rbp-3B0h]
  __int64 v408; // [rsp+10h] [rbp-3B0h]
  char *v409; // [rsp+18h] [rbp-3A8h]
  __int64 *v410; // [rsp+18h] [rbp-3A8h]
  char *v411; // [rsp+18h] [rbp-3A8h]
  _QWORD ***v412; // [rsp+20h] [rbp-3A0h]
  char *v413; // [rsp+20h] [rbp-3A0h]
  __int64 v414; // [rsp+20h] [rbp-3A0h]
  _QWORD **v415; // [rsp+20h] [rbp-3A0h]
  __int64 v416; // [rsp+20h] [rbp-3A0h]
  __int64 *v417; // [rsp+20h] [rbp-3A0h]
  char *v418; // [rsp+28h] [rbp-398h]
  signed __int64 v419; // [rsp+28h] [rbp-398h]
  __int64 v420; // [rsp+28h] [rbp-398h]
  __int64 v421; // [rsp+28h] [rbp-398h]
  _QWORD *v422; // [rsp+28h] [rbp-398h]
  __m128 *v423; // [rsp+28h] [rbp-398h]
  __int64 *v424; // [rsp+28h] [rbp-398h]
  __int64 *v425; // [rsp+28h] [rbp-398h]
  __int64 *v426; // [rsp+30h] [rbp-390h]
  const void *v427; // [rsp+30h] [rbp-390h]
  __m128i *v428; // [rsp+30h] [rbp-390h]
  __int64 v429; // [rsp+40h] [rbp-380h]
  char *v430; // [rsp+40h] [rbp-380h]
  __int64 *v431; // [rsp+40h] [rbp-380h]
  __int64 *v432; // [rsp+40h] [rbp-380h]
  char v433; // [rsp+50h] [rbp-370h]
  __int64 *v434; // [rsp+50h] [rbp-370h]
  __int64 v435; // [rsp+50h] [rbp-370h]
  __int64 v436; // [rsp+50h] [rbp-370h]
  __int64 v437; // [rsp+50h] [rbp-370h]
  __int64 v438; // [rsp+60h] [rbp-360h]
  char *v439; // [rsp+60h] [rbp-360h]
  char *v440; // [rsp+60h] [rbp-360h]
  __int64 v441; // [rsp+60h] [rbp-360h]
  _QWORD *v442; // [rsp+60h] [rbp-360h]
  __int64 *v443; // [rsp+60h] [rbp-360h]
  __int64 *v444; // [rsp+60h] [rbp-360h]
  __int64 v445; // [rsp+60h] [rbp-360h]
  char *src; // [rsp+68h] [rbp-358h]
  void *srca; // [rsp+68h] [rbp-358h]
  __m128i *srcj; // [rsp+68h] [rbp-358h]
  char *srcb; // [rsp+68h] [rbp-358h]
  __int8 *srcl; // [rsp+68h] [rbp-358h]
  char *srcm; // [rsp+68h] [rbp-358h]
  char *srck; // [rsp+68h] [rbp-358h]
  void *srcn; // [rsp+68h] [rbp-358h]
  _QWORD *srco; // [rsp+68h] [rbp-358h]
  char *srcc; // [rsp+68h] [rbp-358h]
  void *srcd; // [rsp+68h] [rbp-358h]
  void *srce; // [rsp+68h] [rbp-358h]
  __int64 *srcf; // [rsp+68h] [rbp-358h]
  __int64 *srcg; // [rsp+68h] [rbp-358h]
  _QWORD *srch; // [rsp+68h] [rbp-358h]
  void *srcp; // [rsp+68h] [rbp-358h]
  __int8 *srcq; // [rsp+68h] [rbp-358h]
  __int64 *srci; // [rsp+68h] [rbp-358h]
  __int64 v464; // [rsp+70h] [rbp-350h]
  _BYTE **v465; // [rsp+70h] [rbp-350h]
  __int64 *v467; // [rsp+80h] [rbp-340h] BYREF
  __int64 *v468; // [rsp+88h] [rbp-338h]
  __int64 v469; // [rsp+90h] [rbp-330h]
  __m128i v470; // [rsp+A0h] [rbp-320h] BYREF
  __int64 v471; // [rsp+B0h] [rbp-310h]
  unsigned int v472; // [rsp+B8h] [rbp-308h]
  __int64 **v473; // [rsp+C0h] [rbp-300h] BYREF
  __int64 v474; // [rsp+C8h] [rbp-2F8h] BYREF
  __int64 *v475; // [rsp+D0h] [rbp-2F0h] BYREF
  __int64 *v476; // [rsp+D8h] [rbp-2E8h]
  __int64 *v477; // [rsp+E0h] [rbp-2E0h]
  __int64 v478; // [rsp+E8h] [rbp-2D8h]
  __m128i v479; // [rsp+F0h] [rbp-2D0h] BYREF
  __int64 v480; // [rsp+100h] [rbp-2C0h]
  __int64 *v481; // [rsp+110h] [rbp-2B0h]
  __int64 v482; // [rsp+118h] [rbp-2A8h]
  __int64 v483; // [rsp+120h] [rbp-2A0h] BYREF
  __m128i v484; // [rsp+130h] [rbp-290h] BYREF
  __int64 v485; // [rsp+140h] [rbp-280h]
  __m128i v486; // [rsp+150h] [rbp-270h] BYREF
  __int64 v487; // [rsp+160h] [rbp-260h] BYREF
  __int64 v488; // [rsp+168h] [rbp-258h]
  __int64 v489[2]; // [rsp+170h] [rbp-250h] BYREF
  char v490; // [rsp+180h] [rbp-240h] BYREF
  __m128i v491; // [rsp+190h] [rbp-230h]
  __int64 v492; // [rsp+1A0h] [rbp-220h]
  void *s2[2]; // [rsp+1B0h] [rbp-210h] BYREF
  __m128i dest[4]; // [rsp+1C0h] [rbp-200h] BYREF
  char *v495; // [rsp+208h] [rbp-1B8h]
  unsigned int v496; // [rsp+210h] [rbp-1B0h]
  char v497; // [rsp+218h] [rbp-1A8h] BYREF

  v9 = *(_QWORD *)a1;
  v10 = sub_15E0FD0(208);
  v12 = sub_16321A0(v9, (__int64)v10, v11);
  v13 = *(_QWORD *)a1;
  v14 = v12;
  v15 = sub_15E0FD0(207);
  v17 = sub_16321A0(v13, (__int64)v15, v16);
  v18 = *(_QWORD *)a1;
  v19 = v17;
  v20 = sub_15E0FD0(4);
  v22 = sub_16321A0(v18, (__int64)v20, v21);
  if ( *(_QWORD *)(a1 + 24) )
  {
    if ( !v14 || !v22 )
    {
      if ( v19 )
        goto LABEL_129;
      goto LABEL_5;
    }
  }
  else if ( !v14 || !*(_QWORD *)(v14 + 8) || !v22 || !*(_QWORD *)(v22 + 8) )
  {
    if ( !v19 || !*(_QWORD *)(v19 + 8) )
      return 0;
    if ( !v14 || !v22 )
      goto LABEL_129;
  }
  v486 = 0u;
  v487 = 0;
  v488 = 0;
  v95 = *(_QWORD *)(v14 + 8);
  if ( !v95 )
    goto LABEL_115;
  v408 = v19;
  do
  {
    v96 = sub_1648700(v95);
    v95 = *(_QWORD *)(v95 + 8);
    v97 = v96;
    if ( *((_BYTE *)v96 + 16) != 78 )
      continue;
    s2[0] = dest;
    s2[1] = (void *)0x100000000LL;
    v473 = &v475;
    v474 = 0x100000000LL;
    sub_14A87C0((__int64)s2, (__int64)&v473, (__int64)v96);
    if ( (_DWORD)v474 )
    {
      v98 = *((_DWORD *)v97 + 5) & 0xFFFFFFF;
      v464 = *(_QWORD *)(v97[3 * (1 - v98)] + 24LL);
      v99 = sub_1649C60(v97[-3 * v98]);
      v100 = v99;
      if ( (_DWORD)v488 )
      {
        v101 = 1;
        v102 = ((unsigned int)v99 >> 9) ^ ((unsigned int)v99 >> 4);
        v103 = 0;
        v104 = (v488 - 1) & v102;
        v105 = (__int64 *)(v486.m128i_i64[1] + 8LL * v104);
        v106 = *v105;
        if ( v100 == *v105 )
        {
LABEL_105:
          v107 = v473;
          v108 = &v473[(unsigned int)v474];
          if ( v473 != v108 )
          {
            do
            {
              v109 = *v107++;
              sub_15F20C0(v109);
            }
            while ( v108 != v107 );
          }
          goto LABEL_107;
        }
        while ( v106 != -8 )
        {
          if ( v106 == -16 && !v103 )
            v103 = v105;
          v104 = (v488 - 1) & (v101 + v104);
          v105 = (__int64 *)(v486.m128i_i64[1] + 8LL * v104);
          v106 = *v105;
          if ( v100 == *v105 )
            goto LABEL_105;
          ++v101;
        }
        if ( v103 )
          v105 = v103;
        ++v486.m128i_i64[0];
        v388 = v487 + 1;
        if ( 4 * ((int)v487 + 1) < (unsigned int)(3 * v488) )
        {
          if ( (int)v488 - HIDWORD(v487) - v388 > (unsigned int)v488 >> 3 )
            goto LABEL_712;
          sub_1353F00((__int64)&v486, v488);
          if ( !(_DWORD)v488 )
          {
LABEL_750:
            LODWORD(v487) = v487 + 1;
            BUG();
          }
          v398 = 0;
          v399 = (v488 - 1) & v102;
          v400 = 1;
          v105 = (__int64 *)(v486.m128i_i64[1] + 8LL * v399);
          v388 = v487 + 1;
          v401 = *v105;
          if ( v100 == *v105 )
            goto LABEL_712;
          while ( v401 != -8 )
          {
            if ( v401 == -16 && !v398 )
              v398 = v105;
            v399 = (v488 - 1) & (v400 + v399);
            v105 = (__int64 *)(v486.m128i_i64[1] + 8LL * v399);
            v401 = *v105;
            if ( v100 == *v105 )
              goto LABEL_712;
            ++v400;
          }
          goto LABEL_724;
        }
      }
      else
      {
        ++v486.m128i_i64[0];
      }
      sub_1353F00((__int64)&v486, 2 * v488);
      if ( !(_DWORD)v488 )
        goto LABEL_750;
      v388 = v487 + 1;
      v389 = (v488 - 1) & (((unsigned int)v100 >> 9) ^ ((unsigned int)v100 >> 4));
      v105 = (__int64 *)(v486.m128i_i64[1] + 8LL * v389);
      v390 = *v105;
      if ( v100 == *v105 )
        goto LABEL_712;
      v397 = 1;
      v398 = 0;
      while ( v390 != -8 )
      {
        if ( !v398 && v390 == -16 )
          v398 = v105;
        v389 = (v488 - 1) & (v397 + v389);
        v105 = (__int64 *)(v486.m128i_i64[1] + 8LL * v389);
        v390 = *v105;
        if ( v100 == *v105 )
          goto LABEL_712;
        ++v397;
      }
LABEL_724:
      if ( v398 )
        v105 = v398;
LABEL_712:
      LODWORD(v487) = v388;
      if ( *v105 != -8 )
        --HIDWORD(v487);
      *v105 = v100;
      v391 = (__int64 *)s2[0];
      srci = (__int64 *)((char *)s2[0] + 16 * LODWORD(s2[1]));
      if ( s2[0] != srci )
      {
        do
        {
          v393 = *v391;
          v394 = v391[1];
          v470.m128i_i64[0] = v464;
          v470.m128i_i64[1] = v393;
          v395 = (_QWORD *)sub_18B7900(a1 + 104, &v470);
          v396 = sub_18BB8E0(v395, v394);
          *((_BYTE *)v396 + 24) = 0;
          v479.m128i_i64[0] = v100;
          v479.m128i_i64[1] = v394;
          v480 = 0;
          v392 = (__m128i *)v396[1];
          if ( v392 == (__m128i *)v396[2] )
          {
            sub_18B49F0((const __m128i **)v396, v392, &v479);
          }
          else
          {
            if ( v392 )
            {
              a2 = (__m128)_mm_load_si128(&v479);
              *v392 = (__m128i)a2;
              v392[1].m128i_i64[0] = v480;
              v392 = (__m128i *)v396[1];
            }
            v396[1] = (char *)v392 + 24;
          }
          v391 += 2;
        }
        while ( srci != v391 );
      }
      goto LABEL_105;
    }
LABEL_107:
    if ( !v97[1] )
      sub_15F20C0(v97);
    if ( v473 != &v475 )
      _libc_free((unsigned __int64)v473);
    if ( s2[0] != dest )
      _libc_free((unsigned __int64)s2[0]);
  }
  while ( v95 );
  v19 = v408;
  v95 = v486.m128i_i64[1];
LABEL_115:
  j___libc_free_0(v95);
  if ( v19 )
LABEL_129:
    sub_18BE580(
      a1,
      v19,
      a2,
      *(double *)si128.m128_u64,
      *(double *)a4.m128i_i64,
      *(double *)a5.m128i_i64,
      v23,
      v24,
      a8,
      a9);
LABEL_5:
  v25 = (_QWORD *)a1;
  if ( !*(_QWORD *)(a1 + 32) )
  {
    v467 = 0;
    v468 = 0;
    v469 = 0;
    v470 = 0u;
    v471 = 0;
    v472 = 0;
    sub_18BFB30((__int64 *)a1, (__int64)&v467, (__int64)&v470);
    if ( !(_DWORD)v471 )
      goto LABEL_155;
    v112 = *(_QWORD *)(a1 + 24);
    if ( !v112 )
      goto LABEL_178;
    v113 = (_BYTE **)v470.m128i_i64[1];
    v486 = 0u;
    v487 = 0;
    LODWORD(v488) = 0;
    v114 = v470.m128i_i64[1] + 56LL * v472;
    if ( v470.m128i_i64[1] == v114 )
    {
      v116 = 0;
      v416 = v112 + 8;
      v421 = *(_QWORD *)(v112 + 24);
      if ( v112 + 8 != v421 )
      {
        v117 = 0;
        goto LABEL_368;
      }
      goto LABEL_177;
    }
    while ( 1 )
    {
      v115 = *v113;
      v465 = v113;
      if ( *v113 != (_BYTE *)-8LL && v115 != (_BYTE *)-4LL )
        break;
      v113 += 7;
      if ( (_BYTE **)v114 == v113 )
      {
        v116 = 0;
        v117 = 0;
        v416 = v112 + 8;
        v421 = *(_QWORD *)(v112 + 24);
        if ( v421 == v112 + 8 )
          goto LABEL_177;
LABEL_368:
        v261 = v117;
        while ( 1 )
        {
          v405 = *(char **)(v421 + 64);
          if ( *(char **)(v421 + 56) != v405 )
            break;
LABEL_473:
          srcp = (void *)v116;
          v326 = sub_220EEE0(v421);
          v116 = (__int64)srcp;
          v421 = v326;
          if ( v326 == v416 )
          {
            v117 = v261;
            goto LABEL_475;
          }
        }
        v411 = *(char **)(v421 + 56);
        while ( 1 )
        {
          srch = *(_QWORD **)v411;
          if ( *(_DWORD *)(*(_QWORD *)v411 + 8LL) == 1 )
          {
            v262 = *(_QWORD **)(*(_QWORD *)v411 + 96LL);
            if ( v262 )
              break;
          }
LABEL_472:
          v411 += 8;
          if ( v405 == v411 )
            goto LABEL_473;
        }
        v431 = (__int64 *)v262[4];
        if ( (__int64 *)v262[3] == v431 )
          goto LABEL_384;
        v434 = (__int64 *)v262[3];
        while ( 2 )
        {
          v263 = *v434;
          v264 = (void *)v434[1];
          if ( v261 )
          {
            v265 = (v261 - 1) & (37 * v263);
            v266 = (_QWORD *)(v116 + 16LL * v265);
            v267 = *v266;
            if ( v263 == *v266 )
            {
LABEL_377:
              v268 = v266[1] & 0xFFFFFFFFFFFFFFFCLL;
              if ( (v266[1] & 2) != 0 )
              {
                v269 = *(void ***)v268;
                v270 = *(_QWORD *)v268 + 8LL * *(unsigned int *)(v268 + 8);
LABEL_379:
                if ( (void **)v270 != v269 )
                {
                  v271 = v269;
                  do
                  {
                    v272 = *v271;
                    s2[1] = v264;
                    ++v271;
                    s2[0] = v272;
                    *(_WORD *)(sub_18B7900(a1 + 104, (const __m128i *)s2) + 24) = 256;
                  }
                  while ( (void **)v270 != v271 );
                }
              }
              else
              {
                v269 = (void **)(v266 + 1);
                v270 = (__int64)(v266 + 2);
                if ( v268 )
                  goto LABEL_379;
              }
LABEL_382:
              v434 += 2;
              v261 = v488;
              v116 = v486.m128i_i64[1];
              if ( v431 == v434 )
              {
                v262 = (_QWORD *)srch[12];
                if ( !v262 )
                  goto LABEL_472;
LABEL_384:
                v402 = (__int64 *)v262[7];
                if ( (__int64 *)v262[6] == v402 )
                  goto LABEL_409;
                v432 = (__int64 *)v262[6];
                while ( 2 )
                {
                  v273 = *v432;
                  v274 = (void *)v432[1];
                  if ( v261 )
                  {
                    LODWORD(v275) = (v261 - 1) & (37 * v273);
                    v276 = (_QWORD *)(v116 + 16LL * (unsigned int)v275);
                    v277 = *v276;
                    if ( v273 == *v276 )
                    {
LABEL_388:
                      v278 = v276[1] & 0xFFFFFFFFFFFFFFFCLL;
                      if ( (v276[1] & 2) != 0 )
                      {
                        v279 = *(void ***)v278;
                        v435 = *(_QWORD *)v278 + 8LL * *(unsigned int *)(v278 + 8);
LABEL_390:
                        if ( (void **)v435 != v279 )
                        {
                          v280 = v279;
                          do
                          {
                            v282 = *v280;
                            s2[1] = v274;
                            s2[0] = v282;
                            v283 = sub_18B7900(a1 + 104, (const __m128i *)s2);
                            v479.m128i_i64[0] = (__int64)srch;
                            v281 = *(_BYTE **)(v283 + 40);
                            if ( v281 == *(_BYTE **)(v283 + 48) )
                            {
                              v445 = v283;
                              sub_18BB3E0(v283 + 32, v281, &v479);
                              v283 = v445;
                            }
                            else
                            {
                              if ( v281 )
                              {
                                *(_QWORD *)v281 = srch;
                                v281 = *(_BYTE **)(v283 + 40);
                              }
                              *(_QWORD *)(v283 + 40) = v281 + 8;
                            }
                            *(_BYTE *)(v283 + 24) = 0;
                            ++v280;
                          }
                          while ( (void **)v435 != v280 );
                        }
                      }
                      else
                      {
                        v279 = (void **)(v276 + 1);
                        if ( v278 )
                        {
                          v435 = (__int64)(v276 + 2);
                          goto LABEL_390;
                        }
                      }
LABEL_407:
                      v432 += 2;
                      v261 = v488;
                      v116 = v486.m128i_i64[1];
                      if ( v402 == v432 )
                      {
                        v262 = (_QWORD *)srch[12];
                        if ( !v262 )
                          goto LABEL_472;
LABEL_409:
                        v403 = (__int64 *)v262[10];
                        if ( (__int64 *)v262[9] == v403 )
                          goto LABEL_439;
                        v287 = (__int64 *)v262[9];
                        while ( 2 )
                        {
                          if ( !v261 )
                          {
                            ++v486.m128i_i64[0];
                            goto LABEL_512;
                          }
                          v288 = (v261 - 1) & (37 * *v287);
                          v289 = (_QWORD *)(v116 + 16LL * v288);
                          v290 = *v289;
                          if ( *v287 != *v289 )
                          {
                            v331 = 1;
                            v332 = 0;
                            while ( v290 != -1 )
                            {
                              if ( !v332 && v290 == -2 )
                                v332 = v289;
                              v288 = (v261 - 1) & (v331 + v288);
                              v289 = (_QWORD *)(v116 + 16LL * v288);
                              v290 = *v289;
                              if ( *v287 == *v289 )
                                goto LABEL_413;
                              ++v331;
                            }
                            if ( v332 )
                              v289 = v332;
                            ++v486.m128i_i64[0];
                            v333 = v487 + 1;
                            if ( 4 * ((int)v487 + 1) < 3 * v261 )
                            {
                              if ( v261 - (v333 + HIDWORD(v487)) > v261 >> 3 )
                                goto LABEL_501;
                              sub_18834E0((__int64)&v486, v261);
                              if ( (_DWORD)v488 )
                              {
                                v335 = v488 - 1;
                                v333 = v487 + 1;
                                v336 = (v488 - 1) & (37 * *v287);
                                v289 = (_QWORD *)(v486.m128i_i64[1] + 16LL * v336);
                                v337 = *v289;
                                if ( *v289 != *v287 )
                                {
                                  v338 = (_QWORD *)(v486.m128i_i64[1] + 16LL * (v335 & (37 * (unsigned int)*v287)));
                                  v339 = 1;
                                  v289 = 0;
                                  while ( 1 )
                                  {
                                    if ( v337 == -1 )
                                    {
                                      if ( !v289 )
                                        v289 = v338;
                                      goto LABEL_501;
                                    }
                                    if ( v337 == -2 && !v289 )
                                      v289 = v338;
                                    v336 = v335 & (v339 + v336);
                                    v338 = (_QWORD *)(v486.m128i_i64[1] + 16LL * v336);
                                    v337 = *v338;
                                    if ( *v287 == *v338 )
                                      break;
                                    ++v339;
                                  }
                                  v289 = (_QWORD *)(v486.m128i_i64[1] + 16LL * v336);
                                }
LABEL_501:
                                LODWORD(v487) = v333;
                                if ( *v289 != -1 )
                                  --HIDWORD(v487);
                                v334 = *v287;
                                v289[1] = 0;
                                *v289 = v334;
                                goto LABEL_437;
                              }
                              goto LABEL_751;
                            }
LABEL_512:
                            sub_18834E0((__int64)&v486, 2 * v261);
                            if ( (_DWORD)v488 )
                            {
                              v333 = v487 + 1;
                              v340 = (v488 - 1) & (37 * *v287);
                              v289 = (_QWORD *)(v486.m128i_i64[1] + 16LL * v340);
                              v341 = *v289;
                              if ( *v289 != *v287 )
                              {
                                v342 = 1;
                                v343 = 0;
                                while ( v341 != -1 )
                                {
                                  if ( !v343 && v341 == -2 )
                                    v343 = v289;
                                  v340 = (v488 - 1) & (v342 + v340);
                                  v289 = (_QWORD *)(v486.m128i_i64[1] + 16LL * v340);
                                  v341 = *v289;
                                  if ( *v287 == *v289 )
                                    goto LABEL_501;
                                  ++v342;
                                }
                                if ( v343 )
                                  v289 = v343;
                              }
                              goto LABEL_501;
                            }
LABEL_751:
                            LODWORD(v487) = v487 + 1;
                            BUG();
                          }
LABEL_413:
                          v291 = v289[1] & 0xFFFFFFFFFFFFFFFCLL;
                          if ( (v289[1] & 2) != 0 )
                          {
                            v292 = *(void ***)v291;
                            v436 = *(_QWORD *)v291 + 8LL * *(unsigned int *)(v291 + 8);
                          }
                          else
                          {
                            v292 = (void **)(v289 + 1);
                            if ( !v291 )
                              goto LABEL_437;
                            v436 = (__int64)(v289 + 2);
                          }
                          if ( (void **)v436 == v292 )
                            goto LABEL_437;
LABEL_416:
                          s2[0] = *v292;
                          s2[1] = (void *)v287[1];
                          v293 = sub_18B7900(a1 + 104, (const __m128i *)s2);
                          v294 = *(_QWORD **)(v293 + 72);
                          v295 = v293;
                          v296 = (_QWORD *)(v293 + 64);
                          if ( !v294 )
                          {
                            v299 = (_QWORD *)(v293 + 64);
                            goto LABEL_435;
                          }
                          v297 = (char *)v287[3];
                          v298 = (char *)v287[2];
                          v299 = (_QWORD *)(v293 + 64);
                          v300 = v297 - v298;
LABEL_418:
                          v301 = (char *)v294[5];
                          v302 = (char *)v294[4];
                          if ( v301 - v302 > v300 )
                            v301 = &v302[v300];
                          v303 = (char *)v287[2];
                          if ( v302 == v301 )
                          {
LABEL_486:
                            if ( v303 == v297 )
                            {
LABEL_487:
                              v299 = v294;
                              v294 = (_QWORD *)v294[2];
LABEL_426:
                              if ( !v294 )
                              {
                                if ( v299 == v296 )
                                  goto LABEL_435;
                                v304 = (_QWORD *)v299[4];
                                v305 = v299[5] - (_QWORD)v304;
                                if ( v300 > v305 )
                                  v297 = &v298[v305];
                                if ( v298 == v297 )
                                {
LABEL_490:
                                  if ( (_QWORD *)v299[5] != v304 )
                                    goto LABEL_435;
                                }
                                else
                                {
                                  while ( *(_QWORD *)v298 >= *v304 )
                                  {
                                    if ( *(_QWORD *)v298 > *v304 )
                                      goto LABEL_436;
                                    v298 += 8;
                                    ++v304;
                                    if ( v297 == v298 )
                                      goto LABEL_490;
                                  }
LABEL_435:
                                  v479.m128i_i64[0] = (__int64)(v287 + 2);
                                  v299 = sub_18B80E0((_QWORD *)(v295 + 56), v299, v479.m128i_i64);
                                }
LABEL_436:
                                ++v292;
                                *((_WORD *)v299 + 40) = 256;
                                if ( (void **)v436 == v292 )
                                {
LABEL_437:
                                  v261 = v488;
                                  v116 = v486.m128i_i64[1];
                                  v287 += 5;
                                  if ( v403 == v287 )
                                  {
                                    v262 = (_QWORD *)srch[12];
                                    if ( !v262 )
                                      goto LABEL_472;
LABEL_439:
                                    v404 = (__int64 *)v262[13];
                                    if ( (__int64 *)v262[12] == v404 )
                                      goto LABEL_472;
                                    v306 = (__int64 *)v262[12];
                                    while ( 2 )
                                    {
                                      if ( !v261 )
                                      {
                                        ++v486.m128i_i64[0];
                                        goto LABEL_550;
                                      }
                                      v307 = (v261 - 1) & (37 * *v306);
                                      v308 = (_QWORD *)(v116 + 16LL * v307);
                                      v309 = *v308;
                                      if ( *v308 != *v306 )
                                      {
                                        v352 = 1;
                                        v353 = 0;
                                        while ( v309 != -1 )
                                        {
                                          if ( v309 == -2 && !v353 )
                                            v353 = v308;
                                          v307 = (v261 - 1) & (v352 + v307);
                                          v308 = (_QWORD *)(v116 + 16LL * v307);
                                          v309 = *v308;
                                          if ( *v306 == *v308 )
                                            goto LABEL_443;
                                          ++v352;
                                        }
                                        if ( v353 )
                                          v308 = v353;
                                        ++v486.m128i_i64[0];
                                        v354 = v487 + 1;
                                        if ( 4 * ((int)v487 + 1) < 3 * v261 )
                                        {
                                          if ( v261 - (v354 + HIDWORD(v487)) > v261 >> 3 )
                                            goto LABEL_539;
                                          sub_18834E0((__int64)&v486, v261);
                                          if ( (_DWORD)v488 )
                                          {
                                            v356 = (v488 - 1) & (37 * *v306);
                                            v354 = v487 + 1;
                                            v357 = (_QWORD *)(v486.m128i_i64[1] + 16LL * v356);
                                            v358 = *v357;
                                            v308 = v357;
                                            if ( *v357 != *v306 )
                                            {
                                              v359 = 1;
                                              v308 = 0;
                                              while ( 1 )
                                              {
                                                if ( v358 == -1 )
                                                {
                                                  if ( !v308 )
                                                    v308 = v357;
                                                  goto LABEL_539;
                                                }
                                                if ( !v308 && v358 == -2 )
                                                  v308 = v357;
                                                v356 = (v488 - 1) & (v359 + v356);
                                                v357 = (_QWORD *)(v486.m128i_i64[1] + 16LL * v356);
                                                v358 = *v357;
                                                if ( *v306 == *v357 )
                                                  break;
                                                ++v359;
                                              }
                                              v308 = (_QWORD *)(v486.m128i_i64[1] + 16LL * v356);
                                            }
LABEL_539:
                                            LODWORD(v487) = v354;
                                            if ( *v308 != -1 )
                                              --HIDWORD(v487);
                                            v355 = *v306;
                                            v308[1] = 0;
                                            *v308 = v355;
                                            goto LABEL_471;
                                          }
                                          goto LABEL_754;
                                        }
LABEL_550:
                                        sub_18834E0((__int64)&v486, 2 * v261);
                                        if ( (_DWORD)v488 )
                                        {
                                          v354 = v487 + 1;
                                          v360 = (v488 - 1) & (37 * *v306);
                                          v308 = (_QWORD *)(v486.m128i_i64[1] + 16LL * v360);
                                          v361 = *v308;
                                          if ( *v308 != *v306 )
                                          {
                                            v362 = 1;
                                            v363 = 0;
                                            while ( v361 != -1 )
                                            {
                                              if ( v361 == -2 && !v363 )
                                                v363 = v308;
                                              v360 = (v488 - 1) & (v362 + v360);
                                              v308 = (_QWORD *)(v486.m128i_i64[1] + 16LL * v360);
                                              v361 = *v308;
                                              if ( *v306 == *v308 )
                                                goto LABEL_539;
                                              ++v362;
                                            }
                                            if ( v363 )
                                              v308 = v363;
                                          }
                                          goto LABEL_539;
                                        }
LABEL_754:
                                        LODWORD(v487) = v487 + 1;
                                        BUG();
                                      }
LABEL_443:
                                      v310 = v308[1] & 0xFFFFFFFFFFFFFFFCLL;
                                      if ( (v308[1] & 2) != 0 )
                                      {
                                        v311 = *(void ***)v310;
                                        v437 = *(_QWORD *)v310 + 8LL * *(unsigned int *)(v310 + 8);
                                      }
                                      else
                                      {
                                        v311 = (void **)(v308 + 1);
                                        if ( !v310 )
                                          goto LABEL_471;
                                        v437 = (__int64)(v308 + 2);
                                      }
                                      if ( (void **)v437 == v311 )
                                        goto LABEL_471;
LABEL_446:
                                      s2[0] = *v311;
                                      s2[1] = (void *)v306[1];
                                      v312 = sub_18B7900(a1 + 104, (const __m128i *)s2);
                                      v313 = *(_QWORD **)(v312 + 72);
                                      v314 = v312;
                                      v315 = (_QWORD *)(v312 + 64);
                                      if ( !v313 )
                                      {
                                        v318 = (_QWORD *)(v312 + 64);
                                        goto LABEL_465;
                                      }
                                      v316 = (char *)v306[3];
                                      v317 = (char *)v306[2];
                                      v318 = (_QWORD *)(v312 + 64);
                                      v319 = v316 - v317;
LABEL_448:
                                      v320 = (char *)v313[5];
                                      v321 = (char *)v313[4];
                                      if ( v320 - v321 > v319 )
                                        v320 = &v321[v319];
                                      v322 = (char *)v306[2];
                                      if ( v321 == v320 )
                                      {
LABEL_484:
                                        if ( v316 == v322 )
                                        {
LABEL_485:
                                          v318 = v313;
                                          v313 = (_QWORD *)v313[2];
LABEL_456:
                                          if ( !v313 )
                                          {
                                            if ( v318 == v315 )
                                              goto LABEL_465;
                                            v323 = (_QWORD *)v318[4];
                                            v324 = v318[5] - (_QWORD)v323;
                                            if ( v319 > v324 )
                                              v316 = &v317[v324];
                                            if ( v317 == v316 )
                                            {
LABEL_488:
                                              if ( (_QWORD *)v318[5] != v323 )
                                                goto LABEL_465;
                                            }
                                            else
                                            {
                                              while ( *(_QWORD *)v317 >= *v323 )
                                              {
                                                if ( *(_QWORD *)v317 > *v323 )
                                                  goto LABEL_466;
                                                v317 += 8;
                                                ++v323;
                                                if ( v316 == v317 )
                                                  goto LABEL_488;
                                              }
LABEL_465:
                                              v479.m128i_i64[0] = (__int64)(v306 + 2);
                                              v318 = sub_18B80E0((_QWORD *)(v314 + 56), v318, v479.m128i_i64);
                                            }
LABEL_466:
                                            v479.m128i_i64[0] = (__int64)srch;
                                            v325 = (_BYTE *)v318[12];
                                            if ( v325 == (_BYTE *)v318[13] )
                                            {
                                              sub_18BB3E0((__int64)(v318 + 11), v325, &v479);
                                            }
                                            else
                                            {
                                              if ( v325 )
                                              {
                                                *(_QWORD *)v325 = srch;
                                                v325 = (_BYTE *)v318[12];
                                              }
                                              v318[12] = v325 + 8;
                                            }
                                            *((_BYTE *)v318 + 80) = 0;
                                            if ( (void **)v437 == ++v311 )
                                            {
LABEL_471:
                                              v261 = v488;
                                              v116 = v486.m128i_i64[1];
                                              v306 += 5;
                                              if ( v404 == v306 )
                                                goto LABEL_472;
                                              continue;
                                            }
                                            goto LABEL_446;
                                          }
                                          goto LABEL_448;
                                        }
                                      }
                                      else
                                      {
                                        while ( *(_QWORD *)v321 >= *(_QWORD *)v322 )
                                        {
                                          if ( *(_QWORD *)v321 > *(_QWORD *)v322 )
                                            goto LABEL_485;
                                          v321 += 8;
                                          v322 += 8;
                                          if ( v320 == v321 )
                                            goto LABEL_484;
                                        }
                                      }
                                      break;
                                    }
                                    v313 = (_QWORD *)v313[3];
                                    goto LABEL_456;
                                  }
                                  continue;
                                }
                                goto LABEL_416;
                              }
                              goto LABEL_418;
                            }
                          }
                          else
                          {
                            while ( *(_QWORD *)v302 >= *(_QWORD *)v303 )
                            {
                              if ( *(_QWORD *)v302 > *(_QWORD *)v303 )
                                goto LABEL_487;
                              v302 += 8;
                              v303 += 8;
                              if ( v301 == v302 )
                                goto LABEL_486;
                            }
                          }
                          break;
                        }
                        v294 = (_QWORD *)v294[3];
                        goto LABEL_426;
                      }
                      continue;
                    }
                    v284 = 1;
                    v285 = 0;
                    while ( v277 != -1 )
                    {
                      if ( !v285 && v277 == -2 )
                        v285 = v276;
                      v275 = (v261 - 1) & ((_DWORD)v275 + v284);
                      v276 = (_QWORD *)(v116 + 16 * v275);
                      v277 = *v276;
                      if ( v273 == *v276 )
                        goto LABEL_388;
                      ++v284;
                    }
                    if ( v285 )
                      v276 = v285;
                    ++v486.m128i_i64[0];
                    v286 = v487 + 1;
                    if ( 4 * ((int)v487 + 1) < 3 * v261 )
                    {
                      if ( v261 - (v286 + HIDWORD(v487)) <= v261 >> 3 )
                      {
                        sub_18834E0((__int64)&v486, v261);
                        if ( !(_DWORD)v488 )
                          goto LABEL_752;
                        v344 = 1;
                        v345 = 0;
                        v346 = (v488 - 1) & (37 * v273);
                        v286 = v487 + 1;
                        v276 = (_QWORD *)(v486.m128i_i64[1] + 16LL * v346);
                        v347 = *v276;
                        if ( v273 != *v276 )
                        {
                          while ( v347 != -1 )
                          {
                            if ( v347 == -2 && !v345 )
                              v345 = v276;
                            v346 = (v488 - 1) & (v344 + v346);
                            v276 = (_QWORD *)(v486.m128i_i64[1] + 16LL * v346);
                            v347 = *v276;
                            if ( v273 == *v276 )
                              goto LABEL_404;
                            ++v344;
                          }
                          if ( v345 )
                            v276 = v345;
                        }
                      }
LABEL_404:
                      LODWORD(v487) = v286;
                      if ( *v276 != -1 )
                        --HIDWORD(v487);
                      *v276 = v273;
                      v276[1] = 0;
                      goto LABEL_407;
                    }
                  }
                  else
                  {
                    ++v486.m128i_i64[0];
                  }
                  break;
                }
                sub_18834E0((__int64)&v486, 2 * v261);
                if ( !(_DWORD)v488 )
                {
LABEL_752:
                  LODWORD(v487) = v487 + 1;
                  BUG();
                }
                v348 = (v488 - 1) & (37 * v273);
                v286 = v487 + 1;
                v276 = (_QWORD *)(v486.m128i_i64[1] + 16LL * v348);
                v349 = *v276;
                if ( v273 != *v276 )
                {
                  v350 = 1;
                  v351 = 0;
                  while ( v349 != -1 )
                  {
                    if ( v349 == -2 && !v351 )
                      v351 = v276;
                    v348 = (v488 - 1) & (v350 + v348);
                    v276 = (_QWORD *)(v486.m128i_i64[1] + 16LL * v348);
                    v349 = *v276;
                    if ( v273 == *v276 )
                      goto LABEL_404;
                    ++v350;
                  }
                  if ( v351 )
                    v276 = v351;
                }
                goto LABEL_404;
              }
              continue;
            }
            v364 = 1;
            v365 = 0;
            while ( v267 != -1 )
            {
              if ( !v365 && v267 == -2 )
                v365 = v266;
              v265 = (v261 - 1) & (v364 + v265);
              v266 = (_QWORD *)(v116 + 16LL * v265);
              v267 = *v266;
              if ( *v266 == v263 )
                goto LABEL_377;
              ++v364;
            }
            if ( v365 )
              v266 = v365;
            ++v486.m128i_i64[0];
            v366 = v487 + 1;
            if ( 4 * ((int)v487 + 1) < 3 * v261 )
            {
              if ( v261 - (v366 + HIDWORD(v487)) <= v261 >> 3 )
              {
                sub_18834E0((__int64)&v486, v261);
                if ( !(_DWORD)v488 )
                  goto LABEL_753;
                v367 = 1;
                v368 = 0;
                v369 = (v488 - 1) & (37 * v263);
                v366 = v487 + 1;
                v266 = (_QWORD *)(v486.m128i_i64[1] + 16LL * v369);
                v370 = *v266;
                if ( v263 != *v266 )
                {
                  while ( v370 != -1 )
                  {
                    if ( !v368 && v370 == -2 )
                      v368 = v266;
                    v369 = (v488 - 1) & (v367 + v369);
                    v266 = (_QWORD *)(v486.m128i_i64[1] + 16LL * v369);
                    v370 = *v266;
                    if ( *v266 == v263 )
                      goto LABEL_563;
                    ++v367;
                  }
                  if ( v368 )
                    v266 = v368;
                }
              }
LABEL_563:
              LODWORD(v487) = v366;
              if ( *v266 != -1 )
                --HIDWORD(v487);
              *v266 = v263;
              v266[1] = 0;
              goto LABEL_382;
            }
          }
          else
          {
            ++v486.m128i_i64[0];
          }
          break;
        }
        sub_18834E0((__int64)&v486, 2 * v261);
        if ( !(_DWORD)v488 )
        {
LABEL_753:
          LODWORD(v487) = v487 + 1;
          BUG();
        }
        v371 = (v488 - 1) & (37 * v263);
        v366 = v487 + 1;
        v266 = (_QWORD *)(v486.m128i_i64[1] + 16LL * v371);
        v372 = *v266;
        if ( *v266 != v263 )
        {
          v373 = 1;
          v374 = 0;
          while ( v372 != -1 )
          {
            if ( v372 == -2 && !v374 )
              v374 = v266;
            v371 = (v488 - 1) & (v373 + v371);
            v266 = (_QWORD *)(v486.m128i_i64[1] + 16LL * v371);
            v372 = *v266;
            if ( *v266 == v263 )
              goto LABEL_563;
            ++v373;
          }
          if ( v374 )
            v266 = v374;
        }
        goto LABEL_563;
      }
    }
    if ( (_BYTE **)v114 == v113 )
    {
      v116 = 0;
      v117 = 0;
      v416 = v112 + 8;
      v421 = *(_QWORD *)(v112 + 24);
      if ( v112 + 8 != v421 )
        goto LABEL_368;
      goto LABEL_177;
    }
    while ( 1 )
    {
      if ( *v115 )
        goto LABEL_363;
      v244 = (int *)sub_161E970((__int64)v115);
      v246 = v245;
      sub_16C1840(s2);
      sub_16C1A90((int *)s2, v244, v246);
      sub_16C1AA0(s2, &v479);
      v247 = v488;
      v248 = v479.m128i_i64[0];
      if ( !(_DWORD)v488 )
        break;
      v249 = v486.m128i_i32[2];
      v250 = (v488 - 1) & (37 * v479.m128i_i32[0]);
      v251 = (__int64 *)(v486.m128i_i64[1] + 16LL * v250);
      v252 = *v251;
      if ( v479.m128i_i64[0] != *v251 )
      {
        v380 = 0;
        v381 = 1;
        while ( v252 != -1 )
        {
          if ( v252 == -2 && !v380 )
            v380 = v251;
          v250 = (v488 - 1) & (v381 + v250);
          v251 = (__int64 *)(v486.m128i_i64[1] + 16LL * v250);
          v252 = *v251;
          if ( v479.m128i_i64[0] == *v251 )
            goto LABEL_353;
          ++v381;
        }
        if ( v380 )
          v251 = v380;
        ++v486.m128i_i64[0];
        v375 = v487 + 1;
        if ( 4 * ((int)v487 + 1) < (unsigned int)(3 * v488) )
        {
          if ( (int)v488 - HIDWORD(v487) - v375 <= (unsigned int)v488 >> 3 )
          {
LABEL_582:
            sub_18834E0((__int64)&v486, v247);
            sub_1882CA0((__int64)&v486, v479.m128i_i64, s2);
            v251 = (__int64 *)s2[0];
            v248 = v479.m128i_i64[0];
            v375 = v487 + 1;
          }
          LODWORD(v487) = v375;
          if ( *v251 != -1 )
            --HIDWORD(v487);
          *v251 = v248;
          v251[1] = 0;
LABEL_586:
          v251[1] = (__int64)v115;
          goto LABEL_363;
        }
LABEL_581:
        v247 = 2 * v488;
        goto LABEL_582;
      }
LABEL_353:
      v253 = v251[1];
      v254 = v253 & 0xFFFFFFFFFFFFFFFCLL;
      if ( (v253 & 0xFFFFFFFFFFFFFFFCLL) == 0 )
        goto LABEL_586;
      if ( (v253 & 2) == 0 )
      {
        v255 = sub_22077B0(48);
        if ( v255 )
        {
          *(_QWORD *)v255 = v255 + 16;
          *(_QWORD *)(v255 + 8) = 0x400000000LL;
        }
        v256 = v255 & 0xFFFFFFFFFFFFFFFCLL;
        v251[1] = v255 | 2;
        v257 = *(unsigned int *)((v255 & 0xFFFFFFFFFFFFFFFCLL) + 8);
        if ( (unsigned int)v257 >= *(_DWORD *)(v256 + 12) )
        {
          sub_16CD150(v256, (const void *)(v256 + 16), 0, 8, v252, v249);
          v257 = *(unsigned int *)(v256 + 8);
        }
        *(_QWORD *)(*(_QWORD *)v256 + 8 * v257) = v254;
        ++*(_DWORD *)(v256 + 8);
        v254 = v251[1] & 0xFFFFFFFFFFFFFFFCLL;
      }
      v258 = *(unsigned int *)(v254 + 8);
      if ( (unsigned int)v258 >= *(_DWORD *)(v254 + 12) )
      {
        sub_16CD150(v254, (const void *)(v254 + 16), 0, 8, v252, v249);
        v258 = *(unsigned int *)(v254 + 8);
      }
      *(_QWORD *)(*(_QWORD *)v254 + 8 * v258) = v115;
      ++*(_DWORD *)(v254 + 8);
LABEL_363:
      v259 = v465 + 7;
      if ( v465 + 7 != (_BYTE **)v114 )
      {
        while ( 1 )
        {
          v115 = *v259;
          v465 = v259;
          if ( *v259 != (_BYTE *)-8LL && v115 != (_BYTE *)-4LL )
            break;
          v259 += 7;
          if ( (_BYTE **)v114 == v259 )
            goto LABEL_367;
        }
        if ( v259 != (_BYTE **)v114 )
          continue;
      }
LABEL_367:
      v117 = v488;
      v116 = v486.m128i_i64[1];
      v260 = *(_QWORD *)(a1 + 24);
      v416 = v260 + 8;
      v421 = *(_QWORD *)(v260 + 24);
      if ( v421 != v260 + 8 )
        goto LABEL_368;
LABEL_475:
      if ( v117 )
      {
        v327 = (_QWORD *)v116;
        v328 = (_QWORD *)(v116 + 16LL * v117);
        do
        {
          if ( *v327 <= 0xFFFFFFFFFFFFFFFDLL )
          {
            v329 = v327[1];
            if ( (v329 & 2) != 0 )
            {
              v330 = (unsigned __int64 *)(v329 & 0xFFFFFFFFFFFFFFFCLL);
              if ( v330 )
              {
                if ( (unsigned __int64 *)*v330 != v330 + 2 )
                  _libc_free(*v330);
                j_j___libc_free_0(v330, 48);
              }
            }
          }
          v327 += 2;
        }
        while ( v328 != v327 );
        v116 = v486.m128i_i64[1];
      }
LABEL_177:
      j___libc_free_0(v116);
LABEL_178:
      LODWORD(v474) = 0;
      v476 = &v474;
      v477 = &v474;
      v139 = *(__int64 **)(a1 + 136);
      v140 = *(__int64 **)(a1 + 144);
      v475 = 0;
      v478 = 0;
      v444 = v140;
      if ( v140 == v139 )
      {
        sub_18B4900(
          a1,
          a2,
          *(double *)si128.m128_u64,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128i_i64,
          v110,
          v111,
          a8,
          a9);
        goto LABEL_154;
      }
      v433 = 0;
      v118 = v139;
      while ( 2 )
      {
        v141 = v472;
        v486 = 0u;
        v487 = 0;
        if ( v472 )
        {
          v119 = (_BYTE *)*v118;
          v142 = (v472 - 1) & (((unsigned int)*v118 >> 9) ^ ((unsigned int)*v118 >> 4));
          v143 = (__int64 *)(v470.m128i_i64[1] + 56LL * v142);
          v144 = (_BYTE *)*v143;
          if ( *v143 == *v118 )
          {
LABEL_182:
            v145 = v143[4];
            v146 = (char *)(v143 + 2);
            if ( v146 == (char *)v145 )
            {
              if ( *(_QWORD *)(a1 + 24) )
                goto LABEL_133;
              goto LABEL_136;
            }
            srcd = (void *)v118[1];
            do
            {
              v147 = **(_QWORD **)(v145 + 32);
              if ( (*(_BYTE *)(v147 + 80) & 1) == 0 )
                goto LABEL_131;
              v148 = sub_18B8B00((__int64 *)a1, *(_QWORD *)(v147 - 24), (unsigned __int64)srcd + *(_QWORD *)(v145 + 40));
              if ( !v148 )
                goto LABEL_131;
              v149 = sub_1649C60(v148);
              if ( *(_BYTE *)(v149 + 16) )
                goto LABEL_131;
              v429 = v149;
              v150 = sub_1649960(v149);
              if ( v151 != 18
                || *(_QWORD *)v150 ^ 0x75705F6178635F5FLL | *((_QWORD *)v150 + 1) ^ 0x75747269765F6572LL
                || *((_WORD *)v150 + 8) != 27745 )
              {
                sub_18B8E80((__int64)s2, v429, v145 + 32);
                v152 = (__m128 *)v486.m128i_i64[1];
                if ( v486.m128i_i64[1] == v487 )
                {
                  sub_18BBFB0((const __m128i **)&v486, (const __m128i *)v486.m128i_i64[1], (const __m128i *)s2);
                }
                else
                {
                  if ( v486.m128i_i64[1] )
                  {
                    si128 = (__m128)_mm_load_si128((const __m128i *)s2);
                    *(__m128 *)v486.m128i_i64[1] = si128;
                    a4 = _mm_load_si128(dest);
                    v152[1] = (__m128)a4;
                    v152 = (__m128 *)v486.m128i_i64[1];
                  }
                  v486.m128i_i64[1] = (__int64)&v152[2];
                }
              }
              v145 = sub_220EF30(v145);
            }
            while ( v146 != (char *)v145 );
LABEL_206:
            v157 = v486.m128i_i64[1];
            v158 = v486.m128i_i64[0];
            v159 = *(_QWORD **)(a1 + 24);
            if ( v486.m128i_i64[1] == v486.m128i_i64[0] )
              goto LABEL_131;
            if ( !v159 )
              goto LABEL_210;
            if ( *(_BYTE *)*v118 )
            {
              v159 = 0;
              goto LABEL_210;
            }
            v177 = (__int8 *)sub_161E970(*v118);
            v178 = (size_t)v176;
            if ( !v177 )
            {
              dest[0].m128i_i8[0] = 0;
              s2[0] = dest;
              s2[1] = 0;
              goto LABEL_244;
            }
            v179 = dest;
            v479.m128i_i64[0] = (__int64)v176;
            s2[0] = dest;
            if ( (unsigned __int64)v176 > 0xF )
            {
              srcq = v177;
              v387 = (__m128i *)sub_22409D0(s2, &v479, 0);
              v177 = srcq;
              s2[0] = v387;
              v386 = v387;
              dest[0].m128i_i64[0] = v479.m128i_i64[0];
            }
            else
            {
              if ( v176 == (void *)1 )
              {
                dest[0].m128i_i8[0] = *v177;
                goto LABEL_695;
              }
              if ( !v176 )
              {
LABEL_695:
                s2[1] = v176;
                *((_BYTE *)v176 + (_QWORD)v179) = 0;
LABEL_244:
                v180 = v159[12];
                v181 = v159 + 11;
                if ( !v180 )
                {
                  v182 = (__int64)(v159 + 11);
                  goto LABEL_261;
                }
                v182 = (__int64)(v159 + 11);
                srcf = v118;
                v183 = s2[0];
                v184 = s2[1];
LABEL_247:
                while ( 2 )
                {
                  v185 = *(_QWORD *)(v180 + 40);
                  v186 = (size_t)v184;
                  if ( v185 <= (unsigned __int64)v184 )
                    v186 = *(_QWORD *)(v180 + 40);
                  if ( v186
                    && (v422 = v181,
                        v427 = v183,
                        LODWORD(v187) = memcmp(*(const void **)(v180 + 32), v183, v186),
                        v183 = v427,
                        v181 = v422,
                        (_DWORD)v187) )
                  {
LABEL_253:
                    if ( (int)v187 < 0 )
                    {
LABEL_246:
                      v180 = *(_QWORD *)(v180 + 24);
                      if ( !v180 )
                        goto LABEL_255;
                      continue;
                    }
                  }
                  else
                  {
                    v187 = v185 - (_QWORD)v184;
                    if ( (__int64)(v185 - (_QWORD)v184) < 0x80000000LL )
                    {
                      if ( v187 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
                        goto LABEL_246;
                      goto LABEL_253;
                    }
                  }
                  break;
                }
                v182 = v180;
                v180 = *(_QWORD *)(v180 + 16);
                if ( !v180 )
                {
LABEL_255:
                  v188 = v184;
                  v118 = srcf;
                  if ( v181 == (_QWORD *)v182 )
                    goto LABEL_261;
                  v189 = *(_QWORD *)(v182 + 40);
                  v190 = (size_t)v184;
                  if ( v189 <= (unsigned __int64)v184 )
                    v190 = *(_QWORD *)(v182 + 40);
                  if ( v190 )
                  {
                    LODWORD(v191) = memcmp(v183, *(const void **)(v182 + 32), v190);
                    v188 = v184;
                    if ( (_DWORD)v191 )
                    {
LABEL_260:
                      if ( (int)v191 < 0 )
                        goto LABEL_261;
LABEL_262:
                      sub_2240A30(s2);
                      v192 = *(unsigned __int64 **)(v182 + 120);
                      if ( !v192 )
                      {
                        v192 = (unsigned __int64 *)(v182 + 112);
                        goto LABEL_270;
                      }
                      v193 = v118[1];
                      v194 = (unsigned __int64 *)(v182 + 112);
                      while ( 1 )
                      {
                        v195 = (unsigned __int64 *)v192[2];
                        v196 = v192[3];
                        if ( v192[4] < v193 )
                        {
                          v192 = v194;
                          v195 = (unsigned __int64 *)v196;
                        }
                        if ( !v195 )
                          break;
                        v194 = v192;
                        v192 = v195;
                      }
                      if ( (unsigned __int64 *)(v182 + 112) == v192 || v193 < v192[4] )
                      {
LABEL_270:
                        s2[0] = v118 + 1;
                        v192 = sub_18BE220((_QWORD *)(v182 + 104), v192, (unsigned __int64 **)s2);
                      }
                      v159 = v192 + 5;
                      v157 = v486.m128i_i64[1];
                      v158 = v486.m128i_i64[0];
LABEL_210:
                      if ( !(unsigned __int8)sub_18B85C0(
                                               a1,
                                               v158,
                                               (v157 - v158) >> 5,
                                               (__int64)(v118 + 2),
                                               (__int64)v159) )
                      {
                        v433 |= sub_18BDA40(
                                  a1,
                                  (__int64 *)v486.m128i_i64[0],
                                  (v486.m128i_i64[1] - v486.m128i_i64[0]) >> 5,
                                  (__int64)(v118 + 2),
                                  v159,
                                  a2,
                                  *(double *)si128.m128_u64,
                                  *(double *)a4.m128i_i64,
                                  *(double *)a5.m128i_i64,
                                  v110,
                                  v111,
                                  a8,
                                  a9,
                                  v160,
                                  *v118,
                                  v118[1]);
                        sub_18BD0B0(
                          (__int64 **)a1,
                          (__int64 *)v486.m128i_i64[0],
                          (v486.m128i_i64[1] - v486.m128i_i64[0]) >> 5,
                          (__int64)(v118 + 2),
                          v159,
                          a2,
                          *(double *)si128.m128_u64,
                          *(double *)a4.m128i_i64,
                          *(double *)a5.m128i_i64,
                          v377,
                          v378,
                          a8,
                          a9,
                          v376,
                          (_BYTE *)*v118,
                          v118[1]);
                      }
                      if ( !*(_BYTE *)(a1 + 80) )
                        goto LABEL_131;
                      srce = (void *)v486.m128i_i64[1];
                      v161 = v486.m128i_i64[0];
                      if ( v486.m128i_i64[1] == v486.m128i_i64[0] )
                        goto LABEL_131;
                      v426 = v118;
                      while ( 2 )
                      {
                        if ( !*(_BYTE *)(v161 + 25) )
                          goto LABEL_215;
                        v163 = sub_1649960(*(_QWORD *)v161);
                        v164 = v162;
                        if ( v163 )
                        {
                          v479.m128i_i64[0] = v162;
                          v165 = (void *)v162;
                          s2[0] = dest;
                          if ( v162 > 0xF )
                          {
                            s2[0] = (void *)sub_22409D0(s2, &v479, 0);
                            v379 = (__m128i *)s2[0];
                            dest[0].m128i_i64[0] = v479.m128i_i64[0];
                          }
                          else
                          {
                            if ( v162 == 1 )
                            {
                              dest[0].m128i_i8[0] = *v163;
                              v166 = dest;
                              goto LABEL_221;
                            }
                            if ( !v162 )
                            {
                              v166 = dest;
                              goto LABEL_221;
                            }
                            v379 = dest;
                          }
                          memcpy(v379, v163, v164);
                          v165 = (void *)v479.m128i_i64[0];
                          v166 = (__m128i *)s2[0];
LABEL_221:
                          s2[1] = v165;
                          *((_BYTE *)v165 + (_QWORD)v166) = 0;
                        }
                        else
                        {
                          dest[0].m128i_i8[0] = 0;
                          s2[0] = dest;
                          s2[1] = 0;
                        }
                        if ( !v475 )
                        {
                          v175 = &v474;
                          goto LABEL_273;
                        }
                        v167 = v161;
                        v168 = s2[0];
                        v169 = &v474;
                        v170 = v475;
                        v171 = s2[1];
                        while ( 1 )
                        {
LABEL_225:
                          v172 = v170[5];
                          v173 = (size_t)v171;
                          if ( v172 <= (unsigned __int64)v171 )
                            v173 = v170[5];
                          if ( v173 )
                          {
                            LODWORD(v174) = memcmp((const void *)v170[4], v168, v173);
                            if ( (_DWORD)v174 )
                              goto LABEL_231;
                          }
                          v174 = v172 - (_QWORD)v171;
                          if ( (__int64)(v172 - (_QWORD)v171) >= 0x80000000LL )
                            break;
                          if ( v174 > (__int64)0xFFFFFFFF7FFFFFFFLL )
                          {
LABEL_231:
                            if ( (int)v174 >= 0 )
                              break;
                          }
                          v170 = (__int64 *)v170[3];
                          if ( !v170 )
                            goto LABEL_233;
                        }
                        v169 = v170;
                        v170 = (__int64 *)v170[2];
                        if ( v170 )
                          goto LABEL_225;
LABEL_233:
                        v161 = v167;
                        v175 = v169;
                        if ( v169 != &v474 && sub_18B4C50(v168, (size_t)v171, (const void *)v169[4], v169[5]) >= 0 )
                          goto LABEL_235;
LABEL_273:
                        v197 = (__m128 *)sub_22077B0(72);
                        v198 = v175;
                        v175 = (__int64 *)v197;
                        v197[2].m128_u64[0] = (unsigned __int64)&v197[3];
                        if ( s2[0] == dest )
                        {
                          a5 = _mm_load_si128(dest);
                          v197[3] = (__m128)a5;
                        }
                        else
                        {
                          v197[2].m128_u64[0] = (unsigned __int64)s2[0];
                          v197[3].m128_u64[0] = dest[0].m128i_i64[0];
                        }
                        v199 = s2[1];
                        v197[4].m128_u64[0] = 0;
                        v423 = v197 + 3;
                        v197[2].m128_u64[1] = (unsigned __int64)v199;
                        s2[0] = dest;
                        s2[1] = 0;
                        dest[0].m128i_i8[0] = 0;
                        v200 = sub_18BF480(&v473, v198, (__int64)&v197[2]);
                        v202 = (__int64 *)v200;
                        v203 = v201;
                        if ( v201 )
                        {
                          if ( v200 || &v474 == v201 )
                          {
LABEL_278:
                            v204 = 1;
LABEL_279:
                            sub_220F040(v204, v175, v203, &v474);
                            ++v478;
                            goto LABEL_235;
                          }
                          v383 = v201[5];
                          v382 = v383;
                          if ( (unsigned __int64)v199 <= v383 )
                            v383 = (size_t)v199;
                          if ( !v383
                            || (v425 = v203,
                                v384 = memcmp((const void *)v175[4], (const void *)v203[4], v383),
                                v203 = v425,
                                (v385 = v384) == 0) )
                          {
                            v204 = 0;
                            if ( (__int64)((__int64)v199 - v382) >= 0x80000000LL )
                              goto LABEL_279;
                            if ( (__int64)((__int64)v199 - v382) <= (__int64)0xFFFFFFFF7FFFFFFFLL )
                              goto LABEL_278;
                            v385 = (_DWORD)v199 - v382;
                          }
                          v204 = v385 >> 31;
                          goto LABEL_279;
                        }
                        v205 = v175[4];
                        if ( v423 != (__m128 *)v205 )
                          j_j___libc_free_0(v205, v175[6] + 1);
                        v206 = v175;
                        v175 = v202;
                        j_j___libc_free_0(v206, 72);
LABEL_235:
                        v175[8] = *(_QWORD *)v161;
                        sub_2240A30(s2);
LABEL_215:
                        v161 += 32;
                        if ( srce != (void *)v161 )
                          continue;
                        break;
                      }
                      v118 = v426;
LABEL_131:
                      if ( *(_QWORD *)(a1 + 24) )
                      {
                        v119 = (_BYTE *)*v118;
LABEL_133:
                        if ( !*v119 )
                        {
                          v207 = &v479;
                          v208 = (int *)sub_161E970((__int64)v119);
                          v210 = v209;
                          sub_16C1840(s2);
                          sub_16C1A90((int *)s2, v208, v210);
                          sub_16C1AA0(s2, &v479);
                          v211 = (__int64 *)v118[7];
                          v212 = (__int64 *)v118[6];
                          v430 = (char *)v479.m128i_i64[0];
                          while ( v211 != v212 )
                          {
                            v213 = *v212++;
                            sub_18BAD00(v213, v430);
                          }
                          v214 = v118[12];
                          v417 = v118 + 10;
                          if ( (__int64 *)v214 != v118 + 10 )
                          {
                            v410 = v118;
                            while ( 1 )
                            {
                              v424 = *(__int64 **)(v214 + 96);
                              if ( v424 == *(__int64 **)(v214 + 88) )
                                goto LABEL_346;
                              srcg = *(__int64 **)(v214 + 88);
                              v428 = v207;
                              do
                              {
                                v216 = *srcg;
                                v479.m128i_i64[0] = (__int64)v430;
                                v217 = *(_QWORD **)(v216 + 96);
                                if ( v217
                                  || (sub_18B9090(s2),
                                      v217 = s2[0],
                                      s2[0] = 0,
                                      v218 = *(_QWORD **)(v216 + 96),
                                      *(_QWORD *)(v216 + 96) = v217,
                                      !v218) )
                                {
                                  v215 = (_BYTE *)v217[1];
                                  if ( v215 == (_BYTE *)v217[2] )
                                    goto LABEL_344;
                                }
                                else
                                {
                                  v219 = v218[12];
                                  if ( v218[13] != v219 )
                                  {
                                    v220 = v218;
                                    v221 = v218[12];
                                    v222 = v218[13];
                                    do
                                    {
                                      v223 = *(_QWORD *)(v221 + 16);
                                      if ( v223 )
                                        j_j___libc_free_0(v223, *(_QWORD *)(v221 + 32) - v223);
                                      v221 += 40;
                                    }
                                    while ( v222 != v221 );
                                    v219 = v220[12];
                                    v218 = v220;
                                  }
                                  if ( v219 )
                                    j_j___libc_free_0(v219, v218[14] - v219);
                                  v224 = v218[9];
                                  if ( v218[10] != v224 )
                                  {
                                    v225 = v218;
                                    v226 = v218[9];
                                    v227 = v218[10];
                                    do
                                    {
                                      v228 = *(_QWORD *)(v226 + 16);
                                      if ( v228 )
                                        j_j___libc_free_0(v228, *(_QWORD *)(v226 + 32) - v228);
                                      v226 += 40;
                                    }
                                    while ( v227 != v226 );
                                    v224 = v225[9];
                                    v218 = v225;
                                  }
                                  if ( v224 )
                                    j_j___libc_free_0(v224, v218[11] - v224);
                                  v229 = v218[6];
                                  if ( v229 )
                                    j_j___libc_free_0(v229, v218[8] - v229);
                                  v230 = v218[3];
                                  if ( v230 )
                                    j_j___libc_free_0(v230, v218[5] - v230);
                                  if ( *v218 )
                                    j_j___libc_free_0(*v218, v218[2] - *v218);
                                  j_j___libc_free_0(v218, 120);
                                  v231 = s2[0];
                                  if ( s2[0] )
                                  {
                                    v232 = *((_QWORD *)s2[0] + 12);
                                    if ( *((_QWORD *)s2[0] + 13) != v232 )
                                    {
                                      v233 = s2[0];
                                      v234 = *((_QWORD *)s2[0] + 12);
                                      v235 = *((_QWORD *)s2[0] + 13);
                                      do
                                      {
                                        v236 = *(_QWORD *)(v234 + 16);
                                        if ( v236 )
                                          j_j___libc_free_0(v236, *(_QWORD *)(v234 + 32) - v236);
                                        v234 += 40;
                                      }
                                      while ( v235 != v234 );
                                      v232 = v233[12];
                                      v231 = v233;
                                    }
                                    if ( v232 )
                                      j_j___libc_free_0(v232, v231[14] - v232);
                                    v237 = v231[9];
                                    if ( v231[10] != v237 )
                                    {
                                      v238 = v231;
                                      v239 = v231[9];
                                      v240 = v231[10];
                                      do
                                      {
                                        v241 = *(_QWORD *)(v239 + 16);
                                        if ( v241 )
                                          j_j___libc_free_0(v241, *(_QWORD *)(v239 + 32) - v241);
                                        v239 += 40;
                                      }
                                      while ( v240 != v239 );
                                      v237 = v238[9];
                                      v231 = v238;
                                    }
                                    if ( v237 )
                                      j_j___libc_free_0(v237, v231[11] - v237);
                                    v242 = v231[6];
                                    if ( v242 )
                                      j_j___libc_free_0(v242, v231[8] - v242);
                                    v243 = v231[3];
                                    if ( v243 )
                                      j_j___libc_free_0(v243, v231[5] - v243);
                                    if ( *v231 )
                                      j_j___libc_free_0(*v231, v231[2] - *v231);
                                    j_j___libc_free_0(v231, 120);
                                  }
                                  v217 = *(_QWORD **)(v216 + 96);
                                  v215 = (_BYTE *)v217[1];
                                  if ( v215 == (_BYTE *)v217[2] )
                                  {
LABEL_344:
                                    sub_9CA200((__int64)v217, v215, v428);
                                    goto LABEL_294;
                                  }
                                }
                                if ( v215 )
                                {
                                  *(_QWORD *)v215 = v479.m128i_i64[0];
                                  v215 = (_BYTE *)v217[1];
                                }
                                v217[1] = v215 + 8;
LABEL_294:
                                ++srcg;
                              }
                              while ( v424 != srcg );
                              v207 = v428;
LABEL_346:
                              v214 = sub_220EEE0(v214);
                              if ( (__int64 *)v214 == v417 )
                              {
                                v118 = v410;
                                break;
                              }
                            }
                          }
                        }
                      }
                      if ( v486.m128i_i64[0] )
                        j_j___libc_free_0(v486.m128i_i64[0], v487 - v486.m128i_i64[0]);
LABEL_136:
                      v118 += 15;
                      if ( v444 != v118 )
                        continue;
                      if ( *(_BYTE *)(a1 + 80) )
                      {
                        for ( i = v476; i != &v474; i = (__int64 *)sub_220EEE0(i) )
                        {
                          v121 = i[8];
                          srco = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD, __int64))(a1 + 88))(
                                             *(_QWORD *)(a1 + 96),
                                             v121);
                          sub_15CA470((__int64)s2, (__int64)"wholeprogramdevirt", (__int64)"Devirtualized", 13, v121);
                          sub_15CAB20((__int64)s2, "devirtualized ", 0xEu);
                          v122 = (char *)sub_1649960(v121);
                          sub_15C9800((__int64)&v479, "FunctionName", 12, v122, v123);
                          v486.m128i_i64[0] = (__int64)&v487;
                          sub_18B4BA0(v486.m128i_i64, v479.m128i_i64[0], v479.m128i_i64[0] + v479.m128i_i64[1]);
                          v489[0] = (__int64)&v490;
                          sub_18B4BA0(v489, v481, (__int64)v481 + v482);
                          v491 = _mm_load_si128(&v484);
                          v492 = v485;
                          sub_15CAC60((__int64)s2, &v486);
                          sub_2240A30(v489);
                          sub_2240A30(&v486);
                          sub_143AA50(srco, (__int64)s2);
                          if ( v481 != &v483 )
                            j_j___libc_free_0(v481, v483 + 1);
                          sub_2240A30(&v479);
                          s2[0] = &unk_49ECF68;
                          srcc = v495;
                          v124 = &v495[88 * v496];
                          if ( v495 != v124 )
                          {
                            do
                            {
                              v124 -= 88;
                              v125 = (char *)*((_QWORD *)v124 + 4);
                              if ( v125 != v124 + 48 )
                                j_j___libc_free_0(v125, *((_QWORD *)v124 + 6) + 1LL);
                              if ( *(char **)v124 != v124 + 16 )
                                j_j___libc_free_0(*(_QWORD *)v124, *((_QWORD *)v124 + 2) + 1LL);
                            }
                            while ( srcc != v124 );
                            v124 = v495;
                          }
                          if ( v124 != &v497 )
                            _libc_free((unsigned __int64)v124);
                        }
                      }
                      sub_18B4900(
                        a1,
                        a2,
                        *(double *)si128.m128_u64,
                        *(double *)a4.m128i_i64,
                        *(double *)a5.m128i_i64,
                        v110,
                        v111,
                        a8,
                        a9);
                      if ( v433 )
                      {
                        v128 = v467;
                        v129 = v468;
                        while ( v129 != v128 )
                        {
                          v130 = v128;
                          v128 += 14;
                          sub_18BA960(
                            (__int64 *)a1,
                            v130,
                            a2,
                            *(double *)si128.m128_u64,
                            *(double *)a4.m128i_i64,
                            *(double *)a5.m128i_i64,
                            v126,
                            v127,
                            a8,
                            a9);
                        }
                      }
LABEL_154:
                      sub_18B5EE0(v475);
LABEL_155:
                      if ( v472 )
                      {
                        v131 = (_QWORD *)v470.m128i_i64[1];
                        v132 = v470.m128i_i64[1] + 56LL * v472;
                        do
                        {
                          if ( *v131 != -4 && *v131 != -8 )
                            sub_18B52F0(v131[3]);
                          v131 += 7;
                        }
                        while ( (_QWORD *)v132 != v131 );
                      }
                      j___libc_free_0(v470.m128i_i64[1]);
                      v133 = v468;
                      v134 = v467;
                      if ( v468 != v467 )
                      {
                        do
                        {
                          v135 = v134[11];
                          if ( v135 )
                            j_j___libc_free_0(v135, v134[13] - v135);
                          v136 = v134[8];
                          if ( v136 )
                            j_j___libc_free_0(v136, v134[10] - v136);
                          v137 = v134[5];
                          if ( v137 )
                            j_j___libc_free_0(v137, v134[7] - v137);
                          v138 = v134[2];
                          if ( v138 )
                            j_j___libc_free_0(v138, v134[4] - v138);
                          v134 += 14;
                        }
                        while ( v133 != v134 );
                        v134 = v467;
                      }
                      if ( v134 )
                        j_j___libc_free_0(v134, v469 - (_QWORD)v134);
                      return 1;
                    }
                  }
                  v191 = (signed __int64)v188 - v189;
                  if ( (__int64)((__int64)v188 - v189) >= 0x80000000LL )
                    goto LABEL_262;
                  if ( v191 > (__int64)0xFFFFFFFF7FFFFFFFLL )
                    goto LABEL_260;
LABEL_261:
                  v479.m128i_i64[0] = (__int64)s2;
                  v182 = sub_18BD640(v159 + 10, (_QWORD *)v182, (__m128i **)&v479);
                  goto LABEL_262;
                }
                goto LABEL_247;
              }
              v386 = dest;
            }
            memcpy(v386, v177, v178);
            v176 = (void *)v479.m128i_i64[0];
            v179 = (__m128i *)s2[0];
            goto LABEL_695;
          }
          v153 = 0;
          v154 = 1;
          while ( v144 != (_BYTE *)-4LL )
          {
            if ( !v153 && v144 == (_BYTE *)-8LL )
              v153 = v143;
            v142 = (v472 - 1) & (v154 + v142);
            v143 = (__int64 *)(v470.m128i_i64[1] + 56LL * v142);
            v144 = (_BYTE *)*v143;
            if ( v119 == (_BYTE *)*v143 )
              goto LABEL_182;
            ++v154;
          }
          if ( v153 )
            v143 = v153;
          ++v470.m128i_i64[0];
          v155 = v471 + 1;
          if ( 4 * ((int)v471 + 1) < 3 * v472 )
          {
            if ( v472 - HIDWORD(v471) - v155 > v472 >> 3 )
            {
LABEL_203:
              LODWORD(v471) = v155;
              if ( *v143 != -4 )
                --HIDWORD(v471);
              v156 = *v118;
              *((_DWORD *)v143 + 4) = 0;
              v143[3] = 0;
              *v143 = v156;
              v143[4] = (__int64)(v143 + 2);
              v143[5] = (__int64)(v143 + 2);
              v143[6] = 0;
              goto LABEL_206;
            }
LABEL_238:
            sub_18BF870((__int64)&v470, v141);
            sub_18BE320((__int64)&v470, v118, s2);
            v143 = (__int64 *)s2[0];
            v155 = v471 + 1;
            goto LABEL_203;
          }
        }
        else
        {
          ++v470.m128i_i64[0];
        }
        break;
      }
      v141 = 2 * v472;
      goto LABEL_238;
    }
    ++v486.m128i_i64[0];
    goto LABEL_581;
  }
  v26 = *(char **)(a1 + 136);
  v409 = *(char **)(a1 + 144);
  if ( v26 == v409 )
    goto LABEL_85;
  v27 = v26 + 80;
  v28 = *(char **)(a1 + 32);
  while ( 2 )
  {
    v29 = *((_QWORD *)v27 - 10);
    v30 = (void *)*((_QWORD *)v27 - 9);
    src = v28;
    v31 = sub_161E970(v29);
    v33 = src;
    v34 = (__int8 *)v31;
    v35 = v32;
    if ( !v31 )
    {
      v40 = dest;
      dest[0].m128i_i8[0] = 0;
      v38 = src + 88;
      s2[1] = 0;
      s2[0] = dest;
      v39 = (char *)*((_QWORD *)src + 12);
      if ( !v39 )
        goto LABEL_66;
      goto LABEL_13;
    }
    v486.m128i_i64[0] = v32;
    v36 = (void *)v32;
    s2[0] = dest;
    if ( v32 > 0xF )
    {
      v440 = src;
      srcl = v34;
      v79 = (__m128i *)sub_22409D0(s2, &v486, 0);
      v34 = srcl;
      v33 = v440;
      s2[0] = v79;
      v80 = v79;
      dest[0].m128i_i64[0] = v486.m128i_i64[0];
      goto LABEL_82;
    }
    if ( v32 != 1 )
    {
      if ( !v32 )
      {
        v37 = dest;
        goto LABEL_12;
      }
      v80 = dest;
LABEL_82:
      srcm = v33;
      memcpy(v80, v34, v35);
      v36 = (void *)v486.m128i_i64[0];
      v37 = (__m128i *)s2[0];
      v33 = srcm;
      goto LABEL_12;
    }
    dest[0].m128i_i8[0] = *v34;
    v37 = dest;
LABEL_12:
    s2[1] = v36;
    v38 = v33 + 88;
    *((_BYTE *)v36 + (_QWORD)v37) = 0;
    v39 = (char *)*((_QWORD *)v33 + 12);
    v40 = (__m128i *)s2[0];
    if ( !v39 )
      goto LABEL_77;
LABEL_13:
    srca = v30;
    v41 = v38;
    v438 = v29;
    v42 = v38;
    v418 = v27;
    v43 = s2[1];
    v412 = (_QWORD ***)v25;
    while ( 1 )
    {
LABEL_15:
      v44 = *((_QWORD *)v39 + 5);
      v45 = (size_t)v43;
      if ( v44 <= (unsigned __int64)v43 )
        v45 = *((_QWORD *)v39 + 5);
      if ( v45 )
      {
        v406 = v40;
        LODWORD(v46) = memcmp(*((const void **)v39 + 4), v40, v45);
        v40 = v406;
        if ( (_DWORD)v46 )
          goto LABEL_21;
      }
      v46 = v44 - (_QWORD)v43;
      if ( (__int64)(v44 - (_QWORD)v43) >= 0x80000000LL )
        break;
      if ( v46 > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
LABEL_21:
        if ( (int)v46 >= 0 )
          break;
      }
      v39 = (char *)*((_QWORD *)v39 + 3);
      if ( !v39 )
        goto LABEL_23;
    }
    v41 = v39;
    v39 = (char *)*((_QWORD *)v39 + 2);
    if ( v39 )
      goto LABEL_15;
LABEL_23:
    v47 = v42;
    v48 = v41;
    v49 = (signed __int64)v43;
    v50 = (unsigned __int64)srca;
    v51 = v438;
    v27 = v418;
    v25 = v412;
    if ( v48 == v47 )
      goto LABEL_77;
    v52 = *((_QWORD *)v48 + 5);
    v53 = v49;
    if ( v52 <= v49 )
      v53 = *((_QWORD *)v48 + 5);
    if ( v53
      && (v419 = v49,
          v439 = v48,
          srcj = v40,
          LODWORD(v54) = memcmp(v40, *((const void **)v48 + 4), v53),
          v40 = srcj,
          v48 = v439,
          v49 = v419,
          (_DWORD)v54) )
    {
LABEL_30:
      if ( (int)v54 < 0 )
        goto LABEL_77;
    }
    else
    {
      v54 = v49 - v52;
      if ( (__int64)(v49 - v52) < 0x80000000LL )
      {
        if ( v54 > (__int64)0xFFFFFFFF7FFFFFFFLL )
          goto LABEL_30;
LABEL_77:
        if ( v40 != dest )
          j_j___libc_free_0(v40, dest[0].m128i_i64[0] + 1);
        goto LABEL_66;
      }
    }
    if ( v40 != dest )
    {
      srck = v48;
      j_j___libc_free_0(v40, dest[0].m128i_i64[0] + 1);
      v48 = srck;
    }
    v55 = (char *)*((_QWORD *)v48 + 15);
    if ( v55 )
    {
      srcb = v48 + 112;
      do
      {
        while ( 1 )
        {
          v56 = *((_QWORD *)v55 + 2);
          v57 = *((_QWORD *)v55 + 3);
          if ( *((_QWORD *)v55 + 4) >= v50 )
            break;
          v55 = (char *)*((_QWORD *)v55 + 3);
          if ( !v57 )
            goto LABEL_38;
        }
        srcb = v55;
        v55 = (char *)*((_QWORD *)v55 + 2);
      }
      while ( v56 );
LABEL_38:
      if ( v48 + 112 != srcb && *((_QWORD *)srcb + 4) <= v50 )
      {
        v420 = (__int64)(v27 - 64);
        if ( *((_DWORD *)srcb + 10) == 1 )
        {
          v82 = *v412;
          v83 = (__int64 *)sub_1643270(**v412);
          v84 = *((_QWORD *)srcb + 7);
          v85 = *((_QWORD *)srcb + 6);
          s2[1] = 0;
          v414 = v84;
          v441 = v85;
          s2[0] = dest;
          v86 = sub_1644EA0(v83, dest, 0, 0);
          v87 = sub_1632080((__int64)v82, v441, v414, v86, 0);
          if ( s2[0] != dest )
            _libc_free((unsigned __int64)s2[0]);
          v486.m128i_i64[0] = v87;
          s2[1] = &v486;
          v479.m128i_i8[0] = 0;
          s2[0] = v25;
          dest[0].m128i_i64[0] = (__int64)&v479;
          sub_18B7020((__int64 *)s2, v420);
          if ( *((char **)v27 + 2) == v27 )
          {
            v58 = (__int64)v27;
          }
          else
          {
            v442 = v25;
            v88 = *((_QWORD *)v27 + 2);
            do
            {
              sub_18B7020((__int64 *)s2, v88 + 56);
              v88 = sub_220EEE0(v88);
            }
            while ( (char *)v88 != v27 );
            v25 = v442;
            v58 = *((_QWORD *)v27 + 2);
          }
        }
        else
        {
          v58 = *((_QWORD *)v27 + 2);
        }
        if ( (char *)v58 != v27 )
        {
          v413 = v27;
          v59 = v25;
          v60 = v58;
          while ( 1 )
          {
            v61 = (char *)*((_QWORD *)srcb + 12);
            if ( !v61 )
              goto LABEL_63;
            v62 = *(__int64 **)(v60 + 40);
            v63 = *(__int64 **)(v60 + 32);
            v64 = srcb + 88;
            v65 = (char *)v62 - (char *)v63;
            do
            {
              v66 = (char *)*((_QWORD *)v61 + 5);
              v67 = (char *)*((_QWORD *)v61 + 4);
              if ( v66 - v67 > v65 )
                v66 = &v67[v65];
              v68 = *(__int64 **)(v60 + 32);
              if ( v67 != v66 )
              {
                while ( *(_QWORD *)v67 >= (unsigned __int64)*v68 )
                {
                  if ( *(_QWORD *)v67 > (unsigned __int64)*v68 )
                    goto LABEL_69;
                  v67 += 8;
                  ++v68;
                  if ( v66 == v67 )
                    goto LABEL_68;
                }
LABEL_53:
                v61 = (char *)*((_QWORD *)v61 + 3);
                continue;
              }
LABEL_68:
              if ( v62 != v68 )
                goto LABEL_53;
LABEL_69:
              v64 = v61;
              v61 = (char *)*((_QWORD *)v61 + 2);
            }
            while ( v61 );
            if ( v64 != srcb + 88 )
            {
              v69 = (_QWORD *)*((_QWORD *)v64 + 4);
              v70 = *((_QWORD *)v64 + 5) - (_QWORD)v69;
              if ( v65 > v70 )
                v62 = (__int64 *)((char *)v63 + v70);
              if ( v63 == v62 )
              {
LABEL_70:
                if ( *((_QWORD **)v64 + 5) == v69 )
                {
LABEL_71:
                  v72 = *((_DWORD *)v64 + 14);
                  switch ( v72 )
                  {
                    case 2:
                      v76 = sub_18B6670(*v59, v59[5], v51, v50, v63, v65 >> 3, "unique_member", 0xDu);
                      sub_18B7320(
                        (__int64)v59,
                        v60 + 56,
                        byte_3F871B3,
                        0,
                        *((_QWORD *)v64 + 8) != 0,
                        (__int64)v76,
                        a2,
                        *(double *)si128.m128_u64,
                        *(double *)a4.m128i_i64,
                        *(double *)a5.m128i_i64,
                        v77,
                        v78,
                        a8,
                        a9);
                      break;
                    case 3:
                      v407 = sub_18B6A10(v59, v51, v50, v63, v65 >> 3, v59[7], "byte", 4u, *((_DWORD *)v64 + 18));
                      v73 = sub_18B6A10(
                              v59,
                              v51,
                              v50,
                              *(__int64 **)(v60 + 32),
                              (__int64)(*(_QWORD *)(v60 + 40) - *(_QWORD *)(v60 + 32)) >> 3,
                              v59[5],
                              "bit",
                              3u,
                              *((_DWORD *)v64 + 19));
                      sub_18B96C0(
                        (__int64)v59,
                        v60 + 56,
                        byte_3F871B3,
                        0,
                        v407,
                        v73,
                        *(double *)a2.m128_u64,
                        *(double *)si128.m128_u64,
                        *(double *)a4.m128i_i64,
                        *(double *)a5.m128i_i64,
                        v74,
                        v75,
                        a8,
                        a9);
                      break;
                    case 1:
                      sub_18B71D0(
                        (__int64)v59,
                        v60 + 56,
                        byte_3F871B3,
                        0,
                        *((_QWORD *)v64 + 8),
                        *(double *)a2.m128_u64,
                        *(double *)si128.m128_u64,
                        *(double *)a4.m128i_i64,
                        *(double *)a5.m128i_i64,
                        v23,
                        v24,
                        a8,
                        a9);
                      break;
                  }
                }
              }
              else
              {
                v71 = *(__int64 **)(v60 + 32);
                while ( (unsigned __int64)*v71 >= *v69 )
                {
                  if ( (unsigned __int64)*v71 > *v69 )
                    goto LABEL_71;
                  ++v71;
                  ++v69;
                  if ( v62 == v71 )
                    goto LABEL_70;
                }
              }
            }
LABEL_63:
            v60 = sub_220EEE0(v60);
            if ( (char *)v60 == v413 )
            {
              v25 = v59;
              v27 = v413;
              break;
            }
          }
        }
        if ( *((_DWORD *)srcb + 10) == 2 )
        {
          v415 = (_QWORD **)*v25;
          v443 = (__int64 *)sub_1643270(*(_QWORD **)*v25);
          sub_18B61E0((__int64 *)s2, v51, v50, 0, 0, v89, "branch_funnel", 0xDu);
          v90 = s2[0];
          v486.m128i_i64[0] = (__int64)&v487;
          v486.m128i_i64[1] = 0;
          srcn = s2[1];
          v91 = sub_1644EA0(v443, &v487, 0, 0);
          v92 = sub_1632080((__int64)v415, (__int64)v90, (__int64)srcn, v91, 0);
          if ( (__int64 *)v486.m128i_i64[0] != &v487 )
            _libc_free(v486.m128i_u64[0]);
          sub_2240A30(s2);
          LOBYTE(s2[0]) = 0;
          sub_18BD000(
            (__int64)v25,
            v420,
            v92,
            s2,
            a2,
            *(double *)si128.m128_u64,
            *(double *)a4.m128i_i64,
            *(double *)a5.m128i_i64,
            v93,
            v94,
            a8,
            a9);
        }
      }
    }
LABEL_66:
    if ( v409 != v27 + 40 )
    {
      v28 = (char *)v25[4];
      v27 += 120;
      continue;
    }
    break;
  }
LABEL_85:
  sub_18B4900(a1, a2, *(double *)si128.m128_u64, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, v23, v24, a8, a9);
  return 1;
}
