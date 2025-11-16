// Function: sub_18895E0
// Address: 0x18895e0
//
__int64 __fastcall sub_18895E0(
        __int64 **a1,
        __int64 *a2,
        __int64 a3,
        __int64 ***a4,
        unsigned __int64 a5,
        __m128 a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13,
        __int64 a14,
        __int64 *a15,
        __int64 a16)
{
  __int64 ***v16; // r15
  unsigned int v18; // esi
  __int64 v19; // rdi
  int v20; // r12d
  __int64 v21; // rbx
  __int64 *v22; // r15
  __int64 v24; // rcx
  unsigned int v25; // edx
  __int64 *v26; // rax
  __int64 v27; // r8
  __int64 *v28; // r13
  int v29; // edx
  __int64 v30; // rax
  __int64 v31; // rbx
  char *v32; // r14
  __int64 v33; // r8
  __int64 v34; // r15
  __int64 v35; // rbx
  __int64 v36; // rdi
  unsigned int v37; // ecx
  __int64 *v38; // rdx
  __int64 v39; // r9
  __int64 v40; // rax
  __int64 v41; // r13
  _QWORD *v42; // rax
  __int64 v43; // rdx
  __int64 v44; // r14
  _BOOL4 v45; // r8d
  __int64 v46; // rax
  int v47; // esi
  __int64 v48; // rcx
  unsigned int v49; // edx
  __int64 *v50; // rax
  __int64 **v51; // rdi
  __int64 v52; // r12
  __int64 v53; // r13
  __int64 *v54; // rbx
  __int64 v55; // r13
  unsigned int v56; // edx
  _QWORD *v57; // rsi
  __int64 v58; // rax
  __int64 v59; // r14
  __int64 v60; // rax
  int v61; // edx
  __int64 v62; // r8
  __int64 v63; // r12
  __int64 v64; // rbx
  __int64 v65; // rsi
  __int64 v66; // r12
  __int64 v67; // r13
  __int64 v68; // rdi
  char *v69; // rax
  char *v70; // rax
  char *v71; // rbx
  __int64 v72; // r13
  char *v73; // rbx
  __int64 v74; // rsi
  __int64 v75; // r8
  bool v76; // bl
  char *v77; // r9
  __int64 v78; // r15
  char *v79; // rsi
  __int64 *v80; // r13
  __int64 v81; // r12
  __int64 **v82; // rax
  __int64 v83; // rcx
  __int64 *v84; // rax
  __int64 v85; // r12
  __int64 v86; // rbx
  __int64 v87; // rax
  double v88; // xmm4_8
  double v89; // xmm5_8
  size_t v90; // r8
  unsigned int v91; // esi
  __int64 v92; // rdi
  __int64 v93; // rax
  unsigned int v94; // ebx
  __int64 v95; // rcx
  unsigned int v96; // edx
  __int64 *v97; // rax
  __int64 v98; // r9
  __int64 *v99; // r13
  unsigned int v100; // esi
  __int64 v101; // r10
  int v102; // edx
  char *v103; // rsi
  _QWORD *v104; // rax
  _QWORD *v105; // rdi
  char *v106; // rdx
  __int64 **v107; // rcx
  unsigned __int64 v108; // rax
  __int64 **v109; // r14
  unsigned int v110; // r8d
  __int64 **v111; // rax
  __int64 *v112; // rdi
  __int64 *v113; // rbx
  __int64 v114; // r12
  __int64 v115; // rax
  __int64 *v116; // rax
  __int64 v117; // rax
  __int64 *v118; // r12
  unsigned int v119; // esi
  __int64 v120; // rcx
  int v121; // edx
  __int64 *v122; // r8
  int v123; // edx
  int v124; // r10d
  int v125; // r10d
  _QWORD *v126; // r9
  _QWORD *v127; // r8
  __int64 v128; // r15
  int v129; // r9d
  __int64 v130; // rcx
  int v131; // ebx
  __int64 *v132; // r11
  int v133; // edi
  __int64 *v134; // r11
  char *v135; // rbx
  __int64 v136; // r12
  __int64 *v137; // rdx
  __int64 v138; // r14
  unsigned int v139; // eax
  __int64 v140; // rcx
  unsigned __int64 v141; // r12
  __int64 v142; // rax
  unsigned __int64 v143; // rax
  unsigned __int64 v144; // rcx
  unsigned __int64 v145; // rsi
  __int64 *v146; // rax
  __int64 *v147; // rax
  char *v148; // rsi
  __int64 v149; // rax
  int v150; // eax
  int v151; // eax
  unsigned int v152; // eax
  __int64 v153; // rsi
  unsigned __int64 v154; // r14
  _QWORD *v155; // rax
  int v156; // ebx
  __int64 v157; // rax
  double v158; // xmm4_8
  double v159; // xmm5_8
  __int64 v160; // r12
  __int64 *v161; // rax
  __int64 v162; // rdi
  __int64 v163; // rdi
  __int64 v164; // rax
  __int64 v165; // r14
  char v166; // al
  char *v167; // rbx
  char *v168; // r12
  char *v169; // rbx
  __int64 v170; // r12
  __int64 v171; // rdi
  __int64 result; // rax
  __int64 *v173; // r11
  int v174; // r10d
  _QWORD *v175; // r9
  __int64 v176; // rdx
  int v177; // r15d
  unsigned int v178; // r13d
  unsigned int v179; // esi
  __int64 v180; // rdi
  __int64 v181; // rax
  int v182; // r14d
  __int64 v183; // rcx
  unsigned int v184; // edx
  __int64 *v185; // rax
  __int64 v186; // r8
  __int64 v187; // rbx
  __int64 *v188; // rbx
  __int64 v189; // r8
  int v190; // edx
  unsigned int v191; // esi
  __int64 *v192; // rbx
  __int64 *v193; // rax
  __int64 v194; // r12
  __int64 v195; // rax
  unsigned int v196; // eax
  __int64 v197; // rsi
  __int64 *v198; // rax
  __int64 v199; // rax
  double v200; // xmm4_8
  double v201; // xmm5_8
  int v202; // r9d
  int v203; // r8d
  __int64 v204; // r14
  __int64 *v205; // r14
  __int64 v206; // rdx
  char *v207; // rsi
  __int64 v208; // r9
  _BYTE *v209; // r12
  char v210; // al
  char v211; // al
  __int64 v212; // rax
  __int64 ***v213; // r15
  __int64 **v214; // r12
  __int64 *v215; // rax
  __int64 v216; // rdi
  __int64 ***v217; // rax
  __int64 *v218; // r12
  double v219; // xmm4_8
  double v220; // xmm5_8
  char v221; // al
  __int64 v222; // r14
  __int64 v223; // rdx
  __int64 v224; // rdx
  char v225; // al
  __int64 *v226; // r14
  __int64 v227; // rdx
  char *v228; // rsi
  int v229; // r13d
  int v230; // r12d
  __int64 v231; // rax
  unsigned int v232; // r15d
  int v233; // r8d
  int v234; // r9d
  __int64 v235; // rax
  __int64 v236; // r8
  __int64 *v237; // rbx
  __int64 v238; // rdx
  void **v239; // rdi
  __int64 v240; // rdi
  _BYTE *v241; // rax
  const char *v242; // rax
  char *v243; // r15
  size_t v244; // rax
  __int64 v245; // rcx
  size_t v246; // rdx
  unsigned int v247; // eax
  __int64 v248; // rsi
  char *v249; // r14
  unsigned int v250; // ebx
  bool v251; // r15
  char v252; // r12
  _WORD *v253; // rdx
  __int64 v254; // rdx
  void **v255; // rdi
  __int64 v256; // rax
  _QWORD *v257; // rdx
  __int64 v258; // rax
  __int64 v259; // rax
  unsigned int v260; // eax
  unsigned int v261; // esi
  __int64 v262; // r12
  char *v263; // rax
  char *v264; // rbx
  _QWORD *v265; // rax
  int v266; // r8d
  int v267; // r9d
  __int64 v268; // rdx
  __int64 **v269; // rbx
  __int64 v270; // rax
  __int64 **v271; // r13
  __int64 v272; // r12
  _BYTE *v273; // r13
  size_t v274; // rbx
  _BYTE *v275; // r14
  size_t v276; // r15
  __int64 *v277; // rax
  __int64 **v278; // rax
  __int64 v279; // rax
  __int64 v280; // rbx
  __int64 v281; // rax
  __int64 *v282; // r13
  __int64 *v283; // rcx
  __int64 *v284; // rdx
  int v285; // esi
  __int64 v286; // rax
  _QWORD *v287; // r14
  size_t v288; // rax
  __int64 v289; // r15
  __int64 *v290; // rsi
  __int64 v291; // rdi
  __int64 v292; // rdx
  int v293; // r8d
  size_t v294; // r9
  char v295; // al
  int v296; // r13d
  unsigned __int64 *v297; // r13
  __int64 v298; // rax
  unsigned __int64 v299; // rcx
  __int64 v300; // rsi
  unsigned __int8 *v301; // rsi
  _QWORD *v302; // rax
  _QWORD *v303; // r15
  unsigned __int64 *v304; // r14
  __int64 v305; // rax
  unsigned __int64 v306; // rsi
  __int64 v307; // rsi
  unsigned __int8 *v308; // rsi
  __int64 v309; // rdx
  __int64 v310; // rax
  __int64 v311; // rax
  __int64 v312; // rbx
  __int64 *v313; // rax
  _QWORD *v314; // rax
  __int64 v315; // rax
  size_t v316; // r8
  size_t v317; // rdx
  char *v318; // rcx
  __int64 *v319; // rdi
  char *v320; // rsi
  _QWORD *v321; // rax
  __int64 v322; // rax
  __int64 ***v323; // r10
  __int64 ***v324; // rax
  double v325; // xmm4_8
  double v326; // xmm5_8
  char *v327; // rax
  char *v328; // rdx
  int v329; // r9d
  __int64 *v330; // rdx
  __int64 *v331; // rax
  _QWORD *v332; // rcx
  int v333; // r11d
  __int64 *v334; // r10
  __int64 v335; // rax
  unsigned __int64 v336; // rax
  int v337; // eax
  _QWORD *v338; // rax
  __int64 v339; // rax
  int v340; // ecx
  __int64 **v341; // rdx
  __int64 **v342; // rcx
  __int64 v343; // r13
  int v344; // esi
  __int64 *v345; // rdi
  _QWORD *v346; // rax
  int v347; // r9d
  __int64 *v348; // rdi
  int v349; // edi
  __int64 **v350; // rsi
  int v351; // r9d
  __int64 *v352; // rdi
  signed __int64 v353; // [rsp+0h] [rbp-3A0h]
  __int64 ***v354; // [rsp+8h] [rbp-398h]
  __int64 ***v355; // [rsp+8h] [rbp-398h]
  size_t v357; // [rsp+10h] [rbp-390h]
  __int64 ***v358; // [rsp+18h] [rbp-388h]
  __int64 *v359; // [rsp+18h] [rbp-388h]
  size_t v360; // [rsp+18h] [rbp-388h]
  __int64 v361; // [rsp+20h] [rbp-380h]
  __int64 *v362; // [rsp+20h] [rbp-380h]
  __int64 n; // [rsp+30h] [rbp-370h]
  size_t na; // [rsp+30h] [rbp-370h]
  size_t ne; // [rsp+30h] [rbp-370h]
  int nb; // [rsp+30h] [rbp-370h]
  char nc; // [rsp+30h] [rbp-370h]
  int nd; // [rsp+30h] [rbp-370h]
  size_t ng; // [rsp+30h] [rbp-370h]
  size_t nf; // [rsp+30h] [rbp-370h]
  size_t nh; // [rsp+30h] [rbp-370h]
  int ni; // [rsp+30h] [rbp-370h]
  __int64 v375; // [rsp+38h] [rbp-368h]
  __int64 v376; // [rsp+38h] [rbp-368h]
  unsigned int v377; // [rsp+38h] [rbp-368h]
  __int64 *v379; // [rsp+40h] [rbp-360h]
  __int64 ***v380; // [rsp+48h] [rbp-358h]
  __int64 ***v381; // [rsp+48h] [rbp-358h]
  __int64 *v382; // [rsp+48h] [rbp-358h]
  __int64 v383; // [rsp+48h] [rbp-358h]
  unsigned int v384; // [rsp+50h] [rbp-350h]
  __int64 v385; // [rsp+50h] [rbp-350h]
  int v386; // [rsp+50h] [rbp-350h]
  int v387; // [rsp+50h] [rbp-350h]
  __int64 ***v389; // [rsp+58h] [rbp-348h]
  char *v390; // [rsp+58h] [rbp-348h]
  __int64 *v391; // [rsp+58h] [rbp-348h]
  __int64 v392; // [rsp+58h] [rbp-348h]
  __int64 v393; // [rsp+58h] [rbp-348h]
  __int64 v394; // [rsp+60h] [rbp-340h]
  __int64 *v395; // [rsp+60h] [rbp-340h]
  __int64 ***v396; // [rsp+60h] [rbp-340h]
  __int64 v397; // [rsp+60h] [rbp-340h]
  __int64 v398; // [rsp+60h] [rbp-340h]
  __int64 v399; // [rsp+60h] [rbp-340h]
  __int64 v400; // [rsp+60h] [rbp-340h]
  __int64 v401; // [rsp+60h] [rbp-340h]
  _BOOL4 v402; // [rsp+68h] [rbp-338h]
  __int64 *v403; // [rsp+68h] [rbp-338h]
  char *v404; // [rsp+68h] [rbp-338h]
  __int64 v405; // [rsp+70h] [rbp-330h] BYREF
  char *v406; // [rsp+78h] [rbp-328h]
  __int64 v407; // [rsp+80h] [rbp-320h]
  __int64 v408[2]; // [rsp+90h] [rbp-310h] BYREF
  __int16 v409; // [rsp+A0h] [rbp-300h]
  __int64 v410[2]; // [rsp+B0h] [rbp-2F0h] BYREF
  __int16 v411; // [rsp+C0h] [rbp-2E0h]
  __int64 v412; // [rsp+D0h] [rbp-2D0h] BYREF
  __int64 v413; // [rsp+D8h] [rbp-2C8h]
  __int64 v414; // [rsp+E0h] [rbp-2C0h]
  unsigned int v415; // [rsp+E8h] [rbp-2B8h]
  __int64 v416; // [rsp+F0h] [rbp-2B0h] BYREF
  __int64 v417; // [rsp+F8h] [rbp-2A8h]
  __int64 v418; // [rsp+100h] [rbp-2A0h]
  unsigned int v419; // [rsp+108h] [rbp-298h]
  __int64 v420; // [rsp+110h] [rbp-290h] BYREF
  __int64 v421; // [rsp+118h] [rbp-288h]
  __int64 v422; // [rsp+120h] [rbp-280h]
  unsigned int v423; // [rsp+128h] [rbp-278h]
  _QWORD v424[2]; // [rsp+130h] [rbp-270h] BYREF
  char v425; // [rsp+140h] [rbp-260h] BYREF
  _QWORD v426[2]; // [rsp+150h] [rbp-250h] BYREF
  char v427; // [rsp+160h] [rbp-240h] BYREF
  char *v428; // [rsp+170h] [rbp-230h] BYREF
  char *v429; // [rsp+178h] [rbp-228h]
  char *v430; // [rsp+180h] [rbp-220h]
  char *v431; // [rsp+188h] [rbp-218h]
  char *v432; // [rsp+190h] [rbp-210h]
  char *v433; // [rsp+198h] [rbp-208h]
  void *v434; // [rsp+1A0h] [rbp-200h] BYREF
  __int64 v435; // [rsp+1A8h] [rbp-1F8h]
  __int64 v436; // [rsp+1B0h] [rbp-1F0h]
  __int64 v437; // [rsp+1B8h] [rbp-1E8h]
  int v438; // [rsp+1C0h] [rbp-1E0h]
  _QWORD *v439; // [rsp+1C8h] [rbp-1D8h]
  __int64 *v440; // [rsp+1D0h] [rbp-1D0h] BYREF
  __int64 v441; // [rsp+1D8h] [rbp-1C8h]
  __int64 v442; // [rsp+1E0h] [rbp-1C0h]
  __int64 v443; // [rsp+1E8h] [rbp-1B8h]
  int v444; // [rsp+1F0h] [rbp-1B0h]
  _QWORD *v445; // [rsp+1F8h] [rbp-1A8h]
  char *v446; // [rsp+200h] [rbp-1A0h] BYREF
  char *v447; // [rsp+208h] [rbp-198h]
  char *v448; // [rsp+210h] [rbp-190h]
  _QWORD *v449; // [rsp+218h] [rbp-188h]
  __int64 v450; // [rsp+220h] [rbp-180h]
  int v451; // [rsp+228h] [rbp-178h]
  __int64 *v452; // [rsp+230h] [rbp-170h]
  __int64 v453; // [rsp+238h] [rbp-168h]
  __int64 **v454; // [rsp+250h] [rbp-150h] BYREF
  __int64 v455; // [rsp+258h] [rbp-148h]
  _WORD v456[64]; // [rsp+260h] [rbp-140h] BYREF
  __int64 *v457; // [rsp+2E0h] [rbp-C0h] BYREF
  __int64 v458; // [rsp+2E8h] [rbp-B8h]
  __int64 v459; // [rsp+2F0h] [rbp-B0h] BYREF
  unsigned int v460; // [rsp+2F8h] [rbp-A8h]

  v16 = a4;
  v412 = 0;
  v413 = 0;
  v414 = 0;
  v415 = 0;
  v394 = 48 * a3;
  if ( !a3 )
  {
    v405 = 0;
    v32 = 0;
    v407 = 0;
    goto LABEL_14;
  }
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = a2;
  while ( 1 )
  {
    v28 = &v22[v21];
    if ( !v18 )
    {
      ++v412;
LABEL_8:
      v18 *= 2;
LABEL_9:
      sub_1883320((__int64)&v412, v18);
      sub_1882BF0((__int64)&v412, &v22[v21], &v457);
      v26 = v457;
      v24 = *v28;
      v29 = v414 + 1;
      goto LABEL_140;
    }
    v24 = *v28;
    v25 = (v18 - 1) & (((unsigned int)*v28 >> 9) ^ ((unsigned int)*v28 >> 4));
    v26 = (__int64 *)(v19 + 16LL * v25);
    v27 = *v26;
    if ( *v28 == *v26 )
      goto LABEL_4;
    v386 = 1;
    v134 = 0;
    while ( v27 != -4 )
    {
      if ( v134 || v27 != -8 )
        v26 = v134;
      v25 = (v18 - 1) & (v386 + v25);
      v27 = *(_QWORD *)(v19 + 16LL * v25);
      if ( v24 == v27 )
      {
        v26 = (__int64 *)(v19 + 16LL * v25);
        goto LABEL_4;
      }
      ++v386;
      v134 = v26;
      v26 = (__int64 *)(v19 + 16LL * v25);
    }
    if ( v134 )
      v26 = v134;
    ++v412;
    v29 = v414 + 1;
    if ( 4 * ((int)v414 + 1) >= 3 * v18 )
      goto LABEL_8;
    if ( v18 - (v29 + HIDWORD(v414)) <= v18 >> 3 )
      goto LABEL_9;
LABEL_140:
    LODWORD(v414) = v29;
    if ( *v26 != -4 )
      --HIDWORD(v414);
    *v26 = v24;
    v26[1] = 0;
LABEL_4:
    v26[1] = v21;
    v21 = (unsigned int)++v20;
    if ( v20 == a3 )
      break;
    v19 = v413;
    v18 = v415;
  }
  v406 = 0;
  v16 = a4;
  v30 = sub_22077B0(v394);
  v31 = v30 + v394;
  v405 = v30;
  v407 = v30 + v394;
  v32 = (char *)(v30 + v394);
  do
  {
    if ( v30 )
    {
      *(_DWORD *)(v30 + 8) = 0;
      *(_QWORD *)(v30 + 16) = 0;
      *(_QWORD *)(v30 + 24) = v30 + 8;
      *(_QWORD *)(v30 + 32) = v30 + 8;
      *(_QWORD *)(v30 + 40) = 0;
    }
    v30 += 48;
  }
  while ( v30 != v31 );
LABEL_14:
  v406 = v32;
  v416 = 0;
  v417 = 0;
  n = 8 * a5;
  v418 = 0;
  v419 = 0;
  v380 = &v16[a5];
  if ( v380 == v16 )
    goto LABEL_28;
  v389 = v16;
  v384 = 0;
  v354 = v16;
  while ( 2 )
  {
    v33 = (__int64)(*v389 + 3);
    v454 = *v389;
    v34 = v33;
    v35 = v33 + 8LL * (_QWORD)v454[1];
    if ( v33 != v35 )
    {
      do
      {
        if ( v415 )
        {
          v36 = *(_QWORD *)(*(_QWORD *)v34 + 8 * (1LL - *(unsigned int *)(*(_QWORD *)v34 + 8LL)));
          v37 = (v415 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
          v38 = (__int64 *)(v413 + 16LL * v37);
          v39 = *v38;
          if ( v36 == *v38 )
          {
LABEL_19:
            if ( v38 != (__int64 *)(v413 + 16LL * v415) )
            {
              v40 = v38[1];
              v457 = (__int64 *)v384;
              v41 = v405 + 48 * v40;
              v42 = sub_1534CB0(v41, (unsigned __int64 *)&v457);
              v44 = v43;
              if ( v43 )
              {
                v45 = 1;
                if ( !v42 && v43 != v41 + 8 )
                  v45 = *(_QWORD *)(v43 + 32) > (unsigned __int64)v384;
                v402 = v45;
                v46 = sub_22077B0(40);
                *(_QWORD *)(v46 + 32) = v457;
                sub_220F040(v402, v46, v44, v41 + 8);
                ++*(_QWORD *)(v41 + 40);
              }
            }
          }
          else
          {
            v123 = 1;
            while ( v39 != -4 )
            {
              v124 = v123 + 1;
              v37 = (v415 - 1) & (v123 + v37);
              v38 = (__int64 *)(v413 + 16LL * v37);
              v39 = *v38;
              if ( v36 == *v38 )
                goto LABEL_19;
              v123 = v124;
            }
          }
        }
        v34 += 8;
      }
      while ( v35 != v34 );
    }
    v47 = v419;
    if ( !v419 )
    {
      ++v416;
LABEL_172:
      v47 = 2 * v419;
LABEL_173:
      sub_18745B0((__int64)&v416, v47);
      sub_1872120((__int64)&v416, (__int64 *)&v454, &v457);
      v50 = v457;
      v48 = (__int64)v454;
      v133 = v418 + 1;
      goto LABEL_131;
    }
    v48 = (__int64)v454;
    v49 = (v419 - 1) & (((unsigned int)v454 >> 9) ^ ((unsigned int)v454 >> 4));
    v50 = (__int64 *)(v417 + 16LL * v49);
    v51 = (__int64 **)*v50;
    if ( v454 == (__int64 **)*v50 )
      goto LABEL_26;
    v131 = 1;
    v132 = 0;
    while ( v51 != (__int64 **)-8LL )
    {
      if ( v132 || v51 != (__int64 **)-16LL )
        v50 = v132;
      v49 = (v419 - 1) & (v131 + v49);
      v323 = (__int64 ***)(v417 + 16LL * v49);
      v51 = *v323;
      if ( v454 == *v323 )
      {
        v50 = (__int64 *)(v417 + 16LL * v49);
        goto LABEL_26;
      }
      ++v131;
      v132 = v50;
      v50 = (__int64 *)(v417 + 16LL * v49);
    }
    if ( v132 )
      v50 = v132;
    ++v416;
    v133 = v418 + 1;
    if ( 4 * ((int)v418 + 1) >= 3 * v419 )
      goto LABEL_172;
    if ( v419 - HIDWORD(v418) - v133 <= v419 >> 3 )
      goto LABEL_173;
LABEL_131:
    LODWORD(v418) = v133;
    if ( *v50 != -8 )
      --HIDWORD(v418);
    *v50 = v48;
    v50[1] = 0;
LABEL_26:
    ++v389;
    v50[1] = v384++;
    if ( v380 != v389 )
      continue;
    break;
  }
  v16 = v354;
  v32 = v406;
LABEL_28:
  v395 = &a15[a16];
  if ( v395 != a15 )
  {
    v403 = a15;
    v52 = (__int64)v32;
    v381 = v16;
    while ( 1 )
    {
      v53 = *v403;
      if ( v407 == v52 )
      {
        sub_187FC10(&v405, v52);
        v52 = (__int64)v406;
      }
      else
      {
        if ( v52 )
        {
          *(_DWORD *)(v52 + 8) = 0;
          *(_QWORD *)(v52 + 16) = 0;
          *(_QWORD *)(v52 + 24) = v52 + 8;
          *(_QWORD *)(v52 + 32) = v52 + 8;
          *(_QWORD *)(v52 + 40) = 0;
          v52 = (__int64)v406;
        }
        v52 += 48;
        v406 = (char *)v52;
      }
      v54 = (__int64 *)(v53 + 24);
      v55 = v53 + 24 + 8LL * *(_QWORD *)(v53 + 16);
      if ( (__int64 *)v55 != v54 )
        break;
LABEL_46:
      if ( v395 == ++v403 )
      {
        v16 = v381;
        v32 = (char *)v52;
        goto LABEL_48;
      }
    }
    while ( 2 )
    {
      v59 = *v54;
      if ( !v419 )
      {
        ++v416;
        goto LABEL_40;
      }
      v56 = (v419 - 1) & (((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4));
      v57 = (_QWORD *)(v417 + 16LL * v56);
      v58 = *v57;
      if ( v59 != *v57 )
      {
        v125 = 1;
        v126 = 0;
        while ( v58 != -8 )
        {
          if ( v58 == -16 && !v126 )
            v126 = v57;
          v56 = (v419 - 1) & (v125 + v56);
          v57 = (_QWORD *)(v417 + 16LL * v56);
          v58 = *v57;
          if ( v59 == *v57 )
            goto LABEL_37;
          ++v125;
        }
        if ( v126 )
          v57 = v126;
        ++v416;
        v61 = v418 + 1;
        if ( 4 * ((int)v418 + 1) >= 3 * v419 )
        {
LABEL_40:
          sub_18745B0((__int64)&v416, 2 * v419);
          if ( !v419 )
            goto LABEL_523;
          LODWORD(v60) = (v419 - 1) & (((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4));
          v61 = v418 + 1;
          v57 = (_QWORD *)(v417 + 16LL * (unsigned int)v60);
          v62 = *v57;
          if ( v59 != *v57 )
          {
            v174 = 1;
            v175 = 0;
            while ( v62 != -8 )
            {
              if ( v62 == -16 && !v175 )
                v175 = v57;
              v60 = (v419 - 1) & ((_DWORD)v60 + v174);
              v57 = (_QWORD *)(v417 + 16 * v60);
              v62 = *v57;
              if ( v59 == *v57 )
                goto LABEL_42;
              ++v174;
            }
            if ( v175 )
              v57 = v175;
          }
        }
        else if ( v419 - HIDWORD(v418) - v61 <= v419 >> 3 )
        {
          sub_18745B0((__int64)&v416, v419);
          if ( !v419 )
          {
LABEL_523:
            LODWORD(v418) = v418 + 1;
            BUG();
          }
          v127 = 0;
          LODWORD(v128) = (v419 - 1) & (((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4));
          v129 = 1;
          v61 = v418 + 1;
          v57 = (_QWORD *)(v417 + 16LL * (unsigned int)v128);
          v130 = *v57;
          if ( v59 != *v57 )
          {
            while ( v130 != -8 )
            {
              if ( v130 == -16 && !v127 )
                v127 = v57;
              v128 = (v419 - 1) & ((_DWORD)v128 + v129);
              v57 = (_QWORD *)(v417 + 16 * v128);
              v130 = *v57;
              if ( v59 == *v57 )
                goto LABEL_42;
              ++v129;
            }
            if ( v127 )
              v57 = v127;
          }
        }
LABEL_42:
        LODWORD(v418) = v61;
        if ( *v57 != -8 )
          --HIDWORD(v418);
        *v57 = v59;
        v57[1] = 0;
      }
LABEL_37:
      ++v54;
      sub_A19EB0((_QWORD *)(v52 - 48), v57 + 1);
      if ( (__int64 *)v55 == v54 )
      {
        v52 = (__int64)v406;
        goto LABEL_46;
      }
      continue;
    }
  }
LABEL_48:
  v63 = v405;
  sub_1880A60((__int64 *)&v457, v405, 0xAAAAAAAAAAAAAAABLL * ((__int64)&v32[-v405] >> 4));
  if ( v459 )
    sub_18894E0(v63, v32, v459, v458);
  else
    sub_1888AF0(v63, v32);
  v64 = v459;
  v65 = 48 * v458;
  v66 = v459 + 48 * v458;
  if ( v459 != v66 )
  {
    do
    {
      v67 = *(_QWORD *)(v64 + 16);
      while ( v67 )
      {
        sub_1876060(*(_QWORD *)(v67 + 24));
        v68 = v67;
        v67 = *(_QWORD *)(v67 + 16);
        j_j___libc_free_0(v68, 40);
      }
      v64 += 48;
    }
    while ( v66 != v64 );
    v66 = v459;
    v65 = 48 * v458;
  }
  j_j___libc_free_0(v66, v65);
  v428 = 0;
  v429 = 0;
  v430 = 0;
  v69 = (char *)sub_22077B0(24);
  v428 = v69;
  v430 = v69 + 24;
  if ( v69 )
  {
    *(_QWORD *)v69 = 0;
    *((_QWORD *)v69 + 1) = 0;
    *((_QWORD *)v69 + 2) = 0;
  }
  v429 = v69 + 24;
  if ( a5 > 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v431 = 0;
  v432 = 0;
  v433 = 0;
  if ( a5 )
  {
    v70 = (char *)sub_22077B0(n);
    v71 = &v70[n];
    v431 = v70;
    v433 = &v70[n];
    if ( v70 != &v70[n] )
      memset(v70, 0, n);
    v432 = v71;
    v72 = v405;
    v73 = v406;
    if ( v406 == (char *)v405 )
      goto LABEL_81;
  }
  else
  {
    v72 = v405;
    v73 = v406;
    if ( v406 == (char *)v405 )
      goto LABEL_64;
  }
  do
  {
    v74 = v72;
    v72 += 48;
    sub_187DE00((__int64 *)&v428, v74);
  }
  while ( v73 != (char *)v72 );
  if ( !a5 )
  {
LABEL_64:
    v75 = (__int64)v428;
    v76 = 1;
    v77 = v429;
    v404 = 0;
    v390 = 0;
    if ( v429 == v428 )
    {
      v446 = 0;
      v447 = 0;
      v448 = 0;
      v385 = 0;
      v353 = 0;
      v78 = sub_1632FA0((__int64)*a1);
      goto LABEL_66;
    }
    goto LABEL_84;
  }
LABEL_81:
  v76 = *((_BYTE *)**v16 + 16) == 3;
  v404 = (char *)sub_22077B0(n);
  v390 = &v404[n];
  if ( v404 != &v404[n] )
    memset(v404, 0, n);
  v75 = (__int64)v428;
  v77 = v429;
  if ( v429 != v428 )
  {
LABEL_84:
    v103 = v404;
    do
    {
      v104 = *(_QWORD **)v75;
      v105 = *(_QWORD **)(v75 + 8);
      if ( v105 != *(_QWORD **)v75 )
      {
        v106 = v103 + 8;
        do
        {
          v107 = v16[*v104];
          if ( (*((_BYTE *)*v107 + 16) == 3) != v76 )
            sub_16BD130("Type identifier may not contain both global variables and functions", 1u);
          ++v104;
          *((_QWORD *)v106 - 1) = v107;
          v103 = v106;
          v106 += 8;
        }
        while ( v105 != v104 );
      }
      v75 += 24;
    }
    while ( v77 != (char *)v75 );
  }
  v353 = v390 - v404;
  v397 = (v390 - v404) >> 3;
  v385 = v397;
  if ( !v76 )
  {
    v108 = *((unsigned int *)a1 + 6);
    if ( (unsigned int)v108 > 0x20 )
    {
      if ( (unsigned int)(v108 - 47) <= 1 )
      {
        v457 = 0;
        v458 = 0;
        v459 = 0;
        v109 = (__int64 **)v404;
        v460 = 0;
        if ( v404 == v390 )
        {
LABEL_404:
          v324 = (__int64 ***)sub_1599A20((__int64 **)a1[10]);
          sub_1880F60((__int64)a1, a2, a3, v324, (__int64)&v457, a6, a7, a8, a9, v325, v326, a12, a13);
          j___libc_free_0(v458);
          goto LABEL_191;
        }
        while ( 1 )
        {
          v113 = *v109;
          v114 = **v109;
          if ( !(unsigned __int8)sub_15E3650(v114, 0) )
            goto LABEL_97;
          v115 = sub_159C470((__int64)a1[11], (__int64)a1[13], 0);
          v454 = (__int64 **)sub_1624210(v115);
          v116 = (__int64 *)sub_15E0530(v114);
          v117 = sub_1627350(v116, (__int64 *)&v454, (__int64 *)1, 0, 1);
          sub_1627100(v114, "wasm.index", 0xAu, v117);
          v118 = a1[13];
          v119 = v460;
          a1[13] = (__int64 *)((char *)v118 + 1);
          if ( !v119 )
            break;
          v110 = (v119 - 1) & (((unsigned int)v113 >> 9) ^ ((unsigned int)v113 >> 4));
          v111 = (__int64 **)(v458 + 16LL * v110);
          v112 = *v111;
          if ( v113 != *v111 )
          {
            v340 = 1;
            v341 = 0;
            while ( v112 != (__int64 *)-8LL )
            {
              if ( !v341 && v112 == (__int64 *)-16LL )
                v341 = v111;
              v110 = (v119 - 1) & (v340 + v110);
              v111 = (__int64 **)(v458 + 16LL * v110);
              v112 = *v111;
              if ( v113 == *v111 )
                goto LABEL_96;
              ++v340;
            }
            if ( v341 )
              v111 = v341;
            v457 = (__int64 *)((char *)v457 + 1);
            v121 = v459 + 1;
            if ( 4 * ((int)v459 + 1) < 3 * v119 )
            {
              if ( v119 - HIDWORD(v459) - v121 <= v119 >> 3 )
              {
                sub_18745B0((__int64)&v457, v119);
                if ( !v460 )
                {
LABEL_522:
                  LODWORD(v459) = v459 + 1;
                  BUG();
                }
                v342 = 0;
                LODWORD(v343) = (v460 - 1) & (((unsigned int)v113 >> 9) ^ ((unsigned int)v113 >> 4));
                v344 = 1;
                v121 = v459 + 1;
                v111 = (__int64 **)(v458 + 16LL * (unsigned int)v343);
                v345 = *v111;
                if ( *v111 != v113 )
                {
                  while ( v345 != (__int64 *)-8LL )
                  {
                    if ( v345 == (__int64 *)-16LL && !v342 )
                      v342 = v111;
                    v343 = (v460 - 1) & ((_DWORD)v343 + v344);
                    v111 = (__int64 **)(v458 + 16 * v343);
                    v345 = *v111;
                    if ( v113 == *v111 )
                      goto LABEL_103;
                    ++v344;
                  }
                  if ( v342 )
                    v111 = v342;
                }
              }
              goto LABEL_103;
            }
LABEL_101:
            sub_18745B0((__int64)&v457, 2 * v119);
            if ( !v460 )
              goto LABEL_522;
            LODWORD(v120) = (v460 - 1) & (((unsigned int)v113 >> 9) ^ ((unsigned int)v113 >> 4));
            v121 = v459 + 1;
            v111 = (__int64 **)(v458 + 16LL * (unsigned int)v120);
            v122 = *v111;
            if ( v113 != *v111 )
            {
              v349 = 1;
              v350 = 0;
              while ( v122 != (__int64 *)-8LL )
              {
                if ( v122 == (__int64 *)-16LL && !v350 )
                  v350 = v111;
                v120 = (v460 - 1) & ((_DWORD)v120 + v349);
                v111 = (__int64 **)(v458 + 16 * v120);
                v122 = *v111;
                if ( v113 == *v111 )
                  goto LABEL_103;
                ++v349;
              }
              if ( v350 )
                v111 = v350;
            }
LABEL_103:
            LODWORD(v459) = v121;
            if ( *v111 != (__int64 *)-8LL )
              --HIDWORD(v459);
            *v111 = v113;
            v111[1] = 0;
          }
LABEL_96:
          v111[1] = v118;
LABEL_97:
          if ( v390 == (char *)++v109 )
            goto LABEL_404;
        }
        v457 = (__int64 *)((char *)v457 + 1);
        goto LABEL_101;
      }
LABEL_244:
      sub_16BD130("Unsupported architecture for jump tables", 1u);
    }
    v176 = 0x1A000000ALL;
    if ( !_bittest64(&v176, v108) )
      goto LABEL_244;
    v420 = 0;
    v421 = 0;
    v422 = 0;
    v423 = 0;
    if ( (_DWORD)v108 != 29 )
    {
      if ( (unsigned int)v108 > 0x1D )
      {
        if ( (unsigned int)(v108 - 31) > 1 )
          goto LABEL_244;
        v177 = 8;
LABEL_231:
        if ( v397 )
        {
          v178 = 0;
          v179 = 0;
          v180 = 0;
          v181 = 0;
          v182 = 0;
          while ( 1 )
          {
            v188 = (__int64 *)&v404[8 * v181];
            if ( !v179 )
              break;
            v183 = *v188;
            v184 = (v179 - 1) & (((unsigned int)*v188 >> 9) ^ ((unsigned int)*v188 >> 4));
            v185 = (__int64 *)(v180 + 16LL * v184);
            v186 = *v185;
            if ( *v185 == *v188 )
              goto LABEL_234;
            v333 = 1;
            v334 = 0;
            while ( v186 != -8 )
            {
              if ( v186 == -16 && !v334 )
                v334 = v185;
              v184 = (v179 - 1) & (v333 + v184);
              v185 = (__int64 *)(v180 + 16LL * v184);
              v186 = *v185;
              if ( v183 == *v185 )
                goto LABEL_234;
              ++v333;
            }
            if ( v334 )
              v185 = v334;
            ++v420;
            v190 = v422 + 1;
            if ( 4 * ((int)v422 + 1) >= 3 * v179 )
              goto LABEL_238;
            if ( v179 - (v190 + HIDWORD(v422)) <= v179 >> 3 )
            {
              sub_18745B0((__int64)&v420, v179);
              sub_1872120((__int64)&v420, v188, &v457);
              v183 = *v188;
              v185 = v457;
              v190 = v422 + 1;
            }
LABEL_240:
            LODWORD(v422) = v190;
            if ( *v185 != -8 )
              --HIDWORD(v422);
            *v185 = v183;
            v185[1] = 0;
LABEL_234:
            v187 = v178;
            v178 += v177;
            v185[1] = v187;
            v181 = (unsigned int)++v182;
            if ( v182 == v397 )
              goto LABEL_246;
            v180 = v421;
            v179 = v423;
          }
          ++v420;
LABEL_238:
          sub_18745B0((__int64)&v420, 2 * v179);
          if ( !v423 )
          {
            LODWORD(v422) = v422 + 1;
            BUG();
          }
          v189 = *v188;
          v190 = v422 + 1;
          v191 = (v423 - 1) & (((unsigned int)v189 >> 9) ^ ((unsigned int)v189 >> 4));
          v185 = (__int64 *)(v421 + 16LL * v191);
          v183 = *v185;
          if ( *v188 != *v185 )
          {
            v351 = 1;
            v352 = 0;
            while ( v183 != -8 )
            {
              if ( v183 == -16 && !v352 )
                v352 = v185;
              v191 = (v423 - 1) & (v351 + v191);
              v185 = (__int64 *)(v421 + 16LL * v191);
              v183 = *v185;
              if ( v189 == *v185 )
                goto LABEL_240;
              ++v351;
            }
            v183 = *v188;
            if ( v352 )
              v185 = v352;
          }
          goto LABEL_240;
        }
LABEL_246:
        LOWORD(v459) = 259;
        v192 = *a1;
        v457 = (__int64 *)".cfi.jumptable";
        v193 = (__int64 *)sub_1643270((_QWORD *)*v192);
        v194 = sub_16453E0(v193, 0);
        v195 = sub_1648B60(120);
        v355 = (__int64 ***)v195;
        if ( v195 )
          sub_15E2490(v195, v194, 8, (__int64)&v457, (__int64)v192);
        v196 = *((_DWORD *)a1 + 6);
        if ( v196 != 29 )
        {
          if ( v196 > 0x1D )
          {
            if ( v196 - 31 > 1 )
              goto LABEL_244;
            v197 = 8;
            goto LABEL_252;
          }
          if ( (v196 & 0xFFFFFFFD) != 1 )
            goto LABEL_244;
        }
        v197 = 4;
LABEL_252:
        v198 = sub_1645D80(a1[6], v197);
        v362 = sub_1645D80(v198, v397);
        v199 = sub_1647190(v362, 0);
        v358 = (__int64 ***)sub_15A4A70(v355, v199);
        sub_1880F60((__int64)a1, a2, a3, v358, (__int64)&v420, a6, a7, a8, a9, v200, v201, a12, a13);
        v203 = 0;
        v387 = 0;
        if ( v397 )
        {
          v204 = 0;
          do
          {
            v212 = *(_QWORD *)&v404[8 * v204];
            v213 = *(__int64 ****)v212;
            v214 = **(__int64 ****)v212;
            nc = *(_BYTE *)(v212 + 16);
            v215 = (__int64 *)sub_159C470((__int64)a1[12], 0, 0);
            v216 = (__int64)a1[12];
            v457 = v215;
            v458 = sub_159C470(v216, v204, 0);
            BYTE4(v454) = 0;
            v217 = (__int64 ***)sub_15A2E80((__int64)v362, (__int64)v358, &v457, 2u, 1u, (__int64)&v454, 0);
            v218 = (__int64 *)sub_15A4510(v217, v214, 0);
            if ( *(_BYTE *)(*(_QWORD *)&v404[8 * v204] + 17LL) )
            {
              if ( nc )
              {
                v205 = a1[1];
                v207 = (char *)sub_1649960((__int64)v213);
                v457 = &v459;
                if ( v207 )
                {
                  sub_18736F0((__int64 *)&v457, v207, (__int64)&v207[v206]);
                }
                else
                {
                  v458 = 0;
                  LOBYTE(v459) = 0;
                }
                sub_1880940((__int64)(v205 + 23), (__int64)&v457);
                if ( v457 != &v459 )
                  j_j___libc_free_0(v457, v459 + 1);
LABEL_259:
                v208 = (__int64)*a1;
                LOWORD(v459) = 257;
                v209 = (_BYTE *)sub_15E57E0(
                                  (__int64)v213[3],
                                  0,
                                  (_BYTE)v213[4] & 0xF,
                                  (__int64)&v457,
                                  (__int64)v218,
                                  v208);
                v210 = (_BYTE)v213[4] & 0x30 | v209[32] & 0xCF;
                v209[32] = v210;
                if ( (v210 & 0xFu) - 7 <= 1 || (v210 & 0x30) != 0 && (v210 & 0xF) != 9 )
                  v209[33] |= 0x40u;
                sub_164B7C0((__int64)v209, (__int64)v213);
                if ( (v209[23] & 0x20) != 0 )
                {
                  v454 = (__int64 **)sub_1649960((__int64)v209);
                  LOWORD(v459) = 773;
                  v457 = (__int64 *)&v454;
                  v455 = v309;
                  v458 = (__int64)".cfi";
                  sub_164B780((__int64)v213, (__int64 *)&v457);
                }
                sub_1887680((__int64)v213, (__int64)v209, 1, *(double *)a6.m128_u64, a7, a8);
                v211 = *((_BYTE *)v213 + 32);
                if ( (v211 & 0xFu) - 7 > 1 )
                {
                  *((_BYTE *)v213 + 32) = v211 & 0xCF | 0x10;
                  if ( (v211 & 0xF) != 9 )
                    *((_BYTE *)v213 + 33) |= 0x40u;
                }
                goto LABEL_264;
              }
              v222 = (__int64)*a1;
              v454 = (__int64 **)sub_1649960((__int64)v213);
              v455 = v223;
              v457 = (__int64 *)&v454;
              LOWORD(v459) = 773;
              v458 = (__int64)".cfi_jt";
              v224 = sub_15E57E0((__int64)v213[3], 0, 0, (__int64)&v457, (__int64)v218, v222);
              v225 = *(_BYTE *)(v224 + 32) & 0xCF | 0x10;
              *(_BYTE *)(v224 + 32) = v225;
              if ( (v225 & 0xF) != 9 )
                *(_BYTE *)(v224 + 33) |= 0x40u;
              v226 = a1[1];
              v228 = (char *)sub_1649960((__int64)v213);
              v457 = &v459;
              if ( v228 )
              {
                sub_18736F0((__int64 *)&v457, v228, (__int64)&v228[v227]);
              }
              else
              {
                v458 = 0;
                LOBYTE(v459) = 0;
              }
              sub_1880940((__int64)(v226 + 29), (__int64)&v457);
              if ( v457 != &v459 )
                j_j___libc_free_0(v457, v459 + 1);
            }
            else if ( nc )
            {
              goto LABEL_259;
            }
            v221 = (_BYTE)v213[4] & 0xF;
            if ( ((v221 + 14) & 0xFu) > 3 && ((v221 + 7) & 0xFu) > 1 )
              sub_1887680((__int64)v213, (__int64)v218, 0, *(double *)a6.m128_u64, a7, a8);
            else
              sub_1887A80(a1, (__int64)v213, v218, 0, a6, a7, a8, a9, v219, v220, a12, a13);
LABEL_264:
            v204 = (unsigned int)++v387;
          }
          while ( v387 != v397 );
        }
        v424[1] = 0;
        v424[0] = &v425;
        v426[0] = &v427;
        v425 = 0;
        v434 = &unk_49EFBE0;
        v440 = (__int64 *)&unk_49EFBE0;
        v454 = (__int64 **)v456;
        v455 = 0x1000000000LL;
        v426[1] = 0;
        v427 = 0;
        v438 = 1;
        v437 = 0;
        v436 = 0;
        v435 = 0;
        v439 = v424;
        v444 = 1;
        v443 = 0;
        v442 = 0;
        v441 = 0;
        v445 = v426;
        if ( (unsigned __int64)(2 * v397) > 0x10 )
          sub_16CD150((__int64)&v454, v456, 2 * v397, 8, v203, v202);
        v229 = *((_DWORD *)a1 + 6);
        if ( v229 != 1 && v229 != 29 )
          goto LABEL_288;
        v377 = 0;
        v249 = v404;
        if ( v404 == v390 )
          goto LABEL_405;
        v250 = 0;
        while ( 1 )
        {
          v252 = *(_BYTE *)(*(_QWORD *)v249 + 16LL);
          if ( !v252 )
            goto LABEL_314;
          v410[0] = sub_1560340((_QWORD *)(**(_QWORD **)v249 + 112LL), -1, "target-features", 0xFu);
          v251 = sub_155D460(v410, 0);
          if ( !v251 )
          {
            v457 = &v459;
            v458 = 0x600000000LL;
            v327 = (char *)sub_155D8B0(v410);
            v447 = v328;
            v446 = v327;
            sub_16D2880(&v446, (__int64)&v457, 44, -1, 1, v329);
            v330 = &v457[2 * (unsigned int)v458];
            if ( v457 != v330 )
            {
              v331 = v457;
              while ( 1 )
              {
                if ( v331[1] == 11 )
                {
                  v332 = (_QWORD *)*v331;
                  if ( *(_QWORD *)*v331 == 0x6D2D626D7568742DLL
                    && *((_WORD *)v332 + 4) == 25711
                    && *((_BYTE *)v332 + 10) == 101 )
                  {
                    goto LABEL_430;
                  }
                  if ( *v332 == 0x6D2D626D7568742BLL && *((_WORD *)v332 + 4) == 25711 && *((_BYTE *)v332 + 10) == 101 )
                    break;
                }
                v331 += 2;
                if ( v330 == v331 )
                  goto LABEL_410;
              }
              v251 = v252;
LABEL_430:
              if ( v457 != &v459 )
                _libc_free((unsigned __int64)v457);
              goto LABEL_311;
            }
LABEL_410:
            if ( v457 != &v459 )
              _libc_free((unsigned __int64)v457);
          }
          v251 = v229 == 29;
LABEL_311:
          if ( v251 )
          {
            ++v377;
            v249 += 8;
            if ( v390 == v249 )
              goto LABEL_315;
          }
          else
          {
LABEL_314:
            ++v250;
            v249 += 8;
            if ( v390 == v249 )
            {
LABEL_315:
              if ( v250 > v377 )
              {
                v229 = 1;
                goto LABEL_288;
              }
LABEL_405:
              v229 = 29;
LABEL_288:
              if ( v397 )
              {
                v230 = 0;
                v231 = 0;
                v232 = v455;
                while ( 1 )
                {
                  v236 = v232;
                  v237 = **(__int64 ***)&v404[8 * v231];
                  if ( (unsigned int)(v229 - 31) <= 1 )
                  {
                    v254 = v437;
                    if ( (unsigned __int64)(v436 - v437) <= 5 )
                    {
                      v310 = sub_16E7EE0((__int64)&v434, "jmp ${", 6u);
                      v236 = v232;
                      v255 = (void **)v310;
                    }
                    else
                    {
                      *(_DWORD *)v437 = 544238954;
                      v255 = &v434;
                      *(_WORD *)(v254 + 4) = 31524;
                      v437 += 6;
                    }
                    v256 = sub_16E7A90((__int64)v255, v236);
                    v257 = *(_QWORD **)(v256 + 24);
                    if ( *(_QWORD *)(v256 + 16) - (_QWORD)v257 <= 7u )
                    {
                      sub_16E7EE0(v256, ":c}@plt\n", 8u);
                    }
                    else
                    {
                      *v257 = 0xA746C70407D633ALL;
                      *(_QWORD *)(v256 + 24) += 8LL;
                    }
                    v258 = v437;
                    if ( (unsigned __int64)(v436 - v437) <= 0xE )
                    {
                      sub_16E7EE0((__int64)&v434, "int3\nint3\nint3\n", 0xFu);
                    }
                    else
                    {
                      *(_DWORD *)(v437 + 8) = 1852377651;
                      *(_QWORD *)v258 = 0x746E690A33746E69LL;
                      *(_WORD *)(v258 + 12) = 13172;
                      *(_BYTE *)(v258 + 14) = 10;
                      v437 += 15;
                    }
                  }
                  else
                  {
                    if ( (v229 & 0xFFFFFFFD) == 1 )
                    {
                      v253 = (_WORD *)v437;
                      if ( (unsigned __int64)(v436 - v437) <= 2 )
                      {
                        v311 = sub_16E7EE0((__int64)&v434, "b $", 3u);
                        v236 = v232;
                        v239 = (void **)v311;
                      }
                      else
                      {
                        *(_BYTE *)(v437 + 2) = 36;
                        v239 = &v434;
                        *v253 = 8290;
                        v437 += 3;
                      }
                    }
                    else
                    {
                      if ( v229 != 29 )
                        goto LABEL_244;
                      v238 = v437;
                      if ( (unsigned __int64)(v436 - v437) <= 4 )
                      {
                        v259 = sub_16E7EE0((__int64)&v434, "b.w $", 5u);
                        v236 = v232;
                        v239 = (void **)v259;
                      }
                      else
                      {
                        *(_DWORD *)v437 = 544681570;
                        v239 = &v434;
                        *(_BYTE *)(v238 + 4) = 36;
                        v437 += 5;
                      }
                    }
                    v240 = sub_16E7A90((__int64)v239, v236);
                    v241 = *(_BYTE **)(v240 + 24);
                    if ( *(_BYTE **)(v240 + 16) == v241 )
                    {
                      sub_16E7EE0(v240, "\n", 1u);
                    }
                    else
                    {
                      *v241 = 10;
                      ++*(_QWORD *)(v240 + 24);
                    }
                  }
                  v242 = "s";
                  if ( v232 )
                    v242 = ",s";
                  v243 = (char *)v242;
                  v244 = strlen(v242);
                  v245 = v443;
                  v246 = v244;
                  if ( v442 - v443 >= v244 )
                    break;
                  sub_16E7EE0((__int64)&v440, v243, v244);
                  v235 = (unsigned int)v455;
                  if ( (unsigned int)v455 >= HIDWORD(v455) )
                    goto LABEL_306;
LABEL_291:
                  v454[v235] = v237;
                  v232 = v455 + 1;
                  LODWORD(v455) = v455 + 1;
                  v231 = (unsigned int)++v230;
                  if ( v230 == v397 )
                    goto LABEL_327;
                }
                v234 = v244;
                if ( (_DWORD)v244 )
                {
                  v247 = 0;
                  do
                  {
                    v248 = v247++;
                    *(_BYTE *)(v245 + v248) = v243[v248];
                  }
                  while ( v247 < (unsigned int)v246 );
                }
                v443 += v246;
                v235 = (unsigned int)v455;
                if ( (unsigned int)v455 < HIDWORD(v455) )
                  goto LABEL_291;
LABEL_306:
                sub_16CD150((__int64)&v454, v456, 0, 8, v233, v234);
                v235 = (unsigned int)v455;
                goto LABEL_291;
              }
LABEL_327:
              v260 = *((_DWORD *)a1 + 6);
              if ( v260 != 29 )
              {
                if ( v260 > 0x1D )
                {
                  if ( v260 - 31 > 1 )
                    goto LABEL_244;
                  v261 = 8;
LABEL_331:
                  sub_15E4CC0((__int64)v355, v261);
                  if ( *((_DWORD *)a1 + 7) != 15 )
                    sub_15E0D50((__int64)v355, -1, 18);
                  if ( v229 == 1 )
                  {
                    v312 = (__int64)v355;
                    v335 = sub_15E0530((__int64)v355);
                    v316 = 11;
                    v317 = 15;
                    v318 = "-thumb-mode";
                    v319 = (__int64 *)v335;
                    v320 = "target-features";
                  }
                  else
                  {
                    if ( v229 != 29 )
                      goto LABEL_335;
                    v312 = (__int64)v355;
                    v313 = (__int64 *)sub_15E0530((__int64)v355);
                    v314 = sub_155D020(v313, "target-features", 0xFu, "+thumb-mode", 0xBu);
                    sub_15E0DA0((__int64)v355, -1, (__int64)v314);
                    v315 = sub_15E0530((__int64)v355);
                    v316 = 9;
                    v317 = 10;
                    v318 = "cortex-a8";
                    v319 = (__int64 *)v315;
                    v320 = "target-cpu";
                  }
                  v321 = sub_155D020(v319, v320, v317, v318, v316);
                  sub_15E0DA0(v312, -1, (__int64)v321);
LABEL_335:
                  sub_15E0D50((__int64)v355, -1, 30);
                  v457 = (__int64 *)"entry";
                  LOWORD(v459) = 259;
                  v262 = **a1;
                  v263 = (char *)sub_22077B0(64);
                  v264 = v263;
                  if ( v263 )
                    sub_157FB60(v263, v262, (__int64)&v457, (__int64)v355, 0);
                  v265 = (_QWORD *)sub_157E9C0((__int64)v264);
                  v268 = (unsigned int)v455;
                  v447 = v264;
                  v449 = v265;
                  v457 = &v459;
                  v446 = 0;
                  v450 = 0;
                  v451 = 0;
                  v452 = 0;
                  v453 = 0;
                  v448 = v264 + 40;
                  v458 = 0x1000000000LL;
                  if ( (unsigned int)v455 > 0x10 )
                  {
                    sub_16CD150((__int64)&v457, &v459, (unsigned int)v455, 8, v266, v267);
                    v268 = (unsigned int)v455;
                  }
                  v269 = v454;
                  v270 = (unsigned int)v458;
                  v271 = &v454[v268];
                  if ( v454 != v271 )
                  {
                    do
                    {
                      v272 = **v269;
                      if ( HIDWORD(v458) <= (unsigned int)v270 )
                      {
                        sub_16CD150((__int64)&v457, &v459, 0, 8, v266, v267);
                        v270 = (unsigned int)v458;
                      }
                      ++v269;
                      v457[v270] = v272;
                      v270 = (unsigned int)(v458 + 1);
                      LODWORD(v458) = v458 + 1;
                    }
                    while ( v271 != v269 );
                  }
                  if ( v443 != v441 )
                    sub_16E7BA0((__int64 *)&v440);
                  v273 = (_BYTE *)*v445;
                  v274 = v445[1];
                  if ( v437 != v435 )
                    sub_16E7BA0((__int64 *)&v434);
                  v275 = (_BYTE *)*v439;
                  v276 = v439[1];
                  v382 = v457;
                  v392 = (unsigned int)v458;
                  v277 = (__int64 *)sub_1643270(v449);
                  v278 = (__int64 **)sub_1644EA0(v277, v382, v392, 0);
                  v279 = sub_15EE570(v278, v275, v276, v273, v274, 1, 0, 0);
                  v409 = 257;
                  v383 = v279;
                  v379 = (__int64 *)v454;
                  v280 = v453;
                  v281 = *(_QWORD *)(*(_QWORD *)v279 + 24LL);
                  v282 = v452;
                  v411 = 257;
                  v393 = v281;
                  v283 = &v452[7 * v453];
                  if ( v452 == v283 )
                  {
                    v360 = (unsigned int)v455;
                    ni = v455 + 1;
                    v346 = sub_1648AB0(72, (int)v455 + 1, 16 * (int)v453);
                    v293 = ni;
                    v294 = v360;
                    v287 = v346;
                    if ( v346 )
                    {
                      v288 = v360;
                      v289 = (__int64)v287;
                      goto LABEL_354;
                    }
                  }
                  else
                  {
                    v284 = v452;
                    v285 = 0;
                    do
                    {
                      v286 = v284[5] - v284[4];
                      v284 += 7;
                      v285 += v286 >> 3;
                    }
                    while ( v283 != v284 );
                    v357 = (unsigned int)v455;
                    v359 = &v452[7 * v453];
                    nd = v455 + 1;
                    v287 = sub_1648AB0(72, (int)v455 + 1 + v285, 16 * (int)v453);
                    if ( v287 )
                    {
                      v288 = v357;
                      v289 = (__int64)v287;
                      v290 = v282;
                      LODWORD(v291) = 0;
                      do
                      {
                        v292 = v290[5] - v290[4];
                        v290 += 7;
                        v291 = (unsigned int)(v292 >> 3) + (unsigned int)v291;
                      }
                      while ( v359 != v290 );
                      v293 = v291 + nd;
                      v294 = v291 + v357;
LABEL_354:
                      ng = v288;
                      sub_15F1EA0((__int64)v287, **(_QWORD **)(v393 + 16), 54, (__int64)&v287[-3 * v294 - 3], v293, 0);
                      v287[7] = 0;
                      sub_15F5B40((__int64)v287, v393, v383, v379, ng, (__int64)v410, v282, v280);
LABEL_355:
                      v295 = *(_BYTE *)(*v287 + 8LL);
                      if ( v295 == 16 )
                        v295 = *(_BYTE *)(**(_QWORD **)(*v287 + 16LL) + 8LL);
                      if ( (unsigned __int8)(v295 - 1) <= 5u || *((_BYTE *)v287 + 16) == 76 )
                      {
                        v296 = v451;
                        if ( v450 )
                          sub_1625C10((__int64)v287, 3, v450);
                        sub_15F2440((__int64)v287, v296);
                      }
                      if ( v447 )
                      {
                        v297 = (unsigned __int64 *)v448;
                        sub_157E9D0((__int64)(v447 + 40), (__int64)v287);
                        v298 = v287[3];
                        v299 = *v297;
                        v287[4] = v297;
                        v299 &= 0xFFFFFFFFFFFFFFF8LL;
                        v287[3] = v299 | v298 & 7;
                        *(_QWORD *)(v299 + 8) = v287 + 3;
                        *v297 = *v297 & 7 | (unsigned __int64)(v287 + 3);
                      }
                      sub_164B780(v289, v408);
                      if ( v446 )
                      {
                        v410[0] = (__int64)v446;
                        sub_1623A60((__int64)v410, (__int64)v446, 2);
                        v300 = v287[6];
                        if ( v300 )
                          sub_161E7C0((__int64)(v287 + 6), v300);
                        v301 = (unsigned __int8 *)v410[0];
                        v287[6] = v410[0];
                        if ( v301 )
                          sub_1623210((__int64)v410, v301, (__int64)(v287 + 6));
                      }
                      v411 = 257;
                      v302 = sub_1648A60(56, 0);
                      v303 = v302;
                      if ( v302 )
                        sub_15F82A0((__int64)v302, (__int64)v449, 0);
                      if ( v447 )
                      {
                        v304 = (unsigned __int64 *)v448;
                        sub_157E9D0((__int64)(v447 + 40), (__int64)v303);
                        v305 = v303[3];
                        v306 = *v304;
                        v303[4] = v304;
                        v306 &= 0xFFFFFFFFFFFFFFF8LL;
                        v303[3] = v306 | v305 & 7;
                        *(_QWORD *)(v306 + 8) = v303 + 3;
                        *v304 = *v304 & 7 | (unsigned __int64)(v303 + 3);
                      }
                      sub_164B780((__int64)v303, v410);
                      if ( v446 )
                      {
                        v408[0] = (__int64)v446;
                        sub_1623A60((__int64)v408, (__int64)v446, 2);
                        v307 = v303[6];
                        if ( v307 )
                          sub_161E7C0((__int64)(v303 + 6), v307);
                        v308 = (unsigned __int8 *)v408[0];
                        v303[6] = v408[0];
                        if ( v308 )
                          sub_1623210((__int64)v408, v308, (__int64)(v303 + 6));
                      }
                      if ( v457 != &v459 )
                        _libc_free((unsigned __int64)v457);
                      if ( v446 )
                        sub_161E7C0((__int64)&v446, (__int64)v446);
                      if ( v454 != (__int64 **)v456 )
                        _libc_free((unsigned __int64)v454);
                      sub_16E7BC0((__int64 *)&v440);
                      sub_16E7BC0((__int64 *)&v434);
                      sub_2240A30(v426);
                      sub_2240A30(v424);
                      j___libc_free_0(v421);
                      goto LABEL_191;
                    }
                  }
                  v289 = 0;
                  v287 = 0;
                  goto LABEL_355;
                }
                if ( (v260 & 0xFFFFFFFD) != 1 )
                  goto LABEL_244;
              }
              v261 = 4;
              goto LABEL_331;
            }
          }
        }
      }
      if ( (v108 & 0xFFFFFFFD) != 1 )
        goto LABEL_244;
    }
    v177 = 4;
    goto LABEL_231;
  }
  v446 = 0;
  v447 = 0;
  v448 = 0;
  v135 = v404;
  v78 = sub_1632FA0((__int64)*a1);
  if ( v404 != v390 )
  {
    v79 = v447;
    do
    {
      v136 = **(_QWORD **)v135;
      v137 = *(__int64 **)(v136 - 24);
      v457 = v137;
      if ( v448 == v79 )
      {
        sub_12F5DA0((__int64)&v446, v79, &v457);
      }
      else
      {
        if ( v79 )
        {
          *(_QWORD *)v79 = v137;
          v79 = v447;
        }
        v447 = v79 + 8;
      }
      v138 = *(_QWORD *)(v136 + 24);
      v139 = sub_15A9FE0(v78, v138);
      v140 = 1;
      v141 = v139;
      while ( 2 )
      {
        switch ( *(_BYTE *)(v138 + 8) )
        {
          case 0:
          case 8:
          case 0xA:
          case 0xC:
          case 0x10:
            v149 = *(_QWORD *)(v138 + 32);
            v138 = *(_QWORD *)(v138 + 24);
            v140 *= v149;
            continue;
          case 1:
            v142 = 16;
            break;
          case 2:
            v142 = 32;
            break;
          case 3:
          case 9:
            v142 = 64;
            break;
          case 4:
            v142 = 80;
            break;
          case 5:
          case 6:
            v142 = 128;
            break;
          case 7:
            v399 = v140;
            v151 = sub_15A9520(v78, 0);
            v140 = v399;
            v142 = (unsigned int)(8 * v151);
            break;
          case 0xB:
            v142 = *(_DWORD *)(v138 + 8) >> 8;
            break;
          case 0xD:
            v401 = v140;
            v155 = (_QWORD *)sub_15A9930(v78, v138);
            v140 = v401;
            v142 = 8LL * *v155;
            break;
          case 0xE:
            v361 = v140;
            ne = *(_QWORD *)(v138 + 24);
            v376 = *(_QWORD *)(v138 + 32);
            v152 = sub_15A9FE0(v78, ne);
            v153 = ne;
            v400 = 1;
            v140 = v361;
            v154 = v152;
            while ( 2 )
            {
              switch ( *(_BYTE *)(v153 + 8) )
              {
                case 0:
                case 8:
                case 0xA:
                case 0xC:
                case 0x10:
                  v339 = v400 * *(_QWORD *)(v153 + 32);
                  v153 = *(_QWORD *)(v153 + 24);
                  v400 = v339;
                  continue;
                case 1:
                  v322 = 16;
                  goto LABEL_395;
                case 2:
                  v322 = 32;
                  goto LABEL_395;
                case 3:
                case 9:
                  v322 = 64;
                  goto LABEL_395;
                case 4:
                  v322 = 80;
                  goto LABEL_395;
                case 5:
                case 6:
                  v322 = 128;
                  goto LABEL_395;
                case 7:
                  v337 = sub_15A9520(v78, 0);
                  v140 = v361;
                  v322 = (unsigned int)(8 * v337);
                  goto LABEL_395;
                case 0xB:
                  v322 = *(_DWORD *)(v153 + 8) >> 8;
                  goto LABEL_395;
                case 0xD:
                  v338 = (_QWORD *)sub_15A9930(v78, v153);
                  v140 = v361;
                  v322 = 8LL * *v338;
                  goto LABEL_395;
                case 0xE:
                  nh = *(_QWORD *)(v153 + 32);
                  v336 = sub_12BE0A0(v78, *(_QWORD *)(v153 + 24));
                  v140 = v361;
                  v322 = 8 * nh * v336;
LABEL_395:
                  v142 = 8 * v154 * v376 * ((v154 + ((unsigned __int64)(v400 * v322 + 7) >> 3) - 1) / v154);
                  break;
                case 0xF:
                  JUMPOUT(0x188C310);
              }
              return result;
            }
          case 0xF:
            v398 = v140;
            v150 = sub_15A9520(v78, *(_DWORD *)(v138 + 8) >> 8);
            v140 = v398;
            v142 = (unsigned int)(8 * v150);
            break;
        }
        break;
      }
      v143 = v141 * ((v141 + ((unsigned __int64)(v142 * v140 + 7) >> 3) - 1) / v141);
      v144 = ((((((((((v143 - 1) | ((v143 - 1) >> 1)) >> 2) | (v143 - 1) | ((v143 - 1) >> 1)) >> 4)
                | (((v143 - 1) | ((v143 - 1) >> 1)) >> 2)
                | (v143 - 1)
                | ((v143 - 1) >> 1)) >> 8)
              | (((((v143 - 1) | ((v143 - 1) >> 1)) >> 2) | (v143 - 1) | ((v143 - 1) >> 1)) >> 4)
              | (((v143 - 1) | ((v143 - 1) >> 1)) >> 2)
              | (v143 - 1)
              | ((v143 - 1) >> 1)) >> 16)
            | (((((((v143 - 1) | ((v143 - 1) >> 1)) >> 2) | (v143 - 1) | ((v143 - 1) >> 1)) >> 4)
              | (((v143 - 1) | ((v143 - 1) >> 1)) >> 2)
              | (v143 - 1)
              | ((v143 - 1) >> 1)) >> 8)
            | (((((v143 - 1) | ((v143 - 1) >> 1)) >> 2) | (v143 - 1) | ((v143 - 1) >> 1)) >> 4)
            | (((v143 - 1) | ((v143 - 1) >> 1)) >> 2)
            | (v143 - 1)
            | ((v143 - 1) >> 1)) >> 32;
      v145 = (v144
            | (((((((((v143 - 1) | ((v143 - 1) >> 1)) >> 2) | (v143 - 1) | ((v143 - 1) >> 1)) >> 4)
                | (((v143 - 1) | ((v143 - 1) >> 1)) >> 2)
                | (v143 - 1)
                | ((v143 - 1) >> 1)) >> 8)
              | (((((v143 - 1) | ((v143 - 1) >> 1)) >> 2) | (v143 - 1) | ((v143 - 1) >> 1)) >> 4)
              | (((v143 - 1) | ((v143 - 1) >> 1)) >> 2)
              | (v143 - 1)
              | ((v143 - 1) >> 1)) >> 16)
            | (((((((v143 - 1) | ((v143 - 1) >> 1)) >> 2) | (v143 - 1) | ((v143 - 1) >> 1)) >> 4)
              | (((v143 - 1) | ((v143 - 1) >> 1)) >> 2)
              | (v143 - 1)
              | ((v143 - 1) >> 1)) >> 8)
            | (((((v143 - 1) | ((v143 - 1) >> 1)) >> 2) | (v143 - 1) | ((v143 - 1) >> 1)) >> 4)
            | (((v143 - 1) | ((v143 - 1) >> 1)) >> 2)
            | (v143 - 1)
            | ((v143 - 1) >> 1))
           - v143
           + 1;
      if ( v145 > 0x20 )
        v145 = ((v143 + 31) & 0xFFFFFFFFFFFFFFE0LL) - v143;
      v146 = sub_1645D80(a1[6], v145);
      v147 = (__int64 *)sub_1598F00((__int64 **)v146);
      v148 = v447;
      v457 = v147;
      if ( v447 == v448 )
      {
        sub_12F5DA0((__int64)&v446, v447, &v457);
        v79 = v447;
      }
      else
      {
        if ( v447 )
        {
          *(_QWORD *)v447 = v147;
          v148 = v447;
        }
        v79 = v148 + 8;
        v447 = v79;
      }
      v135 += 8;
    }
    while ( v390 != v135 );
    goto LABEL_67;
  }
LABEL_66:
  v79 = v447;
LABEL_67:
  v80 = (__int64 *)v446;
  if ( v446 != v79 )
  {
    v79 -= 8;
    v447 = v79;
  }
  v81 = (v79 - v446) >> 3;
  v82 = (__int64 **)sub_15942D0(**a1, (__int64)v446, v81, 0);
  v84 = (__int64 *)sub_159F090(v82, v80, v81, v83);
  v85 = *v84;
  v391 = v84;
  v86 = (__int64)v84;
  LOWORD(v459) = 257;
  v396 = (__int64 ***)sub_1648A60(88, 1u);
  if ( v396 )
    sub_15E51E0((__int64)v396, (__int64)*a1, v85, 1, 8, v86, (__int64)&v457, 0, 0, 0, 0);
  v375 = *v391;
  v87 = sub_15A9930(v78, *v391);
  v457 = 0;
  v458 = 0;
  v459 = 0;
  v460 = 0;
  if ( !v385 )
  {
    sub_1880F60((__int64)a1, a2, a3, v396, (__int64)&v457, a6, a7, a8, a9, v88, v89, a12, a13);
    goto LABEL_189;
  }
  v90 = v87;
  v91 = 0;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  while ( 1 )
  {
    v99 = (__int64 *)&v404[8 * v93];
    if ( !v91 )
    {
      v457 = (__int64 *)((char *)v457 + 1);
LABEL_78:
      na = v90;
      sub_18745B0((__int64)&v457, 2 * v91);
      if ( !v460 )
        goto LABEL_522;
      v95 = *v99;
      v90 = na;
      v100 = (v460 - 1) & (((unsigned int)v95 >> 9) ^ ((unsigned int)v95 >> 4));
      v97 = (__int64 *)(v458 + 16LL * v100);
      v101 = *v97;
      if ( *v97 == *v99 )
      {
LABEL_80:
        v102 = v459 + 1;
      }
      else
      {
        v347 = 1;
        v348 = 0;
        while ( v101 != -8 )
        {
          if ( v101 == -16 && !v348 )
            v348 = v97;
          v100 = (v460 - 1) & (v347 + v100);
          v97 = (__int64 *)(v458 + 16LL * v100);
          v101 = *v97;
          if ( v95 == *v97 )
            goto LABEL_80;
          ++v347;
        }
        v102 = v459 + 1;
        if ( v348 )
          v97 = v348;
      }
      goto LABEL_218;
    }
    v95 = *v99;
    v96 = (v91 - 1) & (((unsigned int)*v99 >> 9) ^ ((unsigned int)*v99 >> 4));
    v97 = (__int64 *)(v92 + 16LL * v96);
    v98 = *v97;
    if ( *v97 == *v99 )
      goto LABEL_74;
    nb = 1;
    v173 = 0;
    while ( v98 != -8 )
    {
      if ( v98 == -16 && !v173 )
        v173 = v97;
      v96 = (v91 - 1) & (nb + v96);
      v97 = (__int64 *)(v92 + 16LL * v96);
      v98 = *v97;
      if ( v95 == *v97 )
        goto LABEL_74;
      ++nb;
    }
    if ( v173 )
      v97 = v173;
    v457 = (__int64 *)((char *)v457 + 1);
    v102 = v459 + 1;
    if ( 4 * ((int)v459 + 1) >= 3 * v91 )
      goto LABEL_78;
    if ( v91 - (v102 + HIDWORD(v459)) <= v91 >> 3 )
    {
      nf = v90;
      sub_18745B0((__int64)&v457, v91);
      sub_1872120((__int64)&v457, v99, &v454);
      v95 = *v99;
      v97 = (__int64 *)v454;
      v90 = nf;
      v102 = v459 + 1;
    }
LABEL_218:
    LODWORD(v459) = v102;
    if ( *v97 != -8 )
      --HIDWORD(v459);
    *v97 = v95;
    v97[1] = 0;
LABEL_74:
    v97[1] = *(_QWORD *)(v90 + 16LL * v94++ + 16);
    v93 = v94;
    if ( v385 == v94 )
      break;
    v92 = v458;
    v91 = v460;
  }
  v156 = 0;
  sub_1880F60((__int64)a1, a2, a3, v396, (__int64)&v457, a6, a7, a8, a9, v88, v89, a12, a13);
  v157 = 0;
  do
  {
    v160 = **(_QWORD **)&v404[8 * v157];
    v161 = (__int64 *)sub_159C470((__int64)a1[9], 0, 0);
    v162 = (__int64)a1[9];
    v440 = v161;
    v441 = sub_159C470(v162, (unsigned int)(2 * v156), 0);
    v163 = *v391;
    BYTE4(v454) = 0;
    v164 = sub_15A2E80(v163, (__int64)v396, &v440, 2u, 0, (__int64)&v454, 0);
    v456[0] = 257;
    v165 = sub_15E57E0(
             *(_QWORD *)(*(_QWORD *)(v375 + 16) + 8LL * (unsigned int)(2 * v156)),
             0,
             *(_BYTE *)(v160 + 32) & 0xF,
             (__int64)&v454,
             v164,
             (__int64)*a1);
    v166 = *(_BYTE *)(v160 + 32) & 0x30 | *(_BYTE *)(v165 + 32) & 0xCF;
    *(_BYTE *)(v165 + 32) = v166;
    if ( (v166 & 0xFu) - 7 <= 1 || (v166 & 0x30) != 0 && (v166 & 0xF) != 9 )
      *(_BYTE *)(v165 + 33) |= 0x40u;
    sub_164B7C0(v165, v160);
    sub_164D160(v160, v165, a6, a7, a8, a9, v158, v159, a12, a13);
    sub_15E55B0(v160);
    v157 = (unsigned int)++v156;
  }
  while ( v385 != v156 );
LABEL_189:
  j___libc_free_0(v458);
  if ( v446 )
    j_j___libc_free_0(v446, v448 - v446);
LABEL_191:
  if ( v404 )
    j_j___libc_free_0(v404, v353);
  if ( v431 )
    j_j___libc_free_0(v431, v433 - v431);
  v167 = v429;
  v168 = v428;
  if ( v429 != v428 )
  {
    do
    {
      if ( *(_QWORD *)v168 )
        j_j___libc_free_0(*(_QWORD *)v168, *((_QWORD *)v168 + 2) - *(_QWORD *)v168);
      v168 += 24;
    }
    while ( v167 != v168 );
    v168 = v428;
  }
  if ( v168 )
    j_j___libc_free_0(v168, v430 - v168);
  j___libc_free_0(v417);
  v169 = v406;
  v170 = v405;
  if ( v406 != (char *)v405 )
  {
    do
    {
      v171 = *(_QWORD *)(v170 + 16);
      v170 += 48;
      sub_1876060(v171);
    }
    while ( v169 != (char *)v170 );
    v170 = v405;
  }
  if ( v170 )
    j_j___libc_free_0(v170, v407 - v170);
  return j___libc_free_0(v413);
}
