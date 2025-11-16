// Function: sub_1C70910
// Address: 0x1c70910
//
__int64 __fastcall sub_1C70910(
        __int64 *a1,
        __int64 a2,
        __m128 si128,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r12
  __int64 *v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rbx
  unsigned __int64 *v14; // rdi
  unsigned __int64 *v15; // rax
  __int64 i; // r14
  unsigned __int64 v17; // rax
  _QWORD *v18; // r9
  unsigned __int64 v19; // rsi
  _QWORD *v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rdx
  _QWORD *v24; // rsi
  _QWORD *v25; // rdi
  _QWORD *v26; // r15
  int v27; // r8d
  int v28; // r9d
  __int64 v29; // rbx
  _DWORD *v30; // rax
  __int64 v31; // rsi
  _BYTE *v32; // rdx
  unsigned __int64 v33; // r12
  _DWORD *j; // rdx
  __int64 v35; // rax
  __int64 v36; // rdi
  __int64 v37; // rax
  __int64 *v38; // rax
  __int64 *v39; // rsi
  __int64 v40; // rcx
  __int64 v41; // rdx
  _QWORD *v42; // r14
  _QWORD *v43; // rbx
  __int64 v44; // rdi
  unsigned __int64 v45; // rax
  __int64 v46; // rsi
  __int64 v47; // rcx
  __int64 v48; // rdi
  __int64 *v49; // rbx
  __int64 *v50; // r12
  __int64 v51; // rdi
  __int64 v53; // rdx
  __int64 v54; // r12
  unsigned __int64 v55; // rdx
  __int64 v56; // r9
  __int64 v57; // rcx
  __int64 *v58; // rax
  __int64 *v59; // r8
  __int64 v60; // rdi
  __int64 v61; // r14
  char v62; // al
  _QWORD *v63; // r15
  unsigned __int64 v64; // rax
  unsigned __int8 v65; // dl
  unsigned __int64 v66; // rax
  unsigned __int64 v67; // rdx
  unsigned __int64 v68; // r10
  __int64 v69; // r13
  __int64 v70; // r12
  unsigned __int64 *v71; // r14
  _QWORD *v72; // r10
  __int64 v73; // r15
  _DWORD *v74; // rcx
  unsigned __int64 v75; // rbx
  int v76; // eax
  _QWORD *v77; // rdx
  _QWORD *v78; // r8
  _QWORD *v79; // rax
  __int64 v80; // rdi
  __int64 v81; // rsi
  char v82; // al
  _QWORD *v83; // r8
  _QWORD *v84; // rax
  __int64 v85; // rcx
  __int64 v86; // rdx
  __int64 v87; // rax
  _QWORD *v88; // rax
  _QWORD *v89; // rdx
  _BOOL8 v90; // rdi
  unsigned __int64 v91; // rbx
  __int64 v92; // r12
  __int64 v93; // r15
  int v94; // r13d
  int *v95; // r14
  int v96; // r10d
  int *v97; // r9
  int *v98; // rax
  __int64 v99; // rcx
  __int64 v100; // rdx
  __int64 v101; // rax
  int *v102; // rax
  __int64 v103; // rdx
  _BOOL8 v104; // rdi
  int *v105; // r11
  unsigned __int64 v106; // rdi
  _QWORD *v107; // r14
  _QWORD *v108; // rdx
  unsigned __int64 v109; // rdi
  _QWORD *v110; // r8
  _QWORD *v111; // rax
  __int64 v112; // rsi
  __int64 v113; // rcx
  _QWORD *v114; // r8
  _QWORD *v115; // rax
  __int64 v116; // rsi
  __int64 v117; // rcx
  _QWORD *v118; // r12
  _QWORD *v119; // rax
  __int64 v120; // r14
  _QWORD *v121; // rsi
  __int64 v122; // rcx
  __int64 v123; // rax
  _QWORD *v124; // rbx
  int k; // r12d
  _QWORD *v126; // rax
  _QWORD *v127; // r8
  __int64 v128; // rax
  __int64 v129; // rax
  _QWORD *v130; // rax
  _QWORD *v131; // rdx
  _BOOL8 v132; // rdi
  int *v133; // rax
  __int64 v134; // rcx
  __int64 v135; // rdx
  __int64 v136; // rax
  int *v137; // rax
  __int64 v138; // rdx
  _BOOL8 v139; // rdi
  __int64 v140; // rdi
  __int64 v141; // rdi
  _QWORD *v142; // r12
  _QWORD *v143; // rbx
  _QWORD *v144; // rax
  __int64 v145; // rcx
  __int64 v146; // rdx
  __int64 v147; // r13
  unsigned __int64 v148; // rax
  unsigned __int64 v149; // rdx
  unsigned __int8 v150; // cl
  unsigned __int64 v151; // rax
  __int64 v152; // rsi
  _QWORD *v153; // rax
  _QWORD *v154; // rdx
  __int64 v155; // r12
  void *v156; // rbx
  __int64 *v157; // r14
  __int64 v158; // rax
  _QWORD *v159; // rax
  __int64 *v160; // rdx
  __int64 *v161; // r14
  _BOOL4 v162; // ebx
  __int64 v163; // rax
  __int64 v164; // rbx
  __int64 v165; // r14
  _QWORD *v166; // rbx
  unsigned __int8 v167; // al
  __int64 *v168; // r14
  __int64 *v169; // rbx
  __int64 v170; // rdi
  _QWORD *v171; // rbx
  __int64 v172; // r14
  int v173; // r12d
  _QWORD *v174; // rax
  unsigned __int64 v175; // rsi
  _QWORD *v176; // r13
  __int64 v177; // rcx
  __int64 v178; // rdx
  __int64 v179; // rax
  unsigned __int64 v180; // rcx
  _QWORD *v181; // rax
  _QWORD *v182; // rdx
  _BOOL8 v183; // rdi
  __int64 v184; // rax
  int v185; // edi
  _QWORD *v186; // rax
  __int64 *v187; // rdx
  __int64 v188; // rdx
  int v189; // edx
  __int64 v190; // rax
  _QWORD *v191; // rbx
  _QWORD *v192; // r12
  __int64 v193; // rax
  __int64 v194; // rdx
  __int64 v195; // rax
  double v196; // xmm4_8
  double v197; // xmm5_8
  __int64 v198; // rdi
  __int64 v199; // rsi
  unsigned __int64 v200; // rax
  _QWORD *v201; // r12
  void *v202; // rsi
  _QWORD *v203; // rax
  __int64 v204; // rcx
  __int64 v205; // rdx
  _QWORD *v206; // r13
  int v207; // eax
  __int64 v208; // r12
  const char *v209; // r15
  size_t v210; // rdx
  size_t v211; // r14
  size_t v212; // rdx
  const char *v213; // rdi
  size_t v214; // rbx
  __int64 v215; // rbx
  const char *v216; // r14
  size_t v217; // rdx
  size_t v218; // r15
  size_t v219; // rdx
  const char *v220; // rdi
  size_t v221; // rbx
  int v222; // eax
  bool v223; // al
  _QWORD *v224; // rax
  _QWORD *v225; // rsi
  __int64 v226; // rcx
  __int64 v227; // rdx
  __int64 v228; // r12
  __int64 v229; // rdx
  _QWORD *v230; // rsi
  __int64 v231; // rdi
  __m128 *v232; // rax
  __int64 v233; // rdi
  _BYTE *v234; // rax
  __int64 v235; // r12
  const char *v236; // rax
  size_t v237; // rdx
  __m128 *v238; // rdi
  char *v239; // rsi
  size_t v240; // r14
  unsigned __int64 v241; // rax
  __m128i v242; // xmm0
  __int64 v243; // rdi
  void *v244; // rax
  void *v245; // rax
  __int64 v246; // rdi
  __m128 *v247; // rax
  __int64 v248; // rax
  _QWORD *v249; // rsi
  _QWORD *v250; // rax
  _QWORD *v251; // rdx
  _BOOL8 v252; // rdi
  __int64 v253; // rax
  _QWORD *v254; // r15
  _QWORD *v255; // r12
  int v256; // eax
  __int64 v257; // r14
  const char *v258; // rax
  size_t v259; // rdx
  size_t v260; // rbx
  const char *v261; // rsi
  size_t v262; // rdx
  const char *v263; // rdi
  size_t v264; // r14
  int v265; // eax
  __int64 v266; // r14
  const char *v267; // rax
  size_t v268; // rdx
  size_t v269; // rbx
  const char *v270; // rsi
  size_t v271; // rdx
  const char *v272; // rdi
  size_t v273; // r14
  _QWORD *v274; // rdi
  __int64 v275; // rax
  __int64 v276; // r14
  _QWORD *v277; // r12
  __int64 v278; // rax
  int v279; // r13d
  __int64 v280; // rax
  _QWORD *v281; // rbx
  __int64 v282; // rdx
  __int64 v283; // rcx
  unsigned __int64 v284; // r15
  _QWORD *v285; // rbx
  _QWORD *v286; // rax
  __int64 v287; // rcx
  __int64 v288; // rdx
  __int64 v289; // rax
  _QWORD *v290; // rax
  _QWORD *v291; // rdx
  _BOOL8 v292; // rdi
  _QWORD *v293; // r13
  __int64 v294; // rax
  _QWORD *v295; // rbx
  __int64 v296; // rdx
  __int64 v297; // rax
  __int64 v298; // r12
  const char *v299; // rax
  size_t v300; // rdx
  __m128 *v301; // rdi
  char *v302; // rsi
  size_t v303; // r14
  unsigned __int64 v304; // rax
  __m128i v305; // xmm0
  __int64 v306; // rdi
  _BYTE *v307; // rax
  __int64 v308; // rax
  _QWORD *v309; // rdi
  int v310; // r8d
  _QWORD *v311; // rdi
  int v312; // edx
  __int64 v313; // rdx
  unsigned __int64 *v314; // rdi
  __int64 *v315; // rdx
  _QWORD *v316; // r12
  _QWORD *v317; // rbx
  __int64 v318; // rsi
  _QWORD *v319; // r12
  const char *v320; // rax
  size_t v321; // rdx
  void *v322; // rdi
  char *v323; // rsi
  size_t v324; // r13
  unsigned __int64 v325; // rax
  _BOOL4 v326; // r8d
  __int64 v327; // rax
  int v328; // esi
  _QWORD *v329; // rcx
  unsigned int v330; // edx
  __int64 v331; // rdi
  __int64 v332; // rax
  __int64 v333; // rax
  int v334; // esi
  _QWORD *v335; // rcx
  unsigned int v336; // edx
  __int64 v337; // rdi
  __int64 v338; // rax
  __int64 v339; // r12
  __m128 *v340; // rax
  const char *v341; // rax
  size_t v342; // rdx
  _BYTE *v343; // rdi
  char *v344; // rsi
  _BYTE *v345; // rax
  size_t v346; // r13
  __int64 v347; // rax
  _QWORD *v348; // [rsp+20h] [rbp-220h]
  unsigned __int8 v349; // [rsp+30h] [rbp-210h]
  _QWORD *v350; // [rsp+30h] [rbp-210h]
  _QWORD *v351; // [rsp+30h] [rbp-210h]
  _QWORD *v352; // [rsp+30h] [rbp-210h]
  char v353; // [rsp+47h] [rbp-1F9h]
  _QWORD *v354; // [rsp+48h] [rbp-1F8h]
  _QWORD *v355; // [rsp+50h] [rbp-1F0h]
  _QWORD *v356; // [rsp+50h] [rbp-1F0h]
  __int64 v357; // [rsp+50h] [rbp-1F0h]
  __int64 *v358; // [rsp+50h] [rbp-1F0h]
  _QWORD *v359; // [rsp+58h] [rbp-1E8h]
  int *v360; // [rsp+58h] [rbp-1E8h]
  _QWORD *v361; // [rsp+58h] [rbp-1E8h]
  int v362; // [rsp+58h] [rbp-1E8h]
  unsigned __int64 v363; // [rsp+58h] [rbp-1E8h]
  _QWORD *v364; // [rsp+60h] [rbp-1E0h]
  __int64 v365; // [rsp+60h] [rbp-1E0h]
  int *v366; // [rsp+60h] [rbp-1E0h]
  _QWORD *v367; // [rsp+60h] [rbp-1E0h]
  __int64 v368; // [rsp+60h] [rbp-1E0h]
  int *v369; // [rsp+60h] [rbp-1E0h]
  _QWORD *v370; // [rsp+60h] [rbp-1E0h]
  int *v371; // [rsp+60h] [rbp-1E0h]
  int *v372; // [rsp+60h] [rbp-1E0h]
  _QWORD *v373; // [rsp+60h] [rbp-1E0h]
  char v374; // [rsp+60h] [rbp-1E0h]
  int v375; // [rsp+60h] [rbp-1E0h]
  _BOOL4 v376; // [rsp+60h] [rbp-1E0h]
  __int64 v377; // [rsp+68h] [rbp-1D8h]
  _QWORD *v378; // [rsp+68h] [rbp-1D8h]
  __int64 v379; // [rsp+68h] [rbp-1D8h]
  int *v380; // [rsp+68h] [rbp-1D8h]
  int *v381; // [rsp+68h] [rbp-1D8h]
  _QWORD *v382; // [rsp+68h] [rbp-1D8h]
  _QWORD *v383; // [rsp+68h] [rbp-1D8h]
  _QWORD *v384; // [rsp+70h] [rbp-1D0h]
  unsigned int v385; // [rsp+70h] [rbp-1D0h]
  __int64 v386; // [rsp+70h] [rbp-1D0h]
  unsigned __int64 v387; // [rsp+70h] [rbp-1D0h]
  unsigned __int64 v389; // [rsp+80h] [rbp-1C0h] BYREF
  unsigned __int64 v390; // [rsp+88h] [rbp-1B8h] BYREF
  void *v391; // [rsp+90h] [rbp-1B0h] BYREF
  _QWORD v392[2]; // [rsp+98h] [rbp-1A8h] BYREF
  __int64 v393; // [rsp+A8h] [rbp-198h]
  __int64 v394; // [rsp+B0h] [rbp-190h]
  __int64 v395; // [rsp+C0h] [rbp-180h] BYREF
  int v396; // [rsp+C8h] [rbp-178h] BYREF
  int *v397; // [rsp+D0h] [rbp-170h]
  int *v398; // [rsp+D8h] [rbp-168h]
  int *v399; // [rsp+E0h] [rbp-160h]
  __int64 v400; // [rsp+E8h] [rbp-158h]
  void **v401; // [rsp+F0h] [rbp-150h] BYREF
  __int64 v402; // [rsp+F8h] [rbp-148h] BYREF
  __int64 v403; // [rsp+100h] [rbp-140h]
  __int64 v404; // [rsp+108h] [rbp-138h]
  __int64 *v405; // [rsp+110h] [rbp-130h]
  __int64 v406; // [rsp+118h] [rbp-128h]
  __int64 v407; // [rsp+120h] [rbp-120h] BYREF
  __int64 v408; // [rsp+128h] [rbp-118h]
  _QWORD *v409; // [rsp+130h] [rbp-110h]
  __int64 v410; // [rsp+138h] [rbp-108h]
  __int64 v411; // [rsp+140h] [rbp-100h]
  unsigned __int64 v412; // [rsp+148h] [rbp-F8h]
  _QWORD *v413; // [rsp+150h] [rbp-F0h]
  __int64 v414; // [rsp+158h] [rbp-E8h]
  __int64 v415; // [rsp+160h] [rbp-E0h]
  __int64 *v416; // [rsp+168h] [rbp-D8h]
  _BYTE *v417; // [rsp+170h] [rbp-D0h] BYREF
  __int64 v418; // [rsp+178h] [rbp-C8h]
  _BYTE v419[64]; // [rsp+180h] [rbp-C0h] BYREF
  unsigned __int64 *v420; // [rsp+1C0h] [rbp-80h] BYREF
  __int64 v421; // [rsp+1C8h] [rbp-78h]
  __int64 v422; // [rsp+1D0h] [rbp-70h]
  __int64 v423; // [rsp+1D8h] [rbp-68h]
  __int64 v424; // [rsp+1E0h] [rbp-60h]
  unsigned __int64 v425; // [rsp+1E8h] [rbp-58h]
  __int64 v426; // [rsp+1F0h] [rbp-50h]
  __int64 v427; // [rsp+1F8h] [rbp-48h]
  __int64 v428; // [rsp+200h] [rbp-40h]
  __int64 *v429; // [rsp+208h] [rbp-38h]

  v10 = a2 + 24;
  *a1 = sub_1632FA0(a2);
  v409 = 0;
  v410 = 0;
  v411 = 0;
  v413 = 0;
  v414 = 0;
  v415 = 0;
  v408 = 8;
  v407 = sub_22077B0(64);
  v11 = (__int64 *)(v407 + 24);
  v12 = sub_22077B0(512);
  *(_QWORD *)(v407 + 24) = v12;
  v412 = (unsigned __int64)v11;
  v416 = v11;
  v13 = *(_QWORD *)(a2 + 32);
  v410 = v12;
  v411 = v12 + 512;
  v414 = v12;
  v415 = v12 + 512;
  v409 = (_QWORD *)v12;
  v413 = (_QWORD *)v12;
  if ( v13 != a2 + 24 )
  {
    v384 = a1 + 20;
    while ( 1 )
    {
      v14 = (unsigned __int64 *)(v13 - 56);
      if ( !v13 )
        v14 = 0;
      v401 = (void **)v14;
      if ( !sub_15E4F60((__int64)v14) )
        break;
LABEL_3:
      v13 = *(_QWORD *)(v13 + 8);
      if ( v10 == v13 )
      {
        if ( !dword_4FBD480 )
          goto LABEL_24;
        goto LABEL_363;
      }
    }
    if ( (unsigned __int8)sub_1C6EC20((__int64)a1, (unsigned __int64)v401) )
    {
      v154 = v413;
      if ( v413 != (_QWORD *)(v415 - 8) )
      {
        v15 = (unsigned __int64 *)v401;
        if ( v413 )
        {
          *v413 = v401;
          v154 = v413;
        }
        v413 = v154 + 1;
LABEL_9:
        for ( i = v15[1]; i; i = *(_QWORD *)(i + 8) )
        {
          v17 = (unsigned __int64)sub_1648700(i);
          if ( *(_BYTE *)(v17 + 16) == 78 )
          {
            v18 = v384;
            v19 = *(_QWORD *)(*(_QWORD *)((v17 & 0xFFFFFFFFFFFFFFF8LL) + 40) + 56LL);
            v20 = (_QWORD *)a1[21];
            v417 = (_BYTE *)v19;
            if ( !v20 )
              goto LABEL_20;
            do
            {
              while ( 1 )
              {
                v21 = v20[2];
                v22 = v20[3];
                if ( v20[4] >= v19 )
                  break;
                v20 = (_QWORD *)v20[3];
                if ( !v22 )
                  goto LABEL_18;
              }
              v18 = v20;
              v20 = (_QWORD *)v20[2];
            }
            while ( v21 );
LABEL_18:
            if ( v384 == v18 || v18[4] > v19 )
            {
LABEL_20:
              v420 = (unsigned __int64 *)&v417;
              v18 = sub_1C6F7D0(a1 + 19, v18, &v420);
            }
            v377 = (__int64)(v18 + 5);
            v24 = sub_1C6FA30((__int64)(v18 + 5), (__int64 *)&v401);
            if ( v23 )
              sub_1C6E9C0(v377, (__int64)v24, v23, (__int64 *)&v401);
          }
        }
        goto LABEL_3;
      }
      sub_1C6FE60(&v407, &v401);
    }
    v15 = (unsigned __int64 *)v401;
    goto LABEL_9;
  }
  if ( !dword_4FBD480 )
  {
LABEL_471:
    v349 = 0;
    goto LABEL_65;
  }
LABEL_363:
  v231 = (__int64)sub_16E8CB0();
  v232 = *(__m128 **)(v231 + 24);
  if ( *(_QWORD *)(v231 + 16) - (_QWORD)v232 <= 0x18u )
  {
    v231 = sub_16E7EE0(v231, "Initial work list size : ", 0x19u);
  }
  else
  {
    si128 = (__m128)_mm_load_si128((const __m128i *)&xmmword_42D1F90);
    v232[1].m128_i8[8] = 32;
    v232[1].m128_u64[0] = 0x3A20657A69732074LL;
    *v232 = si128;
    *(_QWORD *)(v231 + 24) += 25LL;
  }
  v233 = sub_16E7A90(
           v231,
           ((v411 - (__int64)v409) >> 3)
         + (((__int64)v413 - v414) >> 3)
         + ((((__int64)((__int64)v416 - v412) >> 3) - 1) << 6));
  v234 = *(_BYTE **)(v233 + 24);
  if ( *(_BYTE **)(v233 + 16) == v234 )
  {
    sub_16E7EE0(v233, "\n", 1u);
  }
  else
  {
    *v234 = 10;
    ++*(_QWORD *)(v233 + 24);
  }
LABEL_24:
  v349 = 0;
  v25 = v413;
  if ( v409 == v413 )
    goto LABEL_471;
  do
  {
    while ( 1 )
    {
      if ( (_QWORD *)v414 == v25 )
      {
        v389 = *(_QWORD *)(*(v416 - 1) + 504);
        j_j___libc_free_0(v25, 512);
        v26 = (_QWORD *)v389;
        v53 = *--v416 + 512;
        v414 = *v416;
        v415 = v53;
        v413 = (_QWORD *)(v414 + 504);
      }
      else
      {
        v26 = (_QWORD *)*(v25 - 1);
        v413 = v25 - 1;
        v389 = (unsigned __int64)v26;
      }
      if ( !v26[12] )
        goto LABEL_42;
      v353 = sub_1C6EAF0((__int64)a1, (__int64)v26);
      if ( !v353 )
        goto LABEL_41;
      v29 = v389;
      v30 = v419;
      v31 = 0x1000000000LL;
      v417 = v419;
      v32 = v419;
      v418 = 0x1000000000LL;
      v33 = *(_QWORD *)(v389 + 96);
      if ( v33 )
      {
        if ( v33 > 0x10 )
        {
          v31 = (__int64)v419;
          sub_16CD150((__int64)&v417, v419, v33, 4, v27, v28);
          v32 = v417;
          v30 = &v417[4 * (unsigned int)v418];
        }
        for ( j = &v32[4 * v33]; j != v30; ++v30 )
        {
          if ( v30 )
            *v30 = 0;
        }
        v29 = v389;
        LODWORD(v418) = v33;
        v35 = (unsigned int)v33;
        if ( (*(_BYTE *)(v389 + 18) & 1) == 0 )
          goto LABEL_74;
        goto LABEL_73;
      }
      v54 = *(unsigned __int16 *)(v389 + 18);
      if ( (v54 & 1) == 0 )
      {
        v61 = *(_QWORD *)(v389 + 8);
        if ( !v61 )
          goto LABEL_130;
LABEL_88:
        v63 = a1 + 14;
LABEL_89:
        v64 = (unsigned __int64)sub_1648700(v61);
        v65 = *(_BYTE *)(v64 + 16);
        if ( v65 <= 0x17u )
          goto LABEL_92;
        if ( v65 == 78 )
        {
          v66 = v64 | 4;
        }
        else
        {
          v66 = v64 & 0xFFFFFFFFFFFFFFFBLL;
          if ( v65 != 29 )
          {
LABEL_92:
            v67 = 0;
            v379 = *(_QWORD *)(MEMORY[0x28] + 56LL);
            goto LABEL_93;
          }
        }
        v67 = v66 & 0xFFFFFFFFFFFFFFF8LL;
        v31 = *(_QWORD *)(*(_QWORD *)((v66 & 0xFFFFFFFFFFFFFFF8LL) + 40) + 56LL);
        v379 = v31;
        v68 = (v66 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((v66 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
        if ( (v66 & 4) != 0 )
        {
          if ( (v54 & 1) == 0 )
            goto LABEL_94;
          goto LABEL_204;
        }
LABEL_93:
        v68 = v67 - 24LL * (*(_DWORD *)(v67 + 20) & 0xFFFFFFF);
        if ( (v54 & 1) == 0 )
        {
LABEL_94:
          v69 = *(_QWORD *)(v29 + 88);
          v70 = 0;
          if ( !(_DWORD)v418 )
            goto LABEL_126;
          v386 = v61;
          v71 = (unsigned __int64 *)v68;
          v72 = v63;
          v73 = 4LL * (unsigned int)v418;
          while ( 1 )
          {
            v74 = &v417[v70];
            if ( *(_DWORD *)&v417[v70] == 2000 )
              goto LABEL_124;
            v75 = *v71;
            if ( *v71 == v69 )
              goto LABEL_124;
            if ( *(_BYTE *)(*(_QWORD *)v75 + 8LL) != 15 )
              goto LABEL_123;
            v76 = *(_DWORD *)(*(_QWORD *)v75 + 8LL) >> 8;
            LODWORD(v420) = v76;
            if ( !v76 )
            {
              v77 = (_QWORD *)a1[15];
              if ( !v77 )
                goto LABEL_107;
              v78 = v72;
              v79 = (_QWORD *)a1[15];
              do
              {
                while ( 1 )
                {
                  v80 = v79[2];
                  v81 = v79[3];
                  if ( v79[4] >= v75 )
                    break;
                  v79 = (_QWORD *)v79[3];
                  if ( !v81 )
                    goto LABEL_105;
                }
                v78 = v79;
                v79 = (_QWORD *)v79[2];
              }
              while ( v80 );
LABEL_105:
              if ( v72 != v78 && v78[4] <= v75 )
              {
                v127 = v72;
                do
                {
                  while ( 1 )
                  {
                    v31 = v77[2];
                    v128 = v77[3];
                    if ( v77[4] >= v75 )
                      break;
                    v77 = (_QWORD *)v77[3];
                    if ( !v128 )
                      goto LABEL_192;
                  }
                  v127 = v77;
                  v77 = (_QWORD *)v77[2];
                }
                while ( v31 );
LABEL_192:
                if ( v72 == v127 || v127[4] > v75 )
                {
                  v356 = v72;
                  v361 = v127;
                  v129 = sub_22077B0(48);
                  *(_QWORD *)(v129 + 32) = v75;
                  *(_DWORD *)(v129 + 40) = 0;
                  v368 = v129;
                  v130 = sub_1C70330(a1 + 13, v361, (unsigned __int64 *)(v129 + 32));
                  if ( v131 )
                  {
                    v132 = v130 || v356 == v131 || v131[4] > v75;
                    v31 = v368;
                    sub_220F040(v132, v368, v131, v356);
                    v127 = (_QWORD *)v368;
                    v72 = v356;
                    ++a1[18];
                  }
                  else
                  {
                    v140 = v368;
                    v31 = 48;
                    v370 = v130;
                    j_j___libc_free_0(v140, 48);
                    v72 = v356;
                    v127 = v370;
                  }
                  v74 = &v417[v70];
                }
                v76 = *((_DWORD *)v127 + 10);
                LODWORD(v420) = v76;
              }
              else
              {
LABEL_107:
                v31 = v75;
                v364 = v72;
                v82 = sub_1CA0940(v379, v75, &v420, *a1, a1 + 1, a1 + 7);
                v72 = v364;
                if ( !v82 )
                {
                  *(_DWORD *)&v417[v70] = 2000;
                  goto LABEL_124;
                }
                v83 = v364;
                v84 = (_QWORD *)a1[15];
                if ( !v84 )
                  goto LABEL_115;
                do
                {
                  while ( 1 )
                  {
                    v85 = v84[2];
                    v86 = v84[3];
                    if ( v84[4] >= v75 )
                      break;
                    v84 = (_QWORD *)v84[3];
                    if ( !v86 )
                      goto LABEL_113;
                  }
                  v83 = v84;
                  v84 = (_QWORD *)v84[2];
                }
                while ( v85 );
LABEL_113:
                if ( v83 == v364 || v83[4] > v75 )
                {
LABEL_115:
                  v355 = v364;
                  v359 = v83;
                  v87 = sub_22077B0(48);
                  *(_QWORD *)(v87 + 32) = v75;
                  *(_DWORD *)(v87 + 40) = 0;
                  v365 = v87;
                  v88 = sub_1C70330(a1 + 13, v359, (unsigned __int64 *)(v87 + 32));
                  if ( v89 )
                  {
                    v90 = v355 == v89 || v88 || v75 < v89[4];
                    v31 = v365;
                    sub_220F040(v90, v365, v89, v355);
                    v83 = (_QWORD *)v365;
                    v72 = v355;
                    ++a1[18];
                  }
                  else
                  {
                    v141 = v365;
                    v31 = 48;
                    v373 = v88;
                    j_j___libc_free_0(v141, 48);
                    v72 = v355;
                    v83 = v373;
                  }
                }
                v76 = (int)v420;
                *((_DWORD *)v83 + 10) = (_DWORD)v420;
                v74 = &v417[v70];
              }
            }
            if ( *v74 == 1000 )
            {
              *v74 = v76;
              goto LABEL_124;
            }
            if ( *v74 != v76 )
LABEL_123:
              *v74 = 2000;
LABEL_124:
            v70 += 4;
            v71 += 3;
            v69 += 40;
            if ( v73 == v70 )
            {
              v61 = v386;
              v63 = v72;
LABEL_126:
              v29 = v389;
              v61 = *(_QWORD *)(v61 + 8);
              LOWORD(v54) = *(_WORD *)(v389 + 18);
              if ( !v61 )
              {
                v62 = v54 & 1;
                goto LABEL_128;
              }
              goto LABEL_89;
            }
          }
        }
LABEL_204:
        v387 = v68;
        sub_15E08E0(v29, v31);
        v68 = v387;
        goto LABEL_94;
      }
LABEL_73:
      sub_15E08E0(v29, v31);
      v35 = (unsigned int)v418;
LABEL_74:
      v55 = *(_QWORD *)(v29 + 88);
      if ( (_DWORD)v35 )
      {
        v56 = 4 * v35;
        v57 = 0;
        while ( 1 )
        {
          *(_DWORD *)&v417[v57] = 1000;
          if ( *(_BYTE *)(*(_QWORD *)v55 + 8LL) != 15 )
            goto LABEL_76;
          v58 = (__int64 *)a1[9];
          if ( !v58 )
            goto LABEL_77;
          v59 = a1 + 8;
          do
          {
            while ( 1 )
            {
              v60 = v58[2];
              v31 = v58[3];
              if ( v58[4] >= v55 )
                break;
              v58 = (__int64 *)v58[3];
              if ( !v31 )
                goto LABEL_84;
            }
            v59 = v58;
            v58 = (__int64 *)v58[2];
          }
          while ( v60 );
LABEL_84:
          if ( a1 + 8 == v59 )
            goto LABEL_77;
          if ( v59[4] > v55 )
          {
            v57 += 4;
            v55 += 40LL;
            if ( v56 == v57 )
              break;
          }
          else
          {
LABEL_76:
            *(_DWORD *)&v417[v57] = 2000;
LABEL_77:
            v57 += 4;
            v55 += 40LL;
            if ( v56 == v57 )
              break;
          }
        }
      }
      v29 = v389;
      LOWORD(v54) = *(_WORD *)(v389 + 18);
      v61 = *(_QWORD *)(v389 + 8);
      v62 = v54 & 1;
      if ( v61 )
        goto LABEL_88;
LABEL_128:
      if ( v62 )
        sub_15E08E0(v29, v31);
LABEL_130:
      v91 = *(_QWORD *)(v29 + 88);
      v396 = 0;
      v398 = &v396;
      v399 = &v396;
      v397 = 0;
      v400 = 0;
      if ( !(_DWORD)v418 )
      {
        v36 = 0;
        goto LABEL_39;
      }
      v92 = 4LL * (unsigned int)v418;
      v93 = 0;
      v94 = 0;
      v95 = (int *)(a1 + 8);
      do
      {
        v96 = *(_DWORD *)&v417[v93];
        if ( v96 != 1000 && v96 != 2000 && *(_QWORD *)(v91 + 8) )
        {
          if ( (*(_BYTE *)(v389 + 32) & 0xFu) - 7 > 1 )
          {
            v133 = v397;
            v97 = &v396;
            if ( !v397 )
              goto LABEL_212;
            do
            {
              while ( 1 )
              {
                v134 = *((_QWORD *)v133 + 2);
                v135 = *((_QWORD *)v133 + 3);
                if ( *((_QWORD *)v133 + 4) >= v91 )
                  break;
                v133 = (int *)*((_QWORD *)v133 + 3);
                if ( !v135 )
                  goto LABEL_210;
              }
              v97 = v133;
              v133 = (int *)*((_QWORD *)v133 + 2);
            }
            while ( v134 );
LABEL_210:
            if ( v97 == &v396 || *((_QWORD *)v97 + 4) > v91 )
            {
LABEL_212:
              v362 = *(_DWORD *)&v417[v93];
              v369 = v97;
              v136 = sub_22077B0(48);
              *(_QWORD *)(v136 + 32) = v91;
              *(_DWORD *)(v136 + 40) = 0;
              v381 = (int *)v136;
              v137 = (int *)sub_1C70670(&v395, v369, (unsigned __int64 *)(v136 + 32));
              if ( v138 )
              {
                v139 = v137 || &v396 == (int *)v138 || *(_QWORD *)(v138 + 32) > v91;
                sub_220F040(v139, v381, v138, &v396);
                ++v400;
                v97 = v381;
                v96 = v362;
              }
              else
              {
                v372 = v137;
                j_j___libc_free_0(v381, 48);
                v96 = v362;
                v97 = v372;
              }
            }
          }
          else
          {
            v97 = (int *)(a1 + 8);
            v98 = (int *)a1[9];
            if ( !v98 )
              goto LABEL_143;
            do
            {
              while ( 1 )
              {
                v99 = *((_QWORD *)v98 + 2);
                v100 = *((_QWORD *)v98 + 3);
                if ( *((_QWORD *)v98 + 4) >= v91 )
                  break;
                v98 = (int *)*((_QWORD *)v98 + 3);
                if ( !v100 )
                  goto LABEL_141;
              }
              v97 = v98;
              v98 = (int *)*((_QWORD *)v98 + 2);
            }
            while ( v99 );
LABEL_141:
            if ( v97 == v95 || *((_QWORD *)v97 + 4) > v91 )
            {
LABEL_143:
              v360 = (int *)&v417[v93];
              v366 = v97;
              v101 = sub_22077B0(48);
              *(_QWORD *)(v101 + 32) = v91;
              *(_DWORD *)(v101 + 40) = 0;
              v380 = (int *)v101;
              v102 = (int *)sub_1C704D0(a1 + 7, v366, (unsigned __int64 *)(v101 + 32));
              if ( v103 )
              {
                v104 = v102 || v95 == (int *)v103 || *(_QWORD *)(v103 + 32) > v91;
                sub_220F040(v104, v380, v103, v95);
                v97 = v380;
                v105 = v360;
                ++a1[12];
              }
              else
              {
                v371 = v102;
                j_j___libc_free_0(v380, 48);
                v105 = v360;
                v97 = v371;
              }
              v96 = *v105;
            }
          }
          v97[10] = v96;
          ++v94;
        }
        v93 += 4;
        v91 += 40LL;
      }
      while ( v92 != v93 );
      v36 = (__int64)v397;
      if ( !v94 )
      {
LABEL_39:
        sub_1C6EEC0(v36);
        if ( v417 != v419 )
          _libc_free((unsigned __int64)v417);
LABEL_41:
        v26 = (_QWORD *)v389;
        goto LABEL_42;
      }
      if ( !dword_4FBD480 )
        goto LABEL_153;
      v235 = (__int64)sub_16E8CB0();
      v236 = sub_1649960(v389);
      v238 = *(__m128 **)(v235 + 24);
      v239 = (char *)v236;
      v240 = v237;
      v241 = *(_QWORD *)(v235 + 16) - (_QWORD)v238;
      if ( v237 > v241 )
      {
        v308 = sub_16E7EE0(v235, v239, v237);
        v238 = *(__m128 **)(v308 + 24);
        v235 = v308;
        if ( *(_QWORD *)(v308 + 16) - (_QWORD)v238 > 0x24u )
          goto LABEL_372;
      }
      else
      {
        if ( v237 )
        {
          memcpy(v238, v239, v237);
          v253 = *(_QWORD *)(v235 + 16);
          v238 = (__m128 *)(v240 + *(_QWORD *)(v235 + 24));
          *(_QWORD *)(v235 + 24) = v238;
          v241 = v253 - (_QWORD)v238;
        }
        if ( v241 > 0x24 )
        {
LABEL_372:
          v242 = _mm_load_si128((const __m128i *)&xmmword_42D1FA0);
          v238[2].m128_i32[0] = 543515489;
          v238[2].m128_i8[4] = 40;
          *v238 = (__m128)v242;
          si128 = (__m128)_mm_load_si128((const __m128i *)&xmmword_42D1FB0);
          v238[1] = si128;
          *(_QWORD *)(v235 + 24) += 37LL;
          goto LABEL_373;
        }
      }
      v235 = sub_16E7EE0(v235, " : changed in argument memory space (", 0x25u);
LABEL_373:
      v243 = sub_16E7AB0(v235, v94);
      v244 = *(void **)(v243 + 24);
      if ( *(_QWORD *)(v243 + 16) - (_QWORD)v244 <= 0xBu )
      {
        sub_16E7EE0(v243, " arguments)\n", 0xCu);
      }
      else
      {
        qmemcpy(v244, " arguments)\n", 12);
        *(_QWORD *)(v243 + 24) += 12LL;
      }
LABEL_153:
      v106 = v389;
      v107 = a1 + 7;
      v367 = a1 + 20;
      if ( (*(_BYTE *)(v389 + 32) & 0xFu) - 7 <= 1 )
        goto LABEL_154;
      v155 = (__int64)v398;
      if ( v398 != &v396 )
      {
        while ( 1 )
        {
          v156 = *(void **)(v155 + 32);
          LODWORD(v402) = 0;
          v403 = 0;
          v404 = (__int64)&v402;
          v405 = &v402;
          v406 = 0;
          v420 = 0;
          v422 = 0;
          v423 = 0;
          v424 = 0;
          v425 = 0;
          v426 = 0;
          v427 = 0;
          v428 = 0;
          v429 = 0;
          v421 = 8;
          v420 = (unsigned __int64 *)sub_22077B0(64);
          v157 = (__int64 *)((char *)v420 + ((4 * v421 - 4) & 0xFFFFFFFFFFFFFFF8LL));
          v158 = sub_22077B0(512);
          *v157 = v158;
          v425 = (unsigned __int64)v157;
          v423 = v158;
          v424 = v158 + 512;
          v429 = v157;
          v427 = v158;
          v428 = v158 + 512;
          v422 = v158;
          v426 = v158;
          v391 = v156;
          sub_1C70060((__int64 *)&v420, &v391);
          v391 = v156;
          v159 = sub_1819210((__int64)&v401, (unsigned __int64 *)&v391);
          v161 = v160;
          if ( v160 )
          {
            v162 = v159 || v160 == &v402 || (unsigned __int64)v156 < v160[4];
            v163 = sub_22077B0(40);
            *(_QWORD *)(v163 + 32) = v391;
            sub_220F040(v162, v163, v161, &v402);
            ++v406;
          }
LABEL_259:
          while ( v426 != v422 )
          {
            if ( v426 == v427 )
            {
              v164 = *(_QWORD *)(*(v429 - 1) + 504);
              j_j___libc_free_0(v426, 512);
              v188 = *--v429 + 512;
              v427 = *v429;
              v428 = v188;
              v426 = v427 + 504;
            }
            else
            {
              v164 = *(_QWORD *)(v426 - 8);
              v426 -= 8;
            }
            v165 = *(_QWORD *)(v164 + 8);
            if ( v165 )
            {
              while ( 1 )
              {
                v166 = sub_1648700(v165);
                v167 = *((_BYTE *)v166 + 16);
                if ( v167 > 0x17u )
                  break;
LABEL_297:
                v165 = *(_QWORD *)(v165 + 8);
                if ( !v165 )
                  goto LABEL_259;
              }
              if ( ((v167 - 54) & 0xFA) != 0 && v167 != 25 )
              {
                if ( v167 != 78
                  || (v184 = *(v166 - 3), *(_BYTE *)(v184 + 16))
                  || (*(_BYTE *)(v184 + 33) & 0x20) == 0
                  || (v185 = *(_DWORD *)(v184 + 36), (v185 & 0xFFFFFFFD) != 0x85)
                  && v185 != 137
                  && !(unsigned __int8)sub_1C30260(v185) )
                {
                  v391 = v166;
                  v186 = sub_1819210((__int64)&v401, (unsigned __int64 *)&v391);
                  if ( v187 )
                  {
                    v326 = v186 || v187 == &v402 || (unsigned __int64)v166 < v187[4];
                    v358 = v187;
                    v376 = v326;
                    v327 = sub_22077B0(40);
                    *(_QWORD *)(v327 + 32) = v391;
                    sub_220F040(v376, v327, v358, &v402);
                    ++v406;
                    v391 = v166;
                    sub_1C70060((__int64 *)&v420, &v391);
                  }
                  goto LABEL_297;
                }
              }
              if ( v420 )
              {
                v168 = (__int64 *)v425;
                v169 = v429 + 1;
                if ( (unsigned __int64)(v429 + 1) > v425 )
                {
                  v374 = v353;
                  goto LABEL_269;
                }
                j_j___libc_free_0(v420, 8 * v421);
              }
              sub_1C6F430(v403);
LABEL_302:
              v189 = dword_4FBD3A0;
              v375 = *((_DWORD *)a1 + 50);
              *((_DWORD *)a1 + 50) = v375 + 1;
              if ( v375 + 1 <= v189 || (v36 = (__int64)v397, v189 == -1) )
              {
                v420 = 0;
                LODWORD(v423) = 128;
                v190 = sub_22077B0(0x2000);
                v422 = 0;
                v191 = (_QWORD *)v190;
                v421 = v190;
                v402 = 2;
                v192 = (_QWORD *)(v190 + ((unsigned __int64)(unsigned int)v423 << 6));
                v403 = 0;
                v404 = -8;
                v401 = (void **)&unk_49E6B50;
                v405 = 0;
                if ( (_QWORD *)v190 != v192 )
                {
                  v193 = -8;
                  do
                  {
                    if ( v191 )
                    {
                      v194 = v402;
                      v191[2] = 0;
                      v191[3] = v193;
                      v191[1] = v194 & 6;
                      if ( v193 != -8 && v193 != 0 && v193 != -16 )
                      {
                        sub_1649AC0(v191 + 1, v194 & 0xFFFFFFFFFFFFFFF8LL);
                        v193 = v404;
                      }
                      *v191 = &unk_49E6B50;
                      v191[4] = v405;
                    }
                    v191 += 8;
                  }
                  while ( v192 != v191 );
                  v401 = (void **)&unk_49EE2B0;
                  if ( v193 != -16 && v193 != -8 && v193 )
                    sub_1649B30(&v402);
                }
                LOBYTE(v428) = 0;
                BYTE1(v429) = 1;
                v195 = sub_1AB6FF0(v389, (__int64)&v420, 0);
                v198 = v389;
                v199 = v195;
                v390 = v195;
                v367 = a1 + 20;
                *(_WORD *)(v195 + 32) = *(_WORD *)(v195 + 32) & 0xBFC0 | 0x4007;
                v357 = *(_QWORD *)(v198 + 8);
                if ( !v357 )
                  goto LABEL_420;
LABEL_318:
                v200 = (unsigned __int64)sub_1648700(v357);
                if ( *(_BYTE *)(v200 + 16) != 78 )
                  goto LABEL_317;
                v201 = a1 + 20;
                v202 = *(void **)(*(_QWORD *)((v200 & 0xFFFFFFFFFFFFFFF8LL) + 40) + 56LL);
                v203 = (_QWORD *)a1[21];
                v391 = v202;
                if ( !v203 )
                  goto LABEL_326;
                do
                {
                  while ( 1 )
                  {
                    v204 = v203[2];
                    v205 = v203[3];
                    if ( v203[4] >= (unsigned __int64)v202 )
                      break;
                    v203 = (_QWORD *)v203[3];
                    if ( !v205 )
                      goto LABEL_324;
                  }
                  v201 = v203;
                  v203 = (_QWORD *)v203[2];
                }
                while ( v204 );
LABEL_324:
                if ( v367 == v201 || v201[4] > (unsigned __int64)v202 )
                {
LABEL_326:
                  v401 = &v391;
                  v201 = sub_1C6F7D0(a1 + 19, v201, (unsigned __int64 **)&v401);
                }
                v206 = (_QWORD *)v201[7];
                v350 = v201 + 6;
                if ( !v206 )
                {
                  v223 = v353;
                  v354 = v201 + 6;
                  goto LABEL_345;
                }
                v354 = v201 + 6;
                v348 = v201;
                while ( 2 )
                {
                  while ( 2 )
                  {
                    v208 = v206[4];
                    v209 = sub_1649960(v389);
                    v211 = v210;
                    v213 = sub_1649960(v208);
                    v214 = v212;
                    if ( v211 < v212 )
                    {
                      if ( !v211 )
                        goto LABEL_338;
                      v207 = memcmp(v213, v209, v211);
                      if ( v207 )
                      {
LABEL_337:
                        if ( v207 < 0 )
                        {
LABEL_333:
                          v206 = (_QWORD *)v206[3];
                          if ( !v206 )
                            goto LABEL_344;
                          continue;
                        }
                        goto LABEL_338;
                      }
LABEL_332:
                      if ( v211 > v214 )
                        goto LABEL_333;
                      goto LABEL_338;
                    }
                    break;
                  }
                  if ( v212 )
                  {
                    v207 = memcmp(v213, v209, v212);
                    if ( v207 )
                      goto LABEL_337;
                  }
                  if ( v211 != v214 )
                    goto LABEL_332;
LABEL_338:
                  v215 = v389;
                  v216 = sub_1649960(v206[4]);
                  v218 = v217;
                  v220 = sub_1649960(v215);
                  v221 = v219;
                  if ( v219 > v218 )
                  {
                    if ( !v218 )
                      goto LABEL_391;
                    v222 = memcmp(v220, v216, v218);
                    if ( v222 )
                      goto LABEL_390;
LABEL_342:
                    if ( v221 >= v218 )
                      goto LABEL_391;
                  }
                  else
                  {
                    if ( !v219 || (v222 = memcmp(v220, v216, v219)) == 0 )
                    {
                      if ( v221 == v218 )
                        goto LABEL_391;
                      goto LABEL_342;
                    }
LABEL_390:
                    if ( v222 >= 0 )
                    {
LABEL_391:
                      v201 = v348;
                      v254 = (_QWORD *)v206[2];
                      if ( !v206[3] )
                        goto LABEL_408;
                      v255 = (_QWORD *)v206[3];
                      while ( 1 )
                      {
                        v257 = v389;
                        v258 = sub_1649960(v255[4]);
                        v260 = v259;
                        v261 = v258;
                        v263 = sub_1649960(v257);
                        v264 = v262;
                        if ( v262 <= v260 )
                        {
                          if ( !v262 || (v256 = memcmp(v263, v261, v262)) == 0 )
                          {
                            if ( v264 == v260 )
                              goto LABEL_403;
LABEL_396:
                            if ( v264 >= v260 )
                              goto LABEL_403;
                            goto LABEL_397;
                          }
                        }
                        else
                        {
                          if ( !v260 )
                            goto LABEL_403;
                          v256 = memcmp(v263, v261, v260);
                          if ( !v256 )
                            goto LABEL_396;
                        }
                        if ( v256 >= 0 )
                        {
LABEL_403:
                          v255 = (_QWORD *)v255[3];
                          goto LABEL_398;
                        }
LABEL_397:
                        v354 = v255;
                        v255 = (_QWORD *)v255[2];
LABEL_398:
                        if ( !v255 )
                        {
                          v201 = v348;
LABEL_408:
                          while ( 2 )
                          {
                            if ( v254 )
                            {
LABEL_409:
                              v266 = v254[4];
                              v267 = sub_1649960(v389);
                              v269 = v268;
                              v270 = v267;
                              v272 = sub_1649960(v266);
                              v273 = v271;
                              if ( v269 < v271 )
                              {
                                if ( !v269 )
                                  goto LABEL_407;
                                v265 = memcmp(v272, v270, v269);
                                if ( v265 )
                                  goto LABEL_406;
LABEL_413:
                                if ( v269 <= v273 )
                                {
LABEL_407:
                                  v206 = v254;
                                  v254 = (_QWORD *)v254[2];
                                  continue;
                                }
                              }
                              else
                              {
                                if ( !v271 || (v265 = memcmp(v272, v270, v271)) == 0 )
                                {
                                  if ( v269 == v273 )
                                    goto LABEL_407;
                                  goto LABEL_413;
                                }
LABEL_406:
                                if ( v265 >= 0 )
                                  goto LABEL_407;
                              }
                              v254 = (_QWORD *)v254[3];
                              if ( !v254 )
                                break;
                              goto LABEL_409;
                            }
                            break;
                          }
                          if ( (_QWORD *)v201[8] == v206 && v350 == v354 )
                          {
LABEL_347:
                            sub_1C6F600(v201[7]);
                            v201[7] = 0;
                            v201[10] = 0;
                            v201[8] = v350;
                            v201[9] = v350;
                          }
                          else
                          {
                            for ( ; v206 != v354; --v201[10] )
                            {
                              v274 = v206;
                              v206 = (_QWORD *)sub_220EF30(v206);
                              v275 = sub_220F330(v274, v350);
                              j_j___libc_free_0(v275, 40);
                            }
                          }
LABEL_348:
                          v224 = (_QWORD *)a1[21];
                          if ( !v224 )
                          {
                            v225 = a1 + 20;
                            goto LABEL_355;
                          }
                          v225 = a1 + 20;
                          do
                          {
                            while ( 1 )
                            {
                              v226 = v224[2];
                              v227 = v224[3];
                              if ( v224[4] >= (unsigned __int64)v391 )
                                break;
                              v224 = (_QWORD *)v224[3];
                              if ( !v227 )
                                goto LABEL_353;
                            }
                            v225 = v224;
                            v224 = (_QWORD *)v224[2];
                          }
                          while ( v226 );
LABEL_353:
                          if ( v367 == v225 || v225[4] > (unsigned __int64)v391 )
                          {
LABEL_355:
                            v401 = &v391;
                            v225 = sub_1C6F7D0(a1 + 19, v225, (unsigned __int64 **)&v401);
                          }
                          v228 = (__int64)(v225 + 5);
                          v230 = sub_1C6FA30((__int64)(v225 + 5), (__int64 *)&v390);
                          if ( v229 )
                            sub_1C6E9C0(v228, (__int64)v230, v229, (__int64 *)&v390);
LABEL_317:
                          v357 = *(_QWORD *)(v357 + 8);
                          if ( v357 )
                            goto LABEL_318;
                          v198 = v389;
                          v199 = v390;
LABEL_420:
                          sub_164D160(v198, v199, si128, a4, a5, a6, v196, v197, a9, a10);
                          if ( dword_4FBD480 )
                          {
                            v319 = sub_16E8CB0();
                            v320 = sub_1649960(v389);
                            v322 = (void *)v319[3];
                            v323 = (char *)v320;
                            v324 = v321;
                            v325 = v319[2] - (_QWORD)v322;
                            if ( v325 < v321 )
                            {
                              v332 = sub_16E7EE0((__int64)v319, v323, v321);
                              v322 = *(void **)(v332 + 24);
                              v319 = (_QWORD *)v332;
                              v325 = *(_QWORD *)(v332 + 16) - (_QWORD)v322;
                            }
                            else if ( v321 )
                            {
                              memcpy(v322, v323, v321);
                              v347 = v319[2];
                              v322 = (void *)(v324 + v319[3]);
                              v319[3] = v322;
                              v325 = v347 - (_QWORD)v322;
                            }
                            if ( v325 <= 0xA )
                            {
                              sub_16E7EE0((__int64)v319, " is cloned\n", 0xBu);
                            }
                            else
                            {
                              qmemcpy(v322, " is cloned\n", 11);
                              v319[3] += 11LL;
                            }
                          }
                          v107 = a1 + 7;
                          if ( v398 == &v396 )
                            goto LABEL_448;
                          v276 = (__int64)v398;
                          v277 = a1 + 8;
                          while ( 2 )
                          {
                            v278 = *(_QWORD *)(v276 + 32);
                            v279 = *(_DWORD *)(v276 + 40);
                            v402 = 2;
                            v403 = 0;
                            v404 = v278;
                            if ( v278 != 0 && v278 != -8 && v278 != -16 )
                              sub_164C220((__int64)&v402);
                            v401 = (void **)&unk_49E6B50;
                            v405 = (__int64 *)&v420;
                            if ( !(_DWORD)v423 )
                            {
                              v420 = (unsigned __int64 *)((char *)v420 + 1);
                              goto LABEL_428;
                            }
                            v280 = v404;
                            LODWORD(v282) = (v423 - 1) & (((unsigned int)v404 >> 9) ^ ((unsigned int)v404 >> 4));
                            v281 = (_QWORD *)(v421 + ((unsigned __int64)(unsigned int)v282 << 6));
                            v283 = v281[3];
                            if ( v283 == v404 )
                            {
LABEL_431:
                              v401 = (void **)&unk_49EE2B0;
                              if ( v280 != -8 && v280 != 0 && v280 != -16 )
                                sub_1649B30(&v402);
                              v284 = v281[7];
                              v285 = a1 + 8;
                              v286 = (_QWORD *)a1[9];
                              if ( !v286 )
                                goto LABEL_441;
                              do
                              {
                                while ( 1 )
                                {
                                  v287 = v286[2];
                                  v288 = v286[3];
                                  if ( v286[4] >= v284 )
                                    break;
                                  v286 = (_QWORD *)v286[3];
                                  if ( !v288 )
                                    goto LABEL_439;
                                }
                                v285 = v286;
                                v286 = (_QWORD *)v286[2];
                              }
                              while ( v287 );
LABEL_439:
                              if ( v285 == v277 || v285[4] > v284 )
                              {
LABEL_441:
                                v351 = v285;
                                v289 = sub_22077B0(48);
                                *(_QWORD *)(v289 + 32) = v284;
                                v285 = (_QWORD *)v289;
                                *(_DWORD *)(v289 + 40) = 0;
                                v290 = sub_1C704D0(a1 + 7, v351, (unsigned __int64 *)(v289 + 32));
                                if ( v291 )
                                {
                                  v292 = v277 == v291 || v290 || v284 < v291[4];
                                  sub_220F040(v292, v285, v291, v277);
                                  ++a1[12];
                                }
                                else
                                {
                                  v352 = v290;
                                  j_j___libc_free_0(v285, 48);
                                  v285 = v352;
                                }
                              }
                              *((_DWORD *)v285 + 10) = v279;
                              v276 = sub_220EEE0(v276);
                              if ( (int *)v276 == &v396 )
                              {
                                v107 = a1 + 7;
LABEL_448:
                                v389 = v390;
                                if ( (_BYTE)v428 )
                                {
                                  if ( (_DWORD)v427 )
                                  {
                                    v316 = (_QWORD *)v425;
                                    v317 = (_QWORD *)(v425 + 16LL * (unsigned int)v427);
                                    do
                                    {
                                      if ( *v316 != -4 && *v316 != -8 )
                                      {
                                        v318 = v316[1];
                                        if ( v318 )
                                          sub_161E7C0((__int64)(v316 + 1), v318);
                                      }
                                      v316 += 2;
                                    }
                                    while ( v317 != v316 );
                                  }
                                  j___libc_free_0(v425);
                                }
                                if ( (_DWORD)v423 )
                                {
                                  v293 = (_QWORD *)v421;
                                  v392[0] = 2;
                                  v392[1] = 0;
                                  v294 = -8;
                                  v295 = (_QWORD *)(v421 + ((unsigned __int64)(unsigned int)v423 << 6));
                                  v393 = -8;
                                  v391 = &unk_49E6B50;
                                  v394 = 0;
                                  v402 = 2;
                                  v403 = 0;
                                  v404 = -16;
                                  v401 = (void **)&unk_49E6B50;
                                  v405 = 0;
                                  while ( 1 )
                                  {
                                    v296 = v293[3];
                                    if ( v296 != v294 )
                                    {
                                      v294 = v404;
                                      if ( v296 != v404 )
                                      {
                                        v297 = v293[7];
                                        if ( v297 != 0 && v297 != -8 && v297 != -16 )
                                        {
                                          sub_1649B30(v293 + 5);
                                          v296 = v293[3];
                                        }
                                        v294 = v296;
                                      }
                                    }
                                    *v293 = &unk_49EE2B0;
                                    if ( v294 != 0 && v294 != -8 && v294 != -16 )
                                      sub_1649B30(v293 + 1);
                                    v293 += 8;
                                    if ( v295 == v293 )
                                      break;
                                    v294 = v393;
                                  }
                                  v401 = (void **)&unk_49EE2B0;
                                  if ( v404 != 0 && v404 != -8 && v404 != -16 )
                                    sub_1649B30(&v402);
                                  v391 = &unk_49EE2B0;
                                  if ( v393 != -8 && v393 != 0 && v393 != -16 )
                                    sub_1649B30(v392);
                                }
                                j___libc_free_0(v421);
                                v106 = v389;
                                goto LABEL_154;
                              }
                              continue;
                            }
                            break;
                          }
                          v310 = 1;
                          v311 = 0;
                          while ( v283 != -8 )
                          {
                            if ( !v311 && v283 == -16 )
                              v311 = v281;
                            v282 = ((_DWORD)v423 - 1) & (unsigned int)(v282 + v310);
                            v281 = (_QWORD *)(v421 + (v282 << 6));
                            v283 = v281[3];
                            if ( v404 == v283 )
                              goto LABEL_431;
                            ++v310;
                          }
                          if ( v311 )
                            v281 = v311;
                          v420 = (unsigned __int64 *)((char *)v420 + 1);
                          v312 = v422 + 1;
                          if ( 4 * ((int)v422 + 1) >= (unsigned int)(3 * v423) )
                          {
LABEL_428:
                            sub_12E48B0((__int64)&v420, 2 * v423);
                            if ( (_DWORD)v423 )
                            {
                              v280 = v404;
                              v334 = 1;
                              v335 = 0;
                              v336 = (v423 - 1) & (((unsigned int)v404 >> 9) ^ ((unsigned int)v404 >> 4));
                              v281 = (_QWORD *)(v421 + ((unsigned __int64)v336 << 6));
                              v337 = v281[3];
                              if ( v404 != v337 )
                              {
                                while ( v337 != -8 )
                                {
                                  if ( !v335 && v337 == -16 )
                                    v335 = v281;
                                  v336 = (v423 - 1) & (v334 + v336);
                                  v281 = (_QWORD *)(v421 + ((unsigned __int64)v336 << 6));
                                  v337 = v281[3];
                                  if ( v404 == v337 )
                                    goto LABEL_534;
                                  ++v334;
                                }
                                if ( v335 )
                                  v281 = v335;
                              }
                            }
                            else
                            {
                              v280 = v404;
                              v281 = 0;
                            }
LABEL_534:
                            v312 = v422 + 1;
                          }
                          else if ( (int)v423 - HIDWORD(v422) - v312 <= (unsigned int)v423 >> 3 )
                          {
                            sub_12E48B0((__int64)&v420, v423);
                            if ( (_DWORD)v423 )
                            {
                              v280 = v404;
                              v328 = 1;
                              v329 = 0;
                              v330 = (v423 - 1) & (((unsigned int)v404 >> 9) ^ ((unsigned int)v404 >> 4));
                              v281 = (_QWORD *)(v421 + ((unsigned __int64)v330 << 6));
                              v331 = v281[3];
                              if ( v404 != v331 )
                              {
                                while ( v331 != -8 )
                                {
                                  if ( !v329 && v331 == -16 )
                                    v329 = v281;
                                  v330 = (v423 - 1) & (v328 + v330);
                                  v281 = (_QWORD *)(v421 + ((unsigned __int64)v330 << 6));
                                  v331 = v281[3];
                                  if ( v404 == v331 )
                                    goto LABEL_531;
                                  ++v328;
                                }
                                if ( v329 )
                                  v281 = v329;
                              }
                            }
                            else
                            {
                              v280 = v404;
                              v281 = 0;
                            }
LABEL_531:
                            v312 = v422 + 1;
                          }
                          LODWORD(v422) = v312;
                          v313 = v281[3];
                          v314 = v281 + 1;
                          if ( v313 == -8 )
                          {
                            if ( v280 != -8 )
                              goto LABEL_504;
                          }
                          else
                          {
                            --HIDWORD(v422);
                            if ( v280 != v313 )
                            {
                              if ( v313 != -16 && v313 )
                              {
                                sub_1649B30(v314);
                                v280 = v404;
                                v314 = v281 + 1;
                              }
LABEL_504:
                              v281[3] = v280;
                              if ( v280 != -8 && v280 != 0 && v280 != -16 )
                                sub_1649AC0(v314, v402 & 0xFFFFFFFFFFFFFFF8LL);
                              v280 = v404;
                            }
                          }
                          v315 = v405;
                          v281[5] = 6;
                          v281[6] = 0;
                          v281[4] = v315;
                          v281[7] = 0;
                          goto LABEL_431;
                        }
                      }
                    }
                  }
                  v354 = v206;
                  v206 = (_QWORD *)v206[2];
                  if ( !v206 )
                  {
LABEL_344:
                    v201 = v348;
                    v223 = v354 == v350;
LABEL_345:
                    if ( (_QWORD *)v201[8] == v354 && v223 )
                      goto LABEL_347;
                    goto LABEL_348;
                  }
                  continue;
                }
              }
              goto LABEL_39;
            }
          }
          if ( v420 )
          {
            v168 = (__int64 *)v425;
            v169 = v429 + 1;
            if ( v425 < (unsigned __int64)(v429 + 1) )
            {
              v374 = 0;
              do
              {
LABEL_269:
                v170 = *v168++;
                j_j___libc_free_0(v170, 512);
              }
              while ( v168 < v169 );
              j_j___libc_free_0(v420, 8 * v421);
              sub_1C6F430(v403);
              if ( v374 )
                goto LABEL_302;
              goto LABEL_271;
            }
            j_j___libc_free_0(v420, 8 * v421);
          }
          sub_1C6F430(v403);
LABEL_271:
          v155 = sub_220EEE0(v155);
          if ( (int *)v155 == &v396 )
          {
            if ( !dword_4FBD480 )
              goto LABEL_273;
            goto LABEL_541;
          }
        }
      }
      if ( !dword_4FBD480 )
        goto LABEL_154;
LABEL_541:
      v339 = (__int64)sub_16E8CB0();
      v340 = *(__m128 **)(v339 + 24);
      if ( *(_QWORD *)(v339 + 16) - (_QWORD)v340 <= 0x10u )
      {
        v339 = sub_16E7EE0(v339, "avoid cloning of ", 0x11u);
      }
      else
      {
        si128 = (__m128)_mm_load_si128((const __m128i *)&xmmword_42D1FC0);
        v340[1].m128_i8[0] = 32;
        *v340 = si128;
        *(_QWORD *)(v339 + 24) += 17LL;
      }
      v341 = sub_1649960(v389);
      v343 = *(_BYTE **)(v339 + 24);
      v344 = (char *)v341;
      v345 = *(_BYTE **)(v339 + 16);
      v346 = v342;
      if ( v345 - v343 < v342 )
      {
        v339 = sub_16E7EE0(v339, v344, v342);
        v345 = *(_BYTE **)(v339 + 16);
        v343 = *(_BYTE **)(v339 + 24);
      }
      else if ( v342 )
      {
        memcpy(v343, v344, v342);
        v345 = *(_BYTE **)(v339 + 16);
        v343 = (_BYTE *)(v346 + *(_QWORD *)(v339 + 24));
        *(_QWORD *)(v339 + 24) = v343;
      }
      if ( v345 == v343 )
      {
        sub_16E7EE0(v339, "\n", 1u);
      }
      else
      {
        *v343 = 10;
        ++*(_QWORD *)(v339 + 24);
      }
LABEL_273:
      v107 = a1 + 7;
      v367 = a1 + 20;
      if ( v398 == &v396 )
        goto LABEL_289;
      v171 = a1 + 8;
      v172 = (__int64)v398;
      do
      {
        v173 = *(_DWORD *)(v172 + 40);
        v174 = (_QWORD *)a1[9];
        if ( !v174 )
        {
          v176 = a1 + 8;
LABEL_282:
          v382 = v176;
          v179 = sub_22077B0(48);
          v180 = *(_QWORD *)(v172 + 32);
          *(_DWORD *)(v179 + 40) = 0;
          v176 = (_QWORD *)v179;
          *(_QWORD *)(v179 + 32) = v180;
          v363 = v180;
          v181 = sub_1C704D0(a1 + 7, v382, (unsigned __int64 *)(v179 + 32));
          if ( v182 )
          {
            v183 = v171 == v182 || v181 || v363 < v182[4];
            sub_220F040(v183, v176, v182, v171);
            ++a1[12];
          }
          else
          {
            v383 = v181;
            j_j___libc_free_0(v176, 48);
            v176 = v383;
          }
          goto LABEL_287;
        }
        v175 = *(_QWORD *)(v172 + 32);
        v176 = a1 + 8;
        do
        {
          while ( 1 )
          {
            v177 = v174[2];
            v178 = v174[3];
            if ( v174[4] >= v175 )
              break;
            v174 = (_QWORD *)v174[3];
            if ( !v178 )
              goto LABEL_280;
          }
          v176 = v174;
          v174 = (_QWORD *)v174[2];
        }
        while ( v177 );
LABEL_280:
        if ( v176 == v171 || v176[4] > v175 )
          goto LABEL_282;
LABEL_287:
        *((_DWORD *)v176 + 10) = v173;
        v172 = sub_220EEE0(v172);
      }
      while ( (int *)v172 != &v396 );
      v107 = a1 + 7;
LABEL_289:
      v106 = v389;
LABEL_154:
      sub_1CACE90(v106, *a1, a1 + 1, v107);
      v108 = (_QWORD *)a1[21];
      if ( !v108 )
        goto LABEL_184;
      v109 = v389;
      v110 = v367;
      v111 = (_QWORD *)a1[21];
      do
      {
        while ( 1 )
        {
          v112 = v111[2];
          v113 = v111[3];
          if ( v111[4] >= v389 )
            break;
          v111 = (_QWORD *)v111[3];
          if ( !v113 )
            goto LABEL_159;
        }
        v110 = v111;
        v111 = (_QWORD *)v111[2];
      }
      while ( v112 );
LABEL_159:
      if ( v110 == v367 || v110[4] > v389 )
        goto LABEL_184;
      v114 = v367;
      v115 = (_QWORD *)a1[21];
      do
      {
        while ( 1 )
        {
          v116 = v115[2];
          v117 = v115[3];
          if ( v115[4] >= v389 )
            break;
          v115 = (_QWORD *)v115[3];
          if ( !v117 )
            goto LABEL_165;
        }
        v114 = v115;
        v115 = (_QWORD *)v115[2];
      }
      while ( v116 );
LABEL_165:
      if ( v114 != v367 && v114[4] <= v389 )
      {
        v120 = v114[8];
        goto LABEL_169;
      }
      v118 = a1 + 19;
      v420 = &v389;
      v119 = sub_1C6F7D0(a1 + 19, v114, &v420);
      v108 = (_QWORD *)a1[21];
      v120 = v119[8];
      if ( !v108 )
      {
        v121 = v367;
        goto LABEL_360;
      }
      v109 = v389;
LABEL_169:
      v121 = v367;
      do
      {
        while ( 1 )
        {
          v122 = v108[2];
          v123 = v108[3];
          if ( v108[4] >= v109 )
            break;
          v108 = (_QWORD *)v108[3];
          if ( !v123 )
            goto LABEL_173;
        }
        v121 = v108;
        v108 = (_QWORD *)v108[2];
      }
      while ( v122 );
LABEL_173:
      if ( v121 == v367 || v121[4] > v109 )
      {
        v118 = a1 + 19;
LABEL_360:
        v420 = &v389;
        v121 = sub_1C6F7D0(v118, v121, &v420);
      }
      v124 = v121 + 6;
      for ( k = 0; v124 != (_QWORD *)v120; v120 = sub_220EF30(v120) )
      {
        while ( 1 )
        {
          v420 = *(unsigned __int64 **)(v120 + 32);
          if ( (unsigned __int8)sub_1C6EC20((__int64)a1, (unsigned __int64)v420) )
            break;
LABEL_177:
          v120 = sub_220EF30(v120);
          if ( v124 == (_QWORD *)v120 )
            goto LABEL_183;
        }
        v126 = v413;
        ++k;
        if ( v413 == (_QWORD *)(v415 - 8) )
        {
          sub_1C6FE60(&v407, &v420);
          goto LABEL_177;
        }
        if ( v413 )
        {
          *v413 = v420;
          v126 = v413;
        }
        v413 = v126 + 1;
      }
LABEL_183:
      if ( dword_4FBD480 )
      {
        v245 = sub_16E8CB0();
        v246 = sub_16E7AB0((__int64)v245, k);
        v247 = *(__m128 **)(v246 + 24);
        if ( *(_QWORD *)(v246 + 16) - (_QWORD)v247 <= 0x15u )
        {
          sub_16E7EE0(v246, " callees are affected\n", 0x16u);
        }
        else
        {
          si128 = (__m128)_mm_load_si128((const __m128i *)&xmmword_42D1FD0);
          v247[1].m128_i32[0] = 1702126437;
          v247[1].m128_i16[2] = 2660;
          *v247 = si128;
          *(_QWORD *)(v246 + 24) += 22LL;
        }
      }
LABEL_184:
      sub_1C6EEC0((__int64)v397);
      if ( v417 != v419 )
        _libc_free((unsigned __int64)v417);
      v26 = (_QWORD *)v389;
      v349 = v353;
LABEL_42:
      v37 = **(_QWORD **)(v26[3] + 16LL);
      if ( *(_BYTE *)(v37 + 8) == 15 && !(*(_DWORD *)(v37 + 8) >> 8) )
      {
        v38 = (__int64 *)a1[3];
        if ( !v38 )
          goto LABEL_51;
        v39 = a1 + 2;
        do
        {
          while ( 1 )
          {
            v40 = v38[2];
            v41 = v38[3];
            if ( v38[4] >= (unsigned __int64)v26 )
              break;
            v38 = (__int64 *)v38[3];
            if ( !v41 )
              goto LABEL_49;
          }
          v39 = v38;
          v38 = (__int64 *)v38[2];
        }
        while ( v40 );
LABEL_49:
        if ( v39 == a1 + 2 || v39[4] > (unsigned __int64)v26 )
        {
LABEL_51:
          v42 = (_QWORD *)v26[10];
          v43 = v26 + 9;
          v385 = 1000;
          if ( v42 != v26 + 9 )
          {
            v378 = a1 + 2;
            do
            {
LABEL_54:
              v44 = (__int64)(v42 - 3);
              if ( !v42 )
                v44 = 0;
              v45 = sub_157EBA0(v44);
              if ( *(_BYTE *)(v45 + 16) != 25 )
                goto LABEL_53;
              v46 = *(_QWORD *)(v45 - 24LL * (*(_DWORD *)(v45 + 20) & 0xFFFFFFF));
              if ( !v46 )
                goto LABEL_64;
              if ( *(_BYTE *)(*(_QWORD *)v46 + 8LL) != 15 || *(_DWORD *)(*(_QWORD *)v46 + 8LL) >> 8 )
                goto LABEL_53;
              v47 = *a1;
              LODWORD(v420) = 0;
              if ( !(unsigned __int8)sub_1CA0940(v26, v46, &v420, v47, a1 + 1, a1 + 7) || !(_DWORD)v420 )
                goto LABEL_64;
              if ( v385 != 1000 )
              {
                if ( (_DWORD)v420 != v385 )
                  goto LABEL_64;
LABEL_53:
                v42 = (_QWORD *)v42[1];
                if ( v43 == v42 )
                  break;
                goto LABEL_54;
              }
              v42 = (_QWORD *)v42[1];
              v385 = (unsigned int)v420;
            }
            while ( v43 != v42 );
            if ( v385 != 1000 )
              break;
          }
        }
      }
LABEL_64:
      v25 = v413;
      if ( v413 == v409 )
        goto LABEL_65;
    }
    if ( dword_4FBD480 )
    {
      v298 = (__int64)sub_16E8CB0();
      v299 = sub_1649960((__int64)v26);
      v301 = *(__m128 **)(v298 + 24);
      v302 = (char *)v299;
      v303 = v300;
      v304 = *(_QWORD *)(v298 + 16) - (_QWORD)v301;
      if ( v300 > v304 )
      {
        v338 = sub_16E7EE0(v298, v302, v300);
        v301 = *(__m128 **)(v338 + 24);
        v298 = v338;
        v304 = *(_QWORD *)(v338 + 16) - (_QWORD)v301;
      }
      else if ( v300 )
      {
        memcpy(v301, v302, v300);
        v333 = *(_QWORD *)(v298 + 16);
        v301 = (__m128 *)(v303 + *(_QWORD *)(v298 + 24));
        *(_QWORD *)(v298 + 24) = v301;
        v304 = v333 - (_QWORD)v301;
      }
      if ( v304 <= 0x24 )
      {
        v298 = sub_16E7EE0(v298, " : return memory space is resolved : ", 0x25u);
      }
      else
      {
        v305 = _mm_load_si128((const __m128i *)&xmmword_42D1FE0);
        v301[2].m128_i32[0] = 975201381;
        v301[2].m128_i8[4] = 32;
        *v301 = (__m128)v305;
        si128 = (__m128)_mm_load_si128((const __m128i *)&xmmword_42D1FF0);
        v301[1] = si128;
        *(_QWORD *)(v298 + 24) += 37LL;
      }
      v306 = sub_16E7A90(v298, v385);
      v307 = *(_BYTE **)(v306 + 24);
      if ( *(_BYTE **)(v306 + 16) == v307 )
      {
        sub_16E7EE0(v306, "\n", 1u);
      }
      else
      {
        *v307 = 10;
        ++*(_QWORD *)(v306 + 24);
      }
    }
    v142 = a1 + 2;
    v143 = a1 + 1;
    v144 = (_QWORD *)a1[3];
    if ( !v144 )
      goto LABEL_382;
    do
    {
      while ( 1 )
      {
        v145 = v144[2];
        v146 = v144[3];
        if ( v144[4] >= (unsigned __int64)v26 )
          break;
        v144 = (_QWORD *)v144[3];
        if ( !v146 )
          goto LABEL_231;
      }
      v142 = v144;
      v144 = (_QWORD *)v144[2];
    }
    while ( v145 );
LABEL_231:
    if ( v142 == v378 || v142[4] > (unsigned __int64)v26 )
    {
LABEL_382:
      v248 = sub_22077B0(48);
      v249 = v142;
      *(_QWORD *)(v248 + 32) = v26;
      v142 = (_QWORD *)v248;
      *(_DWORD *)(v248 + 40) = 0;
      v250 = sub_1C70810(v143, v249, (unsigned __int64 *)(v248 + 32));
      if ( v251 )
      {
        v252 = v250 || v378 == v251 || (unsigned __int64)v26 < v251[4];
        sub_220F040(v252, v142, v251, v378);
        ++a1[6];
      }
      else
      {
        v309 = v142;
        v142 = v250;
        j_j___libc_free_0(v309, 48);
      }
    }
    *((_DWORD *)v142 + 10) = v385;
    v147 = v26[1];
    if ( v147 )
    {
      while ( 1 )
      {
LABEL_236:
        v148 = (unsigned __int64)sub_1648700(v147);
        v149 = 0;
        v150 = *(_BYTE *)(v148 + 16);
        if ( v150 > 0x17u )
        {
          v151 = v148 & 0xFFFFFFFFFFFFFFF8LL;
          if ( v150 != 78 && v150 != 29 )
            v151 = 0;
          v149 = v151;
        }
        if ( !*(_QWORD *)(v149 + 8) )
          goto LABEL_235;
        v152 = *a1;
        v420 = *(unsigned __int64 **)(*(_QWORD *)(v149 + 40) + 56LL);
        sub_1CACE90(v420, v152, v143, a1 + 7);
        if ( !(unsigned __int8)sub_1C6EC20((__int64)a1, (unsigned __int64)v420) )
          goto LABEL_235;
        v153 = v413;
        if ( v413 == (_QWORD *)(v415 - 8) )
          break;
        if ( v413 )
        {
          *v413 = v420;
          v153 = v413;
        }
        v413 = v153 + 1;
        v147 = *(_QWORD *)(v147 + 8);
        if ( !v147 )
          goto LABEL_247;
      }
      sub_1C6FE60(&v407, &v420);
LABEL_235:
      v147 = *(_QWORD *)(v147 + 8);
      if ( !v147 )
        goto LABEL_247;
      goto LABEL_236;
    }
LABEL_247:
    v349 = 1;
    v25 = v413;
  }
  while ( v413 != v409 );
LABEL_65:
  v48 = v407;
  if ( v407 )
  {
    v49 = (__int64 *)v412;
    v50 = v416 + 1;
    if ( (unsigned __int64)(v416 + 1) > v412 )
    {
      do
      {
        v51 = *v49++;
        j_j___libc_free_0(v51, 512);
      }
      while ( v50 > v49 );
      v48 = v407;
    }
    j_j___libc_free_0(v48, 8 * v408);
  }
  return v349;
}
