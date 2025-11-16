// Function: sub_188C730
// Address: 0x188c730
//
__int64 __fastcall sub_188C730(
        __int64 *a1,
        __m128 a2,
        __m128 a3,
        __m128i a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v9; // r14
  __int64 v10; // r12
  char *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // rbx
  char *v16; // rax
  __int64 v17; // rdx
  double v18; // xmm4_8
  double v19; // xmm5_8
  __int64 v20; // rbx
  __int64 v21; // rdi
  _QWORD *v22; // rax
  double v23; // xmm4_8
  double v24; // xmm5_8
  __int64 *v25; // r13
  __int64 *v26; // r15
  __int64 *v27; // r13
  __int64 v28; // rbx
  void *v29; // rdx
  int v30; // r8d
  int v31; // r9d
  __int64 v32; // rax
  __int64 *v33; // rbx
  __int64 *v34; // r12
  __int64 v35; // rsi
  __m128i *v36; // rdi
  __int64 v37; // r12
  __int64 *v38; // rbx
  __int64 v39; // rsi
  __int64 v41; // rbx
  void *v42; // rdx
  int v43; // r8d
  int v44; // r9d
  __int64 v45; // rax
  __int64 v46; // rdi
  __int64 v47; // rax
  double v48; // xmm4_8
  double v49; // xmm5_8
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // r15
  __int64 *v53; // r14
  __int64 *v54; // r12
  __int64 v55; // rax
  _QWORD *v56; // rbx
  _QWORD *v57; // r13
  unsigned int v58; // edx
  unsigned __int64 *v59; // rdi
  unsigned __int64 v60; // rcx
  int v61; // esi
  unsigned __int64 v62; // rax
  unsigned __int64 *v63; // r10
  int v64; // ecx
  int v65; // r11d
  __int64 *v66; // rdi
  unsigned int v67; // r13d
  __int64 v68; // rbx
  __int64 v69; // rax
  __int64 v70; // rdx
  __int64 v71; // r14
  __int64 v72; // rax
  bool v73; // cc
  _QWORD *v74; // rax
  size_t v75; // rdx
  int *v76; // rsi
  __int64 v77; // rdi
  bool v78; // si
  __int64 v79; // r10
  _QWORD *v80; // rax
  _QWORD *v81; // r8
  __m128 *v82; // rax
  const __m128i *v83; // rax
  const __m128i *v84; // r15
  const __m128i *v85; // r13
  __int32 v86; // r12d
  __int64 v87; // rbx
  __int64 v88; // r14
  __int64 *v89; // rax
  __int64 *v90; // rdx
  __m128i *v91; // r12
  void **v92; // rdi
  __int64 (__fastcall *v93)(__int64); // rax
  __m128i *v94; // r13
  __int64 v95; // rax
  _BYTE *v96; // rbx
  __int64 v97; // rdi
  __m128i v98; // rax
  char v99; // r12
  size_t v100; // r14
  __int64 v101; // rax
  __int64 v102; // r8
  char v103; // al
  __int64 *v104; // rdx
  int v105; // esi
  int v106; // eax
  __int64 v107; // rax
  unsigned __int64 v108; // r14
  __int64 v109; // rax
  __int64 v110; // rdx
  __int64 v111; // rdx
  int v112; // esi
  unsigned int v113; // ecx
  __int64 v114; // rax
  __int64 v115; // r9
  _QWORD *v116; // rdx
  _BYTE *v117; // rsi
  __m128i *v118; // r12
  void **v119; // rdi
  __int64 (__fastcall *v120)(__int64); // rax
  __m128i *v121; // rbx
  __int64 v122; // rdi
  __int64 *v123; // r14
  __int64 j; // r12
  __int64 v125; // rax
  __int64 *v126; // rax
  _BYTE *v127; // rsi
  _QWORD *v128; // r14
  _QWORD *v129; // rax
  __int64 v130; // r12
  int v131; // ebx
  __int64 v132; // rax
  __int64 v133; // rdx
  __int64 v134; // r13
  int v135; // r13d
  __int64 v136; // rax
  __int64 v137; // rdx
  char v138; // al
  unsigned int v139; // r13d
  int v140; // ebx
  __int64 v141; // rax
  __int64 v142; // rdx
  __int64 v143; // r15
  int v144; // r15d
  __int64 v145; // rax
  __int64 v146; // rdx
  __int64 v147; // rax
  unsigned __int16 *v148; // rax
  bool v149; // bl
  unsigned int v150; // ecx
  unsigned __int16 **v151; // rdx
  unsigned __int16 *v152; // r9
  unsigned __int64 v153; // rcx
  char *v154; // rsi
  int *v155; // r15
  unsigned __int64 v156; // rdx
  int *v157; // rax
  bool v158; // si
  int *v159; // rbx
  _BOOL4 v160; // r9d
  _QWORD *v161; // rax
  unsigned __int64 v162; // r10
  __int64 v163; // rdx
  _QWORD *v164; // rcx
  _QWORD *v165; // rsi
  _BYTE *v166; // rdi
  _BYTE *v167; // rax
  __m128i v168; // rax
  __m128 *v169; // rax
  int v170; // ecx
  unsigned int v171; // eax
  unsigned int i; // r12d
  unsigned __int64 v173; // rdx
  __int64 v174; // rax
  __int64 v175; // rdi
  char v176; // dl
  char v177; // al
  unsigned __int64 v178; // rdx
  _QWORD **v179; // r8
  __int64 *v180; // rax
  __int64 v181; // rax
  const void *v182; // r15
  signed __int64 v183; // r13
  __int64 v184; // rax
  __int64 v185; // rbx
  int *v186; // r12
  unsigned __int64 v187; // rbx
  unsigned __int64 v188; // rdx
  int *v189; // rax
  _BOOL4 v190; // r13d
  int *v191; // rax
  unsigned __int64 v192; // rbx
  int *v193; // r15
  unsigned __int64 v194; // rdx
  _BYTE *v195; // rdi
  _BYTE *v196; // rax
  _QWORD *v197; // rdx
  __int64 v198; // rax
  unsigned int v199; // r12d
  __int64 v200; // rax
  int v201; // edx
  __int64 **v202; // rax
  __int64 *v203; // r13
  __int64 *v204; // r14
  __int64 v205; // rax
  unsigned int v206; // r8d
  unsigned int v207; // edx
  __int64 *v208; // r15
  __int64 v209; // rcx
  unsigned __int64 v210; // rdx
  __int64 *v211; // rbx
  __int64 v212; // r15
  __int64 v213; // rsi
  char *v214; // r13
  char *v215; // rbx
  __int64 v216; // r12
  unsigned __int64 *v217; // r12
  __int64 v218; // rax
  int v219; // r15d
  unsigned __int16 **v220; // rdi
  int v221; // ecx
  __int64 v222; // rax
  int v223; // r10d
  __int64 *v224; // rsi
  int v225; // ecx
  int v226; // esi
  unsigned __int16 **v227; // r8
  int *v228; // r12
  unsigned __int64 v229; // r12
  unsigned __int64 v230; // rax
  unsigned __int32 v231; // eax
  __int64 v232; // rax
  unsigned __int32 v233; // eax
  const __m128i *v234; // rsi
  __m128i *v235; // r12
  const __m128i *v236; // r13
  signed __int64 v237; // rbx
  unsigned __int64 v238; // rax
  __int64 v239; // r9
  __m128i *v240; // r11
  __m128i *k; // rdx
  __int64 v242; // r8
  unsigned __int32 v243; // esi
  __m128i *v244; // rcx
  __m128i *v245; // rax
  __int64 v246; // rdi
  unsigned __int64 v247; // r13
  char *v248; // r12
  char *v249; // r14
  unsigned __int64 v250; // rdx
  char *v251; // rsi
  __int64 v252; // rax
  unsigned __int64 v253; // rax
  unsigned __int64 v254; // rbx
  __int64 v255; // rsi
  signed __int64 v256; // rdx
  __int64 v257; // rsi
  __int64 v258; // rax
  bool v259; // cf
  unsigned __int64 v260; // rax
  __int64 v261; // r15
  __int64 v262; // r15
  __int64 v263; // rax
  char *v264; // r8
  __int64 v265; // r15
  __int64 *v266; // r13
  signed __int64 v267; // rbx
  unsigned __int64 v268; // rax
  __int64 *v269; // r15
  __int64 *v270; // r12
  __int64 v271; // rbx
  __int64 v272; // rdx
  __int64 *v273; // r13
  unsigned __int64 v274; // rax
  char *m; // rdi
  __int64 v276; // rcx
  __int64 v277; // rdx
  char *v278; // rsi
  char *v279; // rax
  __int64 v280; // rdi
  __int64 ii; // rax
  int v282; // esi
  int v283; // eax
  __m128i v284; // xmm4
  __int64 v285; // rax
  __int64 v286; // r12
  __int64 v287; // rax
  __int64 v288; // rax
  __int64 v289; // rbx
  __int64 v290; // rax
  __int64 v291; // r12
  double v292; // xmm4_8
  double v293; // xmm5_8
  __int64 v294; // rbx
  __int64 v295; // rdx
  void *v296; // rax
  void *v297; // rdx
  char v298; // al
  __int64 v299; // rdx
  __int64 v300; // rdi
  __int64 v301; // rax
  __int64 v302; // rbx
  unsigned int jj; // r12d
  __int64 v304; // r13
  __int64 v305; // rdx
  __int64 v306; // rax
  __int64 v307; // rdx
  __int64 v308; // r13
  __int64 v309; // rax
  char v310; // al
  char v311; // al
  _QWORD *v312; // rbx
  char v313; // r13
  _QWORD *v314; // r12
  __int64 v315; // rdi
  int v316; // esi
  unsigned int v317; // r10d
  __int64 *v318; // r9
  int v319; // r8d
  __int64 v320; // rdi
  _BYTE **v321; // rax
  _BYTE **v322; // r8
  _BYTE *v323; // rbx
  _QWORD *v324; // r15
  _QWORD *v325; // r14
  size_t v326; // rdx
  size_t v327; // r13
  char v328; // al
  int v329; // r8d
  _QWORD *v330; // r9
  int v331; // esi
  int v332; // eax
  __int64 v333; // rax
  __int64 v334; // rax
  unsigned __int64 v335; // r13
  __int64 v336; // rax
  unsigned __int64 v337; // r12
  __int64 v338; // rax
  __int64 v339; // rax
  _QWORD *v340; // [rsp+8h] [rbp-378h]
  __int64 v341; // [rsp+20h] [rbp-360h]
  __int64 v342; // [rsp+30h] [rbp-350h]
  int v343; // [rsp+38h] [rbp-348h]
  char *v344; // [rsp+38h] [rbp-348h]
  _QWORD *src; // [rsp+40h] [rbp-340h]
  void *srca; // [rsp+40h] [rbp-340h]
  unsigned __int64 srcb; // [rsp+40h] [rbp-340h]
  __int64 v348; // [rsp+48h] [rbp-338h]
  void *v349; // [rsp+48h] [rbp-338h]
  __int64 v350; // [rsp+48h] [rbp-338h]
  __int32 v351; // [rsp+50h] [rbp-330h]
  bool v352; // [rsp+50h] [rbp-330h]
  __int64 v353; // [rsp+50h] [rbp-330h]
  __int64 v354; // [rsp+50h] [rbp-330h]
  __int64 v355; // [rsp+58h] [rbp-328h]
  char v356; // [rsp+58h] [rbp-328h]
  __int64 v357; // [rsp+58h] [rbp-328h]
  _BYTE *v358; // [rsp+58h] [rbp-328h]
  __int64 v359; // [rsp+58h] [rbp-328h]
  __int64 v360; // [rsp+58h] [rbp-328h]
  __int64 v361; // [rsp+58h] [rbp-328h]
  const __m128i *v362; // [rsp+58h] [rbp-328h]
  __int64 v363; // [rsp+60h] [rbp-320h]
  size_t v364; // [rsp+60h] [rbp-320h]
  char v365; // [rsp+60h] [rbp-320h]
  __int64 *v366; // [rsp+60h] [rbp-320h]
  int v367; // [rsp+60h] [rbp-320h]
  __int64 **v368; // [rsp+60h] [rbp-320h]
  __int64 v369; // [rsp+60h] [rbp-320h]
  int v370; // [rsp+68h] [rbp-318h]
  __int64 v371; // [rsp+68h] [rbp-318h]
  signed __int64 v372; // [rsp+68h] [rbp-318h]
  int *v373; // [rsp+70h] [rbp-310h]
  __int64 *v374; // [rsp+70h] [rbp-310h]
  _BOOL4 v375; // [rsp+70h] [rbp-310h]
  __int64 *v376; // [rsp+70h] [rbp-310h]
  __int64 v377; // [rsp+70h] [rbp-310h]
  int v378; // [rsp+70h] [rbp-310h]
  __int64 v379; // [rsp+78h] [rbp-308h]
  _QWORD *v380; // [rsp+78h] [rbp-308h]
  __int64 *v381; // [rsp+78h] [rbp-308h]
  unsigned __int64 v382; // [rsp+78h] [rbp-308h]
  const __m128i *v383; // [rsp+78h] [rbp-308h]
  __int64 v384; // [rsp+78h] [rbp-308h]
  int *v385; // [rsp+78h] [rbp-308h]
  __int64 v386; // [rsp+78h] [rbp-308h]
  size_t n; // [rsp+80h] [rbp-300h]
  __int64 *na; // [rsp+80h] [rbp-300h]
  __int64 *nb; // [rsp+80h] [rbp-300h]
  size_t ne; // [rsp+80h] [rbp-300h]
  int nc; // [rsp+80h] [rbp-300h]
  size_t nd; // [rsp+80h] [rbp-300h]
  __int64 *v393; // [rsp+88h] [rbp-2F8h]
  char *v394; // [rsp+88h] [rbp-2F8h]
  unsigned int v395; // [rsp+88h] [rbp-2F8h]
  char v396; // [rsp+88h] [rbp-2F8h]
  unsigned __int8 v397; // [rsp+88h] [rbp-2F8h]
  _QWORD v398[2]; // [rsp+90h] [rbp-2F0h] BYREF
  _QWORD v399[2]; // [rsp+A0h] [rbp-2E0h] BYREF
  __int64 v400[4]; // [rsp+B0h] [rbp-2D0h] BYREF
  const __m128i *v401; // [rsp+D0h] [rbp-2B0h] BYREF
  __m128i *v402; // [rsp+D8h] [rbp-2A8h]
  const __m128i *v403; // [rsp+E0h] [rbp-2A0h]
  __m128i v404; // [rsp+F0h] [rbp-290h] BYREF
  __int16 v405; // [rsp+100h] [rbp-280h]
  __m128i v406; // [rsp+110h] [rbp-270h] BYREF
  char v407; // [rsp+120h] [rbp-260h]
  char v408; // [rsp+121h] [rbp-25Fh]
  __m128i v409[2]; // [rsp+130h] [rbp-250h] BYREF
  __m128i v410; // [rsp+150h] [rbp-230h] BYREF
  __int16 v411; // [rsp+160h] [rbp-220h]
  __int64 v412; // [rsp+170h] [rbp-210h] BYREF
  _BYTE **v413; // [rsp+178h] [rbp-208h]
  __int64 v414; // [rsp+180h] [rbp-200h]
  unsigned int v415; // [rsp+188h] [rbp-1F8h]
  _BYTE *v416; // [rsp+190h] [rbp-1F0h] BYREF
  __int64 v417; // [rsp+198h] [rbp-1E8h]
  _BYTE v418[16]; // [rsp+1A0h] [rbp-1E0h] BYREF
  __int64 v419; // [rsp+1B0h] [rbp-1D0h] BYREF
  const __m128i *v420; // [rsp+1B8h] [rbp-1C8h]
  __int64 v421; // [rsp+1C0h] [rbp-1C0h]
  unsigned int v422; // [rsp+1C8h] [rbp-1B8h]
  __m128 v423; // [rsp+1D0h] [rbp-1B0h] BYREF
  __int64 v424; // [rsp+1E0h] [rbp-1A0h]
  unsigned int v425; // [rsp+1E8h] [rbp-198h]
  void *v426[2]; // [rsp+1F0h] [rbp-190h] BYREF
  __int64 *v427; // [rsp+200h] [rbp-180h]
  __int64 *v428; // [rsp+208h] [rbp-178h]
  __int64 v429; // [rsp+210h] [rbp-170h] BYREF
  int v430; // [rsp+218h] [rbp-168h] BYREF
  int *v431; // [rsp+220h] [rbp-160h]
  int *v432; // [rsp+228h] [rbp-158h]
  int *v433; // [rsp+230h] [rbp-150h]
  __int64 v434; // [rsp+238h] [rbp-148h]
  __int64 *v435; // [rsp+240h] [rbp-140h] BYREF
  __int64 v436; // [rsp+248h] [rbp-138h]
  _QWORD v437[2]; // [rsp+250h] [rbp-130h] BYREF
  char v438; // [rsp+260h] [rbp-120h] BYREF
  _QWORD *v439; // [rsp+280h] [rbp-100h]
  __int64 v440; // [rsp+288h] [rbp-F8h]
  _QWORD v441[4]; // [rsp+290h] [rbp-F0h] BYREF
  __m128i v442; // [rsp+2B0h] [rbp-D0h] BYREF
  __m128i v443; // [rsp+2C0h] [rbp-C0h] BYREF

  v9 = (__int64)a1;
  v10 = *a1;
  v11 = sub_15E0FD0(208);
  v13 = sub_16321A0(v10, (__int64)v11, v12);
  v14 = *a1;
  v379 = v13;
  v15 = v13;
  v16 = sub_15E0FD0(108);
  n = sub_16321A0(v14, (__int64)v16, v17);
  if ( !v15 || (v20 = *(_QWORD *)(v15 + 8)) == 0 )
  {
    if ( n && *(_QWORD *)(n + 8) || a1[1] )
    {
      if ( !a1[2] )
      {
LABEL_41:
        v46 = *a1;
        v432 = &v430;
        v433 = &v430;
        v437[0] = &v438;
        v437[1] = 0x400000000LL;
        v439 = v441;
        v416 = v418;
        v430 = 0;
        v431 = 0;
        v434 = 0;
        v435 = 0;
        v436 = 0;
        v440 = 0;
        v441[0] = 0;
        v441[1] = 1;
        v412 = 0;
        v413 = 0;
        v414 = 0;
        v415 = 0;
        v417 = 0x200000000LL;
        v47 = sub_16328F0(v46, "Cross-DSO CFI", 0xDu);
        v419 = 0;
        v342 = v47;
        v50 = *(_QWORD *)(v9 + 8);
        v420 = 0;
        v421 = 0;
        v422 = 0;
        if ( !v50 )
          goto LABEL_111;
        v51 = v50 + 8;
        v426[0] = 0;
        v426[1] = 0;
        v427 = 0;
        v428 = 0;
        v52 = *(_QWORD *)(v51 + 16);
        v363 = v51;
        if ( v52 == v51 )
          goto LABEL_68;
        v355 = v9;
        while ( 1 )
        {
          v53 = *(__int64 **)(v52 + 64);
          v54 = *(__int64 **)(v52 + 56);
          if ( v54 != v53 )
            break;
LABEL_66:
          v52 = sub_220EEE0(v52);
          if ( v52 == v363 )
          {
            v9 = v355;
LABEL_68:
            v66 = *(__int64 **)v9;
            v442.m128i_i64[0] = (__int64)"cfi.functions";
            v443.m128i_i16[0] = 259;
            v348 = sub_1632310((__int64)v66, (__int64)&v442);
            if ( v348 )
            {
              v67 = 0;
              v343 = sub_161F520(v348);
              if ( v343 )
              {
                src = (_QWORD *)v9;
                do
                {
                  v68 = sub_161F530(v348, v67);
                  v69 = sub_161E970(*(_QWORD *)(v68 - 8LL * *(unsigned int *)(v68 + 8)));
                  v71 = v70;
                  v373 = (int *)v69;
                  v72 = sub_15A0FC0(*(_QWORD *)(*(_QWORD *)(v68 + 8 * (1LL - *(unsigned int *)(v68 + 8))) + 136LL));
                  v73 = *(_DWORD *)(v72 + 8) <= 0x40u;
                  v74 = *(_QWORD **)v72;
                  if ( !v73 )
                    v74 = (_QWORD *)*v74;
                  v351 = (int)v74;
                  v75 = v71;
                  v76 = v373;
                  if ( v71 && *(_BYTE *)v373 == 1 )
                  {
                    v75 = v71 - 1;
                    v76 = (int *)((char *)v373 + 1);
                  }
                  v364 = v75;
                  sub_16C1840(&v442);
                  sub_16C1A90(v442.m128i_i32, v76, v364);
                  sub_16C1AA0(&v442, &v423);
                  v77 = src[1];
                  v410.m128i_i64[0] = v423.m128_u64[0];
                  v365 = sub_1634350(v77, v423.m128_u64[0]);
                  if ( v365 )
                  {
                    if ( (unsigned __int8)sub_1880D60((__int64)v426, v410.m128i_i64, &v442) )
                      goto LABEL_256;
                    v78 = v351 != 0 || v342 == 0;
                    if ( !v78 )
                    {
                      v79 = src[1];
                      v80 = *(_QWORD **)(v79 + 16);
                      if ( v80 )
                      {
                        v81 = (_QWORD *)(v79 + 8);
                        do
                        {
                          if ( v410.m128i_i64[0] > v80[4] )
                          {
                            v80 = (_QWORD *)v80[3];
                          }
                          else
                          {
                            v81 = v80;
                            v80 = (_QWORD *)v80[2];
                          }
                        }
                        while ( v80 );
                        if ( v81 != (_QWORD *)(v79 + 8) && v410.m128i_i64[0] >= v81[4] )
                        {
                          if ( (unsigned __int16)(4 * *(unsigned __int8 *)(v79 + 178)) & 0xFFF8
                             | (unsigned __int64)(v81 + 4) & 0xFFFFFFFFFFFFFFF8LL )
                          {
                            v174 = *(_QWORD *)(((unsigned __int16)(4 * *(unsigned __int8 *)(v79 + 178)) & 0xFFF8
                                              | (unsigned __int64)(v81 + 4) & 0xFFFFFFFFFFFFFFF8LL)
                                             + 0x18);
                            v175 = *(_QWORD *)(((unsigned __int16)(4 * *(unsigned __int8 *)(v79 + 178)) & 0xFFF8
                                              | (unsigned __int64)(v81 + 4) & 0xFFFFFFFFFFFFFFF8LL)
                                             + 0x20);
                            if ( v174 != v175 )
                            {
                              do
                              {
                                v176 = *(_BYTE *)(*(_QWORD *)v174 + 12LL);
                                if ( (v176 & 0x20) != 0 && (v176 & 0xFu) - 7 >= 2 )
                                  v78 = v365;
                                v174 += 8;
                              }
                              while ( v175 != v174 );
                              if ( v78 )
                              {
LABEL_256:
                                v442.m128i_i64[1] = v71;
                                v443.m128i_i64[1] = v68;
                                v442.m128i_i64[0] = (__int64)v373;
                                v443.m128i_i32[0] = v351;
                                v177 = sub_1872D70((__int64)&v419, (__int64)&v442, &v423);
                                v178 = v423.m128_u64[0];
                                if ( v177 )
                                {
                                  if ( *(_DWORD *)(v423.m128_u64[0] + 16) )
                                  {
                                    *(_QWORD *)(v423.m128_u64[0] + 24) = v68;
                                    *(_DWORD *)(v178 + 16) = v351;
                                  }
                                }
                                else
                                {
                                  v82 = (__m128 *)sub_1874480((__int64)&v419, (__int64)&v442, v423.m128_u64[0]);
                                  a3 = (__m128)_mm_loadu_si128(&v442);
                                  *v82 = a3;
                                  a4 = _mm_loadu_si128(&v443);
                                  v82[1] = (__m128)a4;
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                  ++v67;
                }
                while ( v343 != v67 );
                v9 = (__int64)src;
              }
              if ( (_DWORD)v421 )
              {
                v83 = v420;
                v84 = &v420[2 * v422];
                if ( v420 != v84 )
                {
                  while ( 1 )
                  {
                    v85 = v83;
                    if ( v83->m128i_i64[0] != -1 && v83->m128i_i64[0] != -2 )
                      break;
                    v83 += 2;
                    if ( v84 == v83 )
                      goto LABEL_110;
                  }
                  if ( v84 != v83 )
                  {
                    v374 = (__int64 *)v9;
                    while ( 1 )
                    {
                      a2 = (__m128)_mm_loadu_si128(v85);
                      v423 = a2;
                      v86 = v85[1].m128i_i32[0];
                      v87 = v85[1].m128i_i64[1];
                      v88 = sub_16321A0(*v374, a2.m128_i64[0], a2.m128_i64[1]);
                      if ( !v88 )
                      {
                        v443.m128i_i16[0] = 261;
                        v179 = (_QWORD **)*v374;
                        v442.m128i_i64[0] = (__int64)&v423;
                        v354 = (__int64)v179;
                        v180 = (__int64 *)sub_1643270(*v179);
                        v360 = sub_16453E0(v180, 0);
                        v181 = sub_1648B60(120);
                        v88 = v181;
                        if ( v181 )
                          sub_15E2490(v181, v360, 0, (__int64)&v442, v354);
                      }
                      if ( (*(_BYTE *)(v88 + 32) & 0xF) == 1 )
                      {
                        *(_BYTE *)(v88 + 32) &= 0xF0u;
                        sub_15A5120(v88);
                        sub_15E0C30(v88);
                        *(_BYTE *)(v88 + 32) &= 0xF0u;
                        sub_15A5120(v88);
                        *(_QWORD *)(v88 + 48) = 0;
                        sub_161FB70(v88);
                      }
                      if ( v86 )
                        break;
                      if ( (*(_BYTE *)(v88 + 32) & 0xF) == 9 )
                      {
                        *(_BYTE *)(v88 + 32) &= 0xF0u;
                        sub_15A5120(v88);
                      }
                      if ( sub_15E4F60(v88) )
                        goto LABEL_243;
LABEL_103:
                      v85 += 2;
                      if ( v85 != v84 )
                      {
                        while ( v85->m128i_i64[0] == -1 || v85->m128i_i64[0] == -2 )
                        {
                          v85 += 2;
                          if ( v84 == v85 )
                            goto LABEL_109;
                        }
                        if ( v84 != v85 )
                          continue;
                      }
LABEL_109:
                      v9 = (__int64)v374;
                      goto LABEL_110;
                    }
                    if ( !sub_15E4F60(v88) )
                      goto LABEL_103;
                    if ( v86 == 2 )
                    {
                      *(_BYTE *)(v88 + 32) = *(_BYTE *)(v88 + 32) & 0xF0 | 9;
                      sub_15A5120(v88);
                    }
LABEL_243:
                    sub_1626EF0(v88, 0x13u);
                    v171 = *(_DWORD *)(v87 + 8);
                    if ( v171 > 2 )
                    {
                      for ( i = 2; i < v171; ++i )
                      {
                        v173 = i - (unsigned __int64)v171;
                        sub_16267C0(v88, 0x13u, *(_QWORD *)(v87 + 8 * v173));
                        v171 = *(_DWORD *)(v87 + 8);
                      }
                    }
                    goto LABEL_103;
                  }
                }
              }
            }
LABEL_110:
            j___libc_free_0(v426[1]);
LABEL_111:
            v89 = *(__int64 **)v9;
            v423 = 0u;
            v424 = 0;
            v425 = 0;
            v90 = (__int64 *)v89[4];
            v349 = v89 + 1;
            v366 = v89 + 3;
            v426[0] = (void *)v89[2];
            v426[1] = v89 + 1;
            v427 = v90;
            v428 = v89 + 3;
            v370 = 0;
            v341 = v9;
LABEL_112:
            if ( v366 == v90 && v366 == v428 && v349 == v426[0] && v349 == v426[1] )
            {
              v123 = (__int64 *)v341;
              v400[1] = (__int64)&v429;
              v400[0] = v341;
              v400[2] = (__int64)&v412;
              if ( v379 )
              {
                for ( j = *(_QWORD *)(v379 + 8); j; j = *(_QWORD *)(j + 8) )
                {
                  v442.m128i_i64[0] = (__int64)sub_1648700(j);
                  v125 = *(_QWORD *)(v442.m128i_i64[0] + 24 * (1LL - (*(_DWORD *)(v442.m128i_i64[0] + 20) & 0xFFFFFFF)));
                  if ( *(_BYTE *)(v125 + 16) != 19 )
                    sub_16BD130("Second argument of llvm.type.test must be metadata", 1u);
                  v126 = sub_1878AE0(v400, *(_QWORD *)(v125 + 24));
                  v127 = (_BYTE *)v126[1];
                  if ( v127 == (_BYTE *)v126[2] )
                  {
                    sub_187FFB0((__int64)v126, v127, &v442);
                  }
                  else
                  {
                    if ( v127 )
                    {
                      *(_QWORD *)v127 = v442.m128i_i64[0];
                      v127 = (_BYTE *)v126[1];
                    }
                    v126[1] = (__int64)(v127 + 8);
                  }
                }
              }
              if ( !n )
                goto LABEL_294;
              v359 = *(_QWORD *)(n + 8);
              if ( !v359 )
                goto LABEL_294;
              v128 = v340;
              while ( 2 )
              {
                if ( *(_DWORD *)(v341 + 24) != 32 )
                  sub_16BD130("llvm.icall.branch.funnel not supported on this target", 1u);
                v129 = sub_1648700(v359);
                v426[0] = 0;
                v426[1] = 0;
                v130 = (__int64)v129;
                v427 = 0;
                v131 = *((_DWORD *)v129 + 5) & 0xFFFFFFF;
                if ( *((char *)v129 + 23) < 0 )
                {
                  v132 = sub_1648A40((__int64)v129);
                  v134 = v132 + v133;
                  if ( *(char *)(v130 + 23) >= 0 )
                  {
                    if ( (unsigned int)(v134 >> 4) )
LABEL_597:
                      BUG();
                  }
                  else if ( (unsigned int)((v134 - sub_1648A40(v130)) >> 4) )
                  {
                    if ( *(char *)(v130 + 23) >= 0 )
                      goto LABEL_597;
                    v135 = *(_DWORD *)(sub_1648A40(v130) + 8);
                    if ( *(char *)(v130 + 23) >= 0 )
LABEL_596:
                      BUG();
                    v136 = sub_1648A40(v130);
                    v138 = *(_DWORD *)(v136 + v137 - 4) - v135;
LABEL_178:
                    v367 = ((_BYTE)v131 - 1 - v138) & 1;
                    if ( (((_BYTE)v131 - 1 - v138) & 1) == 0 )
                      sub_16BD130("number of arguments should be odd", 1u);
                    v139 = ((_BYTE)v131 - 1 - v138) & 1;
                    while ( 2 )
                    {
                      v140 = *(_DWORD *)(v130 + 20) & 0xFFFFFFF;
                      if ( *(char *)(v130 + 23) < 0 )
                      {
                        v141 = sub_1648A40(v130);
                        v143 = v141 + v142;
                        if ( *(char *)(v130 + 23) >= 0 )
                        {
                          if ( (unsigned int)(v143 >> 4) )
                            goto LABEL_597;
                        }
                        else if ( (unsigned int)((v143 - sub_1648A40(v130)) >> 4) )
                        {
                          if ( *(char *)(v130 + 23) >= 0 )
                            goto LABEL_597;
                          v144 = *(_DWORD *)(sub_1648A40(v130) + 8);
                          if ( *(char *)(v130 + 23) >= 0 )
                            goto LABEL_596;
                          v145 = sub_1648A40(v130);
                          if ( v139 == v140 - 1 - (*(_DWORD *)(v145 + v146 - 4) - v144) )
                            goto LABEL_268;
LABEL_186:
                          v147 = sub_1632FA0(*(_QWORD *)v341);
                          v148 = sub_14AC610(
                                   *(unsigned __int16 **)(v130
                                                        + 24
                                                        * (v139 - (unsigned __int64)(*(_DWORD *)(v130 + 20) & 0xFFFFFFF))),
                                   v406.m128i_i64,
                                   v147);
                          v149 = *((_BYTE *)v148 + 16) == 0 || *((_BYTE *)v148 + 16) == 3;
                          if ( !v149 )
                          {
                            v409[0].m128i_i64[0] = 0;
                            sub_16BD130("Expected branch funnel operand to be global value", 1u);
                          }
                          v409[0].m128i_i64[0] = (__int64)v148;
                          if ( v425 )
                          {
                            v150 = (v425 - 1) & (((unsigned int)v148 >> 9) ^ ((unsigned int)v148 >> 4));
                            v151 = (unsigned __int16 **)(v423.m128_u64[1] + 16LL * v150);
                            v152 = *v151;
                            if ( v148 == *v151 )
                            {
                              v153 = (unsigned __int64)v151[1];
                              goto LABEL_190;
                            }
                            v219 = v367;
                            v220 = 0;
                            while ( v152 != (unsigned __int16 *)-8LL )
                            {
                              if ( v220 || v152 != (unsigned __int16 *)-16LL )
                                v151 = v220;
                              v150 = (v425 - 1) & (v219 + v150);
                              v227 = (unsigned __int16 **)(v423.m128_u64[1] + 16LL * v150);
                              v152 = *v227;
                              if ( v148 == *v227 )
                              {
                                v153 = (unsigned __int64)v227[1];
                                goto LABEL_190;
                              }
                              ++v219;
                              v220 = v151;
                              v151 = (unsigned __int16 **)(v423.m128_u64[1] + 16LL * v150);
                            }
                            if ( !v220 )
                              v220 = v151;
                            ++v423.m128_u64[0];
                            v221 = v424 + 1;
                            if ( 4 * ((int)v424 + 1) < 3 * v425 )
                            {
                              if ( v425 - HIDWORD(v424) - v221 > v425 >> 3 )
                                goto LABEL_340;
                              sub_1874760((__int64)&v423, v425);
                              goto LABEL_346;
                            }
                          }
                          else
                          {
                            ++v423.m128_u64[0];
                          }
                          sub_1874760((__int64)&v423, 2 * v425);
LABEL_346:
                          sub_1872330((__int64)&v423, v409[0].m128i_i64, &v442);
                          v220 = (unsigned __int16 **)v442.m128i_i64[0];
                          v148 = (unsigned __int16 *)v409[0].m128i_i64[0];
                          v221 = v424 + 1;
LABEL_340:
                          LODWORD(v424) = v221;
                          if ( *v220 != (unsigned __int16 *)-8LL )
                            --HIDWORD(v424);
                          *v220 = v148;
                          v153 = 0;
                          v220[1] = 0;
LABEL_190:
                          v410.m128i_i64[0] = v153;
                          v154 = (char *)v426[1];
                          if ( v426[1] == v427 )
                          {
                            sub_18726A0((__int64)v426, (_BYTE *)v426[1], &v410);
                            v153 = v410.m128i_i64[0];
                          }
                          else
                          {
                            if ( v426[1] )
                            {
                              *(_QWORD *)v426[1] = v153;
                              v154 = (char *)v426[1];
                            }
                            v426[1] = v154 + 8;
                          }
                          v442.m128i_i64[0] = (__int64)&v442;
                          v155 = v431;
                          v442.m128i_i64[1] = 1;
                          v443.m128i_i64[0] = v153;
                          if ( v431 )
                          {
                            while ( 1 )
                            {
                              v156 = *((_QWORD *)v155 + 6);
                              v157 = (int *)*((_QWORD *)v155 + 3);
                              v158 = 0;
                              if ( v156 > v153 )
                              {
                                v157 = (int *)*((_QWORD *)v155 + 2);
                                v158 = v149;
                              }
                              if ( !v157 )
                                break;
                              v155 = v157;
                            }
                            v159 = v155;
                            if ( !v158 )
                            {
                              if ( v156 < v153 )
                                goto LABEL_202;
LABEL_330:
                              if ( v159 == &v430 )
                              {
                                v162 = 0;
LABEL_217:
                                if ( v139 == 1 )
                                {
                                  v128 = (_QWORD *)v162;
                                }
                                else if ( (_QWORD *)v162 != v128 )
                                {
                                  *(_QWORD *)(*v128 + 8LL) = v162 | *(_QWORD *)(*v128 + 8LL) & 1LL;
                                  *v128 = *(_QWORD *)v162;
                                  *(_QWORD *)(v162 + 8) &= ~1uLL;
                                  *(_QWORD *)v162 = v128;
                                }
                                v139 += 2;
                                continue;
                              }
LABEL_586:
                              v162 = (unsigned __int64)(v159 + 8);
                              goto LABEL_206;
                            }
                          }
                          else
                          {
                            v155 = &v430;
                          }
                          if ( v432 != v155 )
                          {
                            v382 = v153;
                            v218 = sub_220EF80(v155);
                            v153 = v382;
                            v159 = (int *)v218;
                            if ( *(_QWORD *)(v218 + 48) >= v382 )
                              goto LABEL_330;
LABEL_202:
                            if ( !v155 )
                            {
                              v159 = 0;
                              goto LABEL_586;
                            }
                          }
                          v160 = 1;
                          if ( v155 != &v430 )
                            v160 = v153 < *((_QWORD *)v155 + 6);
                          v375 = v160;
                          v161 = (_QWORD *)sub_22077B0(56);
                          v161[5] = 1;
                          v159 = (int *)v161;
                          v161[4] = v161 + 4;
                          v380 = v161 + 4;
                          v161[6] = v443.m128i_i64[0];
                          sub_220F040(v375, v161, v155, &v430);
                          ++v434;
                          v162 = (unsigned __int64)v380;
LABEL_206:
                          if ( (v159[10] & 1) == 0 )
                          {
                            v162 = *((_QWORD *)v159 + 4);
                            if ( (*(_BYTE *)(v162 + 8) & 1) == 0 )
                            {
                              v163 = *(_QWORD *)v162;
                              if ( (*(_BYTE *)(*(_QWORD *)v162 + 8LL) & 1) != 0 )
                              {
                                v162 = *(_QWORD *)v162;
                              }
                              else
                              {
                                v164 = *(_QWORD **)v163;
                                if ( (*(_BYTE *)(*(_QWORD *)v163 + 8LL) & 1) == 0 )
                                {
                                  v165 = (_QWORD *)*v164;
                                  if ( (*(_BYTE *)(*v164 + 8LL) & 1) != 0 )
                                  {
                                    v164 = (_QWORD *)*v164;
                                  }
                                  else
                                  {
                                    v166 = (_BYTE *)*v165;
                                    if ( (*(_BYTE *)(*v165 + 8LL) & 1) == 0 )
                                    {
                                      v167 = sub_1874270(v166);
                                      *v165 = v167;
                                      v166 = v167;
                                    }
                                    *v164 = v166;
                                    v164 = v166;
                                  }
                                  *(_QWORD *)v163 = v164;
                                }
                                *(_QWORD *)v162 = v164;
                                v162 = (unsigned __int64)v164;
                              }
                              *((_QWORD *)v159 + 4) = v162;
                            }
                          }
                          goto LABEL_217;
                        }
                      }
                      break;
                    }
                    if ( v139 != v140 - 1 )
                      goto LABEL_186;
LABEL_268:
                    v182 = v426[0];
                    ++v370;
                    v183 = (char *)v426[1] - (char *)v426[0];
                    v184 = sub_145CBF0((__int64 *)&v435, (char *)v426[1] - (char *)v426[0] + 24, 8);
                    *(_QWORD *)v184 = v130;
                    v185 = v184;
                    *(_DWORD *)(v184 + 8) = v370;
                    *(_QWORD *)(v184 + 16) = v183 >> 3;
                    if ( v183 )
                      memmove((void *)(v184 + 24), v182, v183);
                    v186 = v431;
                    v187 = v185 | 1;
                    v442.m128i_i64[1] = 1;
                    v442.m128i_i64[0] = (__int64)&v442;
                    v443.m128i_i64[0] = v187;
                    if ( v431 )
                    {
                      while ( 1 )
                      {
                        v188 = *((_QWORD *)v186 + 6);
                        v189 = (int *)*((_QWORD *)v186 + 3);
                        if ( v188 > v187 )
                          v189 = (int *)*((_QWORD *)v186 + 2);
                        if ( !v189 )
                          break;
                        v186 = v189;
                      }
                      if ( v187 < v188 )
                      {
                        if ( v432 != v186 )
                          goto LABEL_354;
                        goto LABEL_278;
                      }
                      if ( v188 >= v187 )
                        goto LABEL_351;
LABEL_278:
                      v190 = 1;
                      if ( v186 != &v430 )
                        v190 = v187 < *((_QWORD *)v186 + 6);
LABEL_280:
                      v191 = (int *)sub_22077B0(56);
                      v192 = (unsigned __int64)(v191 + 8);
                      *((_QWORD *)v191 + 5) = 1;
                      v193 = v191;
                      *((_QWORD *)v191 + 4) = v191 + 8;
                      *((_QWORD *)v191 + 6) = v443.m128i_i64[0];
                      sub_220F040(v190, v191, v186, &v430);
                      ++v434;
LABEL_281:
                      if ( (v193[10] & 1) == 0 )
                      {
                        v192 = *((_QWORD *)v193 + 4);
                        if ( (*(_BYTE *)(v192 + 8) & 1) == 0 )
                        {
                          v194 = *(_QWORD *)v192;
                          if ( (*(_BYTE *)(*(_QWORD *)v192 + 8LL) & 1) != 0 )
                          {
                            v192 = *(_QWORD *)v192;
                          }
                          else
                          {
                            v195 = *(_BYTE **)v194;
                            if ( (*(_BYTE *)(*(_QWORD *)v194 + 8LL) & 1) == 0 )
                            {
                              v196 = sub_1874270(v195);
                              *v197 = v196;
                              v195 = v196;
                            }
                            *(_QWORD *)v192 = v195;
                            v192 = (unsigned __int64)v195;
                          }
                          *((_QWORD *)v193 + 4) = v192;
                        }
                      }
                    }
                    else
                    {
                      v186 = &v430;
                      if ( v432 == &v430 )
                      {
                        v186 = &v430;
                        v190 = 1;
                        goto LABEL_280;
                      }
LABEL_354:
                      v222 = sub_220EF80(v186);
                      if ( v187 > *(_QWORD *)(v222 + 48) )
                      {
                        if ( !v186 )
                        {
LABEL_356:
                          v192 = (unsigned __int64)(v186 + 8);
                          v193 = v186;
                          goto LABEL_281;
                        }
                        goto LABEL_278;
                      }
                      v186 = (int *)v222;
LABEL_351:
                      if ( v186 != &v430 )
                        goto LABEL_356;
                      v192 = 0;
                    }
                    if ( (_QWORD *)v192 != v128 )
                    {
                      *(_QWORD *)(*v128 + 8LL) = v192 | *(_QWORD *)(*v128 + 8LL) & 1LL;
                      *v128 = *(_QWORD *)v192;
                      *(_QWORD *)(v192 + 8) &= ~1uLL;
                      *(_QWORD *)v192 = v128;
                    }
                    if ( v426[0] )
                      j_j___libc_free_0(v426[0], (char *)v427 - (char *)v426[0]);
                    v359 = *(_QWORD *)(v359 + 8);
                    if ( v359 )
                      continue;
                    v123 = (__int64 *)v341;
LABEL_294:
                    if ( !v123[1] )
                      goto LABEL_381;
                    v426[0] = 0;
                    v426[1] = 0;
                    v427 = 0;
                    LODWORD(v428) = 0;
                    if ( (_DWORD)v414 )
                    {
                      v321 = v413;
                      v322 = &v413[5 * v415];
                      if ( v413 != v322 )
                      {
                        while ( 1 )
                        {
                          v323 = *v321;
                          v324 = v321;
                          if ( *v321 != (_BYTE *)-4LL && v323 != (_BYTE *)-8LL )
                            break;
                          v321 += 5;
                          if ( v322 == v321 )
                            goto LABEL_296;
                        }
                        if ( v321 != v322 )
                        {
                          nd = (size_t)v123;
                          v325 = &v413[5 * v415];
                          while ( *v323 )
                          {
LABEL_563:
                            v324 += 5;
                            if ( v324 == v325 )
                              goto LABEL_567;
                            while ( *v324 == -8 || *v324 == -4 )
                            {
                              v324 += 5;
                              if ( v325 == v324 )
                                goto LABEL_567;
                            }
                            if ( v324 == v325 )
                            {
LABEL_567:
                              v123 = (__int64 *)nd;
                              goto LABEL_296;
                            }
                            v323 = (_BYTE *)*v324;
                          }
                          v385 = (int *)sub_161E970((__int64)v323);
                          v327 = v326;
                          sub_16C1840(&v442);
                          sub_16C1A90(v442.m128i_i32, v385, v327);
                          sub_16C1AA0(&v442, &v410);
                          v328 = sub_1882CA0((__int64)v426, v410.m128i_i64, &v442);
                          v330 = (_QWORD *)v442.m128i_i64[0];
                          if ( v328 )
                          {
                            v334 = *(_QWORD *)(v442.m128i_i64[0] + 8);
                            v335 = v334 & 0xFFFFFFFFFFFFFFFCLL;
                            if ( (v334 & 0xFFFFFFFFFFFFFFFCLL) != 0 )
                            {
                              v386 = v442.m128i_i64[0];
                              if ( (v334 & 2) == 0 )
                              {
                                v336 = sub_22077B0(48);
                                v330 = (_QWORD *)v386;
                                if ( v336 )
                                {
                                  *(_QWORD *)v336 = v336 + 16;
                                  *(_QWORD *)(v336 + 8) = 0x400000000LL;
                                }
                                v337 = v336 & 0xFFFFFFFFFFFFFFFCLL;
                                *(_QWORD *)(v386 + 8) = v336 | 2;
                                v338 = *(unsigned int *)((v336 & 0xFFFFFFFFFFFFFFFCLL) + 8);
                                if ( (unsigned int)v338 >= *(_DWORD *)(v337 + 12) )
                                {
                                  sub_16CD150(v337, (const void *)(v337 + 16), 0, 8, v329, v386);
                                  v338 = *(unsigned int *)(v337 + 8);
                                  v330 = (_QWORD *)v386;
                                }
                                *(_QWORD *)(*(_QWORD *)v337 + 8 * v338) = v335;
                                ++*(_DWORD *)(v337 + 8);
                                v335 = v330[1] & 0xFFFFFFFFFFFFFFFCLL;
                              }
                              v339 = *(unsigned int *)(v335 + 8);
                              if ( (unsigned int)v339 >= *(_DWORD *)(v335 + 12) )
                              {
                                sub_16CD150(v335, (const void *)(v335 + 16), 0, 8, v329, (int)v330);
                                v339 = *(unsigned int *)(v335 + 8);
                              }
                              *(_QWORD *)(*(_QWORD *)v335 + 8 * v339) = v323;
                              ++*(_DWORD *)(v335 + 8);
                              goto LABEL_563;
                            }
                          }
                          else
                          {
                            v331 = (int)v428;
                            ++v426[0];
                            v332 = (_DWORD)v427 + 1;
                            if ( 4 * ((int)v427 + 1) >= (unsigned int)(3 * (_DWORD)v428) )
                            {
                              v331 = 2 * (_DWORD)v428;
                            }
                            else if ( (int)v428 - HIDWORD(v427) - v332 > (unsigned int)v428 >> 3 )
                            {
                              goto LABEL_559;
                            }
                            sub_18834E0((__int64)v426, v331);
                            sub_1882CA0((__int64)v426, v410.m128i_i64, &v442);
                            v330 = (_QWORD *)v442.m128i_i64[0];
                            v332 = (_DWORD)v427 + 1;
LABEL_559:
                            LODWORD(v427) = v332;
                            if ( *v330 != -1 )
                              --HIDWORD(v427);
                            v333 = v410.m128i_i64[0];
                            v330[1] = 0;
                            *v330 = v333;
                          }
                          v330[1] = v323;
                          goto LABEL_563;
                        }
                      }
                    }
LABEL_296:
                    v198 = v123[1];
                    v199 = (unsigned int)v428;
                    v361 = v198 + 8;
                    v371 = *(_QWORD *)(v198 + 24);
                    if ( v198 + 8 == v371 )
                      goto LABEL_317;
                    v381 = v123;
                    while ( 2 )
                    {
                      v376 = *(__int64 **)(v371 + 64);
                      na = *(__int64 **)(v371 + 56);
                      if ( v376 != na )
                      {
                        while ( 1 )
                        {
                          v200 = *na;
                          if ( !*(_BYTE *)(v381[1] + 176) || (*(_BYTE *)(v200 + 12) & 0x20) != 0 )
                          {
                            v201 = *(_DWORD *)(v200 + 8);
                            if ( !v201 )
                            {
                              v200 = *(_QWORD *)(v200 + 64);
                              v201 = *(_DWORD *)(v200 + 8);
                            }
                            if ( v201 == 1 )
                            {
                              v202 = *(__int64 ***)(v200 + 96);
                              if ( v202 )
                              {
                                v203 = *v202;
                                v204 = v202[1];
                                if ( *v202 != v204 )
                                  break;
                              }
                            }
                          }
LABEL_314:
                          if ( v376 == ++na )
                            goto LABEL_315;
                        }
                        while ( 2 )
                        {
                          v205 = *v203;
                          v410.m128i_i64[0] = *v203;
                          if ( v199 )
                          {
                            v206 = v199 - 1;
                            v207 = (v199 - 1) & (37 * v205);
                            v208 = (__int64 *)((char *)v426[1] + 16 * v207);
                            v209 = *v208;
                            if ( v205 == *v208 )
                            {
LABEL_308:
                              v210 = v208[1] & 0xFFFFFFFFFFFFFFFCLL;
                              if ( (v208[1] & 2) != 0 )
                              {
                                v211 = *(__int64 **)v210;
                                v212 = *(_QWORD *)v210 + 8LL * *(unsigned int *)(v210 + 8);
LABEL_310:
                                if ( v211 != (__int64 *)v212 )
                                {
                                  do
                                  {
                                    v213 = *v211++;
                                    *((_BYTE *)sub_1878AE0(v400, v213) + 24) = 1;
                                  }
                                  while ( (__int64 *)v212 != v211 );
                                  v199 = (unsigned int)v428;
                                }
                              }
                              else
                              {
                                v211 = v208 + 1;
                                v212 = (__int64)(v208 + 2);
                                if ( v210 )
                                  goto LABEL_310;
                              }
LABEL_313:
                              if ( v204 == ++v203 )
                                goto LABEL_314;
                              continue;
                            }
                            v223 = 1;
                            v224 = 0;
                            while ( v209 != -1 )
                            {
                              if ( v209 != -2 || v224 )
                                v208 = v224;
                              v316 = v223 + 1;
                              v317 = v207 + v223;
                              v207 = v206 & v317;
                              v318 = (__int64 *)((char *)v426[1] + 16 * (v206 & v317));
                              v209 = *v318;
                              if ( v205 == *v318 )
                              {
                                v208 = (__int64 *)((char *)v426[1] + 16 * (v206 & v317));
                                goto LABEL_308;
                              }
                              v223 = v316;
                              v224 = v208;
                              v208 = v318;
                            }
                            if ( !v224 )
                              v224 = v208;
                            ++v426[0];
                            v225 = (_DWORD)v427 + 1;
                            if ( 4 * ((int)v427 + 1) < 3 * v199 )
                            {
                              if ( v199 - HIDWORD(v427) - v225 > v199 >> 3 )
                                goto LABEL_363;
                              v226 = v199;
                              goto LABEL_368;
                            }
                          }
                          else
                          {
                            ++v426[0];
                          }
                          break;
                        }
                        v226 = 2 * v199;
LABEL_368:
                        sub_18834E0((__int64)v426, v226);
                        sub_1882CA0((__int64)v426, v410.m128i_i64, &v442);
                        v224 = (__int64 *)v442.m128i_i64[0];
                        v205 = v410.m128i_i64[0];
                        v225 = (_DWORD)v427 + 1;
LABEL_363:
                        LODWORD(v427) = v225;
                        if ( *v224 != -1 )
                          --HIDWORD(v427);
                        *v224 = v205;
                        v224[1] = 0;
                        v199 = (unsigned int)v428;
                        goto LABEL_313;
                      }
LABEL_315:
                      v371 = sub_220EEE0(v371);
                      if ( v361 != v371 )
                        continue;
                      break;
                    }
                    v123 = v381;
LABEL_317:
                    if ( v199 )
                    {
                      v214 = (char *)v426[1];
                      v215 = (char *)v426[1] + 16 * v199;
                      do
                      {
                        if ( *(_QWORD *)v214 <= 0xFFFFFFFFFFFFFFFDLL )
                        {
                          v216 = *((_QWORD *)v214 + 1);
                          if ( (v216 & 2) != 0 )
                          {
                            v217 = (unsigned __int64 *)(v216 & 0xFFFFFFFFFFFFFFFCLL);
                            if ( v217 )
                            {
                              if ( (unsigned __int64 *)*v217 != v217 + 2 )
                                _libc_free(*v217);
                              j_j___libc_free_0(v217, 48);
                            }
                          }
                        }
                        v214 += 16;
                      }
                      while ( v215 != v214 );
                    }
                    j___libc_free_0(v426[1]);
LABEL_381:
                    if ( v434 )
                    {
                      v228 = v432;
                      v401 = 0;
                      v402 = 0;
                      v403 = 0;
                      for ( v426[0] = v432; v426[0] != &v430; v228 = (int *)v426[0] )
                      {
                        if ( (v228[10] & 1) != 0 )
                        {
                          v410.m128i_i32[0] = 0;
                          v229 = (unsigned __int64)(v228 + 8);
                          do
                          {
                            v232 = *(_QWORD *)(v229 + 16);
                            if ( (v232 & 1) != 0 )
                            {
                              v230 = v232 & 0xFFFFFFFFFFFFFFFELL;
                              if ( v230 )
                              {
                                v231 = *(_DWORD *)(v230 + 8);
                                if ( v410.m128i_i32[0] >= v231 )
                                  v231 = v410.m128i_i32[0];
                                v410.m128i_i32[0] = v231;
                              }
                            }
                            else if ( (v232 & 2) != 0 )
                            {
                              v442.m128i_i64[0] = v232 & 0xFFFFFFFFFFFFFFFCLL;
                              if ( (v232 & 0xFFFFFFFFFFFFFFFCLL) != 0 )
                              {
                                v233 = *((_DWORD *)sub_1874D50((__int64)&v412, v442.m128i_i64) + 2);
                                if ( v410.m128i_i32[0] >= v233 )
                                  v233 = v410.m128i_i32[0];
                                v410.m128i_i32[0] = v233;
                              }
                            }
                            v229 = *(_QWORD *)(v229 + 8) & 0xFFFFFFFFFFFFFFFELL;
                          }
                          while ( v229 );
                          v234 = v402;
                          if ( v402 == v403 )
                          {
                            sub_18729B0(&v401, v402, v426, &v410);
                          }
                          else
                          {
                            if ( v402 )
                            {
                              v402->m128i_i64[0] = (__int64)v426[0];
                              v234->m128i_i32[2] = v410.m128i_i32[0];
                              v234 = v402;
                            }
                            v402 = (__m128i *)&v234[1];
                          }
                        }
                        v426[0] = (void *)sub_220EF30(v426[0]);
                      }
                      v235 = v402;
                      v236 = v401;
                      if ( v401 != v402 )
                      {
                        v237 = (char *)v402 - (char *)v401;
                        _BitScanReverse64(&v238, v402 - v401);
                        sub_1873100((__int64)v401, (unsigned __int64)v402, 2LL * (int)(63 - (v238 ^ 0x3F)));
                        if ( v237 <= 256 )
                        {
                          sub_18737F0((__int64)v236, v235->m128i_i64);
                        }
                        else
                        {
                          sub_18737F0((__int64)v236, v236[16].m128i_i64);
                          for ( k = v240; v235 != k; v244->m128i_i32[2] = v243 )
                          {
                            v242 = k->m128i_i64[0];
                            v243 = k->m128i_u32[2];
                            v244 = k;
                            if ( k[-1].m128i_i32[2] > v243 )
                            {
                              v245 = k - 1;
                              do
                              {
                                v246 = v245->m128i_i64[0];
                                v244 = v245--;
                                v245[2].m128i_i64[0] = v246;
                                v245[2].m128i_i32[2] = v245[1].m128i_i32[2];
                              }
                              while ( v243 < v245->m128i_i32[2] );
                            }
                            ++k;
                            v244->m128i_i64[0] = v242;
                          }
                        }
                        v362 = v402;
                        if ( v401 != v402 )
                        {
                          v383 = v401;
                          v368 = (__int64 **)v123;
                          while ( 1 )
                          {
                            v426[0] = 0;
                            v426[1] = 0;
                            v427 = 0;
                            v442 = 0u;
                            v443.m128i_i64[0] = 0;
                            if ( (*(_BYTE *)(v383->m128i_i64[0] + 40) & 1) == 0 )
                            {
                              v377 = 0;
                              v372 = 0;
                              v394 = 0;
                              goto LABEL_457;
                            }
                            v247 = v383->m128i_i64[0] + 32;
                            v248 = 0;
                            v394 = 0;
                            v249 = 0;
                            do
                            {
                              v252 = *(_QWORD *)(v247 + 16);
                              if ( (v252 & 1) != 0 )
                              {
                                v253 = v252 & 0xFFFFFFFFFFFFFFFELL;
                                v254 = v253;
                                if ( v248 == v249 )
                                {
                                  v256 = v248 - v394;
                                  v257 = (v248 - v394) >> 3;
                                  if ( v257 == 0xFFFFFFFFFFFFFFFLL )
                                    sub_4262D8((__int64)"vector::_M_realloc_insert");
                                  v258 = 1;
                                  if ( v257 )
                                    v258 = (v248 - v394) >> 3;
                                  v259 = __CFADD__(v257, v258);
                                  v260 = v257 + v258;
                                  if ( v259 )
                                  {
                                    v262 = 0x7FFFFFFFFFFFFFF8LL;
                                    goto LABEL_436;
                                  }
                                  if ( v260 )
                                  {
                                    v261 = 0xFFFFFFFFFFFFFFFLL;
                                    if ( v260 <= 0xFFFFFFFFFFFFFFFLL )
                                      v261 = v260;
                                    v262 = 8 * v261;
LABEL_436:
                                    v263 = sub_22077B0(v262);
                                    v256 = v248 - v394;
                                    v264 = (char *)v263;
                                    v265 = v263 + v262;
                                  }
                                  else
                                  {
                                    v265 = 0;
                                    v264 = 0;
                                  }
                                  if ( &v264[v256] )
                                    *(_QWORD *)&v264[v256] = v254;
                                  v248 = &v264[v256 + 8];
                                  if ( v256 > 0 )
                                  {
                                    v264 = (char *)memmove(v264, v394, v256);
                                  }
                                  else if ( !v394 )
                                  {
LABEL_441:
                                    v394 = v264;
                                    v249 = (char *)v265;
                                    goto LABEL_418;
                                  }
                                  ne = (size_t)v264;
                                  j_j___libc_free_0(v394, v249 - v394);
                                  v264 = (char *)ne;
                                  goto LABEL_441;
                                }
                                if ( v248 )
                                  *(_QWORD *)v248 = v253;
                                v248 += 8;
                              }
                              else
                              {
                                v250 = v252 & 0xFFFFFFFFFFFFFFFCLL;
                                v410.m128i_i64[0] = v252 & 0xFFFFFFFFFFFFFFFCLL;
                                if ( (v252 & 2) != 0 )
                                {
                                  v251 = (char *)v426[1];
                                  if ( v426[1] == v427 )
                                  {
                                    sub_1273E00((__int64)v426, (_BYTE *)v426[1], &v410);
                                  }
                                  else
                                  {
                                    if ( v426[1] )
                                    {
                                      *(_QWORD *)v426[1] = v250;
                                      v251 = (char *)v426[1];
                                    }
                                    v426[1] = v251 + 8;
                                  }
                                }
                                else
                                {
                                  v255 = v442.m128i_i64[1];
                                  if ( v442.m128i_i64[1] == v443.m128i_i64[0] )
                                  {
                                    sub_18726A0((__int64)&v442, (_BYTE *)v442.m128i_i64[1], &v410);
                                  }
                                  else
                                  {
                                    if ( v442.m128i_i64[1] )
                                    {
                                      *(_QWORD *)v442.m128i_i64[1] = v250;
                                      v255 = v442.m128i_i64[1];
                                    }
                                    v442.m128i_i64[1] = v255 + 8;
                                  }
                                }
                              }
LABEL_418:
                              v247 = *(_QWORD *)(v247 + 8) & 0xFFFFFFFFFFFFFFFELL;
                            }
                            while ( v247 );
                            v266 = (__int64 *)v426[0];
                            v350 = v248 - v394;
                            v372 = v249 - v394;
                            srcb = (v248 - v394) >> 3;
                            v377 = srcb;
                            if ( v426[0] != v426[1] )
                            {
                              v267 = (char *)v426[1] - (char *)v426[0];
                              nb = (__int64 *)v426[1];
                              _BitScanReverse64(&v268, ((char *)v426[1] - (char *)v426[0]) >> 3);
                              sub_187AD60(
                                (char *)v426[0],
                                (char *)v426[1],
                                2LL * (int)(63 - (v268 ^ 0x3F)),
                                (__int64)&v412);
                              if ( v267 <= 128 )
                              {
                                sub_1874E90(v266, nb, (__int64)&v412);
                              }
                              else
                              {
                                v269 = v266 + 16;
                                sub_1874E90(v266, v266 + 16, (__int64)&v412);
                                if ( nb != v266 + 16 )
                                {
                                  v344 = v248;
                                  do
                                  {
                                    v270 = v269;
                                    v410.m128i_i64[0] = (__int64)&v412;
                                    v271 = *v269;
                                    while ( 1 )
                                    {
                                      v272 = *(v270 - 1);
                                      v273 = v270--;
                                      if ( !sub_18756C0(v410.m128i_i64, v271, v272) )
                                        break;
                                      v270[1] = *v270;
                                    }
                                    *v273 = v271;
                                    ++v269;
                                  }
                                  while ( nb != v269 );
                                  v248 = v344;
                                }
                              }
                            }
                            if ( v394 != v248 )
                            {
                              _BitScanReverse64(&v274, srcb);
                              sub_1872E80(v394, v248, 2LL * (int)(63 - (v274 ^ 0x3F)));
                              if ( v350 <= 128 )
                              {
                                sub_1872BB0(v394, v248);
                              }
                              else
                              {
                                sub_1872BB0(v394, v394 + 128);
                                for ( m = v394 + 128; v248 != m; *(_QWORD *)v278 = v276 )
                                {
                                  v276 = *(_QWORD *)m;
                                  v277 = *((_QWORD *)m - 1);
                                  v278 = m;
                                  v279 = m - 8;
                                  if ( *(_DWORD *)(v277 + 8) > *(_DWORD *)(*(_QWORD *)m + 8LL) )
                                  {
                                    do
                                    {
                                      *((_QWORD *)v279 + 1) = v277;
                                      v278 = v279;
                                      v277 = *((_QWORD *)v279 - 1);
                                      v279 -= 8;
                                    }
                                    while ( *(_DWORD *)(v276 + 8) < *(_DWORD *)(v277 + 8) );
                                  }
                                  m += 8;
                                }
                              }
                            }
LABEL_457:
                            sub_18895E0(
                              v368,
                              (__int64 *)v426[0],
                              ((char *)v426[1] - (char *)v426[0]) >> 3,
                              (__int64 ***)v442.m128i_i64[0],
                              (v442.m128i_i64[1] - v442.m128i_i64[0]) >> 3,
                              a2,
                              *(double *)a3.m128_u64,
                              *(double *)a4.m128i_i64,
                              *(double *)a5.m128_u64,
                              v48,
                              v49,
                              a8,
                              a9,
                              v239,
                              (__int64 *)v394,
                              v377);
                            if ( v394 )
                              j_j___libc_free_0(v394, v372);
                            if ( v442.m128i_i64[0] )
                              j_j___libc_free_0(v442.m128i_i64[0], v443.m128i_i64[0] - v442.m128i_i64[0]);
                            if ( v426[0] )
                              j_j___libc_free_0(v426[0], (char *)v427 - (char *)v426[0]);
                            if ( v362 == ++v383 )
                            {
                              v123 = (__int64 *)v368;
                              break;
                            }
                          }
                        }
                      }
                      sub_187E2A0(
                        v123,
                        *(double *)a2.m128_u64,
                        *(double *)a3.m128_u64,
                        *(double *)a4.m128i_i64,
                        *(double *)a5.m128_u64,
                        v48,
                        v49,
                        a8,
                        a9);
                      if ( v123[1] )
                      {
                        v280 = *v123;
                        v442.m128i_i64[0] = (__int64)"aliases";
                        v443.m128i_i16[0] = 259;
                        v384 = sub_1632310(v280, (__int64)&v442);
                        if ( v384 )
                        {
                          v378 = sub_161F520(v384);
                          if ( v378 )
                          {
                            v395 = 0;
                            for ( ii = sub_161F530(v384, 0); ; ii = sub_161F530(v384, v395) )
                            {
                              v294 = ii;
                              v410.m128i_i64[0] = sub_161E970(*(_QWORD *)(ii - 8LL * *(unsigned int *)(ii + 8)));
                              v410.m128i_i64[1] = v295;
                              v296 = (void *)sub_161E970(*(_QWORD *)(v294 + 8 * (1LL - *(unsigned int *)(v294 + 8))));
                              v426[1] = v297;
                              v426[0] = v296;
                              if ( !(unsigned __int8)sub_1872D70((__int64)&v419, (__int64)v426, &v442) )
                                goto LABEL_483;
                              v298 = sub_1872D70((__int64)&v419, (__int64)v426, &v442);
                              v299 = v442.m128i_i64[0];
                              if ( v298 )
                              {
                                if ( *(_DWORD *)(v442.m128i_i64[0] + 16) )
                                  goto LABEL_483;
                                goto LABEL_474;
                              }
                              v282 = v422;
                              ++v419;
                              v283 = v421 + 1;
                              if ( 4 * ((int)v421 + 1) >= 3 * v422 )
                                break;
                              if ( v422 - HIDWORD(v421) - v283 <= v422 >> 3 )
                                goto LABEL_544;
LABEL_471:
                              LODWORD(v421) = v283;
                              if ( *(_QWORD *)v299 != -1 )
                                --HIDWORD(v421);
                              v284 = _mm_loadu_si128((const __m128i *)v426);
                              *(_DWORD *)(v299 + 16) = 0;
                              *(_QWORD *)(v299 + 24) = 0;
                              *(__m128i *)v299 = v284;
LABEL_474:
                              if ( sub_16322F0(*v123, (__int64)v426[0], (__int64)v426[1]) )
                              {
                                v285 = sub_15A0FC0(*(_QWORD *)(*(_QWORD *)(v294 + 8
                                                                                * (2LL - *(unsigned int *)(v294 + 8)))
                                                             + 136LL));
                                if ( *(_DWORD *)(v285 + 8) > 0x40u )
                                  v285 = *(_QWORD *)v285;
                                v286 = *(_QWORD *)v285;
                                v287 = sub_15A0FC0(*(_QWORD *)(*(_QWORD *)(v294 + 8
                                                                                * (3LL - *(unsigned int *)(v294 + 8)))
                                                             + 136LL));
                                if ( *(_DWORD *)(v287 + 8) > 0x40u )
                                  v287 = *(_QWORD *)v287;
                                v369 = *(_QWORD *)v287;
                                v288 = sub_16322F0(*v123, (__int64)v426[0], (__int64)v426[1]);
                                v443.m128i_i16[0] = 257;
                                v289 = sub_15E58A0((__int64)&v442, v288);
                                *(_BYTE *)(v289 + 32) = *(_BYTE *)(v289 + 32) & 0xCF | (16 * (v286 & 3));
                                sub_15A5120(v289);
                                if ( v369 )
                                {
                                  *(_BYTE *)(v289 + 32) = *(_BYTE *)(v289 + 32) & 0xF0 | 4;
                                  sub_15A5120(v289);
                                }
                                v290 = sub_16321A0(*v123, v410.m128i_i64[0], v410.m128i_i64[1]);
                                v291 = v290;
                                if ( v290 )
                                {
                                  sub_164B7C0(v289, v290);
                                  sub_164D160(
                                    v291,
                                    v289,
                                    a2,
                                    *(double *)a3.m128_u64,
                                    *(double *)a4.m128i_i64,
                                    *(double *)a5.m128_u64,
                                    v292,
                                    v293,
                                    a8,
                                    a9);
                                  sub_15E3D00(v291);
                                }
                                else
                                {
                                  v443.m128i_i16[0] = 261;
                                  v442.m128i_i64[0] = (__int64)&v410;
                                  sub_164B780(v289, v442.m128i_i64);
                                }
                              }
LABEL_483:
                              if ( v378 == ++v395 )
                                goto LABEL_497;
                            }
                            v282 = 2 * v422;
LABEL_544:
                            sub_1874300((__int64)&v419, v282);
                            sub_1872D70((__int64)&v419, (__int64)v426, &v442);
                            v299 = v442.m128i_i64[0];
                            v283 = v421 + 1;
                            goto LABEL_471;
                          }
                        }
LABEL_497:
                        if ( v123[1] )
                        {
                          v300 = *v123;
                          v442.m128i_i64[0] = (__int64)"symvers";
                          v443.m128i_i16[0] = 259;
                          v301 = sub_1632310(v300, (__int64)&v442);
                          v302 = v301;
                          if ( v301 )
                          {
                            nc = sub_161F520(v301);
                            if ( nc )
                            {
                              for ( jj = 0; jj != nc; ++jj )
                              {
                                v304 = sub_161F530(v302, jj);
                                v398[0] = sub_161E970(*(_QWORD *)(v304 - 8LL * *(unsigned int *)(v304 + 8)));
                                v398[1] = v305;
                                v306 = sub_161E970(*(_QWORD *)(v304 + 8 * (1LL - *(unsigned int *)(v304 + 8))));
                                v399[1] = v307;
                                v399[0] = v306;
                                if ( (unsigned __int8)sub_1872D70((__int64)&v419, (__int64)v398, &v442) )
                                {
                                  v308 = *v123;
                                  v411 = 261;
                                  v410.m128i_i64[0] = (__int64)v399;
                                  v406.m128i_i64[0] = (__int64)", ";
                                  v405 = 1283;
                                  v404.m128i_i64[1] = (__int64)v398;
                                  v404.m128i_i64[0] = (__int64)".symver ";
                                  v408 = 1;
                                  v407 = 3;
                                  sub_14EC200(v409, &v404, &v406);
                                  sub_14EC200((__m128i *)v426, v409, &v410);
                                  sub_16E2FC0(v442.m128i_i64, (__int64)v426);
                                  if ( v442.m128i_i64[1] > (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL
                                                                            - *(_QWORD *)(v308 + 96)) )
                                    sub_4262D8((__int64)"basic_string::append");
                                  sub_2241490(v308 + 88, (const char *)v442.m128i_i64[0]);
                                  v309 = *(_QWORD *)(v308 + 96);
                                  if ( v309 && *(_BYTE *)(*(_QWORD *)(v308 + 88) + v309 - 1) != 10 )
                                    sub_2240F50(v308 + 88, 10);
                                  sub_2240A30(&v442);
                                }
                              }
                            }
                          }
                        }
                      }
                      if ( v401 )
                        j_j___libc_free_0(v401, (char *)v403 - (char *)v401);
                      v310 = 1;
                    }
                    else
                    {
                      v310 = 0;
                    }
                    v396 = v310;
                    j___libc_free_0(v423.m128_u64[1]);
                    j___libc_free_0(v420);
                    v311 = v396;
                    if ( v416 != v418 )
                    {
                      _libc_free((unsigned __int64)v416);
                      v311 = v396;
                    }
                    if ( v415 )
                    {
                      v312 = v413;
                      v313 = v311;
                      v314 = &v413[5 * v415];
                      do
                      {
                        if ( *v312 != -4 && *v312 != -8 )
                        {
                          v315 = v312[2];
                          if ( v315 )
                            j_j___libc_free_0(v315, v312[4] - v315);
                        }
                        v312 += 5;
                      }
                      while ( v314 != v312 );
                      v311 = v313;
                    }
                    v397 = v311;
                    j___libc_free_0(v413);
                    sub_1605960((__int64)&v435);
                    sub_1873A50((__int64)v431);
                    return v397;
                  }
                }
                break;
              }
              v138 = 0;
              goto LABEL_178;
            }
            v91 = &v442;
            v443.m128i_i64[1] = 0;
            v92 = v426;
            v443.m128i_i64[0] = (__int64)sub_18564C0;
            v93 = sub_18564A0;
            v94 = &v442;
            if ( ((unsigned __int8)sub_18564A0 & 1) == 0 )
              goto LABEL_115;
            while ( 1 )
            {
              v93 = *(__int64 (__fastcall **)(__int64))((char *)v93 + (_QWORD)*v92 - 1);
LABEL_115:
              v95 = v93((__int64)v92);
              v96 = (_BYTE *)v95;
              if ( v95 )
                break;
              while ( 1 )
              {
                v97 = v94[1].m128i_i64[1];
                v93 = (__int64 (__fastcall *)(__int64))v94[1].m128i_i64[0];
                v94 = ++v91;
                v92 = (void **)((char *)v426 + v97);
                if ( ((unsigned __int8)v93 & 1) != 0 )
                  break;
                v95 = v93((__int64)v92);
                v96 = (_BYTE *)v95;
                if ( v95 )
                  goto LABEL_118;
              }
            }
LABEL_118:
            if ( *(_BYTE *)(v95 + 16) == 3 && ((*(_BYTE *)(v95 + 32) & 0xF) == 1 || sub_15E4F60((__int64)v96)) )
              goto LABEL_151;
            LODWORD(v417) = 0;
            sub_1626560((__int64)v96, 19, (__int64)&v416);
            if ( (v96[32] & 0xF) == 1 )
            {
              v356 = 0;
              v352 = 1;
            }
            else
            {
              v352 = sub_15E4F60((__int64)v96);
              v356 = !v352;
            }
            if ( v96[16] )
            {
              v99 = 0;
            }
            else
            {
              v98.m128i_i64[0] = (__int64)sub_1649960((__int64)v96);
              v442 = v98;
              v99 = sub_1872D70((__int64)&v419, (__int64)&v442, &v410);
              if ( !v99 )
              {
                if ( (unsigned __int8)sub_15E3650((__int64)v96, 0) )
                  goto LABEL_129;
                v99 = v352 || v342 == 0;
                if ( !v99 && (v96[32] & 0xFu) - 7 > 1 )
                {
                  v356 = 1;
                  goto LABEL_129;
                }
LABEL_151:
                v118 = &v442;
                v443.m128i_i64[1] = 0;
                v119 = v426;
                v443.m128i_i64[0] = (__int64)sub_1856470;
                v120 = sub_1856440;
                v121 = &v442;
                if ( ((unsigned __int8)sub_1856440 & 1) == 0 )
                  goto LABEL_153;
                while ( 1 )
                {
                  v120 = *(__int64 (__fastcall **)(__int64))((char *)v120 + (_QWORD)*v119 - 1);
LABEL_153:
                  if ( (unsigned __int8)v120((__int64)v119) )
                  {
LABEL_156:
                    v90 = v427;
                    goto LABEL_112;
                  }
                  while ( 1 )
                  {
                    v122 = v121[1].m128i_i64[1];
                    v120 = (__int64 (__fastcall *)(__int64))v121[1].m128i_i64[0];
                    v121 = ++v118;
                    v119 = (void **)((char *)v426 + v122);
                    if ( ((unsigned __int8)v120 & 1) != 0 )
                      break;
                    if ( (unsigned __int8)v120((__int64)v119) )
                      goto LABEL_156;
                  }
                }
              }
              v168.m128i_i64[0] = (__int64)sub_1649960((__int64)v96);
              v442 = v168;
              if ( (unsigned __int8)sub_1872D70((__int64)&v419, (__int64)&v442, &v410) )
              {
                v356 |= *(_DWORD *)(v410.m128i_i64[0] + 16) == 0;
              }
              else
              {
                v169 = (__m128 *)sub_1874480((__int64)&v419, (__int64)&v442, v410.m128i_i64[0]);
                a5 = (__m128)_mm_loadu_si128(&v442);
                v356 = v99;
                v169[1].m128_i32[0] = 0;
                v169[1].m128_u64[1] = 0;
                *v169 = a5;
              }
            }
LABEL_129:
            v100 = 8LL * (unsigned int)v417;
            v353 = (unsigned int)v417;
            srca = v416;
            v101 = sub_145CBF0((__int64 *)&v435, v100 + 24, 8);
            *(_QWORD *)v101 = v96;
            v102 = v101;
            *(_QWORD *)(v101 + 8) = v353;
            *(_BYTE *)(v101 + 17) = v99;
            *(_BYTE *)(v101 + 16) = v356;
            if ( v100 )
            {
              v357 = v101;
              memmove((void *)(v101 + 24), srca, v100);
              v102 = v357;
            }
            v409[0].m128i_i64[0] = v102;
            v410.m128i_i64[0] = (__int64)v96;
            v103 = sub_1872330((__int64)&v423, v410.m128i_i64, &v442);
            v104 = (__int64 *)v442.m128i_i64[0];
            if ( !v103 )
            {
              v105 = v425;
              ++v423.m128_u64[0];
              v106 = v424 + 1;
              if ( 4 * ((int)v424 + 1) >= 3 * v425 )
              {
                v105 = 2 * v425;
              }
              else if ( v425 - HIDWORD(v424) - v106 > v425 >> 3 )
              {
                goto LABEL_134;
              }
              sub_1874760((__int64)&v423, v105);
              sub_1872330((__int64)&v423, v410.m128i_i64, &v442);
              v104 = (__int64 *)v442.m128i_i64[0];
              v106 = v424 + 1;
LABEL_134:
              LODWORD(v424) = v106;
              if ( *v104 != -8 )
                --HIDWORD(v424);
              v107 = v410.m128i_i64[0];
              v104[1] = 0;
              *v104 = v107;
            }
            v104[1] = v409[0].m128i_i64[0];
            v358 = &v416[8 * (unsigned int)v417];
            if ( v416 == v358 )
              goto LABEL_151;
            v108 = (unsigned __int64)v416;
            while ( 2 )
            {
              v109 = *(_QWORD *)v108;
              if ( *(_DWORD *)(*(_QWORD *)v108 + 8LL) != 2 )
                sub_16BD130("All operands of type metadata must have 2 elements", 1u);
              if ( (v96[33] & 0x1C) != 0 )
                sub_16BD130("Bit set element may not be thread-local", 1u);
              if ( v96[16] == 3 && (v96[34] & 0x20) != 0 )
                sub_16BD130("A member of a type identifier may not have an explicit section", 1u);
              v110 = *(_QWORD *)(v109 - 16);
              if ( *(_BYTE *)v110 != 1 )
                sub_16BD130("Type offset must be a constant", 1u);
              if ( *(_BYTE *)(*(_QWORD *)(v110 + 136) + 16LL) != 13 )
                sub_16BD130("Type offset must be an integer constant", 1u);
              v111 = *(_QWORD *)(v109 - 8);
              v112 = v415;
              v410.m128i_i64[0] = v111;
              if ( v415 )
              {
                v113 = (v415 - 1) & (((unsigned int)v111 >> 9) ^ ((unsigned int)v111 >> 4));
                v114 = (__int64)&v413[5 * v113];
                v115 = *(_QWORD *)v114;
                if ( v111 == *(_QWORD *)v114 )
                {
LABEL_146:
                  v116 = *(_QWORD **)(v114 + 24);
                  v117 = *(_BYTE **)(v114 + 32);
                  *(_DWORD *)(v114 + 8) = ++v370;
                  if ( v116 != (_QWORD *)v117 )
                  {
                    if ( v116 )
                    {
                      *v116 = v409[0].m128i_i64[0];
                      v116 = *(_QWORD **)(v114 + 24);
                    }
                    *(_QWORD *)(v114 + 24) = v116 + 1;
LABEL_150:
                    v108 += 8LL;
                    if ( v358 == (_BYTE *)v108 )
                      goto LABEL_151;
                    continue;
                  }
LABEL_239:
                  sub_18726A0(v114 + 16, v117, v409);
                  goto LABEL_150;
                }
                v319 = 1;
                v320 = 0;
                while ( v115 != -4 )
                {
                  if ( !v320 && v115 == -8 )
                    v320 = v114;
                  v113 = (v415 - 1) & (v319 + v113);
                  v114 = (__int64)&v413[5 * v113];
                  v115 = *(_QWORD *)v114;
                  if ( v111 == *(_QWORD *)v114 )
                    goto LABEL_146;
                  ++v319;
                }
                if ( v320 )
                  v114 = v320;
                ++v412;
                v170 = v414 + 1;
                if ( 4 * ((int)v414 + 1) < 3 * v415 )
                {
                  if ( v415 - HIDWORD(v414) - v170 <= v415 >> 3 )
                  {
LABEL_235:
                    sub_1874B30((__int64)&v412, v112);
                    sub_18721D0((__int64)&v412, v410.m128i_i64, &v442);
                    v114 = v442.m128i_i64[0];
                    v111 = v410.m128i_i64[0];
                    v170 = v414 + 1;
                  }
                  LODWORD(v414) = v170;
                  if ( *(_QWORD *)v114 != -4 )
                    --HIDWORD(v414);
                  ++v370;
                  v117 = 0;
                  *(_QWORD *)v114 = v111;
                  *(_QWORD *)(v114 + 16) = 0;
                  *(_QWORD *)(v114 + 24) = 0;
                  *(_QWORD *)(v114 + 32) = 0;
                  *(_DWORD *)(v114 + 8) = v370;
                  goto LABEL_239;
                }
              }
              else
              {
                ++v412;
              }
              break;
            }
            v112 = 2 * v415;
            goto LABEL_235;
          }
        }
        while ( 1 )
        {
          v55 = *v54;
          if ( (*(_BYTE *)(*v54 + 12) & 0x20) != 0 )
          {
            v56 = *(_QWORD **)(v55 + 40);
            v57 = *(_QWORD **)(v55 + 48);
            if ( v57 != v56 )
              break;
          }
LABEL_46:
          if ( v53 == ++v54 )
            goto LABEL_66;
        }
        while ( 1 )
        {
          v61 = (int)v428;
          v62 = *(_QWORD *)(*v56 & 0xFFFFFFFFFFFFFFF8LL);
          v423.m128_u64[0] = v62;
          if ( !(_DWORD)v428 )
            break;
          v58 = ((_DWORD)v428 - 1) & (37 * v62);
          v59 = (unsigned __int64 *)((char *)v426[1] + 8 * v58);
          v60 = *v59;
          if ( v62 != *v59 )
          {
            v65 = 1;
            v63 = 0;
            while ( v60 != -1 )
            {
              if ( v60 != -2 || v63 )
                v59 = v63;
              v58 = ((_DWORD)v428 - 1) & (v65 + v58);
              v60 = *((_QWORD *)v426[1] + v58);
              if ( v62 == v60 )
                goto LABEL_51;
              ++v65;
              v63 = v59;
              v59 = (unsigned __int64 *)((char *)v426[1] + 8 * v58);
            }
            if ( !v63 )
              v63 = v59;
            ++v426[0];
            v64 = (_DWORD)v427 + 1;
            if ( 4 * ((int)v427 + 1) < (unsigned int)(3 * (_DWORD)v428) )
            {
              if ( (int)v428 - HIDWORD(v427) - v64 <= (unsigned int)v428 >> 3 )
              {
LABEL_55:
                sub_142F750((__int64)v426, v61);
                sub_1880D60((__int64)v426, (__int64 *)&v423, &v442);
                v63 = (unsigned __int64 *)v442.m128i_i64[0];
                v62 = v423.m128_u64[0];
                v64 = (_DWORD)v427 + 1;
              }
              LODWORD(v427) = v64;
              if ( *v63 != -1 )
                --HIDWORD(v427);
              *v63 = v62;
              goto LABEL_51;
            }
LABEL_54:
            v61 = 2 * (_DWORD)v428;
            goto LABEL_55;
          }
LABEL_51:
          if ( v57 == ++v56 )
            goto LABEL_46;
        }
        ++v426[0];
        goto LABEL_54;
      }
    }
    else if ( !a1[2] )
    {
      return 0;
    }
    if ( !v379 )
      goto LABEL_5;
    v20 = *(_QWORD *)(v379 + 8);
    if ( !v20 )
      goto LABEL_5;
    goto LABEL_4;
  }
  if ( !a1[2] )
    goto LABEL_41;
  do
  {
LABEL_4:
    v21 = v20;
    v20 = *(_QWORD *)(v20 + 8);
    v22 = sub_1648700(v21);
    sub_187F790(
      v9,
      (__int64)v22,
      a2,
      *(double *)a3.m128_u64,
      *(double *)a4.m128i_i64,
      *(double *)a5.m128_u64,
      v23,
      v24,
      a8,
      a9);
  }
  while ( v20 );
LABEL_5:
  if ( n && *(_QWORD *)(n + 8) )
    sub_16BD130("unexpected call to llvm.icall.branch.funnel during import phase", 1u);
  v25 = *(__int64 **)v9;
  v435 = v437;
  v436 = 0x800000000LL;
  v442.m128i_i64[1] = 0x800000000LL;
  v442.m128i_i64[0] = (__int64)&v443;
  v26 = (__int64 *)v25[4];
  v393 = v25 + 3;
  if ( v26 != v25 + 3 )
  {
    while ( 1 )
    {
      if ( !v26 )
        BUG();
      if ( (*(_BYTE *)(v26 - 3) & 0xFu) - 7 <= 1 )
        goto LABEL_9;
      v27 = v26 - 7;
      v28 = *(_QWORD *)(v9 + 16);
      v426[0] = (void *)sub_1649960((__int64)(v26 - 7));
      v426[1] = v29;
      sub_12C70A0(&v429, (__int64)v426);
      if ( v28 + 192 == sub_187FEE0(v28 + 184, (__int64)&v429) )
      {
        sub_2240A30(&v429);
        v41 = *(_QWORD *)(v9 + 16);
        v426[0] = (void *)sub_1649960((__int64)(v26 - 7));
        v426[1] = v42;
        sub_12C70A0(&v429, (__int64)v426);
        if ( v41 + 240 == sub_187FEE0(v41 + 232, (__int64)&v429) )
        {
          sub_2240A30(&v429);
        }
        else
        {
          sub_2240A30(&v429);
          v45 = v442.m128i_u32[2];
          if ( v442.m128i_i32[2] >= (unsigned __int32)v442.m128i_i32[3] )
          {
            sub_16CD150((__int64)&v442, &v443, 0, 8, v43, v44);
            v45 = v442.m128i_u32[2];
          }
          *(_QWORD *)(v442.m128i_i64[0] + 8 * v45) = v27;
          ++v442.m128i_i32[2];
        }
LABEL_9:
        v26 = (__int64 *)v26[1];
        if ( v26 == v393 )
          goto LABEL_16;
      }
      else
      {
        sub_2240A30(&v429);
        v32 = (unsigned int)v436;
        if ( (unsigned int)v436 >= HIDWORD(v436) )
        {
          sub_16CD150((__int64)&v435, v437, 0, 8, v30, v31);
          v32 = (unsigned int)v436;
        }
        v435[v32] = (__int64)v27;
        LODWORD(v436) = v436 + 1;
        v26 = (__int64 *)v26[1];
        if ( v26 == v393 )
        {
LABEL_16:
          v33 = v435;
          v34 = &v435[(unsigned int)v436];
          if ( v435 != v34 )
          {
            do
            {
              v35 = *v33++;
              sub_1888020(
                (__int64 **)v9,
                v35,
                1,
                a2,
                *(double *)a3.m128_u64,
                *(double *)a4.m128i_i64,
                *(double *)a5.m128_u64,
                v18,
                v19,
                a8,
                a9);
            }
            while ( v34 != v33 );
          }
          v36 = (__m128i *)v442.m128i_i64[0];
          v37 = v442.m128i_i64[0] + 8LL * v442.m128i_u32[2];
          if ( v37 != v442.m128i_i64[0] )
          {
            v38 = (__int64 *)v442.m128i_i64[0];
            do
            {
              v39 = *v38++;
              sub_1888020(
                (__int64 **)v9,
                v39,
                0,
                a2,
                *(double *)a3.m128_u64,
                *(double *)a4.m128i_i64,
                *(double *)a5.m128_u64,
                v18,
                v19,
                a8,
                a9);
            }
            while ( (__int64 *)v37 != v38 );
            v36 = (__m128i *)v442.m128i_i64[0];
          }
          if ( v36 != &v443 )
            _libc_free((unsigned __int64)v36);
          break;
        }
      }
    }
  }
  if ( v435 != v437 )
    _libc_free((unsigned __int64)v435);
  return 1;
}
