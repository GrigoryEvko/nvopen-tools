// Function: sub_1B01A40
// Address: 0x1b01a40
//
__int64 __fastcall sub_1B01A40(
        __m128i *a1,
        unsigned int a2,
        unsigned int a3,
        char a4,
        char a5,
        unsigned __int8 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        char a15,
        char a16,
        unsigned int a17,
        int a18,
        unsigned __int8 a19,
        __m128i *a20,
        __int64 a21,
        __int64 a22,
        __int64 a23,
        __int64 *a24,
        unsigned __int8 a25)
{
  unsigned __int8 v26; // r13
  __int64 *v27; // r12
  unsigned __int64 v28; // rbx
  __int64 v29; // rax
  char *v30; // rax
  const void *v31; // rsi
  __int64 v32; // rdx
  unsigned __int64 v33; // rdx
  char *v34; // r15
  _QWORD *v35; // rsi
  unsigned __int64 v36; // rcx
  _QWORD *v37; // rax
  _DWORD *v38; // rdx
  unsigned int v39; // r13d
  __int64 v41; // rax
  __int64 v42; // rax
  int v43; // ecx
  unsigned int v44; // esi
  __int64 v45; // rax
  __int64 v46; // rbx
  __int64 *v47; // rsi
  __int128 v48; // rdi
  const void *v49; // r14
  size_t v50; // r12
  char *v51; // rbx
  char *v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rcx
  _QWORD *v55; // r8
  _QWORD *v56; // r9
  __int64 **v57; // r12
  __int64 **i; // rbx
  __int64 *v59; // rax
  double v60; // xmm4_8
  double v61; // xmm5_8
  __int64 v62; // r14
  __int64 v63; // rbx
  __int64 v64; // r15
  __int64 v65; // rdi
  __int64 v66; // rax
  __int64 v67; // rsi
  unsigned __int8 *v68; // rsi
  __int64 v69; // rax
  __m128i *v70; // rax
  _QWORD *v71; // rax
  __int64 v72; // rbx
  __int64 *v73; // r13
  __int64 v74; // rcx
  __int64 v75; // rax
  char *v76; // rsi
  char v77; // al
  __int64 v78; // rcx
  _QWORD *v79; // r8
  _QWORD *v80; // r9
  char *v81; // rdx
  unsigned int v82; // eax
  unsigned int v83; // esi
  _QWORD *v84; // rax
  char *v85; // rsi
  char *v86; // rbx
  _QWORD *v87; // r12
  char *v88; // rax
  char *v89; // r12
  unsigned __int64 v90; // rdi
  unsigned int v91; // r13d
  __int64 v92; // r12
  __int64 v93; // rax
  __int64 v94; // rdx
  __int64 v95; // r14
  __int64 v96; // rbx
  __int64 v97; // rcx
  __int64 v98; // r8
  unsigned int v99; // eax
  __int64 v100; // r9
  int v101; // edx
  __int64 v102; // rsi
  __int64 v103; // rcx
  __int64 v104; // rdi
  __int64 v105; // rsi
  __int64 v106; // r12
  __int64 v107; // rcx
  __int64 v108; // r10
  char *v109; // r15
  __int64 v110; // rdx
  unsigned int v111; // ecx
  int v112; // eax
  __int64 v113; // rdx
  _QWORD *v114; // rax
  __int64 v115; // rcx
  unsigned __int64 v116; // rdx
  __int64 v117; // rdx
  __int64 v118; // rdx
  __int64 v119; // rcx
  __int64 v120; // rax
  _BYTE *v121; // rsi
  char *v122; // rsi
  char *v123; // rsi
  __int64 v124; // rax
  __int64 v125; // r12
  _QWORD *v126; // rax
  _QWORD *v127; // r14
  __int64 v128; // rax
  _QWORD *v129; // rbx
  __int64 v130; // rdx
  _BYTE *v131; // r12
  __int64 v132; // r15
  __int64 j; // rbx
  __int64 v134; // rax
  __int64 *v135; // rbx
  __int64 v136; // r14
  unsigned int v137; // edx
  unsigned int v138; // esi
  __int64 v139; // rax
  __int64 v140; // rdi
  __int64 v141; // rax
  __int64 v142; // rcx
  __int64 v143; // r8
  __int64 v144; // r9
  __int64 v145; // r15
  unsigned int v146; // r15d
  __int64 v147; // r10
  __int64 *v148; // rbx
  __int64 *v149; // r12
  __int64 v150; // rax
  unsigned __int64 *v151; // rax
  unsigned __int64 v152; // rdx
  unsigned __int64 *v153; // r13
  __int64 v154; // rax
  __int64 v155; // rax
  _QWORD *v156; // rbx
  _QWORD *v157; // r12
  __int64 v158; // rsi
  __int64 *v159; // rbx
  __int64 v160; // r12
  __int64 v161; // r15
  char v162; // di
  unsigned int v163; // esi
  __int64 v164; // rdx
  __int64 v165; // rax
  __int64 v166; // rcx
  __int64 v167; // rax
  __int64 v168; // rdx
  __int64 v169; // r14
  _QWORD *v170; // rdi
  __int64 v171; // rax
  unsigned __int64 *v172; // rcx
  unsigned __int64 v173; // rdx
  double v174; // xmm4_8
  double v175; // xmm5_8
  __int64 v176; // rdx
  char v177; // cl
  unsigned int v178; // eax
  __int64 v179; // rsi
  __int64 v180; // rdx
  __int64 v181; // rdi
  __int64 v182; // rcx
  unsigned __int64 *v183; // rcx
  unsigned __int64 v184; // rdx
  double v185; // xmm4_8
  double v186; // xmm5_8
  __int64 *v187; // rcx
  __int64 v188; // rax
  __int64 v189; // r15
  _QWORD *v190; // rdi
  __int64 v191; // r14
  _QWORD *v192; // r12
  unsigned int v193; // edx
  __int64 v194; // r13
  __int64 *v195; // r12
  __int64 v196; // rdx
  unsigned __int64 v197; // rcx
  __int64 v198; // rdx
  __int64 v199; // r13
  unsigned int m; // r12d
  __int64 v201; // rdi
  __int64 v202; // rax
  __int64 v203; // rdx
  __int64 v204; // rbx
  __int64 v205; // r14
  unsigned int v206; // ecx
  unsigned int v207; // esi
  __int64 v208; // rax
  __int64 v209; // rdx
  __int64 v210; // rax
  __int64 v211; // rdi
  __int64 v212; // r13
  __int64 v213; // rax
  __int64 **v214; // rbx
  __int64 **v215; // r12
  __int64 v216; // r14
  int v217; // r8d
  int v218; // r9d
  __int64 v219; // rax
  char *v220; // r14
  __int64 *v221; // rbx
  _BYTE *v222; // rsi
  __int64 v223; // r12
  __int64 v224; // r13
  __int64 v225; // r15
  __int64 v226; // rax
  __int64 v227; // r13
  __int64 v228; // rax
  _BYTE *v229; // rax
  __int64 v230; // rdx
  __int64 v231; // rcx
  int v232; // r8d
  int v233; // r9d
  __int64 *v234; // r13
  __int64 *v235; // rbx
  __int64 *v236; // r14
  __int64 v237; // r12
  unsigned __int64 v238; // rax
  char *v239; // rsi
  __int64 *v240; // r13
  __int64 *v241; // rbx
  unsigned __int64 v242; // rax
  double v243; // xmm4_8
  double v244; // xmm5_8
  __int64 v245; // r14
  __int64 *v246; // rax
  __int64 *ii; // rdx
  _QWORD *v248; // r14
  _QWORD *v249; // rax
  _QWORD *v250; // rcx
  _QWORD *v251; // rax
  char *v252; // rax
  char *v253; // rcx
  int v254; // eax
  size_t v255; // r13
  char *v256; // rax
  double v257; // xmm4_8
  double v258; // xmm5_8
  __int64 *v259; // r13
  __int64 *v260; // rbx
  __int64 v261; // r12
  __m128i *v262; // rdi
  __int64 v263; // rdx
  double v264; // xmm4_8
  double v265; // xmm5_8
  _QWORD *v266; // r12
  bool v267; // al
  void *v268; // rax
  void *v269; // rsi
  unsigned __int64 v270; // rbx
  __int64 v271; // rax
  __int64 *v272; // rcx
  size_t v273; // r13
  __int64 v274; // r13
  _QWORD *v275; // r15
  _QWORD *v276; // rbx
  __int64 v277; // rdx
  __int64 v278; // rax
  __int64 v279; // rax
  _DWORD *v280; // rcx
  _DWORD *v281; // rdx
  unsigned __int64 v282; // rdx
  unsigned __int64 v283; // rax
  unsigned __int64 v284; // rcx
  __int64 v285; // rax
  __int64 v286; // rax
  __int64 v287; // rax
  __int64 v288; // rax
  __int64 v289; // rax
  __int64 v290; // rax
  __int64 *v291; // r15
  __int64 *v292; // r12
  __int64 v293; // r15
  _QWORD *v294; // rbx
  _QWORD *v295; // r12
  __int64 v296; // rsi
  __int64 *v297; // rsi
  __int64 *v298; // rcx
  __int64 v299; // rdx
  __int64 *v300; // rax
  __int64 v301; // rsi
  __int64 *v302; // rax
  __int64 *v303; // r15
  __int64 v304; // r14
  __int64 *v305; // rbx
  unsigned int v306; // edx
  void *v307; // rsi
  unsigned int v308; // edx
  __int64 v309; // rcx
  int v310; // r8d
  unsigned int v311; // eax
  __int64 v312; // rdi
  __int64 v313; // r13
  __int64 v314; // r12
  __int64 v315; // rax
  int v316; // r8d
  unsigned int v317; // eax
  void *v318; // rdi
  char v319; // al
  __int64 *v320; // rdx
  int v321; // esi
  int v322; // eax
  void *v323; // rax
  char *v324; // r12
  __int64 v325; // rax
  _QWORD *v326; // rdx
  __int64 v327; // rdi
  _QWORD *v328; // rdx
  __int64 v329; // rsi
  __int64 *v330; // rbx
  __int64 *v331; // r12
  __int64 v332; // rdi
  _BOOL4 v333; // [rsp+34h] [rbp-57Ch]
  __int64 v334; // [rsp+48h] [rbp-568h]
  __int64 v335; // [rsp+50h] [rbp-560h]
  __int64 v336; // [rsp+58h] [rbp-558h]
  char v337; // [rsp+78h] [rbp-538h]
  char v338; // [rsp+79h] [rbp-537h]
  char v339; // [rsp+7Ah] [rbp-536h]
  unsigned __int8 v340; // [rsp+7Bh] [rbp-535h]
  unsigned __int8 v341; // [rsp+7Ch] [rbp-534h]
  __int64 v342; // [rsp+80h] [rbp-530h]
  _QWORD *v343; // [rsp+88h] [rbp-528h]
  __int64 v344; // [rsp+90h] [rbp-520h]
  unsigned __int64 v345; // [rsp+98h] [rbp-518h]
  unsigned int v346; // [rsp+A0h] [rbp-510h]
  __int64 v347; // [rsp+A8h] [rbp-508h]
  __int64 *v348; // [rsp+A8h] [rbp-508h]
  unsigned int src; // [rsp+B0h] [rbp-500h]
  int srca; // [rsp+B0h] [rbp-500h]
  __int64 *srcb; // [rsp+B0h] [rbp-500h]
  char *srcc; // [rsp+B0h] [rbp-500h]
  size_t nb; // [rsp+B8h] [rbp-4F8h]
  int n; // [rsp+B8h] [rbp-4F8h]
  size_t na; // [rsp+B8h] [rbp-4F8h]
  unsigned __int64 v357; // [rsp+C0h] [rbp-4F0h]
  __int64 *v358; // [rsp+C0h] [rbp-4F0h]
  __int64 v360; // [rsp+C8h] [rbp-4E8h]
  __int64 v361; // [rsp+C8h] [rbp-4E8h]
  _BYTE *v362; // [rsp+C8h] [rbp-4E8h]
  __int64 *k; // [rsp+C8h] [rbp-4E8h]
  unsigned int v364; // [rsp+C8h] [rbp-4E8h]
  __int64 *v365; // [rsp+C8h] [rbp-4E8h]
  unsigned __int64 v366; // [rsp+D0h] [rbp-4E0h] BYREF
  __m128i *v367; // [rsp+D8h] [rbp-4D8h] BYREF
  unsigned int v368; // [rsp+E4h] [rbp-4CCh] BYREF
  char *v369; // [rsp+E8h] [rbp-4C8h] BYREF
  char *v370; // [rsp+F0h] [rbp-4C0h] BYREF
  __int64 v371; // [rsp+F8h] [rbp-4B8h] BYREF
  char *v372; // [rsp+100h] [rbp-4B0h] BYREF
  _QWORD *v373; // [rsp+108h] [rbp-4A8h] BYREF
  char *v374; // [rsp+110h] [rbp-4A0h] BYREF
  __int64 *v375; // [rsp+118h] [rbp-498h]
  char *v376; // [rsp+120h] [rbp-490h]
  __int64 *v377; // [rsp+130h] [rbp-480h] BYREF
  __int64 *v378; // [rsp+138h] [rbp-478h]
  __int64 *v379; // [rsp+140h] [rbp-470h]
  _QWORD v380[4]; // [rsp+150h] [rbp-460h] BYREF
  __int64 *v381; // [rsp+170h] [rbp-440h] BYREF
  __int64 *v382; // [rsp+178h] [rbp-438h]
  __int64 v383; // [rsp+180h] [rbp-430h]
  void *v384; // [rsp+190h] [rbp-420h] BYREF
  void *v385; // [rsp+198h] [rbp-418h]
  char *v386; // [rsp+1A0h] [rbp-410h]
  _BYTE *v387; // [rsp+1B0h] [rbp-400h] BYREF
  _BYTE *v388; // [rsp+1B8h] [rbp-3F8h]
  _BYTE *v389; // [rsp+1C0h] [rbp-3F0h]
  void *v390; // [rsp+1D0h] [rbp-3E0h] BYREF
  __int64 v391; // [rsp+1D8h] [rbp-3D8h] BYREF
  char *v392; // [rsp+1E0h] [rbp-3D0h]
  __int64 v393; // [rsp+1E8h] [rbp-3C8h]
  __int64 v394; // [rsp+1F0h] [rbp-3C0h]
  char *v395; // [rsp+200h] [rbp-3B0h] BYREF
  __int64 v396; // [rsp+208h] [rbp-3A8h] BYREF
  __int64 v397; // [rsp+210h] [rbp-3A0h]
  __int64 v398; // [rsp+218h] [rbp-398h]
  __int64 v399; // [rsp+220h] [rbp-390h]
  _QWORD *v400; // [rsp+230h] [rbp-380h] BYREF
  __int64 v401; // [rsp+238h] [rbp-378h]
  _BYTE v402[32]; // [rsp+240h] [rbp-370h] BYREF
  _BYTE v403[16]; // [rsp+260h] [rbp-350h] BYREF
  __int64 v404; // [rsp+270h] [rbp-340h]
  _QWORD v405[3]; // [rsp+288h] [rbp-328h] BYREF
  __int64 v406; // [rsp+2A0h] [rbp-310h] BYREF
  __int64 v407; // [rsp+2A8h] [rbp-308h]
  unsigned int v408; // [rsp+2B8h] [rbp-2F8h]
  _QWORD *v409; // [rsp+2C8h] [rbp-2E8h]
  unsigned int v410; // [rsp+2D8h] [rbp-2D8h]
  char v411; // [rsp+2E0h] [rbp-2D0h]
  char v412; // [rsp+2E9h] [rbp-2C7h]
  __int64 v413; // [rsp+2F0h] [rbp-2C0h] BYREF
  __int64 v414; // [rsp+2F8h] [rbp-2B8h]
  __int64 v415; // [rsp+300h] [rbp-2B0h] BYREF
  unsigned int v416; // [rsp+308h] [rbp-2A8h]
  __m128i v417; // [rsp+340h] [rbp-270h] BYREF
  __int64 v418; // [rsp+350h] [rbp-260h]
  __int64 v419; // [rsp+358h] [rbp-258h]
  __int64 v420; // [rsp+360h] [rbp-250h]
  __int64 *v421; // [rsp+370h] [rbp-240h]
  __int64 v422; // [rsp+378h] [rbp-238h]
  _BYTE v423[32]; // [rsp+380h] [rbp-230h] BYREF
  __int64 *v424; // [rsp+3A0h] [rbp-210h] BYREF
  __int64 v425; // [rsp+3A8h] [rbp-208h] BYREF
  char *v426; // [rsp+3B0h] [rbp-200h] BYREF
  __int64 v427; // [rsp+3B8h] [rbp-1F8h]
  __int64 v428; // [rsp+3C0h] [rbp-1F0h]
  _QWORD *v429; // [rsp+3C8h] [rbp-1E8h]
  unsigned int v430; // [rsp+3D8h] [rbp-1D8h]
  char v431; // [rsp+3E0h] [rbp-1D0h]
  char v432; // [rsp+3E9h] [rbp-1C7h]
  _BYTE v433[440]; // [rsp+3F8h] [rbp-1B8h] BYREF

  v367 = a1;
  v26 = a19;
  v341 = a25;
  v366 = __PAIR64__(a2, a3);
  v27 = a24;
  v336 = sub_13FC520((__int64)a1);
  if ( !v336 )
    return 0;
  v369 = (char *)sub_13FCB50((__int64)v367);
  if ( !v369 )
    return 0;
  v340 = sub_13FBCD0((__int64)v367);
  if ( !v340 )
    return 0;
  v370 = *(char **)v367[2].m128i_i64[0];
  v28 = sub_157EBA0((__int64)v369);
  if ( *(_BYTE *)(v28 + 16) == 26 )
  {
    sub_13FD840(&v371, (__int64)v367);
    if ( !v27 )
      goto LABEL_9;
  }
  else
  {
    v28 = 0;
    sub_13FD840(&v371, (__int64)v367);
    if ( !v27 )
      goto LABEL_32;
  }
  v29 = sub_15E0530(*v27);
  if ( sub_1602790(v29)
    || (v41 = sub_15E0530(*v27),
        v42 = sub_16033E0(v41),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v42 + 48LL))(v42)) )
  {
    sub_15C9090((__int64)&v417, &v371);
    sub_15CA330((__int64)&v424, (__int64)"loop-unroll", (__int64)"UnrollLoop", 10, &v417, (__int64)v370);
    sub_15CAB20((__int64)&v424, "  Applying unrolling strategy...", 0x20u);
    sub_143AA50(v27, (__int64)&v424);
    v424 = (__int64 *)&unk_49ECF68;
    sub_1897B80((__int64)v433);
  }
  if ( !v28 )
    goto LABEL_32;
LABEL_9:
  if ( (*(_DWORD *)(v28 + 20) & 0xFFFFFFF) == 1 )
    goto LABEL_32;
  v30 = v370;
  if ( *(char **)(v28 - 24) != v370 )
    goto LABEL_11;
  if ( sub_1377F70((__int64)&v367[3].m128i_i64[1], *(_QWORD *)(v28 - 48)) )
  {
    v30 = v370;
LABEL_11:
    if ( *(char **)(v28 - 48) != v30 || sub_1377F70((__int64)&v367[3].m128i_i64[1], *(_QWORD *)(v28 - 24)) )
      goto LABEL_32;
  }
  if ( !*((_WORD *)v370 + 9) )
  {
    v345 = v366;
    if ( (_DWORD)v366 )
    {
      if ( (unsigned int)v366 < HIDWORD(v366) )
      {
        HIDWORD(v366) = v366;
        HIDWORD(v345) = v345;
      }
      else
      {
        v340 = (_DWORD)v366 == HIDWORD(v366);
      }
      goto LABEL_17;
    }
    if ( HIDWORD(v366) > 1 )
    {
      v340 = 0;
LABEL_17:
      v400 = v402;
      v401 = 0x400000000LL;
      sub_13F9EC0((__int64)v367, (__int64)&v400);
      v374 = 0;
      v375 = 0;
      v31 = (const void *)v367[2].m128i_i64[0];
      v32 = v367[2].m128i_i64[1];
      v376 = 0;
      v33 = v32 - (_QWORD)v31;
      if ( v33 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_485;
      v34 = 0;
      if ( v33 )
      {
        nb = v33;
        v374 = (char *)sub_22077B0(v33);
        v34 = &v374[nb];
        v376 = &v374[nb];
        memcpy(v374, v31, nb);
      }
      v375 = (__int64 *)v34;
      v338 = v341 & v340;
      if ( (v341 & v340) != 0 )
      {
        v35 = &v400[(unsigned int)v401];
        v338 = v35 != sub_1AFB860(v400, (__int64)v35);
      }
      v337 = 0;
      v339 = a5 & (HIDWORD(v366) != 0 && (_DWORD)v366 == 0);
      if ( a18 )
      {
        v337 = sub_1B0BF10((_DWORD)v367, a18, (_DWORD)a20, a21, a22, a23, v341);
        if ( v337 )
        {
          v293 = sub_13F9E70((__int64)v367);
          v336 = sub_13FC520((__int64)v367);
          LODWORD(v366) = sub_1474190(a21, (__int64)v367, v293);
          a17 = sub_147DD60(a21, (__int64)v367, v293);
        }
      }
      v36 = sub_16D5D50();
      v37 = *(_QWORD **)&dword_4FA0208[2];
      if ( !*(_QWORD *)&dword_4FA0208[2] )
        goto LABEL_43;
      v38 = dword_4FA0208;
      do
      {
        if ( v36 > v37[4] )
        {
          v37 = (_QWORD *)v37[3];
        }
        else
        {
          v38 = v37;
          v37 = (_QWORD *)v37[2];
        }
      }
      while ( v37 );
      if ( v38 == dword_4FA0208 )
        goto LABEL_43;
      if ( v36 < *((_QWORD *)v38 + 4) )
        goto LABEL_43;
      v279 = *((_QWORD *)v38 + 7);
      v280 = v38 + 12;
      if ( !v279 )
        goto LABEL_43;
      v281 = v38 + 12;
      do
      {
        if ( *(_DWORD *)(v279 + 32) < dword_4FB67C8 )
        {
          v279 = *(_QWORD *)(v279 + 24);
        }
        else
        {
          v281 = (_DWORD *)v279;
          v279 = *(_QWORD *)(v279 + 16);
        }
      }
      while ( v279 );
      if ( v281 != v280 && dword_4FB67C8 >= v281[8] && v281[9] )
        v43 = (unsigned __int8)byte_4FB6860;
      else
LABEL_43:
        v43 = (unsigned __int8)sub_1AFBBC0((__int64)v367);
      v44 = HIDWORD(v366);
      if ( v339 && a17 % HIDWORD(v366) )
      {
        v339 = sub_1B12B90((_DWORD)v367, HIDWORD(v366), a6, v43, v26, (_DWORD)a20, a21, a22, a23, v341);
        if ( !v339 )
        {
          if ( !a4 )
          {
            v39 = 0;
            if ( v27 )
            {
              v288 = sub_15E0530(*v27);
              if ( sub_1602790(v288)
                || (v289 = sub_15E0530(*v27),
                    v290 = sub_16033E0(v289),
                    (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v290 + 48LL))(v290)) )
              {
                sub_15C9090((__int64)&v417, &v371);
                sub_15CA330((__int64)&v424, (__int64)"loop-unroll", (__int64)"UnrollLoop", 10, &v417, (__int64)v370);
                sub_15CAB20((__int64)&v424, "    Failed : remainder loops could not be ", 0x2Au);
                sub_15CAB20((__int64)&v424, "generated when assuming runtime trip count", 0x2Au);
                sub_143AA50(v27, (__int64)&v424);
                v424 = (__int64 *)&unk_49ECF68;
                sub_1897B80((__int64)v433);
              }
              v39 = 0;
            }
            goto LABEL_440;
          }
          if ( v27 )
          {
            v285 = sub_15E0530(*v27);
            if ( sub_1602790(v285)
              || (v286 = sub_15E0530(*v27),
                  v287 = sub_16033E0(v286),
                  (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v287 + 48LL))(v287)) )
            {
              sub_15C9090((__int64)&v417, &v371);
              sub_15CA330((__int64)&v424, (__int64)"loop-unroll", (__int64)"UnrollLoop", 10, &v417, (__int64)v370);
              sub_15CAB20((__int64)&v424, "    Note : cannot generate the remainder loop. ", 0x2Fu);
              sub_15CAB20((__int64)&v424, "Will unroll the main loop with side-exits that may hurt performance", 0x43u);
              sub_143AA50(v27, (__int64)&v424);
              v424 = (__int64 *)&unk_49ECF68;
              sub_1897B80((__int64)v433);
            }
          }
        }
        v44 = HIDWORD(v366);
      }
      if ( (_DWORD)v366 )
      {
        a17 = 0;
        v368 = (unsigned int)v366 % v44;
      }
      else
      {
        v282 = a17;
        v283 = v44;
        if ( a17 )
        {
          do
          {
            v284 = v282;
            v282 = v283 % v282;
            v283 = v284;
          }
          while ( v282 );
          v44 = v284;
        }
        a17 = v44;
        v368 = v44;
      }
      if ( (_DWORD)v345 == HIDWORD(v345) )
      {
        if ( v27 )
          sub_1AFEDE0(v27, (__int64 *)&v367, (unsigned int *)&v366);
      }
      else if ( a18 )
      {
        if ( v27 )
          sub_1AFDD90(v27, (__int64 *)&v367, (unsigned int *)&a18);
      }
      else
      {
        v424 = (__int64 *)&v367;
        v425 = (__int64)&v366 + 4;
        if ( a17 && a17 == v368 )
        {
          if ( a17 == 1 )
          {
            if ( v27 && v339 )
              sub_1AFE3E0(v27, (__int64)&v424, (_DWORD *)&v366 + 1);
          }
          else if ( v27 )
          {
            sub_1AFEA80(v27, (__int64)&v424, &a17);
          }
        }
        else if ( v27 )
        {
          sub_1AFE730(v27, (__int64)&v424, &v368);
        }
      }
      if ( a21 )
        sub_1465DB0(a21, v367);
      v333 = sub_1377F70((__int64)&v367[3].m128i_i64[1], *(_QWORD *)(v28 - 24));
      v45 = *(_QWORD *)(v28 - 24LL * v333 - 24);
      v406 = 0;
      v408 = 128;
      v334 = v45;
      v407 = sub_22077B0(0x2000);
      sub_1954940((__int64)&v406);
      v411 = 0;
      v412 = 1;
      v377 = 0;
      v46 = *((_QWORD *)v370 + 6);
      v378 = 0;
      v379 = 0;
      while ( 1 )
      {
        if ( !v46 )
          goto LABEL_573;
        if ( *(_BYTE *)(v46 - 8) != 77 )
          break;
        v424 = (__int64 *)(v46 - 24);
        v47 = v378;
        if ( v378 == v379 )
        {
          sub_1AFF1A0((__int64)&v377, v378, &v424);
        }
        else
        {
          if ( v378 )
          {
            *v378 = v46 - 24;
            v47 = v378;
          }
          v378 = v47 + 1;
        }
        v46 = *(_QWORD *)(v46 + 8);
      }
      memset(v380, 0, 24);
      v381 = 0;
      v382 = 0;
      v383 = 0;
      sub_15CE600((__int64)v380, &v370);
      sub_15CE600((__int64)&v381, &v369);
      sub_1AFCDB0((__int64)v403, (__int64)v367);
      *((_QWORD *)&v48 + 1) = a20;
      *(_QWORD *)&v48 = v403;
      sub_13FF3D0(v48);
      v384 = 0;
      v385 = 0;
      v335 = v405[1];
      v386 = 0;
      v342 = v405[0];
      v49 = (const void *)v367[2].m128i_i64[0];
      v50 = v367[2].m128i_i64[1] - (_QWORD)v49;
      if ( v50 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_485:
        sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
      v51 = 0;
      if ( v50 )
      {
        v52 = (char *)sub_22077B0(v367[2].m128i_i64[1] - (_QWORD)v49);
        v51 = &v52[v50];
        v384 = v52;
        v386 = &v52[v50];
        memcpy(v52, v49, v50);
      }
      v385 = v51;
      v417.m128i_i8[8] |= 1u;
      v417.m128i_i64[0] = 0;
      sub_1AFF3E0((__int64)&v417);
      v421 = (__int64 *)v423;
      v422 = 0x400000000LL;
      v57 = (__int64 **)v367[1].m128i_i64[0];
      for ( i = (__int64 **)v367->m128i_i64[1]; v57 != i; ++i )
      {
        v59 = *i;
        v424 = v59;
        sub_1B01020((__int64)&v417, &v424, v53, v54, v55, v56);
      }
      if ( (unsigned __int8)sub_1626D30(*((_QWORD *)v370 + 7)) )
      {
        v360 = v367[2].m128i_i64[1];
        if ( v367[2].m128i_i64[0] != v360 )
        {
          v62 = v367[2].m128i_i64[0];
          while ( 1 )
          {
            v63 = *(_QWORD *)(*(_QWORD *)v62 + 48LL);
            v64 = *(_QWORD *)v62 + 40LL;
            if ( v63 != v64 )
              break;
LABEL_86:
            v62 += 8;
            if ( v360 == v62 )
              goto LABEL_87;
          }
          while ( v63 )
          {
            if ( *(_BYTE *)(v63 - 8) == 78
              && (v69 = *(_QWORD *)(v63 - 48), !*(_BYTE *)(v69 + 16))
              && (*(_BYTE *)(v69 + 33) & 0x20) != 0
              && (unsigned int)(*(_DWORD *)(v69 + 36) - 35) <= 3 )
            {
              v63 = *(_QWORD *)(v63 + 8);
              if ( v63 == v64 )
                goto LABEL_86;
            }
            else
            {
              v65 = sub_15C70A0(v63 + 24);
              if ( v65 )
              {
                v66 = sub_1AFCF60(v65, HIDWORD(v366));
                sub_15C7080(&v424, v66);
                if ( (__int64 **)(v63 + 24) == &v424 )
                {
                  if ( v424 )
                    sub_161E7C0((__int64)&v424, (__int64)v424);
                }
                else
                {
                  v67 = *(_QWORD *)(v63 + 24);
                  if ( v67 )
                    sub_161E7C0(v63 + 24, v67);
                  v68 = (unsigned __int8 *)v424;
                  *(_QWORD *)(v63 + 24) = v424;
                  if ( v68 )
                    sub_1623210((__int64)&v424, v68, v63 + 24);
                }
              }
              v63 = *(_QWORD *)(v63 + 8);
              if ( v63 == v64 )
                goto LABEL_86;
            }
          }
LABEL_573:
          BUG();
        }
      }
LABEL_87:
      if ( HIDWORD(v366) == 1 )
        goto LABEL_195;
      v346 = 1;
      while ( 1 )
      {
        v387 = 0;
        v70 = (__m128i *)&v415;
        v388 = 0;
        v389 = 0;
        v413 = 0;
        v414 = 1;
        do
        {
          v70->m128i_i64[0] = -8;
          ++v70;
        }
        while ( v70 != &v417 );
        v71 = sub_1B00500((__int64)&v413, (__int64 *)&v367);
        v71[1] = v367;
        if ( v335 != v342 )
          break;
LABEL_178:
        v131 = v387;
        if ( v388 != v387 )
        {
          v362 = v388;
          do
          {
            v132 = *(_QWORD *)(*(_QWORD *)v131 + 48LL);
            for ( j = *(_QWORD *)v131 + 40LL; j != v132; v132 = *(_QWORD *)(v132 + 8) )
            {
              while ( 1 )
              {
                if ( !v132 )
                {
                  sub_1AFD1D0(0, (__int64)&v406);
                  BUG();
                }
                sub_1AFD1D0(v132 - 24, (__int64)&v406);
                if ( *(_BYTE *)(v132 - 8) == 78 )
                {
                  v134 = *(_QWORD *)(v132 - 48);
                  if ( !*(_BYTE *)(v134 + 16) && (*(_BYTE *)(v134 + 33) & 0x20) != 0 && *(_DWORD *)(v134 + 36) == 4 )
                    break;
                }
                v132 = *(_QWORD *)(v132 + 8);
                if ( j == v132 )
                  goto LABEL_189;
              }
              sub_14CE830(a23, v132 - 24);
            }
LABEL_189:
            v131 += 8;
          }
          while ( v362 != v131 );
        }
        if ( (v414 & 1) == 0 )
          j___libc_free_0(v415);
        if ( v387 )
          j_j___libc_free_0(v387, v389 - v387);
        if ( HIDWORD(v366) == ++v346 )
        {
LABEL_195:
          v135 = v377;
          for ( k = v378; k != v135; ++v135 )
          {
            v136 = *v135;
            if ( (_DWORD)v345 == HIDWORD(v345) )
            {
              v176 = 0x17FFFFFFE8LL;
              v177 = *(_BYTE *)(v136 + 23) & 0x40;
              v178 = *(_DWORD *)(v136 + 20) & 0xFFFFFFF;
              if ( v178 )
              {
                v179 = 24LL * *(unsigned int *)(v136 + 56) + 8;
                v180 = 0;
                do
                {
                  v181 = v136 - 24LL * v178;
                  if ( v177 )
                    v181 = *(_QWORD *)(v136 - 8);
                  if ( v336 == *(_QWORD *)(v181 + v179) )
                  {
                    v176 = 24 * v180;
                    goto LABEL_286;
                  }
                  ++v180;
                  v179 += 8;
                }
                while ( v178 != (_DWORD)v180 );
                v176 = 0x17FFFFFFE8LL;
              }
LABEL_286:
              if ( v177 )
                v182 = *(_QWORD *)(v136 - 8);
              else
                v182 = v136 - 24LL * v178;
              sub_164D160(*v135, *(_QWORD *)(v182 + v176), a7, a8, a9, a10, v60, v61, a13, a14);
              sub_157EA20((__int64)(v370 + 40), v136);
              v183 = *(unsigned __int64 **)(v136 + 32);
              v184 = *(_QWORD *)(v136 + 24) & 0xFFFFFFFFFFFFFFF8LL;
              *v183 = v184 | *v183 & 7;
              *(_QWORD *)(v184 + 8) = v183;
              *(_QWORD *)(v136 + 24) &= 7uLL;
              *(_QWORD *)(v136 + 32) = 0;
              sub_164BEC0(v136, v136, v184, (__int64)v183, a7, a8, a9, a10, v185, v186, a13, a14);
            }
            else if ( HIDWORD(v366) > 1 )
            {
              v137 = *(_DWORD *)(v136 + 20) & 0xFFFFFFF;
              if ( v137 )
              {
                v138 = 0;
                v139 = 24LL * *(unsigned int *)(v136 + 56) + 8;
                while ( 1 )
                {
                  v140 = v136 - 24LL * v137;
                  if ( (*(_BYTE *)(v136 + 23) & 0x40) != 0 )
                    v140 = *(_QWORD *)(v136 - 8);
                  if ( v369 == *(char **)(v140 + v139) )
                    break;
                  ++v138;
                  v139 += 8;
                  if ( v137 == v138 )
                    goto LABEL_290;
                }
              }
              else
              {
LABEL_290:
                v138 = -1;
              }
              v141 = sub_15F5350(*v135, v138, 0);
              v145 = v141;
              if ( *(_BYTE *)(v141 + 16) > 0x17u && sub_1377F70((__int64)&v367[3].m128i_i64[1], *(_QWORD *)(v141 + 40)) )
                v145 = sub_1B01300((__int64)&v406, v145)[2];
              sub_1704F80(v136, v145, *(v382 - 1), v142, v143, v144);
            }
          }
          v187 = v381;
          v188 = v382 - v381;
          v364 = v188;
          if ( (_DWORD)v188 )
          {
            v347 = (unsigned int)v188;
            v189 = 0;
            while ( 1 )
            {
              v191 = v187[v189];
              v192 = (_QWORD *)sub_157EBA0(v191);
              v193 = ((int)v189 + 1) % v364;
              v194 = *(_QWORD *)(v380[0] + 8LL * v193);
              if ( v193 && v339 )
              {
                if ( (_DWORD)v345 != HIDWORD(v345) )
                  goto LABEL_295;
              }
              else
              {
                if ( (_DWORD)v345 != HIDWORD(v345) )
                {
                  if ( v193 != v368 && (!a17 || v193 % a17) )
                    goto LABEL_295;
                  goto LABEL_306;
                }
                if ( !v193 )
                {
                  v194 = v334;
LABEL_296:
                  v190 = sub_1648A60(56, 1u);
                  if ( v190 )
                    sub_15F8320((__int64)v190, v194, (__int64)v192);
                  sub_15F20C0(v192);
                  goto LABEL_299;
                }
              }
              if ( !a15 || a16 == 1 && (_DWORD)v189 )
              {
LABEL_295:
                if ( v194 != v334 )
                {
                  if ( v192 )
                  {
                    srca = sub_15F4D60((__int64)v192);
                    na = sub_157EBA0(v191);
                    if ( srca )
                    {
                      v344 = v194;
                      v199 = v191;
                      v343 = v192;
                      for ( m = 0; m != srca; ++m )
                      {
                        v201 = sub_15F4DF0(na, m);
                        if ( *(_QWORD *)(v380[0] + 8 * v189) != v201 )
                        {
                          v202 = sub_157F280(v201);
                          v204 = v203;
                          v205 = v202;
                          while ( v204 != v205 )
                          {
                            v206 = *(_DWORD *)(v205 + 20) & 0xFFFFFFF;
                            if ( v206 )
                            {
                              v207 = 0;
                              v208 = 24LL * *(unsigned int *)(v205 + 56) + 8;
                              while ( 1 )
                              {
                                v209 = v205 - 24LL * v206;
                                if ( (*(_BYTE *)(v205 + 23) & 0x40) != 0 )
                                  v209 = *(_QWORD *)(v205 - 8);
                                if ( v199 == *(_QWORD *)(v209 + v208) )
                                  break;
                                ++v207;
                                v208 += 8;
                                if ( v206 == v207 )
                                  goto LABEL_329;
                              }
                            }
                            else
                            {
LABEL_329:
                              v207 = -1;
                            }
                            sub_15F5350(v205, v207, 0);
                            v210 = *(_QWORD *)(v205 + 32);
                            if ( !v210 )
                              goto LABEL_573;
                            v205 = 0;
                            if ( *(_BYTE *)(v210 - 8) == 77 )
                              v205 = v210 - 24;
                          }
                        }
                      }
                      v194 = v344;
                      v192 = v343;
                    }
                  }
                }
                goto LABEL_296;
              }
LABEL_306:
              v195 = (_QWORD *)((char *)v192 - 24 - 24LL * !v333);
              if ( *v195 )
              {
                v196 = v195[1];
                v197 = v195[2] & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v197 = v196;
                if ( v196 )
                  *(_QWORD *)(v196 + 16) = v197 | *(_QWORD *)(v196 + 16) & 3LL;
              }
              *v195 = v194;
              if ( v194 )
              {
                v198 = *(_QWORD *)(v194 + 8);
                v195[1] = v198;
                if ( v198 )
                  *(_QWORD *)(v198 + 16) = (unsigned __int64)(v195 + 1) | *(_QWORD *)(v198 + 16) & 3LL;
                v195[2] = (v194 + 8) | v195[2] & 3;
                *(_QWORD *)(v194 + 8) = v195;
              }
LABEL_299:
              if ( v347 == ++v189 )
                break;
              v187 = v381;
            }
          }
          v211 = a22;
          if ( a22 )
          {
            if ( HIDWORD(v366) > 1 )
            {
              v348 = v375;
              if ( v375 != (__int64 *)v374 )
              {
                v365 = (__int64 *)v374;
                while ( 1 )
                {
                  v212 = *v365;
                  v213 = sub_15CC510(v211, *v365);
                  v424 = (__int64 *)&v426;
                  v425 = 0x1000000000LL;
                  v214 = *(__int64 ***)(v213 + 24);
                  v215 = *(__int64 ***)(v213 + 32);
                  if ( v215 == v214 )
                  {
                    v291 = (__int64 *)&v426;
                    v292 = (__int64 *)&v426;
                  }
                  else
                  {
                    do
                    {
                      while ( 1 )
                      {
                        v216 = **v214;
                        if ( !sub_1377F70((__int64)&v367[3].m128i_i64[1], v216) )
                          break;
                        if ( v215 == ++v214 )
                          goto LABEL_348;
                      }
                      v219 = (unsigned int)v425;
                      if ( (unsigned int)v425 >= HIDWORD(v425) )
                      {
                        sub_16CD150((__int64)&v424, &v426, 0, 8, v217, v218);
                        v219 = (unsigned int)v425;
                      }
                      ++v214;
                      v424[v219] = v216;
                      LODWORD(v425) = v425 + 1;
                    }
                    while ( v215 != v214 );
LABEL_348:
                    v291 = v424;
                    v292 = &v424[(unsigned int)v425];
                  }
                  if ( v369 == (char *)v212 )
                  {
                    v234 = v382;
                    v235 = v381;
                    v220 = (char *)*(v382 - 1);
                    if ( v381 != v382 )
                    {
                      srcc = (char *)*(v382 - 1);
                      v236 = v292;
                      do
                      {
                        v237 = *v235;
                        v238 = sub_157EBA0(*v235);
                        if ( *(_BYTE *)(v238 + 16) == 26 && (*(_DWORD *)(v238 + 20) & 0xFFFFFFF) == 3 )
                        {
                          v239 = (char *)v237;
                          v292 = v236;
                          v220 = v239;
                          goto LABEL_355;
                        }
                        ++v235;
                      }
                      while ( v234 != v235 );
                      v292 = v236;
                      v220 = srcc;
                    }
                  }
                  else
                  {
                    v220 = *(char **)(*(_QWORD *)(v212 + 56) + 80LL);
                    if ( v220 )
                      v220 -= 24;
                    if ( (char *)v212 != v220 && v369 != v220 )
                      v220 = (char *)sub_1AFCC80(a22, v212, (__int64)v369);
                  }
LABEL_355:
                  if ( v291 != v292 )
                  {
                    srcb = v292;
                    v221 = v291;
                    do
                    {
                      v223 = a22;
                      v224 = *v221;
                      v225 = sub_15CC510(a22, (__int64)v220);
                      v226 = sub_15CC510(v223, v224);
                      *(_BYTE *)(v223 + 72) = 0;
                      v227 = v226;
                      v228 = *(_QWORD *)(v226 + 8);
                      if ( v225 != v228 )
                      {
                        v413 = v227;
                        v229 = sub_1AFB990(*(_QWORD **)(v228 + 24), *(_QWORD *)(v228 + 32), &v413);
                        sub_15CDF70(*(_QWORD *)(v227 + 8) + 24LL, v229);
                        *(_QWORD *)(v227 + 8) = v225;
                        v413 = v227;
                        v222 = *(_BYTE **)(v225 + 32);
                        if ( v222 == *(_BYTE **)(v225 + 40) )
                        {
                          sub_15CE310(v225 + 24, v222, &v413);
                        }
                        else
                        {
                          if ( v222 )
                          {
                            *(_QWORD *)v222 = v227;
                            v222 = *(_BYTE **)(v225 + 32);
                          }
                          v222 += 8;
                          *(_QWORD *)(v225 + 32) = v222;
                        }
                        if ( *(_DWORD *)(v227 + 16) != *(_DWORD *)(*(_QWORD *)(v227 + 8) + 16LL) + 1 )
                          sub_1AFB750(v227, (__int64)v222, v230, v231, v232, v233);
                      }
                      ++v221;
                    }
                    while ( srcb != v221 );
                    v292 = v424;
                  }
                  if ( v292 != (__int64 *)&v426 )
                    _libc_free((unsigned __int64)v292);
                  if ( v348 == ++v365 )
                    break;
                  v211 = a22;
                }
              }
            }
          }
          v240 = v382;
          v241 = v381;
          v395 = 0;
          v396 = 0;
          v397 = 0;
          LODWORD(v398) = 0;
          v413 = 0;
          v414 = 0;
          v415 = 0;
          v416 = 0;
          if ( v382 == v381 )
          {
            v390 = 0;
            v324 = 0;
            v392 = 0;
            v391 = 0;
            goto LABEL_402;
          }
          do
          {
            v242 = sub_157EBA0(*v241);
            if ( (*(_DWORD *)(v242 + 20) & 0xFFFFFFF) == 1 )
            {
              v424 = *(__int64 **)(v242 - 24);
              v245 = sub_1AFF430(
                       (__int64)v424,
                       (__int64)a20,
                       a7,
                       a8,
                       a9,
                       a10,
                       v243,
                       v244,
                       a13,
                       a14,
                       a21,
                       a22,
                       (__int64)&v395);
              if ( v245 )
              {
                sub_1AFFEE0((__int64)&v413, (__int64 *)&v424)[1] = v245;
                v246 = v381;
                for ( ii = v382; v246 != ii; ++v246 )
                {
                  if ( (__int64 *)*v246 == v424 )
                    *v246 = v245;
                }
                v248 = v385;
                v249 = sub_1AFBA50(v384, (__int64)v385, (__int64 *)&v424);
                v250 = v249;
                if ( v248 != v249 )
                {
                  v251 = v249 + 1;
                  if ( v248 == v251 )
                    goto LABEL_394;
                  do
                  {
                    if ( (__int64 *)*v251 != v424 )
                      *v250++ = *v251;
                    ++v251;
                  }
                  while ( v251 != v248 );
                  if ( v248 != v250 )
                  {
LABEL_394:
                    v252 = (char *)v385;
                    if ( v248 != v385 )
                    {
                      v250 = memmove(v250, v248, (_BYTE *)v385 - (_BYTE *)v248);
                      v252 = (char *)v385;
                    }
                    v253 = (char *)v250 + v252 - (char *)v248;
                    if ( v252 != v253 )
                      v385 = v253;
                  }
                }
              }
            }
            ++v241;
          }
          while ( v240 != v241 );
          if ( !(_DWORD)v397 )
            goto LABEL_399;
          v302 = (__int64 *)v396;
          v303 = (__int64 *)(v396 + 16LL * (unsigned int)v398);
          if ( (__int64 *)v396 == v303 )
            goto LABEL_399;
          while ( 1 )
          {
            v304 = *v302;
            v305 = v302;
            if ( *v302 != -8 && v304 != -16 )
              break;
            v302 += 2;
            if ( v303 == v302 )
              goto LABEL_399;
          }
          if ( v303 == v302 )
          {
LABEL_399:
            v324 = 0;
            v390 = 0;
            v391 = 0;
            v392 = 0;
            v254 = v415;
            if ( (_DWORD)v415 )
            {
              v255 = 8LL * (unsigned int)v415;
              v256 = (char *)sub_22077B0(v255);
              v324 = &v256[v255];
              v390 = v256;
              v392 = &v256[v255];
              memset(v256, 0, v255);
              v254 = v415;
            }
            v391 = (__int64)v324;
            if ( v254 )
            {
              v297 = (__int64 *)v414;
              v298 = (__int64 *)(v414 + 16LL * v416);
              if ( (__int64 *)v414 != v298 )
              {
                while ( 1 )
                {
                  v299 = *v297;
                  v300 = v297;
                  if ( *v297 != -16 && v299 != -8 )
                    break;
                  v297 += 2;
                  if ( v298 == v297 )
                    goto LABEL_402;
                }
                if ( v298 != v297 )
                {
                  v301 = 0;
                  do
                  {
                    v300 += 2;
                    *(_QWORD *)((char *)v390 + v301) = v299;
                    if ( v300 == v298 )
                      break;
                    while ( 1 )
                    {
                      v299 = *v300;
                      if ( *v300 != -16 && v299 != -8 )
                        break;
                      v300 += 2;
                      if ( v298 == v300 )
                        goto LABEL_508;
                    }
                    v301 += 8;
                  }
                  while ( v300 != v298 );
LABEL_508:
                  v324 = (char *)v391;
                }
              }
            }
LABEL_402:
            sub_1AFCA80((__int64 *)v390, v324, &a22);
            v259 = (__int64 *)v391;
            v260 = (__int64 *)v390;
            if ( (void *)v391 != v390 )
            {
              do
              {
                v261 = *v260++;
                sub_15CDFB0(a22, v261);
                sub_13FBA80((__int64)a20, v261);
                sub_157F980(v261);
              }
              while ( v259 != v260 );
            }
            v262 = v367;
            if ( (_DWORD)v345 == HIDWORD(v345) )
            {
              sub_1AFD4E0((__int64)v367, 0, (__int64)a20, a21, a22, a23, a7, a8, a9, a10, v257, v258, a13, a14);
              v262 = a20;
              v266 = (_QWORD *)v367->m128i_i64[0];
              sub_1401B00((__int64)a20, v367);
            }
            else
            {
              if ( HIDWORD(v366) <= 1 && !v337 )
              {
                sub_1AFD4E0((__int64)v367, 0, (__int64)a20, a21, a22, a23, a7, a8, a9, a10, v257, v258, a13, a14);
                v266 = (_QWORD *)v367->m128i_i64[0];
                v267 = v367->m128i_i64[0] != 0;
                if ( v341 && v367->m128i_i64[0] )
                {
LABEL_563:
                  v274 = a22;
                  if ( a22 )
                    goto LABEL_418;
                  goto LABEL_420;
                }
LABEL_565:
                v329 = a22;
                v274 = a22;
                if ( a22 )
                {
                  if ( v267 )
                    goto LABEL_418;
                  v330 = v421;
                  v331 = &v421[(unsigned int)v422];
                  if ( v331 != v421 )
                  {
                    while ( 1 )
                    {
                      v332 = *v330++;
                      sub_1AFB400(v332, v329, (__int64)a20, a21, a23, v341, a7, a8, a9, a10, v264, v265, a13, a14);
                      if ( v331 == v330 )
                        break;
                      v329 = a22;
                    }
                  }
                }
LABEL_420:
                v39 = 1;
                if ( (_DWORD)v345 != HIDWORD(v345) )
                {
LABEL_421:
                  sub_15CE080(&v390);
                  j___libc_free_0(v414);
                  j___libc_free_0(v396);
                  if ( v421 != (__int64 *)v423 )
                    _libc_free((unsigned __int64)v421);
                  if ( (v417.m128i_i8[8] & 1) == 0 )
                    j___libc_free_0(v418);
                  sub_15CE080(&v384);
                  sub_15CE080(v405);
                  j___libc_free_0(v404);
                  sub_15CE080(&v381);
                  sub_15CE080(v380);
                  if ( v377 )
                    j_j___libc_free_0(v377, (char *)v379 - (char *)v377);
                  if ( v411 )
                  {
                    if ( v410 )
                    {
                      v294 = v409;
                      v295 = &v409[2 * v410];
                      do
                      {
                        if ( *v294 != -8 && *v294 != -4 )
                        {
                          v296 = v294[1];
                          if ( v296 )
                            sub_161E7C0((__int64)(v294 + 1), v296);
                        }
                        v294 += 2;
                      }
                      while ( v295 != v294 );
                    }
                    j___libc_free_0(v409);
                  }
                  if ( v408 )
                  {
                    v275 = (_QWORD *)v407;
                    v417.m128i_i64[1] = 2;
                    v418 = 0;
                    v276 = (_QWORD *)(v407 + ((unsigned __int64)v408 << 6));
                    v419 = -8;
                    v417.m128i_i64[0] = (__int64)&unk_49E6B50;
                    v424 = (__int64 *)&unk_49E6B50;
                    v277 = -8;
                    v420 = 0;
                    v425 = 2;
                    v426 = 0;
                    v427 = -16;
                    v428 = 0;
                    while ( 1 )
                    {
                      v278 = v275[3];
                      if ( v277 != v278 && v278 != v427 )
                      {
                        sub_1455FA0((__int64)(v275 + 5));
                        v278 = v275[3];
                      }
                      *v275 = &unk_49EE2B0;
                      if ( v278 != -8 && v278 != 0 && v278 != -16 )
                        sub_1649B30(v275 + 1);
                      v275 += 8;
                      if ( v276 == v275 )
                        break;
                      v277 = v419;
                    }
                    v424 = (__int64 *)&unk_49EE2B0;
                    sub_1455FA0((__int64)&v425);
                    v417.m128i_i64[0] = (__int64)&unk_49EE2B0;
                    sub_1455FA0((__int64)&v417.m128i_i64[1]);
                  }
                  j___libc_free_0(v407);
LABEL_440:
                  sub_15CE080(&v374);
                  if ( v400 != (_QWORD *)v402 )
                    _libc_free((unsigned __int64)v400);
                  goto LABEL_33;
                }
LABEL_555:
                v39 = 2;
                goto LABEL_421;
              }
              sub_1AFD4E0((__int64)v367, 1, (__int64)a20, a21, a22, a23, a7, a8, a9, a10, v257, v258, a13, a14);
              v266 = (_QWORD *)v367->m128i_i64[0];
            }
            v267 = v266 != 0;
            if ( v266 && v341 )
            {
              if ( v338 != 1 && v340 )
              {
                v268 = v385;
                v269 = v384;
                v424 = 0;
                v425 = 0;
                v426 = 0;
                v270 = (_BYTE *)v385 - (_BYTE *)v384;
                if ( v385 == v384 )
                {
                  v273 = 0;
                  v272 = 0;
                }
                else
                {
                  if ( v270 > 0x7FFFFFFFFFFFFFF8LL )
                    sub_4261EA(v262, v384, v263);
                  v271 = sub_22077B0((_BYTE *)v385 - (_BYTE *)v384);
                  v269 = v384;
                  v272 = (__int64 *)v271;
                  v268 = v385;
                  v273 = (_BYTE *)v385 - (_BYTE *)v384;
                }
                v424 = v272;
                v425 = (__int64)v272;
                v426 = (char *)v272 + v270;
                if ( v269 != v268 )
                  v272 = (__int64 *)memmove(v272, v269, v273);
                v425 = (__int64)v272 + v273;
                v338 = sub_1AFBCE0(v266, &v424, (__int64)a20);
                sub_15CE080(&v424);
                v274 = a22;
                if ( !a22 )
                  goto LABEL_555;
LABEL_418:
                if ( !v338 )
                {
LABEL_419:
                  sub_1AFB400((__int64)v266, v274, (__int64)a20, a21, a23, v341, a7, a8, a9, a10, v264, v265, a13, a14);
                  goto LABEL_420;
                }
                v325 = sub_13AE450((__int64)a20, *(v382 - 1));
                if ( v266 != (_QWORD *)v325 )
                {
                  if ( !v325 )
                  {
LABEL_556:
                    v328 = v266;
                    do
                    {
                      v327 = (__int64)v328;
                      v328 = (_QWORD *)*v328;
                    }
                    while ( (_QWORD *)v325 != v328 );
                    goto LABEL_554;
                  }
                  v326 = (_QWORD *)v325;
                  while ( 1 )
                  {
                    v326 = (_QWORD *)*v326;
                    if ( v266 == v326 )
                      break;
                    if ( !v326 )
                      goto LABEL_556;
                  }
                }
                v327 = (__int64)v266;
LABEL_554:
                sub_1AE5120(v327, v274, (__int64)a20, a21);
                v274 = a22;
                goto LABEL_419;
              }
              goto LABEL_563;
            }
            goto LABEL_565;
          }
          v306 = v416;
          v307 = (void *)v302[1];
          if ( !v416 )
            goto LABEL_526;
LABEL_518:
          v308 = v306 - 1;
          v309 = v414;
          v310 = 1;
          v311 = v308 & (((unsigned int)v304 >> 9) ^ ((unsigned int)v304 >> 4));
          v312 = *(_QWORD *)(v414 + 16LL * v311);
          if ( v304 == v312 )
            goto LABEL_519;
          while ( v312 != -8 )
          {
            v311 = v308 & (v310 + v311);
            v312 = *(_QWORD *)(v414 + 16LL * v311);
            if ( v304 == v312 )
              goto LABEL_519;
            ++v310;
          }
          v390 = v307;
          while ( 1 )
          {
            v316 = 1;
            v317 = v308 & (((unsigned int)v307 >> 9) ^ ((unsigned int)v307 >> 4));
            v318 = *(void **)(v309 + 16LL * v317);
            if ( v318 != v307 )
            {
              while ( v318 != (void *)-8LL )
              {
                v317 = v308 & (v316 + v317);
                v318 = *(void **)(v309 + 16LL * v317);
                if ( v307 == v318 )
                  goto LABEL_533;
                ++v316;
              }
LABEL_527:
              while ( 2 )
              {
                v313 = a22;
                v314 = sub_15CC510(a22, (__int64)v307);
                v315 = sub_15CC510(v313, v304);
                *(_BYTE *)(v313 + 72) = 0;
                sub_15CE4D0(v315, v314);
LABEL_519:
                v305 += 2;
                if ( v305 == v303 )
                  goto LABEL_399;
                while ( *v305 == -16 || *v305 == -8 )
                {
                  v305 += 2;
                  if ( v303 == v305 )
                    goto LABEL_399;
                }
                if ( v305 == v303 )
                  goto LABEL_399;
                v306 = v416;
                v307 = (void *)v305[1];
                v304 = *v305;
                if ( !v416 )
                {
LABEL_526:
                  v390 = v307;
                  continue;
                }
                goto LABEL_518;
              }
            }
LABEL_533:
            v319 = sub_1AFF330((__int64)&v413, (__int64 *)&v390, &v424);
            v320 = v424;
            if ( v319 )
            {
              v307 = (void *)v424[1];
              goto LABEL_539;
            }
            v321 = v416;
            ++v413;
            v322 = v415 + 1;
            if ( 4 * ((int)v415 + 1) >= 3 * v416 )
              break;
            if ( v416 - HIDWORD(v415) - v322 <= v416 >> 3 )
              goto LABEL_544;
LABEL_536:
            LODWORD(v415) = v322;
            if ( *v320 != -8 )
              --HIDWORD(v415);
            v323 = v390;
            v320[1] = 0;
            v307 = 0;
            *v320 = (__int64)v323;
LABEL_539:
            v390 = v307;
            if ( !v416 )
              goto LABEL_527;
            v309 = v414;
            v308 = v416 - 1;
          }
          v321 = 2 * v416;
LABEL_544:
          sub_1447B20((__int64)&v413, v321);
          sub_1AFF330((__int64)&v413, (__int64 *)&v390, &v424);
          v320 = v424;
          v322 = v415 + 1;
          goto LABEL_536;
        }
      }
      v361 = v335;
      while ( 1 )
      {
        v424 = 0;
        LODWORD(v427) = 128;
        v425 = sub_22077B0(0x2000);
        sub_1954940((__int64)&v424);
        v431 = 0;
        LODWORD(v390) = v346;
        v395 = ".";
        LOWORD(v397) = 2307;
        v432 = 1;
        v396 = (__int64)v390;
        v72 = sub_1AB5760(*(_QWORD *)(v361 - 8), (__int64)&v424, (__int64 *)&v395, 0, 0, 0);
        v372 = (char *)v72;
        v73 = (__int64 *)(*((_QWORD *)v370 + 7) + 72LL);
        sub_15E01D0((__int64)v73, v72);
        v74 = *v73;
        v75 = *(_QWORD *)(v72 + 24);
        *(_QWORD *)(v72 + 32) = v73;
        v76 = v372;
        v74 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v72 + 24) = v74 | v75 & 7;
        *(_QWORD *)(v74 + 8) = v72 + 24;
        *v73 = *v73 & 7 | (v72 + 24);
        v373 = sub_1B007D0(*(_QWORD *)(v361 - 8), (__int64)v76, (__int64)a20, (__int64)&v413);
        if ( v373 )
          break;
LABEL_102:
        v85 = *(char **)(v361 - 8);
        if ( v85 == v370 )
        {
          v159 = v377;
          if ( v378 != v377 )
          {
            v358 = v378;
            while ( 1 )
            {
              v160 = *v159;
              v161 = sub_1B01300((__int64)&v424, *v159)[2];
              v162 = *(_BYTE *)(v161 + 23) & 0x40;
              v163 = *(_DWORD *)(v161 + 20) & 0xFFFFFFF;
              if ( v163 )
              {
                v164 = 24LL * *(unsigned int *)(v161 + 56) + 8;
                v165 = 0;
                while ( 1 )
                {
                  v166 = v161 - 24LL * v163;
                  if ( v162 )
                    v166 = *(_QWORD *)(v161 - 8);
                  if ( v369 == *(char **)(v166 + v164) )
                    break;
                  ++v165;
                  v164 += 8;
                  if ( v163 == (_DWORD)v165 )
                    goto LABEL_270;
                }
                v167 = 24 * v165;
                if ( v162 )
                {
LABEL_258:
                  v168 = *(_QWORD *)(v161 - 8);
                  goto LABEL_259;
                }
              }
              else
              {
LABEL_270:
                v167 = 0x17FFFFFFE8LL;
                if ( v162 )
                  goto LABEL_258;
              }
              v168 = v161 - 24LL * v163;
LABEL_259:
              v169 = *(_QWORD *)(v168 + v167);
              if ( *(_BYTE *)(v169 + 16) > 0x17u
                && v346 > 1
                && sub_1377F70((__int64)&v367[3].m128i_i64[1], *(_QWORD *)(v169 + 40)) )
              {
                v169 = sub_1B01300((__int64)&v406, v169)[2];
              }
              v170 = sub_1B01300((__int64)&v424, v160);
              v171 = v170[2];
              if ( v169 != v171 )
              {
                if ( v171 != -8 && v171 != 0 && v171 != -16 )
                  sub_1649B30(v170);
                v170[2] = v169;
                if ( v169 != 0 && v169 != -8 && v169 != -16 )
                  sub_164C220((__int64)v170);
              }
              ++v159;
              sub_157EA20((__int64)(v372 + 40), v161);
              v172 = *(unsigned __int64 **)(v161 + 32);
              v173 = *(_QWORD *)(v161 + 24) & 0xFFFFFFFFFFFFFFF8LL;
              *v172 = v173 | *v172 & 7;
              *(_QWORD *)(v173 + 8) = v172;
              *(_QWORD *)(v161 + 24) &= 7uLL;
              *(_QWORD *)(v161 + 32) = 0;
              sub_164BEC0(v161, v161, v173, (__int64)v172, a7, a8, a9, a10, v174, v175, a13, a14);
              if ( v358 == v159 )
              {
                v85 = *(char **)(v361 - 8);
                break;
              }
            }
          }
        }
        v86 = v372;
        v87 = sub_1B01300((__int64)&v406, (__int64)v85);
        v88 = (char *)v87[2];
        if ( v86 != v88 )
        {
          if ( v88 + 8 != 0 && v88 != 0 && v88 != (char *)-16LL )
            sub_1649B30(v87);
          v87[2] = v86;
          if ( v86 + 8 != 0 && v86 != 0 && v86 != (char *)-16LL )
            sub_164C220((__int64)v87);
        }
        if ( (_DWORD)v426 )
        {
          v148 = (__int64 *)v425;
          v149 = (__int64 *)(v425 + ((unsigned __int64)(unsigned int)v427 << 6));
          if ( (__int64 *)v425 != v149 )
          {
            while ( 1 )
            {
              v150 = v148[3];
              if ( v150 != -8 && v150 != -16 )
                break;
              v148 += 8;
              if ( v149 == v148 )
                goto LABEL_111;
            }
            while ( v149 != v148 )
            {
              v151 = sub_1B01300((__int64)&v406, v148[3]);
              v152 = v151[2];
              v153 = v151;
              v154 = v148[7];
              if ( v152 != v154 )
              {
                if ( v152 != 0 && v152 != -8 && v152 != -16 )
                {
                  sub_1649B30(v153);
                  v154 = v148[7];
                }
                v153[2] = v154;
                if ( v154 != -8 && v154 != 0 && v154 != -16 )
                  sub_1649AC0(v153, v148[5] & 0xFFFFFFFFFFFFFFF8LL);
              }
              v148 += 8;
              if ( v148 == v149 )
                break;
              while ( 1 )
              {
                v155 = v148[3];
                if ( v155 != -16 && v155 != -8 )
                  break;
                v148 += 8;
                if ( v149 == v148 )
                  goto LABEL_111;
              }
            }
          }
        }
LABEL_111:
        v89 = *(char **)(v361 - 8);
        v90 = sub_157EBA0((__int64)v89);
        if ( !v90 )
          goto LABEL_150;
        v91 = 0;
        n = sub_15F4D60(v90);
        v357 = sub_157EBA0((__int64)v89);
        if ( n )
        {
LABEL_115:
          while ( 1 )
          {
            v92 = sub_15F4DF0(v357, v91);
            if ( !sub_1377F70((__int64)&v367[3].m128i_i64[1], v92) )
            {
              v93 = sub_157F280(v92);
              v95 = v94;
              v96 = v93;
              if ( v93 != v94 )
                break;
            }
            if ( n == ++v91 )
              goto LABEL_149;
          }
          src = v91;
          while ( 1 )
          {
            v97 = 0x17FFFFFFE8LL;
            v98 = *(unsigned int *)(v96 + 56);
            v99 = *(_DWORD *)(v96 + 20) & 0xFFFFFFF;
            v100 = *(_BYTE *)(v96 + 23) & 0x40;
            v101 = v99;
            if ( v99 )
            {
              v102 = 24LL * (unsigned int)v98 + 8;
              v103 = 0;
              do
              {
                v104 = v96 - 24LL * v99;
                if ( (_BYTE)v100 )
                  v104 = *(_QWORD *)(v96 - 8);
                if ( *(_QWORD *)(v361 - 8) == *(_QWORD *)(v104 + v102) )
                {
                  v97 = 24 * v103;
                  goto LABEL_125;
                }
                ++v103;
                v102 += 8;
              }
              while ( v99 != (_DWORD)v103 );
              v97 = 0x17FFFFFFE8LL;
            }
LABEL_125:
            if ( (_BYTE)v100 )
              v105 = *(_QWORD *)(v96 - 8);
            else
              v105 = v96 - 24LL * v99;
            v106 = *(_QWORD *)(v105 + v97);
            v107 = v408;
            if ( !v408 )
              goto LABEL_131;
            v100 = (v408 - 1) & (((unsigned int)v106 >> 9) ^ ((unsigned int)v106 >> 4));
            v105 = v407 + (v100 << 6);
            v108 = *(_QWORD *)(v105 + 24);
            if ( v106 == v108 )
              goto LABEL_129;
            v105 = 1;
            if ( v108 != -8 )
              break;
LABEL_131:
            v109 = v372;
            if ( (_DWORD)v98 == v99 )
            {
              sub_15F55D0(v96, v105, v99, v107, v98, v100);
              v101 = *(_DWORD *)(v96 + 20) & 0xFFFFFFF;
            }
            v110 = (v101 + 1) & 0xFFFFFFF;
            v111 = v110 - 1;
            v112 = v110 | *(_DWORD *)(v96 + 20) & 0xF0000000;
            *(_DWORD *)(v96 + 20) = v112;
            if ( (v112 & 0x40000000) != 0 )
              v113 = *(_QWORD *)(v96 - 8);
            else
              v113 = v96 - 24 * v110;
            v114 = (_QWORD *)(v113 + 24LL * v111);
            if ( *v114 )
            {
              v115 = v114[1];
              v116 = v114[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v116 = v115;
              if ( v115 )
                *(_QWORD *)(v115 + 16) = *(_QWORD *)(v115 + 16) & 3LL | v116;
            }
            *v114 = v106;
            if ( v106 )
            {
              v117 = *(_QWORD *)(v106 + 8);
              v114[1] = v117;
              if ( v117 )
                *(_QWORD *)(v117 + 16) = (unsigned __int64)(v114 + 1) | *(_QWORD *)(v117 + 16) & 3LL;
              v114[2] = (v106 + 8) | v114[2] & 3LL;
              *(_QWORD *)(v106 + 8) = v114;
            }
            v118 = *(_DWORD *)(v96 + 20) & 0xFFFFFFF;
            if ( (*(_BYTE *)(v96 + 23) & 0x40) != 0 )
              v119 = *(_QWORD *)(v96 - 8);
            else
              v119 = v96 - 24 * v118;
            *(_QWORD *)(v119 + 8LL * (unsigned int)(v118 - 1) + 24LL * *(unsigned int *)(v96 + 56) + 8) = v109;
            v120 = *(_QWORD *)(v96 + 32);
            if ( !v120 )
              goto LABEL_573;
            v96 = 0;
            if ( *(_BYTE *)(v120 - 8) == 77 )
              v96 = v120 - 24;
            if ( v95 == v96 )
            {
              ++v91;
              if ( n == src + 1 )
                goto LABEL_149;
              goto LABEL_115;
            }
          }
          while ( 1 )
          {
            v146 = v105 + 1;
            v100 = (v408 - 1) & ((_DWORD)v105 + (_DWORD)v100);
            v105 = v407 + ((unsigned __int64)(unsigned int)v100 << 6);
            v147 = *(_QWORD *)(v105 + 24);
            if ( v106 == v147 )
              break;
            v105 = v146;
            if ( v147 == -8 )
              goto LABEL_131;
          }
LABEL_129:
          v107 = v407 + ((unsigned __int64)v408 << 6);
          if ( v105 != v107 )
            v106 = *(_QWORD *)(v105 + 56);
          goto LABEL_131;
        }
LABEL_149:
        v89 = *(char **)(v361 - 8);
LABEL_150:
        if ( v370 == v89 )
        {
          sub_15CE600((__int64)v380, &v372);
          v89 = *(char **)(v361 - 8);
        }
        if ( v369 == v89 )
          sub_15CE600((__int64)&v381, &v372);
        v121 = v388;
        if ( v388 == v389 )
        {
          sub_1292090((__int64)&v387, v388, &v372);
        }
        else
        {
          if ( v388 )
          {
            *(_QWORD *)v388 = v372;
            v121 = v388;
          }
          v388 = v121 + 8;
        }
        v122 = (char *)v385;
        if ( v385 == v386 )
        {
          sub_1292090((__int64)&v384, v385, &v372);
        }
        else
        {
          if ( v385 )
          {
            *(_QWORD *)v385 = v372;
            v122 = (char *)v385;
          }
          v385 = v122 + 8;
        }
        if ( a22 )
        {
          v123 = *(char **)(v361 - 8);
          if ( v123 == v370 )
          {
            sub_1B01660(a22, (__int64)v372, v381[v346 - 1]);
          }
          else
          {
            v124 = sub_15CC510(a22, (__int64)v123);
            v125 = a22;
            v126 = sub_1B01300((__int64)&v406, **(_QWORD **)(v124 + 8));
            sub_1B01660(v125, (__int64)v372, v126[2]);
          }
        }
        if ( v431 )
        {
          if ( v430 )
          {
            v156 = v429;
            v157 = &v429[2 * v430];
            do
            {
              if ( *v156 != -8 && *v156 != -4 )
              {
                v158 = v156[1];
                if ( v158 )
                  sub_161E7C0((__int64)(v156 + 1), v158);
              }
              v156 += 2;
            }
            while ( v157 != v156 );
          }
          j___libc_free_0(v429);
        }
        if ( (_DWORD)v427 )
        {
          v127 = (_QWORD *)v425;
          v391 = 2;
          v392 = 0;
          v128 = -8;
          v129 = (_QWORD *)(v425 + ((unsigned __int64)(unsigned int)v427 << 6));
          v393 = -8;
          v390 = &unk_49E6B50;
          v394 = 0;
          v396 = 2;
          v397 = 0;
          v398 = -16;
          v395 = (char *)&unk_49E6B50;
          v399 = 0;
          while ( 1 )
          {
            v130 = v127[3];
            if ( v130 != v128 )
            {
              v128 = v398;
              if ( v130 != v398 )
              {
                sub_1455FA0((__int64)(v127 + 5));
                v128 = v127[3];
              }
            }
            *v127 = &unk_49EE2B0;
            if ( v128 != -8 && v128 != 0 && v128 != -16 )
              sub_1649B30(v127 + 1);
            v127 += 8;
            if ( v129 == v127 )
              break;
            v128 = v393;
          }
          v395 = (char *)&unk_49EE2B0;
          sub_1455FA0((__int64)&v396);
          v390 = &unk_49EE2B0;
          sub_1455FA0((__int64)&v391);
        }
        j___libc_free_0(v425);
        v361 -= 8;
        if ( v342 == v361 )
          goto LABEL_178;
      }
      v77 = sub_1AFDA40((__int64)&v413, (__int64 *)&v373, &v395);
      v81 = v395;
      if ( v77 )
      {
LABEL_101:
        sub_1B01020((__int64)&v417, (_QWORD *)v81 + 1, (__int64)v81, v78, v79, v80);
        goto LABEL_102;
      }
      ++v413;
      v82 = ((unsigned int)v414 >> 1) + 1;
      if ( (v414 & 1) != 0 )
      {
        v83 = 4;
        if ( 4 * v82 < 0xC )
        {
LABEL_97:
          if ( v83 - (v82 + HIDWORD(v414)) > v83 >> 3 )
          {
LABEL_98:
            v78 = v414 & 1;
            LODWORD(v414) = v78 | (2 * v82);
            if ( *(_QWORD *)v81 != -8 )
              --HIDWORD(v414);
            v84 = v373;
            *((_QWORD *)v81 + 1) = 0;
            *(_QWORD *)v81 = v84;
            goto LABEL_101;
          }
LABEL_276:
          sub_1B00120((__int64)&v413, v83);
          sub_1AFDA40((__int64)&v413, (__int64 *)&v373, &v395);
          v81 = v395;
          v82 = ((unsigned int)v414 >> 1) + 1;
          goto LABEL_98;
        }
      }
      else
      {
        v83 = v416;
        if ( 3 * v416 > 4 * v82 )
          goto LABEL_97;
      }
      v83 *= 2;
      goto LABEL_276;
    }
    if ( a18 )
    {
      v340 = (BYTE4(v366) ^ 1) & 1;
      goto LABEL_17;
    }
  }
LABEL_32:
  v39 = 0;
LABEL_33:
  if ( v371 )
    sub_161E7C0((__int64)&v371, v371);
  return v39;
}
