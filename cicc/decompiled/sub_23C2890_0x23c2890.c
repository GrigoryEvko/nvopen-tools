// Function: sub_23C2890
// Address: 0x23c2890
//
void __fastcall sub_23C2890(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char a6,
        _BYTE *a7,
        unsigned __int64 a8,
        char a9,
        int a10,
        char a11,
        __int64 a12,
        __int64 a13)
{
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdx
  __int64 v18; // r8
  __int64 v19; // r9
  __m128i *v20; // rax
  unsigned __int64 v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  const void *v26; // r13
  size_t v27; // rbx
  int v28; // eax
  unsigned int v29; // r12d
  __int64 *v30; // r15
  __int64 v31; // rdx
  char *v32; // r14
  int v33; // eax
  __int64 v34; // r13
  __int64 v35; // r12
  __int64 v36; // rbx
  size_t *v37; // r13
  unsigned int v38; // r15d
  unsigned int v39; // edx
  size_t *v40; // rax
  bool v41; // r8
  __int64 v42; // rax
  char *v43; // rdx
  unsigned __int64 v44; // rax
  unsigned __int64 v45; // rcx
  bool v46; // cf
  unsigned __int64 v47; // rax
  char *v48; // r15
  __m128i *v49; // rsi
  size_t v50; // rdx
  unsigned __int64 v51; // r15
  __int64 v52; // rax
  _QWORD *v53; // rdi
  _QWORD *v54; // r12
  unsigned __int64 v55; // rdi
  _QWORD *v56; // rdi
  _QWORD *v57; // r12
  unsigned __int64 v58; // rdi
  unsigned __int64 v59; // rdi
  __int64 v60; // r12
  unsigned __int64 v61; // r13
  unsigned __int64 v62; // rdi
  unsigned __int64 v63; // rdi
  __int64 v64; // r12
  void **v65; // r15
  _QWORD *v66; // rdx
  void *v67; // r8
  signed __int64 v68; // r13
  __int64 v69; // rax
  __int64 v70; // rdx
  unsigned __int64 v71; // rax
  unsigned __int64 v72; // r14
  __int64 v73; // rax
  char *v74; // rcx
  unsigned __int64 v75; // r14
  __int64 v76; // r9
  __int64 v77; // r13
  __int64 v78; // rdx
  __m128i *v79; // rax
  char *v80; // rsi
  __m128i *v81; // r15
  __m128i *v82; // rcx
  _QWORD *v83; // r12
  unsigned int v84; // ebx
  __int64 v85; // rax
  _BYTE *v86; // rsi
  __int64 v87; // rdx
  __int64 v88; // rax
  _BYTE *v89; // rsi
  __int64 v90; // rdx
  unsigned __int64 *v91; // rax
  unsigned __int64 v92; // rax
  unsigned __int64 v93; // rdi
  unsigned __int64 *v94; // rax
  unsigned __int64 *v95; // rdx
  unsigned __int8 *v96; // rcx
  size_t v97; // r14
  int v98; // eax
  int v99; // eax
  __int64 *v100; // rax
  __int64 v101; // rax
  _BYTE *v102; // r12
  __int64 v103; // r14
  __m128i *v104; // r14
  size_t v105; // rax
  unsigned __int64 v106; // rdi
  _QWORD *v107; // r9
  _QWORD *v108; // rax
  _QWORD *v109; // rcx
  __int64 v110; // rax
  __int64 v111; // r14
  unsigned __int64 v112; // rbx
  _QWORD *v113; // rax
  unsigned __int64 v114; // r12
  unsigned __int64 v115; // rsi
  _QWORD *v116; // r8
  __int64 v117; // r15
  _QWORD *v118; // rax
  _QWORD *v119; // rcx
  void **v120; // rax
  __m128i *v121; // rsi
  __int64 v122; // r13
  __int64 v123; // rax
  __int64 v124; // r13
  __int64 *v125; // r12
  __int64 v126; // rax
  unsigned __int64 *v127; // rax
  unsigned __int64 v128; // rcx
  __int64 v129; // rax
  __int64 v130; // rax
  size_t v131; // rdx
  _QWORD *v132; // rdx
  unsigned int v133; // eax
  unsigned __int64 v134; // r13
  __int64 v135; // rax
  __int64 v136; // rax
  __int64 v137; // rax
  __int64 v138; // r13
  __int64 v139; // rsi
  __int64 v140; // rax
  _QWORD *v141; // r13
  unsigned __int64 v142; // rbx
  char v143; // r15
  unsigned __int64 v144; // r10
  unsigned __int64 v145; // rdi
  _QWORD *v146; // r8
  _QWORD *v147; // rax
  _QWORD *v148; // rcx
  __int64 v149; // rax
  __int64 v150; // rax
  _BYTE *v151; // rdx
  _BYTE **v152; // rdi
  __int64 v153; // r15
  _BYTE *v154; // rax
  int v155; // r15d
  _QWORD *v156; // r13
  const char *v157; // rsi
  __int64 v158; // r9
  __int64 v159; // r12
  unsigned int v160; // eax
  __int64 v161; // rdx
  __int64 v162; // r12
  unsigned int v163; // ebx
  void **v164; // r13
  unsigned __int64 v165; // r12
  _QWORD *v166; // rbx
  unsigned __int64 v167; // rdi
  unsigned __int64 v168; // rdi
  _QWORD *v169; // rbx
  unsigned __int64 v170; // rdi
  unsigned __int64 v171; // rdi
  unsigned __int64 v172; // rdi
  __int64 v173; // rbx
  unsigned __int64 v174; // r14
  unsigned __int64 v175; // rdi
  unsigned __int64 v176; // rdi
  unsigned __int64 v177; // rdi
  __int64 v178; // r13
  __int64 v179; // rbx
  _QWORD *v180; // r12
  __int64 v181; // r14
  unsigned __int64 v182; // r8
  __int64 v183; // r12
  __int64 v184; // rbx
  _QWORD *v185; // rdi
  char *v186; // rbx
  char *v187; // r12
  unsigned __int64 v188; // rdi
  unsigned __int64 v189; // rdi
  unsigned __int64 v190; // r13
  unsigned __int64 v191; // r14
  unsigned __int64 v192; // rdi
  _QWORD *v193; // rax
  _QWORD *v194; // r12
  __int64 v195; // rdi
  char v196; // al
  unsigned __int64 v197; // rdx
  unsigned __int64 v198; // rbx
  __int64 v199; // r8
  _QWORD *v200; // rax
  char v201; // al
  size_t v202; // r8
  _QWORD *v203; // rax
  __int64 v204; // r8
  __int64 v205; // rax
  __int64 v206; // rdx
  unsigned __int64 v207; // rax
  unsigned __int64 v208; // r12
  __int64 v209; // rax
  __int64 m128i_i64; // r12
  __int64 *v211; // r8
  __m128i *v212; // rax
  size_t v213; // rdx
  const __m128i *v214; // r12
  __m128i *v215; // r14
  __m128i *v216; // rbx
  __int64 v217; // rdx
  unsigned __int64 v218; // rdi
  __m128i v219; // xmm1
  const __m128i *v220; // rdx
  char *v221; // r14
  __int64 v222; // r8
  _QWORD *v223; // rsi
  unsigned __int64 v224; // rdi
  _QWORD *v225; // rcx
  unsigned __int64 v226; // rdx
  char *v227; // rax
  unsigned __int64 v228; // rdi
  size_t v229; // r15
  void *v230; // rax
  _QWORD *v231; // rax
  __int64 v232; // r10
  _QWORD *v233; // r9
  _QWORD *v234; // rsi
  size_t v235; // rdi
  _QWORD *v236; // rcx
  size_t v237; // rdx
  _QWORD **v238; // rax
  unsigned __int64 v239; // rdi
  char *v240; // rax
  __int64 v241; // rax
  unsigned int v242; // eax
  __int64 *v243; // rax
  __int64 *v244; // rax
  __int64 v245; // rdx
  __int64 v246; // rcx
  __int64 v247; // r8
  __int64 v248; // r9
  __m128i si128; // xmm0
  _BYTE **v250; // rdi
  __int64 v251; // rax
  _WORD *v252; // rdx
  __int64 v253; // rdi
  __int64 v254; // rax
  __int64 v255; // rdx
  __int64 v256; // rax
  __int64 v257; // rax
  __int64 v258; // rax
  __int64 v259; // r13
  __int64 v260; // r13
  void *v261; // rax
  __int64 v262; // r13
  __int64 v263; // rax
  void *v264; // rax
  __int64 v265; // r12
  __int64 v266; // rax
  char *v267; // rsi
  __int64 v268; // r13
  size_t v269; // rdx
  unsigned __int8 *v270; // rsi
  __int64 v271; // rdi
  __int64 v272; // rax
  __int64 v273; // r12
  unsigned __int64 v274; // rcx
  __int64 v275; // rax
  char *v277; // [rsp+70h] [rbp-640h]
  __int64 v278; // [rsp+78h] [rbp-638h]
  _QWORD *v279; // [rsp+80h] [rbp-630h]
  char *v280; // [rsp+80h] [rbp-630h]
  void **v281; // [rsp+88h] [rbp-628h]
  unsigned int *v282; // [rsp+88h] [rbp-628h]
  __int64 j; // [rsp+90h] [rbp-620h]
  __int64 v284; // [rsp+90h] [rbp-620h]
  const __m128i *v285; // [rsp+90h] [rbp-620h]
  unsigned __int64 v286; // [rsp+90h] [rbp-620h]
  unsigned __int64 v287; // [rsp+98h] [rbp-618h]
  __int64 v288; // [rsp+98h] [rbp-618h]
  __int64 **v289; // [rsp+98h] [rbp-618h]
  __int64 *v290; // [rsp+98h] [rbp-618h]
  char v291; // [rsp+A0h] [rbp-610h]
  __int64 v292; // [rsp+A0h] [rbp-610h]
  unsigned __int8 *v293; // [rsp+A0h] [rbp-610h]
  unsigned __int64 v294; // [rsp+A0h] [rbp-610h]
  size_t n; // [rsp+A8h] [rbp-608h]
  size_t na; // [rsp+A8h] [rbp-608h]
  size_t nb; // [rsp+A8h] [rbp-608h]
  __int64 nc; // [rsp+A8h] [rbp-608h]
  size_t nd; // [rsp+A8h] [rbp-608h]
  int v300; // [rsp+B0h] [rbp-600h]
  __int64 **k; // [rsp+B0h] [rbp-600h]
  unsigned __int64 v302; // [rsp+B0h] [rbp-600h]
  _QWORD *v303; // [rsp+B0h] [rbp-600h]
  int v304; // [rsp+B8h] [rbp-5F8h]
  char *v305; // [rsp+B8h] [rbp-5F8h]
  _QWORD *v306; // [rsp+B8h] [rbp-5F8h]
  __int64 v307; // [rsp+B8h] [rbp-5F8h]
  char *v308; // [rsp+B8h] [rbp-5F8h]
  void *srcb; // [rsp+C8h] [rbp-5E8h]
  _QWORD *src; // [rsp+C8h] [rbp-5E8h]
  unsigned __int64 srca; // [rsp+C8h] [rbp-5E8h]
  void *srcc; // [rsp+C8h] [rbp-5E8h]
  void *srcd; // [rsp+C8h] [rbp-5E8h]
  int i; // [rsp+D0h] [rbp-5E0h]
  unsigned int v315; // [rsp+D0h] [rbp-5E0h]
  unsigned int *v316; // [rsp+D0h] [rbp-5E0h]
  __int64 v317; // [rsp+D0h] [rbp-5E0h]
  _QWORD v318[2]; // [rsp+E0h] [rbp-5D0h] BYREF
  _QWORD v319[3]; // [rsp+F0h] [rbp-5C0h] BYREF
  void **v320; // [rsp+108h] [rbp-5A8h] BYREF
  unsigned int v321; // [rsp+110h] [rbp-5A0h] BYREF
  __int64 (__fastcall **v322)(); // [rsp+118h] [rbp-598h]
  char **v323; // [rsp+120h] [rbp-590h] BYREF
  void ***v324; // [rsp+128h] [rbp-588h]
  __int16 v325; // [rsp+130h] [rbp-580h]
  unsigned __int64 v326[3]; // [rsp+140h] [rbp-570h] BYREF
  char v327; // [rsp+158h] [rbp-558h] BYREF
  unsigned __int64 v328[3]; // [rsp+160h] [rbp-550h] BYREF
  char v329; // [rsp+178h] [rbp-538h] BYREF
  _BYTE *v330; // [rsp+180h] [rbp-530h] BYREF
  __int64 v331; // [rsp+188h] [rbp-528h]
  __int64 v332[4]; // [rsp+1A0h] [rbp-510h] BYREF
  __int64 *v333; // [rsp+1C0h] [rbp-4F0h] BYREF
  __int64 v334; // [rsp+1C8h] [rbp-4E8h]
  __int64 v335; // [rsp+1D0h] [rbp-4E0h] BYREF
  __m128i *v336; // [rsp+1E0h] [rbp-4D0h]
  size_t v337; // [rsp+1E8h] [rbp-4C8h]
  __m128i v338; // [rsp+1F0h] [rbp-4C0h] BYREF
  unsigned __int8 *v339; // [rsp+200h] [rbp-4B0h] BYREF
  size_t v340; // [rsp+208h] [rbp-4A8h]
  _BYTE v341[16]; // [rsp+210h] [rbp-4A0h] BYREF
  __m128i *v342; // [rsp+220h] [rbp-490h] BYREF
  size_t v343; // [rsp+228h] [rbp-488h]
  __m128i v344; // [rsp+230h] [rbp-480h] BYREF
  __m128i v345; // [rsp+240h] [rbp-470h] BYREF
  _QWORD v346[2]; // [rsp+250h] [rbp-460h] BYREF
  __m128i *v347; // [rsp+260h] [rbp-450h] BYREF
  size_t v348; // [rsp+268h] [rbp-448h]
  __m128i v349; // [rsp+270h] [rbp-440h] BYREF
  __int16 v350; // [rsp+280h] [rbp-430h]
  unsigned __int64 v351[3]; // [rsp+290h] [rbp-420h] BYREF
  _BYTE v352[24]; // [rsp+2A8h] [rbp-408h] BYREF
  __m128i *v353; // [rsp+2C0h] [rbp-3F0h] BYREF
  size_t v354; // [rsp+2C8h] [rbp-3E8h]
  __m128i v355; // [rsp+2D0h] [rbp-3E0h] BYREF
  _BYTE *v356; // [rsp+2E0h] [rbp-3D0h]
  __int64 v357; // [rsp+2E8h] [rbp-3C8h]
  unsigned __int8 **v358; // [rsp+2F0h] [rbp-3C0h]
  void *v359; // [rsp+300h] [rbp-3B0h] BYREF
  __int64 v360[2]; // [rsp+308h] [rbp-3A8h] BYREF
  _QWORD v361[2]; // [rsp+318h] [rbp-398h] BYREF
  __int64 v362; // [rsp+328h] [rbp-388h]
  void **v363; // [rsp+330h] [rbp-380h]
  void **v364; // [rsp+338h] [rbp-378h]
  void *v365; // [rsp+340h] [rbp-370h]
  _QWORD *v366; // [rsp+348h] [rbp-368h]
  _QWORD *v367; // [rsp+350h] [rbp-360h]
  __int64 v368; // [rsp+358h] [rbp-358h]
  char *v369; // [rsp+360h] [rbp-350h] BYREF
  __int64 v370; // [rsp+368h] [rbp-348h]
  __int64 v371; // [rsp+370h] [rbp-340h]
  _BYTE v372[136]; // [rsp+378h] [rbp-338h] BYREF
  __m128i *v373; // [rsp+400h] [rbp-2B0h] BYREF
  size_t v374; // [rsp+408h] [rbp-2A8h] BYREF
  unsigned __int64 v375; // [rsp+410h] [rbp-2A0h] BYREF
  __m128i v376; // [rsp+418h] [rbp-298h] BYREF
  _QWORD v377[2]; // [rsp+428h] [rbp-288h] BYREF
  _QWORD v378[2]; // [rsp+438h] [rbp-278h] BYREF
  void *v379; // [rsp+448h] [rbp-268h] BYREF
  unsigned __int64 v380[2]; // [rsp+450h] [rbp-260h] BYREF
  __m128i v381; // [rsp+460h] [rbp-250h] BYREF
  _QWORD v382[2]; // [rsp+470h] [rbp-240h] BYREF
  _QWORD v383[2]; // [rsp+480h] [rbp-230h] BYREF
  _QWORD v384[6]; // [rsp+490h] [rbp-220h] BYREF
  unsigned __int8 *v385; // [rsp+4C0h] [rbp-1F0h] BYREF
  __int64 v386; // [rsp+4C8h] [rbp-1E8h]
  __int64 v387; // [rsp+4D0h] [rbp-1E0h]
  __int64 v388; // [rsp+4D8h] [rbp-1D8h] BYREF
  __int64 v389; // [rsp+4E0h] [rbp-1D0h]
  __int64 v390; // [rsp+4E8h] [rbp-1C8h]
  unsigned __int64 *v391; // [rsp+4F0h] [rbp-1C0h]
  char *v392; // [rsp+5A0h] [rbp-110h] BYREF
  __int64 v393; // [rsp+5A8h] [rbp-108h]
  __int64 v394; // [rsp+5B0h] [rbp-100h]
  unsigned __int64 v395; // [rsp+5B8h] [rbp-F8h] BYREF
  void *v396; // [rsp+5C0h] [rbp-F0h]
  __m128i *v397; // [rsp+5C8h] [rbp-E8h] BYREF
  unsigned __int64 *v398; // [rsp+5D0h] [rbp-E0h] BYREF
  __m128i v399; // [rsp+5D8h] [rbp-D8h] BYREF
  __m128i *v400; // [rsp+5E8h] [rbp-C8h] BYREF
  unsigned __int64 *v401; // [rsp+5F0h] [rbp-C0h]
  void *v402; // [rsp+5F8h] [rbp-B8h]
  unsigned __int64 *v403; // [rsp+600h] [rbp-B0h]

  v326[0] = (unsigned __int64)&v327;
  v328[0] = (unsigned __int64)&v329;
  v319[0] = a2;
  v319[1] = a3;
  v318[0] = a4;
  v318[1] = a5;
  v326[1] = 0;
  v326[2] = 8;
  v328[1] = 0;
  v328[2] = 8;
  if ( a6 )
  {
    v392 = "{0}_{1}";
    v394 = (__int64)&v400;
    v390 = 0x100000000LL;
    v398 = (unsigned __int64 *)&a11;
    v397 = (__m128i *)&unk_49DB138;
    v399.m128i_i64[0] = (__int64)&unk_49DB138;
    v399.m128i_i64[1] = a1 + 36;
    v391 = (unsigned __int64 *)&v373;
    v385 = (unsigned __int8 *)&unk_49DD288;
    v393 = 7;
    v395 = 2;
    LOBYTE(v396) = 1;
    v400 = &v399;
    v401 = (unsigned __int64 *)&v397;
    v373 = &v376;
    v374 = 0;
    v375 = 8;
    v386 = 2;
    v387 = 0;
    v388 = 0;
    v389 = 0;
    sub_CB5980((__int64)&v385, 0, 0, 0);
    sub_CB6840((__int64)&v385, (__int64)&v392);
    v385 = (unsigned __int8 *)&unk_49DD388;
    sub_CB5840((__int64)&v385);
    sub_23AE8F0((__int64)v326, (char **)&v373, v13, v14, v15, v16);
    if ( v373 != &v376 )
      _libc_free((unsigned __int64)v373);
    v393 = 7;
    v392 = "{0}.{1}";
    v395 = 2;
    v394 = (__int64)&v400;
    LOBYTE(v396) = 1;
    v398 = (unsigned __int64 *)&a11;
    v397 = (__m128i *)&unk_49DB138;
    v399.m128i_i64[0] = (__int64)&unk_49DB138;
    v400 = &v399;
    v399.m128i_i64[1] = a1 + 36;
    v401 = (unsigned __int64 *)&v397;
  }
  else
  {
    v392 = "{0}";
    v397 = (__m128i *)&unk_49DB138;
    v391 = (unsigned __int64 *)&v373;
    v398 = (unsigned __int64 *)(a1 + 36);
    v390 = 0x100000000LL;
    v393 = 3;
    v385 = (unsigned __int8 *)&unk_49DD288;
    v394 = (__int64)&v399;
    v395 = 1;
    LOBYTE(v396) = 1;
    v399.m128i_i64[0] = (__int64)&v397;
    v373 = &v376;
    v374 = 0;
    v375 = 8;
    v386 = 2;
    v387 = 0;
    v388 = 0;
    v389 = 0;
    sub_CB5980((__int64)&v385, 0, 0, 0);
    sub_CB6840((__int64)&v385, (__int64)&v392);
    v385 = (unsigned __int8 *)&unk_49DD388;
    sub_CB5840((__int64)&v385);
    sub_23AE8F0((__int64)v326, (char **)&v373, v245, v246, v247, v248);
    if ( v373 != &v376 )
      _libc_free((unsigned __int64)v373);
    v393 = 3;
    v392 = "{0}";
    v394 = (__int64)&v399;
    LOBYTE(v396) = 1;
    v397 = (__m128i *)&unk_49DB138;
    v395 = 1;
    v398 = (unsigned __int64 *)(a1 + 36);
    v399.m128i_i64[0] = (__int64)&v397;
  }
  v390 = 0x100000000LL;
  v373 = &v376;
  v374 = 0;
  v385 = (unsigned __int8 *)&unk_49DD288;
  v375 = 8;
  v391 = (unsigned __int64 *)&v373;
  v386 = 2;
  v387 = 0;
  v388 = 0;
  v389 = 0;
  sub_CB5980((__int64)&v385, 0, 0, 0);
  sub_CB6840((__int64)&v385, (__int64)&v392);
  v385 = (unsigned __int8 *)&unk_49DD388;
  sub_CB5840((__int64)&v385);
  sub_23AE8F0((__int64)v328, (char **)&v373, v17, (__int64)v328, v18, v19);
  if ( v373 != &v376 )
    _libc_free((unsigned __int64)v373);
  v370 = 0;
  v369 = v372;
  v392 = "cfgdot-%%%%%%.dot";
  v371 = 128;
  LOWORD(v396) = 259;
  sub_C85490((__int64)&v392, &v369, 1);
  LOWORD(v396) = 261;
  v392 = v369;
  v393 = v370;
  sub_CA0F50((__int64 *)&v330, (void **)&v392);
  v392 = "diff_{0}.pdf";
  v351[0] = (unsigned __int64)v352;
  v390 = 0x100000000LL;
  v397 = (__m128i *)&unk_4A16088;
  v393 = 12;
  v398 = v326;
  v385 = (unsigned __int8 *)&unk_49DD288;
  v391 = v351;
  v394 = (__int64)&v399;
  v395 = 1;
  LOBYTE(v396) = 1;
  v399.m128i_i64[0] = (__int64)&v397;
  v351[1] = 0;
  v351[2] = 20;
  v386 = 2;
  v387 = 0;
  v388 = 0;
  v389 = 0;
  sub_CB5980((__int64)&v385, 0, 0, 0);
  sub_CB6840((__int64)&v385, (__int64)&v392);
  v385 = (unsigned __int8 *)&unk_49DD388;
  sub_CB5840((__int64)&v385);
  v385 = (unsigned __int8 *)&v388;
  v386 = 0;
  v387 = 200;
  sub_23AF980((__int64)&v353, a7, a8);
  v373 = (__m128i *)&unk_49E64B0;
  v20 = v353;
  if ( v353 == &v355 )
  {
    v20 = &v376;
    v376 = _mm_load_si128(&v355);
  }
  else
  {
    v376.m128i_i64[0] = v355.m128i_i64[0];
  }
  v355.m128i_i8[0] = 0;
  v21 = v354;
  v393 = (__int64)v319;
  v353 = &v355;
  v392 = (char *)&unk_49DB108;
  v394 = (__int64)&unk_49DB108;
  v395 = (unsigned __int64)&a9;
  v354 = 0;
  v397 = &v399;
  v396 = &unk_49E64B0;
  if ( v20 == &v376 )
  {
    v399 = _mm_loadu_si128(&v376);
  }
  else
  {
    v397 = v20;
    v399.m128i_i64[0] = v376.m128i_i64[0];
  }
  v374 = (size_t)&v376;
  v401 = v318;
  v398 = (unsigned __int64 *)v21;
  v376.m128i_i8[0] = 0;
  v402 = &unk_4A16088;
  v375 = 0;
  v403 = v328;
  v400 = (__m128i *)&unk_49DB108;
  sub_2240A30(&v374);
  v374 = 16;
  v373 = (__m128i *)"{0}.{1}{2}{3}{4}";
  v375 = (unsigned __int64)v384;
  v376.m128i_i64[0] = 5;
  v377[1] = v393;
  v376.m128i_i8[8] = 1;
  v378[1] = v395;
  v377[0] = &unk_49DB108;
  v378[0] = &unk_49DB108;
  v379 = &unk_49E64B0;
  v380[0] = (unsigned __int64)&v381;
  if ( v397 == &v399 )
  {
    v381 = _mm_loadu_si128(&v399);
  }
  else
  {
    v380[0] = (unsigned __int64)v397;
    v381.m128i_i64[0] = v399.m128i_i64[0];
  }
  v397 = &v399;
  v399.m128i_i8[0] = 0;
  v380[1] = (unsigned __int64)v398;
  v382[0] = &unk_49DB108;
  v398 = 0;
  v382[1] = v401;
  v383[0] = &unk_4A16088;
  v383[1] = v403;
  v384[0] = v383;
  v384[1] = v382;
  v384[2] = &v379;
  v384[3] = v378;
  v384[4] = v377;
  v396 = &unk_49E64B0;
  sub_2240A30((unsigned __int64 *)&v397);
  v392 = (char *)&v395;
  v362 = 0x100000000LL;
  v393 = 0;
  v394 = 200;
  v359 = &unk_49DD288;
  v360[0] = 2;
  v360[1] = 0;
  v361[0] = 0;
  v361[1] = 0;
  v363 = (void **)&v392;
  sub_CB5980((__int64)&v359, 0, 0, 0);
  sub_CB6840((__int64)&v359, (__int64)&v373);
  v359 = &unk_49DD388;
  sub_CB5840((__int64)&v359);
  sub_23AE8F0((__int64)&v385, &v392, v22, v23, v24, v25);
  if ( v392 != (char *)&v395 )
    _libc_free((unsigned __int64)v392);
  v379 = &unk_49E64B0;
  sub_2240A30(v380);
  sub_2240A30((unsigned __int64 *)&v353);
  sub_23B97D0((__int64)&v392, (__int64)v385, v386, a12, a13);
  sub_2241BD0(v332, a13 + 48);
  if ( !sub_2241AC0((__int64)v332, byte_3F871B3) )
  {
    sub_2241BD0((__int64 *)&v373, a12 + 48);
    sub_23AEBB0((__int64)v332, (__int64)&v373);
    sub_2240A30((unsigned __int64 *)&v373);
  }
  v26 = (const void *)v332[0];
  v27 = v332[1];
  v339 = v385;
  v340 = v386;
  v28 = sub_C92610();
  v29 = sub_C92740((__int64)&v395, v26, v27, v28);
  v30 = (__int64 *)(v395 + 8LL * v29);
  v31 = *v30;
  if ( *v30 )
  {
    if ( v31 != -8 )
      goto LABEL_19;
    LODWORD(v397) = (_DWORD)v397 - 1;
  }
  v241 = sub_23AE710(16, 8, v26, v27);
  if ( v241 )
  {
    *(_QWORD *)v241 = v27;
    *(_DWORD *)(v241 + 8) = 0;
  }
  *v30 = v241;
  ++HIDWORD(v396);
  v242 = sub_C929D0((__int64 *)&v395, v29);
  v243 = (__int64 *)(v395 + 8LL * v242);
  v31 = *v243;
  if ( *v243 == -8 || !v31 )
  {
    v244 = v243 + 1;
    do
    {
      do
        v31 = *v244++;
      while ( v31 == -8 );
    }
    while ( !v31 );
  }
LABEL_19:
  v300 = *(_DWORD *)(v31 + 8);
  sub_95CA80((__int64 *)&v373, (__int64)&v339);
  v360[0] = (__int64)v361;
  LOBYTE(v359) = 0;
  sub_23AEDD0(v360, v373, (__int64)v373->m128i_i64 + v374);
  v362 = 0;
  v363 = 0;
  v364 = 0;
  v365 = 0;
  v366 = 0;
  v367 = 0;
  v368 = 0;
  if ( v373 != (__m128i *)&v375 )
    j_j___libc_free_0((unsigned __int64)v373);
  v32 = v392;
  LODWORD(v374) = 0;
  v375 = 0;
  v376.m128i_i64[0] = (__int64)&v374;
  v376.m128i_i64[1] = (__int64)&v374;
  v377[0] = 0;
  n = v393;
  if ( v392 != (char *)v393 )
  {
    v304 = -1;
    for ( i = 0; ; ++i )
    {
      v33 = v304;
      if ( v300 == *((_DWORD *)v32 + 2) )
        v33 = i;
      v34 = *((_QWORD *)v32 + 4);
      v35 = *((_QWORD *)v32 + 5);
      v304 = v33;
      sub_23B4900(&v345, (__int64)v32);
      v36 = (__int64)v363;
      if ( v363 != v364 )
      {
        v353 = &v355;
        sub_23AEDD0((__int64 *)&v353, v345.m128i_i64[0], v345.m128i_i64[0] + v345.m128i_i64[1]);
        if ( v36 )
        {
          *(_QWORD *)v36 = v34;
          *(_QWORD *)(v36 + 16) = v36 + 32;
          *(_QWORD *)(v36 + 8) = v35;
          sub_23AEDD0((__int64 *)(v36 + 16), v353, (__int64)v353->m128i_i64 + v354);
          *(_QWORD *)(v36 + 48) = 0;
          *(_QWORD *)(v36 + 96) = v36 + 144;
          *(_QWORD *)(v36 + 56) = 0;
          *(_QWORD *)(v36 + 64) = 0;
          *(_QWORD *)(v36 + 72) = 0;
          *(_QWORD *)(v36 + 80) = 0;
          *(_QWORD *)(v36 + 88) = 0;
          *(_QWORD *)(v36 + 104) = 1;
          *(_QWORD *)(v36 + 112) = 0;
          *(_QWORD *)(v36 + 120) = 0;
          *(_QWORD *)(v36 + 136) = 0;
          *(_QWORD *)(v36 + 144) = 0;
          *(_QWORD *)(v36 + 152) = v36 + 200;
          *(_QWORD *)(v36 + 160) = 1;
          *(_QWORD *)(v36 + 168) = 0;
          *(_QWORD *)(v36 + 176) = 0;
          *(_QWORD *)(v36 + 192) = 0;
          *(_QWORD *)(v36 + 200) = 0;
          *(_BYTE *)(v36 + 208) = 0;
          *(_DWORD *)(v36 + 128) = 1065353216;
          *(_DWORD *)(v36 + 184) = 1065353216;
        }
        if ( v353 != &v355 )
          j_j___libc_free_0((unsigned __int64)v353);
        v363 += 27;
        goto LABEL_31;
      }
      v43 = (char *)v363 - v362;
      v287 = v362;
      v44 = 0x84BDA12F684BDA13LL * (((__int64)v363 - v362) >> 3);
      if ( v44 == 0x97B425ED097B42LL )
LABEL_475:
        sub_4262D8((__int64)"vector::_M_realloc_insert");
      v45 = 1;
      if ( v44 )
        v45 = 0x84BDA12F684BDA13LL * (((__int64)v363 - v362) >> 3);
      v46 = __CFADD__(v45, v44);
      v47 = v45 - 0x7B425ED097B425EDLL * (((__int64)v363 - v362) >> 3);
      if ( v46 )
        break;
      if ( v47 )
      {
        if ( v47 > 0x97B425ED097B42LL )
          v47 = 0x97B425ED097B42LL;
        v274 = 216 * v47;
        goto LABEL_472;
      }
      v281 = 0;
      v292 = 0;
LABEL_56:
      v48 = &v43[v292];
      v353 = &v355;
      sub_23AEDD0((__int64 *)&v353, v345.m128i_i64[0], v345.m128i_i64[0] + v345.m128i_i64[1]);
      if ( v48 )
      {
        v49 = v353;
        *(_QWORD *)v48 = v34;
        *((_QWORD *)v48 + 2) = v48 + 32;
        v50 = v354;
        *((_QWORD *)v48 + 1) = v35;
        sub_23AEDD0((__int64 *)v48 + 2, v49, (__int64)v49->m128i_i64 + v50);
        *((_QWORD *)v48 + 6) = 0;
        *((_QWORD *)v48 + 12) = v48 + 144;
        *((_QWORD *)v48 + 7) = 0;
        *((_QWORD *)v48 + 8) = 0;
        *((_QWORD *)v48 + 9) = 0;
        *((_QWORD *)v48 + 10) = 0;
        *((_QWORD *)v48 + 11) = 0;
        *((_QWORD *)v48 + 13) = 1;
        *((_QWORD *)v48 + 14) = 0;
        *((_QWORD *)v48 + 15) = 0;
        *((_DWORD *)v48 + 32) = 1065353216;
        *((_QWORD *)v48 + 17) = 0;
        *((_QWORD *)v48 + 18) = 0;
        *((_QWORD *)v48 + 19) = v48 + 200;
        *((_QWORD *)v48 + 20) = 1;
        *((_QWORD *)v48 + 21) = 0;
        *((_QWORD *)v48 + 22) = 0;
        *((_DWORD *)v48 + 46) = 1065353216;
        *((_QWORD *)v48 + 24) = 0;
        *((_QWORD *)v48 + 25) = 0;
        v48[208] = 0;
      }
      if ( v353 != &v355 )
        j_j___libc_free_0((unsigned __int64)v353);
      v51 = v287;
      v52 = sub_23AF510(v287, v36, v292);
      for ( j = sub_23AF510(v36, v36, v52 + 216); v36 != v51; v51 += 216LL )
      {
        v53 = *(_QWORD **)(v51 + 168);
        if ( v53 )
        {
          while ( 1 )
          {
            v54 = (_QWORD *)*v53;
            j_j___libc_free_0((unsigned __int64)v53);
            if ( !v54 )
              break;
            v53 = v54;
          }
        }
        memset(*(void **)(v51 + 152), 0, 8LL * *(_QWORD *)(v51 + 160));
        v55 = *(_QWORD *)(v51 + 152);
        *(_QWORD *)(v51 + 176) = 0;
        *(_QWORD *)(v51 + 168) = 0;
        if ( v55 != v51 + 200 )
          j_j___libc_free_0(v55);
        v56 = *(_QWORD **)(v51 + 112);
        if ( v56 )
        {
          while ( 1 )
          {
            v57 = (_QWORD *)*v56;
            j_j___libc_free_0((unsigned __int64)v56);
            if ( !v57 )
              break;
            v56 = v57;
          }
        }
        memset(*(void **)(v51 + 96), 0, 8LL * *(_QWORD *)(v51 + 104));
        v58 = *(_QWORD *)(v51 + 96);
        *(_QWORD *)(v51 + 120) = 0;
        *(_QWORD *)(v51 + 112) = 0;
        if ( v58 != v51 + 144 )
          j_j___libc_free_0(v58);
        v59 = *(_QWORD *)(v51 + 72);
        if ( v59 )
          j_j___libc_free_0(v59);
        v60 = *(_QWORD *)(v51 + 56);
        v61 = *(_QWORD *)(v51 + 48);
        if ( v60 != v61 )
        {
          do
          {
            v62 = *(_QWORD *)(v61 + 16);
            if ( v62 != v61 + 32 )
              j_j___libc_free_0(v62);
            v61 += 56LL;
          }
          while ( v60 != v61 );
          v61 = *(_QWORD *)(v51 + 48);
        }
        if ( v61 )
          j_j___libc_free_0(v61);
        v63 = *(_QWORD *)(v51 + 16);
        if ( v63 != v51 + 32 )
          j_j___libc_free_0(v63);
      }
      if ( v287 )
        j_j___libc_free_0(v287);
      v362 = v292;
      v363 = (void **)j;
      v364 = v281;
LABEL_31:
      if ( (_QWORD *)v345.m128i_i64[0] != v346 )
        j_j___libc_free_0(v345.m128i_u64[0]);
      v37 = (size_t *)v375;
      v38 = *((_DWORD *)v32 + 2);
      if ( v375 )
      {
        while ( 1 )
        {
          v39 = *((_DWORD *)v37 + 8);
          v40 = (size_t *)v37[3];
          if ( v38 < v39 )
            v40 = (size_t *)v37[2];
          if ( !v40 )
            break;
          v37 = v40;
        }
        if ( v38 >= v39 )
        {
          if ( v38 <= v39 )
            goto LABEL_43;
LABEL_41:
          v41 = 1;
          if ( v37 == &v374 )
          {
LABEL_42:
            v291 = v41;
            v42 = sub_22077B0(0x28u);
            *(_DWORD *)(v42 + 32) = v38;
            *(_DWORD *)(v42 + 36) = i;
            sub_220F040(v291, v42, v37, &v374);
            ++v377[0];
            goto LABEL_43;
          }
LABEL_49:
          v41 = v38 < *((_DWORD *)v37 + 8);
          goto LABEL_42;
        }
        if ( (size_t *)v376.m128i_i64[0] == v37 )
          goto LABEL_41;
      }
      else
      {
        v37 = &v374;
        if ( (size_t *)v376.m128i_i64[0] == &v374 )
        {
          v37 = &v374;
          v41 = 1;
          goto LABEL_42;
        }
      }
      if ( v38 > *(_DWORD *)(sub_220EF80((__int64)v37) + 32) && v37 )
      {
        v41 = 1;
        if ( v37 == &v374 )
          goto LABEL_42;
        goto LABEL_49;
      }
LABEL_43:
      v32 += 144;
      if ( (char *)n == v32 )
      {
        v315 = v304;
        goto LABEL_91;
      }
    }
    v274 = 0x7FFFFFFFFFFFFFB0LL;
LABEL_472:
    v280 = (char *)v363 - v362;
    v286 = v274;
    v275 = sub_22077B0(v274);
    v43 = v280;
    v292 = v275;
    v281 = (void **)(v275 + v286);
    goto LABEL_56;
  }
  v315 = -1;
LABEL_91:
  v64 = v362;
  LOBYTE(v359) = 1;
  v65 = v363;
  if ( (void **)v362 != v363 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v66 = v366;
        if ( v366 == v367 )
          break;
        if ( v366 )
        {
          *v366 = v64;
          v66 = v366;
        }
        v64 += 216;
        v366 = v66 + 1;
        if ( v65 == (void **)v64 )
          goto LABEL_111;
      }
      v67 = v365;
      v68 = (char *)v366 - (_BYTE *)v365;
      v69 = ((char *)v366 - (_BYTE *)v365) >> 3;
      if ( v69 == 0xFFFFFFFFFFFFFFFLL )
        goto LABEL_475;
      v70 = 1;
      if ( v69 )
        v70 = ((char *)v366 - (_BYTE *)v365) >> 3;
      v46 = __CFADD__(v70, v69);
      v71 = v70 + v69;
      if ( v46 )
        break;
      if ( v71 )
      {
        if ( v71 > 0xFFFFFFFFFFFFFFFLL )
          v71 = 0xFFFFFFFFFFFFFFFLL;
        v72 = 8 * v71;
        goto LABEL_105;
      }
      v75 = 0;
      v74 = 0;
LABEL_106:
      if ( &v74[v68] )
        *(_QWORD *)&v74[v68] = v64;
      v76 = (__int64)&v74[v68 + 8];
      if ( v68 > 0 )
      {
        v307 = (__int64)&v74[v68 + 8];
        srcc = v67;
        v240 = (char *)memmove(v74, v67, v68);
        v67 = srcc;
        v76 = v307;
        v74 = v240;
LABEL_409:
        v308 = v74;
        srcd = (void *)v76;
        j_j___libc_free_0((unsigned __int64)v67);
        v74 = v308;
        v76 = (__int64)srcd;
        goto LABEL_110;
      }
      if ( v67 )
        goto LABEL_409;
LABEL_110:
      v64 += 216;
      v365 = v74;
      v366 = (_QWORD *)v76;
      v367 = (_QWORD *)v75;
      if ( v65 == (void **)v64 )
        goto LABEL_111;
    }
    v72 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_105:
    srcb = v365;
    v73 = sub_22077B0(v72);
    v67 = srcb;
    v74 = (char *)v73;
    v75 = v73 + v72;
    goto LABEL_106;
  }
LABEL_111:
  v368 = *((_QWORD *)v365 + v315);
  v278 = v376.m128i_i64[0];
  if ( (size_t *)v376.m128i_i64[0] != &v374 )
  {
    while ( 1 )
    {
      v305 = &v392[144 * *(unsigned int *)(v278 + 32)];
      v77 = v362 + 216LL * *(unsigned int *)(v278 + 36);
      v282 = (unsigned int *)*((_QWORD *)v305 + 16);
      v316 = (unsigned int *)*((_QWORD *)v305 + 15);
      v279 = (_QWORD *)(v77 + 112);
      while ( v282 != v316 )
      {
        v78 = *v316;
        v79 = (__m128i *)*((_QWORD *)v305 + 8);
        if ( !v79 )
          goto LABEL_473;
        v80 = v305 + 56;
        while ( 1 )
        {
          v81 = (__m128i *)v79[1].m128i_i64[0];
          v82 = (__m128i *)v79[1].m128i_i64[1];
          if ( (unsigned int)v78 > v79[2].m128i_i32[0] )
          {
            v79 = (__m128i *)v80;
            v81 = v82;
          }
          if ( !v81 )
            break;
          v80 = (char *)v79;
          v79 = v81;
        }
        if ( v305 + 56 == (char *)v79 || (unsigned int)v78 < v79[2].m128i_i32[0] )
LABEL_473:
          sub_426320((__int64)"map::at");
        v288 = v79[4].m128i_i64[1];
        v284 = v79[5].m128i_i64[0];
        v83 = *(_QWORD **)v305;
        v84 = *((_DWORD *)v305 + 2);
        src = (_QWORD *)(**(_QWORD **)v305 + 144 * v78);
        v85 = src[2];
        v86 = *(_BYTE **)v85;
        if ( *(_QWORD *)v85 )
        {
          v87 = *(_QWORD *)(v85 + 8);
          v345.m128i_i64[0] = (__int64)v346;
          sub_23AE760(v345.m128i_i64, v86, (__int64)&v86[v87]);
        }
        else
        {
          LOBYTE(v346[0]) = 0;
          v345.m128i_i64[0] = (__int64)v346;
          v345.m128i_i64[1] = 0;
        }
        v88 = *(_QWORD *)(*v83 + 144LL * v84 + 16);
        v89 = *(_BYTE **)v88;
        if ( *(_QWORD *)v88 )
        {
          v90 = *(_QWORD *)(v88 + 8);
          v353 = &v355;
          sub_23AE760((__int64 *)&v353, v89, (__int64)&v89[v90]);
          if ( v354 == 0x3FFFFFFFFFFFFFFFLL )
            sub_4262D8((__int64)"basic_string::append");
        }
        else
        {
          v355.m128i_i8[0] = 0;
          v354 = 0;
          v353 = &v355;
        }
        v91 = sub_2241490((unsigned __int64 *)&v353, " ", 1u);
        v347 = &v349;
        if ( (unsigned __int64 *)*v91 == v91 + 2 )
        {
          v349 = _mm_loadu_si128((const __m128i *)v91 + 1);
        }
        else
        {
          v347 = (__m128i *)*v91;
          v349.m128i_i64[0] = v91[2];
        }
        v348 = v91[1];
        *v91 = (unsigned __int64)(v91 + 2);
        v91[1] = 0;
        *((_BYTE *)v91 + 16) = 0;
        v92 = 15;
        v93 = 15;
        if ( v347 != &v349 )
          v93 = v349.m128i_i64[0];
        if ( v348 + v345.m128i_i64[1] <= v93 )
          goto LABEL_133;
        if ( (_QWORD *)v345.m128i_i64[0] != v346 )
          v92 = v346[0];
        if ( v348 + v345.m128i_i64[1] <= v92 )
        {
          v94 = sub_2241130((unsigned __int64 *)&v345, 0, 0, v347, v348);
          v95 = v94 + 2;
          v342 = &v344;
          v96 = (unsigned __int8 *)*v94;
          if ( (unsigned __int64 *)*v94 != v94 + 2 )
          {
LABEL_134:
            v342 = (__m128i *)v96;
            v344.m128i_i64[0] = v94[2];
            goto LABEL_135;
          }
        }
        else
        {
LABEL_133:
          v94 = sub_2241490((unsigned __int64 *)&v347, (char *)v345.m128i_i64[0], v345.m128i_u64[1]);
          v95 = v94 + 2;
          v342 = &v344;
          v96 = (unsigned __int8 *)*v94;
          if ( (unsigned __int64 *)*v94 != v94 + 2 )
            goto LABEL_134;
        }
        v344 = _mm_loadu_si128((const __m128i *)v94 + 1);
LABEL_135:
        v343 = v94[1];
        *v94 = (unsigned __int64)v95;
        v94[1] = 0;
        *((_BYTE *)v94 + 16) = 0;
        if ( v347 != &v349 )
          j_j___libc_free_0((unsigned __int64)v347);
        if ( v353 != &v355 )
          j_j___libc_free_0((unsigned __int64)v353);
        if ( (_QWORD *)v345.m128i_i64[0] != v346 )
          j_j___libc_free_0(v345.m128i_u64[0]);
        v97 = v343;
        v293 = (unsigned __int8 *)v342;
        v98 = sub_C92610();
        v99 = sub_C92860(v83 + 10, v293, v97, v98);
        if ( v99 == -1 )
          v100 = (__int64 *)(v83[10] + 8LL * *((unsigned int *)v83 + 22));
        else
          v100 = (__int64 *)(v83[10] + 8LL * v99);
        v101 = *v100;
        v102 = *(_BYTE **)(v101 + 8);
        v103 = *(_QWORD *)(v101 + 16);
        if ( v342 != &v344 )
          j_j___libc_free_0((unsigned __int64)v342);
        srca = v362 + 216LL * *((unsigned int *)src + 2);
        if ( v102 )
        {
          v347 = &v349;
          sub_23AE760((__int64 *)&v347, v102, (__int64)&v102[v103]);
          v104 = *(__m128i **)(v77 + 56);
          if ( v104 != *(__m128i **)(v77 + 64) )
            goto LABEL_147;
        }
        else
        {
          v347 = &v349;
          v348 = 0;
          v349.m128i_i8[0] = 0;
          v104 = *(__m128i **)(v77 + 56);
          if ( v104 != *(__m128i **)(v77 + 64) )
          {
LABEL_147:
            v353 = &v355;
            if ( v347 == &v349 )
            {
              v355 = _mm_load_si128(&v349);
            }
            else
            {
              v353 = v347;
              v355.m128i_i64[0] = v349.m128i_i64[0];
            }
            v105 = v348;
            v347 = &v349;
            v348 = 0;
            v354 = v105;
            v349.m128i_i8[0] = 0;
            if ( v104 )
            {
              v104->m128i_i64[0] = v288;
              v104->m128i_i64[1] = v284;
              v104[1].m128i_i64[0] = (__int64)v104[2].m128i_i64;
              sub_23AEDD0(v104[1].m128i_i64, v353, (__int64)v353->m128i_i64 + v354);
              v104[3].m128i_i64[0] = srca;
            }
            if ( v353 != &v355 )
              j_j___libc_free_0((unsigned __int64)v353);
            *(_QWORD *)(v77 + 56) += 56LL;
            goto LABEL_154;
          }
        }
        v204 = (__int64)v104->m128i_i64 - *(_QWORD *)(v77 + 48);
        nb = *(_QWORD *)(v77 + 48);
        v205 = 0x6DB6DB6DB6DB6DB7LL * (v204 >> 3);
        if ( v205 == 0x249249249249249LL )
          goto LABEL_475;
        v206 = 1;
        if ( v205 )
          v206 = 0x6DB6DB6DB6DB6DB7LL * (v204 >> 3);
        v46 = __CFADD__(v206, v205);
        v207 = v206 + v205;
        if ( v46 )
        {
          v208 = 0x7FFFFFFFFFFFFFF8LL;
        }
        else
        {
          if ( !v207 )
          {
            v294 = 0;
            m128i_i64 = 56;
            goto LABEL_341;
          }
          if ( v207 > 0x249249249249249LL )
            v207 = 0x249249249249249LL;
          v208 = 56 * v207;
        }
        v277 = &v104->m128i_i8[-*(_QWORD *)(v77 + 48)];
        v209 = sub_22077B0(v208);
        v204 = (__int64)v277;
        v81 = (__m128i *)v209;
        v294 = v208 + v209;
        m128i_i64 = v209 + 56;
LABEL_341:
        v211 = (__int64 *)((char *)v81->m128i_i64 + v204);
        v353 = &v355;
        v212 = v347;
        if ( v347 == &v349 )
        {
          v212 = &v355;
          v355 = _mm_load_si128(&v349);
        }
        else
        {
          v353 = v347;
          v355.m128i_i64[0] = v349.m128i_i64[0];
        }
        v213 = v348;
        v347 = &v349;
        v348 = 0;
        v354 = v213;
        v349.m128i_i8[0] = 0;
        if ( v211 )
        {
          v211[2] = (__int64)(v211 + 4);
          *v211 = v288;
          v290 = v211;
          v211[1] = v284;
          sub_23AEDD0(v211 + 2, v212, (__int64)v212->m128i_i64 + v213);
          v290[6] = srca;
          v212 = v353;
        }
        if ( v212 != &v355 )
          j_j___libc_free_0((unsigned __int64)v212);
        if ( v104 != (__m128i *)nb )
        {
          v214 = (const __m128i *)(nb + 32);
          v285 = v104;
          v215 = v81;
          v216 = v81 + 2;
          while ( 1 )
          {
            if ( v215 )
            {
              v219 = _mm_loadu_si128(v214 - 2);
              v216[-1].m128i_i64[0] = (__int64)v216;
              v216[-2] = v219;
              v220 = (const __m128i *)v214[-1].m128i_i64[0];
              if ( v214 == v220 )
              {
                *v216 = _mm_loadu_si128(v214);
              }
              else
              {
                v216[-1].m128i_i64[0] = (__int64)v220;
                v216->m128i_i64[0] = v214->m128i_i64[0];
              }
              v216[-1].m128i_i64[1] = v214[-1].m128i_i64[1];
              v217 = v214[1].m128i_i64[0];
              v214[-1].m128i_i64[0] = (__int64)v214;
              v214[-1].m128i_i64[1] = 0;
              v214->m128i_i8[0] = 0;
              v216[1].m128i_i64[0] = v217;
            }
            v218 = v214[-1].m128i_u64[0];
            if ( v214 != (const __m128i *)v218 )
              j_j___libc_free_0(v218);
            v216 = (__m128i *)((char *)v216 + 56);
            if ( v285 == (const __m128i *)&v214[1].m128i_u64[1] )
              break;
            v214 = (const __m128i *)((char *)v214 + 56);
            v215 = (__m128i *)((char *)v215 + 56);
          }
          m128i_i64 = (__int64)v215[7].m128i_i64;
        }
        if ( nb )
          j_j___libc_free_0(nb);
        *(_QWORD *)(v77 + 48) = v81;
        *(_QWORD *)(v77 + 56) = m128i_i64;
        *(_QWORD *)(v77 + 64) = v294;
LABEL_154:
        if ( v347 != &v349 )
          j_j___libc_free_0((unsigned __int64)v347);
        v106 = *(_QWORD *)(v77 + 104);
        v107 = *(_QWORD **)(*(_QWORD *)(v77 + 96) + 8 * (srca % v106));
        if ( !v107 )
          goto LABEL_321;
        v108 = (_QWORD *)*v107;
        if ( srca != *(_QWORD *)(*v107 + 8LL) )
        {
          while ( 1 )
          {
            v109 = (_QWORD *)*v108;
            if ( !*v108 )
              break;
            v107 = v108;
            if ( srca % v106 != v109[1] % v106 )
              break;
            v108 = (_QWORD *)*v108;
            if ( srca == v109[1] )
              goto LABEL_161;
          }
LABEL_321:
          na = 8 * (srca % v106);
          v193 = (_QWORD *)sub_22077B0(0x10u);
          v194 = v193;
          if ( v193 )
            *v193 = 0;
          v195 = v77 + 128;
          v193[1] = srca;
          v115 = *(_QWORD *)(v77 + 104);
          v196 = sub_222DA10(v77 + 128, v115, *(_QWORD *)(v77 + 120), 1);
          v198 = v197;
          if ( !v196 )
          {
            v199 = na;
            v200 = *(_QWORD **)(*(_QWORD *)(v77 + 96) + na);
            if ( v200 )
              goto LABEL_325;
LABEL_383:
            *v194 = *(_QWORD *)(v77 + 112);
            *(_QWORD *)(v77 + 112) = v194;
            if ( *v194 )
              *(_QWORD *)(*(_QWORD *)(v77 + 96) + 8LL * (*(_QWORD *)(*v194 + 8LL) % *(_QWORD *)(v77 + 104))) = v194;
            *(_QWORD *)(*(_QWORD *)(v77 + 96) + v199) = v279;
            goto LABEL_326;
          }
          if ( v197 == 1 )
          {
            v221 = (char *)(v77 + 144);
            *(_QWORD *)(v77 + 144) = 0;
            v222 = v77 + 144;
          }
          else
          {
            if ( v197 > 0xFFFFFFFFFFFFFFFLL )
LABEL_478:
              sub_4261EA(v195, v115, v197);
            nc = 8 * v197;
            v221 = (char *)sub_22077B0(8 * v197);
            memset(v221, 0, nc);
            v222 = v77 + 144;
          }
          v223 = *(_QWORD **)(v77 + 112);
          *(_QWORD *)(v77 + 112) = 0;
          if ( !v223 )
          {
LABEL_380:
            v228 = *(_QWORD *)(v77 + 96);
            if ( v228 != v222 )
              j_j___libc_free_0(v228);
            *(_QWORD *)(v77 + 104) = v198;
            *(_QWORD *)(v77 + 96) = v221;
            v199 = 8 * (srca % v198);
            v200 = *(_QWORD **)&v221[v199];
            if ( !v200 )
              goto LABEL_383;
LABEL_325:
            *v194 = *v200;
            **(_QWORD **)(*(_QWORD *)(v77 + 96) + v199) = v194;
LABEL_326:
            ++*(_QWORD *)(v77 + 120);
            goto LABEL_162;
          }
          v224 = 0;
          while ( 1 )
          {
            v225 = v223;
            v223 = (_QWORD *)*v223;
            v226 = v225[1] % v198;
            v227 = &v221[8 * v226];
            if ( *(_QWORD *)v227 )
              break;
            *v225 = *(_QWORD *)(v77 + 112);
            *(_QWORD *)(v77 + 112) = v225;
            *(_QWORD *)v227 = v279;
            if ( !*v225 )
            {
              v224 = v226;
LABEL_376:
              if ( !v223 )
                goto LABEL_380;
              continue;
            }
            *(_QWORD *)&v221[8 * v224] = v225;
            v224 = v226;
            if ( !v223 )
              goto LABEL_380;
          }
          *v225 = **(_QWORD **)v227;
          **(_QWORD **)v227 = v225;
          goto LABEL_376;
        }
LABEL_161:
        if ( !*v107 )
          goto LABEL_321;
LABEL_162:
        ++v316;
      }
      v110 = *(_QWORD *)(v77 + 56);
      v111 = *(_QWORD *)(v77 + 48);
      *(_BYTE *)(v77 + 208) = 1;
      v317 = v110;
      v306 = (_QWORD *)(v77 + 168);
      if ( v111 != v110 )
        break;
LABEL_174:
      v278 = sub_220EEE0(v278);
      if ( (size_t *)v278 == &v374 )
        goto LABEL_175;
    }
    while ( 1 )
    {
      v112 = *(_QWORD *)(v111 + 48);
      v113 = (_QWORD *)sub_22077B0(0x18u);
      v114 = (unsigned __int64)v113;
      if ( v113 )
        *v113 = 0;
      v113[1] = v112;
      v113[2] = v111;
      v115 = *(_QWORD *)(v77 + 160);
      v116 = *(_QWORD **)(*(_QWORD *)(v77 + 152) + 8 * (v112 % v115));
      v117 = 8 * (v112 % v115);
      if ( !v116 )
        goto LABEL_327;
      v118 = (_QWORD *)*v116;
      if ( v112 != *(_QWORD *)(*v116 + 8LL) )
        break;
LABEL_171:
      if ( !*v116 )
        goto LABEL_327;
      j_j___libc_free_0(v114);
LABEL_173:
      v111 += 56;
      if ( v317 == v111 )
        goto LABEL_174;
    }
    while ( 1 )
    {
      v119 = (_QWORD *)*v118;
      if ( !*v118 )
        break;
      v116 = v118;
      if ( v112 % v115 != v119[1] % v115 )
        break;
      v118 = (_QWORD *)*v118;
      if ( v112 == v119[1] )
        goto LABEL_171;
    }
LABEL_327:
    v195 = v77 + 184;
    v201 = sub_222DA10(v77 + 184, v115, *(_QWORD *)(v77 + 176), 1);
    v202 = v197;
    if ( !v201 )
    {
      v203 = *(_QWORD **)(*(_QWORD *)(v77 + 152) + v117);
      if ( v203 )
      {
LABEL_329:
        *(_QWORD *)v114 = *v203;
        **(_QWORD **)(*(_QWORD *)(v77 + 152) + v117) = v114;
LABEL_330:
        ++*(_QWORD *)(v77 + 176);
        goto LABEL_173;
      }
LABEL_399:
      *(_QWORD *)v114 = *(_QWORD *)(v77 + 168);
      *(_QWORD *)(v77 + 168) = v114;
      if ( *(_QWORD *)v114 )
        *(_QWORD *)(*(_QWORD *)(v77 + 152) + 8LL * (*(_QWORD *)(*(_QWORD *)v114 + 8LL) % *(_QWORD *)(v77 + 160))) = v114;
      *(_QWORD *)(*(_QWORD *)(v77 + 152) + v117) = v306;
      goto LABEL_330;
    }
    if ( v197 == 1 )
    {
      v233 = (_QWORD *)(v77 + 200);
      *(_QWORD *)(v77 + 200) = 0;
      v232 = v77 + 200;
    }
    else
    {
      if ( v197 > 0xFFFFFFFFFFFFFFFLL )
        goto LABEL_478;
      v229 = 8 * v197;
      v302 = v197;
      v230 = (void *)sub_22077B0(8 * v197);
      v231 = memset(v230, 0, v229);
      v202 = v302;
      v232 = v77 + 200;
      v233 = v231;
    }
    v234 = *(_QWORD **)(v77 + 168);
    *(_QWORD *)(v77 + 168) = 0;
    if ( !v234 )
    {
LABEL_396:
      v239 = *(_QWORD *)(v77 + 152);
      if ( v239 != v232 )
      {
        nd = v202;
        v303 = v233;
        j_j___libc_free_0(v239);
        v202 = nd;
        v233 = v303;
      }
      *(_QWORD *)(v77 + 160) = v202;
      *(_QWORD *)(v77 + 152) = v233;
      v117 = 8 * (v112 % v202);
      v203 = (_QWORD *)v233[(unsigned __int64)v117 / 8];
      if ( v203 )
        goto LABEL_329;
      goto LABEL_399;
    }
    v235 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v236 = v234;
        v234 = (_QWORD *)*v234;
        v237 = v236[1] % v202;
        v238 = (_QWORD **)&v233[v237];
        if ( !*v238 )
          break;
        *v236 = **v238;
        **v238 = v236;
LABEL_392:
        if ( !v234 )
          goto LABEL_396;
      }
      *v236 = *(_QWORD *)(v77 + 168);
      *(_QWORD *)(v77 + 168) = v236;
      *v238 = v306;
      if ( !*v236 )
      {
        v235 = v237;
        goto LABEL_392;
      }
      v233[v235] = v236;
      v235 = v237;
      if ( !v234 )
        goto LABEL_396;
    }
  }
LABEL_175:
  sub_23AF340(v375);
  v321 = 0;
  v322 = sub_2241E40();
  sub_CB7040((__int64)&v373, v330, v331, (__int64)&v321);
  if ( v321 )
  {
    v264 = sub_CB72A0();
    v265 = sub_904010((__int64)v264, "Error: ");
    (*((void (__fastcall **)(__m128i **, __int64 (__fastcall **)(), _QWORD))*v322 + 4))(&v353, v322, v321);
    v266 = sub_CB6200(v265, (unsigned __int8 *)v353, v354);
    v267 = "\n";
    sub_904010(v266, "\n");
    if ( v353 != &v355 )
    {
      v267 = (char *)(v355.m128i_i64[0] + 1);
      j_j___libc_free_0((unsigned __int64)v353);
    }
    sub_CB5B00((int *)&v373, (__int64)v267);
    goto LABEL_255;
  }
  v350 = 257;
  v320 = &v359;
  v325 = 1;
  v323 = (char **)&v373;
  v324 = &v320;
  sub_CA0F50((__int64 *)&v333, (void **)&v347);
  v120 = *v324;
  v345.m128i_i64[0] = (__int64)v346;
  sub_23AEDD0(v345.m128i_i64, v120[1], (__int64)v120[2] + (_QWORD)v120[1]);
  if ( v334 )
  {
    v121 = (__m128i *)&v333;
    v122 = sub_904010((__int64)v323, "digraph \"");
    goto LABEL_179;
  }
  if ( v345.m128i_i64[1] )
  {
    v121 = &v345;
    v122 = sub_904010((__int64)v323, "digraph \"");
LABEL_179:
    sub_C67200((__int64 *)&v353, (__int64)v121);
    v123 = sub_CB6200(v122, (unsigned __int8 *)v353, v354);
    sub_904010(v123, "\" {\n");
    sub_2240A30((unsigned __int64 *)&v353);
    goto LABEL_180;
  }
  sub_904010((__int64)v323, "digraph unnamed {\n");
LABEL_180:
  v124 = (__int64)v323;
  if ( v334 )
  {
    v268 = sub_904010((__int64)v323, "\tlabel=\"");
    sub_C67200((__int64 *)&v353, (__int64)&v333);
    v269 = v354;
    v270 = (unsigned __int8 *)v353;
    v271 = v268;
LABEL_466:
    v272 = sub_CB6200(v271, v270, v269);
    sub_904010(v272, "\";\n");
    sub_2240A30((unsigned __int64 *)&v353);
    v124 = (__int64)v323;
    goto LABEL_182;
  }
  if ( v345.m128i_i64[1] )
  {
    v273 = sub_904010((__int64)v323, "\tlabel=\"");
    sub_C67200((__int64 *)&v353, (__int64)&v345);
    v269 = v354;
    v270 = (unsigned __int8 *)v353;
    v271 = v273;
    goto LABEL_466;
  }
LABEL_182:
  v353 = &v355;
  sub_23AE760((__int64 *)&v353, "\tsize=\"190, 190\";\n", (__int64)"");
  sub_CB6200(v124, (unsigned __int8 *)v353, v354);
  if ( v353 != &v355 )
    j_j___libc_free_0((unsigned __int64)v353);
  sub_904010((__int64)v323, "\n");
  if ( (_QWORD *)v345.m128i_i64[0] != v346 )
    j_j___libc_free_0(v345.m128i_u64[0]);
  v289 = (__int64 **)(*v324)[9];
  if ( v289 != (*v324)[8] )
  {
    for ( k = (__int64 **)(*v324)[8]; v289 != k; ++k )
    {
      v125 = *k;
      v126 = (*k)[1];
      v345.m128i_i64[0] = **k;
      v345.m128i_i64[1] = v126;
      sub_95CA80((__int64 *)&v353, (__int64)&v345);
      v127 = sub_2241130((unsigned __int64 *)&v353, 0, 0, "color=", 6u);
      v336 = &v338;
      if ( (unsigned __int64 *)*v127 == v127 + 2 )
      {
        v338 = _mm_loadu_si128((const __m128i *)v127 + 1);
      }
      else
      {
        v336 = (__m128i *)*v127;
        v338.m128i_i64[0] = v127[2];
      }
      v128 = v127[1];
      *((_BYTE *)v127 + 16) = 0;
      v337 = v128;
      *v127 = (unsigned __int64)(v127 + 2);
      v127[1] = 0;
      if ( v353 != &v355 )
        j_j___libc_free_0((unsigned __int64)v353);
      v129 = sub_904010((__int64)v323, "\tNode");
      v130 = sub_CB5A80(v129, (unsigned __int64)v125);
      sub_904010(v130, " [shape=");
      if ( (_BYTE)v325 )
      {
        sub_904010((__int64)v323, "none,");
        v131 = v337;
        if ( !v337 )
          goto LABEL_194;
      }
      else
      {
        sub_904010((__int64)v323, "record,");
        v131 = v337;
        if ( !v337 )
          goto LABEL_194;
      }
      v258 = sub_CB6200((__int64)v323, (unsigned __int8 *)v336, v131);
      sub_904010(v258, ",");
LABEL_194:
      sub_904010((__int64)v323, "label=");
      if ( (_BYTE)v325 )
      {
        v132 = (_QWORD *)v125[14];
        v133 = 0;
        v134 = 1;
        if ( v132 )
        {
          do
          {
            v132 = (_QWORD *)*v132;
            ++v133;
            if ( !v132 )
            {
              v134 = v133;
              goto LABEL_200;
            }
          }
          while ( v133 != 64 );
          v134 = 65;
        }
LABEL_200:
        v135 = sub_904010((__int64)v323, "<<table border=\"0\" cellborder=\"1\" cellspacing=\"0\"");
        v136 = sub_904010(v135, " cellpadding=\"0\"><tr><td align=\"text\" colspan=\"");
        v137 = sub_CB59D0(v136, v134);
        sub_904010(v137, "\">");
      }
      else
      {
        sub_904010((__int64)v323, "\"{");
      }
      v138 = (__int64)v323;
      v139 = (__int64)(v125 + 2);
      if ( (_BYTE)v325 )
      {
        sub_2241BD0((__int64 *)&v353, v139);
        v140 = sub_CB6200(v138, (unsigned __int8 *)v353, v354);
        sub_904010(v140, "</td>");
        sub_2240A30((unsigned __int64 *)&v353);
      }
      else
      {
        sub_2241BD0(v345.m128i_i64, v139);
        sub_C67200((__int64 *)&v353, (__int64)&v345);
        sub_CB6200(v138, (unsigned __int8 *)v353, v354);
        sub_2240A30((unsigned __int64 *)&v353);
        sub_2240A30((unsigned __int64 *)&v345);
      }
      sub_23B0820((__int64 *)&v342, byte_3F871B3);
      if ( v343 )
      {
        v259 = sub_904010((__int64)v323, "|");
        sub_C67200((__int64 *)&v353, (__int64)&v342);
        sub_CB6200(v259, (unsigned __int8 *)v353, v354);
        sub_2240A30((unsigned __int64 *)&v353);
      }
      sub_23B0820(v345.m128i_i64, byte_3F871B3);
      if ( v345.m128i_i64[1] )
      {
        v260 = sub_904010((__int64)v323, "|");
        sub_C67200((__int64 *)&v353, (__int64)&v345);
        sub_CB6200(v260, (unsigned __int8 *)v353, v354);
        sub_2240A30((unsigned __int64 *)&v353);
      }
      if ( (_QWORD *)v345.m128i_i64[0] != v346 )
        j_j___libc_free_0(v345.m128i_u64[0]);
      if ( v342 != &v344 )
        j_j___libc_free_0((unsigned __int64)v342);
      v340 = 0;
      v339 = v341;
      v357 = 0x100000000LL;
      v341[0] = 0;
      v354 = 0;
      v353 = (__m128i *)&unk_49DD210;
      v355 = 0u;
      v356 = 0;
      v358 = &v339;
      sub_CB5980((__int64)&v353, 0, 0, 0);
      v141 = (_QWORD *)v125[14];
      if ( !(_BYTE)v325 )
      {
        if ( !v141 )
          goto LABEL_436;
LABEL_213:
        v142 = 0;
        v143 = 0;
        while ( 2 )
        {
          v144 = v141[1];
          v145 = v125[20];
          v146 = *(_QWORD **)(v125[19] + 8 * (v144 % v145));
          if ( !v146 )
            goto LABEL_480;
          v147 = (_QWORD *)*v146;
          if ( *(_QWORD *)(*v146 + 8LL) != v144 )
          {
            while ( 1 )
            {
              v148 = (_QWORD *)*v147;
              if ( !*v147 )
                break;
              v146 = v147;
              if ( v141[1] % v145 != v148[1] % v145 )
                break;
              v147 = (_QWORD *)*v147;
              if ( v148[1] == v144 )
                goto LABEL_219;
            }
LABEL_480:
            BUG();
          }
LABEL_219:
          if ( !*v146 )
            goto LABEL_480;
          v149 = *(_QWORD *)(*v146 + 16LL);
          v342 = &v344;
          sub_23AEDD0((__int64 *)&v342, *(_BYTE **)(v149 + 16), *(_QWORD *)(v149 + 16) + *(_QWORD *)(v149 + 24));
          if ( v343 )
          {
            v150 = v355.m128i_i64[1];
            v151 = v356;
            if ( (_BYTE)v325 )
            {
              if ( v355.m128i_i64[1] - (__int64)v356 <= 0x16uLL )
              {
                v250 = (_BYTE **)sub_CB6200((__int64)&v353, "<td colspan=\"1\" port=\"s", 0x17u);
              }
              else
              {
                si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CB30);
                v356[22] = 115;
                *((_DWORD *)v151 + 4) = 1953656688;
                v250 = (_BYTE **)&v353;
                *((_WORD *)v151 + 10) = 8765;
                *(__m128i *)v151 = si128;
                v356 += 23;
              }
              v251 = sub_CB59D0((__int64)v250, v142);
              v252 = *(_WORD **)(v251 + 32);
              v253 = v251;
              if ( *(_QWORD *)(v251 + 24) - (_QWORD)v252 <= 1u )
              {
                v253 = sub_CB6200(v251, "\">", 2u);
              }
              else
              {
                *v252 = 15906;
                *(_QWORD *)(v251 + 32) += 2LL;
              }
              v254 = sub_CB6200(v253, (unsigned __int8 *)v342, v343);
              v255 = *(_QWORD *)(v254 + 32);
              if ( (unsigned __int64)(*(_QWORD *)(v254 + 24) - v255) <= 4 )
              {
                sub_CB6200(v254, "</td>", 5u);
              }
              else
              {
                *(_DWORD *)v255 = 1685335868;
                *(_BYTE *)(v255 + 4) = 62;
                *(_QWORD *)(v254 + 32) += 5LL;
              }
            }
            else
            {
              if ( !v142 )
                goto LABEL_223;
              if ( v356 == (_BYTE *)v355.m128i_i64[1] )
              {
                sub_CB6200((__int64)&v353, (unsigned __int8 *)"|", 1u);
                v151 = v356;
                v150 = v355.m128i_i64[1];
LABEL_223:
                if ( (unsigned __int64)(v150 - (_QWORD)v151) > 1 )
                  goto LABEL_224;
LABEL_439:
                v152 = (_BYTE **)sub_CB6200((__int64)&v353, "<s", 2u);
              }
              else
              {
                *v356 = 124;
                v151 = v356 + 1;
                v356 = v151;
                if ( v355.m128i_i64[1] - (__int64)v151 <= 1uLL )
                  goto LABEL_439;
LABEL_224:
                v152 = (_BYTE **)&v353;
                *(_WORD *)v151 = 29500;
                v356 += 2;
              }
              v153 = sub_CB59D0((__int64)v152, v142);
              v154 = *(_BYTE **)(v153 + 32);
              if ( *(_BYTE **)(v153 + 24) == v154 )
              {
                v153 = sub_CB6200(v153, (unsigned __int8 *)">", 1u);
              }
              else
              {
                *v154 = 62;
                ++*(_QWORD *)(v153 + 32);
              }
              sub_C67200(v345.m128i_i64, (__int64)&v342);
              sub_CB6200(v153, (unsigned __int8 *)v345.m128i_i64[0], v345.m128i_u64[1]);
              if ( (_QWORD *)v345.m128i_i64[0] != v346 )
                j_j___libc_free_0(v345.m128i_u64[0]);
            }
            if ( v342 != &v344 )
              j_j___libc_free_0((unsigned __int64)v342);
            v143 = 1;
LABEL_232:
            v141 = (_QWORD *)*v141;
            if ( !v141 )
              goto LABEL_432;
          }
          else
          {
            if ( v342 == &v344 )
              goto LABEL_232;
            j_j___libc_free_0((unsigned __int64)v342);
            v141 = (_QWORD *)*v141;
            if ( !v141 )
            {
LABEL_432:
              if ( v143 )
              {
                if ( !(_BYTE)v325 )
                  goto LABEL_434;
                goto LABEL_238;
              }
              goto LABEL_239;
            }
          }
          if ( ++v142 == 64 )
          {
            if ( !v143 )
              goto LABEL_239;
            if ( (_BYTE)v325 )
              sub_904010((__int64)&v353, "<td colspan=\"1\" port=\"s64\">truncated...</td>");
            else
              sub_904010((__int64)&v353, "|<s64>truncated...");
            if ( (_BYTE)v325 )
              goto LABEL_238;
LABEL_434:
            sub_904010((__int64)v323, "|");
            if ( (_BYTE)v325 )
            {
LABEL_238:
              sub_CB6200((__int64)v323, v339, v340);
              goto LABEL_239;
            }
            v256 = sub_904010((__int64)v323, "{");
            v257 = sub_CB6200(v256, v339, v340);
            sub_904010(v257, "}");
            if ( (_BYTE)v325 )
              goto LABEL_240;
LABEL_436:
            sub_904010((__int64)v323, "}\"");
            goto LABEL_241;
          }
          continue;
        }
      }
      sub_904010((__int64)&v353, "</tr><tr>");
      if ( v141 )
        goto LABEL_213;
LABEL_239:
      if ( !(_BYTE)v325 )
        goto LABEL_436;
LABEL_240:
      sub_904010((__int64)v323, "</tr></table>>");
LABEL_241:
      v155 = 0;
      sub_904010((__int64)v323, "];\n");
      v156 = (_QWORD *)v125[14];
      if ( v156 )
      {
        while ( 1 )
        {
          sub_23B2260((__int64 *)&v323, (unsigned __int64)v125, v155, (__int64)v156);
          v156 = (_QWORD *)*v156;
          ++v155;
          if ( !v156 )
            break;
          if ( v155 == 64 )
          {
            do
            {
              sub_23B2260((__int64 *)&v323, (unsigned __int64)v125, 64, (__int64)v156);
              v156 = (_QWORD *)*v156;
            }
            while ( v156 );
            break;
          }
        }
      }
      v353 = (__m128i *)&unk_49DD210;
      sub_CB5840((__int64)&v353);
      if ( v339 != v341 )
        j_j___libc_free_0((unsigned __int64)v339);
      if ( v336 != &v338 )
        j_j___libc_free_0((unsigned __int64)v336);
    }
  }
  v157 = "}\n";
  sub_904010((__int64)v323, "}\n");
  if ( v333 != &v335 )
  {
    v157 = (const char *)(v335 + 1);
    j_j___libc_free_0((unsigned __int64)v333);
  }
  if ( v376.m128i_i64[1] != v375 )
    sub_CB5AE0((__int64 *)&v373);
  sub_CB7080((__int64)&v373, (__int64)v157);
  sub_CB5B00((int *)&v373, (__int64)v157);
LABEL_255:
  v159 = *(_QWORD *)(a1 + 40);
  sub_23B36B0((__int64 *)&v373, (__int64)v385, v386, (__int64)v330, v331, v158, v351[0]);
  sub_CB6200(v159, (unsigned __int8 *)v373, v374);
  sub_2240A30((unsigned __int64 *)&v373);
  v376.m128i_i16[4] = 260;
  v373 = (__m128i *)&v330;
  v160 = sub_C823F0((__int64)&v373, 1);
  v162 = v161;
  v163 = v160;
  if ( v160 )
  {
    v261 = sub_CB72A0();
    v262 = sub_904010((__int64)v261, "Error: ");
    (*(void (__fastcall **)(__m128i **, __int64, _QWORD))(*(_QWORD *)v162 + 32LL))(&v373, v162, v163);
    v263 = sub_CB6200(v262, (unsigned __int8 *)v373, v374);
    sub_904010(v263, "\n");
    sub_2240A30((unsigned __int64 *)&v373);
  }
  if ( v365 )
    j_j___libc_free_0((unsigned __int64)v365);
  v164 = v363;
  v165 = v362;
  if ( v363 != (void **)v362 )
  {
    do
    {
      v166 = *(_QWORD **)(v165 + 168);
      while ( v166 )
      {
        v167 = (unsigned __int64)v166;
        v166 = (_QWORD *)*v166;
        j_j___libc_free_0(v167);
      }
      memset(*(void **)(v165 + 152), 0, 8LL * *(_QWORD *)(v165 + 160));
      v168 = *(_QWORD *)(v165 + 152);
      *(_QWORD *)(v165 + 176) = 0;
      *(_QWORD *)(v165 + 168) = 0;
      if ( v168 != v165 + 200 )
        j_j___libc_free_0(v168);
      v169 = *(_QWORD **)(v165 + 112);
      while ( v169 )
      {
        v170 = (unsigned __int64)v169;
        v169 = (_QWORD *)*v169;
        j_j___libc_free_0(v170);
      }
      memset(*(void **)(v165 + 96), 0, 8LL * *(_QWORD *)(v165 + 104));
      v171 = *(_QWORD *)(v165 + 96);
      *(_QWORD *)(v165 + 120) = 0;
      *(_QWORD *)(v165 + 112) = 0;
      if ( v171 != v165 + 144 )
        j_j___libc_free_0(v171);
      v172 = *(_QWORD *)(v165 + 72);
      if ( v172 )
        j_j___libc_free_0(v172);
      v173 = *(_QWORD *)(v165 + 56);
      v174 = *(_QWORD *)(v165 + 48);
      if ( v173 != v174 )
      {
        do
        {
          v175 = *(_QWORD *)(v174 + 16);
          if ( v175 != v174 + 32 )
            j_j___libc_free_0(v175);
          v174 += 56LL;
        }
        while ( v173 != v174 );
        v174 = *(_QWORD *)(v165 + 48);
      }
      if ( v174 )
        j_j___libc_free_0(v174);
      v176 = *(_QWORD *)(v165 + 16);
      if ( v176 != v165 + 32 )
        j_j___libc_free_0(v176);
      v165 += 216LL;
    }
    while ( v164 != (void **)v165 );
    v165 = v362;
  }
  if ( v165 )
    j_j___libc_free_0(v165);
  sub_2240A30((unsigned __int64 *)v360);
  sub_2240A30((unsigned __int64 *)v332);
  if ( HIDWORD(v402) )
  {
    v177 = (unsigned __int64)v401;
    if ( (_DWORD)v402 )
    {
      v178 = 8LL * (unsigned int)v402;
      v179 = 0;
      do
      {
        v180 = *(_QWORD **)(v177 + v179);
        if ( v180 && v180 != (_QWORD *)-8LL )
        {
          v181 = *v180 + 41LL;
          sub_2240A30(v180 + 1);
          sub_C7D6A0((__int64)v180, v181, 8);
          v177 = (unsigned __int64)v401;
        }
        v179 += 8;
      }
      while ( v178 != v179 );
    }
  }
  else
  {
    v177 = (unsigned __int64)v401;
  }
  _libc_free(v177);
  sub_2240A30((unsigned __int64 *)&v398);
  if ( HIDWORD(v396) )
  {
    v182 = v395;
    if ( (_DWORD)v396 )
    {
      v183 = 8LL * (unsigned int)v396;
      v184 = 0;
      do
      {
        v185 = *(_QWORD **)(v182 + v184);
        if ( v185 && v185 != (_QWORD *)-8LL )
        {
          sub_C7D6A0((__int64)v185, *v185 + 17LL, 8);
          v182 = v395;
        }
        v184 += 8;
      }
      while ( v183 != v184 );
    }
  }
  else
  {
    v182 = v395;
  }
  _libc_free(v182);
  v186 = (char *)v393;
  v187 = v392;
  if ( (char *)v393 != v392 )
  {
    do
    {
      v188 = *((_QWORD *)v187 + 15);
      if ( v188 )
        j_j___libc_free_0(v188);
      v189 = *((_QWORD *)v187 + 12);
      if ( v189 )
        j_j___libc_free_0(v189);
      v190 = *((_QWORD *)v187 + 8);
      while ( v190 )
      {
        v191 = v190;
        sub_23AFE30(*(_QWORD **)(v190 + 24));
        v192 = *(_QWORD *)(v190 + 40);
        v190 = *(_QWORD *)(v190 + 16);
        if ( v192 != v191 + 56 )
          j_j___libc_free_0(v192);
        j_j___libc_free_0(v191);
      }
      v187 += 144;
    }
    while ( v186 != v187 );
    v187 = v392;
  }
  if ( v187 )
    j_j___libc_free_0((unsigned __int64)v187);
  if ( v385 != (unsigned __int8 *)&v388 )
    _libc_free((unsigned __int64)v385);
  if ( (_BYTE *)v351[0] != v352 )
    _libc_free(v351[0]);
  sub_2240A30((unsigned __int64 *)&v330);
  if ( v369 != v372 )
    _libc_free((unsigned __int64)v369);
  if ( (char *)v328[0] != &v329 )
    _libc_free(v328[0]);
  if ( (char *)v326[0] != &v327 )
    _libc_free(v326[0]);
}
