// Function: sub_2EFFE00
// Address: 0x2effe00
//
__int64 __fastcall sub_2EFFE00(__int64 a1)
{
  __int64 v1; // r14
  _BYTE *v2; // rbx
  __int64 v3; // rax
  __m128i v4; // xmm1
  void (__fastcall *v5)(__m128i *, __m128i *, __int64); // rax
  __m128i v6; // xmm0
  __int64 v7; // r12
  __int64 i; // r14
  unsigned __int64 *v9; // rbx
  unsigned __int64 *v10; // r12
  unsigned __int64 v11; // r13
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  _BYTE *v16; // rsi
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  int v29; // ebx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  unsigned int v36; // r12d
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  unsigned __int64 v40; // rdi
  unsigned __int64 v41; // rdi
  unsigned __int64 v42; // rdi
  unsigned __int64 v43; // rdi
  __m128i *v44; // r12
  __int64 v45; // rax
  _QWORD *v46; // r15
  __int64 *v47; // rcx
  __int64 *v48; // r14
  _QWORD *v49; // rax
  int v50; // eax
  __int64 v51; // rdx
  unsigned __int64 v52; // rax
  unsigned __int64 v53; // rax
  unsigned int v54; // esi
  int *v55; // rbx
  unsigned __int64 **v56; // r14
  __int64 v57; // r8
  unsigned int v58; // edx
  _DWORD *v59; // rdi
  int v60; // ecx
  unsigned int v61; // esi
  int v62; // r12d
  int v63; // r12d
  __int64 v64; // r9
  unsigned int v65; // edx
  _DWORD *v66; // r10
  int v67; // edi
  int v68; // eax
  int v69; // eax
  _BYTE *v70; // r15
  __m128i *v71; // rsi
  _QWORD *v72; // rax
  char v73; // dl
  __int64 v74; // rbx
  unsigned int v75; // eax
  __int64 v76; // r12
  char v77; // al
  unsigned int v78; // r12d
  __int64 v79; // rax
  __int64 v80; // r13
  __int64 v81; // rcx
  __int64 v82; // rdx
  __int64 v83; // r15
  bool v84; // al
  __int64 v85; // r8
  __int64 v86; // r9
  _QWORD *v87; // rdx
  __int64 v88; // rcx
  _QWORD *v89; // rax
  _QWORD *v90; // rax
  int v91; // edi
  int v92; // edx
  __int64 v93; // r10
  int v94; // edi
  unsigned int v95; // ecx
  int v96; // r8d
  int v97; // esi
  int v98; // esi
  __int64 v99; // rdi
  unsigned int v100; // eax
  int v101; // r10d
  int v102; // ecx
  unsigned int v103; // ecx
  __int64 v104; // rdx
  __int64 *v105; // rcx
  __int64 *v106; // r15
  __int64 v107; // rbx
  _QWORD *v108; // rax
  __int64 v109; // rdx
  __int64 v110; // rsi
  __m128i *v111; // rdi
  __int64 v112; // r13
  __int64 v113; // rdx
  __m128i *v114; // rdx
  __m128i si128; // xmm0
  int v116; // r11d
  int v117; // eax
  int v118; // r12d
  int v119; // r12d
  __int64 v120; // r9
  int v121; // esi
  _DWORD *v122; // rcx
  unsigned int v123; // edx
  int v124; // r8d
  __int64 v125; // r8
  __int64 v126; // r9
  __int64 v127; // rax
  __int64 v128; // rcx
  _QWORD *v129; // rax
  unsigned int *v130; // rbx
  unsigned int *v131; // r15
  unsigned int v132; // r13d
  __int64 v133; // rcx
  int v134; // eax
  int v135; // edx
  unsigned int v136; // eax
  int v137; // esi
  __int64 v138; // r14
  __m128i *v139; // rdx
  __m128i v140; // xmm0
  __m128i *v141; // rdx
  __m128i v142; // xmm0
  int v143; // edi
  _QWORD *v144; // rax
  __m128i *v145; // rdi
  __m128i *v146; // rdx
  __int64 *v147; // r12
  __int64 v148; // rax
  unsigned __int64 v149; // r13
  _QWORD *v150; // rax
  int v151; // eax
  int v152; // eax
  unsigned int v153; // esi
  __int64 *v154; // rdi
  __int64 *v155; // rax
  unsigned int v156; // esi
  unsigned int v157; // edx
  __int64 v158; // r11
  __m128i *v159; // rdx
  __m128i v160; // xmm0
  __m128i v161; // xmm0
  __int64 v162; // rsi
  __int64 v163; // rax
  __int64 v164; // rax
  __int64 v165; // rax
  __int64 v166; // rbx
  int v167; // r13d
  __int64 v168; // rax
  __int64 v169; // rdx
  __int64 v170; // r12
  __int64 v171; // r15
  __int64 v172; // r13
  __int128 v173; // rax
  __int64 v174; // rbx
  __int64 v175; // r12
  __int64 v176; // rsi
  __int64 v177; // rcx
  __int64 v178; // r13
  unsigned int *v179; // rbx
  unsigned int *v180; // r14
  unsigned int v181; // r12d
  __int64 v182; // rdx
  __int64 v183; // rax
  __int64 *v184; // rbx
  int v185; // eax
  unsigned int v186; // esi
  int v187; // ecx
  _QWORD *v188; // rax
  __int64 v189; // rdx
  int v190; // eax
  __int64 v191; // r14
  char *v192; // rsi
  size_t v193; // rax
  __m128i *v194; // rdi
  size_t v195; // r12
  unsigned __int64 v196; // rax
  __int64 v197; // rax
  __m128i v198; // xmm0
  _BYTE *v199; // rax
  __int64 result; // rax
  __int64 v201; // rax
  __int64 v202; // r15
  unsigned int v203; // esi
  __int64 v204; // rdi
  __int64 v205; // rbx
  __int64 v206; // r13
  int v207; // r12d
  __int64 v208; // rsi
  int v209; // r10d
  __int64 v210; // r8
  _DWORD *v211; // rdx
  _DWORD *v212; // rax
  int v213; // ecx
  int v214; // edi
  __int64 *v215; // rdi
  __int64 *v216; // rax
  unsigned int v217; // esi
  __int64 v218; // r11
  __m128i *v219; // rdx
  __m128i v220; // xmm0
  __m128i v221; // xmm0
  __m128i v222; // xmm0
  int v223; // edi
  __int64 v224; // r8
  void *v225; // rdx
  _BYTE *v226; // rax
  __int64 v227; // r8
  void *v228; // rdx
  _BYTE *v229; // rax
  __int64 v230; // r13
  __m128i *v231; // rdx
  __m128i v232; // xmm0
  __int64 v233; // r8
  void *v234; // rax
  _BYTE *v235; // rax
  _BYTE *v236; // rax
  unsigned int v237; // r13d
  int v238; // edx
  unsigned __int64 v239; // r15
  __int64 v240; // rax
  _QWORD *v241; // rdx
  unsigned int **v242; // r12
  unsigned int **v243; // rbx
  unsigned int *v244; // r13
  __int64 v245; // rdi
  _BYTE *v246; // rax
  __int64 v247; // rdi
  _BYTE *v248; // rax
  unsigned int v249; // ecx
  int v250; // r11d
  int v251; // edi
  _DWORD *v252; // rsi
  int v253; // esi
  unsigned int v254; // r14d
  _DWORD *v255; // rcx
  int v256; // edi
  _QWORD *v257; // rax
  _QWORD *v258; // r13
  _QWORD *v259; // rbx
  _BYTE *v260; // rdi
  unsigned int v261; // r15d
  int v262; // eax
  size_t v263; // rdx
  _BYTE *v264; // rdi
  __int64 v265; // rsi
  __m128i *v266; // rdx
  const __m128i *v267; // rax
  unsigned __int64 v268; // rcx
  unsigned int *v269; // rbx
  unsigned int *v270; // r13
  unsigned int v271; // r15d
  __int64 v272; // rsi
  __m128i *v273; // rdx
  const __m128i *v274; // rax
  const __m128i *v275; // rcx
  int v276; // esi
  int v277; // r11d
  __int64 v278; // rsi
  int v279; // r11d
  __int64 v280; // rsi
  int v281; // r9d
  __int64 v282; // [rsp+0h] [rbp-820h]
  _QWORD *v283; // [rsp+18h] [rbp-808h]
  __int64 v284; // [rsp+18h] [rbp-808h]
  unsigned __int64 v285; // [rsp+20h] [rbp-800h]
  unsigned int *v286; // [rsp+20h] [rbp-800h]
  void *src; // [rsp+28h] [rbp-7F8h]
  void *srca; // [rsp+28h] [rbp-7F8h]
  __int32 srcb; // [rsp+28h] [rbp-7F8h]
  _BYTE *v290; // [rsp+30h] [rbp-7F0h]
  __int64 v291; // [rsp+30h] [rbp-7F0h]
  __int64 v292; // [rsp+38h] [rbp-7E8h]
  __int64 v293; // [rsp+38h] [rbp-7E8h]
  __int64 v294; // [rsp+38h] [rbp-7E8h]
  unsigned __int64 v295; // [rsp+38h] [rbp-7E8h]
  __int64 v296; // [rsp+38h] [rbp-7E8h]
  _BYTE *v297; // [rsp+48h] [rbp-7D8h]
  __int64 v298; // [rsp+48h] [rbp-7D8h]
  int v299; // [rsp+48h] [rbp-7D8h]
  int v300; // [rsp+48h] [rbp-7D8h]
  __int64 v301; // [rsp+48h] [rbp-7D8h]
  __int64 v302; // [rsp+48h] [rbp-7D8h]
  __int64 v303; // [rsp+48h] [rbp-7D8h]
  __int32 v304; // [rsp+50h] [rbp-7D0h]
  unsigned int v305; // [rsp+50h] [rbp-7D0h]
  _QWORD *v306; // [rsp+58h] [rbp-7C8h]
  _BYTE *v307; // [rsp+58h] [rbp-7C8h]
  __int64 v308; // [rsp+58h] [rbp-7C8h]
  unsigned int v309; // [rsp+58h] [rbp-7C8h]
  __int128 v310; // [rsp+58h] [rbp-7C8h]
  unsigned int *v311; // [rsp+58h] [rbp-7C8h]
  __int64 v312; // [rsp+58h] [rbp-7C8h]
  _QWORD *v313; // [rsp+60h] [rbp-7C0h]
  __m128i *v314; // [rsp+60h] [rbp-7C0h]
  __int64 v315; // [rsp+60h] [rbp-7C0h]
  __int64 v316; // [rsp+60h] [rbp-7C0h]
  _QWORD *v317; // [rsp+60h] [rbp-7C0h]
  __int64 v318; // [rsp+60h] [rbp-7C0h]
  __int64 v319; // [rsp+60h] [rbp-7C0h]
  __int64 v320; // [rsp+60h] [rbp-7C0h]
  __int64 v321; // [rsp+68h] [rbp-7B8h]
  __int64 *v322; // [rsp+68h] [rbp-7B8h]
  __int64 v323; // [rsp+68h] [rbp-7B8h]
  int v324; // [rsp+68h] [rbp-7B8h]
  __int64 *v325; // [rsp+68h] [rbp-7B8h]
  unsigned __int64 v326; // [rsp+68h] [rbp-7B8h]
  unsigned int v327; // [rsp+68h] [rbp-7B8h]
  int v328; // [rsp+68h] [rbp-7B8h]
  __int64 *v329; // [rsp+68h] [rbp-7B8h]
  __int64 v330; // [rsp+68h] [rbp-7B8h]
  __int64 v331; // [rsp+78h] [rbp-7A8h] BYREF
  _QWORD *v332; // [rsp+80h] [rbp-7A0h] BYREF
  __int64 v333; // [rsp+88h] [rbp-798h]
  _QWORD v334[8]; // [rsp+90h] [rbp-790h] BYREF
  unsigned __int64 v335[38]; // [rsp+D0h] [rbp-750h] BYREF
  __int64 v336; // [rsp+200h] [rbp-620h] BYREF
  __int64 *v337; // [rsp+208h] [rbp-618h]
  __int64 v338; // [rsp+210h] [rbp-610h]
  void (__fastcall *v339)(__int64 *, __int64); // [rsp+218h] [rbp-608h]
  __int64 v340; // [rsp+220h] [rbp-600h] BYREF
  __int64 *v341; // [rsp+260h] [rbp-5C0h]
  unsigned int v342; // [rsp+268h] [rbp-5B8h]
  int v343; // [rsp+26Ch] [rbp-5B4h]
  __int64 v344[24]; // [rsp+270h] [rbp-5B0h] BYREF
  char v345[8]; // [rsp+330h] [rbp-4F0h] BYREF
  unsigned __int64 v346; // [rsp+338h] [rbp-4E8h]
  void (__fastcall *v347)(char *, char *, __int64); // [rsp+340h] [rbp-4E0h]
  void (__fastcall *v348)(char *, __int64); // [rsp+348h] [rbp-4D8h]
  char v349[64]; // [rsp+350h] [rbp-4D0h] BYREF
  __m128i *v350; // [rsp+390h] [rbp-490h] BYREF
  __int64 v351; // [rsp+398h] [rbp-488h]
  _BYTE v352[192]; // [rsp+3A0h] [rbp-480h] BYREF
  _QWORD v353[2]; // [rsp+460h] [rbp-3C0h] BYREF
  __int64 (__fastcall *v354)(_QWORD *, _QWORD *, int); // [rsp+470h] [rbp-3B0h]
  void (__fastcall *v355)(__int64 *, void **); // [rsp+478h] [rbp-3A8h]
  char *v356; // [rsp+4C0h] [rbp-360h]
  char v357; // [rsp+4D0h] [rbp-350h] BYREF
  __m128i v358; // [rsp+590h] [rbp-290h] BYREF
  void (__fastcall *v359)(__m128i *, __m128i *, __int64); // [rsp+5A0h] [rbp-280h] BYREF
  __int64 (__fastcall *v360)(__int64); // [rsp+5A8h] [rbp-278h]
  char v361[64]; // [rsp+5B0h] [rbp-270h] BYREF
  __m128i *v362; // [rsp+5F0h] [rbp-230h] BYREF
  __int64 v363; // [rsp+5F8h] [rbp-228h]
  _BYTE v364[192]; // [rsp+600h] [rbp-220h] BYREF
  _BYTE *v365; // [rsp+6C0h] [rbp-160h] BYREF
  _BYTE s[24]; // [rsp+6C8h] [rbp-158h] BYREF
  __int64 (__fastcall *v367)(__int64); // [rsp+6E0h] [rbp-140h] BYREF
  __int64 v368; // [rsp+6F8h] [rbp-128h]
  __int64 v369; // [rsp+700h] [rbp-120h]
  __int64 v370; // [rsp+708h] [rbp-118h]
  __int64 v371; // [rsp+710h] [rbp-110h]
  __int64 v372; // [rsp+718h] [rbp-108h]
  __int64 *v373; // [rsp+720h] [rbp-100h]
  unsigned __int64 **v374; // [rsp+728h] [rbp-F8h] BYREF
  __int64 v375; // [rsp+730h] [rbp-F0h] BYREF
  unsigned __int64 *v376; // [rsp+738h] [rbp-E8h] BYREF
  unsigned __int64 *v377; // [rsp+740h] [rbp-E0h]
  __int64 v378; // [rsp+748h] [rbp-D8h]
  __int64 v379; // [rsp+750h] [rbp-D0h]
  int v380; // [rsp+758h] [rbp-C8h]
  __int64 v381; // [rsp+760h] [rbp-C0h]
  __int64 v382; // [rsp+768h] [rbp-B8h]
  __int64 v383; // [rsp+770h] [rbp-B0h]
  unsigned int v384; // [rsp+778h] [rbp-A8h]
  char v385; // [rsp+780h] [rbp-A0h]

  v1 = a1;
  v355 = sub_2EF10E0;
  v2 = *(_BYTE **)(a1 + 16);
  v354 = sub_2EEE540;
  v3 = *(_QWORD *)(a1 + 32);
  v353[0] = a1;
  v321 = v3;
  *(_QWORD *)&s[16] = 0;
  v368 = 0;
  v369 = 0;
  v370 = 0;
  LODWORD(v371) = 0;
  v372 = 0;
  v373 = 0;
  v374 = 0;
  LODWORD(v375) = 0;
  v376 = 0;
  v377 = 0;
  v378 = 0;
  v380 = 2;
  v381 = 0;
  v382 = 0;
  v383 = 0;
  v384 = 0;
  v385 = 0;
  sub_352B0C0(&v365);
  v358.m128i_i64[1] = (__int64)v353;
  v4 = _mm_loadu_si128((const __m128i *)s);
  v365 = v2;
  v358.m128i_i64[0] = (__int64)sub_2EEE4A0;
  v5 = *(void (__fastcall **)(__m128i *, __m128i *, __int64))&s[16];
  *(_QWORD *)&s[16] = sub_BD8F60;
  v6 = _mm_loadu_si128(&v358);
  v359 = v5;
  v360 = v367;
  v358 = v4;
  v367 = sub_BD8DA0;
  *(__m128i *)s = v6;
  if ( v5 )
    v5(&v358, &v358, 3);
  v379 = v321;
  v7 = *(_QWORD *)(v321 + 328);
  if ( v7 != v321 + 320 )
  {
    do
    {
      sub_352B2D0(&v365, v7);
      for ( i = *(_QWORD *)(v7 + 56); v7 + 48 != i; i = *(_QWORD *)(i + 8) )
        sub_352BBD0(&v365, i);
      v7 = *(_QWORD *)(v7 + 8);
    }
    while ( v321 + 320 != v7 );
    v1 = a1;
  }
  if ( !v380 )
  {
    *(_QWORD *)(v1 + 776) = v321;
    *(_DWORD *)(v1 + 792) = *(_DWORD *)(v321 + 120);
    sub_2E708A0(v1 + 672);
    sub_352CE10(&v365, v1 + 672);
  }
  sub_C7D6A0(v382, 16LL * v384, 8);
  v9 = v377;
  v10 = v376;
  if ( v377 != v376 )
  {
    do
    {
      v11 = *v10;
      if ( *v10 )
      {
        v12 = *(_QWORD *)(v11 + 176);
        if ( v12 != v11 + 192 )
          _libc_free(v12);
        v13 = *(_QWORD *)(v11 + 88);
        if ( v13 != v11 + 104 )
          _libc_free(v13);
        sub_C7D6A0(*(_QWORD *)(v11 + 64), 8LL * *(unsigned int *)(v11 + 80), 8);
        sub_2399F40(*(unsigned __int64 **)(v11 + 32), *(unsigned __int64 **)(v11 + 40));
        v14 = *(_QWORD *)(v11 + 32);
        if ( v14 )
          j_j___libc_free_0(v14);
        v15 = *(_QWORD *)(v11 + 8);
        if ( v15 != v11 + 24 )
          _libc_free(v15);
        j_j___libc_free_0(v11);
      }
      ++v10;
    }
    while ( v9 != v10 );
    v10 = v376;
  }
  if ( v10 )
    j_j___libc_free_0((unsigned __int64)v10);
  sub_C7D6A0((__int64)v373, 16LL * (unsigned int)v375, 8);
  v16 = (_BYTE *)(16LL * (unsigned int)v371);
  sub_C7D6A0(v369, (__int64)v16, 8);
  if ( *(_QWORD *)&s[16] )
  {
    v16 = s;
    (*(void (__fastcall **)(_BYTE *, _BYTE *, __int64))&s[16])(s, s, 3);
  }
  if ( v354 )
  {
    v16 = v353;
    v354(v353, v353, 3);
  }
  v19 = *(_QWORD *)(v1 + 32);
  v285 = *(_QWORD *)(v19 + 320) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v285 == v19 + 320 )
    goto LABEL_87;
  v20 = *(_QWORD *)(v19 + 328);
  v332 = v334;
  memset(v335, 0, sizeof(v335));
  v333 = 0x800000000LL;
  v335[1] = (unsigned __int64)&v335[4];
  v335[12] = (unsigned __int64)&v335[14];
  v337 = &v340;
  v340 = v20;
  LODWORD(v335[2]) = 8;
  BYTE4(v335[3]) = 1;
  HIDWORD(v335[13]) = 8;
  v338 = 0x100000008LL;
  LODWORD(v339) = 0;
  BYTE4(v339) = 1;
  v341 = v344;
  v343 = 8;
  v336 = 1;
  v21 = *(unsigned int *)(v20 + 120);
  v22 = *(_QWORD *)(v20 + 112);
  v344[2] = v20;
  v344[1] = v22;
  v344[0] = v22 + 8 * v21;
  v342 = 1;
  sub_2EFF3E0((__int64)&v336, (__int64)v16, v20, v344[0], v17, v18);
  sub_C8CD80((__int64)&v358, (__int64)v361, (__int64)v335, v23, v24, v25);
  v363 = 0x800000000LL;
  v29 = v335[13];
  v362 = (__m128i *)v364;
  if ( LODWORD(v335[13]) )
  {
    v265 = LODWORD(v335[13]);
    v266 = (__m128i *)v364;
    if ( LODWORD(v335[13]) > 8 )
    {
      sub_2E3C030((__int64)&v362, LODWORD(v335[13]), (__int64)v364, v26, v27, v28);
      v266 = v362;
      v265 = LODWORD(v335[13]);
    }
    v267 = (const __m128i *)v335[12];
    v268 = v335[12] + 24 * v265;
    if ( v335[12] != v268 )
    {
      do
      {
        if ( v266 )
        {
          *v266 = _mm_loadu_si128(v267);
          v266[1].m128i_i64[0] = v267[1].m128i_i64[0];
        }
        v267 = (const __m128i *)((char *)v267 + 24);
        v266 = (__m128i *)((char *)v266 + 24);
      }
      while ( (const __m128i *)v268 != v267 );
    }
    LODWORD(v363) = v29;
  }
  sub_2EFF5C0((__int64)&v365, (__int64)&v358);
  sub_C8CD80((__int64)v345, (__int64)v349, (__int64)&v336, v30, v31, v32);
  v36 = v342;
  v350 = (__m128i *)v352;
  v351 = 0x800000000LL;
  if ( v342 )
  {
    v272 = v342;
    v273 = (__m128i *)v352;
    if ( v342 > 8 )
    {
      sub_2E3C030((__int64)&v350, v342, (__int64)v352, v33, v34, v35);
      v273 = v350;
      v272 = v342;
    }
    v274 = (const __m128i *)v341;
    v275 = (const __m128i *)&v341[3 * v272];
    if ( v341 != (__int64 *)v275 )
    {
      do
      {
        if ( v273 )
        {
          *v273 = _mm_loadu_si128(v274);
          v273[1].m128i_i64[0] = v274[1].m128i_i64[0];
        }
        v274 = (const __m128i *)((char *)v274 + 24);
        v273 = (__m128i *)((char *)v273 + 24);
      }
      while ( v275 != v274 );
    }
    LODWORD(v351) = v36;
  }
  sub_2EFF5C0((__int64)v353, (__int64)v345);
  sub_2EFF7A0((__int64)v353, (__int64)&v365, (__int64)&v332, v37, v38, v39);
  if ( v356 != &v357 )
    _libc_free((unsigned __int64)v356);
  if ( BYTE4(v355) )
  {
    v40 = (unsigned __int64)v350;
    if ( v350 == (__m128i *)v352 )
      goto LABEL_37;
    goto LABEL_36;
  }
  _libc_free(v353[1]);
  v40 = (unsigned __int64)v350;
  if ( v350 != (__m128i *)v352 )
LABEL_36:
    _libc_free(v40);
LABEL_37:
  if ( !BYTE4(v348) )
    _libc_free(v346);
  if ( v373 != &v375 )
    _libc_free((unsigned __int64)v373);
  if ( s[20] )
  {
    v41 = (unsigned __int64)v362;
    if ( v362 == (__m128i *)v364 )
      goto LABEL_44;
    goto LABEL_43;
  }
  _libc_free(*(unsigned __int64 *)s);
  v41 = (unsigned __int64)v362;
  if ( v362 != (__m128i *)v364 )
LABEL_43:
    _libc_free(v41);
LABEL_44:
  if ( BYTE4(v360) )
  {
    v42 = (unsigned __int64)v341;
    if ( v341 == v344 )
      goto LABEL_47;
    goto LABEL_46;
  }
  _libc_free(v358.m128i_u64[1]);
  v42 = (unsigned __int64)v341;
  if ( v341 != v344 )
LABEL_46:
    _libc_free(v42);
LABEL_47:
  if ( !BYTE4(v339) )
  {
    _libc_free((unsigned __int64)v337);
    v43 = v335[12];
    if ( (unsigned __int64 *)v335[12] == &v335[14] )
      goto LABEL_50;
    goto LABEL_49;
  }
  v43 = v335[12];
  if ( (unsigned __int64 *)v335[12] != &v335[14] )
LABEL_49:
    _libc_free(v43);
LABEL_50:
  if ( !BYTE4(v335[3]) )
    _libc_free(v335[1]);
  v283 = v332;
  v306 = &v332[(unsigned int)v333];
  if ( v332 != v306 )
  {
    v282 = v1;
    v44 = &v358;
    src = (void *)(v1 + 600);
    while ( 1 )
    {
      v45 = *(v306 - 1);
      LODWORD(v369) = 0;
      v370 = 0;
      v353[0] = v45;
      v371 = 0;
      v365 = &s[8];
      *(_QWORD *)s = 0x600000000LL;
      v372 = 0;
      v373 = 0;
      v374 = &v376;
      v375 = 0;
      v46 = sub_2EEFC50((__int64)src, v353);
      v358 = (__m128i)(unsigned __int64)&v359;
      sub_2EF9210((__int64)&v365, (__int64)(v46 + 5), (__int64)v44);
      if ( (void (__fastcall **)(__m128i *, __m128i *, __int64))v358.m128i_i64[0] != &v359 )
        _libc_free(v358.m128i_u64[0]);
      v358 = (__m128i)(unsigned __int64)&v359;
      sub_2EF9210((__int64)&v365, (__int64)(v46 + 9), (__int64)v44);
      if ( (void (__fastcall **)(__m128i *, __m128i *, __int64))v358.m128i_i64[0] != &v359 )
        _libc_free(v358.m128i_u64[0]);
      v47 = *(__int64 **)(v353[0] + 64LL);
      v322 = &v47[*(unsigned int *)(v353[0] + 72LL)];
      if ( v47 != v322 )
      {
        v48 = *(__int64 **)(v353[0] + 64LL);
        do
        {
          v358.m128i_i64[0] = *v48;
          v49 = sub_2EEFC50((__int64)src, v44);
          if ( *(_BYTE *)v49 )
          {
            v313 = v49;
            sub_2EF9210((__int64)&v365, (__int64)(v49 + 9), (__int64)&v374);
            sub_2EF9210((__int64)&v365, (__int64)(v313 + 13), (__int64)&v374);
          }
          ++v48;
        }
        while ( v322 != v48 );
      }
      v50 = v375;
      v323 = (__int64)(v46 + 13);
      v51 = v46[13] + 1LL;
      if ( (_DWORD)v375 )
      {
        v46[13] = v51;
        v52 = (4 * v50 / 3u + 1) | ((unsigned __int64)(4 * v50 / 3u + 1) >> 1);
        v53 = (((v52 >> 2) | v52) >> 4) | (v52 >> 2) | v52;
        v54 = ((((v53 >> 8) | v53) >> 16) | (v53 >> 8) | v53) + 1;
        if ( *((_DWORD *)v46 + 32) < v54 )
          sub_2E29BA0(v323, v54);
      }
      else
      {
        v46[13] = v51;
      }
      v55 = (int *)v374;
      v56 = (unsigned __int64 **)((char *)v374 + 4 * (unsigned int)v375);
      if ( v374 != v56 )
        break;
LABEL_78:
      if ( v56 != &v376 )
        _libc_free((unsigned __int64)v56);
      sub_C7D6A0(v371, 4LL * (unsigned int)v373, 4);
      if ( v365 != &s[8] )
        _libc_free((unsigned __int64)v365);
      if ( v283 == --v306 )
      {
        v1 = v282;
        v306 = v332;
        goto LABEL_84;
      }
    }
    v314 = v44;
    while ( 1 )
    {
      v61 = *((_DWORD *)v46 + 32);
      if ( !v61 )
        break;
      v57 = v46[14];
      v58 = (v61 - 1) & (37 * *v55);
      v59 = (_DWORD *)(v57 + 4LL * v58);
      v60 = *v59;
      if ( *v55 == *v59 )
      {
LABEL_69:
        if ( v56 == (unsigned __int64 **)++v55 )
          goto LABEL_77;
      }
      else
      {
        v116 = 1;
        v66 = 0;
        while ( v60 != -1 )
        {
          if ( v66 || v60 != -2 )
            v59 = v66;
          v58 = (v61 - 1) & (v116 + v58);
          v60 = *(_DWORD *)(v57 + 4LL * v58);
          if ( *v55 == v60 )
            goto LABEL_69;
          ++v116;
          v66 = v59;
          v59 = (_DWORD *)(v57 + 4LL * v58);
        }
        v117 = *((_DWORD *)v46 + 30);
        if ( !v66 )
          v66 = v59;
        ++v46[13];
        v68 = v117 + 1;
        if ( 4 * v68 < 3 * v61 )
        {
          if ( v61 - *((_DWORD *)v46 + 31) - v68 > v61 >> 3 )
            goto LABEL_74;
          sub_2E29BA0(v323, v61);
          v118 = *((_DWORD *)v46 + 32);
          if ( !v118 )
          {
LABEL_544:
            ++*((_DWORD *)v46 + 30);
            BUG();
          }
          v119 = v118 - 1;
          v120 = v46[14];
          v121 = 1;
          v122 = 0;
          v123 = v119 & (37 * *v55);
          v66 = (_DWORD *)(v120 + 4LL * v123);
          v124 = *v66;
          v68 = *((_DWORD *)v46 + 30) + 1;
          if ( *v55 == *v66 )
            goto LABEL_74;
          while ( v124 != -1 )
          {
            if ( !v122 && v124 == -2 )
              v122 = v66;
            v279 = v121 + 1;
            v280 = v119 & (v123 + v121);
            v66 = (_DWORD *)(v120 + 4 * v280);
            v123 = v280;
            v124 = *v66;
            if ( *v55 == *v66 )
              goto LABEL_74;
            v121 = v279;
          }
          goto LABEL_173;
        }
LABEL_72:
        sub_2E29BA0(v323, 2 * v61);
        v62 = *((_DWORD *)v46 + 32);
        if ( !v62 )
          goto LABEL_544;
        v63 = v62 - 1;
        v64 = v46[14];
        v65 = v63 & (37 * *v55);
        v66 = (_DWORD *)(v64 + 4LL * v65);
        v67 = *v66;
        v68 = *((_DWORD *)v46 + 30) + 1;
        if ( *v55 == *v66 )
          goto LABEL_74;
        v276 = 1;
        v122 = 0;
        while ( v67 != -1 )
        {
          if ( v67 == -2 && !v122 )
            v122 = v66;
          v277 = v276 + 1;
          v278 = v63 & (v65 + v276);
          v66 = (_DWORD *)(v64 + 4 * v278);
          v65 = v278;
          v67 = *v66;
          if ( *v55 == *v66 )
            goto LABEL_74;
          v276 = v277;
        }
LABEL_173:
        if ( v122 )
          v66 = v122;
LABEL_74:
        *((_DWORD *)v46 + 30) = v68;
        if ( *v66 != -1 )
          --*((_DWORD *)v46 + 31);
        v69 = *v55++;
        *v66 = v69;
        if ( v56 == (unsigned __int64 **)v55 )
        {
LABEL_77:
          v44 = v314;
          v56 = v374;
          goto LABEL_78;
        }
      }
    }
    ++v46[13];
    goto LABEL_72;
  }
LABEL_84:
  if ( v306 != v334 )
    _libc_free((unsigned __int64)v306);
  v19 = *(_QWORD *)(v1 + 32);
  v285 = v19 + 320;
LABEL_87:
  v70 = *(_BYTE **)(v19 + 328);
  srca = (void *)(v1 + 600);
  if ( v70 != (_BYTE *)v285 )
  {
    while ( 1 )
    {
      v71 = (__m128i *)&v365;
      v365 = v70;
      v72 = sub_2EEFC50((__int64)srca, &v365);
      s[20] = 1;
      v73 = 1;
      v297 = v72;
      v365 = 0;
      *(_QWORD *)s = &v367;
      *(_QWORD *)&s[8] = 8;
      *(_DWORD *)&s[16] = 0;
      v74 = *((_QWORD *)v70 + 7);
      v290 = v70 + 48;
      if ( (_BYTE *)v74 != v70 + 48 )
        break;
LABEL_92:
      v70 = (_BYTE *)*((_QWORD *)v70 + 1);
      if ( v70 == (_BYTE *)v285 )
        goto LABEL_183;
    }
    while ( 1 )
    {
      if ( *(_WORD *)(v74 + 68) && *(_WORD *)(v74 + 68) != 68 )
      {
LABEL_90:
        if ( !v73 )
          _libc_free(*(unsigned __int64 *)s);
        goto LABEL_92;
      }
      ++v365;
      if ( !v73 )
      {
        v75 = 4 * (*(_DWORD *)&s[12] - *(_DWORD *)&s[16]);
        if ( v75 < 0x20 )
          v75 = 32;
        if ( *(_DWORD *)&s[8] > v75 )
        {
          sub_C8C990((__int64)&v365, (__int64)v71);
          goto LABEL_101;
        }
        v71 = (__m128i *)0xFFFFFFFFLL;
        memset(*(void **)s, -1, 8LL * *(unsigned int *)&s[8]);
      }
      *(_QWORD *)&s[12] = 0;
LABEL_101:
      v76 = *(_QWORD *)(v74 + 32);
      if ( *(_BYTE *)v76 || (v77 = *(_BYTE *)(v76 + 3), (v77 & 0x10) == 0) )
      {
        v71 = (__m128i *)"Expected first PHI operand to be a register def";
        sub_2EF0A60(v1, "Expected first PHI operand to be a register def", *(_QWORD *)(v74 + 32), 0, 0);
        goto LABEL_104;
      }
      if ( (*(_WORD *)(v76 + 2) & 0xFF0) != 0 || (v77 & 0x20) != 0 || (*(_BYTE *)(v76 + 4) & 0xE) != 0 )
      {
        v71 = (__m128i *)"Unexpected flag on PHI operand";
        sub_2EF0A60(v1, "Unexpected flag on PHI operand", *(_QWORD *)(v74 + 32), 0, 0);
      }
      if ( *(int *)(v76 + 8) >= 0 )
      {
        v71 = (__m128i *)"Expected first PHI operand to be a virtual register";
        sub_2EF0A60(v1, "Expected first PHI operand to be a virtual register", v76, 0, 0);
      }
      v78 = 1;
      v324 = *(_DWORD *)(v74 + 40) & 0xFFFFFF;
      if ( v324 != 1 )
      {
        v292 = (__int64)v70;
        while ( 1 )
        {
          while ( 1 )
          {
            v79 = *(_QWORD *)(v74 + 32);
            v80 = v79 + 40LL * v78;
            if ( !*(_BYTE *)v80 )
              break;
            v103 = v78;
            v104 = v79 + 40LL * v78;
            v71 = (__m128i *)"Expected PHI operand to be a register";
            v78 += 2;
            sub_2EF0A60(v1, "Expected PHI operand to be a register", v104, v103, 0);
            if ( v324 == v78 )
            {
LABEL_145:
              v70 = (_BYTE *)v292;
              goto LABEL_146;
            }
          }
          if ( (*(_BYTE *)(v80 + 3) & 0x20) != 0
            || (*(_BYTE *)(v80 + 4) & 0xE) != 0
            || (*(_WORD *)(v80 + 2) & 0xFF0) != 0 )
          {
            sub_2EF0A60(v1, "Unexpected flag on PHI operand", v79 + 40LL * v78, v78, 0);
            v79 = *(_QWORD *)(v74 + 32);
          }
          v81 = v78 + 1;
          v82 = v79 + 40 * v81;
          if ( *(_BYTE *)v82 != 4 )
          {
            v71 = (__m128i *)"Expected PHI operand to be a basic block";
            sub_2EF0A60(v1, "Expected PHI operand to be a basic block", v82, v81, 0);
            goto LABEL_119;
          }
          v83 = *(_QWORD *)(v82 + 24);
          v71 = (__m128i *)v292;
          v315 = v79 + 40 * v81;
          v84 = sub_2E322C0(v83, v292);
          v87 = (_QWORD *)v315;
          v88 = v78 + 1;
          if ( !v84 )
          {
            v71 = (__m128i *)"PHI input is not a predecessor block";
            sub_2EF0A60(v1, "PHI input is not a predecessor block", v315, v88, 0);
            goto LABEL_119;
          }
          if ( *v297 )
            break;
LABEL_119:
          v78 += 2;
          if ( v324 == v78 )
            goto LABEL_145;
        }
        if ( !s[20] )
          goto LABEL_178;
        v89 = *(_QWORD **)s;
        v88 = *(unsigned int *)&s[12];
        v87 = (_QWORD *)(*(_QWORD *)s + 8LL * *(unsigned int *)&s[12]);
        if ( *(_QWORD **)s != v87 )
        {
          while ( v83 != *v89 )
          {
            if ( v87 == ++v89 )
              goto LABEL_180;
          }
          goto LABEL_132;
        }
LABEL_180:
        if ( *(_DWORD *)&s[12] < *(_DWORD *)&s[8] )
        {
          ++*(_DWORD *)&s[12];
          *v87 = v83;
          ++v365;
        }
        else
        {
LABEL_178:
          sub_C8CC70((__int64)&v365, v83, (__int64)v87, v88, v85, v86);
        }
LABEL_132:
        v71 = &v358;
        v358.m128i_i64[0] = v83;
        v90 = sub_2EEFC50((__int64)srca, &v358);
        if ( (*(_BYTE *)(v80 + 4) & 1) != 0 || !*(_BYTE *)v90 )
          goto LABEL_119;
        v91 = *((_DWORD *)v90 + 24);
        v92 = *(_DWORD *)(v80 + 8);
        v93 = v90[10];
        if ( v91 )
        {
          v94 = v91 - 1;
          v95 = v94 & (37 * v92);
          v71 = (__m128i *)v95;
          v96 = *(_DWORD *)(v93 + 4LL * v95);
          if ( v92 == v96 )
            goto LABEL_119;
          v97 = 1;
          while ( v96 != -1 )
          {
            v281 = v97 + 1;
            v95 = v94 & (v97 + v95);
            v71 = (__m128i *)v95;
            v96 = *(_DWORD *)(v93 + 4LL * v95);
            if ( v92 == v96 )
              goto LABEL_119;
            v97 = v281;
          }
        }
        v98 = *((_DWORD *)v90 + 32);
        v99 = v90[14];
        if ( !v98 )
          goto LABEL_460;
        v71 = (__m128i *)(unsigned int)(v98 - 1);
        v100 = (unsigned int)v71 & (37 * v92);
        v101 = *(_DWORD *)(v99 + 4LL * v100);
        if ( v92 != v101 )
        {
          v102 = 1;
          while ( v101 != -1 )
          {
            v100 = (unsigned int)v71 & (v102 + v100);
            v101 = *(_DWORD *)(v99 + 4LL * v100);
            if ( v92 == v101 )
              goto LABEL_119;
            ++v102;
          }
LABEL_460:
          v71 = (__m128i *)"PHI operand is not live-out from predecessor";
          sub_2EF0A60(v1, "PHI operand is not live-out from predecessor", v80, v78, 0);
          goto LABEL_119;
        }
        goto LABEL_119;
      }
LABEL_146:
      if ( *v297 )
      {
        v105 = (__int64 *)*((_QWORD *)v70 + 8);
        v325 = &v105[*((unsigned int *)v70 + 18)];
        if ( v105 != v325 )
        {
          v316 = v74;
          v307 = v70;
          v106 = (__int64 *)*((_QWORD *)v70 + 8);
          while ( 1 )
          {
            v107 = *v106;
            if ( s[20] )
            {
              v108 = *(_QWORD **)s;
              v109 = *(_QWORD *)s + 8LL * *(unsigned int *)&s[12];
              if ( *(_QWORD *)s != v109 )
              {
                while ( v107 != *v108 )
                {
                  if ( (_QWORD *)v109 == ++v108 )
                    goto LABEL_157;
                }
                goto LABEL_154;
              }
LABEL_157:
              sub_2EF06E0(v1, "Missing PHI operand", v316);
              v110 = v107;
              v111 = (__m128i *)&v336;
              v112 = *(_QWORD *)(v1 + 16);
              sub_2E31000(&v336, v107);
              if ( !v338 )
LABEL_533:
                sub_4263D6(v111, v110, v113);
              v71 = (__m128i *)v112;
              v339(&v336, v112);
              v114 = *(__m128i **)(v112 + 32);
              if ( *(_QWORD *)(v112 + 24) - (_QWORD)v114 <= 0x27u )
              {
                v71 = (__m128i *)" is a predecessor according to the CFG.\n";
                sub_CB6200(v112, " is a predecessor according to the CFG.\n", 0x28u);
              }
              else
              {
                si128 = _mm_load_si128((const __m128i *)&xmmword_4453E90);
                v114[2].m128i_i64[0] = 0xA2E474643206568LL;
                *v114 = si128;
                v114[1] = _mm_load_si128((const __m128i *)&xmmword_4453EA0);
                *(_QWORD *)(v112 + 32) += 40LL;
              }
              if ( !v338 )
                goto LABEL_154;
              v71 = (__m128i *)&v336;
              ++v106;
              ((void (__fastcall *)(__int64 *, __int64 *, __int64))v338)(&v336, &v336, 3);
              if ( v325 == v106 )
              {
LABEL_155:
                v70 = v307;
                v74 = v316;
                break;
              }
            }
            else
            {
              v71 = (__m128i *)*v106;
              if ( !sub_C8CA60((__int64)&v365, v107) )
                goto LABEL_157;
LABEL_154:
              if ( v325 == ++v106 )
                goto LABEL_155;
            }
          }
        }
      }
LABEL_104:
      if ( (*(_BYTE *)v74 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v74 + 44) & 8) != 0 )
          v74 = *(_QWORD *)(v74 + 8);
      }
      v74 = *(_QWORD *)(v74 + 8);
      v73 = s[20];
      if ( v290 == (_BYTE *)v74 )
        goto LABEL_90;
    }
  }
LABEL_183:
  sub_2EF9CC0(v1);
  v127 = *(_QWORD *)(v1 + 32);
  v293 = v1 + 600;
  v128 = *(_QWORD *)(v127 + 328);
  v298 = v127 + 320;
  v326 = v128;
  if ( v128 != v127 + 320 )
  {
    v308 = v1;
    do
    {
      v365 = (_BYTE *)v326;
      v129 = sub_2EEFC50(v293, &v365);
      if ( *((_DWORD *)v129 + 38) )
      {
        v130 = (unsigned int *)v129[18];
        v131 = &v130[*((unsigned int *)v129 + 40)];
        if ( v130 != v131 )
        {
          while ( *v130 > 0xFFFFFFFD )
          {
            if ( v131 == ++v130 )
              goto LABEL_185;
          }
          if ( v130 != v131 )
          {
            v317 = v129;
            do
            {
              v132 = *v130;
              v133 = v317[6];
              v134 = *((_DWORD *)v317 + 16);
              if ( v134 )
              {
                v135 = v134 - 1;
                v136 = (v134 - 1) & (37 * v132);
                v137 = *(_DWORD *)(v133 + 4LL * v136);
                if ( v132 == v137 )
                {
LABEL_195:
                  sub_2EF03A0(v308, "Virtual register killed in block, but needed live out.", v326);
                  v138 = *(_QWORD *)(v308 + 16);
                  v139 = *(__m128i **)(v138 + 32);
                  if ( *(_QWORD *)(v138 + 24) - (_QWORD)v139 <= 0x10u )
                  {
                    v138 = sub_CB6200(v138, "Virtual register ", 0x11u);
                  }
                  else
                  {
                    v140 = _mm_load_si128((const __m128i *)&xmmword_42EF1B0);
                    v139[1].m128i_i8[0] = 32;
                    *v139 = v140;
                    *(_QWORD *)(v138 + 32) += 17LL;
                  }
                  v110 = v132;
                  v111 = (__m128i *)&v332;
                  sub_2FF6320(&v332, v132, 0, 0, 0);
                  if ( !v334[0] )
                    goto LABEL_533;
                  ((void (__fastcall *)(_QWORD **, __int64))v334[1])(&v332, v138);
                  v141 = *(__m128i **)(v138 + 32);
                  if ( *(_QWORD *)(v138 + 24) - (_QWORD)v141 <= 0x19u )
                  {
                    sub_CB6200(v138, " is used after the block.\n", 0x1Au);
                  }
                  else
                  {
                    v142 = _mm_load_si128((const __m128i *)&xmmword_42EF1C0);
                    qmemcpy(&v141[1], "he block.\n", 10);
                    *v141 = v142;
                    *(_QWORD *)(v138 + 32) += 26LL;
                  }
                  if ( v334[0] )
                    ((void (__fastcall *)(_QWORD **, _QWORD **, __int64))v334[0])(&v332, &v332, 3);
                }
                else
                {
                  v143 = 1;
                  while ( v137 != -1 )
                  {
                    v125 = (unsigned int)(v143 + 1);
                    v136 = v135 & (v143 + v136);
                    v137 = *(_DWORD *)(v133 + 4LL * v136);
                    if ( v137 == v132 )
                      goto LABEL_195;
                    ++v143;
                  }
                }
              }
              if ( ++v130 == v131 )
                break;
              while ( *v130 > 0xFFFFFFFD )
              {
                if ( v131 == ++v130 )
                  goto LABEL_185;
              }
            }
            while ( v130 != v131 );
          }
        }
      }
LABEL_185:
      v326 = *(_QWORD *)(v326 + 8);
    }
    while ( v298 != v326 );
    v1 = v308;
    v127 = *(_QWORD *)(v308 + 32);
    v128 = v127 + 320;
    v298 = v127 + 320;
  }
  if ( v298 != (*(_QWORD *)(v127 + 320) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v365 = *(_BYTE **)(v127 + 328);
    v144 = sub_2EEFC50(v1 + 600, &v365);
    if ( *((_DWORD *)v144 + 38) )
    {
      v269 = (unsigned int *)v144[18];
      v270 = &v269[*((unsigned int *)v144 + 40)];
      if ( v269 != v270 )
      {
        while ( *v269 > 0xFFFFFFFD )
        {
          if ( v270 == ++v269 )
            goto LABEL_217;
        }
        while ( v270 != v269 )
        {
          v271 = *v269++;
          sub_2EEFF60(v1, "Virtual register defs don't dominate all uses.", *(__int64 **)(v1 + 32));
          sub_2EEF700(v1, v271);
          if ( v269 == v270 )
            break;
          while ( *v269 > 0xFFFFFFFD )
          {
            if ( v270 == ++v269 )
              goto LABEL_217;
          }
        }
      }
    }
  }
LABEL_217:
  v145 = *(__m128i **)(v1 + 632);
  if ( v145 )
  {
    v146 = *(__m128i **)(v1 + 64);
    srcb = v146[4].m128i_i32[0];
    if ( srcb )
    {
      v299 = 0;
      while ( 1 )
      {
        v147 = (__int64 *)sub_2E29D60(v145, v299 | 0x80000000, (__int64)v146, v128, v125, v126);
        v148 = *(_QWORD *)(v1 + 32);
        v149 = *(_QWORD *)(v148 + 328);
        v318 = v148 + 320;
        if ( v149 != v148 + 320 )
          break;
LABEL_241:
        if ( srcb == ++v299 )
          goto LABEL_244;
        v145 = *(__m128i **)(v1 + 632);
      }
      v327 = v299 | 0x80000000;
      v309 = 37 * (v299 | 0x80000000);
      while ( 1 )
      {
        v365 = (_BYTE *)v149;
        v150 = sub_2EEFC50(v1 + 600, &v365);
        v146 = (__m128i *)v150[18];
        v151 = *((_DWORD *)v150 + 40);
        if ( v151 )
        {
          v152 = v151 - 1;
          v153 = v152 & v309;
          v128 = v146->m128i_u32[v152 & v309];
          if ( v327 == (_DWORD)v128 )
          {
LABEL_224:
            v154 = (__int64 *)*v147;
            if ( v147 != (__int64 *)*v147 )
            {
              v155 = (__int64 *)v147[3];
              v128 = *(unsigned int *)(v149 + 24);
              if ( v147 == v155 )
              {
                v155 = (__int64 *)v147[1];
                v157 = (unsigned int)v128 >> 7;
                v147[3] = (__int64)v155;
                v156 = *((_DWORD *)v155 + 4);
                if ( (unsigned int)v128 >> 7 == v156 )
                {
                  if ( v147 != v155 )
                    goto LABEL_353;
                }
                else
                {
LABEL_227:
                  if ( v157 >= v156 )
                  {
                    if ( v147 == v155 )
                    {
LABEL_388:
                      v147[3] = (__int64)v155;
                      goto LABEL_233;
                    }
                    while ( v156 < v157 )
                    {
                      v155 = (__int64 *)*v155;
                      if ( v147 == v155 )
                        goto LABEL_388;
                      v156 = *((_DWORD *)v155 + 4);
                    }
LABEL_231:
                    v147[3] = (__int64)v155;
                    if ( v147 == v155 )
                      goto LABEL_233;
                  }
                  else
                  {
                    if ( v154 != v155 )
                    {
                      while ( 1 )
                      {
                        v155 = (__int64 *)v155[1];
                        if ( v154 == v155 )
                          break;
                        if ( *((_DWORD *)v155 + 4) <= v157 )
                          goto LABEL_231;
                      }
                    }
                    v147[3] = (__int64)v155;
                  }
                  if ( *((_DWORD *)v155 + 4) == v157 )
                    goto LABEL_353;
                }
              }
              else
              {
                v156 = *((_DWORD *)v155 + 4);
                v157 = (unsigned int)v128 >> 7;
                if ( v156 != (unsigned int)v128 >> 7 )
                  goto LABEL_227;
LABEL_353:
                v146 = (__m128i *)(v155[(((unsigned int)v128 >> 6) & 1) + 3] & (1LL << v128));
                if ( v146 )
                  goto LABEL_240;
              }
            }
LABEL_233:
            sub_2EF03A0(v1, "LiveVariables: Block missing from AliveBlocks", v149);
            v158 = *(_QWORD *)(v1 + 16);
            v159 = *(__m128i **)(v158 + 32);
            if ( *(_QWORD *)(v158 + 24) - (_QWORD)v159 <= 0x10u )
            {
              v158 = sub_CB6200(*(_QWORD *)(v1 + 16), "Virtual register ", 0x11u);
            }
            else
            {
              v160 = _mm_load_si128((const __m128i *)&xmmword_42EF1B0);
              v159[1].m128i_i8[0] = 32;
              *v159 = v160;
              *(_QWORD *)(v158 + 32) += 17LL;
            }
            v110 = v327;
            v111 = (__m128i *)v353;
            v294 = v158;
            sub_2FF6320(v353, v327, 0, 0, 0);
            if ( !v354 )
              goto LABEL_533;
            v355(v353, (void **)v294);
            v146 = *(__m128i **)(v294 + 32);
            if ( *(_QWORD *)(v294 + 24) - (_QWORD)v146 <= 0x20u )
            {
              sub_CB6200(v294, " must be live through the block.\n", 0x21u);
            }
            else
            {
              v161 = _mm_load_si128((const __m128i *)&xmmword_42EF1D0);
              v146[2].m128i_i8[0] = 10;
              *v146 = v161;
              v146[1] = _mm_load_si128((const __m128i *)&xmmword_42EF1E0);
              *(_QWORD *)(v294 + 32) += 33LL;
            }
            if ( v354 )
              v354(v353, v353, 3);
            goto LABEL_240;
          }
          v214 = 1;
          while ( (_DWORD)v128 != -1 )
          {
            v125 = (unsigned int)(v214 + 1);
            v153 = v152 & (v214 + v153);
            v128 = v146->m128i_u32[v153];
            if ( v327 == (_DWORD)v128 )
              goto LABEL_224;
            ++v214;
          }
        }
        v215 = (__int64 *)*v147;
        if ( v147 == (__int64 *)*v147 )
          goto LABEL_240;
        v216 = (__int64 *)v147[3];
        v128 = *(unsigned int *)(v149 + 24);
        if ( v147 == v216 )
        {
          v216 = (__int64 *)v147[1];
          v146 = (__m128i *)((unsigned int)v128 >> 7);
          v147[3] = (__int64)v216;
          v217 = *((_DWORD *)v216 + 4);
          if ( (_DWORD)v146 == v217 )
          {
            if ( v147 == v216 )
              goto LABEL_240;
LABEL_343:
            v146 = (__m128i *)(v216[(((unsigned int)v128 >> 6) & 1) + 3] & (1LL << v128));
            if ( v146 )
            {
              sub_2EF03A0(v1, "LiveVariables: Block should not be in AliveBlocks", v149);
              v218 = *(_QWORD *)(v1 + 16);
              v219 = *(__m128i **)(v218 + 32);
              if ( *(_QWORD *)(v218 + 24) - (_QWORD)v219 <= 0x10u )
              {
                v218 = sub_CB6200(*(_QWORD *)(v1 + 16), "Virtual register ", 0x11u);
              }
              else
              {
                v220 = _mm_load_si128((const __m128i *)&xmmword_42EF1B0);
                v219[1].m128i_i8[0] = 32;
                *v219 = v220;
                *(_QWORD *)(v218 + 32) += 17LL;
              }
              v110 = v327;
              v111 = (__m128i *)v345;
              v296 = v218;
              sub_2FF6320(v345, v327, 0, 0, 0);
              if ( !v347 )
                goto LABEL_533;
              v348(v345, v296);
              v146 = *(__m128i **)(v296 + 32);
              if ( *(_QWORD *)(v296 + 24) - (_QWORD)v146 <= 0x26u )
              {
                sub_CB6200(v296, " is not needed live through the block.\n", 0x27u);
              }
              else
              {
                v221 = _mm_load_si128((const __m128i *)&xmmword_42EF1F0);
                v146[2].m128i_i32[0] = 1668246626;
                v146[2].m128i_i16[2] = 11883;
                *v146 = v221;
                v222 = _mm_load_si128((const __m128i *)&xmmword_42EF200);
                v146[2].m128i_i8[6] = 10;
                v146[1] = v222;
                *(_QWORD *)(v296 + 32) += 39LL;
              }
              if ( v347 )
                v347(v345, v345, 3);
            }
            goto LABEL_240;
          }
        }
        else
        {
          v217 = *((_DWORD *)v216 + 4);
          v146 = (__m128i *)((unsigned int)v128 >> 7);
          if ( v217 == (_DWORD)v146 )
            goto LABEL_343;
        }
        if ( v217 <= (unsigned int)v146 )
        {
          if ( v147 == v216 )
          {
LABEL_412:
            v147[3] = (__int64)v216;
            goto LABEL_240;
          }
          while ( v217 < (unsigned int)v146 )
          {
            v216 = (__int64 *)*v216;
            if ( v147 == v216 )
              goto LABEL_412;
            v217 = *((_DWORD *)v216 + 4);
          }
LABEL_341:
          v147[3] = (__int64)v216;
          if ( v147 == v216 )
            goto LABEL_240;
        }
        else
        {
          if ( v215 != v216 )
          {
            while ( 1 )
            {
              v216 = (__int64 *)v216[1];
              if ( v215 == v216 )
                break;
              if ( *((_DWORD *)v216 + 4) <= (unsigned int)v146 )
                goto LABEL_341;
            }
          }
          v147[3] = (__int64)v216;
        }
        if ( *((_DWORD *)v216 + 4) == (_DWORD)v146 )
          goto LABEL_343;
LABEL_240:
        v149 = *(_QWORD *)(v149 + 8);
        if ( v318 == v149 )
          goto LABEL_241;
      }
    }
    if ( !*(_QWORD *)(v1 + 640) )
      goto LABEL_266;
    goto LABEL_261;
  }
LABEL_244:
  v146 = *(__m128i **)(v1 + 64);
  if ( *(_QWORD *)(v1 + 640) )
  {
    v304 = v146[4].m128i_i32[0];
    if ( !v304 )
    {
LABEL_261:
      v174 = 0;
      v175 = *(unsigned int *)(*(_QWORD *)(v1 + 56) + 44LL);
      if ( (_DWORD)v175 )
      {
        do
        {
          v176 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v1 + 640) + 424LL) + 8 * v174);
          if ( v176 )
            sub_2EF2540(v1, v176, v174, 0, 0);
          ++v174;
        }
        while ( v174 != v175 );
      }
      v146 = *(__m128i **)(v1 + 64);
      goto LABEL_266;
    }
    v328 = 0;
    while ( 1 )
    {
      v162 = v328 & 0x7FFFFFFF;
      v163 = *(_QWORD *)(v146[3].m128i_i64[1] + 16 * v162 + 8);
      if ( v163 )
      {
        if ( (*(_BYTE *)(v163 + 4) & 8) != 0 )
        {
          while ( 1 )
          {
            v163 = *(_QWORD *)(v163 + 32);
            if ( !v163 )
              break;
            if ( (*(_BYTE *)(v163 + 4) & 8) == 0 )
              goto LABEL_249;
          }
        }
        else
        {
LABEL_249:
          v164 = *(_QWORD *)(v1 + 640);
          if ( *(_DWORD *)(v164 + 160) > (v328 & 0x7FFFFFFFu)
            && (v165 = *(_QWORD *)(v164 + 152), (v166 = *(_QWORD *)(v165 + 8 * v162)) != 0) )
          {
            v167 = *(_DWORD *)(v166 + 112);
            sub_2EF2540(v1, *(_QWORD *)(v165 + 8 * v162), v167, 0, 0);
            if ( *(_QWORD *)(v166 + 104) )
            {
              v168 = sub_2EBF1E0(*(_QWORD *)(v1 + 64), v167);
              v170 = *(_QWORD *)(v166 + 104);
              if ( v170 )
              {
                v171 = 0;
                v172 = 0;
                *(_QWORD *)&v310 = ~v169;
                *((_QWORD *)&v310 + 1) = ~v168;
                while ( 1 )
                {
                  *((_QWORD *)&v173 + 1) = *(_QWORD *)(v170 + 112);
                  *(_QWORD *)&v173 = *(_QWORD *)(v170 + 120);
                  if ( *((_QWORD *)&v173 + 1) & v172 | (unsigned __int64)v173 & v171 )
                  {
                    sub_2EEFF60(v1, "Lane masks of sub ranges overlap in live interval", *(__int64 **)(v1 + 32));
                    v227 = *(_QWORD *)(v1 + 16);
                    v228 = *(void **)(v227 + 32);
                    if ( *(_QWORD *)(v227 + 24) - (_QWORD)v228 <= 0xEu )
                    {
                      v227 = sub_CB6200(*(_QWORD *)(v1 + 16), "- interval:    ", 0xFu);
                    }
                    else
                    {
                      qmemcpy(v228, "- interval:    ", 15);
                      *(_QWORD *)(v227 + 32) += 15LL;
                    }
                    v302 = v227;
                    sub_2E0B730(v166, v227);
                    v229 = *(_BYTE **)(v302 + 32);
                    if ( (unsigned __int64)v229 >= *(_QWORD *)(v302 + 24) )
                    {
                      sub_CB5D20(v302, 10);
                    }
                    else
                    {
                      *(_QWORD *)(v302 + 32) = v229 + 1;
                      *v229 = 10;
                    }
                    *(_QWORD *)&v173 = *(_QWORD *)(v170 + 120);
                    *((_QWORD *)&v173 + 1) = *(_QWORD *)(v170 + 112);
                  }
                  if ( (v310 & v173) != 0 )
                  {
                    sub_2EEFF60(v1, "Subrange lanemask is invalid", *(__int64 **)(v1 + 32));
                    v224 = *(_QWORD *)(v1 + 16);
                    v225 = *(void **)(v224 + 32);
                    if ( *(_QWORD *)(v224 + 24) - (_QWORD)v225 <= 0xEu )
                    {
                      v224 = sub_CB6200(*(_QWORD *)(v1 + 16), "- interval:    ", 0xFu);
                    }
                    else
                    {
                      qmemcpy(v225, "- interval:    ", 15);
                      *(_QWORD *)(v224 + 32) += 15LL;
                    }
                    v301 = v224;
                    sub_2E0B730(v166, v224);
                    v226 = *(_BYTE **)(v301 + 32);
                    if ( (unsigned __int64)v226 >= *(_QWORD *)(v301 + 24) )
                    {
                      sub_CB5D20(v301, 10);
                    }
                    else
                    {
                      *(_QWORD *)(v301 + 32) = v226 + 1;
                      *v226 = 10;
                    }
                  }
                  if ( !*(_DWORD *)(v170 + 8) )
                  {
                    sub_2EEFF60(v1, "Subrange must not be empty", *(__int64 **)(v1 + 32));
                    sub_2EEFB40(v1, v170, *(_DWORD *)(v166 + 112), *(_QWORD *)(v170 + 112), *(_QWORD *)(v170 + 120));
                  }
                  v172 |= *(_QWORD *)(v170 + 112);
                  v171 |= *(_QWORD *)(v170 + 120);
                  sub_2EF2540(v1, v170, *(_DWORD *)(v166 + 112), *(_QWORD *)(v170 + 112), *(_QWORD *)(v170 + 120));
                  if ( sub_2E0A1A0(v166, v170) )
                    goto LABEL_255;
                  sub_2EEFF60(v1, "A Subrange is not covered by the main range", *(__int64 **)(v1 + 32));
                  v233 = *(_QWORD *)(v1 + 16);
                  v234 = *(void **)(v233 + 32);
                  if ( *(_QWORD *)(v233 + 24) - (_QWORD)v234 <= 0xEu )
                  {
                    v233 = sub_CB6200(*(_QWORD *)(v1 + 16), "- interval:    ", 0xFu);
                  }
                  else
                  {
                    qmemcpy(v234, "- interval:    ", 15);
                    *(_QWORD *)(v233 + 32) += 15LL;
                  }
                  v303 = v233;
                  sub_2E0B730(v166, v233);
                  v235 = *(_BYTE **)(v303 + 32);
                  if ( (unsigned __int64)v235 >= *(_QWORD *)(v303 + 24) )
                  {
                    sub_CB5D20(v303, 10);
LABEL_255:
                    v170 = *(_QWORD *)(v170 + 104);
                    if ( !v170 )
                      break;
                  }
                  else
                  {
                    *(_QWORD *)(v303 + 32) = v235 + 1;
                    *v235 = 10;
                    v170 = *(_QWORD *)(v170 + 104);
                    if ( !v170 )
                      break;
                  }
                }
              }
            }
            v236 = *(_BYTE **)(v1 + 640);
            LODWORD(v368) = 0;
            v365 = v236;
            *(_QWORD *)s = &s[16];
            *(_QWORD *)&s[8] = 0x800000000LL;
            sub_3157150(s, 0);
            v237 = sub_2E0BE90(&v365, v166);
            if ( v237 > 1 )
            {
              v239 = 0;
              sub_2EEFF60(v1, "Multiple connected components in live interval", *(__int64 **)(v1 + 32));
              sub_2EEF440(*(_QWORD *)(v1 + 16), v166);
              v312 = v166;
              v320 = v237;
              do
              {
                v240 = sub_CB59D0(*(_QWORD *)(v1 + 16), v239);
                v241 = *(_QWORD **)(v240 + 32);
                if ( *(_QWORD *)(v240 + 24) - (_QWORD)v241 <= 7u )
                {
                  sub_CB6200(v240, ": valnos", 8u);
                }
                else
                {
                  *v241 = 0x736F6E6C6176203ALL;
                  *(_QWORD *)(v240 + 32) += 8LL;
                }
                v242 = *(unsigned int ***)(v312 + 64);
                v243 = &v242[*(unsigned int *)(v312 + 72)];
                while ( v243 != v242 )
                {
                  while ( 1 )
                  {
                    v244 = *v242;
                    if ( *(_DWORD *)(*(_QWORD *)s + 4LL * **v242) == (_DWORD)v239 )
                      break;
                    if ( v243 == ++v242 )
                      goto LABEL_439;
                  }
                  v245 = *(_QWORD *)(v1 + 16);
                  v246 = *(_BYTE **)(v245 + 32);
                  if ( (unsigned __int64)v246 >= *(_QWORD *)(v245 + 24) )
                  {
                    v245 = sub_CB5D20(v245, 32);
                  }
                  else
                  {
                    *(_QWORD *)(v245 + 32) = v246 + 1;
                    *v246 = 32;
                  }
                  ++v242;
                  sub_CB59D0(v245, *v244);
                }
LABEL_439:
                v247 = *(_QWORD *)(v1 + 16);
                v248 = *(_BYTE **)(v247 + 32);
                if ( (unsigned __int64)v248 >= *(_QWORD *)(v247 + 24) )
                {
                  sub_CB5D20(v247, 10);
                }
                else
                {
                  *(_QWORD *)(v247 + 32) = v248 + 1;
                  *v248 = 10;
                }
                ++v239;
              }
              while ( v320 != v239 );
            }
            if ( *(_BYTE **)s != &s[16] )
              _libc_free(*(unsigned __int64 *)s);
          }
          else
          {
            sub_2EEFF60(v1, "Missing live interval for virtual register", *(__int64 **)(v1 + 32));
            v111 = &v358;
            v230 = *(_QWORD *)(v1 + 16);
            v110 = v328 | 0x80000000;
            sub_2FF6320(&v358, v110, *(_QWORD *)(v1 + 56), 0, 0);
            if ( !v359 )
              goto LABEL_533;
            ((void (__fastcall *)(__m128i *, __int64))v360)(&v358, v230);
            v231 = *(__m128i **)(v230 + 32);
            if ( *(_QWORD *)(v230 + 24) - (_QWORD)v231 <= 0x17u )
            {
              sub_CB6200(v230, " still has defs or uses\n", 0x18u);
            }
            else
            {
              v232 = _mm_load_si128((const __m128i *)&xmmword_4453EB0);
              v231[1].m128i_i64[0] = 0xA7365737520726FLL;
              *v231 = v232;
              *(_QWORD *)(v230 + 32) += 24LL;
            }
            if ( v359 )
              v359(&v358, &v358, 3);
          }
        }
      }
      if ( v304 == ++v328 )
        goto LABEL_261;
      v146 = *(__m128i **)(v1 + 64);
    }
  }
LABEL_266:
  v177 = *(_QWORD *)(v1 + 32);
  if ( (*(_BYTE *)(v146->m128i_i64[0] + 344) & 4) == 0 )
    goto LABEL_296;
  v284 = v177 + 320;
  v295 = *(_QWORD *)(v177 + 328);
  if ( v177 + 320 == v295 )
    goto LABEL_296;
  v178 = v1;
  do
  {
    v179 = *(unsigned int **)(v295 + 192);
    v311 = v179;
    v180 = (unsigned int *)sub_2E33140(v295);
    if ( v179 != v180 )
    {
      while ( 1 )
      {
        v181 = *v180;
        sub_E922F0(*(_QWORD **)(v178 + 56), *v180);
        if ( 2 * v182 != 2 )
          goto LABEL_271;
        v183 = *(_QWORD *)(v178 + 56);
        if ( v181 < *(_DWORD *)(v183 + 16) && *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v183 + 248) + 16LL) + v181) )
        {
          if ( (*(_QWORD *)(*(_QWORD *)(v178 + 200) + 8LL * (v181 >> 6)) & (1LL << v181)) == 0
            || v181 < *(_DWORD *)(v178 + 264) )
          {
            goto LABEL_271;
          }
        }
        else if ( v181 < *(_DWORD *)(v178 + 264)
               && (*(_QWORD *)(*(_QWORD *)(v178 + 200) + 8LL * (v181 >> 6)) & (1LL << v181)) != 0 )
        {
          goto LABEL_271;
        }
        v184 = *(__int64 **)(v295 + 64);
        v329 = &v184[*(unsigned int *)(v295 + 72)];
        if ( v184 == v329 )
        {
LABEL_271:
          v180 += 6;
          if ( v311 == v180 )
            break;
        }
        else
        {
          v305 = v181;
          v286 = v180;
          v300 = 37 * v181;
          v291 = 24LL * v181;
          do
          {
            while ( 1 )
            {
              v331 = *v184;
              v188 = sub_2EEFC50(v178 + 600, &v331);
              v189 = v188[10];
              v190 = *((_DWORD *)v188 + 24);
              if ( v190 )
              {
                v185 = v190 - 1;
                v186 = v185 & v300;
                v187 = *(_DWORD *)(v189 + 4LL * (v185 & (unsigned int)v300));
                if ( v305 == v187 )
                  goto LABEL_280;
                v223 = 1;
                while ( v187 != -1 )
                {
                  v186 = v185 & (v223 + v186);
                  v187 = *(_DWORD *)(v189 + 4LL * v186);
                  if ( v305 == v187 )
                    goto LABEL_280;
                  ++v223;
                }
              }
              sub_2EF03A0(v178, "Live in register not found to be live out from predecessor.", v295);
              v191 = *(_QWORD *)(v178 + 16);
              v192 = (char *)(*(_QWORD *)(*(_QWORD *)(v178 + 56) + 72LL)
                            + *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(v178 + 56) + 8LL) + v291));
              if ( !v192 )
                goto LABEL_356;
              v193 = strlen(v192);
              v194 = *(__m128i **)(v191 + 32);
              v195 = v193;
              v196 = *(_QWORD *)(v191 + 24) - (_QWORD)v194;
              if ( v195 > v196 )
              {
                v191 = sub_CB6200(v191, (unsigned __int8 *)v192, v195);
LABEL_356:
                v194 = *(__m128i **)(v191 + 32);
                if ( *(_QWORD *)(v191 + 24) - (_QWORD)v194 > 0x1Eu )
                  goto LABEL_287;
                goto LABEL_357;
              }
              if ( v195 )
              {
                memcpy(v194, v192, v195);
                v197 = *(_QWORD *)(v191 + 24);
                v194 = (__m128i *)(v195 + *(_QWORD *)(v191 + 32));
                *(_QWORD *)(v191 + 32) = v194;
                v196 = v197 - (_QWORD)v194;
              }
              if ( v196 > 0x1E )
              {
LABEL_287:
                v198 = _mm_load_si128((const __m128i *)&xmmword_4453EC0);
                qmemcpy(&v194[1], " live out from ", 15);
                *v194 = v198;
                *(_QWORD *)(v191 + 32) += 31LL;
                goto LABEL_288;
              }
LABEL_357:
              v191 = sub_CB6200(v191, " not found to be live out from ", 0x1Fu);
LABEL_288:
              v110 = v331;
              v111 = (__m128i *)v335;
              sub_2E31000(v335, v331);
              if ( !v335[2] )
                goto LABEL_533;
              ((void (__fastcall *)(unsigned __int64 *, __int64))v335[3])(v335, v191);
              v199 = *(_BYTE **)(v191 + 32);
              if ( (unsigned __int64)v199 >= *(_QWORD *)(v191 + 24) )
              {
                sub_CB5D20(v191, 10);
              }
              else
              {
                *(_QWORD *)(v191 + 32) = v199 + 1;
                *v199 = 10;
              }
              if ( v335[2] )
                break;
LABEL_280:
              if ( v329 == ++v184 )
                goto LABEL_293;
            }
            ++v184;
            ((void (__fastcall *)(unsigned __int64 *, unsigned __int64 *, __int64))v335[2])(v335, v335, 3);
          }
          while ( v329 != v184 );
LABEL_293:
          v180 = v286 + 6;
          if ( v311 == v286 + 6 )
            break;
        }
      }
    }
    v295 = *(_QWORD *)(v295 + 8);
  }
  while ( v284 != v295 );
  v177 = *(_QWORD *)(v178 + 32);
  v1 = v178;
LABEL_296:
  if ( *(_DWORD *)(v177 + 704) )
  {
    v257 = *(_QWORD **)(v177 + 696);
    v258 = &v257[4 * *(unsigned int *)(v177 + 712)];
    if ( v257 != v258 )
    {
      while ( 1 )
      {
        v259 = v257;
        if ( *v257 != -8192 && *v257 != -4096 )
          break;
        v257 += 4;
        if ( v258 == v257 )
          goto LABEL_297;
      }
      if ( v258 != v257 )
      {
        while ( 1 )
        {
          v260 = (_BYTE *)*v259;
          *(_QWORD *)s = &s[16];
          *(_QWORD *)&s[8] = 0x100000000LL;
          v365 = v260;
          v261 = *((_DWORD *)v259 + 4);
          if ( v261 && s != (_BYTE *)(v259 + 1) )
          {
            v263 = 8;
            v264 = &s[16];
            if ( v261 == 1
              || (sub_C8D5F0((__int64)s, &s[16], v261, 8u, (__int64)s, v261),
                  v264 = *(_BYTE **)s,
                  (v263 = 8LL * *((unsigned int *)v259 + 4)) != 0) )
            {
              memcpy(v264, (const void *)v259[1], v263);
            }
            *(_DWORD *)&s[8] = v261;
            v260 = v365;
          }
          v262 = *((_DWORD *)v260 + 11);
          if ( (v262 & 4) != 0 || (v262 & 8) == 0 )
          {
            if ( (*(_QWORD *)(*((_QWORD *)v260 + 2) + 24LL) & 0x80u) != 0LL )
              goto LABEL_471;
          }
          else if ( sub_2E88A90((__int64)v260, 128, 1) )
          {
            goto LABEL_471;
          }
          sub_2EEFF60(v1, "Call site info referencing instruction that is not call", *(__int64 **)(v1 + 32));
LABEL_471:
          if ( *(_BYTE **)s != &s[16] )
            _libc_free(*(unsigned __int64 *)s);
          v259 += 4;
          if ( v259 != v258 )
          {
            while ( *v259 == -8192 || *v259 == -4096 )
            {
              v259 += 4;
              if ( v258 == v259 )
                goto LABEL_477;
            }
            if ( v258 != v259 )
              continue;
          }
LABEL_477:
          v177 = *(_QWORD *)(v1 + 32);
          break;
        }
      }
    }
  }
LABEL_297:
  result = sub_B92180(*(_QWORD *)v177);
  if ( result )
  {
    v365 = 0;
    v201 = *(_QWORD *)(v1 + 32);
    memset(s, 0, sizeof(s));
    v201 += 320;
    v202 = *(_QWORD *)(v201 + 8);
    v319 = v201;
    if ( v201 == v202 )
    {
      v204 = 0;
      v208 = 0;
      return sub_C7D6A0(v204, v208, 4);
    }
    v330 = v1;
    v203 = 0;
    v204 = 0;
    while ( 1 )
    {
      v205 = *(_QWORD *)(v202 + 56);
      v206 = v202 + 48;
      if ( v202 + 48 != v205 )
        break;
LABEL_306:
      v202 = *(_QWORD *)(v202 + 8);
      if ( v319 == v202 )
      {
        v208 = 4LL * v203;
        return sub_C7D6A0(v204, v208, 4);
      }
    }
    while ( 1 )
    {
      v207 = *(_DWORD *)(v205 + 64);
      if ( v207 )
        break;
LABEL_304:
      if ( (*(_BYTE *)v205 & 4) != 0 )
      {
        v205 = *(_QWORD *)(v205 + 8);
        if ( v206 == v205 )
          goto LABEL_306;
      }
      else
      {
        while ( (*(_BYTE *)(v205 + 44) & 8) != 0 )
          v205 = *(_QWORD *)(v205 + 8);
        v205 = *(_QWORD *)(v205 + 8);
        if ( v206 == v205 )
          goto LABEL_306;
      }
    }
    if ( v203 )
    {
      v209 = 1;
      LODWORD(v210) = (v203 - 1) & (37 * v207);
      v211 = (_DWORD *)(v204 + 4LL * (unsigned int)v210);
      v212 = 0;
      v213 = *v211;
      if ( v207 == *v211 )
      {
LABEL_327:
        sub_2EF06E0(v330, "Instruction has a duplicated value tracking number", v205);
LABEL_328:
        v203 = *(_DWORD *)&s[16];
        v204 = *(_QWORD *)s;
        goto LABEL_304;
      }
      while ( v213 != -1 )
      {
        if ( !v212 && v213 == -2 )
          v212 = v211;
        v210 = (v203 - 1) & ((_DWORD)v210 + v209);
        v211 = (_DWORD *)(v204 + 4 * v210);
        v213 = *v211;
        if ( v207 == *v211 )
          goto LABEL_327;
        ++v209;
      }
      if ( !v212 )
        v212 = v211;
      ++v365;
      v238 = *(_DWORD *)&s[8] + 1;
      if ( 4 * (*(_DWORD *)&s[8] + 1) < 3 * v203 )
      {
        if ( v203 - (v238 + *(_DWORD *)&s[12]) <= v203 >> 3 )
        {
          sub_A08C50((__int64)&v365, v203);
          if ( !*(_DWORD *)&s[16] )
          {
LABEL_545:
            ++*(_DWORD *)&s[8];
            BUG();
          }
          v253 = 1;
          v254 = (*(_DWORD *)&s[16] - 1) & (37 * v207);
          v238 = *(_DWORD *)&s[8] + 1;
          v255 = 0;
          v212 = (_DWORD *)(*(_QWORD *)s + 4LL * v254);
          v256 = *v212;
          if ( v207 != *v212 )
          {
            while ( v256 != -1 )
            {
              if ( v256 == -2 && !v255 )
                v255 = v212;
              v254 = (*(_DWORD *)&s[16] - 1) & (v253 + v254);
              v212 = (_DWORD *)(*(_QWORD *)s + 4LL * v254);
              v256 = *v212;
              if ( v207 == *v212 )
                goto LABEL_423;
              ++v253;
            }
            if ( v255 )
              v212 = v255;
          }
        }
        goto LABEL_423;
      }
    }
    else
    {
      ++v365;
    }
    sub_A08C50((__int64)&v365, 2 * v203);
    if ( !*(_DWORD *)&s[16] )
      goto LABEL_545;
    v249 = (*(_DWORD *)&s[16] - 1) & (37 * v207);
    v238 = *(_DWORD *)&s[8] + 1;
    v212 = (_DWORD *)(*(_QWORD *)s + 4LL * v249);
    v250 = *v212;
    if ( v207 != *v212 )
    {
      v251 = 1;
      v252 = 0;
      while ( v250 != -1 )
      {
        if ( !v252 && v250 == -2 )
          v252 = v212;
        v249 = (*(_DWORD *)&s[16] - 1) & (v251 + v249);
        v212 = (_DWORD *)(*(_QWORD *)s + 4LL * v249);
        v250 = *v212;
        if ( v207 == *v212 )
          goto LABEL_423;
        ++v251;
      }
      if ( v252 )
        v212 = v252;
    }
LABEL_423:
    *(_DWORD *)&s[8] = v238;
    if ( *v212 != -1 )
      --*(_DWORD *)&s[12];
    *v212 = v207;
    goto LABEL_328;
  }
  return result;
}
