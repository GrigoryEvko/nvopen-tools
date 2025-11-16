// Function: sub_1F53550
// Address: 0x1f53550
//
_BOOL8 __fastcall sub_1F53550(__int64 a1, __int64 *a2)
{
  __int64 v2; // rdx
  __int64 v3; // r14
  __int64 v4; // r12
  __int64 v5; // rdi
  __int64 (*v6)(); // rcx
  __int64 *v7; // rax
  __int64 v8; // rdi
  __int64 (*v9)(); // rcx
  __int64 v10; // rdx
  __int64 (*v11)(void); // rdx
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  _QWORD *v24; // rax
  char **v25; // rax
  __int64 v26; // rax
  int v27; // eax
  __int64 v28; // rdx
  _QWORD *v29; // rax
  _QWORD *i; // rdx
  int v31; // eax
  __int64 v32; // rdx
  _DWORD *v33; // rax
  _DWORD *j; // rdx
  int v35; // eax
  __int64 v36; // rdx
  _DWORD *v37; // rax
  _DWORD *m; // rdx
  void *v39; // rdi
  unsigned int v40; // eax
  __int64 v41; // rdx
  void *v42; // rdi
  unsigned int v43; // eax
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // r15
  __int64 v47; // rbx
  __int64 v48; // rax
  char **v49; // r12
  char **v50; // rbx
  unsigned __int64 v51; // rdi
  _QWORD *v53; // rdx
  _QWORD *v54; // rax
  _QWORD *v55; // r12
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // r12
  unsigned int v59; // esi
  __int64 v60; // rdi
  unsigned int v61; // edx
  __int64 v62; // rax
  __int64 v63; // r13
  __int64 v64; // rdx
  __int16 v65; // ax
  __int64 v66; // rax
  int v67; // r12d
  int v68; // eax
  __int64 *v69; // rax
  __int64 v70; // r15
  __int64 v71; // r12
  __int64 v72; // r14
  int v73; // ecx
  unsigned int v74; // edi
  int *v75; // rbx
  int v76; // r9d
  unsigned __int64 v77; // r13
  __int64 v78; // rax
  __int64 v79; // rdx
  __int64 v80; // rax
  __int64 v81; // rbx
  int v82; // r11d
  unsigned __int8 v83; // r9
  __int64 v84; // rcx
  unsigned int v85; // eax
  char v86; // si
  char *v87; // rdx
  char **v88; // rax
  char *v89; // rax
  __int64 v90; // r14
  bool v91; // al
  char **v92; // rdx
  __int64 v93; // rax
  char *v94; // rax
  __int64 v95; // rdx
  __int64 v96; // r11
  char *v97; // rax
  int v98; // r12d
  __int64 v99; // rcx
  __int64 v100; // rax
  unsigned int *v101; // rsi
  __int64 v102; // rdx
  __int64 v103; // rdx
  __int64 v104; // rbx
  unsigned int *v105; // rdx
  int v106; // r13d
  _DWORD *v107; // rax
  int v108; // r15d
  __int64 v109; // rsi
  __int64 v110; // r12
  unsigned __int64 v111; // rax
  unsigned __int64 v112; // rdx
  unsigned __int64 v113; // r12
  __int64 ii; // r12
  unsigned int v115; // esi
  unsigned int v116; // edi
  __int64 v117; // rcx
  unsigned int v118; // r10d
  unsigned __int64 *v119; // rax
  unsigned __int64 v120; // r9
  unsigned int v121; // r8d
  __int64 *v122; // rax
  __int64 v123; // rdx
  __int64 v124; // rax
  __int64 v125; // r12
  unsigned int v126; // esi
  __int64 v127; // rdi
  unsigned int *v128; // rax
  unsigned int v129; // edx
  __int64 v130; // rdi
  char *v131; // rdi
  char *v132; // rax
  _BYTE *v133; // rax
  __int64 v134; // r8
  unsigned __int64 v135; // r9
  unsigned __int64 v136; // rdx
  unsigned __int64 v137; // rbx
  __int64 jj; // rbx
  __int64 v139; // r12
  char v140; // al
  __int64 v141; // r12
  unsigned __int64 v142; // rax
  unsigned int v143; // edx
  __int64 v144; // r15
  __int64 v145; // r8
  __int64 *v146; // r13
  __int64 v147; // rcx
  unsigned __int64 v148; // rax
  __int64 v149; // rsi
  unsigned int v150; // ecx
  unsigned int v151; // r9d
  __int64 *v152; // rdx
  __int64 v153; // rdi
  __int64 v154; // rbx
  __int64 v155; // rax
  unsigned __int64 v156; // rdx
  char *v157; // rax
  __int64 v158; // rbx
  __int64 v159; // r13
  __int64 v160; // r14
  char v161; // al
  __int64 v162; // rdi
  __int64 v163; // rdx
  __int64 v164; // rax
  __int64 v165; // r8
  unsigned int v166; // r10d
  unsigned __int64 v167; // rsi
  __int64 v168; // r9
  __int64 v169; // rax
  __int64 v170; // r12
  __int64 v171; // r8
  __int64 v172; // r9
  __int64 v173; // rax
  __int64 v174; // rcx
  unsigned __int64 v175; // rax
  __int64 v176; // rdi
  __int64 v177; // rcx
  unsigned int v178; // esi
  __int64 *v179; // rdx
  __int64 v180; // rax
  unsigned int v181; // r10d
  __int64 v182; // r12
  __int64 v183; // rax
  __int64 v184; // rdx
  int v185; // edx
  int v186; // r10d
  int v187; // edi
  int v188; // edx
  int v189; // r10d
  __int64 *v190; // r9
  int v191; // edi
  int v192; // edx
  int v193; // r11d
  unsigned __int64 *v194; // r8
  int v195; // eax
  int v196; // ecx
  int v197; // r9d
  int v198; // r9d
  __int64 v199; // r8
  unsigned int v200; // ecx
  __int64 v201; // rsi
  int v202; // r11d
  __int64 *v203; // r10
  int v204; // r9d
  __int64 v205; // rdi
  unsigned int v206; // esi
  int v207; // r11d
  unsigned int *v208; // r10
  int v209; // r9d
  int v210; // r9d
  __int64 v211; // r8
  int v212; // r11d
  unsigned int v213; // ecx
  __int64 v214; // rsi
  int v215; // r8d
  __int64 v216; // rsi
  unsigned int v217; // r12d
  int v218; // r10d
  int v219; // r10d
  int v220; // r10d
  __int64 v221; // r9
  unsigned int v222; // eax
  unsigned __int64 v223; // rdx
  int v224; // r11d
  unsigned __int64 *v225; // rdi
  unsigned int v226; // r13d
  __int64 v227; // rbx
  int v228; // r10d
  int v229; // r10d
  __int64 v230; // r9
  int v231; // r11d
  unsigned int v232; // edx
  unsigned __int64 v233; // rax
  unsigned int v234; // edi
  int v235; // r8d
  unsigned int v236; // r9d
  unsigned int v237; // esi
  __int64 v238; // rdx
  int v239; // edx
  __int64 v240; // rdx
  char *v241; // rdx
  _QWORD *v242; // rcx
  _QWORD *v243; // rdx
  __int64 v244; // rsi
  __int64 v245; // rdx
  int v246; // r11d
  int *v247; // r10
  __int64 v248; // rax
  _DWORD *v249; // rdi
  unsigned int v250; // eax
  int v251; // eax
  unsigned __int64 v252; // rax
  unsigned __int64 v253; // rax
  int v254; // ebx
  __int64 v255; // r12
  _DWORD *v256; // rax
  __int64 v257; // rdx
  _DWORD *k; // rdx
  _QWORD *v259; // rdi
  unsigned int v260; // eax
  int v261; // eax
  unsigned __int64 v262; // rax
  unsigned __int64 v263; // rax
  int v264; // ebx
  __int64 v265; // r12
  _QWORD *v266; // rax
  __int64 v267; // rdx
  _QWORD *kk; // rdx
  _DWORD *v269; // rdi
  unsigned int v270; // eax
  int v271; // eax
  unsigned __int64 v272; // rax
  unsigned __int64 v273; // rax
  int v274; // ebx
  __int64 v275; // r12
  _DWORD *v276; // rax
  __int64 v277; // rdx
  _DWORD *n; // rdx
  __int64 v279; // rax
  bool v280; // zf
  unsigned int v281; // ebx
  int v282; // ebx
  __int64 v283; // r13
  __int64 v284; // rax
  __int64 v285; // rbx
  unsigned int v286; // r13d
  __int64 v287; // r15
  __int64 v288; // r10
  int v289; // r8d
  int v290; // eax
  char *v291; // rdi
  unsigned int v292; // eax
  const __m128i *v293; // r14
  __int32 v294; // r8d
  __int64 v295; // rsi
  __int64 v296; // rdx
  __int64 v297; // rax
  __int64 v298; // rax
  unsigned __int64 v299; // r13
  signed int v300; // ebx
  __int64 v301; // rdi
  unsigned int v302; // eax
  __int64 v303; // rcx
  char **v304; // r9
  int v305; // esi
  unsigned int v306; // ecx
  int v307; // r10d
  int *v308; // rdi
  __int64 v309; // rax
  __int64 *v310; // rsi
  unsigned int v311; // edi
  char **v312; // r9
  int v313; // esi
  unsigned int v314; // ecx
  int v315; // r10d
  __int64 v316; // rcx
  _QWORD *v317; // rdx
  _QWORD *v318; // rax
  __int64 v319; // rbx
  __int64 v320; // rdx
  __int64 v321; // rcx
  __int64 v322; // r8
  _BYTE *v323; // r9
  char *v324; // rax
  _BYTE *v325; // rsi
  int v326; // r10d
  __int64 v327; // rcx
  int v328; // edi
  int v329; // edx
  int v330; // r11d
  int v331; // r11d
  __int64 v332; // r10
  unsigned int v333; // ecx
  __int64 v334; // r8
  int v335; // edi
  __int64 v336; // rsi
  int v337; // r10d
  int v338; // r10d
  __int64 v339; // r9
  __int64 v340; // rcx
  unsigned int v341; // ebx
  int v342; // esi
  __int64 v343; // rdi
  int v344; // r11d
  int v345; // r8d
  _DWORD *v346; // rax
  _QWORD *v347; // rax
  _DWORD *v348; // rax
  __int128 v349; // [rsp-20h] [rbp-280h]
  __int64 v350; // [rsp-8h] [rbp-268h]
  __int64 v351; // [rsp+18h] [rbp-248h]
  __int64 v352; // [rsp+20h] [rbp-240h]
  __int64 v353; // [rsp+28h] [rbp-238h]
  __int64 v354; // [rsp+30h] [rbp-230h]
  __int64 v355; // [rsp+38h] [rbp-228h]
  __int64 v356; // [rsp+48h] [rbp-218h]
  __int64 v357; // [rsp+50h] [rbp-210h]
  __int64 v358; // [rsp+58h] [rbp-208h]
  char *v359; // [rsp+60h] [rbp-200h]
  char *v360; // [rsp+68h] [rbp-1F8h]
  __int64 v361; // [rsp+70h] [rbp-1F0h]
  unsigned int v362; // [rsp+78h] [rbp-1E8h]
  char v363; // [rsp+7Eh] [rbp-1E2h]
  char v364; // [rsp+7Fh] [rbp-1E1h]
  char v365; // [rsp+80h] [rbp-1E0h]
  __int64 v366; // [rsp+80h] [rbp-1E0h]
  unsigned int v367; // [rsp+88h] [rbp-1D8h]
  __int64 v368; // [rsp+90h] [rbp-1D0h]
  __int64 v369; // [rsp+90h] [rbp-1D0h]
  __int64 *v370; // [rsp+90h] [rbp-1D0h]
  __int64 v371; // [rsp+90h] [rbp-1D0h]
  unsigned __int64 v372; // [rsp+98h] [rbp-1C8h]
  __int32 v373; // [rsp+98h] [rbp-1C8h]
  bool v374; // [rsp+A0h] [rbp-1C0h]
  __int64 *v375; // [rsp+A0h] [rbp-1C0h]
  __int64 v376; // [rsp+A8h] [rbp-1B8h]
  unsigned __int64 v377; // [rsp+A8h] [rbp-1B8h]
  int v378; // [rsp+A8h] [rbp-1B8h]
  __int64 v379; // [rsp+A8h] [rbp-1B8h]
  __int64 v380; // [rsp+B0h] [rbp-1B0h]
  __int64 v381; // [rsp+B0h] [rbp-1B0h]
  __int64 v382; // [rsp+B0h] [rbp-1B0h]
  __int64 v383; // [rsp+B0h] [rbp-1B0h]
  char v384; // [rsp+B8h] [rbp-1A8h]
  int v385; // [rsp+B8h] [rbp-1A8h]
  unsigned int v386; // [rsp+BCh] [rbp-1A4h]
  char *v387; // [rsp+C0h] [rbp-1A0h]
  unsigned __int8 v388; // [rsp+C0h] [rbp-1A0h]
  char v389; // [rsp+C0h] [rbp-1A0h]
  __int64 v390; // [rsp+C0h] [rbp-1A0h]
  __int64 v391; // [rsp+C0h] [rbp-1A0h]
  __int64 v392; // [rsp+C8h] [rbp-198h]
  __int64 v393; // [rsp+C8h] [rbp-198h]
  __int64 v394; // [rsp+C8h] [rbp-198h]
  __int16 v395; // [rsp+D0h] [rbp-190h]
  unsigned __int8 v396; // [rsp+D0h] [rbp-190h]
  __int64 v397; // [rsp+D0h] [rbp-190h]
  __int64 v398; // [rsp+D0h] [rbp-190h]
  int v399; // [rsp+D0h] [rbp-190h]
  _QWORD *v400; // [rsp+D0h] [rbp-190h]
  __int64 v401; // [rsp+D0h] [rbp-190h]
  unsigned int v402; // [rsp+D0h] [rbp-190h]
  __int64 v403; // [rsp+D0h] [rbp-190h]
  int v404; // [rsp+D0h] [rbp-190h]
  unsigned int v405; // [rsp+D0h] [rbp-190h]
  int v406; // [rsp+D0h] [rbp-190h]
  int v407; // [rsp+D0h] [rbp-190h]
  __int64 v408; // [rsp+D0h] [rbp-190h]
  __int64 v409; // [rsp+D8h] [rbp-188h]
  __int64 v410; // [rsp+D8h] [rbp-188h]
  int v411; // [rsp+D8h] [rbp-188h]
  __int64 v412; // [rsp+D8h] [rbp-188h]
  char v413; // [rsp+D8h] [rbp-188h]
  unsigned __int64 v414; // [rsp+E0h] [rbp-180h] BYREF
  unsigned __int64 v415; // [rsp+E8h] [rbp-178h] BYREF
  int *v416; // [rsp+F0h] [rbp-170h] BYREF
  __int64 v417; // [rsp+F8h] [rbp-168h]
  _DWORD v418[4]; // [rsp+100h] [rbp-160h] BYREF
  __m128i v419; // [rsp+110h] [rbp-150h] BYREF
  __int64 v420; // [rsp+120h] [rbp-140h]
  __int64 v421; // [rsp+128h] [rbp-138h]
  __int64 v422; // [rsp+130h] [rbp-130h]
  __int64 v423; // [rsp+140h] [rbp-120h] BYREF
  __int64 v424; // [rsp+148h] [rbp-118h]
  char *v425; // [rsp+150h] [rbp-110h] BYREF
  unsigned int v426; // [rsp+158h] [rbp-108h]
  _BYTE v427[48]; // [rsp+230h] [rbp-30h] BYREF

  v2 = 0;
  v3 = a1;
  *(_QWORD *)(a1 + 232) = a2;
  v4 = a2[1];
  *(_QWORD *)(a1 + 264) = a2[5];
  v5 = a2[2];
  v6 = *(__int64 (**)())(*(_QWORD *)v5 + 40LL);
  v7 = a2;
  if ( v6 != sub_1D00B00 )
  {
    v2 = ((__int64 (__fastcall *)(__int64, __int64 *, _QWORD))v6)(v5, a2, 0);
    v7 = *(__int64 **)(v3 + 232);
  }
  *(_QWORD *)(v3 + 240) = v2;
  v8 = v7[2];
  v9 = *(__int64 (**)())(*(_QWORD *)v8 + 112LL);
  v10 = 0;
  if ( v9 != sub_1D00B10 )
  {
    v10 = ((__int64 (__fastcall *)(__int64, __int64 *, _QWORD))v9)(v8, a2, 0);
    v7 = *(__int64 **)(v3 + 232);
  }
  *(_QWORD *)(v3 + 248) = v10;
  v11 = *(__int64 (**)(void))(*(_QWORD *)v7[2] + 128LL);
  v12 = 0;
  if ( v11 != sub_1D0B140 )
    v12 = v11();
  v13 = *(_QWORD *)(v3 + 8);
  *(_QWORD *)(v3 + 256) = v12;
  v14 = sub_160F9A0(v13, (__int64)&unk_4FC4534, 1u);
  v15 = v14;
  if ( v14 )
    v15 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v14 + 104LL))(v14, &unk_4FC4534);
  *(_QWORD *)(v3 + 272) = v15;
  v16 = sub_160F9A0(*(_QWORD *)(v3 + 8), (__int64)&unk_4FC450C, 1u);
  v17 = v16;
  if ( v16 )
    v17 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v16 + 104LL))(v16, &unk_4FC450C);
  *(_QWORD *)(v3 + 280) = v17;
  v18 = sub_160F9A0(*(_QWORD *)(v3 + 8), (__int64)&unk_4F96DB4, 1u);
  if ( v18 && (v19 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v18 + 104LL))(v18, &unk_4F96DB4)) != 0 )
    v20 = *(_QWORD *)(v19 + 160);
  else
    v20 = 0;
  *(_QWORD *)(v3 + 288) = v20;
  *(_DWORD *)(v3 + 296) = sub_1700720(v4);
  if ( (unsigned __int8)sub_1636880(v3, *a2) )
    *(_DWORD *)(v3 + 296) = 0;
  v24 = *(_QWORD **)(**(_QWORD **)(v3 + 264) + 352LL);
  *v24 &= ~1uLL;
  v25 = &v425;
  v423 = 0;
  v424 = 1;
  do
  {
    *(_DWORD *)v25 = -1;
    v25 += 7;
  }
  while ( v25 != (char **)v427 );
  v26 = *(_QWORD *)(v3 + 232);
  v352 = v26 + 320;
  v353 = *(_QWORD *)(v26 + 328);
  if ( v353 != v26 + 320 )
  {
    v374 = 0;
    v358 = v3 + 312;
    v356 = v3 + 552;
    v351 = v3 + 584;
    v355 = v3 + 344;
    v354 = v3 + 448;
    while ( 1 )
    {
      ++*(_QWORD *)(v3 + 312);
      *(_QWORD *)(v3 + 304) = v353;
      v27 = *(_DWORD *)(v3 + 328);
      if ( !v27 )
      {
        v22 = *(unsigned int *)(v3 + 332);
        if ( !(_DWORD)v22 )
          goto LABEL_26;
        v28 = *(unsigned int *)(v3 + 336);
        if ( (unsigned int)v28 > 0x40 )
        {
          j___libc_free_0(*(_QWORD *)(v3 + 320));
          *(_QWORD *)(v3 + 320) = 0;
          *(_QWORD *)(v3 + 328) = 0;
          *(_DWORD *)(v3 + 336) = 0;
          goto LABEL_26;
        }
LABEL_23:
        v29 = *(_QWORD **)(v3 + 320);
        for ( i = &v29[2 * v28]; i != v29; v29 += 2 )
          *v29 = -8;
        *(_QWORD *)(v3 + 328) = 0;
        goto LABEL_26;
      }
      v21 = (unsigned int)(4 * v27);
      v28 = *(unsigned int *)(v3 + 336);
      if ( (unsigned int)v21 < 0x40 )
        v21 = 64;
      if ( (unsigned int)v21 >= (unsigned int)v28 )
        goto LABEL_23;
      v259 = *(_QWORD **)(v3 + 320);
      v260 = v27 - 1;
      if ( !v260 )
        break;
      _BitScanReverse(&v260, v260);
      v21 = 33 - (v260 ^ 0x1F);
      v261 = 1 << (33 - (v260 ^ 0x1F));
      if ( v261 < 64 )
        v261 = 64;
      if ( (_DWORD)v28 != v261 )
      {
        v262 = (4 * v261 / 3u + 1) | ((unsigned __int64)(4 * v261 / 3u + 1) >> 1);
        v263 = ((v262 | (v262 >> 2)) >> 4)
             | v262
             | (v262 >> 2)
             | ((((v262 | (v262 >> 2)) >> 4) | v262 | (v262 >> 2)) >> 8);
        v264 = (v263 | (v263 >> 16)) + 1;
        v265 = 16 * ((v263 | (v263 >> 16)) + 1);
        goto LABEL_374;
      }
      *(_QWORD *)(v3 + 328) = 0;
      v347 = &v259[2 * (unsigned int)v28];
      do
      {
        if ( v259 )
          *v259 = -8;
        v259 += 2;
      }
      while ( v347 != v259 );
LABEL_26:
      v31 = *(_DWORD *)(v3 + 568);
      ++*(_QWORD *)(v3 + 552);
      if ( !v31 )
      {
        if ( !*(_DWORD *)(v3 + 572) )
          goto LABEL_32;
        v32 = *(unsigned int *)(v3 + 576);
        if ( (unsigned int)v32 > 0x40 )
        {
          j___libc_free_0(*(_QWORD *)(v3 + 560));
          *(_QWORD *)(v3 + 560) = 0;
          *(_QWORD *)(v3 + 568) = 0;
          *(_DWORD *)(v3 + 576) = 0;
          goto LABEL_32;
        }
LABEL_29:
        v33 = *(_DWORD **)(v3 + 560);
        for ( j = &v33[2 * v32]; j != v33; v33 += 2 )
          *v33 = -1;
        *(_QWORD *)(v3 + 568) = 0;
        goto LABEL_32;
      }
      v21 = (unsigned int)(4 * v31);
      v32 = *(unsigned int *)(v3 + 576);
      if ( (unsigned int)v21 < 0x40 )
        v21 = 64;
      if ( (unsigned int)v32 <= (unsigned int)v21 )
        goto LABEL_29;
      v249 = *(_DWORD **)(v3 + 560);
      v250 = v31 - 1;
      if ( !v250 )
      {
        v255 = 1024;
        v254 = 128;
LABEL_361:
        j___libc_free_0(v249);
        *(_DWORD *)(v3 + 576) = v254;
        v256 = (_DWORD *)sub_22077B0(v255);
        v257 = *(unsigned int *)(v3 + 576);
        *(_QWORD *)(v3 + 568) = 0;
        *(_QWORD *)(v3 + 560) = v256;
        for ( k = &v256[2 * v257]; k != v256; v256 += 2 )
        {
          if ( v256 )
            *v256 = -1;
        }
        goto LABEL_32;
      }
      _BitScanReverse(&v250, v250);
      v21 = 33 - (v250 ^ 0x1F);
      v251 = 1 << (33 - (v250 ^ 0x1F));
      if ( v251 < 64 )
        v251 = 64;
      if ( (_DWORD)v32 != v251 )
      {
        v252 = (4 * v251 / 3u + 1) | ((unsigned __int64)(4 * v251 / 3u + 1) >> 1);
        v253 = ((v252 | (v252 >> 2)) >> 4)
             | v252
             | (v252 >> 2)
             | ((((v252 | (v252 >> 2)) >> 4) | v252 | (v252 >> 2)) >> 8);
        v254 = (v253 | (v253 >> 16)) + 1;
        v255 = 8 * ((v253 | (v253 >> 16)) + 1);
        goto LABEL_361;
      }
      *(_QWORD *)(v3 + 568) = 0;
      v346 = &v249[2 * v32];
      do
      {
        if ( v249 )
          *v249 = -1;
        v249 += 2;
      }
      while ( v346 != v249 );
LABEL_32:
      v35 = *(_DWORD *)(v3 + 600);
      ++*(_QWORD *)(v3 + 584);
      if ( !v35 )
      {
        if ( !*(_DWORD *)(v3 + 604) )
          goto LABEL_38;
        v36 = *(unsigned int *)(v3 + 608);
        if ( (unsigned int)v36 > 0x40 )
        {
          j___libc_free_0(*(_QWORD *)(v3 + 592));
          *(_QWORD *)(v3 + 592) = 0;
          *(_QWORD *)(v3 + 600) = 0;
          *(_DWORD *)(v3 + 608) = 0;
          goto LABEL_38;
        }
LABEL_35:
        v37 = *(_DWORD **)(v3 + 592);
        for ( m = &v37[2 * v36]; m != v37; v37 += 2 )
          *v37 = -1;
        *(_QWORD *)(v3 + 600) = 0;
        goto LABEL_38;
      }
      v21 = (unsigned int)(4 * v35);
      v36 = *(unsigned int *)(v3 + 608);
      if ( (unsigned int)v21 < 0x40 )
        v21 = 64;
      if ( (unsigned int)v36 <= (unsigned int)v21 )
        goto LABEL_35;
      v269 = *(_DWORD **)(v3 + 592);
      v270 = v35 - 1;
      if ( !v270 )
      {
        v275 = 1024;
        v274 = 128;
LABEL_387:
        j___libc_free_0(v269);
        *(_DWORD *)(v3 + 608) = v274;
        v276 = (_DWORD *)sub_22077B0(v275);
        v277 = *(unsigned int *)(v3 + 608);
        *(_QWORD *)(v3 + 600) = 0;
        *(_QWORD *)(v3 + 592) = v276;
        for ( n = &v276[2 * v277]; n != v276; v276 += 2 )
        {
          if ( v276 )
            *v276 = -1;
        }
        goto LABEL_38;
      }
      _BitScanReverse(&v270, v270);
      v21 = 33 - (v270 ^ 0x1F);
      v271 = 1 << (33 - (v270 ^ 0x1F));
      if ( v271 < 64 )
        v271 = 64;
      if ( (_DWORD)v36 != v271 )
      {
        v272 = (4 * v271 / 3u + 1) | ((unsigned __int64)(4 * v271 / 3u + 1) >> 1);
        v273 = ((v272 | (v272 >> 2)) >> 4)
             | v272
             | (v272 >> 2)
             | ((((v272 | (v272 >> 2)) >> 4) | v272 | (v272 >> 2)) >> 8);
        v274 = (v273 | (v273 >> 16)) + 1;
        v275 = 8 * ((v273 | (v273 >> 16)) + 1);
        goto LABEL_387;
      }
      *(_QWORD *)(v3 + 600) = 0;
      v348 = &v269[2 * v36];
      do
      {
        if ( v269 )
          *v269 = -1;
        v269 += 2;
      }
      while ( v348 != v269 );
LABEL_38:
      ++*(_QWORD *)(v3 + 344);
      v39 = *(void **)(v3 + 360);
      if ( v39 == *(void **)(v3 + 352) )
        goto LABEL_43;
      v40 = 4 * (*(_DWORD *)(v3 + 372) - *(_DWORD *)(v3 + 376));
      v41 = *(unsigned int *)(v3 + 368);
      if ( v40 < 0x20 )
        v40 = 32;
      if ( (unsigned int)v41 <= v40 )
      {
        memset(v39, -1, 8 * v41);
LABEL_43:
        *(_QWORD *)(v3 + 372) = 0;
        goto LABEL_44;
      }
      sub_16CC920(v355);
LABEL_44:
      ++*(_QWORD *)(v3 + 448);
      v42 = *(void **)(v3 + 464);
      if ( v42 == *(void **)(v3 + 456) )
        goto LABEL_49;
      v43 = 4 * (*(_DWORD *)(v3 + 476) - *(_DWORD *)(v3 + 480));
      v44 = *(unsigned int *)(v3 + 472);
      if ( v43 < 0x20 )
        v43 = 32;
      if ( (unsigned int)v44 <= v43 )
      {
        memset(v42, -1, 8 * v44);
LABEL_49:
        *(_QWORD *)(v3 + 476) = 0;
        goto LABEL_50;
      }
      sub_16CC920(v354);
LABEL_50:
      v45 = *(_QWORD *)(v3 + 304);
      v46 = v3;
      v386 = 0;
      v47 = *(_QWORD *)(v45 + 32);
      v357 = v45 + 24;
      v414 = v47;
      if ( v47 != v45 + 24 )
      {
        while ( 1 )
        {
          if ( !v47 )
            BUG();
          v48 = v47;
          if ( (*(_BYTE *)v47 & 4) == 0 && (*(_BYTE *)(v47 + 46) & 8) != 0 )
          {
            do
              v48 = *(_QWORD *)(v48 + 8);
            while ( (*(_BYTE *)(v48 + 46) & 8) != 0 );
          }
          v415 = *(_QWORD *)(v48 + 8);
          if ( (unsigned __int16)(**(_WORD **)(v47 + 16) - 12) <= 1u )
            goto LABEL_56;
          v53 = *(_QWORD **)(v46 + 464);
          v54 = *(_QWORD **)(v46 + 456);
          if ( v53 == v54 )
          {
            v55 = &v54[*(unsigned int *)(v46 + 476)];
            if ( v54 == v55 )
            {
              v57 = *(_QWORD *)(v46 + 456);
            }
            else
            {
              do
              {
                if ( *v54 == v47 )
                  break;
                ++v54;
              }
              while ( v55 != v54 );
              v57 = (__int64)v55;
            }
          }
          else
          {
            v55 = &v53[*(unsigned int *)(v46 + 472)];
            v54 = sub_16CC9F0(v354, v47);
            if ( *v54 == v47 )
            {
              v240 = *(_QWORD *)(v46 + 464);
              if ( v240 == *(_QWORD *)(v46 + 456) )
                v21 = *(unsigned int *)(v46 + 476);
              else
                v21 = *(unsigned int *)(v46 + 472);
              v57 = v240 + 8 * v21;
            }
            else
            {
              v56 = *(_QWORD *)(v46 + 464);
              if ( v56 != *(_QWORD *)(v46 + 456) )
              {
                v57 = *(unsigned int *)(v46 + 472);
                v54 = (_QWORD *)(v56 + 8 * v57);
                goto LABEL_75;
              }
              v54 = (_QWORD *)(v56 + 8LL * *(unsigned int *)(v46 + 476));
              v57 = (__int64)v54;
            }
          }
          while ( (_QWORD *)v57 != v54 && *v54 >= 0xFFFFFFFFFFFFFFFELL )
            ++v54;
LABEL_75:
          if ( v55 != v54 )
            goto LABEL_56;
          v58 = v414;
          if ( **(_WORD **)(v414 + 16) == 14 )
          {
            v279 = *(_QWORD *)(v414 + 32);
            v280 = *(_QWORD *)(v46 + 280) == 0;
            v373 = *(_DWORD *)(v279 + 8);
            v281 = *(_DWORD *)(v414 + 40);
            v416 = v418;
            v405 = v281;
            v417 = 0x400000000LL;
            if ( v280 )
              goto LABEL_400;
            v57 = *(unsigned int *)(v279 + 8);
            LODWORD(v417) = 1;
            v418[0] = v57;
            if ( v281 <= 1 )
              goto LABEL_531;
            v282 = *(_DWORD *)(v279 + 48);
            v57 = (__int64)v418;
            v283 = 120;
            v284 = 1;
            while ( 1 )
            {
              *(_DWORD *)(v57 + 4 * v284) = v282;
              v284 = (unsigned int)(v417 + 1);
              LODWORD(v417) = v417 + 1;
              if ( v283 == 80LL * ((v405 - 2) >> 1) + 120 )
                break;
              v282 = *(_DWORD *)(*(_QWORD *)(v58 + 32) + v283 + 8);
              if ( HIDWORD(v417) <= (unsigned int)v284 )
              {
                sub_16CD150((__int64)&v416, v418, 0, 4, v22, v23);
                v284 = (unsigned int)v417;
              }
              v57 = (__int64)v416;
              v283 += 80;
            }
            v405 = *(_DWORD *)(v58 + 40);
LABEL_400:
            if ( v405 <= 1 )
            {
LABEL_531:
              v413 = (*(__int64 *)v58 >> 2) & 1;
              if ( ((*(__int64 *)v58 >> 2) & 1) == 0 && (*(_BYTE *)(v58 + 46) & 8) != 0 )
              {
LABEL_416:
                v298 = v58;
                do
                  v298 = *(_QWORD *)(v298 + 8);
                while ( (*(_BYTE *)(v298 + 46) & 8) != 0 );
                goto LABEL_418;
              }
              v299 = *(_QWORD *)(v58 + 8);
LABEL_419:
              *(_QWORD *)(v58 + 16) = *(_QWORD *)(*(_QWORD *)(v46 + 240) + 8LL) + 576LL;
              v300 = *(_DWORD *)(v58 + 40) - 1;
              if ( v300 > 0 )
              {
                do
                  sub_1E16C90(v58, v300--, v57, v21, v22, (_BYTE *)v23);
                while ( v300 );
              }
LABEL_421:
              v301 = *(_QWORD *)(v46 + 280);
              if ( v301 )
                sub_1DBF6C0(v301, *(_QWORD *)(v46 + 304), v414, v299, v416, (unsigned int)v417);
              if ( v416 != v418 )
                _libc_free((unsigned __int64)v416);
              v58 = v414;
              goto LABEL_77;
            }
            v285 = 40;
            v413 = 0;
            v286 = 1;
            v370 = (__int64 *)(v58 + 64);
            v394 = v46;
            while ( 1 )
            {
LABEL_408:
              v57 = *(_QWORD *)(v58 + 32);
              v292 = v286;
              v286 += 2;
              v293 = (const __m128i *)(v57 + v285);
              if ( (*(_BYTE *)(v57 + v285 + 4) & 1) != 0 )
                goto LABEL_407;
              v294 = v293->m128i_i32[2];
              v382 = *(_QWORD *)(v57 + 40LL * (v292 + 1) + 24);
              v389 = ((v293->m128i_i8[3] & 0x40) != 0) & (((unsigned __int8)v293->m128i_i8[3] >> 4) ^ 1);
              if ( v389 && v286 < v405 )
              {
                v302 = v286;
                while ( 1 )
                {
                  v303 = v57 + 40LL * v302;
                  if ( v294 == *(_DWORD *)(v303 + 8) )
                    break;
                  v302 += 2;
                  if ( v302 >= v405 )
                    goto LABEL_410;
                }
                *(_BYTE *)(v303 + 3) |= 0x40u;
                v389 = 0;
                v293->m128i_i8[3] &= ~0x40u;
              }
LABEL_410:
              v385 = v294;
              v366 = *(_QWORD *)(v58 + 24);
              v379 = *(_QWORD *)(v366 + 56);
              v295 = *(_QWORD *)(*(_QWORD *)(v394 + 240) + 8LL) + 960LL;
              if ( (*(_BYTE *)(v58 + 46) & 4) != 0 )
              {
                v287 = (__int64)sub_1E0B640(v379, v295, v370, 0);
                sub_1DD6E10(v366, (__int64 *)v58, v287);
                v288 = v379;
                v289 = v385;
              }
              else
              {
                v287 = (__int64)sub_1E0B640(v379, v295, v370, 0);
                sub_1DD5BA0((__int64 *)(v366 + 16), v287);
                v296 = *(_QWORD *)v58;
                v297 = *(_QWORD *)v287;
                *(_QWORD *)(v287 + 8) = v58;
                v289 = v385;
                v288 = v379;
                v296 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)v287 = v296 | v297 & 7;
                *(_QWORD *)(v296 + 8) = v287;
                *(_QWORD *)v58 = v287 | *(_QWORD *)v58 & 7LL;
              }
              v378 = v289;
              v419.m128i_i32[2] = v373;
              v290 = v382 & 0xFFF;
              v381 = v288;
              v420 = 0;
              v421 = 0;
              v419.m128i_i64[0] = (v290 << 8) | 0x10000000u;
              v422 = 0;
              sub_1E1A9C0(v287, v288, &v419);
              sub_1E1A9C0(v287, v381, v293);
              v22 = (unsigned int)v378;
              if ( !v413 )
              {
                *(_BYTE *)(*(_QWORD *)(v287 + 32) + 4LL) |= 1u;
                v414 = v287;
              }
              v291 = *(char **)(v394 + 272);
              v413 = v389 & (v291 != 0);
              if ( !v413 )
                break;
              if ( v378 > 0 )
                goto LABEL_407;
              v285 += 80;
              sub_1DCCCA0(v291, v378, v58, v287);
              if ( v286 >= v405 )
              {
LABEL_414:
                v46 = v394;
                v298 = v58;
                if ( (*(_BYTE *)v58 & 4) == 0 && (*(_BYTE *)(v58 + 46) & 8) != 0 )
                  goto LABEL_416;
LABEL_418:
                v299 = *(_QWORD *)(v298 + 8);
                if ( !v413 )
                  goto LABEL_419;
                sub_1E16240(v58);
                goto LABEL_421;
              }
            }
            v413 = 1;
LABEL_407:
            v285 += 80;
            if ( v286 >= v405 )
              goto LABEL_414;
            goto LABEL_408;
          }
LABEL_77:
          v59 = *(_DWORD *)(v46 + 336);
          ++v386;
          if ( !v59 )
          {
            ++*(_QWORD *)(v46 + 312);
LABEL_508:
            sub_1DC6D40(v358, 2 * v59);
            v330 = *(_DWORD *)(v46 + 336);
            if ( !v330 )
              goto LABEL_606;
            v331 = v330 - 1;
            v332 = *(_QWORD *)(v46 + 320);
            v333 = v331 & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
            v329 = *(_DWORD *)(v46 + 328) + 1;
            v62 = v332 + 16LL * v333;
            v334 = *(_QWORD *)v62;
            if ( v58 != *(_QWORD *)v62 )
            {
              v335 = 1;
              v336 = 0;
              while ( v334 != -8 )
              {
                if ( v334 == -16 && !v336 )
                  v336 = v62;
                v333 = v331 & (v335 + v333);
                v62 = v332 + 16LL * v333;
                v334 = *(_QWORD *)v62;
                if ( *(_QWORD *)v62 == v58 )
                  goto LABEL_500;
                ++v335;
              }
              if ( v336 )
                v62 = v336;
            }
            goto LABEL_500;
          }
          v60 = *(_QWORD *)(v46 + 320);
          v61 = (v59 - 1) & (((unsigned int)v58 >> 4) ^ ((unsigned int)v58 >> 9));
          v62 = v60 + 16LL * v61;
          v63 = *(_QWORD *)v62;
          if ( v58 == *(_QWORD *)v62 )
            goto LABEL_79;
          v326 = 1;
          v327 = 0;
          while ( v63 != -8 )
          {
            if ( v63 == -16 && !v327 )
              v327 = v62;
            v61 = (v59 - 1) & (v326 + v61);
            v62 = v60 + 16LL * v61;
            v63 = *(_QWORD *)v62;
            if ( *(_QWORD *)v62 == v58 )
              goto LABEL_79;
            ++v326;
          }
          v328 = *(_DWORD *)(v46 + 328);
          if ( v327 )
            v62 = v327;
          ++*(_QWORD *)(v46 + 312);
          v329 = v328 + 1;
          if ( 4 * (v328 + 1) >= 3 * v59 )
            goto LABEL_508;
          if ( v59 - *(_DWORD *)(v46 + 332) - v329 <= v59 >> 3 )
          {
            sub_1DC6D40(v358, v59);
            v337 = *(_DWORD *)(v46 + 336);
            if ( !v337 )
            {
LABEL_606:
              ++*(_DWORD *)(v46 + 328);
              BUG();
            }
            v338 = v337 - 1;
            v339 = *(_QWORD *)(v46 + 320);
            v340 = 0;
            v341 = v338 & (((unsigned int)v58 >> 4) ^ ((unsigned int)v58 >> 9));
            v329 = *(_DWORD *)(v46 + 328) + 1;
            v342 = 1;
            v62 = v339 + 16LL * v341;
            v343 = *(_QWORD *)v62;
            if ( v58 != *(_QWORD *)v62 )
            {
              while ( v343 != -8 )
              {
                if ( v343 == -16 && !v340 )
                  v340 = v62;
                v341 = v338 & (v342 + v341);
                v62 = v339 + 16LL * v341;
                v343 = *(_QWORD *)v62;
                if ( *(_QWORD *)v62 == v58 )
                  goto LABEL_500;
                ++v342;
              }
              if ( v340 )
                v62 = v340;
            }
          }
LABEL_500:
          *(_DWORD *)(v46 + 328) = v329;
          if ( *(_QWORD *)v62 != -8 )
            --*(_DWORD *)(v46 + 332);
          *(_QWORD *)v62 = v58;
          v63 = v414;
          *(_DWORD *)(v62 + 8) = v386;
LABEL_79:
          if ( !sub_1F4DD40(v355, v63) )
          {
            v65 = **(_WORD **)(v63 + 16);
            if ( v65 == 15 )
            {
              v309 = *(_QWORD *)(v63 + 32);
              v67 = *(_DWORD *)(v309 + 8);
              v68 = *(_DWORD *)(v309 + 48);
              if ( v67 <= 0 )
              {
LABEL_83:
                if ( v68 > 0 )
                {
                  v416 = (int *)__PAIR64__(v68, v67);
                  sub_1F4E3A0((__int64)&v419, v356, (int *)&v416, (int *)&v416 + 1);
                  sub_1F4E620(v46, v67);
                }
                goto LABEL_85;
              }
LABEL_457:
              if ( v68 <= 0 )
              {
                v416 = (int *)__PAIR64__(v67, v68);
                sub_1F4E3A0((__int64)&v419, v351, (int *)&v416, (int *)&v416 + 1);
                v69 = *(__int64 **)(v46 + 352);
                if ( *(__int64 **)(v46 + 360) != v69 )
                  goto LABEL_86;
                goto LABEL_459;
              }
LABEL_85:
              v69 = *(__int64 **)(v46 + 352);
              if ( *(__int64 **)(v46 + 360) != v69 )
              {
LABEL_86:
                sub_16CCBA0(v355, v63);
                goto LABEL_87;
              }
LABEL_459:
              v64 = *(unsigned int *)(v46 + 372);
              v310 = &v69[v64];
              v311 = *(_DWORD *)(v46 + 372);
              if ( v69 != v310 )
              {
                v21 = 0;
                do
                {
                  v64 = *v69;
                  if ( *v69 == v63 )
                    goto LABEL_87;
                  if ( v64 == -2 )
                    v21 = (__int64)v69;
                  ++v69;
                }
                while ( v310 != v69 );
                if ( v21 )
                {
                  *(_QWORD *)v21 = v63;
                  --*(_DWORD *)(v46 + 376);
                  ++*(_QWORD *)(v46 + 344);
                  goto LABEL_87;
                }
              }
              if ( v311 < *(_DWORD *)(v46 + 368) )
              {
                *(_DWORD *)(v46 + 372) = v311 + 1;
                *v310 = v63;
                ++*(_QWORD *)(v46 + 344);
                goto LABEL_87;
              }
              goto LABEL_86;
            }
            if ( (v65 & 0xFFFD) == 8 )
            {
              v66 = *(_QWORD *)(v63 + 32);
              v67 = *(_DWORD *)(v66 + 8);
              v68 = *(_DWORD *)(v66 + 88);
              if ( v67 <= 0 )
                goto LABEL_83;
              goto LABEL_457;
            }
          }
LABEL_87:
          v392 = *(_QWORD *)(v414 + 16);
          if ( !*(_DWORD *)(v414 + 40) )
            goto LABEL_56;
          v409 = v46;
          v23 = 0;
          v70 = v414;
          v71 = 0;
          v72 = *(unsigned int *)(v414 + 40);
          do
          {
            while ( 1 )
            {
              v78 = 40 * v71 + *(_QWORD *)(v70 + 32);
              if ( *(_BYTE *)v78 || (*(_BYTE *)(v78 + 3) & 0x10) != 0 || (*(_WORD *)(v78 + 2) & 0xFF0) == 0 )
                goto LABEL_94;
              LODWORD(v80) = sub_1E16AB0(v70, v71, v64, v21, v22, (_BYTE *)v23);
              v79 = *(_QWORD *)(v70 + 32);
              v80 = (unsigned int)v80;
              v81 = v79 + 40 * v71;
              v21 = v79 + 40LL * (unsigned int)v80;
              v64 = *(unsigned int *)(v81 + 8);
              v82 = *(_DWORD *)(v21 + 8);
              if ( (_DWORD)v64 == v82 )
                break;
              v83 = *(_BYTE *)(v81 + 4) & 1;
              if ( v83 && (*(_DWORD *)v21 & 0xFFF00) == 0 )
              {
                if ( v82 < 0 )
                {
                  v388 = *(_BYTE *)(v81 + 4) & 1;
                  v404 = *(_DWORD *)(v21 + 8);
                  v248 = sub_1F3AD60(
                           *(_QWORD *)(v409 + 240),
                           v392,
                           v71,
                           *(_QWORD **)(v409 + 248),
                           *(_QWORD *)(v409 + 232));
                  v82 = v404;
                  v83 = v388;
                  if ( v248 )
                  {
                    sub_1E69410(*(__int64 **)(v409 + 264), v404, v248, 0);
                    v83 = v388;
                    v82 = v404;
                  }
                }
                v396 = v83;
                sub_1E310D0(v81, v82);
                *(_DWORD *)v81 &= 0xFFF000FF;
                v23 = v396;
                goto LABEL_94;
              }
              if ( (v424 & 1) != 0 )
              {
                v22 = (__int64)&v425;
                v73 = 3;
              }
              else
              {
                v84 = v426;
                v22 = (__int64)v425;
                if ( !v426 )
                {
                  v234 = v424;
                  ++v423;
                  v75 = 0;
                  v235 = ((unsigned int)v424 >> 1) + 1;
                  goto LABEL_300;
                }
                v73 = v426 - 1;
              }
              v74 = v73 & (37 * v64);
              v75 = (int *)(v22 + 56LL * v74);
              v76 = *v75;
              if ( (_DWORD)v64 != *v75 )
              {
                v246 = 1;
                v247 = 0;
                while ( v76 != -1 )
                {
                  if ( !v247 && v76 == -2 )
                    v247 = v75;
                  v74 = v73 & (v246 + v74);
                  v75 = (int *)(v22 + 56LL * v74);
                  v76 = *v75;
                  if ( (_DWORD)v64 == *v75 )
                    goto LABEL_91;
                  ++v246;
                }
                v234 = v424;
                v236 = 12;
                v84 = 4;
                if ( v247 )
                  v75 = v247;
                ++v423;
                v235 = ((unsigned int)v424 >> 1) + 1;
                if ( (v424 & 1) != 0 )
                {
LABEL_301:
                  if ( v236 <= 4 * v235 )
                  {
                    v390 = (unsigned int)v80;
                    v406 = v64;
                    sub_1F53020((__int64)&v423, 2 * v84, v64, v84, v235);
                    LODWORD(v64) = v406;
                    v80 = v390;
                    if ( (v424 & 1) != 0 )
                    {
                      v304 = &v425;
                      v305 = 3;
                    }
                    else
                    {
                      v304 = (char **)v425;
                      if ( !v426 )
                      {
LABEL_607:
                        LODWORD(v424) = (2 * ((unsigned int)v424 >> 1) + 2) | v424 & 1;
                        BUG();
                      }
                      v305 = v426 - 1;
                    }
                    v306 = v305 & (37 * v406);
                    v75 = (int *)&v304[7 * v306];
                    v234 = v424;
                    v22 = (unsigned int)*v75;
                    if ( v406 != (_DWORD)v22 )
                    {
                      v307 = 1;
                      v308 = 0;
                      while ( (_DWORD)v22 != -1 )
                      {
                        if ( !v308 && (_DWORD)v22 == -2 )
                          v308 = v75;
                        v306 = v305 & (v306 + v307);
                        v75 = (int *)&v304[7 * v306];
                        v22 = (unsigned int)*v75;
                        if ( v406 == (_DWORD)v22 )
                          goto LABEL_439;
                        ++v307;
                      }
                      goto LABEL_437;
                    }
                  }
                  else
                  {
                    v237 = v84 - HIDWORD(v424) - v235;
                    v22 = (unsigned int)v84 >> 3;
                    if ( v237 <= (unsigned int)v22 )
                    {
                      v391 = (unsigned int)v80;
                      v407 = v64;
                      sub_1F53020((__int64)&v423, v84, v64, v84, v22);
                      LODWORD(v64) = v407;
                      v80 = v391;
                      if ( (v424 & 1) != 0 )
                      {
                        v312 = &v425;
                        v313 = 3;
                      }
                      else
                      {
                        v312 = (char **)v425;
                        if ( !v426 )
                          goto LABEL_607;
                        v313 = v426 - 1;
                      }
                      v314 = v313 & (37 * v407);
                      v75 = (int *)&v312[7 * v314];
                      v234 = v424;
                      v22 = (unsigned int)*v75;
                      if ( v407 != (_DWORD)v22 )
                      {
                        v315 = 1;
                        v308 = 0;
                        while ( (_DWORD)v22 != -1 )
                        {
                          if ( (_DWORD)v22 == -2 && !v308 )
                            v308 = v75;
                          v314 = v313 & (v314 + v315);
                          v75 = (int *)&v312[7 * v314];
                          v22 = (unsigned int)*v75;
                          if ( v407 == (_DWORD)v22 )
                            goto LABEL_439;
                          ++v315;
                        }
LABEL_437:
                        if ( v308 )
                          v75 = v308;
LABEL_439:
                        v234 = v424;
                      }
                    }
                  }
                  v21 = 2 * (v234 >> 1) + 2;
                  LODWORD(v424) = v21 | v234 & 1;
                  if ( *v75 != -1 )
                    --HIDWORD(v424);
                  *v75 = v64;
                  *((_QWORD *)v75 + 1) = v75 + 6;
                  v77 = (v80 << 32) | (unsigned int)v71;
                  v64 = 0;
                  *((_QWORD *)v75 + 2) = 0x400000000LL;
                  goto LABEL_93;
                }
                v84 = v426;
LABEL_300:
                v236 = 3 * v84;
                goto LABEL_301;
              }
LABEL_91:
              v64 = (unsigned int)v75[4];
              v21 = (unsigned int)v75[5];
              v77 = ((unsigned __int64)(unsigned int)v80 << 32) | (unsigned int)v71;
              if ( (unsigned int)v21 <= (unsigned int)v64 )
              {
                sub_16CD150((__int64)(v75 + 2), v75 + 6, 0, 8, v22, v76);
                v64 = (unsigned int)v75[4];
              }
LABEL_93:
              v23 = 1;
              *(_QWORD *)(*((_QWORD *)v75 + 1) + 8 * v64) = v77;
              ++v75[4];
LABEL_94:
              if ( v72 == ++v71 )
                goto LABEL_105;
            }
            ++v71;
            v23 = 1;
          }
          while ( v72 != v71 );
LABEL_105:
          v364 = v23;
          v46 = v409;
          if ( !(_BYTE)v23 )
            goto LABEL_56;
          v85 = (unsigned int)v424 >> 1;
          if ( (unsigned int)v424 >> 1 != 1 )
            goto LABEL_115;
          v86 = v424 & 1;
          if ( (v424 & 1) != 0 )
          {
            v87 = v427;
            v88 = &v425;
            goto LABEL_109;
          }
          v21 = v426;
          v88 = (char **)v425;
          v87 = &v425[56 * v426];
          if ( v425 != v87 )
          {
            do
            {
LABEL_109:
              if ( *(_DWORD *)v88 <= 0xFFFFFFFD )
                break;
              v88 += 7;
            }
            while ( v87 != (char *)v88 );
          }
          if ( *((_DWORD *)v88 + 4) != 1 )
          {
LABEL_327:
            v90 = v414;
LABEL_328:
            if ( v86 )
            {
              v241 = v427;
              v94 = (char *)&v425;
              v360 = v427;
              do
              {
LABEL_330:
                if ( *(_DWORD *)v94 <= 0xFFFFFFFD )
                  break;
                v94 += 56;
              }
              while ( v241 != v94 );
              v387 = v94;
              if ( v86 )
                goto LABEL_333;
LABEL_119:
              v94 = v425;
              v21 = v426;
            }
            else
            {
              v21 = v426;
              v94 = v425;
              v387 = v425;
              v241 = &v425[56 * v426];
              v360 = v241;
              if ( v425 != v241 )
                goto LABEL_330;
            }
            v95 = 56 * v21;
            goto LABEL_121;
          }
          v89 = v88[1];
          v90 = v414;
          v21 = *(unsigned int *)v89;
          v22 = *((unsigned int *)v89 + 1);
          if ( *(_DWORD *)(*(_QWORD *)(v414 + 32) + 40 * v22 + 8) == *(_DWORD *)(*(_QWORD *)(v414 + 32)
                                                                               + 40LL * (unsigned int)v21
                                                                               + 8) )
            goto LABEL_328;
          v91 = sub_1F50270(v409, &v414, &v415, v21, v22, v386, 0);
          v21 = v350;
          v374 = v91;
          if ( v91 )
          {
            sub_1F4DE20((__int64)&v423);
LABEL_56:
            v414 = v415;
            goto LABEL_57;
          }
          v85 = (unsigned int)v424 >> 1;
LABEL_115:
          v86 = v424 & 1;
          if ( v85 )
            goto LABEL_327;
          if ( v86 )
          {
            v92 = &v425;
            v93 = 28;
          }
          else
          {
            v21 = v426;
            v92 = (char **)v425;
            v93 = 7LL * v426;
          }
          v90 = v414;
          v387 = (char *)&v92[v93];
          v360 = (char *)&v92[v93];
          if ( !v86 )
            goto LABEL_119;
LABEL_333:
          v94 = (char *)&v425;
          v95 = 224;
LABEL_121:
          v96 = v409;
          v359 = &v94[v95];
          if ( &v94[v95] == v387 )
            goto LABEL_321;
          v97 = v387;
          v98 = *((_DWORD *)v387 + 4);
          if ( v98 )
          {
LABEL_123:
            v99 = *((_QWORD *)v97 + 1);
            v100 = *(_QWORD *)(v90 + 32);
            LODWORD(v22) = 0;
            v380 = 8LL * (unsigned int)(v98 - 1);
            v101 = (unsigned int *)(v99 + 4);
            do
            {
              v102 = *v101;
              v101 += 2;
              v22 = ((*(_BYTE *)(v100 + 40 * v102 + 4) & 4) != 0) | (unsigned int)v22;
            }
            while ( v101 != (unsigned int *)(v99 + v380 + 12) );
            v363 = v22;
            v410 = 0;
            v372 = 0;
            v367 = ((unsigned int)v90 >> 9) ^ ((unsigned int)v90 >> 4);
            v98 = 0;
            v384 = 0;
            v103 = v99;
            v361 = 2LL * ((unsigned int)((_BYTE)v22 == 0) + 1);
            v365 = v364;
            v375 = (__int64 *)(v90 + 64);
            v104 = v96;
            while ( 2 )
            {
              v105 = (unsigned int *)(v410 + v103);
              v106 = *(_DWORD *)(v100 + 40LL * v105[1] + 8);
              v21 = 40LL * *v105;
              v107 = (_DWORD *)(v21 + v100);
              v393 = v21;
              v108 = v107[2];
              if ( v106 == v108 )
              {
                v365 = 0;
                v130 = v410;
                if ( v380 == v410 )
                {
LABEL_154:
                  v411 = v108;
                  v96 = v104;
                  if ( v365 )
                  {
                    if ( v363 )
                      goto LABEL_156;
                    goto LABEL_182;
                  }
                  if ( v384 )
                  {
                    v183 = *(_QWORD *)(v90 + 32);
                    v184 = v183 + 40LL * *(unsigned int *)(v90 + 40);
                    if ( v183 != v184 )
                    {
                      while ( 1 )
                      {
                        if ( !*(_BYTE *)v183 && v108 == *(_DWORD *)(v183 + 8) )
                        {
                          v21 = *(unsigned __int8 *)(v183 + 3);
                          if ( (v21 & 0x10) == 0 )
                            break;
                        }
                        v183 += 40;
                        if ( v184 == v183 )
                          goto LABEL_175;
                      }
                      v21 = (unsigned int)v21 | 0x40;
                      *(_BYTE *)(v183 + 3) = v21;
                    }
                  }
                  goto LABEL_175;
                }
                goto LABEL_150;
              }
              v368 = *(_QWORD *)(v90 + 24);
              v376 = *(_QWORD *)(v368 + 56);
              v395 = (*v107 >> 8) & 0xFFF;
              v109 = *(_QWORD *)(*(_QWORD *)(v104 + 240) + 8LL) + 960LL;
              if ( (*(_BYTE *)(v90 + 46) & 4) != 0 )
              {
                v110 = (__int64)sub_1E0B640(v376, v109, v375, 0);
                sub_1DD6E10(v368, (__int64 *)v90, v110);
              }
              else
              {
                v110 = (__int64)sub_1E0B640(v376, v109, v375, 0);
                sub_1DD5BA0((__int64 *)(v368 + 16), v110);
                v163 = *(_QWORD *)v90;
                v164 = *(_QWORD *)v110;
                *(_QWORD *)(v110 + 8) = v90;
                v163 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)v110 = v163 | v164 & 7;
                *(_QWORD *)(v163 + 8) = v110;
                *(_QWORD *)v90 = v110 | *(_QWORD *)v90 & 7LL;
              }
              v419.m128i_i64[0] = 0x10000000;
              v420 = 0;
              v419.m128i_i32[2] = v106;
              v421 = 0;
              v422 = 0;
              sub_1E1A9C0(v110, v376, &v419);
              v420 = 0;
              v419.m128i_i32[2] = v108;
              v421 = 0;
              v419.m128i_i64[0] = (unsigned __int16)(v395 & 0xFFF) << 8;
              v422 = 0;
              sub_1E1A9C0(v110, v376, &v419);
              v377 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v104 + 264) + 24LL) + 16LL * (v108 & 0x7FFFFFFF))
                   & 0xFFFFFFFFFFFFFFF8LL;
              if ( v395 )
              {
                v111 = 0;
                if ( v106 >= 0 )
                  v111 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v104 + 264) + 24LL) + 16LL * (v108 & 0x7FFFFFFF))
                       & 0xFFFFFFFFFFFFFFF8LL;
                v377 = v111;
              }
              v112 = *(_QWORD *)v90 & 0xFFFFFFFFFFFFFFF8LL;
              if ( !v112 )
                BUG();
              v113 = *(_QWORD *)v90 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (*(_QWORD *)v112 & 4) == 0 && (*(_BYTE *)(v112 + 46) & 4) != 0 )
              {
                for ( ii = *(_QWORD *)v112; ; ii = *(_QWORD *)v113 )
                {
                  v113 = ii & 0xFFFFFFFFFFFFFFF8LL;
                  if ( (*(_BYTE *)(v113 + 46) & 4) == 0 )
                    break;
                }
              }
              v115 = *(_DWORD *)(v104 + 336);
              if ( v115 )
              {
                v116 = v115 - 1;
                v117 = *(_QWORD *)(v104 + 320);
                v118 = (v115 - 1) & (((unsigned int)v113 >> 9) ^ ((unsigned int)v113 >> 4));
                v119 = (unsigned __int64 *)(v117 + 16LL * v118);
                v120 = *v119;
                if ( v113 == *v119 )
                {
LABEL_141:
                  ++v386;
                  goto LABEL_142;
                }
                v193 = 1;
                v194 = 0;
                while ( v120 != -8 )
                {
                  if ( v120 != -16 || v194 )
                    v119 = v194;
                  v118 = v116 & (v193 + v118);
                  v120 = *(_QWORD *)(v117 + 16LL * v118);
                  if ( v113 == v120 )
                    goto LABEL_141;
                  ++v193;
                  v194 = v119;
                  v119 = (unsigned __int64 *)(v117 + 16LL * v118);
                }
                if ( !v194 )
                  v194 = v119;
                v195 = *(_DWORD *)(v104 + 328);
                ++*(_QWORD *)(v104 + 312);
                v196 = v195 + 1;
                if ( 4 * (v195 + 1) < 3 * v115 )
                {
                  if ( v115 - *(_DWORD *)(v104 + 332) - v196 <= v115 >> 3 )
                  {
                    v402 = ((unsigned int)v113 >> 9) ^ ((unsigned int)v113 >> 4);
                    sub_1DC6D40(v358, v115);
                    v228 = *(_DWORD *)(v104 + 336);
                    if ( !v228 )
                    {
LABEL_609:
                      ++*(_DWORD *)(v104 + 328);
                      BUG();
                    }
                    v229 = v228 - 1;
                    v225 = 0;
                    v230 = *(_QWORD *)(v104 + 320);
                    v231 = 1;
                    v232 = v229 & v402;
                    v196 = *(_DWORD *)(v104 + 328) + 1;
                    v194 = (unsigned __int64 *)(v230 + 16LL * (v229 & v402));
                    v233 = *v194;
                    if ( v113 != *v194 )
                    {
                      while ( v233 != -8 )
                      {
                        if ( v233 == -16 && !v225 )
                          v225 = v194;
                        v232 = v229 & (v231 + v232);
                        v194 = (unsigned __int64 *)(v230 + 16LL * v232);
                        v233 = *v194;
                        if ( v113 == *v194 )
                          goto LABEL_246;
                        ++v231;
                      }
                      goto LABEL_285;
                    }
                  }
                  goto LABEL_246;
                }
              }
              else
              {
                ++*(_QWORD *)(v104 + 312);
              }
              sub_1DC6D40(v358, 2 * v115);
              v219 = *(_DWORD *)(v104 + 336);
              if ( !v219 )
                goto LABEL_609;
              v220 = v219 - 1;
              v221 = *(_QWORD *)(v104 + 320);
              v196 = *(_DWORD *)(v104 + 328) + 1;
              v222 = v220 & (((unsigned int)v113 >> 9) ^ ((unsigned int)v113 >> 4));
              v194 = (unsigned __int64 *)(v221 + 16LL * v222);
              v223 = *v194;
              if ( *v194 != v113 )
              {
                v224 = 1;
                v225 = 0;
                while ( v223 != -8 )
                {
                  if ( v223 == -16 && !v225 )
                    v225 = v194;
                  v222 = v220 & (v224 + v222);
                  v194 = (unsigned __int64 *)(v221 + 16LL * v222);
                  v223 = *v194;
                  if ( v113 == *v194 )
                    goto LABEL_246;
                  ++v224;
                }
LABEL_285:
                if ( v225 )
                  v194 = v225;
              }
LABEL_246:
              *(_DWORD *)(v104 + 328) = v196;
              if ( *v194 != -8 )
                --*(_DWORD *)(v104 + 332);
              *v194 = v113;
              *((_DWORD *)v194 + 2) = v386;
              v115 = *(_DWORD *)(v104 + 336);
              ++v386;
              v117 = *(_QWORD *)(v104 + 320);
              if ( !v115 )
              {
                ++*(_QWORD *)(v104 + 312);
                goto LABEL_250;
              }
              v116 = v115 - 1;
LABEL_142:
              v121 = v116 & v367;
              v122 = (__int64 *)(v117 + 16LL * (v116 & v367));
              v123 = *v122;
              if ( v90 == *v122 )
              {
LABEL_143:
                *((_DWORD *)v122 + 2) = v386;
                v124 = *(_QWORD *)(v104 + 280);
                if ( !v124 )
                {
                  v125 = *(_QWORD *)(v90 + 32) + v393;
                  if ( (((*(_BYTE *)(v125 + 3) & 0x40) != 0) & ((*(_BYTE *)(v125 + 3) >> 4) ^ 1)) != 0 )
                    goto LABEL_146;
                  if ( v106 >= 0 )
                    goto LABEL_147;
LABEL_195:
                  if ( v108 >= 0 )
                    goto LABEL_147;
                  goto LABEL_196;
                }
                v372 = sub_1DC1550(*(_QWORD *)(v124 + 272), v113, 0) & 0xFFFFFFFFFFFFFFF8LL | 4;
                if ( v106 >= 0 )
                {
                  v125 = *(_QWORD *)(v90 + 32) + v393;
                  if ( (((*(_BYTE *)(v125 + 3) & 0x40) != 0) & ((*(_BYTE *)(v125 + 3) >> 4) ^ 1)) != 0 )
                    goto LABEL_146;
LABEL_147:
                  sub_1E310D0(v125, v106);
                  *(_DWORD *)v125 &= 0xFFF000FF;
                  v126 = *(_DWORD *)(v104 + 576);
                  if ( v126 )
                  {
                    v22 = v126 - 1;
                    v127 = *(_QWORD *)(v104 + 560);
                    v21 = (unsigned int)v22 & (37 * v106);
                    v128 = (unsigned int *)(v127 + 8 * v21);
                    v129 = *v128;
                    if ( v106 == *v128 )
                    {
LABEL_149:
                      v128[1] = v108;
                      v130 = v410;
                      v98 = v106;
                      if ( v380 == v410 )
                        goto LABEL_154;
LABEL_150:
                      v410 = v130 + 8;
                      v103 = *((_QWORD *)v387 + 1);
                      v100 = *(_QWORD *)(v90 + 32);
                      continue;
                    }
                    v186 = 1;
                    v23 = 0;
                    while ( v129 != -1 )
                    {
                      if ( v129 == -2 && !v23 )
                        v23 = (__int64)v128;
                      v21 = (unsigned int)v22 & (v186 + (_DWORD)v21);
                      v128 = (unsigned int *)(v127 + 8LL * (unsigned int)v21);
                      v129 = *v128;
                      if ( v106 == *v128 )
                        goto LABEL_149;
                      ++v186;
                    }
                    v187 = *(_DWORD *)(v104 + 568);
                    if ( v23 )
                      v128 = (unsigned int *)v23;
                    ++*(_QWORD *)(v104 + 552);
                    v188 = v187 + 1;
                    if ( 4 * (v187 + 1) < 3 * v126 )
                    {
                      v21 = v126 - *(_DWORD *)(v104 + 572) - v188;
                      if ( (unsigned int)v21 <= v126 >> 3 )
                      {
                        sub_1392B70(v356, v126);
                        v215 = *(_DWORD *)(v104 + 576);
                        if ( !v215 )
                        {
LABEL_605:
                          ++*(_DWORD *)(v104 + 568);
                          BUG();
                        }
                        v22 = (unsigned int)(v215 - 1);
                        v216 = *(_QWORD *)(v104 + 560);
                        v23 = 0;
                        v217 = v22 & (37 * v106);
                        v218 = 1;
                        v188 = *(_DWORD *)(v104 + 568) + 1;
                        v128 = (unsigned int *)(v216 + 8LL * v217);
                        v21 = *v128;
                        if ( (_DWORD)v21 != v106 )
                        {
                          while ( (_DWORD)v21 != -1 )
                          {
                            if ( (_DWORD)v21 == -2 && !v23 )
                              v23 = (__int64)v128;
                            v217 = v22 & (v218 + v217);
                            v128 = (unsigned int *)(v216 + 8LL * v217);
                            v21 = *v128;
                            if ( v106 == (_DWORD)v21 )
                              goto LABEL_228;
                            ++v218;
                          }
                          if ( v23 )
                            v128 = (unsigned int *)v23;
                        }
                      }
LABEL_228:
                      *(_DWORD *)(v104 + 568) = v188;
                      if ( *v128 != -1 )
                        --*(_DWORD *)(v104 + 572);
                      *v128 = v106;
                      v128[1] = 0;
                      goto LABEL_149;
                    }
                  }
                  else
                  {
                    ++*(_QWORD *)(v104 + 552);
                  }
                  sub_1392B70(v356, 2 * v126);
                  v204 = *(_DWORD *)(v104 + 576);
                  if ( !v204 )
                    goto LABEL_605;
                  v23 = (unsigned int)(v204 - 1);
                  v205 = *(_QWORD *)(v104 + 560);
                  v21 = (unsigned int)v23 & (37 * v106);
                  v188 = *(_DWORD *)(v104 + 568) + 1;
                  v128 = (unsigned int *)(v205 + 8 * v21);
                  v206 = *v128;
                  if ( *v128 != v106 )
                  {
                    v207 = 1;
                    v208 = 0;
                    while ( v206 != -1 )
                    {
                      if ( !v208 && v206 == -2 )
                        v208 = v128;
                      v22 = (unsigned int)(v207 + 1);
                      v21 = (unsigned int)v23 & (v207 + (_DWORD)v21);
                      v128 = (unsigned int *)(v205 + 8LL * (unsigned int)v21);
                      v206 = *v128;
                      if ( v106 == *v128 )
                        goto LABEL_228;
                      ++v207;
                    }
                    if ( v208 )
                      v128 = v208;
                  }
                  goto LABEL_228;
                }
                v165 = *(_QWORD *)(v104 + 280);
                v166 = v106 & 0x7FFFFFFF;
                v167 = *(unsigned int *)(v165 + 408);
                v168 = v106 & 0x7FFFFFFF;
                v169 = 8 * v168;
                if ( (v106 & 0x7FFFFFFFu) < (unsigned int)v167 )
                {
                  v170 = *(_QWORD *)(*(_QWORD *)(v165 + 400) + 8LL * v166);
                  if ( v170 )
                  {
LABEL_199:
                    v399 = *(_DWORD *)(v170 + 72);
                    v171 = sub_145CBF0((__int64 *)(v165 + 296), 16, 16);
                    *(_DWORD *)v171 = v399;
                    *(_QWORD *)(v171 + 8) = v372;
                    v173 = *(unsigned int *)(v170 + 72);
                    if ( (unsigned int)v173 >= *(_DWORD *)(v170 + 76) )
                    {
                      v403 = v171;
                      sub_16CD150(v170 + 64, (const void *)(v170 + 80), 0, 8, v171, v172);
                      v173 = *(unsigned int *)(v170 + 72);
                      v171 = v403;
                    }
                    *(_QWORD *)(*(_QWORD *)(v170 + 64) + 8 * v173) = v171;
                    ++*(_DWORD *)(v170 + 72);
                    v174 = *(_QWORD *)(*(_QWORD *)(v104 + 280) + 272LL);
                    v175 = v90;
                    if ( (*(_BYTE *)(v90 + 46) & 4) != 0 )
                    {
                      do
                        v175 = *(_QWORD *)v175 & 0xFFFFFFFFFFFFFFF8LL;
                      while ( (*(_BYTE *)(v175 + 46) & 4) != 0 );
                    }
                    v176 = *(_QWORD *)(v174 + 368);
                    v177 = *(unsigned int *)(v174 + 384);
                    if ( (_DWORD)v177 )
                    {
                      v178 = (v177 - 1) & (((unsigned int)v175 >> 9) ^ ((unsigned int)v175 >> 4));
                      v179 = (__int64 *)(v176 + 16LL * v178);
                      v172 = *v179;
                      if ( *v179 == v175 )
                        goto LABEL_205;
                      v185 = 1;
                      while ( v172 != -8 )
                      {
                        v344 = v185 + 1;
                        v178 = (v177 - 1) & (v185 + v178);
                        v179 = (__int64 *)(v176 + 16LL * v178);
                        v172 = *v179;
                        if ( *v179 == v175 )
                          goto LABEL_205;
                        v185 = v344;
                      }
                    }
                    v177 *= 16;
                    v179 = (__int64 *)(v176 + v177);
LABEL_205:
                    v180 = v179[1];
                    v420 = v171;
                    v419.m128i_i64[1] = v361 | v180 & 0xFFFFFFFFFFFFFFF8LL;
                    *((_QWORD *)&v349 + 1) = v419.m128i_i64[1];
                    *(_QWORD *)&v349 = v372;
                    v419.m128i_i64[0] = v372;
                    sub_1DB8610(v170, v372, (__int64)v179, v177, v171, v172, v349, v171);
                    v125 = *(_QWORD *)(v90 + 32) + v393;
                    if ( (((*(_BYTE *)(v125 + 3) & 0x40) != 0) & ((*(_BYTE *)(v125 + 3) >> 4) ^ 1)) == 0 )
                    {
                      if ( v108 >= 0 )
                        goto LABEL_147;
LABEL_196:
                      sub_1E69410(*(__int64 **)(v104 + 264), v106, v377, 0);
                      goto LABEL_147;
                    }
LABEL_146:
                    *(_BYTE *)(v125 + 3) &= ~0x40u;
                    v384 = v364;
                    if ( v106 >= 0 )
                      goto LABEL_147;
                    goto LABEL_195;
                  }
                }
                v181 = v166 + 1;
                if ( (unsigned int)v167 >= v181 )
                  goto LABEL_209;
                v238 = v181;
                if ( v181 < v167 )
                {
                  *(_DWORD *)(v165 + 408) = v181;
                  goto LABEL_209;
                }
                if ( v181 <= v167 )
                {
LABEL_209:
                  v182 = *(_QWORD *)(v165 + 400);
                }
                else
                {
                  if ( v181 > (unsigned __int64)*(unsigned int *)(v165 + 412) )
                  {
                    v362 = v181;
                    v371 = *(_QWORD *)(v104 + 280);
                    v408 = v181;
                    sub_16CD150(v165 + 400, (const void *)(v165 + 416), v181, 8, v165, v168);
                    v165 = v371;
                    v168 = v106 & 0x7FFFFFFF;
                    v181 = v362;
                    v169 = 8 * v168;
                    v167 = *(unsigned int *)(v371 + 408);
                    v238 = v408;
                  }
                  v182 = *(_QWORD *)(v165 + 400);
                  v242 = (_QWORD *)(v182 + 8 * v238);
                  v243 = (_QWORD *)(v182 + 8 * v167);
                  v244 = *(_QWORD *)(v165 + 416);
                  if ( v242 != v243 )
                  {
                    do
                      *v243++ = v244;
                    while ( v242 != v243 );
                    v182 = *(_QWORD *)(v165 + 400);
                  }
                  *(_DWORD *)(v165 + 408) = v181;
                }
                v369 = v168;
                v400 = (_QWORD *)v165;
                *(_QWORD *)(v169 + v182) = sub_1DBA290(v106);
                v170 = *(_QWORD *)(v400[50] + 8 * v369);
                sub_1DBB110(v400, v170);
                v165 = *(_QWORD *)(v104 + 280);
                goto LABEL_199;
              }
              break;
            }
            v189 = 1;
            v190 = 0;
            while ( v123 != -8 )
            {
              if ( !v190 && v123 == -16 )
                v190 = v122;
              v121 = v116 & (v189 + v121);
              v122 = (__int64 *)(v117 + 16LL * v121);
              v123 = *v122;
              if ( v90 == *v122 )
                goto LABEL_143;
              ++v189;
            }
            v191 = *(_DWORD *)(v104 + 328);
            if ( v190 )
              v122 = v190;
            ++*(_QWORD *)(v104 + 312);
            v192 = v191 + 1;
            if ( 4 * (v191 + 1) >= 3 * v115 )
            {
LABEL_250:
              sub_1DC6D40(v358, 2 * v115);
              v197 = *(_DWORD *)(v104 + 336);
              if ( !v197 )
                goto LABEL_611;
              v198 = v197 - 1;
              v199 = *(_QWORD *)(v104 + 320);
              v200 = v198 & v367;
              v192 = *(_DWORD *)(v104 + 328) + 1;
              v122 = (__int64 *)(v199 + 16LL * (v198 & v367));
              v201 = *v122;
              if ( v90 != *v122 )
              {
                v202 = 1;
                v203 = 0;
                while ( v201 != -8 )
                {
                  if ( !v203 && v201 == -16 )
                    v203 = v122;
                  v200 = v198 & (v202 + v200);
                  v122 = (__int64 *)(v199 + 16LL * v200);
                  v201 = *v122;
                  if ( v90 == *v122 )
                    goto LABEL_237;
                  ++v202;
                }
LABEL_254:
                if ( v203 )
                  v122 = v203;
              }
            }
            else if ( v115 - (v192 + *(_DWORD *)(v104 + 332)) <= v115 >> 3 )
            {
              sub_1DC6D40(v358, v115);
              v209 = *(_DWORD *)(v104 + 336);
              if ( !v209 )
              {
LABEL_611:
                ++*(_DWORD *)(v104 + 328);
                BUG();
              }
              v210 = v209 - 1;
              v203 = 0;
              v211 = *(_QWORD *)(v104 + 320);
              v212 = 1;
              v213 = v210 & v367;
              v192 = *(_DWORD *)(v104 + 328) + 1;
              v122 = (__int64 *)(v211 + 16LL * (v210 & v367));
              v214 = *v122;
              if ( v90 != *v122 )
              {
                while ( v214 != -8 )
                {
                  if ( v214 == -16 && !v203 )
                    v203 = v122;
                  v213 = v210 & (v212 + v213);
                  v122 = (__int64 *)(v211 + 16LL * v213);
                  v214 = *v122;
                  if ( v90 == *v122 )
                    goto LABEL_237;
                  ++v212;
                }
                goto LABEL_254;
              }
            }
LABEL_237:
            *(_DWORD *)(v104 + 328) = v192;
            if ( *v122 != -8 )
              --*(_DWORD *)(v104 + 332);
            *v122 = v90;
            *((_DWORD *)v122 + 2) = 0;
            goto LABEL_143;
          }
          while ( 2 )
          {
            v411 = 0;
            v384 = 0;
            v372 = 0;
LABEL_182:
            v158 = *(_QWORD *)(v90 + 32);
            v159 = v158 + 40LL * *(unsigned int *)(v90 + 40);
            if ( v158 == v159 )
            {
              v363 = 0;
            }
            else
            {
              v398 = v90;
              v160 = v96;
              do
              {
                while ( 1 )
                {
                  if ( !*(_BYTE *)v158 && v411 == *(_DWORD *)(v158 + 8) )
                  {
                    v161 = *(_BYTE *)(v158 + 3);
                    if ( (v161 & 0x10) == 0 )
                      break;
                  }
                  v158 += 40;
                  if ( v159 == v158 )
                    goto LABEL_191;
                }
                if ( (v161 & 0x40) != 0 )
                {
                  v384 = 1;
                  *(_BYTE *)(v158 + 3) = v161 & 0xBF;
                }
                v162 = v158;
                v158 += 40;
                sub_1E310D0(v162, v98);
              }
              while ( v159 != v158 );
LABEL_191:
              v96 = v160;
              v363 = 0;
              v90 = v398;
            }
LABEL_156:
            if ( v384 )
            {
              v131 = *(char **)(v96 + 272);
              if ( v131 )
              {
                v397 = v96;
                v132 = sub_1DCC790(v131, v411);
                v419.m128i_i64[0] = v90;
                v133 = sub_1F4C640(*((_QWORD **)v132 + 4), *((_QWORD *)v132 + 5), v419.m128i_i64);
                v96 = v397;
                if ( v133 != *(_BYTE **)(v22 + 40) )
                {
                  sub_1DCBB50(v22 + 32, v133);
                  v136 = *(_QWORD *)v90 & 0xFFFFFFFFFFFFFFF8LL;
                  if ( !v136 )
                    BUG();
                  v137 = *(_QWORD *)v90 & 0xFFFFFFFFFFFFFFF8LL;
                  if ( (*(_QWORD *)v136 & 4) == 0 && (*(_BYTE *)(v136 + 46) & 4) != 0 )
                  {
                    for ( jj = *(_QWORD *)v136; ; jj = *(_QWORD *)v137 )
                    {
                      v137 = jj & 0xFFFFFFFFFFFFFFF8LL;
                      if ( (*(_BYTE *)(v137 + 46) & 4) == 0 )
                        break;
                    }
                  }
                  v139 = *(_QWORD *)(v397 + 272);
                  v140 = sub_1E1AFE0(v137, v411, *(_QWORD **)(v139 + 360), 0, v134, v135);
                  v96 = v397;
                  if ( v140 )
                  {
                    v324 = sub_1DCC790((char *)v139, v411);
                    v419.m128i_i64[0] = v137;
                    v96 = v397;
                    v325 = (_BYTE *)*((_QWORD *)v324 + 5);
                    if ( v325 == *((_BYTE **)v324 + 6) )
                    {
                      sub_1DCC370((__int64)(v324 + 32), v325, &v419);
                      v96 = v397;
                    }
                    else
                    {
                      if ( v325 )
                      {
                        *(_QWORD *)v325 = v137;
                        v325 = (_BYTE *)*((_QWORD *)v324 + 5);
                      }
                      *((_QWORD *)v324 + 5) = v325 + 8;
                    }
                  }
                }
              }
            }
            v141 = *(_QWORD *)(v96 + 280);
            if ( !v141 )
              goto LABEL_175;
            v142 = *(unsigned int *)(v141 + 408);
            v143 = v411 & 0x7FFFFFFF;
            v144 = v411 & 0x7FFFFFFF;
            v145 = 8 * v144;
            if ( (v411 & 0x7FFFFFFFu) >= (unsigned int)v142
              || (v146 = *(__int64 **)(*(_QWORD *)(v141 + 400) + 8LL * v143)) == 0 )
            {
              v226 = v143 + 1;
              if ( (unsigned int)v142 >= v143 + 1 )
                goto LABEL_289;
              v245 = v226;
              if ( v226 < v142 )
              {
                *(_DWORD *)(v141 + 408) = v226;
                goto LABEL_289;
              }
              if ( v226 <= v142 )
              {
LABEL_289:
                v227 = *(_QWORD *)(v141 + 400);
              }
              else
              {
                if ( v226 > (unsigned __int64)*(unsigned int *)(v141 + 412) )
                {
                  v383 = v96;
                  sub_16CD150(v141 + 400, (const void *)(v141 + 416), v226, 8, 8 * v411, v23);
                  v142 = *(unsigned int *)(v141 + 408);
                  v96 = v383;
                  v145 = 8LL * (v411 & 0x7FFFFFFF);
                  v245 = v226;
                }
                v227 = *(_QWORD *)(v141 + 400);
                v316 = *(_QWORD *)(v141 + 416);
                v317 = (_QWORD *)(v227 + 8 * v245);
                v318 = (_QWORD *)(v227 + 8 * v142);
                if ( v317 != v318 )
                {
                  do
                    *v318++ = v316;
                  while ( v317 != v318 );
                  v227 = *(_QWORD *)(v141 + 400);
                }
                *(_DWORD *)(v141 + 408) = v226;
              }
              v401 = v96;
              *(_QWORD *)(v145 + v227) = sub_1DBA290(v411);
              v146 = *(__int64 **)(*(_QWORD *)(v141 + 400) + 8 * v144);
              sub_1DBB110((_QWORD *)v141, (__int64)v146);
              v96 = v401;
              v141 = *(_QWORD *)(v401 + 280);
            }
            v147 = *(_QWORD *)(v141 + 272);
            v148 = v90;
            if ( (*(_BYTE *)(v90 + 46) & 4) != 0 )
            {
              do
                v148 = *(_QWORD *)v148 & 0xFFFFFFFFFFFFFFF8LL;
              while ( (*(_BYTE *)(v148 + 46) & 4) != 0 );
            }
            v149 = *(_QWORD *)(v147 + 368);
            v150 = *(_DWORD *)(v147 + 384);
            if ( !v150 )
              goto LABEL_319;
            v151 = (v150 - 1) & (((unsigned int)v148 >> 9) ^ ((unsigned int)v148 >> 4));
            v152 = (__int64 *)(v149 + 16LL * v151);
            v153 = *v152;
            if ( *v152 != v148 )
            {
              v239 = 1;
              while ( v153 != -8 )
              {
                v345 = v239 + 1;
                v151 = (v150 - 1) & (v239 + v151);
                v152 = (__int64 *)(v149 + 16LL * v151);
                v153 = *v152;
                if ( *v152 == v148 )
                  goto LABEL_173;
                v239 = v345;
              }
LABEL_319:
              v152 = (__int64 *)(v149 + 16LL * v150);
            }
LABEL_173:
            v154 = v152[1];
            v412 = v96;
            v155 = sub_1DB3C70(v146, v154);
            v96 = v412;
            v156 = v154 & 0xFFFFFFFFFFFFFFF8LL | (v363 == 0 ? 4LL : 2LL);
            if ( v156 == *(_QWORD *)(v155 + 8) )
            {
              sub_1DB4410((__int64)v146, v372, v156, 0);
              v96 = v412;
            }
LABEL_175:
            v387 += 56;
            v95 = (__int64)v360;
            v157 = v387;
            if ( v387 != v360 )
            {
              do
              {
                if ( *(_DWORD *)v157 <= 0xFFFFFFFD )
                  break;
                v157 += 56;
              }
              while ( v157 != v360 );
              v387 = v157;
            }
            if ( v359 != v387 )
            {
              v97 = v387;
              v90 = v414;
              v98 = *((_DWORD *)v387 + 4);
              if ( v98 )
                goto LABEL_123;
              continue;
            }
            break;
          }
          v90 = v414;
          v46 = v96;
LABEL_321:
          if ( **(_WORD **)(v90 + 16) == 8 )
          {
            v319 = *(_QWORD *)(*(_QWORD *)(v90 + 32) + 144LL);
            sub_1E16C90(v90, 3u, v95, v21, v22, (_BYTE *)v23);
            **(_DWORD **)(v414 + 32) = ((v319 & 0xFFF) << 8) | **(_DWORD **)(v414 + 32) & 0xFFF000FF;
            v320 = *(_QWORD *)(v414 + 32);
            v321 = *(_BYTE *)(v320 + 44) & 1;
            *(_BYTE *)(v320 + 4) = v321 | *(_BYTE *)(v320 + 4) & 0xFE;
            sub_1E16C90(v414, 1u, v320, v321, v322, v323);
            *(_QWORD *)(v414 + 16) = *(_QWORD *)(*(_QWORD *)(v46 + 240) + 8LL) + 960LL;
          }
          sub_1F4DE20((__int64)&v423);
          v414 = v415;
          v374 = v364;
LABEL_57:
          v47 = v414;
          if ( v414 == v357 )
          {
            v3 = v46;
            break;
          }
        }
      }
      v353 = *(_QWORD *)(v353 + 8);
      if ( v352 == v353 )
        goto LABEL_60;
    }
    v265 = 2048;
    v264 = 128;
LABEL_374:
    j___libc_free_0(v259);
    *(_DWORD *)(v3 + 336) = v264;
    v266 = (_QWORD *)sub_22077B0(v265);
    v267 = *(unsigned int *)(v3 + 336);
    *(_QWORD *)(v3 + 328) = 0;
    *(_QWORD *)(v3 + 320) = v266;
    for ( kk = &v266[2 * v267]; kk != v266; v266 += 2 )
    {
      if ( v266 )
        *v266 = -8;
    }
    goto LABEL_26;
  }
  v374 = 0;
LABEL_60:
  if ( *(_QWORD *)(v3 + 280) )
    sub_1E926D0(*(_QWORD *)(v3 + 232), v3, (__int64)"After two-address instruction pass", 1);
  if ( (v424 & 1) != 0 )
  {
    v50 = (char **)v427;
    v49 = &v425;
    goto LABEL_66;
  }
  v49 = (char **)v425;
  if ( !v426 )
    goto LABEL_535;
  v50 = (char **)&v425[56 * v426];
  do
  {
LABEL_66:
    while ( 1 )
    {
      if ( *(_DWORD *)v49 <= 0xFFFFFFFD )
      {
        v51 = (unsigned __int64)v49[1];
        if ( (char **)v51 != v49 + 3 )
          break;
      }
      v49 += 7;
      if ( v49 == v50 )
        goto LABEL_69;
    }
    _libc_free(v51);
    v49 += 7;
  }
  while ( v49 != v50 );
LABEL_69:
  if ( (v424 & 1) == 0 )
  {
    v49 = (char **)v425;
LABEL_535:
    j___libc_free_0(v49);
  }
  return v374;
}
