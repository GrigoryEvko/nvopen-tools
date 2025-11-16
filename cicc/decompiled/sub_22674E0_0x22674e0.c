// Function: sub_22674E0
// Address: 0x22674e0
//
__int64 __fastcall sub_22674E0(
        int a1,
        __int64 a2,
        __int64 *a3,
        __int64 a4,
        int a5,
        __int64 a6,
        __m128i a7,
        __int64 a8,
        __int64 a9,
        __m128i *a10,
        __int64 a11,
        unsigned __int64 *a12,
        __int64 a13,
        __int64 a14)
{
  _QWORD *v14; // rsi
  __int128 v15; // kr00_16
  char *v16; // r15
  __m128i *v17; // rdi
  char *v18; // r13
  __int64 (__fastcall *v19)(__int64); // rax
  char *v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rbx
  char v23; // al
  __m128i v24; // rax
  char v25; // bl
  __m128i v26; // kr10_16
  unsigned __int8 v27; // bl
  int v28; // eax
  unsigned int v29; // r8d
  _QWORD *v30; // r10
  __m128i *v31; // r15
  __m128i *v32; // r13
  __m128i *v33; // rdi
  __int64 (__fastcall *v34)(__int64); // rax
  __int64 v35; // rdi
  __int64 v36; // rax
  unsigned int v37; // r8d
  __int64 *v38; // r10
  __int64 v39; // rcx
  __int64 v40; // rbx
  unsigned int v41; // esi
  int v42; // r12d
  __int64 v43; // rsi
  __int64 v44; // rax
  unsigned __int64 *v45; // r13
  unsigned __int64 *v46; // r15
  __int64 v47; // r14
  unsigned __int64 v48; // r12
  _QWORD *v49; // rdi
  int i; // ebx
  __int64 *v51; // rax
  __int64 *v52; // r15
  char v53; // si
  _QWORD *v54; // r15
  _QWORD *v55; // rax
  _QWORD *v56; // rdx
  __int8 v57; // cl
  __int64 *v58; // rsi
  _QWORD **v59; // r14
  __m128i *v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rdx
  __m128i v63; // xmm2
  __m128i v64; // xmm3
  __m128i v65; // xmm4
  __m128i v66; // xmm5
  __m128i v67; // xmm6
  __m128i v68; // xmm7
  __int64 v69; // rax
  __m128i v70; // xmm1
  __m128i v71; // xmm2
  __m128i v72; // xmm3
  __m128i v73; // xmm4
  __m128i v74; // xmm5
  __m128i v75; // xmm6
  _BYTE *v76; // rdx
  __int64 v77; // rdx
  __m128i v78; // xmm0
  __m128i v79; // xmm7
  __int64 v80; // rax
  volatile signed __int32 *v81; // r12
  _QWORD *v82; // rax
  _QWORD *v83; // rbx
  __int64 v84; // rax
  __int64 v85; // r15
  __m128i v86; // xmm2
  __int64 v87; // rax
  char v88; // al
  volatile signed __int32 *v89; // r13
  signed __int32 v90; // eax
  volatile signed __int32 *v91; // r13
  _QWORD *v92; // rax
  __int64 (__fastcall *v93)(unsigned __int64 *, unsigned __int64 *, int); // rdx
  __int64 (__fastcall *v94)(volatile signed __int32 ***); // rax
  unsigned int v95; // eax
  __m128i v96; // xmm1
  __int64 (__fastcall *v97)(volatile signed __int32 ***); // rdx
  unsigned int v98; // r15d
  __int64 v99; // rsi
  __int64 v100; // rdx
  __int64 v101; // rcx
  signed __int32 v102; // eax
  signed __int32 v103; // eax
  _QWORD **v104; // r12
  __int64 v105; // rsi
  _QWORD *v106; // r13
  unsigned __int64 v107; // rbx
  __int64 v108; // rax
  __int64 v109; // rdx
  __int64 v110; // rax
  __int64 v111; // rsi
  unsigned int v112; // r13d
  unsigned __int64 *v113; // rbx
  unsigned __int64 *v114; // r12
  unsigned __int64 v115; // r14
  __int64 v116; // rbx
  unsigned __int64 v117; // r12
  __int64 v118; // rsi
  __int64 v119; // rdi
  unsigned __int64 v120; // r8
  __int64 v121; // r12
  __int64 v122; // rbx
  _QWORD *v123; // rdi
  unsigned __int64 v124; // r8
  __int64 v125; // r12
  __int64 v126; // rbx
  _QWORD *v127; // rdi
  unsigned __int64 v128; // r8
  __int64 v129; // r12
  __int64 v130; // rbx
  _QWORD *v131; // rdi
  __m128i v133; // rax
  __m128i v134; // kr20_16
  int v135; // eax
  unsigned int v136; // r8d
  __int64 *v137; // r9
  __int64 v138; // rax
  __int64 v139; // rdx
  unsigned int v140; // r8d
  __int64 *v141; // r9
  __int64 v142; // rcx
  _BYTE *v143; // rax
  const char *v144; // r13
  size_t v145; // rdx
  size_t v146; // r15
  __m128i v147; // rax
  int v148; // eax
  unsigned int v149; // r11d
  _QWORD *v150; // r10
  __int64 v151; // rax
  unsigned int v152; // r11d
  __int64 *v153; // r10
  __int64 v154; // rcx
  __m128i v155; // xmm2
  int v156; // r14d
  __m128i v157; // rax
  __m128i v158; // kr30_16
  int v159; // eax
  unsigned int v160; // r9d
  _QWORD *v161; // r11
  __int64 v162; // rax
  unsigned int v163; // r9d
  __int64 *v164; // r11
  __int64 v165; // rcx
  unsigned __int32 v166; // eax
  _QWORD *v167; // rbx
  __int64 v168; // r12
  __int64 v169; // rsi
  __m128i v170; // xmm3
  __m128i v171; // xmm4
  __m128i v172; // xmm5
  __m128i v173; // xmm6
  __m128i v174; // xmm7
  __m128i v175; // xmm1
  __m128i v176; // xmm3
  signed __int32 v177; // eax
  signed __int32 v178; // eax
  signed __int32 v179; // eax
  _DWORD *v180; // rdx
  unsigned __int64 v181; // rdx
  __int64 *v182; // rbx
  __int64 v183; // rdx
  __int64 v184; // rax
  __m128i *v185; // rsi
  __int64 v186; // r8
  __int64 v187; // r9
  int v188; // eax
  __int64 v189; // rax
  __m128i v190; // xmm0
  __int64 v191; // rdx
  size_t v192; // r14
  const char *v193; // r15
  int v194; // eax
  int v195; // eax
  __int64 v196; // rax
  unsigned int v197; // r8d
  int v198; // eax
  int v199; // eax
  __int64 v200; // rax
  unsigned int v201; // eax
  int v202; // eax
  unsigned int v203; // r8d
  _QWORD *v204; // r9
  int v205; // eax
  unsigned int v206; // r12d
  _QWORD *v207; // rcx
  __int64 v208; // rdi
  const char *v209; // r12
  size_t v210; // rdx
  size_t v211; // rbx
  int v212; // eax
  int v213; // eax
  __int64 v214; // rax
  __int64 v215; // rax
  _QWORD *v216; // rcx
  __int64 v217; // rbx
  __int64 v218; // rax
  unsigned int v219; // r8d
  _QWORD *v220; // r9
  _QWORD *v221; // rcx
  __int64 v222; // rax
  __int64 v223; // r12
  __int64 v224; // r13
  unsigned __int64 v225; // rcx
  unsigned __int64 v226; // r13
  unsigned __int64 v227; // rax
  __int64 v228; // rax
  __int64 v229; // rax
  __int64 v230; // rdx
  char *v231; // rcx
  __int64 v232; // r15
  __int64 v233; // r13
  const char *v234; // r12
  size_t v235; // rdx
  size_t v236; // r14
  int v237; // eax
  int v238; // eax
  unsigned __int64 v239; // rax
  int v240; // eax
  char v241; // cl
  char v242; // dl
  __int64 v243; // r15
  __int64 v244; // r13
  const char *v245; // r12
  size_t v246; // rdx
  size_t v247; // r14
  int v248; // eax
  int v249; // eax
  unsigned __int64 v250; // rax
  int v251; // eax
  char v252; // cl
  char v253; // dl
  unsigned __int64 v254; // r8
  __int64 v255; // r12
  __int64 v256; // rbx
  _QWORD *v257; // rdi
  unsigned __int64 v258; // rbx
  unsigned __int64 v259; // rax
  unsigned __int64 v260; // r12
  unsigned __int64 v261; // rdx
  __int64 v262; // rax
  unsigned __int64 v263; // r12
  void *v264; // rdi
  __int64 v265; // rax
  __int64 v266; // rdx
  __int64 v267; // rcx
  char v268; // al
  __m128i v269; // [rsp+30h] [rbp-1340h]
  __int64 v270; // [rsp+30h] [rbp-1340h]
  unsigned __int64 v271; // [rsp+38h] [rbp-1338h]
  unsigned __int64 v272; // [rsp+40h] [rbp-1330h]
  __int64 v273; // [rsp+40h] [rbp-1330h]
  unsigned __int64 v274; // [rsp+48h] [rbp-1328h]
  unsigned __int64 v277; // [rsp+60h] [rbp-1310h]
  __int64 *v280; // [rsp+78h] [rbp-12F8h]
  __int64 *v281; // [rsp+80h] [rbp-12F0h]
  __int64 v282; // [rsp+80h] [rbp-12F0h]
  __int64 v283; // [rsp+80h] [rbp-12F0h]
  _QWORD *v284; // [rsp+80h] [rbp-12F0h]
  __int64 v285; // [rsp+80h] [rbp-12F0h]
  unsigned __int64 v286; // [rsp+88h] [rbp-12E8h]
  unsigned __int64 v287; // [rsp+88h] [rbp-12E8h]
  int v288; // [rsp+98h] [rbp-12D8h]
  _QWORD *v289; // [rsp+98h] [rbp-12D8h]
  int v290; // [rsp+A8h] [rbp-12C8h]
  unsigned __int64 v291; // [rsp+B0h] [rbp-12C0h]
  __int64 v292; // [rsp+B0h] [rbp-12C0h]
  __int64 *v293; // [rsp+B8h] [rbp-12B8h]
  unsigned int v294; // [rsp+B8h] [rbp-12B8h]
  __int64 *v295; // [rsp+B8h] [rbp-12B8h]
  __int64 *v296; // [rsp+B8h] [rbp-12B8h]
  unsigned int n; // [rsp+C0h] [rbp-12B0h]
  unsigned __int64 *na; // [rsp+C0h] [rbp-12B0h]
  size_t nb; // [rsp+C0h] [rbp-12B0h]
  unsigned int nc; // [rsp+C0h] [rbp-12B0h]
  unsigned int nd; // [rsp+C0h] [rbp-12B0h]
  unsigned int ne; // [rsp+C0h] [rbp-12B0h]
  size_t nf; // [rsp+C0h] [rbp-12B0h]
  size_t ng; // [rsp+C0h] [rbp-12B0h]
  unsigned int nh; // [rsp+C0h] [rbp-12B0h]
  unsigned __int64 v306; // [rsp+C8h] [rbp-12A8h]
  __int32 v307; // [rsp+C8h] [rbp-12A8h]
  __int64 v308; // [rsp+C8h] [rbp-12A8h]
  __int64 v309; // [rsp+C8h] [rbp-12A8h]
  __int64 v310; // [rsp+C8h] [rbp-12A8h]
  __int64 v311; // [rsp+D0h] [rbp-12A0h] BYREF
  int v312; // [rsp+DCh] [rbp-1294h] BYREF
  char v313; // [rsp+EFh] [rbp-1281h] BYREF
  __int64 v314; // [rsp+F0h] [rbp-1280h] BYREF
  _QWORD **v315; // [rsp+F8h] [rbp-1278h] BYREF
  unsigned __int64 v316; // [rsp+100h] [rbp-1270h] BYREF
  __int64 v317; // [rsp+108h] [rbp-1268h]
  __int64 v318; // [rsp+110h] [rbp-1260h]
  unsigned __int64 v319; // [rsp+120h] [rbp-1250h] BYREF
  __int64 v320; // [rsp+128h] [rbp-1248h]
  __int64 v321; // [rsp+130h] [rbp-1240h]
  unsigned __int64 v322; // [rsp+140h] [rbp-1230h] BYREF
  __int64 v323; // [rsp+148h] [rbp-1228h]
  __int64 v324; // [rsp+150h] [rbp-1220h]
  size_t v325; // [rsp+160h] [rbp-1210h] BYREF
  __int64 v326; // [rsp+168h] [rbp-1208h]
  unsigned __int64 *v327; // [rsp+180h] [rbp-11F0h] BYREF
  __int64 v328; // [rsp+188h] [rbp-11E8h]
  unsigned __int64 *v329; // [rsp+190h] [rbp-11E0h]
  __int64 *v330; // [rsp+1A0h] [rbp-11D0h] BYREF
  __int64 *v331; // [rsp+1A8h] [rbp-11C8h]
  __int64 *v332; // [rsp+1B0h] [rbp-11C0h]
  __m128i *v333; // [rsp+1C0h] [rbp-11B0h] BYREF
  __int64 v334; // [rsp+1C8h] [rbp-11A8h]
  __m128i v335; // [rsp+1D0h] [rbp-11A0h] BYREF
  __m128i v336; // [rsp+1E0h] [rbp-1190h] BYREF
  void (__fastcall *v337)(__m128i *, __m128i *, __int64); // [rsp+1F0h] [rbp-1180h]
  __int64 v338; // [rsp+1F8h] [rbp-1178h]
  __m128i v339; // [rsp+200h] [rbp-1170h] BYREF
  __int64 (__fastcall *v340)(unsigned __int64 *, unsigned __int64 *, int); // [rsp+210h] [rbp-1160h]
  __int64 (__fastcall *v341)(volatile signed __int32 ***); // [rsp+218h] [rbp-1158h]
  _OWORD v342[2]; // [rsp+220h] [rbp-1150h] BYREF
  __int64 v343; // [rsp+240h] [rbp-1130h]
  __m128i v344; // [rsp+250h] [rbp-1120h] BYREF
  __int64 v345; // [rsp+260h] [rbp-1110h]
  _QWORD v346[3]; // [rsp+268h] [rbp-1108h] BYREF
  __m128i v347; // [rsp+280h] [rbp-10F0h] BYREF
  __m128i v348; // [rsp+290h] [rbp-10E0h] BYREF
  __m128i si128; // [rsp+2A0h] [rbp-10D0h]
  __m128i v350; // [rsp+2B0h] [rbp-10C0h]
  char v351; // [rsp+2C0h] [rbp-10B0h]
  _BYTE v352[16]; // [rsp+2D0h] [rbp-10A0h] BYREF
  __int64 (__fastcall *v353)(__int64); // [rsp+2E0h] [rbp-1090h]
  __int64 v354; // [rsp+2E8h] [rbp-1088h]
  __int64 (__fastcall *v355)(__int64); // [rsp+2F0h] [rbp-1080h]
  __int64 v356; // [rsp+2F8h] [rbp-1078h]
  __int64 (__fastcall *v357)(__int64 *); // [rsp+300h] [rbp-1070h]
  __int64 v358; // [rsp+308h] [rbp-1068h]
  char v359; // [rsp+310h] [rbp-1060h] BYREF
  unsigned __int64 v360[10]; // [rsp+330h] [rbp-1040h] BYREF
  pthread_mutex_t mutex; // [rsp+380h] [rbp-FF0h] BYREF
  pthread_cond_t v362[2]; // [rsp+3A8h] [rbp-FC8h] BYREF
  int v363; // [rsp+408h] [rbp-F68h]
  __m128i v364; // [rsp+440h] [rbp-F30h] BYREF
  __m128i v365; // [rsp+450h] [rbp-F20h] BYREF
  __m128i v366; // [rsp+460h] [rbp-F10h] BYREF
  __m128i v367; // [rsp+470h] [rbp-F00h] BYREF
  __m128i v368; // [rsp+480h] [rbp-EF0h] BYREF
  __m128i v369; // [rsp+490h] [rbp-EE0h] BYREF
  __m128i *v370; // [rsp+4A0h] [rbp-ED0h]
  __int64 v371; // [rsp+4A8h] [rbp-EC8h]
  __m128i v372; // [rsp+4B0h] [rbp-EC0h] BYREF
  _BYTE v373[1784]; // [rsp+4C0h] [rbp-EB0h] BYREF
  __int32 v374; // [rsp+BB8h] [rbp-7B8h]
  __m128i v375; // [rsp+BC0h] [rbp-7B0h] BYREF
  __m128i v376; // [rsp+BD0h] [rbp-7A0h] BYREF
  _BYTE v377[96]; // [rsp+BE0h] [rbp-790h] BYREF
  _BYTE v378[1784]; // [rsp+C40h] [rbp-730h] BYREF
  __int32 v379; // [rsp+1338h] [rbp-38h]

  v311 = a2;
  v14 = (_QWORD *)*a3;
  v312 = a1;
  v318 = 0x1000000000LL;
  v321 = 0x1000000000LL;
  v324 = 0x1800000000LL;
  v316 = 0;
  v317 = 0;
  v319 = 0;
  v320 = 0;
  v322 = 0;
  v323 = 0;
  sub_BA9680(&v375, v14);
  v288 = 0;
  v272 = *(_QWORD *)&v377[32];
  v347 = _mm_load_si128(&v375);
  v277 = *(_QWORD *)&v377[40];
  v348 = _mm_load_si128(&v376);
  v286 = *(_QWORD *)&v377[48];
  si128 = _mm_load_si128((const __m128i *)v377);
  v306 = *(_QWORD *)&v377[56];
  v350 = _mm_load_si128((const __m128i *)&v377[16]);
  v271 = *(_QWORD *)&v377[64];
  v274 = *(_QWORD *)&v377[72];
  v15 = *(_OWORD *)&v377[80];
  while ( *(_OWORD *)&v348 != __PAIR128__(v306, v286)
       || *(_OWORD *)&v347 != __PAIR128__(v277, v272)
       || *(_OWORD *)&v350 != v15
       || *(_OWORD *)&si128 != __PAIR128__(v274, v271) )
  {
    v16 = v352;
    v354 = 0;
    v17 = &v347;
    v353 = sub_C11C50;
    v18 = v352;
    v356 = 0;
    v355 = sub_C11C70;
    v358 = 0;
    v357 = sub_C11C90;
    v19 = sub_C11C30;
    if ( ((unsigned __int8)sub_C11C30 & 1) == 0 )
      goto LABEL_5;
    while ( 1 )
    {
      v19 = *(__int64 (__fastcall **)(__int64))((char *)v19 + v17->m128i_i64[0] - 1);
LABEL_5:
      v20 = (char *)v19((__int64)v17);
      if ( v20 )
        break;
      while ( 1 )
      {
        v16 += 16;
        if ( v16 == &v359 )
LABEL_390:
          BUG();
        v21 = *((_QWORD *)v18 + 3);
        v19 = (__int64 (__fastcall *)(__int64))*((_QWORD *)v18 + 2);
        v18 = v16;
        v17 = (__m128i *)((char *)&v347 + v21);
        if ( ((unsigned __int8)v19 & 1) != 0 )
          break;
        v20 = (char *)v19((__int64)v17);
        if ( v20 )
          goto LABEL_9;
      }
    }
LABEL_9:
    v22 = (__int64)v20;
    v23 = *v20;
    if ( !v23 )
    {
      if ( sub_B2FC80(v22) )
        goto LABEL_177;
      v156 = sub_225FE40(v22);
      v157.m128i_i64[0] = (__int64)sub_BD5D20(v22);
      v158 = v157;
      v364 = v157;
      v159 = sub_C92610();
      v160 = sub_C92740((__int64)&v316, (const void *)v158.m128i_i64[0], v158.m128i_u64[1], v159);
      v161 = (_QWORD *)(v316 + 8LL * v160);
      if ( *v161 )
      {
        if ( *v161 == -8 )
        {
          LODWORD(v318) = v318 - 1;
          goto LABEL_194;
        }
      }
      else
      {
LABEL_194:
        v296 = (__int64 *)(v316 + 8LL * v160);
        nd = v160;
        v162 = sub_C7D670(v158.m128i_i64[1] + 17, 8);
        v163 = nd;
        v164 = v296;
        v165 = v162;
        if ( v158.m128i_i64[1] )
        {
          v283 = v162;
          memcpy((void *)(v162 + 16), (const void *)v158.m128i_i64[0], v158.m128i_u64[1]);
          v163 = nd;
          v164 = v296;
          v165 = v283;
        }
        *(_BYTE *)(v165 + v158.m128i_i64[1] + 16) = 0;
        *(_QWORD *)v165 = v158.m128i_i64[1];
        *(_DWORD *)(v165 + 8) = v156;
        *v164 = v165;
        ++HIDWORD(v317);
        sub_C929D0((__int64 *)&v316, v163);
      }
LABEL_177:
      ++v288;
      v133.m128i_i64[0] = (__int64)sub_BD5D20(v22);
      v134 = v133;
      v364 = v133;
      v135 = sub_C92610();
      v136 = sub_C92740(a8, (const void *)v134.m128i_i64[0], v134.m128i_u64[1], v135);
      v137 = (__int64 *)(*(_QWORD *)a8 + 8LL * v136);
      if ( !*v137 )
        goto LABEL_180;
      if ( *v137 == -8 )
      {
        --*(_DWORD *)(a8 + 16);
LABEL_180:
        v281 = v137;
        v294 = v136;
        v138 = sub_C7D670(v134.m128i_i64[1] + 17, 8);
        v139 = v134.m128i_i64[1];
        v140 = v294;
        v141 = v281;
        v142 = v138;
        if ( v134.m128i_i64[1] )
        {
          v270 = v138;
          memcpy((void *)(v138 + 16), (const void *)v134.m128i_i64[0], v134.m128i_u64[1]);
          v139 = v134.m128i_i64[1];
          v140 = v294;
          v141 = v281;
          v142 = v270;
        }
        *(_BYTE *)(v142 + v139 + 16) = 0;
        *(_QWORD *)v142 = v139;
        *(_DWORD *)(v142 + 8) = v288;
        *v141 = v142;
        ++*(_DWORD *)(a8 + 12);
        sub_C929D0((__int64 *)a8, v140);
        goto LABEL_11;
      }
      goto LABEL_11;
    }
    if ( v23 == 1 )
    {
      v143 = (_BYTE *)sub_B325F0(v22);
      if ( !v143 )
        goto LABEL_13;
      if ( !*v143 )
      {
        v144 = sub_BD5D20((__int64)v143);
        v146 = v145;
        v147.m128i_i64[0] = (__int64)sub_BD5D20(v22);
        v364.m128i_i64[0] = (__int64)v144;
        v364.m128i_i64[1] = v146;
        v365 = v147;
        v148 = sub_C92610();
        v149 = sub_C92740((__int64)&v322, v144, v146, v148);
        v150 = (_QWORD *)(v322 + 8LL * v149);
        if ( !*v150 )
          goto LABEL_188;
        if ( *v150 == -8 )
        {
          LODWORD(v324) = v324 - 1;
LABEL_188:
          v295 = (__int64 *)(v322 + 8LL * v149);
          nc = v149;
          v151 = sub_C7D670(v146 + 25, 8);
          v152 = nc;
          v153 = v295;
          v154 = v151;
          if ( v146 )
          {
            v285 = v151;
            memcpy((void *)(v151 + 24), v144, v146);
            v152 = nc;
            v153 = v295;
            v154 = v285;
          }
          v155 = _mm_load_si128(&v365);
          *(_BYTE *)(v154 + v146 + 24) = 0;
          *(_QWORD *)v154 = v146;
          *(__m128i *)(v154 + 8) = v155;
          *v153 = v154;
          ++HIDWORD(v323);
          sub_C929D0((__int64 *)&v322, v152);
        }
      }
    }
LABEL_11:
    v24.m128i_i64[0] = (__int64)sub_BD5D20(v22);
    v25 = *(_BYTE *)(v22 + 32);
    v26 = v24;
    v364 = v24;
    v27 = v25 & 0xF;
    v28 = sub_C92610();
    v29 = sub_C92740((__int64)&v319, (const void *)v26.m128i_i64[0], v26.m128i_u64[1], v28);
    v30 = (_QWORD *)(v319 + 8LL * v29);
    if ( *v30 )
    {
      if ( *v30 != -8 )
        goto LABEL_13;
      LODWORD(v321) = v321 - 1;
    }
    v293 = (__int64 *)(v319 + 8LL * v29);
    n = v29;
    v36 = sub_C7D670(v26.m128i_i64[1] + 17, 8);
    v37 = n;
    v38 = v293;
    v39 = v36;
    if ( v26.m128i_i64[1] )
    {
      v282 = v36;
      memcpy((void *)(v36 + 16), (const void *)v26.m128i_i64[0], v26.m128i_u64[1]);
      v37 = n;
      v38 = v293;
      v39 = v282;
    }
    *(_BYTE *)(v39 + v26.m128i_i64[1] + 16) = 0;
    *(_QWORD *)v39 = v26.m128i_i64[1];
    *(_DWORD *)(v39 + 8) = v27;
    *v38 = v39;
    ++HIDWORD(v320);
    sub_C929D0((__int64 *)&v319, v37);
LABEL_13:
    v31 = &v364;
    v365.m128i_i64[1] = 0;
    v32 = &v364;
    v33 = &v347;
    v366.m128i_i64[1] = 0;
    v365.m128i_i64[0] = (__int64)sub_C11BA0;
    v367.m128i_i64[1] = 0;
    v366.m128i_i64[0] = (__int64)sub_C11BD0;
    v367.m128i_i64[0] = (__int64)sub_C11C00;
    v34 = sub_C11B70;
    if ( ((unsigned __int8)sub_C11B70 & 1) == 0 )
      goto LABEL_15;
LABEL_14:
    v34 = *(__int64 (__fastcall **)(__int64))((char *)v34 + v33->m128i_i64[0] - 1);
LABEL_15:
    while ( !(unsigned __int8)v34((__int64)v33) )
    {
      if ( ++v31 == &v368 )
        goto LABEL_390;
      v35 = v32[1].m128i_i64[1];
      v34 = (__int64 (__fastcall *)(__int64))v32[1].m128i_i64[0];
      v32 = v31;
      v33 = (__m128i *)((char *)&v347 + v35);
      if ( ((unsigned __int8)v34 & 1) != 0 )
        goto LABEL_14;
    }
  }
  sub_22660A0(&v325, a6, (__int64)&v316, (__int64)&v322);
  v40 = (__int64)(v326 - v325) >> 5;
  v41 = a5;
  v42 = v40;
  if ( (int)v40 <= a5 )
    v41 = v40;
  v43 = v41 | 0x100000000LL;
  sub_23CD6E0(v352, v43);
  v327 = 0;
  v328 = 0;
  v329 = 0;
  if ( (unsigned __int64)(int)v40 > 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::reserve");
  if ( !(_DWORD)v40 )
    goto LABEL_51;
  v44 = sub_22077B0(8LL * (int)v40);
  v45 = (unsigned __int64 *)v328;
  v46 = v327;
  na = (unsigned __int64 *)v44;
  if ( (unsigned __int64 *)v328 == v327 )
    goto LABEL_38;
  v47 = v44;
  while ( 2 )
  {
    while ( 2 )
    {
      v48 = *v46;
      if ( v47 )
      {
        *(_QWORD *)v47 = v48;
        *v46 = 0;
        goto LABEL_33;
      }
      if ( !v48 )
      {
LABEL_33:
        ++v46;
        v47 += 8;
        if ( v45 == v46 )
          goto LABEL_37;
        continue;
      }
      break;
    }
    v49 = (_QWORD *)*v46++;
    v47 = 8;
    sub_B6E710(v49);
    v43 = 8;
    j_j___libc_free_0(v48);
    if ( v45 != v46 )
      continue;
    break;
  }
LABEL_37:
  v42 = v40;
  v46 = v327;
LABEL_38:
  if ( v46 )
  {
    v43 = (char *)v329 - (char *)v46;
    j_j___libc_free_0((unsigned __int64)v46);
  }
  v327 = na;
  v328 = (__int64)na;
  v329 = &na[(int)v40];
  if ( (int)v40 > 0 )
  {
    for ( i = 0; v42 != i; ++i )
    {
      while ( 1 )
      {
        v51 = (__int64 *)sub_22077B0(8u);
        v52 = v51;
        if ( v51 )
          sub_B6EEA0(v51);
        v53 = *(_BYTE *)(a4 + 232);
        v375.m128i_i64[0] = (__int64)v52;
        sub_B6F950(v52, v53);
        v43 = v328;
        if ( (unsigned __int64 *)v328 != v329 )
          break;
        sub_2265790((unsigned __int64 *)&v327, (unsigned __int64 *)v328, v375.m128i_i64);
        v54 = (_QWORD *)v375.m128i_i64[0];
LABEL_49:
        if ( !v54 )
          goto LABEL_43;
        ++i;
        sub_B6E710(v54);
        v43 = 8;
        j_j___libc_free_0((unsigned __int64)v54);
        if ( v42 == i )
          goto LABEL_51;
      }
      if ( !v328 )
      {
        v328 = 8;
        v54 = (_QWORD *)v375.m128i_i64[0];
        goto LABEL_49;
      }
      *(_QWORD *)v328 = v375.m128i_i64[0];
      v328 += 8;
LABEL_43:
      ;
    }
  }
LABEL_51:
  v314 = 0;
  if ( *(_BYTE *)(a6 + 688) )
  {
    v43 = 0;
    v188 = sub_16832F0(&v314, 0);
    if ( v188 )
    {
      if ( (unsigned int)(v188 - 5) > 1 )
        sub_C64ED0("GNU Jobserver support requested, but an error occurred", 1u);
      v375.m128i_i64[0] = (__int64)&v376;
      v364.m128i_i64[0] = 90;
      v189 = sub_22409D0((__int64)&v375, (unsigned __int64 *)&v364, 0);
      v43 = 1;
      v375.m128i_i64[0] = v189;
      v376.m128i_i64[0] = v364.m128i_i64[0];
      *(__m128i *)v189 = _mm_load_si128((const __m128i *)&xmmword_4281AD0);
      v190 = _mm_load_si128((const __m128i *)&xmmword_4281AE0);
      qmemcpy((void *)(v189 + 80), "jobserver'", 10);
      *(__m128i *)(v189 + 16) = v190;
      *(__m128i *)(v189 + 32) = _mm_load_si128((const __m128i *)&xmmword_4281AF0);
      *(__m128i *)(v189 + 48) = _mm_load_si128((const __m128i *)&xmmword_4281B00);
      a7 = _mm_load_si128((const __m128i *)&xmmword_4281B10);
      *(__m128i *)(v189 + 64) = a7;
      v191 = v375.m128i_i64[0];
      v375.m128i_i64[1] = v364.m128i_i64[0];
      *(_BYTE *)(v375.m128i_i64[0] + v364.m128i_i64[0]) = 0;
      sub_CEB590(&v375, 1, v191, (char *)0x6576726573626F6ALL);
      if ( (__m128i *)v375.m128i_i64[0] != &v376 )
      {
        v43 = v376.m128i_i64[0] + 1;
        j_j___libc_free_0(v375.m128i_u64[0]);
      }
    }
  }
  v330 = 0;
  v331 = 0;
  v332 = 0;
  v313 = 1;
  v273 = v326;
  if ( v325 != v326 )
  {
    nb = v325;
    v269.m128i_i64[0] = (__int64)&v312;
    v269.m128i_i64[1] = (__int64)&v311;
    v307 = 0;
    do
    {
      v347.m128i_i64[0] = 0;
      v348.m128i_i32[2] = 128;
      v55 = (_QWORD *)sub_C7D670(0x2000, 8);
      v348.m128i_i64[0] = 0;
      v347.m128i_i64[1] = (__int64)v55;
      v375.m128i_i64[1] = 2;
      v56 = &v55[8 * (unsigned __int64)v348.m128i_u32[2]];
      v375.m128i_i64[0] = (__int64)&unk_49DD7B0;
      v376.m128i_i64[0] = 0;
      v376.m128i_i64[1] = -4096;
      for ( *(_QWORD *)v377 = 0; v56 != v55; v55 += 8 )
      {
        if ( v55 )
        {
          v57 = v375.m128i_i8[8];
          v55[2] = 0;
          v55[3] = -4096;
          *v55 = &unk_49DD7B0;
          v55[1] = v57 & 6;
          v55[4] = *(_QWORD *)v377;
        }
      }
      v351 = 0;
      v364.m128i_i32[0] = v307;
      v364.m128i_i64[1] = nb;
      sub_29A8CE0(&v315, *a3, &v347, sub_2260000, &v364);
      v58 = v331;
      v375.m128i_i64[0] = 0;
      if ( v331 == v332 )
      {
        sub_A28060((__int64)&v330, v331, &v375);
      }
      else
      {
        if ( v331 )
        {
          *v331 = 0;
          v58 = v331;
        }
        v331 = v58 + 1;
      }
      *(_QWORD *)&v377[8] = 0x100000000LL;
      v59 = v315;
      memset(v342, 0, sizeof(v342));
      v343 = 0;
      *(_QWORD *)&v377[16] = &v364;
      v315 = 0;
      v364 = (__m128i)(unsigned __int64)&v365;
      v365.m128i_i8[0] = 0;
      v375.m128i_i64[1] = 0;
      v376 = 0u;
      *(_QWORD *)v377 = 0;
      v375.m128i_i64[0] = (__int64)&unk_49DD210;
      sub_CB5980((__int64)&v375, 0, 0, 0);
      sub_A3ACE0((__int64)v59, (__int64)&v375, 0, 0, 0, 0);
      v333 = &v335;
      sub_2260190(
        (__int64 *)&v333,
        **(_BYTE ***)&v377[16],
        **(_QWORD **)&v377[16] + *(_QWORD *)(*(_QWORD *)&v377[16] + 8LL));
      v375.m128i_i64[0] = (__int64)&unk_49DD210;
      sub_CB5840((__int64)&v375);
      if ( (__m128i *)v364.m128i_i64[0] != &v365 )
        j_j___libc_free_0(v364.m128i_u64[0]);
      v364.m128i_i64[0] = (__int64)&v314;
      v364.m128i_i64[1] = (__int64)&v327;
      v365 = v269;
      v366.m128i_i64[0] = a4;
      v366.m128i_i64[1] = a14;
      v367.m128i_i64[0] = a11;
      v367.m128i_i64[1] = (__int64)&v313;
      v368.m128i_i64[0] = (__int64)v342;
      v368.m128i_i64[1] = (__int64)a12;
      v369.m128i_i64[0] = a13;
      v369.m128i_i64[1] = (__int64)&v330;
      v60 = v333;
      if ( v333 == &v335 )
      {
        v170 = _mm_load_si128(&v335);
        v335.m128i_i8[0] = 0;
        v372 = v170;
        v62 = v334;
        qmemcpy(v373, (const void *)a6, sizeof(v373));
        v374 = v307;
        v334 = 0;
        v171 = _mm_load_si128(&v364);
        v172 = _mm_load_si128(&v365);
        *(_QWORD *)&v377[64] = &v377[80];
        v173 = _mm_load_si128(&v366);
        v174 = _mm_load_si128(&v367);
        v175 = _mm_load_si128(&v368);
        v176 = _mm_load_si128(&v369);
        v375 = v171;
        v376 = v172;
        *(__m128i *)v377 = v173;
        *(__m128i *)&v377[16] = v174;
        *(__m128i *)&v377[32] = v175;
        *(__m128i *)&v377[48] = v176;
      }
      else
      {
        v61 = v335.m128i_i64[0];
        v335.m128i_i8[0] = 0;
        v372.m128i_i64[0] = v61;
        v333 = &v335;
        v62 = v334;
        qmemcpy(v373, (const void *)a6, sizeof(v373));
        v334 = 0;
        v374 = v307;
        v63 = _mm_load_si128(&v364);
        v64 = _mm_load_si128(&v365);
        *(_QWORD *)&v377[64] = &v377[80];
        v65 = _mm_load_si128(&v366);
        v66 = _mm_load_si128(&v367);
        v67 = _mm_load_si128(&v368);
        v68 = _mm_load_si128(&v369);
        v375 = v63;
        v376 = v64;
        *(__m128i *)v377 = v65;
        *(__m128i *)&v377[16] = v66;
        *(__m128i *)&v377[32] = v67;
        *(__m128i *)&v377[48] = v68;
        if ( v60 != &v372 )
        {
          *(_QWORD *)&v377[64] = v60;
          *(_QWORD *)&v377[80] = v372.m128i_i64[0];
          goto LABEL_67;
        }
      }
      *(__m128i *)&v377[80] = _mm_load_si128(&v372);
LABEL_67:
      *(_QWORD *)&v377[72] = v62;
      v372.m128i_i8[0] = 0;
      v370 = &v372;
      v371 = 0;
      qmemcpy(v378, v373, sizeof(v378));
      v337 = 0;
      v379 = v307;
      v69 = sub_22077B0(0x780u);
      if ( v69 )
      {
        v70 = _mm_load_si128(&v375);
        v71 = _mm_load_si128(&v376);
        v72 = _mm_load_si128((const __m128i *)v377);
        v73 = _mm_load_si128((const __m128i *)&v377[16]);
        *(_QWORD *)(v69 + 96) = v69 + 112;
        v74 = _mm_load_si128((const __m128i *)&v377[32]);
        v75 = _mm_load_si128((const __m128i *)&v377[48]);
        *(__m128i *)v69 = v70;
        v76 = *(_BYTE **)&v377[64];
        *(__m128i *)(v69 + 16) = v71;
        *(__m128i *)(v69 + 32) = v72;
        *(__m128i *)(v69 + 48) = v73;
        *(__m128i *)(v69 + 64) = v74;
        *(__m128i *)(v69 + 80) = v75;
        if ( v76 == &v377[80] )
        {
          *(__m128i *)(v69 + 112) = _mm_load_si128((const __m128i *)&v377[80]);
        }
        else
        {
          *(_QWORD *)(v69 + 96) = v76;
          *(_QWORD *)(v69 + 112) = *(_QWORD *)&v377[80];
        }
        v77 = *(_QWORD *)&v377[72];
        *(_QWORD *)&v377[72] = 0;
        v377[80] = 0;
        *(_QWORD *)(v69 + 104) = v77;
        LODWORD(v77) = v379;
        *(_QWORD *)&v377[64] = &v377[80];
        qmemcpy((void *)(v69 + 128), v378, 0x6F8u);
        *(_DWORD *)(v69 + 1912) = v77;
      }
      v336.m128i_i64[0] = v69;
      v78 = _mm_load_si128(&v336);
      v345 = (__int64)sub_2260EF0;
      v79 = _mm_load_si128(&v344);
      v337 = 0;
      v338 = v346[0];
      v346[0] = sub_2266030;
      v336 = v79;
      v344 = v78;
      v80 = sub_22077B0(0x58u);
      v81 = (volatile signed __int32 *)v80;
      if ( v80 )
      {
        *(_BYTE *)(v80 + 36) = 0;
        *(_QWORD *)(v80 + 8) = 0x100000001LL;
        *(_QWORD *)(v80 + 24) = 0;
        *(_DWORD *)(v80 + 32) = 0;
        *(_QWORD *)v80 = &unk_4A083C0;
        *(_DWORD *)(v80 + 40) = 0;
        *(_QWORD *)(v80 + 16) = &unk_4A08390;
        v82 = (_QWORD *)sub_22077B0(0x10u);
        v83 = v82;
        if ( v82 )
        {
          *v82 = 0;
          v82[1] = 0;
          sub_222D5E0((__int64)v82);
          *v83 = &unk_49EF518;
        }
        v84 = v345;
        *((_QWORD *)v81 + 6) = v83;
        v85 = (__int64)(v81 + 4);
        v86 = _mm_load_si128(&v344);
        *((_QWORD *)v81 + 9) = v84;
        v87 = v346[0];
        *(__m128i *)(v81 + 14) = v86;
        *((_QWORD *)v81 + 10) = v87;
        if ( &_pthread_key_create )
          _InterlockedAdd(v81 + 2, 1u);
        else
          ++*((_DWORD *)v81 + 2);
        v88 = *((_BYTE *)v81 + 36);
        *((_BYTE *)v81 + 36) = 1;
        if ( v88 )
          goto LABEL_384;
        v89 = v81 + 2;
        if ( &_pthread_key_create )
        {
          v90 = _InterlockedExchangeAdd(v89, 0xFFFFFFFF);
        }
        else
        {
          v90 = *((_DWORD *)v81 + 2);
          *((_DWORD *)v81 + 2) = v90 - 1;
        }
        if ( v90 == 1
          && (((*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v81 + 16LL))(v81), &_pthread_key_create)
            ? (v177 = _InterlockedExchangeAdd(v81 + 3, 0xFFFFFFFF))
            : (v177 = *((_DWORD *)v81 + 3), *((_DWORD *)v81 + 3) = v177 - 1),
              v177 == 1) )
        {
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v81 + 24LL))(v81);
          if ( &_pthread_key_create )
          {
LABEL_213:
            _InterlockedAdd(v89, 1u);
            v91 = v81;
            goto LABEL_82;
          }
        }
        else if ( &_pthread_key_create )
        {
          goto LABEL_213;
        }
        ++*((_DWORD *)v81 + 2);
        v91 = v81;
      }
      else
      {
        v85 = 16;
        if ( v345 )
          ((void (__fastcall *)(__m128i *, __m128i *, __int64))v345)(&v344, &v344, 3);
        v268 = MEMORY[0x24];
        MEMORY[0x24] = 1;
        if ( v268 )
LABEL_384:
          sub_42641C(1u);
        v91 = 0;
      }
LABEL_82:
      v340 = 0;
      v92 = (_QWORD *)sub_22077B0(0x10u);
      if ( v92 )
      {
        *v92 = v85;
        v91 = 0;
        v92[1] = v81;
      }
      v339.m128i_i64[0] = (__int64)v92;
      v93 = sub_2261E30;
      v94 = sub_225FF40;
      v340 = sub_2261E30;
      v341 = sub_225FF40;
      if ( &_pthread_key_create )
      {
        v95 = pthread_mutex_lock(&mutex);
        if ( v95 )
          sub_4264C5(v95);
        v93 = v340;
        v94 = v341;
      }
      a7 = _mm_load_si128(&v339);
      v96 = _mm_load_si128(&v344);
      v345 = (__int64)v93;
      v97 = (__int64 (__fastcall *)(volatile signed __int32 ***))v346[0];
      v340 = 0;
      v346[0] = v94;
      v341 = v97;
      v346[1] = 0;
      v339 = v96;
      v344 = a7;
      sub_2265350(v360, &v344);
      if ( v345 )
        ((void (__fastcall *)(__m128i *, __m128i *, __int64))v345)(&v344, &v344, 3);
      v98 = v363
          + -858993459 * ((__int64)(v360[4] - v360[2]) >> 3)
          + -858993459 * ((__int64)(v360[6] - v360[7]) >> 3)
          + 4 * (3 * ((__int64)(v360[9] - v360[5]) >> 3) - 3);
      if ( &_pthread_key_create )
        pthread_mutex_unlock(&mutex);
      sub_2210B50(v362);
      v99 = v98;
      sub_23CCA50(v352, v98);
      if ( v340 )
      {
        v99 = (__int64)&v339;
        v340((unsigned __int64 *)&v339, (unsigned __int64 *)&v339, 3);
      }
      if ( v91 )
      {
        if ( &_pthread_key_create )
        {
          v102 = _InterlockedExchangeAdd(v91 + 2, 0xFFFFFFFF);
        }
        else
        {
          v102 = *((_DWORD *)v91 + 2);
          v100 = (unsigned int)(v102 - 1);
          *((_DWORD *)v91 + 2) = v100;
        }
        if ( v102 == 1 )
        {
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v91 + 16LL))(v91);
          if ( &_pthread_key_create )
          {
            v178 = _InterlockedExchangeAdd(v91 + 3, 0xFFFFFFFF);
          }
          else
          {
            v178 = *((_DWORD *)v91 + 3);
            v100 = (unsigned int)(v178 - 1);
            *((_DWORD *)v91 + 3) = v100;
          }
          if ( v178 == 1 )
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v91 + 24LL))(v91);
        }
      }
      if ( v337 )
      {
        v99 = (__int64)&v336;
        v337(&v336, &v336, 3);
      }
      if ( *(_BYTE **)&v377[64] != &v377[80] )
      {
        v99 = *(_QWORD *)&v377[80] + 1LL;
        j_j___libc_free_0(*(unsigned __int64 *)&v377[64]);
      }
      if ( v370 != &v372 )
      {
        v99 = v372.m128i_i64[0] + 1;
        j_j___libc_free_0((unsigned __int64)v370);
      }
      if ( v81 )
      {
        if ( &_pthread_key_create )
        {
          v103 = _InterlockedExchangeAdd(v81 + 2, 0xFFFFFFFF);
        }
        else
        {
          v103 = *((_DWORD *)v81 + 2);
          v100 = (unsigned int)(v103 - 1);
          *((_DWORD *)v81 + 2) = v100;
        }
        if ( v103 == 1 )
        {
          (*(void (__fastcall **)(volatile signed __int32 *, __int64))(*(_QWORD *)v81 + 16LL))(v81, v99);
          if ( &_pthread_key_create )
          {
            v179 = _InterlockedExchangeAdd(v81 + 3, 0xFFFFFFFF);
          }
          else
          {
            v179 = *((_DWORD *)v81 + 3);
            v100 = (unsigned int)(v179 - 1);
            *((_DWORD *)v81 + 3) = v100;
          }
          if ( v179 == 1 )
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v81 + 24LL))(v81);
        }
      }
      if ( v333 != &v335 )
      {
        v99 = v335.m128i_i64[0] + 1;
        j_j___libc_free_0((unsigned __int64)v333);
      }
      if ( v59 )
      {
        sub_BA9C10(v59, v99, v100, v101);
        v99 = 880;
        j_j___libc_free_0((unsigned __int64)v59);
      }
      v104 = v315;
      ++v307;
      if ( v315 )
      {
        sub_BA9C10(v315, v99, v100, v101);
        j_j___libc_free_0((unsigned __int64)v104);
      }
      if ( v351 )
      {
        v166 = v350.m128i_u32[2];
        v351 = 0;
        if ( v350.m128i_i32[2] )
        {
          v167 = (_QWORD *)si128.m128i_i64[1];
          v168 = si128.m128i_i64[1] + 16LL * v350.m128i_u32[2];
          do
          {
            if ( *v167 != -4096 && *v167 != -8192 )
            {
              v169 = v167[1];
              if ( v169 )
                sub_B91220((__int64)(v167 + 1), v169);
            }
            v167 += 2;
          }
          while ( (_QWORD *)v168 != v167 );
          v166 = v350.m128i_u32[2];
        }
        sub_C7D6A0(si128.m128i_i64[1], 16LL * v166, 8);
      }
      v105 = v348.m128i_u32[2];
      if ( v348.m128i_i32[2] )
      {
        v106 = (_QWORD *)v347.m128i_i64[1];
        v364.m128i_i64[1] = 2;
        v365.m128i_i64[0] = 0;
        v364.m128i_i64[0] = (__int64)&unk_49DD7B0;
        v107 = v347.m128i_i64[1] + ((unsigned __int64)v348.m128i_u32[2] << 6);
        v375.m128i_i64[0] = (__int64)&unk_49DD7B0;
        v108 = -4096;
        v365.m128i_i64[1] = -4096;
        v366.m128i_i64[0] = 0;
        v375.m128i_i64[1] = 2;
        v376.m128i_i64[0] = 0;
        v376.m128i_i64[1] = -8192;
        *(_QWORD *)v377 = 0;
        while ( 1 )
        {
          v109 = v106[3];
          if ( v109 != v108 )
          {
            v108 = v376.m128i_i64[1];
            if ( v109 != v376.m128i_i64[1] )
            {
              v110 = v106[7];
              if ( v110 != -4096 && v110 != 0 && v110 != -8192 )
              {
                sub_BD60C0(v106 + 5);
                v109 = v106[3];
              }
              v108 = v109;
            }
          }
          *v106 = &unk_49DB368;
          if ( v108 != 0 && v108 != -4096 && v108 != -8192 )
            sub_BD60C0(v106 + 1);
          v106 += 8;
          if ( (_QWORD *)v107 == v106 )
            break;
          v108 = v365.m128i_i64[1];
        }
        v375.m128i_i64[0] = (__int64)&unk_49DB368;
        if ( v376.m128i_i64[1] != -4096 && v376.m128i_i64[1] != 0 && v376.m128i_i64[1] != -8192 )
          sub_BD60C0(&v375.m128i_i64[1]);
        v364.m128i_i64[0] = (__int64)&unk_49DB368;
        if ( v365.m128i_i64[1] != 0 && v365.m128i_i64[1] != -4096 && v365.m128i_i64[1] != -8192 )
          sub_BD60C0(&v364.m128i_i64[1]);
        v105 = v348.m128i_u32[2];
      }
      v43 = v105 << 6;
      sub_C7D6A0(v347.m128i_i64[1], v43, 8);
      nb += 32LL;
    }
    while ( v273 != nb );
  }
  sub_23CCD50(v352);
  if ( v314 && (unsigned int)sub_1682740(&v314, (char *)v43) )
    sub_C64ED0("GNU Jobserver support requested, but an error occurred", 1u);
  if ( *(_QWORD *)a14 )
  {
    v111 = 0;
    if ( (*(unsigned int (__fastcall **)(_QWORD, _QWORD))a14)(*(_QWORD *)(a14 + 8), 0) )
    {
      v112 = 1;
      goto LABEL_140;
    }
  }
  if ( *(int *)(a6 + 1560) >= 0 )
  {
    if ( v313 )
    {
      v258 = a12[1];
      v259 = *a12;
      v260 = v258 + 1;
      if ( (unsigned __int64 *)*a12 == a12 + 2 )
        v261 = 15;
      else
        v261 = a12[2];
      if ( v260 > v261 )
      {
        sub_2240BB0(a12, a12[1], 0, 0, 1u);
        v259 = *a12;
      }
      *(_BYTE *)(v259 + v258) = 0;
      a12[1] = v260;
      *(_BYTE *)(*a12 + v258 + 1) = 0;
    }
    else
    {
      *a3 = 0;
    }
    v180 = *(_DWORD **)(a6 + 1776);
    if ( (*v180 & 4) != 0 )
      *v180 ^= 4u;
  }
  v181 = (unsigned __int64)v331;
  if ( v330 == v331 )
  {
    v344 = 0u;
    v345 = 0x1000000000LL;
    goto LABEL_298;
  }
  v182 = v330;
  while ( 2 )
  {
    v344 = (__m128i)(unsigned __int64)v346;
    v364.m128i_i64[0] = (__int64)&unk_49DD288;
    v345 = 0;
    v367.m128i_i64[0] = (__int64)&v344;
    v364.m128i_i64[1] = 2;
    v365 = 0u;
    v366.m128i_i64[0] = 0;
    v366.m128i_i64[1] = 0x100000000LL;
    sub_CB5980((__int64)&v364, 0, 0, 0);
    sub_A3ACE0(*v182, (__int64)&v364, 0, 0, 0, 0);
    v185 = a10;
    v348.m128i_i64[0] = (__int64)"<split-module>";
    v348.m128i_i64[1] = 14;
    memset(v377, 0, 0x58u);
    v347 = v344;
    sub_A01950(
      (__int64)v342,
      (__int64)a10,
      (__int64)&v375,
      0,
      v186,
      v187,
      a7,
      (const __m128i *)v344.m128i_i64[0],
      v344.m128i_u64[1]);
    if ( v377[80] && (v377[80] = 0, *(_QWORD *)&v377[64]) )
    {
      v185 = (__m128i *)&v377[48];
      (*(void (__fastcall **)(_BYTE *, _BYTE *, __int64))&v377[64])(&v377[48], &v377[48], 3);
      if ( v377[40] )
      {
LABEL_247:
        v377[40] = 0;
        if ( *(_QWORD *)&v377[24] )
        {
          v185 = (__m128i *)&v377[8];
          (*(void (__fastcall **)(_BYTE *, _BYTE *, __int64))&v377[24])(&v377[8], &v377[8], 3);
          if ( v377[0] )
            goto LABEL_249;
          goto LABEL_236;
        }
      }
    }
    else if ( v377[40] )
    {
      goto LABEL_247;
    }
    if ( v377[0] )
    {
LABEL_249:
      v377[0] = 0;
      if ( v376.m128i_i64[0] )
      {
        v185 = &v375;
        ((void (__fastcall *)(__m128i *, __m128i *, __int64))v376.m128i_i64[0])(&v375, &v375, 3);
      }
    }
LABEL_236:
    v183 = BYTE8(v342[0]) & 1;
    BYTE8(v342[0]) = (2 * v183) | BYTE8(v342[0]) & 0xFD;
    if ( (_BYTE)v183 )
      sub_C64ED0("Failed to read the Bitcode", 1u);
    v184 = *(_QWORD *)&v342[0];
    *(_QWORD *)&v342[0] = 0;
    *v182 = v184;
    if ( (BYTE8(v342[0]) & 2) != 0 )
      sub_904700(v342);
    if ( (BYTE8(v342[0]) & 1) != 0 )
    {
      if ( *(_QWORD *)&v342[0] )
        (*(void (**)(void))(**(_QWORD **)&v342[0] + 8LL))();
    }
    else if ( *(_QWORD *)&v342[0] )
    {
      v291 = *(_QWORD *)&v342[0];
      sub_BA9C10(*(_QWORD ***)&v342[0], (__int64)v185, v183, (unsigned int)(2 * v183));
      j_j___libc_free_0(v291);
    }
    v364.m128i_i64[0] = (__int64)&unk_49DD388;
    sub_CB5840((__int64)&v364);
    if ( (_QWORD *)v344.m128i_i64[0] != v346 )
      _libc_free(v344.m128i_u64[0]);
    if ( v331 != ++v182 )
      continue;
    break;
  }
  v280 = v182;
  v344 = 0u;
  v345 = 0x1000000000LL;
  if ( v182 != v330 )
  {
    v287 = (unsigned __int64)v330;
    v192 = 0;
    v193 = 0;
    while ( 1 )
    {
      v292 = *(_QWORD *)v287 + 24LL;
      v308 = *(_QWORD *)(*(_QWORD *)v287 + 32LL);
      if ( v308 != v292 )
        break;
LABEL_296:
      v287 += 8LL;
      if ( v280 == (__int64 *)v287 )
      {
        v181 = (unsigned __int64)v330;
        goto LABEL_298;
      }
    }
    while ( 2 )
    {
      v208 = v308 - 56;
      if ( !v308 )
        v208 = 0;
      v209 = sub_BD5D20(v208);
      v211 = v210;
      nf = *(_QWORD *)a8 + 8LL * *(unsigned int *)(a8 + 8);
      v212 = sub_C92610();
      v213 = sub_C92860((__int64 *)a8, v209, v211, v212);
      if ( v213 == -1 )
        v214 = *(_QWORD *)a8 + 8LL * *(unsigned int *)(a8 + 8);
      else
        v214 = *(_QWORD *)a8 + 8LL * v213;
      if ( v214 != nf )
      {
        v192 = v211;
        v193 = v209;
        goto LABEL_278;
      }
      v194 = sub_C92610();
      v195 = sub_C92860(v344.m128i_i64, v193, v192, v194);
      if ( v195 == -1 || (v196 = v344.m128i_i64[0] + 8LL * v195, v196 == v344.m128i_i64[0] + 8LL * v344.m128i_u32[2]) )
      {
        v290 = 1;
        v197 = 0;
      }
      else
      {
        v197 = *(_DWORD *)(*(_QWORD *)v196 + 8LL);
        v290 = v197 + 1;
      }
      ne = v197;
      v198 = sub_C92610();
      v199 = sub_C92860((__int64 *)a8, v193, v192, v198);
      if ( v199 == -1 || (v200 = *(_QWORD *)a8 + 8LL * v199, v200 == *(_QWORD *)a8 + 8LL * *(unsigned int *)(a8 + 8)) )
        v201 = 0;
      else
        v201 = *(_DWORD *)(*(_QWORD *)v200 + 8LL);
      v375.m128i_i64[0] = (__int64)v209;
      v375.m128i_i64[1] = v211;
      v376.m128i_i64[0] = __PAIR64__(ne, v201);
      v202 = sub_C92610();
      v203 = sub_C92740(a9, v209, v211, v202);
      v204 = (_QWORD *)(*(_QWORD *)a9 + 8LL * v203);
      if ( *v204 )
      {
        if ( *v204 != -8 )
        {
LABEL_276:
          v375.m128i_i64[0] = (__int64)v193;
          v375.m128i_i64[1] = v192;
          v205 = sub_C92610();
          v206 = sub_C92740((__int64)&v344, (const void *)v375.m128i_i64[0], v375.m128i_u64[1], v205);
          v207 = (_QWORD *)(v344.m128i_i64[0] + 8LL * v206);
          if ( *v207 )
          {
            if ( *v207 != -8 )
              goto LABEL_278;
            LODWORD(v345) = v345 - 1;
          }
          ng = v344.m128i_i64[0] + 8LL * v206;
          v215 = sub_C7D670(v192 + 17, 8);
          v216 = (_QWORD *)ng;
          v217 = v215;
          if ( v192 )
          {
            memcpy((void *)(v215 + 16), v193, v192);
            v216 = (_QWORD *)ng;
          }
          *(_BYTE *)(v217 + v192 + 16) = 0;
          *(_QWORD *)v217 = v192;
          *(_DWORD *)(v217 + 8) = v290;
          *v216 = v217;
          ++v344.m128i_i32[3];
          sub_C929D0(v344.m128i_i64, v206);
LABEL_278:
          v308 = *(_QWORD *)(v308 + 8);
          if ( v292 == v308 )
            goto LABEL_296;
          continue;
        }
        --*(_DWORD *)(a9 + 16);
      }
      break;
    }
    v289 = v204;
    nh = v203;
    v218 = sub_C7D670(v211 + 17, 8);
    v219 = nh;
    v220 = v289;
    v221 = (_QWORD *)v218;
    if ( v211 )
    {
      v284 = (_QWORD *)v218;
      memcpy((void *)(v218 + 16), v209, v211);
      v219 = nh;
      v220 = v289;
      v221 = v284;
    }
    v222 = v376.m128i_i64[0];
    *((_BYTE *)v221 + v211 + 16) = 0;
    *v221 = v211;
    v221[1] = v222;
    *v220 = v221;
    ++*(_DWORD *)(a9 + 12);
    sub_C929D0((__int64 *)a9, v219);
    goto LABEL_276;
  }
  v181 = (unsigned __int64)v182;
LABEL_298:
  v223 = (__int64)((__int64)v331 - v181) >> 3;
  v224 = v223;
  if ( !(_DWORD)v223 )
  {
LABEL_319:
    v228 = *a3;
LABEL_320:
    v232 = *(_QWORD *)(v228 + 32);
    v309 = v228 + 24;
    if ( v232 == v228 + 24 )
      goto LABEL_335;
    while ( 1 )
    {
      v233 = v232 - 56;
      if ( !v232 )
        v233 = 0;
      v234 = sub_BD5D20(v233);
      v236 = v235;
      v237 = sub_C92610();
      v238 = sub_C92860((__int64 *)&v319, v234, v236, v237);
      if ( v238 == -1 )
        goto LABEL_324;
      v239 = v319 + 8LL * v238;
      if ( v239 == v319 + 8LL * (unsigned int)v320 )
        goto LABEL_324;
      v240 = *(_DWORD *)(*(_QWORD *)v239 + 8LL);
      v241 = v240 & 0xF;
      if ( (unsigned int)(v240 - 7) <= 1 )
        break;
      v242 = v241 | *(_BYTE *)(v233 + 32) & 0xF0;
      *(_BYTE *)(v233 + 32) = v242;
      if ( (v240 & 0xFu) - 7 <= 1 )
        goto LABEL_323;
      if ( (v242 & 0x30) == 0 )
        goto LABEL_324;
      if ( v241 == 9 )
      {
        v232 = *(_QWORD *)(v232 + 8);
        if ( v309 == v232 )
        {
LABEL_334:
          v228 = *a3;
LABEL_335:
          v243 = *(_QWORD *)(v228 + 16);
          v310 = v228 + 8;
          if ( v228 + 8 == v243 )
          {
LABEL_349:
            v112 = 1;
            goto LABEL_350;
          }
          while ( 1 )
          {
LABEL_340:
            v244 = v243 - 56;
            if ( !v243 )
              v244 = 0;
            v245 = sub_BD5D20(v244);
            v247 = v246;
            v248 = sub_C92610();
            v249 = sub_C92860((__int64 *)&v319, v245, v247, v248);
            if ( v249 == -1 )
              goto LABEL_339;
            v250 = v319 + 8LL * v249;
            if ( v250 == v319 + 8LL * (unsigned int)v320 )
              goto LABEL_339;
            v251 = *(_DWORD *)(*(_QWORD *)v250 + 8LL);
            v252 = v251 & 0xF;
            if ( (unsigned int)(v251 - 7) <= 1 )
              break;
            v253 = v252 | *(_BYTE *)(v244 + 32) & 0xF0;
            *(_BYTE *)(v244 + 32) = v253;
            if ( (v251 & 0xFu) - 7 <= 1 )
              goto LABEL_338;
            if ( (v253 & 0x30) == 0 )
              goto LABEL_339;
            if ( v252 != 9 )
              goto LABEL_338;
            v243 = *(_QWORD *)(v243 + 8);
            if ( v310 == v243 )
              goto LABEL_349;
          }
          *(_WORD *)(v244 + 32) = *(_WORD *)(v244 + 32) & 0xFCC0 | v251 & 0xF;
LABEL_338:
          *(_BYTE *)(v244 + 33) |= 0x40u;
LABEL_339:
          v243 = *(_QWORD *)(v243 + 8);
          if ( v310 == v243 )
            goto LABEL_349;
          goto LABEL_340;
        }
      }
      else
      {
LABEL_323:
        *(_BYTE *)(v233 + 33) |= 0x40u;
LABEL_324:
        v232 = *(_QWORD *)(v232 + 8);
        if ( v309 == v232 )
          goto LABEL_334;
      }
    }
    *(_WORD *)(v233 + 32) = *(_WORD *)(v233 + 32) & 0xFCC0 | v240 & 0xF;
    goto LABEL_323;
  }
  if ( v331 == (__int64 *)v181 )
  {
    if ( *(_QWORD *)(*(_QWORD *)v181 + 240LL) )
      goto LABEL_308;
LABEL_365:
    v112 = 0;
    *a3 = 0;
    goto LABEL_350;
  }
  v225 = 0;
  v226 = 0;
  do
  {
    if ( sub_2241AC0(*(_QWORD *)(v181 + 8 * v225) + 232LL, off_4C5D110) )
    {
      v181 = (unsigned __int64)v330;
      v227 = (unsigned __int64)&v330[v226];
      goto LABEL_304;
    }
    v181 = (unsigned __int64)v330;
    v225 = (unsigned int)(v226 + 1);
    v226 = v225;
  }
  while ( v225 < v331 - v330 );
  v227 = (unsigned __int64)v330;
LABEL_304:
  if ( !*(_QWORD *)(*(_QWORD *)v227 + 240LL) )
    goto LABEL_365;
  if ( (_DWORD)v223 == 1 )
  {
    v228 = *(_QWORD *)v181;
    *a3 = *(_QWORD *)v181;
    goto LABEL_320;
  }
  v224 = (__int64)((__int64)v331 - v181) >> 3;
LABEL_308:
  v347 = (__m128i)(unsigned __int64)&v348;
  v348.m128i_i8[0] = 0;
  v364 = (__m128i)(unsigned __int64)&v365;
  v365.m128i_i8[0] = 0;
  v375 = 0u;
  v376 = 0u;
  *(_QWORD *)v377 = 0;
  if ( v224 )
  {
    v262 = sub_22077B0(8 * ((unsigned __int64)(v224 + 63) >> 6));
    v375.m128i_i32[2] = 0;
    v263 = v262 + 8 * ((unsigned __int64)(v224 + 63) >> 6);
    v264 = (void *)v262;
    v375.m128i_i64[0] = v262;
    v265 = v224 + 63;
    *(_QWORD *)v377 = v263;
    if ( v224 >= 0 )
      v265 = v224;
    v266 = (__int64)v264 + 8 * (v265 >> 6);
    v267 = v224 % 64;
    if ( v224 % 64 < 0 )
    {
      LODWORD(v267) = v267 + 64;
      v266 -= 8;
    }
    *(_QWORD *)&v342[0] = v266;
    DWORD2(v342[0]) = v267;
    v376.m128i_i64[0] = v266;
    v376.m128i_i32[2] = v267;
    if ( v264 )
      memset(v264, 0, *(_QWORD *)v377 - (_QWORD)v264);
  }
  else
  {
    *(_QWORD *)&v342[0] = 0;
    DWORD2(v342[0]) = 0;
    v376.m128i_i64[0] = 0;
    v376.m128i_i32[2] = 0;
  }
  v229 = sub_3099E10(&v330, &v375, &v347, &v364, a4);
  if ( v229 )
  {
    *a3 = v229;
    if ( v375.m128i_i64[0] )
      j_j___libc_free_0(v375.m128i_u64[0]);
    if ( (__m128i *)v364.m128i_i64[0] != &v365 )
      j_j___libc_free_0(v364.m128i_u64[0]);
    if ( (__m128i *)v347.m128i_i64[0] != &v348 )
      j_j___libc_free_0(v347.m128i_u64[0]);
    goto LABEL_319;
  }
  sub_CEB590(&v347, 1, v230, v231);
  *a3 = 0;
  if ( v375.m128i_i64[0] )
    j_j___libc_free_0(v375.m128i_u64[0]);
  if ( (__m128i *)v364.m128i_i64[0] != &v365 )
    j_j___libc_free_0(v364.m128i_u64[0]);
  if ( (__m128i *)v347.m128i_i64[0] != &v348 )
    j_j___libc_free_0(v347.m128i_u64[0]);
  v112 = 0;
LABEL_350:
  v111 = v344.m128i_u32[3];
  v254 = v344.m128i_i64[0];
  if ( v344.m128i_i32[3] && v344.m128i_i32[2] )
  {
    v255 = 8LL * v344.m128i_u32[2];
    v256 = 0;
    do
    {
      v257 = *(_QWORD **)(v254 + v256);
      if ( v257 != (_QWORD *)-8LL && v257 )
      {
        v111 = *v257 + 17LL;
        sub_C7D6A0((__int64)v257, v111, 8);
        v254 = v344.m128i_i64[0];
      }
      v256 += 8;
    }
    while ( v256 != v255 );
  }
  _libc_free(v254);
LABEL_140:
  if ( v330 )
  {
    v111 = (char *)v332 - (char *)v330;
    j_j___libc_free_0((unsigned __int64)v330);
  }
  v113 = (unsigned __int64 *)v328;
  v114 = v327;
  if ( (unsigned __int64 *)v328 != v327 )
  {
    do
    {
      v115 = *v114;
      if ( *v114 )
      {
        sub_B6E710((_QWORD *)*v114);
        v111 = 8;
        j_j___libc_free_0(v115);
      }
      ++v114;
    }
    while ( v113 != v114 );
    v114 = v327;
  }
  if ( v114 )
  {
    v111 = (char *)v329 - (char *)v114;
    j_j___libc_free_0((unsigned __int64)v114);
  }
  sub_23CD060(v352, v111);
  v116 = v326;
  v117 = v325;
  if ( v326 != v325 )
  {
    do
    {
      v118 = *(unsigned int *)(v117 + 24);
      v119 = *(_QWORD *)(v117 + 8);
      v117 += 32LL;
      sub_C7D6A0(v119, 16 * v118, 8);
    }
    while ( v116 != v117 );
    v117 = v325;
  }
  if ( v117 )
    j_j___libc_free_0(v117);
  v120 = v322;
  if ( HIDWORD(v323) && (_DWORD)v323 )
  {
    v121 = 8LL * (unsigned int)v323;
    v122 = 0;
    do
    {
      v123 = *(_QWORD **)(v120 + v122);
      if ( v123 != (_QWORD *)-8LL && v123 )
      {
        sub_C7D6A0((__int64)v123, *v123 + 25LL, 8);
        v120 = v322;
      }
      v122 += 8;
    }
    while ( v121 != v122 );
  }
  _libc_free(v120);
  if ( HIDWORD(v320) )
  {
    v124 = v319;
    if ( (_DWORD)v320 )
    {
      v125 = 8LL * (unsigned int)v320;
      v126 = 0;
      do
      {
        v127 = *(_QWORD **)(v124 + v126);
        if ( v127 != (_QWORD *)-8LL && v127 )
        {
          sub_C7D6A0((__int64)v127, *v127 + 17LL, 8);
          v124 = v319;
        }
        v126 += 8;
      }
      while ( v126 != v125 );
    }
  }
  else
  {
    v124 = v319;
  }
  _libc_free(v124);
  if ( HIDWORD(v317) )
  {
    v128 = v316;
    if ( (_DWORD)v317 )
    {
      v129 = 8LL * (unsigned int)v317;
      v130 = 0;
      do
      {
        v131 = *(_QWORD **)(v128 + v130);
        if ( v131 != (_QWORD *)-8LL && v131 )
        {
          sub_C7D6A0((__int64)v131, *v131 + 17LL, 8);
          v128 = v316;
        }
        v130 += 8;
      }
      while ( v130 != v129 );
    }
  }
  else
  {
    v128 = v316;
  }
  _libc_free(v128);
  return v112;
}
