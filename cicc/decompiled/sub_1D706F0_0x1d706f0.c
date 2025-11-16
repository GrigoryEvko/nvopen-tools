// Function: sub_1D706F0
// Address: 0x1d706f0
//
__int64 __fastcall sub_1D706F0(
        __int64 a1,
        __int64 a2,
        double a3,
        double a4,
        __m128 a5,
        __m128 a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rax
  __int64 **v12; // rax
  __int64 **v13; // rdi
  __int64 v14; // r8
  __int64 v15; // r9
  double v16; // xmm4_8
  double v17; // xmm5_8
  __int64 v18; // rax
  __m128 v19; // xmm0
  __int64 *v20; // r13
  unsigned int v21; // r12d
  __int64 v22; // rax
  __int64 *v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rax
  __int64 v26; // rdx
  int v27; // edi
  unsigned __int64 v28; // rsi
  unsigned __int64 v29; // rsi
  unsigned int i; // eax
  __int64 v31; // rsi
  unsigned int v32; // eax
  __int64 v33; // rax
  __int64 v34; // rbx
  __int64 v35; // r13
  __int64 v36; // r14
  int v37; // r12d
  int v38; // r12d
  __int64 v39; // rdx
  __int64 v40; // r12
  _QWORD *v41; // rax
  __int64 v42; // r14
  __int64 v43; // r11
  __int64 v44; // r12
  __int64 v45; // rax
  __int64 *v46; // rax
  __int64 v47; // rax
  __int64 v48; // rbx
  __int64 *v49; // rax
  int v50; // eax
  unsigned int v51; // esi
  __int64 *v52; // rdx
  __int64 v53; // rbx
  __int64 v54; // rdi
  __int64 v55; // rdx
  int v56; // r8d
  __int64 v57; // rcx
  __int64 *v58; // rdi
  unsigned __int64 v59; // rcx
  unsigned __int64 v60; // rcx
  unsigned int j; // eax
  __int64 *v62; // rcx
  __int64 v63; // r10
  __int64 v64; // r9
  __int64 v65; // rax
  __int64 v66; // r14
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // r8
  __int64 *v70; // r9
  _QWORD *v71; // rax
  __int64 *v72; // rax
  __int64 v73; // r14
  __int64 v74; // r12
  __int64 v75; // rax
  __int64 v76; // r14
  __int64 v77; // rbx
  _QWORD *v78; // rax
  unsigned int v79; // esi
  __int64 v80; // rdx
  __int64 v81; // rcx
  __int64 v82; // rdi
  int v83; // r9d
  unsigned __int64 v84; // r8
  unsigned __int64 v85; // r8
  int v86; // eax
  __int64 v87; // r8
  unsigned int k; // eax
  __int64 *v89; // rbx
  __int64 v90; // r11
  unsigned int v91; // eax
  unsigned __int64 *v92; // rax
  __int64 *v93; // rdx
  __int64 *v94; // rdi
  __int64 v95; // rax
  __int64 v96; // rbx
  unsigned int v97; // eax
  __int64 v98; // r14
  __int64 v99; // rax
  __int64 v100; // rdx
  __int64 v101; // rbx
  __int64 v102; // rax
  int v103; // r8d
  int v104; // r9d
  __int64 v105; // rax
  __int64 *v106; // rax
  __int64 v107; // rdi
  char v108; // r15
  unsigned int v109; // esi
  __int64 v110; // rax
  __int64 v111; // rdx
  __int64 *v112; // rdx
  __int64 v113; // r10
  __int64 v114; // r9
  __int64 v115; // rax
  __int64 v116; // rcx
  __m128 *v117; // rdx
  __m128 *v118; // rdi
  char v119; // dl
  __int64 v120; // rax
  char v121; // cl
  __int64 v122; // rdx
  __int64 v123; // r8
  __int64 v124; // r13
  __int64 *v125; // r8
  __int64 *v126; // r15
  __int64 v127; // rsi
  __int64 v128; // rdi
  char v129; // cl
  __int64 v130; // r9
  __int64 v131; // rsi
  __int64 v132; // r10
  __int64 v133; // rax
  __int64 v134; // rcx
  __int64 v135; // r13
  char v136; // si
  unsigned int v137; // edx
  __int64 v138; // r9
  __int64 v139; // rcx
  __int64 v140; // r11
  __int64 v141; // r9
  __int64 *v142; // r13
  __int64 *v143; // r12
  __int64 *v144; // rcx
  __int64 v145; // r9
  __int64 v146; // r8
  __int64 *v147; // rdi
  __int64 *v148; // rax
  __int64 *v149; // rdx
  unsigned int v150; // eax
  __int64 **v151; // rax
  __int64 **v152; // rdx
  _QWORD *v153; // rcx
  int v154; // edx
  int v155; // r10d
  __int64 v156; // rax
  unsigned int v157; // edi
  __int64 v158; // rsi
  _QWORD *v159; // rcx
  int v160; // edx
  int v161; // edi
  unsigned __int64 v162; // rax
  unsigned __int64 v163; // rax
  unsigned __int64 v164; // rsi
  unsigned int m; // eax
  __int64 *v166; // rsi
  __int64 v167; // r10
  unsigned int v168; // eax
  __int64 v169; // rax
  __int64 v170; // r9
  __int64 *v171; // rax
  bool v172; // zf
  __int64 v173; // rax
  __int64 *v174; // r13
  __int64 *v175; // rax
  __int64 v176; // rcx
  __int64 *v177; // r12
  _QWORD *v178; // r8
  int v179; // edx
  unsigned int v180; // eax
  __int64 *v181; // rsi
  __int64 v182; // rdi
  char *v183; // rsi
  _QWORD *v184; // rax
  int v185; // r8d
  __int64 *v186; // rax
  char v187; // r8
  __int64 *v188; // rax
  __int64 v189; // r12
  int v190; // r8d
  unsigned int v191; // edx
  __int64 *v192; // rax
  __int64 v193; // r9
  int ii; // esi
  int v196; // r9d
  __int64 *v197; // r15
  __int64 v198; // r8
  __int64 v199; // rbx
  __int64 n; // rax
  __int64 *v201; // rdi
  int v202; // r11d
  int v203; // r10d
  unsigned int v204; // ecx
  __int64 *v205; // rdx
  __int64 v206; // r14
  int v207; // r11d
  __int64 v208; // r13
  unsigned int v209; // ecx
  __int64 *v210; // rdx
  __int64 v211; // r14
  int v212; // edx
  int v213; // r8d
  unsigned int v214; // ecx
  __int64 *v215; // rax
  __int64 v216; // rdx
  _QWORD *v217; // rsi
  int v218; // r8d
  _QWORD *v219; // rdi
  unsigned __int32 v220; // eax
  __int64 *v221; // rdx
  __int64 v222; // rcx
  char *v223; // rsi
  _QWORD *v224; // rax
  int v225; // r8d
  unsigned int v226; // eax
  __int64 **v227; // rax
  __int64 **v228; // rdx
  int v229; // edx
  unsigned int v230; // eax
  unsigned int v231; // r12d
  char v232; // al
  __int64 v233; // rdi
  __int64 v234; // rcx
  int v235; // edx
  __int64 v236; // r8
  int v237; // edx
  int v238; // r9d
  int v239; // r8d
  __int64 *v240; // rdi
  __int64 *v241; // rcx
  __int64 v242; // r14
  int v243; // esi
  __int64 v244; // rdi
  int v245; // r13d
  __int64 **v246; // rdx
  __int64 **v247; // rax
  __int64 v248; // r9
  __int64 v249; // rax
  __int64 v250; // r14
  __int64 v251; // rdx
  __int64 v252; // rcx
  __int64 v253; // r8
  __int64 *v254; // r9
  _QWORD *v255; // rax
  __int64 v256; // r14
  __int64 v257; // rax
  char v258; // di
  unsigned int v259; // esi
  __int64 v260; // rdx
  __int64 v261; // rax
  __int64 v262; // rcx
  __int64 *v263; // rax
  __int64 *v264; // rax
  unsigned int v265; // edi
  __int64 *v266; // rsi
  __int64 v267; // rax
  __int64 **v268; // rax
  __int64 **v269; // rdx
  __int64 v270; // rdx
  __int64 v271; // rsi
  int v272; // r10d
  __int64 v273; // rcx
  __int64 *v274; // rax
  __int64 v275; // r11
  int v276; // eax
  __int64 v277; // rax
  int v278; // ecx
  __int64 v279; // rcx
  __int64 *v280; // rax
  __int64 v281; // rsi
  unsigned __int64 v282; // rcx
  __int64 v283; // rcx
  __int64 v284; // rdx
  __int64 v285; // rcx
  __int64 v286; // r13
  __int64 *v287; // rdx
  bool v288; // cc
  __int64 v289; // rdi
  unsigned int v290; // esi
  __int64 v291; // r11
  __int64 v292; // r9
  __int64 v293; // r10
  unsigned int v294; // r8d
  unsigned __int64 v295; // rcx
  unsigned __int64 v296; // rcx
  unsigned int v297; // eax
  __int64 v298; // r8
  __int64 v299; // rdx
  char v300; // cl
  unsigned int v301; // eax
  __int64 v302; // rsi
  __int64 v303; // rdi
  __int64 v304; // rdx
  __int64 v305; // rax
  int v306; // eax
  int v307; // eax
  __int64 v308; // rax
  unsigned int v309; // eax
  __int64 v310; // rax
  char v311; // al
  __int64 *v312; // rdx
  __int64 v313; // rsi
  int v314; // r9d
  unsigned int v315; // edx
  __int64 *v316; // rax
  __int64 v317; // r11
  __int64 v318; // rdx
  char v319; // al
  _QWORD *v320; // rdx
  __int64 v321; // rsi
  int v322; // r9d
  unsigned int v323; // edx
  __int64 *v324; // rax
  __int64 v325; // r11
  int v326; // eax
  __int64 v327; // rax
  int v328; // eax
  int v329; // r10d
  unsigned int v330; // esi
  int v331; // eax
  int v332; // eax
  __int64 v333; // rax
  int v334; // eax
  int v335; // r10d
  unsigned int v336; // esi
  int v337; // eax
  int v338; // eax
  __int64 v339; // rax
  __int64 v340; // rdx
  int v341; // eax
  __int64 v342; // rax
  int v343; // eax
  int v344; // r10d
  int v345; // eax
  unsigned int v346; // eax
  unsigned int v347; // eax
  unsigned int v348; // ebx
  char v349; // al
  __int64 v350; // rdi
  __int64 **v351; // rdx
  __int64 **v352; // rax
  int v353; // eax
  int v354; // edi
  __int64 *v355; // rsi
  __int64 v356; // rax
  int v357; // eax
  __int64 v358; // [rsp+18h] [rbp-A18h]
  char v359; // [rsp+2Fh] [rbp-A01h]
  __int64 *v360; // [rsp+38h] [rbp-9F8h]
  __int64 v361; // [rsp+40h] [rbp-9F0h]
  int v362; // [rsp+48h] [rbp-9E8h]
  __int64 v363; // [rsp+48h] [rbp-9E8h]
  __int64 v365; // [rsp+60h] [rbp-9D0h]
  _QWORD *v366; // [rsp+60h] [rbp-9D0h]
  __int64 v367; // [rsp+60h] [rbp-9D0h]
  __int64 v368; // [rsp+60h] [rbp-9D0h]
  __int64 v369; // [rsp+60h] [rbp-9D0h]
  __int64 v370; // [rsp+60h] [rbp-9D0h]
  _QWORD *v371; // [rsp+70h] [rbp-9C0h]
  __int64 v372; // [rsp+70h] [rbp-9C0h]
  __int64 v373; // [rsp+70h] [rbp-9C0h]
  __int64 v374; // [rsp+70h] [rbp-9C0h]
  __int64 *v375; // [rsp+80h] [rbp-9B0h]
  __int64 *v376; // [rsp+80h] [rbp-9B0h]
  __int64 v377; // [rsp+80h] [rbp-9B0h]
  __m128i v378; // [rsp+90h] [rbp-9A0h] BYREF
  __m128i v379; // [rsp+A0h] [rbp-990h] BYREF
  __m128i v380; // [rsp+B0h] [rbp-980h] BYREF
  __int64 *v381; // [rsp+C0h] [rbp-970h]
  _BYTE v382[12]; // [rsp+C8h] [rbp-968h]
  _BYTE s[72]; // [rsp+D8h] [rbp-958h] BYREF
  const char *v384; // [rsp+120h] [rbp-910h] BYREF
  __int64 v385; // [rsp+128h] [rbp-908h]
  _BYTE v386[128]; // [rsp+130h] [rbp-900h] BYREF
  __m128i v387; // [rsp+1B0h] [rbp-880h] BYREF
  _WORD v388[64]; // [rsp+1C0h] [rbp-870h] BYREF
  __int64 v389; // [rsp+240h] [rbp-7F0h] BYREF
  int v390; // [rsp+248h] [rbp-7E8h] BYREF
  __int64 v391; // [rsp+250h] [rbp-7E0h]
  int *v392; // [rsp+258h] [rbp-7D8h]
  int *v393; // [rsp+260h] [rbp-7D0h]
  __int64 v394; // [rsp+268h] [rbp-7C8h]
  _BYTE *v395; // [rsp+270h] [rbp-7C0h] BYREF
  __int64 v396; // [rsp+278h] [rbp-7B8h]
  _BYTE v397[512]; // [rsp+280h] [rbp-7B0h] BYREF
  __int64 *v398; // [rsp+480h] [rbp-5B0h] BYREF
  __int64 v399; // [rsp+488h] [rbp-5A8h]
  __int64 **v400; // [rsp+490h] [rbp-5A0h] BYREF
  unsigned int v401; // [rsp+498h] [rbp-598h]
  __int64 *v402; // [rsp+510h] [rbp-520h] BYREF
  __int64 v403; // [rsp+518h] [rbp-518h]
  _BYTE v404[368]; // [rsp+520h] [rbp-510h] BYREF
  __int64 v405; // [rsp+690h] [rbp-3A0h] BYREF
  __int64 v406; // [rsp+698h] [rbp-398h]
  __int64 v407; // [rsp+6A0h] [rbp-390h]
  __int64 v408; // [rsp+6A8h] [rbp-388h]
  __int64 v409; // [rsp+6B0h] [rbp-380h]
  __int64 v410; // [rsp+6B8h] [rbp-378h] BYREF
  __int64 v411; // [rsp+6C0h] [rbp-370h]
  _QWORD *v412; // [rsp+6C8h] [rbp-368h] BYREF
  int v413; // [rsp+6D0h] [rbp-360h]
  __int64 *v414; // [rsp+7C8h] [rbp-268h] BYREF
  __int64 v415; // [rsp+7D0h] [rbp-260h]
  _BYTE v416[256]; // [rsp+7D8h] [rbp-258h] BYREF
  __int64 v417; // [rsp+8D8h] [rbp-158h] BYREF
  _BYTE *v418; // [rsp+8E0h] [rbp-150h]
  _BYTE *v419; // [rsp+8E8h] [rbp-148h]
  __int64 v420; // [rsp+8F0h] [rbp-140h]
  int v421; // [rsp+8F8h] [rbp-138h]
  _BYTE v422[304]; // [rsp+900h] [rbp-130h] BYREF

  v11 = *(_QWORD *)(a1 + 928);
  v405 = 0;
  v409 = v11;
  v12 = &v412;
  v406 = 0;
  v407 = 0;
  v408 = 0;
  v410 = 0;
  v411 = 1;
  do
    *v12++ = (__int64 *)-8LL;
  while ( v12 != &v414 );
  v417 = 0;
  v414 = (__int64 *)v416;
  v415 = 0x2000000000LL;
  v396 = 0x2000000000LL;
  v399 = 0x2000000000LL;
  v395 = v397;
  v13 = *(__int64 ***)(a1 + 920);
  v418 = v422;
  v419 = v422;
  v420 = 32;
  v421 = 0;
  v398 = (__int64 *)&v400;
  v375 = (__int64 *)sub_1599EF0(v13);
  v18 = (unsigned int)v399;
  if ( (unsigned int)v399 >= HIDWORD(v399) )
  {
    sub_16CD150((__int64)&v398, &v400, 0, 16, v14, v15);
    v18 = (unsigned int)v399;
  }
  v19 = (__m128)_mm_loadu_si128((const __m128i *)(a1 + 936));
  *(__m128 *)&v398[2 * v18] = v19;
  v20 = v398;
  v21 = v399 + 1;
  LODWORD(v399) = v21;
  while ( v21 )
  {
    v22 = v21--;
    v23 = &v20[2 * v22 - 2];
    v24 = *v23;
    v25 = v23[1];
    LODWORD(v399) = v21;
    v380.m128i_i64[0] = v24;
    v380.m128i_i64[1] = v25;
    if ( *(_BYTE *)(v24 + 16) <= 0x17u )
      v380.m128i_i64[1] = 0;
    v26 = *(unsigned int *)(a2 + 24);
    if ( (_DWORD)v26 )
    {
      LODWORD(v14) = v380.m128i_i32[2];
      v15 = *(_QWORD *)(a2 + 8);
      v27 = 1;
      v28 = (((((unsigned __int32)v380.m128i_i32[2] >> 9) ^ ((unsigned __int32)v380.m128i_i32[2] >> 4)
             | ((unsigned __int64)(((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4)) << 32))
            - 1
            - ((unsigned __int64)(((unsigned __int32)v380.m128i_i32[2] >> 9) ^ ((unsigned __int32)v380.m128i_i32[2] >> 4)) << 32)) >> 22)
          ^ ((((unsigned __int32)v380.m128i_i32[2] >> 9) ^ ((unsigned __int32)v380.m128i_i32[2] >> 4)
            | ((unsigned __int64)(((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4)) << 32))
           - 1
           - ((unsigned __int64)(((unsigned __int32)v380.m128i_i32[2] >> 9) ^ ((unsigned __int32)v380.m128i_i32[2] >> 4)) << 32));
      v29 = ((9 * (((v28 - 1 - (v28 << 13)) >> 8) ^ (v28 - 1 - (v28 << 13)))) >> 15)
          ^ (9 * (((v28 - 1 - (v28 << 13)) >> 8) ^ (v28 - 1 - (v28 << 13))));
      for ( i = (v26 - 1) & (((v29 - 1 - (v29 << 27)) >> 31) ^ (v29 - 1 - ((_DWORD)v29 << 27))); ; i = (v26 - 1) & v32 )
      {
        v31 = v15 + 24LL * i;
        if ( *(_OWORD *)v31 == __PAIR128__(v380.m128i_u64[1], v24) )
          break;
        if ( *(_QWORD *)v31 == -8 && *(_QWORD *)(v31 + 8) == -8 )
          goto LABEL_14;
        v32 = v27 + i;
        ++v27;
      }
      if ( v31 != v15 + 24 * v26 )
        continue;
    }
LABEL_14:
    v33 = (unsigned int)v396;
    if ( (unsigned int)v396 >= HIDWORD(v396) )
    {
      sub_16CD150((__int64)&v395, v397, 0, 16, v14, v15);
      v33 = (unsigned int)v396;
    }
    a5 = (__m128)_mm_load_si128(&v380);
    *(__m128 *)&v395[16 * v33] = a5;
    v35 = v380.m128i_i64[1];
    v34 = v380.m128i_i64[0];
    LODWORD(v396) = v396 + 1;
    v36 = *(_QWORD *)(v380.m128i_i64[1] + 8);
    v365 = *(_QWORD *)(v380.m128i_i64[0] + 40);
    if ( v36 )
    {
      while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v36) + 16) - 25) > 9u )
      {
        v36 = *(_QWORD *)(v36 + 8);
        if ( !v36 )
          goto LABEL_46;
      }
      v37 = 0;
      while ( 1 )
      {
        v36 = *(_QWORD *)(v36 + 8);
        if ( !v36 )
          break;
        while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v36) + 16) - 25) <= 9u )
        {
          v36 = *(_QWORD *)(v36 + 8);
          ++v37;
          if ( !v36 )
            goto LABEL_22;
        }
      }
LABEL_22:
      v38 = v37 + 1;
      if ( v35 == v365 )
      {
LABEL_23:
        if ( *(_BYTE *)(v34 + 16) != 79 )
        {
          v248 = *(_QWORD *)(v35 + 48);
          v388[0] = 259;
          if ( v248 )
            v248 -= 24;
          v387.m128i_i64[0] = (__int64)"sunk_phi";
          v374 = v248;
          v369 = *(_QWORD *)(a1 + 920);
          v249 = sub_1648B60(64);
          v250 = v249;
          if ( v249 )
          {
            sub_15F1EA0(v249, v369, 53, 0, 0, v374);
            *(_DWORD *)(v250 + 56) = v38;
            sub_164B780(v250, v387.m128i_i64);
            sub_1648880(v250, *(_DWORD *)(v250 + 56), 1);
          }
          sub_1D6B240(a2, v380.m128i_i64)[2] = v250;
          v387.m128i_i64[0] = v250;
          sub_1D6FF60((__int64)&v410, v387.m128i_i64, v251, v252, v253, v254);
          do
          {
            v35 = *(_QWORD *)(v35 + 8);
            if ( !v35 )
              goto LABEL_528;
            v255 = sub_1648700(v35);
          }
          while ( (unsigned __int8)(*((_BYTE *)v255 + 16) - 25) > 9u );
          v21 = v399;
LABEL_358:
          v256 = v255[5];
          v257 = 0x17FFFFFFE8LL;
          v258 = *(_BYTE *)(v34 + 23) & 0x40;
          v259 = *(_DWORD *)(v34 + 20) & 0xFFFFFFF;
          if ( v259 )
          {
            v260 = 24LL * *(unsigned int *)(v34 + 56) + 8;
            v261 = 0;
            do
            {
              v262 = v34 - 24LL * v259;
              if ( v258 )
                v262 = *(_QWORD *)(v34 - 8);
              if ( v256 == *(_QWORD *)(v262 + v260) )
              {
                v257 = 24 * v261;
                goto LABEL_365;
              }
              ++v261;
              v260 += 8;
            }
            while ( v259 != (_DWORD)v261 );
            v257 = 0x17FFFFFFE8LL;
          }
LABEL_365:
          if ( v258 )
          {
            v14 = *(_QWORD *)(*(_QWORD *)(v34 - 8) + v257);
            if ( HIDWORD(v399) > v21 )
              goto LABEL_367;
LABEL_372:
            v370 = v14;
            sub_16CD150((__int64)&v398, &v400, 0, 16, v14, v15);
            v21 = v399;
            v14 = v370;
          }
          else
          {
            v14 = *(_QWORD *)(v34 - 24LL * v259 + v257);
            if ( HIDWORD(v399) <= v21 )
              goto LABEL_372;
          }
LABEL_367:
          v263 = &v398[2 * v21];
          *v263 = v14;
          v263[1] = v256;
          v21 = v399 + 1;
          LODWORD(v399) = v399 + 1;
          while ( 1 )
          {
            v35 = *(_QWORD *)(v35 + 8);
            if ( !v35 )
              goto LABEL_31;
            v255 = sub_1648700(v35);
            if ( (unsigned __int8)(*((_BYTE *)v255 + 16) - 25) <= 9u )
              goto LABEL_358;
          }
        }
        v384 = sub_1649960(v34);
        v385 = v39;
        v388[0] = 261;
        v387.m128i_i64[0] = (__int64)&v384;
        v40 = *(_QWORD *)(v34 - 72);
        v41 = sub_1648A60(56, 3u);
        v42 = (__int64)v41;
        if ( v41 )
        {
          v371 = v41;
          v366 = v41 - 9;
          sub_15F1EA0((__int64)v41, *v375, 55, (__int64)(v41 - 9), 3, v34);
          sub_1593B40(v366, v40);
          sub_1593B40((_QWORD *)(v42 - 48), (__int64)v375);
          sub_1593B40((_QWORD *)(v42 - 24), (__int64)v375);
          sub_164B780(v42, v387.m128i_i64);
          v43 = (__int64)v371;
        }
        else
        {
          v43 = 0;
        }
        sub_15F4370(v43, v34, 0, 0);
        sub_1D6B240(a2, v380.m128i_i64)[2] = v42;
        sub_1412190((__int64)&v417, v42);
        v44 = *(_QWORD *)(v34 - 48);
        v45 = (unsigned int)v399;
        if ( (unsigned int)v399 >= HIDWORD(v399) )
        {
          sub_16CD150((__int64)&v398, &v400, 0, 16, v14, v15);
          v45 = (unsigned int)v399;
        }
        v46 = &v398[2 * v45];
        *v46 = v44;
        v46[1] = v35;
        v47 = (unsigned int)(v399 + 1);
        LODWORD(v399) = v47;
        v48 = *(_QWORD *)(v34 - 24);
        if ( HIDWORD(v399) <= (unsigned int)v47 )
        {
          sub_16CD150((__int64)&v398, &v400, 0, 16, v14, v15);
          v47 = (unsigned int)v399;
        }
        v49 = &v398[2 * v47];
        *v49 = v48;
        v49[1] = v35;
        v21 = v399 + 1;
        LODWORD(v399) = v399 + 1;
        goto LABEL_31;
      }
    }
    else
    {
LABEL_46:
      v38 = 0;
      if ( v35 == v365 )
        goto LABEL_23;
    }
    v64 = *(_QWORD *)(v35 + 48);
    v388[0] = 259;
    if ( v64 )
      v64 -= 24;
    v387.m128i_i64[0] = (__int64)"sunk_phi";
    v372 = v64;
    v368 = *(_QWORD *)(a1 + 920);
    v65 = sub_1648B60(64);
    v66 = v65;
    if ( v65 )
    {
      sub_15F1EA0(v65, v368, 53, 0, 0, v372);
      *(_DWORD *)(v66 + 56) = v38;
      sub_164B780(v66, v387.m128i_i64);
      sub_1648880(v66, *(_DWORD *)(v66 + 56), 1);
    }
    sub_1D6B240(a2, v380.m128i_i64)[2] = v66;
    v387.m128i_i64[0] = v66;
    sub_1D6FF60((__int64)&v410, v387.m128i_i64, v67, v68, v69, v70);
    do
    {
      v35 = *(_QWORD *)(v35 + 8);
      if ( !v35 )
      {
LABEL_528:
        v21 = v399;
        goto LABEL_31;
      }
      v71 = sub_1648700(v35);
    }
    while ( (unsigned __int8)(*((_BYTE *)v71 + 16) - 25) > 9u );
    v21 = v399;
LABEL_58:
    v73 = v71[5];
    if ( HIDWORD(v399) <= v21 )
    {
      sub_16CD150((__int64)&v398, &v400, 0, 16, v14, v15);
      v21 = v399;
    }
    v72 = &v398[2 * v21];
    *v72 = v34;
    v72[1] = v73;
    v21 = v399 + 1;
    LODWORD(v399) = v399 + 1;
    while ( 1 )
    {
      v35 = *(_QWORD *)(v35 + 8);
      if ( !v35 )
        break;
      v71 = sub_1648700(v35);
      if ( (unsigned __int8)(*((_BYTE *)v71 + 16) - 25) <= 9u )
        goto LABEL_58;
    }
LABEL_31:
    v20 = v398;
  }
  if ( v20 != (__int64 *)&v400 )
    _libc_free((unsigned __int64)v20);
  v50 = v396;
  if ( !(_DWORD)v396 )
    goto LABEL_81;
  do
  {
    v51 = *(_DWORD *)(a2 + 24);
    v52 = (__int64 *)&v395[16 * v50 - 16];
    v53 = *v52;
    v54 = v52[1];
    LODWORD(v396) = v50 - 1;
    v376 = (__int64 *)v53;
    v367 = v54;
    v384 = (const char *)v53;
    v385 = v54;
    if ( !v51 )
    {
      ++*(_QWORD *)a2;
LABEL_506:
      v51 *= 2;
      goto LABEL_507;
    }
    v55 = *(_QWORD *)(a2 + 8);
    v56 = 1;
    v57 = ((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4);
    v58 = 0;
    v59 = (((v57 | ((unsigned __int64)(((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4)) << 32)) - 1 - (v57 << 32)) >> 22)
        ^ ((v57 | ((unsigned __int64)(((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4)) << 32)) - 1 - (v57 << 32));
    v60 = ((9 * (((v59 - 1 - (v59 << 13)) >> 8) ^ (v59 - 1 - (v59 << 13)))) >> 15)
        ^ (9 * (((v59 - 1 - (v59 << 13)) >> 8) ^ (v59 - 1 - (v59 << 13))));
    for ( j = (v51 - 1) & (((v60 - 1 - (v60 << 27)) >> 31) ^ (v60 - 1 - ((_DWORD)v60 << 27))); ; j = (v51 - 1) & v346 )
    {
      v62 = (__int64 *)(v55 + 24LL * j);
      v63 = *v62;
      if ( v53 == *v62 && v367 == v62[1] )
      {
        v74 = v62[2];
        goto LABEL_64;
      }
      if ( v63 == -8 )
        break;
      if ( v63 == -16 && v62[1] == -16 && !v58 )
        v58 = (__int64 *)(v55 + 24LL * j);
LABEL_534:
      v346 = v56 + j;
      ++v56;
    }
    if ( v62[1] != -8 )
      goto LABEL_534;
    v357 = *(_DWORD *)(a2 + 16);
    if ( v58 )
      v62 = v58;
    ++*(_QWORD *)a2;
    v341 = v357 + 1;
    if ( 4 * v341 >= 3 * v51 )
      goto LABEL_506;
    v340 = v53;
    if ( v51 - *(_DWORD *)(a2 + 20) - v341 <= v51 >> 3 )
    {
LABEL_507:
      sub_1D6AF90(a2, v51);
      sub_1D66970(a2, (__int64 *)&v384, &v398);
      v62 = v398;
      v340 = (__int64)v384;
      v341 = *(_DWORD *)(a2 + 16) + 1;
    }
    *(_DWORD *)(a2 + 16) = v341;
    if ( *v62 != -8 || v62[1] != -8 )
      --*(_DWORD *)(a2 + 20);
    *v62 = v340;
    v342 = v385;
    v74 = 0;
    v62[2] = 0;
    v62[1] = v342;
LABEL_64:
    if ( *(_BYTE *)(v74 + 16) == 79 )
    {
      v310 = 0;
      v288 = *(_BYTE *)(*(_QWORD *)(v53 - 48) + 16LL) <= 0x17u;
      v387.m128i_i64[0] = *(_QWORD *)(v53 - 48);
      if ( !v288 )
        v310 = v367;
      v387.m128i_i64[1] = v310;
      v311 = sub_1D66970(a2, v387.m128i_i64, &v398);
      v312 = v398;
      if ( v311 )
      {
        v313 = v398[2];
LABEL_464:
        v314 = v408 - 1;
        while ( 1 )
        {
          if ( !(_DWORD)v408 )
            goto LABEL_469;
          v315 = v314 & (((unsigned int)v313 >> 9) ^ ((unsigned int)v313 >> 4));
          v316 = (__int64 *)(v406 + 16LL * v315);
          v317 = *v316;
          if ( v313 != *v316 )
            break;
LABEL_466:
          if ( (__int64 *)(v406 + 16LL * (unsigned int)v408) == v316 )
            goto LABEL_469;
          v313 = v316[1];
        }
        v328 = 1;
        while ( v317 != -8 )
        {
          v329 = v328 + 1;
          v315 = v314 & (v328 + v315);
          v316 = (__int64 *)(v406 + 16LL * v315);
          v317 = *v316;
          if ( v313 == *v316 )
            goto LABEL_466;
          v328 = v329;
        }
LABEL_469:
        sub_1593B40((_QWORD *)(v74 - 48), v313);
        v318 = 0;
        if ( *(_BYTE *)(*(_QWORD *)(v53 - 24) + 16LL) > 0x17u )
          v318 = v367;
        v398 = *(__int64 **)(v53 - 24);
        v399 = v318;
        v319 = sub_1D66970(a2, (__int64 *)&v398, (__int64 **)&v380);
        v320 = (_QWORD *)v380.m128i_i64[0];
        if ( v319 )
        {
          v321 = *(_QWORD *)(v380.m128i_i64[0] + 16);
          goto LABEL_473;
        }
        v336 = *(_DWORD *)(a2 + 24);
        v337 = *(_DWORD *)(a2 + 16);
        ++*(_QWORD *)a2;
        v338 = v337 + 1;
        if ( 4 * v338 >= 3 * v336 )
        {
          v336 *= 2;
        }
        else if ( v336 - *(_DWORD *)(a2 + 20) - v338 > v336 >> 3 )
        {
          goto LABEL_500;
        }
        sub_1D6AF90(a2, v336);
        sub_1D66970(a2, (__int64 *)&v398, (__int64 **)&v380);
        v320 = (_QWORD *)v380.m128i_i64[0];
        v338 = *(_DWORD *)(a2 + 16) + 1;
LABEL_500:
        *(_DWORD *)(a2 + 16) = v338;
        if ( *v320 != -8 || v320[1] != -8 )
          --*(_DWORD *)(a2 + 20);
        v321 = 0;
        *v320 = v398;
        v339 = v399;
        v320[2] = 0;
        v320[1] = v339;
LABEL_473:
        v322 = v408 - 1;
        while ( 1 )
        {
          if ( !(_DWORD)v408 )
            goto LABEL_478;
          v323 = v322 & (((unsigned int)v321 >> 9) ^ ((unsigned int)v321 >> 4));
          v324 = (__int64 *)(v406 + 16LL * v323);
          v325 = *v324;
          if ( *v324 != v321 )
            break;
LABEL_475:
          if ( (__int64 *)(v406 + 16LL * (unsigned int)v408) == v324 )
            goto LABEL_478;
          v321 = v324[1];
        }
        v334 = 1;
        while ( v325 != -8 )
        {
          v335 = v334 + 1;
          v323 = v322 & (v334 + v323);
          v324 = (__int64 *)(v406 + 16LL * v323);
          v325 = *v324;
          if ( v321 == *v324 )
            goto LABEL_475;
          v334 = v335;
        }
LABEL_478:
        sub_1593B40((_QWORD *)(v74 - 24), v321);
        v79 = *(_DWORD *)(a2 + 24);
        if ( v79 )
          goto LABEL_71;
LABEL_479:
        ++*(_QWORD *)a2;
LABEL_480:
        v79 *= 2;
LABEL_481:
        sub_1D6AF90(a2, v79);
        sub_1D66970(a2, (__int64 *)&v384, &v398);
        v89 = v398;
        v80 = (__int64)v384;
        v326 = *(_DWORD *)(a2 + 16) + 1;
LABEL_482:
        *(_DWORD *)(a2 + 16) = v326;
        if ( *v89 != -8 || v89[1] != -8 )
          --*(_DWORD *)(a2 + 20);
        *v89 = v80;
        v327 = v385;
        v89[2] = 0;
        v89[1] = v327;
        goto LABEL_80;
      }
      v330 = *(_DWORD *)(a2 + 24);
      v331 = *(_DWORD *)(a2 + 16);
      ++*(_QWORD *)a2;
      v332 = v331 + 1;
      if ( 4 * v332 >= 3 * v330 )
      {
        v330 *= 2;
      }
      else if ( v330 - *(_DWORD *)(a2 + 20) - v332 > v330 >> 3 )
      {
        goto LABEL_491;
      }
      sub_1D6AF90(a2, v330);
      sub_1D66970(a2, v387.m128i_i64, &v398);
      v312 = v398;
      v332 = *(_DWORD *)(a2 + 16) + 1;
LABEL_491:
      *(_DWORD *)(a2 + 16) = v332;
      if ( *v312 != -8 || v312[1] != -8 )
        --*(_DWORD *)(a2 + 20);
      v313 = 0;
      *v312 = v387.m128i_i64[0];
      v333 = v387.m128i_i64[1];
      v312[2] = 0;
      v312[1] = v333;
      goto LABEL_464;
    }
    v373 = *(_QWORD *)(v53 + 40);
    v75 = 0;
    if ( *(_BYTE *)(v53 + 16) == 77 )
      v75 = v53;
    v76 = v75;
    v77 = *(_QWORD *)(v367 + 8);
    if ( !v77 )
      goto LABEL_70;
    while ( 1 )
    {
      v78 = sub_1648700(v77);
      if ( (unsigned __int8)(*((_BYTE *)v78 + 16) - 25) <= 9u )
        break;
      v77 = *(_QWORD *)(v77 + 8);
      if ( !v77 )
        goto LABEL_70;
    }
LABEL_420:
    v286 = v78[5];
    v287 = v376;
    if ( v373 == v367 )
    {
      v299 = 0x17FFFFFFE8LL;
      v300 = *(_BYTE *)(v76 + 23) & 0x40;
      v301 = *(_DWORD *)(v76 + 20) & 0xFFFFFFF;
      if ( v301 )
      {
        v302 = 0;
        v303 = 24LL * *(unsigned int *)(v76 + 56) + 8;
        do
        {
          v304 = v76 - 24LL * v301;
          if ( v300 )
            v304 = *(_QWORD *)(v76 - 8);
          if ( v286 == *(_QWORD *)(v304 + v303) )
          {
            v299 = 24 * v302;
            goto LABEL_439;
          }
          ++v302;
          v303 += 8;
        }
        while ( v301 != (_DWORD)v302 );
        v299 = 0x17FFFFFFE8LL;
      }
LABEL_439:
      if ( v300 )
        v305 = *(_QWORD *)(v76 - 8);
      else
        v305 = v76 - 24LL * v301;
      v287 = *(__int64 **)(v305 + v299);
    }
    v288 = *((_BYTE *)v287 + 16) <= 0x17u;
    v289 = 0;
    v290 = *(_DWORD *)(a2 + 24);
    v398 = v287;
    if ( !v288 )
      v289 = v286;
    v399 = v289;
    if ( !v290 )
    {
      ++*(_QWORD *)a2;
LABEL_451:
      v290 *= 2;
      goto LABEL_452;
    }
    v291 = *(_QWORD *)(a2 + 8);
    v292 = v290 - 1;
    v293 = 0;
    v362 = 1;
    v294 = (unsigned int)v289 >> 9;
    v295 = (((v294 ^ ((unsigned int)v289 >> 4)
            | ((unsigned __int64)(((unsigned int)v287 >> 9) ^ ((unsigned int)v287 >> 4)) << 32))
           - 1
           - ((unsigned __int64)(v294 ^ ((unsigned int)v289 >> 4)) << 32)) >> 22)
         ^ ((v294 ^ ((unsigned int)v289 >> 4)
           | ((unsigned __int64)(((unsigned int)v287 >> 9) ^ ((unsigned int)v287 >> 4)) << 32))
          - 1
          - ((unsigned __int64)(v294 ^ ((unsigned int)v289 >> 4)) << 32));
    v296 = 9 * (((v295 - 1 - (v295 << 13)) >> 8) ^ (v295 - 1 - (v295 << 13)));
    v297 = v292
         & ((((v296 ^ (v296 >> 15)) - 1 - ((v296 ^ (v296 >> 15)) << 27)) >> 31)
          ^ ((v296 ^ (v296 >> 15)) - 1 - (((unsigned int)v296 ^ (unsigned int)(v296 >> 15)) << 27)));
    while ( 2 )
    {
      v273 = v291 + 24LL * v297;
      v298 = *(_QWORD *)v273;
      if ( v287 == *(__int64 **)v273 && v289 == *(_QWORD *)(v273 + 8) )
      {
        v270 = *(_QWORD *)(v273 + 16);
        goto LABEL_399;
      }
      if ( v298 != -8 )
      {
        if ( v298 == -16 && *(_QWORD *)(v273 + 8) == -16 && !v293 )
          v293 = v291 + 24LL * v297;
        goto LABEL_459;
      }
      if ( *(_QWORD *)(v273 + 8) != -8 )
      {
LABEL_459:
        v309 = v362 + v297;
        ++v362;
        v297 = v292 & v309;
        continue;
      }
      break;
    }
    v345 = *(_DWORD *)(a2 + 16);
    if ( v293 )
      v273 = v293;
    ++*(_QWORD *)a2;
    v307 = v345 + 1;
    if ( 4 * v307 >= 3 * v290 )
      goto LABEL_451;
    if ( v290 - *(_DWORD *)(a2 + 20) - v307 <= v290 >> 3 )
    {
LABEL_452:
      sub_1D6AF90(a2, v290);
      sub_1D66970(a2, (__int64 *)&v398, (__int64 **)&v387);
      v273 = v387.m128i_i64[0];
      v287 = v398;
      v307 = *(_DWORD *)(a2 + 16) + 1;
    }
    *(_DWORD *)(a2 + 16) = v307;
    if ( *(_QWORD *)v273 != -8 || *(_QWORD *)(v273 + 8) != -8 )
      --*(_DWORD *)(a2 + 20);
    *(_QWORD *)v273 = v287;
    v308 = v399;
    v270 = 0;
    *(_QWORD *)(v273 + 16) = 0;
    *(_QWORD *)(v273 + 8) = v308;
LABEL_399:
    v271 = v406 + 16LL * (unsigned int)v408;
    v272 = v408 - 1;
    while ( (_DWORD)v408 )
    {
      v273 = v272 & (((unsigned int)v270 >> 9) ^ ((unsigned int)v270 >> 4));
      v274 = (__int64 *)(v406 + 16 * v273);
      v275 = *v274;
      if ( v270 != *v274 )
      {
        v306 = 1;
        while ( v275 != -8 )
        {
          v292 = (unsigned int)(v306 + 1);
          v273 = v272 & (unsigned int)(v306 + v273);
          v274 = (__int64 *)(v406 + 16LL * (unsigned int)v273);
          v275 = *v274;
          if ( v270 == *v274 )
            goto LABEL_401;
          v306 = v292;
        }
        break;
      }
LABEL_401:
      if ( v274 == (__int64 *)v271 )
        break;
      v270 = v274[1];
    }
    v276 = *(_DWORD *)(v74 + 20) & 0xFFFFFFF;
    if ( v276 == *(_DWORD *)(v74 + 56) )
    {
      v363 = v270;
      sub_15F55D0(v74, v271, v270, v273, v406, v292);
      v270 = v363;
      v276 = *(_DWORD *)(v74 + 20) & 0xFFFFFFF;
    }
    v277 = (v276 + 1) & 0xFFFFFFF;
    v278 = v277 | *(_DWORD *)(v74 + 20) & 0xF0000000;
    *(_DWORD *)(v74 + 20) = v278;
    if ( (v278 & 0x40000000) != 0 )
      v279 = *(_QWORD *)(v74 - 8);
    else
      v279 = v74 - 24 * v277;
    v280 = (__int64 *)(v279 + 24LL * (unsigned int)(v277 - 1));
    if ( *v280 )
    {
      v281 = v280[1];
      v282 = v280[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v282 = v281;
      if ( v281 )
        *(_QWORD *)(v281 + 16) = *(_QWORD *)(v281 + 16) & 3LL | v282;
    }
    *v280 = v270;
    if ( v270 )
    {
      v283 = *(_QWORD *)(v270 + 8);
      v280[1] = v283;
      if ( v283 )
        *(_QWORD *)(v283 + 16) = (unsigned __int64)(v280 + 1) | *(_QWORD *)(v283 + 16) & 3LL;
      v280[2] = v280[2] & 3 | (v270 + 8);
      *(_QWORD *)(v270 + 8) = v280;
    }
    v284 = *(_DWORD *)(v74 + 20) & 0xFFFFFFF;
    if ( (*(_BYTE *)(v74 + 23) & 0x40) != 0 )
      v285 = *(_QWORD *)(v74 - 8);
    else
      v285 = v74 - 24 * v284;
    *(_QWORD *)(v285 + 8LL * (unsigned int)(v284 - 1) + 24LL * *(unsigned int *)(v74 + 56) + 8) = v286;
    while ( 1 )
    {
      v77 = *(_QWORD *)(v77 + 8);
      if ( !v77 )
        break;
      v78 = sub_1648700(v77);
      if ( (unsigned __int8)(*((_BYTE *)v78 + 16) - 25) <= 9u )
        goto LABEL_420;
    }
LABEL_70:
    v79 = *(_DWORD *)(a2 + 24);
    if ( !v79 )
      goto LABEL_479;
LABEL_71:
    v80 = (__int64)v384;
    v81 = v385;
    v83 = 1;
    v84 = (((((unsigned int)v385 >> 9) ^ ((unsigned int)v385 >> 4)
           | ((unsigned __int64)(((unsigned int)v384 >> 9) ^ ((unsigned int)v384 >> 4)) << 32))
          - 1
          - ((unsigned __int64)(((unsigned int)v385 >> 9) ^ ((unsigned int)v385 >> 4)) << 32)) >> 22)
        ^ ((((unsigned int)v385 >> 9) ^ ((unsigned int)v385 >> 4)
          | ((unsigned __int64)(((unsigned int)v384 >> 9) ^ ((unsigned int)v384 >> 4)) << 32))
         - 1
         - ((unsigned __int64)(((unsigned int)v385 >> 9) ^ ((unsigned int)v385 >> 4)) << 32));
    v85 = ((9 * (((v84 - 1 - (v84 << 13)) >> 8) ^ (v84 - 1 - (v84 << 13)))) >> 15)
        ^ (9 * (((v84 - 1 - (v84 << 13)) >> 8) ^ (v84 - 1 - (v84 << 13))));
    v86 = ((v85 - 1 - (v85 << 27)) >> 31) ^ (v85 - 1 - ((_DWORD)v85 << 27));
    v87 = 0;
    for ( k = (v79 - 1) & v86; ; k = (v79 - 1) & v91 )
    {
      v82 = *(_QWORD *)(a2 + 8);
      v89 = (__int64 *)(v82 + 24LL * k);
      v90 = *v89;
      if ( (const char *)*v89 == v384 && v89[1] == v385 )
        break;
      if ( v90 == -8 )
      {
        if ( v89[1] == -8 )
        {
          v353 = *(_DWORD *)(a2 + 16);
          if ( v87 )
            v89 = (__int64 *)v87;
          ++*(_QWORD *)a2;
          v326 = v353 + 1;
          if ( 4 * v326 >= 3 * v79 )
            goto LABEL_480;
          v81 = v79 - *(_DWORD *)(a2 + 20) - v326;
          if ( (unsigned int)v81 > v79 >> 3 )
            goto LABEL_482;
          goto LABEL_481;
        }
      }
      else if ( v90 == -16 && v89[1] == -16 && !v87 )
      {
        v87 = v82 + 24LL * k;
      }
      v91 = v83 + k;
      ++v83;
    }
LABEL_80:
    v89[2] = sub_1D6A950(
               (__int64)&v405,
               v74,
               v19,
               a4,
               *(double *)a5.m128_u64,
               *(double *)a6.m128_u64,
               v16,
               v17,
               a9,
               a10,
               v80,
               v81,
               v87);
    v50 = v396;
  }
  while ( (_DWORD)v396 );
LABEL_81:
  if ( !byte_4FC2500 && v421 != HIDWORD(v420) )
    goto LABEL_520;
  v398 = 0;
  v399 = 1;
  v359 = byte_4FC25E0;
  v92 = (unsigned __int64 *)&v400;
  do
  {
    *v92 = -8;
    v92 += 2;
    *(v92 - 1) = -8;
  }
  while ( v92 != (unsigned __int64 *)&v402 );
  v93 = (__int64 *)s;
  v380.m128i_i64[0] = 0;
  v94 = (__int64 *)s;
  v402 = (__int64 *)v404;
  v403 = 0x800000000LL;
  v95 = 0;
  v380.m128i_i64[1] = (__int64)s;
  v381 = (__int64 *)s;
  *(_QWORD *)v382 = 8;
  *(_DWORD *)&v382[8] = 0;
  if ( !(_DWORD)v415 )
  {
    if ( (v399 & 1) == 0 )
    {
      v359 = 1;
      goto LABEL_236;
    }
LABEL_238:
    v187 = sub_1D66970(a2, (__int64 *)(a1 + 936), &v398);
    v188 = v398;
    if ( !v187 )
      v188 = (__int64 *)(*(_QWORD *)(a2 + 8) + 24LL * *(unsigned int *)(a2 + 24));
    v189 = v188[2];
    v190 = v408 - 1;
    while ( (_DWORD)v408 )
    {
      v191 = v190 & (((unsigned int)v189 >> 9) ^ ((unsigned int)v189 >> 4));
      v192 = (__int64 *)(v406 + 16LL * v191);
      v193 = *v192;
      if ( *v192 != v189 )
      {
        v343 = 1;
        while ( v193 != -8 )
        {
          v344 = v343 + 1;
          v191 = v190 & (v343 + v191);
          v192 = (__int64 *)(v406 + 16LL * v191);
          v193 = *v192;
          if ( v189 == *v192 )
            goto LABEL_242;
          v343 = v344;
        }
        goto LABEL_245;
      }
LABEL_242:
      if ( v192 == (__int64 *)(v406 + 16LL * (unsigned int)v408) )
        goto LABEL_245;
      v189 = v192[1];
    }
    goto LABEL_245;
  }
  v358 = a2;
  while ( 2 )
  {
    v96 = *v414;
    v380.m128i_i64[0] = v95 + 1;
    if ( v94 == v93 )
    {
      *(_QWORD *)&v382[4] = 0;
      goto LABEL_384;
    }
    v97 = 4 * (*(_DWORD *)&v382[4] - *(_DWORD *)&v382[8]);
    if ( v97 < 0x20 )
      v97 = 32;
    if ( *(_DWORD *)v382 <= v97 )
    {
      memset(v94, -1, 8LL * *(unsigned int *)v382);
      v93 = (__int64 *)v380.m128i_i64[1];
      *(_QWORD *)&v382[4] = 0;
      if ( v381 != (__int64 *)v380.m128i_i64[1] )
        goto LABEL_92;
LABEL_384:
      v265 = 0;
LABEL_385:
      if ( v265 < *(_DWORD *)v382 )
      {
        *(_DWORD *)&v382[4] = v265 + 1;
        *v93 = v96;
        ++v380.m128i_i64[0];
        goto LABEL_93;
      }
LABEL_92:
      sub_16CCBA0((__int64)&v380, v96);
      goto LABEL_93;
    }
    sub_16CC920((__int64)&v380);
    v264 = (__int64 *)v380.m128i_i64[1];
    if ( v381 != (__int64 *)v380.m128i_i64[1] )
      goto LABEL_92;
    v265 = *(_DWORD *)&v382[4];
    v93 = (__int64 *)(v380.m128i_i64[1] + 8LL * *(unsigned int *)&v382[4]);
    if ( (__int64 *)v380.m128i_i64[1] == v93 )
      goto LABEL_385;
    v266 = 0;
    while ( v96 != *v264 )
    {
      if ( *v264 == -2 )
        v266 = v264;
      if ( v93 == ++v264 )
      {
        if ( !v266 )
          goto LABEL_385;
        *v266 = v96;
        --*(_DWORD *)&v382[8];
        ++v380.m128i_i64[0];
        break;
      }
    }
LABEL_93:
    v98 = v96;
    v99 = sub_157F280(*(_QWORD *)(v96 + 40));
    v377 = v100;
    v101 = v99;
    if ( v99 != v100 )
    {
      while ( 2 )
      {
        if ( v101 == v98 )
          goto LABEL_95;
        v387.m128i_i64[0] = v98;
        v385 = 0x800000000LL;
        v384 = v386;
        v387.m128i_i64[1] = v101;
        sub_1D705A0((__int64)&v398, &v387);
        v105 = (unsigned int)v385;
        if ( (unsigned int)v385 >= HIDWORD(v385) )
        {
          sub_16CD150((__int64)&v384, v386, 0, 16, v103, v104);
          v105 = (unsigned int)v385;
        }
        v106 = (__int64 *)&v384[16 * v105];
        v107 = 0;
        v108 = 1;
        *v106 = v98;
        v106[1] = v101;
        v390 = 0;
        v391 = 0;
        v109 = v385 + 1;
        v387.m128i_i64[0] = (__int64)v388;
        v387.m128i_i64[1] = 0x800000000LL;
        v392 = &v390;
        v393 = &v390;
        v110 = 0;
        LODWORD(v385) = v109;
        v394 = 0;
        if ( !v109 )
          goto LABEL_137;
LABEL_103:
        v111 = v109--;
        v112 = (__int64 *)&v384[16 * v111 - 16];
        v113 = *v112;
        v114 = v112[1];
        LODWORD(v385) = v109;
        v378.m128i_i64[0] = v113;
        v378.m128i_i64[1] = v114;
        if ( !v110 )
        {
          v115 = v387.m128i_u32[2];
          v116 = v387.m128i_i64[0];
          v117 = (__m128 *)(v387.m128i_i64[0] + 16LL * v387.m128i_u32[2]);
          if ( (__m128 *)v387.m128i_i64[0] != v117 )
          {
            v118 = (__m128 *)v387.m128i_i64[0];
            while ( v113 != v118->m128_u64[0] || v114 != v118->m128_u64[1] )
            {
              if ( v117 == ++v118 )
                goto LABEL_199;
            }
            if ( v117 != v118 )
            {
              if ( !v109 )
                goto LABEL_191;
              goto LABEL_111;
            }
          }
LABEL_199:
          if ( v387.m128i_u32[2] > 7uLL )
          {
            while ( 1 )
            {
              sub_1D66260(&v389, (const __m128i *)(v116 + 16 * v115 - 16));
              v172 = v387.m128i_i32[2] == 1;
              v115 = (unsigned int)--v387.m128i_i32[2];
              if ( v172 )
                break;
              v116 = v387.m128i_i64[0];
            }
            sub_1D66260(&v389, &v378);
          }
          else
          {
            if ( v387.m128i_i32[2] >= (unsigned __int32)v387.m128i_i32[3] )
            {
              sub_16CD150((__int64)&v387, v388, 0, 16, v387.m128i_i32[2], v114);
              v117 = (__m128 *)(v387.m128i_i64[0] + 16LL * v387.m128i_u32[2]);
            }
            a6 = (__m128)_mm_load_si128(&v378);
            *v117 = a6;
            ++v387.m128i_i32[2];
          }
          goto LABEL_113;
        }
        sub_1D66260(&v389, &v378);
        if ( v119 )
        {
LABEL_113:
          v120 = v378.m128i_i64[0];
          v121 = *(_BYTE *)(v378.m128i_i64[0] + 23);
          v122 = *(_DWORD *)(v378.m128i_i64[0] + 20) & 0xFFFFFFF;
          if ( (v121 & 0x40) != 0 )
            v123 = *(_QWORD *)(v378.m128i_i64[0] - 8);
          else
            v123 = v378.m128i_i64[0] - 24 * v122;
          v124 = v123 + 24LL * *(unsigned int *)(v378.m128i_i64[0] + 56) + 8;
          if ( v124 + 8 * v122 != v124 )
          {
            v125 = (__int64 *)(v124 + 8 * v122);
            v126 = (__int64 *)v124;
            while ( 1 )
            {
              v127 = 0x17FFFFFFE8LL;
              v128 = *v126;
              v129 = v121 & 0x40;
              if ( (_DWORD)v122 )
              {
                v130 = 24LL * *(unsigned int *)(v120 + 56) + 8;
                v131 = 0;
                do
                {
                  v132 = v120 - 24LL * (unsigned int)v122;
                  if ( v129 )
                    v132 = *(_QWORD *)(v120 - 8);
                  if ( v128 == *(_QWORD *)(v132 + v130) )
                  {
                    v127 = 24 * v131;
                    goto LABEL_124;
                  }
                  ++v131;
                  v130 += 8;
                }
                while ( (_DWORD)v122 != (_DWORD)v131 );
                v127 = 0x17FFFFFFE8LL;
              }
LABEL_124:
              if ( v129 )
                v133 = *(_QWORD *)(v120 - 8);
              else
                v133 = v120 - 24 * v122;
              v134 = 0x17FFFFFFE8LL;
              v135 = *(_QWORD *)(v133 + v127);
              v136 = *(_BYTE *)(v378.m128i_i64[1] + 23) & 0x40;
              v137 = *(_DWORD *)(v378.m128i_i64[1] + 20) & 0xFFFFFFF;
              if ( v137 )
              {
                v138 = 24LL * *(unsigned int *)(v378.m128i_i64[1] + 56) + 8;
                v139 = 0;
                do
                {
                  v140 = v378.m128i_i64[1] - 24LL * v137;
                  if ( v136 )
                    v140 = *(_QWORD *)(v378.m128i_i64[1] - 8);
                  if ( v128 == *(_QWORD *)(v140 + v138) )
                  {
                    v134 = 24 * v139;
                    goto LABEL_133;
                  }
                  ++v139;
                  v138 += 8;
                }
                while ( v137 != (_DWORD)v139 );
                v134 = 0x17FFFFFFE8LL;
              }
LABEL_133:
              if ( v136 )
              {
                v141 = *(_QWORD *)(*(_QWORD *)(v378.m128i_i64[1] - 8) + v134);
                if ( v135 == v141 )
                  goto LABEL_167;
              }
              else
              {
                v141 = *(_QWORD *)(v378.m128i_i64[1] - 24LL * v137 + v134);
                if ( v135 == v141 )
                {
LABEL_167:
                  if ( v125 == ++v126 )
                    goto LABEL_190;
                  goto LABEL_168;
                }
              }
              if ( *(_BYTE *)(v135 + 16) != 77 || *(_BYTE *)(v141 + 16) != 77 )
                goto LABEL_136;
              if ( (v411 & 1) != 0 )
              {
                v153 = &v412;
                v154 = 31;
              }
              else
              {
                v153 = v412;
                if ( !v413 )
                  goto LABEL_136;
                v154 = v413 - 1;
              }
              v155 = 1;
              v156 = ((unsigned int)v135 >> 4) ^ ((unsigned int)v135 >> 9);
              v157 = v156 & v154;
              v158 = v153[(unsigned int)v156 & v154];
              if ( v158 != v135 )
                break;
LABEL_173:
              if ( *(_QWORD *)(v135 + 40) != *(_QWORD *)(v141 + 40) )
                goto LABEL_136;
              if ( (v399 & 1) != 0 )
              {
                v159 = &v400;
                v160 = 7;
LABEL_176:
                v161 = 1;
                v162 = (((unsigned int)v141 >> 9) ^ ((unsigned int)v141 >> 4) | (unsigned __int64)(v156 << 32))
                     - 1
                     - ((unsigned __int64)(((unsigned int)v141 >> 9) ^ ((unsigned int)v141 >> 4)) << 32);
                v163 = ((v162 >> 22) ^ v162) - 1 - (((v162 >> 22) ^ v162) << 13);
                v164 = ((9 * ((v163 >> 8) ^ v163)) >> 15) ^ (9 * ((v163 >> 8) ^ v163));
                for ( m = v160 & (((v164 - 1 - (v164 << 27)) >> 31) ^ (v164 - 1 - ((_DWORD)v164 << 27))); ; m = v160 & v168 )
                {
                  v166 = &v159[2 * m];
                  v167 = *v166;
                  if ( v135 == *v166 )
                  {
                    if ( v141 == v166[1] )
                      goto LABEL_167;
                    if ( v167 != -8 )
                      goto LABEL_179;
                  }
                  else if ( v167 != -8 )
                  {
                    goto LABEL_179;
                  }
                  if ( v166[1] == -8 )
                    goto LABEL_187;
LABEL_179:
                  v168 = v161 + m;
                  ++v161;
                }
              }
              v159 = v400;
              if ( v401 )
              {
                v160 = v401 - 1;
                goto LABEL_176;
              }
LABEL_187:
              v360 = v125;
              v379.m128i_i64[1] = v141;
              v361 = v141;
              v379.m128i_i64[0] = v135;
              sub_1D705A0((__int64)&v398, &v379);
              v169 = (unsigned int)v385;
              v170 = v361;
              v125 = v360;
              if ( (unsigned int)v385 >= HIDWORD(v385) )
              {
                sub_16CD150((__int64)&v384, v386, 0, 16, (int)v360, v361);
                v169 = (unsigned int)v385;
                v125 = v360;
                v170 = v361;
              }
              ++v126;
              v171 = (__int64 *)&v384[16 * v169];
              *v171 = v135;
              v171[1] = v170;
              LODWORD(v385) = v385 + 1;
              if ( v125 == v126 )
                goto LABEL_190;
LABEL_168:
              v120 = v378.m128i_i64[0];
              v121 = *(_BYTE *)(v378.m128i_i64[0] + 23);
              v122 = *(_DWORD *)(v378.m128i_i64[0] + 20) & 0xFFFFFFF;
            }
            while ( v158 != -8 )
            {
              v157 = v154 & (v155 + v157);
              v158 = v153[v157];
              if ( v135 == v158 )
                goto LABEL_173;
              ++v155;
            }
LABEL_136:
            v107 = v391;
            v108 = 0;
LABEL_137:
            sub_1D5AE70(v107);
            if ( (_WORD *)v387.m128i_i64[0] != v388 )
              _libc_free(v387.m128i_u64[0]);
            if ( v384 != v386 )
              _libc_free((unsigned __int64)v384);
            v142 = v402;
            v143 = &v402[2 * (unsigned int)v403];
            if ( v108 )
            {
              if ( v143 == v402 )
                goto LABEL_295;
              v197 = v402;
              while ( 1 )
              {
                v198 = *v197;
                v199 = v197[1];
                v387.m128i_i64[0] = v198;
                n = v198;
                v201 = (__int64 *)(v406 + 16LL * (unsigned int)v408);
                v202 = v408 - 1;
                while ( 1 )
                {
                  if ( !(_DWORD)v408 )
                  {
                    if ( v198 == n )
                    {
                      v208 = v199;
                      v199 = v198;
                    }
                    else
                    {
LABEL_270:
                      v207 = v408 - 1;
                      while ( 1 )
                      {
                        v387.m128i_i64[0] = v199;
                        if ( *(_BYTE *)(n + 16) != 77 )
                          n = 0;
                        v208 = n;
                        for ( n = v199; ; n = v210[1] )
                        {
                          if ( !(_DWORD)v408 )
                            goto LABEL_278;
                          v209 = v207 & (((unsigned int)n >> 9) ^ ((unsigned int)n >> 4));
                          v210 = (__int64 *)(v406 + 16LL * v209);
                          v211 = *v210;
                          if ( *v210 != n )
                            break;
LABEL_275:
                          if ( v210 == v201 )
                            goto LABEL_278;
                        }
                        v212 = 1;
                        while ( v211 != -8 )
                        {
                          v213 = v212 + 1;
                          v209 = v207 & (v212 + v209);
                          v210 = (__int64 *)(v406 + 16LL * v209);
                          v211 = *v210;
                          if ( n == *v210 )
                            goto LABEL_275;
                          v212 = v213;
                        }
LABEL_278:
                        if ( n == v199 )
                          break;
                        v199 = v208;
                      }
                      if ( (_DWORD)v408 )
                      {
                        v203 = v408 - 1;
                        goto LABEL_286;
                      }
                    }
                    ++v405;
                    goto LABEL_318;
                  }
                  v203 = v408 - 1;
                  v204 = v202 & (((unsigned int)n >> 9) ^ ((unsigned int)n >> 4));
                  v205 = (__int64 *)(v406 + 16LL * v204);
                  v206 = *v205;
                  if ( n != *v205 )
                    break;
LABEL_266:
                  if ( v201 == v205 )
                    goto LABEL_308;
                  n = v205[1];
                }
                v229 = 1;
                while ( v206 != -8 )
                {
                  v245 = v229 + 1;
                  v204 = v202 & (v229 + v204);
                  v205 = (__int64 *)(v406 + 16LL * v204);
                  v206 = *v205;
                  if ( n == *v205 )
                    goto LABEL_266;
                  v229 = v245;
                }
LABEL_308:
                if ( v198 != n )
                  goto LABEL_270;
                v208 = v199;
                v199 = v198;
LABEL_286:
                v214 = v203 & (((unsigned int)v199 >> 9) ^ ((unsigned int)v199 >> 4));
                v215 = (__int64 *)(v406 + 16LL * v214);
                v216 = *v215;
                if ( v199 != *v215 )
                {
                  v239 = 1;
                  v240 = 0;
                  while ( v216 != -8 )
                  {
                    if ( v216 == -16 && !v240 )
                      v240 = v215;
                    v214 = v203 & (v239 + v214);
                    v215 = (__int64 *)(v406 + 16LL * v214);
                    v216 = *v215;
                    if ( v199 == *v215 )
                      goto LABEL_287;
                    ++v239;
                  }
                  if ( v240 )
                    v215 = v240;
                  ++v405;
                  v235 = v407 + 1;
                  if ( 4 * ((int)v407 + 1) < (unsigned int)(3 * v408) )
                  {
                    if ( (int)v408 - HIDWORD(v407) - v235 > (unsigned int)v408 >> 3 )
                    {
LABEL_320:
                      LODWORD(v407) = v235;
                      if ( *v215 != -8 )
                        --HIDWORD(v407);
                      *v215 = v199;
                      v215[1] = v208;
                      goto LABEL_287;
                    }
                    sub_176F940((__int64)&v405, v408);
                    if ( (_DWORD)v408 )
                    {
                      v241 = 0;
                      LODWORD(v242) = (v408 - 1) & (((unsigned int)v199 >> 9) ^ ((unsigned int)v199 >> 4));
                      v235 = v407 + 1;
                      v243 = 1;
                      v215 = (__int64 *)(v406 + 16LL * (unsigned int)v242);
                      v244 = *v215;
                      if ( *v215 != v199 )
                      {
                        while ( v244 != -8 )
                        {
                          if ( !v241 && v244 == -16 )
                            v241 = v215;
                          v242 = ((_DWORD)v408 - 1) & (unsigned int)(v242 + v243);
                          v215 = (__int64 *)(v406 + 16 * v242);
                          v244 = *v215;
                          if ( v199 == *v215 )
                            goto LABEL_320;
                          ++v243;
                        }
                        if ( v241 )
                          v215 = v241;
                      }
                      goto LABEL_320;
                    }
LABEL_590:
                    LODWORD(v407) = v407 + 1;
                    BUG();
                  }
LABEL_318:
                  sub_176F940((__int64)&v405, 2 * v408);
                  if ( (_DWORD)v408 )
                  {
                    LODWORD(v234) = (v408 - 1) & (((unsigned int)v199 >> 9) ^ ((unsigned int)v199 >> 4));
                    v235 = v407 + 1;
                    v215 = (__int64 *)(v406 + 16LL * (unsigned int)v234);
                    v236 = *v215;
                    if ( *v215 != v199 )
                    {
                      v354 = 1;
                      v355 = 0;
                      while ( v236 != -8 )
                      {
                        if ( !v355 && v236 == -16 )
                          v355 = v215;
                        v234 = ((_DWORD)v408 - 1) & (unsigned int)(v234 + v354);
                        v215 = (__int64 *)(v406 + 16 * v234);
                        v236 = *v215;
                        if ( v199 == *v215 )
                          goto LABEL_320;
                        ++v354;
                      }
                      if ( v355 )
                        v215 = v355;
                    }
                    goto LABEL_320;
                  }
                  goto LABEL_590;
                }
LABEL_287:
                sub_164D160(
                  v387.m128i_i64[0],
                  v208,
                  v19,
                  a4,
                  *(double *)a5.m128_u64,
                  *(double *)a6.m128_u64,
                  v16,
                  v17,
                  a9,
                  a10);
                if ( (v411 & 1) != 0 )
                {
                  v217 = &v412;
                  v218 = 31;
                  goto LABEL_289;
                }
                v217 = v412;
                v218 = v413 - 1;
                if ( v413 )
                {
LABEL_289:
                  v219 = (_QWORD *)v387.m128i_i64[0];
                  v220 = v218
                       & (((unsigned __int32)v387.m128i_i32[0] >> 9)
                        ^ ((unsigned __int32)v387.m128i_i32[0] >> 4));
                  v221 = &v217[v220];
                  v222 = *v221;
                  if ( *v221 == v387.m128i_i64[0] )
                  {
LABEL_290:
                    *v221 = -16;
                    ++HIDWORD(v411);
                    LODWORD(v411) = (2 * ((unsigned int)v411 >> 1) - 2) | v411 & 1;
                    v223 = (char *)&v414[(unsigned int)v415];
                    v224 = sub_1D5A7E0(v414, (__int64)v223, v387.m128i_i64);
                    if ( v224 + 1 != (_QWORD *)v223 )
                    {
                      memmove(v224, v224 + 1, v223 - (char *)(v224 + 1));
                      v225 = v415;
                    }
                    LODWORD(v415) = v225 - 1;
                    goto LABEL_293;
                  }
                  v237 = 1;
                  while ( v222 != -8 )
                  {
                    v238 = v237 + 1;
                    v220 = v218 & (v237 + v220);
                    v221 = &v217[v220];
                    v222 = *v221;
                    if ( v387.m128i_i64[0] == *v221 )
                      goto LABEL_290;
                    v237 = v238;
                  }
                }
                else
                {
LABEL_293:
                  v219 = (_QWORD *)v387.m128i_i64[0];
                }
                sub_15F20C0(v219);
                v197 += 2;
                if ( v143 == v197 )
                {
LABEL_295:
                  v398 = (__int64 *)((char *)v398 + 1);
                  v226 = (unsigned int)v399 >> 1;
                  if ( !((unsigned int)v399 >> 1) && !HIDWORD(v399) )
                    goto LABEL_303;
                  if ( (v399 & 1) != 0 )
                  {
                    v228 = &v402;
                    v227 = (__int64 **)&v400;
                    goto LABEL_301;
                  }
                  if ( 4 * v226 >= v401 || v401 <= 0x40 )
                  {
                    v227 = v400;
                    v228 = &v400[2 * v401];
                    if ( v400 != v228 )
                    {
                      do
                      {
LABEL_301:
                        *v227 = (__int64 *)-8LL;
                        v227 += 2;
                        *(v227 - 1) = (__int64 *)-8LL;
                      }
                      while ( v227 != v228 );
                    }
                    v399 &= 1u;
                    goto LABEL_303;
                  }
                  if ( v226 && (v347 = v226 - 1) != 0 )
                  {
                    _BitScanReverse(&v347, v347);
                    v348 = 1 << (33 - (v347 ^ 0x1F));
                    if ( v348 - 9 <= 0x36 )
                    {
                      v348 = 64;
                      j___libc_free_0(v400);
                      v349 = v399;
                      v350 = 1024;
                      goto LABEL_569;
                    }
                    if ( v348 == v401 )
                    {
                      sub_1D67FB0((__int64)&v398);
                      goto LABEL_303;
                    }
                    j___libc_free_0(v400);
                    v349 = v399 | 1;
                    LOBYTE(v399) = v399 | 1;
                    if ( v348 > 8 )
                    {
                      v350 = 16LL * v348;
LABEL_569:
                      LOBYTE(v399) = v349 & 0xFE;
                      v356 = sub_22077B0(v350);
                      v401 = v348;
                      v400 = (__int64 **)v356;
                    }
                  }
                  else
                  {
                    j___libc_free_0(v400);
                    LOBYTE(v399) = v399 | 1;
                  }
                  v399 &= 1u;
                  if ( v399 )
                  {
                    v351 = &v402;
                    v352 = (__int64 **)&v400;
                  }
                  else
                  {
                    v352 = v400;
                    v351 = &v400[2 * v401];
                    if ( v400 == v351 )
                      goto LABEL_303;
                  }
                  do
                  {
                    if ( v352 )
                    {
                      *v352 = (__int64 *)-8LL;
                      v352[1] = (__int64 *)-8LL;
                    }
                    v352 += 2;
                  }
                  while ( v352 != v351 );
LABEL_303:
                  v94 = v381;
                  v93 = (__int64 *)v380.m128i_i64[1];
                  LODWORD(v403) = 0;
LABEL_217:
                  if ( !(_DWORD)v415 )
                    goto LABEL_230;
                  goto LABEL_218;
                }
              }
            }
            if ( v143 != v402 )
            {
              v144 = v381;
              v145 = v380.m128i_i64[1];
              do
              {
LABEL_146:
                v146 = *v142;
                if ( v144 != (__int64 *)v145 )
                  goto LABEL_144;
                v147 = &v144[*(unsigned int *)&v382[4]];
                if ( v147 != v144 )
                {
                  v148 = v144;
                  v149 = 0;
                  while ( v146 != *v148 )
                  {
                    if ( *v148 == -2 )
                      v149 = v148;
                    if ( v147 == ++v148 )
                    {
                      if ( !v149 )
                        goto LABEL_204;
                      v142 += 2;
                      *v149 = v146;
                      v144 = v381;
                      --*(_DWORD *)&v382[8];
                      v145 = v380.m128i_i64[1];
                      ++v380.m128i_i64[0];
                      if ( v143 != v142 )
                        goto LABEL_146;
                      goto LABEL_155;
                    }
                  }
                  goto LABEL_145;
                }
LABEL_204:
                if ( *(_DWORD *)&v382[4] < *(_DWORD *)v382 )
                {
                  ++*(_DWORD *)&v382[4];
                  *v147 = v146;
                  v145 = v380.m128i_i64[1];
                  ++v380.m128i_i64[0];
                  v144 = v381;
                }
                else
                {
LABEL_144:
                  sub_16CCBA0((__int64)&v380, *v142);
                  v144 = v381;
                  v145 = v380.m128i_i64[1];
                }
LABEL_145:
                v142 += 2;
              }
              while ( v143 != v142 );
            }
LABEL_155:
            v398 = (__int64 *)((char *)v398 + 1);
            v150 = (unsigned int)v399 >> 1;
            if ( !((unsigned int)v399 >> 1) && !HIDWORD(v399) )
              goto LABEL_163;
            if ( (v399 & 1) != 0 )
            {
              v152 = &v402;
              v151 = (__int64 **)&v400;
              goto LABEL_161;
            }
            if ( 4 * v150 >= v401 || v401 <= 0x40 )
            {
              v151 = v400;
              v152 = &v400[2 * v401];
              if ( v400 != v152 )
              {
                do
                {
LABEL_161:
                  *v151 = (__int64 *)-8LL;
                  v151 += 2;
                  *(v151 - 1) = (__int64 *)-8LL;
                }
                while ( v151 != v152 );
              }
              v399 &= 1u;
              goto LABEL_163;
            }
            if ( v150 && (v230 = v150 - 1) != 0 )
            {
              _BitScanReverse(&v230, v230);
              v231 = 1 << (33 - (v230 ^ 0x1F));
              if ( v231 - 9 <= 0x36 )
              {
                v231 = 64;
                j___libc_free_0(v400);
                v232 = v399;
                v233 = 1024;
LABEL_388:
                LOBYTE(v399) = v232 & 0xFE;
                v267 = sub_22077B0(v233);
                v401 = v231;
                v400 = (__int64 **)v267;
              }
              else
              {
                if ( v231 == v401 )
                {
                  v399 &= 1u;
                  if ( v399 )
                  {
                    v269 = &v402;
                    v268 = (__int64 **)&v400;
                  }
                  else
                  {
                    v268 = v400;
                    v269 = &v400[2 * v231];
                  }
                  do
                  {
                    if ( v268 )
                    {
                      *v268 = (__int64 *)-8LL;
                      v268[1] = (__int64 *)-8LL;
                    }
                    v268 += 2;
                  }
                  while ( v268 != v269 );
                  goto LABEL_163;
                }
                j___libc_free_0(v400);
                v232 = v399 | 1;
                LOBYTE(v399) = v399 | 1;
                if ( v231 > 8 )
                {
                  v233 = 16LL * v231;
                  goto LABEL_388;
                }
              }
            }
            else
            {
              j___libc_free_0(v400);
              LOBYTE(v399) = v399 | 1;
            }
            v399 &= 1u;
            if ( v399 )
            {
              v246 = &v402;
              v247 = (__int64 **)&v400;
            }
            else
            {
              v247 = v400;
              v246 = &v400[2 * v401];
              if ( v400 == v246 )
                goto LABEL_163;
            }
            do
            {
              if ( v247 )
              {
                *v247 = (__int64 *)-8LL;
                v247[1] = (__int64 *)-8LL;
              }
              v247 += 2;
            }
            while ( v247 != v246 );
LABEL_163:
            LODWORD(v403) = 0;
            if ( !v101 )
              BUG();
LABEL_95:
            v102 = *(_QWORD *)(v101 + 32);
            if ( !v102 )
              BUG();
            v101 = 0;
            if ( *(_BYTE *)(v102 - 8) == 77 )
              v101 = v102 - 24;
            if ( v377 == v101 )
              goto LABEL_210;
            continue;
          }
        }
        break;
      }
LABEL_190:
      v109 = v385;
      if ( !(_DWORD)v385 )
      {
LABEL_191:
        v107 = v391;
        v108 = 1;
        goto LABEL_137;
      }
LABEL_111:
      v110 = v394;
      goto LABEL_103;
    }
LABEL_210:
    v94 = v381;
    v93 = (__int64 *)v380.m128i_i64[1];
    if ( v359 )
    {
      v173 = *(unsigned int *)&v382[4];
      if ( v381 != (__int64 *)v380.m128i_i64[1] )
        v173 = *(unsigned int *)v382;
      v174 = &v381[v173];
      if ( v174 == v381 )
        goto LABEL_217;
      v175 = v381;
      while ( 1 )
      {
        v176 = *v175;
        v177 = v175;
        if ( (unsigned __int64)*v175 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v174 == ++v175 )
          goto LABEL_217;
      }
      if ( v175 == v174 )
        goto LABEL_217;
      v387.m128i_i64[0] = *v175;
      if ( (v411 & 1) != 0 )
      {
LABEL_221:
        v178 = &v412;
        v179 = 31;
        goto LABEL_222;
      }
      while ( 1 )
      {
        v178 = v412;
        if ( v413 )
        {
          v179 = v413 - 1;
LABEL_222:
          v180 = v179 & (((unsigned int)v176 >> 9) ^ ((unsigned int)v176 >> 4));
          v181 = &v178[v180];
          v182 = *v181;
          if ( *v181 != v176 )
          {
            for ( ii = 1; ; ii = v196 )
            {
              if ( v182 == -8 )
                goto LABEL_226;
              v196 = ii + 1;
              v180 = v179 & (ii + v180);
              v181 = &v178[v180];
              v182 = *v181;
              if ( *v181 == v176 )
                break;
            }
          }
          *v181 = -16;
          ++HIDWORD(v411);
          LODWORD(v411) = (2 * ((unsigned int)v411 >> 1) - 2) | v411 & 1;
          v183 = (char *)&v414[(unsigned int)v415];
          v184 = sub_1D5A7E0(v414, (__int64)v183, v387.m128i_i64);
          if ( v184 + 1 != (_QWORD *)v183 )
          {
            memmove(v184, v184 + 1, v183 - (char *)(v184 + 1));
            v185 = v415;
          }
          LODWORD(v415) = v185 - 1;
        }
LABEL_226:
        v186 = v177 + 1;
        if ( v177 + 1 == v174 )
          break;
        while ( 1 )
        {
          v176 = *v186;
          v177 = v186;
          if ( (unsigned __int64)*v186 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v174 == ++v186 )
            goto LABEL_229;
        }
        if ( v186 == v174 )
          break;
        v387.m128i_i64[0] = *v186;
        if ( (v411 & 1) != 0 )
          goto LABEL_221;
      }
LABEL_229:
      v94 = v381;
      v93 = (__int64 *)v380.m128i_i64[1];
      if ( !(_DWORD)v415 )
      {
LABEL_230:
        v359 = 1;
        a2 = v358;
        goto LABEL_231;
      }
LABEL_218:
      v95 = v380.m128i_i64[0];
      continue;
    }
    break;
  }
  a2 = v358;
LABEL_231:
  if ( v93 != v94 )
    _libc_free((unsigned __int64)v94);
  if ( v402 != (__int64 *)v404 )
    _libc_free((unsigned __int64)v402);
  if ( (v399 & 1) == 0 )
LABEL_236:
    j___libc_free_0(v400);
  if ( v359 )
    goto LABEL_238;
LABEL_520:
  v189 = 0;
  sub_1D5C0A0(
    (__int64)&v405,
    *(__int64 ***)(a1 + 920),
    v19,
    a4,
    *(double *)a5.m128_u64,
    *(double *)a6.m128_u64,
    v16,
    v17,
    a9,
    a10);
LABEL_245:
  if ( v395 != v397 )
    _libc_free((unsigned __int64)v395);
  if ( v419 != v418 )
    _libc_free((unsigned __int64)v419);
  if ( v414 != (__int64 *)v416 )
    _libc_free((unsigned __int64)v414);
  if ( (v411 & 1) == 0 )
    j___libc_free_0(v412);
  j___libc_free_0(v406);
  return v189;
}
