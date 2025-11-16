// Function: sub_292A4F0
// Address: 0x292a4f0
//
char __fastcall sub_292A4F0(_BYTE **a1, _QWORD *a2)
{
  __int64 v2; // r12
  __int64 v4; // rbx
  _BYTE *v5; // rdx
  _BYTE *v6; // rax
  _BYTE *v7; // rdi
  _BYTE *v8; // rcx
  __int64 v9; // r14
  _QWORD *v10; // rax
  __int64 v11; // r13
  __int64 v12; // rsi
  __int64 v13; // r8
  unsigned __int64 v14; // rsi
  _QWORD *v15; // rax
  int v16; // ecx
  _QWORD *v17; // rdx
  __int64 v18; // rdi
  __int128 v19; // rax
  __m128i *v20; // rsi
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  _QWORD *v24; // rdi
  size_t v25; // rdx
  __int64 v26; // r12
  __int64 v27; // rax
  unsigned __int8 *v28; // rbx
  __int64 v29; // r14
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // rbx
  int v37; // eax
  __int64 v38; // rcx
  _QWORD *v39; // rax
  __int64 *v40; // rsi
  __int64 v41; // rdi
  _QWORD *v42; // rdx
  char result; // al
  __int64 v44; // rax
  unsigned int v45; // eax
  __int64 v46; // rax
  __int16 v47; // ax
  __int64 v48; // rsi
  __int64 v49; // rsi
  __int64 v50; // r8
  unsigned __int64 v51; // rsi
  _QWORD *v52; // rax
  int v53; // ecx
  _QWORD *v54; // rdx
  __int64 v55; // rdi
  __int64 v56; // rbx
  _BYTE *v57; // rax
  __int64 v58; // r14
  __int64 v59; // rax
  unsigned __int8 *v60; // rbx
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // rcx
  __int64 v64; // rdx
  __int64 v65; // rcx
  __int64 v66; // r8
  __int64 v67; // r9
  __int64 v68; // rdi
  __int64 v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // r9
  __int64 v73; // r12
  __int64 v74; // r9
  __int64 v75; // r14
  unsigned __int64 v76; // rsi
  _QWORD *v77; // rax
  int v78; // ecx
  _QWORD *v79; // rdx
  unsigned __int64 v80; // rax
  __int64 v81; // rsi
  __m128i v82; // rax
  _BYTE *v83; // rax
  __m128i v84; // xmm3
  _BYTE *v85; // rdi
  unsigned __int64 v86; // rcx
  unsigned __int64 v87; // rax
  __int64 v88; // rcx
  _BYTE *v89; // rdx
  __int16 v90; // cx
  __int64 v91; // r10
  unsigned __int64 v92; // rax
  _QWORD *v93; // rax
  __int64 v94; // rbx
  __int64 v95; // r9
  _BYTE *v96; // rax
  __int64 v97; // rdx
  _BYTE *v98; // r14
  _BYTE *v99; // r12
  __int64 v100; // rdx
  unsigned int v101; // esi
  __int16 v102; // cx
  unsigned __int64 v103; // rax
  _QWORD *v104; // rax
  __int64 v105; // r9
  __int64 v106; // rbx
  _BYTE *v107; // rax
  __int64 v108; // r14
  _BYTE *v109; // r14
  _BYTE *v110; // r12
  __int64 v111; // rdx
  unsigned int v112; // esi
  _BYTE *v113; // r14
  __int64 v114; // rdx
  unsigned __int64 v115; // rcx
  __m128i *v116; // r9
  unsigned __int64 v117; // rsi
  __int64 v118; // r8
  int v119; // eax
  unsigned __int64 *v120; // rdi
  __int64 v121; // rax
  __int64 v122; // rax
  __int64 v123; // rbx
  __m128i v124; // rax
  bool v125; // zf
  _BYTE *v126; // r8
  unsigned __int64 v127; // rcx
  unsigned __int64 v128; // rax
  _BYTE *v129; // rdx
  __int64 v130; // rsi
  __int64 v131; // rax
  __int64 v132; // rax
  __int64 v133; // r11
  __int64 v134; // rax
  __int64 v135; // rbx
  __int64 **v136; // rax
  _QWORD *v137; // rax
  unsigned __int64 v138; // r11
  __int64 v139; // rdx
  __int64 v140; // rdi
  __int64 v141; // r8
  unsigned __int64 v142; // rax
  __int64 v143; // r8
  __int64 v144; // r9
  _BYTE *v145; // r14
  __int64 v146; // rdx
  __m128i *v147; // rbx
  unsigned __int64 v148; // rcx
  unsigned __int64 v149; // rsi
  int v150; // eax
  unsigned __int64 *v151; // rdi
  __int64 v152; // rax
  __int64 v153; // rdx
  __int64 v154; // rcx
  __int64 v155; // r8
  __int64 v156; // r9
  unsigned __int64 v157; // rax
  __int64 v158; // rax
  unsigned __int64 v159; // rdx
  unsigned __int64 v160; // rax
  unsigned __int16 v161; // cx
  __int64 v162; // r14
  __int16 v163; // dx
  __int64 v164; // r8
  char v165; // al
  char v166; // dl
  __int16 v167; // cx
  _BYTE *v168; // rax
  __int64 v169; // r14
  _QWORD *v170; // rax
  __int64 v171; // rax
  _BYTE *v172; // rdi
  __int64 v173; // r8
  unsigned int v174; // ebx
  _QWORD *v175; // rax
  unsigned int v176; // esi
  __int64 v177; // r9
  __int64 v178; // r8
  int v179; // r11d
  __int64 v180; // r10
  unsigned int v181; // edx
  __int64 *v182; // rcx
  __int64 v183; // rdi
  int v184; // eax
  _QWORD *v185; // rax
  __int64 v186; // rax
  __int64 v187; // r12
  __int64 v188; // rax
  __int64 v189; // rax
  __int64 v190; // rdx
  unsigned __int64 v191; // rcx
  char v192; // si
  unsigned __int64 v193; // rax
  unsigned __int16 v194; // cx
  __int16 v195; // ax
  __int16 v196; // dx
  __int64 v197; // rdx
  __int64 v198; // rcx
  __int64 v199; // r8
  __int64 v200; // r9
  _BYTE *v201; // rax
  __int64 v202; // rbx
  __int64 v203; // rdx
  __int64 v204; // r8
  __int64 v205; // r9
  __int16 v206; // ax
  char v207; // dl
  __int64 v208; // rax
  const char *v209; // rax
  __int64 v210; // rdx
  _BYTE *v211; // rdx
  int v212; // ecx
  __int64 v213; // r11
  __int16 v214; // ax
  __int16 v215; // dx
  unsigned __int64 v216; // rax
  _BYTE *v217; // r9
  __int64 v218; // rax
  __m128i v219; // rax
  char v220; // si
  unsigned __int64 v221; // rcx
  unsigned __int64 v222; // rax
  int v223; // ecx
  int v224; // eax
  __int64 v225; // rax
  __int64 v226; // r11
  __int16 v227; // ax
  __int16 v228; // dx
  unsigned __int64 v229; // rax
  unsigned __int64 v230; // rax
  _BYTE *v231; // rax
  __int64 v232; // r14
  _BYTE *v233; // r14
  __int64 v234; // r14
  unsigned __int8 *v235; // rax
  unsigned __int64 v236; // rax
  __int64 v237; // rax
  unsigned __int64 v238; // rsi
  __int64 v239; // rax
  __int64 *v240; // r12
  __int64 *v241; // r14
  __int64 v242; // r8
  unsigned int v243; // eax
  _QWORD *v244; // rcx
  __int64 v245; // rdi
  unsigned int v246; // esi
  int v247; // eax
  __int64 *v248; // rax
  __int64 v249; // rcx
  _BYTE *v250; // rax
  __int64 v251; // rbx
  __int64 v252; // rdi
  __int8 *v253; // rbx
  __int64 v254; // rdx
  unsigned __int64 v255; // rax
  unsigned __int16 v256; // cx
  unsigned __int64 v257; // rax
  _QWORD *v258; // rax
  __int64 v259; // rax
  _BYTE *v260; // rdi
  __int64 v261; // r9
  __m128i v262; // xmm7
  __int64 v263; // rdi
  _BYTE *v264; // rax
  __int64 v265; // rdx
  _BYTE *v266; // rdx
  __int16 v267; // cx
  __int64 v268; // r11
  unsigned __int64 v269; // rax
  int v270; // ecx
  unsigned __int64 v271; // rax
  unsigned __int64 v272; // rax
  __int64 v273; // rdi
  __int64 v274; // r8
  __int64 v275; // rax
  _BYTE *v276; // r9
  __int16 v277; // cx
  _QWORD *v278; // rax
  __int64 v279; // rbx
  _BYTE *v280; // rax
  _BYTE *v281; // r13
  _BYTE *v282; // r12
  __int64 v283; // rdx
  unsigned int v284; // esi
  __int64 v285; // r8
  __int64 v286; // r9
  _BYTE *v287; // rbx
  __int64 v288; // rdx
  unsigned __int64 v289; // rcx
  unsigned __int64 v290; // rsi
  int v291; // eax
  __int64 v292; // rsi
  __m128i *v293; // rcx
  unsigned __int64 *v294; // rdi
  __int64 v295; // rax
  __int64 v296; // rdi
  unsigned __int64 v297; // rsi
  unsigned __int64 v298; // rsi
  int v299; // r11d
  _QWORD *v300; // r10
  int v301; // eax
  unsigned __int64 v302; // r12
  __int64 v303; // rdi
  __int64 v304; // [rsp+8h] [rbp-1B8h]
  __int64 v305; // [rsp+10h] [rbp-1B0h]
  __int16 v306; // [rsp+10h] [rbp-1B0h]
  char v307; // [rsp+18h] [rbp-1A8h]
  __int64 v308; // [rsp+18h] [rbp-1A8h]
  __int64 v309; // [rsp+18h] [rbp-1A8h]
  __int64 v310; // [rsp+18h] [rbp-1A8h]
  __int64 v311; // [rsp+18h] [rbp-1A8h]
  int v312; // [rsp+18h] [rbp-1A8h]
  __int64 v313; // [rsp+18h] [rbp-1A8h]
  __int64 v314; // [rsp+18h] [rbp-1A8h]
  char v315; // [rsp+20h] [rbp-1A0h]
  unsigned int v316; // [rsp+20h] [rbp-1A0h]
  __int64 v317; // [rsp+20h] [rbp-1A0h]
  unsigned __int64 v318; // [rsp+20h] [rbp-1A0h]
  int v319; // [rsp+20h] [rbp-1A0h]
  __int64 v320; // [rsp+20h] [rbp-1A0h]
  _BYTE *v321; // [rsp+20h] [rbp-1A0h]
  __int64 v322; // [rsp+20h] [rbp-1A0h]
  __int64 v323; // [rsp+20h] [rbp-1A0h]
  __int64 v324; // [rsp+20h] [rbp-1A0h]
  unsigned __int64 v325; // [rsp+20h] [rbp-1A0h]
  __int64 v326; // [rsp+20h] [rbp-1A0h]
  __int64 v327; // [rsp+28h] [rbp-198h]
  unsigned __int8 *v328; // [rsp+28h] [rbp-198h]
  __int64 v329; // [rsp+28h] [rbp-198h]
  __int64 v330; // [rsp+28h] [rbp-198h]
  int v331; // [rsp+28h] [rbp-198h]
  __int64 v332; // [rsp+28h] [rbp-198h]
  _QWORD *v333; // [rsp+28h] [rbp-198h]
  __int64 v334; // [rsp+28h] [rbp-198h]
  char v335; // [rsp+28h] [rbp-198h]
  __int64 v336; // [rsp+30h] [rbp-190h]
  unsigned __int64 v337; // [rsp+30h] [rbp-190h]
  __int64 v338; // [rsp+30h] [rbp-190h]
  unsigned __int64 v339; // [rsp+30h] [rbp-190h]
  unsigned __int8 *v340; // [rsp+30h] [rbp-190h]
  unsigned int v341; // [rsp+30h] [rbp-190h]
  __int64 v342; // [rsp+30h] [rbp-190h]
  __int8 *v343; // [rsp+30h] [rbp-190h]
  __int64 v344; // [rsp+30h] [rbp-190h]
  unsigned __int64 v345; // [rsp+38h] [rbp-188h]
  unsigned __int64 v346; // [rsp+38h] [rbp-188h]
  char v347; // [rsp+38h] [rbp-188h]
  unsigned int v348; // [rsp+38h] [rbp-188h]
  __int64 v349; // [rsp+38h] [rbp-188h]
  _QWORD *v350; // [rsp+40h] [rbp-180h] BYREF
  size_t n; // [rsp+48h] [rbp-178h]
  _QWORD src[6]; // [rsp+50h] [rbp-170h] BYREF
  char v353; // [rsp+80h] [rbp-140h]
  char v354; // [rsp+81h] [rbp-13Fh]
  __int128 v355; // [rsp+90h] [rbp-130h] BYREF
  char *v356; // [rsp+A0h] [rbp-120h]
  __int64 v357; // [rsp+A8h] [rbp-118h]
  __int64 v358; // [rsp+B0h] [rbp-110h]
  __m128i v359; // [rsp+C0h] [rbp-100h] BYREF
  __m128i v360; // [rsp+D0h] [rbp-F0h] BYREF
  __int16 v361; // [rsp+E0h] [rbp-E0h]
  __m128i v362; // [rsp+F0h] [rbp-D0h] BYREF
  unsigned __int128 v363; // [rsp+100h] [rbp-C0h]
  __int64 v364; // [rsp+110h] [rbp-B0h]
  __m128i v365; // [rsp+120h] [rbp-A0h] BYREF
  __m128i v366; // [rsp+130h] [rbp-90h]
  __int64 v367; // [rsp+140h] [rbp-80h]
  __m128i v368; // [rsp+150h] [rbp-70h] BYREF
  char *v369; // [rsp+160h] [rbp-60h]
  __int64 v370; // [rsp+168h] [rbp-58h]
  _BYTE *v371; // [rsp+170h] [rbp-50h]
  __int16 v372; // [rsp+178h] [rbp-48h]
  __int64 v373[8]; // [rsp+180h] [rbp-40h] BYREF

  v4 = (__int64)(a1 + 12);
  v5 = (_BYTE *)*a2;
  a1[12] = (_BYTE *)*a2;
  v6 = (_BYTE *)a2[1];
  a1[13] = v6;
  *((_BYTE *)a1 + 136) = (__int64)a2[2] >> 2;
  *((_BYTE *)a1 + 136) &= 1u;
  v7 = a1[5];
  v8 = a1[6];
  if ( v7 <= v5 )
  {
    *((_BYTE *)a1 + 137) = v6 > v8;
  }
  else
  {
    *((_BYTE *)a1 + 137) = 1;
    v5 = v7;
  }
  a1[14] = v5;
  v9 = (__int64)(a1 + 22);
  if ( v6 > v8 )
    v6 = v8;
  a1[15] = v6;
  a1[16] = (_BYTE *)(v6 - v5);
  v10 = (_QWORD *)(a2[2] & 0xFFFFFFFFFFFFFFF8LL);
  a1[18] = v10;
  a1[19] = (_BYTE *)*v10;
  v11 = v10[3];
  sub_D5F1F0((__int64)(a1 + 22), v11);
  v12 = *(_QWORD *)(v11 + 48);
  v368.m128i_i64[0] = v12;
  if ( !v12 || (sub_B96E90((__int64)&v368, v12, 1), (v13 = v368.m128i_i64[0]) == 0) )
  {
    sub_93FB40((__int64)(a1 + 22), 0);
    v13 = v368.m128i_i64[0];
    goto LABEL_155;
  }
  v14 = *((unsigned int *)a1 + 46);
  v15 = a1[22];
  v16 = *((_DWORD *)a1 + 46);
  v17 = &v15[2 * v14];
  if ( v15 == v17 )
  {
LABEL_157:
    v157 = *((unsigned int *)a1 + 47);
    if ( v14 >= v157 )
    {
      v238 = v14 + 1;
      if ( v157 < v238 )
      {
        v314 = v368.m128i_i64[0];
        sub_C8D5F0((__int64)(a1 + 22), a1 + 24, v238, 0x10u, v368.m128i_i64[0], (__int64)(a1 + 24));
        v13 = v314;
        v17 = &a1[22][16 * *((unsigned int *)a1 + 46)];
      }
      *v17 = 0;
      v17[1] = v13;
      v13 = v368.m128i_i64[0];
      ++*((_DWORD *)a1 + 46);
    }
    else
    {
      if ( v17 )
      {
        *(_DWORD *)v17 = 0;
        v17[1] = v13;
        v16 = *((_DWORD *)a1 + 46);
        v13 = v368.m128i_i64[0];
      }
      *((_DWORD *)a1 + 46) = v16 + 1;
    }
LABEL_155:
    if ( !v13 )
      goto LABEL_13;
    goto LABEL_12;
  }
  while ( *(_DWORD *)v15 )
  {
    v15 += 2;
    if ( v17 == v15 )
      goto LABEL_157;
  }
  v15[1] = v368.m128i_i64[0];
LABEL_12:
  sub_B91220((__int64)&v368, v13);
LABEL_13:
  v361 = 267;
  v18 = (__int64)a1[4];
  v365.m128i_i64[0] = (__int64)".";
  LOWORD(v367) = 259;
  v359.m128i_i64[0] = v4;
  v354 = 1;
  src[2] = ".";
  v353 = 3;
  *(_QWORD *)&v19 = sub_BD5D20(v18);
  v355 = v19;
  v356 = ".";
  v357 = src[3];
  LOWORD(v358) = 773;
  v362.m128i_i64[0] = (__int64)&v355;
  v362.m128i_i64[1] = v2;
  v363 = __PAIR128__(v359.m128i_u64[1], v4);
  LOWORD(v364) = 2818;
  v368.m128i_i64[0] = (__int64)&v362;
  v369 = ".";
  v368.m128i_i64[1] = v327;
  LOWORD(v371) = 770;
  v370 = v365.m128i_i64[1];
  v20 = &v368;
  sub_CA0F50((__int64 *)&v350, (void **)&v368);
  v24 = a1[40];
  if ( v350 == src )
  {
    v25 = n;
    if ( n )
    {
      if ( n == 1 )
      {
        *(_BYTE *)v24 = src[0];
      }
      else
      {
        v20 = (__m128i *)src;
        memcpy(v24, src, n);
      }
      v25 = n;
      v24 = a1[40];
    }
    a1[41] = (_BYTE *)v25;
    *((_BYTE *)v24 + v25) = 0;
    v24 = v350;
    goto LABEL_19;
  }
  v25 = (size_t)(a1 + 42);
  v20 = (__m128i *)n;
  v21 = src[0];
  if ( v24 == a1 + 42 )
  {
    a1[40] = v350;
    a1[41] = v20;
    a1[42] = (_BYTE *)v21;
  }
  else
  {
    v25 = (size_t)a1[42];
    a1[40] = v350;
    a1[41] = v20;
    a1[42] = (_BYTE *)v21;
    if ( v24 )
    {
      v350 = v24;
      src[0] = v25;
      goto LABEL_19;
    }
  }
  v350 = src;
  v24 = src;
LABEL_19:
  n = 0;
  *(_BYTE *)v24 = 0;
  if ( v350 != src )
  {
    v20 = (__m128i *)(src[0] + 1LL);
    j_j___libc_free_0((unsigned __int64)v350);
  }
  v26 = *((_QWORD *)a1[18] + 3);
  switch ( *(_BYTE *)v26 )
  {
    case '"':
    case '(':
      goto LABEL_400;
    case '=':
      v340 = *(unsigned __int8 **)(v26 - 32);
      sub_B91FC0(v365.m128i_i64, *((_QWORD *)a1[18] + 3));
      v122 = *(_QWORD *)(*(_QWORD *)(v26 - 32) + 8LL);
      if ( (unsigned int)*(unsigned __int8 *)(v122 + 8) - 17 <= 1 )
        v122 = **(_QWORD **)(v122 + 16);
      v331 = *(_DWORD *)(v122 + 8) >> 8;
      if ( *((_BYTE *)a1 + 137) )
      {
        v174 = 8 * *((_DWORD *)a1 + 32);
        v175 = (_QWORD *)sub_BD5C60(v26);
        v123 = sub_BCD140(v175, v174);
      }
      else
      {
        v123 = *(_QWORD *)(v26 + 8);
      }
      v124.m128i_i64[0] = sub_9208B0((__int64)*a1, v123);
      v125 = a1[9] == 0;
      v126 = a1[16];
      v368 = v124;
      if ( !v125 )
      {
        v127 = (unsigned __int64)a1[11];
        v346 = (a1[14] - a1[5]) / v127;
        v128 = (a1[15] - a1[5]) / v127;
        v129 = a1[4];
        LOWORD(v127) = *((_WORD *)v129 + 1);
        v130 = *((_QWORD *)v129 + 9);
        LOWORD(v371) = 259;
        v316 = v128;
        _BitScanReverse64(&v128, 1LL << v127);
        LODWORD(v127) = (unsigned __int8)(63 - (v128 ^ 0x3F));
        v368.m128i_i64[0] = (__int64)"load";
        BYTE1(v127) = 1;
        v131 = sub_A82CA0((unsigned int **)a1 + 22, v130, (int)v129, v127, 0, (__int64)&v368);
        v368.m128i_i64[0] = 0x190000000ALL;
        v310 = v131;
        sub_B47C00(v131, v26, v368.m128i_i32, 2);
        LOWORD(v371) = 259;
        v368.m128i_i64[0] = (__int64)"vec";
        v132 = sub_2918880((__int64 *)a1 + 22, v310, v346, v316, &v368, v310);
        v347 = 0;
        v133 = v132;
        goto LABEL_135;
      }
      if ( a1[8] && *(_BYTE *)(*(_QWORD *)(v26 + 8) + 8LL) == 12 )
      {
        v254 = (__int64)a1[4];
        _BitScanReverse64(&v255, 1LL << *(_WORD *)(v254 + 2));
        LOBYTE(v256) = 63 - (v255 ^ 0x3F);
        HIBYTE(v256) = 1;
        v257 = sub_291B4B0((__int64 *)a1 + 22, *(_QWORD *)(v254 + 72), v254, v256, "load");
        v133 = sub_291C8F0((__int64)*a1, (unsigned int **)a1 + 22, v257, (__int64)a1[8]);
        if ( a1[14] != a1[5] || a1[15] < a1[6] )
        {
          v313 = a1[14] - a1[5];
          v325 = v133;
          v348 = 8 * *((_DWORD *)a1 + 32);
          v258 = (_QWORD *)sub_BD5C60(v26);
          v259 = sub_BCD140(v258, v348);
          v260 = *a1;
          LOWORD(v371) = 259;
          v368.m128i_i64[0] = (__int64)"extract";
          v133 = sub_291AEB0(v260, (__int64)(a1 + 22), v325, v259, v313, &v368);
        }
        v261 = *(_QWORD *)(v26 + 8);
        v347 = 0;
        if ( *(_DWORD *)(v261 + 8) >> 8 > (unsigned __int64)(8LL * (_QWORD)a1[16]) )
        {
          LOWORD(v371) = 257;
          v133 = sub_A82F30((unsigned int **)a1 + 22, v133, v261, (__int64)&v368, 0);
        }
      }
      else
      {
        if ( a1[14] == a1[5] && a1[15] == a1[6] )
        {
          v311 = v124.m128i_i64[0];
          v318 = (unsigned __int64)v126;
          if ( sub_29191E0((__int64)*a1, (__int64)a1[7], v123) )
          {
            v206 = *(_WORD *)(v26 + 2);
LABEL_229:
            v207 = v206 & 1;
            v208 = *(_QWORD *)(*(_QWORD *)(v26 - 32) + 8LL);
            if ( (unsigned int)*(unsigned __int8 *)(v208 + 8) - 17 <= 1 )
              v208 = **(_QWORD **)(v208 + 16);
            v319 = sub_291AE20((__int64)a1, *(_DWORD *)(v208 + 8) >> 8, v207);
            v209 = sub_BD5D20(v26);
            v368.m128i_i64[1] = v210;
            v211 = a1[4];
            v368.m128i_i64[0] = (__int64)v209;
            LOWORD(v371) = 261;
            _BitScanReverse64((unsigned __int64 *)&v209, 1LL << *((_WORD *)v211 + 1));
            v212 = (unsigned __int8)(63 - ((unsigned __int8)v209 ^ 0x3F));
            BYTE1(v212) = 1;
            v213 = sub_A82CA0(
                     (unsigned int **)a1 + 22,
                     *((_QWORD *)v211 + 9),
                     v319,
                     v212,
                     *(_BYTE *)(v26 + 2) & 1,
                     (__int64)&v368);
            v214 = *(_WORD *)(v26 + 2);
            if ( (v214 & 1) != 0 )
            {
              v215 = *(_WORD *)(v213 + 2) & 0xFC7F;
              *(_BYTE *)(v213 + 72) = *(_BYTE *)(v26 + 72);
              *(_WORD *)(v213 + 2) = v215 | v214 & 0x380;
            }
            v320 = v213;
            if ( sub_B46500((unsigned __int8 *)v213) )
            {
              _BitScanReverse64(&v216, 1LL << (*(_WORD *)(v26 + 2) >> 1));
              *(_WORD *)(v320 + 2) = *(_WORD *)(v320 + 2) & 0xFF81 | (2 * (63 - (v216 ^ 0x3F)));
            }
            sub_F57A40(v320, v26);
            v133 = v320;
            if ( v365.m128i_i64[0] || __PAIR128__(v365.m128i_u64[1], 0) != v366.m128i_u64[0] || v366.m128i_i64[1] )
            {
              sub_E00EB0(&v368, v365.m128i_i64, a1[14] - a1[12], *(_QWORD *)(v320 + 8), (__int64)*a1);
              sub_B9A100(v320, v368.m128i_i64);
              v133 = v320;
            }
            v217 = a1[7];
            v347 = 0;
            if ( v217[8] == 12 && *(_BYTE *)(v123 + 8) == 12 && *((_DWORD *)v217 + 2) >> 8 < *(_DWORD *)(v123 + 8) >> 8 )
            {
              v321 = a1[7];
              v368.m128i_i64[0] = (__int64)"load.ext";
              LOWORD(v371) = 259;
              v218 = sub_A82F30((unsigned int **)a1 + 22, v133, v123, (__int64)&v368, 0);
              v133 = v218;
              v347 = **a1;
              if ( v347 )
              {
                v368.m128i_i64[0] = (__int64)"endian_shift";
                LOWORD(v371) = 259;
                v347 = 0;
                v133 = sub_920C00(
                         (unsigned int **)a1 + 22,
                         v218,
                         (unsigned int)((*(_DWORD *)(v123 + 8) >> 8) - (*((_DWORD *)v321 + 2) >> 8)),
                         (__int64)&v368,
                         0,
                         0);
              }
            }
            goto LABEL_135;
          }
          if ( v318 < (unsigned __int64)(v311 + 7) >> 3 && a1[7][8] == 12 && *(_BYTE *)(v123 + 8) == 12 )
          {
            v206 = *(_WORD *)(v26 + 2);
            if ( (v206 & 1) == 0 )
              goto LABEL_229;
          }
        }
        v322 = sub_BCE3C0((__int64 *)a1[31], v331);
        v219.m128i_i64[0] = (__int64)sub_BD5D20(v26);
        v220 = -1;
        v368 = v219;
        v219.m128i_i64[0] = (__int64)a1[4];
        LOWORD(v371) = 261;
        _BitScanReverse64(&v221, 1LL << *(_WORD *)(v219.m128i_i64[0] + 2));
        v222 = -(__int64)((0x8000000000000000LL >> ((unsigned __int8)v221 ^ 0x3Fu)) | (a1[14] - a1[5]))
             & ((0x8000000000000000LL >> ((unsigned __int8)v221 ^ 0x3Fu)) | (a1[14] - a1[5]));
        if ( v222 )
        {
          _BitScanReverse64(&v222, v222);
          v220 = 63 - (v222 ^ 0x3F);
        }
        v223 = 256;
        v306 = *(_WORD *)(v26 + 2) & 1;
        LOBYTE(v223) = v220;
        v312 = v223;
        v224 = sub_291C360((__int64 *)a1, (__int64)(a1 + 22), v322);
        v225 = sub_A82CA0((unsigned int **)a1 + 22, v123, v224, v312, v306, (__int64)&v368);
        v226 = v225;
        if ( v365.m128i_i64[0] || __PAIR128__(v365.m128i_u64[1], 0) != v366.m128i_u64[0] || v366.m128i_i64[1] )
        {
          v323 = v225;
          sub_E00EB0(&v368, v365.m128i_i64, a1[14] - a1[12], *(_QWORD *)(v225 + 8), (__int64)*a1);
          sub_B9A100(v323, v368.m128i_i64);
          v226 = v323;
        }
        v227 = *(_WORD *)(v26 + 2);
        if ( (v227 & 1) != 0 )
        {
          v228 = *(_WORD *)(v226 + 2) & 0xFC7F;
          *(_BYTE *)(v226 + 72) = *(_BYTE *)(v26 + 72);
          *(_WORD *)(v226 + 2) = v228 | v227 & 0x380;
        }
        v324 = v226;
        v368.m128i_i64[0] = 0x190000000ALL;
        sub_B47C00(v226, v26, v368.m128i_i32, 2);
        v347 = 1;
        v133 = v324;
      }
LABEL_135:
      v134 = sub_291C8F0((__int64)*a1, (unsigned int **)a1 + 22, v133, v123);
      v135 = v134;
      if ( *((_BYTE *)a1 + 137) )
      {
        sub_A88F30((__int64)(a1 + 22), *(_QWORD *)(v26 + 40), *(_QWORD *)(v26 + 32), 1);
        v317 = *(_QWORD *)(v26 + 8);
        v136 = (__int64 **)sub_BCE3C0((__int64 *)a1[31], v331);
        v332 = sub_ACADE0(v136);
        LOWORD(v371) = 257;
        v137 = sub_BD2C40(80, 1u);
        v138 = (unsigned __int64)v137;
        if ( v137 )
        {
          v139 = v332;
          v333 = v137;
          sub_B4D190((__int64)v137, v317, v139, (__int64)&v368, 0, 0, 0, 0);
          v138 = (unsigned __int64)v333;
        }
        v140 = (__int64)*a1;
        v141 = a1[14] - a1[12];
        v334 = v138;
        v368.m128i_i64[0] = (__int64)"insert";
        LOWORD(v371) = 259;
        v142 = sub_291CC20(v140, (__int64)(a1 + 22), v138, v135, v141, &v368);
        sub_BD84D0(v26, v142);
        sub_BD84D0(v334, v26);
        sub_BD72D0(v334, v26);
      }
      else
      {
        sub_BD84D0(v26, v134);
      }
      v368 = (__m128i)4uLL;
      v145 = a1[2];
      v369 = (char *)v26;
      if ( v26 != -4096 && v26 != -8192 )
        sub_BD73F0((__int64)&v368);
      v146 = *((unsigned int *)v145 + 56);
      v147 = &v368;
      v148 = *((_QWORD *)v145 + 27);
      v149 = v146 + 1;
      v150 = *((_DWORD *)v145 + 56);
      if ( v146 + 1 > (unsigned __int64)*((unsigned int *)v145 + 57) )
      {
        v252 = (__int64)(v145 + 216);
        if ( v148 > (unsigned __int64)&v368 || (unsigned __int64)&v368 >= v148 + 24 * v146 )
        {
          sub_D6B130(v252, v149, v146, v148, v143, v144);
          v146 = *((unsigned int *)v145 + 56);
          v147 = &v368;
          v148 = *((_QWORD *)v145 + 27);
          v150 = *((_DWORD *)v145 + 56);
        }
        else
        {
          v253 = &v368.m128i_i8[-v148];
          sub_D6B130(v252, v149, v146, v148, v143, v144);
          v148 = *((_QWORD *)v145 + 27);
          v146 = *((unsigned int *)v145 + 56);
          v147 = (__m128i *)&v253[v148];
          v150 = *((_DWORD *)v145 + 56);
        }
      }
      v151 = (unsigned __int64 *)(v148 + 24 * v146);
      if ( v151 )
      {
        *v151 = 4;
        v152 = v147[1].m128i_i64[0];
        v151[1] = 0;
        v151[2] = v152;
        if ( v152 != 0 && v152 != -4096 && v152 != -8192 )
          sub_BD6050(v151, v147->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL);
        v150 = *((_DWORD *)v145 + 56);
      }
      *((_DWORD *)v145 + 56) = v150 + 1;
      if ( v369 + 4096 != 0 && v369 != 0 && v369 != (char *)-8192LL )
        sub_BD60C0(&v368);
      if ( sub_F50EE0(v340, 0) )
      {
        v233 = a1[2];
        v368 = (__m128i)4uLL;
        v234 = (__int64)(v233 + 216);
        v369 = (char *)v340;
        LOBYTE(v153) = v340 != 0;
        if ( v340 + 4096 != 0 && v340 != 0 && v340 != (unsigned __int8 *)-8192LL )
          sub_BD73F0((__int64)&v368);
        sub_D6B260(v234, v368.m128i_i8, v153, v154, v155, v156);
        if ( v369 != 0 && v369 + 4096 != 0 && v369 != (char *)-8192LL )
          sub_BD60C0(&v368);
      }
      return (v347 | *(_WORD *)(v26 + 2) & 1) ^ 1;
    case '>':
      v328 = *(unsigned __int8 **)(v26 - 32);
      sub_B91FC0(v359.m128i_i64, *((_QWORD *)a1[18] + 3));
      v80 = *(_QWORD *)(v26 - 64);
      v81 = *(_QWORD *)(v80 + 8);
      v345 = v80;
      if ( *(_BYTE *)(v81 + 8) == 14 )
      {
        v235 = sub_BD4CB0(
                 (unsigned __int8 *)v80,
                 (void (__fastcall *)(__int64, unsigned __int8 *))nullsub_96,
                 (__int64)&v365);
        if ( *v235 == 60 )
        {
          v368.m128i_i64[0] = (__int64)v235;
          sub_2928360((__int64)(a1[2] + 424), v368.m128i_i64);
        }
        v81 = *(_QWORD *)(v345 + 8);
      }
      v337 = (unsigned __int64)a1[16];
      v82.m128i_i64[0] = sub_9208B0((__int64)*a1, v81);
      v368 = v82;
      if ( v337 < (unsigned __int64)(v82.m128i_i64[0] + 7) >> 3 )
      {
        v341 = 8 * *((_DWORD *)a1 + 32);
        v170 = (_QWORD *)sub_BD5C60(v26);
        v171 = sub_BCD140(v170, v341);
        v172 = *a1;
        v173 = a1[14] - a1[12];
        LOWORD(v371) = 259;
        v368.m128i_i64[0] = (__int64)"extract";
        v345 = sub_291AEB0(v172, (__int64)(a1 + 22), v345, v171, v173, &v368);
      }
      v83 = a1[9];
      if ( v83 )
      {
        v84 = _mm_load_si128(&v360);
        v362 = _mm_load_si128(&v359);
        v363 = (unsigned __int128)v84;
        v85 = *(_BYTE **)(v345 + 8);
        v338 = v345;
        if ( v83 != v85 )
        {
          v86 = (unsigned __int64)a1[11];
          v339 = (a1[14] - a1[5]) / v86;
          v87 = (a1[15] - a1[5]) / v86;
          v88 = (__int64)a1[10];
          if ( (_DWORD)v87 - (_DWORD)v339 != 1 )
          {
            v237 = sub_BCDA70((__int64 *)a1[10], (int)v87 - (int)v339);
            v85 = *(_BYTE **)(v345 + 8);
            v88 = v237;
          }
          v329 = v345;
          if ( v85 != (_BYTE *)v88 )
            v329 = sub_291C8F0((__int64)*a1, (unsigned int **)a1 + 22, v345, v88);
          v89 = a1[4];
          v90 = *((_WORD *)v89 + 1);
          v91 = *((_QWORD *)v89 + 9);
          v304 = (__int64)v89;
          v365.m128i_i64[0] = (__int64)"load";
          v305 = v91;
          _BitScanReverse64(&v92, 1LL << v90);
          LOWORD(v367) = 259;
          v307 = 63 - (v92 ^ 0x3F);
          LOWORD(v371) = 257;
          v93 = sub_BD2C40(80, 1u);
          v94 = (__int64)v93;
          if ( v93 )
            sub_B4D190((__int64)v93, v305, v304, (__int64)&v368, 0, v307, 0, 0);
          (*(void (__fastcall **)(_BYTE *, __int64, __m128i *, _BYTE *, _BYTE *))(*(_QWORD *)a1[33] + 16LL))(
            a1[33],
            v94,
            &v365,
            a1[29],
            a1[30]);
          v96 = a1[22];
          v97 = 16LL * *((unsigned int *)a1 + 46);
          if ( v96 != &v96[v97] )
          {
            v98 = &v96[v97];
            v308 = v26;
            v99 = a1[22];
            do
            {
              v100 = *((_QWORD *)v99 + 1);
              v101 = *(_DWORD *)v99;
              v99 += 16;
              sub_B99FD0(v94, v101, v100);
            }
            while ( v98 != v99 );
            v9 = (__int64)(a1 + 22);
            v26 = v308;
          }
          v368.m128i_i64[0] = (__int64)"vec";
          LOWORD(v371) = 259;
          v338 = sub_2918170(v9, v94, v329, (unsigned int)v339, &v368, v95);
        }
        v102 = *((_WORD *)a1[4] + 1);
        v309 = (__int64)a1[4];
        LOWORD(v371) = 257;
        _BitScanReverse64(&v103, 1LL << v102);
        v315 = 63 - (v103 ^ 0x3F);
        v104 = sub_BD2C40(80, unk_3F10A10);
        v106 = (__int64)v104;
        if ( v104 )
          sub_B4D3C0((__int64)v104, v338, v309, 0, v315, v105, 0, 0);
        (*(void (__fastcall **)(_BYTE *, __int64, __m128i *, _QWORD, _QWORD))(*(_QWORD *)a1[33] + 16LL))(
          a1[33],
          v106,
          &v368,
          *(_QWORD *)(v9 + 56),
          *(_QWORD *)(v9 + 64));
        v107 = a1[22];
        v108 = 16LL * *((unsigned int *)a1 + 46);
        if ( v107 != &v107[v108] )
        {
          v330 = v26;
          v109 = &v107[v108];
          v110 = a1[22];
          do
          {
            v111 = *((_QWORD *)v110 + 1);
            v112 = *(_DWORD *)v110;
            v110 += 16;
            sub_B99FD0(v106, v112, v111);
          }
          while ( v109 != v110 );
          v26 = v330;
        }
        v368.m128i_i64[0] = 0x190000000ALL;
        sub_B47C00(v106, v26, v368.m128i_i32, 2);
        if ( v362.m128i_i64[0] || __PAIR128__(v362.m128i_u64[1], 0) != (unsigned __int64)v363 || *((_QWORD *)&v363 + 1) )
        {
          sub_E00EB0(&v368, v362.m128i_i64, a1[14] - a1[12], *(_QWORD *)(v338 + 8), (__int64)*a1);
          sub_B9A100(v106, v368.m128i_i64);
        }
        v368 = (__m128i)4uLL;
        v113 = a1[2];
        v369 = (char *)v26;
        if ( v26 != -8192 && v26 != -4096 )
          sub_BD73F0((__int64)&v368);
        v114 = *((unsigned int *)v113 + 56);
        v115 = *((unsigned int *)v113 + 57);
        v116 = &v368;
        v117 = *((_QWORD *)v113 + 27);
        v118 = v114 + 1;
        v119 = *((_DWORD *)v113 + 56);
        if ( v114 + 1 > v115 )
        {
          v296 = (__int64)(v113 + 216);
          if ( v117 > (unsigned __int64)&v368 || (unsigned __int64)&v368 >= v117 + 24 * v114 )
          {
            sub_D6B130(v296, v114 + 1, v114, v115, v118, (__int64)&v368);
            v114 = *((unsigned int *)v113 + 56);
            v117 = *((_QWORD *)v113 + 27);
            v116 = &v368;
            v119 = *((_DWORD *)v113 + 56);
          }
          else
          {
            v343 = &v368.m128i_i8[-v117];
            sub_D6B130(v296, v114 + 1, v114, v115, v118, (__int64)&v368);
            v117 = *((_QWORD *)v113 + 27);
            v114 = *((unsigned int *)v113 + 56);
            v116 = (__m128i *)&v343[v117];
            v119 = *((_DWORD *)v113 + 56);
          }
        }
        v120 = (unsigned __int64 *)(v117 + 24 * v114);
        if ( v120 )
        {
          *v120 = 4;
          v121 = v116[1].m128i_i64[0];
          v120[1] = 0;
          v120[2] = v121;
          if ( v121 != 0 && v121 != -4096 && v121 != -8192 )
            sub_BD6050(v120, v116->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL);
          v119 = *((_DWORD *)v113 + 56);
        }
        *((_DWORD *)v113 + 56) = v119 + 1;
        if ( v369 != 0 && v369 + 4096 != 0 && v369 != (char *)-8192LL )
          sub_BD60C0(&v368);
        sub_29228E0(
          (__int64)a1[3],
          *((_BYTE *)a1 + 137),
          8LL * (_QWORD)a1[14],
          8LL * (_QWORD)a1[16],
          v26,
          v106,
          *(_QWORD *)(v106 - 32),
          v345);
        return 1;
      }
      if ( !a1[8] || *(_BYTE *)(*(_QWORD *)(v345 + 8) + 8LL) != 12 )
      {
        if ( a1[14] == a1[5] && a1[15] == a1[6] && sub_29191E0((__int64)*a1, *(_QWORD *)(v345 + 8), (__int64)a1[7]) )
        {
          v345 = sub_291C8F0((__int64)*a1, (unsigned int **)a1 + 22, v345, (__int64)a1[7]);
          v158 = *(_QWORD *)(*(_QWORD *)(v26 - 32) + 8LL);
          if ( (unsigned int)*(unsigned __int8 *)(v158 + 8) - 17 <= 1 )
            v158 = **(_QWORD **)(v158 + 16);
          v159 = sub_291AE20((__int64)a1, *(_DWORD *)(v158 + 8) >> 8, *(_BYTE *)(v26 + 2) & 1);
          _BitScanReverse64(&v160, 1LL << *((_WORD *)a1[4] + 1));
          LOBYTE(v161) = 63 - (v160 ^ 0x3F);
          HIBYTE(v161) = 1;
          v162 = sub_2463EC0((__int64 *)a1 + 22, v345, v159, v161, *(_BYTE *)(v26 + 2) & 1);
        }
        else
        {
          v188 = *(_QWORD *)(*(_QWORD *)(v26 - 32) + 8LL);
          if ( (unsigned int)*(unsigned __int8 *)(v188 + 8) - 17 <= 1 )
            v188 = **(_QWORD **)(v188 + 16);
          v189 = sub_BCE3C0((__int64 *)a1[31], *(_DWORD *)(v188 + 8) >> 8);
          v190 = sub_291C360((__int64 *)a1, (__int64)(a1 + 22), v189);
          _BitScanReverse64(&v191, 1LL << *((_WORD *)a1[4] + 1));
          v192 = -1;
          v193 = -(__int64)((0x8000000000000000LL >> ((unsigned __int8)v191 ^ 0x3Fu)) | (a1[14] - a1[5]))
               & ((0x8000000000000000LL >> ((unsigned __int8)v191 ^ 0x3Fu)) | (a1[14] - a1[5]));
          if ( v193 )
          {
            _BitScanReverse64(&v193, v193);
            v192 = 63 - (v193 ^ 0x3F);
          }
          LOBYTE(v194) = v192;
          HIBYTE(v194) = 1;
          v162 = sub_2463EC0((__int64 *)a1 + 22, v345, v190, v194, *(_BYTE *)(v26 + 2) & 1);
        }
        v368.m128i_i64[0] = 0x190000000ALL;
        sub_B47C00(v162, v26, v368.m128i_i32, 2);
        if ( v359.m128i_i64[0] || __PAIR128__(v359.m128i_u64[1], 0) != v360.m128i_u64[0] || v360.m128i_i64[1] )
        {
          sub_E00EB0(&v368, v359.m128i_i64, a1[14] - a1[12], *(_QWORD *)(v345 + 8), (__int64)*a1);
          sub_B9A100(v162, v368.m128i_i64);
        }
        v195 = *(_WORD *)(v26 + 2);
        if ( (v195 & 1) != 0 )
        {
          v196 = *(_WORD *)(v162 + 2) & 0xFC7F;
          *(_BYTE *)(v162 + 72) = *(_BYTE *)(v26 + 72);
          *(_WORD *)(v162 + 2) = v196 | v195 & 0x380;
        }
        if ( sub_B46500((unsigned __int8 *)v162) )
        {
          _BitScanReverse64(&v236, 1LL << (*(_WORD *)(v26 + 2) >> 1));
          *(_WORD *)(v162 + 2) = *(_WORD *)(v162 + 2) & 0xFF81 | (2 * (63 - (v236 ^ 0x3F)));
        }
        sub_29228E0(
          (__int64)a1[3],
          *((_BYTE *)a1 + 137),
          8LL * (_QWORD)a1[14],
          8LL * (_QWORD)a1[16],
          v26,
          v162,
          *(_QWORD *)(v162 - 32),
          *(_QWORD *)(v162 - 64));
        v201 = a1[2];
        v369 = (char *)v26;
        v368 = (__m128i)4uLL;
        v202 = (__int64)(v201 + 216);
        if ( v26 != -8192 && v26 != -4096 )
          sub_BD73F0((__int64)&v368);
        sub_D6B260(v202, v368.m128i_i8, v197, v198, v199, v200);
        if ( v369 + 4096 != 0 && v369 != 0 && v369 != (char *)-8192LL )
          sub_BD60C0(&v368);
        if ( sub_F50EE0(v328, 0) )
        {
          v249 = (__int64)v328;
          v250 = a1[2];
          v368 = (__m128i)4uLL;
          v251 = (__int64)(v250 + 216);
          v369 = (char *)v328;
          LOBYTE(v203) = v328 + 4096 != 0;
          if ( ((v328 != 0) & (unsigned __int8)v203) != 0 && v328 != (unsigned __int8 *)-8192LL )
            sub_BD73F0((__int64)&v368);
          sub_D6B260(v251, v368.m128i_i8, v203, v249, v204, v205);
          if ( v369 != 0 && v369 + 4096 != 0 && v369 != (char *)-8192LL )
            sub_BD60C0(&v368);
        }
        result = 0;
        if ( *(_BYTE **)(v162 - 32) == a1[4] && *(_BYTE **)(*(_QWORD *)(v162 - 64) + 8LL) == a1[7] )
          return !(*(_WORD *)(v26 + 2) & 1);
        return result;
      }
      v262 = _mm_load_si128(&v360);
      v263 = (__int64)*a1;
      v365 = _mm_load_si128(&v359);
      v366 = v262;
      v368.m128i_i64[0] = sub_9208B0(v263, *(_QWORD *)(v345 + 8));
      v264 = a1[8];
      v368.m128i_i64[1] = v265;
      if ( v368.m128i_i64[0] != *((_DWORD *)v264 + 2) >> 8 )
      {
        v266 = a1[4];
        v267 = *((_WORD *)v266 + 1);
        v268 = *((_QWORD *)v266 + 9);
        LOWORD(v371) = 259;
        _BitScanReverse64(&v269, 1LL << v267);
        v270 = (unsigned __int8)(63 - (v269 ^ 0x3F));
        v368.m128i_i64[0] = (__int64)"oldload";
        BYTE1(v270) = 1;
        v271 = sub_A82CA0((unsigned int **)a1 + 22, v268, (int)v266, v270, 0, (__int64)&v368);
        v272 = sub_291C8F0((__int64)*a1, (unsigned int **)a1 + 22, v271, (__int64)a1[8]);
        v273 = (__int64)*a1;
        LOWORD(v371) = 259;
        v274 = a1[12] - a1[5];
        v368.m128i_i64[0] = (__int64)"insert";
        v345 = sub_291CC20(v273, (__int64)(a1 + 22), v272, *(_QWORD *)(v26 - 64), v274, &v368);
      }
      v275 = sub_291C8F0((__int64)*a1, (unsigned int **)a1 + 22, v345, (__int64)a1[7]);
      v276 = a1[4];
      v349 = v275;
      v277 = *((_WORD *)v276 + 1);
      LOWORD(v371) = 257;
      v326 = (__int64)v276;
      _BitScanReverse64((unsigned __int64 *)&v275, 1LL << v277);
      v335 = 63 - (v275 ^ 0x3F);
      v278 = sub_BD2C40(80, unk_3F10A10);
      v279 = (__int64)v278;
      if ( v278 )
        sub_B4D3C0((__int64)v278, v349, v326, 0, v335, v326, 0, 0);
      (*(void (__fastcall **)(_BYTE *, __int64, __m128i *, _BYTE *, _BYTE *))(*(_QWORD *)a1[33] + 16LL))(
        a1[33],
        v279,
        &v368,
        a1[29],
        a1[30]);
      v280 = a1[22];
      if ( v280 != &v280[16 * *((unsigned int *)a1 + 46)] )
      {
        v342 = v26;
        v281 = a1[22];
        v282 = &v280[16 * *((unsigned int *)a1 + 46)];
        do
        {
          v283 = *((_QWORD *)v281 + 1);
          v284 = *(_DWORD *)v281;
          v281 += 16;
          sub_B99FD0(v279, v284, v283);
        }
        while ( v282 != v281 );
        v26 = v342;
      }
      v368.m128i_i64[0] = 0x190000000ALL;
      sub_B47C00(v279, v26, v368.m128i_i32, 2);
      if ( v365.m128i_i64[0] || __PAIR128__(v365.m128i_u64[1], 0) != v366.m128i_u64[0] || v366.m128i_i64[1] )
      {
        sub_E00EB0(&v368, v365.m128i_i64, a1[14] - a1[12], *(_QWORD *)(v349 + 8), (__int64)*a1);
        sub_B9A100(v279, v368.m128i_i64);
      }
      sub_29228E0(
        (__int64)a1[3],
        *((_BYTE *)a1 + 137),
        8LL * (_QWORD)a1[14],
        8LL * (_QWORD)a1[16],
        v26,
        v279,
        *(_QWORD *)(v279 - 32),
        *(_QWORD *)(v279 - 64));
      v287 = a1[2];
      v369 = (char *)v26;
      v368 = (__m128i)4uLL;
      if ( v26 != -8192 && v26 != -4096 )
        sub_BD73F0((__int64)&v368);
      v288 = *((unsigned int *)v287 + 56);
      v289 = *((unsigned int *)v287 + 57);
      v290 = v288 + 1;
      v291 = *((_DWORD *)v287 + 56);
      if ( v288 + 1 > v289 )
      {
        v302 = *((_QWORD *)v287 + 27);
        v303 = (__int64)(v287 + 216);
        if ( v302 > (unsigned __int64)&v368 || (unsigned __int64)&v368 >= v302 + 24 * v288 )
        {
          sub_D6B130(v303, v290, v288, v289, v285, v286);
          v288 = *((unsigned int *)v287 + 56);
          v292 = *((_QWORD *)v287 + 27);
          v293 = &v368;
          v291 = *((_DWORD *)v287 + 56);
        }
        else
        {
          sub_D6B130(v303, v290, v288, v289, v285, v286);
          v292 = *((_QWORD *)v287 + 27);
          v288 = *((unsigned int *)v287 + 56);
          v293 = (__m128i *)((char *)&v368 + v292 - v302);
          v291 = *((_DWORD *)v287 + 56);
        }
      }
      else
      {
        v292 = *((_QWORD *)v287 + 27);
        v293 = &v368;
      }
      v294 = (unsigned __int64 *)(v292 + 24 * v288);
      if ( v294 )
      {
        *v294 = 4;
        v295 = v293[1].m128i_i64[0];
        v294[1] = 0;
        v294[2] = v295;
        if ( v295 != -4096 && v295 != 0 && v295 != -8192 )
          sub_BD6050(v294, v293->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL);
        v291 = *((_DWORD *)v287 + 56);
      }
      *((_DWORD *)v287 + 56) = v291 + 1;
      if ( v369 + 4096 != 0 && v369 != 0 && v369 != (char *)-8192LL )
        sub_BD60C0(&v368);
      return 1;
    case 'T':
      v46 = (__int64)a1[28];
      v368 = (__m128i)(unsigned __int64)v9;
      v369 = 0;
      v370 = v46;
      if ( v46 != -4096 && v46 != 0 && v46 != -8192 )
        sub_BD73F0((__int64)&v368.m128i_i64[1]);
      v47 = *((_WORD *)a1 + 120);
      v371 = a1[29];
      v372 = v47;
      sub_B33910(v373, (__int64 *)a1 + 22);
      v48 = (__int64)a1[19];
      if ( *(_BYTE *)v48 == 84 )
      {
        v164 = sub_AA5190(*(_QWORD *)(v48 + 40));
        if ( v164 )
        {
          v165 = v163;
          v166 = HIBYTE(v163);
        }
        else
        {
          v166 = 0;
          v165 = 0;
        }
        LOBYTE(v167) = v165;
        HIBYTE(v167) = v166;
        sub_A88F30((__int64)(a1 + 22), *((_QWORD *)a1[19] + 5), v164, v167);
      }
      else
      {
        sub_D5F1F0((__int64)(a1 + 22), v48);
      }
      v49 = *((_QWORD *)a1[19] + 6);
      v365.m128i_i64[0] = v49;
      if ( v49 && (sub_B96E90((__int64)&v365, v49, 1), (v50 = v365.m128i_i64[0]) != 0) )
      {
        v51 = *((unsigned int *)a1 + 46);
        v52 = a1[22];
        v53 = *((_DWORD *)a1 + 46);
        v54 = &v52[2 * v51];
        if ( v52 != v54 )
        {
          while ( *(_DWORD *)v52 )
          {
            v52 += 2;
            if ( v54 == v52 )
              goto LABEL_249;
          }
          v52[1] = v365.m128i_i64[0];
LABEL_58:
          sub_B91220((__int64)&v365, v50);
          goto LABEL_59;
        }
LABEL_249:
        v229 = *((unsigned int *)a1 + 47);
        if ( v51 >= v229 )
        {
          v297 = v51 + 1;
          if ( v229 < v297 )
          {
            v344 = v365.m128i_i64[0];
            sub_C8D5F0((__int64)(a1 + 22), a1 + 24, v297, 0x10u, v365.m128i_i64[0], (__int64)(a1 + 24));
            v50 = v344;
            v54 = &a1[22][16 * *((unsigned int *)a1 + 46)];
          }
          *v54 = 0;
          v54[1] = v50;
          v50 = v365.m128i_i64[0];
          ++*((_DWORD *)a1 + 46);
        }
        else
        {
          if ( v54 )
          {
            *(_DWORD *)v54 = 0;
            v54[1] = v50;
            v53 = *((_DWORD *)a1 + 46);
            v50 = v365.m128i_i64[0];
          }
          *((_DWORD *)a1 + 46) = v53 + 1;
        }
      }
      else
      {
        sub_93FB40((__int64)(a1 + 22), 0);
        v50 = v365.m128i_i64[0];
      }
      if ( v50 )
        goto LABEL_58;
LABEL_59:
      v55 = (__int64)*a1;
      v56 = *((_QWORD *)a1[19] + 1);
      v57 = a1[14];
      LOWORD(v367) = 257;
      v336 = v57 - a1[5];
      v362.m128i_i32[2] = sub_AE43F0(v55, v56);
      if ( v362.m128i_i32[2] > 0x40u )
        sub_C43690((__int64)&v362, v336, 0);
      else
        v362.m128i_i64[0] = v336;
      v58 = sub_291C070((__int64)(a1 + 22), (__int64)a1[4], (__int64)&v362, v56, v365.m128i_i64);
      if ( v362.m128i_i32[2] > 0x40u && v362.m128i_i64[0] )
        j_j___libc_free_0_0(v362.m128i_u64[0]);
      v59 = *(_QWORD *)(v26 - 8);
      v60 = a1[19];
      v61 = v59 + 32LL * (*(_DWORD *)(v26 + 4) & 0x7FFFFFF);
      if ( v59 != v61 )
      {
        do
        {
          while ( 1 )
          {
            if ( v60 == *(unsigned __int8 **)v59 )
            {
              if ( v60 )
              {
                v62 = *(_QWORD *)(v59 + 8);
                **(_QWORD **)(v59 + 16) = v62;
                if ( v62 )
                  *(_QWORD *)(v62 + 16) = *(_QWORD *)(v59 + 16);
              }
              *(_QWORD *)v59 = v58;
              if ( v58 )
                break;
            }
            v59 += 32;
            if ( v61 == v59 )
              goto LABEL_75;
          }
          v63 = *(_QWORD *)(v58 + 16);
          *(_QWORD *)(v59 + 8) = v63;
          if ( v63 )
            *(_QWORD *)(v63 + 16) = v59 + 8;
          *(_QWORD *)(v59 + 16) = v58 + 16;
          *(_QWORD *)(v58 + 16) = v59;
          v59 += 32;
        }
        while ( v61 != v59 );
LABEL_75:
        v60 = a1[19];
      }
      if ( sub_F50EE0(v60, 0) )
      {
        v231 = a1[2];
        v365 = (__m128i)4uLL;
        LOBYTE(v64) = v60 + 4096 != 0;
        v232 = (__int64)(v231 + 216);
        v366.m128i_i64[0] = (__int64)v60;
        if ( ((v60 != 0) & (unsigned __int8)v64) != 0 && v60 != (unsigned __int8 *)-8192LL )
          sub_BD73F0((__int64)&v365);
        sub_D6B260(v232, v365.m128i_i8, v64, v65, v66, v67);
        LOBYTE(v65) = v366.m128i_i64[0] != -4096;
        LOBYTE(v64) = v366.m128i_i64[0] != 0;
        if ( ((v366.m128i_i64[0] != 0) & (unsigned __int8)v65) != 0 && v366.m128i_i64[0] != -8192 )
          sub_BD60C0(&v365);
      }
      sub_2915D60(a1, v26, v64, v65, v66, v67);
      v68 = (__int64)a1[20];
      v365.m128i_i64[0] = v26;
      sub_2929FB0(v68, v365.m128i_i64, v69, v70, v71, v72);
      v73 = v368.m128i_i64[0];
      if ( v370 )
      {
        sub_A88F30(v368.m128i_i64[0], v370, (__int64)v371, v372);
        v73 = v368.m128i_i64[0];
      }
      else
      {
        *(_QWORD *)(v368.m128i_i64[0] + 48) = 0;
        *(_QWORD *)(v73 + 56) = 0;
        *(_WORD *)(v73 + 64) = 0;
      }
      v365.m128i_i64[0] = v373[0];
      if ( v373[0] && (sub_B96E90((__int64)&v365, v373[0], 1), (v75 = v365.m128i_i64[0]) != 0) )
      {
        v76 = *(unsigned int *)(v73 + 8);
        v77 = *(_QWORD **)v73;
        v78 = *(_DWORD *)(v73 + 8);
        v79 = (_QWORD *)(*(_QWORD *)v73 + 16 * v76);
        if ( *(_QWORD **)v73 != v79 )
        {
          while ( *(_DWORD *)v77 )
          {
            v77 += 2;
            if ( v79 == v77 )
              goto LABEL_256;
          }
          v77[1] = v365.m128i_i64[0];
          goto LABEL_86;
        }
LABEL_256:
        v230 = *(unsigned int *)(v73 + 12);
        if ( v76 >= v230 )
        {
          v298 = v76 + 1;
          if ( v230 < v298 )
          {
            sub_C8D5F0(v73, (const void *)(v73 + 16), v298, 0x10u, v73 + 16, v74);
            v79 = (_QWORD *)(*(_QWORD *)v73 + 16LL * *(unsigned int *)(v73 + 8));
          }
          *v79 = 0;
          v79[1] = v75;
          ++*(_DWORD *)(v73 + 8);
          v75 = v365.m128i_i64[0];
        }
        else
        {
          if ( v79 )
          {
            *(_DWORD *)v79 = 0;
            v79[1] = v75;
            v78 = *(_DWORD *)(v73 + 8);
            v75 = v365.m128i_i64[0];
          }
          *(_DWORD *)(v73 + 8) = v78 + 1;
        }
      }
      else
      {
        sub_93FB40(v73, 0);
        v75 = v365.m128i_i64[0];
      }
      if ( v75 )
LABEL_86:
        sub_B91220((__int64)&v365, v75);
      if ( v373[0] )
        sub_B91220((__int64)v373, v373[0]);
      if ( v370 != 0 && v370 != -4096 && v370 != -8192 )
        sub_BD60C0(&v368.m128i_i64[1]);
      return 1;
    case 'U':
      v44 = *(_QWORD *)(v26 - 32);
      if ( !v44 || *(_BYTE *)v44 || *(_QWORD *)(v44 + 24) != *(_QWORD *)(v26 + 80) )
        goto LABEL_400;
      v45 = *(_DWORD *)(v44 + 36);
      if ( v45 > 0xF5 )
        return sub_2921E80((__int64)a1, *((_QWORD *)a1[18] + 3), v25, v21, v22, v23);
      if ( v45 > 0xED )
      {
        switch ( v45 )
        {
          case 0xEEu:
          case 0xF0u:
          case 0xF1u:
            result = sub_29287A0((__int64)a1, *((_QWORD *)a1[18] + 3));
            break;
          case 0xF3u:
          case 0xF5u:
            result = sub_2924A90((__int64)a1, *((_QWORD *)a1[18] + 3));
            break;
          default:
            return sub_2921E80((__int64)a1, *((_QWORD *)a1[18] + 3), v25, v21, v22, v23);
        }
      }
      else
      {
        if ( !v45 )
LABEL_400:
          sub_42CA58(*((_QWORD *)a1[18] + 3), v20);
        return sub_2921E80((__int64)a1, *((_QWORD *)a1[18] + 3), v25, v21, v22, v23);
      }
      return result;
    case 'V':
      v27 = sub_291C360((__int64 *)a1, (__int64)(a1 + 22), *((_QWORD *)a1[19] + 1));
      v28 = a1[19];
      v29 = v27;
      if ( *(unsigned __int8 **)(v26 - 64) == v28 )
      {
        sub_AC2B30(v26 - 64, v27);
        v28 = a1[19];
      }
      if ( *(unsigned __int8 **)(v26 - 32) == v28 )
      {
        sub_AC2B30(v26 - 32, v29);
        v28 = a1[19];
      }
      if ( sub_F50EE0(v28, 0) )
      {
        v168 = a1[2];
        v369 = (char *)v28;
        LOBYTE(v30) = v28 + 4096 != 0;
        v368 = (__m128i)4uLL;
        v169 = (__int64)(v168 + 216);
        if ( ((v28 != 0) & (unsigned __int8)v30) != 0 && v28 != (unsigned __int8 *)-8192LL )
          sub_BD73F0((__int64)&v368);
        sub_D6B260(v169, v368.m128i_i8, v30, v31, v32, v33);
        LOBYTE(v31) = v369 != 0;
        LOBYTE(v30) = v369 + 4096 != 0;
        if ( ((unsigned __int8)v30 & (v369 != 0)) != 0 && v369 != (char *)-8192LL )
          sub_BD60C0(&v368);
      }
      sub_2915D60(a1, v26, v30, v31, v32, v33);
      v36 = (__int64)a1[21];
      v365.m128i_i64[0] = v26;
      v37 = *(_DWORD *)(v36 + 16);
      if ( v37 )
      {
        v176 = *(_DWORD *)(v36 + 24);
        if ( v176 )
        {
          v177 = v176 - 1;
          v178 = *(_QWORD *)(v36 + 8);
          v179 = 1;
          v180 = 0;
          v181 = v177 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
          v182 = (__int64 *)(v178 + 8LL * v181);
          v183 = *v182;
          if ( v26 == *v182 )
            return 1;
          while ( v183 != -4096 )
          {
            if ( !v180 && v183 == -8192 )
              v180 = (__int64)v182;
            v181 = v177 & (v179 + v181);
            v182 = (__int64 *)(v178 + 8LL * v181);
            v183 = *v182;
            if ( v26 == *v182 )
              return 1;
            ++v179;
          }
          if ( !v180 )
            v180 = (__int64)v182;
          v184 = v37 + 1;
          v368.m128i_i64[0] = v180;
          ++*(_QWORD *)v36;
          if ( 4 * v184 < 3 * v176 )
          {
            if ( v176 - *(_DWORD *)(v36 + 20) - v184 > v176 >> 3 )
            {
LABEL_193:
              *(_DWORD *)(v36 + 16) = v184;
              v185 = (_QWORD *)v368.m128i_i64[0];
              if ( *(_QWORD *)v368.m128i_i64[0] != -4096 )
                --*(_DWORD *)(v36 + 20);
              *v185 = v365.m128i_i64[0];
              v186 = *(unsigned int *)(v36 + 40);
              v187 = v365.m128i_i64[0];
              if ( v186 + 1 > (unsigned __int64)*(unsigned int *)(v36 + 44) )
              {
                sub_C8D5F0(v36 + 32, (const void *)(v36 + 48), v186 + 1, 8u, v178, v177);
                v186 = *(unsigned int *)(v36 + 40);
              }
              *(_QWORD *)(*(_QWORD *)(v36 + 32) + 8 * v186) = v187;
              ++*(_DWORD *)(v36 + 40);
              return 1;
            }
LABEL_285:
            sub_2404320(v36, v176);
            sub_23FE510(v36, v365.m128i_i64, &v368);
            v184 = *(_DWORD *)(v36 + 16) + 1;
            goto LABEL_193;
          }
        }
        else
        {
          v368.m128i_i64[0] = 0;
          ++*(_QWORD *)v36;
        }
        v176 *= 2;
        goto LABEL_285;
      }
      v38 = *(unsigned int *)(v36 + 40);
      v39 = *(_QWORD **)(v36 + 32);
      v40 = &v39[v38];
      v41 = (8 * v38) >> 3;
      if ( (8 * v38) >> 5 )
      {
        v42 = &v39[4 * ((8 * v38) >> 5)];
        while ( v26 != *v39 )
        {
          if ( v26 == v39[1] )
          {
            ++v39;
            goto LABEL_35;
          }
          if ( v26 == v39[2] )
          {
            v39 += 2;
            goto LABEL_35;
          }
          if ( v26 == v39[3] )
          {
            v39 += 3;
            goto LABEL_35;
          }
          v39 += 4;
          if ( v42 == v39 )
          {
            v41 = v40 - v39;
            goto LABEL_291;
          }
        }
        goto LABEL_35;
      }
LABEL_291:
      if ( v41 == 2 )
      {
LABEL_360:
        if ( v26 != *v39 )
        {
          ++v39;
          goto LABEL_294;
        }
        goto LABEL_35;
      }
      if ( v41 != 3 )
      {
        if ( v41 != 1 )
          goto LABEL_295;
LABEL_294:
        if ( v26 != *v39 )
          goto LABEL_295;
        goto LABEL_35;
      }
      if ( v26 != *v39 )
      {
        ++v39;
        goto LABEL_360;
      }
LABEL_35:
      if ( v40 != v39 )
        return 1;
LABEL_295:
      if ( v38 + 1 > (unsigned __int64)*(unsigned int *)(v36 + 44) )
      {
        sub_C8D5F0(v36 + 32, (const void *)(v36 + 48), v38 + 1, 8u, v34, v35);
        v40 = (__int64 *)(*(_QWORD *)(v36 + 32) + 8LL * *(unsigned int *)(v36 + 40));
      }
      *v40 = v26;
      v239 = (unsigned int)(*(_DWORD *)(v36 + 40) + 1);
      *(_DWORD *)(v36 + 40) = v239;
      if ( (unsigned int)v239 > 8 )
      {
        v240 = *(__int64 **)(v36 + 32);
        v241 = &v240[v239];
        while ( 1 )
        {
          v246 = *(_DWORD *)(v36 + 24);
          if ( !v246 )
            break;
          v242 = *(_QWORD *)(v36 + 8);
          v243 = (v246 - 1) & (((unsigned int)*v240 >> 9) ^ ((unsigned int)*v240 >> 4));
          v244 = (_QWORD *)(v242 + 8LL * v243);
          v245 = *v244;
          if ( *v244 != *v240 )
          {
            v299 = 1;
            v300 = 0;
            while ( v245 != -4096 )
            {
              if ( !v300 && v245 == -8192 )
                v300 = v244;
              v243 = (v246 - 1) & (v299 + v243);
              v244 = (_QWORD *)(v242 + 8LL * v243);
              v245 = *v244;
              if ( *v240 == *v244 )
                goto LABEL_300;
              ++v299;
            }
            if ( v300 )
              v244 = v300;
            v368.m128i_i64[0] = (__int64)v244;
            v301 = *(_DWORD *)(v36 + 16);
            ++*(_QWORD *)v36;
            v247 = v301 + 1;
            if ( 4 * v247 < 3 * v246 )
            {
              if ( v246 - *(_DWORD *)(v36 + 20) - v247 <= v246 >> 3 )
              {
LABEL_304:
                sub_2404320(v36, v246);
                sub_23FE510(v36, v240, &v368);
                v247 = *(_DWORD *)(v36 + 16) + 1;
              }
              *(_DWORD *)(v36 + 16) = v247;
              v248 = (__int64 *)v368.m128i_i64[0];
              if ( *(_QWORD *)v368.m128i_i64[0] != -4096 )
                --*(_DWORD *)(v36 + 20);
              *v248 = *v240;
              goto LABEL_300;
            }
LABEL_303:
            v246 *= 2;
            goto LABEL_304;
          }
LABEL_300:
          if ( v241 == ++v240 )
            return 1;
        }
        v368.m128i_i64[0] = 0;
        ++*(_QWORD *)v36;
        goto LABEL_303;
      }
      return 1;
    default:
      BUG();
  }
}
