// Function: sub_1FAE680
// Address: 0x1fae680
//
__int64 __fastcall sub_1FAE680(__int64 **a1, __int64 a2, __m128 a3, __int64 a4, __int64 a5, int a6, int a7)
{
  __int64 *v9; // rax
  __m128i v10; // xmm1
  __m128i v11; // xmm2
  int v12; // ecx
  int v13; // r8d
  int v14; // r9d
  __int64 v15; // rax
  __int64 v16; // rsi
  unsigned __int8 *v17; // rax
  __int64 v18; // r14
  const void **v19; // rax
  char v20; // bl
  char v21; // r12
  bool v22; // bl
  bool v23; // r12
  __int64 *v24; // rax
  __int64 v25; // r12
  __int64 v27; // r14
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rcx
  unsigned __int8 v31; // dl
  __int64 v32; // r12
  void *v33; // rax
  __int64 v34; // r12
  __int64 v35; // rax
  __int64 *v36; // rax
  __int64 v37; // rsi
  __m128i v38; // xmm4
  char *v39; // rax
  unsigned __int8 v40; // r12
  const void **v41; // rbx
  bool v42; // zf
  __int64 *v43; // rdi
  __int64 v44; // r14
  __int64 v45; // rax
  __int64 (*v46)(); // rax
  __int64 *v47; // r12
  __int128 v48; // rax
  bool v49; // r8
  __int64 (*v50)(); // rax
  __int64 (*v51)(); // rax
  unsigned int v52; // r11d
  __int64 *v53; // rdi
  __int64 (*v54)(); // rax
  __int16 v55; // ax
  __int64 v56; // rax
  __int64 v57; // rax
  __int16 v58; // ax
  int v59; // eax
  __int16 v60; // dx
  unsigned __int16 v61; // ax
  unsigned int *v62; // rax
  __int64 v63; // rcx
  __int64 v64; // rdx
  __int64 v65; // rax
  __int64 v66; // r14
  __int64 *v67; // rdi
  unsigned __int8 *v68; // rdx
  __int64 v69; // r8
  __int64 v70; // r9
  __int64 (__fastcall *v71)(__int64, __int64, unsigned int, __int64, unsigned int, __int64); // rax
  __int64 (*v72)(); // rax
  char v73; // al
  unsigned int v74; // r11d
  __int64 *v75; // rax
  __int64 v76; // rcx
  __int64 v77; // r14
  __int64 v78; // r15
  __int128 v79; // rax
  __int128 v80; // rax
  unsigned __int64 v81; // rax
  __int16 *v82; // rdx
  __int64 *v83; // rax
  __int64 v84; // rdx
  __int64 v85; // r15
  __int64 v86; // r14
  __int128 v87; // rax
  __int128 v88; // rax
  unsigned __int64 v89; // rax
  __int16 *v90; // rdx
  __int64 *v91; // rax
  __int64 v92; // rbx
  __int64 v93; // rax
  __int64 v94; // r10
  __int64 v95; // r14
  unsigned int v96; // r11d
  unsigned int v97; // esi
  __int64 v98; // r12
  __int64 v99; // rbx
  __int64 v100; // rdi
  char v101; // al
  __int64 v102; // r10
  unsigned __int8 v103; // cl
  unsigned int v104; // esi
  __int64 *v105; // rsi
  __int64 v106; // rax
  bool v107; // dl
  __int64 v108; // rcx
  __int64 v109; // rcx
  __int64 v110; // rax
  bool v111; // al
  unsigned int v112; // eax
  __int64 v113; // rax
  __int64 v114; // rdx
  __int64 v115; // r15
  __int64 *v116; // r14
  unsigned __int64 v117; // rax
  __int16 *v118; // rdx
  __int64 *v119; // rax
  __int64 v120; // rax
  __int64 v121; // rdi
  __int64 (*v122)(); // rax
  char v123; // al
  __int64 v124; // rcx
  char v125; // al
  __int64 *v126; // r14
  __int128 *v127; // r15
  unsigned int v128; // eax
  unsigned __int64 v129; // rax
  __int16 *v130; // rdx
  __int64 *v131; // rax
  __int64 (*v132)(); // rax
  __int64 v133; // rax
  unsigned int *v134; // rcx
  __int64 v135; // r14
  __int64 *v136; // rdi
  unsigned __int8 *v137; // rdx
  __int64 v138; // r8
  __int64 v139; // r9
  __int64 (__fastcall *v140)(__int64, __int64, unsigned int, __int64, unsigned int, __int64); // rax
  __int64 (*v141)(); // rax
  unsigned int *v142; // rdx
  __int64 v143; // r14
  __int64 *v144; // rdi
  unsigned __int8 *v145; // rdx
  __int64 v146; // r8
  __int64 v147; // r9
  __int64 (__fastcall *v148)(__int64, __int64, unsigned int, __int64, unsigned int, __int64); // rax
  __int64 (*v149)(); // rax
  unsigned int *v150; // rcx
  __int64 v151; // rsi
  __int64 *v152; // rax
  __int64 v153; // r15
  __int64 *v154; // rdi
  unsigned __int8 *v155; // rcx
  __int64 v156; // r8
  __int64 v157; // r9
  __int64 (__fastcall *v158)(__int64, __int64, unsigned int, __int64, unsigned int, __int64); // r10
  __int64 (*v159)(); // rax
  char v160; // al
  __int64 v161; // rcx
  __int64 *v162; // r14
  __int128 v163; // rax
  unsigned __int64 v164; // rax
  __int16 *v165; // rdx
  __int128 v166; // rax
  __int64 v167; // rcx
  unsigned int *v168; // rdx
  __int64 *v169; // rdi
  unsigned __int8 *v170; // rdx
  __int64 v171; // r8
  __int64 v172; // r9
  __int64 (__fastcall *v173)(__int64, __int64, unsigned int, __int64, unsigned int, __int64); // rax
  __int64 (*v174)(); // rax
  char v175; // al
  __int64 v176; // rcx
  unsigned int *v177; // rcx
  __int64 v178; // r14
  __int64 *v179; // rdi
  unsigned __int8 *v180; // rcx
  __int64 v181; // r8
  __int64 v182; // r9
  __int64 (__fastcall *v183)(__int64, __int64, unsigned int, __int64, unsigned int, __int64); // r10
  __int64 (*v184)(); // rax
  __int64 *v185; // r14
  unsigned int v186; // r15d
  __int64 v187; // rax
  __int64 v188; // rdx
  unsigned int *v189; // rdx
  __int64 v190; // r14
  __int64 *v191; // rdi
  unsigned __int8 *v192; // rdx
  __int64 v193; // r8
  __int64 v194; // r9
  __int64 (__fastcall *v195)(__int64, __int64, unsigned int, __int64, unsigned int, __int64); // rax
  __int64 (*v196)(); // rax
  char v197; // al
  unsigned int v198; // r15d
  __int128 v199; // rax
  __int128 v200; // rax
  unsigned __int64 v201; // rax
  __int16 *v202; // rdx
  __int64 v203; // rdx
  __int128 v204; // rax
  char v205; // al
  __int64 v206; // rcx
  __int64 v207; // rax
  unsigned int *v208; // rcx
  __int64 v209; // r14
  __int64 *v210; // rdi
  unsigned __int8 *v211; // rcx
  __int64 v212; // r8
  __int64 v213; // r9
  __int64 (__fastcall *v214)(__int64, __int64, unsigned int, __int64, unsigned int, __int64); // r10
  __int64 (*v215)(); // rax
  char v216; // al
  unsigned int v217; // r15d
  __int128 v218; // rax
  __int128 v219; // rax
  unsigned __int64 v220; // rax
  __int16 *v221; // rdx
  __int64 *v222; // rax
  __int64 v223; // rdx
  __int64 v224; // rax
  __int64 v225; // rsi
  __int64 v226; // rax
  __int64 v227; // rax
  __int64 v228; // rcx
  __int64 v229; // r14
  __int64 v230; // r15
  unsigned __int64 v231; // rax
  unsigned int v232; // esi
  __int16 *v233; // rdx
  __int64 *v234; // r14
  __int64 v235; // rdx
  __int64 v236; // r15
  __int128 *v237; // rax
  unsigned __int64 v238; // rax
  __int16 *v239; // rdx
  char v240; // al
  __int64 *v241; // r15
  unsigned int v242; // eax
  __int128 v243; // rax
  __int128 v244; // rax
  unsigned __int64 v245; // rax
  __int16 *v246; // rdx
  __int64 *v247; // rax
  __int64 v248; // rcx
  char v249; // al
  __int64 *v250; // r15
  __int64 v251; // rcx
  __int128 v252; // rax
  __int128 v253; // rax
  unsigned __int64 v254; // rax
  __int16 *v255; // rdx
  __int64 v256; // rcx
  __int64 v257; // rax
  __int64 *v258; // r14
  unsigned int v259; // r15d
  __int64 v260; // rax
  __int64 v261; // rdx
  __int64 v262; // r9
  __int64 v263; // rdx
  __int64 *v264; // rax
  __int64 v265; // rdx
  __int64 v266; // rax
  __int64 v267; // rcx
  __int64 v268; // rax
  char v269; // al
  __int64 v270; // rax
  __int64 *v271; // r14
  __int64 v272; // rcx
  __int128 v273; // rax
  __int64 *v274; // r15
  __int128 v275; // rax
  unsigned __int64 v276; // rax
  __int16 *v277; // rdx
  __int128 v278; // rax
  __int128 *v279; // r15
  unsigned __int64 v280; // rax
  __int16 *v281; // rdx
  __int64 v282; // rax
  __int64 v283; // rcx
  __int64 v284; // rcx
  char v285; // al
  __int128 v286; // [rsp-30h] [rbp-110h]
  __int128 v287; // [rsp-30h] [rbp-110h]
  __int128 v288; // [rsp-20h] [rbp-100h]
  __int128 v289; // [rsp-20h] [rbp-100h]
  __m128i v290; // [rsp-20h] [rbp-100h]
  __int128 v291; // [rsp-10h] [rbp-F0h]
  __int64 v292; // [rsp+0h] [rbp-E0h]
  unsigned int v293; // [rsp+0h] [rbp-E0h]
  unsigned int v294; // [rsp+0h] [rbp-E0h]
  unsigned int v295; // [rsp+0h] [rbp-E0h]
  unsigned int v296; // [rsp+0h] [rbp-E0h]
  unsigned int v297; // [rsp+0h] [rbp-E0h]
  unsigned int v298; // [rsp+0h] [rbp-E0h]
  unsigned int v299; // [rsp+0h] [rbp-E0h]
  unsigned int v300; // [rsp+0h] [rbp-E0h]
  _BYTE *v301; // [rsp+8h] [rbp-D8h]
  bool v302; // [rsp+17h] [rbp-C9h]
  __int64 v303; // [rsp+18h] [rbp-C8h]
  int v304; // [rsp+20h] [rbp-C0h]
  unsigned int v305; // [rsp+24h] [rbp-BCh]
  unsigned int v306; // [rsp+24h] [rbp-BCh]
  bool v307; // [rsp+24h] [rbp-BCh]
  __int64 v308; // [rsp+28h] [rbp-B8h]
  __int64 *v309; // [rsp+28h] [rbp-B8h]
  unsigned int v310; // [rsp+28h] [rbp-B8h]
  bool v311; // [rsp+28h] [rbp-B8h]
  bool v312; // [rsp+28h] [rbp-B8h]
  unsigned int v313; // [rsp+28h] [rbp-B8h]
  unsigned int v314; // [rsp+28h] [rbp-B8h]
  unsigned int v315; // [rsp+28h] [rbp-B8h]
  unsigned int v316; // [rsp+28h] [rbp-B8h]
  unsigned int v317; // [rsp+28h] [rbp-B8h]
  unsigned int v318; // [rsp+28h] [rbp-B8h]
  unsigned int v319; // [rsp+28h] [rbp-B8h]
  unsigned int v320; // [rsp+28h] [rbp-B8h]
  unsigned int v321; // [rsp+28h] [rbp-B8h]
  __int64 v322; // [rsp+30h] [rbp-B0h]
  bool v323; // [rsp+30h] [rbp-B0h]
  __int64 *v324; // [rsp+30h] [rbp-B0h]
  unsigned int v325; // [rsp+30h] [rbp-B0h]
  __int64 *v326; // [rsp+30h] [rbp-B0h]
  unsigned int v327; // [rsp+30h] [rbp-B0h]
  unsigned int v328; // [rsp+30h] [rbp-B0h]
  unsigned int v329; // [rsp+30h] [rbp-B0h]
  unsigned int v330; // [rsp+30h] [rbp-B0h]
  unsigned int v331; // [rsp+30h] [rbp-B0h]
  unsigned __int16 v332; // [rsp+38h] [rbp-A8h]
  unsigned __int16 v333; // [rsp+38h] [rbp-A8h]
  __int64 v334; // [rsp+38h] [rbp-A8h]
  bool v335; // [rsp+38h] [rbp-A8h]
  __int64 v336; // [rsp+40h] [rbp-A0h]
  __int64 v337; // [rsp+40h] [rbp-A0h]
  __m128i v338; // [rsp+40h] [rbp-A0h]
  unsigned int v339; // [rsp+40h] [rbp-A0h]
  unsigned int v340; // [rsp+40h] [rbp-A0h]
  __int64 *v341; // [rsp+40h] [rbp-A0h]
  unsigned int v342; // [rsp+40h] [rbp-A0h]
  unsigned int v343; // [rsp+40h] [rbp-A0h]
  __int64 v344; // [rsp+50h] [rbp-90h]
  unsigned __int8 v345; // [rsp+50h] [rbp-90h]
  __int128 v346; // [rsp+50h] [rbp-90h]
  __int128 v347; // [rsp+50h] [rbp-90h]
  __int64 *v348; // [rsp+50h] [rbp-90h]
  __int128 v349; // [rsp+50h] [rbp-90h]
  __int128 v350; // [rsp+50h] [rbp-90h]
  unsigned int v351; // [rsp+50h] [rbp-90h]
  __int64 *v352; // [rsp+50h] [rbp-90h]
  unsigned int v353; // [rsp+50h] [rbp-90h]
  __int128 v354; // [rsp+50h] [rbp-90h]
  __int64 v355; // [rsp+60h] [rbp-80h]
  __m128i v356; // [rsp+60h] [rbp-80h]
  __int64 *v357; // [rsp+60h] [rbp-80h]
  __int64 v358; // [rsp+60h] [rbp-80h]
  __int128 v359; // [rsp+60h] [rbp-80h]
  unsigned int v360; // [rsp+60h] [rbp-80h]
  unsigned int v361; // [rsp+60h] [rbp-80h]
  __int128 v362; // [rsp+60h] [rbp-80h]
  __int128 v363; // [rsp+60h] [rbp-80h]
  __int128 v364; // [rsp+60h] [rbp-80h]
  __int64 *v365; // [rsp+60h] [rbp-80h]
  unsigned int v366; // [rsp+60h] [rbp-80h]
  __int64 v367; // [rsp+70h] [rbp-70h]
  unsigned int v368; // [rsp+70h] [rbp-70h]
  __int128 v369; // [rsp+70h] [rbp-70h]
  __int128 v370; // [rsp+70h] [rbp-70h]
  __int64 v371; // [rsp+70h] [rbp-70h]
  __int128 v372; // [rsp+70h] [rbp-70h]
  unsigned int v373; // [rsp+70h] [rbp-70h]
  __int128 v374; // [rsp+70h] [rbp-70h]
  __int64 v375; // [rsp+70h] [rbp-70h]
  __int64 *v376; // [rsp+70h] [rbp-70h]
  __int128 *v377; // [rsp+70h] [rbp-70h]
  __int128 v378; // [rsp+70h] [rbp-70h]
  __int128 v379; // [rsp+70h] [rbp-70h]
  unsigned int v380; // [rsp+70h] [rbp-70h]
  __int128 v381; // [rsp+70h] [rbp-70h]
  __int128 v382; // [rsp+70h] [rbp-70h]
  unsigned int v383; // [rsp+70h] [rbp-70h]
  unsigned int v384; // [rsp+70h] [rbp-70h]
  unsigned int v385; // [rsp+80h] [rbp-60h] BYREF
  const void **v386; // [rsp+88h] [rbp-58h]
  __int64 v387; // [rsp+90h] [rbp-50h] BYREF
  int v388; // [rsp+98h] [rbp-48h]
  __int64 v389; // [rsp+A0h] [rbp-40h] BYREF
  int v390; // [rsp+A8h] [rbp-38h]

  v9 = *(__int64 **)(a2 + 32);
  v10 = _mm_loadu_si128((const __m128i *)v9);
  v11 = _mm_loadu_si128((const __m128i *)(v9 + 5));
  v303 = *v9;
  v304 = *((_DWORD *)v9 + 2);
  v308 = v9[5];
  v305 = *((_DWORD *)v9 + 12);
  v336 = sub_1D23470(v10.m128i_i64[0], v10.m128i_i64[1], v10.m128i_i64[1], v308, a6, a7);
  v15 = sub_1D23470(v11.m128i_i64[0], v11.m128i_i64[1], v11.m128i_i64[1], v12, v13, v14);
  v16 = *(_QWORD *)(a2 + 72);
  v344 = v15;
  v17 = *(unsigned __int8 **)(a2 + 40);
  v18 = *v17;
  v19 = (const void **)*((_QWORD *)v17 + 1);
  v387 = v16;
  LOBYTE(v385) = v18;
  v386 = v19;
  if ( v16 )
    sub_1623A60((__int64)&v387, v16, 2);
  v20 = *(_BYTE *)(a2 + 80);
  v388 = *(_DWORD *)(a2 + 64);
  v21 = v20;
  v22 = (v20 & 0x40) != 0;
  v23 = (v21 & 0x10) != 0;
  v322 = **a1;
  v332 = *(_WORD *)(a2 + 80);
  if ( (_BYTE)v18 )
  {
    if ( (unsigned __int8)(v18 - 14) > 0x5Fu )
      goto LABEL_5;
  }
  else if ( !sub_1F58D20((__int64)&v385) )
  {
    goto LABEL_5;
  }
  v24 = sub_1FA8C50((__int64)a1, a2, *(double *)a3.m128_u64, *(double *)v10.m128i_i64, v11);
  if ( v24 )
    goto LABEL_7;
LABEL_5:
  v302 = v344 != 0 && v336 != 0;
  if ( v302 )
  {
    v25 = (__int64)sub_1D332F0(
                     *a1,
                     77,
                     (__int64)&v387,
                     v385,
                     v386,
                     v332,
                     *(double *)a3.m128_u64,
                     *(double *)v10.m128i_i64,
                     v11,
                     v10.m128i_i64[0],
                     v10.m128i_u64[1],
                     *(_OWORD *)&v11);
    goto LABEL_8;
  }
  v24 = sub_1F77C50(a1, a2, *(double *)a3.m128_u64, *(double *)v10.m128i_i64, v11);
  if ( v24 )
  {
LABEL_7:
    v25 = (__int64)v24;
    goto LABEL_8;
  }
  v301 = (_BYTE *)(v322 + 792);
  if ( v344 )
  {
    v27 = *(_QWORD *)(v344 + 88);
    if ( *(void **)(v27 + 32) == sub_16982C0() )
    {
      v29 = *(_QWORD *)(v27 + 40);
      if ( (*(_BYTE *)(v29 + 26) & 7) == 3 )
      {
        v28 = v29 + 8;
        goto LABEL_19;
      }
    }
    else if ( (*(_BYTE *)(v27 + 50) & 7) == 3 )
    {
      v28 = v27 + 32;
LABEL_19:
      if ( (*(_BYTE *)(v28 + 18) & 8) == 0 || (*(_BYTE *)(v322 + 792) & 2) != 0 || v22 )
      {
        v24 = (__int64 *)v10.m128i_i64[0];
        goto LABEL_7;
      }
    }
    if ( v305 == v304 && v308 == v303 && ((*(_BYTE *)(v322 + 792) & 2) != 0 || v23) )
    {
LABEL_29:
      v25 = (__int64)sub_1D364E0((__int64)*a1, (__int64)&v387, v385, v386, 0, 0.0, *(double *)v10.m128i_i64, v11);
      goto LABEL_8;
    }
    v30 = (__int64)a1[1];
    v31 = *((_BYTE *)a1 + 24);
    goto LABEL_43;
  }
  if ( v308 == v303 && v305 == v304 && ((*(_BYTE *)(v322 + 792) & 2) != 0 || v23) )
    goto LABEL_29;
  v30 = (__int64)a1[1];
  v31 = *((_BYTE *)a1 + 24);
  if ( v336 )
  {
    v32 = *(_QWORD *)(v336 + 88);
    v337 = (__int64)a1[1];
    v345 = *((_BYTE *)a1 + 24);
    v33 = sub_16982C0();
    v31 = v345;
    v30 = v337;
    v34 = *(void **)(v32 + 32) == v33 ? *(_QWORD *)(v32 + 40) + 8LL : v32 + 32;
    if ( (*(_BYTE *)(v34 + 18) & 7) == 3 && ((*(_BYTE *)(v322 + 792) & 0x20) != 0 || v22) )
    {
      if ( (unsigned __int8)sub_1F79A30(
                              v308,
                              v305,
                              v345,
                              v337,
                              v301,
                              0,
                              *(double *)a3.m128_u64,
                              *(double *)v10.m128i_i64,
                              *(double *)v11.m128i_i64) )
      {
        v103 = *((_BYTE *)a1 + 24);
        v104 = v11.m128i_u32[2];
        v102 = v11.m128i_i64[0];
        goto LABEL_102;
      }
      v31 = *((_BYTE *)a1 + 24);
      if ( !v31
        || ((v30 = (__int64)a1[1], v35 = 1, (_BYTE)v18 == 1)
         || (_BYTE)v18 && (v35 = (unsigned __int8)v18, *(_QWORD *)(v30 + 8 * v18 + 120)))
        && !*(_BYTE *)(v30 + 259 * v35 + 2584) )
      {
        v25 = sub_1D309E0(
                *a1,
                162,
                (__int64)&v387,
                v385,
                v386,
                v332,
                *(double *)a3.m128_u64,
                *(double *)v10.m128i_i64,
                *(double *)v11.m128i_i64,
                *(_OWORD *)&v11);
        goto LABEL_8;
      }
    }
  }
LABEL_43:
  if ( (unsigned __int8)sub_1F79A30(
                          v308,
                          v305,
                          v31,
                          v30,
                          v301,
                          0,
                          *(double *)a3.m128_u64,
                          *(double *)v10.m128i_i64,
                          *(double *)v11.m128i_i64) )
  {
    v47 = *a1;
    *(_QWORD *)&v48 = sub_1F7A040(
                        v11.m128i_i64[0],
                        v11.m128i_u32[2],
                        *a1,
                        *((_BYTE *)a1 + 24),
                        0,
                        *(double *)a3.m128_u64,
                        *(double *)v10.m128i_i64,
                        v11);
    v25 = (__int64)sub_1D332F0(
                     v47,
                     76,
                     (__int64)&v387,
                     v385,
                     v386,
                     v332,
                     *(double *)a3.m128_u64,
                     *(double *)v10.m128i_i64,
                     v11,
                     v10.m128i_i64[0],
                     v10.m128i_u64[1],
                     v48);
    goto LABEL_8;
  }
  if ( (*(_BYTE *)(v322 + 792) & 2) == 0 || *(_WORD *)(v308 + 24) != 76 )
  {
LABEL_46:
    v36 = *(__int64 **)(a2 + 32);
    v37 = *(_QWORD *)(a2 + 72);
    v38 = _mm_loadu_si128((const __m128i *)(v36 + 5));
    v367 = *v36;
    v355 = v36[5];
    v39 = *(char **)(a2 + 40);
    v338 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 32));
    v40 = *v39;
    v41 = (const void **)*((_QWORD *)v39 + 1);
    v389 = v37;
    v346 = (__int128)v38;
    if ( v37 )
      sub_1623A60((__int64)&v389, v37, 2);
    v42 = *((_BYTE *)a1 + 24) == 0;
    v43 = a1[1];
    v390 = *(_DWORD *)(a2 + 64);
    v44 = **a1;
    if ( !v42 && ((v45 = 1, v40 == 1) || v40 && (v45 = v40, v43[v40 + 15])) )
    {
      v49 = *((_BYTE *)v43 + 259 * v45 + 2522) == 0;
      v50 = *(__int64 (**)())(*v43 + 920);
      if ( v50 == sub_1F3CBF0
        || (v335 = v49,
            v292 = v40,
            v285 = ((__int64 (__fastcall *)(__int64 *, _QWORD, const void **))v50)(v43, v40, v41),
            v49 = v335,
            !v285) )
      {
LABEL_60:
        if ( !v49 )
          goto LABEL_53;
LABEL_61:
        v333 = *(_WORD *)(a2 + 80);
        if ( (*(_BYTE *)(v44 + 792) & 2) != 0 )
        {
          v323 = 1;
        }
        else
        {
          v323 = (*(_BYTE *)(a2 + 81) & 2) != 0 || (*(_BYTE *)(a2 + 81) & 8) != 0;
          if ( !v323 && *(_DWORD *)(v44 + 816) )
          {
            if ( !v49 )
              goto LABEL_53;
            v52 = 100;
            v132 = *(__int64 (**)())(**(_QWORD **)((*a1)[4] + 16) + 64LL);
            if ( v132 == sub_1D12D30
              || (v307 = v49, v133 = v132(), v52 = 100, v49 = v307, (v121 = v133) == 0)
              || (v122 = *(__int64 (**)())(*(_QWORD *)v133 + 88LL), v122 == sub_1F6BBD0) )
            {
LABEL_65:
              v53 = a1[1];
              v54 = *(__int64 (**)())(*v53 + 256);
              if ( v54 == sub_1F3CA50 )
              {
                v55 = *(_WORD *)(v367 + 24);
                if ( v55 == 78 )
                  goto LABEL_67;
                if ( *(_WORD *)(v355 + 24) != 78 )
                {
                  v107 = 0;
                  goto LABEL_114;
                }
              }
              else
              {
                v124 = v292;
                v313 = v52;
                LOBYTE(v124) = v40;
                v292 = v124;
                v125 = ((__int64 (__fastcall *)(__int64 *, _QWORD, const void **))v54)(v53, (unsigned int)v124, v41);
                v52 = v313;
                v107 = v125;
                v55 = *(_WORD *)(v367 + 24);
                if ( v55 == 78 )
                {
                  if ( v107 )
                  {
LABEL_174:
                    v185 = *a1;
                    v186 = v292;
                    v361 = v52;
                    LOBYTE(v186) = v40;
                    v187 = sub_1D309E0(
                             *a1,
                             162,
                             (__int64)&v389,
                             v186,
                             v41,
                             0,
                             *(double *)a3.m128_u64,
                             *(double *)v10.m128i_i64,
                             *(double *)v11.m128i_i64,
                             v346);
                    v131 = sub_1D3A900(
                             v185,
                             v361,
                             (__int64)&v389,
                             v186,
                             v41,
                             v333,
                             a3,
                             *(double *)v10.m128i_i64,
                             v11,
                             **(_QWORD **)(v367 + 32),
                             *(__int16 **)(*(_QWORD *)(v367 + 32) + 8LL),
                             *(_OWORD *)(*(_QWORD *)(v367 + 32) + 40LL),
                             v187,
                             v188);
                    goto LABEL_132;
                  }
LABEL_67:
                  v56 = *(_QWORD *)(v367 + 48);
                  if ( v56 && !*(_QWORD *)(v56 + 32) )
                    goto LABEL_174;
                  if ( *(_WORD *)(v355 + 24) == 78 )
                  {
                    v57 = *(_QWORD *)(v355 + 48);
                    if ( v57 )
                    {
                      if ( !*(_QWORD *)(v57 + 32) )
                      {
LABEL_131:
                        v126 = *a1;
                        v373 = v52;
                        v127 = *(__int128 **)(v355 + 32);
                        v128 = v292;
                        LOBYTE(v128) = v40;
                        v295 = v128;
                        v129 = sub_1D309E0(
                                 *a1,
                                 162,
                                 (__int64)&v389,
                                 v128,
                                 v41,
                                 0,
                                 *(double *)a3.m128_u64,
                                 *(double *)v10.m128i_i64,
                                 *(double *)v11.m128i_i64,
                                 *v127);
                        v131 = sub_1D3A900(
                                 v126,
                                 v373,
                                 (__int64)&v389,
                                 v295,
                                 v41,
                                 v333,
                                 a3,
                                 *(double *)v10.m128i_i64,
                                 v11,
                                 v129,
                                 v130,
                                 *(__int128 *)((char *)v127 + 40),
                                 v338.m128i_i64[0],
                                 v338.m128i_i64[1]);
LABEL_132:
                        v92 = (__int64)v131;
                        goto LABEL_89;
                      }
                    }
                  }
LABEL_70:
                  if ( *(_WORD *)(v367 + 24) != 157 )
                    goto LABEL_71;
                  v134 = *(unsigned int **)(v367 + 32);
                  v135 = *(_QWORD *)v134;
                  if ( *(_WORD *)(*(_QWORD *)v134 + 24LL) == 78 )
                  {
                    v136 = a1[1];
                    v137 = (unsigned __int8 *)(*(_QWORD *)(v135 + 40) + 16LL * v134[2]);
                    v138 = *v137;
                    v139 = *((_QWORD *)v137 + 1);
                    v140 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64, unsigned int, __int64))(*v136 + 888);
                    if ( v140 == sub_1F3CF40 )
                    {
                      v141 = *(__int64 (**)())(*v136 + 880);
                      if ( v141 == sub_1D5A410 )
                        goto LABEL_143;
                      v316 = v52;
                      v240 = ((__int64 (__fastcall *)(__int64 *, _QWORD, const void **, _QWORD, __int64))v141)(
                               v136,
                               v40,
                               v41,
                               (unsigned __int8)v138,
                               v139);
                      v52 = v316;
                    }
                    else
                    {
                      v248 = v292;
                      v317 = v52;
                      LOBYTE(v248) = v40;
                      v292 = v248;
                      v240 = v140((__int64)v136, v52, v248, (__int64)v41, v138, v139);
                      v52 = v317;
                    }
                    if ( v240 )
                    {
                      v241 = *a1;
                      v242 = v292;
                      v343 = v52;
                      LOBYTE(v242) = v40;
                      v298 = v242;
                      *(_QWORD *)&v243 = sub_1D309E0(
                                           *a1,
                                           162,
                                           (__int64)&v389,
                                           v242,
                                           v41,
                                           0,
                                           *(double *)a3.m128_u64,
                                           *(double *)v10.m128i_i64,
                                           *(double *)v11.m128i_i64,
                                           v346);
                      v378 = v243;
                      *(_QWORD *)&v244 = sub_1D309E0(
                                           *a1,
                                           157,
                                           (__int64)&v389,
                                           v298,
                                           v41,
                                           0,
                                           *(double *)a3.m128_u64,
                                           *(double *)v10.m128i_i64,
                                           *(double *)v11.m128i_i64,
                                           *(_OWORD *)(*(_QWORD *)(v135 + 32) + 40LL));
                      v364 = v244;
                      v245 = sub_1D309E0(
                               *a1,
                               157,
                               (__int64)&v389,
                               v298,
                               v41,
                               0,
                               *(double *)a3.m128_u64,
                               *(double *)v10.m128i_i64,
                               *(double *)v11.m128i_i64,
                               *(_OWORD *)*(_QWORD *)(v135 + 32));
                      v247 = sub_1D3A900(
                               v241,
                               v343,
                               (__int64)&v389,
                               v298,
                               v41,
                               v333,
                               a3,
                               *(double *)v10.m128i_i64,
                               v11,
                               v245,
                               v246,
                               v364,
                               v378,
                               *((__int64 *)&v378 + 1));
                      goto LABEL_211;
                    }
                  }
LABEL_143:
                  if ( *(_WORD *)(v355 + 24) != 157 )
                    goto LABEL_147;
                  goto LABEL_144;
                }
                if ( *(_WORD *)(v355 + 24) != 78 )
                {
LABEL_114:
                  if ( v55 != 162 )
                  {
                    v302 = v107;
                    goto LABEL_70;
                  }
                  v109 = *(_QWORD *)(v367 + 32);
                  if ( *(_WORD *)(*(_QWORD *)v109 + 24LL) != 78 )
                  {
                    v302 = v107;
                    if ( *(_WORD *)(v367 + 24) != 157 )
                    {
LABEL_71:
                      if ( *(_WORD *)(v355 + 24) != 157 )
                      {
                        v58 = *(_WORD *)(v367 + 24);
                        goto LABEL_73;
                      }
LABEL_144:
                      v142 = *(unsigned int **)(v355 + 32);
                      v143 = *(_QWORD *)v142;
                      if ( *(_WORD *)(*(_QWORD *)v142 + 24LL) != 78 )
                        goto LABEL_147;
                      v144 = a1[1];
                      v145 = (unsigned __int8 *)(*(_QWORD *)(v143 + 40) + 16LL * v142[2]);
                      v146 = *v145;
                      v147 = *((_QWORD *)v145 + 1);
                      v148 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64, unsigned int, __int64))(*v144 + 888);
                      if ( v148 == sub_1F3CF40 )
                      {
                        v149 = *(__int64 (**)())(*v144 + 880);
                        if ( v149 == sub_1D5A410 )
                          goto LABEL_147;
                        v318 = v52;
                        v249 = ((__int64 (__fastcall *)(__int64 *, _QWORD, const void **, _QWORD, __int64))v149)(
                                 v144,
                                 v40,
                                 v41,
                                 (unsigned __int8)v146,
                                 v147);
                        v52 = v318;
                      }
                      else
                      {
                        v256 = v292;
                        v319 = v52;
                        LOBYTE(v256) = v40;
                        v292 = v256;
                        v249 = v148((__int64)v144, v52, v256, (__int64)v41, v146, v147);
                        v52 = v319;
                      }
                      if ( !v249 )
                      {
LABEL_147:
                        v58 = *(_WORD *)(v367 + 24);
                        if ( v58 != 157 )
                          goto LABEL_73;
                        v150 = *(unsigned int **)(v367 + 32);
                        v151 = *(_QWORD *)v150;
                        if ( *(_WORD *)(*(_QWORD *)v150 + 24LL) != 162 )
                          goto LABEL_74;
                        v152 = *(__int64 **)(v151 + 32);
                        v153 = *v152;
                        if ( *(_WORD *)(*v152 + 24) != 78 )
                          goto LABEL_152;
                        v154 = a1[1];
                        v155 = (unsigned __int8 *)(*(_QWORD *)(v151 + 40) + 16LL * v150[2]);
                        v156 = *v155;
                        v157 = *((_QWORD *)v155 + 1);
                        v158 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64, unsigned int, __int64))(*v154 + 888);
                        if ( v158 == sub_1F3CF40 )
                        {
                          v159 = *(__int64 (**)())(*v154 + 880);
                          if ( v159 == sub_1D5A410 )
                          {
LABEL_152:
                            v58 = 157;
                            goto LABEL_73;
                          }
                          v314 = v52;
                          v160 = ((__int64 (__fastcall *)(__int64 *, _QWORD, const void **, _QWORD, __int64))v159)(
                                   v154,
                                   v40,
                                   v41,
                                   (unsigned __int8)v156,
                                   v157);
                          v52 = v314;
                        }
                        else
                        {
                          v266 = v292;
                          v320 = v52;
                          LOBYTE(v266) = v40;
                          v292 = v266;
                          v160 = v158((__int64)v154, v52, v266, (__int64)v41, v156, v157);
                          v52 = v320;
                        }
                        if ( v160 )
                          goto LABEL_155;
                        v58 = *(_WORD *)(v367 + 24);
LABEL_73:
                        if ( v58 != 162 )
                          goto LABEL_74;
                        v167 = *(_QWORD *)(v367 + 32);
                        if ( *(_WORD *)(*(_QWORD *)v167 + 24LL) != 157 )
                          goto LABEL_74;
                        v168 = *(unsigned int **)(*(_QWORD *)v167 + 32LL);
                        v153 = *(_QWORD *)v168;
                        if ( *(_WORD *)(*(_QWORD *)v168 + 24LL) != 78 )
                          goto LABEL_74;
                        v169 = a1[1];
                        v170 = (unsigned __int8 *)(*(_QWORD *)(v153 + 40) + 16LL * v168[2]);
                        v171 = *v170;
                        v172 = *((_QWORD *)v170 + 1);
                        v173 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64, unsigned int, __int64))(*v169 + 888);
                        if ( v173 == sub_1F3CF40 )
                        {
                          v174 = *(__int64 (**)())(*v169 + 880);
                          if ( v174 == sub_1D5A410 )
                            goto LABEL_74;
                          v315 = v52;
                          v175 = ((__int64 (__fastcall *)(__int64 *, _QWORD, const void **, _QWORD, __int64))v174)(
                                   v169,
                                   v40,
                                   v41,
                                   (unsigned __int8)v171,
                                   v172);
                          v52 = v315;
                        }
                        else
                        {
                          v267 = v292;
                          v321 = v52;
                          LOBYTE(v267) = v40;
                          v292 = v267;
                          v175 = v173((__int64)v169, v52, v267, (__int64)v41, v171, v172);
                          v52 = v321;
                        }
                        if ( !v175 )
                        {
LABEL_74:
                          if ( !v302 )
                            goto LABEL_53;
                          v59 = *(unsigned __int16 *)(v367 + 24);
                          v60 = *(_WORD *)(v367 + 24);
                          if ( v323 )
                          {
                            if ( v52 != v59 )
                            {
                              if ( v52 != *(unsigned __int16 *)(v355 + 24) )
                              {
                                if ( v60 != 157 )
                                {
                                  v61 = *(_WORD *)(v355 + 24);
                                  goto LABEL_80;
                                }
LABEL_177:
                                v189 = *(unsigned int **)(v367 + 32);
                                v375 = *(_QWORD *)v189;
                                if ( v52 == *(unsigned __int16 *)(*(_QWORD *)v189 + 24LL) )
                                {
                                  v190 = *(_QWORD *)(*(_QWORD *)(v375 + 32) + 80LL);
                                  if ( *(_WORD *)(v190 + 24) == 78 )
                                  {
                                    v191 = a1[1];
                                    v192 = (unsigned __int8 *)(*(_QWORD *)(v375 + 40) + 16LL * v189[2]);
                                    v193 = *v192;
                                    v194 = *((_QWORD *)v192 + 1);
                                    v195 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64, unsigned int, __int64))(*v191 + 888);
                                    if ( v195 == sub_1F3CF40 )
                                    {
                                      v196 = *(__int64 (**)())(*v191 + 880);
                                      if ( v196 == sub_1D5A410 )
                                        goto LABEL_165;
                                      v325 = v52;
                                      v197 = ((__int64 (__fastcall *)(__int64 *, _QWORD, const void **, _QWORD, __int64))v196)(
                                               v191,
                                               v40,
                                               v41,
                                               (unsigned __int8)v193,
                                               v194);
                                      v52 = v325;
                                    }
                                    else
                                    {
                                      v284 = v292;
                                      v331 = v52;
                                      LOBYTE(v284) = v40;
                                      v292 = v284;
                                      v197 = v195((__int64)v191, v52, v284, (__int64)v41, v193, v194);
                                      v52 = v331;
                                    }
                                    if ( v197 )
                                    {
                                      v198 = v292;
                                      v339 = v52;
                                      LOBYTE(v198) = v40;
                                      v326 = *a1;
                                      *(_QWORD *)&v199 = sub_1D309E0(
                                                           *a1,
                                                           162,
                                                           (__int64)&v389,
                                                           v198,
                                                           v41,
                                                           0,
                                                           *(double *)a3.m128_u64,
                                                           *(double *)v10.m128i_i64,
                                                           *(double *)v11.m128i_i64,
                                                           v346);
                                      v362 = v199;
                                      *(_QWORD *)&v200 = sub_1D309E0(
                                                           *a1,
                                                           157,
                                                           (__int64)&v389,
                                                           v198,
                                                           v41,
                                                           0,
                                                           *(double *)a3.m128_u64,
                                                           *(double *)v10.m128i_i64,
                                                           *(double *)v11.m128i_i64,
                                                           *(_OWORD *)(*(_QWORD *)(v190 + 32) + 40LL));
                                      v349 = v200;
                                      v201 = sub_1D309E0(
                                               *a1,
                                               157,
                                               (__int64)&v389,
                                               v198,
                                               v41,
                                               0,
                                               *(double *)a3.m128_u64,
                                               *(double *)v10.m128i_i64,
                                               *(double *)v11.m128i_i64,
                                               *(_OWORD *)*(_QWORD *)(v190 + 32));
                                      v286 = v349;
                                      v294 = v198;
                                      v348 = v326;
                                      v116 = sub_1D3A900(
                                               v326,
                                               v339,
                                               (__int64)&v389,
                                               v198,
                                               v41,
                                               v333,
                                               a3,
                                               *(double *)v10.m128i_i64,
                                               v11,
                                               v201,
                                               v202,
                                               v286,
                                               v362,
                                               *((__int64 *)&v362 + 1));
                                      v115 = v203;
                                      *(_QWORD *)&v204 = sub_1D309E0(
                                                           *a1,
                                                           157,
                                                           (__int64)&v389,
                                                           v294,
                                                           v41,
                                                           0,
                                                           *(double *)a3.m128_u64,
                                                           *(double *)v10.m128i_i64,
                                                           *(double *)v11.m128i_i64,
                                                           *(_OWORD *)(*(_QWORD *)(v375 + 32) + 40LL));
                                      v359 = v204;
                                      v117 = sub_1D309E0(
                                               *a1,
                                               157,
                                               (__int64)&v389,
                                               v294,
                                               v41,
                                               0,
                                               *(double *)a3.m128_u64,
                                               *(double *)v10.m128i_i64,
                                               *(double *)v11.m128i_i64,
                                               *(_OWORD *)*(_QWORD *)(v375 + 32));
                                      goto LABEL_122;
                                    }
                                  }
                                }
LABEL_165:
                                v61 = *(_WORD *)(v355 + 24);
                                if ( v52 == v61 )
                                {
                                  v176 = *(_QWORD *)(*(_QWORD *)(v355 + 32) + 80LL);
                                  if ( *(_WORD *)(v176 + 24) == 157 )
                                  {
                                    v177 = *(unsigned int **)(v176 + 32);
                                    v178 = *(_QWORD *)v177;
                                    if ( *(_WORD *)(*(_QWORD *)v177 + 24LL) != 78 )
                                    {
LABEL_170:
                                      v61 = *(_WORD *)(v355 + 24);
                                      goto LABEL_80;
                                    }
                                    v179 = a1[1];
                                    v180 = (unsigned __int8 *)(*(_QWORD *)(v178 + 40) + 16LL * v177[2]);
                                    v181 = *v180;
                                    v182 = *((_QWORD *)v180 + 1);
                                    v183 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64, unsigned int, __int64))(*v179 + 888);
                                    if ( v183 == sub_1F3CF40 )
                                    {
                                      v184 = *(__int64 (**)())(*v179 + 880);
                                      if ( v184 == sub_1D5A410 )
                                        goto LABEL_170;
                                      v380 = v52;
                                      v269 = ((__int64 (__fastcall *)(__int64 *, _QWORD, const void **, _QWORD, __int64))v184)(
                                               v179,
                                               v40,
                                               v41,
                                               (unsigned __int8)v181,
                                               v182);
                                      v52 = v380;
                                    }
                                    else
                                    {
                                      v282 = v292;
                                      v383 = v52;
                                      LOBYTE(v282) = v40;
                                      v292 = v282;
                                      v269 = v183((__int64)v179, v52, v282, (__int64)v41, v181, v182);
                                      v52 = v383;
                                    }
                                    if ( v269 )
                                    {
                                      v270 = *(_QWORD *)(v178 + 32);
                                      v271 = *a1;
                                      v272 = v292;
                                      v330 = v52;
                                      LOBYTE(v272) = v40;
                                      v354 = (__int128)_mm_loadu_si128((const __m128i *)v270);
                                      v300 = v272;
                                      *(_QWORD *)&v273 = sub_1D309E0(
                                                           *a1,
                                                           157,
                                                           (__int64)&v389,
                                                           v272,
                                                           v41,
                                                           0,
                                                           *(double *)a3.m128_u64,
                                                           *(double *)v10.m128i_i64,
                                                           *(double *)v11.m128i_i64,
                                                           *(_OWORD *)(v270 + 40));
                                      v274 = *a1;
                                      v381 = v273;
                                      *(_QWORD *)&v275 = sub_1D309E0(
                                                           *a1,
                                                           157,
                                                           (__int64)&v389,
                                                           v300,
                                                           v41,
                                                           0,
                                                           *(double *)a3.m128_u64,
                                                           *(double *)v10.m128i_i64,
                                                           *(double *)v11.m128i_i64,
                                                           v354);
                                      v276 = sub_1D309E0(
                                               v274,
                                               162,
                                               (__int64)&v389,
                                               v300,
                                               v41,
                                               0,
                                               *(double *)a3.m128_u64,
                                               *(double *)v10.m128i_i64,
                                               *(double *)v11.m128i_i64,
                                               v275);
                                      *(_QWORD *)&v278 = sub_1D3A900(
                                                           v271,
                                                           v330,
                                                           (__int64)&v389,
                                                           v300,
                                                           v41,
                                                           v333,
                                                           a3,
                                                           *(double *)v10.m128i_i64,
                                                           v11,
                                                           v276,
                                                           v277,
                                                           v381,
                                                           v338.m128i_i64[0],
                                                           v338.m128i_i64[1]);
                                      v279 = *(__int128 **)(v355 + 32);
                                      v382 = v278;
                                      v280 = sub_1D309E0(
                                               *a1,
                                               162,
                                               (__int64)&v389,
                                               v300,
                                               v41,
                                               0,
                                               *(double *)a3.m128_u64,
                                               *(double *)v10.m128i_i64,
                                               *(double *)v11.m128i_i64,
                                               *v279);
                                      v131 = sub_1D3A900(
                                               v271,
                                               v330,
                                               (__int64)&v389,
                                               v300,
                                               v41,
                                               v333,
                                               a3,
                                               *(double *)v10.m128i_i64,
                                               v11,
                                               v280,
                                               v281,
                                               *(__int128 *)((char *)v279 + 40),
                                               v382,
                                               *((__int64 *)&v382 + 1));
                                      goto LABEL_132;
                                    }
                                    v61 = *(_WORD *)(v355 + 24);
                                  }
                                }
LABEL_80:
                                if ( v61 == 157 )
                                {
                                  v62 = *(unsigned int **)(v355 + 32);
                                  v63 = *(_QWORD *)v62;
                                  if ( v52 == *(unsigned __int16 *)(*(_QWORD *)v62 + 24LL) )
                                  {
                                    v64 = v62[2];
                                    v65 = *(_QWORD *)(v63 + 32);
                                    v66 = *(_QWORD *)(v65 + 80);
                                    v356 = _mm_loadu_si128((const __m128i *)v65);
                                    v347 = (__int128)_mm_loadu_si128((const __m128i *)(v65 + 40));
                                    if ( *(_WORD *)(v66 + 24) == 78 )
                                    {
                                      v67 = a1[1];
                                      v68 = (unsigned __int8 *)(*(_QWORD *)(v63 + 40) + 16 * v64);
                                      v69 = *v68;
                                      v70 = *((_QWORD *)v68 + 1);
                                      v71 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64, unsigned int, __int64))(*v67 + 888);
                                      if ( v71 == sub_1F3CF40 )
                                      {
                                        v72 = *(__int64 (**)())(*v67 + 880);
                                        if ( v72 == sub_1D5A410 )
                                          goto LABEL_53;
                                        v368 = v52;
                                        v73 = ((__int64 (__fastcall *)(__int64 *, _QWORD, const void **, _QWORD, __int64))v72)(
                                                v67,
                                                v40,
                                                v41,
                                                (unsigned __int8)v69,
                                                v70);
                                        v74 = v368;
                                      }
                                      else
                                      {
                                        v283 = v292;
                                        v384 = v52;
                                        LOBYTE(v283) = v40;
                                        v292 = v283;
                                        v73 = v71((__int64)v67, v52, v283, (__int64)v41, v69, v70);
                                        v74 = v384;
                                      }
                                      v306 = v74;
                                      if ( v73 )
                                      {
                                        v75 = *(__int64 **)(v66 + 32);
                                        v76 = v292;
                                        v77 = *v75;
                                        v78 = v75[1];
                                        LOBYTE(v76) = v40;
                                        v309 = *a1;
                                        v293 = v76;
                                        *(_QWORD *)&v79 = sub_1D309E0(
                                                            *a1,
                                                            157,
                                                            (__int64)&v389,
                                                            v76,
                                                            v41,
                                                            0,
                                                            *(double *)a3.m128_u64,
                                                            *(double *)v10.m128i_i64,
                                                            *(double *)v11.m128i_i64,
                                                            *(_OWORD *)(v75 + 5));
                                        *((_QWORD *)&v288 + 1) = v78;
                                        *(_QWORD *)&v288 = v77;
                                        v324 = *a1;
                                        v369 = v79;
                                        *(_QWORD *)&v80 = sub_1D309E0(
                                                            *a1,
                                                            157,
                                                            (__int64)&v389,
                                                            v293,
                                                            v41,
                                                            0,
                                                            *(double *)a3.m128_u64,
                                                            *(double *)v10.m128i_i64,
                                                            *(double *)v11.m128i_i64,
                                                            v288);
                                        v81 = sub_1D309E0(
                                                v324,
                                                162,
                                                (__int64)&v389,
                                                v293,
                                                v41,
                                                0,
                                                *(double *)a3.m128_u64,
                                                *(double *)v10.m128i_i64,
                                                *(double *)v11.m128i_i64,
                                                v80);
                                        v83 = sub_1D3A900(
                                                v309,
                                                v306,
                                                (__int64)&v389,
                                                v293,
                                                v41,
                                                v333,
                                                a3,
                                                *(double *)v10.m128i_i64,
                                                v11,
                                                v81,
                                                v82,
                                                v369,
                                                v338.m128i_i64[0],
                                                v338.m128i_i64[1]);
                                        v85 = v84;
                                        v86 = (__int64)v83;
                                        *(_QWORD *)&v87 = sub_1D309E0(
                                                            *a1,
                                                            157,
                                                            (__int64)&v389,
                                                            v293,
                                                            v41,
                                                            0,
                                                            *(double *)a3.m128_u64,
                                                            *(double *)v10.m128i_i64,
                                                            *(double *)v11.m128i_i64,
                                                            v347);
                                        v289 = (__int128)v356;
                                        v357 = *a1;
                                        v370 = v87;
                                        *(_QWORD *)&v88 = sub_1D309E0(
                                                            *a1,
                                                            157,
                                                            (__int64)&v389,
                                                            v293,
                                                            v41,
                                                            0,
                                                            *(double *)a3.m128_u64,
                                                            *(double *)v10.m128i_i64,
                                                            *(double *)v11.m128i_i64,
                                                            v289);
                                        v89 = sub_1D309E0(
                                                v357,
                                                162,
                                                (__int64)&v389,
                                                v293,
                                                v41,
                                                0,
                                                *(double *)a3.m128_u64,
                                                *(double *)v10.m128i_i64,
                                                *(double *)v11.m128i_i64,
                                                v88);
                                        v91 = sub_1D3A900(
                                                v309,
                                                v306,
                                                (__int64)&v389,
                                                v293,
                                                v41,
                                                v333,
                                                a3,
                                                *(double *)v10.m128i_i64,
                                                v11,
                                                v89,
                                                v90,
                                                v370,
                                                v86,
                                                v85);
LABEL_88:
                                        v92 = (__int64)v91;
                                        goto LABEL_89;
                                      }
                                    }
                                  }
                                }
LABEL_53:
                                if ( v389 )
                                  sub_161E7C0((__int64)&v389, v389);
                                goto LABEL_55;
                              }
                              v226 = *(_QWORD *)(*(_QWORD *)(v355 + 32) + 80LL);
                              if ( *(_WORD *)(v226 + 24) != 78 )
                                goto LABEL_164;
LABEL_201:
                              v227 = *(_QWORD *)(v226 + 32);
                              v228 = v292;
                              v351 = v52;
                              v229 = *(_QWORD *)(v227 + 40);
                              v230 = *(_QWORD *)(v227 + 48);
                              LOBYTE(v228) = v40;
                              v297 = v228;
                              v376 = *a1;
                              v231 = sub_1D309E0(
                                       *a1,
                                       162,
                                       (__int64)&v389,
                                       v228,
                                       v41,
                                       0,
                                       *(double *)a3.m128_u64,
                                       *(double *)v10.m128i_i64,
                                       *(double *)v11.m128i_i64,
                                       *(_OWORD *)v227);
                              v290 = v338;
                              *((_QWORD *)&v287 + 1) = v230;
                              v232 = v351;
                              *(_QWORD *)&v287 = v229;
                              v342 = v351;
                              v352 = v376;
                              v234 = sub_1D3A900(
                                       v376,
                                       v232,
                                       (__int64)&v389,
                                       v297,
                                       v41,
                                       v333,
                                       a3,
                                       *(double *)v10.m128i_i64,
                                       v11,
                                       v231,
                                       v233,
                                       v287,
                                       v290.m128i_i64[0],
                                       v290.m128i_i64[1]);
                              v236 = v235;
                              v237 = *(__int128 **)(v355 + 32);
                              v377 = v237;
                              v238 = sub_1D309E0(
                                       *a1,
                                       162,
                                       (__int64)&v389,
                                       v297,
                                       v41,
                                       0,
                                       *(double *)a3.m128_u64,
                                       *(double *)v10.m128i_i64,
                                       *(double *)v11.m128i_i64,
                                       *v237);
                              v91 = sub_1D3A900(
                                      v352,
                                      v342,
                                      (__int64)&v389,
                                      v297,
                                      v41,
                                      v333,
                                      a3,
                                      *(double *)v10.m128i_i64,
                                      v11,
                                      v238,
                                      v239,
                                      *(__int128 *)((char *)v377 + 40),
                                      (__int64)v234,
                                      v236);
                              goto LABEL_88;
                            }
                            v206 = *(_QWORD *)(v367 + 32);
                            v224 = *(_QWORD *)(v206 + 80);
                            if ( *(_WORD *)(v224 + 24) == 78 )
                            {
                              v225 = *(_QWORD *)(v367 + 48);
                              if ( v225 )
                              {
                                if ( !*(_QWORD *)(v225 + 32) )
                                {
                                  v257 = *(_QWORD *)(v224 + 48);
                                  if ( v257 )
                                  {
                                    if ( !*(_QWORD *)(v257 + 32) )
                                    {
                                      v258 = *a1;
                                      v259 = v292;
                                      v366 = v52;
                                      LOBYTE(v259) = v40;
                                      v260 = sub_1D309E0(
                                               *a1,
                                               162,
                                               (__int64)&v389,
                                               v259,
                                               v41,
                                               0,
                                               *(double *)a3.m128_u64,
                                               *(double *)v10.m128i_i64,
                                               *(double *)v11.m128i_i64,
                                               v346);
                                      v262 = v261;
                                      v263 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v367 + 32) + 80LL) + 32LL);
                                      v264 = sub_1D3A900(
                                               v258,
                                               v366,
                                               (__int64)&v389,
                                               v259,
                                               v41,
                                               v333,
                                               a3,
                                               *(double *)v10.m128i_i64,
                                               v11,
                                               *(_QWORD *)v263,
                                               *(__int16 **)(v263 + 8),
                                               *(_OWORD *)(v263 + 40),
                                               v260,
                                               v262);
                                      v92 = (__int64)sub_1D3A900(
                                                       v258,
                                                       v366,
                                                       (__int64)&v389,
                                                       v259,
                                                       v41,
                                                       v333,
                                                       a3,
                                                       *(double *)v10.m128i_i64,
                                                       v11,
                                                       **(_QWORD **)(v367 + 32),
                                                       *(__int16 **)(*(_QWORD *)(v367 + 32) + 8LL),
                                                       *(_OWORD *)(*(_QWORD *)(v367 + 32) + 40LL),
                                                       (__int64)v264,
                                                       v265);
LABEL_89:
                                      if ( v389 )
                                        sub_161E7C0((__int64)&v389, v389);
                                      if ( v92 )
                                      {
                                        v25 = v92;
                                        sub_1F81BC0((__int64)a1, v92);
                                        goto LABEL_8;
                                      }
LABEL_55:
                                      v25 = 0;
                                      goto LABEL_8;
                                    }
                                  }
                                }
                              }
                              if ( v52 != *(unsigned __int16 *)(v355 + 24) )
                                goto LABEL_164;
                            }
                            else if ( v52 != *(unsigned __int16 *)(v355 + 24) )
                            {
                              goto LABEL_190;
                            }
                            v226 = *(_QWORD *)(*(_QWORD *)(v355 + 32) + 80LL);
                            if ( *(_WORD *)(v226 + 24) == 78 )
                              goto LABEL_201;
                          }
                          else
                          {
                            if ( v52 != v59 )
                              goto LABEL_164;
                            v206 = *(_QWORD *)(v367 + 32);
                          }
LABEL_190:
                          v207 = *(_QWORD *)(v206 + 80);
                          if ( *(_WORD *)(v207 + 24) == 157 )
                          {
                            v208 = *(unsigned int **)(v207 + 32);
                            v209 = *(_QWORD *)v208;
                            if ( *(_WORD *)(*(_QWORD *)v208 + 24LL) == 78 )
                            {
                              v210 = a1[1];
                              v211 = (unsigned __int8 *)(*(_QWORD *)(v209 + 40) + 16LL * v208[2]);
                              v212 = *v211;
                              v213 = *((_QWORD *)v211 + 1);
                              v214 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64, unsigned int, __int64))(*v210 + 888);
                              if ( v214 == sub_1F3CF40 )
                              {
                                v215 = *(__int64 (**)())(*v210 + 880);
                                if ( v215 == sub_1D5A410 )
                                  goto LABEL_164;
                                v327 = v52;
                                v216 = ((__int64 (__fastcall *)(__int64 *, _QWORD, const void **, _QWORD, __int64))v215)(
                                         v210,
                                         v40,
                                         v41,
                                         (unsigned __int8)v212,
                                         v213);
                                v52 = v327;
                              }
                              else
                              {
                                v268 = v292;
                                v329 = v52;
                                LOBYTE(v268) = v40;
                                v292 = v268;
                                v216 = v214((__int64)v210, v52, v268, (__int64)v41, v212, v213);
                                v52 = v329;
                              }
                              if ( v216 )
                              {
                                v217 = v292;
                                v328 = v52;
                                LOBYTE(v217) = v40;
                                v341 = *a1;
                                *(_QWORD *)&v218 = sub_1D309E0(
                                                     *a1,
                                                     162,
                                                     (__int64)&v389,
                                                     v217,
                                                     v41,
                                                     0,
                                                     *(double *)a3.m128_u64,
                                                     *(double *)v10.m128i_i64,
                                                     *(double *)v11.m128i_i64,
                                                     v346);
                                v363 = v218;
                                *(_QWORD *)&v219 = sub_1D309E0(
                                                     *a1,
                                                     157,
                                                     (__int64)&v389,
                                                     v217,
                                                     v41,
                                                     0,
                                                     *(double *)a3.m128_u64,
                                                     *(double *)v10.m128i_i64,
                                                     *(double *)v11.m128i_i64,
                                                     *(_OWORD *)(*(_QWORD *)(v209 + 32) + 40LL));
                                v350 = v219;
                                v220 = sub_1D309E0(
                                         *a1,
                                         157,
                                         (__int64)&v389,
                                         v217,
                                         v41,
                                         0,
                                         *(double *)a3.m128_u64,
                                         *(double *)v10.m128i_i64,
                                         *(double *)v11.m128i_i64,
                                         *(_OWORD *)*(_QWORD *)(v209 + 32));
                                v222 = sub_1D3A900(
                                         v341,
                                         v328,
                                         (__int64)&v389,
                                         v217,
                                         v41,
                                         v333,
                                         a3,
                                         *(double *)v10.m128i_i64,
                                         v11,
                                         v220,
                                         v221,
                                         v350,
                                         v363,
                                         *((__int64 *)&v363 + 1));
                                v119 = sub_1D3A900(
                                         v341,
                                         v328,
                                         (__int64)&v389,
                                         v217,
                                         v41,
                                         v333,
                                         a3,
                                         *(double *)v10.m128i_i64,
                                         v11,
                                         **(_QWORD **)(v367 + 32),
                                         *(__int16 **)(*(_QWORD *)(v367 + 32) + 8LL),
                                         *(_OWORD *)(*(_QWORD *)(v367 + 32) + 40LL),
                                         (__int64)v222,
                                         v223);
                                goto LABEL_123;
                              }
                              v60 = *(_WORD *)(v367 + 24);
                            }
                          }
LABEL_164:
                          if ( v60 != 157 )
                            goto LABEL_165;
                          goto LABEL_177;
                        }
LABEL_155:
                        v360 = v52;
                        v161 = v292;
                        v162 = *a1;
                        LOBYTE(v161) = v40;
                        v296 = v161;
                        *(_QWORD *)&v163 = sub_1D309E0(
                                             *a1,
                                             157,
                                             (__int64)&v389,
                                             v161,
                                             v41,
                                             0,
                                             *(double *)a3.m128_u64,
                                             *(double *)v10.m128i_i64,
                                             *(double *)v11.m128i_i64,
                                             *(_OWORD *)(*(_QWORD *)(v153 + 32) + 40LL));
                        v374 = v163;
                        v164 = sub_1D309E0(
                                 *a1,
                                 157,
                                 (__int64)&v389,
                                 v296,
                                 v41,
                                 0,
                                 *(double *)a3.m128_u64,
                                 *(double *)v10.m128i_i64,
                                 *(double *)v11.m128i_i64,
                                 *(_OWORD *)*(_QWORD *)(v153 + 32));
                        *(_QWORD *)&v166 = sub_1D3A900(
                                             v162,
                                             v360,
                                             (__int64)&v389,
                                             v296,
                                             v41,
                                             v333,
                                             a3,
                                             *(double *)v10.m128i_i64,
                                             v11,
                                             v164,
                                             v165,
                                             v374,
                                             v346,
                                             *((__int64 *)&v346 + 1));
                        v92 = sub_1D309E0(
                                v162,
                                162,
                                (__int64)&v389,
                                v296,
                                v41,
                                0,
                                *(double *)a3.m128_u64,
                                *(double *)v10.m128i_i64,
                                *(double *)v11.m128i_i64,
                                v166);
                        goto LABEL_89;
                      }
                      v250 = *a1;
                      v251 = v292;
                      v353 = v52;
                      LOBYTE(v251) = v40;
                      v299 = v251;
                      *(_QWORD *)&v252 = sub_1D309E0(
                                           *a1,
                                           157,
                                           (__int64)&v389,
                                           v251,
                                           v41,
                                           0,
                                           *(double *)a3.m128_u64,
                                           *(double *)v10.m128i_i64,
                                           *(double *)v11.m128i_i64,
                                           *(_OWORD *)(*(_QWORD *)(v143 + 32) + 40LL));
                      v379 = v252;
                      v365 = *a1;
                      *(_QWORD *)&v253 = sub_1D309E0(
                                           *a1,
                                           157,
                                           (__int64)&v389,
                                           v299,
                                           v41,
                                           0,
                                           *(double *)a3.m128_u64,
                                           *(double *)v10.m128i_i64,
                                           *(double *)v11.m128i_i64,
                                           *(_OWORD *)*(_QWORD *)(v143 + 32));
                      v254 = sub_1D309E0(
                               v365,
                               162,
                               (__int64)&v389,
                               v299,
                               v41,
                               0,
                               *(double *)a3.m128_u64,
                               *(double *)v10.m128i_i64,
                               *(double *)v11.m128i_i64,
                               v253);
                      v247 = sub_1D3A900(
                               v250,
                               v353,
                               (__int64)&v389,
                               v299,
                               v41,
                               v333,
                               a3,
                               *(double *)v10.m128i_i64,
                               v11,
                               v254,
                               v255,
                               v379,
                               v338.m128i_i64[0],
                               v338.m128i_i64[1]);
LABEL_211:
                      v92 = (__int64)v247;
                      goto LABEL_89;
                    }
                    goto LABEL_143;
                  }
                  if ( !v107 )
                  {
                    v110 = *(_QWORD *)(v367 + 48);
                    if ( !v110 )
                      goto LABEL_70;
                    if ( *(_QWORD *)(v110 + 32) )
                      goto LABEL_70;
                    v310 = v52;
                    v111 = sub_1D18C00(*(_QWORD *)v109, 1, *(_DWORD *)(v109 + 8));
                    v52 = v310;
                    if ( !v111 )
                      goto LABEL_70;
                    v109 = *(_QWORD *)(v367 + 32);
                  }
                  v339 = v52;
                  v112 = v292;
                  v291 = v346;
                  v372 = (__int128)_mm_loadu_si128((const __m128i *)*(_QWORD *)(*(_QWORD *)v109 + 32LL));
                  LOBYTE(v112) = v40;
                  v348 = *a1;
                  v294 = v112;
                  v359 = (__int128)_mm_loadu_si128((const __m128i *)(*(_QWORD *)(*(_QWORD *)v109 + 32LL) + 40LL));
                  v113 = sub_1D309E0(
                           *a1,
                           162,
                           (__int64)&v389,
                           v112,
                           v41,
                           0,
                           *(double *)a3.m128_u64,
                           *(double *)v10.m128i_i64,
                           *(double *)v11.m128i_i64,
                           v291);
                  v115 = v114;
                  v116 = (__int64 *)v113;
                  v117 = sub_1D309E0(
                           *a1,
                           162,
                           (__int64)&v389,
                           v294,
                           v41,
                           0,
                           *(double *)a3.m128_u64,
                           *(double *)v10.m128i_i64,
                           *(double *)v11.m128i_i64,
                           v372);
LABEL_122:
                  v119 = sub_1D3A900(
                           v348,
                           v339,
                           (__int64)&v389,
                           v294,
                           v41,
                           v333,
                           a3,
                           *(double *)v10.m128i_i64,
                           v11,
                           v117,
                           v118,
                           v359,
                           (__int64)v116,
                           v115);
LABEL_123:
                  v92 = (__int64)v119;
                  goto LABEL_89;
                }
                if ( v107 )
                  goto LABEL_131;
              }
              v107 = 0;
              v108 = *(_QWORD *)(v355 + 48);
              if ( v108 && !*(_QWORD *)(v108 + 32) )
                goto LABEL_131;
              goto LABEL_114;
            }
            goto LABEL_126;
          }
        }
        v51 = *(__int64 (**)())(**(_QWORD **)((*a1)[4] + 16) + 64LL);
        if ( v51 == sub_1D12D30
          || (v311 = v49, v120 = v51(), v49 = v311, (v121 = v120) == 0)
          || (v122 = *(__int64 (**)())(*(_QWORD *)v120 + 88LL), v122 == sub_1F6BBD0) )
        {
LABEL_64:
          v52 = v49 + 99;
          goto LABEL_65;
        }
LABEL_126:
        v312 = v49;
        v123 = ((__int64 (__fastcall *)(__int64, _QWORD))v122)(v121, *((unsigned int *)a1 + 5));
        v49 = v312;
        if ( v123 )
          goto LABEL_53;
        goto LABEL_64;
      }
    }
    else
    {
      v46 = *(__int64 (**)())(*v43 + 920);
      if ( v46 == sub_1F3CBF0 )
        goto LABEL_53;
      v292 = v40;
      if ( !((unsigned __int8 (__fastcall *)(__int64 *, _QWORD, const void **))v46)(v43, v40, v41) )
        goto LABEL_53;
      v49 = 0;
    }
    if ( !*((_BYTE *)a1 + 24) )
      goto LABEL_61;
    v105 = a1[1];
    v106 = 1;
    if ( v40 == 1 || v40 && (v106 = v40, v105[v40 + 15]) )
    {
      if ( (*((_BYTE *)v105 + 259 * v106 + 2521) & 0xFB) == 0 )
        goto LABEL_61;
    }
    goto LABEL_60;
  }
  v93 = *(_QWORD *)(v308 + 32);
  v94 = *(_QWORD *)v93;
  v95 = *(_QWORD *)(v93 + 40);
  v96 = *(_DWORD *)(v93 + 8);
  v371 = *(_QWORD *)(v93 + 8);
  v97 = *(_DWORD *)(v93 + 48);
  v98 = *(_QWORD *)v93;
  v99 = *(_QWORD *)(v93 + 48);
  v100 = v95;
  if ( *(_QWORD *)v93 != v303
    || v96 != v304
    || (v334 = *(_QWORD *)v93,
        v340 = *(_DWORD *)(v93 + 8),
        v205 = sub_1F79A30(
                 v95,
                 v97,
                 *((_BYTE *)a1 + 24),
                 (__int64)a1[1],
                 v301,
                 0,
                 *(double *)a3.m128_u64,
                 *(double *)v10.m128i_i64,
                 *(double *)v11.m128i_i64),
        v100 = v95,
        v96 = v340,
        v94 = v334,
        !v205) )
  {
    if ( v100 != v303 )
      goto LABEL_46;
    if ( v97 != v304 )
      goto LABEL_46;
    v358 = v94;
    v101 = sub_1F79A30(
             v98,
             v96,
             *((_BYTE *)a1 + 24),
             (__int64)a1[1],
             v301,
             0,
             *(double *)a3.m128_u64,
             *(double *)v10.m128i_i64,
             *(double *)v11.m128i_i64);
    v102 = v358;
    if ( !v101 )
      goto LABEL_46;
    v103 = *((_BYTE *)a1 + 24);
    v104 = v371;
LABEL_102:
    v24 = sub_1F7A040(v102, v104, *a1, v103, 0, *(double *)a3.m128_u64, *(double *)v10.m128i_i64, v11);
    goto LABEL_7;
  }
  v25 = (__int64)sub_1F7A040(
                   v95,
                   v99,
                   *a1,
                   *((_BYTE *)a1 + 24),
                   0,
                   *(double *)a3.m128_u64,
                   *(double *)v10.m128i_i64,
                   v11);
LABEL_8:
  if ( v387 )
    sub_161E7C0((__int64)&v387, v387);
  return v25;
}
