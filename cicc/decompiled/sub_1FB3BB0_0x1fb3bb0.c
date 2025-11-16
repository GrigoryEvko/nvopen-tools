// Function: sub_1FB3BB0
// Address: 0x1fb3bb0
//
__int64 __fastcall sub_1FB3BB0(__int64 **a1, _QWORD *a2, double a3, __m128i a4, __m128i a5)
{
  __int64 **v5; // r15
  __int64 v6; // rbx
  const __m128i *v7; // roff
  __m128 v8; // xmm0
  __int64 v9; // r12
  __int64 v10; // r13
  __int64 v11; // rax
  __int8 v12; // dl
  __int64 v13; // rax
  __int64 v14; // r13
  __int64 *v15; // rax
  __int8 *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 result; // rax
  __int64 *v20; // r14
  __int64 v21; // rsi
  __int128 *v22; // r12
  __int32 v23; // eax
  __int64 v24; // rdx
  __int64 v25; // rsi
  __int64 v26; // rsi
  __int64 v27; // rdi
  __int64 v28; // rsi
  __int64 *v29; // r14
  __int64 v30; // rax
  __int64 v31; // r14
  unsigned __int16 v32; // r14
  unsigned __int16 v33; // dx
  unsigned int *v34; // r13
  __m128i v35; // xmm1
  __int64 v36; // rdx
  __int64 v37; // rax
  __int8 v38; // r8
  __int64 v39; // rax
  char v40; // r9
  const void **v41; // rdx
  __m128i v42; // xmm3
  const void **v43; // rax
  int v44; // eax
  char v45; // r8
  char v46; // r9
  __int64 v47; // rdx
  unsigned int v48; // eax
  __int64 v49; // rsi
  __int64 *v50; // r15
  __int64 v51; // rsi
  __int32 v52; // eax
  int v53; // eax
  __int64 v54; // rax
  __int64 v55; // rax
  char v56; // r8
  __int64 *v57; // rdi
  unsigned __int16 v58; // r13
  bool (__fastcall *v59)(__int64, __int64, unsigned __int8); // rax
  bool v60; // al
  __int64 v61; // rcx
  __int64 v62; // r8
  __int64 v63; // r9
  __int64 v64; // r13
  __int64 v65; // rbx
  __int64 v66; // r14
  __int64 v67; // rax
  __int64 v68; // rdx
  __int64 v69; // rax
  __int64 v70; // rax
  __int8 v71; // dl
  __int64 v72; // rax
  unsigned int v73; // r15d
  unsigned __int8 v74; // al
  unsigned int v75; // r12d
  char v76; // al
  __int64 v77; // r10
  __int64 v78; // rax
  __int64 *v79; // rax
  __int64 v80; // rax
  int v81; // edx
  __int64 v82; // rdx
  unsigned __int64 v83; // rdi
  __m128i v84; // xmm4
  char v85; // r8
  unsigned int v86; // r12d
  unsigned int v87; // eax
  __int64 v88; // rsi
  __int64 *v89; // r12
  __m128i *v90; // r10
  __int32 v91; // eax
  unsigned int v92; // eax
  __int64 v93; // rax
  __int64 v94; // rsi
  __int128 *v95; // r14
  __int64 *v96; // r13
  __int32 v97; // eax
  __m128i v98; // rax
  __m128i *v99; // r10
  __int64 v100; // rsi
  __int64 v101; // r13
  __int64 *v102; // r14
  __int32 v103; // eax
  __int64 v104; // r13
  __int64 v105; // r10
  __int64 v106; // rdx
  __int64 v107; // r14
  __int64 v108; // rsi
  __int64 *v109; // r15
  __int64 v110; // rsi
  unsigned int v111; // r8d
  __int64 v112; // r13
  char v113; // di
  __int64 v114; // rax
  unsigned int v115; // eax
  unsigned int v116; // r8d
  int v117; // r9d
  unsigned __int64 v118; // rax
  __int128 v119; // rax
  __int128 v120; // kr00_16
  __int64 v121; // rsi
  __int64 *v122; // r14
  __int32 v123; // eax
  __int64 v124; // rdx
  __int64 *v125; // rdi
  unsigned __int8 *v126; // rdx
  __int64 v127; // rax
  __int64 v128; // rdx
  __int64 v129; // rsi
  unsigned int v130; // ecx
  __int64 *v131; // rdi
  unsigned int v132; // edx
  __int64 v133; // rax
  bool v134; // al
  __int8 v135; // r8
  __int64 *v136; // rcx
  __int64 v137; // rdx
  __int64 v138; // rdx
  __int64 (*v139)(); // rax
  __int64 v140; // rsi
  __int64 *v141; // rdi
  __int64 v142; // rax
  __int128 v143; // rax
  __int64 *v144; // rdi
  __int64 v145; // rax
  __int64 *v146; // r15
  __int64 v147; // r12
  __int64 v148; // rdx
  __int64 v149; // r13
  __int64 v150; // rdx
  bool (__fastcall *v151)(__int64, __int64, unsigned __int8); // rax
  __int64 v152; // rax
  __m128i v153; // xmm6
  __int64 v154; // rcx
  __int64 *v155; // rdi
  __int64 v156; // rdx
  __int64 v157; // rcx
  __int64 v158; // r8
  __int64 v159; // r9
  unsigned int v160; // eax
  __int32 v161; // r14d
  unsigned int v162; // edx
  unsigned __int64 v163; // rax
  int v164; // eax
  unsigned int v165; // r14d
  int v166; // eax
  __int64 v167; // r13
  __int64 v168; // rax
  __int64 v169; // rax
  __int64 *v170; // rdi
  __int64 v171; // r11
  __int128 *v172; // rax
  const void **v173; // rdx
  const void **v174; // rbx
  __int64 v175; // r12
  __int64 v176; // rax
  unsigned __int64 v177; // rdx
  unsigned __int64 v178; // r13
  __m128i *v179; // r10
  __int64 v180; // rdx
  __int64 *v181; // rdi
  __int64 v182; // rdx
  unsigned __int16 v183; // ax
  __int32 v184; // eax
  __int64 *v185; // rdi
  __int64 v186; // rax
  __int64 *v187; // rdi
  __int64 v188; // r13
  __int64 v189; // rdx
  __int64 v190; // r14
  __m128i v191; // rax
  __int64 *v192; // rdi
  __int64 v193; // rax
  __int64 v194; // rsi
  const void ***v195; // rcx
  __int64 v196; // rax
  int v197; // edx
  __int64 v198; // r9
  __int64 v199; // rdx
  __int64 *v200; // rax
  __int64 v201; // rdx
  __int64 v202; // rax
  __int64 v203; // rax
  char v204; // dl
  const void **v205; // rax
  char v206; // al
  __int64 v207; // rdx
  __int32 v208; // eax
  __int64 *v209; // r12
  __int64 (__fastcall *v210)(__int64 *, __int64); // rbx
  __int64 v211; // rax
  unsigned __int8 v212; // al
  __m128i *v213; // r10
  unsigned int v214; // ebx
  unsigned int v215; // eax
  int v216; // eax
  __int64 *v217; // r14
  __int64 *v218; // rdi
  __int128 v219; // rax
  __int64 v220; // rdx
  bool v221; // al
  __int64 v222; // r14
  __int64 v223; // rax
  __int64 v224; // rax
  __int64 v225; // rdx
  __int64 v226; // r13
  __int64 v227; // rax
  __int64 v228; // rax
  char v229; // al
  __int64 v230; // rdx
  __int64 v231; // rbx
  __int64 *v232; // r13
  __int64 *v233; // r15
  __int64 v234; // rdx
  __int64 v235; // rax
  char v236; // dl
  __int64 v237; // rax
  char *v238; // rax
  char v239; // dl
  __int64 v240; // rax
  const void **v241; // r9
  __int64 v242; // rdx
  __int64 v243; // rcx
  __int64 v244; // r8
  __int64 v245; // r9
  __int64 v246; // rdx
  __int64 v247; // rcx
  __int64 v248; // r8
  __int64 v249; // r9
  unsigned int v250; // r14d
  const void **v251; // rdx
  __int64 v252; // rax
  int v253; // edx
  bool v254; // al
  __int64 v255; // rcx
  __int64 v256; // rdx
  int v257; // r13d
  __int64 v258; // rdx
  __int64 v259; // rcx
  __int64 v260; // r8
  __int64 v261; // r9
  unsigned int v262; // eax
  __m128i *v263; // r10
  __int64 v264; // r13
  __int64 v265; // r9
  __int64 v266; // rbx
  __int64 *v267; // r14
  __int64 v268; // r8
  __int32 v269; // eax
  __int64 v270; // rbx
  __int64 v271; // r8
  __int64 v272; // rax
  unsigned int v273; // edx
  unsigned int v274; // ebx
  __int64 i; // r13
  __int64 v276; // rcx
  __int64 v277; // r8
  __m128i v278; // rax
  int v279; // r8d
  __int64 v280; // rax
  __int64 *v281; // r12
  __int64 v282; // rax
  const void ***v283; // rcx
  __int64 v284; // rsi
  __int64 v285; // rax
  __int64 v286; // rdx
  __int64 v287; // r12
  int v288; // r8d
  __int64 v289; // rax
  __int64 v290; // rcx
  __int64 *v291; // rax
  __int64 *v292; // r15
  __int64 v293; // r12
  __int64 v294; // r13
  __int64 v295; // rdx
  unsigned int *v296; // rax
  __int64 v297; // rcx
  __int64 v298; // rax
  __int8 v299; // dl
  __int64 v300; // rax
  char v301; // al
  __int64 v302; // rdx
  __int64 v303; // rdx
  unsigned int v304; // r14d
  unsigned int v305; // r12d
  __int64 v306; // rcx
  unsigned int v307; // r8d
  int v308; // r9d
  __int64 v309; // rdx
  int v310; // r13d
  __int64 v311; // rsi
  __int64 *v312; // r15
  __int64 v313; // r13
  __int64 v314; // r12
  __int64 v315; // rdx
  __int64 *v316; // r8
  __int64 v317; // rax
  _QWORD *v318; // r13
  __int64 v319; // rax
  __int64 v320; // rdi
  __int64 v321; // rax
  __int64 v322; // rcx
  signed int v323; // edx
  __int64 *v324; // r13
  __int32 v325; // eax
  __int64 v326; // rdi
  __m128i v327; // rax
  __int64 *v328; // rdi
  __int64 v329; // rax
  unsigned __int64 v330; // rdx
  __int64 v331; // rdx
  __int128 v332; // [rsp-30h] [rbp-250h]
  __m128i v333; // [rsp-20h] [rbp-240h]
  __int128 v334; // [rsp-20h] [rbp-240h]
  __int64 v335; // [rsp-20h] [rbp-240h]
  unsigned __int64 v336; // [rsp-18h] [rbp-238h]
  __int128 v337; // [rsp-10h] [rbp-230h]
  __int128 v338; // [rsp-10h] [rbp-230h]
  __int128 v339; // [rsp-10h] [rbp-230h]
  __int128 v340; // [rsp-10h] [rbp-230h]
  __int128 v341; // [rsp-10h] [rbp-230h]
  __int128 v342; // [rsp-10h] [rbp-230h]
  int v343; // [rsp+0h] [rbp-220h]
  int v344; // [rsp+0h] [rbp-220h]
  int v345; // [rsp+0h] [rbp-220h]
  __int64 v346; // [rsp+8h] [rbp-218h]
  __int64 v347; // [rsp+8h] [rbp-218h]
  __int64 v348; // [rsp+8h] [rbp-218h]
  int v349; // [rsp+10h] [rbp-210h]
  __int64 v350; // [rsp+10h] [rbp-210h]
  int v351; // [rsp+10h] [rbp-210h]
  int v352; // [rsp+10h] [rbp-210h]
  __int64 v353; // [rsp+18h] [rbp-208h]
  __int64 v354; // [rsp+18h] [rbp-208h]
  __int64 v355; // [rsp+18h] [rbp-208h]
  unsigned __int64 v356; // [rsp+28h] [rbp-1F8h]
  unsigned int v357; // [rsp+38h] [rbp-1E8h]
  int v358; // [rsp+3Ch] [rbp-1E4h]
  __int64 v359; // [rsp+40h] [rbp-1E0h]
  unsigned int v360; // [rsp+40h] [rbp-1E0h]
  __int64 v361; // [rsp+48h] [rbp-1D8h]
  __int64 v362; // [rsp+50h] [rbp-1D0h]
  unsigned int v363; // [rsp+50h] [rbp-1D0h]
  __int64 v364; // [rsp+50h] [rbp-1D0h]
  char v365; // [rsp+50h] [rbp-1D0h]
  __int64 v366; // [rsp+58h] [rbp-1C8h]
  __int64 v367; // [rsp+58h] [rbp-1C8h]
  int v368; // [rsp+58h] [rbp-1C8h]
  const void **v369; // [rsp+58h] [rbp-1C8h]
  __int64 v370; // [rsp+58h] [rbp-1C8h]
  char v371; // [rsp+60h] [rbp-1C0h]
  _QWORD *v372; // [rsp+60h] [rbp-1C0h]
  char v373; // [rsp+60h] [rbp-1C0h]
  unsigned __int16 v374; // [rsp+60h] [rbp-1C0h]
  unsigned int v375; // [rsp+60h] [rbp-1C0h]
  unsigned int v376; // [rsp+60h] [rbp-1C0h]
  __int64 v377; // [rsp+60h] [rbp-1C0h]
  unsigned int v378; // [rsp+60h] [rbp-1C0h]
  unsigned int v379; // [rsp+60h] [rbp-1C0h]
  __int8 v380; // [rsp+60h] [rbp-1C0h]
  unsigned __int8 v381; // [rsp+60h] [rbp-1C0h]
  __int64 v382; // [rsp+60h] [rbp-1C0h]
  __m128i v383; // [rsp+70h] [rbp-1B0h] BYREF
  __int128 v384; // [rsp+80h] [rbp-1A0h]
  __m128i v385; // [rsp+90h] [rbp-190h] BYREF
  __m128i v386; // [rsp+A0h] [rbp-180h] BYREF
  char v387[8]; // [rsp+B0h] [rbp-170h] BYREF
  __int64 v388; // [rsp+B8h] [rbp-168h]
  __int64 v389; // [rsp+C0h] [rbp-160h] BYREF
  __int64 v390; // [rsp+C8h] [rbp-158h]
  unsigned __int64 v391; // [rsp+D0h] [rbp-150h] BYREF
  __int64 v392; // [rsp+D8h] [rbp-148h]
  _BYTE v393[128]; // [rsp+E0h] [rbp-140h] BYREF
  __m128i v394; // [rsp+160h] [rbp-C0h] BYREF
  __int64 v395[22]; // [rsp+170h] [rbp-B0h] BYREF

  v5 = a1;
  v6 = (__int64)a2;
  v7 = (const __m128i *)a2[4];
  v8 = (__m128)_mm_loadu_si128(v7);
  v9 = v7->m128i_i64[0];
  v10 = v7->m128i_u32[2];
  v11 = a2[5];
  v385 = (__m128i)v8;
  v12 = *(_BYTE *)v11;
  v13 = *(_QWORD *)(v11 + 8);
  LODWORD(v384) = v10;
  v14 = 16 * v10;
  v386.m128i_i64[1] = v13;
  v15 = *a1;
  v386.m128i_i8[0] = v12;
  v16 = (__int8 *)sub_1E0A0C0(v15[4]);
  v17 = a2[5];
  v18 = v14 + *(_QWORD *)(v9 + 40);
  if ( *(_BYTE *)v17 == *(_BYTE *)v18 && (*(_QWORD *)(v18 + 8) == *(_QWORD *)(v17 + 8) || *(_BYTE *)v18) )
    return v385.m128i_i64[0];
  v20 = *a1;
  if ( *(_WORD *)(v9 + 24) == 145 )
  {
    v21 = a2[9];
    v22 = *(__int128 **)(v9 + 32);
    v394.m128i_i64[0] = v21;
    if ( v21 )
    {
      v385.m128i_i64[0] = (__int64)&v394;
      sub_1623A60((__int64)&v394, v21, 2);
    }
    v23 = *(_DWORD *)(v6 + 64);
    v385.m128i_i64[0] = (__int64)&v394;
    v394.m128i_i32[2] = v23;
    result = sub_1D309E0(
               v20,
               145,
               (__int64)&v394,
               v386.m128i_u32[0],
               (const void **)v386.m128i_i64[1],
               0,
               *(double *)v8.m128_u64,
               *(double *)a4.m128i_i64,
               *(double *)a5.m128i_i64,
               *v22);
    goto LABEL_10;
  }
  v26 = v385.m128i_i64[0];
  v27 = (__int64)*a1;
  v383.m128i_i8[0] = *v16;
  if ( sub_1D23600(v27, v385.m128i_i64[0]) )
  {
    v28 = *(_QWORD *)(v6 + 72);
    v29 = *v5;
    v394.m128i_i64[0] = v28;
    if ( v28 )
      sub_1623A60((__int64)&v394, v28, 2);
    v394.m128i_i32[2] = *(_DWORD *)(v6 + 64);
    v30 = sub_1D309E0(
            v29,
            145,
            (__int64)&v394,
            v386.m128i_u32[0],
            (const void **)v386.m128i_i64[1],
            0,
            *(double *)v8.m128_u64,
            *(double *)a4.m128i_i64,
            *(double *)a5.m128i_i64,
            *(_OWORD *)&v385);
    v26 = v394.m128i_i64[0];
    v31 = v30;
    if ( v394.m128i_i64[0] )
      sub_161E7C0((__int64)&v394, v394.m128i_i64[0]);
    result = v31;
    if ( v31 != v6 )
      return result;
  }
  v32 = *(_WORD *)(v9 + 24);
  v33 = v32;
  if ( (unsigned __int16)(v32 - 142) > 2u )
  {
    v54 = *(_QWORD *)(v6 + 48);
    if ( v54 && !*(_QWORD *)(v54 + 32) && *(_WORD *)(*(_QWORD *)(v54 + 16) + 24LL) == 144 )
      return 0;
    v383.m128i_i8[0] ^= 1u;
    if ( v32 == 106 )
    {
      if ( !*((_BYTE *)v5 + 25)
        || *((_BYTE *)v5 + 24)
        || (v93 = *(_QWORD *)(v9 + 48)) == 0
        || *(_QWORD *)(v93 + 32)
        || v386.m128i_i8[0] == 2 )
      {
        if ( *((_DWORD *)v5 + 4) != 2 )
          goto LABEL_44;
        goto LABEL_102;
      }
      v235 = *(_QWORD *)(**(_QWORD **)(v9 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v9 + 32) + 8LL);
      v236 = *(_BYTE *)v235;
      v388 = *(_QWORD *)(v235 + 8);
      v237 = *(_QWORD *)(v9 + 40);
      v387[0] = v236;
      v238 = (char *)(v14 + v237);
      v239 = *v238;
      v390 = *((_QWORD *)v238 + 1);
      v240 = *(_QWORD *)(v6 + 40);
      LOBYTE(v389) = v239;
      v241 = *(const void ***)(v240 + 8);
      LOBYTE(v240) = *(_BYTE *)v240;
      v364 = (__int64)v241;
      v392 = (__int64)v241;
      LOBYTE(v391) = v240;
      v368 = sub_1D15970(v387);
      v378 = sub_1D159A0((char *)&v389, v26, v242, v243, v244, v245, v343, v346, v349, v353);
      v379 = v378 / (unsigned int)sub_1D159A0((char *)&v391, v26, v246, v247, v248, v249, v344, v347, v351, v354);
      v250 = sub_1F7DEB0((_QWORD *)(*v5)[6], v391, v364, v379 * v368, 0);
      v369 = v251;
      v252 = *(_QWORD *)(*(_QWORD *)(v9 + 32) + 40LL);
      v253 = *(unsigned __int16 *)(v252 + 24);
      if ( v253 == 32 || v253 == 10 )
      {
        v316 = v5[1];
        if ( !*((_BYTE *)v5 + 25) || (_BYTE)v250 && v316[(unsigned __int8)v250 + 15] )
        {
          v317 = *(_QWORD *)(v252 + 88);
          v318 = *(_QWORD **)(v317 + 24);
          if ( *(_DWORD *)(v317 + 32) > 0x40u )
            v318 = (_QWORD *)*v318;
          v319 = *v316;
          *(_QWORD *)&v384 = v5[1];
          v320 = (*v5)[4];
          v385.m128i_i64[0] = *(_QWORD *)(v319 + 48);
          v321 = sub_1E0A0C0(v320);
          v322 = ((__int64 (__fastcall *)(_QWORD, __int64))v385.m128i_i64[0])(v384, v321);
          if ( v383.m128i_i8[0] )
            v323 = (_DWORD)v318 * v379;
          else
            v323 = ((_DWORD)v318 + 1) * v379 - 1;
          v394.m128i_i64[0] = *(_QWORD *)(v6 + 72);
          if ( v394.m128i_i64[0] )
          {
            v383.m128i_i64[0] = v322;
            LODWORD(v384) = v323;
            v385.m128i_i64[0] = (__int64)&v394;
            sub_1F6CA20(v394.m128i_i64);
            v322 = v383.m128i_i64[0];
            v323 = v384;
          }
          v324 = *v5;
          v325 = *(_DWORD *)(v6 + 64);
          v326 = (__int64)*v5;
          *(_QWORD *)&v384 = &v394;
          v394.m128i_i32[2] = v325;
          v327.m128i_i64[0] = sub_1D38BB0(
                                v326,
                                v323,
                                (__int64)&v394,
                                v322,
                                0,
                                0,
                                (__m128i)v8,
                                *(double *)a4.m128i_i64,
                                a5,
                                0);
          v328 = *v5;
          v385 = v327;
          v329 = sub_1D32840(
                   v328,
                   v250,
                   v369,
                   **(_QWORD **)(v9 + 32),
                   *(_QWORD *)(*(_QWORD *)(v9 + 32) + 8LL),
                   *(double *)v8.m128_u64,
                   *(double *)a4.m128i_i64,
                   *(double *)a5.m128i_i64);
          v383.m128i_i64[0] = (__int64)&v394;
          *(_QWORD *)&v384 = sub_1D332F0(
                               v324,
                               106,
                               (__int64)&v394,
                               (unsigned int)v391,
                               (const void **)v392,
                               0,
                               *(double *)v8.m128_u64,
                               *(double *)a4.m128i_i64,
                               a5,
                               v329,
                               v330,
                               *(_OWORD *)&v385);
          v385.m128i_i64[0] = v331;
          goto LABEL_213;
        }
      }
      v33 = *(_WORD *)(v9 + 24);
    }
    if ( v33 == 134 )
    {
      if ( sub_1D18C00(v9, 1, v384) )
      {
        v125 = v5[1];
        v126 = (unsigned __int8 *)(v14 + *(_QWORD *)(v9 + 40));
        v127 = *v126;
        v128 = *((_QWORD *)v126 + 1);
        v129 = (unsigned __int8)v127;
        if ( !*((_BYTE *)v5 + 24)
          || ((v130 = 1, (_BYTE)v127 == 1) || (_BYTE)v127 && (v130 = (unsigned __int8)v127, v125[v127 + 15]))
          && !*((_BYTE *)v125 + 259 * v130 + 2556) )
        {
          v139 = *(__int64 (**)())(*v125 + 800);
          if ( v139 != sub_1D12DF0 )
          {
            if ( ((unsigned __int8 (__fastcall *)(__int64 *, __int64, __int64, _QWORD, __int64))v139)(
                   v125,
                   v129,
                   v128,
                   v386.m128i_u32[0],
                   v386.m128i_i64[1]) )
            {
              v140 = *(_QWORD *)(v9 + 72);
              v391 = v140;
              if ( v140 )
                sub_1623A60((__int64)&v391, v140, 2);
              v141 = *v5;
              LODWORD(v392) = *(_DWORD *)(v9 + 64);
              v142 = *(_QWORD *)(v9 + 32);
              v338 = *(_OWORD *)(v142 + 40);
              v385 = _mm_loadu_si128((const __m128i *)v142);
              *(_QWORD *)&v143 = sub_1D309E0(
                                   v141,
                                   145,
                                   (__int64)&v391,
                                   v386.m128i_u32[0],
                                   (const void **)v386.m128i_i64[1],
                                   0,
                                   *(double *)v8.m128_u64,
                                   *(double *)a4.m128i_i64,
                                   *(double *)a5.m128i_i64,
                                   v338);
              v144 = *v5;
              v384 = v143;
              v145 = sub_1D309E0(
                       v144,
                       145,
                       (__int64)&v391,
                       v386.m128i_u32[0],
                       (const void **)v386.m128i_i64[1],
                       0,
                       *(double *)v8.m128_u64,
                       *(double *)a4.m128i_i64,
                       *(double *)a5.m128i_i64,
                       *(_OWORD *)(*(_QWORD *)(v9 + 32) + 80LL));
              v146 = *v5;
              v147 = v145;
              v149 = v148;
              v394.m128i_i64[0] = *(_QWORD *)(v6 + 72);
              if ( v394.m128i_i64[0] )
              {
                v383.m128i_i64[0] = (__int64)&v394;
                sub_1F6CA20(v394.m128i_i64);
              }
              v394.m128i_i32[2] = *(_DWORD *)(v6 + 64);
              v383.m128i_i64[0] = (__int64)&v394;
              result = (__int64)sub_1D3A900(
                                  v146,
                                  0x86u,
                                  (__int64)&v394,
                                  v386.m128i_u32[0],
                                  (const void **)v386.m128i_i64[1],
                                  0,
                                  v8,
                                  *(double *)a4.m128i_i64,
                                  a5,
                                  v385.m128i_u64[0],
                                  (__int16 *)v385.m128i_i64[1],
                                  v384,
                                  v147,
                                  v149);
              if ( v394.m128i_i64[0] )
              {
                *(_QWORD *)&v384 = v150;
                v385.m128i_i64[0] = result;
                sub_161E7C0(v383.m128i_i64[0], v394.m128i_i64[0]);
                v150 = v384;
                result = v385.m128i_i64[0];
              }
              if ( v391 )
              {
                *(_QWORD *)&v384 = v150;
                v385.m128i_i64[0] = result;
                sub_161E7C0((__int64)&v391, v391);
                return v385.m128i_i64[0];
              }
              return result;
            }
          }
        }
      }
      v33 = *(_WORD *)(v9 + 24);
    }
    if ( v33 != 122 )
    {
LABEL_38:
      if ( *((_DWORD *)v5 + 4) != 2 )
        goto LABEL_39;
LABEL_102:
      v56 = v386.m128i_i8[0];
      if ( v386.m128i_i8[0] )
      {
LABEL_103:
        if ( (unsigned __int8)(v56 - 14) > 0x5Fu )
        {
          if ( (unsigned __int16)(v33 - 118) > 2u )
            goto LABEL_45;
          goto LABEL_40;
        }
        goto LABEL_152;
      }
LABEL_157:
      v374 = v33;
      v134 = sub_1F58D20((__int64)&v386);
      v33 = v374;
      if ( !v134 )
      {
        if ( (unsigned __int16)(v374 - 118) > 2u )
          goto LABEL_122;
        goto LABEL_40;
      }
LABEL_152:
      if ( v33 == 158 )
      {
        if ( sub_1D18C00(v9, 1, v384) )
        {
          v133 = *(_QWORD *)(v9 + 32);
          if ( *(_WORD *)(*(_QWORD *)v133 + 24LL) == 104 && sub_1D18C00(*(_QWORD *)v133, 1, *(_DWORD *)(v133 + 8)) )
          {
            v296 = *(unsigned int **)(v9 + 32);
            v297 = *(_QWORD *)v296;
            v298 = *(_QWORD *)(*(_QWORD *)v296 + 40LL) + 16LL * v296[2];
            v382 = v297;
            v299 = *(_BYTE *)v298;
            v300 = *(_QWORD *)(v298 + 8);
            v394.m128i_i8[0] = v299;
            v394.m128i_i64[1] = v300;
            v301 = sub_1F7E0F0((__int64)&v394);
            v370 = v302;
            v365 = v301;
            if ( v301 == sub_1F7E0F0((__int64)&v386) && (v365 || v370 == v303) )
            {
              v383.m128i_i64[0] = (__int64)&v394;
              v304 = 0;
              v305 = *(_DWORD *)(v382 + 56);
              *(_QWORD *)&v384 = v6;
              v307 = sub_1D15970(&v386);
              v309 = v305 % v307;
              v310 = v305 / v307;
              v385.m128i_i64[0] = (__int64)v395;
              v394.m128i_i64[0] = (__int64)v395;
              v394.m128i_i64[1] = 0x800000000LL;
              while ( v304 != v305 )
              {
                v311 = v304;
                v304 += v310;
                sub_1D23890(
                  (__int64)&v394,
                  (const __m128i *)(*(_QWORD *)(v382 + 32) + 40 * v311),
                  v309,
                  v306,
                  v307,
                  v308);
              }
              v312 = *v5;
              v313 = v394.m128i_i64[0];
              v314 = v394.m128i_u32[2];
              v391 = *(_QWORD *)(v384 + 72);
              if ( v391 )
                sub_1F6CA20((__int64 *)&v391);
              *((_QWORD *)&v342 + 1) = v314;
              *(_QWORD *)&v342 = v313;
              LODWORD(v392) = *(_DWORD *)(v384 + 64);
              v383.m128i_i64[0] = (__int64)sub_1D359D0(
                                             v312,
                                             104,
                                             (__int64)&v391,
                                             v386.m128i_i64[0],
                                             (const void **)v386.m128i_i64[1],
                                             0,
                                             *(double *)v8.m128_u64,
                                             *(double *)a4.m128i_i64,
                                             a5,
                                             v342);
              *(_QWORD *)&v384 = v315;
              sub_17CD270((__int64 *)&v391);
              v83 = v394.m128i_i64[0];
              result = v383.m128i_i64[0];
              if ( v394.m128i_i64[0] == v385.m128i_i64[0] )
                return result;
              goto LABEL_75;
            }
          }
        }
        v33 = *(_WORD *)(v9 + 24);
      }
LABEL_39:
      if ( (unsigned __int16)(v33 - 118) <= 2u )
      {
LABEL_40:
        v55 = *(_QWORD *)(v9 + 48);
        if ( !v55 )
        {
LABEL_109:
          v94 = *(_QWORD *)(v6 + 72);
          v95 = *(__int128 **)(v9 + 32);
          v96 = *v5;
          v394.m128i_i64[0] = v94;
          if ( v94 )
          {
            v385.m128i_i64[0] = (__int64)&v394;
            sub_1623A60((__int64)&v394, v94, 2);
          }
          v97 = *(_DWORD *)(v6 + 64);
          *(_QWORD *)&v384 = &v394;
          v394.m128i_i32[2] = v97;
          v98.m128i_i64[0] = sub_1D309E0(
                               v96,
                               145,
                               (__int64)&v394,
                               v386.m128i_u32[0],
                               (const void **)v386.m128i_i64[1],
                               0,
                               *(double *)v8.m128_u64,
                               *(double *)a4.m128i_i64,
                               *(double *)a5.m128i_i64,
                               *v95);
          v99 = &v394;
          v385 = v98;
          if ( v394.m128i_i64[0] )
          {
            sub_161E7C0(v384, v394.m128i_i64[0]);
            v99 = (__m128i *)v384;
          }
          v100 = *(_QWORD *)(v6 + 72);
          v101 = *(_QWORD *)(v9 + 32);
          v102 = *v5;
          v394.m128i_i64[0] = v100;
          if ( v100 )
          {
            *(_QWORD *)&v384 = v99;
            sub_1623A60((__int64)v99, v100, 2);
            v99 = (__m128i *)v384;
          }
          v103 = *(_DWORD *)(v6 + 64);
          *(_QWORD *)&v384 = v99;
          v394.m128i_i32[2] = v103;
          v104 = sub_1D309E0(
                   v102,
                   145,
                   (__int64)v99,
                   v386.m128i_u32[0],
                   (const void **)v386.m128i_i64[1],
                   0,
                   *(double *)v8.m128_u64,
                   *(double *)a4.m128i_i64,
                   *(double *)a5.m128i_i64,
                   *(_OWORD *)(v101 + 40));
          v105 = v384;
          v107 = v106;
          if ( v394.m128i_i64[0] )
          {
            sub_161E7C0(v384, v394.m128i_i64[0]);
            v105 = v384;
          }
          v108 = *(_QWORD *)(v6 + 72);
          v109 = *v5;
          v394.m128i_i64[0] = v108;
          if ( v108 )
          {
            *(_QWORD *)&v384 = v105;
            sub_1623A60(v105, v108, 2);
            v105 = v384;
          }
          v394.m128i_i32[2] = *(_DWORD *)(v6 + 64);
          v110 = *(unsigned __int16 *)(v9 + 24);
          *((_QWORD *)&v337 + 1) = v107;
          *(_QWORD *)&v337 = v104;
          v333 = v385;
          v385.m128i_i64[0] = v105;
          result = (__int64)sub_1D332F0(
                              v109,
                              v110,
                              v105,
                              v386.m128i_u32[0],
                              (const void **)v386.m128i_i64[1],
                              0,
                              *(double *)v8.m128_u64,
                              *(double *)a4.m128i_i64,
                              a5,
                              v333.m128i_i64[0],
                              v333.m128i_u64[1],
                              v337);
          v25 = v394.m128i_i64[0];
          if ( v394.m128i_i64[0] )
            goto LABEL_11;
          return result;
        }
        while ( *(_WORD *)(*(_QWORD *)(v55 + 16) + 24LL) == 145 )
        {
          v55 = *(_QWORD *)(v55 + 32);
          if ( !v55 )
            goto LABEL_109;
        }
      }
LABEL_44:
      v56 = v386.m128i_i8[0];
      if ( v386.m128i_i8[0] )
      {
LABEL_45:
        if ( (unsigned __int8)(v56 - 14) <= 0x5Fu )
          goto LABEL_46;
        v111 = sub_1F6C8D0(v56);
        goto LABEL_123;
      }
      if ( sub_1F58D20((__int64)&v386) )
        goto LABEL_46;
LABEL_122:
      v111 = sub_1F58D40((__int64)&v386);
LABEL_123:
      v112 = *(_QWORD *)(v9 + 40) + v14;
      v113 = *(_BYTE *)v112;
      v114 = *(_QWORD *)(v112 + 8);
      v394.m128i_i8[0] = v113;
      v394.m128i_i64[1] = v114;
      if ( v113 )
      {
        v115 = sub_1F6C8D0(v113);
      }
      else
      {
        v376 = v111;
        v115 = sub_1F58D40((__int64)&v394);
        v116 = v376;
      }
      LODWORD(v392) = v115;
      if ( v115 > 0x40 )
      {
        v375 = v116;
        sub_16A4EF0((__int64)&v391, 0, 0);
        v116 = v375;
      }
      else
      {
        v391 = 0;
      }
      if ( v116 )
      {
        if ( v116 > 0x40 )
        {
          sub_16A5260(&v391, 0, v116);
        }
        else
        {
          v118 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v116);
          if ( (unsigned int)v392 > 0x40 )
            *(_QWORD *)v391 |= v118;
          else
            v391 |= v118;
        }
      }
      *(_QWORD *)&v119 = sub_1D3CC80(
                           *v5,
                           v385.m128i_i64[0],
                           v385.m128i_i64[1],
                           (__int64)&v391,
                           v116,
                           v117,
                           (__m128i)v8,
                           *(double *)a4.m128i_i64,
                           a5);
      v120 = v119;
      if ( (_QWORD)v119 )
      {
        v121 = *(_QWORD *)(v6 + 72);
        v122 = *v5;
        v394.m128i_i64[0] = v121;
        if ( v121 )
        {
          v385.m128i_i64[0] = (__int64)&v394;
          v384 = v119;
          sub_1623A60((__int64)&v394, v121, 2);
          v120 = v384;
        }
        v123 = *(_DWORD *)(v6 + 64);
        v385.m128i_i64[0] = (__int64)&v394;
        v394.m128i_i32[2] = v123;
        result = sub_1D309E0(
                   v122,
                   145,
                   (__int64)&v394,
                   v386.m128i_u32[0],
                   (const void **)v386.m128i_i64[1],
                   0,
                   *(double *)v8.m128_u64,
                   *(double *)a4.m128i_i64,
                   *(double *)a5.m128i_i64,
                   v120);
        if ( v394.m128i_i64[0] )
        {
          v383.m128i_i64[0] = v124;
          *(_QWORD *)&v384 = result;
          sub_161E7C0(v385.m128i_i64[0], v394.m128i_i64[0]);
          v124 = v383.m128i_i64[0];
          result = v384;
        }
        if ( (unsigned int)v392 > 0x40 && v391 )
        {
          *(_QWORD *)&v384 = v124;
          v385.m128i_i64[0] = result;
          j_j___libc_free_0_0(v391);
          return v385.m128i_i64[0];
        }
        return result;
      }
      if ( (unsigned int)v392 > 0x40 && v391 )
        j_j___libc_free_0_0(v391);
LABEL_46:
      if ( *((_BYTE *)v5 + 25) )
      {
        v57 = v5[1];
        v58 = *(_WORD *)(v9 + 24);
        v59 = *(bool (__fastcall **)(__int64, __int64, unsigned __int8))(*v57 + 1136);
        if ( v59 == sub_1F6BB70 )
        {
          if ( !v386.m128i_i8[0] || !v57[v386.m128i_u8[0] + 15] )
          {
            if ( v58 == 107 )
              goto LABEL_231;
LABEL_161:
            v135 = v386.m128i_i8[0];
            if ( v58 != 158 )
              goto LABEL_162;
            if ( v386.m128i_i8[0] )
            {
              if ( (unsigned __int8)(v386.m128i_i8[0] - 14) > 0x5Fu )
              {
LABEL_216:
                v200 = *(__int64 **)(v9 + 32);
                v201 = *v200;
                v202 = *((unsigned int *)v200 + 2);
                v385 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v9 + 32));
                v203 = *(_QWORD *)(v201 + 40) + 16 * v202;
                v204 = *(_BYTE *)v203;
                v205 = *(const void ***)(v203 + 8);
                LOBYTE(v391) = v204;
                v392 = (__int64)v205;
                if ( v204 )
                {
                  if ( (unsigned __int8)(v204 - 14) > 0x5Fu )
                    goto LABEL_162;
                }
                else
                {
                  v380 = v135;
                  v254 = sub_1F58D20((__int64)&v391);
                  v135 = v380;
                  if ( !v254 )
                    goto LABEL_162;
                }
                v206 = sub_1D15870((char *)&v391);
                if ( v386.m128i_i8[0] == v206 )
                {
                  if ( v206 || (v135 = 0, v386.m128i_i64[1] == v207) )
                  {
                    if ( !*((_BYTE *)v5 + 24) || sub_1F6C830((__int64)v5[1], 0x6Au, v391) )
                    {
                      v394.m128i_i64[0] = *(_QWORD *)(v6 + 72);
                      if ( v394.m128i_i64[0] )
                      {
                        *(_QWORD *)&v384 = &v394;
                        sub_1F6CA20(v394.m128i_i64);
                      }
                      v208 = *(_DWORD *)(v6 + 64);
                      v209 = v5[1];
                      *(_QWORD *)&v384 = &v394;
                      v394.m128i_i32[2] = v208;
                      v210 = *(__int64 (__fastcall **)(__int64 *, __int64))(*v209 + 48);
                      v211 = sub_1E0A0C0((*v5)[4]);
                      v212 = v210(v209, v211);
                      v213 = &v394;
                      v214 = v212;
                      v215 = 0;
                      if ( !v383.m128i_i8[0] )
                      {
                        v216 = sub_1D15970(&v391);
                        v213 = (__m128i *)v384;
                        v215 = v216 - 1;
                      }
                      v217 = *v5;
                      v218 = *v5;
                      *(_QWORD *)&v384 = v213;
                      *(_QWORD *)&v219 = sub_1D38BB0(
                                           (__int64)v218,
                                           v215,
                                           (__int64)v213,
                                           v214,
                                           0,
                                           0,
                                           (__m128i)v8,
                                           *(double *)a4.m128i_i64,
                                           a5,
                                           0);
                      v383.m128i_i64[0] = v384;
                      *(_QWORD *)&v384 = sub_1D332F0(
                                           v217,
                                           106,
                                           v384,
                                           v386.m128i_u32[0],
                                           (const void **)v386.m128i_i64[1],
                                           0,
                                           *(double *)v8.m128_u64,
                                           *(double *)a4.m128i_i64,
                                           a5,
                                           v385.m128i_i64[0],
                                           v385.m128i_u64[1],
                                           v219);
                      v385.m128i_i64[0] = v220;
                      goto LABEL_213;
                    }
                    goto LABEL_231;
                  }
LABEL_162:
                  if ( v135 )
                  {
                    if ( (unsigned __int8)(v135 - 14) <= 0x5Fu )
                    {
LABEL_164:
                      v58 = *(_WORD *)(v9 + 24);
                      goto LABEL_165;
                    }
                  }
                  else if ( sub_1F58D20((__int64)&v386) )
                  {
                    goto LABEL_164;
                  }
                  if ( (unsigned __int8)sub_1FB1D70((__int64)v5, v6, 0) )
                    return v6;
                  goto LABEL_164;
                }
LABEL_231:
                v135 = v386.m128i_i8[0];
                goto LABEL_162;
              }
            }
            else
            {
              v385.m128i_i8[0] = 0;
              v221 = sub_1F58D20((__int64)&v386);
              v135 = 0;
              if ( !v221 )
                goto LABEL_216;
            }
LABEL_165:
            if ( ((v58 - 66) & 0xFFFD) != 0
              || !sub_1D18C00(v9, 1, v384)
              || (unsigned __int8)sub_1D18C40(v9, 1)
              || ((v183 = *(_WORD *)(v9 + 24), *((_BYTE *)v5 + 24)) || v183 != 68)
              && !sub_1F6C830((__int64)v5[1], v183, v386.m128i_u8[0]) )
            {
              if ( *((_BYTE *)v5 + 25) )
                goto LABEL_168;
              if ( *(_WORD *)(v9 + 24) != 109 )
                goto LABEL_168;
              v222 = **(_QWORD **)(v9 + 32);
              if ( (unsigned __int16)(*(_WORD *)(v222 + 24) - 142) > 2u )
                goto LABEL_168;
              LOBYTE(v223) = sub_1F7E0F0((__int64)&v386);
              v385.m128i_i64[0] = v223;
              v224 = *(_QWORD *)(v222 + 32);
              v226 = v225;
              *(_QWORD *)&v384 = &v394;
              v227 = *(_QWORD *)(*(_QWORD *)v224 + 40LL);
              LOBYTE(v225) = *(_BYTE *)v227;
              v228 = *(_QWORD *)(v227 + 8);
              v394.m128i_i8[0] = v225;
              v394.m128i_i64[1] = v228;
              v229 = sub_1F7E0F0((__int64)&v394);
              if ( v229 != v385.m128i_i8[0] || !v229 && v230 != v226 )
              {
LABEL_168:
                v136 = sub_1F77270(v5, v6, *(double *)v8.m128_u64, *(double *)a4.m128i_i64, a5);
                result = 0;
                if ( v136 )
                  return (__int64)v136;
                return result;
              }
              v231 = *(_QWORD *)(v9 + 32);
              v232 = *(__int64 **)(v222 + 32);
              v385.m128i_i64[0] = v384;
              v233 = *v5;
              sub_1F80610(v384, *(_QWORD *)v231);
              v339 = *(_OWORD *)(v231 + 40);
              v336 = v232[1];
              v335 = *v232;
              v383.m128i_i64[0] = v384;
              *(_QWORD *)&v384 = sub_1D332F0(
                                   v233,
                                   109,
                                   v384,
                                   v386.m128i_u32[0],
                                   (const void **)v386.m128i_i64[1],
                                   0,
                                   *(double *)v8.m128_u64,
                                   *(double *)a4.m128i_i64,
                                   a5,
                                   v335,
                                   v336,
                                   v339);
              v385.m128i_i64[0] = v234;
            }
            else
            {
              v394.m128i_i64[0] = *(_QWORD *)(v6 + 72);
              if ( v394.m128i_i64[0] )
              {
                v385.m128i_i64[0] = (__int64)&v394;
                sub_1F6CA20(v394.m128i_i64);
              }
              v184 = *(_DWORD *)(v6 + 64);
              v185 = *v5;
              v385.m128i_i64[0] = (__int64)&v394;
              v394.m128i_i32[2] = v184;
              v186 = sub_1D309E0(
                       v185,
                       145,
                       (__int64)&v394,
                       v386.m128i_u32[0],
                       (const void **)v386.m128i_i64[1],
                       0,
                       *(double *)v8.m128_u64,
                       *(double *)a4.m128i_i64,
                       *(double *)a5.m128i_i64,
                       *(_OWORD *)*(_QWORD *)(v9 + 32));
              v187 = *v5;
              v188 = v186;
              v190 = v189;
              v334 = *(_OWORD *)(*(_QWORD *)(v9 + 32) + 40LL);
              *(_QWORD *)&v384 = &v394;
              v191.m128i_i64[0] = sub_1D309E0(
                                    v187,
                                    145,
                                    (__int64)&v394,
                                    v386.m128i_u32[0],
                                    (const void **)v386.m128i_i64[1],
                                    0,
                                    *(double *)v8.m128_u64,
                                    *(double *)a4.m128i_i64,
                                    *(double *)a5.m128i_i64,
                                    v334);
              v192 = *v5;
              v385 = v191;
              v193 = sub_1D252B0(
                       (__int64)v192,
                       v386.m128i_u32[0],
                       v386.m128i_i64[1],
                       *(unsigned __int8 *)(*(_QWORD *)(v9 + 40) + 16LL),
                       *(_QWORD *)(*(_QWORD *)(v9 + 40) + 24LL));
              v194 = *(unsigned __int16 *)(v9 + 24);
              v195 = (const void ***)v193;
              v196 = *(_QWORD *)(v9 + 32);
              v383.m128i_i64[0] = (__int64)&v394;
              *((_QWORD *)&v332 + 1) = v190;
              *(_QWORD *)&v332 = v188;
              *(_QWORD *)&v384 = sub_1D37470(
                                   *v5,
                                   v194,
                                   (__int64)&v394,
                                   v195,
                                   v197,
                                   v198,
                                   v332,
                                   *(_OWORD *)&v385,
                                   *(_OWORD *)(v196 + 80));
              v385.m128i_i64[0] = v199;
            }
LABEL_213:
            sub_17CD270((__int64 *)v383.m128i_i64[0]);
            return v384;
          }
        }
        else if ( !((unsigned __int8 (__fastcall *)(__int64 *, _QWORD, _QWORD, __int64))v59)(
                     v57,
                     v58,
                     v386.m128i_u32[0],
                     v386.m128i_i64[1]) )
        {
          v58 = *(_WORD *)(v9 + 24);
LABEL_53:
          if ( v58 == 107 )
          {
            if ( !*((_BYTE *)v5 + 25) )
            {
              v391 = (unsigned __int64)v393;
              v392 = 0x800000000LL;
              if ( *(_DWORD *)(v9 + 56) )
              {
                v361 = *(unsigned int *)(v9 + 56);
                v64 = v346;
                v357 = 0;
                v359 = 0;
                v385.m128i_i32[0] = 0;
                v358 = 0;
                v366 = (__int64)v5;
                v350 = v6;
                v65 = 0;
                v66 = v9;
                do
                {
                  v67 = *(_QWORD *)(v66 + 32) + 40 * v65;
                  v68 = *(_QWORD *)v67;
                  v69 = *(unsigned int *)(v67 + 8);
                  if ( *(_WORD *)(v68 + 24) != 48 )
                  {
                    if ( v385.m128i_i32[0] == 1 )
                    {
                      v5 = (__int64 **)v366;
                      v6 = v350;
                      v9 = v66;
                      if ( (_BYTE *)v391 != v393 )
                        _libc_free(v391);
                      v58 = *(_WORD *)(v66 + 24);
                      goto LABEL_161;
                    }
                    v357 = v69;
                    v359 = v68;
                    v358 = v65;
                    v385.m128i_i32[0] = 1;
                  }
                  v70 = *(_QWORD *)(v68 + 40) + 16 * v69;
                  v71 = *(_BYTE *)v70;
                  v72 = *(_QWORD *)(v70 + 8);
                  v394.m128i_i8[0] = v71;
                  v394.m128i_i64[1] = v72;
                  if ( v71 )
                    v73 = word_42FA680[(unsigned __int8)(v71 - 14)];
                  else
                    v73 = sub_1F58D30((__int64)&v394);
                  if ( v386.m128i_i8[0] )
                  {
                    switch ( v386.m128i_i8[0] )
                    {
                      case 0xE:
                      case 0xF:
                      case 0x10:
                      case 0x11:
                      case 0x12:
                      case 0x13:
                      case 0x14:
                      case 0x15:
                      case 0x16:
                      case 0x17:
                      case 0x38:
                      case 0x39:
                      case 0x3A:
                      case 0x3B:
                      case 0x3C:
                      case 0x3D:
                        v74 = 2;
                        break;
                      case 0x18:
                      case 0x19:
                      case 0x1A:
                      case 0x1B:
                      case 0x1C:
                      case 0x1D:
                      case 0x1E:
                      case 0x1F:
                      case 0x20:
                      case 0x3E:
                      case 0x3F:
                      case 0x40:
                      case 0x41:
                      case 0x42:
                      case 0x43:
                        v74 = 3;
                        break;
                      case 0x21:
                      case 0x22:
                      case 0x23:
                      case 0x24:
                      case 0x25:
                      case 0x26:
                      case 0x27:
                      case 0x28:
                      case 0x44:
                      case 0x45:
                      case 0x46:
                      case 0x47:
                      case 0x48:
                      case 0x49:
                        v74 = 4;
                        break;
                      case 0x29:
                      case 0x2A:
                      case 0x2B:
                      case 0x2C:
                      case 0x2D:
                      case 0x2E:
                      case 0x2F:
                      case 0x30:
                      case 0x4A:
                      case 0x4B:
                      case 0x4C:
                      case 0x4D:
                      case 0x4E:
                      case 0x4F:
                        v74 = 5;
                        break;
                      case 0x31:
                      case 0x32:
                      case 0x33:
                      case 0x34:
                      case 0x35:
                      case 0x36:
                      case 0x50:
                      case 0x51:
                      case 0x52:
                      case 0x53:
                      case 0x54:
                      case 0x55:
                        v74 = 6;
                        break;
                      case 0x37:
                        v74 = 7;
                        break;
                      case 0x56:
                      case 0x57:
                      case 0x58:
                      case 0x62:
                      case 0x63:
                      case 0x64:
                        v74 = 8;
                        break;
                      case 0x59:
                      case 0x5A:
                      case 0x5B:
                      case 0x5C:
                      case 0x5D:
                      case 0x65:
                      case 0x66:
                      case 0x67:
                      case 0x68:
                      case 0x69:
                        v74 = 9;
                        break;
                      case 0x5E:
                      case 0x5F:
                      case 0x60:
                      case 0x61:
                      case 0x6A:
                      case 0x6B:
                      case 0x6C:
                      case 0x6D:
                        v74 = 10;
                        break;
                    }
                    v362 = 0;
                  }
                  else
                  {
                    v74 = sub_1F596B0((__int64)&v386);
                    v362 = v137;
                  }
                  v75 = v74;
                  v372 = *(_QWORD **)(*(_QWORD *)v366 + 48LL);
                  v76 = sub_1D15020(v74, v73);
                  v77 = 0;
                  if ( !v76 )
                  {
                    v76 = sub_1F593D0(v372, v75, v362, v73);
                    v77 = v138;
                  }
                  LOBYTE(v64) = v76;
                  v78 = (unsigned int)v392;
                  if ( (unsigned int)v392 >= HIDWORD(v392) )
                  {
                    v377 = v77;
                    sub_16CD150((__int64)&v391, v393, 0, 16, v62, v63);
                    v78 = (unsigned int)v392;
                    v77 = v377;
                  }
                  v79 = (__int64 *)(v391 + 16 * v78);
                  ++v65;
                  *v79 = v64;
                  v79[1] = v77;
                  v80 = (unsigned int)v392;
                  v81 = v392 + 1;
                  LODWORD(v392) = v392 + 1;
                }
                while ( v361 != v65 );
                v5 = (__int64 **)v366;
                if ( !v385.m128i_i32[0] )
                  goto LABEL_72;
                v385.m128i_i64[0] = (__int64)v395;
                v394.m128i_i64[0] = (__int64)v395;
                v394.m128i_i64[1] = 0x800000000LL;
                if ( v81 )
                {
                  *(_QWORD *)&v384 = v80;
                  for ( i = 0; ; ++i )
                  {
                    v281 = *(__int64 **)v366;
                    v282 = 16 * i;
                    if ( v358 == (_DWORD)i )
                    {
                      v283 = (const void ***)(v391 + v282);
                      v284 = *(_QWORD *)(v359 + 72);
                      v389 = v284;
                      if ( v284 )
                      {
                        v383.m128i_i64[0] = v391 + v282;
                        sub_1623A60((__int64)&v389, v284, 2);
                        v283 = (const void ***)v383.m128i_i64[0];
                      }
                      LODWORD(v390) = *(_DWORD *)(v359 + 64);
                      v356 = v357 | v356 & 0xFFFFFFFF00000000LL;
                      *((_QWORD *)&v340 + 1) = v356;
                      *(_QWORD *)&v340 = v359;
                      v285 = sub_1D309E0(
                               v281,
                               145,
                               (__int64)&v389,
                               *(unsigned int *)v283,
                               v283[1],
                               0,
                               *(double *)v8.m128_u64,
                               *(double *)a4.m128i_i64,
                               *(double *)a5.m128i_i64,
                               v340);
                      v383.m128i_i64[0] = v286;
                      v287 = v285;
                      if ( v389 )
                        sub_161E7C0((__int64)&v389, v389);
                      sub_1F81BC0(v366, v287);
                      v289 = v394.m128i_u32[2];
                      if ( v394.m128i_i32[2] >= (unsigned __int32)v394.m128i_i32[3] )
                      {
                        sub_16CD150((__int64)&v394, (const void *)v385.m128i_i64[0], 0, 16, v288, v63);
                        v289 = v394.m128i_u32[2];
                      }
                      v290 = v383.m128i_i64[0];
                      v291 = (__int64 *)(v394.m128i_i64[0] + 16 * v289);
                      *v291 = v287;
                      v291[1] = v290;
                      ++v394.m128i_i32[2];
                    }
                    else
                    {
                      v276 = *(_QWORD *)(v391 + 16 * i);
                      v277 = *(_QWORD *)(v391 + v282 + 8);
                      v389 = 0;
                      LODWORD(v390) = 0;
                      v278.m128i_i64[0] = (__int64)sub_1D2B300(v281, 0x30u, (__int64)&v389, v276, v277, v63);
                      v383 = v278;
                      if ( v389 )
                        sub_161E7C0((__int64)&v389, v389);
                      v280 = v394.m128i_u32[2];
                      if ( v394.m128i_i32[2] >= (unsigned __int32)v394.m128i_i32[3] )
                      {
                        sub_16CD150((__int64)&v394, (const void *)v385.m128i_i64[0], 0, 16, v279, v63);
                        v280 = v394.m128i_u32[2];
                      }
                      *(__m128i *)(v394.m128i_i64[0] + 16 * v280) = _mm_load_si128(&v383);
                      ++v394.m128i_i32[2];
                    }
                    if ( (_QWORD)v384 == i )
                      break;
                  }
                }
                v292 = *(__int64 **)v366;
                v293 = v394.m128i_i64[0];
                v294 = v394.m128i_u32[2];
                v389 = *(_QWORD *)(v350 + 72);
                if ( v389 )
                  sub_1F6CA20(&v389);
                *((_QWORD *)&v341 + 1) = v294;
                *(_QWORD *)&v341 = v293;
                LODWORD(v390) = *(_DWORD *)(v350 + 64);
                v383.m128i_i64[0] = (__int64)sub_1D359D0(
                                               v292,
                                               107,
                                               (__int64)&v389,
                                               v386.m128i_u32[0],
                                               (const void **)v386.m128i_i64[1],
                                               0,
                                               *(double *)v8.m128_u64,
                                               *(double *)a4.m128i_i64,
                                               a5,
                                               v341);
                *(_QWORD *)&v384 = v295;
                sub_17CD270(&v389);
                v82 = v384;
                result = v383.m128i_i64[0];
                if ( v394.m128i_i64[0] != v385.m128i_i64[0] )
                {
                  v385.m128i_i64[0] = v383.m128i_i64[0];
                  _libc_free(v394.m128i_u64[0]);
                  result = v385.m128i_i64[0];
                  v82 = v384;
                }
              }
              else
              {
LABEL_72:
                result = (__int64)sub_1D2B530(*v5, v386.m128i_u32[0], v386.m128i_i64[1], v61, v62, v63);
              }
              v83 = v391;
              if ( (_BYTE *)v391 == v393 )
                return result;
              *(_QWORD *)&v384 = v82;
LABEL_75:
              v385.m128i_i64[0] = result;
              _libc_free(v83);
              return v385.m128i_i64[0];
            }
            goto LABEL_231;
          }
          goto LABEL_161;
        }
      }
      result = sub_1F84730(v5, v6, *(double *)v8.m128_u64, a4, a5);
      if ( result )
        return result;
      v60 = sub_1D18C00(v9, 1, v384);
      v58 = *(_WORD *)(v9 + 24);
      if ( v60 && v58 == 185 )
      {
        if ( (*(_WORD *)(v9 + 26) & 0x380) == 0 && (*(_BYTE *)(v9 + 26) & 8) == 0 )
        {
          v255 = *(unsigned __int8 *)(v9 + 88);
          v256 = *(_QWORD *)(v9 + 96);
          v385.m128i_i64[0] = (__int64)&v394;
          v394.m128i_i8[0] = v255;
          v381 = v255;
          v394.m128i_i64[1] = v256;
          v383.m128i_i64[0] = v256;
          v257 = sub_1D159A0(v394.m128i_i8, 1, v256, v255, v62, v63, v343, v346, v349, v353);
          v262 = sub_1D159A0(v386.m128i_i8, 1, v258, v259, v260, v261, v345, v348, v352, v355);
          v263 = &v394;
          if ( v262 > ((v257 + 7) & 0xFFFFFFF8) )
          {
            v264 = *(_QWORD *)(v9 + 104);
            v265 = v383.m128i_i64[0];
            v266 = *(_QWORD *)(v9 + 32);
            v267 = *v5;
            v268 = v381;
            v394.m128i_i64[0] = *(_QWORD *)(v9 + 72);
            if ( v394.m128i_i64[0] )
            {
              *(_QWORD *)&v384 = v381;
              *((_QWORD *)&v384 + 1) = v383.m128i_i64[0];
              sub_1F6CA20((__int64 *)v385.m128i_i64[0]);
              v268 = v381;
              v265 = v383.m128i_i64[0];
              v263 = (__m128i *)v385.m128i_i64[0];
            }
            v269 = *(_DWORD *)(v9 + 64);
            v385.m128i_i64[0] = (__int64)v263;
            v394.m128i_i32[2] = v269;
            v270 = sub_1D2B590(
                     v267,
                     (*(_BYTE *)(v9 + 27) >> 2) & 3,
                     (__int64)v263,
                     v386.m128i_u32[0],
                     v386.m128i_i64[1],
                     v264,
                     *(_OWORD *)v266,
                     *(_QWORD *)(v266 + 40),
                     *(_QWORD *)(v266 + 48),
                     v268,
                     v265);
            sub_17CD270((__int64 *)v385.m128i_i64[0]);
            sub_1D44C70((__int64)*v5, v9, 1, v270, 1u);
            return v270;
          }
        }
        goto LABEL_231;
      }
      goto LABEL_53;
    }
    if ( !sub_1D18C00(v9, 1, v384) )
    {
LABEL_174:
      v33 = *(_WORD *)(v9 + 24);
      goto LABEL_38;
    }
    v131 = v5[1];
    if ( *((_BYTE *)v5 + 24) )
    {
      v132 = 1;
      if ( v386.m128i_i8[0] != 1 )
      {
        if ( !v386.m128i_i8[0] )
          goto LABEL_156;
        v132 = v386.m128i_u8[0];
        if ( !v131[v386.m128i_u8[0] + 15] )
        {
LABEL_150:
          v33 = *(_WORD *)(v9 + 24);
          if ( *((_DWORD *)v5 + 4) != 2 )
            goto LABEL_39;
          v56 = v386.m128i_i8[0];
          goto LABEL_103;
        }
      }
      if ( (*((_BYTE *)v131 + 259 * v132 + 2544) & 0xFB) != 0 )
        goto LABEL_174;
    }
    v151 = *(bool (__fastcall **)(__int64, __int64, unsigned __int8))(*v131 + 1136);
    if ( v151 != sub_1F6BB70 )
    {
      if ( !((unsigned __int8 (__fastcall *)(__int64 *, __int64, _QWORD, __int64))v151)(
              v131,
              122,
              v386.m128i_u32[0],
              v386.m128i_i64[1]) )
        goto LABEL_174;
LABEL_195:
      v152 = *(_QWORD *)(v9 + 32);
      v153 = _mm_loadu_si128((const __m128i *)(v152 + 40));
      v154 = *(_QWORD *)(v152 + 40);
      LODWORD(v152) = *(_DWORD *)(v152 + 48);
      v394.m128i_i64[0] = 0;
      v367 = v154;
      v155 = *v5;
      v363 = v152;
      v394.m128i_i64[1] = 1;
      v395[0] = 0;
      v395[1] = 1;
      sub_1D1F820((__int64)v155, v153.m128i_i64[0], v153.m128i_i64[1], (unsigned __int64 *)&v394, 0);
      v160 = sub_1D159C0((__int64)&v386, v153.m128i_i64[0], v156, v157, v158, v159);
      v161 = v394.m128i_i32[2];
      v162 = v160;
      if ( v394.m128i_i32[2] > 0x40u )
      {
        v360 = v160;
        v164 = sub_16A5810((__int64)&v394);
        v162 = v360;
      }
      else if ( v394.m128i_i64[0] << (64 - v394.m128i_i8[8]) == -1 )
      {
        v164 = 64;
      }
      else
      {
        _BitScanReverse64(&v163, ~(v394.m128i_i64[0] << (64 - v394.m128i_i8[8])));
        v164 = v163 ^ 0x3F;
      }
      v165 = v161 - v164;
      if ( !v162 || (_BitScanReverse(&v162, v162), v165 <= 31 - (v162 ^ 0x1F)) )
      {
        v391 = *(_QWORD *)(v6 + 72);
        if ( v391 )
        {
          v385.m128i_i64[0] = (__int64)&v394;
          sub_1F6CA20((__int64 *)&v391);
        }
        v166 = *(_DWORD *)(v6 + 64);
        v167 = (__int64)v5[1];
        *(_QWORD *)&v384 = &v394;
        LODWORD(v392) = v166;
        v168 = sub_1E0A0C0((*v5)[4]);
        v169 = sub_1F40B60(v167, v386.m128i_u32[0], v386.m128i_i64[1], v168, 1);
        v170 = *v5;
        v171 = v169;
        v172 = *(__int128 **)(v9 + 32);
        v174 = v173;
        v385.m128i_i64[0] = v171;
        v175 = sub_1D309E0(
                 v170,
                 145,
                 (__int64)&v391,
                 v386.m128i_u32[0],
                 (const void **)v386.m128i_i64[1],
                 0,
                 *(double *)v8.m128_u64,
                 *(double *)a4.m128i_i64,
                 *(double *)a5.m128i_i64,
                 *v172);
        v176 = v363;
        v178 = v177;
        v179 = &v394;
        v180 = *(_QWORD *)(v367 + 40) + 16LL * v363;
        if ( v385.m128i_i8[0] != *(_BYTE *)v180 || !v385.m128i_i8[0] && *(const void ***)(v180 + 8) != v174 )
        {
          v271 = v385.m128i_u32[0];
          v385.m128i_i64[0] = v384;
          v272 = sub_1D323C0(
                   *v5,
                   v153.m128i_i64[0],
                   v153.m128i_i64[1],
                   (__int64)&v391,
                   v271,
                   v174,
                   *(double *)v8.m128_u64,
                   *(double *)a4.m128i_i64,
                   *(double *)a5.m128i_i64);
          v274 = v273;
          v367 = v272;
          sub_1F81BC0((__int64)v5, v272);
          v179 = (__m128i *)v384;
          v176 = v274;
        }
        v181 = *v5;
        v383.m128i_i64[0] = (__int64)v179;
        *(_QWORD *)&v384 = sub_1D332F0(
                             v181,
                             122,
                             (__int64)&v391,
                             v386.m128i_u32[0],
                             (const void **)v386.m128i_i64[1],
                             0,
                             *(double *)v8.m128_u64,
                             *(double *)a4.m128i_i64,
                             a5,
                             v175,
                             v178,
                             __PAIR128__(v176 | v153.m128i_i64[1] & 0xFFFFFFFF00000000LL, v367));
        v385.m128i_i64[0] = v182;
        sub_17CD270((__int64 *)&v391);
        sub_135E100(v395);
        sub_135E100((__int64 *)v383.m128i_i64[0]);
        return v384;
      }
      sub_135E100(v395);
      sub_135E100(v394.m128i_i64);
      goto LABEL_174;
    }
    if ( v386.m128i_i8[0] )
    {
      if ( !v131[v386.m128i_u8[0] + 15] )
        goto LABEL_150;
      goto LABEL_195;
    }
LABEL_156:
    v33 = *(_WORD *)(v9 + 24);
    if ( *((_DWORD *)v5 + 4) != 2 )
      goto LABEL_39;
    goto LABEL_157;
  }
  v34 = *(unsigned int **)(v9 + 32);
  v35 = _mm_load_si128(&v386);
  v36 = *(_QWORD *)v34;
  v37 = v34[2];
  v385 = v35;
  v38 = v35.m128i_i8[0];
  v39 = *(_QWORD *)(v36 + 40) + 16 * v37;
  v40 = *(_BYTE *)v39;
  v41 = *(const void ***)(v39 + 8);
  v394 = v35;
  LOBYTE(v391) = v40;
  v392 = (__int64)v41;
  if ( v40 == v35.m128i_i8[0] )
  {
    if ( v40 || v41 == (const void **)v394.m128i_i64[1] )
    {
      v42 = _mm_load_si128(&v385);
      LOBYTE(v391) = v40;
      v392 = (__int64)v41;
      v394 = v42;
      v43 = (const void **)v42.m128i_i64[1];
LABEL_22:
      if ( v40 || v41 == v43 )
        return *(_QWORD *)v34;
      goto LABEL_97;
    }
  }
  else if ( v40 )
  {
    v383.m128i_i64[0] = (__int64)v41;
    v44 = sub_1F6C8D0(v40);
    v47 = v383.m128i_i64[0];
    LODWORD(v384) = v44;
    goto LABEL_26;
  }
  v371 = v40;
  v383.m128i_i64[0] = (__int64)v41;
  v53 = sub_1F58D40((__int64)&v391);
  v45 = v35.m128i_i8[0];
  v46 = v371;
  LODWORD(v384) = v53;
  v47 = v383.m128i_i64[0];
LABEL_26:
  if ( v45 )
  {
    v383.m128i_i64[0] = v47;
    v48 = sub_1F6C8D0(v45);
  }
  else
  {
    v373 = v46;
    v383.m128i_i64[0] = v47;
    v48 = sub_1F58D40((__int64)&v394);
    v38 = 0;
    v40 = v373;
  }
  v41 = (const void **)v383.m128i_i64[0];
  if ( v48 > (unsigned int)v384 )
  {
    v49 = *(_QWORD *)(v6 + 72);
    v50 = *v5;
    v394.m128i_i64[0] = v49;
    if ( v49 )
    {
      v385.m128i_i64[0] = (__int64)&v394;
      sub_1623A60((__int64)&v394, v49, 2);
      v51 = *(unsigned __int16 *)(v9 + 24);
    }
    else
    {
      v51 = v32;
    }
    v52 = *(_DWORD *)(v6 + 64);
    v385.m128i_i64[0] = (__int64)&v394;
    v394.m128i_i32[2] = v52;
    result = sub_1D309E0(
               v50,
               v51,
               (__int64)&v394,
               v386.m128i_u32[0],
               (const void **)v386.m128i_i64[1],
               0,
               *(double *)v8.m128_u64,
               *(double *)v35.m128i_i64,
               *(double *)a5.m128i_i64,
               *(_OWORD *)v34);
    goto LABEL_10;
  }
  v84 = _mm_load_si128(&v385);
  LOBYTE(v391) = v40;
  v392 = v383.m128i_i64[0];
  v394 = v84;
  if ( v40 == v38 )
  {
    v43 = (const void **)v394.m128i_i64[1];
    goto LABEL_22;
  }
  if ( v40 )
  {
    v86 = sub_1F6C8D0(v40);
    goto LABEL_88;
  }
LABEL_97:
  v385.m128i_i8[0] = v38;
  v92 = sub_1F58D40((__int64)&v391);
  v85 = v385.m128i_i8[0];
  v86 = v92;
LABEL_88:
  if ( v85 )
    v87 = sub_1F6C8D0(v85);
  else
    v87 = sub_1F58D40((__int64)&v394);
  if ( v87 >= v86 )
    return *(_QWORD *)v34;
  v88 = *(_QWORD *)(v6 + 72);
  v89 = *v5;
  v90 = &v394;
  v394.m128i_i64[0] = v88;
  if ( v88 )
  {
    v385.m128i_i64[0] = (__int64)&v394;
    sub_1623A60((__int64)&v394, v88, 2);
    v90 = (__m128i *)v385.m128i_i64[0];
  }
  v91 = *(_DWORD *)(v6 + 64);
  v385.m128i_i64[0] = (__int64)v90;
  v394.m128i_i32[2] = v91;
  result = sub_1D309E0(
             v89,
             145,
             (__int64)v90,
             v386.m128i_u32[0],
             (const void **)v386.m128i_i64[1],
             0,
             *(double *)v8.m128_u64,
             *(double *)v35.m128i_i64,
             *(double *)a5.m128i_i64,
             *(_OWORD *)v34);
LABEL_10:
  v25 = v394.m128i_i64[0];
  if ( v394.m128i_i64[0] )
  {
LABEL_11:
    v383.m128i_i64[0] = v24;
    *(_QWORD *)&v384 = result;
    sub_161E7C0(v385.m128i_i64[0], v25);
    return v384;
  }
  return result;
}
