// Function: sub_1175E90
// Address: 0x1175e90
//
unsigned __int8 *__fastcall sub_1175E90(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r14
  __int64 v7; // r12
  __m128i v8; // xmm0
  __m128i v9; // xmm1
  __m128i v10; // xmm3
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned __int8 **v17; // rcx
  unsigned __int64 v18; // r8
  __int64 v19; // r9
  unsigned __int8 **v20; // rax
  unsigned __int8 *v21; // rdi
  unsigned __int8 *v22; // rax
  __int64 v23; // rdx
  unsigned __int8 v24; // al
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  int v30; // edx
  __int64 v31; // rdi
  unsigned int v32; // eax
  __int64 *v33; // rsi
  __int64 v34; // r8
  __int64 v35; // r15
  unsigned int *v36; // r14
  __int64 v37; // r11
  __int64 v38; // r10
  __int64 v39; // rdx
  __int64 *v40; // r9
  __int64 v41; // r13
  __int64 v42; // rbx
  __int64 v43; // rdi
  __int64 v44; // rax
  __int64 v45; // rsi
  __int64 v46; // rax
  __int64 v47; // r10
  __int64 v48; // rcx
  __int64 v49; // rdi
  __int64 v50; // rdx
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rdx
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // r13
  __int64 v59; // rax
  __int64 v60; // rdi
  __int64 v61; // rdx
  char *v62; // rax
  char *v63; // rsi
  __int64 v64; // rdx
  char *v65; // rdx
  __int64 v66; // rdx
  __int64 v67; // rax
  __int64 *v68; // r13
  __int64 v69; // rax
  __int64 v70; // rcx
  __m128i *v71; // rax
  __m128i *v72; // rax
  _QWORD *v73; // rcx
  unsigned __int64 v74; // rax
  unsigned __int64 v75; // rcx
  char v76; // al
  __int64 v77; // r15
  __int64 v78; // rax
  __int64 v79; // r15
  __int64 v80; // rax
  __int64 v81; // rdi
  __int64 v82; // rdx
  char *v83; // rdx
  __int64 v84; // rax
  char v85; // dl
  __int64 v86; // r13
  __int64 v87; // r15
  __int64 v88; // rdx
  __int64 v89; // rax
  unsigned __int64 v90; // rdx
  bool v91; // al
  __int64 v92; // rdx
  bool v93; // r15
  _BYTE *v94; // r13
  const __m128i *v95; // rax
  __int64 v96; // r14
  const __m128i *v97; // r15
  __int64 v98; // rdx
  __int64 v99; // rsi
  unsigned __int64 v100; // rax
  int v101; // esi
  unsigned __int64 v102; // rax
  __int64 v103; // rbx
  __m128i v104; // xmm4
  __m128i v105; // xmm5
  unsigned __int64 v106; // xmm6_8
  __m128i v107; // xmm7
  __int64 v108; // rbx
  _BYTE *v109; // rax
  __int64 v110; // rdx
  __int64 v111; // rdx
  __int64 v112; // rbx
  __int64 v113; // rax
  unsigned __int8 **v114; // rbx
  unsigned __int8 **v115; // r12
  unsigned __int8 *v116; // rdi
  __int64 v117; // r15
  unsigned int v118; // r10d
  __int64 v119; // rdx
  __int64 v120; // rax
  __int64 v121; // r8
  int v122; // r13d
  __int64 v123; // rax
  __int64 v124; // r13
  __int64 v125; // rax
  unsigned __int8 *v126; // r15
  unsigned __int8 *v127; // rax
  unsigned __int8 *v128; // r13
  __int64 v129; // rax
  __int64 *v130; // r15
  __int64 v131; // rax
  __int64 v132; // rdx
  __int64 *v133; // rax
  __int64 v134; // r14
  __int64 *v135; // r12
  __int64 *v136; // r15
  unsigned __int8 *v137; // r8
  unsigned __int8 **v138; // rax
  unsigned __int8 *v139; // r8
  unsigned __int8 **v140; // rax
  unsigned __int8 *v141; // r8
  unsigned __int8 **v142; // rax
  unsigned __int8 *v143; // r8
  unsigned __int8 **v144; // rax
  __int64 v145; // rdx
  __int64 *v146; // rax
  __int64 v147; // rdx
  __int64 *v148; // rax
  __int64 *v149; // r15
  __int64 v150; // rdx
  __int64 *v151; // rax
  __int64 *v152; // r15
  char v153; // dl
  __int64 *v154; // rax
  __int64 *v155; // r15
  unsigned int v156; // r15d
  bool v157; // al
  __int64 v158; // rdi
  __int64 v159; // rsi
  _QWORD *v160; // rdx
  int v161; // eax
  bool v162; // al
  __int64 v163; // r13
  _BYTE *v164; // rax
  unsigned __int8 *v165; // rdx
  unsigned int v166; // r15d
  __int64 *v167; // rax
  __int64 v168; // rax
  unsigned __int8 *v169; // r8
  unsigned __int8 **v170; // rax
  __int64 *v171; // rax
  __int64 v172; // r13
  __m128i *v173; // r8
  int v174; // r9d
  int v175; // r9d
  __m128i *v176; // rcx
  int v177; // esi
  unsigned int i; // edx
  __m128i *v179; // rax
  __int64 v180; // rdi
  unsigned __int64 v181; // rcx
  __int64 v182; // r13
  unsigned __int64 v183; // rbx
  __m128i *v184; // r9
  int v185; // edx
  __int64 v186; // rcx
  __m128i *v187; // rax
  __int64 v188; // rdi
  int *v189; // rdx
  int v190; // eax
  __int64 v191; // rax
  __int64 v192; // rdx
  __int64 v193; // r15
  __int64 *v194; // r12
  __int64 *v195; // r14
  char v196; // di
  __int64 **v197; // r10
  int v198; // edx
  unsigned int v199; // esi
  __int64 **v200; // rax
  __int64 *v201; // r8
  char v202; // si
  unsigned int v203; // edx
  unsigned int v204; // edx
  unsigned __int32 v205; // ecx
  int v206; // edi
  unsigned int v207; // r8d
  unsigned int v208; // r8d
  int v209; // r9d
  unsigned int v210; // esi
  unsigned __int8 *v211; // rax
  unsigned int v212; // r15d
  int v213; // eax
  __int64 *v214; // r15
  __int64 v215; // rdx
  __int64 v216; // rax
  __int64 v217; // rsi
  __int64 *v218; // rdx
  __int64 v219; // rax
  __int64 *v220; // r12
  __int64 v221; // rax
  __int64 v222; // r8
  char v223; // r9
  char v224; // al
  __int64 v225; // rax
  char v226; // dh
  char v227; // dl
  char v228; // bl
  char v229; // si
  __int64 v230; // rdx
  char v231; // al
  __int64 v232; // rsi
  __int16 v233; // cx
  __int64 v234; // rbx
  __int64 v235; // r15
  __int64 v236; // r13
  __int64 v237; // rbx
  __int64 v238; // r13
  __int64 v239; // rdx
  unsigned int v240; // esi
  signed __int64 v241; // rdx
  __m128i v242; // rax
  __int64 v243; // r13
  __int64 v244; // rax
  __int64 v245; // r8
  __int64 v246; // r9
  __int64 v247; // rcx
  __int64 v248; // rdx
  unsigned __int8 *v249; // rax
  unsigned __int8 *v250; // rdi
  __int64 v251; // rcx
  unsigned __int8 *v252; // rsi
  __int64 v253; // rcx
  __int64 **v254; // r9
  int v255; // ecx
  unsigned int v256; // esi
  unsigned __int32 v257; // edx
  unsigned __int32 v258; // ecx
  __m128i *v259; // r8
  int v260; // r11d
  __int64 v261; // r9
  int v262; // ecx
  int v263; // ecx
  __m128i *v264; // rsi
  int v265; // edi
  unsigned int ii; // edx
  __int64 v267; // r8
  __int64 v268; // r9
  int v269; // ecx
  int v270; // ecx
  int v271; // edi
  unsigned int n; // edx
  __int64 v273; // r8
  __int64 v274; // r10
  int v275; // esi
  int v276; // esi
  __int64 **v277; // rdi
  int v278; // r8d
  unsigned int m; // edx
  __int64 *v280; // rcx
  char v281; // dl
  __int64 v282; // r10
  int v283; // esi
  int v284; // esi
  int v285; // r8d
  unsigned int k; // edx
  __int64 *v287; // rcx
  __int64 v288; // rax
  unsigned __int8 *v289; // r8
  unsigned __int8 *v290; // r8
  unsigned __int8 **v291; // rax
  signed __int64 v292; // rcx
  unsigned int v293; // edx
  unsigned int v294; // edx
  unsigned int v295; // edx
  __m128i *v296; // r9
  int v297; // r8d
  int v298; // r8d
  __m128i *v299; // rcx
  int v300; // esi
  unsigned int j; // edx
  __int64 v302; // rdi
  unsigned int v303; // edx
  __m128i *v304; // r9
  int v305; // r8d
  int v306; // r8d
  int v307; // esi
  unsigned int jj; // edx
  __int64 v309; // rdi
  unsigned int v310; // edx
  unsigned int v311; // edx
  unsigned int v312; // edx
  __int64 v313; // [rsp+8h] [rbp-278h]
  __int64 *v314; // [rsp+28h] [rbp-258h]
  _QWORD *v315; // [rsp+30h] [rbp-250h]
  unsigned int v316; // [rsp+38h] [rbp-248h]
  char v317; // [rsp+38h] [rbp-248h]
  char v318; // [rsp+40h] [rbp-240h]
  __int64 *v319; // [rsp+40h] [rbp-240h]
  char v320; // [rsp+40h] [rbp-240h]
  __int64 v321; // [rsp+48h] [rbp-238h]
  __int64 v322; // [rsp+48h] [rbp-238h]
  unsigned int v323; // [rsp+48h] [rbp-238h]
  __int64 *v324; // [rsp+48h] [rbp-238h]
  unsigned __int64 v325; // [rsp+50h] [rbp-230h]
  __int64 v326; // [rsp+50h] [rbp-230h]
  _BYTE *v327; // [rsp+50h] [rbp-230h]
  __int64 v328; // [rsp+50h] [rbp-230h]
  __int64 *v329; // [rsp+50h] [rbp-230h]
  unsigned __int8 *v330; // [rsp+50h] [rbp-230h]
  __int64 v331; // [rsp+50h] [rbp-230h]
  unsigned __int8 *v332; // [rsp+50h] [rbp-230h]
  __int64 v333; // [rsp+50h] [rbp-230h]
  __int64 v334; // [rsp+50h] [rbp-230h]
  __int64 v335; // [rsp+58h] [rbp-228h]
  char v336; // [rsp+58h] [rbp-228h]
  unsigned __int8 *v337; // [rsp+58h] [rbp-228h]
  unsigned int v338; // [rsp+58h] [rbp-228h]
  unsigned __int8 *v339; // [rsp+58h] [rbp-228h]
  unsigned int *v340; // [rsp+60h] [rbp-220h]
  __int64 *v341; // [rsp+60h] [rbp-220h]
  __int64 v342; // [rsp+60h] [rbp-220h]
  __int64 v343; // [rsp+60h] [rbp-220h]
  __int64 v344; // [rsp+60h] [rbp-220h]
  unsigned __int8 **v345; // [rsp+60h] [rbp-220h]
  __int64 v346; // [rsp+60h] [rbp-220h]
  int v347; // [rsp+60h] [rbp-220h]
  __int64 v348; // [rsp+60h] [rbp-220h]
  unsigned __int8 *v349; // [rsp+60h] [rbp-220h]
  __int64 v350; // [rsp+68h] [rbp-218h]
  __int64 v351; // [rsp+70h] [rbp-210h] BYREF
  __int64 v352; // [rsp+78h] [rbp-208h] BYREF
  __int64 **v353; // [rsp+80h] [rbp-200h] BYREF
  __m128i *v354; // [rsp+88h] [rbp-1F8h]
  char v355[16]; // [rsp+90h] [rbp-1F0h] BYREF
  unsigned int *v356; // [rsp+A0h] [rbp-1E0h]
  char v357; // [rsp+B0h] [rbp-1D0h]
  _QWORD v358[4]; // [rsp+C0h] [rbp-1C0h] BYREF
  __int16 v359; // [rsp+E0h] [rbp-1A0h]
  __int64 v360[4]; // [rsp+F0h] [rbp-190h] BYREF
  _QWORD *v361; // [rsp+110h] [rbp-170h]
  __int64 *v362; // [rsp+118h] [rbp-168h]
  __int64 *v363; // [rsp+120h] [rbp-160h] BYREF
  __int64 v364; // [rsp+128h] [rbp-158h]
  __int64 v365; // [rsp+130h] [rbp-150h] BYREF
  unsigned int v366; // [rsp+138h] [rbp-148h]
  __int16 v367; // [rsp+140h] [rbp-140h]
  __m128i v368; // [rsp+1B0h] [rbp-D0h] BYREF
  __m128i v369; // [rsp+1C0h] [rbp-C0h] BYREF
  unsigned __int64 v370; // [rsp+1D0h] [rbp-B0h] BYREF
  __int64 v371; // [rsp+1D8h] [rbp-A8h]
  __m128i v372; // [rsp+1E0h] [rbp-A0h]
  __int64 v373; // [rsp+1F0h] [rbp-90h]
  char v374; // [rsp+240h] [rbp-40h] BYREF

  v6 = (__int64 *)a1;
  v7 = a2;
  v8 = _mm_loadu_si128(a1 + 6);
  v9 = _mm_loadu_si128(a1 + 7);
  v10 = _mm_loadu_si128(a1 + 9);
  v11 = a1[10].m128i_i64[0];
  v370 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v371 = a2;
  v373 = v11;
  v368 = v8;
  v369 = v9;
  v372 = v10;
  v12 = sub_1020E10(a2, &v368, a3, a4, a5, a6);
  if ( v12 )
  {
    if ( !*(_QWORD *)(a2 + 16) )
      return 0;
    v13 = v12;
    sub_10A5FE0(a1[2].m128i_i64[1], a2);
    if ( a2 == v13 )
    {
      v13 = sub_ACADE0(*(__int64 ***)(a2 + 8));
      if ( *(_QWORD *)(v13 + 16) )
        goto LABEL_5;
    }
    else if ( *(_QWORD *)(v13 + 16) )
    {
LABEL_5:
      sub_BD84D0(a2, v13);
      return (unsigned __int8 *)v7;
    }
    if ( *(_BYTE *)v13 > 0x1Cu && (*(_BYTE *)(v13 + 7) & 0x10) == 0 && (*(_BYTE *)(a2 + 7) & 0x10) != 0 )
      sub_BD6B90((unsigned __int8 *)v13, (unsigned __int8 *)a2);
    goto LABEL_5;
  }
  v15 = (__int64)sub_1172FC0((__int64)a1, a2);
  if ( v15 )
    return (unsigned __int8 *)v15;
  v350 = sub_1171EF0((__int64)a1, a2);
  if ( v350 )
    return (unsigned __int8 *)v350;
  v20 = *(unsigned __int8 ***)(a2 - 8);
  v21 = *v20;
  v22 = v20[4];
  v23 = *v21;
  if ( (unsigned __int8)v23 > 0x1Cu )
  {
    v24 = *v22;
    if ( v24 > 0x1Cu && (_BYTE)v23 == v24 )
    {
      if ( (unsigned __int8)sub_BD36B0((__int64)v21) )
      {
        v15 = (__int64)sub_1174320((__int64)v6, a2, v23, (__int64)v17, v18, v19);
        if ( v15 )
          return (unsigned __int8 *)v15;
      }
    }
  }
  if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) != 14 )
    goto LABEL_23;
  v124 = *(_QWORD *)(a2 + 40);
  v125 = sub_AA5190(v124);
  if ( v125 )
  {
    if ( v125 == v124 + 48 )
      goto LABEL_23;
  }
  v126 = **(unsigned __int8 ***)(a2 - 8);
  v127 = sub_BD3990(v126, a2);
  v369.m128i_i8[12] = 1;
  v128 = v127;
  v369.m128i_i32[2] = 0;
  v368.m128i_i64[1] = (__int64)&v370;
  v369.m128i_i64[0] = 0x100000004LL;
  v370 = (unsigned __int64)v126;
  v368.m128i_i64[0] = 1;
  if ( v126 == v127 )
    goto LABEL_23;
  v129 = 4LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v130 = *(__int64 **)(a2 - 8);
    v17 = (unsigned __int8 **)&v130[v129];
    v329 = &v130[v129];
  }
  else
  {
    v329 = (__int64 *)a2;
    v130 = (__int64 *)(a2 - v129 * 8);
  }
  v131 = (v129 * 8) >> 7;
  if ( !v131 )
  {
    v23 = (char *)v329 - (char *)v130;
    v168 = ((char *)v329 - (char *)v130) >> 5;
    if ( (char *)v329 - (char *)v130 == 64 )
    {
      v345 = (unsigned __int8 **)v130;
      v290 = (unsigned __int8 *)*v130;
      goto LABEL_444;
    }
LABEL_279:
    if ( v168 != 3 )
    {
      if ( v168 != 1 )
        goto LABEL_286;
      goto LABEL_281;
    }
    v289 = (unsigned __int8 *)*v130;
    if ( !v369.m128i_i8[12] )
      goto LABEL_462;
    v23 = v368.m128i_i64[1];
    a2 = v368.m128i_i64[1] + 8LL * v369.m128i_u32[1];
    while ( a2 != v23 )
    {
      if ( v289 == *(unsigned __int8 **)v23 )
        goto LABEL_442;
      v23 += 8;
    }
    if ( v369.m128i_i32[1] < (unsigned __int32)v369.m128i_i32[0] )
    {
      ++v369.m128i_i32[1];
      *(_QWORD *)a2 = v289;
      ++v368.m128i_i64[0];
    }
    else
    {
LABEL_462:
      a2 = *v130;
      v349 = (unsigned __int8 *)*v130;
      sub_C8CC70((__int64)&v368, *v130, v23, (__int64)v17, (__int64)v289, v19);
      v289 = v349;
      if ( !(_BYTE)v23 )
        goto LABEL_442;
    }
    if ( v128 != sub_BD3990(v289, a2) )
      goto LABEL_209;
LABEL_442:
    v345 = (unsigned __int8 **)(v130 + 4);
    goto LABEL_443;
  }
  v132 = 1;
  v345 = (unsigned __int8 **)&v130[16 * v131];
  v133 = v6;
  v134 = a2;
  v135 = v130;
  v136 = v133;
  while ( 1 )
  {
    v137 = (unsigned __int8 *)*v135;
    if ( (_BYTE)v132 )
    {
      v138 = (unsigned __int8 **)v368.m128i_i64[1];
      a2 = v368.m128i_i64[1] + 8LL * v369.m128i_u32[1];
      if ( v368.m128i_i64[1] != a2 )
      {
        while ( v137 != *v138 )
        {
          if ( (unsigned __int8 **)a2 == ++v138 )
            goto LABEL_221;
        }
        goto LABEL_189;
      }
LABEL_221:
      if ( v369.m128i_i32[1] < (unsigned __int32)v369.m128i_i32[0] )
      {
        ++v369.m128i_i32[1];
        *(_QWORD *)a2 = v137;
        ++v368.m128i_i64[0];
LABEL_207:
        if ( v128 != sub_BD3990(v137, a2) )
        {
          v146 = v136;
          v130 = v135;
          v7 = v134;
          v6 = v146;
          goto LABEL_209;
        }
        v132 = v369.m128i_u8[12];
        goto LABEL_189;
      }
    }
    a2 = *v135;
    v337 = (unsigned __int8 *)*v135;
    sub_C8CC70((__int64)&v368, *v135, v132, (__int64)v17, (__int64)v137, v19);
    v137 = v337;
    v19 = v145;
    v132 = v369.m128i_u8[12];
    if ( (_BYTE)v19 )
      goto LABEL_207;
LABEL_189:
    v139 = (unsigned __int8 *)v135[4];
    if ( !(_BYTE)v132 )
      goto LABEL_212;
    v140 = (unsigned __int8 **)v368.m128i_i64[1];
    a2 = v368.m128i_i64[1] + 8LL * v369.m128i_u32[1];
    if ( v368.m128i_i64[1] != a2 )
    {
      while ( v139 != *v140 )
      {
        if ( (unsigned __int8 **)a2 == ++v140 )
          goto LABEL_223;
      }
      goto LABEL_194;
    }
LABEL_223:
    if ( v369.m128i_i32[1] < (unsigned __int32)v369.m128i_i32[0] )
    {
      ++v369.m128i_i32[1];
      *(_QWORD *)a2 = v139;
      ++v368.m128i_i64[0];
    }
    else
    {
LABEL_212:
      a2 = v135[4];
      sub_C8CC70((__int64)&v368, a2, v132, (__int64)v17, (__int64)v139, v19);
      v139 = (unsigned __int8 *)a2;
      v19 = v147;
      v132 = v369.m128i_u8[12];
      if ( !(_BYTE)v19 )
        goto LABEL_194;
    }
    if ( v128 != sub_BD3990(v139, a2) )
    {
      v148 = v136;
      v149 = v135;
      v7 = v134;
      v130 = v149 + 4;
      v6 = v148;
      goto LABEL_209;
    }
    v132 = v369.m128i_u8[12];
LABEL_194:
    v141 = (unsigned __int8 *)v135[8];
    if ( (_BYTE)v132 )
    {
      v142 = (unsigned __int8 **)v368.m128i_i64[1];
      a2 = v368.m128i_i64[1] + 8LL * v369.m128i_u32[1];
      if ( v368.m128i_i64[1] != a2 )
      {
        while ( v141 != *v142 )
        {
          if ( (unsigned __int8 **)a2 == ++v142 )
            goto LABEL_225;
        }
        goto LABEL_199;
      }
LABEL_225:
      if ( v369.m128i_i32[1] < (unsigned __int32)v369.m128i_i32[0] )
      {
        ++v369.m128i_i32[1];
        *(_QWORD *)a2 = v141;
        ++v368.m128i_i64[0];
LABEL_216:
        if ( v128 != sub_BD3990(v141, a2) )
        {
          v151 = v136;
          v152 = v135;
          v7 = v134;
          v130 = v152 + 8;
          v6 = v151;
          goto LABEL_209;
        }
        v132 = v369.m128i_u8[12];
        goto LABEL_199;
      }
    }
    a2 = v135[8];
    sub_C8CC70((__int64)&v368, a2, v132, (__int64)v17, (__int64)v141, v19);
    v141 = (unsigned __int8 *)a2;
    v19 = v150;
    v132 = v369.m128i_u8[12];
    if ( (_BYTE)v19 )
      goto LABEL_216;
LABEL_199:
    v143 = (unsigned __int8 *)v135[12];
    if ( !(_BYTE)v132 )
      goto LABEL_218;
    v144 = (unsigned __int8 **)v368.m128i_i64[1];
    a2 = v369.m128i_u32[1];
    v132 = v368.m128i_i64[1] + 8LL * v369.m128i_u32[1];
    if ( v368.m128i_i64[1] != v132 )
    {
      while ( v143 != *v144 )
      {
        if ( (unsigned __int8 **)v132 == ++v144 )
          goto LABEL_227;
      }
      goto LABEL_204;
    }
LABEL_227:
    if ( v369.m128i_i32[1] < (unsigned __int32)v369.m128i_i32[0] )
    {
      a2 = (unsigned int)++v369.m128i_i32[1];
      *(_QWORD *)v132 = v143;
      ++v368.m128i_i64[0];
    }
    else
    {
LABEL_218:
      a2 = v135[12];
      sub_C8CC70((__int64)&v368, a2, v132, (__int64)v17, (__int64)v143, v19);
      v143 = (unsigned __int8 *)a2;
      if ( !v153 )
        goto LABEL_204;
    }
    if ( v128 != sub_BD3990(v143, a2) )
    {
      v154 = v136;
      v155 = v135;
      v7 = v134;
      v130 = v155 + 12;
      v6 = v154;
      goto LABEL_209;
    }
LABEL_204:
    v135 += 16;
    if ( v135 == (__int64 *)v345 )
      break;
    v132 = v369.m128i_u8[12];
  }
  v167 = v136;
  v130 = v135;
  v7 = v134;
  v6 = v167;
  v23 = (char *)v329 - (char *)v130;
  v168 = ((char *)v329 - (char *)v130) >> 5;
  if ( (char *)v329 - (char *)v130 != 64 )
    goto LABEL_279;
LABEL_443:
  v290 = *v345;
  if ( !v369.m128i_i8[12] )
    goto LABEL_453;
LABEL_444:
  v291 = (unsigned __int8 **)v368.m128i_i64[1];
  v23 = v369.m128i_u32[1];
  v17 = (unsigned __int8 **)(v368.m128i_i64[1] + 8LL * v369.m128i_u32[1]);
  if ( (unsigned __int8 **)v368.m128i_i64[1] != v17 )
  {
    while ( *v291 != v290 )
    {
      if ( v17 == ++v291 )
        goto LABEL_449;
    }
LABEL_448:
    v130 = (__int64 *)(v345 + 4);
LABEL_281:
    v169 = (unsigned __int8 *)*v130;
    if ( v369.m128i_i8[12] )
    {
      v170 = (unsigned __int8 **)v368.m128i_i64[1];
      v23 = v369.m128i_u32[1];
      a2 = v368.m128i_i64[1] + 8LL * v369.m128i_u32[1];
      if ( v368.m128i_i64[1] != a2 )
      {
        while ( v169 != *v170 )
        {
          if ( (unsigned __int8 **)a2 == ++v170 )
            goto LABEL_466;
        }
        goto LABEL_286;
      }
LABEL_466:
      if ( v369.m128i_i32[1] < (unsigned __int32)v369.m128i_i32[0] )
      {
        ++v369.m128i_i32[1];
        *(_QWORD *)a2 = v169;
        ++v368.m128i_i64[0];
LABEL_422:
        if ( v128 != sub_BD3990(v169, a2) )
          goto LABEL_209;
        goto LABEL_286;
      }
    }
    a2 = *v130;
    v348 = *v130;
    sub_C8CC70((__int64)&v368, *v130, v23, (__int64)v17, (__int64)v169, v19);
    v169 = (unsigned __int8 *)v348;
    if ( v281 )
      goto LABEL_422;
LABEL_286:
    v26 = *(_QWORD *)(v7 + 8);
    v367 = 257;
    v7 = sub_B52210((__int64)v128, v26, (__int64)&v363, 0, 0);
    if ( v369.m128i_i8[12] )
      return (unsigned __int8 *)v7;
LABEL_261:
    v158 = v368.m128i_i64[1];
    goto LABEL_240;
  }
LABEL_449:
  if ( v369.m128i_i32[1] < (unsigned __int32)v369.m128i_i32[0] )
  {
    ++v369.m128i_i32[1];
    *v17 = v290;
    ++v368.m128i_i64[0];
    goto LABEL_451;
  }
LABEL_453:
  a2 = (__int64)v290;
  v339 = v290;
  sub_C8CC70((__int64)&v368, (__int64)v290, v23, (__int64)v17, (__int64)v290, v19);
  v290 = v339;
  if ( !(_BYTE)v23 )
    goto LABEL_448;
LABEL_451:
  v130 = (__int64 *)v345;
  if ( v128 == sub_BD3990(v290, a2) )
    goto LABEL_448;
LABEL_209:
  if ( v130 == v329 )
    goto LABEL_286;
  if ( !v369.m128i_i8[12] )
    _libc_free(v368.m128i_i64[1], a2);
LABEL_23:
  if ( (unsigned __int8)sub_1171B00((__int64)v6, v7, v23, (__int64)v17, v18, v19) )
    return 0;
  v25 = *(_QWORD *)(v7 + 16);
  if ( v25 && !*(_QWORD *)(v25 + 8) )
  {
    if ( *(_BYTE *)(*(_QWORD *)(v7 + 8) + 8LL) == 12 )
    {
      if ( (unsigned __int8)sub_1174BB0((__int64)v6, v7) )
        return 0;
      v25 = *(_QWORD *)(v7 + 16);
    }
    v83 = *(char **)(v25 + 24);
    v84 = *((_QWORD *)v83 + 2);
    if ( v84 )
    {
      if ( !*(_QWORD *)(v84 + 8) )
      {
        v85 = *v83;
        if ( ((unsigned __int8)(v85 - 41) <= 0x12u || v85 == 63) && v7 == *(_QWORD *)(v84 + 24) )
        {
          v16 = sub_ACADE0(*(__int64 ***)(v7 + 8));
          return sub_F162A0((__int64)v6, v7, v16);
        }
      }
    }
  }
  v26 = 3;
  v318 = sub_BD3660(v7, 3);
  if ( !v318 )
  {
    v86 = *(_QWORD *)(v7 + 16);
    v363 = &v365;
    v364 = 0x600000000LL;
    if ( !v86 )
    {
LABEL_145:
      if ( (*(_DWORD *)(v7 + 4) & 0x7FFFFFF) == 0 )
        goto LABEL_131;
      v94 = 0;
      v321 = 8LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
      v95 = (const __m128i *)v6;
      v96 = 0;
      v97 = v95;
      do
      {
        v98 = *(_QWORD *)(v7 - 8);
        v99 = *(_QWORD *)(v98 + 32LL * *(unsigned int *)(v7 + 72) + v96);
        v100 = *(_QWORD *)(v99 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v100 == v99 + 48 )
        {
          v102 = 0;
        }
        else
        {
          if ( !v100 )
            goto LABEL_550;
          v101 = *(unsigned __int8 *)(v100 - 24);
          v102 = v100 - 24;
          if ( (unsigned int)(v101 - 30) >= 0xB )
            v102 = 0;
        }
        v103 = *(_QWORD *)(v98 + 4 * v96);
        v104 = _mm_loadu_si128(v97 + 6);
        v105 = _mm_loadu_si128(v97 + 7);
        v326 = 4 * v96;
        v106 = _mm_loadu_si128(v97 + 8).m128i_u64[0];
        v26 = (__int64)&v368;
        v373 = v97[10].m128i_i64[0];
        v107 = _mm_loadu_si128(v97 + 9);
        v368 = v104;
        v370 = v106;
        v369 = v105;
        v371 = v102;
        v372 = v107;
        v336 = sub_9B6260(v103, &v368, 0);
        if ( v336 )
        {
          if ( !v94 )
          {
            v159 = 4LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
            if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
            {
              v160 = *(_QWORD **)(v7 - 8);
              v26 = (__int64)&v160[v159];
            }
            else
            {
              v160 = (_QWORD *)(v7 - v159 * 8);
              v26 = v7;
            }
            if ( v160 == (_QWORD *)v26 )
            {
LABEL_267:
              v26 = 1;
              v94 = (_BYTE *)sub_ACD640(*(_QWORD *)(v7 + 8), 1, 0);
            }
            else
            {
              while ( 1 )
              {
                v94 = (_BYTE *)*v160;
                if ( *(_BYTE *)*v160 == 17 )
                {
                  v27 = *((unsigned int *)v94 + 8);
                  if ( (unsigned int)v27 > 0x40 )
                  {
                    v315 = v160;
                    v316 = *((_DWORD *)v94 + 8);
                    v161 = sub_C444A0((__int64)(v94 + 24));
                    v27 = v316;
                    v160 = v315;
                    v162 = v316 == v161;
                  }
                  else
                  {
                    v162 = *((_QWORD *)v94 + 3) == 0;
                  }
                  if ( !v162 )
                    break;
                }
                v160 += 4;
                if ( (_QWORD *)v26 == v160 )
                  goto LABEL_267;
              }
            }
          }
          if ( v94 != (_BYTE *)v103 )
          {
            if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
              v108 = *(_QWORD *)(v7 - 8) + v326;
            else
              v108 = v7 + v326 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
            v109 = *(_BYTE **)v108;
            if ( *(_QWORD *)v108 )
            {
              v26 = *(_QWORD *)(v108 + 16);
              v110 = *(_QWORD *)(v108 + 8);
              *(_QWORD *)v26 = v110;
              if ( v110 )
              {
                v26 = *(_QWORD *)(v108 + 16);
                *(_QWORD *)(v110 + 16) = v26;
              }
            }
            *(_QWORD *)v108 = v94;
            if ( v94 )
            {
              v111 = *((_QWORD *)v94 + 2);
              v26 = (__int64)(v94 + 16);
              *(_QWORD *)(v108 + 8) = v111;
              if ( v111 )
                *(_QWORD *)(v111 + 16) = v108 + 8;
              *(_QWORD *)(v108 + 16) = v26;
              *((_QWORD *)v94 + 2) = v108;
            }
            if ( *v109 > 0x1Cu )
            {
              v368.m128i_i64[0] = (__int64)v109;
              v26 = (__int64)&v368;
              v327 = v109;
              v112 = v97[2].m128i_i64[1] + 2096;
              sub_11715E0(v112, v368.m128i_i64);
              v113 = *((_QWORD *)v327 + 2);
              if ( v113 )
              {
                if ( !*(_QWORD *)(v113 + 8) )
                {
                  v26 = (__int64)&v368;
                  v368.m128i_i64[0] = *(_QWORD *)(v113 + 24);
                  sub_11715E0(v112, v368.m128i_i64);
                }
              }
            }
            v27 = (__int64)&v363[(unsigned int)v364];
            if ( (__int64 *)v27 == v363 )
            {
              v318 = v336;
            }
            else
            {
              v328 = v7;
              v114 = (unsigned __int8 **)&v363[(unsigned int)v364];
              v115 = (unsigned __int8 **)v363;
              do
              {
                v116 = *v115++;
                sub_B44F30(v116);
              }
              while ( v114 != v115 );
              v7 = v328;
              v318 = v336;
            }
          }
        }
        v96 += 8;
      }
      while ( v96 != v321 );
      v6 = (__int64 *)v97;
      if ( !v318 )
      {
LABEL_131:
        if ( v363 != &v365 )
          _libc_free(v363, v26);
        goto LABEL_26;
      }
      v158 = (__int64)v363;
      if ( v363 == &v365 )
        return (unsigned __int8 *)v7;
LABEL_240:
      _libc_free(v158, v26);
      return (unsigned __int8 *)v7;
    }
    while ( 1 )
    {
      v87 = *(_QWORD *)(v86 + 24);
      if ( *(_BYTE *)v87 != 82 )
      {
        v88 = *(_QWORD *)(v87 + 16);
        if ( !v88
          || *(_QWORD *)(v88 + 8)
          || *(_BYTE *)v87 != 58
          || v7 != *(_QWORD *)(v87 - 64) && v7 != *(_QWORD *)(v87 - 32) )
        {
          goto LABEL_131;
        }
        v89 = (unsigned int)v364;
        v27 = HIDWORD(v364);
        v90 = (unsigned int)v364 + 1LL;
        if ( v90 > HIDWORD(v364) )
        {
          v26 = (__int64)&v365;
          sub_C8D5F0((__int64)&v363, &v365, v90, 8u, v28, v29);
          v89 = (unsigned int)v364;
        }
        v363[v89] = v87;
        LODWORD(v364) = v364 + 1;
        v87 = *(_QWORD *)(*(_QWORD *)(v87 + 16) + 24LL);
        if ( *(_BYTE *)v87 != 82 )
          goto LABEL_131;
      }
      if ( *(_BYTE *)(*(_QWORD *)(v7 + 8) + 8LL) != 12
        || (*(_WORD *)(v87 + 2) & 0x3Fu) - 32 > 1
        || **(_BYTE **)(v87 - 32) > 0x15u )
      {
        goto LABEL_131;
      }
      v344 = *(_QWORD *)(v87 - 32);
      v91 = sub_AC30F0(v344);
      v92 = v344;
      v93 = v91;
      if ( !v91 )
      {
        if ( *(_BYTE *)v344 == 17 )
        {
          v156 = *(_DWORD *)(v344 + 32);
          if ( v156 <= 0x40 )
            v157 = *(_QWORD *)(v344 + 24) == 0;
          else
            v157 = v156 == (unsigned int)sub_C444A0(v344 + 24);
          goto LABEL_236;
        }
        v346 = *(_QWORD *)(v344 + 8);
        v27 = v346;
        if ( (unsigned int)*(unsigned __int8 *)(v346 + 8) - 17 > 1 )
          goto LABEL_131;
        v26 = 0;
        v330 = (unsigned __int8 *)v92;
        v164 = sub_AD7630(v92, 0, v92);
        v165 = v330;
        v27 = v346;
        if ( v164 && *v164 == 17 )
        {
          v166 = *((_DWORD *)v164 + 8);
          if ( v166 <= 0x40 )
            v157 = *((_QWORD *)v164 + 3) == 0;
          else
            v157 = v166 == (unsigned int)sub_C444A0((__int64)(v164 + 24));
LABEL_236:
          if ( !v157 )
            goto LABEL_131;
          goto LABEL_144;
        }
        if ( *(_BYTE *)(v346 + 8) != 17 )
          goto LABEL_131;
        v27 = 0;
        v347 = *(_DWORD *)(v346 + 32);
        while ( v347 != (_DWORD)v27 )
        {
          v26 = (unsigned int)v27;
          v323 = v27;
          v332 = v165;
          v211 = (unsigned __int8 *)sub_AD69F0(v165, (unsigned int)v27);
          if ( !v211 )
            goto LABEL_131;
          v26 = *v211;
          v165 = v332;
          v27 = v323;
          if ( (_BYTE)v26 != 13 )
          {
            if ( (_BYTE)v26 != 17 )
              goto LABEL_131;
            v212 = *((_DWORD *)v211 + 8);
            if ( v212 <= 0x40 )
            {
              v93 = *((_QWORD *)v211 + 3) == 0;
            }
            else
            {
              v213 = sub_C444A0((__int64)(v211 + 24));
              v165 = v332;
              v27 = v323;
              v93 = v212 == v213;
            }
            if ( !v93 )
              goto LABEL_131;
          }
          v27 = (unsigned int)(v27 + 1);
        }
        if ( !v93 )
          goto LABEL_131;
      }
LABEL_144:
      v86 = *(_QWORD *)(v86 + 8);
      if ( !v86 )
        goto LABEL_145;
    }
  }
LABEL_26:
  v30 = *(_DWORD *)(v7 + 4) & 0x7FFFFFF;
  if ( v30 )
  {
    v31 = *(_QWORD *)(v7 - 8);
    v32 = 0;
    v27 = v31;
    while ( 1 )
    {
      v33 = *(__int64 **)v27;
      ++v32;
      if ( **(_BYTE **)v27 != 84 )
        break;
      v27 += 32;
      if ( v30 == v32 )
        goto LABEL_171;
    }
    v363 = *(__int64 **)v27;
    if ( v30 != v32 )
    {
      while ( 1 )
      {
        v27 = *(_QWORD *)(v31 + 32LL * v32);
        if ( !v27 )
          break;
        if ( v33 != (__int64 *)v27 && *(_BYTE *)v27 != 84 )
          goto LABEL_33;
        if ( ++v32 == v30 )
          goto LABEL_114;
      }
LABEL_550:
      BUG();
    }
  }
  else
  {
LABEL_171:
    v363 = 0;
  }
LABEL_114:
  v26 = (__int64)&v363;
  v368.m128i_i64[0] = 0;
  v368.m128i_i64[1] = (__int64)&v370;
  v369.m128i_i64[0] = 16;
  v369.m128i_i32[2] = 0;
  v369.m128i_i8[12] = 1;
  if ( (unsigned __int8)sub_116D410(v7, &v363, (__int64)&v368, v27, v28, v29) )
  {
    v163 = (__int64)v363;
    if ( *(_QWORD *)(v7 + 16) )
    {
      sub_10A5FE0(v6[5], v7);
      if ( v7 == v163 )
        v163 = sub_ACADE0(*(__int64 ***)(v7 + 8));
      if ( !*(_QWORD *)(v163 + 16)
        && *(_BYTE *)v163 > 0x1Cu
        && (*(_BYTE *)(v163 + 7) & 0x10) == 0
        && (*(_BYTE *)(v7 + 7) & 0x10) != 0 )
      {
        sub_BD6B90((unsigned __int8 *)v163, (unsigned __int8 *)v7);
      }
      v26 = v163;
      sub_BD84D0(v7, v163);
    }
    else
    {
      v7 = 0;
    }
    if ( v369.m128i_i8[12] )
      return (unsigned __int8 *)v7;
    goto LABEL_261;
  }
  if ( !v369.m128i_i8[12] )
    _libc_free(v368.m128i_i64[1], &v363);
LABEL_33:
  v368.m128i_i64[0] = *(_QWORD *)(v7 + 40);
  sub_1171210((__int64)v355, (__int64)(v6 + 49), v368.m128i_i64);
  v340 = v356;
  if ( v357 )
  {
    v117 = *(_QWORD *)(v7 - 8);
    v118 = *(_DWORD *)(v7 + 72);
    v119 = v356[4];
    v120 = *(_DWORD *)(v7 + 4) & 0x7FFFFFF;
    v121 = 8 * v120;
    v122 = *(_DWORD *)(v7 + 4) & 0x7FFFFFF;
    if ( v119 + v120 > (unsigned __int64)v356[5] )
    {
      v334 = 8 * v120;
      v338 = *(_DWORD *)(v7 + 72);
      sub_C8D5F0((__int64)(v356 + 2), v356 + 6, v119 + v120, 8u, v121, v119 + v120);
      v121 = v334;
      v118 = v338;
      v119 = v340[4];
    }
    v123 = 32LL * v118;
    if ( v117 + v123 != v117 + v121 + v123 )
    {
      memcpy((void *)(*((_QWORD *)v340 + 1) + 8 * v119), (const void *)(v117 + v123), v121);
      LODWORD(v119) = v340[4];
    }
    v340[4] = v119 + v122;
  }
  else
  {
    v34 = 0;
    v35 = 8LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
    if ( (*(_DWORD *)(v7 + 4) & 0x7FFFFFF) != 0 )
    {
      v341 = v6;
      v36 = v356;
      do
      {
        v37 = *(_QWORD *)(v7 - 8);
        v38 = 32LL * *(unsigned int *)(v7 + 72);
        v39 = *(_QWORD *)(*((_QWORD *)v36 + 1) + v34);
        v40 = (__int64 *)(v37 + v38 + v34);
        v41 = *v40;
        if ( v39 != *v40 )
        {
          v42 = *(_QWORD *)(v37 + 4 * v34);
          v43 = 4 * v34;
          v44 = 0x1FFFFFFFE0LL;
          v45 = 0x7FFFFFFF8LL;
          if ( (*(_DWORD *)(v7 + 4) & 0x7FFFFFF) != 0 )
          {
            v46 = 0;
            v47 = v37 + v38;
            do
            {
              v45 = 8 * v46;
              if ( v39 == *(_QWORD *)(v47 + 8 * v46) )
              {
                v44 = 32 * v46;
                goto LABEL_42;
              }
              ++v46;
            }
            while ( (*(_DWORD *)(v7 + 4) & 0x7FFFFFF) != (_DWORD)v46 );
            v44 = 0x1FFFFFFFE0LL;
            v45 = 0x7FFFFFFF8LL;
          }
LABEL_42:
          v48 = *(_QWORD *)(v37 + v44);
          if ( v48 )
          {
            *v40 = v39;
            v49 = *(_QWORD *)(v7 - 8) + v43;
            if ( *(_QWORD *)v49 )
            {
              v50 = *(_QWORD *)(v49 + 8);
              **(_QWORD **)(v49 + 16) = v50;
              if ( v50 )
                *(_QWORD *)(v50 + 16) = *(_QWORD *)(v49 + 16);
            }
            *(_QWORD *)v49 = v48;
            v51 = *(_QWORD *)(v48 + 16);
            *(_QWORD *)(v49 + 8) = v51;
            if ( v51 )
              *(_QWORD *)(v51 + 16) = v49 + 8;
            *(_QWORD *)(v49 + 16) = v48 + 16;
            *(_QWORD *)(v48 + 16) = v49;
            v52 = *(_QWORD *)(v7 - 8);
          }
          else
          {
            *v40 = v39;
            v52 = *(_QWORD *)(v7 - 8);
            v81 = v52 + v43;
            if ( *(_QWORD *)v81 )
            {
              v82 = *(_QWORD *)(v81 + 8);
              **(_QWORD **)(v81 + 16) = v82;
              if ( v82 )
                *(_QWORD *)(v82 + 16) = *(_QWORD *)(v81 + 16);
              *(_QWORD *)v81 = 0;
              v52 = *(_QWORD *)(v7 - 8);
            }
          }
          *(_QWORD *)(v52 + 32LL * *(unsigned int *)(v7 + 72) + v45) = v41;
          v53 = *(_QWORD *)(v7 - 8) + v44;
          if ( *(_QWORD *)v53 )
          {
            v54 = *(_QWORD *)(v53 + 8);
            **(_QWORD **)(v53 + 16) = v54;
            if ( v54 )
              *(_QWORD *)(v54 + 16) = *(_QWORD *)(v53 + 16);
          }
          *(_QWORD *)v53 = v42;
          if ( v42 )
          {
            v55 = *(_QWORD *)(v42 + 16);
            *(_QWORD *)(v53 + 8) = v55;
            if ( v55 )
              *(_QWORD *)(v55 + 16) = v53 + 8;
            *(_QWORD *)(v53 + 16) = v42 + 16;
            *(_QWORD *)(v42 + 16) = v53;
          }
        }
        v34 += 8;
      }
      while ( v34 != v35 );
      v6 = v341;
    }
  }
  v56 = sub_AA5930(*(_QWORD *)(v7 + 40));
  v342 = v57;
  v58 = v56;
  while ( v342 != v58 )
  {
    if ( v7 != v58 )
    {
      if ( (unsigned __int8)sub_B46130(v7, v58, 0) )
      {
        if ( *(_QWORD *)(v7 + 16) )
        {
          sub_10A5FE0(v6[5], v7);
          if ( !*(_QWORD *)(v58 + 16)
            && *(_BYTE *)v58 > 0x1Cu
            && (*(_BYTE *)(v58 + 7) & 0x10) == 0
            && (*(_BYTE *)(v7 + 7) & 0x10) != 0 )
          {
            sub_BD6B90((unsigned __int8 *)v58, (unsigned __int8 *)v7);
          }
          sub_BD84D0(v7, v58);
          return (unsigned __int8 *)v7;
        }
        return 0;
      }
      if ( !v58 )
        goto LABEL_549;
    }
    v59 = *(_QWORD *)(v58 + 32);
    if ( !v59 )
      goto LABEL_550;
    v58 = 0;
    if ( *(_BYTE *)(v59 - 24) == 84 )
      v58 = v59 - 24;
  }
  v60 = *(_QWORD *)(v7 + 8);
  if ( *(_BYTE *)(v60 + 8) == 12 )
  {
    v242.m128i_i64[0] = sub_BCAE30(v60);
    v243 = v6[11];
    v368 = v242;
    v244 = sub_CA1930(&v368);
    v247 = *(_QWORD *)(v243 + 40);
    v248 = v244;
    v249 = *(unsigned __int8 **)(v243 + 32);
    v250 = &v249[v247];
    v251 = v247 >> 2;
    if ( v251 > 0 )
    {
      v252 = &v249[4 * v251];
      while ( 1 )
      {
        v253 = *v249;
        if ( v248 == v253 )
          break;
        v253 = v249[1];
        if ( v248 == v253 )
        {
          ++v249;
          break;
        }
        v253 = v249[2];
        if ( v248 == v253 )
        {
          v249 += 2;
          break;
        }
        v253 = v249[3];
        if ( v248 == v253 )
        {
          v249 += 3;
          break;
        }
        v249 += 4;
        if ( v252 == v249 )
          goto LABEL_469;
      }
LABEL_370:
      if ( v250 != v249 )
        goto LABEL_71;
LABEL_371:
      v15 = sub_116E930((__int64)v6, v7, v248, v253, v245, v246);
      if ( !v15 )
        goto LABEL_71;
      return (unsigned __int8 *)v15;
    }
LABEL_469:
    v292 = v250 - v249;
    if ( v250 - v249 != 2 )
    {
      if ( v292 != 3 )
      {
        v253 = v292 - 1;
        if ( v253 )
          goto LABEL_371;
        goto LABEL_472;
      }
      v253 = *v249;
      if ( v248 == v253 )
        goto LABEL_370;
      ++v249;
    }
    v253 = *v249;
    if ( v248 == v253 )
      goto LABEL_370;
    ++v249;
LABEL_472:
    v253 = *v249;
    if ( v248 != v253 )
      goto LABEL_371;
    goto LABEL_370;
  }
LABEL_71:
  v335 = v6[10];
  v61 = 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
  {
    v62 = *(char **)(v7 - 8);
    v63 = &v62[v61];
  }
  else
  {
    v63 = (char *)v7;
    v62 = (char *)(v7 - v61);
  }
  v64 = v61 >> 7;
  if ( !v64 )
  {
LABEL_358:
    v241 = v63 - v62;
    if ( v63 - v62 != 64 )
    {
      if ( v241 != 96 )
      {
        if ( v241 != 32 )
          goto LABEL_81;
        goto LABEL_361;
      }
      if ( **(_BYTE **)v62 != 17 )
        goto LABEL_80;
      v62 += 32;
    }
    if ( **(_BYTE **)v62 != 17 )
      goto LABEL_80;
    v62 += 32;
LABEL_361:
    if ( **(_BYTE **)v62 != 17 )
      goto LABEL_80;
    goto LABEL_81;
  }
  v65 = &v62[128 * v64];
  while ( **(_BYTE **)v62 == 17 )
  {
    if ( **((_BYTE **)v62 + 4) != 17 )
    {
      v62 += 32;
      break;
    }
    if ( **((_BYTE **)v62 + 8) != 17 )
    {
      v62 += 64;
      break;
    }
    if ( **((_BYTE **)v62 + 12) != 17 )
    {
      v62 += 96;
      break;
    }
    v62 += 128;
    if ( v62 == v65 )
      goto LABEL_358;
  }
LABEL_80:
  if ( v63 != v62 )
    goto LABEL_15;
LABEL_81:
  v66 = *(_QWORD *)(v7 + 40);
  v67 = 0;
  v351 = v66;
  if ( v66 )
    v67 = (unsigned int)(*(_DWORD *)(v66 + 44) + 1);
  if ( (unsigned int)v67 >= *(_DWORD *)(v335 + 32) || !*(_QWORD *)(*(_QWORD *)(v335 + 24) + 8 * v67) )
  {
LABEL_15:
    v16 = sub_116D950(v7, v6[4]);
    if ( !v16 )
      return 0;
    return sub_F162A0((__int64)v6, v7, v16);
  }
  v68 = (__int64 *)sub_BD5C60(v7);
  v69 = 0;
  if ( v351 )
    v69 = (unsigned int)(*(_DWORD *)(v351 + 44) + 1);
  if ( (unsigned int)v69 >= *(_DWORD *)(v335 + 32) )
LABEL_549:
    BUG();
  v70 = **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v335 + 24) + 8 * v69) + 8LL);
  v71 = (__m128i *)&v365;
  v363 = 0;
  v364 = 1;
  v352 = v70;
  do
  {
    v71->m128i_i64[0] = -4096;
    ++v71;
  }
  while ( v71 != &v368 );
  v368.m128i_i64[0] = 0;
  v368.m128i_i64[1] = 1;
  v72 = &v369;
  do
  {
    v72->m128i_i64[0] = -4096;
    ++v72;
  }
  while ( v72 != (__m128i *)&v374 );
  v73 = (_QWORD *)(v70 + 48);
  v354 = &v368;
  v353 = &v363;
  v74 = *v73 & 0xFFFFFFFFFFFFFFF8LL;
  v325 = v74;
  if ( (_QWORD *)v74 == v73 )
    goto LABEL_550;
  if ( !v74 )
    goto LABEL_550;
  v75 = *v73 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (unsigned int)*(unsigned __int8 *)(v74 - 24) - 30 > 0xA )
    goto LABEL_550;
  v76 = *(_BYTE *)(v74 - 24);
  if ( v76 != 31 )
  {
    if ( v76 != 32 )
      goto LABEL_99;
    v171 = *(__int64 **)(v325 - 32);
    v172 = v171[4];
    v343 = *v171;
    if ( (v368.m128i_i8[8] & 1) != 0 )
    {
      v173 = &v369;
      v174 = 8;
    }
    else
    {
      v256 = v369.m128i_u32[2];
      v173 = (__m128i *)v369.m128i_i64[0];
      v174 = v369.m128i_i32[2];
      if ( !v369.m128i_i32[2] )
      {
        v257 = v368.m128i_u32[2];
        ++v368.m128i_i64[0];
        v179 = 0;
        v258 = ((unsigned __int32)v368.m128i_i32[2] >> 1) + 1;
        goto LABEL_389;
      }
    }
    v175 = v174 - 1;
    v176 = 0;
    v177 = 1;
    for ( i = v175 & (((unsigned int)v172 >> 4) ^ ((unsigned int)v172 >> 9)); ; i = v175 & v312 )
    {
      v179 = &v173[i];
      v180 = v179->m128i_i64[0];
      if ( v172 == v179->m128i_i64[0] )
        goto LABEL_294;
      if ( v180 == -4096 )
        break;
      if ( v176 || v180 != -8192 )
        v179 = v176;
      v312 = v177 + i;
      v176 = v179;
      ++v177;
    }
    v257 = v368.m128i_u32[2];
    v256 = 8;
    if ( v176 )
      v179 = v176;
    ++v368.m128i_i64[0];
    v258 = ((unsigned __int32)v368.m128i_i32[2] >> 1) + 1;
    if ( (v368.m128i_i8[8] & 1) == 0 )
      v256 = v369.m128i_u32[2];
LABEL_389:
    if ( 4 * v258 < 3 * v256 )
    {
      if ( v256 - v368.m128i_i32[3] - v258 > v256 >> 3 )
        goto LABEL_391;
      sub_FB9E50((__int64)&v368, v256);
      if ( (v368.m128i_i8[8] & 1) != 0 )
      {
        v296 = &v369;
        v297 = 8;
        goto LABEL_501;
      }
      v297 = v369.m128i_i32[2];
      v296 = (__m128i *)v369.m128i_i64[0];
      if ( v369.m128i_i32[2] )
      {
LABEL_501:
        v298 = v297 - 1;
        v299 = 0;
        v300 = 1;
        for ( j = v298 & (((unsigned int)v172 >> 9) ^ ((unsigned int)v172 >> 4)); ; j = v298 & v310 )
        {
          v179 = &v296[j];
          v302 = v179->m128i_i64[0];
          if ( v172 == v179->m128i_i64[0] )
            break;
          if ( v302 == -4096 )
            goto LABEL_522;
          if ( v302 != -8192 || v299 )
            v179 = v299;
          v310 = v300 + j;
          v299 = v179;
          ++v300;
        }
LABEL_503:
        v257 = v368.m128i_u32[2];
LABEL_391:
        v368.m128i_i32[2] = (2 * (v257 >> 1) + 2) | v257 & 1;
        if ( v179->m128i_i64[0] != -4096 )
          --v368.m128i_i32[3];
        v179->m128i_i64[0] = v172;
        v179->m128i_i32[2] = 0;
LABEL_294:
        v181 = v325;
        ++v179->m128i_i32[2];
        v331 = ((*(_DWORD *)(v325 - 20) & 0x7FFFFFFu) >> 1) - 1;
        if ( !v331 )
        {
LABEL_98:
          if ( *(_QWORD *)(v343 + 8) == *(_QWORD *)(v7 + 8) )
          {
            v214 = *(__int64 **)(v7 - 8);
            v215 = 4LL * *(unsigned int *)(v7 + 72);
            v216 = *(_DWORD *)(v7 + 4) & 0x7FFFFFF;
            v217 = v215 * 8 + 8 * v216;
            v218 = &v214[v215];
            v219 = 4 * v216;
            v324 = (__int64 *)((char *)v214 + v217);
            if ( (*(_BYTE *)(v7 + 7) & 0x40) == 0 )
              v214 = (__int64 *)(v7 - v219 * 8);
            v314 = &v214[v219];
            if ( v324 == v218 || &v214[v219] == v214 )
            {
LABEL_386:
              v350 = v343;
            }
            else
            {
              v320 = 0;
              v317 = 0;
              v333 = v7;
              v220 = v218;
              do
              {
                v221 = *v220;
                v222 = *v214;
                v360[1] = (__int64)&v368;
                v358[0] = v221;
                v313 = v222;
                v360[0] = (__int64)&v363;
                v360[2] = v335;
                v360[3] = (__int64)&v352;
                v361 = v358;
                v362 = &v351;
                v223 = sub_1170890(v360, v222);
                v224 = 0;
                if ( !v223 && (v288 = sub_AD63D0(v313), (v224 = sub_1170890(v360, v288)) == 0) || v317 && v320 != v224 )
                {
                  v7 = v333;
                  goto LABEL_99;
                }
                v320 = v224;
                v214 += 4;
                ++v220;
                v317 = 1;
              }
              while ( v214 != v314 && v324 != v220 );
              v7 = v333;
              if ( !v224 )
                goto LABEL_386;
              v225 = sub_AA5190(v351);
              v228 = v227;
              if ( v225 )
              {
                v229 = v226;
                v230 = v225;
                v231 = v229;
                v232 = v351;
                LOBYTE(v233) = v228;
                HIBYTE(v233) = v231;
                if ( v230 == v351 + 48 )
                  goto LABEL_99;
              }
              else
              {
                v232 = v351;
                v230 = 0;
                v233 = 0;
              }
              sub_A88F30(v6[4], v232, v230, v233);
              v234 = v6[4];
              v359 = 257;
              v235 = sub_AD62B0(*(_QWORD *)(v343 + 8));
              v350 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v234 + 80) + 16LL))(
                       *(_QWORD *)(v234 + 80),
                       30,
                       v343,
                       v235);
              if ( !v350 )
              {
                LOWORD(v361) = 257;
                v350 = sub_B504D0(30, v343, v235, (__int64)v360, 0, 0);
                (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v234 + 88) + 16LL))(
                  *(_QWORD *)(v234 + 88),
                  v350,
                  v358,
                  *(_QWORD *)(v234 + 56),
                  *(_QWORD *)(v234 + 64));
                v236 = 16LL * *(unsigned int *)(v234 + 8);
                v237 = *(_QWORD *)v234;
                v238 = v237 + v236;
                while ( v238 != v237 )
                {
                  v239 = *(_QWORD *)(v237 + 8);
                  v240 = *(_DWORD *)v237;
                  v237 += 16;
                  sub_B99FD0(v350, v240, v239);
                }
              }
            }
          }
          goto LABEL_99;
        }
        v182 = 0;
        v183 = v181;
        v322 = v7;
        v319 = v6;
        while ( 1 )
        {
          v191 = 32;
          if ( (_DWORD)v182 != -2 )
            v191 = 32LL * (unsigned int)(2 * v182 + 3);
          v192 = *(_QWORD *)(v183 - 32);
          v193 = (__int64)v353;
          ++v182;
          v194 = *(__int64 **)(v192 + v191);
          v195 = *(__int64 **)(v192 + 32LL * (unsigned int)(2 * v182));
          v196 = (_BYTE)v353[1] & 1;
          if ( v196 )
          {
            v197 = v353 + 2;
            v198 = 7;
          }
          else
          {
            v204 = *((_DWORD *)v353 + 6);
            v197 = (__int64 **)v353[2];
            if ( !v204 )
            {
              v208 = *((_DWORD *)v353 + 2);
              *v353 = (__int64 *)((char *)*v353 + 1);
              v200 = 0;
              v209 = (v208 >> 1) + 1;
LABEL_318:
              v210 = v204;
              goto LABEL_319;
            }
            v198 = v204 - 1;
          }
          v199 = v198 & (((unsigned int)v195 >> 9) ^ ((unsigned int)v195 >> 4));
          v200 = &v197[2 * v199];
          v201 = *v200;
          if ( *v200 == v195 )
            goto LABEL_305;
          v254 = 0;
          v255 = 1;
          while ( v201 != (__int64 *)-4096LL )
          {
            if ( v201 == (__int64 *)-8192LL && !v254 )
              v254 = v200;
            v199 = v198 & (v255 + v199);
            v200 = &v197[2 * v199];
            v201 = *v200;
            if ( v195 == *v200 )
              goto LABEL_305;
            ++v255;
          }
          v208 = *((_DWORD *)v353 + 2);
          v210 = 8;
          if ( v254 )
            v200 = v254;
          *v353 = (__int64 *)((char *)*v353 + 1);
          v209 = (v208 >> 1) + 1;
          if ( !v196 )
          {
            v204 = *(_DWORD *)(v193 + 24);
            goto LABEL_318;
          }
LABEL_319:
          if ( 4 * v209 >= 3 * v210 )
          {
            sub_116FE70(v193, 2 * v210);
            if ( (*(_BYTE *)(v193 + 8) & 1) != 0 )
            {
              v282 = v193 + 16;
              v283 = 8;
            }
            else
            {
              v283 = *(_DWORD *)(v193 + 24);
              v282 = *(_QWORD *)(v193 + 16);
              if ( !v283 )
              {
LABEL_547:
                *(_DWORD *)(v193 + 8) = (2 * (*(_DWORD *)(v193 + 8) >> 1) + 2) | *(_DWORD *)(v193 + 8) & 1;
                BUG();
              }
            }
            v284 = v283 - 1;
            v277 = 0;
            v285 = 1;
            for ( k = v284 & (((unsigned int)v195 >> 9) ^ ((unsigned int)v195 >> 4)); ; k = v284 & v293 )
            {
              v200 = (__int64 **)(v282 + 16LL * k);
              v287 = *v200;
              if ( v195 == *v200 )
                break;
              if ( v287 == (__int64 *)-4096LL )
              {
LABEL_429:
                if ( v277 )
                  v200 = v277;
                goto LABEL_420;
              }
              if ( v277 || v287 != (__int64 *)-8192LL )
                v200 = v277;
              v293 = v285 + k;
              v277 = v200;
              ++v285;
            }
            goto LABEL_420;
          }
          if ( v210 - *(_DWORD *)(v193 + 12) - v209 <= v210 >> 3 )
          {
            sub_116FE70(v193, v210);
            if ( (*(_BYTE *)(v193 + 8) & 1) != 0 )
            {
              v274 = v193 + 16;
              v275 = 8;
            }
            else
            {
              v275 = *(_DWORD *)(v193 + 24);
              v274 = *(_QWORD *)(v193 + 16);
              if ( !v275 )
                goto LABEL_547;
            }
            v276 = v275 - 1;
            v277 = 0;
            v278 = 1;
            for ( m = v276 & (((unsigned int)v195 >> 9) ^ ((unsigned int)v195 >> 4)); ; m = v276 & v295 )
            {
              v200 = (__int64 **)(v274 + 16LL * m);
              v280 = *v200;
              if ( v195 == *v200 )
                break;
              if ( v280 == (__int64 *)-4096LL )
                goto LABEL_429;
              if ( v277 || v280 != (__int64 *)-8192LL )
                v200 = v277;
              v295 = v278 + m;
              v277 = v200;
              ++v278;
            }
LABEL_420:
            v208 = *(_DWORD *)(v193 + 8);
          }
          *(_DWORD *)(v193 + 8) = (2 * (v208 >> 1) + 2) | v208 & 1;
          if ( *v200 != (__int64 *)-4096LL )
            --*(_DWORD *)(v193 + 12);
          *v200 = v195;
          v200[1] = 0;
LABEL_305:
          v200[1] = v194;
          v193 = (__int64)v354;
          v202 = v354->m128i_i8[8] & 1;
          if ( v202 )
          {
            v184 = v354 + 1;
            v185 = 7;
          }
          else
          {
            v203 = v354[1].m128i_u32[2];
            v184 = (__m128i *)v354[1].m128i_i64[0];
            if ( !v203 )
            {
              v205 = v354->m128i_u32[2];
              ++v354->m128i_i64[0];
              v187 = 0;
              v206 = (v205 >> 1) + 1;
              goto LABEL_311;
            }
            v185 = v203 - 1;
          }
          LODWORD(v186) = v185 & (((unsigned int)v194 >> 9) ^ ((unsigned int)v194 >> 4));
          v187 = &v184[(unsigned int)v186];
          v188 = v187->m128i_i64[0];
          if ( v194 != (__int64 *)v187->m128i_i64[0] )
          {
            v259 = 0;
            v260 = 1;
            while ( v188 != -4096 )
            {
              if ( !v259 && v188 == -8192 )
                v259 = v187;
              v186 = v185 & (unsigned int)(v186 + v260);
              v187 = &v184[v186];
              v188 = v187->m128i_i64[0];
              if ( v194 == (__int64 *)v187->m128i_i64[0] )
                goto LABEL_298;
              ++v260;
            }
            v205 = v354->m128i_u32[2];
            if ( v259 )
              v187 = v259;
            ++v354->m128i_i64[0];
            v207 = 8;
            v206 = (v205 >> 1) + 1;
            if ( !v202 )
            {
              v203 = *(_DWORD *)(v193 + 24);
LABEL_311:
              v207 = v203;
            }
            if ( 4 * v206 >= 3 * v207 )
            {
              sub_FB9E50(v193, 2 * v207);
              if ( (*(_BYTE *)(v193 + 8) & 1) != 0 )
              {
                v268 = v193 + 16;
                v269 = 8;
              }
              else
              {
                v269 = *(_DWORD *)(v193 + 24);
                v268 = *(_QWORD *)(v193 + 16);
                if ( !v269 )
                  goto LABEL_547;
              }
              v270 = v269 - 1;
              v264 = 0;
              v271 = 1;
              for ( n = v270 & (((unsigned int)v194 >> 9) ^ ((unsigned int)v194 >> 4)); ; n = v270 & v303 )
              {
                v187 = (__m128i *)(v268 + 16LL * n);
                v273 = v187->m128i_i64[0];
                if ( v194 == (__int64 *)v187->m128i_i64[0] )
                  break;
                if ( v273 == -4096 )
                {
LABEL_414:
                  if ( v264 )
                    v187 = v264;
                  break;
                }
                if ( v273 != -8192 || v264 )
                  v187 = v264;
                v303 = v271 + n;
                v264 = v187;
                ++v271;
              }
            }
            else
            {
              if ( v207 - *(_DWORD *)(v193 + 12) - v206 > v207 >> 3 )
              {
LABEL_314:
                *(_DWORD *)(v193 + 8) = (2 * (v205 >> 1) + 2) | v205 & 1;
                if ( v187->m128i_i64[0] != -4096 )
                  --*(_DWORD *)(v193 + 12);
                v187->m128i_i64[0] = (__int64)v194;
                v189 = &v187->m128i_i32[2];
                v187->m128i_i32[2] = 0;
                v190 = 1;
                goto LABEL_299;
              }
              sub_FB9E50(v193, v207);
              if ( (*(_BYTE *)(v193 + 8) & 1) != 0 )
              {
                v261 = v193 + 16;
                v262 = 8;
              }
              else
              {
                v262 = *(_DWORD *)(v193 + 24);
                v261 = *(_QWORD *)(v193 + 16);
                if ( !v262 )
                  goto LABEL_547;
              }
              v263 = v262 - 1;
              v264 = 0;
              v265 = 1;
              for ( ii = v263 & (((unsigned int)v194 >> 9) ^ ((unsigned int)v194 >> 4)); ; ii = v263 & v294 )
              {
                v187 = (__m128i *)(v261 + 16LL * ii);
                v267 = v187->m128i_i64[0];
                if ( v194 == (__int64 *)v187->m128i_i64[0] )
                  break;
                if ( v267 == -4096 )
                  goto LABEL_414;
                if ( v267 != -8192 || v264 )
                  v187 = v264;
                v294 = v265 + ii;
                v264 = v187;
                ++v265;
              }
            }
            v205 = *(_DWORD *)(v193 + 8);
            goto LABEL_314;
          }
LABEL_298:
          v189 = &v187->m128i_i32[2];
          v190 = v187->m128i_i32[2] + 1;
LABEL_299:
          *v189 = v190;
          if ( v331 == v182 )
          {
            v7 = v322;
            v6 = v319;
            goto LABEL_98;
          }
        }
      }
LABEL_548:
      v368.m128i_i32[2] = (2 * ((unsigned __int32)v368.m128i_i32[2] >> 1) + 2) | v368.m128i_i8[8] & 1;
      BUG();
    }
    sub_FB9E50((__int64)&v368, 2 * v256);
    if ( (v368.m128i_i8[8] & 1) != 0 )
    {
      v304 = &v369;
      v305 = 8;
    }
    else
    {
      v305 = v369.m128i_i32[2];
      v304 = (__m128i *)v369.m128i_i64[0];
      if ( !v369.m128i_i32[2] )
        goto LABEL_548;
    }
    v306 = v305 - 1;
    v299 = 0;
    v307 = 1;
    for ( jj = v306 & (((unsigned int)v172 >> 9) ^ ((unsigned int)v172 >> 4)); ; jj = v306 & v311 )
    {
      v179 = &v304[jj];
      v309 = v179->m128i_i64[0];
      if ( v172 == v179->m128i_i64[0] )
        break;
      if ( v309 == -4096 )
      {
LABEL_522:
        if ( v299 )
          v179 = v299;
        goto LABEL_503;
      }
      if ( v309 != -8192 || v299 )
        v179 = v299;
      v311 = v307 + jj;
      v299 = v179;
      ++v307;
    }
    goto LABEL_503;
  }
  if ( (*(_DWORD *)(v75 - 20) & 0x7FFFFFF) != 1 )
  {
    v77 = *(_QWORD *)(v75 - 56);
    v343 = *(_QWORD *)(v75 - 120);
    v78 = sub_ACD6D0(v68);
    sub_11702B0((__int64 *)&v353, v78, v77);
    v79 = *(_QWORD *)(v325 - 88);
    v80 = sub_ACD720(v68);
    sub_11702B0((__int64 *)&v353, v80, v79);
    goto LABEL_98;
  }
LABEL_99:
  if ( (v368.m128i_i8[8] & 1) == 0 )
    sub_C7D6A0(v369.m128i_i64[0], 16LL * v369.m128i_u32[2], 8);
  if ( (v364 & 1) == 0 )
    sub_C7D6A0(v365, 16LL * v366, 8);
  if ( !v350 )
    goto LABEL_15;
  if ( !*(_QWORD *)(v7 + 16) )
    return 0;
  sub_10A5FE0(v6[5], v7);
  if ( v7 == v350 )
    v350 = sub_ACADE0(*(__int64 ***)(v7 + 8));
  if ( !*(_QWORD *)(v350 + 16)
    && *(_BYTE *)v350 > 0x1Cu
    && (*(_BYTE *)(v350 + 7) & 0x10) == 0
    && (*(_BYTE *)(v7 + 7) & 0x10) != 0 )
  {
    sub_BD6B90((unsigned __int8 *)v350, (unsigned __int8 *)v7);
  }
  sub_BD84D0(v7, v350);
  return (unsigned __int8 *)v7;
}
