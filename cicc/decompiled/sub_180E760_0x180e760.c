// Function: sub_180E760
// Address: 0x180e760
//
__int64 __fastcall sub_180E760(
        __int64 *a1,
        __m128 a2,
        __m128 a3,
        __m128i a4,
        __m128i a5,
        double a6,
        double a7,
        __m128i a8,
        __m128 a9)
{
  __int64 v9; // rbx
  unsigned __int8 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __m128 *v23; // rdi
  __int64 *v24; // rsi
  __int64 v25; // rcx
  __int64 v26; // rdx
  unsigned __int64 v27; // r14
  __int64 v28; // rax
  __int64 v29; // rcx
  char v30; // si
  int v31; // r9d
  __int64 v32; // rax
  unsigned __int64 v33; // r12
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rax
  char v37; // cl
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // rax
  __int64 v41; // r12
  __int64 v42; // r14
  __int64 v43; // r15
  __int64 v44; // r8
  __int64 v45; // rax
  unsigned __int64 v46; // r8
  unsigned __int64 v47; // r15
  bool v48; // al
  char v49; // al
  __int64 v50; // rax
  unsigned int v51; // eax
  char v52; // al
  __int64 v53; // r8
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  _QWORD *v57; // r12
  __int64 v58; // rsi
  __int64 v59; // rax
  __int64 v60; // r13
  __int64 v61; // r12
  __int64 v62; // r14
  __int64 v63; // rbx
  _QWORD *v64; // r14
  unsigned int v65; // r13d
  char v66; // al
  __int64 v67; // rdx
  __int64 v68; // rax
  __int64 v69; // r8
  __int64 v70; // rax
  __int64 v71; // rsi
  char v72; // al
  char v73; // r8
  bool v74; // al
  double v75; // xmm4_8
  double v76; // xmm5_8
  __int64 v77; // rax
  unsigned __int8 *v78; // rsi
  int v79; // r8d
  int v80; // r9d
  __int64 v81; // r12
  __int64 v82; // r14
  __int64 v83; // r13
  _QWORD *v84; // rdi
  _QWORD *v85; // rdi
  unsigned __int64 v86; // rdx
  __int64 *v87; // r13
  __int64 v88; // rax
  __int64 v89; // r14
  __int64 v90; // rax
  unsigned __int64 v91; // r15
  int v92; // r8d
  int v93; // r9d
  __int16 v94; // cx
  __int64 v95; // rax
  __m128 *v96; // rax
  __int64 v97; // r12
  unsigned int v99; // edx
  _QWORD *v100; // rax
  _QWORD *v101; // r15
  unsigned __int64 v102; // rsi
  __int64 v103; // rax
  __int64 v104; // rsi
  __int64 v105; // rdx
  unsigned __int8 *v106; // rsi
  double v107; // xmm4_8
  double v108; // xmm5_8
  __int64 v109; // r14
  __int64 v110; // rax
  __int64 *v111; // rax
  _QWORD *v112; // rcx
  __m128i *v113; // rdx
  char v114; // cl
  void *p_s; // rsi
  int v116; // ecx
  unsigned __int64 v117; // rax
  unsigned __int8 *v118; // r12
  unsigned __int8 *v119; // r14
  unsigned int v120; // esi
  __int64 v121; // rdi
  __int64 v122; // rcx
  unsigned int v123; // edx
  _QWORD *v124; // rax
  __int64 v125; // r8
  int v126; // edx
  __int64 *v127; // rax
  __int64 *v128; // r12
  __int64 *v129; // r15
  unsigned int v130; // edx
  __int64 *v131; // rax
  __int64 v132; // r10
  __int64 v133; // rbx
  __int64 v134; // r13
  __int64 v135; // rax
  __int64 v136; // rsi
  __int64 v137; // rcx
  int v138; // esi
  __int64 *v139; // r9
  int v140; // edx
  __int64 v141; // rax
  char v142; // r12
  unsigned __int8 v143; // r15
  __int64 v144; // r14
  _QWORD *v145; // rsi
  __int64 v146; // r12
  __int64 v147; // rax
  __int64 v148; // rax
  __int64 v149; // r13
  __int64 **v150; // rax
  __int64 v151; // rdx
  __int64 v152; // rcx
  __int64 v153; // rax
  __int64 v154; // r12
  _BYTE *v155; // rax
  __int64 v156; // r13
  __int64 v157; // r12
  __int64 v158; // rsi
  int v159; // edx
  unsigned __int64 v160; // rax
  __int64 v161; // rdi
  __int64 v162; // rdx
  unsigned __int8 *v163; // rsi
  __int64 v164; // rax
  __int64 v165; // rax
  __int64 **v166; // rdi
  _BYTE *v167; // r13
  __int64 v168; // rdx
  __int64 v169; // rcx
  __int64 v170; // rax
  __int64 v171; // r13
  __int64 v172; // r12
  __int64 v173; // rsi
  unsigned __int8 *v174; // rsi
  unsigned __int8 *v175; // rsi
  unsigned __int8 *v176; // r12
  __int64 v177; // r9
  double v178; // xmm4_8
  double v179; // xmm5_8
  __int64 ***v180; // r13
  __int64 v181; // rdi
  __int64 **v182; // r15
  __int64 v183; // rax
  __int64 v184; // rax
  __int64 v185; // r9
  __int64 v186; // rax
  __int64 v187; // rsi
  __int64 v188; // rsi
  __int64 v189; // rdx
  unsigned __int8 *v190; // rsi
  __int64 v191; // rcx
  __int64 v192; // rax
  __int64 v193; // rdi
  __int64 v194; // r12
  __int64 v195; // rax
  __int64 v196; // rax
  __int64 v197; // r12
  _QWORD *v198; // rax
  __int64 v199; // rdx
  __int64 v200; // rax
  __int64 v201; // rax
  __int64 v202; // rdi
  __int64 v203; // r12
  __int64 v204; // rax
  __int64 v205; // rax
  __int64 v206; // rax
  __int64 v207; // r12
  __int64 v208; // rax
  double v209; // xmm0_8
  __int64 v210; // rcx
  int v211; // r9d
  int v212; // r8d
  size_t v213; // r12
  __int64 *v214; // rdx
  __int64 *v215; // r15
  __int64 **v216; // rdi
  __int64 v217; // rax
  __int64 v218; // rax
  __int64 v219; // r14
  _QWORD *v220; // rax
  unsigned __int8 *v221; // rsi
  __int64 v222; // rdi
  unsigned __int64 v223; // rax
  unsigned __int64 v224; // r9
  unsigned __int64 v225; // r8
  __int64 v226; // rsi
  __int64 v227; // rax
  __int64 v228; // rax
  __int64 v229; // rax
  __int64 v230; // rcx
  __int64 v231; // rax
  __int64 v232; // rax
  __int64 **v233; // rax
  __int64 v234; // rdx
  __int64 v235; // rcx
  __int64 v236; // rax
  __int64 v237; // rax
  unsigned __int8 *v238; // rsi
  __int64 v239; // r14
  __int64 v240; // rax
  unsigned __int8 *v241; // rsi
  __int64 v242; // rsi
  __int64 v243; // rdx
  __int64 v244; // rdi
  unsigned int v245; // eax
  unsigned int v246; // ecx
  _QWORD **v247; // r12
  _QWORD **v248; // rbx
  _QWORD *v249; // rdi
  int v250; // r11d
  _QWORD *v251; // r10
  int v252; // r13d
  unsigned __int64 *v253; // rdi
  __int64 v254; // rax
  __int64 v255; // r12
  __int64 v256; // rdx
  _QWORD *v257; // rax
  __int64 v258; // r8
  __int64 v259; // rax
  __int64 v260; // r14
  unsigned __int64 v261; // r15
  __m128 *v262; // rdx
  int v263; // esi
  _QWORD *v264; // rdi
  int v265; // edx
  __int64 v266; // rax
  __int64 v267; // rax
  int v268; // r14d
  __int64 *v269; // r11
  int v270; // [rsp-8h] [rbp-828h]
  __int64 *v271; // [rsp+8h] [rbp-818h]
  __int64 v272; // [rsp+8h] [rbp-818h]
  __int64 v273; // [rsp+30h] [rbp-7F0h]
  __int64 v274; // [rsp+30h] [rbp-7F0h]
  __int64 v275; // [rsp+30h] [rbp-7F0h]
  __int64 v276; // [rsp+30h] [rbp-7F0h]
  int v277; // [rsp+30h] [rbp-7F0h]
  unsigned __int64 v278; // [rsp+38h] [rbp-7E8h]
  __int64 v279; // [rsp+40h] [rbp-7E0h]
  unsigned __int8 *v280; // [rsp+40h] [rbp-7E0h]
  unsigned __int8 *v281; // [rsp+48h] [rbp-7D8h]
  __int64 *j; // [rsp+50h] [rbp-7D0h]
  unsigned int v283; // [rsp+50h] [rbp-7D0h]
  int v284; // [rsp+58h] [rbp-7C8h]
  unsigned __int8 v285; // [rsp+5Fh] [rbp-7C1h]
  __int64 v286; // [rsp+60h] [rbp-7C0h]
  __int64 v287; // [rsp+60h] [rbp-7C0h]
  __int64 v288; // [rsp+68h] [rbp-7B8h]
  char v289; // [rsp+68h] [rbp-7B8h]
  __int64 v290; // [rsp+70h] [rbp-7B0h]
  unsigned int v291; // [rsp+70h] [rbp-7B0h]
  unsigned __int64 *v292; // [rsp+70h] [rbp-7B0h]
  unsigned __int64 v293; // [rsp+70h] [rbp-7B0h]
  __int64 v294; // [rsp+70h] [rbp-7B0h]
  __int64 v295; // [rsp+70h] [rbp-7B0h]
  __int64 v296; // [rsp+70h] [rbp-7B0h]
  _QWORD *v297; // [rsp+78h] [rbp-7A8h]
  __int64 v298; // [rsp+78h] [rbp-7A8h]
  __int64 v299; // [rsp+78h] [rbp-7A8h]
  __int64 v300; // [rsp+78h] [rbp-7A8h]
  int v301; // [rsp+80h] [rbp-7A0h]
  unsigned __int64 v302; // [rsp+80h] [rbp-7A0h]
  __int64 *v303; // [rsp+88h] [rbp-798h]
  __int64 v304; // [rsp+98h] [rbp-788h]
  __int64 v305; // [rsp+98h] [rbp-788h]
  unsigned __int8 *v306; // [rsp+A8h] [rbp-778h] BYREF
  __int64 v307; // [rsp+B0h] [rbp-770h] BYREF
  __int64 v308; // [rsp+B8h] [rbp-768h] BYREF
  _QWORD v309[2]; // [rsp+C0h] [rbp-760h] BYREF
  unsigned __int64 v310; // [rsp+D0h] [rbp-750h]
  unsigned __int8 *v311[2]; // [rsp+E0h] [rbp-740h] BYREF
  __int16 v312; // [rsp+F0h] [rbp-730h]
  __int64 v313; // [rsp+100h] [rbp-720h] BYREF
  __int64 v314; // [rsp+108h] [rbp-718h]
  __int64 v315; // [rsp+110h] [rbp-710h]
  unsigned int v316; // [rsp+118h] [rbp-708h]
  unsigned __int8 *v317; // [rsp+120h] [rbp-700h] BYREF
  __int64 v318; // [rsp+128h] [rbp-6F8h]
  __int64 *v319; // [rsp+130h] [rbp-6F0h]
  _QWORD *v320; // [rsp+138h] [rbp-6E8h]
  __int64 v321[5]; // [rsp+170h] [rbp-6B0h] BYREF
  int v322; // [rsp+198h] [rbp-688h]
  __int64 v323; // [rsp+1A0h] [rbp-680h]
  __int64 v324; // [rsp+1A8h] [rbp-678h]
  const char *v325; // [rsp+1C0h] [rbp-660h] BYREF
  __int64 v326; // [rsp+1C8h] [rbp-658h]
  __int64 v327; // [rsp+1D0h] [rbp-650h]
  _QWORD *v328; // [rsp+1D8h] [rbp-648h]
  __int64 v329; // [rsp+1E0h] [rbp-640h]
  int v330; // [rsp+1E8h] [rbp-638h]
  __int64 v331; // [rsp+1F0h] [rbp-630h]
  __int64 v332; // [rsp+1F8h] [rbp-628h]
  unsigned __int8 *v333[2]; // [rsp+210h] [rbp-610h] BYREF
  __int64 v334; // [rsp+220h] [rbp-600h]
  __int64 v335; // [rsp+228h] [rbp-5F8h]
  __int64 v336; // [rsp+230h] [rbp-5F0h]
  int v337; // [rsp+238h] [rbp-5E8h]
  __int64 v338; // [rsp+240h] [rbp-5E0h]
  __int64 v339; // [rsp+248h] [rbp-5D8h]
  __m128i v340[8]; // [rsp+260h] [rbp-5C0h] BYREF
  void *s; // [rsp+2E0h] [rbp-540h] BYREF
  __int64 v342; // [rsp+2E8h] [rbp-538h]
  unsigned __int64 v343[2]; // [rsp+2F0h] [rbp-530h] BYREF
  int v344; // [rsp+300h] [rbp-520h]
  _QWORD v345[8]; // [rsp+308h] [rbp-518h] BYREF
  __int64 v346; // [rsp+348h] [rbp-4D8h] BYREF
  __int64 v347; // [rsp+350h] [rbp-4D0h]
  unsigned __int64 v348; // [rsp+358h] [rbp-4C8h]
  __m128 v349; // [rsp+360h] [rbp-4C0h] BYREF
  unsigned __int64 v350[3]; // [rsp+370h] [rbp-4B0h] BYREF
  _BYTE v351[64]; // [rsp+388h] [rbp-498h] BYREF
  __int64 v352; // [rsp+3C8h] [rbp-458h]
  __int64 v353; // [rsp+3D0h] [rbp-450h]
  unsigned __int64 v354; // [rsp+3D8h] [rbp-448h]
  size_t n[2]; // [rsp+3E0h] [rbp-440h] BYREF
  __m128i v356; // [rsp+3F0h] [rbp-430h] BYREF
  __m128i v357; // [rsp+400h] [rbp-420h] BYREF
  unsigned __int64 v358; // [rsp+410h] [rbp-410h]
  __int64 v359; // [rsp+448h] [rbp-3D8h]
  __int64 i; // [rsp+450h] [rbp-3D0h]
  __int64 v361; // [rsp+458h] [rbp-3C8h]
  unsigned __int8 *v362; // [rsp+460h] [rbp-3C0h] BYREF
  __int64 v363; // [rsp+468h] [rbp-3B8h]
  unsigned __int64 *v364; // [rsp+470h] [rbp-3B0h] BYREF
  _QWORD *v365; // [rsp+478h] [rbp-3A8h]
  __int64 v366; // [rsp+480h] [rbp-3A0h]
  int v367; // [rsp+488h] [rbp-398h] BYREF
  __int64 v368; // [rsp+490h] [rbp-390h]
  __int64 v369; // [rsp+498h] [rbp-388h]
  __int64 v370; // [rsp+4C8h] [rbp-358h]
  __int64 v371; // [rsp+4D0h] [rbp-350h]
  unsigned __int64 v372; // [rsp+4D8h] [rbp-348h]
  _QWORD v373[2]; // [rsp+4E0h] [rbp-340h] BYREF
  unsigned __int64 v374; // [rsp+4F0h] [rbp-330h]
  char v375[64]; // [rsp+508h] [rbp-318h] BYREF
  __int64 v376; // [rsp+548h] [rbp-2D8h]
  __int64 v377; // [rsp+550h] [rbp-2D0h]
  __int64 v378; // [rsp+558h] [rbp-2C8h]

  v285 = byte_4FA8580;
  if ( !byte_4FA8580 )
    return 0;
  v9 = (__int64)a1;
  if ( byte_4FA82E0 )
  {
    v56 = *(_QWORD *)(*a1 + 80);
    if ( !v56 )
      BUG();
    v57 = *(_QWORD **)(v56 + 24);
    if ( v57 )
      v57 -= 3;
    if ( *(_QWORD **)(a1[1] + 720) == v57 )
    {
      v267 = v57[4];
      if ( v267 == v57[5] + 40LL || !v267 )
      {
        v362 = 0;
        v364 = 0;
        v365 = (_QWORD *)sub_16498A0(0);
        v366 = 0;
        v367 = 0;
        v368 = 0;
        v369 = 0;
        v363 = 0;
        BUG();
      }
      v57 = (_QWORD *)(v267 - 24);
    }
    v362 = 0;
    v365 = (_QWORD *)sub_16498A0((__int64)v57);
    v366 = 0;
    v367 = 0;
    v368 = 0;
    v369 = 0;
    v363 = v57[5];
    v364 = v57 + 3;
    v58 = v57[6];
    n[0] = v58;
    if ( v58 )
    {
      sub_1623A60((__int64)n, v58, 2);
      if ( v362 )
        sub_161E7C0((__int64)&v362, (__int64)v362);
      v58 = n[0];
      v362 = (unsigned __int8 *)n[0];
      if ( n[0] )
        sub_1623210((__int64)n, (unsigned __int8 *)n[0], (__int64)&v362);
    }
    v59 = sub_1632FA0(*(_QWORD *)(*a1 + 40));
    v60 = *a1;
    v305 = v59;
    if ( (*(_BYTE *)(*a1 + 18) & 1) != 0 )
    {
      sub_15E08E0(v60, v58);
      v61 = *(_QWORD *)(v60 + 88);
      v62 = v61 + 40LL * *(_QWORD *)(v60 + 96);
      if ( (*(_BYTE *)(v60 + 18) & 1) != 0 )
      {
        sub_15E08E0(v60, v58);
        v61 = *(_QWORD *)(v60 + 88);
      }
    }
    else
    {
      v61 = *(_QWORD *)(v60 + 88);
      v62 = v61 + 40LL * *(_QWORD *)(v60 + 96);
    }
    if ( v61 == v62 )
    {
LABEL_176:
      if ( v362 )
        sub_161E7C0((__int64)&v362, (__int64)v362);
      goto LABEL_3;
    }
    v63 = v62;
    while ( 1 )
    {
      while ( !(unsigned __int8)sub_15E0450(v61) )
      {
        v61 += 40;
        if ( v63 == v61 )
          goto LABEL_175;
      }
      v64 = **(_QWORD ***)(*(_QWORD *)v61 + 16LL);
      v65 = sub_15E0370(v61);
      if ( !v65 )
        v65 = sub_15A9FE0(v305, (__int64)v64);
      s = ".byval";
      LOWORD(v343[0]) = 259;
      if ( (*(_BYTE *)(v61 + 23) & 0x20) == 0 )
        break;
      v325 = sub_1649960(v61);
      v340[1].m128i_i16[0] = 261;
      v340[0].m128i_i64[0] = (__int64)&v325;
      v66 = v343[0];
      v326 = v67;
      if ( LOBYTE(v343[0]) )
      {
        if ( LOBYTE(v343[0]) != 1 )
        {
          v113 = (__m128i *)v340[0].m128i_i64[0];
          v114 = 5;
          goto LABEL_179;
        }
        a8 = _mm_loadu_si128(v340);
        v349 = (__m128)a8;
        v350[0] = v340[1].m128i_u64[0];
      }
      else
      {
        LOWORD(v350[0]) = 256;
      }
LABEL_165:
      v99 = *(_DWORD *)(sub_1632FA0(*(_QWORD *)(*(_QWORD *)(v363 + 56) + 40LL)) + 4);
      v356.m128i_i16[0] = 257;
      v291 = v99;
      v100 = sub_1648A60(64, 1u);
      v101 = v100;
      if ( v100 )
        sub_15F8BC0((__int64)v100, v64, v291, 0, (__int64)n, 0);
      if ( v363 )
      {
        v292 = v364;
        sub_157E9D0(v363 + 40, (__int64)v101);
        v102 = *v292;
        v103 = v101[3] & 7LL;
        v101[4] = v292;
        v102 &= 0xFFFFFFFFFFFFFFF8LL;
        v101[3] = v102 | v103;
        *(_QWORD *)(v102 + 8) = v101 + 3;
        *v292 = *v292 & 7 | (unsigned __int64)(v101 + 3);
      }
      sub_164B780((__int64)v101, (__int64 *)&v349);
      if ( v362 )
      {
        v321[0] = (__int64)v362;
        sub_1623A60((__int64)v321, (__int64)v362, 2);
        v104 = v101[6];
        v105 = (__int64)(v101 + 6);
        if ( v104 )
        {
          sub_161E7C0((__int64)(v101 + 6), v104);
          v105 = (__int64)(v101 + 6);
        }
        v106 = (unsigned __int8 *)v321[0];
        v101[6] = v321[0];
        if ( v106 )
          sub_1623210((__int64)v321, v106, v105);
      }
      sub_15F8A20((__int64)v101, v65);
      sub_164D160(
        v61,
        (__int64)v101,
        a2,
        *(double *)a3.m128_u64,
        *(double *)a4.m128i_i64,
        *(double *)a5.m128i_i64,
        v107,
        v108,
        *(double *)a8.m128i_i64,
        a9);
      v293 = (unsigned int)sub_15A9FE0(v305, (__int64)v64);
      v109 = v293 * ((v293 + ((unsigned __int64)(sub_127FA20(v305, (__int64)v64) + 7) >> 3) - 1) / v293);
      v110 = sub_1643360(v365);
      v111 = (__int64 *)sub_159C470(v110, v109, 0);
      v112 = (_QWORD *)v61;
      v61 += 40;
      sub_15E7430((__int64 *)&v362, v101, v65, v112, v65, v111, 0, 0, 0, 0, 0);
      if ( v63 == v61 )
      {
LABEL_175:
        v9 = (__int64)a1;
        goto LABEL_176;
      }
    }
    v113 = v340;
    v114 = 2;
    LODWORD(v333[0]) = *(_DWORD *)(v61 + 32);
    v340[0].m128i_i64[0] = (__int64)"Arg";
    v340[0].m128i_i64[1] = (__int64)v333[0];
    v340[1].m128i_i16[0] = 2307;
    v66 = 3;
LABEL_179:
    p_s = s;
    if ( BYTE1(v343[0]) != 1 )
    {
      p_s = &s;
      v66 = 2;
    }
    v349.m128_u64[0] = (unsigned __int64)v113;
    v349.m128_u64[1] = (unsigned __int64)p_s;
    LOBYTE(v350[0]) = v114;
    BYTE1(v350[0]) = v66;
    goto LABEL_165;
  }
LABEL_3:
  v10 = *(unsigned __int8 **)(*(_QWORD *)v9 + 80LL);
  LOBYTE(v365) = 0;
  v346 = 0;
  v347 = 0;
  v348 = 0;
  if ( v10 )
    v10 -= 24;
  memset(v340, 0, sizeof(v340));
  v340[1].m128i_i32[2] = 8;
  v340[0].m128i_i64[1] = (__int64)&v340[2].m128i_i64[1];
  v340[1].m128i_i64[0] = (__int64)&v340[2].m128i_i64[1];
  v342 = (__int64)v345;
  v343[0] = (unsigned __int64)v345;
  v343[1] = 0x100000008LL;
  v345[0] = v10;
  v362 = v10;
  v344 = 0;
  s = (void *)1;
  sub_144A690(&v346, (__int64)&v362);
  sub_16CCEE0(n, (__int64)&v357.m128i_i64[1], 8, (__int64)v340);
  v11 = v340[6].m128i_i64[1];
  v340[6].m128i_i64[1] = 0;
  v359 = v11;
  v12 = v340[7].m128i_i64[0];
  v340[7].m128i_i64[0] = 0;
  i = v12;
  v13 = v340[7].m128i_i64[1];
  v340[7].m128i_i64[1] = 0;
  v361 = v13;
  sub_16CCEE0(&v349, (__int64)v351, 8, (__int64)&s);
  v14 = v346;
  v346 = 0;
  v352 = v14;
  v15 = v347;
  v347 = 0;
  v353 = v15;
  v16 = v348;
  v348 = 0;
  v354 = v16;
  sub_16CCEE0(&v362, (__int64)&v367, 8, (__int64)&v349);
  v17 = v352;
  v352 = 0;
  v370 = v17;
  v18 = v353;
  v353 = 0;
  v371 = v18;
  v19 = v354;
  v354 = 0;
  v372 = v19;
  sub_16CCEE0(v373, (__int64)v375, 8, (__int64)n);
  v20 = v359;
  v359 = 0;
  v376 = v20;
  v21 = i;
  i = 0;
  v377 = v21;
  v22 = v361;
  v361 = 0;
  v378 = v22;
  if ( v352 )
    j_j___libc_free_0(v352, v354 - v352);
  if ( v350[0] != v349.m128_u64[1] )
    _libc_free(v350[0]);
  if ( v359 )
    j_j___libc_free_0(v359, v361 - v359);
  if ( v356.m128i_i64[0] != n[1] )
    _libc_free(v356.m128i_u64[0]);
  if ( v346 )
    j_j___libc_free_0(v346, v348 - v346);
  if ( v343[0] != v342 )
    _libc_free(v343[0]);
  if ( v340[6].m128i_i64[1] )
    j_j___libc_free_0(v340[6].m128i_i64[1], v340[7].m128i_i64[1] - v340[6].m128i_i64[1]);
  if ( v340[1].m128i_i64[0] != v340[0].m128i_i64[1] )
    _libc_free(v340[1].m128i_u64[0]);
  v23 = &v349;
  v24 = (__int64 *)v351;
  sub_16CCCB0(&v349, (__int64)v351, (__int64)&v362);
  v25 = v371;
  v26 = v370;
  v352 = 0;
  v353 = 0;
  v354 = 0;
  v27 = v371 - v370;
  if ( v371 == v370 )
  {
    v27 = 0;
    v28 = 0;
  }
  else
  {
    if ( v27 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_414;
    v28 = sub_22077B0(v371 - v370);
    v25 = v371;
    v26 = v370;
  }
  v352 = v28;
  v353 = v28;
  v354 = v28 + v27;
  if ( v26 == v25 )
  {
    v29 = v28;
  }
  else
  {
    v29 = v28 + v25 - v26;
    do
    {
      if ( v28 )
      {
        *(_QWORD *)v28 = *(_QWORD *)v26;
        v30 = *(_BYTE *)(v26 + 24);
        *(_BYTE *)(v28 + 24) = v30;
        if ( v30 )
          *(__m128i *)(v28 + 8) = _mm_loadu_si128((const __m128i *)(v26 + 8));
      }
      v28 += 32;
      v26 += 32;
    }
    while ( v28 != v29 );
  }
  v23 = (__m128 *)n;
  v24 = &v357.m128i_i64[1];
  v353 = v29;
  sub_16CCCB0(n, (__int64)&v357.m128i_i64[1], (__int64)v373);
  v32 = v377;
  v26 = v376;
  v359 = 0;
  i = 0;
  v361 = 0;
  v33 = v377 - v376;
  if ( v377 == v376 )
  {
    v35 = 0;
    goto LABEL_33;
  }
  if ( v33 > 0x7FFFFFFFFFFFFFE0LL )
LABEL_414:
    sub_4261EA(v23, v24, v26);
  v34 = sub_22077B0(v377 - v376);
  v26 = v376;
  v35 = v34;
  v32 = v377;
LABEL_33:
  v359 = v35;
  i = v35;
  v361 = v35 + v33;
  if ( v26 == v32 )
  {
    v36 = v35;
  }
  else
  {
    v36 = v35 + v32 - v26;
    do
    {
      if ( v35 )
      {
        *(_QWORD *)v35 = *(_QWORD *)v26;
        v37 = *(_BYTE *)(v26 + 24);
        *(_BYTE *)(v35 + 24) = v37;
        if ( v37 )
          *(__m128i *)(v35 + 8) = _mm_loadu_si128((const __m128i *)(v26 + 8));
      }
      v35 += 32;
      v26 += 32;
    }
    while ( v35 != v36 );
    v35 = v359;
  }
  for ( i = v36; ; v36 = i )
  {
    v38 = v353;
    v39 = v352;
    if ( v353 - v352 != v36 - v35 )
      goto LABEL_42;
    if ( v352 == v353 )
      break;
    v71 = v35;
    while ( *(_QWORD *)v39 == *(_QWORD *)v71 )
    {
      v72 = *(_BYTE *)(v39 + 24);
      v73 = *(_BYTE *)(v71 + 24);
      if ( v72 && v73 )
        v74 = *(_DWORD *)(v39 + 16) == *(_DWORD *)(v71 + 16);
      else
        v74 = v72 == v73;
      if ( !v74 )
        break;
      v39 += 32;
      v71 += 32;
      if ( v353 == v39 )
        goto LABEL_110;
    }
LABEL_42:
    v40 = *(_QWORD *)(v353 - 32);
    v41 = *(_QWORD *)(v40 + 48);
    v42 = v40 + 40;
    if ( v40 + 40 != v41 )
    {
      while ( 1 )
      {
        v43 = v41;
        v41 = *(_QWORD *)(v41 + 8);
        v44 = v43 - 24;
        switch ( *(_BYTE *)(v43 - 8) )
        {
          case 0x18:
          case 0x1A:
          case 0x1B:
          case 0x1C:
          case 0x1F:
          case 0x21:
          case 0x22:
          case 0x23:
          case 0x24:
          case 0x25:
          case 0x26:
          case 0x27:
          case 0x28:
          case 0x29:
          case 0x2A:
          case 0x2B:
          case 0x2C:
          case 0x2D:
          case 0x2E:
          case 0x2F:
          case 0x30:
          case 0x31:
          case 0x32:
          case 0x33:
          case 0x34:
          case 0x36:
          case 0x37:
          case 0x38:
          case 0x39:
          case 0x3A:
          case 0x3B:
          case 0x3C:
          case 0x3D:
          case 0x3E:
          case 0x3F:
          case 0x40:
          case 0x41:
          case 0x42:
          case 0x43:
          case 0x44:
          case 0x45:
          case 0x46:
          case 0x47:
          case 0x48:
          case 0x49:
          case 0x4A:
          case 0x4B:
          case 0x4C:
          case 0x4D:
          case 0x4F:
          case 0x50:
          case 0x51:
          case 0x52:
          case 0x53:
          case 0x54:
          case 0x55:
          case 0x56:
          case 0x57:
          case 0x58:
            goto LABEL_47;
          case 0x19:
          case 0x1E:
          case 0x20:
            v45 = *(unsigned int *)(v9 + 824);
            if ( (unsigned int)v45 >= *(_DWORD *)(v9 + 828) )
            {
              sub_16CD150(v9 + 816, (const void *)(v9 + 832), 0, 8, v44, v31);
              v45 = *(unsigned int *)(v9 + 824);
              v44 = v43 - 24;
            }
            v38 = *(_QWORD *)(v9 + 816);
            *(_QWORD *)(v38 + 8 * v45) = v44;
            ++*(_DWORD *)(v9 + 824);
            goto LABEL_47;
          case 0x1D:
            goto LABEL_49;
          case 0x35:
            v304 = v43 - 24;
            if ( (unsigned __int8)sub_180D640(*(_QWORD *)(v9 + 8), v43 - 24) )
            {
              v51 = (unsigned int)(1 << *(_WORD *)(v43 - 6)) >> 1;
              if ( v51 < *(_DWORD *)(v9 + 896) )
                v51 = *(_DWORD *)(v9 + 896);
              *(_DWORD *)(v9 + 896) = v51;
              v52 = sub_15F8F00(v304);
              v53 = v43 - 24;
              if ( v52 )
              {
                v68 = *(unsigned int *)(v9 + 536);
                if ( (unsigned int)v68 >= *(_DWORD *)(v9 + 540) )
                {
                  sub_16CD150(v9 + 528, (const void *)(v9 + 544), 0, 8, v304, v31);
                  v68 = *(unsigned int *)(v9 + 536);
                  v53 = v43 - 24;
                }
                v38 = *(_QWORD *)(v9 + 528);
                *(_QWORD *)(v38 + 8 * v68) = v53;
                ++*(_DWORD *)(v9 + 536);
                if ( v42 == v41 )
                  goto LABEL_48;
              }
              else
              {
                v54 = *(unsigned int *)(v9 + 3712);
                if ( (unsigned int)v54 >= *(_DWORD *)(v9 + 3716) )
                {
                  sub_16CD150(v9 + 3704, (const void *)(v9 + 3720), 0, 8, v304, v31);
                  v54 = *(unsigned int *)(v9 + 3712);
                  v53 = v43 - 24;
                }
                v38 = *(_QWORD *)(v9 + 3704);
                *(_QWORD *)(v38 + 8 * v54) = v53;
                ++*(_DWORD *)(v9 + 3712);
                if ( v42 == v41 )
                  goto LABEL_48;
              }
            }
            else if ( (unsigned __int8)sub_15F8F00(v304) && (v69 = v43 - 24, *(_DWORD *)(v9 + 536)) )
            {
              v70 = *(unsigned int *)(v9 + 680);
              if ( (unsigned int)v70 >= *(_DWORD *)(v9 + 684) )
              {
                sub_16CD150(v9 + 672, (const void *)(v9 + 688), 0, 8, v304, v31);
                v70 = *(unsigned int *)(v9 + 680);
                v69 = v43 - 24;
              }
              v38 = *(_QWORD *)(v9 + 672);
              *(_QWORD *)(v38 + 8 * v70) = v69;
              ++*(_DWORD *)(v9 + 680);
              if ( v42 == v41 )
                goto LABEL_48;
            }
            else
            {
LABEL_47:
              if ( v42 == v41 )
                goto LABEL_48;
            }
            break;
          case 0x4E:
            v55 = *(_QWORD *)(v43 - 48);
            if ( *(_BYTE *)(v55 + 16) || !*(_DWORD *)(v55 + 36) )
            {
LABEL_49:
              v46 = v44 & 0xFFFFFFFFFFFFFFF8LL;
              v47 = v46;
              if ( *(_BYTE *)(v46 + 16) != 78 )
                goto LABEL_47;
              v48 = *(_BYTE *)(*(_QWORD *)(v46 - 24) + 16LL) == 20
                 && !sub_15F41F0(v46, *(_QWORD *)(v9 + 3808))
                 && *(_QWORD *)(*(_QWORD *)(v9 + 8) + 720LL) != v47;
              *(_BYTE *)(v9 + 3800) |= v48;
              v49 = sub_1560260((_QWORD *)(v47 + 56), -1, 39);
              if ( !v49 )
              {
                v50 = *(_QWORD *)(v47 - 24);
                if ( *(_BYTE *)(v50 + 16) )
                {
                  v49 = *(_BYTE *)(v9 + 3801);
                }
                else
                {
                  s = *(void **)(v50 + 112);
                  v49 = *(_BYTE *)(v9 + 3801) | sub_1560260(&s, -1, 39);
                }
              }
              *(_BYTE *)(v9 + 3801) = v49;
              if ( v42 == v41 )
                goto LABEL_48;
            }
            else
            {
              sub_180E4A0(v9, v43 - 24, v38, v39, v44, v31);
              if ( v42 == v41 )
                goto LABEL_48;
            }
            continue;
        }
      }
    }
LABEL_48:
    sub_17D3A30((__int64)&v349);
    v35 = v359;
  }
LABEL_110:
  if ( v35 )
    j_j___libc_free_0(v35, v361 - v35);
  if ( v356.m128i_i64[0] != n[1] )
    _libc_free(v356.m128i_u64[0]);
  if ( v352 )
    j_j___libc_free_0(v352, v354 - v352);
  if ( v350[0] != v349.m128_u64[1] )
    _libc_free(v350[0]);
  if ( v376 )
    j_j___libc_free_0(v376, v378 - v376);
  if ( v374 != v373[1] )
    _libc_free(v374);
  if ( v370 )
    j_j___libc_free_0(v370, v372 - v370);
  if ( v364 != (unsigned __int64 *)v363 )
    _libc_free((unsigned __int64)v364);
  if ( !*(_DWORD *)(v9 + 536) && !*(_DWORD *)(v9 + 3712) )
    return 0;
  sub_180C840((_QWORD *)v9, *(_QWORD *)(*(_QWORD *)v9 + 40LL));
  sub_1808150(
    v9,
    a2,
    *(double *)a3.m128_u64,
    *(double *)a4.m128i_i64,
    *(double *)a5.m128i_i64,
    v75,
    v76,
    *(double *)a8.m128i_i64,
    a9);
  if ( !*(_DWORD *)(v9 + 536) )
    return v285;
  v306 = 0;
  v77 = sub_1626D20(*(_QWORD *)v9);
  if ( v77 )
  {
    sub_15C7110(&v362, *(_DWORD *)(v77 + 28), 0, v77, 0);
    v306 = v362;
    if ( v362 )
    {
      sub_1623210((__int64)&v362, v362, (__int64)&v306);
      v362 = 0;
    }
    sub_17CD270((__int64 *)&v362);
  }
  v279 = **(_QWORD **)(v9 + 528);
  sub_17CE510((__int64)&v317, v279, 0, 0, 0);
  v362 = v306;
  if ( v306 )
  {
    sub_1623A60((__int64)&v362, (__int64)v306, 2);
    v78 = v317;
    if ( v317 )
      goto LABEL_135;
LABEL_136:
    v317 = v362;
    if ( v362 )
    {
      sub_1623210((__int64)&v362, v362, (__int64)&v317);
      v362 = 0;
    }
  }
  else
  {
    v78 = v317;
    if ( v317 )
    {
LABEL_135:
      sub_161E7C0((__int64)&v317, (__int64)v78);
      goto LABEL_136;
    }
  }
  sub_17CD270((__int64 *)&v362);
  v81 = *(_QWORD *)(v9 + 672);
  v82 = *(_QWORD *)(v279 + 40);
  v83 = v81 + 8LL * *(unsigned int *)(v9 + 680);
  while ( v83 != v81 )
  {
    while ( 1 )
    {
      v84 = *(_QWORD **)v81;
      if ( v82 == *(_QWORD *)(*(_QWORD *)v81 + 40LL) )
        break;
      v81 += 8;
      if ( v83 == v81 )
        goto LABEL_143;
    }
    v81 += 8;
    sub_15F22F0(v84, v279);
  }
LABEL_143:
  v85 = *(_QWORD **)(v9 + 3760);
  if ( v85 )
    sub_15F22F0(v85, v279);
  v86 = *(unsigned int *)(v9 + 536);
  v362 = (unsigned __int8 *)&v364;
  v363 = 0x1000000000LL;
  if ( (unsigned int)v86 > 0x10 )
  {
    sub_16CD150((__int64)&v362, &v364, v86, 56, v79, v80);
    v86 = *(unsigned int *)(v9 + 536);
  }
  v87 = *(__int64 **)(v9 + 528);
  for ( j = &v87[v86]; j != v87; LODWORD(v363) = v363 + 1 )
  {
    v97 = *v87;
    a2 = 0;
    v358 = 0;
    *(_OWORD *)n = 0;
    v356 = 0;
    v357 = 0;
    n[0] = (size_t)sub_1649960(v97);
    if ( (unsigned __int8)sub_15F8BF0(v97) )
    {
      v88 = *(_QWORD *)(v97 - 24);
      if ( *(_BYTE *)(v88 + 16) != 13 )
        BUG();
      if ( *(_DWORD *)(v88 + 32) <= 0x40u )
        v288 = *(_QWORD *)(v88 + 24);
      else
        v288 = **(_QWORD **)(v88 + 24);
    }
    else
    {
      v288 = 1;
    }
    v89 = *(_QWORD *)(v97 + 56);
    v90 = sub_15F2050(v97);
    v290 = sub_1632FA0(v90);
    v91 = (unsigned int)sub_15A9FE0(v290, v89);
    n[1] = v91 * v288 * ((v91 + ((unsigned __int64)(sub_127FA20(v290, v89) + 7) >> 3) - 1) / v91);
    v94 = *(_WORD *)(v97 + 18);
    v357.m128i_i64[0] = v97;
    v356.m128i_i64[1] = (unsigned int)(1 << v94) >> 1;
    v95 = (unsigned int)v363;
    if ( (unsigned int)v363 >= HIDWORD(v363) )
    {
      sub_16CD150((__int64)&v362, &v364, 0, 56, v92, v93);
      v95 = (unsigned int)v363;
    }
    a3 = (__m128)_mm_loadu_si128((const __m128i *)n);
    ++v87;
    v96 = (__m128 *)&v362[56 * v95];
    *v96 = a3;
    a4 = _mm_loadu_si128(&v356);
    v96[1] = (__m128)a4;
    a5 = _mm_loadu_si128(&v357);
    v96[2] = (__m128)a5;
    v96[3].m128_u64[0] = v358;
  }
  v116 = *(_DWORD *)(v9 + 504);
  v117 = (unsigned __int64)*(int *)(*(_QWORD *)(v9 + 8) + 224LL) >> 1;
  if ( v117 < 1LL << v116 )
    v117 = 1LL << v116;
  sub_1AA5170(v309, &v362, 1LL << v116, v117);
  v118 = v362;
  v313 = 0;
  v314 = 0;
  v315 = 0;
  v316 = 0;
  v119 = &v362[56 * (unsigned int)v363];
  if ( v362 != v119 )
  {
    v120 = 0;
    v121 = 0;
    while ( v120 )
    {
      v122 = *((_QWORD *)v118 + 4);
      v123 = (v120 - 1) & (((unsigned int)v122 >> 9) ^ ((unsigned int)v122 >> 4));
      v124 = (_QWORD *)(v121 + 16LL * v123);
      v125 = *v124;
      if ( v122 == *v124 )
      {
LABEL_188:
        v124[1] = v118;
        v118 += 56;
        if ( v119 == v118 )
          goto LABEL_197;
        goto LABEL_189;
      }
      v250 = 1;
      v251 = 0;
      while ( v125 != -8 )
      {
        if ( !v251 && v125 == -16 )
          v251 = v124;
        v123 = (v120 - 1) & (v250 + v123);
        v124 = (_QWORD *)(v121 + 16LL * v123);
        v125 = *v124;
        if ( v122 == *v124 )
          goto LABEL_188;
        ++v250;
      }
      if ( v251 )
        v124 = v251;
      ++v313;
      v126 = v315 + 1;
      if ( 4 * ((int)v315 + 1) >= 3 * v120 )
        goto LABEL_192;
      if ( v120 - (v126 + HIDWORD(v315)) <= v120 >> 3 )
        goto LABEL_193;
LABEL_194:
      LODWORD(v315) = v126;
      if ( *v124 != -8 )
        --HIDWORD(v315);
      v124[1] = 0;
      v124[1] = v118;
      v118 += 56;
      *v124 = v122;
      if ( v119 == v118 )
        goto LABEL_197;
LABEL_189:
      v121 = v314;
      v120 = v316;
    }
    ++v313;
LABEL_192:
    v120 *= 2;
LABEL_193:
    sub_180DCD0((__int64)&v313, v120);
    sub_180D320((__int64)&v313, (__int64 *)v118 + 4, n);
    v122 = *((_QWORD *)v118 + 4);
    v124 = (_QWORD *)n[0];
    v126 = v315 + 1;
    goto LABEL_194;
  }
LABEL_197:
  v127 = *(__int64 **)(v9 + 3432);
  v128 = v127 + 1;
  v129 = &v127[4 * *(unsigned int *)(v9 + 3440)];
  if ( v127 == v129 )
    goto LABEL_218;
  v294 = v9;
  while ( 1 )
  {
    v138 = v316;
    if ( !v316 )
    {
      ++v313;
LABEL_212:
      v138 = 2 * v316;
      goto LABEL_213;
    }
    v130 = (v316 - 1) & (((unsigned int)*v128 >> 9) ^ ((unsigned int)*v128 >> 4));
    v131 = (__int64 *)(v314 + 16LL * v130);
    v132 = *v131;
    if ( *v128 == *v131 )
    {
      v133 = v131[1];
      goto LABEL_201;
    }
    v252 = 1;
    v139 = 0;
    while ( v132 != -8 )
    {
      if ( v139 || v132 != -16 )
        v131 = v139;
      v130 = (v316 - 1) & (v252 + v130);
      v269 = (__int64 *)(v314 + 16LL * v130);
      v132 = *v269;
      if ( *v128 == *v269 )
      {
        v133 = v269[1];
        goto LABEL_201;
      }
      ++v252;
      v139 = v131;
      v131 = (__int64 *)(v314 + 16LL * v130);
    }
    if ( !v139 )
      v139 = v131;
    ++v313;
    v140 = v315 + 1;
    if ( 4 * ((int)v315 + 1) >= 3 * v316 )
      goto LABEL_212;
    if ( v316 - HIDWORD(v315) - v140 > v316 >> 3 )
      goto LABEL_214;
LABEL_213:
    sub_180DCD0((__int64)&v313, v138);
    sub_180D320((__int64)&v313, v128, n);
    v139 = (__int64 *)n[0];
    v140 = v315 + 1;
LABEL_214:
    LODWORD(v315) = v140;
    if ( *v139 != -8 )
      --HIDWORD(v315);
    v141 = *v128;
    v133 = 0;
    v139[1] = 0;
    *v139 = v141;
LABEL_201:
    *(_QWORD *)(v133 + 16) = *(_QWORD *)(v133 + 8);
    v134 = sub_15C70A0((__int64)&v306);
    if ( v134 )
    {
      v135 = sub_15C70A0(*(v128 - 1) + 48);
      if ( v135 )
      {
        v136 = *(_QWORD *)(v135 - 8LL * *(unsigned int *)(v135 + 8));
        if ( *(_BYTE *)v136 != 15 )
          v136 = *(_QWORD *)(v136 - 8LL * *(unsigned int *)(v136 + 8));
        v137 = *(_QWORD *)(v134 - 8LL * *(unsigned int *)(v134 + 8));
        if ( *(_BYTE *)v137 != 15 )
          v137 = *(_QWORD *)(v137 - 8LL * *(unsigned int *)(v137 + 8));
        if ( v136 == v137 )
        {
          v245 = *(_DWORD *)(v135 + 4);
          if ( v245 )
          {
            v246 = *(_DWORD *)(v133 + 48);
            if ( v246 && v245 > v246 )
              v245 = *(_DWORD *)(v133 + 48);
            *(_DWORD *)(v133 + 48) = v245;
          }
        }
      }
    }
    if ( v129 == v128 + 3 )
      break;
    v128 += 4;
  }
  v9 = v294;
LABEL_218:
  sub_1AA3870(v340, &v362);
  v278 = v310;
  v289 = byte_4FA83C0;
  if ( byte_4FA83C0 )
    v289 = (v310 <= 0x10000) & (*(_BYTE *)(*(_QWORD *)(v9 + 8) + 228LL) ^ 1);
  v142 = *(_BYTE *)(v9 + 3800);
  if ( v142 )
  {
    v142 = 0;
  }
  else if ( !*(_BYTE *)(v9 + 3801) )
  {
    v143 = byte_4FA74A0;
    if ( byte_4FA74A0 )
    {
      v144 = 0;
      goto LABEL_224;
    }
    v142 = v289;
  }
  v143 = 0;
  v289 = v142;
  v144 = sub_18051B0(v9, (__int64 *)&v317, (__int64)v309, 0);
LABEL_224:
  if ( v289 )
  {
    v145 = *(_QWORD **)(v9 + 488);
    n[0] = (size_t)"asan_local_stack_base";
    v356.m128i_i16[0] = 259;
    v295 = (__int64)sub_17CEAE0((__int64 *)&v317, v145, 0, (__int64 *)n);
    v146 = *(_QWORD *)(*(_QWORD *)v9 + 40LL);
    v147 = sub_1643350(v320);
    v148 = sub_1632210(v146, (__int64)"__asan_option_detect_stack_use_after_return", 43, v147);
    v356.m128i_i16[0] = 257;
    v149 = v148;
    v150 = (__int64 **)sub_1643350(v320);
    v153 = sub_15A06D0(v150, (__int64)"__asan_option_detect_stack_use_after_return", v151, v152);
    LOWORD(v350[0]) = 257;
    v154 = v153;
    v155 = sub_156E5B0((__int64 *)&v317, v149, (__int64)&v349);
    v156 = sub_12AA0C0((__int64 *)&v317, 0x21u, v155, v154, (__int64)n);
    v157 = sub_1AA92B0(v156, v279, 0, 0, 0, 0);
    sub_17CE510((__int64)n, v157, 0, 0, 0);
    v349.m128_u64[0] = (unsigned __int64)v306;
    if ( v306 )
    {
      sub_1623A60((__int64)&v349, (__int64)v306, 2);
      v158 = n[0];
      if ( n[0] )
        goto LABEL_227;
LABEL_228:
      n[0] = v349.m128_u64[0];
      if ( v349.m128_u64[0] )
      {
        sub_1623210((__int64)&v349, (unsigned __int8 *)v349.m128_u64[0], (__int64)n);
        v349.m128_u64[0] = 0;
      }
    }
    else
    {
      v158 = n[0];
      if ( n[0] )
      {
LABEL_227:
        sub_161E7C0((__int64)n, v158);
        goto LABEL_228;
      }
    }
    sub_17CD270((__int64 *)&v349);
    v159 = 0;
    v160 = 64;
    v284 = 0;
    if ( v278 > 0x40 )
    {
      do
      {
        v160 *= 2LL;
        ++v159;
      }
      while ( v278 > v160 );
      v284 = v159;
    }
    v161 = *(_QWORD *)(v9 + 488);
    LOWORD(v350[0]) = 257;
    s = (void *)sub_15A0680(v161, v278, 0);
    v162 = *(_QWORD *)(v9 + 8LL * v284 + 904);
    v286 = sub_1285290((__int64 *)n, *(_QWORD *)(v162 + 24), v162, (int)&s, 1, (__int64)&v349, 0);
    sub_17050D0((__int64 *)&v317, v279);
    v349.m128_u64[0] = (unsigned __int64)v306;
    if ( v306 )
    {
      sub_1623A60((__int64)&v349, (__int64)v306, 2);
      v163 = v317;
      if ( v317 )
        goto LABEL_235;
LABEL_236:
      v317 = (unsigned __int8 *)v349.m128_u64[0];
      if ( v349.m128_u64[0] )
      {
        sub_1623210((__int64)&v349, (unsigned __int8 *)v349.m128_u64[0], (__int64)&v317);
        v349.m128_u64[0] = 0;
      }
    }
    else
    {
      v163 = v317;
      if ( v317 )
      {
LABEL_235:
        sub_161E7C0((__int64)&v317, (__int64)v163);
        goto LABEL_236;
      }
    }
    sub_17CD270((__int64 *)&v349);
    v164 = sub_15A0680(*(_QWORD *)(v9 + 488), 0, 0);
    v165 = sub_1802930(*(_QWORD *)(v9 + 488), (__int64 *)&v317, v156, v286, v157, v164);
    v166 = *(__int64 ***)(v9 + 488);
    LOWORD(v350[0]) = 257;
    v167 = (_BYTE *)v165;
    v281 = (unsigned __int8 *)v165;
    v170 = sub_15A06D0(v166, (__int64)&v317, v168, v169);
    v171 = sub_12AA0C0((__int64 *)&v317, 0x20u, v167, v170, (__int64)&v349);
    v172 = sub_1AA92B0(v171, v279, 0, 0, 0, 0);
    sub_17050D0((__int64 *)n, v172);
    v349.m128_u64[0] = (unsigned __int64)v306;
    if ( v306 )
    {
      sub_1623A60((__int64)&v349, (__int64)v306, 2);
      v173 = n[0];
      if ( n[0] )
        goto LABEL_240;
LABEL_241:
      n[0] = v349.m128_u64[0];
      if ( v349.m128_u64[0] )
      {
        sub_1623210((__int64)&v349, (unsigned __int8 *)v349.m128_u64[0], (__int64)n);
        v349.m128_u64[0] = 0;
      }
    }
    else
    {
      v173 = n[0];
      if ( n[0] )
      {
LABEL_240:
        sub_161E7C0((__int64)n, v173);
        goto LABEL_241;
      }
    }
    sub_17CD270((__int64 *)&v349);
    if ( v143 )
      v144 = sub_18051B0(v9, (__int64 *)n, (__int64)v309, 1);
    sub_17050D0((__int64 *)&v317, v279);
    v349.m128_u64[0] = (unsigned __int64)v306;
    if ( v306 )
    {
      sub_1623A60((__int64)&v349, (__int64)v306, 2);
      v174 = v317;
      if ( v317 )
        goto LABEL_247;
LABEL_248:
      v317 = (unsigned __int8 *)v349.m128_u64[0];
      if ( v349.m128_u64[0] )
      {
        sub_1623210((__int64)&v349, (unsigned __int8 *)v349.m128_u64[0], (__int64)&v317);
        v349.m128_u64[0] = 0;
      }
    }
    else
    {
      v174 = v317;
      if ( v317 )
      {
LABEL_247:
        sub_161E7C0((__int64)&v317, (__int64)v174);
        goto LABEL_248;
      }
    }
    sub_17CD270((__int64 *)&v349);
    v144 = sub_1802930(*(_QWORD *)(v9 + 488), (__int64 *)&v317, v171, v144, v172, (__int64)v281);
    v349.m128_u64[0] = (unsigned __int64)v306;
    if ( v306 )
    {
      sub_1623A60((__int64)&v349, (__int64)v306, 2);
      v175 = v317;
      if ( v317 )
        goto LABEL_252;
LABEL_253:
      v317 = (unsigned __int8 *)v349.m128_u64[0];
      if ( v349.m128_u64[0] )
      {
        sub_1623210((__int64)&v349, (unsigned __int8 *)v349.m128_u64[0], (__int64)&v317);
        v349.m128_u64[0] = 0;
      }
    }
    else
    {
      v175 = v317;
      if ( v317 )
      {
LABEL_252:
        sub_161E7C0((__int64)&v317, (__int64)v175);
        goto LABEL_253;
      }
    }
    sub_17CD270((__int64 *)&v349);
    sub_12A8F50((__int64 *)&v317, v144, v295, 0);
    sub_17CD270((__int64 *)n);
    v143 = v289;
  }
  else
  {
    v295 = v144;
    v281 = (unsigned __int8 *)sub_15A0680(*(_QWORD *)(v9 + 488), 0, 0);
    v284 = -1;
    if ( v143 )
    {
      v143 = 0;
      v295 = sub_18051B0(v9, (__int64 *)&v317, (__int64)v309, 1);
      v144 = v295;
    }
  }
  v176 = v362;
  v283 = v143;
  v280 = &v362[56 * (unsigned int)v363];
  if ( v362 != v280 )
  {
    do
    {
      v180 = (__int64 ***)*((_QWORD *)v176 + 4);
      sub_1AEA710(v180, v295, v9 + 16, v283, *((unsigned int *)v176 + 10), 0);
      LOWORD(v350[0]) = 257;
      v181 = *(_QWORD *)(v9 + 488);
      v182 = *v180;
      LOWORD(v343[0]) = 257;
      v183 = sub_15A0680(v181, *((_QWORD *)v176 + 5), 0);
      v184 = sub_12899C0((__int64 *)&v317, v144, v183, (__int64)&s, 0, 0);
      v177 = v184;
      if ( v182 != *(__int64 ***)v184 )
      {
        if ( *(_BYTE *)(v184 + 16) <= 0x10u )
        {
          v177 = sub_15A46C0(46, (__int64 ***)v184, v182, 0);
        }
        else
        {
          v356.m128i_i16[0] = 257;
          v185 = sub_15FDBD0(46, v184, (__int64)v182, (__int64)n, 0);
          if ( v318 )
          {
            v273 = v185;
            v271 = v319;
            sub_157E9D0(v318 + 40, v185);
            v185 = v273;
            v186 = *(_QWORD *)(v273 + 24);
            v187 = *v271;
            *(_QWORD *)(v273 + 32) = v271;
            v187 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v273 + 24) = v187 | v186 & 7;
            *(_QWORD *)(v187 + 8) = v273 + 24;
            *v271 = *v271 & 7 | (v273 + 24);
          }
          v274 = v185;
          sub_164B780(v185, (__int64 *)&v349);
          v177 = v274;
          if ( v317 )
          {
            v333[0] = v317;
            sub_1623A60((__int64)v333, (__int64)v317, 2);
            v177 = v274;
            v188 = *(_QWORD *)(v274 + 48);
            v189 = v274 + 48;
            if ( v188 )
            {
              v272 = v274;
              v275 = v274 + 48;
              sub_161E7C0(v275, v188);
              v177 = v272;
              v189 = v275;
            }
            v190 = v333[0];
            *(unsigned __int8 **)(v177 + 48) = v333[0];
            if ( v190 )
            {
              v276 = v177;
              sub_1623210((__int64)v333, v190, v189);
              v177 = v276;
            }
          }
        }
      }
      v176 += 56;
      sub_164D160(
        (__int64)v180,
        v177,
        a2,
        *(double *)a3.m128_u64,
        *(double *)a4.m128i_i64,
        *(double *)a5.m128i_i64,
        v178,
        v179,
        *(double *)a8.m128i_i64,
        a9);
    }
    while ( v280 != v176 );
  }
  v191 = *(_QWORD *)(v9 + 496);
  v356.m128i_i16[0] = 257;
  v287 = sub_12AA3B0((__int64 *)&v317, 0x2Eu, v144, v191, (__int64)n);
  v192 = sub_15A0680(*(_QWORD *)(v9 + 488), 1102416563, 0);
  sub_12A8F50((__int64 *)&v317, v192, v287, 0);
  v193 = *(_QWORD *)(v9 + 488);
  v194 = *(_QWORD *)(v9 + 496);
  v356.m128i_i16[0] = 257;
  LOWORD(v350[0]) = 257;
  v195 = sub_15A0680(v193, *(_DWORD *)(*(_QWORD *)(v9 + 8) + 224LL) / 8, 0);
  v196 = sub_12899C0((__int64 *)&v317, v144, v195, (__int64)&v349, 0, 0);
  v197 = sub_12AA3B0((__int64 *)&v317, 0x2Eu, v196, v194, (__int64)n);
  v198 = sub_1801990(*(__int64 **)(*(_QWORD *)v9 + 40LL), (char *)v340[0].m128i_i64[0], v340[0].m128i_u32[2], 1);
  v199 = *(_QWORD *)(v9 + 488);
  v356.m128i_i16[0] = 257;
  v200 = sub_12A95D0((__int64 *)&v317, (__int64)v198, v199, (__int64)n);
  sub_12A8F50((__int64 *)&v317, v200, v197, 0);
  v201 = *(_QWORD *)(v9 + 8);
  v356.m128i_i16[0] = 257;
  v202 = *(_QWORD *)(v9 + 488);
  LOWORD(v350[0]) = 257;
  v203 = *(_QWORD *)(v9 + 496);
  v204 = sub_15A0680(v202, *(_DWORD *)(v201 + 224) / 4, 0);
  v205 = sub_12899C0((__int64 *)&v317, v144, v204, (__int64)&v349, 0, 0);
  v206 = sub_12AA3B0((__int64 *)&v317, 0x2Eu, v205, v203, (__int64)n);
  v356.m128i_i16[0] = 257;
  v207 = v206;
  v208 = sub_12A95D0((__int64 *)&v317, *(_QWORD *)v9, *(_QWORD *)(v9 + 488), (__int64)n);
  sub_12A8F50((__int64 *)&v317, v208, v207, 0);
  v209 = sub_1AA4160(n, &v362, v309);
  v296 = sub_1804A80(*(_QWORD *)(v9 + 8), v144, (__int64 *)&v317, v209, *(double *)a3.m128_u64, *(double *)a4.m128i_i64);
  sub_1806600((_QWORD *)v9, n[0], n[0], 0, LODWORD(n[1]), (__int64 *)&v317, v296);
  v211 = *(_DWORD *)(v9 + 3440);
  v212 = v270;
  if ( v211 )
  {
    ((void (__fastcall *)(__m128 *, unsigned __int8 **, _QWORD *, __int64))sub_1AA3EA0)(&v349, &v362, v309, v210);
    v254 = *(_QWORD *)(v9 + 3432);
    v255 = v254 + 8;
    if ( v254 == v254 + 32LL * *(unsigned int *)(v9 + 3440) )
    {
LABEL_367:
      if ( (unsigned __int64 *)v349.m128_u64[0] != v350 )
        _libc_free(v349.m128_u64[0]);
      goto LABEL_270;
    }
    v300 = v254 + 32LL * *(unsigned int *)(v9 + 3440);
    while ( 1 )
    {
      v263 = v316;
      if ( v316 )
      {
        LODWORD(v256) = (v316 - 1) & (((unsigned int)*(_QWORD *)v255 >> 9) ^ ((unsigned int)*(_QWORD *)v255 >> 4));
        v257 = (_QWORD *)(v314 + 16LL * (unsigned int)v256);
        v258 = *v257;
        if ( *(_QWORD *)v255 == *v257 )
        {
LABEL_353:
          v259 = v257[1];
          goto LABEL_354;
        }
        v268 = 1;
        v264 = 0;
        while ( v258 != -8 )
        {
          if ( v258 == -16 && !v264 )
            v264 = v257;
          v256 = (v316 - 1) & ((_DWORD)v256 + v268);
          v257 = (_QWORD *)(v314 + 16 * v256);
          v258 = *v257;
          if ( *(_QWORD *)v255 == *v257 )
            goto LABEL_353;
          ++v268;
        }
        if ( !v264 )
          v264 = v257;
        ++v313;
        v265 = v315 + 1;
        if ( 4 * ((int)v315 + 1) < 3 * v316 )
        {
          if ( v316 - HIDWORD(v315) - v265 > v316 >> 3 )
            goto LABEL_364;
          goto LABEL_363;
        }
      }
      else
      {
        ++v313;
      }
      v263 = 2 * v316;
LABEL_363:
      sub_180DCD0((__int64)&v313, v263);
      sub_180D320((__int64)&v313, (__int64 *)v255, &s);
      v264 = s;
      v265 = v315 + 1;
LABEL_364:
      LODWORD(v315) = v265;
      if ( *v264 != -8 )
        --HIDWORD(v315);
      v266 = *(_QWORD *)v255;
      v264[1] = 0;
      *v264 = v266;
      v259 = 0;
LABEL_354:
      v260 = *(_QWORD *)(v259 + 40) / v309[0];
      v261 = (unsigned __int64)(v309[0] + *(_QWORD *)(v255 + 8) - 1LL) / v309[0] + v260;
      sub_17CE510((__int64)&s, *(_QWORD *)(v255 - 8), 0, 0, 0);
      v262 = &v349;
      if ( *(_BYTE *)(v255 + 16) )
        v262 = (__m128 *)n;
      sub_1806600((_QWORD *)v9, n[0], v262->m128_u64[0], v260, v261, (__int64 *)&s, v296);
      v212 = v270;
      if ( s )
        sub_161E7C0((__int64)&s, (__int64)s);
      if ( v300 == v255 + 24 )
        goto LABEL_367;
      v255 += 32;
    }
  }
LABEL_270:
  v213 = LODWORD(n[1]);
  s = v343;
  v342 = 0x4000000000LL;
  if ( LODWORD(n[1]) > 0x40 )
  {
    sub_16CD150((__int64)&s, v343, LODWORD(n[1]), 1, v212, v211);
    LODWORD(v342) = v213;
    v253 = (unsigned __int64 *)s;
  }
  else
  {
    LODWORD(v342) = n[1];
    if ( !LODWORD(n[1]) )
      goto LABEL_272;
    v253 = v343;
  }
  memset(v253, 0, v213);
LABEL_272:
  v214 = *(__int64 **)(v9 + 816);
  v349.m128_u64[0] = (unsigned __int64)v350;
  v349.m128_u64[1] = 0x4000000000LL;
  v303 = &v214[*(unsigned int *)(v9 + 824)];
  if ( v214 == v303 )
  {
    v247 = *(_QWORD ***)(v9 + 528);
    v248 = &v247[*(unsigned int *)(v9 + 536)];
    if ( v248 != v247 )
      goto LABEL_311;
  }
  else
  {
    v215 = v214;
    v277 = 64LL << v284;
    do
    {
      v239 = *v215;
      v240 = sub_16498A0(*v215);
      v321[0] = 0;
      v321[3] = v240;
      v321[4] = 0;
      v322 = 0;
      v323 = 0;
      v324 = 0;
      v321[1] = *(_QWORD *)(v239 + 40);
      v321[2] = v239 + 24;
      v241 = *(unsigned __int8 **)(v239 + 48);
      v333[0] = v241;
      if ( v241 )
      {
        sub_1623A60((__int64)v333, (__int64)v241, 2);
        if ( v321[0] )
          sub_161E7C0((__int64)v321, v321[0]);
        v321[0] = (__int64)v333[0];
        if ( v333[0] )
          sub_1623210((__int64)v333, v333[0], (__int64)v321);
      }
      v242 = sub_15A0680(*(_QWORD *)(v9 + 488), 1172321806, 0);
      sub_12A8F50(v321, v242, v287, 0);
      if ( v289 )
      {
        v216 = *(__int64 ***)(v9 + 488);
        LOWORD(v334) = 257;
        v217 = sub_15A06D0(v216, v242, v243, 257);
        v218 = sub_12AA0C0(v321, 0x21u, v281, v217, (__int64)v333);
        sub_1AA6B00(v218, v239, &v307, &v308, 0);
        v219 = v307;
        v220 = (_QWORD *)sub_16498A0(v307);
        v325 = 0;
        v328 = v220;
        v329 = 0;
        v330 = 0;
        v331 = 0;
        v332 = 0;
        v326 = *(_QWORD *)(v219 + 40);
        v327 = v219 + 24;
        v221 = *(unsigned __int8 **)(v219 + 48);
        v333[0] = v221;
        if ( v221 )
        {
          sub_1623A60((__int64)v333, (__int64)v221, 2);
          if ( v325 )
            sub_161E7C0((__int64)&v325, (__int64)v325);
          v325 = (const char *)v333[0];
          if ( v333[0] )
            sub_1623210((__int64)v333, v333[0], (__int64)&v325);
        }
        if ( v284 > 4 )
        {
          LOWORD(v334) = 257;
          v244 = *(_QWORD *)(v9 + 488);
          v311[0] = v281;
          v311[1] = (unsigned __int8 *)sub_15A0680(v244, v278, 0);
          sub_1285290(
            (__int64 *)&v325,
            *(_QWORD *)(*(_QWORD *)(v9 + 8 * (v284 + 124LL)) + 24LL),
            *(_QWORD *)(v9 + 8 * (v284 + 124LL)),
            (int)v311,
            2,
            (__int64)v333,
            0);
        }
        else
        {
          v222 = v349.m128_u32[2];
          v223 = (unsigned __int64)v277 / v309[0];
          v224 = v223;
          v225 = v223;
          if ( v223 >= v349.m128_u32[2] )
          {
            if ( (unsigned __int64)v277 / v309[0] > v349.m128_u32[2] )
            {
              if ( v223 > v349.m128_u32[3] )
              {
                v302 = (unsigned __int64)v277 / v309[0];
                sub_16CD150((__int64)&v349, v350, v302, 1, v223, v223);
                v222 = v349.m128_u32[2];
                v224 = v302;
              }
              v226 = v349.m128_u64[0];
              if ( v224 != v222 )
              {
                v301 = v224;
                memset((void *)(v349.m128_u64[0] + v222), 245, v224 - v222);
                v226 = v349.m128_u64[0];
                LODWORD(v224) = v301;
              }
              v349.m128_i32[2] = v224;
              v225 = (unsigned int)v224;
            }
            else
            {
              v226 = v349.m128_u64[0];
              v225 = v349.m128_u32[2];
            }
          }
          else
          {
            v349.m128_i32[2] = v223;
            v226 = v349.m128_u64[0];
          }
          sub_1806600((_QWORD *)v9, v226, v226, 0, v225, (__int64 *)&v325, v296);
          v227 = *(_QWORD *)(v9 + 8);
          LOWORD(v334) = 257;
          v228 = sub_15A0680(*(_QWORD *)(v9 + 488), v277 - *(_DWORD *)(v227 + 224) / 8, 0);
          v229 = sub_12899C0((__int64 *)&v325, (__int64)v281, v228, (__int64)v333, 0, 0);
          v230 = *(_QWORD *)(v9 + 496);
          v312 = 257;
          LOWORD(v334) = 257;
          v231 = sub_12AA3B0((__int64 *)&v325, 0x2Eu, v229, v230, (__int64)v311);
          v297 = sub_156E5B0((__int64 *)&v325, v231, (__int64)v333);
          LOWORD(v334) = 257;
          v232 = sub_16471D0(v328, 0);
          v298 = sub_12AA3B0((__int64 *)&v325, 0x2Eu, (__int64)v297, v232, (__int64)v333);
          v233 = (__int64 **)sub_1643330(v328);
          v236 = sub_15A06D0(v233, 46, v234, v235);
          sub_12A8F50((__int64 *)&v325, v236, v298, 0);
        }
        v299 = v308;
        v237 = sub_16498A0(v308);
        v333[0] = 0;
        v335 = v237;
        v336 = 0;
        v337 = 0;
        v338 = 0;
        v339 = 0;
        v333[1] = *(unsigned __int8 **)(v299 + 40);
        v334 = v299 + 24;
        v238 = *(unsigned __int8 **)(v299 + 48);
        v311[0] = v238;
        if ( v238 )
        {
          sub_1623A60((__int64)v311, (__int64)v238, 2);
          if ( v333[0] )
            sub_161E7C0((__int64)v333, (__int64)v333[0]);
          v333[0] = v311[0];
          if ( v311[0] )
            sub_1623210((__int64)v311, v311[0], (__int64)v333);
        }
        sub_1806600((_QWORD *)v9, n[0], (__int64)s, 0, LODWORD(n[1]), (__int64 *)v333, v296);
        if ( v333[0] )
          sub_161E7C0((__int64)v333, (__int64)v333[0]);
        if ( v325 )
          sub_161E7C0((__int64)&v325, (__int64)v325);
      }
      else
      {
        sub_1806600((_QWORD *)v9, n[0], (__int64)s, 0, LODWORD(n[1]), v321, v296);
      }
      if ( v321[0] )
        sub_161E7C0((__int64)v321, v321[0]);
      ++v215;
    }
    while ( v303 != v215 );
    v247 = *(_QWORD ***)(v9 + 528);
    v248 = &v247[*(unsigned int *)(v9 + 536)];
    while ( v248 != v247 )
    {
LABEL_311:
      v249 = *v247++;
      sub_15F20C0(v249);
    }
    if ( (unsigned __int64 *)v349.m128_u64[0] != v350 )
      _libc_free(v349.m128_u64[0]);
  }
  if ( s != v343 )
    _libc_free((unsigned __int64)s);
  if ( (__m128i *)n[0] != &v356 )
    _libc_free(n[0]);
  if ( (__m128i *)v340[0].m128i_i64[0] != &v340[1] )
    _libc_free(v340[0].m128i_u64[0]);
  j___libc_free_0(v314);
  if ( v362 != (unsigned __int8 *)&v364 )
    _libc_free((unsigned __int64)v362);
  sub_17CD270((__int64 *)&v317);
  sub_17CD270((__int64 *)&v306);
  return v285;
}
