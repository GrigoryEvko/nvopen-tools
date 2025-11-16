// Function: sub_28873F0
// Address: 0x28873f0
//
__int64 __fastcall sub_28873F0(
        __int64 a1,
        __int64 **a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 *a6,
        __int64 a7,
        __int64 *a8,
        unsigned int a9,
        unsigned int a10,
        char a11,
        unsigned int a12,
        __m128i *a13,
        unsigned int *a14,
        unsigned int *a15,
        _BYTE *a16,
        __int64 *a17)
{
  __int64 v17; // rbx
  __m128i *v18; // r12
  __int64 v19; // rcx
  __int64 v20; // r14
  __int64 v21; // rax
  unsigned __int64 *v22; // r14
  unsigned __int64 *v23; // r15
  unsigned __int64 v24; // rdi
  _QWORD *v25; // r15
  _QWORD *v26; // r14
  unsigned __int64 v27; // rsi
  _QWORD *v28; // rax
  _QWORD *v29; // rdi
  __int64 v30; // rcx
  __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // rax
  _QWORD *v37; // rdi
  bool v38; // r14
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rdi
  __int64 v49; // rax
  unsigned __int8 v50; // r14
  _QWORD *v51; // r13
  _QWORD *v52; // rbx
  unsigned __int64 v53; // rsi
  _QWORD *v54; // rax
  _QWORD *v55; // rdi
  __int64 v56; // rcx
  __int64 v57; // rdx
  __int64 v58; // rax
  _QWORD *v59; // rdi
  __int64 v60; // rcx
  __int64 v61; // rdx
  __int64 v63; // r14
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // r9
  unsigned __int64 *v68; // r14
  unsigned __int64 v69; // r8
  unsigned __int64 *v70; // r15
  unsigned __int64 v71; // rdi
  __m128i v72; // xmm4
  __m128i v73; // xmm2
  __int64 v74; // rax
  __int64 v75; // r14
  __int64 v76; // rax
  __int64 v77; // r8
  unsigned __int64 *v78; // r14
  unsigned __int64 *v79; // r12
  unsigned __int64 *v80; // rbx
  unsigned __int64 v81; // rdi
  __int64 v82; // r14
  __int64 v83; // rax
  unsigned __int64 *v84; // r14
  unsigned __int64 *v85; // r15
  unsigned __int64 v86; // rdi
  __int64 v87; // rax
  __int64 v88; // rax
  __int64 v89; // r14
  __int64 v90; // rax
  __int64 v91; // rax
  __int64 v92; // rcx
  __int64 v93; // r8
  unsigned __int64 *v94; // r14
  unsigned __int64 *v95; // rbx
  unsigned __int64 *v96; // r14
  unsigned __int64 v97; // rdi
  __int64 v98; // rcx
  __int64 v99; // r8
  unsigned __int64 v100; // rax
  __int64 v101; // r14
  __int64 v102; // rax
  __int64 v103; // rcx
  __int64 v104; // r8
  unsigned __int64 v105; // rax
  __int64 v106; // r8
  unsigned __int64 *v107; // r14
  unsigned __int64 *v108; // rbx
  unsigned __int64 *v109; // r14
  unsigned __int64 v110; // rdi
  __int32 v111; // eax
  unsigned int v112; // eax
  __int64 v113; // r14
  __int64 v114; // rax
  __int64 v115; // rdx
  __int64 v116; // r9
  __int64 v117; // rcx
  __int64 v118; // r8
  unsigned __int64 *v119; // r14
  unsigned __int64 *v120; // rbx
  unsigned __int64 *v121; // r14
  unsigned __int64 v122; // rdi
  __int64 v123; // rcx
  __int64 v124; // r8
  __int64 v125; // r14
  __int64 v126; // rax
  __int64 v127; // r8
  unsigned __int64 *v128; // r14
  unsigned __int64 *v129; // rbx
  unsigned __int64 *v130; // r14
  unsigned __int64 v131; // rdi
  __int64 v132; // r14
  __int64 v133; // rax
  __int64 v134; // rdx
  __int64 v135; // rcx
  __int64 v136; // r9
  __m128i v137; // xmm2
  __m128i v138; // xmm6
  __m128i v139; // xmm0
  __int64 v140; // rcx
  __int64 v141; // r8
  __int64 v142; // r14
  __int64 v143; // rax
  unsigned __int32 v144; // edx
  unsigned __int64 v145; // rcx
  unsigned int v146; // eax
  unsigned int v147; // esi
  unsigned __int64 v148; // rax
  __int64 v149; // r14
  __int64 v150; // rax
  __int64 v151; // rcx
  __int64 v152; // r8
  unsigned __int64 *v153; // r14
  unsigned __int64 *v154; // rbx
  unsigned __int64 *v155; // r14
  unsigned __int64 v156; // rdi
  int v157; // eax
  __int64 v158; // rax
  __int64 v159; // rax
  __int64 v160; // rax
  __int64 v161; // rax
  __int64 v162; // rax
  __int64 v163; // rax
  __int64 v164; // r14
  __int64 v165; // rax
  __int64 v166; // r14
  __int64 v167; // rax
  __int64 v168; // r14
  __int64 v169; // rax
  unsigned int v170; // r14d
  __int64 v171; // r15
  __int64 v172; // rax
  __int64 v173; // r15
  __int64 v174; // rax
  __int64 v175; // rdx
  __int64 v176; // rcx
  __int64 v177; // r8
  __int64 v178; // r9
  __m128i v179; // xmm6
  __m128i v180; // xmm4
  __int64 v181; // r8
  __int64 v182; // rdx
  __int64 v183; // rcx
  __int64 v184; // r8
  __int64 v185; // r9
  __int64 v186; // r14
  __int64 v187; // rdx
  __int64 v188; // rcx
  __m128i v189; // xmm3
  __m128i v190; // xmm0
  __m128i v191; // xmm1
  unsigned __int64 v192; // rax
  __int64 v193; // rax
  __int32 v194; // edx
  unsigned __int64 v195; // r8
  int v196; // eax
  __int64 v197; // rax
  __int64 v198; // rax
  __int64 v199; // rax
  __int64 v200; // rax
  __int64 v201; // r14
  __int64 v202; // rax
  __int64 v203; // rax
  __int64 v204; // rax
  __m128i *v205; // rsi
  __int64 v206; // rdx
  __int64 v207; // rcx
  __int64 v208; // r8
  __int64 v209; // r9
  __int64 v210; // rdi
  __int64 v211; // r8
  _QWORD *v212; // rax
  unsigned int v213; // edx
  __int64 v214; // rax
  __int64 v215; // rcx
  __m128i *v216; // rsi
  unsigned __int64 v217; // rax
  __int64 v218; // r9
  __int64 v219; // rdx
  unsigned int v220; // eax
  __int64 v221; // rax
  __int64 v222; // rax
  __int8 *v223; // rsi
  size_t v224; // rdx
  __int64 v225; // rax
  __int64 v226; // rax
  __int64 v227; // rax
  __int32 v228; // edx
  unsigned int v229; // ecx
  __int64 v230; // r14
  __int64 v231; // rax
  unsigned __int64 *v232; // r8
  unsigned __int64 *v233; // r12
  unsigned __int64 *v234; // rbx
  unsigned __int64 v235; // rdi
  __int64 v236; // rax
  __int64 v237; // rax
  __int64 v238; // rdx
  __int64 v239; // rcx
  __int64 v240; // r8
  __int64 v241; // rdx
  __int64 v242; // rcx
  __int64 v243; // r9
  __m128i v244; // xmm2
  __m128i v245; // xmm3
  __m128i v246; // xmm0
  __int64 v247; // rcx
  unsigned int v248; // esi
  unsigned int v249; // r8d
  __int64 v250; // rax
  __int64 v251; // rax
  __int64 v252; // rax
  __int64 v253; // rax
  __int64 v254; // rax
  __int64 v255; // rax
  __int64 v256; // rax
  __int64 v257; // rax
  __int64 v258; // rax
  __int64 v259; // rax
  unsigned __int64 v260; // rax
  __int64 v261; // rcx
  unsigned int v262; // edx
  unsigned int v263; // ecx
  unsigned int v264; // edx
  unsigned __int64 v265; // r8
  unsigned int v266; // ebx
  unsigned int v267; // eax
  bool v268; // zf
  __int64 v269; // rax
  __int64 v270; // rax
  __int64 v271; // r15
  __int64 v272; // rdx
  _QWORD *v273; // rax
  __int64 v274; // rdx
  __int64 v275; // rcx
  __int64 v276; // r8
  __int64 v277; // rdx
  __int64 v278; // rcx
  __int64 v279; // r9
  __m128i v280; // xmm0
  __m128i v281; // xmm1
  __m128i v282; // xmm2
  __int64 v283; // rax
  __int64 v284; // rax
  __int64 v285; // [rsp+10h] [rbp-4C0h]
  bool v286; // [rsp+18h] [rbp-4B8h]
  __int64 v287; // [rsp+18h] [rbp-4B8h]
  __int64 v288; // [rsp+20h] [rbp-4B0h]
  unsigned int v289; // [rsp+2Ch] [rbp-4A4h]
  __int64 v290; // [rsp+30h] [rbp-4A0h]
  unsigned __int64 v291; // [rsp+30h] [rbp-4A0h]
  __int64 v292; // [rsp+30h] [rbp-4A0h]
  __int64 v293; // [rsp+30h] [rbp-4A0h]
  __int64 v295; // [rsp+40h] [rbp-490h]
  __int64 v297; // [rsp+48h] [rbp-488h]
  int v298; // [rsp+50h] [rbp-480h]
  int v300; // [rsp+60h] [rbp-470h]
  __int64 v301; // [rsp+60h] [rbp-470h]
  __int64 v302; // [rsp+60h] [rbp-470h]
  __int64 v303; // [rsp+60h] [rbp-470h]
  __m128i v304; // [rsp+60h] [rbp-470h]
  __int64 v305; // [rsp+60h] [rbp-470h]
  __int64 v306; // [rsp+60h] [rbp-470h]
  __int64 v307; // [rsp+60h] [rbp-470h]
  bool v308; // [rsp+74h] [rbp-45Ch]
  bool v309; // [rsp+75h] [rbp-45Bh]
  bool v310; // [rsp+76h] [rbp-45Ah]
  unsigned __int8 v311; // [rsp+77h] [rbp-459h]
  __int64 v312; // [rsp+78h] [rbp-458h]
  unsigned int v313; // [rsp+80h] [rbp-450h]
  bool v314; // [rsp+88h] [rbp-448h]
  __int64 v315; // [rsp+88h] [rbp-448h]
  __int64 v316; // [rsp+90h] [rbp-440h]
  unsigned int v317; // [rsp+90h] [rbp-440h]
  __int64 v318; // [rsp+90h] [rbp-440h]
  __int64 v319; // [rsp+A0h] [rbp-430h] BYREF
  __int64 v320; // [rsp+A8h] [rbp-428h] BYREF
  __m128i v321; // [rsp+B0h] [rbp-420h] BYREF
  __m128i v322; // [rsp+C0h] [rbp-410h]
  __m128i v323; // [rsp+D0h] [rbp-400h]
  __int64 v324; // [rsp+E0h] [rbp-3F0h]
  __m128i v325; // [rsp+F0h] [rbp-3E0h] BYREF
  _QWORD v326[2]; // [rsp+100h] [rbp-3D0h] BYREF
  _QWORD *v327; // [rsp+110h] [rbp-3C0h]
  _QWORD v328[4]; // [rsp+120h] [rbp-3B0h] BYREF
  __m128i v329; // [rsp+140h] [rbp-390h] BYREF
  __m256i v330; // [rsp+150h] [rbp-380h] BYREF
  __m128i v331; // [rsp+170h] [rbp-360h] BYREF
  __m128i v332; // [rsp+180h] [rbp-350h]
  _BYTE *v333; // [rsp+190h] [rbp-340h] BYREF
  __int64 v334; // [rsp+198h] [rbp-338h]
  _BYTE v335[320]; // [rsp+1A0h] [rbp-330h] BYREF
  char v336; // [rsp+2E0h] [rbp-1F0h]
  int v337; // [rsp+2E4h] [rbp-1ECh]
  __int64 v338; // [rsp+2E8h] [rbp-1E8h]
  __m128i v339; // [rsp+2F0h] [rbp-1E0h] BYREF
  __m128i v340; // [rsp+300h] [rbp-1D0h] BYREF
  __m128i v341; // [rsp+310h] [rbp-1C0h]
  __m128i v342; // [rsp+320h] [rbp-1B0h] BYREF
  __m128i v343; // [rsp+330h] [rbp-1A0h] BYREF
  unsigned __int64 *v344; // [rsp+340h] [rbp-190h] BYREF
  unsigned int v345; // [rsp+348h] [rbp-188h]
  _BYTE v346[320]; // [rsp+350h] [rbp-180h] BYREF
  char v347; // [rsp+490h] [rbp-40h]
  int v348; // [rsp+494h] [rbp-3Ch]
  __int64 v349; // [rsp+498h] [rbp-38h]
  __m128i *v350; // [rsp+518h] [rbp+48h]

  v17 = a1;
  v18 = (__m128i *)a14;
  v298 = a3;
  v19 = a13->m128i_u32[2];
  if ( !(_DWORD)v19 )
    v288 = a13->m128i_i64[0];
  sub_D4BD20(&v319, a1, a3, v19, a5, (__int64)a6);
  v20 = *a8;
  v316 = **(_QWORD **)(a1 + 32);
  v21 = sub_B2BE50(*a8);
  if ( sub_B6EA50(v21)
    || (v87 = sub_B2BE50(v20),
        v88 = sub_B6F970(v87),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v88 + 48LL))(v88)) )
  {
    sub_B157E0((__int64)&v329, &v319);
    sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v329, v316);
    sub_B18290((__int64)&v339, "  Analyzing unrolling strategy...", 0x21u);
    sub_1049740(a8, (__int64)&v339);
    v22 = v344;
    v339.m128i_i64[0] = (__int64)&unk_49D9D40;
    v23 = &v344[10 * v345];
    if ( v344 != v23 )
    {
      do
      {
        v23 -= 10;
        v24 = v23[4];
        if ( (unsigned __int64 *)v24 != v23 + 6 )
          j_j___libc_free_0(v24);
        if ( (unsigned __int64 *)*v23 != v23 + 2 )
          j_j___libc_free_0(*v23);
      }
      while ( v22 != v23 );
      v23 = v344;
    }
    if ( v23 != (unsigned __int64 *)v346 )
      _libc_free((unsigned __int64)v23);
  }
  if ( a14[16] > 1 )
  {
    v82 = *a8;
    v83 = sub_B2BE50(*a8);
    if ( sub_B6EA50(v83)
      || (v160 = sub_B2BE50(v82),
          v161 = sub_B6F970(v160),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v161 + 48LL))(v161)) )
    {
      sub_B157E0((__int64)&v329, &v319);
      sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v329, v316);
      sub_B18290((__int64)&v339, "    Reminder: loop accesses local arrays with approximate size of ", 0x42u);
      sub_B169E0(v329.m128i_i64, "UP.AccessedLocalArraySize", 25, a14[16]);
      sub_23FD640((__int64)&v339, (__int64)&v329);
      if ( (__m128i *)v330.m256i_i64[2] != &v331 )
        j_j___libc_free_0(v330.m256i_u64[2]);
      if ( (__m256i *)v329.m128i_i64[0] != &v330 )
        j_j___libc_free_0(v329.m128i_u64[0]);
      sub_1049740(a8, (__int64)&v339);
      v84 = v344;
      v339.m128i_i64[0] = (__int64)&unk_49D9D40;
      v85 = &v344[10 * v345];
      if ( v344 != v85 )
      {
        do
        {
          v85 -= 10;
          v86 = v85[4];
          if ( (unsigned __int64 *)v86 != v85 + 6 )
            j_j___libc_free_0(v86);
          if ( (unsigned __int64 *)*v85 != v85 + 2 )
            j_j___libc_free_0(*v85);
        }
        while ( v84 != v85 );
        v85 = v344;
      }
      if ( v85 != (unsigned __int64 *)v346 )
        _libc_free((unsigned __int64)v85);
    }
  }
  v25 = sub_C52410();
  v26 = v25 + 1;
  v27 = sub_C959E0();
  v28 = (_QWORD *)v25[2];
  if ( v28 )
  {
    v29 = v25 + 1;
    do
    {
      while ( 1 )
      {
        v30 = v28[2];
        v31 = v28[3];
        if ( v27 <= v28[4] )
          break;
        v28 = (_QWORD *)v28[3];
        if ( !v31 )
          goto LABEL_19;
      }
      v29 = v28;
      v28 = (_QWORD *)v28[2];
    }
    while ( v30 );
LABEL_19:
    if ( v26 != v29 && v27 >= v29[4] )
      v26 = v29;
  }
  if ( v26 == (_QWORD *)((char *)sub_C52410() + 8) )
    goto LABEL_58;
  v36 = v26[7];
  v34 = (__int64)(v26 + 6);
  if ( !v36 )
    goto LABEL_58;
  v27 = (unsigned int)dword_50031C8;
  v37 = v26 + 6;
  do
  {
    while ( 1 )
    {
      v33 = *(_QWORD *)(v36 + 16);
      v32 = *(_QWORD *)(v36 + 24);
      if ( *(_DWORD *)(v36 + 32) >= dword_50031C8 )
        break;
      v36 = *(_QWORD *)(v36 + 24);
      if ( !v32 )
        goto LABEL_28;
    }
    v37 = (_QWORD *)v36;
    v36 = *(_QWORD *)(v36 + 16);
  }
  while ( v33 );
LABEL_28:
  if ( v37 == (_QWORD *)v34 || dword_50031C8 < *((_DWORD *)v37 + 8) )
  {
LABEL_58:
    v310 = 0;
    v300 = 0;
  }
  else
  {
    v300 = *((_DWORD *)v37 + 9);
    v310 = v300 > 0;
  }
  v38 = 0;
  v39 = sub_D49300(v17, v27, v32, v33, v34, v35);
  v312 = v39;
  if ( v39 )
  {
    v27 = (unsigned __int64)"llvm.loop.unroll.full";
    v312 = sub_2A11940(v39, "llvm.loop.unroll.full", 21);
    v38 = v312 != 0;
  }
  v313 = sub_287EE90(v17, v27, v40, v41, v42, v43);
  v48 = sub_D49300(v17, v27, v44, v45, v46, v47);
  if ( v48 )
  {
    v49 = sub_2A11940(v48, "llvm.loop.unroll.enable", 23);
    v309 = v49 != 0;
    v308 = (v312 | v49) != 0;
  }
  else
  {
    v308 = v38;
    v309 = 0;
  }
  v314 = v313 != 0;
  v50 = v313 != 0 || v38;
  if ( !v50 )
    v50 = v310 || v309;
  v311 = v50;
  if ( !*a15 )
  {
    v63 = *a8;
    v64 = sub_B2BE50(*a8);
    if ( sub_B6EA50(v64)
      || (v158 = sub_B2BE50(v63),
          v159 = sub_B6F970(v158),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v159 + 48LL))(v159)) )
    {
      sub_B157E0((__int64)&v329, &v319);
      sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v329, v316);
      sub_B18290((__int64)&v339, "    Trying to unroll by #pragma...", 0x22u);
      sub_1049740(a8, (__int64)&v339);
      v68 = v344;
      v339.m128i_i64[0] = (__int64)&unk_49D9D40;
      v69 = 80LL * v345;
      v70 = (unsigned __int64 *)((char *)v344 + v69);
      if ( v344 != (unsigned __int64 *)((char *)v344 + v69) )
      {
        do
        {
          v70 -= 10;
          v71 = v70[4];
          if ( (unsigned __int64 *)v71 != v70 + 6 )
            j_j___libc_free_0(v71);
          if ( (unsigned __int64 *)*v70 != v70 + 2 )
            j_j___libc_free_0(*v70);
        }
        while ( v68 != v70 );
        v70 = v344;
      }
      if ( v70 != (unsigned __int64 *)v346 )
        _libc_free((unsigned __int64)v70);
    }
    v289 = qword_5002C28;
    v72 = _mm_loadu_si128(a13 + 2);
    v73 = _mm_loadu_si128(a13 + 1);
    v74 = a13[3].m128i_i64[0];
    v321 = _mm_loadu_si128(a13);
    v324 = v74;
    v322 = v73;
    v323 = v72;
    if ( v300 > 0 )
    {
      if ( *((_BYTE *)a14 + 46) )
      {
        v69 = sub_2880E70(v321.m128i_i64, (__int64)a14, qword_5003248, v66, v69);
        if ( v69 < *a14 )
        {
          a14[5] = qword_5003248;
LABEL_208:
          *(_WORD *)((char *)a14 + 47) = 257;
LABEL_209:
          *((_BYTE *)a14 + 45) |= v314;
          goto LABEL_55;
        }
      }
    }
    sub_D4BD20(&v320, v17, v65, v66, v69, v67);
    v301 = **(_QWORD **)(v17 + 32);
    if ( !v313 )
    {
      if ( !v312 )
      {
LABEL_73:
        v75 = *a8;
        v76 = sub_B2BE50(*a8);
        if ( sub_B6EA50(v76)
          || (v162 = sub_B2BE50(v75),
              v163 = sub_B6F970(v162),
              (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v163 + 48LL))(v163)) )
        {
          sub_B157E0((__int64)&v329, &v320);
          sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v329, v301);
          sub_B18290((__int64)&v339, "      Failed: \"#pragma unroll\" is not set", 0x29u);
          sub_1049740(a8, (__int64)&v339);
          v339.m128i_i64[0] = (__int64)&unk_49D9D40;
          v77 = 10LL * v345;
          v78 = &v344[v77];
          if ( v344 != &v344[v77] )
          {
            v79 = &v344[v77];
            v302 = v17;
            v80 = v344;
            do
            {
              v79 -= 10;
              v81 = v79[4];
              if ( (unsigned __int64 *)v81 != v79 + 6 )
                j_j___libc_free_0(v81);
              if ( (unsigned __int64 *)*v79 != v79 + 2 )
                j_j___libc_free_0(*v79);
            }
            while ( v80 != v79 );
            v18 = (__m128i *)a14;
            v17 = v302;
            v78 = v344;
          }
          if ( v78 != (unsigned __int64 *)v346 )
            _libc_free((unsigned __int64)v78);
        }
        goto LABEL_146;
      }
      v286 = 0;
      goto LABEL_211;
    }
    v89 = *a8;
    v90 = sub_B2BE50(*a8);
    if ( sub_B6EA50(v90)
      || (v236 = sub_B2BE50(v89),
          v237 = sub_B6F970(v236),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v237 + 48LL))(v237)) )
    {
      sub_B157E0((__int64)&v329, &v320);
      sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v329, v301);
      sub_B18290((__int64)&v339, "      Loop has \"#pragma unroll ", 0x1Fu);
      sub_B169E0(v329.m128i_i64, "PragmaCount", 11, v313);
      v91 = sub_23FD640((__int64)&v339, (__int64)&v329);
      sub_B18290(v91, "\"", 1u);
      if ( (__m128i *)v330.m256i_i64[2] != &v331 )
        j_j___libc_free_0(v330.m256i_u64[2]);
      if ( (__m256i *)v329.m128i_i64[0] != &v330 )
        j_j___libc_free_0(v329.m128i_u64[0]);
      sub_1049740(a8, (__int64)&v339);
      v92 = (__int64)v344;
      v339.m128i_i64[0] = (__int64)&unk_49D9D40;
      v93 = 10LL * v345;
      v94 = &v344[v93];
      if ( v344 != &v344[v93] )
      {
        v290 = v17;
        v95 = &v344[v93];
        v96 = v344;
        do
        {
          v95 -= 10;
          v97 = v95[4];
          if ( (unsigned __int64 *)v97 != v95 + 6 )
            j_j___libc_free_0(v97);
          if ( (unsigned __int64 *)*v95 != v95 + 2 )
            j_j___libc_free_0(*v95);
        }
        while ( v96 != v95 );
        v17 = v290;
        v94 = v344;
      }
      if ( v94 != (unsigned __int64 *)v346 )
        _libc_free((unsigned __int64)v94);
    }
    if ( *((_BYTE *)a14 + 46) || !(a12 % v313) )
    {
      v192 = sub_2880E70(v321.m128i_i64, (__int64)a14, v313, v92, v93 * 8);
      v98 = v289;
      v291 = v289;
      if ( v192 < v289 )
      {
LABEL_227:
        if ( v320 )
          sub_B91220((__int64)&v320, v320);
        a14[5] = v313;
        if ( !v314 && !v310 )
          goto LABEL_209;
        goto LABEL_208;
      }
      if ( *((_BYTE *)a14 + 46) || !(a12 % v313) )
      {
LABEL_124:
        v100 = sub_2880E70(v321.m128i_i64, (__int64)a14, v313, v98, v99);
        v286 = v100 >= v291;
        if ( !v312 )
        {
          if ( v100 < v291 )
            goto LABEL_146;
          goto LABEL_126;
        }
LABEL_211:
        v149 = *a8;
        v150 = sub_B2BE50(*a8);
        if ( sub_B6EA50(v150)
          || (v252 = sub_B2BE50(v149),
              v253 = sub_B6F970(v252),
              (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v253 + 48LL))(v253)) )
        {
          sub_B157E0((__int64)&v329, &v320);
          sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v329, v301);
          sub_B18290((__int64)&v339, "      Loop has \"#pragma unroll\" (full unroll directive)", 0x37u);
          sub_1049740(a8, (__int64)&v339);
          v151 = (__int64)v344;
          v339.m128i_i64[0] = (__int64)&unk_49D9D40;
          v152 = 10LL * v345;
          v153 = &v344[v152];
          if ( v344 != &v344[v152] )
          {
            v293 = v17;
            v154 = &v344[v152];
            v155 = v344;
            do
            {
              v154 -= 10;
              v156 = v154[4];
              if ( (unsigned __int64 *)v156 != v154 + 6 )
                j_j___libc_free_0(v156);
              if ( (unsigned __int64 *)*v154 != v154 + 2 )
                j_j___libc_free_0(*v154);
            }
            while ( v155 != v154 );
            v17 = v293;
            v153 = v344;
          }
          if ( v153 != (unsigned __int64 *)v346 )
            _libc_free((unsigned __int64)v153);
        }
        if ( a9 )
        {
          v157 = 6;
          if ( a14[16] <= 6 )
            v157 = a14[16];
          v289 *= v157;
          if ( sub_2880E70(v321.m128i_i64, (__int64)a14, a9, v151, v152 * 8) < (unsigned __int64)v289 )
          {
            v313 = a9;
            goto LABEL_227;
          }
        }
        else
        {
          v201 = *a8;
          v202 = sub_B2BE50(*a8);
          if ( sub_B6EA50(v202)
            || (v269 = sub_B2BE50(v201),
                v270 = sub_B6F970(v269),
                (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v270 + 48LL))(v270)) )
          {
            sub_B157E0((__int64)&v329, &v320);
            sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v329, v301);
            sub_B18290((__int64)&v339, "      Failed: trip count could not be statically calculated", 0x3Bu);
            sub_1049740(a8, (__int64)&v339);
            v339.m128i_i64[0] = (__int64)&unk_49D9D40;
            sub_23FD590((__int64)&v344);
          }
          if ( !v286 )
            goto LABEL_146;
        }
LABEL_126:
        v101 = *a8;
        v102 = sub_B2BE50(*a8);
        if ( sub_B6EA50(v102)
          || (v221 = sub_B2BE50(v101),
              v222 = sub_B6F970(v221),
              (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v222 + 48LL))(v222)) )
        {
          sub_B157E0((__int64)&v329, &v320);
          sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v329, v301);
          sub_B18290((__int64)&v339, "      Failed: estimated unrolled loop size ", 0x2Bu);
          v105 = sub_2880E70(v321.m128i_i64, (__int64)a14, a9, v103, v104);
          sub_B16B10(v329.m128i_i64, "UnrolledLoopSize", 16, v105);
          v287 = sub_23FD640((__int64)&v339, (__int64)&v329);
          sub_B18290(v287, " exceeds threshold ", 0x13u);
          sub_B169E0(v325.m128i_i64, "Threshold", 9, v289);
          sub_23FD640(v287, (__int64)&v325);
          if ( v327 != v328 )
            j_j___libc_free_0((unsigned __int64)v327);
          if ( (_QWORD *)v325.m128i_i64[0] != v326 )
            j_j___libc_free_0(v325.m128i_u64[0]);
          if ( (__m128i *)v330.m256i_i64[2] != &v331 )
            j_j___libc_free_0(v330.m256i_u64[2]);
          if ( (__m256i *)v329.m128i_i64[0] != &v330 )
            j_j___libc_free_0(v329.m128i_u64[0]);
          sub_1049740(a8, (__int64)&v339);
          v339.m128i_i64[0] = (__int64)&unk_49D9D40;
          v106 = 10LL * v345;
          v107 = &v344[v106];
          if ( v344 != &v344[v106] )
          {
            v292 = v17;
            v108 = &v344[v106];
            v109 = v344;
            do
            {
              v108 -= 10;
              v110 = v108[4];
              if ( (unsigned __int64 *)v110 != v108 + 6 )
                j_j___libc_free_0(v110);
              if ( (unsigned __int64 *)*v108 != v108 + 2 )
                j_j___libc_free_0(*v108);
            }
            while ( v109 != v108 );
            v17 = v292;
            v107 = v344;
          }
          if ( v107 != (unsigned __int64 *)v346 )
            _libc_free((unsigned __int64)v107);
        }
        if ( !v312 && !v313 )
          goto LABEL_73;
LABEL_146:
        if ( v320 )
          sub_B91220((__int64)&v320, v320);
        if ( a9 && v311 )
        {
          v111 = qword_5002C28;
          if ( v18->m128i_i32[0] >= (unsigned int)qword_5002C28 )
            v111 = v18->m128i_i32[0];
          v18->m128i_i32[0] = v111;
          v112 = v18->m128i_u32[3];
          if ( (unsigned int)qword_5002C28 >= v112 )
            v112 = qword_5002C28;
          v18->m128i_i32[3] = v112;
        }
        v113 = *a8;
        v114 = sub_B2BE50(*a8);
        if ( sub_B6EA50(v114)
          || (v197 = sub_B2BE50(v113),
              v198 = sub_B6F970(v197),
              (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v198 + 48LL))(v198)) )
        {
          sub_B157E0((__int64)&v329, &v319);
          sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v329, v316);
          sub_B18290((__int64)&v339, "    Trying to fully unroll...", 0x1Du);
          sub_1049740(a8, (__int64)&v339);
          v117 = (__int64)v344;
          v339.m128i_i64[0] = (__int64)&unk_49D9D40;
          v118 = 10LL * v345;
          v119 = &v344[v118];
          if ( v344 != &v344[v118] )
          {
            v303 = v17;
            v120 = &v344[v118];
            v121 = v344;
            do
            {
              v120 -= 10;
              v122 = v120[4];
              if ( (unsigned __int64 *)v122 != v120 + 6 )
                j_j___libc_free_0(v122);
              v115 = (__int64)(v120 + 2);
              if ( (unsigned __int64 *)*v120 != v120 + 2 )
                j_j___libc_free_0(*v120);
            }
            while ( v121 != v120 );
            v17 = v303;
            v119 = v344;
          }
          if ( v119 != (unsigned __int64 *)v346 )
            _libc_free((unsigned __int64)v119);
        }
        v18[1].m128i_i32[1] = 0;
        if ( a9 )
        {
          v18[1].m128i_i32[1] = a9;
          v321 = _mm_loadu_si128(a13);
          v322 = _mm_loadu_si128(a13 + 1);
          v304 = _mm_loadu_si128(a13 + 2);
          v324 = a13[3].m128i_i64[0];
          v323 = v304;
          sub_D4BD20(&v320, v17, v115, v117, v118 * 8, v116);
          v305 = **(_QWORD **)(v17 + 32);
          if ( a9 <= v18[2].m128i_i32[1] )
          {
            v193 = sub_2880E70(v321.m128i_i64, (__int64)v18, 0, v123, v124);
            v194 = v18->m128i_i32[0];
            v195 = v193;
            v196 = 6;
            if ( v18[4].m128i_i32[0] <= 6u )
              v196 = v18[4].m128i_i32[0];
            if ( v195 < (unsigned int)(v194 * v196) )
              goto LABEL_278;
            v339.m128i_i64[0] = sub_2885680(
                                  (_QWORD *)v17,
                                  a9,
                                  a6,
                                  a7,
                                  a2,
                                  v18->m128i_i32[1] * v194 / 0x64u,
                                  v18[3].m128i_u32[2]);
            v339.m128i_i32[2] = v228;
            if ( (_BYTE)v228 )
            {
              v229 = v18->m128i_u32[1];
              if ( v339.m128i_i32[1] > 0x28F5C27u )
              {
                v229 = 100;
              }
              else if ( v339.m128i_i32[0] && v229 > (unsigned int)(100 * v339.m128i_i32[1]) / v339.m128i_i32[0] )
              {
                v229 = (unsigned int)(100 * v339.m128i_i32[1]) / v339.m128i_i32[0];
              }
              if ( v18->m128i_i32[0] * v229 / 0x64 > v339.m128i_i32[0] )
              {
LABEL_278:
                if ( v320 )
                  sub_B91220((__int64)&v320, v320);
                v18[1].m128i_i32[1] = a9;
                *a16 = 0;
                goto LABEL_55;
              }
            }
            v230 = *a8;
            v231 = sub_B2BE50(*a8);
            if ( sub_B6EA50(v231)
              || (v283 = sub_B2BE50(v230),
                  v284 = sub_B6F970(v283),
                  (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v284 + 48LL))(v284)) )
            {
              sub_B157E0((__int64)&v329, &v320);
              sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v329, v305);
              sub_B18290((__int64)&v339, "      Failed: the estimated unrolled loop size is too large", 0x3Bu);
              sub_1049740(a8, (__int64)&v339);
              v232 = v344;
              v339.m128i_i64[0] = (__int64)&unk_49D9D40;
              if ( v344 != &v344[10 * v345] )
              {
                v350 = v18;
                v233 = &v344[10 * v345];
                v307 = v17;
                v234 = v344;
                do
                {
                  v233 -= 10;
                  v235 = v233[4];
                  if ( (unsigned __int64 *)v235 != v233 + 6 )
                    j_j___libc_free_0(v235);
                  if ( (unsigned __int64 *)*v233 != v233 + 2 )
                    j_j___libc_free_0(*v233);
                }
                while ( v234 != v233 );
                v17 = v307;
                v232 = v344;
                v18 = v350;
              }
              if ( v232 != (unsigned __int64 *)v346 )
                _libc_free((unsigned __int64)v232);
            }
          }
          else
          {
            v125 = *a8;
            v126 = sub_B2BE50(*a8);
            if ( sub_B6EA50(v126)
              || (v258 = sub_B2BE50(v125),
                  v259 = sub_B6F970(v258),
                  (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v259 + 48LL))(v259)) )
            {
              sub_B157E0((__int64)&v329, &v320);
              sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v329, v305);
              sub_B18290((__int64)&v339, "      Failed: trip count ", 0x19u);
              sub_B169E0(v329.m128i_i64, "FullUnrollTripCount", 19, a9);
              v295 = sub_23FD640((__int64)&v339, (__int64)&v329);
              sub_B18290(v295, " exceeds threshold ", 0x13u);
              sub_B169E0(v325.m128i_i64, "FullUnrollTripCount", 19, v18[2].m128i_u32[1]);
              sub_23FD640(v295, (__int64)&v325);
              if ( v327 != v328 )
                j_j___libc_free_0((unsigned __int64)v327);
              if ( (_QWORD *)v325.m128i_i64[0] != v326 )
                j_j___libc_free_0(v325.m128i_u64[0]);
              if ( (__m128i *)v330.m256i_i64[2] != &v331 )
                j_j___libc_free_0(v330.m256i_u64[2]);
              if ( (__m256i *)v329.m128i_i64[0] != &v330 )
                j_j___libc_free_0(v329.m128i_u64[0]);
              sub_1049740(a8, (__int64)&v339);
              v339.m128i_i64[0] = (__int64)&unk_49D9D40;
              v127 = 10LL * v345;
              v128 = &v344[v127];
              if ( v344 != &v344[v127] )
              {
                v306 = v17;
                v129 = &v344[v127];
                v130 = v344;
                do
                {
                  v129 -= 10;
                  v131 = v129[4];
                  if ( (unsigned __int64 *)v131 != v129 + 6 )
                    j_j___libc_free_0(v131);
                  if ( (unsigned __int64 *)*v129 != v129 + 2 )
                    j_j___libc_free_0(*v129);
                }
                while ( v130 != v129 );
                v17 = v306;
                v128 = v344;
              }
              if ( v128 != (unsigned __int64 *)v346 )
                _libc_free((unsigned __int64)v128);
            }
          }
          if ( v320 )
            sub_B91220((__int64)&v320, v320);
        }
        else
        {
          v164 = *a8;
          v165 = sub_B2BE50(*a8);
          if ( sub_B6EA50(v165)
            || (v225 = sub_B2BE50(v164),
                v226 = sub_B6F970(v225),
                (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v226 + 48LL))(v226)) )
          {
            sub_B157E0((__int64)&v329, &v319);
            sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v329, v316);
            sub_B18290((__int64)&v339, "      Failed: trip count could not be statically calculated", 0x3Bu);
            sub_1049740(a8, (__int64)&v339);
            v339.m128i_i64[0] = (__int64)&unk_49D9D40;
            sub_23FD590((__int64)&v344);
          }
        }
        v132 = *a8;
        v133 = sub_B2BE50(*a8);
        if ( sub_B6EA50(v133)
          || (v199 = sub_B2BE50(v132),
              v200 = sub_B6F970(v199),
              (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v200 + 48LL))(v200)) )
        {
          sub_B157E0((__int64)&v329, &v319);
          sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v329, v316);
          sub_B18290((__int64)&v339, "    Trying upper-bound unrolling...", 0x23u);
          sub_1049740(a8, (__int64)&v339);
          v339.m128i_i64[0] = (__int64)&unk_49D9D40;
          sub_23FD590((__int64)&v344);
        }
        if ( a9 )
        {
          v166 = *a8;
          v167 = sub_B2BE50(*a8);
          if ( sub_B6EA50(v167)
            || (v250 = sub_B2BE50(v166),
                v251 = sub_B6F970(v250),
                (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v251 + 48LL))(v251)) )
          {
            sub_B157E0((__int64)&v329, &v319);
            sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v329, v316);
            sub_B18290((__int64)&v339, "      Failed: trip count is known", 0x21u);
            sub_1049740(a8, (__int64)&v339);
            v339.m128i_i64[0] = (__int64)&unk_49D9D40;
            sub_23FD590((__int64)&v344);
          }
LABEL_242:
          v168 = *a8;
          v169 = sub_B2BE50(*a8);
          if ( sub_B6EA50(v169)
            || (v203 = sub_B2BE50(v168),
                v204 = sub_B6F970(v203),
                (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v204 + 48LL))(v204)) )
          {
            sub_B157E0((__int64)&v329, &v319);
            sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v329, v316);
            sub_B18290((__int64)&v339, "    Trying loop peeling...", 0x1Au);
            sub_1049740(a8, (__int64)&v339);
            v339.m128i_i64[0] = (__int64)&unk_49D9D40;
            sub_23FD590((__int64)&v344);
          }
          sub_2A05E00(v17, v288, (_DWORD)a15, a9, v298, (_DWORD)a6, a5, v18->m128i_i32[0]);
          v170 = *a15;
          if ( *a15 )
          {
            v18[2].m128i_i8[13] = 0;
            v18[1].m128i_i32[1] = 1;
            goto LABEL_55;
          }
          v171 = *a8;
          v172 = sub_B2BE50(*a8);
          if ( sub_B6EA50(v172)
            || (v256 = sub_B2BE50(v171),
                v257 = sub_B6F970(v256),
                (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v257 + 48LL))(v257)) )
          {
            sub_B157E0((__int64)&v329, &v319);
            sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v329, v316);
            sub_B18290((__int64)&v339, "      Failed: cannot do loop peeling", 0x24u);
            sub_1049740(a8, (__int64)&v339);
            v339.m128i_i64[0] = (__int64)&unk_49D9D40;
            sub_23FD590((__int64)&v344);
          }
          if ( a9 )
            v18[2].m128i_i8[12] |= v311;
          v173 = *a8;
          v174 = sub_B2BE50(*a8);
          if ( sub_B6EA50(v174)
            || (v254 = sub_B2BE50(v173),
                v255 = sub_B6F970(v254),
                (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v255 + 48LL))(v255)) )
          {
            sub_B157E0((__int64)&v329, &v319);
            sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v329, v316);
            sub_B18290((__int64)&v339, "    Trying static partial unrolling...", 0x26u);
            sub_1049740(a8, (__int64)&v339);
            v339.m128i_i64[0] = (__int64)&unk_49D9D40;
            sub_23FD590((__int64)&v344);
          }
          v179 = _mm_loadu_si128(a13 + 2);
          v339 = _mm_loadu_si128(a13);
          v180 = _mm_loadu_si128(a13 + 1);
          v342.m128i_i64[0] = a13[3].m128i_i64[0];
          v340 = v180;
          v341 = v179;
          sub_D4BD20(&v325, v17, v175, v176, v177, v178);
          v329.m128i_i64[0] = **(_QWORD **)(v17 + 32);
          if ( a9 )
          {
            if ( !v18[2].m128i_i8[12] )
            {
              sub_287F950(a8, &v325, v329.m128i_i64);
              goto LABEL_256;
            }
            if ( *(_QWORD *)(v17 + 16) != *(_QWORD *)(v17 + 8) )
            {
              sub_287FAB0(a8, &v325, v329.m128i_i64);
LABEL_256:
              if ( v325.m128i_i64[0] )
                sub_B91220((__int64)&v325, v325.m128i_i64[0]);
              v18[1].m128i_i32[1] = v170;
              if ( v308 && a9 != 0 && a9 != v170 && (unsigned __int8)sub_287ED80(*a8) )
              {
                v315 = **(_QWORD **)(v17 + 32);
                sub_D4BD20(&v321, v17, v274, v275, v276, v315);
                sub_B157E0((__int64)&v325, &v321);
                sub_B17640(
                  (__int64)&v339,
                  (__int64)"loop-unroll",
                  (__int64)"FullUnrollAsDirectedTooLarge",
                  28,
                  &v325,
                  v315);
                sub_B18290(
                  (__int64)&v339,
                  "Unable to fully unroll loop as directed by unroll pragma because unrolled size is too large.",
                  0x5Cu);
                v280 = _mm_loadu_si128((const __m128i *)&v340.m128i_u64[1]);
                v281 = _mm_loadu_si128(&v342);
                v329.m128i_i32[2] = v339.m128i_i32[2];
                v282 = _mm_loadu_si128(&v343);
                *(__m128i *)&v330.m256i_u64[1] = v280;
                v329.m128i_i8[12] = v339.m128i_i8[12];
                v331 = v281;
                v330.m256i_i64[0] = v340.m128i_i64[0];
                v332 = v282;
                v329.m128i_i64[0] = (__int64)&unk_49D9D40;
                v330.m256i_i64[3] = v341.m128i_i64[1];
                v333 = v335;
                v334 = 0x400000000LL;
                if ( v345 )
                  sub_28821D0((__int64)&v333, (__int64)&v344, v277, v278, (__int64)&v344, v279);
                v336 = v347;
                v337 = v348;
                v338 = v349;
                v329.m128i_i64[0] = (__int64)&unk_49D9DB0;
                v339.m128i_i64[0] = (__int64)&unk_49D9D40;
                sub_23FD590((__int64)&v344);
                if ( v321.m128i_i64[0] )
                  sub_B91220((__int64)&v321, v321.m128i_i64[0]);
                sub_1049740(a8, (__int64)&v329);
                v329.m128i_i64[0] = (__int64)&unk_49D9D40;
                sub_23FD590((__int64)&v333);
              }
              if ( v18->m128i_i32[3] != -1 && !v18[1].m128i_i32[1] && v309 && (unsigned __int8)sub_287ED80(*a8) )
              {
                v186 = **(_QWORD **)(v17 + 32);
                sub_D4BD20(&v321, v17, v182, v183, v184, v185);
                sub_B157E0((__int64)&v325, &v321);
                sub_B17640((__int64)&v339, (__int64)"loop-unroll", (__int64)"UnrollAsDirectedTooLarge", 24, &v325, v186);
                sub_B18290(
                  (__int64)&v339,
                  "Unable to unroll loop as directed by unroll(enable) pragma because unrolled size is too large.",
                  0x5Eu);
                v189 = _mm_loadu_si128((const __m128i *)&v340.m128i_u64[1]);
                v190 = _mm_loadu_si128(&v342);
                v329.m128i_i32[2] = v339.m128i_i32[2];
                v191 = _mm_loadu_si128(&v343);
                *(__m128i *)&v330.m256i_u64[1] = v189;
                v329.m128i_i8[12] = v339.m128i_i8[12];
                v331 = v190;
                v330.m256i_i64[0] = v340.m128i_i64[0];
                v332 = v191;
                v329.m128i_i64[0] = (__int64)&unk_49D9D40;
                v330.m256i_i64[3] = v341.m128i_i64[1];
                v333 = v335;
                v334 = 0x400000000LL;
                if ( v345 )
                  sub_28821D0((__int64)&v333, (__int64)&v344, v187, v188, (__int64)&v344, v345);
                v336 = v347;
                v337 = v348;
                v338 = v349;
                v329.m128i_i64[0] = (__int64)&unk_49D9DB0;
                v339.m128i_i64[0] = (__int64)&unk_49D9D40;
                sub_23FD590((__int64)&v344);
                if ( v321.m128i_i64[0] )
                  sub_B91220((__int64)&v321, v321.m128i_i64[0]);
                sub_1049740(a8, (__int64)&v329);
                v329.m128i_i64[0] = (__int64)&unk_49D9D40;
                sub_23FD590((__int64)&v333);
              }
              goto LABEL_55;
            }
            v247 = v18[1].m128i_u32[1];
            if ( !(_DWORD)v247 )
              v247 = a9;
            if ( v18->m128i_i32[3] == -1 )
            {
              v248 = v18[1].m128i_u32[3];
              v249 = a9;
              goto LABEL_385;
            }
            v317 = v247;
            v260 = sub_2880E70(v339.m128i_i64, (__int64)v18, v247, v247, v181);
            v261 = v317;
            if ( v260 > v18->m128i_u32[3] )
            {
              v262 = v18[2].m128i_u32[2];
              v263 = v18->m128i_u32[3];
              if ( v262 + 1 >= v263 )
                v263 = v262 + 1;
              v261 = (v263 - v262) / ((unsigned int)v288 - v262);
            }
            v248 = v18[1].m128i_u32[3];
            if ( v248 <= (unsigned int)v261 )
              v261 = v248;
            if ( (_DWORD)v261 )
            {
              while ( 1 )
              {
                v249 = v261;
                v264 = a9 % (unsigned int)v261;
                v261 = (unsigned int)(v261 - 1);
                if ( !(v264 | v249 & (unsigned int)v261) )
                  break;
                if ( !(_DWORD)v261 )
                  goto LABEL_435;
              }
              if ( v18[2].m128i_i8[14] )
              {
                if ( v249 == 1 )
                  goto LABEL_419;
LABEL_385:
                if ( v248 <= v249 )
                  v249 = v248;
                v170 = v249;
                if ( v249 )
                  goto LABEL_256;
                goto LABEL_388;
              }
LABEL_424:
              if ( v249 != 1 )
              {
                v248 = v18[1].m128i_u32[3];
                goto LABEL_385;
              }
            }
            else
            {
LABEL_435:
              if ( v18[2].m128i_i8[14] )
              {
LABEL_419:
                v265 = v18[1].m128i_u32[2];
                if ( (_DWORD)v265 )
                {
                  v318 = v17;
                  v266 = v18[1].m128i_u32[2];
                  do
                  {
                    v265 = sub_2880E70(v339.m128i_i64, (__int64)v18, v266, v261, v265);
                    if ( v265 <= v18->m128i_u32[3] )
                    {
                      v249 = v266;
                      v17 = v318;
                      goto LABEL_424;
                    }
                    v266 >>= 1;
                  }
                  while ( v266 );
                  v17 = v318;
                }
              }
            }
LABEL_388:
            v170 = 0;
            sub_287F7F0(a8, &v325, v329.m128i_i64);
            goto LABEL_256;
          }
          sub_287FC10(a8, &v325, v329.m128i_i64);
          v205 = (__m128i *)v325.m128i_i64[0];
          if ( v325.m128i_i64[0] )
            sub_B91220((__int64)&v325, v325.m128i_i64[0]);
          if ( v312 && (unsigned __int8)sub_287ED80(*a8) )
          {
            v297 = **(_QWORD **)(v17 + 32);
            sub_D4BD20(&v321, v17, v238, v239, v240, v297);
            sub_B157E0((__int64)&v325, &v321);
            sub_B17640(
              (__int64)&v339,
              (__int64)"loop-unroll",
              (__int64)"CantFullUnrollAsDirectedRuntimeTripCount",
              40,
              &v325,
              v297);
            sub_B18290(
              (__int64)&v339,
              "Unable to fully unroll loop as directed by unroll(full) pragma because loop has a runtime trip count.",
              0x65u);
            v244 = _mm_loadu_si128((const __m128i *)&v340.m128i_u64[1]);
            v245 = _mm_loadu_si128(&v342);
            v329.m128i_i32[2] = v339.m128i_i32[2];
            v246 = _mm_loadu_si128(&v343);
            *(__m128i *)&v330.m256i_u64[1] = v244;
            v329.m128i_i8[12] = v339.m128i_i8[12];
            v331 = v245;
            v330.m256i_i64[0] = v340.m128i_i64[0];
            v332 = v246;
            v329.m128i_i64[0] = (__int64)&unk_49D9D40;
            v330.m256i_i64[3] = v341.m128i_i64[1];
            v333 = v335;
            v334 = 0x400000000LL;
            if ( v345 )
              sub_28821D0((__int64)&v333, (__int64)&v344, v241, v242, (__int64)&v344, v243);
            v336 = v347;
            v337 = v348;
            v338 = v349;
            v329.m128i_i64[0] = (__int64)&unk_49D9DB0;
            v339.m128i_i64[0] = (__int64)&unk_49D9D40;
            sub_23FD590((__int64)&v344);
            if ( v321.m128i_i64[0] )
              sub_B91220((__int64)&v321, v321.m128i_i64[0]);
            v205 = &v329;
            sub_1049740(a8, (__int64)&v329);
            v329.m128i_i64[0] = (__int64)&unk_49D9D40;
            sub_23FD590((__int64)&v333);
          }
          if ( (unsigned __int8)sub_287ED80(*a8) )
          {
            sub_B157E0((__int64)&v329, &v319);
            sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v329, v316);
            sub_B18290((__int64)&v339, "    Trying runtime unrolling...", 0x1Fu);
            v205 = &v339;
            sub_1049740(a8, (__int64)&v339);
            v339.m128i_i64[0] = (__int64)&unk_49D9D40;
            sub_23FD590((__int64)&v344);
          }
          v210 = sub_D49300(v17, (__int64)v205, v206, v207, v208, v209);
          if ( v210 && sub_2A11940(v210, "llvm.loop.unroll.runtime.disable", 32) )
          {
            if ( !(unsigned __int8)sub_287ED80(*a8) )
              goto LABEL_339;
          }
          else
          {
            if ( a10 && !v18[3].m128i_i8[0] && v18[2].m128i_i32[0] > a10 )
              goto LABEL_339;
            sub_B2EE70((__int64)&v339, *(_QWORD *)(**(_QWORD **)(v17 + 32) + 72LL), 0);
            if ( v340.m128i_i8[0] )
            {
              v325.m128i_i64[0] = sub_F6EC60(v17, 0);
              if ( v325.m128i_i8[4] )
              {
                if ( v325.m128i_i32[0] < (unsigned int)dword_5002B48 )
                {
                  if ( (unsigned __int8)sub_287ED80(*a8) )
                  {
                    sub_B157E0((__int64)&v329, &v319);
                    sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v329, v316);
                    sub_B18290((__int64)&v339, "      Failed : runtime trip count is too small", 0x2Eu);
                    sub_1049740(a8, (__int64)&v339);
                    v339.m128i_i64[0] = (__int64)&unk_49D9D40;
                    sub_23FD590((__int64)&v344);
                  }
                  goto LABEL_340;
                }
                v18[2].m128i_i8[15] = 1;
              }
            }
            if ( v310 || v314 || v309 )
            {
              v18[2].m128i_i8[13] = 1;
LABEL_308:
              if ( !(_BYTE)qword_50021A8 )
              {
LABEL_315:
                if ( v18[4].m128i_i8[5]
                  && (v216 = a13,
                      v267 = sub_2882FC0(v17, (__int64)a13, a17, (__int64)v18, a6, (__int64)a8),
                      (LODWORD(v215) = v267) != 0) )
                {
                  v18[1].m128i_i32[1] = v267;
                  v268 = v18[2].m128i_i8[14] == 0;
                  v18[4].m128i_i8[4] = qword_5001D48;
                  if ( !v268 )
                    goto LABEL_324;
                }
                else
                {
                  v215 = v18[1].m128i_u32[1];
                  if ( !(_DWORD)v215 )
                  {
                    v215 = v18[1].m128i_u32[2];
                    v18[1].m128i_i32[1] = v215;
                    if ( !(_DWORD)v215 )
                    {
                      if ( !v18[2].m128i_i8[14] )
                        goto LABEL_330;
                      goto LABEL_433;
                    }
                  }
                  while ( 1 )
                  {
                    v216 = v18;
                    v217 = sub_2880E70(a13->m128i_i64, (__int64)v18, 0, v215, v211);
                    LODWORD(v215) = v18[1].m128i_i32[1];
                    v211 = v217;
                    if ( v217 <= v18->m128i_u32[3] )
                      break;
                    v215 = (unsigned int)v215 >> 1;
                    v18[1].m128i_i32[1] = v215;
                    if ( !(_DWORD)v215 )
                    {
                      if ( v18[2].m128i_i8[14] )
                        goto LABEL_324;
                      goto LABEL_330;
                    }
                  }
                  if ( v18[2].m128i_i8[14] )
                    goto LABEL_324;
                  if ( !(_DWORD)v215 )
                    goto LABEL_330;
                }
                v219 = a12 % (unsigned int)v215;
                if ( a12 % (unsigned int)v215 )
                {
                  while ( 1 )
                  {
                    v215 = (unsigned int)v215 >> 1;
                    if ( !(_DWORD)v215 )
                      break;
                    v219 = a12 % (unsigned int)v215;
                    if ( !(a12 % (unsigned int)v215) )
                    {
                      v18[1].m128i_i32[1] = v215;
                      goto LABEL_334;
                    }
                  }
                  v18[1].m128i_i32[1] = 0;
LABEL_334:
                  if ( (unsigned int)sub_287EE90(v17, (__int64)v216, v219, v215, v211, v218)
                    && !v18[2].m128i_i8[14]
                    && (unsigned __int8)sub_287ED80(*a8) )
                  {
                    sub_B157E0((__int64)&v329, &v319);
                    sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v329, v316);
                    sub_B18290((__int64)&v339, "      Warning : remainder loops not allowed", 0x2Bu);
                    sub_1049740(a8, (__int64)&v339);
                    v339.m128i_i64[0] = (__int64)&unk_49D9D40;
                    sub_23FD590((__int64)&v344);
                  }
                  LODWORD(v215) = v18[1].m128i_i32[1];
                }
LABEL_324:
                v220 = v18[1].m128i_u32[3];
                if ( v220 < (unsigned int)v215 )
                {
                  v18[1].m128i_i32[1] = v220;
LABEL_326:
                  if ( a10 && a10 < v220 )
                  {
                    v220 = a10;
                    v18[1].m128i_i32[1] = a10;
                  }
                  if ( v220 > 1 )
                    goto LABEL_55;
LABEL_330:
                  v18[1].m128i_i32[1] = 0;
                  goto LABEL_55;
                }
LABEL_433:
                v220 = v215;
                goto LABEL_326;
              }
              v339.m128i_i64[0] = v17;
              v329.m128i_i32[0] = v288;
              v212 = *(_QWORD **)v17;
              if ( *(_QWORD *)v17 )
              {
                v213 = 1;
                do
                {
                  v212 = (_QWORD *)*v212;
                  ++v213;
                }
                while ( v212 );
                v214 = v17;
                if ( v213 > 1 )
                  goto LABEL_313;
              }
              v271 = sub_DCF3A0(a6, (char *)v17, 1);
              if ( sub_D96A50(v271) || *(_WORD *)(v271 + 24) )
                goto LABEL_453;
              v272 = *(_QWORD *)(v271 + 32);
              v273 = *(_QWORD **)(v272 + 24);
              if ( *(_DWORD *)(v272 + 32) > 0x40u )
                v273 = (_QWORD *)*v273;
              if ( (unsigned int)dword_5002288 <= (unsigned __int64)v273 )
              {
LABEL_453:
                v214 = v339.m128i_i64[0];
LABEL_313:
                if ( *(_QWORD *)(v214 + 16) == *(_QWORD *)(v214 + 8) )
                {
                  if ( (unsigned int)dword_5002368 >= v329.m128i_i32[0] )
                    goto LABEL_315;
                  sub_2881ED0(a8, v339.m128i_i64, (unsigned int *)&v329);
                }
                else
                {
                  sub_287FD70(a8, v339.m128i_i64);
                }
                goto LABEL_339;
              }
              sub_287FF00(a8, v339.m128i_i64);
LABEL_339:
              v18[1].m128i_i32[1] = 0;
LABEL_340:
              v311 = 0;
              goto LABEL_55;
            }
            if ( v18[2].m128i_i8[13] )
              goto LABEL_308;
            if ( !(unsigned __int8)sub_287ED80(*a8) )
              goto LABEL_339;
          }
          sub_B157E0((__int64)&v329, &v319);
          sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v329, v316);
          sub_B18290((__int64)&v339, "      Failed : runtime unrolling is disabled", 0x2Cu);
          sub_1049740(a8, (__int64)&v339);
          v339.m128i_i64[0] = (__int64)&unk_49D9D40;
          sub_23FD590((__int64)&v344);
          goto LABEL_339;
        }
        v137 = _mm_loadu_si128(a13 + 2);
        v138 = _mm_loadu_si128(a13);
        v139 = _mm_loadu_si128(a13 + 1);
        v331.m128i_i64[0] = a13[3].m128i_i64[0];
        v329 = v138;
        *(__m128i *)v330.m256i_i8 = v139;
        *(__m128i *)&v330.m256i_u64[2] = v137;
        sub_D4BD20(&v321, v17, v134, v135, 0, v136);
        v142 = **(_QWORD **)(v17 + 32);
        if ( !a10 )
        {
          if ( !(unsigned __int8)sub_287ED80(*a8) )
            goto LABEL_286;
          sub_B157E0((__int64)&v325, &v321);
          sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v325, v142);
          v223 = "    Failed: the max trip count could not be statically calculated";
          v224 = 65;
          goto LABEL_346;
        }
        if ( !v18[3].m128i_i8[1] )
        {
          if ( !(unsigned __int8)sub_287ED80(*a8) )
            goto LABEL_286;
          sub_B157E0((__int64)&v325, &v321);
          sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v325, v142);
          v223 = "    Failed: Upper-bound unrolling is disabled";
          goto LABEL_345;
        }
        if ( v313 )
        {
          if ( (unsigned __int8)sub_287ED80(*a8) )
          {
            sub_B157E0((__int64)&v325, &v321);
            sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v325, v142);
            v223 = "    Failed: should not upper-bound unroll the loop when user provide pragma unroll count";
            v224 = 88;
            goto LABEL_346;
          }
          goto LABEL_286;
        }
        if ( v312 )
        {
          v227 = sub_2880E70(v329.m128i_i64, (__int64)v18, 0, v140, v141);
          v144 = v18->m128i_i32[0];
          v145 = v227;
          v146 = v18[4].m128i_u32[0];
        }
        else
        {
          if ( !a11 && v18[4].m128i_i32[0] <= 1u )
            goto LABEL_349;
          v143 = sub_2880E70(v329.m128i_i64, (__int64)v18, 0, v140, v141);
          v144 = v18->m128i_i32[0];
          v145 = v143;
          v146 = v18[4].m128i_u32[0];
          if ( a10 > v146 )
          {
            v147 = qword_5002D08;
            v148 = v144;
            goto LABEL_200;
          }
        }
        v147 = qword_5002448;
        if ( v146 > 6 )
          v146 = 6;
        v148 = v144 * v146;
LABEL_200:
        if ( v148 > v145 && a10 <= v147 && !v330.m256i_i32[7] )
        {
          if ( v321.m128i_i64[0] )
            sub_B91220((__int64)&v321, v321.m128i_i64[0]);
          v18[1].m128i_i32[1] = a10;
          *a16 = 1;
          goto LABEL_55;
        }
LABEL_349:
        if ( (unsigned __int8)sub_287ED80(*a8) )
        {
          sub_B157E0((__int64)&v325, &v321);
          sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v325, v142);
          v223 = "    Failed: upper-bound may be not profitable";
LABEL_345:
          v224 = 45;
LABEL_346:
          sub_B18290((__int64)&v339, v223, v224);
          sub_1049740(a8, (__int64)&v339);
          v339.m128i_i64[0] = (__int64)&unk_49D9D40;
          sub_23FD590((__int64)&v344);
        }
LABEL_286:
        if ( v321.m128i_i64[0] )
          sub_B91220((__int64)&v321, v321.m128i_i64[0]);
        goto LABEL_242;
      }
    }
    else
    {
      v291 = v289;
    }
    if ( (unsigned __int8)sub_287ED80(*a8) )
    {
      sub_B157E0((__int64)&v329, &v320);
      sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v329, v301);
      sub_B18290((__int64)&v339, "      Failed: not allow remainder loops", 0x27u);
      sub_1049740(a8, (__int64)&v339);
      v339.m128i_i64[0] = (__int64)&unk_49D9D40;
      sub_23FD590((__int64)&v344);
    }
    if ( (unsigned __int8)sub_287ED80(*a8) )
    {
      sub_B157E0((__int64)&v329, &v320);
      sub_B17430((__int64)&v339, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v329, v301);
      sub_B18290((__int64)&v339, "               and pragma count ", 0x20u);
      sub_B169E0(v329.m128i_i64, "PInfo.PragmaCount", 17, v313);
      v285 = sub_23FD640((__int64)&v339, (__int64)&v329);
      sub_B18290(v285, " does not divide trip multiple ", 0x1Fu);
      sub_B169E0(v325.m128i_i64, "TripMultiple", 12, a12);
      sub_23FD640(v285, (__int64)&v325);
      if ( v327 != v328 )
        j_j___libc_free_0((unsigned __int64)v327);
      if ( (_QWORD *)v325.m128i_i64[0] != v326 )
        j_j___libc_free_0(v325.m128i_u64[0]);
      if ( (__m128i *)v330.m256i_i64[2] != &v331 )
        j_j___libc_free_0(v330.m256i_u64[2]);
      if ( (__m256i *)v329.m128i_i64[0] != &v330 )
        j_j___libc_free_0(v329.m128i_u64[0]);
      sub_1049740(a8, (__int64)&v339);
      v339.m128i_i64[0] = (__int64)&unk_49D9D40;
      sub_23FD590((__int64)&v344);
    }
    goto LABEL_124;
  }
  v51 = sub_C52410();
  v52 = v51 + 1;
  v53 = sub_C959E0();
  v54 = (_QWORD *)v51[2];
  if ( v54 )
  {
    v55 = v51 + 1;
    do
    {
      while ( 1 )
      {
        v56 = v54[2];
        v57 = v54[3];
        if ( v53 <= v54[4] )
          break;
        v54 = (_QWORD *)v54[3];
        if ( !v57 )
          goto LABEL_43;
      }
      v55 = v54;
      v54 = (_QWORD *)v54[2];
    }
    while ( v56 );
LABEL_43:
    if ( v52 != v55 && v53 >= v55[4] )
      v52 = v55;
  }
  if ( v52 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v58 = v52[7];
    if ( v58 )
    {
      v59 = v52 + 6;
      do
      {
        while ( 1 )
        {
          v60 = *(_QWORD *)(v58 + 16);
          v61 = *(_QWORD *)(v58 + 24);
          if ( *(_DWORD *)(v58 + 32) >= dword_50031C8 )
            break;
          v58 = *(_QWORD *)(v58 + 24);
          if ( !v61 )
            goto LABEL_52;
        }
        v59 = (_QWORD *)v58;
        v58 = *(_QWORD *)(v58 + 16);
      }
      while ( v60 );
LABEL_52:
      if ( v52 + 6 != v59 && dword_50031C8 >= *((_DWORD *)v59 + 8) && *((int *)v59 + 9) > 0 )
        sub_C64ED0("Cannot specify both explicit peel count and explicit unroll count", 0);
    }
  }
  a14[5] = 1;
  *((_BYTE *)a14 + 45) = 0;
  v311 = 1;
LABEL_55:
  if ( v319 )
    sub_B91220((__int64)&v319, v319);
  return v311;
}
