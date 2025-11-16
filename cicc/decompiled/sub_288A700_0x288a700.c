// Function: sub_288A700
// Address: 0x288a700
//
__int64 __fastcall sub_288A700(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 **a5,
        __int64 a6,
        __int64 *a7,
        __int64 *a8,
        __int64 a9,
        __int64 *a10,
        unsigned __int8 a11,
        int a12,
        char a13,
        char a14,
        __int8 a15,
        __int64 a16,
        __int64 a17,
        unsigned __int16 a18,
        unsigned __int16 a19,
        unsigned __int16 a20,
        unsigned __int16 a21,
        unsigned __int16 a22,
        __int64 a23,
        __int64 a24)
{
  _QWORD *v25; // rax
  unsigned int v26; // ebx
  __int64 v27; // r13
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int8 *v31; // rax
  size_t v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  unsigned __int64 *v37; // r13
  unsigned __int64 *v38; // r14
  unsigned __int64 v39; // rdi
  const char **v40; // rsi
  __int64 v41; // rdx
  __int64 i; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rax
  bool v46; // r15
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  char v51; // r14
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  __int64 v56; // r15
  char v57; // r14
  __int32 v58; // eax
  unsigned int v59; // r14d
  __int64 *v60; // rbx
  __int64 *v61; // r13
  unsigned int v62; // eax
  char v63; // al
  __int64 v64; // rdx
  __int64 v65; // r8
  char v66; // r13
  __int64 v67; // rsi
  __int64 v68; // rcx
  __int64 v69; // r8
  unsigned int v70; // ebx
  __int64 *v71; // rax
  __int64 v72; // rdx
  __int64 v73; // rcx
  __int64 v74; // r8
  __int64 v75; // r9
  __int64 *v76; // rax
  __int64 v77; // rdx
  __int64 v78; // rcx
  __int64 v79; // r8
  __int64 v80; // r9
  __int64 v82; // r15
  __int64 v83; // rax
  __int64 v84; // rdx
  __int64 v85; // rcx
  __int64 v86; // r8
  __int64 v87; // r9
  __int64 j; // rcx
  unsigned __int8 v89; // di
  __int64 v90; // rax
  _BYTE **v91; // rax
  _BYTE *v92; // rax
  unsigned __int8 v93; // dl
  _BYTE **v94; // rax
  size_t v95; // rdx
  __int8 *v96; // rsi
  __int64 v97; // rdx
  __int64 v98; // rax
  __int64 v99; // r12
  __int64 v100; // rax
  __int8 *v101; // rsi
  size_t v102; // rdx
  __int64 v103; // r12
  __int64 v104; // rax
  __int64 v105; // rax
  unsigned __int8 v106; // di
  __int64 v107; // rax
  __int64 *v108; // rax
  __int64 v109; // rax
  __int64 v110; // rax
  __int64 v111; // rax
  __int64 v112; // rax
  __int64 v113; // rax
  unsigned __int8 v114; // dl
  __int64 *v115; // rax
  __int64 v116; // rdx
  __int64 v117; // rcx
  __int64 v118; // r8
  __int64 v119; // r9
  __int64 v120; // rax
  unsigned __int8 v121; // dl
  __int64 v122; // rax
  __int64 v123; // r14
  __int64 v124; // rdx
  __int64 v125; // rcx
  __int64 v126; // r8
  __int64 v127; // r9
  __int64 v128; // rax
  unsigned __int8 v129; // dl
  __int64 v130; // rax
  __int64 v131; // rax
  unsigned __int64 *v132; // r14
  unsigned __int64 *v133; // r15
  unsigned __int64 v134; // rdi
  __int64 v135; // rax
  __int64 v136; // rax
  unsigned int v137; // r15d
  __int64 v138; // rax
  __int64 v139; // rdx
  unsigned __int64 v140; // rax
  __int64 v141; // r14
  int v142; // eax
  unsigned int v143; // ebx
  int v144; // r15d
  __int64 v145; // rsi
  _QWORD *v146; // rax
  _QWORD *v147; // rdx
  __int64 v148; // rax
  __int64 v149; // rax
  __int64 v150; // rdx
  __int64 v151; // rcx
  __int64 v152; // r8
  __int64 v153; // r9
  unsigned int v154; // eax
  _QWORD *v155; // r12
  __int64 v156; // rax
  _QWORD *v157; // r13
  __int64 v158; // rdx
  __int64 v159; // rax
  __int64 v160; // rax
  __int64 v161; // rax
  __int64 v162; // r14
  __int64 v163; // rbx
  __int64 v164; // rcx
  __int64 v165; // r9
  __m128i v166; // xmm0
  __m128i v167; // xmm2
  __int64 v168; // r8
  __int64 v169; // rax
  _QWORD *v170; // rbx
  _QWORD *v171; // r12
  __int64 v172; // rsi
  unsigned __int64 v173; // r8
  int v174; // eax
  char *v175; // rsi
  __int64 v176; // rbx
  size_t v177; // rax
  unsigned int v178; // edx
  const char *v179; // r14
  __int8 *v180; // rsi
  __int64 v181; // rdi
  size_t v182; // rdx
  __int64 v183; // rax
  const char *v184; // rbx
  __int64 v185; // r14
  __int64 v186; // [rsp+8h] [rbp-698h]
  unsigned int v187; // [rsp+10h] [rbp-690h]
  __int32 v188; // [rsp+20h] [rbp-680h]
  unsigned int v189; // [rsp+20h] [rbp-680h]
  unsigned int v190; // [rsp+20h] [rbp-680h]
  unsigned int v193; // [rsp+58h] [rbp-648h]
  __int64 v194; // [rsp+58h] [rbp-648h]
  unsigned int v197; // [rsp+70h] [rbp-630h]
  __int64 v198; // [rsp+70h] [rbp-630h]
  __int64 v200; // [rsp+80h] [rbp-620h]
  char v201; // [rsp+97h] [rbp-609h] BYREF
  __int64 v202; // [rsp+98h] [rbp-608h] BYREF
  __int64 v203; // [rsp+A0h] [rbp-600h] BYREF
  __int64 v204; // [rsp+A8h] [rbp-5F8h] BYREF
  __m128i v205; // [rsp+B0h] [rbp-5F0h] BYREF
  __m128i v206; // [rsp+C0h] [rbp-5E0h] BYREF
  unsigned int v207; // [rsp+D4h] [rbp-5CCh]
  char v208; // [rsp+DCh] [rbp-5C4h]
  __int64 v209; // [rsp+100h] [rbp-5A0h] BYREF
  int v210; // [rsp+10Ch] [rbp-594h]
  unsigned int v211; // [rsp+114h] [rbp-58Ch]
  int v212; // [rsp+128h] [rbp-578h]
  char v213; // [rsp+12Ch] [rbp-574h]
  unsigned __int8 v214; // [rsp+12Dh] [rbp-573h]
  char v215; // [rsp+12Eh] [rbp-572h]
  __int8 v216; // [rsp+12Fh] [rbp-571h]
  __int8 v217; // [rsp+130h] [rbp-570h]
  __int8 v218; // [rsp+132h] [rbp-56Eh]
  __int8 v219; // [rsp+144h] [rbp-55Ch]
  __int32 v220; // [rsp+148h] [rbp-558h]
  __int8 v221; // [rsp+14Ch] [rbp-554h]
  __int64 *v222; // [rsp+150h] [rbp-550h] BYREF
  __int64 v223; // [rsp+158h] [rbp-548h]
  _BYTE v224[64]; // [rsp+160h] [rbp-540h] BYREF
  __int64 *v225; // [rsp+1A0h] [rbp-500h] BYREF
  __int64 v226; // [rsp+1A8h] [rbp-4F8h] BYREF
  __int64 v227; // [rsp+1B0h] [rbp-4F0h] BYREF
  __int64 v228; // [rsp+1B8h] [rbp-4E8h]
  __int64 *v229; // [rsp+1C0h] [rbp-4E0h]
  __int64 v230; // [rsp+1D0h] [rbp-4D0h] BYREF
  __int64 *v231; // [rsp+1F0h] [rbp-4B0h] BYREF
  __int64 **v232; // [rsp+1F8h] [rbp-4A8h]
  __int64 v233; // [rsp+200h] [rbp-4A0h] BYREF
  int v234; // [rsp+208h] [rbp-498h]
  char v235; // [rsp+20Ch] [rbp-494h]
  __int64 *v236; // [rsp+210h] [rbp-490h] BYREF
  __int64 v237; // [rsp+220h] [rbp-480h] BYREF
  __m128i v238; // [rsp+310h] [rbp-390h] BYREF
  __int64 v239; // [rsp+320h] [rbp-380h] BYREF
  __m128i v240; // [rsp+328h] [rbp-378h]
  __int64 v241; // [rsp+338h] [rbp-368h]
  _OWORD v242[2]; // [rsp+340h] [rbp-360h] BYREF
  _QWORD v243[2]; // [rsp+360h] [rbp-340h] BYREF
  _BYTE v244[324]; // [rsp+370h] [rbp-330h] BYREF
  int v245; // [rsp+4B4h] [rbp-1ECh]
  __int64 v246; // [rsp+4B8h] [rbp-1E8h]
  const char *v247; // [rsp+4C0h] [rbp-1E0h] BYREF
  __int64 v248; // [rsp+4C8h] [rbp-1D8h]
  const char *v249; // [rsp+4D0h] [rbp-1D0h]
  __int64 v250; // [rsp+4D8h] [rbp-1C8h]
  _QWORD *v251; // [rsp+4E8h] [rbp-1B8h]
  unsigned int v252; // [rsp+4F8h] [rbp-1A8h]
  char v253; // [rsp+500h] [rbp-1A0h]
  unsigned __int64 *v254; // [rsp+510h] [rbp-190h] BYREF
  unsigned int v255; // [rsp+518h] [rbp-188h]
  _BYTE v256[384]; // [rsp+520h] [rbp-180h] BYREF

  v25 = *(_QWORD **)a1;
  if ( *(_QWORD *)a1 )
  {
    v26 = 1;
    do
    {
      v25 = (_QWORD *)*v25;
      ++v26;
    }
    while ( v25 );
    sub_D4BD20(&v202, a1, a3, (__int64)a4, (__int64)a5, a6);
    v200 = **(_QWORD **)(a1 + 32);
    v27 = *a7;
    v28 = sub_B2BE50(*a7);
    if ( sub_B6EA50(v28) )
      goto LABEL_5;
  }
  else
  {
    v26 = 1;
    sub_D4BD20(&v202, a1, a3, (__int64)a4, (__int64)a5, a6);
    v200 = **(_QWORD **)(a1 + 32);
    v27 = *a7;
    v113 = sub_B2BE50(*a7);
    if ( sub_B6EA50(v113) )
    {
      sub_B157E0((__int64)&v238, &v202);
      v26 = 1;
      sub_B17430((__int64)&v247, (__int64)"loop-unroll", (__int64)"tryToUnrollLoop", 15, &v238, v200);
      goto LABEL_133;
    }
  }
  v111 = sub_B2BE50(v27);
  v112 = sub_B6F970(v111);
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v112 + 48LL))(v112) )
    goto LABEL_22;
LABEL_5:
  sub_B157E0((__int64)&v238, &v202);
  sub_B17430((__int64)&v247, (__int64)"loop-unroll", (__int64)"tryToUnrollLoop", 15, &v238, v200);
  if ( v26 > 1 )
  {
    sub_B18290((__int64)&v247, "Starting analysis in nested loop (loop depth : ", 0x2Fu);
    sub_B169E0(v238.m128i_i64, "LoopDepth", 9, v26);
    v29 = sub_23FD640((__int64)&v247, (__int64)&v238);
    sub_B18290(v29, ")", 1u);
    if ( (_OWORD *)v240.m128i_i64[1] != v242 )
      j_j___libc_free_0(v240.m128i_u64[1]);
    if ( (__int64 *)v238.m128i_i64[0] != &v239 )
      j_j___libc_free_0(v238.m128i_u64[0]);
    goto LABEL_10;
  }
LABEL_133:
  sub_B18290((__int64)&v247, "Starting analysis in loop", 0x19u);
LABEL_10:
  v30 = **(_QWORD **)(a1 + 32);
  if ( v30 && *(_QWORD *)(v30 + 72) )
  {
    sub_B18290((__int64)&v247, ", in function F[", 0x10u);
    v31 = (__int8 *)sub_BD5D20(*(_QWORD *)(**(_QWORD **)(a1 + 32) + 72LL));
    sub_B18290((__int64)&v247, v31, v32);
    sub_B18290((__int64)&v247, "]", 1u);
  }
  sub_B18290((__int64)&v247, "...", 3u);
  sub_1049740(a7, (__int64)&v247);
  v37 = v254;
  v247 = (const char *)&unk_49D9D40;
  v38 = &v254[10 * v255];
  if ( v254 != v38 )
  {
    do
    {
      v38 -= 10;
      v39 = v38[4];
      if ( (unsigned __int64 *)v39 != v38 + 6 )
        j_j___libc_free_0(v39);
      if ( (unsigned __int64 *)*v38 != v38 + 2 )
        j_j___libc_free_0(*v38);
    }
    while ( v37 != v38 );
    v38 = v254;
  }
  if ( v38 != (unsigned __int64 *)v256 )
    _libc_free((unsigned __int64)v38);
LABEL_22:
  v40 = (const char **)a1;
  sub_D4BD20(&v231, a1, v33, v34, v35, v36);
  if ( !v231 )
    goto LABEL_31;
  sub_D4BD20(&v238, a1, v41, i, v43, v44);
  v45 = sub_B10CD0((__int64)&v238);
  v41 = *(unsigned __int8 *)(v45 - 16);
  if ( (v41 & 2) != 0 )
  {
    if ( *(_DWORD *)(v45 - 24) != 2 )
    {
LABEL_25:
      v46 = 0;
      goto LABEL_26;
    }
    v105 = *(_QWORD *)(v45 - 32);
  }
  else
  {
    i = (*(_WORD *)(v45 - 16) >> 6) & 0xF;
    if ( ((*(_WORD *)(v45 - 16) >> 6) & 0xF) != 2 )
      goto LABEL_25;
    v41 = 8LL * (((unsigned __int8)v41 >> 2) & 0xF);
    v105 = v45 - 16 - v41;
  }
  if ( !*(_QWORD *)(v105 + 8) )
    goto LABEL_25;
  sub_D4BD20(&v247, a1, v41, i, v43, v44);
  for ( i = sub_B10CD0((__int64)&v247); ; i = v110 )
  {
    v106 = *(_BYTE *)(i - 16);
    if ( (v106 & 2) != 0 )
    {
      v107 = *(_QWORD *)(i - 32);
      if ( *(_DWORD *)(i - 24) != 2 )
        goto LABEL_118;
    }
    else
    {
      v41 = i - 16;
      v43 = (*(_WORD *)(i - 16) >> 6) & 0xF;
      if ( ((*(_WORD *)(i - 16) >> 6) & 0xF) != 2 )
        goto LABEL_127;
      v107 = v41 - 8LL * ((v106 >> 2) & 0xF);
    }
    v110 = *(_QWORD *)(v107 + 8);
    if ( !v110 )
      break;
  }
  v41 = i - 16;
  if ( (*(_BYTE *)(i - 16) & 2) != 0 )
  {
LABEL_118:
    v108 = *(__int64 **)(i - 32);
    goto LABEL_119;
  }
LABEL_127:
  i = 8LL * ((v106 >> 2) & 0xF);
  v108 = (__int64 *)(v41 - i);
LABEL_119:
  v109 = *v108;
  if ( v247 )
  {
    v198 = v109;
    sub_B91220((__int64)&v247, (__int64)v247);
    v109 = v198;
  }
  v46 = v109 != 0;
LABEL_26:
  if ( v238.m128i_i64[0] )
    sub_B91220((__int64)&v238, v238.m128i_i64[0]);
  v40 = (const char **)v231;
  if ( v231 )
    sub_B91220((__int64)&v231, (__int64)v231);
  if ( v46 )
  {
    v82 = *a7;
    v83 = sub_B2BE50(*a7);
    if ( sub_B6EA50(v83)
      || (v135 = sub_B2BE50(v82),
          v136 = sub_B6F970(v135),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v136 + 48LL))(v136)) )
    {
      sub_B157E0((__int64)&v238, &v202);
      sub_B17430((__int64)&v247, (__int64)"loop-unroll", (__int64)"tryToUnrollLoop", 15, &v238, v200);
      sub_B18290((__int64)&v247, "  Loop is from an inlined function: inlined into \"", 0x32u);
      sub_D4BD20(&v225, a1, v84, v85, v86, v87);
      for ( j = sub_B10CD0((__int64)&v225); ; j = v98 )
      {
        v89 = *(_BYTE *)(j - 16);
        if ( (v89 & 2) != 0 )
        {
          v90 = *(_QWORD *)(j - 32);
          if ( *(_DWORD *)(j - 24) != 2 )
            goto LABEL_94;
        }
        else
        {
          v97 = j - 16;
          if ( ((*(_WORD *)(j - 16) >> 6) & 0xF) != 2 )
            goto LABEL_175;
          v90 = v97 - 8LL * ((v89 >> 2) & 0xF);
        }
        v98 = *(_QWORD *)(v90 + 8);
        if ( !v98 )
          break;
      }
      v97 = j - 16;
      if ( (*(_BYTE *)(j - 16) & 2) != 0 )
      {
LABEL_94:
        v91 = *(_BYTE ***)(j - 32);
        goto LABEL_95;
      }
LABEL_175:
      v91 = (_BYTE **)(v97 - 8LL * ((v89 >> 2) & 0xF));
LABEL_95:
      v92 = *v91;
      if ( *v92 == 16
        || ((v93 = *(v92 - 16), (v93 & 2) != 0)
          ? (v94 = (_BYTE **)*((_QWORD *)v92 - 4))
          : (v94 = (_BYTE **)&v92[-8 * ((v93 >> 2) & 0xF) - 16]),
            (v92 = *v94) != 0) )
      {
        v114 = *(v92 - 16);
        if ( (v114 & 2) != 0 )
          v115 = (__int64 *)*((_QWORD *)v92 - 4);
        else
          v115 = (__int64 *)&v92[-8 * ((v114 >> 2) & 0xF) - 16];
        v96 = (__int8 *)*v115;
        if ( *v115 )
          v96 = (__int8 *)sub_B91420(*v115);
        else
          v95 = 0;
      }
      else
      {
        v95 = 0;
        v96 = (__int8 *)byte_3F871B3;
      }
      sub_B18290((__int64)&v247, v96, v95);
      sub_B18290((__int64)&v247, ":", 1u);
      sub_D4BD20(&v222, a1, v116, v117, v118, v119);
      v120 = sub_B10CD0((__int64)&v222);
      v121 = *(_BYTE *)(v120 - 16);
      if ( (v121 & 2) != 0 )
      {
        if ( *(_DWORD *)(v120 - 24) == 2 )
        {
          v122 = *(_QWORD *)(v120 - 32);
          goto LABEL_144;
        }
      }
      else if ( ((*(_WORD *)(v120 - 16) >> 6) & 0xF) == 2 )
      {
        v122 = v120 - 16 - 8LL * ((v121 >> 2) & 0xF);
LABEL_144:
        sub_B169E0(v238.m128i_i64, "LineNumber", 10, *(_DWORD *)(*(_QWORD *)(v122 + 8) + 4LL));
        v123 = sub_23FD640((__int64)&v247, (__int64)&v238);
        sub_B18290(v123, ":", 1u);
        sub_D4BD20(&v209, a1, v124, v125, v126, v127);
        v128 = sub_B10CD0((__int64)&v209);
        v129 = *(_BYTE *)(v128 - 16);
        if ( (v129 & 2) != 0 )
        {
          if ( *(_DWORD *)(v128 - 24) == 2 )
          {
            v130 = *(_QWORD *)(v128 - 32);
LABEL_147:
            sub_B169E0((__int64 *)&v231, "ColumnNumber", 12, *(unsigned __int16 *)(*(_QWORD *)(v130 + 8) + 2LL));
            v131 = sub_23FD640(v123, (__int64)&v231);
            sub_B18290(v131, "\"", 1u);
            if ( v236 != &v237 )
              j_j___libc_free_0((unsigned __int64)v236);
            if ( v231 != &v233 )
              j_j___libc_free_0((unsigned __int64)v231);
            if ( v209 )
              sub_B91220((__int64)&v209, v209);
            if ( (_OWORD *)v240.m128i_i64[1] != v242 )
              j_j___libc_free_0(v240.m128i_u64[1]);
            if ( (__int64 *)v238.m128i_i64[0] != &v239 )
              j_j___libc_free_0(v238.m128i_u64[0]);
            if ( v222 )
              sub_B91220((__int64)&v222, (__int64)v222);
            if ( v225 )
              sub_B91220((__int64)&v225, (__int64)v225);
            v40 = &v247;
            sub_1049740(a7, (__int64)&v247);
            v132 = v254;
            v247 = (const char *)&unk_49D9D40;
            v133 = &v254[10 * v255];
            if ( v254 != v133 )
            {
              do
              {
                v133 -= 10;
                v134 = v133[4];
                if ( (unsigned __int64 *)v134 != v133 + 6 )
                {
                  v40 = (const char **)(v133[6] + 1);
                  j_j___libc_free_0(v134);
                }
                if ( (unsigned __int64 *)*v133 != v133 + 2 )
                {
                  v40 = (const char **)(v133[2] + 1);
                  j_j___libc_free_0(*v133);
                }
              }
              while ( v132 != v133 );
              v133 = v254;
            }
            if ( v133 != (unsigned __int64 *)v256 )
              _libc_free((unsigned __int64)v133);
            goto LABEL_31;
          }
        }
        else if ( ((*(_WORD *)(v128 - 16) >> 6) & 0xF) == 2 )
        {
          v130 = v128 - 16 - 8LL * ((v129 >> 2) & 0xF);
          goto LABEL_147;
        }
        BUG();
      }
      BUG();
    }
  }
LABEL_31:
  v51 = sub_F6E5D0(a1, (__int64)v40, v41, i, v43, v44);
  v197 = v51 & 2;
  if ( (v51 & 2) != 0 )
  {
    v103 = *a7;
    v104 = sub_B2BE50(*a7);
    if ( sub_B6EA50(v104)
      || (v148 = sub_B2BE50(v103),
          v149 = sub_B6F970(v148),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v149 + 48LL))(v149)) )
    {
      sub_B157E0((__int64)&v238, &v202);
      sub_B17430((__int64)&v247, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v238, v200);
      sub_B18290(
        (__int64)&v247,
        "  Unrolling is disabled by source code \"#pragma unroll 1\" or previous unrolling",
        0x4Fu);
      sub_1049740(a7, (__int64)&v247);
      v247 = (const char *)&unk_49D9D40;
      sub_23FD590((__int64)&v254);
    }
    v197 = 0;
    goto LABEL_87;
  }
  if ( *(_QWORD *)a1
    && (unsigned int)sub_F6E690(*(_QWORD *)a1, (__int64)v40, v47, v48, v49, v50) == 5
    && (unsigned int)sub_F6E5D0(a1, (__int64)v40, v47, v48, v49, v50) != 5 )
  {
    if ( !(unsigned __int8)sub_287ED80(*a7) )
      goto LABEL_87;
    sub_B157E0((__int64)&v238, &v202);
    sub_B17430((__int64)&v247, (__int64)"loop-unroll", (__int64)"tryToUnrollLoop", 15, &v238, v200);
    v101 = "  Not unrolling loop since parent loop has llvm.loop.unroll_and_jam.";
    v102 = 68;
    goto LABEL_108;
  }
  if ( (unsigned int)sub_F6E690(a1, (__int64)v40, v47, v48, v49, v50) == 5 )
  {
    if ( (unsigned int)sub_F6E5D0(a1, (__int64)v40, v52, v53, v54, v55) != 5 )
      goto LABEL_87;
    if ( (unsigned __int8)sub_D4B3D0(a1) )
      goto LABEL_36;
LABEL_106:
    v99 = *a7;
    v100 = sub_B2BE50(*a7);
    if ( !sub_B6EA50(v100) )
    {
      v160 = sub_B2BE50(v99);
      v161 = sub_B6F970(v160);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v161 + 48LL))(v161) )
        goto LABEL_87;
    }
    sub_B157E0((__int64)&v238, &v202);
    sub_B17430((__int64)&v247, (__int64)"loop-unroll", (__int64)"tryToUnrollLoop", 15, &v238, v200);
    v101 = "  Not unrolling : loop not in normal form";
    v102 = 41;
LABEL_108:
    sub_B18290((__int64)&v247, v101, v102);
    sub_1049740(a7, (__int64)&v247);
    v247 = (const char *)&unk_49D9D40;
    sub_23FD590((__int64)&v254);
    goto LABEL_87;
  }
  if ( !(unsigned __int8)sub_D4B3D0(a1) )
    goto LABEL_106;
LABEL_36:
  if ( a14 && (v51 & 1) == 0 )
    goto LABEL_87;
  v56 = *(_QWORD *)(**(_QWORD **)(a1 + 32) + 72LL);
  v57 = sub_B2D610(v56, 47);
  if ( !v57 )
    v57 = sub_B2D610(v56, 18);
  sub_2880090((__int64)&v209, a1, (__int64)a4, (__int64)a5, a8, a9, (__int64)a7, a12, a17, a16, a18, a19, a20, a23);
  v204 = sub_2A04E50(a1, a4, a5, a21, a22, 1);
  if ( !(_DWORD)v209 && (!v213 || !v210) && !v57 )
    goto LABEL_87;
  v232 = &v236;
  v231 = 0;
  v233 = 32;
  v234 = 0;
  v235 = 1;
  sub_30AB790(a1, a6, &v231);
  sub_2880BC0((__int64)&v206, a1, (__int64)a5, (__int64)&v231, v212);
  if ( !(unsigned __int8)sub_2880E50((__int64)&v206) )
    goto LABEL_85;
  if ( !v206.m128i_i32[2] )
    v188 = v206.m128i_i32[0];
  if ( v57 )
  {
    v58 = v188 + 1;
    if ( v188 + 1 < (unsigned int)v209 )
      v58 = v209;
    LODWORD(v209) = v58;
  }
  v189 = v207;
  if ( v207 )
  {
    if ( (unsigned __int8)sub_287ED80(*a7) )
    {
      sub_B157E0((__int64)&v238, &v202);
      sub_B17430((__int64)&v247, (__int64)"loop-unroll", (__int64)"tryToUnrollLoop", 15, &v238, v200);
      sub_B18290((__int64)&v247, "  Not unrolling : loop contains function calls that may be inlined later", 0x48u);
      sub_1049740(a7, (__int64)&v247);
      v247 = (const char *)&unk_49D9D40;
      sub_23FD590((__int64)&v254);
    }
    goto LABEL_85;
  }
  v222 = (__int64 *)v224;
  v223 = 0x800000000LL;
  sub_D46D90(a1, (__int64)&v222);
  if ( &v222[(unsigned int)v223] == v222 )
  {
    v193 = 1;
  }
  else
  {
    v187 = v26;
    v59 = 0;
    v60 = &v222[(unsigned int)v223];
    v193 = 1;
    v61 = v222;
    do
    {
      v62 = sub_DBA790((__int64)a4, a1, *v61);
      if ( v62 && (!v59 || v62 < v59) )
      {
        v193 = v62;
        v59 = v62;
      }
      ++v61;
    }
    while ( v60 != v61 );
    v26 = v187;
    if ( v59 )
    {
      v63 = 0;
      if ( !LOBYTE(qword_500A488[8]) )
        v215 &= v208;
      goto LABEL_62;
    }
  }
  v138 = sub_D47930(a1);
  v186 = v138;
  if ( !v138 )
    goto LABEL_196;
  v139 = v138 + 48;
  v140 = *(_QWORD *)(v138 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v140 == v139 )
    goto LABEL_196;
  if ( !v140 )
    BUG();
  v141 = v140 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v140 - 24) - 30 > 0xA || (v142 = sub_B46E30(v141)) == 0 )
  {
LABEL_196:
    v186 = sub_D46F00(a1);
    if ( v186 )
      goto LABEL_202;
    goto LABEL_197;
  }
  v190 = v26;
  v143 = 0;
  v144 = v142;
  while ( 1 )
  {
    v145 = sub_B46EC0(v141, v143);
    if ( !*(_BYTE *)(a1 + 84) )
      break;
    v146 = *(_QWORD **)(a1 + 64);
    v147 = &v146[*(unsigned int *)(a1 + 76)];
    if ( v146 == v147 )
      goto LABEL_201;
    while ( v145 != *v146 )
    {
      if ( v147 == ++v146 )
        goto LABEL_201;
    }
LABEL_194:
    if ( v144 == ++v143 )
    {
      v26 = v190;
      goto LABEL_196;
    }
  }
  if ( sub_C8CA60(a1 + 56, v145) )
    goto LABEL_194;
LABEL_201:
  v26 = v190;
LABEL_202:
  v193 = sub_DE5E70(a4, a1, v186);
LABEL_197:
  if ( !LOBYTE(qword_500A488[8]) )
    v215 &= v208;
  v59 = 0;
  v189 = sub_DBB070((__int64)a4, a1, 0);
  v63 = sub_DBA820((__int64)a4, a1);
LABEL_62:
  v201 = 0;
  v66 = sub_28873F0(
          a1,
          a5,
          a2,
          a3,
          a6,
          a4,
          (__int64)&v231,
          a7,
          v59,
          v189,
          v63,
          v193,
          &v206,
          (unsigned int *)&v209,
          (unsigned int *)&v204,
          &v201,
          a10);
  if ( v211 )
  {
    if ( !LOBYTE(qword_500A488[8]) )
      v214 &= v208;
    if ( (_DWORD)v204 )
    {
      if ( (unsigned __int8)sub_287ED80(*a7) )
      {
        v162 = **(_QWORD **)(a1 + 32);
        sub_D4BD20(&v203, a1, v150, v151, v152, v153);
        sub_B157E0((__int64)&v205, &v203);
        sub_B17430((__int64)&v247, (__int64)"loop-unroll", (__int64)"Peeled", 6, &v205, v162);
        sub_B18290((__int64)&v247, " peeled loop by ", 0x10u);
        sub_B169E0((__int64 *)&v225, "PeelCount", 9, v204);
        v163 = sub_23FD640((__int64)&v247, (__int64)&v225);
        sub_B18290(v163, " iterations", 0xBu);
        v238.m128i_i32[2] = *(_DWORD *)(v163 + 8);
        v238.m128i_i8[12] = *(_BYTE *)(v163 + 12);
        v239 = *(_QWORD *)(v163 + 16);
        v166 = _mm_loadu_si128((const __m128i *)(v163 + 24));
        v238.m128i_i64[0] = (__int64)&unk_49D9D40;
        v240 = v166;
        v241 = *(_QWORD *)(v163 + 40);
        v242[0] = _mm_loadu_si128((const __m128i *)(v163 + 48));
        v167 = _mm_loadu_si128((const __m128i *)(v163 + 64));
        v243[0] = v244;
        v243[1] = 0x400000000LL;
        v242[1] = v167;
        v168 = *(unsigned int *)(v163 + 88);
        if ( (_DWORD)v168 )
          sub_28821D0((__int64)v243, v163 + 80, (__int64)v244, v164, v168, v165);
        v244[320] = *(_BYTE *)(v163 + 416);
        v245 = *(_DWORD *)(v163 + 420);
        v246 = *(_QWORD *)(v163 + 424);
        v238.m128i_i64[0] = (__int64)&unk_49D9D78;
        if ( v229 != &v230 )
          j_j___libc_free_0((unsigned __int64)v229);
        if ( v225 != &v227 )
          j_j___libc_free_0((unsigned __int64)v225);
        v247 = (const char *)&unk_49D9D40;
        sub_23FD590((__int64)&v254);
        if ( v203 )
          sub_B91220((__int64)&v203, v203);
        sub_1049740(a7, (__int64)&v238);
        v238.m128i_i64[0] = (__int64)&unk_49D9D40;
        sub_23FD590((__int64)v243);
      }
      v247 = 0;
      LODWORD(v250) = 128;
      v248 = sub_C7D670(0x2000, 8);
      sub_23FE7B0((__int64)&v247);
      v253 = 0;
      if ( (unsigned __int8)sub_2A07DE0(a1, v204, a3, (_DWORD)a4, a2, a6, a11, (__int64)&v247) )
      {
        sub_2A13F00(a1, 1, a3, (_DWORD)a4, a2, a6, (__int64)a5, 0);
        v197 = 1;
        if ( BYTE6(v204) )
          sub_D4A9E0(a1);
      }
      if ( v253 )
      {
        v169 = v252;
        v253 = 0;
        if ( v252 )
        {
          v170 = v251;
          v171 = &v251[2 * v252];
          do
          {
            if ( *v170 != -8192 && *v170 != -4096 )
            {
              v172 = v170[1];
              if ( v172 )
                sub_B91220((__int64)(v170 + 1), v172);
            }
            v170 += 2;
          }
          while ( v171 != v170 );
          v169 = v252;
        }
        sub_C7D6A0((__int64)v251, 16 * v169, 8);
      }
      v154 = v250;
      if ( (_DWORD)v250 )
      {
        v155 = (_QWORD *)v248;
        v226 = 2;
        v227 = 0;
        v156 = -4096;
        v157 = (_QWORD *)(v248 + ((unsigned __int64)(unsigned int)v250 << 6));
        v228 = -4096;
        v225 = (__int64 *)&unk_49DD7B0;
        v229 = 0;
        v238.m128i_i64[1] = 2;
        v239 = 0;
        v240 = (__m128i)0xFFFFFFFFFFFFE000LL;
        v238.m128i_i64[0] = (__int64)&unk_49DD7B0;
        while ( 1 )
        {
          v158 = v155[3];
          if ( v156 != v158 )
          {
            v156 = v240.m128i_i64[0];
            if ( v158 != v240.m128i_i64[0] )
            {
              v159 = v155[7];
              if ( v159 != 0 && v159 != -4096 && v159 != -8192 )
              {
                sub_BD60C0(v155 + 5);
                v158 = v155[3];
              }
              v156 = v158;
            }
          }
          *v155 = &unk_49DB368;
          if ( v156 != -4096 && v156 != 0 && v156 != -8192 )
            sub_BD60C0(v155 + 1);
          v155 += 8;
          if ( v157 == v155 )
            break;
          v156 = v228;
        }
        v238.m128i_i64[0] = (__int64)&unk_49DB368;
        if ( v240.m128i_i64[0] != 0 && v240.m128i_i64[0] != -4096 && v240.m128i_i64[0] != -8192 )
          sub_BD60C0(&v238.m128i_i64[1]);
        v225 = (__int64 *)&unk_49DB368;
        if ( v228 != -4096 && v228 != 0 && v228 != -8192 )
          sub_BD60C0(&v226);
        v154 = v250;
      }
      sub_C7D6A0(v248, (unsigned __int64)v154 << 6, 8);
    }
    else
    {
      if ( !a13 )
        goto LABEL_67;
      v137 = v189;
      if ( v189 < v59 )
        v137 = v59;
      if ( v211 >= v137 )
      {
LABEL_67:
        v67 = v214;
        if ( v59 )
        {
          v67 = 0;
        }
        else
        {
          v64 = v193 % v211;
          if ( !(v193 % v211) )
            v67 = 0;
        }
        v214 = v67;
        v194 = sub_D49300(a1, v67, v64, v211, v65, (unsigned int)v204);
        if ( (_BYTE)qword_50028A8 && v26 > 1 )
        {
          v173 = sub_2880E70(v206.m128i_i64, (__int64)&v209, 0, v68, v69);
          v174 = 3;
          if ( v26 >= 3 )
            v174 = v26;
          if ( v173 < (unsigned int)(v210 * v174) )
            v218 = 1;
        }
        if ( (unsigned __int8)sub_287ED80(*a7) )
        {
          sub_B157E0((__int64)&v238, &v202);
          sub_B17430((__int64)&v247, (__int64)"loop-unroll", (__int64)"tryToUnrollLoop", 15, &v238, v200);
          sub_B18290((__int64)&v247, "    Success! Unrolling strategy :", 0x21u);
          sub_1049740(a7, (__int64)&v247);
          v247 = (const char *)&unk_49D9D40;
          sub_23FD590((__int64)&v254);
        }
        if ( !(unsigned __int8)sub_287ED80(*a7) )
        {
LABEL_75:
          v205.m128i_i64[0] = 0;
          v238.m128i_i8[10] = 0;
          v238.m128i_i32[0] = v211;
          v238.m128i_i8[4] = v217;
          v238.m128i_i8[6] = v216;
          v238.m128i_i8[7] = v218;
          v238.m128i_i8[5] = v214;
          v238.m128i_i8[8] = a15;
          v239 = sub_D4A330(a1);
          v238.m128i_i8[9] = v219;
          v240.m128i_i32[0] = v220;
          v240.m128i_i8[4] = v221;
          v70 = sub_2A15A20(
                  a1,
                  a3,
                  a4,
                  a2,
                  a6,
                  a5,
                  v238.m128i_i64[0],
                  v238.m128i_i64[1],
                  v239,
                  v240.m128i_i64[0],
                  a7,
                  a11,
                  &v205,
                  a24);
          if ( v70 )
          {
            if ( v205.m128i_i64[0] )
            {
              v247 = "llvm.loop.unroll.followup_all";
              v248 = 29;
              v249 = "llvm.loop.unroll.followup_remainder";
              v250 = 35;
              v71 = (__int64 *)sub_F6E0D0(v194, (__int64)&v247, 2, byte_3F871B3, 0);
              v226 = v72;
              v225 = v71;
              if ( (_BYTE)v72 )
                sub_D49440(v205.m128i_i64[0], (__int64)v225, v72, v73, v74, v75);
            }
            v197 = 2;
            if ( v70 != 2 )
            {
              v247 = "llvm.loop.unroll.followup_all";
              v248 = 29;
              v249 = "llvm.loop.unroll.followup_unrolled";
              v250 = 34;
              v76 = (__int64 *)sub_F6E0D0(v194, (__int64)&v247, 2, byte_3F871B3, 0);
              v226 = v77;
              v225 = v76;
              if ( (_BYTE)v77 )
              {
                sub_D49440(a1, (__int64)v225, v77, v78, v79, v80);
                v197 = v70;
              }
              else
              {
                v197 = v70;
                if ( v66 )
                  sub_D4A9E0(a1);
              }
            }
          }
          goto LABEL_83;
        }
        sub_B157E0((__int64)&v238, &v202);
        sub_B17430((__int64)&v247, (__int64)"loop-unroll", (__int64)"tryToUnrollLoop", 15, &v238, v200);
        if ( (_DWORD)v204 )
        {
          sub_B18290((__int64)&v247, "      loop peeling by ", 0x16u);
          sub_B169E0(v238.m128i_i64, "PeelCount", 9, v204);
          v183 = sub_23FD640((__int64)&v247, (__int64)&v238);
          v180 = " iterations";
          v182 = 11;
          v181 = v183;
          goto LABEL_274;
        }
        if ( v59 )
        {
          if ( v201 )
          {
LABEL_239:
            sub_B18290((__int64)&v247, "      fully unroll by known upper bound", 0x27u);
          }
          else
          {
            if ( v211 < v59 )
            {
              sub_B18290((__int64)&v247, "      partially unroll by factor of ", 0x24u);
              sub_B169E0(v238.m128i_i64, "UP.Count", 8, v211);
              v175 = " with remainder loop";
              v176 = sub_23FD640((__int64)&v247, (__int64)&v238);
              if ( !(v59 % v211) )
                v175 = (char *)byte_3F871B3;
              v177 = strlen(v175);
              sub_B18290(v176, v175, v177);
              v178 = v59 % v211;
              v179 = byte_3F871B3;
              if ( v178 && v218 )
                v179 = " and remainder loop will be fully unrolled";
              v180 = (__int8 *)v179;
              v181 = v176;
              v182 = strlen(v179);
              goto LABEL_274;
            }
            sub_B18290((__int64)&v247, "      fully unroll to straight-line code", 0x28u);
          }
        }
        else
        {
          if ( v201 )
            goto LABEL_239;
          sub_B18290((__int64)&v247, "      runtime unroll by factor of ", 0x22u);
          sub_B169E0(v238.m128i_i64, "UP.Count", 8, v211);
          v184 = " and remainder loop will be fully unrolled";
          v185 = sub_23FD640((__int64)&v247, (__int64)&v238);
          sub_B18290(v185, " with remainder loop", 0x14u);
          if ( !v218 )
            v184 = byte_3F871B3;
          v180 = (__int8 *)v184;
          v181 = v185;
          v182 = strlen(v184);
LABEL_274:
          sub_B18290(v181, v180, v182);
          if ( (_OWORD *)v240.m128i_i64[1] != v242 )
            j_j___libc_free_0(v240.m128i_u64[1]);
          if ( (__int64 *)v238.m128i_i64[0] != &v239 )
            j_j___libc_free_0(v238.m128i_u64[0]);
        }
        sub_1049740(a7, (__int64)&v247);
        v247 = (const char *)&unk_49D9D40;
        sub_23FD590((__int64)&v254);
        goto LABEL_75;
      }
    }
  }
LABEL_83:
  if ( v222 != (__int64 *)v224 )
    _libc_free((unsigned __int64)v222);
LABEL_85:
  if ( !v235 )
    _libc_free((unsigned __int64)v232);
LABEL_87:
  if ( v202 )
    sub_B91220((__int64)&v202, v202);
  return v197;
}
