// Function: sub_1F8E5A0
// Address: 0x1f8e5a0
//
__int64 *__fastcall sub_1F8E5A0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        __int8 a5,
        double a6,
        __m128i a7,
        __m128i a8)
{
  bool v8; // cc
  unsigned __int8 v12; // r14
  __int64 v13; // rdx
  char v14; // al
  __int64 v15; // rdx
  __int64 v16; // r13
  unsigned int v17; // r12d
  unsigned int v18; // eax
  __int64 v19; // rdi
  __int64 (*v20)(); // rax
  __m128i v21; // rax
  __int64 *v22; // r12
  char *v23; // rax
  unsigned __int8 v24; // dl
  __int64 v25; // rsi
  __int64 v26; // rdi
  double v27; // xmm0_8
  __int64 *v28; // rax
  __int64 *v29; // rdi
  unsigned __int64 v30; // rdx
  __m128 v31; // rax
  __int64 v32; // rdx
  int v33; // r10d
  __int64 *v34; // rdi
  __int64 *v35; // rax
  __int64 v36; // rdx
  __int64 v37; // r13
  __int64 *v38; // r12
  __int64 *v39; // rdi
  __int64 *v40; // r12
  __int64 v41; // rdx
  unsigned __int64 v42; // r13
  __int64 *v43; // r12
  __int64 v44; // rdx
  unsigned __int64 v45; // r13
  __int64 v46; // rdx
  unsigned __int64 v47; // rax
  __int64 *v48; // rdi
  __int64 v49; // r13
  __int128 v50; // rax
  __int64 *v51; // rdi
  __int128 v52; // rax
  const void **v53; // r13
  __int64 v54; // r14
  __int64 v55; // rax
  unsigned int v56; // r12d
  __int64 *v57; // rax
  __int64 v58; // rdx
  __int64 *v59; // rdi
  __int64 *v60; // rax
  __int64 v61; // rdx
  __int64 *v62; // rdi
  unsigned __int64 v63; // rdx
  unsigned __int64 v64; // rdx
  __int64 v65; // rdx
  char *v66; // rax
  __int64 v67; // rsi
  char v68; // bl
  const void **v69; // rax
  __int64 v70; // r14
  __int64 v71; // r13
  int v72; // eax
  __int64 v73; // rdi
  __int64 (__fastcall *v74)(__int64, __int64, unsigned __int64, __int64, __int64); // r12
  __int64 v75; // rax
  const void **v76; // rdx
  const void **v77; // r14
  _DWORD *v78; // rax
  __int64 v79; // rdx
  _QWORD *v80; // rax
  __int64 *v81; // rbx
  _QWORD *v82; // r12
  __int64 v83; // rdx
  __int64 v84; // r13
  __int64 *v85; // rdi
  __int64 v86; // r8
  __int64 v87; // r9
  __int64 v88; // rax
  __int64 v89; // rdx
  __int64 *v90; // r14
  __int16 *v91; // rdx
  __int64 v92; // rdx
  __int64 v93; // rdx
  _QWORD *v94; // r12
  _QWORD *v95; // rax
  __int64 v96; // rdx
  __int64 v97; // r13
  _QWORD *v98; // r12
  __int128 v99; // rax
  __int64 *v100; // rdi
  __int64 v101; // rax
  __int64 *v102; // rdi
  __int64 v103; // rcx
  __int16 *v104; // rdx
  __int64 v105; // r8
  __int64 v106; // r9
  __int64 v107; // rax
  const void **v108; // r8
  __int64 v109; // r14
  __int64 v110; // rdx
  __int64 *v111; // r13
  __int16 *v112; // rdx
  __int64 v113; // rdx
  __int64 v114; // rbx
  __int64 v115; // r13
  __int64 v116; // r14
  __int64 v117; // rax
  __int64 v118; // r15
  unsigned __int64 v119; // rdx
  unsigned __int64 v120; // rsi
  __int128 v121; // [rsp-30h] [rbp-240h]
  __int128 v122; // [rsp-20h] [rbp-230h]
  __int128 v123; // [rsp-20h] [rbp-230h]
  __int128 v124; // [rsp-10h] [rbp-220h]
  __int128 v125; // [rsp-10h] [rbp-220h]
  __int128 v126; // [rsp-10h] [rbp-220h]
  __int128 v127; // [rsp+0h] [rbp-210h]
  const void **v128; // [rsp+10h] [rbp-200h]
  unsigned __int8 v129; // [rsp+10h] [rbp-200h]
  __int128 v130; // [rsp+20h] [rbp-1F0h]
  __int64 *v131; // [rsp+30h] [rbp-1E0h]
  __int64 *v132; // [rsp+30h] [rbp-1E0h]
  __int128 v133; // [rsp+30h] [rbp-1E0h]
  unsigned __int64 v134; // [rsp+38h] [rbp-1D8h]
  __m128i v135; // [rsp+40h] [rbp-1D0h] BYREF
  __int64 *v136; // [rsp+50h] [rbp-1C0h]
  unsigned __int64 v137; // [rsp+58h] [rbp-1B8h]
  unsigned int v138; // [rsp+64h] [rbp-1ACh]
  unsigned __int64 v139; // [rsp+68h] [rbp-1A8h]
  __int64 v140; // [rsp+70h] [rbp-1A0h]
  _QWORD *v141; // [rsp+78h] [rbp-198h]
  __int128 v142; // [rsp+80h] [rbp-190h]
  __m128 v143; // [rsp+90h] [rbp-180h]
  __int64 v144; // [rsp+A0h] [rbp-170h]
  unsigned __int64 v145; // [rsp+A8h] [rbp-168h]
  __m128 v146; // [rsp+B0h] [rbp-160h]
  __int128 v147; // [rsp+C0h] [rbp-150h]
  __int64 *v148; // [rsp+D0h] [rbp-140h]
  __int64 v149; // [rsp+D8h] [rbp-138h]
  __int64 *v150; // [rsp+E0h] [rbp-130h]
  __int64 v151; // [rsp+E8h] [rbp-128h]
  __int64 *v152; // [rsp+F0h] [rbp-120h]
  unsigned __int64 v153; // [rsp+F8h] [rbp-118h]
  __int64 *v154; // [rsp+100h] [rbp-110h]
  __int64 v155; // [rsp+108h] [rbp-108h]
  __int64 *v156; // [rsp+110h] [rbp-100h]
  __int64 v157; // [rsp+118h] [rbp-F8h]
  __int64 *v158; // [rsp+120h] [rbp-F0h]
  __int64 v159; // [rsp+128h] [rbp-E8h]
  __int64 *v160; // [rsp+130h] [rbp-E0h]
  __int64 v161; // [rsp+138h] [rbp-D8h]
  __int64 *v162; // [rsp+140h] [rbp-D0h]
  __int64 v163; // [rsp+148h] [rbp-C8h]
  __int64 *v164; // [rsp+150h] [rbp-C0h]
  __int64 v165; // [rsp+158h] [rbp-B8h]
  __int64 *v166; // [rsp+160h] [rbp-B0h]
  __int64 v167; // [rsp+168h] [rbp-A8h]
  __int64 *v168; // [rsp+170h] [rbp-A0h]
  __int64 v169; // [rsp+178h] [rbp-98h]
  char v170; // [rsp+183h] [rbp-8Dh] BYREF
  unsigned int v171; // [rsp+184h] [rbp-8Ch] BYREF
  __int64 v172; // [rsp+188h] [rbp-88h] BYREF
  unsigned int v173; // [rsp+190h] [rbp-80h] BYREF
  __int64 v174; // [rsp+198h] [rbp-78h]
  __int64 v175; // [rsp+1A0h] [rbp-70h] BYREF
  const void **v176; // [rsp+1A8h] [rbp-68h]
  __int64 v177; // [rsp+1B0h] [rbp-60h] BYREF
  int v178; // [rsp+1B8h] [rbp-58h]
  __int64 v179; // [rsp+1C0h] [rbp-50h] BYREF
  _QWORD *v180; // [rsp+1C8h] [rbp-48h] BYREF
  __int64 v181; // [rsp+1D0h] [rbp-40h]

  v8 = *((_DWORD *)a1 + 4) <= 2;
  *(_QWORD *)&v147 = a2;
  *((_QWORD *)&v147 + 1) = a3;
  v146.m128_i8[0] = a5;
  if ( !v8 )
    return 0;
  v12 = a5;
  v141 = (_QWORD *)v147;
  v139 = 16LL * DWORD2(v147);
  v13 = *(_QWORD *)(v147 + 40) + v139;
  v14 = *(_BYTE *)v13;
  v15 = *(_QWORD *)(v13 + 8);
  LOBYTE(v173) = v14;
  v174 = v15;
  if ( !v14 )
  {
    if ( sub_1F58D20((__int64)&v173) && sub_1F596B0((__int64)&v173) == 9 )
      goto LABEL_6;
    v14 = v173;
    if ( !(_BYTE)v173 )
    {
      if ( !sub_1F58D20((__int64)&v173) )
        return 0;
      v14 = sub_1F596B0((__int64)&v173);
LABEL_29:
      if ( v14 == 10 )
        goto LABEL_6;
      return 0;
    }
LABEL_28:
    if ( (unsigned __int8)(v14 - 14) <= 0x5Fu )
    {
      switch ( v14 )
      {
        case '^':
        case '_':
        case '`':
        case 'a':
        case 'j':
        case 'k':
        case 'l':
        case 'm':
          goto LABEL_6;
        default:
          return 0;
      }
    }
    goto LABEL_29;
  }
  if ( (unsigned __int8)(v14 - 14) <= 0x5Fu )
  {
    switch ( v14 )
    {
      case 'Y':
      case 'Z':
      case '[':
      case '\\':
      case ']':
      case 'e':
      case 'f':
      case 'g':
      case 'h':
      case 'i':
        goto LABEL_6;
      default:
        v14 = v173;
        goto LABEL_28;
    }
  }
  if ( v14 != 9 )
    goto LABEL_28;
LABEL_6:
  v16 = *(_QWORD *)(*a1 + 32LL);
  v17 = sub_1F44350(a1[1], v173, v174, v16);
  if ( !v17 )
    return 0;
  v18 = sub_1F44410(a1[1], v173, v174, v16);
  v19 = a1[1];
  v170 = 0;
  v171 = v18;
  v20 = *(__int64 (**)())(*(_QWORD *)v19 + 1440LL);
  if ( v20 == sub_1F6BBB0 )
    return 0;
  v21.m128i_i64[0] = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, _QWORD, unsigned int *, char *, _QWORD))v20)(
                       v19,
                       v147,
                       *((_QWORD *)&v147 + 1),
                       *a1,
                       v17,
                       &v171,
                       &v170,
                       v12);
  v22 = (__int64 *)v21.m128i_i64[0];
  v135 = v21;
  if ( !v21.m128i_i64[0] )
    return 0;
  sub_1F81BC0((__int64)a1, v21.m128i_i64[0]);
  v138 = v171;
  if ( v171 )
  {
    v23 = (char *)(v141[5] + v139);
    v24 = *v23;
    v25 = v141[9];
    v128 = (const void **)*((_QWORD *)v23 + 1);
    if ( v170 )
    {
      a7 = _mm_load_si128(&v135);
      v179 = v141[9];
      v144 = v24;
      *(_QWORD *)&v142 = &v179;
      v146 = (__m128)a7;
      if ( v25 )
        sub_1623A60((__int64)&v179, v25, 2);
      v26 = *a1;
      v27 = 1.5;
      LODWORD(v180) = *((_DWORD *)v141 + 16);
      v28 = sub_1D364E0(v26, v142, (unsigned int)v144, v128, 0, 1.5, *(double *)a7.m128i_i64, a8);
      v29 = (__int64 *)*a1;
      v137 = v30;
      v136 = v28;
      v31.m128_u64[0] = (unsigned __int64)sub_1D332F0(
                                            v29,
                                            78,
                                            v142,
                                            (unsigned int)v144,
                                            v128,
                                            a4,
                                            1.5,
                                            *(double *)a7.m128i_i64,
                                            a8,
                                            (__int64)v28,
                                            v30,
                                            v147);
      v143 = v31;
      sub_1F81BC0((__int64)a1, v31.m128_i64[0]);
      v131 = sub_1D332F0(
               (__int64 *)*a1,
               77,
               v142,
               (unsigned int)v144,
               v128,
               a4,
               1.5,
               *(double *)a7.m128i_i64,
               a8,
               v143.m128_i64[0],
               v143.m128_u64[1],
               v147);
      v168 = v131;
      v143.m128_u64[0] = (unsigned __int64)v131;
      v169 = v32;
      v143.m128_u64[1] = (unsigned int)v32 | v143.m128_u64[1] & 0xFFFFFFFF00000000LL;
      sub_1F81BC0((__int64)a1, (__int64)v131);
      v33 = 0;
      do
      {
        v34 = (__int64 *)*a1;
        v146.m128_u64[0] = (unsigned __int64)v22;
        LODWORD(v140) = v33;
        v35 = sub_1D332F0(
                v34,
                78,
                v142,
                (unsigned int)v144,
                v128,
                a4,
                1.5,
                *(double *)a7.m128i_i64,
                a8,
                (__int64)v22,
                v146.m128_u64[1],
                __PAIR128__(v146.m128_u64[1], (unsigned __int64)v22));
        v37 = v36;
        v38 = v35;
        sub_1F81BC0((__int64)a1, (__int64)v35);
        v39 = (__int64 *)*a1;
        *((_QWORD *)&v124 + 1) = v37;
        *(_QWORD *)&v124 = v38;
        v143.m128_u64[0] = (unsigned __int64)v131;
        v40 = sub_1D332F0(
                v39,
                78,
                v142,
                (unsigned int)v144,
                v128,
                a4,
                1.5,
                *(double *)a7.m128i_i64,
                a8,
                (__int64)v131,
                v143.m128_u64[1],
                v124);
        v166 = v40;
        v167 = v41;
        v42 = (unsigned int)v41 | v37 & 0xFFFFFFFF00000000LL;
        sub_1F81BC0((__int64)a1, (__int64)v40);
        *((_QWORD *)&v125 + 1) = v42;
        *(_QWORD *)&v125 = v40;
        v43 = sub_1D332F0(
                (__int64 *)*a1,
                77,
                v142,
                (unsigned int)v144,
                v128,
                a4,
                1.5,
                *(double *)a7.m128i_i64,
                a8,
                (__int64)v136,
                v137,
                v125);
        v164 = v43;
        v165 = v44;
        v45 = (unsigned int)v44 | v42 & 0xFFFFFFFF00000000LL;
        sub_1F81BC0((__int64)a1, (__int64)v43);
        *((_QWORD *)&v126 + 1) = v45;
        *(_QWORD *)&v126 = v43;
        v162 = sub_1D332F0(
                 (__int64 *)*a1,
                 78,
                 v142,
                 (unsigned int)v144,
                 v128,
                 a4,
                 1.5,
                 *(double *)a7.m128i_i64,
                 a8,
                 v146.m128_i64[0],
                 v146.m128_u64[1],
                 v126);
        v22 = v162;
        v146.m128_u64[0] = (unsigned __int64)v162;
        v163 = v46;
        v146.m128_u64[1] = (unsigned int)v46 | v146.m128_u64[1] & 0xFFFFFFFF00000000LL;
        sub_1F81BC0((__int64)a1, (__int64)v162);
        v33 = v140 + 1;
      }
      while ( v138 != (_DWORD)v140 + 1 );
      if ( !v12 )
      {
        v22 = sub_1D332F0(
                (__int64 *)*a1,
                78,
                v142,
                (unsigned int)v144,
                v128,
                a4,
                1.5,
                *(double *)a7.m128i_i64,
                a8,
                v146.m128_i64[0],
                v146.m128_u64[1],
                v147);
        v160 = v22;
        v146.m128_u64[0] = (unsigned __int64)v22;
        v161 = v93;
        v146.m128_u64[1] = (unsigned int)v93 | v146.m128_u64[1] & 0xFFFFFFFF00000000LL;
        sub_1F81BC0((__int64)a1, (__int64)v22);
      }
      if ( v179 )
        sub_161E7C0(v142, v179);
      v47 = v146.m128_u64[1];
    }
    else
    {
      a8 = _mm_load_si128(&v135);
      v179 = v141[9];
      v136 = (__int64 *)v24;
      *(_QWORD *)&v142 = &v179;
      v143 = (__m128)a8;
      if ( v25 )
        sub_1623A60((__int64)&v179, v25, 2);
      v48 = (__int64 *)*a1;
      v49 = v142;
      LODWORD(v180) = *((_DWORD *)v141 + 16);
      *(_QWORD *)&v50 = sub_1D364E0((__int64)v48, v142, (unsigned int)v136, v128, 0, -3.0, *(double *)a7.m128i_i64, a8);
      v51 = (__int64 *)*a1;
      v27 = -0.5;
      *(_QWORD *)&v142 = v49;
      v130 = v50;
      *(_QWORD *)&v52 = sub_1D364E0((__int64)v51, v49, (unsigned int)v136, v128, 0, -0.5, *(double *)a7.m128i_i64, a8);
      v53 = v128;
      v129 = v12;
      v54 = v142;
      v127 = v52;
      v144 = 0;
      v55 = (__int64)v22;
      v56 = (unsigned int)v136;
      LODWORD(v140) = 0;
      v145 = 0;
      do
      {
        v62 = (__int64 *)*a1;
        v143.m128_u64[0] = v55;
        v136 = sub_1D332F0(
                 v62,
                 78,
                 v54,
                 v56,
                 v53,
                 a4,
                 -0.5,
                 *(double *)a7.m128i_i64,
                 a8,
                 v147,
                 *((unsigned __int64 *)&v147 + 1),
                 __PAIR128__(v143.m128_u64[1], v55));
        v137 = v63;
        sub_1F81BC0((__int64)a1, (__int64)v136);
        v132 = sub_1D332F0(
                 (__int64 *)*a1,
                 78,
                 v54,
                 v56,
                 v53,
                 a4,
                 -0.5,
                 *(double *)a7.m128i_i64,
                 a8,
                 (__int64)v136,
                 v137,
                 *(_OWORD *)&v143);
        v134 = v64;
        sub_1F81BC0((__int64)a1, (__int64)v132);
        *(_QWORD *)&v133 = sub_1D332F0(
                             (__int64 *)*a1,
                             76,
                             v54,
                             v56,
                             v53,
                             a4,
                             -0.5,
                             *(double *)a7.m128i_i64,
                             a8,
                             (__int64)v132,
                             v134,
                             v130);
        *((_QWORD *)&v133 + 1) = v65;
        sub_1F81BC0((__int64)a1, v133);
        LODWORD(v140) = v140 + 1;
        v145 = 0;
        if ( v146.m128_i8[0] || v138 > (unsigned int)v140 )
        {
          v57 = sub_1D332F0(
                  (__int64 *)*a1,
                  78,
                  v54,
                  v56,
                  v53,
                  a4,
                  -0.5,
                  *(double *)a7.m128i_i64,
                  a8,
                  v143.m128_i64[0],
                  v143.m128_u64[1],
                  v127);
          v158 = v57;
          v159 = v58;
        }
        else
        {
          v57 = sub_1D332F0(
                  (__int64 *)*a1,
                  78,
                  v54,
                  v56,
                  v53,
                  a4,
                  -0.5,
                  *(double *)a7.m128i_i64,
                  a8,
                  (__int64)v136,
                  v137,
                  v127);
          v156 = v57;
          v157 = v58;
        }
        v144 = (__int64)v57;
        v136 = v57;
        v145 = (unsigned int)v58 | v145 & 0xFFFFFFFF00000000LL;
        sub_1F81BC0((__int64)a1, (__int64)v57);
        v59 = (__int64 *)*a1;
        v144 = (__int64)v136;
        v60 = sub_1D332F0(v59, 78, v54, v56, v53, a4, -0.5, *(double *)a7.m128i_i64, a8, (__int64)v136, v145, v133);
        v155 = v61;
        v154 = v60;
        v143.m128_u64[0] = (unsigned __int64)v60;
        v136 = v60;
        v143.m128_u64[1] = (unsigned int)v61 | v143.m128_u64[1] & 0xFFFFFFFF00000000LL;
        sub_1F81BC0((__int64)a1, (__int64)v60);
        v55 = (__int64)v136;
      }
      while ( v138 != (_DWORD)v140 );
      v12 = v129;
      v22 = v136;
      if ( v179 )
        sub_161E7C0(v142, v179);
      v47 = v143.m128_u64[1];
    }
    v153 = v47;
    v152 = v22;
    v135.m128i_i64[0] = (__int64)v22;
    v135.m128i_i64[1] = (unsigned int)v47 | v135.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    if ( !v12 )
    {
      v66 = (char *)(v141[5] + v139);
      v67 = v141[9];
      v68 = *v66;
      v69 = (const void **)*((_QWORD *)v66 + 1);
      v177 = v67;
      v176 = v69;
      LOBYTE(v175) = v68;
      v146.m128_u64[0] = (unsigned __int64)&v177;
      if ( v67 )
        sub_1623A60((__int64)&v177, v67, 2);
      v70 = a1[1];
      v71 = v175;
      v72 = *((_DWORD *)v141 + 16);
      v144 = (__int64)v176;
      v178 = v72;
      v73 = *(_QWORD *)(*a1 + 32LL);
      v74 = *(__int64 (__fastcall **)(__int64, __int64, unsigned __int64, __int64, __int64))(*(_QWORD *)v70 + 264LL);
      v143.m128_u64[0] = *(_QWORD *)(*a1 + 48LL);
      v75 = sub_1E0A0C0(v73);
      LOBYTE(v144) = v74(v70, v75, v143.m128_u64[0], v71, v144);
      v77 = v76;
      if ( v68 )
        v143.m128_i32[0] = ((unsigned __int8)(v68 - 14) < 0x60u) + 134;
      else
        v143.m128_i32[0] = 134 - (!sub_1F58D20((__int64)&v175) - 1);
      v172 = sub_1560340((_QWORD *)(**(_QWORD **)(*a1 + 32LL) + 112LL), -1, "denormal-fp-math", 0x10u);
      v78 = (_DWORD *)sub_155D8B0(&v172);
      if ( v79 == 4 && *v78 == 1701143913 )
      {
        v94 = sub_1D15FA0((unsigned int)v175, (__int64)v176);
        v141 = sub_16982C0();
        if ( v94 == v141 )
          sub_169C580(&v180, (__int64)v141);
        else
          sub_1698390((__int64)&v180, (__int64)v94);
        if ( v180 == v141 )
          sub_16A1F30((__int64)&v180, 0, v27, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64);
        else
          sub_169B400((__int64)&v180, 0);
        v95 = sub_1D36490(*a1, v142, v146.m128_i64[0], v175, v176, 0, v27, *(double *)a7.m128i_i64, a8);
        v97 = v96;
        v98 = v95;
        *(_QWORD *)&v99 = sub_1D364E0(
                            *a1,
                            v146.m128_i64[0],
                            (unsigned int)v175,
                            v176,
                            0,
                            0.0,
                            *(double *)a7.m128i_i64,
                            a8);
        v100 = (__int64 *)*a1;
        v142 = v99;
        v101 = sub_1D309E0(
                 v100,
                 163,
                 v146.m128_i64[0],
                 (unsigned int)v175,
                 v176,
                 0,
                 0.0,
                 *(double *)a7.m128i_i64,
                 *(double *)a8.m128i_i64,
                 v147);
        v102 = (__int64 *)*a1;
        v103 = (unsigned __int8)v144;
        v140 = v101;
        v139 = (unsigned __int8)v144;
        *(_QWORD *)&v147 = v102;
        v144 = v101;
        v145 = (unsigned __int64)v104;
        v107 = sub_1D28D50(v102, 0x14u, (__int64)v104, v103, v105, v106);
        *((_QWORD *)&v121 + 1) = v97;
        *(_QWORD *)&v121 = v98;
        v108 = v77;
        v109 = v146.m128_u64[0];
        v111 = sub_1D3A900(
                 (__int64 *)v147,
                 0x89u,
                 v146.m128_i64[0],
                 v139,
                 v108,
                 0,
                 (__m128)0LL,
                 *(double *)a7.m128i_i64,
                 a8,
                 v144,
                 (__int16 *)v145,
                 v121,
                 v107,
                 v110);
        v148 = sub_1D3A900(
                 (__int64 *)*a1,
                 v143.m128_u32[0],
                 v109,
                 (unsigned int)v175,
                 v176,
                 0,
                 (__m128)0LL,
                 *(double *)a7.m128i_i64,
                 a8,
                 (unsigned __int64)v111,
                 v112,
                 v142,
                 v135.m128i_i64[0],
                 v135.m128i_i64[1]);
        v22 = v148;
        v149 = v113;
        v135.m128i_i64[0] = (__int64)v148;
        v135.m128i_i64[1] = (unsigned int)v113 | v135.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        sub_1F81BC0((__int64)a1, v140);
        sub_1F81BC0((__int64)a1, (__int64)v111);
        sub_1F81BC0((__int64)a1, (__int64)v22);
        if ( v180 == v141 )
        {
          v114 = v181;
          if ( v181 )
          {
            v115 = v181 + 32LL * *(_QWORD *)(v181 - 8);
            if ( v181 != v115 )
            {
              do
              {
                v115 -= 32;
                if ( v141 == *(_QWORD **)(v115 + 8) )
                {
                  v116 = *(_QWORD *)(v115 + 16);
                  if ( v116 )
                  {
                    v117 = 32LL * *(_QWORD *)(v116 - 8);
                    v118 = v116 + v117;
                    while ( v116 != v118 )
                    {
                      v118 -= 32;
                      if ( v141 == *(_QWORD **)(v118 + 8) )
                      {
                        v119 = *(_QWORD *)(v118 + 16);
                        if ( v119 )
                        {
                          v120 = v119 + 32LL * *(_QWORD *)(v119 - 8);
                          if ( v119 != v120 )
                          {
                            do
                            {
                              v144 = v119;
                              *(_QWORD *)&v147 = v120 - 32;
                              sub_127D120((_QWORD *)(v120 - 24));
                              v120 = v147;
                              v119 = v144;
                            }
                            while ( v144 != (_QWORD)v147 );
                          }
                          j_j_j___libc_free_0_0(v119 - 8);
                        }
                      }
                      else
                      {
                        sub_1698460(v118 + 8);
                      }
                    }
                    j_j_j___libc_free_0_0(v116 - 8);
                  }
                }
                else
                {
                  sub_1698460(v115 + 8);
                }
              }
              while ( v114 != v115 );
            }
            j_j_j___libc_free_0_0(v114 - 8);
          }
        }
        else
        {
          sub_1698460((__int64)&v180);
        }
      }
      else
      {
        v80 = sub_1D364E0(*a1, v146.m128_i64[0], (unsigned int)v175, v176, 0, 0.0, *(double *)a7.m128i_i64, a8);
        v81 = (__int64 *)*a1;
        v82 = v80;
        v84 = v83;
        v85 = (__int64 *)*a1;
        v144 = (unsigned __int8)v144;
        v88 = sub_1D28D50(v85, 0x11u, v83, (unsigned __int8)v144, v86, v87);
        *((_QWORD *)&v122 + 1) = v84;
        *(_QWORD *)&v122 = v82;
        v90 = sub_1D3A900(
                v81,
                0x89u,
                v146.m128_i64[0],
                v144,
                v77,
                0,
                (__m128)0LL,
                *(double *)a7.m128i_i64,
                a8,
                v147,
                *((__int16 **)&v147 + 1),
                v122,
                v88,
                v89);
        *((_QWORD *)&v123 + 1) = v84;
        *(_QWORD *)&v123 = v82;
        v150 = sub_1D3A900(
                 (__int64 *)*a1,
                 v143.m128_u32[0],
                 v146.m128_i64[0],
                 (unsigned int)v175,
                 v176,
                 0,
                 (__m128)0LL,
                 *(double *)a7.m128i_i64,
                 a8,
                 (unsigned __int64)v90,
                 v91,
                 v123,
                 v135.m128i_i64[0],
                 v135.m128i_i64[1]);
        v22 = v150;
        v135.m128i_i64[0] = (__int64)v150;
        v151 = v92;
        v135.m128i_i64[1] = (unsigned int)v92 | v135.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        sub_1F81BC0((__int64)a1, (__int64)v90);
        sub_1F81BC0((__int64)a1, (__int64)v22);
      }
      if ( v177 )
        sub_161E7C0(v146.m128_i64[0], v177);
    }
  }
  return v22;
}
