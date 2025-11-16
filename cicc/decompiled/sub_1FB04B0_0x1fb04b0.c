// Function: sub_1FB04B0
// Address: 0x1fb04b0
//
__int64 __fastcall sub_1FB04B0(__int64 **a1, __int64 a2, double a3, __int64 a4, __int64 a5, int a6, int a7)
{
  __int64 v8; // rbx
  __int64 v9; // rax
  __m128i v10; // xmm1
  __m128i v11; // xmm2
  unsigned int v12; // ecx
  __int64 v13; // rcx
  __int64 v14; // r13
  int v15; // ecx
  int v16; // r8d
  int v17; // r9d
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rax
  unsigned __int8 v21; // r14
  const void **v22; // rax
  char v23; // cl
  __int64 *v24; // rax
  __int64 v25; // r13
  __int64 *v27; // rax
  int v28; // eax
  int v29; // eax
  __int64 v30; // r13
  __int64 **v31; // r15
  __int64 **v32; // rax
  __int64 **v33; // rdx
  __int64 *v34; // rcx
  int v35; // r8d
  int v36; // r9d
  char v37; // r13
  __int16 v38; // ax
  __int64 v39; // r15
  __int64 **v40; // rax
  __int64 **v41; // rdx
  __int64 v42; // r15
  __int64 **v43; // rax
  __int64 ***v44; // r8
  unsigned __int8 v45; // dl
  __int64 v46; // rcx
  char v47; // r15
  int v48; // edx
  __int64 v49; // rsi
  const __m128i *v50; // roff
  __int64 v51; // r13
  __int64 v52; // r14
  unsigned __int64 v53; // r15
  char *v54; // rax
  char v55; // dl
  __int64 v56; // rax
  int v57; // r8d
  __int64 v58; // rbx
  char v59; // al
  __int64 *v60; // rdi
  __int64 (*v61)(); // r9
  __int64 v62; // rdi
  bool v63; // al
  int *v64; // r15
  __int64 v65; // rsi
  int *v66; // r13
  void *v67; // rax
  __int64 v68; // rax
  bool v69; // al
  __int64 v70; // rax
  __int64 v71; // rax
  __int64 v72; // r13
  __int64 v73; // r15
  __int64 *v74; // rdx
  __m128i v75; // xmm4
  __m128i v76; // xmm5
  int v77; // ecx
  int v78; // r8d
  int v79; // r9d
  __int64 v80; // rdi
  bool v81; // al
  __int64 v82; // rdi
  bool v83; // al
  char v84; // al
  __int64 *v85; // r13
  _QWORD *v86; // rax
  __int64 v87; // rdx
  __int64 v88; // r15
  _QWORD *v89; // r14
  _QWORD *v90; // rax
  unsigned __int64 v91; // rdx
  int *v92; // r15
  int *v93; // r13
  int *v94; // r15
  int *v95; // rbx
  __int64 v96; // rax
  _QWORD *v97; // rax
  unsigned __int64 v98; // rdx
  __int128 v99; // rax
  __int64 *v100; // rsi
  __int64 v101; // rax
  __int64 v102; // rax
  __int64 *v103; // rax
  __int64 v104; // r13
  __int64 v105; // r15
  int v106; // edx
  __int16 v107; // cx
  __int64 v108; // rdx
  __int64 v109; // rax
  __int64 v110; // rax
  int v111; // ecx
  __int64 v112; // rdi
  __int128 v113; // rax
  char v114; // al
  char v115; // si
  __int64 *v116; // rdi
  __int64 v117; // rdx
  __int64 v118; // rax
  __int64 v119; // rax
  int v120; // eax
  __int64 (*v121)(); // rax
  __int64 *v122; // rax
  __int64 v123; // rdi
  __int64 *v124; // r9
  __int64 *v125; // rbx
  char v126; // al
  __int64 *v127; // rax
  unsigned __int16 v128; // r8
  __int64 *v129; // rax
  _BYTE **v130; // rdi
  __int64 v131; // rt1
  __int64 v132; // rax
  char v133; // al
  __int64 *v134; // rdi
  __int64 *v135; // r12
  __int128 v136; // rax
  __int128 v137; // [rsp-10h] [rbp-1B0h]
  __int64 **v138; // [rsp+10h] [rbp-190h]
  __int64 **v139; // [rsp+18h] [rbp-188h]
  __int64 v140; // [rsp+18h] [rbp-188h]
  __int16 *v141; // [rsp+20h] [rbp-180h]
  __int64 **v142; // [rsp+20h] [rbp-180h]
  __m128i v143; // [rsp+20h] [rbp-180h]
  __int16 *v144; // [rsp+38h] [rbp-168h]
  __int64 **v145; // [rsp+38h] [rbp-168h]
  __int64 **v146; // [rsp+38h] [rbp-168h]
  __int64 v147; // [rsp+38h] [rbp-168h]
  bool v148; // [rsp+40h] [rbp-160h]
  unsigned int v149; // [rsp+4Ch] [rbp-154h]
  bool v150; // [rsp+50h] [rbp-150h]
  bool v151; // [rsp+50h] [rbp-150h]
  __m128i v152; // [rsp+50h] [rbp-150h]
  bool v153; // [rsp+62h] [rbp-13Eh]
  bool v154; // [rsp+63h] [rbp-13Dh]
  unsigned int v155; // [rsp+64h] [rbp-13Ch]
  bool v156; // [rsp+68h] [rbp-138h]
  __int64 v157; // [rsp+70h] [rbp-130h]
  unsigned __int16 v158; // [rsp+78h] [rbp-128h]
  __int64 v159; // [rsp+80h] [rbp-120h]
  __int64 v160; // [rsp+88h] [rbp-118h]
  int v161; // [rsp+88h] [rbp-118h]
  unsigned __int16 v162; // [rsp+88h] [rbp-118h]
  __m128i v163; // [rsp+90h] [rbp-110h] BYREF
  __m128i v164; // [rsp+A0h] [rbp-100h]
  __m128i v165; // [rsp+B0h] [rbp-F0h]
  bool v166; // [rsp+CBh] [rbp-D5h] BYREF
  int v167; // [rsp+CCh] [rbp-D4h] BYREF
  unsigned int v168; // [rsp+D0h] [rbp-D0h] BYREF
  const void **v169; // [rsp+D8h] [rbp-C8h]
  __int64 v170; // [rsp+E0h] [rbp-C0h] BYREF
  int v171; // [rsp+E8h] [rbp-B8h]
  unsigned int v172; // [rsp+F0h] [rbp-B0h] BYREF
  __int64 v173; // [rsp+F8h] [rbp-A8h]
  __int64 v174; // [rsp+100h] [rbp-A0h] BYREF
  int v175; // [rsp+108h] [rbp-98h]
  __int64 v176[6]; // [rsp+110h] [rbp-90h] BYREF
  bool *v177; // [rsp+140h] [rbp-60h] BYREF
  __int64 **v178; // [rsp+148h] [rbp-58h] BYREF
  int *v179; // [rsp+150h] [rbp-50h]
  __int64 *v180; // [rsp+158h] [rbp-48h]
  unsigned int *v181; // [rsp+160h] [rbp-40h]

  v8 = a2;
  v9 = *(_QWORD *)(a2 + 32);
  v10 = _mm_loadu_si128((const __m128i *)v9);
  v11 = _mm_loadu_si128((const __m128i *)(v9 + 40));
  v160 = *(_QWORD *)v9;
  v12 = *(_DWORD *)(v9 + 8);
  v165 = v10;
  v155 = v12;
  v13 = *(_QWORD *)(v9 + 40);
  LODWORD(v9) = *(_DWORD *)(v9 + 48);
  v163 = v11;
  v159 = v13;
  v149 = v9;
  v14 = sub_1D23470(v10.m128i_i64[0], v10.m128i_i64[1], v10.m128i_i64[1], v13, a6, a7);
  v18 = sub_1D23470(v11.m128i_i64[0], v11.m128i_i64[1], v11.m128i_i64[1], v15, v16, v17);
  v19 = *(_QWORD *)(a2 + 72);
  v164.m128i_i64[0] = v18;
  v20 = *(_QWORD *)(v8 + 40);
  v21 = *(_BYTE *)v20;
  v22 = *(const void ***)(v20 + 8);
  v170 = v19;
  LOBYTE(v168) = v21;
  v169 = v22;
  if ( v19 )
    sub_1623A60((__int64)&v170, v19, 2);
  v171 = *(_DWORD *)(v8 + 64);
  v157 = **a1;
  v158 = *(_WORD *)(v8 + 80);
  v23 = *(_BYTE *)(v8 + 80);
  v156 = (v23 & 0x40) != 0;
  v154 = (v23 & 0x10) != 0;
  v150 = (*(_BYTE *)(v8 + 81) & 8) != 0;
  if ( v21 )
  {
    if ( (unsigned __int8)(v21 - 14) > 0x5Fu )
      goto LABEL_5;
  }
  else if ( !sub_1F58D20((__int64)&v168) )
  {
    goto LABEL_5;
  }
  v27 = sub_1FA8C50((__int64)a1, v8, a3, *(double *)v10.m128i_i64, v11);
  if ( v27 )
  {
LABEL_13:
    v25 = (__int64)v27;
    goto LABEL_8;
  }
LABEL_5:
  v153 = v164.m128i_i64[0] != 0 && v14 != 0;
  if ( v153 )
  {
    v24 = sub_1D332F0(
            *a1,
            78,
            (__int64)&v170,
            v168,
            v169,
            v158,
            a3,
            *(double *)v10.m128i_i64,
            v11,
            v165.m128i_i64[0],
            v165.m128i_u64[1],
            *(_OWORD *)&v163);
LABEL_7:
    v25 = (__int64)v24;
    goto LABEL_8;
  }
  v28 = *(unsigned __int16 *)(v160 + 24);
  if ( v28 == 11 || v28 == 33 || (unsigned __int8)sub_1D16930(v160) )
  {
    v29 = *(unsigned __int16 *)(v159 + 24);
    if ( v29 != 11 && v29 != 33 && !(unsigned __int8)sub_1D16930(v159) )
    {
      v24 = sub_1D332F0(
              *a1,
              78,
              (__int64)&v170,
              v168,
              v169,
              v158,
              a3,
              *(double *)v10.m128i_i64,
              v11,
              v163.m128i_i64[0],
              v163.m128i_u64[1],
              *(_OWORD *)&v165);
      goto LABEL_7;
    }
  }
  if ( v164.m128i_i64[0] )
  {
    v30 = *(_QWORD *)(v164.m128i_i64[0] + 88);
    a3 = 1.0;
    v144 = (__int16 *)sub_1698280();
    sub_169D3F0((__int64)v176, 1.0);
    sub_169E320(&v178, v176, v144);
    sub_1698460((__int64)v176);
    sub_16A3360((__int64)&v177, *(__int16 **)(v30 + 32), 0, (bool *)v176);
    v31 = v178;
    v145 = *(__int64 ***)(v30 + 32);
    v32 = (__int64 **)sub_16982C0();
    v148 = 0;
    v33 = v32;
    if ( v145 == v31 )
    {
      v142 = v32;
      v62 = v30 + 32;
      if ( v31 == v32 )
      {
        v69 = sub_169CB90(v62, (__int64)&v178);
        v31 = v178;
        v33 = v142;
        v148 = v69;
      }
      else
      {
        v63 = sub_1698510(v62, (__int64)&v178);
        v31 = v178;
        v148 = v63;
        v33 = v142;
      }
    }
    if ( v31 == v33 )
    {
      v64 = v179;
      if ( v179 )
      {
        v65 = 8LL * *((_QWORD *)v179 - 1);
        v66 = &v179[v65];
        if ( v179 != &v179[v65] )
        {
          do
          {
            v66 -= 8;
            sub_127D120((_QWORD *)v66 + 1);
          }
          while ( v64 != v66 );
        }
        j_j_j___libc_free_0_0(v64 - 2);
      }
    }
    else
    {
      sub_1698460((__int64)&v178);
    }
    if ( v148 )
    {
      v27 = (__int64 *)v165.m128i_i64[0];
      goto LABEL_13;
    }
  }
  v34 = sub_1F77C50(a1, v8, a3, *(double *)v10.m128i_i64, v11);
  v25 = (__int64)v34;
  if ( v34 )
    goto LABEL_8;
  v37 = *(_BYTE *)(v157 + 792) & 2;
  if ( v37 )
  {
    if ( !v164.m128i_i64[0] )
    {
LABEL_31:
      v38 = *(_WORD *)(v160 + 24);
      if ( v38 == 78 )
      {
        v71 = *(_QWORD *)(v160 + 32);
        v72 = v159;
        v73 = *(_QWORD *)v71;
        v74 = *(__int64 **)(v71 + 40);
        v75 = _mm_loadu_si128((const __m128i *)v71);
        v76 = _mm_loadu_si128((const __m128i *)(v71 + 40));
        if ( *(_WORD *)(v159 + 24) != 104 )
          v72 = 0;
        v152 = v75;
        if ( *(_WORD *)(v73 + 24) != 104 )
          v73 = 0;
        v143 = v76;
        if ( *((_WORD *)v74 + 12) != 104 )
          v74 = v34;
        v147 = (__int64)v74;
        if ( !sub_1D23470(v75.m128i_i64[0], v75.m128i_i64[1], v75.m128i_i64[1], (int)v34, v35, v36)
          && (!v73 || !(unsigned __int8)sub_1D23510(v73))
          && (v164.m128i_i64[0] && sub_1D23470(v143.m128i_i64[0], v143.m128i_i64[1], v143.m128i_i64[1], v77, v78, v79)
           || v72 && v147 && (unsigned __int8)sub_1D23510(v72) && (unsigned __int8)sub_1D23510(v147)) )
        {
          *(_QWORD *)&v113 = sub_1D332F0(
                               *a1,
                               78,
                               (__int64)&v170,
                               v168,
                               v169,
                               v158,
                               a3,
                               *(double *)v10.m128i_i64,
                               v11,
                               v143.m128i_i64[0],
                               v143.m128i_u64[1],
                               *(_OWORD *)&v163);
          v24 = sub_1D332F0(
                  *a1,
                  78,
                  (__int64)&v170,
                  v168,
                  v169,
                  v158,
                  a3,
                  *(double *)v10.m128i_i64,
                  v11,
                  v152.m128i_i64[0],
                  v152.m128i_u64[1],
                  v113);
          goto LABEL_7;
        }
        v38 = *(_WORD *)(v160 + 24);
      }
      if ( v38 == 76 && sub_1D18C00(v160, 1, v155) )
      {
        v96 = *(_QWORD *)(v160 + 32);
        if ( *(_QWORD *)v96 == *(_QWORD *)(v96 + 40) && *(_DWORD *)(v96 + 8) == *(_DWORD *)(v96 + 48) )
        {
          v97 = sub_1D364E0((__int64)*a1, (__int64)&v170, v168, v169, 0, 2.0, *(double *)v10.m128i_i64, v11);
          *(_QWORD *)&v99 = sub_1D332F0(
                              *a1,
                              78,
                              (__int64)&v170,
                              v168,
                              v169,
                              v158,
                              2.0,
                              *(double *)v10.m128i_i64,
                              v11,
                              (__int64)v97,
                              v98,
                              *(_OWORD *)&v163);
          v24 = sub_1D332F0(
                  *a1,
                  78,
                  (__int64)&v170,
                  v168,
                  v169,
                  v158,
                  2.0,
                  *(double *)v10.m128i_i64,
                  v11,
                  **(_QWORD **)(v160 + 32),
                  *(_QWORD *)(*(_QWORD *)(v160 + 32) + 8LL),
                  v99);
          goto LABEL_7;
        }
      }
LABEL_33:
      if ( v164.m128i_i64[0] )
      {
        v39 = *(_QWORD *)(v164.m128i_i64[0] + 88);
        goto LABEL_35;
      }
LABEL_74:
      v46 = (__int64)a1[1];
      v45 = *((_BYTE *)a1 + 24);
      goto LABEL_44;
    }
  }
  else
  {
    if ( !v156 || !v154 )
    {
      if ( !v150 )
        goto LABEL_33;
      goto LABEL_31;
    }
    if ( !v164.m128i_i64[0] )
    {
      if ( !v150 )
        goto LABEL_74;
      goto LABEL_31;
    }
  }
  v39 = *(_QWORD *)(v164.m128i_i64[0] + 88);
  v67 = sub_16982C0();
  v34 = 0;
  if ( *(void **)(v39 + 32) == v67 )
    v68 = *(_QWORD *)(v39 + 40) + 8LL;
  else
    v68 = v39 + 32;
  if ( (*(_BYTE *)(v68 + 18) & 7) == 3 )
  {
    v25 = v163.m128i_i64[0];
    goto LABEL_8;
  }
  if ( v37 || v150 )
    goto LABEL_31;
LABEL_35:
  v141 = (__int16 *)sub_1698280();
  sub_169D3F0((__int64)v176, 2.0);
  sub_169E320(&v178, v176, v141);
  sub_1698460((__int64)v176);
  sub_16A3360((__int64)&v177, *(__int16 **)(v39 + 32), 0, (bool *)v176);
  v138 = v178;
  v139 = *(__int64 ***)(v39 + 32);
  v40 = (__int64 **)sub_16982C0();
  v41 = v138;
  v151 = 0;
  v146 = v40;
  if ( v139 == v138 )
  {
    v82 = v39 + 32;
    if ( v138 == v40 )
      v83 = sub_169CB90(v82, (__int64)&v178);
    else
      v83 = sub_1698510(v82, (__int64)&v178);
    v41 = v178;
    v151 = v83;
  }
  if ( v41 == v146 )
  {
    v94 = v179;
    if ( v179 )
    {
      if ( v179 != &v179[8 * *((_QWORD *)v179 - 1)] )
      {
        v140 = v8;
        v95 = &v179[8 * *((_QWORD *)v179 - 1)];
        do
        {
          v95 -= 8;
          sub_127D120((_QWORD *)v95 + 1);
        }
        while ( v94 != v95 );
        v8 = v140;
      }
      j_j_j___libc_free_0_0(v94 - 2);
    }
  }
  else
  {
    sub_1698460((__int64)&v178);
  }
  if ( v151 )
  {
    v24 = sub_1D332F0(
            *a1,
            76,
            (__int64)&v170,
            v168,
            v169,
            v158,
            2.0,
            *(double *)v10.m128i_i64,
            v11,
            v165.m128i_i64[0],
            v165.m128i_u64[1],
            *(_OWORD *)&v165);
    goto LABEL_7;
  }
  a3 = -1.0;
  v42 = *(_QWORD *)(v164.m128i_i64[0] + 88);
  sub_169D3F0((__int64)v176, -1.0);
  v164.m128i_i64[0] = (__int64)&v178;
  sub_169E320(&v178, v176, v141);
  sub_1698460((__int64)v176);
  sub_16A3360((__int64)&v177, *(__int16 **)(v42 + 32), 0, (bool *)v176);
  v43 = v178;
  v44 = &v178;
  if ( *(__int64 ***)(v42 + 32) == v178 )
  {
    v80 = v42 + 32;
    if ( v178 == v146 )
      v81 = sub_169CB90(v80, v164.m128i_i64[0]);
    else
      v81 = sub_1698510(v80, v164.m128i_i64[0]);
    v44 = (__int64 ***)v164.m128i_i64[0];
    v151 = v81;
    v43 = v178;
  }
  if ( v43 == v146 )
  {
    v92 = v179;
    if ( v179 )
    {
      v93 = &v179[8 * *((_QWORD *)v179 - 1)];
      if ( v179 != v93 )
      {
        do
        {
          v93 -= 8;
          sub_127D120((_QWORD *)v93 + 1);
        }
        while ( v92 != v93 );
      }
      j_j_j___libc_free_0_0(v92 - 2);
    }
  }
  else
  {
    sub_1698460((__int64)v44);
  }
  v45 = *((_BYTE *)a1 + 24);
  if ( v151 )
  {
    if ( !v45
      || ((v46 = (__int64)a1[1], v70 = 1, v21 == 1) || v21 && (v70 = v21, *(_QWORD *)(v46 + 8LL * v21 + 120)))
      && !*(_BYTE *)(v46 + 259 * v70 + 2584) )
    {
      v25 = sub_1D309E0(
              *a1,
              162,
              (__int64)&v170,
              v168,
              v169,
              0,
              -1.0,
              *(double *)v10.m128i_i64,
              *(double *)v11.m128i_i64,
              *(_OWORD *)&v165);
      goto LABEL_8;
    }
  }
  else
  {
    v46 = (__int64)a1[1];
  }
LABEL_44:
  v47 = sub_1F79A30(
          v160,
          v155,
          v45,
          v46,
          (_BYTE *)(v157 + 792),
          0,
          a3,
          *(double *)v10.m128i_i64,
          *(double *)v11.m128i_i64);
  if ( v47 )
  {
    v84 = sub_1F79A30(
            v159,
            v149,
            *((_BYTE *)a1 + 24),
            (__int64)a1[1],
            (_BYTE *)(v157 + 792),
            0,
            a3,
            *(double *)v10.m128i_i64,
            *(double *)v11.m128i_i64);
    if ( v84 )
    {
      if ( v47 == 2 || v84 == 2 )
      {
        v85 = *a1;
        v86 = sub_1F7A040(
                v163.m128i_i64[0],
                v163.m128i_u32[2],
                *a1,
                *((_BYTE *)a1 + 24),
                0,
                a3,
                *(double *)v10.m128i_i64,
                v11);
        v88 = v87;
        v89 = v86;
        v90 = sub_1F7A040(
                v165.m128i_i64[0],
                v165.m128i_u32[2],
                *a1,
                *((_BYTE *)a1 + 24),
                0,
                a3,
                *(double *)v10.m128i_i64,
                v11);
        *((_QWORD *)&v137 + 1) = v88;
        *(_QWORD *)&v137 = v89;
        v25 = (__int64)sub_1D332F0(
                         v85,
                         78,
                         (__int64)&v170,
                         v168,
                         v169,
                         v158,
                         a3,
                         *(double *)v10.m128i_i64,
                         v11,
                         (__int64)v90,
                         v91,
                         v137);
        goto LABEL_8;
      }
    }
  }
  if ( v156 && v154 )
  {
    v48 = *(unsigned __int16 *)(v160 + 24);
    if ( v48 == 134 || *(_WORD *)(v159 + 24) == 134 )
    {
      v100 = a1[1];
      v101 = 1;
      if ( v21 == 1 || v21 && (v101 = v21, v100[v21 + 15]) )
      {
        if ( !*((_BYTE *)v100 + 259 * v101 + 2585) )
        {
          v165 = _mm_load_si128(&v163);
          if ( v48 == 134 )
          {
            v155 = v149;
            v102 = v159;
            v159 = v160;
            v160 = v102;
          }
          v103 = *(__int64 **)(v159 + 32);
          v104 = v103[5];
          v105 = v103[10];
          v106 = *(unsigned __int16 *)(v104 + 24);
          v107 = *(_WORD *)(v105 + 24);
          if ( (v106 == 11 || v106 == 33) && (v107 == 11 || v107 == 33) )
          {
            v108 = *v103;
            if ( *(_WORD *)(*v103 + 24) == 137 )
            {
              v109 = *(_QWORD *)(v108 + 32);
              if ( v160 == *(_QWORD *)v109 && v155 == *(_DWORD *)(v109 + 8) )
              {
                v110 = *(_QWORD *)(v109 + 40);
                v111 = *(unsigned __int16 *)(v110 + 24);
                if ( v111 == 11 || v111 == 33 )
                {
                  v112 = *(_QWORD *)(v110 + 88);
                  a3 = 0.0;
                  v164.m128i_i64[0] = v108;
                  if ( (unsigned __int8)sub_1F7B5F0(v112 + 24, 0.0) )
                  {
                    switch ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v164.m128i_i64[0] + 32) + 80LL) + 84LL) )
                    {
                      case 2:
                      case 3:
                      case 0xA:
                      case 0xB:
                      case 0x12:
                      case 0x13:
                        break;
                      case 4:
                      case 5:
                      case 0xC:
                      case 0xD:
                      case 0x14:
                      case 0x15:
                        v131 = v105;
                        v105 = v104;
                        v104 = v131;
                        break;
                      default:
                        goto LABEL_49;
                    }
                    a3 = 1.0;
                    if ( (unsigned __int8)sub_1F7B5F0(*(_QWORD *)(v104 + 88) + 24LL, -1.0) )
                    {
                      v132 = *(_QWORD *)(v105 + 88);
                      v164.m128i_i64[0] = 0x3FF0000000000000LL;
                      v133 = sub_1F7B5F0(v132 + 24, 1.0);
                      a3 = *(double *)v164.m128i_i64;
                      if ( v133 )
                      {
                        if ( sub_1F6C830((__int64)a1[1], 0xA2u, v21) )
                        {
                          v135 = *a1;
                          v165.m128i_i64[0] = v160;
                          v165.m128i_i64[1] = v155 | v165.m128i_i64[1] & 0xFFFFFFFF00000000LL;
                          *(_QWORD *)&v136 = sub_1D309E0(
                                               v135,
                                               163,
                                               (__int64)&v170,
                                               v168,
                                               v169,
                                               0,
                                               a3,
                                               *(double *)v10.m128i_i64,
                                               *(double *)v11.m128i_i64,
                                               __PAIR128__(v165.m128i_u64[1], v160));
                          v25 = sub_1D309E0(
                                  v135,
                                  162,
                                  (__int64)&v170,
                                  v168,
                                  v169,
                                  0,
                                  a3,
                                  *(double *)v10.m128i_i64,
                                  *(double *)v11.m128i_i64,
                                  v136);
                          goto LABEL_8;
                        }
                      }
                    }
                    if ( (unsigned __int8)sub_1F7B5F0(*(_QWORD *)(v104 + 88) + 24LL, a3) )
                    {
                      a3 = -1.0;
                      if ( (unsigned __int8)sub_1F7B5F0(*(_QWORD *)(v105 + 88) + 24LL, -1.0) )
                      {
                        v134 = *a1;
                        v165.m128i_i64[0] = v160;
                        v165.m128i_i64[1] = v155 | v165.m128i_i64[1] & 0xFFFFFFFF00000000LL;
                        v25 = sub_1D309E0(
                                v134,
                                163,
                                (__int64)&v170,
                                v168,
                                v169,
                                0,
                                -1.0,
                                *(double *)v10.m128i_i64,
                                *(double *)v11.m128i_i64,
                                __PAIR128__(v165.m128i_u64[1], v160));
                        goto LABEL_8;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
LABEL_49:
  v49 = *(_QWORD *)(v8 + 72);
  v50 = *(const __m128i **)(v8 + 32);
  v51 = v50->m128i_i64[0];
  v52 = v50[2].m128i_i64[1];
  v53 = v50[3].m128i_u64[0];
  v164 = _mm_loadu_si128(v50);
  v165.m128i_i64[0] = v52;
  v54 = *(char **)(v8 + 40);
  v55 = *v54;
  v56 = *((_QWORD *)v54 + 1);
  v174 = v49;
  LOBYTE(v172) = v55;
  v173 = v56;
  if ( v49 )
    sub_1623A60((__int64)&v174, v49, 2);
  v57 = *(unsigned __int16 *)(v8 + 80);
  v175 = *(_DWORD *)(v8 + 64);
  v58 = **a1;
  v59 = *(_BYTE *)(v58 + 792);
  if ( (v59 & 4) == 0 )
    goto LABEL_56;
  if ( *(_DWORD *)(v58 + 816) )
  {
    if ( (v59 & 2) == 0 )
    {
LABEL_56:
      if ( v174 )
        sub_161E7C0((__int64)&v174, v174);
      v25 = 0;
      goto LABEL_8;
    }
    v60 = a1[1];
    v61 = *(__int64 (**)())(*v60 + 920);
    if ( v61 == sub_1F3CBF0 )
      goto LABEL_55;
  }
  else
  {
    v60 = a1[1];
    v61 = *(__int64 (**)())(*v60 + 920);
    if ( v61 == sub_1F3CBF0 )
      goto LABEL_119;
  }
  v163.m128i_i32[0] = v57;
  v114 = ((__int64 (__fastcall *)(__int64 *, _QWORD, __int64))v61)(v60, v172, v173);
  v57 = v163.m128i_i32[0];
  if ( !v114 )
  {
    v59 = *(_BYTE *)(v58 + 792);
LABEL_119:
    if ( (v59 & 2) == 0 )
      goto LABEL_56;
LABEL_55:
    if ( !*((_BYTE *)a1 + 24) )
      goto LABEL_56;
    v116 = a1[1];
    v117 = (unsigned __int8)v172;
    v115 = 0;
    goto LABEL_152;
  }
  v115 = *((_BYTE *)a1 + 24);
  v116 = a1[1];
  if ( v115 )
  {
    v117 = (unsigned __int8)v172;
    v118 = 1;
    if ( (_BYTE)v172 != 1 )
    {
      if ( !(_BYTE)v172 )
        goto LABEL_56;
      v118 = (unsigned __int8)v172;
      if ( !v116[(unsigned __int8)v172 + 15] )
        goto LABEL_56;
    }
    if ( (*((_BYTE *)v116 + 259 * v118 + 2521) & 0xFB) != 0 )
    {
      v115 = 0;
      if ( (*(_BYTE *)(v58 + 792) & 2) == 0 )
        goto LABEL_56;
      goto LABEL_152;
    }
    if ( (*(_BYTE *)(v58 + 792) & 2) != 0 )
    {
LABEL_152:
      v119 = 1;
      if ( ((_BYTE)v117 == 1 || (_BYTE)v117 && (v119 = (unsigned __int8)v117, v116[v117 + 15]))
        && !*((_BYTE *)v116 + 259 * v119 + 2522) )
      {
        v120 = 100;
      }
      else
      {
        v120 = 99;
        if ( !v115 )
          goto LABEL_56;
      }
      goto LABEL_155;
    }
  }
  v120 = 99;
LABEL_155:
  v167 = v120;
  v121 = *(__int64 (**)())(*v116 + 256);
  if ( v121 != sub_1F3CA50 )
  {
    v163.m128i_i32[0] = v57;
    v126 = ((__int64 (__fastcall *)(__int64 *, _QWORD, __int64))v121)(v116, v172, v173);
    v57 = v163.m128i_i32[0];
    v153 = v126;
  }
  v161 = v57;
  v166 = v153;
  v176[0] = (__int64)&v166;
  v176[2] = (__int64)&v167;
  v176[3] = (__int64)&v174;
  v163.m128i_i64[0] = (__int64)v176;
  v176[1] = (__int64)a1;
  v176[4] = (__int64)&v172;
  v122 = sub_1F78AA0((__int64)v176, v51, v52, v53, v57, (int)&v174, a3, *(double *)v10.m128i_i64, v11);
  v123 = v163.m128i_i64[0];
  v124 = &v174;
  v125 = v122;
  if ( !v122 )
  {
    v163.m128i_i32[0] = v161;
    v127 = sub_1F78AA0(
             v123,
             v165.m128i_i64[0],
             v164.m128i_i64[0],
             v164.m128i_u64[1],
             v161,
             (int)&v174,
             a3,
             *(double *)v10.m128i_i64,
             v11);
    v128 = v163.m128i_i16[0];
    v124 = &v174;
    v125 = v127;
    if ( !v127 )
    {
      v180 = &v174;
      v177 = &v166;
      v179 = &v167;
      v162 = v163.m128i_i16[0];
      v163.m128i_i64[0] = (__int64)&v177;
      v178 = a1;
      v181 = &v172;
      v129 = sub_1F79160(&v177, v51, v52, v53, v128, (int)&v174, a3, *(double *)v10.m128i_i64, v11);
      v130 = (_BYTE **)v163.m128i_i64[0];
      v124 = &v174;
      v125 = v129;
      if ( !v129 )
      {
        v163.m128i_i64[0] = (__int64)&v174;
        v125 = sub_1F79160(
                 v130,
                 v165.m128i_i64[0],
                 v164.m128i_i64[0],
                 v164.m128i_i64[1],
                 v162,
                 (int)&v174,
                 a3,
                 *(double *)v10.m128i_i64,
                 v11);
        if ( !v125 )
          goto LABEL_56;
        v124 = (__int64 *)v163.m128i_i64[0];
      }
    }
  }
  if ( v174 )
    sub_161E7C0((__int64)v124, v174);
  v25 = (__int64)v125;
  sub_1F81BC0((__int64)a1, (__int64)v125);
LABEL_8:
  if ( v170 )
    sub_161E7C0((__int64)&v170, v170);
  return v25;
}
