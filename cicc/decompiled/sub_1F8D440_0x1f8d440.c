// Function: sub_1F8D440
// Address: 0x1f8d440
//
__int64 *__fastcall sub_1F8D440(__int64 a1, __int64 a2, __m128 a3)
{
  __int64 v4; // r12
  __int64 *v5; // rax
  __int64 v6; // rcx
  __m128i v7; // xmm1
  __m128i v8; // xmm2
  bool v9; // al
  int v10; // edx
  char *v11; // rdx
  __int64 v12; // rsi
  const void **v13; // rbx
  unsigned __int8 v14; // cl
  unsigned int v15; // ebx
  unsigned __int16 v16; // si
  __int64 *v17; // rdi
  int v18; // edx
  void *v19; // rax
  __int64 v20; // r15
  void *v21; // r9
  bool v22; // r13
  void *v23; // rax
  void *v24; // r9
  __int64 v25; // rsi
  __int64 *v26; // r13
  __int64 *v27; // rax
  __int64 v28; // rsi
  __int64 *v29; // r14
  int v30; // edx
  void *v31; // rax
  __int64 v32; // r15
  __int64 v33; // rax
  bool v34; // r13
  void *v35; // rax
  void *v36; // r9
  __int64 v37; // rsi
  __int64 *v38; // r13
  int v39; // eax
  __int64 v40; // r14
  void *v41; // r15
  void *v42; // rax
  __int64 v44; // rdi
  bool v45; // al
  __int64 v46; // rdi
  bool v47; // al
  int v48; // edx
  int v49; // edx
  __int64 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // rbx
  __int64 v53; // r15
  __int64 v54; // rax
  __int64 v55; // rsi
  __int64 v56; // rbx
  __int64 v57; // r15
  __int64 v58; // r14
  void *v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rax
  __int64 v62; // r14
  __int64 v63; // rdx
  __int64 v64; // r15
  __int64 v65; // rdi
  bool v66; // al
  __int64 v67; // rdi
  bool v68; // al
  int v69; // eax
  __int64 v70; // rdx
  __int64 v71; // rdi
  int v72; // eax
  __int64 *v73; // r13
  __int128 v74; // rax
  __int64 v75; // rsi
  __int64 *v76; // r13
  __int64 *v77; // rax
  __int64 v78; // r13
  __int64 v79; // r12
  __int64 v80; // rax
  int v81; // edx
  __int64 v82; // rdi
  int v83; // edx
  __int64 *v84; // r13
  __int128 v85; // rax
  __int64 v86; // r15
  __int64 v87; // rsi
  __int64 v88; // rbx
  __int64 v89; // rax
  __int64 v90; // rdx
  const void **v91; // r15
  __int64 *v92; // r12
  double v93; // rax
  const void **v94; // rcx
  double v95; // xmm0_8
  __int128 v96; // rax
  __int128 v97; // rax
  __int64 v98; // rdi
  __int64 (*v99)(); // rax
  __int64 *v100; // r12
  __int128 v101; // rax
  __int128 v102; // [rsp-20h] [rbp-140h]
  __int16 *v103; // [rsp+8h] [rbp-118h]
  void *v104; // [rsp+18h] [rbp-108h]
  unsigned int v105; // [rsp+18h] [rbp-108h]
  void *v106; // [rsp+20h] [rbp-100h]
  void *v107; // [rsp+20h] [rbp-100h]
  void *v108; // [rsp+20h] [rbp-100h]
  __int16 *v109; // [rsp+28h] [rbp-F8h]
  __int16 *v110; // [rsp+28h] [rbp-F8h]
  bool v111; // [rsp+28h] [rbp-F8h]
  unsigned int v112; // [rsp+28h] [rbp-F8h]
  unsigned int v113; // [rsp+28h] [rbp-F8h]
  int v114; // [rsp+30h] [rbp-F0h]
  int v115; // [rsp+34h] [rbp-ECh]
  int v116; // [rsp+38h] [rbp-E8h]
  unsigned __int16 v117; // [rsp+3Ch] [rbp-E4h]
  unsigned __int8 v118; // [rsp+3Eh] [rbp-E2h]
  bool v119; // [rsp+3Fh] [rbp-E1h]
  __int64 v120; // [rsp+60h] [rbp-C0h]
  const void **v121; // [rsp+68h] [rbp-B8h]
  __m128i v122; // [rsp+70h] [rbp-B0h]
  __int64 v123; // [rsp+80h] [rbp-A0h]
  __int64 v124; // [rsp+88h] [rbp-98h]
  __int64 v126; // [rsp+98h] [rbp-88h]
  __int64 v127; // [rsp+A0h] [rbp-80h] BYREF
  int v128; // [rsp+A8h] [rbp-78h]
  __int64 v129[4]; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v130; // [rsp+D0h] [rbp-50h] BYREF
  void *v131; // [rsp+D8h] [rbp-48h] BYREF
  __int64 v132; // [rsp+E0h] [rbp-40h]

  v4 = 0;
  v5 = *(__int64 **)(a2 + 32);
  v6 = *v5;
  v7 = _mm_loadu_si128((const __m128i *)v5);
  v8 = _mm_loadu_si128((const __m128i *)(v5 + 5));
  v116 = *((_DWORD *)v5 + 2);
  v123 = *v5;
  v114 = *((_DWORD *)v5 + 12);
  v124 = v5[5];
  v115 = *((_DWORD *)v5 + 22);
  v120 = v5[10];
  v122 = _mm_loadu_si128((const __m128i *)v5 + 5);
  v9 = *(_WORD *)(*v5 + 24) == 33 || *(_WORD *)(*v5 + 24) == 11;
  v10 = *(unsigned __int16 *)(v124 + 24);
  if ( v9 )
    v4 = v6;
  if ( v10 == 11 || (v126 = 0, v10 == 33) )
    v126 = v124;
  v11 = *(char **)(a2 + 40);
  v12 = *(_QWORD *)(a2 + 72);
  v13 = (const void **)*((_QWORD *)v11 + 1);
  v14 = *v11;
  v127 = v12;
  v121 = v13;
  v118 = v14;
  v15 = v14;
  if ( v12 )
  {
    sub_1623A60((__int64)&v127, v12, 2);
    v9 = *(_WORD *)(v123 + 24) == 33 || *(_WORD *)(v123 + 24) == 11;
  }
  v16 = *(_WORD *)(a2 + 80);
  v17 = *(__int64 **)a1;
  v128 = *(_DWORD *)(a2 + 64);
  v117 = v16;
  if ( (*(_BYTE *)(*v17 + 792) & 2) != 0 )
  {
    if ( v9 )
    {
      v30 = *(unsigned __int16 *)(v124 + 24);
      if ( v30 == 33 || v30 == 11 )
      {
        v48 = *(unsigned __int16 *)(v120 + 24);
        if ( v48 == 11 || v48 == 33 )
          goto LABEL_76;
      }
    }
  }
  else
  {
    v119 = (*(_BYTE *)(a2 + 81) & 2) != 0 || (*(_BYTE *)(a2 + 81) & 8) != 0;
    if ( v9 )
    {
      v18 = *(unsigned __int16 *)(v124 + 24);
      if ( v18 == 33 || v18 == 11 )
      {
        v49 = *(unsigned __int16 *)(v120 + 24);
        if ( v49 == 11 || v49 == 33 )
        {
LABEL_76:
          v29 = sub_1D3A900(
                  v17,
                  0x63u,
                  (__int64)&v127,
                  v15,
                  v121,
                  0,
                  a3,
                  *(double *)v7.m128i_i64,
                  v8,
                  v7.m128i_u64[0],
                  (__int16 *)v7.m128i_i64[1],
                  *(_OWORD *)&v8,
                  v122.m128i_i64[0],
                  v122.m128i_i64[1]);
          goto LABEL_60;
        }
      }
    }
    if ( !v119 )
    {
      if ( !v4 )
      {
LABEL_36:
        if ( !v126 )
        {
LABEL_45:
          v9 = *(_WORD *)(v123 + 24) == 11 || *(_WORD *)(v123 + 24) == 33;
          goto LABEL_46;
        }
        v32 = *(_QWORD *)(v126 + 88);
        v21 = sub_16982C0();
LABEL_38:
        v107 = v21;
        a3 = (__m128)0x3FF0000000000000uLL;
        v110 = (__int16 *)sub_1698280();
        sub_169D3F0((__int64)v129, 1.0);
        sub_169E320(&v131, v129, v110);
        sub_1698460((__int64)v129);
        v34 = 0;
        sub_16A3360((__int64)&v130, *(__int16 **)(v32 + 32), 0, (bool *)v129);
        v35 = v131;
        v36 = v107;
        if ( *(void **)(v32 + 32) == v131 )
        {
          v46 = v32 + 32;
          if ( v131 == v107 )
            v47 = sub_169CB90(v46, (__int64)&v131);
          else
            v47 = sub_1698510(v46, (__int64)&v131);
          v36 = v107;
          v34 = v47;
          v35 = v131;
        }
        if ( v36 == v35 )
        {
          v54 = v132;
          if ( v132 )
          {
            v55 = 32LL * *(_QWORD *)(v132 - 8);
            if ( v132 != v132 + v55 )
            {
              v113 = v15;
              v56 = v132 + v55;
              v57 = v132;
              do
              {
                v56 -= 32;
                sub_127D120((_QWORD *)(v56 + 8));
              }
              while ( v57 != v56 );
              v15 = v113;
              v54 = v57;
            }
            j_j_j___libc_free_0_0(v54 - 8);
          }
        }
        else
        {
          sub_1698460((__int64)&v131);
        }
        if ( v34 )
        {
          v37 = *(_QWORD *)(a2 + 72);
          v38 = *(__int64 **)a1;
          v130 = v37;
          if ( v37 )
            sub_1623A60((__int64)&v130, v37, 2);
          LODWORD(v131) = *(_DWORD *)(a2 + 64);
          v27 = sub_1D332F0(
                  v38,
                  76,
                  (__int64)&v130,
                  v15,
                  v121,
                  0,
                  1.0,
                  *(double *)v7.m128i_i64,
                  v8,
                  v7.m128i_i64[0],
                  v7.m128i_u64[1],
                  *(_OWORD *)&v122);
LABEL_21:
          v28 = v130;
          v29 = v27;
          if ( !v130 )
            goto LABEL_60;
          goto LABEL_22;
        }
        goto LABEL_45;
      }
      v19 = sub_16982C0();
      v20 = *(_QWORD *)(v4 + 88);
      v21 = v19;
LABEL_14:
      v106 = v21;
      a3 = (__m128)0x3FF0000000000000uLL;
      v109 = (__int16 *)sub_1698280();
      sub_169D3F0((__int64)v129, 1.0);
      sub_169E320(&v131, v129, v109);
      sub_1698460((__int64)v129);
      v22 = 0;
      sub_16A3360((__int64)&v130, *(__int16 **)(v20 + 32), 0, (bool *)v129);
      v23 = v131;
      v24 = v106;
      if ( *(void **)(v20 + 32) == v131 )
      {
        v44 = v20 + 32;
        if ( v131 == v106 )
          v45 = sub_169CB90(v44, (__int64)&v131);
        else
          v45 = sub_1698510(v44, (__int64)&v131);
        v24 = v106;
        v22 = v45;
        v23 = v131;
      }
      if ( v24 == v23 )
      {
        v50 = v132;
        if ( v132 )
        {
          v51 = 32LL * *(_QWORD *)(v132 - 8);
          if ( v132 != v132 + v51 )
          {
            v112 = v15;
            v52 = v132 + v51;
            v53 = v132;
            do
            {
              v52 -= 32;
              sub_127D120((_QWORD *)(v52 + 8));
            }
            while ( v53 != v52 );
            v15 = v112;
            v50 = v53;
          }
          j_j_j___libc_free_0_0(v50 - 8);
        }
      }
      else
      {
        sub_1698460((__int64)&v131);
      }
      if ( v22 )
      {
        v25 = *(_QWORD *)(a2 + 72);
        v26 = *(__int64 **)a1;
        v130 = v25;
        if ( v25 )
          sub_1623A60((__int64)&v130, v25, 2);
        LODWORD(v131) = *(_DWORD *)(a2 + 64);
        v27 = sub_1D332F0(
                v26,
                76,
                (__int64)&v130,
                v15,
                v121,
                0,
                1.0,
                *(double *)v7.m128i_i64,
                v8,
                v8.m128i_i64[0],
                v8.m128i_u64[1],
                *(_OWORD *)&v122);
        goto LABEL_21;
      }
      goto LABEL_36;
    }
  }
  if ( v4 )
  {
    v31 = sub_16982C0();
    v20 = *(_QWORD *)(v4 + 88);
    v21 = v31;
    if ( *(void **)(v20 + 32) == v31 )
    {
      if ( (*(_BYTE *)(*(_QWORD *)(v20 + 40) + 26LL) & 7) == 3 )
        goto LABEL_59;
    }
    else if ( (*(_BYTE *)(v20 + 50) & 7) == 3 )
    {
      goto LABEL_59;
    }
    if ( !v126 )
    {
      v119 = 1;
      goto LABEL_14;
    }
LABEL_31:
    v32 = *(_QWORD *)(v126 + 88);
    v33 = v32 + 32;
    if ( *(void **)(v32 + 32) == v21 )
      v33 = *(_QWORD *)(v32 + 40) + 8LL;
    if ( (*(_BYTE *)(v33 + 18) & 7) != 3 )
    {
      if ( !v4 )
      {
        v119 = 1;
        goto LABEL_38;
      }
      v119 = 1;
      v20 = *(_QWORD *)(v4 + 88);
      goto LABEL_14;
    }
LABEL_59:
    v29 = (__int64 *)v122.m128i_i64[0];
    goto LABEL_60;
  }
  if ( v126 )
  {
    v21 = sub_16982C0();
    goto LABEL_31;
  }
  v119 = 1;
LABEL_46:
  if ( !v9 && !(unsigned __int8)sub_1D16930(v123)
    || (v39 = *(unsigned __int16 *)(v124 + 24), v39 == 11)
    || v39 == 33
    || (unsigned __int8)sub_1D16930(v124) )
  {
    if ( !v119 )
      goto LABEL_53;
    if ( *(_WORD *)(v120 + 24) == 78 )
    {
      v80 = *(_QWORD *)(v120 + 32);
      if ( *(_QWORD *)v80 == v123 && *(_DWORD *)(v80 + 8) == v116 )
      {
        v81 = *(unsigned __int16 *)(v124 + 24);
        if ( v81 != 11 && v81 != 33 )
        {
          if ( !(unsigned __int8)sub_1D16930(v124) )
            goto LABEL_52;
          v80 = *(_QWORD *)(v120 + 32);
        }
        v82 = *(_QWORD *)(v80 + 40);
        v83 = *(unsigned __int16 *)(v82 + 24);
        if ( v83 == 11 || v83 == 33 )
          goto LABEL_141;
        if ( (unsigned __int8)sub_1D16930(v82) )
        {
          v80 = *(_QWORD *)(v120 + 32);
LABEL_141:
          v84 = *(__int64 **)a1;
          *(_QWORD *)&v85 = sub_1D332F0(
                              *(__int64 **)a1,
                              76,
                              (__int64)&v127,
                              v15,
                              v121,
                              v117,
                              *(double *)a3.m128_u64,
                              *(double *)v7.m128i_i64,
                              v8,
                              v8.m128i_i64[0],
                              v8.m128i_u64[1],
                              *(_OWORD *)(v80 + 40));
          v29 = sub_1D332F0(
                  v84,
                  78,
                  (__int64)&v127,
                  v15,
                  v121,
                  v117,
                  *(double *)a3.m128_u64,
                  *(double *)v7.m128i_i64,
                  v8,
                  v7.m128i_i64[0],
                  v7.m128i_u64[1],
                  v85);
          goto LABEL_60;
        }
      }
    }
LABEL_52:
    if ( *(_WORD *)(v123 + 24) == 78 )
    {
      v69 = *(unsigned __int16 *)(v124 + 24);
      if ( v69 == 11 || v69 == 33 || (unsigned __int8)sub_1D16930(v124) )
      {
        v70 = *(_QWORD *)(v123 + 32);
        v71 = *(_QWORD *)(v70 + 40);
        v72 = *(unsigned __int16 *)(v71 + 24);
        if ( v72 == 11 || v72 == 33 )
          goto LABEL_118;
        if ( (unsigned __int8)sub_1D16930(v71) )
        {
          v70 = *(_QWORD *)(v123 + 32);
LABEL_118:
          v73 = *(__int64 **)a1;
          *(_QWORD *)&v74 = sub_1D332F0(
                              *(__int64 **)a1,
                              78,
                              (__int64)&v127,
                              v15,
                              v121,
                              v117,
                              *(double *)a3.m128_u64,
                              *(double *)v7.m128i_i64,
                              v8,
                              v8.m128i_i64[0],
                              v8.m128i_u64[1],
                              *(_OWORD *)(v70 + 40));
          v29 = sub_1D3A900(
                  v73,
                  0x63u,
                  (__int64)&v127,
                  v15,
                  v121,
                  0,
                  a3,
                  *(double *)v7.m128i_i64,
                  v8,
                  **(_QWORD **)(v123 + 32),
                  *(__int16 **)(*(_QWORD *)(v123 + 32) + 8LL),
                  v74,
                  v122.m128i_i64[0],
                  v122.m128i_i64[1]);
          goto LABEL_60;
        }
      }
    }
LABEL_53:
    if ( !v126 )
      goto LABEL_73;
    v40 = *(_QWORD *)(v126 + 88);
    v103 = (__int16 *)sub_1698280();
    sub_169D3F0((__int64)v129, 1.0);
    sub_169E320(&v131, v129, v103);
    sub_1698460((__int64)v129);
    sub_16A3360((__int64)&v130, *(__int16 **)(v40 + 32), 0, (bool *)v129);
    v41 = v131;
    v104 = *(void **)(v40 + 32);
    v42 = sub_16982C0();
    v111 = 0;
    v108 = v42;
    if ( v104 == v41 )
    {
      v65 = v40 + 32;
      if ( v41 == v42 )
        v66 = sub_169CB90(v65, (__int64)&v131);
      else
        v66 = sub_1698510(v65, (__int64)&v131);
      v41 = v131;
      v111 = v66;
    }
    if ( v41 == v108 )
    {
      v86 = v132;
      if ( v132 )
      {
        v87 = 32LL * *(_QWORD *)(v132 - 8);
        if ( v132 != v132 + v87 )
        {
          v105 = v15;
          v88 = v132 + v87;
          do
          {
            v88 -= 32;
            sub_127D120((_QWORD *)(v88 + 8));
          }
          while ( v86 != v88 );
          v15 = v105;
        }
        j_j_j___libc_free_0_0(v86 - 8);
      }
    }
    else
    {
      sub_1698460((__int64)&v131);
    }
    if ( v111 )
    {
      v29 = sub_1D332F0(
              *(__int64 **)a1,
              76,
              (__int64)&v127,
              v15,
              v121,
              0,
              1.0,
              *(double *)v7.m128i_i64,
              v8,
              v7.m128i_i64[0],
              v7.m128i_u64[1],
              *(_OWORD *)&v122);
      goto LABEL_60;
    }
    v58 = *(_QWORD *)(v126 + 88);
    sub_169D3F0((__int64)v129, -1.0);
    sub_169E320(&v131, v129, v103);
    sub_1698460((__int64)v129);
    sub_16A3360((__int64)&v130, *(__int16 **)(v58 + 32), 0, (bool *)v129);
    v59 = v131;
    if ( *(void **)(v58 + 32) == v131 )
    {
      v67 = v58 + 32;
      if ( v131 == v108 )
        v68 = sub_169CB90(v67, (__int64)&v131);
      else
        v68 = sub_1698510(v67, (__int64)&v131);
      v111 = v68;
      v59 = v131;
    }
    if ( v59 == v108 )
    {
      v78 = v132;
      if ( v132 )
      {
        v79 = v132 + 32LL * *(_QWORD *)(v132 - 8);
        if ( v132 != v79 )
        {
          do
          {
            v79 -= 32;
            sub_127D120((_QWORD *)(v79 + 8));
          }
          while ( v78 != v79 );
        }
        j_j_j___libc_free_0_0(v78 - 8);
      }
    }
    else
    {
      sub_1698460((__int64)&v131);
    }
    if ( v111 )
    {
      if ( !*(_BYTE *)(a1 + 24) )
      {
LABEL_103:
        v62 = sub_1D309E0(
                *(__int64 **)a1,
                162,
                (__int64)&v127,
                v15,
                v121,
                0,
                -1.0,
                *(double *)v7.m128i_i64,
                *(double *)v8.m128i_i64,
                *(_OWORD *)&v7);
        v64 = v63;
        sub_1F81BC0(a1, v62);
        *((_QWORD *)&v102 + 1) = v64;
        *(_QWORD *)&v102 = v62;
        v29 = sub_1D332F0(
                *(__int64 **)a1,
                76,
                (__int64)&v127,
                v15,
                v121,
                0,
                -1.0,
                *(double *)v7.m128i_i64,
                v8,
                v122.m128i_i64[0],
                v122.m128i_u64[1],
                v102);
        goto LABEL_60;
      }
      v60 = *(_QWORD *)(a1 + 8);
      v61 = 1;
      if ( v118 != 1 )
      {
        if ( !v118 )
        {
          if ( *(_WORD *)(v123 + 24) != 162 )
            goto LABEL_150;
          goto LABEL_160;
        }
        v61 = v118;
        if ( !*(_QWORD *)(v60 + 8LL * v118 + 120) )
        {
          if ( *(_WORD *)(v123 + 24) != 162 )
            goto LABEL_150;
          goto LABEL_169;
        }
      }
      if ( !*(_BYTE *)(v60 + 259 * v61 + 2584) )
        goto LABEL_103;
    }
    if ( *(_WORD *)(v123 + 24) != 162 )
      goto LABEL_150;
    v60 = *(_QWORD *)(a1 + 8);
    v61 = 1;
    if ( v118 == 1 )
    {
LABEL_159:
      if ( !*(_BYTE *)(v60 + 259 * v61 + 2433) )
      {
LABEL_162:
        v100 = *(__int64 **)a1;
        *(_QWORD *)&v101 = sub_1D309E0(
                             *(__int64 **)a1,
                             162,
                             (__int64)&v127,
                             v15,
                             v121,
                             v117,
                             -1.0,
                             *(double *)v7.m128i_i64,
                             *(double *)v8.m128i_i64,
                             *(_OWORD *)&v8);
        v29 = sub_1D3A900(
                v100,
                0x63u,
                (__int64)&v127,
                v15,
                v121,
                0,
                (__m128)0xBFF0000000000000LL,
                *(double *)v7.m128i_i64,
                v8,
                **(_QWORD **)(v123 + 32),
                *(__int16 **)(*(_QWORD *)(v123 + 32) + 8LL),
                v101,
                v122.m128i_i64[0],
                v122.m128i_i64[1]);
        goto LABEL_60;
      }
      goto LABEL_160;
    }
    if ( !v118 )
    {
LABEL_160:
      if ( sub_1D18C00(v124, 1, v114) )
      {
        v98 = *(_QWORD *)(a1 + 8);
        v99 = *(__int64 (**)())(*(_QWORD *)v98 + 328LL);
        if ( v99 == sub_1F3CA70
          || !((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD, const void **))v99)(
                v98,
                *(_QWORD *)(v126 + 88) + 24LL,
                v15,
                v121) )
        {
          goto LABEL_162;
        }
      }
LABEL_150:
      if ( v119 )
      {
        if ( v123 == v120 && v116 == v115 )
        {
          v90 = v15;
          v91 = v121;
          v92 = *(__int64 **)a1;
          v93 = 1.0;
          v94 = v121;
          goto LABEL_157;
        }
        if ( *(_WORD *)(v120 + 24) == 162 )
        {
          v89 = *(_QWORD *)(v120 + 32);
          if ( *(_QWORD *)v89 == v123 && v116 == *(_DWORD *)(v89 + 8) )
          {
            v90 = v15;
            v91 = v121;
            v92 = *(__int64 **)a1;
            v93 = -1.0;
            v94 = v121;
LABEL_157:
            v95 = v93;
            *(_QWORD *)&v96 = sub_1D364E0((__int64)v92, (__int64)&v127, v90, v94, 0, v93, *(double *)v7.m128i_i64, v8);
            *(_QWORD *)&v97 = sub_1D332F0(
                                v92,
                                76,
                                (__int64)&v127,
                                v15,
                                v91,
                                v117,
                                v95,
                                *(double *)v7.m128i_i64,
                                v8,
                                v8.m128i_i64[0],
                                v8.m128i_u64[1],
                                v96);
            v29 = sub_1D332F0(
                    v92,
                    78,
                    (__int64)&v127,
                    v15,
                    v91,
                    v117,
                    v95,
                    *(double *)v7.m128i_i64,
                    v8,
                    v7.m128i_i64[0],
                    v7.m128i_u64[1],
                    v97);
            goto LABEL_60;
          }
        }
      }
LABEL_73:
      v29 = 0;
      goto LABEL_60;
    }
    v61 = v118;
LABEL_169:
    if ( *(_QWORD *)(v60 + 8LL * (int)v61 + 120) )
      goto LABEL_159;
    goto LABEL_160;
  }
  v75 = *(_QWORD *)(a2 + 72);
  v130 = v75;
  v76 = *(__int64 **)a1;
  if ( v75 )
    sub_1623A60((__int64)&v130, v75, 2);
  LODWORD(v131) = *(_DWORD *)(a2 + 64);
  v77 = sub_1D3A900(
          v76,
          0x63u,
          (__int64)&v130,
          v15,
          v121,
          0,
          a3,
          *(double *)v7.m128i_i64,
          v8,
          v8.m128i_u64[0],
          (__int16 *)v8.m128i_i64[1],
          *(_OWORD *)&v7,
          v122.m128i_i64[0],
          v122.m128i_i64[1]);
  v28 = v130;
  v29 = v77;
  if ( v130 )
LABEL_22:
    sub_161E7C0((__int64)&v130, v28);
LABEL_60:
  if ( v127 )
    sub_161E7C0((__int64)&v127, v127);
  return v29;
}
