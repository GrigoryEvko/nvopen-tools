// Function: sub_124D500
// Address: 0x124d500
//
__int64 __fastcall sub_124D500(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rdx
  int v7; // eax
  __int64 v8; // rdi
  __int64 v9; // rbx
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // rax
  __m128i *v14; // r15
  __int64 v15; // r12
  unsigned __int64 v16; // rax
  __int64 *v17; // rax
  __int64 *v18; // r13
  __int64 v19; // r15
  bool v20; // r14
  bool v21; // bl
  bool v22; // r12
  __int64 v23; // rdx
  __int64 v24; // rcx
  unsigned int v25; // eax
  __int64 v26; // rdi
  char v27; // al
  bool v28; // bl
  char *v29; // r14
  char v30; // al
  __int64 (*v31)(); // rax
  char v32; // al
  _QWORD *v33; // rbx
  int v34; // ebx
  char v35; // al
  __int64 v36; // r12
  char *v37; // r14
  __int64 *v38; // r14
  unsigned __int64 v39; // rax
  void *v40; // rax
  char v41; // al
  __int64 v42; // rdx
  _QWORD *v43; // rdx
  __int64 v44; // rax
  const __m128i *v45; // r13
  int v46; // r12d
  __int64 v47; // rbx
  __int64 v48; // r15
  int v49; // r13d
  char *v50; // r12
  __int64 v51; // r14
  unsigned __int64 v52; // rax
  unsigned int v53; // eax
  int v54; // ecx
  int v55; // eax
  unsigned int v56; // edx
  int v57; // ecx
  __int64 v58; // r12
  unsigned __int64 v59; // rax
  unsigned int v60; // eax
  __int64 v61; // rcx
  char *v62; // r12
  __int64 v63; // r13
  unsigned __int64 v64; // rax
  unsigned int v65; // esi
  const __m128i *v66; // r14
  __m128i *v67; // r13
  char *v68; // r14
  __int64 v69; // rbx
  unsigned __int64 v70; // rax
  unsigned int v71; // eax
  int v72; // esi
  __int64 v73; // rcx
  unsigned int v74; // edx
  __int64 v75; // rax
  _QWORD *v76; // rbx
  __int64 v77; // rax
  unsigned int *v78; // r14
  __int64 result; // rax
  unsigned int *v80; // rbx
  _QWORD *v81; // r13
  unsigned int *v82; // r12
  __int64 v83; // r13
  unsigned int v84; // eax
  __int64 v85; // rdi
  _QWORD *v86; // rbx
  __int64 v87; // r8
  __int64 v88; // rax
  void *v89; // rax
  __int64 v90; // rdi
  void *v91; // rax
  _QWORD *v92; // rax
  int v93; // ecx
  __int64 v94; // rsi
  int v95; // ecx
  unsigned int v96; // edx
  __int64 *v97; // rax
  __int64 v98; // rdi
  unsigned __int64 v99; // r12
  int v100; // eax
  __int64 *v101; // rbx
  unsigned __int64 v102; // rdx
  __int64 *v103; // rax
  _QWORD *v104; // rax
  int v105; // eax
  int v106; // r8d
  __int64 v108; // [rsp+18h] [rbp-180h]
  __int64 v109; // [rsp+28h] [rbp-170h]
  unsigned __int64 v110; // [rsp+30h] [rbp-168h]
  __int64 v111; // [rsp+38h] [rbp-160h]
  __int64 v112; // [rsp+40h] [rbp-158h]
  int v113; // [rsp+48h] [rbp-150h]
  int v114; // [rsp+58h] [rbp-140h]
  char v115; // [rsp+5Ch] [rbp-13Ch]
  int v116; // [rsp+5Ch] [rbp-13Ch]
  __int64 v117; // [rsp+60h] [rbp-138h]
  __m128i *v118; // [rsp+60h] [rbp-138h]
  char v120; // [rsp+70h] [rbp-128h]
  __int64 *v122; // [rsp+80h] [rbp-118h]
  __int64 v123; // [rsp+80h] [rbp-118h]
  __int64 v124; // [rsp+88h] [rbp-110h]
  __int64 v125; // [rsp+88h] [rbp-110h]
  __int64 v126; // [rsp+90h] [rbp-108h]
  __int32 v127; // [rsp+90h] [rbp-108h]
  const __m128i *v128; // [rsp+90h] [rbp-108h]
  int v129; // [rsp+90h] [rbp-108h]
  char *v130; // [rsp+90h] [rbp-108h]
  int v131; // [rsp+90h] [rbp-108h]
  __m128i *v132; // [rsp+90h] [rbp-108h]
  __int64 v133; // [rsp+90h] [rbp-108h]
  const __m128i *v134; // [rsp+98h] [rbp-100h] BYREF
  __m128i *v135; // [rsp+A0h] [rbp-F8h]
  const __m128i *v136; // [rsp+A8h] [rbp-F0h]
  const __m128i *v137; // [rsp+B8h] [rbp-E0h] BYREF
  __m128i *v138; // [rsp+C0h] [rbp-D8h]
  const __m128i *v139; // [rsp+C8h] [rbp-D0h]
  __m128i v140; // [rsp+D8h] [rbp-C0h] BYREF
  __m128i v141; // [rsp+E8h] [rbp-B0h] BYREF
  char v142; // [rsp+F8h] [rbp-A0h]
  char v143; // [rsp+F9h] [rbp-9Fh]
  unsigned __int8 v144[16]; // [rsp+108h] [rbp-90h] BYREF
  _QWORD *v145; // [rsp+118h] [rbp-80h]
  __int64 v146; // [rsp+120h] [rbp-78h]
  __int16 v147; // [rsp+128h] [rbp-70h]
  __int64 v148; // [rsp+138h] [rbp-60h] BYREF
  bool v149; // [rsp+140h] [rbp-58h]
  unsigned int *v150; // [rsp+148h] [rbp-50h]
  unsigned int *v151; // [rsp+150h] [rbp-48h]
  __int64 v152; // [rsp+158h] [rbp-40h]
  int v153; // [rsp+160h] [rbp-38h]

  v4 = *(_QWORD *)a1;
  v5 = *(_QWORD *)a2;
  v6 = *(_QWORD *)(v4 + 112);
  LOBYTE(v4) = *(_BYTE *)(v6 + 12);
  v150 = 0;
  v151 = 0;
  v152 = 0;
  v153 = 0;
  v149 = (v4 & 2) != 0;
  LOBYTE(v4) = *(_BYTE *)(v6 + 12);
  v140.m128i_i64[0] = (__int64)".symtab";
  v148 = a1;
  v147 = 257;
  v108 = v5;
  v143 = 1;
  v142 = 3;
  v110 = sub_E71CB0(v5, (size_t *)&v140, 2, 0, (v4 & 2) == 0 ? 16 : 24, (__int64)v144, 0, -1, 0);
  *(_BYTE *)(v110 + 32) = ((*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 112LL) + 12LL) & 2) != 0) + 2;
  v7 = sub_124CD00((_QWORD *)a1, v110);
  v8 = a1;
  v9 = a1 + 32;
  *(_DWORD *)(v9 + 56) = v7;
  v10 = 0;
  v112 = sub_124BD80(v8, *(_BYTE *)(v110 + 32));
  sub_124CF30((__int64)&v148, 0, 0, 0, 0, 0, 0, 0);
  v11 = *(_QWORD *)(v9 - 32);
  v136 = 0;
  v139 = 0;
  v135 = 0;
  v138 = 0;
  v12 = *(_QWORD *)(v11 + 8);
  v134 = 0;
  v13 = *(unsigned int *)(v11 + 16);
  v137 = 0;
  v111 = v13;
  v124 = v12 + 40 * v13;
  if ( v124 != v12 )
  {
    v126 = v12;
    do
    {
      v14 = *(__m128i **)v12;
      v15 = *(_QWORD *)(v12 + 8);
      v12 += 40;
      v16 = sub_C94890(v14, v15);
      v10 = (__int64)v14;
      sub_C0CA60(v9, (__int64)v14, (v16 << 32) | (unsigned int)v15);
    }
    while ( v124 != v12 );
    v12 = v126;
  }
  v115 = 0;
  v127 = 0;
  v17 = *(__int64 **)(a2 + 56);
  v122 = &v17[*(unsigned int *)(a2 + 64)];
  if ( v17 == v122 )
  {
LABEL_66:
    v113 = 0;
    goto LABEL_67;
  }
  v117 = v9;
  v18 = *(__int64 **)(a2 + 56);
  v109 = v12;
  do
  {
    v19 = *v18;
    v20 = (*(_BYTE *)(*v18 + 9) & 8) != 0;
    v120 = *(_BYTE *)(*v18 + 9) & 8;
    v21 = sub_EA16D0(*v18);
    v22 = sub_EA16F0(v19);
    v23 = *(unsigned int *)(*(_QWORD *)a1 + 192LL);
    v24 = *(_QWORD *)(*(_QWORD *)a1 + 176LL);
    if ( (_DWORD)v23
      && (v10 = (unsigned int)(v23 - 1),
          v25 = v10 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4)),
          v26 = *(_QWORD *)(v24 + 16LL * v25),
          v23 = 1,
          v26 != v19) )
    {
      while ( v26 != -4096 )
      {
        v25 = v10 & (v23 + v25);
        v26 = *(_QWORD *)(v24 + 16LL * v25);
        if ( v19 == v26 )
        {
          v23 = 1;
          goto LABEL_14;
        }
        LODWORD(v23) = v23 + 1;
      }
      v23 = 0;
      v27 = *(_BYTE *)(v19 + 9) & 0x70;
      v28 = v20 || v22 || v21;
      if ( v28 )
        goto LABEL_15;
    }
    else
    {
LABEL_14:
      v27 = *(_BYTE *)(v19 + 9) & 0x70;
      v28 = v20 || v22 || v21;
      if ( v28 )
      {
LABEL_15:
        if ( v27 != 32 )
          goto LABEL_23;
        goto LABEL_16;
      }
    }
    if ( v27 != 32 )
    {
      if ( (_DWORD)v23 )
        goto LABEL_11;
      goto LABEL_21;
    }
LABEL_16:
    v29 = *(char **)(v19 + 24);
    *(_BYTE *)(v19 + 8) |= 8u;
    v30 = *v29;
    if ( *v29 != 4 )
      goto LABEL_52;
    v31 = *(__int64 (**)())(*((_QWORD *)v29 - 1) + 56LL);
    if ( v31 != sub_E4C910 )
    {
      v114 = v23;
      v41 = ((__int64 (__fastcall *)(char *, __int64, __int64, __int64))v31)(v29 - 8, v10, v23, v24);
      LODWORD(v23) = v114;
      if ( v41 )
        goto LABEL_11;
      v30 = *v29;
LABEL_52:
      if ( v30 == 2 && *(_WORD *)(v29 + 1) == 29 )
        goto LABEL_11;
    }
    if ( v28 )
      goto LABEL_23;
    if ( (_DWORD)v23 )
      goto LABEL_11;
    if ( (*(_BYTE *)(v19 + 9) & 0x70) == 0x20 && !*(_QWORD *)v19 )
    {
      if ( *(char *)(v19 + 8) < 0
        || (*(_BYTE *)(v19 + 8) |= 8u, v89 = sub_E807D0(*(_QWORD *)(v19 + 24)), (*(_QWORD *)v19 = v89) == 0) )
      {
        v10 = v19;
        sub_E5C930((__int64 *)a2, v19);
        goto LABEL_11;
      }
    }
LABEL_21:
    if ( (*(_BYTE *)(v19 + 8) & 2) != 0 || (unsigned int)sub_EA1630(v19) == 3 )
      goto LABEL_11;
LABEL_23:
    v32 = *(_BYTE *)(v19 + 8);
    if ( (v32 & 2) != 0 )
    {
      v33 = *(_QWORD **)v19;
      if ( !*(_QWORD *)v19 )
      {
        if ( (*(_BYTE *)(v19 + 9) & 0x70) != 0x20 || v32 < 0 )
          goto LABEL_56;
        v90 = *(_QWORD *)(v19 + 24);
        *(_BYTE *)(v19 + 8) = v32 | 8;
        v91 = sub_E807D0(v90);
        *(_QWORD *)v19 = v91;
        if ( !v91 )
        {
          v32 = *(_BYTE *)(v19 + 8);
LABEL_56:
          v42 = 0;
          if ( (v32 & 1) != 0 )
          {
            v101 = *(__int64 **)(v19 - 8);
            v42 = *v101;
            v33 = v101 + 3;
          }
          v145 = v33;
          v147 = 1283;
          *(_QWORD *)v144 = "Undefined temporary symbol ";
LABEL_59:
          v146 = v42;
          v10 = 0;
          sub_E66880(v108, 0, (__int64)v144);
          goto LABEL_11;
        }
      }
    }
    v141.m128i_i64[0] = 0;
    v140 = (__m128i)(unsigned __int64)v19;
    v141.m128i_i32[3] = v127;
    v34 = sub_EA1780(v19);
    if ( *(_QWORD *)v19 || (*(_BYTE *)(v19 + 9) & 0x70) != 0x20 )
    {
      if ( off_4C5D170 == *(_UNKNOWN **)v19 )
        goto LABEL_49;
    }
    else
    {
      if ( *(char *)(v19 + 8) < 0 )
      {
        if ( !off_4C5D170 )
        {
LABEL_49:
          v141.m128i_i32[2] = 65521;
          goto LABEL_30;
        }
        goto LABEL_111;
      }
      *(_BYTE *)(v19 + 8) |= 8u;
      v40 = sub_E807D0(*(_QWORD *)(v19 + 24));
      *(_QWORD *)v19 = v40;
      if ( off_4C5D170 == v40 )
        goto LABEL_49;
    }
    v35 = *(_BYTE *)(v19 + 9) & 0x70;
    if ( ((v35 - 48) & 0xE0) == 0 )
    {
      if ( v35 == 64 )
        v141.m128i_i32[2] = *(_DWORD *)(v19 + 16);
      else
        v141.m128i_i32[2] = 65522;
      goto LABEL_30;
    }
    v43 = *(_QWORD **)v19;
    if ( *(_QWORD *)v19 )
      goto LABEL_61;
    if ( v35 != 32 )
      goto LABEL_113;
LABEL_111:
    if ( *(char *)(v19 + 8) < 0
      || (*(_BYTE *)(v19 + 8) |= 8u, v92 = sub_E807D0(*(_QWORD *)(v19 + 24)), *(_QWORD *)v19 = v92, (v43 = v92) == 0) )
    {
LABEL_113:
      if ( !v120 && v22 )
      {
        v93 = *(_DWORD *)(a3 + 24);
        v94 = *(_QWORD *)(a3 + 8);
        if ( v93 )
        {
          v95 = v93 - 1;
          v96 = v95 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
          v97 = (__int64 *)(v94 + 16LL * v96);
          v98 = *v97;
          if ( v19 == *v97 )
          {
LABEL_117:
            v141.m128i_i32[2] = *((_DWORD *)v97 + 2);
            if ( v141.m128i_i32[2] > 0xFEFFu )
LABEL_64:
              v115 = 1;
LABEL_30:
            v36 = 4;
            v37 = ".L0 ";
            if ( (*(_BYTE *)(v19 + 8) & 1) != 0 )
            {
              v38 = *(__int64 **)(v19 - 8);
              v36 = *v38;
              v37 = (char *)(v38 + 3);
              if ( !v36 )
              {
                v36 = 4;
                v37 = ".L0 ";
              }
            }
            if ( (unsigned int)sub_EA1630(v19) != 3 )
            {
              v140.m128i_i64[1] = (__int64)v37;
              v141.m128i_i64[0] = v36;
              v39 = sub_C94890(v37, v36);
              sub_C0CA60(v117, (__int64)v37, (v39 << 32) | (unsigned int)v36);
            }
            if ( v34 )
            {
              v10 = (__int64)v138;
              if ( v138 == v139 )
              {
                sub_124B8C0(&v137, v138, &v140);
              }
              else
              {
                if ( v138 )
                {
                  *v138 = _mm_loadu_si128(&v140);
                  *(__m128i *)(v10 + 16) = _mm_loadu_si128(&v141);
                  v10 = (__int64)v138;
                }
                v10 += 32;
                v138 = (__m128i *)v10;
              }
            }
            else
            {
              v10 = (__int64)v135;
              if ( v135 == v136 )
              {
                sub_124B8C0(&v134, v135, &v140);
              }
              else
              {
                if ( v135 )
                {
                  *v135 = _mm_loadu_si128(&v140);
                  *(__m128i *)(v10 + 16) = _mm_loadu_si128(&v141);
                  v10 = (__int64)v135;
                }
                v10 += 32;
                v135 = (__m128i *)v10;
              }
            }
            goto LABEL_11;
          }
          v105 = 1;
          while ( v98 != -4096 )
          {
            v106 = v105 + 1;
            v96 = v95 & (v105 + v96);
            v97 = (__int64 *)(v94 + 16LL * v96);
            v98 = *v97;
            if ( v19 == *v97 )
              goto LABEL_117;
            v105 = v106;
          }
        }
      }
      v141.m128i_i32[2] = 0;
      goto LABEL_30;
    }
LABEL_61:
    v44 = v43[1];
    if ( (*(_BYTE *)(v44 + 48) & 8) == 0 )
    {
      if ( (*(_BYTE *)(v19 + 8) & 1) != 0 )
      {
        v103 = *(__int64 **)(v19 - 8);
        v42 = *v103;
        v104 = v103 + 3;
      }
      else
      {
        v42 = 0;
        v104 = 0;
      }
      v145 = v104;
      v147 = 1283;
      *(_QWORD *)v144 = "Undefined section reference: ";
      goto LABEL_59;
    }
    if ( *(_DWORD *)(a1 + 24) != 1
      || (v102 = *(_QWORD *)(v44 + 136), v102 <= 3)
      || *(_DWORD *)(*(_QWORD *)(v44 + 128) + v102 - 4) != 1870095406 )
    {
      v141.m128i_i32[2] = *(_DWORD *)(v44 + 36);
      if ( v141.m128i_i32[2] > 0xFEFFu )
        goto LABEL_64;
      goto LABEL_30;
    }
LABEL_11:
    ++v127;
    ++v18;
  }
  while ( v122 != v18 );
  v9 = v117;
  v12 = v109;
  if ( !v115 )
    goto LABEL_66;
  v147 = 257;
  v143 = 1;
  v140.m128i_i64[0] = (__int64)".symtab_shndx";
  v142 = 3;
  v99 = sub_E71CB0(v108, (size_t *)&v140, 18, 0, 4, (__int64)v144, 0, -1, 0);
  v100 = sub_124CD00((_QWORD *)a1, v99);
  *(_BYTE *)(v99 + 32) = 2;
  v113 = v100;
LABEL_67:
  sub_C0D290(v9);
  if ( v111 )
    *(_QWORD *)(v12 + 32) = 0;
  v45 = v134;
  v46 = 1;
  v118 = v135;
  if ( v135 != v134 )
  {
    v123 = v9;
    v47 = v12;
    do
    {
      if ( v124 != v47 )
      {
        v128 = v45;
        v48 = v47;
        v49 = v46;
        do
        {
          v54 = v49++;
          if ( *(_QWORD *)(v48 + 32) > (unsigned __int64)v128[1].m128i_u32[3] )
          {
            v45 = v128;
            v47 = v48;
            goto LABEL_76;
          }
          v50 = *(char **)v48;
          v51 = *(_QWORD *)(v48 + 8);
          v48 += 40;
          v52 = sub_C94890(v50, v51);
          v53 = sub_C0C3A0(v123, v50, (v52 << 32) | (unsigned int)v51);
          sub_124CF30((__int64)&v148, v53, 4u, 0, 0, 0, 0xFFF1u, 1);
        }
        while ( v124 != v48 );
        v46 = v49;
        v45 = v128;
        v47 = v48;
      }
      v54 = v46;
LABEL_76:
      v129 = v54;
      v55 = sub_EA1630(v45->m128i_i64[0]);
      v56 = 0;
      v57 = v129;
      if ( v55 != 3 )
      {
        v58 = v45[1].m128i_i64[0];
        v116 = v129;
        v130 = (char *)v45->m128i_i64[1];
        v59 = sub_C94890(v130, v58);
        v60 = sub_C0C3A0(v123, v130, (v59 << 32) | (unsigned int)v58);
        v57 = v116;
        v56 = v60;
      }
      v46 = v57 + 1;
      *(_DWORD *)(v45->m128i_i64[0] + 16) = v57;
      v61 = (__int64)v45;
      v45 += 2;
      sub_124D200((__int64 *)a2, (__int64)&v148, v56, v61);
    }
    while ( v118 != v45 );
    v12 = v47;
    v9 = v123;
  }
  if ( v124 != v12 )
  {
    v131 = v46;
    do
    {
      v62 = *(char **)v12;
      v63 = *(_QWORD *)(v12 + 8);
      v12 += 40;
      v64 = sub_C94890(v62, v63);
      v65 = sub_C0C3A0(v9, v62, (v64 << 32) | (unsigned int)v63);
      sub_124CF30((__int64)&v148, v65, 4u, 0, 0, 0, 0xFFF1u, 1);
      ++v131;
    }
    while ( v124 != v12 );
    v46 = v131;
  }
  v66 = v137;
  *(_DWORD *)(a1 + 80) = v46;
  v132 = v138;
  if ( v138 != v66 )
  {
    v125 = v9;
    v67 = (__m128i *)v66;
    do
    {
      v68 = (char *)v67->m128i_i64[1];
      v69 = v67[1].m128i_i64[0];
      v70 = sub_C94890(v68, v69);
      v71 = sub_C0C3A0(v125, v68, (v70 << 32) | (unsigned int)v69);
      v72 = v46;
      v73 = (__int64)v67;
      ++v46;
      v74 = v71;
      v75 = v67->m128i_i64[0];
      v67 += 2;
      *(_DWORD *)(v75 + 16) = v72;
      sub_124D200((__int64 *)a2, (__int64)&v148, v74, v73);
    }
    while ( v132 != v67 );
  }
  v76 = *(_QWORD **)(a1 + 8);
  v77 = (*(__int64 (__fastcall **)(_QWORD *))(*v76 + 80LL))(v76);
  v78 = v150;
  result = v77 + v76[4] - v76[2];
  *(_QWORD *)(v110 + 184) = v112;
  *(_QWORD *)(v110 + 192) = result;
  v80 = v151;
  if ( v78 != v151 )
  {
    v81 = *(_QWORD **)(a1 + 8);
    v133 = (*(__int64 (__fastcall **)(_QWORD *))(*v81 + 80LL))(v81) + v81[4] - v81[2];
    v82 = v78;
    v83 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8LL * (unsigned int)(v113 - 1));
    do
    {
      v84 = *v82;
      v85 = *(_QWORD *)(a1 + 8);
      if ( *(_DWORD *)(a1 + 16) != 1 )
        v84 = _byteswap_ulong(v84);
      ++v82;
      *(_DWORD *)v144 = v84;
      sub_CB6200(v85, v144, 4u);
    }
    while ( v80 != v82 );
    v86 = *(_QWORD **)(a1 + 8);
    v87 = (*(__int64 (__fastcall **)(_QWORD *))(*v86 + 80LL))(v86);
    v88 = v86[4] - v86[2];
    *(_QWORD *)(v83 + 184) = v133;
    result = v87 + v88;
    *(_QWORD *)(v83 + 192) = result;
  }
  if ( v137 )
    result = j_j___libc_free_0(v137, (char *)v139 - (char *)v137);
  if ( v134 )
    result = j_j___libc_free_0(v134, (char *)v136 - (char *)v134);
  if ( v150 )
    return j_j___libc_free_0(v150, v152 - (_QWORD)v150);
  return result;
}
