// Function: sub_25F27C0
// Address: 0x25f27c0
//
unsigned __int64 *__fastcall sub_25F27C0(
        unsigned __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int64 v8; // rbx
  bool v9; // al
  _BOOL4 v10; // esi
  bool v11; // zf
  __int64 v12; // rax
  _BYTE *v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r9
  __int64 v18; // rcx
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r8
  __m128i *v24; // rdx
  const __m128i *v25; // rax
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // r9
  const __m128i *v30; // rcx
  unsigned __int64 v31; // r13
  __int64 v32; // rax
  __int64 v33; // r8
  __m128i *v34; // rdx
  const __m128i *v35; // rax
  __int64 v36; // rcx
  unsigned __int64 v37; // r13
  __int64 v38; // rax
  unsigned __int64 v39; // r8
  __m128i *v40; // rdx
  const __m128i *v41; // rax
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // rdx
  unsigned __int64 v44; // rsi
  __int64 v45; // r15
  unsigned __int64 v46; // rsi
  unsigned __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  char v50; // al
  char v51; // cl
  __int64 *v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  __int64 v56; // rbx
  __int64 v57; // rcx
  __int64 v58; // r8
  __int64 v59; // r9
  __int64 v60; // rcx
  unsigned __int64 v61; // r13
  __m128i *v62; // rax
  const __m128i *v63; // rdx
  __m128i *v64; // rcx
  unsigned __int64 v65; // rdi
  __m128i *v66; // rax
  __int64 v67; // r15
  char v68; // r14
  __int64 *v69; // rax
  __int64 *v70; // rdx
  __int64 v71; // rax
  unsigned __int64 v72; // r13
  bool v73; // al
  __int64 *v74; // rdx
  __int64 v75; // rcx
  unsigned __int64 v76; // r13
  __int64 *v77; // rax
  unsigned __int64 v78; // rax
  int v79; // edx
  __int64 v80; // rax
  unsigned __int64 v82; // r13
  __int64 *v83; // rax
  __int64 *v84; // rax
  unsigned __int64 v85; // rax
  unsigned __int64 v86; // rcx
  int v87; // edx
  __int64 v88; // rax
  __int64 v89; // rax
  bool v90; // al
  bool v91; // al
  __int64 *v92; // rdx
  __int64 v93; // r8
  __int64 v94; // r9
  __int64 v95; // rcx
  __int64 *v96; // rax
  unsigned __int64 v97; // rax
  unsigned __int64 v98; // rsi
  int v99; // edx
  __int64 v100; // rax
  unsigned __int64 v101; // r9
  __int64 *v102; // rax
  __int64 *v103; // rax
  unsigned __int64 v104; // [rsp+8h] [rbp-458h]
  unsigned __int64 v105; // [rsp+10h] [rbp-450h]
  __int64 v106; // [rsp+20h] [rbp-440h]
  unsigned __int64 v109; // [rsp+50h] [rbp-410h]
  unsigned int v110; // [rsp+50h] [rbp-410h]
  _BOOL4 v111; // [rsp+58h] [rbp-408h]
  unsigned int v112; // [rsp+5Ch] [rbp-404h]
  __int64 v113; // [rsp+60h] [rbp-400h]
  __int64 v114; // [rsp+68h] [rbp-3F8h]
  unsigned int v115; // [rsp+68h] [rbp-3F8h]
  unsigned __int64 v116; // [rsp+68h] [rbp-3F8h]
  __int64 v117; // [rsp+70h] [rbp-3F0h] BYREF
  __int64 *v118; // [rsp+78h] [rbp-3E8h]
  __int64 v119; // [rsp+80h] [rbp-3E0h]
  int v120; // [rsp+88h] [rbp-3D8h]
  char v121; // [rsp+8Ch] [rbp-3D4h]
  char v122; // [rsp+90h] [rbp-3D0h] BYREF
  char v123[8]; // [rsp+B0h] [rbp-3B0h] BYREF
  unsigned __int64 v124; // [rsp+B8h] [rbp-3A8h]
  char v125; // [rsp+CCh] [rbp-394h]
  _BYTE v126[64]; // [rsp+D0h] [rbp-390h] BYREF
  unsigned __int64 v127; // [rsp+110h] [rbp-350h]
  unsigned __int64 v128; // [rsp+118h] [rbp-348h]
  unsigned __int64 v129; // [rsp+120h] [rbp-340h]
  char v130[8]; // [rsp+130h] [rbp-330h] BYREF
  unsigned __int64 v131; // [rsp+138h] [rbp-328h]
  char v132; // [rsp+14Ch] [rbp-314h]
  _BYTE v133[64]; // [rsp+150h] [rbp-310h] BYREF
  unsigned __int64 v134; // [rsp+190h] [rbp-2D0h]
  __int64 v135; // [rsp+198h] [rbp-2C8h]
  __m128i *v136; // [rsp+1A0h] [rbp-2C0h]
  char v137[8]; // [rsp+1B0h] [rbp-2B0h] BYREF
  unsigned __int64 v138; // [rsp+1B8h] [rbp-2A8h]
  char v139; // [rsp+1CCh] [rbp-294h]
  _BYTE v140[64]; // [rsp+1D0h] [rbp-290h] BYREF
  unsigned __int64 v141; // [rsp+210h] [rbp-250h]
  unsigned __int64 v142; // [rsp+218h] [rbp-248h]
  unsigned __int64 v143; // [rsp+220h] [rbp-240h]
  char v144[8]; // [rsp+230h] [rbp-230h] BYREF
  unsigned __int64 v145; // [rsp+238h] [rbp-228h]
  char v146; // [rsp+24Ch] [rbp-214h]
  _BYTE v147[64]; // [rsp+250h] [rbp-210h] BYREF
  __m128i *v148; // [rsp+290h] [rbp-1D0h]
  __m128i *v149; // [rsp+298h] [rbp-1C8h]
  __int8 *v150; // [rsp+2A0h] [rbp-1C0h]
  __int64 v151; // [rsp+2B0h] [rbp-1B0h] BYREF
  __int64 *v152; // [rsp+2B8h] [rbp-1A8h]
  __int64 v153; // [rsp+2C0h] [rbp-1A0h]
  int v154; // [rsp+2C8h] [rbp-198h]
  char v155; // [rsp+2CCh] [rbp-194h]
  __int64 v156; // [rsp+2D0h] [rbp-190h] BYREF
  unsigned __int64 v157; // [rsp+310h] [rbp-150h] BYREF
  __int64 v158; // [rsp+318h] [rbp-148h]
  __int64 v159; // [rsp+320h] [rbp-140h]
  __m128i v160[8]; // [rsp+330h] [rbp-130h] BYREF
  __m128i v161[11]; // [rsp+3B0h] [rbp-B0h] BYREF

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  v118 = (__int64 *)&v122;
  v117 = 0;
  v119 = 4;
  v120 = 0;
  v121 = 1;
  sub_25EFFB0(a1, 0, a3, a4, a5, a6);
  v8 = a1[1];
  v106 = v8 - 32;
  v113 = v8 - 32;
  v9 = sub_25F0310(a2);
  v10 = v9;
  v11 = !v9;
  v12 = 0;
  if ( !v11 )
    v12 = a2;
  v111 = v10;
  *(_QWORD *)(v8 - 16) = v12;
  v161[0].m128i_i64[1] = (__int64)v161[2].m128i_i64;
  v161[1].m128i_i64[0] = 0x100000008LL;
  memset(&v161[6], 0, 24);
  v161[1].m128i_i32[2] = 0;
  v161[1].m128i_i8[12] = 1;
  v161[2].m128i_i64[0] = a2;
  v161[0].m128i_i64[0] = 1;
  v160[0].m128i_i64[0] = a2;
  v160[1].m128i_i8[0] = 0;
  sub_25F2740((unsigned __int64 *)&v161[6], v160);
  v13 = v133;
  sub_C8CD80((__int64)v130, (__int64)v133, (__int64)v161, v14, v15, v16);
  v18 = v161[6].m128i_i64[1];
  v19 = v161[6].m128i_u64[0];
  v134 = 0;
  v135 = 0;
  v136 = 0;
  v20 = v161[6].m128i_i64[1] - v161[6].m128i_i64[0];
  if ( v161[6].m128i_i64[1] == v161[6].m128i_i64[0] )
  {
    v22 = 0;
    v23 = 0;
  }
  else
  {
    if ( v20 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_167;
    v114 = v161[6].m128i_i64[1] - v161[6].m128i_i64[0];
    v21 = sub_22077B0(v161[6].m128i_i64[1] - v161[6].m128i_i64[0]);
    v18 = v161[6].m128i_i64[1];
    v19 = v161[6].m128i_u64[0];
    v22 = v114;
    v23 = v21;
  }
  v24 = (__m128i *)(v23 + v22);
  v134 = v23;
  v135 = v23;
  v136 = v24;
  if ( v18 != v19 )
  {
    v24 = (__m128i *)v23;
    v25 = (const __m128i *)v19;
    do
    {
      if ( v24 )
      {
        *v24 = _mm_loadu_si128(v25);
        v13 = (_BYTE *)v25[1].m128i_i64[0];
        v24[1].m128i_i64[0] = (__int64)v13;
      }
      v25 = (const __m128i *)((char *)v25 + 24);
      v24 = (__m128i *)((char *)v24 + 24);
    }
    while ( v25 != (const __m128i *)v18 );
    v23 += 8 * (((unsigned __int64)&v25[-2].m128i_u64[1] - v19) >> 3) + 24;
  }
  v135 = v23;
  if ( v19 )
  {
    v13 = (_BYTE *)(v161[7].m128i_i64[0] - v19);
    j_j___libc_free_0(v19);
  }
  if ( !v161[1].m128i_i8[12] )
    _libc_free(v161[0].m128i_u64[1]);
  sub_24FFDC0((__int64)v130, (__int64)v13, (__int64)v24, v18, v23, v17);
  v13 = v126;
  sub_C8CD80((__int64)v123, (__int64)v126, (__int64)v130, v26, v27, v28);
  v30 = (const __m128i *)v135;
  v19 = v134;
  v127 = 0;
  v128 = 0;
  v129 = 0;
  v31 = v135 - v134;
  if ( v135 == v134 )
  {
    v33 = 0;
  }
  else
  {
    if ( v31 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_167;
    v32 = sub_22077B0(v135 - v134);
    v30 = (const __m128i *)v135;
    v19 = v134;
    v33 = v32;
  }
  v127 = v33;
  v128 = v33;
  v129 = v33 + v31;
  if ( (const __m128i *)v19 != v30 )
  {
    v34 = (__m128i *)v33;
    v35 = (const __m128i *)v19;
    do
    {
      if ( v34 )
      {
        *v34 = _mm_loadu_si128(v35);
        v34[1].m128i_i64[0] = v35[1].m128i_i64[0];
      }
      v35 = (const __m128i *)((char *)v35 + 24);
      v34 = (__m128i *)((char *)v34 + 24);
    }
    while ( v35 != v30 );
    v33 += 8 * (((unsigned __int64)&v35[-2].m128i_u64[1] - v19) >> 3) + 24;
  }
  v128 = v33;
  if ( v19 )
    j_j___libc_free_0(v19);
  if ( !v132 )
    _libc_free(v131);
  memset(v160, 0, 0x78u);
  v160[1].m128i_i32[0] = 8;
  v13 = v140;
  v160[0].m128i_i64[1] = (__int64)v160[2].m128i_i64;
  v160[1].m128i_i8[12] = 1;
  sub_C8CD80((__int64)v137, (__int64)v140, (__int64)v160, 0, v33, v29);
  v36 = v160[6].m128i_i64[1];
  v19 = v160[6].m128i_u64[0];
  v141 = 0;
  v142 = 0;
  v143 = 0;
  v37 = v160[6].m128i_i64[1] - v160[6].m128i_i64[0];
  if ( v160[6].m128i_i64[1] == v160[6].m128i_i64[0] )
  {
    v39 = 0;
  }
  else
  {
    if ( v37 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_167;
    v38 = sub_22077B0(v160[6].m128i_i64[1] - v160[6].m128i_i64[0]);
    v36 = v160[6].m128i_i64[1];
    v19 = v160[6].m128i_u64[0];
    v39 = v38;
  }
  v141 = v39;
  v142 = v39;
  v143 = v39 + v37;
  if ( v36 != v19 )
  {
    v40 = (__m128i *)v39;
    v41 = (const __m128i *)v19;
    do
    {
      if ( v40 )
      {
        *v40 = _mm_loadu_si128(v41);
        v40[1].m128i_i64[0] = v41[1].m128i_i64[0];
      }
      v41 = (const __m128i *)((char *)v41 + 24);
      v40 = (__m128i *)((char *)v40 + 24);
    }
    while ( (const __m128i *)v36 != v41 );
    v39 += 8 * ((v36 - 24 - v19) >> 3) + 24;
  }
  v142 = v39;
  if ( v19 )
    j_j___libc_free_0(v19);
  if ( !v160[1].m128i_i8[12] )
    _libc_free(v160[0].m128i_u64[1]);
  v42 = v127;
  v43 = v128;
  v112 = v111;
  while ( 1 )
  {
    v44 = v141;
    if ( v43 - v42 != v142 - v141 )
      goto LABEL_44;
    if ( v43 == v42 )
      break;
    while ( *(_QWORD *)v42 == *(_QWORD *)v44 )
    {
      v51 = *(_BYTE *)(v42 + 16);
      if ( v51 != *(_BYTE *)(v44 + 16) || v51 && *(_QWORD *)(v42 + 8) != *(_QWORD *)(v44 + 8) )
        break;
      v42 += 24LL;
      v44 += 24LL;
      if ( v42 == v43 )
        goto LABEL_52;
    }
LABEL_44:
    v45 = *(_QWORD *)(v43 - 24);
    v46 = a2;
    sub_B19AA0(a4, a2, v45);
    if ( !v50 )
      goto LABEL_45;
    v71 = *(_QWORD *)(v45 + 16);
    if ( !v71 )
      goto LABEL_99;
    while ( (unsigned __int8)(**(_BYTE **)(v71 + 24) - 30) > 0xAu )
    {
      v71 = *(_QWORD *)(v71 + 8);
      if ( !v71 )
        goto LABEL_99;
    }
    if ( sub_25F0310(v45) )
    {
      v72 = v128;
      v109 = v127;
      v73 = sub_25F0310(v45);
      v48 = 0;
      if ( v73 )
      {
        v76 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v72 - v109) >> 3);
        v48 = (unsigned int)v76;
        if ( v112 < (unsigned int)v76 )
        {
          *(_QWORD *)(v8 - 16) = v45;
          v112 = v76;
        }
      }
      if ( v121 )
      {
        v77 = v118;
        v75 = HIDWORD(v119);
        v74 = &v118[HIDWORD(v119)];
        if ( v118 != v74 )
        {
          while ( v45 != *v77 )
          {
            if ( v74 == ++v77 )
              goto LABEL_115;
          }
LABEL_94:
          v78 = *(unsigned int *)(v8 - 24);
          v47 = *(unsigned int *)(v8 - 20);
          v79 = *(_DWORD *)(v8 - 24);
          if ( v78 < v47 )
            goto LABEL_95;
          goto LABEL_112;
        }
LABEL_115:
        if ( HIDWORD(v119) < (unsigned int)v119 )
        {
          ++HIDWORD(v119);
          *v74 = v45;
          ++v117;
          goto LABEL_94;
        }
      }
      v46 = v45;
      v110 = v48;
      sub_C8CC70((__int64)&v117, v45, (__int64)v74, v75, v48, v49);
      v78 = *(unsigned int *)(v8 - 24);
      v47 = *(unsigned int *)(v8 - 20);
      v48 = v110;
      v79 = *(_DWORD *)(v8 - 24);
      if ( v78 < v47 )
      {
LABEL_95:
        v80 = *(_QWORD *)(v8 - 32) + 16 * v78;
        if ( v80 )
        {
          *(_QWORD *)v80 = v45;
          *(_DWORD *)(v80 + 8) = v48;
          v79 = *(_DWORD *)(v8 - 24);
        }
        v43 = (unsigned int)(v79 + 1);
        *(_DWORD *)(v8 - 24) = v43;
        goto LABEL_46;
      }
LABEL_112:
      v46 = (unsigned int)v48 | v105 & 0xFFFFFFFF00000000LL;
      v43 = v78 + 1;
      v105 = v46;
      v82 = v46;
      if ( v47 < v78 + 1 )
      {
        v46 = v8 - 16;
        sub_C8D5F0(v106, (const void *)(v8 - 16), v43, 0x10u, v48, v49);
        v78 = *(unsigned int *)(v8 - 24);
      }
      v83 = (__int64 *)(*(_QWORD *)(v8 - 32) + 16 * v78);
      *v83 = v45;
      v83[1] = v82;
      ++*(_DWORD *)(v8 - 24);
LABEL_46:
      sub_24FFDC0((__int64)v123, v46, v43, v47, v48, v49);
      v42 = v127;
      v43 = v128;
    }
    else
    {
LABEL_45:
      v42 = v127;
      v128 -= 24LL;
      v43 = v127;
      if ( v128 != v127 )
        goto LABEL_46;
    }
  }
LABEL_52:
  if ( !sub_25F0310(a2) )
  {
    v56 = a1[1];
    v113 = v56;
    if ( v56 == a1[2] )
    {
      sub_25EFFB0(a1, v56, (__int64)v52, (__int64)a1, v54, v55);
      v113 = a1[1] - 32;
    }
    else
    {
      if ( v56 )
      {
        *(_QWORD *)(v56 + 8) = 0;
        *(_QWORD *)v56 = v56 + 16;
        *(_QWORD *)(v56 + 16) = 0;
        *(_BYTE *)(v56 + 24) = 0;
        v113 = a1[1];
      }
      a1[1] = v113 + 32;
    }
    v112 = 0;
    goto LABEL_58;
  }
  if ( !v121 )
    goto LABEL_156;
  v84 = v118;
  v53 = HIDWORD(v119);
  v52 = &v118[HIDWORD(v119)];
  if ( v118 == v52 )
  {
LABEL_160:
    if ( HIDWORD(v119) < (unsigned int)v119 )
    {
      ++HIDWORD(v119);
      *v52 = a2;
      ++v117;
      goto LABEL_124;
    }
LABEL_156:
    sub_C8CC70((__int64)&v117, a2, (__int64)v52, v53, v54, v55);
    goto LABEL_124;
  }
  while ( a2 != *v84 )
  {
    if ( v52 == ++v84 )
      goto LABEL_160;
  }
LABEL_124:
  v85 = *(unsigned int *)(v8 - 24);
  v86 = *(unsigned int *)(v8 - 20);
  v87 = *(_DWORD *)(v8 - 24);
  if ( v85 >= v86 )
  {
    if ( v86 < v85 + 1 )
    {
      sub_C8D5F0(v106, (const void *)(v8 - 16), v85 + 1, 0x10u, v54, v55);
      v85 = *(unsigned int *)(v8 - 24);
    }
    v103 = (__int64 *)(*(_QWORD *)(v8 - 32) + 16 * v85);
    *v103 = a2;
    v103[1] = v111;
    ++*(_DWORD *)(v8 - 24);
  }
  else
  {
    v88 = *(_QWORD *)(v8 - 32) + 16 * v85;
    if ( v88 )
    {
      *(_QWORD *)v88 = a2;
      *(_DWORD *)(v88 + 8) = v111;
      v87 = *(_DWORD *)(v8 - 24);
    }
    *(_DWORD *)(v8 - 24) = v87 + 1;
  }
  v89 = *(_QWORD *)(a2 + 16);
  if ( !v89 )
  {
LABEL_99:
    *(_BYTE *)(v8 - 8) = 1;
    goto LABEL_100;
  }
  while ( (unsigned __int8)(**(_BYTE **)(v89 + 24) - 30) > 0xAu )
  {
    v89 = *(_QWORD *)(v89 + 8);
    if ( !v89 )
      goto LABEL_99;
  }
LABEL_58:
  v157 = 0;
  v152 = &v156;
  v153 = 0x100000008LL;
  v158 = 0;
  v159 = 0;
  v154 = 0;
  v155 = 1;
  v156 = a2;
  v151 = 1;
  v161[0].m128i_i64[0] = a2;
  v161[1].m128i_i8[8] = 0;
  sub_25F2780((__int64)&v157, v161);
  sub_23EC7E0((__int64)&v151);
  v13 = v147;
  sub_C8CD80((__int64)v144, (__int64)v147, (__int64)&v151, v57, v58, v59);
  v60 = v158;
  v19 = v157;
  v148 = 0;
  v149 = 0;
  v150 = 0;
  v61 = v158 - v157;
  if ( v158 == v157 )
  {
    v62 = 0;
    goto LABEL_61;
  }
  if ( v61 > 0x7FFFFFFFFFFFFFE0LL )
LABEL_167:
    sub_4261EA(v19, v13, v20);
  v62 = (__m128i *)sub_22077B0(v158 - v157);
  v60 = v158;
  v19 = v157;
LABEL_61:
  v148 = v62;
  v149 = v62;
  v150 = &v62->m128i_i8[v61];
  if ( v60 == v19 )
  {
    v64 = v62;
  }
  else
  {
    v63 = (const __m128i *)v19;
    v64 = (__m128i *)((char *)v62 + v60 - v19);
    do
    {
      if ( v62 )
      {
        *v62 = _mm_loadu_si128(v63);
        v62[1] = _mm_loadu_si128(v63 + 1);
      }
      v62 += 2;
      v63 += 2;
    }
    while ( v64 != v62 );
  }
  v149 = v64;
  if ( v19 )
    j_j___libc_free_0(v19);
  if ( !v155 )
    _libc_free((unsigned __int64)v152);
  memset(v161, 0, 0x78u);
  v161[1].m128i_i32[0] = 8;
  v161[1].m128i_i8[12] = 1;
  v65 = (unsigned __int64)v149;
  v161[0].m128i_i64[1] = (__int64)v161[2].m128i_i64;
  v66 = v148;
  while ( 2 )
  {
    if ( (__m128i *)v65 != v66 )
    {
      while ( 2 )
      {
        v67 = *(_QWORD *)(v65 - 32);
        v68 = sub_B19720(a3, a2, v67);
        if ( !v121 )
        {
          v90 = sub_C8CA60((__int64)&v117, v67) != 0;
          goto LABEL_133;
        }
        v69 = v118;
        v70 = &v118[HIDWORD(v119)];
        if ( v118 == v70 )
        {
LABEL_132:
          v90 = 0;
LABEL_133:
          if ( v68 == 1 && !v90 && sub_25F0310(v67) )
          {
            v91 = sub_25F0310(v67);
            v95 = 0;
            if ( v91 )
            {
              v95 = 1;
              if ( !v112 )
              {
                v112 = 1;
                *(_QWORD *)(v113 + 16) = v67;
              }
            }
            if ( !v121 )
              goto LABEL_149;
            v96 = v118;
            v92 = &v118[HIDWORD(v119)];
            if ( v118 == v92 )
            {
LABEL_150:
              if ( HIDWORD(v119) < (unsigned int)v119 )
              {
                ++HIDWORD(v119);
                *v92 = v67;
                ++v117;
                goto LABEL_144;
              }
LABEL_149:
              v115 = v95;
              sub_C8CC70((__int64)&v117, v67, (__int64)v92, v95, v93, v94);
              v95 = v115;
            }
            else
            {
              while ( v67 != *v96 )
              {
                if ( v92 == ++v96 )
                  goto LABEL_150;
              }
            }
LABEL_144:
            v97 = *(unsigned int *)(v113 + 8);
            v98 = *(unsigned int *)(v113 + 12);
            v99 = *(_DWORD *)(v113 + 8);
            if ( v97 >= v98 )
            {
              v101 = v95 | v104 & 0xFFFFFFFF00000000LL;
              v104 = v101;
              if ( v98 < v97 + 1 )
              {
                v116 = v101;
                sub_C8D5F0(v113, (const void *)(v113 + 16), v97 + 1, 0x10u, v93, v101);
                v97 = *(unsigned int *)(v113 + 8);
                v101 = v116;
              }
              v102 = (__int64 *)(*(_QWORD *)v113 + 16 * v97);
              *v102 = v67;
              v102[1] = v101;
              ++*(_DWORD *)(v113 + 8);
            }
            else
            {
              v100 = *(_QWORD *)v113 + 16 * v97;
              if ( v100 )
              {
                *(_QWORD *)v100 = v67;
                *(_DWORD *)(v100 + 8) = v95;
                v99 = *(_DWORD *)(v113 + 8);
              }
              *(_DWORD *)(v113 + 8) = v99 + 1;
            }
LABEL_78:
            sub_23EC7E0((__int64)v144);
            v65 = (unsigned __int64)v149;
            if ( v149 == v148 )
              goto LABEL_79;
            continue;
          }
        }
        else
        {
          while ( v67 != *v69 )
          {
            if ( v70 == ++v69 )
              goto LABEL_132;
          }
        }
        break;
      }
      v66 = v148;
      v149 -= 2;
      v65 = (unsigned __int64)v148;
      if ( v149 == v148 )
        continue;
      goto LABEL_78;
    }
    break;
  }
LABEL_79:
  if ( v65 )
    j_j___libc_free_0(v65);
  if ( !v146 )
    _libc_free(v145);
LABEL_100:
  if ( v141 )
    j_j___libc_free_0(v141);
  if ( !v139 )
    _libc_free(v138);
  if ( v127 )
    j_j___libc_free_0(v127);
  if ( !v125 )
    _libc_free(v124);
  if ( !v121 )
    _libc_free((unsigned __int64)v118);
  return a1;
}
