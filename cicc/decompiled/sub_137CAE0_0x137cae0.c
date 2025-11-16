// Function: sub_137CAE0
// Address: 0x137cae0
//
__int64 __fastcall sub_137CAE0(__int64 a1, __int64 *a2, __int64 a3, _QWORD *a4)
{
  __int64 v4; // r14
  __int64 v5; // rsi
  char *v6; // rbx
  int v7; // r15d
  char *v8; // r14
  unsigned int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r12
  unsigned int v13; // edx
  int v14; // ecx
  __int64 v15; // r8
  __int64 v16; // rbx
  __int64 v17; // rax
  const __m128i *v18; // rax
  size_t v19; // rax
  __int8 *v20; // rax
  const __m128i *v21; // rax
  size_t v22; // rax
  __int8 *v23; // rax
  __int64 v24; // rax
  const __m128i *v25; // rax
  unsigned __int64 v26; // rax
  const __m128i *v27; // rsi
  char *v28; // rdi
  __int64 v29; // rdx
  const __m128i *v30; // rcx
  const __m128i *v31; // r8
  unsigned __int64 v32; // r15
  __int64 v33; // rax
  __m128i *v34; // rdi
  __m128i *v35; // rdx
  const __m128i *v36; // rax
  const __m128i *v37; // r8
  unsigned __int64 v38; // r12
  __int64 v39; // rax
  __m128i *v40; // rcx
  const __m128i *v41; // rdx
  __int64 v42; // rdi
  size_t v43; // rdx
  const __m128i *v44; // rcx
  __int64 v45; // r12
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rsi
  void *v49; // rdi
  unsigned int v50; // eax
  __int64 v51; // rdx
  __int64 *v52; // rdi
  unsigned int v53; // eax
  __int64 v54; // rdx
  __int64 *v55; // rax
  __int64 v56; // rdx
  __int64 v57; // r8
  __int64 v58; // rbx
  __int64 v59; // r12
  __int64 v60; // rdi
  int v62; // r10d
  __int64 *v63; // r9
  __int64 *v64; // r8
  unsigned int v65; // r13d
  int v66; // r9d
  __int64 v67; // rsi
  __int64 v68; // rax
  int v69; // r10d
  __int64 *v70; // r9
  __int64 v74; // [rsp+30h] [rbp-370h] BYREF
  __int64 v75; // [rsp+38h] [rbp-368h]
  __int64 v76; // [rsp+40h] [rbp-360h]
  unsigned int v77; // [rsp+48h] [rbp-358h]
  __int64 v78; // [rsp+50h] [rbp-350h]
  __int64 v79; // [rsp+58h] [rbp-348h]
  __int64 v80; // [rsp+60h] [rbp-340h]
  _QWORD v81[16]; // [rsp+70h] [rbp-330h] BYREF
  __int64 v82; // [rsp+F0h] [rbp-2B0h] BYREF
  _QWORD *v83; // [rsp+F8h] [rbp-2A8h]
  _QWORD *v84; // [rsp+100h] [rbp-2A0h]
  __int64 v85; // [rsp+108h] [rbp-298h]
  int v86; // [rsp+110h] [rbp-290h]
  _QWORD v87[8]; // [rsp+118h] [rbp-288h] BYREF
  const __m128i *v88; // [rsp+158h] [rbp-248h] BYREF
  size_t v89; // [rsp+160h] [rbp-240h]
  char *v90; // [rsp+168h] [rbp-238h]
  char v91[8]; // [rsp+170h] [rbp-230h] BYREF
  __int64 v92; // [rsp+178h] [rbp-228h]
  unsigned __int64 v93; // [rsp+180h] [rbp-220h]
  _BYTE v94[64]; // [rsp+198h] [rbp-208h] BYREF
  const __m128i *v95; // [rsp+1D8h] [rbp-1C8h]
  size_t v96; // [rsp+1E0h] [rbp-1C0h]
  __int8 *v97; // [rsp+1E8h] [rbp-1B8h]
  char v98[8]; // [rsp+1F0h] [rbp-1B0h] BYREF
  __int64 v99; // [rsp+1F8h] [rbp-1A8h]
  unsigned __int64 v100; // [rsp+200h] [rbp-1A0h]
  _BYTE v101[64]; // [rsp+218h] [rbp-188h] BYREF
  __int64 v102; // [rsp+258h] [rbp-148h]
  __int64 v103; // [rsp+260h] [rbp-140h]
  unsigned __int64 v104; // [rsp+268h] [rbp-138h]
  __m128i v105; // [rsp+270h] [rbp-130h] BYREF
  unsigned __int64 v106; // [rsp+280h] [rbp-120h]
  __int64 v107; // [rsp+288h] [rbp-118h]
  __int64 v108; // [rsp+290h] [rbp-110h]
  _QWORD v109[2]; // [rsp+298h] [rbp-108h] BYREF
  __int64 v110; // [rsp+2A8h] [rbp-F8h]
  char *v111; // [rsp+2B0h] [rbp-F0h]
  char *v112; // [rsp+2B8h] [rbp-E8h]
  __int64 v113; // [rsp+2C0h] [rbp-E0h]
  __int64 v114; // [rsp+2C8h] [rbp-D8h]
  __int64 v115; // [rsp+2D0h] [rbp-D0h]
  const __m128i *v116; // [rsp+2D8h] [rbp-C8h]
  const __m128i *v117; // [rsp+2E0h] [rbp-C0h]
  __int8 *v118; // [rsp+2E8h] [rbp-B8h]
  char v119[8]; // [rsp+2F0h] [rbp-B0h] BYREF
  __int64 v120; // [rsp+2F8h] [rbp-A8h]
  unsigned __int64 v121; // [rsp+300h] [rbp-A0h]
  char v122[64]; // [rsp+318h] [rbp-88h] BYREF
  const __m128i *v123; // [rsp+358h] [rbp-48h]
  const __m128i *v124; // [rsp+360h] [rbp-40h]
  unsigned __int64 v125; // [rsp+368h] [rbp-38h]

  v4 = a1;
  *(_QWORD *)(a1 + 64) = a2;
  v5 = a2[10];
  v74 = 0;
  if ( v5 )
    v5 -= 24;
  v75 = 0;
  v76 = 0;
  v77 = 0;
  v78 = 0;
  v79 = 0;
  v80 = 0;
  v105 = 0u;
  v106 = 0;
  v107 = 0;
  v108 = 0;
  v109[0] = 0;
  v109[1] = 0;
  v110 = 0;
  v111 = 0;
  v112 = 0;
  v113 = 0;
  v114 = 0;
  v115 = 0;
  v116 = 0;
  sub_137C180((__int64)&v105, v5);
  sub_137C590((__int64)&v105);
  v6 = v111;
  if ( v112 != v111 )
  {
    v7 = 0;
    v8 = v112;
    while ( v8 - v6 == 8 )
    {
LABEL_16:
      ++v7;
      sub_137C590((__int64)&v105);
      v8 = v112;
      v6 = v111;
      if ( v112 == v111 )
      {
        v4 = a1;
        goto LABEL_18;
      }
    }
    while ( 1 )
    {
      v12 = *(_QWORD *)v6;
      if ( !v77 )
        break;
      v9 = (v77 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v10 = (__int64 *)(v75 + 16LL * v9);
      v11 = *v10;
      if ( v12 == *v10 )
      {
LABEL_8:
        v6 += 8;
        *((_DWORD *)v10 + 2) = v7;
        if ( v8 == v6 )
          goto LABEL_16;
      }
      else
      {
        v62 = 1;
        v63 = 0;
        while ( v11 != -8 )
        {
          if ( v11 == -16 && !v63 )
            v63 = v10;
          v9 = (v77 - 1) & (v62 + v9);
          v10 = (__int64 *)(v75 + 16LL * v9);
          v11 = *v10;
          if ( v12 == *v10 )
            goto LABEL_8;
          ++v62;
        }
        if ( v63 )
          v10 = v63;
        ++v74;
        v14 = v76 + 1;
        if ( 4 * ((int)v76 + 1) < 3 * v77 )
        {
          if ( v77 - HIDWORD(v76) - v14 <= v77 >> 3 )
          {
            sub_137BC70((__int64)&v74, v77);
            if ( !v77 )
            {
LABEL_157:
              LODWORD(v76) = v76 + 1;
              BUG();
            }
            v64 = 0;
            v65 = (v77 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
            v66 = 1;
            v14 = v76 + 1;
            v10 = (__int64 *)(v75 + 16LL * v65);
            v67 = *v10;
            if ( v12 != *v10 )
            {
              while ( v67 != -8 )
              {
                if ( !v64 && v67 == -16 )
                  v64 = v10;
                v65 = (v77 - 1) & (v66 + v65);
                v10 = (__int64 *)(v75 + 16LL * v65);
                v67 = *v10;
                if ( v12 == *v10 )
                  goto LABEL_13;
                ++v66;
              }
              if ( v64 )
                v10 = v64;
            }
          }
          goto LABEL_13;
        }
LABEL_11:
        sub_137BC70((__int64)&v74, 2 * v77);
        if ( !v77 )
          goto LABEL_157;
        v13 = (v77 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v14 = v76 + 1;
        v10 = (__int64 *)(v75 + 16LL * v13);
        v15 = *v10;
        if ( v12 != *v10 )
        {
          v69 = 1;
          v70 = 0;
          while ( v15 != -8 )
          {
            if ( v15 == -16 && !v70 )
              v70 = v10;
            v13 = (v77 - 1) & (v69 + v13);
            v10 = (__int64 *)(v75 + 16LL * v13);
            v15 = *v10;
            if ( v12 == *v10 )
              goto LABEL_13;
            ++v69;
          }
          if ( v70 )
            v10 = v70;
        }
LABEL_13:
        LODWORD(v76) = v14;
        if ( *v10 != -8 )
          --HIDWORD(v76);
        v6 += 8;
        *((_DWORD *)v10 + 2) = 0;
        *v10 = v12;
        *((_DWORD *)v10 + 2) = v7;
        if ( v8 == v6 )
          goto LABEL_16;
      }
    }
    ++v74;
    goto LABEL_11;
  }
LABEL_18:
  if ( v114 )
    j_j___libc_free_0(v114, (char *)v116 - v114);
  if ( v111 )
    j_j___libc_free_0(v111, v113 - (_QWORD)v111);
  if ( v109[0] )
    j_j___libc_free_0(v109[0], v110 - v109[0]);
  j___libc_free_0(v106);
  v88 = 0;
  v16 = a2[10];
  v89 = 0;
  v90 = 0;
  v86 = 0;
  if ( v16 )
    v16 -= 24;
  memset(v81, 0, sizeof(v81));
  LODWORD(v81[3]) = 8;
  v81[1] = &v81[5];
  v81[2] = &v81[5];
  v83 = v87;
  v84 = v87;
  v87[0] = v16;
  v85 = 0x100000008LL;
  v82 = 1;
  v105.m128i_i64[1] = sub_157EBA0(v16);
  v105.m128i_i64[0] = v16;
  LODWORD(v106) = 0;
  sub_136D560(&v88, 0, &v105);
  sub_136D710((__int64)&v82);
  sub_16CCEE0(v98, v101, 8, v81);
  v17 = v81[13];
  memset(&v81[13], 0, 24);
  v102 = v17;
  v103 = v81[14];
  v104 = v81[15];
  sub_16CCEE0(v91, v94, 8, &v82);
  v18 = v88;
  v88 = 0;
  v95 = v18;
  v19 = v89;
  v89 = 0;
  v96 = v19;
  v20 = v90;
  v90 = 0;
  v97 = v20;
  sub_16CCEE0(&v105, v109, 8, v91);
  v21 = v95;
  v95 = 0;
  v116 = v21;
  v22 = v96;
  v96 = 0;
  v117 = (const __m128i *)v22;
  v23 = v97;
  v97 = 0;
  v118 = v23;
  sub_16CCEE0(v119, v122, 8, v98);
  v24 = v102;
  v102 = 0;
  v123 = (const __m128i *)v24;
  v25 = (const __m128i *)v103;
  v103 = 0;
  v124 = v25;
  v26 = v104;
  v104 = 0;
  v125 = v26;
  if ( v95 )
    j_j___libc_free_0(v95, v97 - (__int8 *)v95);
  if ( v93 != v92 )
    _libc_free(v93);
  if ( v102 )
    j_j___libc_free_0(v102, v104 - v102);
  if ( v100 != v99 )
    _libc_free(v100);
  if ( v88 )
    j_j___libc_free_0(v88, v90 - (char *)v88);
  if ( v84 != v83 )
    _libc_free((unsigned __int64)v84);
  if ( v81[13] )
    j_j___libc_free_0(v81[13], v81[15] - v81[13]);
  if ( v81[2] != v81[1] )
    _libc_free(v81[2]);
  v27 = (const __m128i *)v94;
  v28 = v91;
  sub_16CCCB0(v91, v94, &v105);
  v30 = v117;
  v31 = v116;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  v32 = (char *)v117 - (char *)v116;
  if ( v117 == v116 )
  {
    v34 = 0;
  }
  else
  {
    if ( v32 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_156;
    v33 = sub_22077B0((char *)v117 - (char *)v116);
    v30 = v117;
    v31 = v116;
    v34 = (__m128i *)v33;
  }
  v95 = v34;
  v96 = (size_t)v34;
  v97 = &v34->m128i_i8[v32];
  if ( v30 != v31 )
  {
    v35 = v34;
    v36 = v31;
    do
    {
      if ( v35 )
      {
        *v35 = _mm_loadu_si128(v36);
        v35[1].m128i_i64[0] = v36[1].m128i_i64[0];
      }
      v36 = (const __m128i *)((char *)v36 + 24);
      v35 = (__m128i *)((char *)v35 + 24);
    }
    while ( v36 != v30 );
    v34 = (__m128i *)((char *)v34 + 8 * ((unsigned __int64)((char *)&v36[-2].m128i_u64[1] - (char *)v31) >> 3) + 24);
  }
  v96 = (size_t)v34;
  v28 = v98;
  sub_16CCCB0(v98, v101, v119);
  v27 = v124;
  v37 = v123;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v38 = (char *)v124 - (char *)v123;
  if ( v124 != v123 )
  {
    if ( v38 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v39 = sub_22077B0((char *)v124 - (char *)v123);
      v27 = v124;
      v37 = v123;
      goto LABEL_54;
    }
LABEL_156:
    sub_4261EA(v28, v27, v29);
  }
  v39 = 0;
LABEL_54:
  v102 = v39;
  v103 = v39;
  v104 = v39 + v38;
  if ( v37 == v27 )
  {
    v42 = v39;
  }
  else
  {
    v40 = (__m128i *)v39;
    v41 = v37;
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
    while ( v41 != v27 );
    v42 = v102;
    v39 += 8 * ((unsigned __int64)((char *)&v41[-2].m128i_u64[1] - (char *)v37) >> 3) + 24;
  }
  v103 = v39;
LABEL_61:
  v43 = v96;
  v44 = v95;
  if ( v96 - (_QWORD)v95 == v39 - v42 )
    goto LABEL_70;
  while ( 1 )
  {
    do
    {
      v45 = *(_QWORD *)(v43 - 24);
      sub_1378050(v4, v45);
      sub_1378260(v4, v45);
      v46 = sub_157EBA0(v45);
      if ( (unsigned int)sub_15F4D60(v46) > 1
        && !(unsigned __int8)sub_1379A50(v4, v45)
        && !(unsigned __int8)sub_137A790(v4, v45)
        && !(unsigned __int8)sub_13797A0(v4, v45)
        && !(unsigned __int8)sub_137A080(v4, v45)
        && !(unsigned __int8)sub_137A800(v4, v45, a3, (__int64)&v74)
        && !(unsigned __int8)sub_137A320(v4, v45)
        && !(unsigned __int8)sub_137A3F0(v4, v45, a4) )
      {
        sub_137A690(v4, v45);
      }
      v96 -= 24LL;
      v44 = v95;
      v43 = v96;
      if ( (const __m128i *)v96 != v95 )
      {
        sub_136D710((__int64)v91);
        v42 = v102;
        v39 = v103;
        goto LABEL_61;
      }
      v42 = v102;
    }
    while ( v96 - (_QWORD)v95 != v103 - v102 );
LABEL_70:
    if ( v44 == (const __m128i *)v43 )
      break;
    v47 = v42;
    while ( v44->m128i_i64[0] == *(_QWORD *)v47 && v44[1].m128i_i32[0] == *(_DWORD *)(v47 + 16) )
    {
      v44 = (const __m128i *)((char *)v44 + 24);
      v47 += 24;
      if ( (const __m128i *)v43 == v44 )
        goto LABEL_75;
    }
  }
LABEL_75:
  v48 = v104 - v42;
  if ( v42 )
    j_j___libc_free_0(v42, v48);
  if ( v100 != v99 )
    _libc_free(v100);
  if ( v95 )
  {
    v48 = v97 - (__int8 *)v95;
    j_j___libc_free_0(v95, v97 - (__int8 *)v95);
  }
  if ( v93 != v92 )
    _libc_free(v93);
  if ( v123 )
  {
    v48 = v125 - (_QWORD)v123;
    j_j___libc_free_0(v123, v125 - (_QWORD)v123);
  }
  if ( v121 != v120 )
    _libc_free(v121);
  if ( v116 )
  {
    v48 = v118 - (__int8 *)v116;
    j_j___libc_free_0(v116, v118 - (__int8 *)v116);
  }
  if ( v106 != v105.m128i_i64[1] )
    _libc_free(v106);
  ++*(_QWORD *)(v4 + 72);
  v49 = *(void **)(v4 + 88);
  if ( v49 == *(void **)(v4 + 80) )
  {
LABEL_96:
    *(_QWORD *)(v4 + 100) = 0;
  }
  else
  {
    v50 = 4 * (*(_DWORD *)(v4 + 100) - *(_DWORD *)(v4 + 104));
    v51 = *(unsigned int *)(v4 + 96);
    if ( v50 < 0x20 )
      v50 = 32;
    if ( v50 >= (unsigned int)v51 )
    {
      v48 = 0xFFFFFFFFLL;
      memset(v49, -1, 8 * v51);
      goto LABEL_96;
    }
    sub_16CC920(v4 + 72);
  }
  ++*(_QWORD *)(v4 + 240);
  v52 = *(__int64 **)(v4 + 256);
  if ( v52 == *(__int64 **)(v4 + 248) )
  {
LABEL_102:
    *(_QWORD *)(v4 + 268) = 0;
  }
  else
  {
    v53 = 4 * (*(_DWORD *)(v4 + 268) - *(_DWORD *)(v4 + 272));
    v54 = *(unsigned int *)(v4 + 264);
    if ( v53 < 0x20 )
      v53 = 32;
    if ( v53 >= (unsigned int)v54 )
    {
      v48 = 0xFFFFFFFFLL;
      memset(v52, -1, 8 * v54);
      goto LABEL_102;
    }
    v52 = (__int64 *)(v4 + 240);
    sub_16CC920(v4 + 240);
  }
  if ( byte_4F98900 )
  {
    if ( !qword_4F987E8
      || (v52 = a2, v55 = (__int64 *)sub_1649960(a2), v57 = v56, v43 = qword_4F987E8, qword_4F987E8 == v57)
      && (!qword_4F987E8 || (v48 = (__int64)qword_4F987E0, v52 = v55, !memcmp(v55, qword_4F987E0, qword_4F987E8))) )
    {
      v68 = sub_16BA580(v52, v48, v43);
      sub_13779F0(v4, v68);
    }
  }
  v58 = v79;
  v59 = v78;
  if ( v79 != v78 )
  {
    do
    {
      v60 = *(_QWORD *)(v59 + 8);
      v59 += 32;
      j___libc_free_0(v60);
    }
    while ( v58 != v59 );
    v59 = v78;
  }
  if ( v59 )
    j_j___libc_free_0(v59, v80 - v59);
  return j___libc_free_0(v75);
}
