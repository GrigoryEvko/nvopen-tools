// Function: sub_1DE12F0
// Address: 0x1de12f0
//
void __fastcall sub_1DE12F0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v3; // rcx
  __int64 v4; // r15
  __int64 v5; // rcx
  __int64 v6; // rax
  unsigned int v7; // edx
  const void *v8; // r8
  __int64 v9; // r14
  __int64 v10; // r13
  __int64 v11; // rax
  char *v12; // rcx
  signed __int64 v13; // rdx
  _QWORD *v14; // rdi
  const __m128i *v15; // rcx
  const __m128i *v16; // rdx
  _BYTE *v17; // rsi
  __m128i *v18; // rax
  __int64 v19; // rsi
  __m128i *v20; // rcx
  __m128i *v21; // rax
  __m128i *v22; // rax
  __int8 *v23; // rax
  const __m128i *v24; // rcx
  unsigned __int64 v25; // r13
  __m128i *v26; // rax
  __m128i *v27; // rcx
  __m128i *v28; // rax
  __m128i *v29; // rax
  __int8 *v30; // rax
  char *v31; // r15
  char *v32; // rax
  __int64 *v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // rsi
  __int64 *v36; // r12
  __int64 v37; // r8
  unsigned int v38; // edx
  __int64 *v39; // rax
  __int64 v40; // rdi
  unsigned int v41; // esi
  __int64 v42; // r13
  int v43; // esi
  int v44; // esi
  __int64 v45; // r9
  int v46; // edx
  unsigned int v47; // ecx
  __int64 v48; // r8
  __int64 v49; // rdx
  char *v50; // rdi
  const __m128i **v51; // r12
  unsigned __int64 v52; // rcx
  const __m128i *v53; // rsi
  const __m128i *v54; // rax
  const __m128i *v55; // rdi
  __int64 v56; // r8
  __m128i *v57; // r15
  signed __int64 v58; // r13
  __int64 v59; // rax
  __m128i *v60; // rdx
  unsigned __int64 v61; // rax
  __m128i *v62; // rsi
  unsigned __int64 v63; // rsi
  unsigned __int64 v64; // rdx
  __int64 v65; // rdi
  __int64 v66; // rcx
  unsigned __int64 v67; // rax
  unsigned __int64 v68; // r8
  __int64 v69; // rax
  int v70; // r11d
  __int64 *v71; // r10
  int v72; // edi
  int v73; // esi
  int v74; // esi
  __int64 *v75; // r10
  __int64 v76; // r9
  int v77; // r11d
  unsigned int v78; // ecx
  __int64 v79; // r8
  char *v80; // rax
  __int64 v81; // rsi
  int v82; // r11d
  __int64 v83; // [rsp+8h] [rbp-338h]
  __int64 v84; // [rsp+8h] [rbp-338h]
  __int64 v85; // [rsp+8h] [rbp-338h]
  __int64 v86; // [rsp+8h] [rbp-338h]
  const void *v87; // [rsp+8h] [rbp-338h]
  char *v88; // [rsp+8h] [rbp-338h]
  __int64 v89; // [rsp+10h] [rbp-330h] BYREF
  _QWORD *v90; // [rsp+18h] [rbp-328h]
  _QWORD *v91; // [rsp+20h] [rbp-320h]
  __int64 v92; // [rsp+28h] [rbp-318h]
  int v93; // [rsp+30h] [rbp-310h]
  _QWORD v94[8]; // [rsp+38h] [rbp-308h] BYREF
  const __m128i *v95; // [rsp+78h] [rbp-2C8h] BYREF
  const __m128i *v96; // [rsp+80h] [rbp-2C0h]
  __int64 v97; // [rsp+88h] [rbp-2B8h]
  _QWORD v98[16]; // [rsp+90h] [rbp-2B0h] BYREF
  _QWORD v99[2]; // [rsp+110h] [rbp-230h] BYREF
  unsigned __int64 v100; // [rsp+120h] [rbp-220h]
  _BYTE v101[64]; // [rsp+138h] [rbp-208h] BYREF
  __m128i *v102; // [rsp+178h] [rbp-1C8h]
  __m128i *v103; // [rsp+180h] [rbp-1C0h]
  __int8 *v104; // [rsp+188h] [rbp-1B8h]
  _QWORD v105[2]; // [rsp+190h] [rbp-1B0h] BYREF
  unsigned __int64 v106; // [rsp+1A0h] [rbp-1A0h]
  char v107[64]; // [rsp+1B8h] [rbp-188h] BYREF
  __m128i *v108; // [rsp+1F8h] [rbp-148h]
  __m128i *v109; // [rsp+200h] [rbp-140h]
  __int8 *v110; // [rsp+208h] [rbp-138h]
  _QWORD v111[2]; // [rsp+210h] [rbp-130h] BYREF
  unsigned __int64 v112; // [rsp+220h] [rbp-120h]
  char v113[64]; // [rsp+238h] [rbp-108h] BYREF
  __m128i *v114; // [rsp+278h] [rbp-C8h]
  __m128i *v115; // [rsp+280h] [rbp-C0h]
  __int8 *v116; // [rsp+288h] [rbp-B8h]
  __m128i v117; // [rsp+290h] [rbp-B0h] BYREF
  unsigned __int64 v118; // [rsp+2A0h] [rbp-A0h]
  char v119[64]; // [rsp+2B8h] [rbp-88h] BYREF
  __m128i *v120; // [rsp+2F8h] [rbp-48h]
  __m128i *v121; // [rsp+300h] [rbp-40h]
  __int8 *v122; // [rsp+308h] [rbp-38h]

  v1 = a1 + 136;
  v3 = *(_QWORD *)(a1 + 128);
  v4 = *(_QWORD *)(v3 + 328);
  v5 = v3 + 320;
  if ( v5 != v4 )
  {
    v6 = v4;
    v7 = 0;
    do
    {
      v6 = *(_QWORD *)(v6 + 8);
      ++v7;
    }
    while ( v5 != v6 );
    v8 = *(const void **)(a1 + 136);
    if ( v7 > (unsigned __int64)((__int64)(*(_QWORD *)(a1 + 152) - (_QWORD)v8) >> 3) )
    {
      v9 = 8LL * v7;
      v10 = *(_QWORD *)(a1 + 144) - (_QWORD)v8;
      if ( v7 )
      {
        v11 = sub_22077B0(8LL * v7);
        v8 = *(const void **)(a1 + 136);
        v12 = (char *)v11;
        v13 = *(_QWORD *)(a1 + 144) - (_QWORD)v8;
        if ( v13 <= 0 )
        {
LABEL_7:
          if ( !v8 )
          {
LABEL_8:
            *(_QWORD *)(a1 + 136) = v12;
            *(_QWORD *)(a1 + 144) = &v12[v10];
            *(_QWORD *)(a1 + 152) = &v12[v9];
            goto LABEL_9;
          }
          v81 = *(_QWORD *)(a1 + 152) - (_QWORD)v8;
LABEL_104:
          v88 = v12;
          j_j___libc_free_0(v8, v81);
          v12 = v88;
          goto LABEL_8;
        }
      }
      else
      {
        v13 = *(_QWORD *)(a1 + 144) - (_QWORD)v8;
        v12 = 0;
        if ( v10 <= 0 )
          goto LABEL_7;
      }
      v87 = v8;
      v80 = (char *)memmove(v12, v8, v13);
      v8 = v87;
      v12 = v80;
      v81 = *(_QWORD *)(a1 + 152) - (_QWORD)v87;
      goto LABEL_104;
    }
  }
LABEL_9:
  memset(v98, 0, sizeof(v98));
  v95 = 0;
  v98[1] = &v98[5];
  v98[2] = &v98[5];
  v90 = v94;
  v91 = v94;
  v96 = 0;
  v97 = 0;
  v92 = 0x100000008LL;
  v117.m128i_i64[1] = *(_QWORD *)(v4 + 88);
  v94[0] = v4;
  v117.m128i_i64[0] = v4;
  LODWORD(v98[3]) = 8;
  v93 = 0;
  v89 = 1;
  sub_1DE02F0(&v95, 0, &v117);
  sub_1DE0470((__int64)&v89);
  v14 = v111;
  sub_16CCCB0(v111, (__int64)v113, (__int64)v98);
  v15 = (const __m128i *)v98[14];
  v16 = (const __m128i *)v98[13];
  v114 = 0;
  v115 = 0;
  v116 = 0;
  v17 = (_BYTE *)(v98[14] - v98[13]);
  if ( v98[14] == v98[13] )
  {
    v19 = 0;
    v18 = 0;
  }
  else
  {
    if ( (unsigned __int64)v17 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_130;
    v83 = v98[14] - v98[13];
    v18 = (__m128i *)sub_22077B0(v98[14] - v98[13]);
    v15 = (const __m128i *)v98[14];
    v16 = (const __m128i *)v98[13];
    v19 = v83;
  }
  v114 = v18;
  v115 = v18;
  v116 = &v18->m128i_i8[v19];
  if ( v16 == v15 )
  {
    v20 = v18;
  }
  else
  {
    v20 = (__m128i *)((char *)v18 + (char *)v15 - (char *)v16);
    do
    {
      if ( v18 )
        *v18 = _mm_loadu_si128(v16);
      ++v18;
      ++v16;
    }
    while ( v20 != v18 );
  }
  v115 = v20;
  sub_16CCEE0(&v117, (__int64)v119, 8, (__int64)v111);
  v21 = v114;
  v14 = v99;
  v17 = v101;
  v114 = 0;
  v120 = v21;
  v22 = v115;
  v115 = 0;
  v121 = v22;
  v23 = v116;
  v116 = 0;
  v122 = v23;
  sub_16CCCB0(v99, (__int64)v101, (__int64)&v89);
  v24 = v96;
  v16 = v95;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v25 = (char *)v96 - (char *)v95;
  if ( v96 == v95 )
  {
    v25 = 0;
    v26 = 0;
    goto LABEL_20;
  }
  if ( v25 > 0x7FFFFFFFFFFFFFF0LL )
LABEL_130:
    sub_4261EA(v14, v17, v16);
  v26 = (__m128i *)sub_22077B0((char *)v96 - (char *)v95);
  v24 = v96;
  v16 = v95;
LABEL_20:
  v102 = v26;
  v103 = v26;
  v104 = &v26->m128i_i8[v25];
  if ( v16 == v24 )
  {
    v27 = v26;
  }
  else
  {
    v27 = (__m128i *)((char *)v26 + (char *)v24 - (char *)v16);
    do
    {
      if ( v26 )
        *v26 = _mm_loadu_si128(v16);
      ++v26;
      ++v16;
    }
    while ( v26 != v27 );
  }
  v103 = v27;
  sub_16CCEE0(v105, (__int64)v107, 8, (__int64)v99);
  v28 = v102;
  v102 = 0;
  v108 = v28;
  v29 = v103;
  v103 = 0;
  v109 = v29;
  v30 = v104;
  v104 = 0;
  v110 = v30;
  sub_1DE0A50((__int64)v105, (__int64)&v117, v1);
  if ( v108 )
    j_j___libc_free_0(v108, v110 - (__int8 *)v108);
  if ( v106 != v105[1] )
    _libc_free(v106);
  if ( v102 )
    j_j___libc_free_0(v102, v104 - (__int8 *)v102);
  if ( v100 != v99[1] )
    _libc_free(v100);
  if ( v120 )
    j_j___libc_free_0(v120, v122 - (__int8 *)v120);
  if ( v118 != v117.m128i_i64[1] )
    _libc_free(v118);
  if ( v114 )
    j_j___libc_free_0(v114, v116 - (__int8 *)v114);
  if ( v112 != v111[1] )
    _libc_free(v112);
  if ( v95 )
    j_j___libc_free_0(v95, v97 - (_QWORD)v95);
  if ( v91 != v90 )
    _libc_free((unsigned __int64)v91);
  if ( v98[13] )
    j_j___libc_free_0(v98[13], v98[15] - v98[13]);
  if ( v98[2] != v98[1] )
    _libc_free(v98[2]);
  v31 = *(char **)(a1 + 144);
  v32 = *(char **)(a1 + 136);
  if ( v32 == v31 )
    goto LABEL_116;
  v33 = (__int64 *)(v31 - 8);
  if ( v32 < v31 - 8 )
  {
    do
    {
      v34 = *(_QWORD *)v32;
      v35 = *v33;
      v32 += 8;
      --v33;
      *((_QWORD *)v32 - 1) = v35;
      v33[1] = v34;
    }
    while ( v33 > (__int64 *)v32 );
    v32 = *(char **)(a1 + 136);
    v31 = *(char **)(a1 + 144);
    if ( v32 == v31 )
    {
LABEL_116:
      v50 = v31;
      v51 = (const __m128i **)(a1 + 64);
      v52 = 0;
      goto LABEL_65;
    }
  }
  v36 = (__int64 *)v32;
  v84 = a1 + 160;
  while ( 1 )
  {
    v41 = *(_DWORD *)(a1 + 184);
    v42 = ((char *)v36 - v32) >> 3;
    if ( !v41 )
    {
      ++*(_QWORD *)(a1 + 160);
      goto LABEL_59;
    }
    v37 = *(_QWORD *)(a1 + 168);
    v38 = (v41 - 1) & (((unsigned int)*v36 >> 9) ^ ((unsigned int)*v36 >> 4));
    v39 = (__int64 *)(v37 + 16LL * v38);
    v40 = *v39;
    if ( *v36 != *v39 )
      break;
LABEL_55:
    ++v36;
    *((_DWORD *)v39 + 2) = v42;
    if ( v31 == (char *)v36 )
      goto LABEL_64;
LABEL_56:
    v32 = *(char **)(a1 + 136);
  }
  v70 = 1;
  v71 = 0;
  while ( v40 != -8 )
  {
    if ( !v71 && v40 == -16 )
      v71 = v39;
    v38 = (v41 - 1) & (v70 + v38);
    v39 = (__int64 *)(v37 + 16LL * v38);
    v40 = *v39;
    if ( *v36 == *v39 )
      goto LABEL_55;
    ++v70;
  }
  v72 = *(_DWORD *)(a1 + 176);
  if ( v71 )
    v39 = v71;
  ++*(_QWORD *)(a1 + 160);
  v46 = v72 + 1;
  if ( 4 * (v72 + 1) >= 3 * v41 )
  {
LABEL_59:
    sub_1DE05B0(v84, 2 * v41);
    v43 = *(_DWORD *)(a1 + 184);
    if ( !v43 )
      goto LABEL_132;
    v44 = v43 - 1;
    v45 = *(_QWORD *)(a1 + 168);
    v46 = *(_DWORD *)(a1 + 176) + 1;
    v47 = v44 & (((unsigned int)*v36 >> 9) ^ ((unsigned int)*v36 >> 4));
    v39 = (__int64 *)(v45 + 16LL * v47);
    v48 = *v39;
    if ( *v36 != *v39 )
    {
      v82 = 1;
      v75 = 0;
      while ( v48 != -8 )
      {
        if ( v48 == -16 && !v75 )
          v75 = v39;
        v47 = v44 & (v82 + v47);
        v39 = (__int64 *)(v45 + 16LL * v47);
        v48 = *v39;
        if ( *v36 == *v39 )
          goto LABEL_61;
        ++v82;
      }
LABEL_99:
      if ( v75 )
        v39 = v75;
      goto LABEL_61;
    }
    goto LABEL_61;
  }
  if ( v41 - *(_DWORD *)(a1 + 180) - v46 <= v41 >> 3 )
  {
    sub_1DE05B0(v84, v41);
    v73 = *(_DWORD *)(a1 + 184);
    if ( !v73 )
    {
LABEL_132:
      ++*(_DWORD *)(a1 + 176);
      BUG();
    }
    v74 = v73 - 1;
    v75 = 0;
    v76 = *(_QWORD *)(a1 + 168);
    v77 = 1;
    v46 = *(_DWORD *)(a1 + 176) + 1;
    v78 = v74 & (((unsigned int)*v36 >> 9) ^ ((unsigned int)*v36 >> 4));
    v39 = (__int64 *)(v76 + 16LL * v78);
    v79 = *v39;
    if ( *v39 != *v36 )
    {
      while ( v79 != -8 )
      {
        if ( v79 == -16 && !v75 )
          v75 = v39;
        v78 = v74 & (v77 + v78);
        v39 = (__int64 *)(v76 + 16LL * v78);
        v79 = *v39;
        if ( *v36 == *v39 )
          goto LABEL_61;
        ++v77;
      }
      goto LABEL_99;
    }
  }
LABEL_61:
  *(_DWORD *)(a1 + 176) = v46;
  if ( *v39 != -8 )
    --*(_DWORD *)(a1 + 180);
  v49 = *v36++;
  *((_DWORD *)v39 + 2) = -1;
  *((_DWORD *)v39 + 2) = v42;
  *v39 = v49;
  if ( v31 != (char *)v36 )
    goto LABEL_56;
LABEL_64:
  v31 = *(char **)(a1 + 144);
  v50 = *(char **)(a1 + 136);
  v51 = (const __m128i **)(a1 + 64);
  v52 = (v31 - v50) >> 3;
  if ( (unsigned __int64)(v31 - v50) > 0x2AAAAAAAAAAAAAA8LL )
    sub_4262D8((__int64)"vector::reserve");
LABEL_65:
  v53 = *(const __m128i **)(a1 + 64);
  v54 = v53;
  if ( 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(a1 + 80) - (_QWORD)v53) >> 3) < v52 )
  {
    v55 = *(const __m128i **)(a1 + 72);
    v56 = 24 * v52;
    v57 = 0;
    v58 = (char *)v55 - (char *)v53;
    if ( v52 )
    {
      v85 = 24 * v52;
      v59 = sub_22077B0(24 * v52);
      v53 = *(const __m128i **)(a1 + 64);
      v55 = *(const __m128i **)(a1 + 72);
      v56 = v85;
      v57 = (__m128i *)v59;
      v54 = v53;
    }
    if ( v53 != v55 )
    {
      v60 = v57;
      do
      {
        if ( v60 )
        {
          *v60 = _mm_loadu_si128(v54);
          v60[1].m128i_i64[0] = v54[1].m128i_i64[0];
        }
        v54 = (const __m128i *)((char *)v54 + 24);
        v60 = (__m128i *)((char *)v60 + 24);
      }
      while ( v55 != v54 );
      v55 = v53;
    }
    if ( v55 )
    {
      v86 = v56;
      j_j___libc_free_0(v55, *(_QWORD *)(a1 + 80) - (_QWORD)v55);
      v56 = v86;
    }
    *(_QWORD *)(a1 + 64) = v57;
    v50 = *(char **)(a1 + 136);
    *(_QWORD *)(a1 + 72) = (char *)v57 + v58;
    *(_QWORD *)(a1 + 80) = (char *)v57 + v56;
    v31 = *(char **)(a1 + 144);
  }
  v117.m128i_i64[0] = 0;
  if ( v31 == v50 )
  {
    v65 = *(_QWORD *)(a1 + 16);
    v66 = *(_QWORD *)(a1 + 8);
    v64 = 0;
    v68 = 0xAAAAAAAAAAAAAAABLL * ((v65 - v66) >> 3);
    goto LABEL_86;
  }
  LODWORD(v61) = 0;
  do
  {
    v62 = *(__m128i **)(a1 + 72);
    if ( v62 == *(__m128i **)(a1 + 80) )
    {
      sub_13699D0(v51, v62, &v117);
    }
    else
    {
      if ( v62 )
      {
        v62->m128i_i32[0] = v61;
        v62->m128i_i64[1] = 0;
        v62[1].m128i_i64[0] = 0;
        v62 = *(__m128i **)(a1 + 72);
      }
      *(_QWORD *)(a1 + 72) = (char *)v62 + 24;
    }
    v61 = v117.m128i_i64[0] + 1;
    v63 = (__int64)(*(_QWORD *)(a1 + 144) - *(_QWORD *)(a1 + 136)) >> 3;
    v117.m128i_i64[0] = v61;
    v64 = v63;
  }
  while ( v61 < v63 );
  v65 = *(_QWORD *)(a1 + 16);
  v66 = *(_QWORD *)(a1 + 8);
  v67 = 0xAAAAAAAAAAAAAAABLL * ((v65 - v66) >> 3);
  v68 = v67;
  if ( v63 > v67 )
  {
    sub_1369B80((const __m128i **)(a1 + 8), v63 - v67);
  }
  else
  {
LABEL_86:
    if ( v64 < v68 )
    {
      v69 = v66 + 24 * v64;
      if ( v65 != v69 )
        *(_QWORD *)(a1 + 16) = v69;
    }
  }
}
