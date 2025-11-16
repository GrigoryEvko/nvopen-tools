// Function: sub_2E41F60
// Address: 0x2e41f60
//
void __fastcall sub_2E41F60(__int64 a1)
{
  __int64 v2; // r12
  __int64 v3; // rdx
  __int64 v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // rax
  unsigned __int64 v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r8
  int v14; // ecx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r8
  unsigned int v18; // ecx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  char *v22; // r13
  char *v23; // rax
  char *v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rsi
  __int64 *v27; // rbx
  __int64 v28; // r9
  __int64 *v29; // rdi
  int v30; // r11d
  unsigned int v31; // edx
  __int64 *v32; // rax
  __int64 v33; // r8
  unsigned int v34; // esi
  __int64 v35; // r12
  int v36; // ecx
  int v37; // ecx
  __int64 v38; // r9
  unsigned int v39; // edx
  int v40; // eax
  __int64 v41; // r8
  __int64 v42; // rax
  char *v43; // rcx
  const __m128i **v44; // rbx
  unsigned __int64 v45; // rsi
  unsigned __int64 v46; // rdi
  const __m128i *v47; // rax
  const __m128i *v48; // rcx
  __m128i *v49; // r13
  __int8 *v50; // r12
  __int64 v51; // rax
  __m128i *v52; // rdx
  unsigned __int64 v53; // rax
  __m128i *v54; // rsi
  unsigned __int64 v55; // rsi
  unsigned __int64 v56; // rdx
  int v57; // eax
  int v58; // ecx
  int v59; // ecx
  __int64 v60; // r9
  __int64 *v61; // r10
  int v62; // r11d
  unsigned int v63; // edx
  __int64 v64; // r8
  __int64 v65; // rdi
  __int64 v66; // rcx
  unsigned __int64 v67; // rax
  unsigned __int64 v68; // r9
  __int64 v69; // rax
  __int64 v70; // rsi
  __m128i *v71; // rdx
  const __m128i *v72; // rax
  unsigned __int64 v73; // rdi
  __int64 v74; // rsi
  __m128i *v75; // rdx
  const __m128i *v76; // rax
  const __m128i *v77; // rdi
  int v78; // r11d
  unsigned int v79; // [rsp+8h] [rbp-768h]
  int v80; // [rsp+18h] [rbp-758h]
  __int64 v81; // [rsp+20h] [rbp-750h] BYREF
  __int64 *v82; // [rsp+28h] [rbp-748h]
  int v83; // [rsp+30h] [rbp-740h]
  int v84; // [rsp+34h] [rbp-73Ch]
  int v85; // [rsp+38h] [rbp-738h]
  char v86; // [rsp+3Ch] [rbp-734h]
  __int64 v87; // [rsp+40h] [rbp-730h] BYREF
  __int64 *v88; // [rsp+80h] [rbp-6F0h]
  unsigned int v89; // [rsp+88h] [rbp-6E8h]
  int v90; // [rsp+8Ch] [rbp-6E4h]
  __int64 v91; // [rsp+90h] [rbp-6E0h] BYREF
  __int64 v92; // [rsp+98h] [rbp-6D8h]
  __int64 v93; // [rsp+A0h] [rbp-6D0h]
  unsigned __int64 v94[38]; // [rsp+150h] [rbp-620h] BYREF
  char v95[8]; // [rsp+280h] [rbp-4F0h] BYREF
  unsigned __int64 v96; // [rsp+288h] [rbp-4E8h]
  char v97; // [rsp+29Ch] [rbp-4D4h]
  char v98[64]; // [rsp+2A0h] [rbp-4D0h] BYREF
  __m128i *v99; // [rsp+2E0h] [rbp-490h] BYREF
  __int64 v100; // [rsp+2E8h] [rbp-488h]
  _BYTE v101[192]; // [rsp+2F0h] [rbp-480h] BYREF
  char v102[8]; // [rsp+3B0h] [rbp-3C0h] BYREF
  unsigned __int64 v103; // [rsp+3B8h] [rbp-3B8h]
  char v104; // [rsp+3CCh] [rbp-3A4h]
  char *v105; // [rsp+410h] [rbp-360h]
  char v106; // [rsp+420h] [rbp-350h] BYREF
  char v107[8]; // [rsp+4E0h] [rbp-290h] BYREF
  unsigned __int64 v108; // [rsp+4E8h] [rbp-288h]
  char v109; // [rsp+4FCh] [rbp-274h]
  char v110[64]; // [rsp+500h] [rbp-270h] BYREF
  __m128i *v111; // [rsp+540h] [rbp-230h] BYREF
  __int64 v112; // [rsp+548h] [rbp-228h]
  _BYTE v113[192]; // [rsp+550h] [rbp-220h] BYREF
  _QWORD v114[3]; // [rsp+610h] [rbp-160h] BYREF
  char v115; // [rsp+62Ch] [rbp-144h]
  char *v116; // [rsp+670h] [rbp-100h]
  char v117; // [rsp+680h] [rbp-F0h] BYREF

  v2 = a1 + 136;
  v3 = *(_QWORD *)(a1 + 128);
  v4 = *(_QWORD *)(v3 + 328);
  v5 = v3 + 320;
  if ( v5 == v4 )
  {
    v7 = 0;
  }
  else
  {
    v6 = v4;
    LODWORD(v7) = 0;
    do
    {
      v6 = *(_QWORD *)(v6 + 8);
      LODWORD(v7) = v7 + 1;
    }
    while ( v5 != v6 );
    v7 = (unsigned int)v7;
  }
  sub_2E3A980(a1 + 136, v7);
  memset(v94, 0, sizeof(v94));
  v94[1] = (unsigned __int64)&v94[4];
  v8 = *(unsigned int *)(v4 + 120);
  v94[12] = (unsigned __int64)&v94[14];
  v82 = &v87;
  v92 = *(_QWORD *)(v4 + 112);
  v91 = v92 + 8 * v8;
  v87 = v4;
  v93 = v4;
  LODWORD(v94[2]) = 8;
  BYTE4(v94[3]) = 1;
  HIDWORD(v94[13]) = 8;
  v83 = 8;
  v85 = 0;
  v86 = 1;
  v88 = &v91;
  v90 = 8;
  v84 = 1;
  v81 = 1;
  v89 = 1;
  sub_2E3BE50((__int64)&v81, v7, (__int64)v94, v91, v9, v10);
  sub_C8CD80((__int64)v107, (__int64)v110, (__int64)v94, v11, v12, (__int64)v107);
  v14 = v94[13];
  v111 = (__m128i *)v113;
  v112 = 0x800000000LL;
  if ( LODWORD(v94[13]) )
  {
    v70 = LODWORD(v94[13]);
    v71 = (__m128i *)v113;
    if ( LODWORD(v94[13]) > 8 )
    {
      v80 = v94[13];
      sub_2E3C030((__int64)&v111, LODWORD(v94[13]), (__int64)v113, LODWORD(v94[13]), v13, (__int64)v107);
      v71 = v111;
      v70 = LODWORD(v94[13]);
      v14 = v80;
    }
    v72 = (const __m128i *)v94[12];
    v73 = v94[12] + 24 * v70;
    if ( v94[12] != v73 )
    {
      do
      {
        if ( v71 )
        {
          *v71 = _mm_loadu_si128(v72);
          v71[1].m128i_i64[0] = v72[1].m128i_i64[0];
        }
        v72 = (const __m128i *)((char *)v72 + 24);
        v71 = (__m128i *)((char *)v71 + 24);
      }
      while ( (const __m128i *)v73 != v72 );
    }
    LODWORD(v112) = v14;
  }
  sub_2E3C0D0((__int64)v114, (__int64)v107);
  sub_C8CD80((__int64)v95, (__int64)v98, (__int64)&v81, v15, v16, (__int64)v95);
  v18 = v89;
  v99 = (__m128i *)v101;
  v100 = 0x800000000LL;
  if ( v89 )
  {
    v74 = v89;
    v75 = (__m128i *)v101;
    if ( v89 > 8 )
    {
      v79 = v89;
      sub_2E3C030((__int64)&v99, v89, (__int64)v101, v89, v17, (__int64)v95);
      v75 = v99;
      v74 = v89;
      v18 = v79;
    }
    v76 = (const __m128i *)v88;
    v77 = (const __m128i *)&v88[3 * v74];
    if ( v88 != (__int64 *)v77 )
    {
      do
      {
        if ( v75 )
        {
          *v75 = _mm_loadu_si128(v76);
          v75[1].m128i_i64[0] = v76[1].m128i_i64[0];
        }
        v76 = (const __m128i *)((char *)v76 + 24);
        v75 = (__m128i *)((char *)v75 + 24);
      }
      while ( v77 != v76 );
    }
    LODWORD(v100) = v18;
  }
  sub_2E3C0D0((__int64)v102, (__int64)v95);
  sub_2E41930((__int64)v102, (__int64)v114, v2, v19, v20, v21);
  if ( v105 != &v106 )
    _libc_free((unsigned __int64)v105);
  if ( !v104 )
    _libc_free(v103);
  if ( v99 != (__m128i *)v101 )
    _libc_free((unsigned __int64)v99);
  if ( !v97 )
    _libc_free(v96);
  if ( v116 != &v117 )
    _libc_free((unsigned __int64)v116);
  if ( !v115 )
    _libc_free(v114[1]);
  if ( v111 != (__m128i *)v113 )
    _libc_free((unsigned __int64)v111);
  if ( !v109 )
    _libc_free(v108);
  if ( v88 != &v91 )
    _libc_free((unsigned __int64)v88);
  if ( !v86 )
    _libc_free((unsigned __int64)v82);
  if ( (unsigned __int64 *)v94[12] != &v94[14] )
    _libc_free(v94[12]);
  if ( !BYTE4(v94[3]) )
    _libc_free(v94[1]);
  v22 = *(char **)(a1 + 144);
  v23 = *(char **)(a1 + 136);
  if ( v23 == v22 )
    goto LABEL_103;
  v24 = v22 - 8;
  if ( v23 < v22 - 8 )
  {
    do
    {
      v25 = *(_QWORD *)v23;
      v26 = *(_QWORD *)v24;
      v23 += 8;
      v24 -= 8;
      *((_QWORD *)v23 - 1) = v26;
      *((_QWORD *)v24 + 1) = v25;
    }
    while ( v24 > v23 );
    v23 = *(char **)(a1 + 136);
    v22 = *(char **)(a1 + 144);
    if ( v23 == v22 )
    {
LABEL_103:
      v43 = v22;
      v44 = (const __m128i **)(a1 + 64);
      v45 = 0;
      goto LABEL_47;
    }
  }
  v27 = (__int64 *)v23;
  while ( 1 )
  {
    v34 = *(_DWORD *)(a1 + 184);
    v35 = ((char *)v27 - v23) >> 3;
    if ( !v34 )
    {
      ++*(_QWORD *)(a1 + 160);
      goto LABEL_41;
    }
    v28 = *(_QWORD *)(a1 + 168);
    v29 = 0;
    v30 = 1;
    v31 = (v34 - 1) & (((unsigned int)*v27 >> 9) ^ ((unsigned int)*v27 >> 4));
    v32 = (__int64 *)(v28 + 16LL * v31);
    v33 = *v32;
    if ( *v27 != *v32 )
      break;
LABEL_37:
    ++v27;
    *((_DWORD *)v32 + 2) = v35;
    if ( v22 == (char *)v27 )
      goto LABEL_46;
LABEL_38:
    v23 = *(char **)(a1 + 136);
  }
  while ( v33 != -4096 )
  {
    if ( !v29 && v33 == -8192 )
      v29 = v32;
    v31 = (v34 - 1) & (v30 + v31);
    v32 = (__int64 *)(v28 + 16LL * v31);
    v33 = *v32;
    if ( *v27 == *v32 )
      goto LABEL_37;
    ++v30;
  }
  if ( !v29 )
    v29 = v32;
  v57 = *(_DWORD *)(a1 + 176);
  ++*(_QWORD *)(a1 + 160);
  v40 = v57 + 1;
  if ( 4 * v40 >= 3 * v34 )
  {
LABEL_41:
    sub_2E3D7D0(a1 + 160, 2 * v34);
    v36 = *(_DWORD *)(a1 + 184);
    if ( !v36 )
      goto LABEL_118;
    v37 = v36 - 1;
    v38 = *(_QWORD *)(a1 + 168);
    v39 = v37 & (((unsigned int)*v27 >> 9) ^ ((unsigned int)*v27 >> 4));
    v40 = *(_DWORD *)(a1 + 176) + 1;
    v29 = (__int64 *)(v38 + 16LL * v39);
    v41 = *v29;
    if ( *v29 != *v27 )
    {
      v78 = 1;
      v61 = 0;
      while ( v41 != -4096 )
      {
        if ( !v61 && v41 == -8192 )
          v61 = v29;
        v39 = v37 & (v78 + v39);
        v29 = (__int64 *)(v38 + 16LL * v39);
        v41 = *v29;
        if ( *v27 == *v29 )
          goto LABEL_43;
        ++v78;
      }
LABEL_79:
      if ( v61 )
        v29 = v61;
      goto LABEL_43;
    }
    goto LABEL_43;
  }
  if ( v34 - *(_DWORD *)(a1 + 180) - v40 <= v34 >> 3 )
  {
    sub_2E3D7D0(a1 + 160, v34);
    v58 = *(_DWORD *)(a1 + 184);
    if ( !v58 )
    {
LABEL_118:
      ++*(_DWORD *)(a1 + 176);
      BUG();
    }
    v59 = v58 - 1;
    v60 = *(_QWORD *)(a1 + 168);
    v61 = 0;
    v62 = 1;
    v63 = v59 & (((unsigned int)*v27 >> 9) ^ ((unsigned int)*v27 >> 4));
    v40 = *(_DWORD *)(a1 + 176) + 1;
    v29 = (__int64 *)(v60 + 16LL * v63);
    v64 = *v29;
    if ( *v27 != *v29 )
    {
      while ( v64 != -4096 )
      {
        if ( v64 == -8192 && !v61 )
          v61 = v29;
        v63 = v59 & (v62 + v63);
        v29 = (__int64 *)(v60 + 16LL * v63);
        v64 = *v29;
        if ( *v27 == *v29 )
          goto LABEL_43;
        ++v62;
      }
      goto LABEL_79;
    }
  }
LABEL_43:
  *(_DWORD *)(a1 + 176) = v40;
  if ( *v29 != -4096 )
    --*(_DWORD *)(a1 + 180);
  v42 = *v27++;
  *((_DWORD *)v29 + 2) = -1;
  *v29 = v42;
  *((_DWORD *)v29 + 2) = v35;
  if ( v22 != (char *)v27 )
    goto LABEL_38;
LABEL_46:
  v22 = *(char **)(a1 + 144);
  v43 = *(char **)(a1 + 136);
  v44 = (const __m128i **)(a1 + 64);
  v45 = (v22 - v43) >> 3;
  if ( (unsigned __int64)(v22 - v43) > 0x2AAAAAAAAAAAAAA8LL )
    sub_4262D8((__int64)"vector::reserve");
LABEL_47:
  v46 = *(_QWORD *)(a1 + 64);
  v47 = (const __m128i *)v46;
  if ( 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(a1 + 80) - v46) >> 3) < v45 )
  {
    v48 = *(const __m128i **)(a1 + 72);
    v49 = 0;
    v50 = &v48->m128i_i8[-v46];
    if ( v45 )
    {
      v51 = sub_22077B0(24 * v45);
      v46 = *(_QWORD *)(a1 + 64);
      v48 = *(const __m128i **)(a1 + 72);
      v49 = (__m128i *)v51;
      v47 = (const __m128i *)v46;
    }
    if ( v48 != (const __m128i *)v46 )
    {
      v52 = v49;
      do
      {
        if ( v52 )
        {
          *v52 = _mm_loadu_si128(v47);
          v52[1].m128i_i64[0] = v47[1].m128i_i64[0];
        }
        v47 = (const __m128i *)((char *)v47 + 24);
        v52 = (__m128i *)((char *)v52 + 24);
      }
      while ( v48 != v47 );
    }
    if ( v46 )
      j_j___libc_free_0(v46);
    *(_QWORD *)(a1 + 64) = v49;
    v43 = *(char **)(a1 + 136);
    *(_QWORD *)(a1 + 72) = &v50[(_QWORD)v49];
    *(_QWORD *)(a1 + 80) = (char *)v49 + 24 * v45;
    v22 = *(char **)(a1 + 144);
  }
  v114[0] = 0;
  if ( v43 == v22 )
  {
    v65 = *(_QWORD *)(a1 + 16);
    v66 = *(_QWORD *)(a1 + 8);
    v56 = 0;
    v68 = 0xAAAAAAAAAAAAAAABLL * ((v65 - v66) >> 3);
    goto LABEL_83;
  }
  LODWORD(v53) = 0;
  do
  {
    v54 = *(__m128i **)(a1 + 72);
    if ( v54 == *(__m128i **)(a1 + 80) )
    {
      sub_FDDEB0(v44, v54, v114);
    }
    else
    {
      if ( v54 )
      {
        v54->m128i_i32[0] = v53;
        v54->m128i_i64[1] = 0;
        v54[1].m128i_i64[0] = 0;
        v54 = *(__m128i **)(a1 + 72);
      }
      *(_QWORD *)(a1 + 72) = (char *)v54 + 24;
    }
    v53 = v114[0] + 1LL;
    v55 = (__int64)(*(_QWORD *)(a1 + 144) - *(_QWORD *)(a1 + 136)) >> 3;
    v114[0] = v53;
    v56 = v55;
  }
  while ( v53 < v55 );
  v65 = *(_QWORD *)(a1 + 16);
  v66 = *(_QWORD *)(a1 + 8);
  v67 = 0xAAAAAAAAAAAAAAABLL * ((v65 - v66) >> 3);
  v68 = v67;
  if ( v55 > v67 )
  {
    sub_FDE060((const __m128i **)(a1 + 8), v55 - v67);
  }
  else
  {
LABEL_83:
    if ( v68 > v56 )
    {
      v69 = v66 + 24 * v56;
      if ( v65 != v69 )
        *(_QWORD *)(a1 + 16) = v69;
    }
  }
}
