// Function: sub_1E2DAF0
// Address: 0x1e2daf0
//
void __fastcall sub_1E2DAF0(__int64 a1, __int64 a2)
{
  int v3; // ebx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r13
  int v7; // r13d
  __int64 v8; // rax
  __int64 v9; // rdx
  int v10; // eax
  int v11; // ebx
  __int64 v12; // rdx
  __int64 v13; // rax
  __m128i *v14; // rax
  const __m128i *v15; // rax
  __m128i *v16; // rax
  __int8 *v17; // rax
  const __m128i *v18; // rax
  __m128i *v19; // rax
  __int8 *v20; // rax
  const __m128i *v21; // rax
  const __m128i *v22; // rax
  __int8 *v23; // rax
  __int64 *v24; // rdi
  const __m128i *v25; // rcx
  const __m128i *v26; // rdx
  unsigned __int64 v27; // rsi
  __m128i *v28; // rax
  __int64 v29; // rsi
  __m128i *v30; // rcx
  const __m128i *v31; // rcx
  __int64 v32; // rax
  __m128i *v33; // rdi
  __m128i *v34; // rax
  __m128i *v35; // rdx
  __m128i *v36; // rax
  __int64 *v37; // rcx
  __int64 *v38; // rdx
  signed __int64 v39; // [rsp+10h] [rbp-3A0h]
  unsigned __int64 v40; // [rsp+10h] [rbp-3A0h]
  __int64 v41; // [rsp+70h] [rbp-340h]
  _QWORD v43[16]; // [rsp+80h] [rbp-330h] BYREF
  __int64 v44; // [rsp+100h] [rbp-2B0h] BYREF
  _QWORD *v45; // [rsp+108h] [rbp-2A8h]
  _QWORD *v46; // [rsp+110h] [rbp-2A0h]
  __int64 v47; // [rsp+118h] [rbp-298h]
  int v48; // [rsp+120h] [rbp-290h]
  _QWORD v49[8]; // [rsp+128h] [rbp-288h] BYREF
  const __m128i *v50; // [rsp+168h] [rbp-248h] BYREF
  __m128i *v51; // [rsp+170h] [rbp-240h]
  __int8 *v52; // [rsp+178h] [rbp-238h]
  __int64 v53; // [rsp+180h] [rbp-230h] BYREF
  __int64 v54; // [rsp+188h] [rbp-228h]
  unsigned __int64 v55; // [rsp+190h] [rbp-220h]
  _BYTE v56[64]; // [rsp+1A8h] [rbp-208h] BYREF
  const __m128i *v57; // [rsp+1E8h] [rbp-1C8h]
  __m128i *v58; // [rsp+1F0h] [rbp-1C0h]
  __int8 *v59; // [rsp+1F8h] [rbp-1B8h]
  __int64 v60; // [rsp+200h] [rbp-1B0h] BYREF
  __int64 v61; // [rsp+208h] [rbp-1A8h]
  unsigned __int64 v62; // [rsp+210h] [rbp-1A0h]
  _BYTE v63[64]; // [rsp+228h] [rbp-188h] BYREF
  __m128i *v64; // [rsp+268h] [rbp-148h]
  __m128i *v65; // [rsp+270h] [rbp-140h]
  __int8 *v66; // [rsp+278h] [rbp-138h]
  __m128i v67; // [rsp+280h] [rbp-130h] BYREF
  unsigned __int64 v68; // [rsp+290h] [rbp-120h]
  char v69[64]; // [rsp+2A8h] [rbp-108h] BYREF
  const __m128i *v70; // [rsp+2E8h] [rbp-C8h]
  const __m128i *v71; // [rsp+2F0h] [rbp-C0h]
  __int8 *v72; // [rsp+2F8h] [rbp-B8h]
  __int64 v73; // [rsp+300h] [rbp-B0h] BYREF
  __int64 v74; // [rsp+308h] [rbp-A8h]
  unsigned __int64 v75; // [rsp+310h] [rbp-A0h]
  char v76[64]; // [rsp+328h] [rbp-88h] BYREF
  const __m128i *v77; // [rsp+368h] [rbp-48h]
  const __m128i *v78; // [rsp+370h] [rbp-40h]
  __int8 *v79; // [rsp+378h] [rbp-38h]

  if ( !(*(_DWORD *)(**(_QWORD **)(**(_QWORD **)(a1 - 24) + 16LL) + 8LL) >> 8) || *(_BYTE *)(a2 + 1745) )
    return;
  v3 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( *(char *)(a1 + 23) < 0 )
  {
    v4 = sub_1648A40(a1);
    v6 = v4 + v5;
    if ( *(char *)(a1 + 23) >= 0 )
    {
      if ( (unsigned int)(v6 >> 4) )
        goto LABEL_96;
    }
    else if ( (unsigned int)((v6 - sub_1648A40(a1)) >> 4) )
    {
      if ( *(char *)(a1 + 23) < 0 )
      {
        v7 = *(_DWORD *)(sub_1648A40(a1) + 8);
        if ( *(char *)(a1 + 23) >= 0 )
          BUG();
        v8 = sub_1648A40(a1);
        v10 = *(_DWORD *)(v8 + v9 - 4) - v7;
        goto LABEL_12;
      }
LABEL_96:
      BUG();
    }
  }
  v10 = 0;
LABEL_12:
  v11 = v3 - 1 - v10;
  if ( !v11 )
    return;
  v41 = 0;
LABEL_14:
  v12 = **(_QWORD **)(a1 + 24 * (v41 - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
  v50 = 0;
  memset(v43, 0, sizeof(v43));
  v43[2] = &v43[5];
  v43[1] = &v43[5];
  v49[0] = v12;
  v46 = v49;
  v45 = v49;
  v47 = 0x100000008LL;
  LODWORD(v43[3]) = 8;
  v51 = 0;
  v52 = 0;
  v48 = 0;
  v13 = *(_QWORD *)(v12 + 16);
  v67.m128i_i64[0] = v12;
  v67.m128i_i64[1] = v13;
  v44 = 1;
  sub_1E2D820(&v50, 0, &v67);
  sub_1E2D9A0((__int64)&v44);
  sub_16CCEE0(&v60, (__int64)v63, 8, (__int64)v43);
  v14 = (__m128i *)v43[13];
  memset(&v43[13], 0, 24);
  v64 = v14;
  v65 = (__m128i *)v43[14];
  v66 = (__int8 *)v43[15];
  sub_16CCEE0(&v53, (__int64)v56, 8, (__int64)&v44);
  v15 = v50;
  v50 = 0;
  v57 = v15;
  v16 = v51;
  v51 = 0;
  v58 = v16;
  v17 = v52;
  v52 = 0;
  v59 = v17;
  sub_16CCEE0(&v67, (__int64)v69, 8, (__int64)&v53);
  v18 = v57;
  v57 = 0;
  v70 = v18;
  v19 = v58;
  v58 = 0;
  v71 = v19;
  v20 = v59;
  v59 = 0;
  v72 = v20;
  sub_16CCEE0(&v73, (__int64)v76, 8, (__int64)&v60);
  v21 = v64;
  v64 = 0;
  v77 = v21;
  v22 = v65;
  v65 = 0;
  v78 = v22;
  v23 = v66;
  v66 = 0;
  v79 = v23;
  if ( v57 )
    j_j___libc_free_0(v57, v59 - (__int8 *)v57);
  if ( v55 != v54 )
    _libc_free(v55);
  if ( v64 )
    j_j___libc_free_0(v64, v66 - (__int8 *)v64);
  if ( v62 != v61 )
    _libc_free(v62);
  if ( v50 )
    j_j___libc_free_0(v50, v52 - (__int8 *)v50);
  if ( v46 != v45 )
    _libc_free((unsigned __int64)v46);
  if ( v43[13] )
    j_j___libc_free_0(v43[13], v43[15] - v43[13]);
  if ( v43[2] != v43[1] )
    _libc_free(v43[2]);
  v24 = &v53;
  sub_16CCCB0(&v53, (__int64)v56, (__int64)&v67);
  v25 = v71;
  v26 = v70;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v27 = (char *)v71 - (char *)v70;
  if ( v71 == v70 )
  {
    v29 = 0;
    v28 = 0;
  }
  else
  {
    if ( v27 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_95;
    v39 = (char *)v71 - (char *)v70;
    v28 = (__m128i *)sub_22077B0((char *)v71 - (char *)v70);
    v25 = v71;
    v26 = v70;
    v29 = v39;
  }
  v57 = v28;
  v58 = v28;
  v59 = &v28->m128i_i8[v29];
  if ( v25 == v26 )
  {
    v30 = v28;
  }
  else
  {
    v30 = (__m128i *)((char *)v28 + (char *)v25 - (char *)v26);
    do
    {
      if ( v28 )
        *v28 = _mm_loadu_si128(v26);
      ++v28;
      ++v26;
    }
    while ( v28 != v30 );
  }
  v24 = &v60;
  v58 = v30;
  sub_16CCCB0(&v60, (__int64)v63, (__int64)&v73);
  v26 = v78;
  v31 = v77;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v27 = (char *)v78 - (char *)v77;
  if ( v78 == v77 )
  {
    v33 = 0;
  }
  else
  {
    if ( v27 > 0x7FFFFFFFFFFFFFF0LL )
LABEL_95:
      sub_4261EA(v24, v27, v26);
    v40 = (char *)v78 - (char *)v77;
    v32 = sub_22077B0((char *)v78 - (char *)v77);
    v26 = v78;
    v31 = v77;
    v27 = v40;
    v33 = (__m128i *)v32;
  }
  v64 = v33;
  v65 = v33;
  v66 = &v33->m128i_i8[v27];
  if ( v26 == v31 )
  {
    v35 = v33;
  }
  else
  {
    v34 = v33;
    v35 = (__m128i *)((char *)v33 + (char *)v26 - (char *)v31);
    do
    {
      if ( v34 )
        *v34 = _mm_loadu_si128(v31);
      ++v34;
      ++v31;
    }
    while ( v34 != v35 );
  }
  v65 = v35;
LABEL_47:
  v36 = v58;
  v37 = (__int64 *)v57;
  if ( (char *)v58 - (char *)v57 == (char *)v35 - (char *)v33 )
    goto LABEL_68;
  while ( (unsigned __int8)(*(_BYTE *)(v36[-1].m128i_i64[0] + 8) - 1) > 5u )
  {
    v37 = (__int64 *)v57;
    v58 = --v36;
    if ( v36 != v57 )
    {
      sub_1E2D9A0((__int64)&v53);
      v33 = v64;
      v35 = v65;
      goto LABEL_47;
    }
    v33 = v64;
    if ( (char *)v36 - (char *)v57 == (char *)v65 - (char *)v64 )
    {
LABEL_68:
      if ( v37 == (__int64 *)v36 )
      {
LABEL_73:
        if ( v33 )
          j_j___libc_free_0(v33, v66 - (__int8 *)v33);
        if ( v62 != v61 )
          _libc_free(v62);
        if ( v57 )
          j_j___libc_free_0(v57, v59 - (__int8 *)v57);
        if ( v55 != v54 )
          _libc_free(v55);
        if ( v77 )
          j_j___libc_free_0(v77, v79 - (__int8 *)v77);
        if ( v75 != v74 )
          _libc_free(v75);
        if ( v70 )
          j_j___libc_free_0(v70, v72 - (__int8 *)v70);
        if ( v68 != v67.m128i_i64[1] )
          _libc_free(v68);
        if ( v11 == ++v41 )
          return;
        goto LABEL_14;
      }
      v38 = (__int64 *)v33;
      while ( *v37 == *v38 && v37[1] == v38[1] )
      {
        v37 += 2;
        v38 += 2;
        if ( v37 == (__int64 *)v36 )
          goto LABEL_73;
      }
    }
  }
  *(_BYTE *)(a2 + 1745) = 1;
  if ( v33 )
    j_j___libc_free_0(v33, v66 - (__int8 *)v33);
  if ( v62 != v61 )
    _libc_free(v62);
  if ( v57 )
    j_j___libc_free_0(v57, v59 - (__int8 *)v57);
  if ( v55 != v54 )
    _libc_free(v55);
  if ( v77 )
    j_j___libc_free_0(v77, v79 - (__int8 *)v77);
  if ( v75 != v74 )
    _libc_free(v75);
  if ( v70 )
    j_j___libc_free_0(v70, v72 - (__int8 *)v70);
  if ( v68 != v67.m128i_i64[1] )
    _libc_free(v68);
}
