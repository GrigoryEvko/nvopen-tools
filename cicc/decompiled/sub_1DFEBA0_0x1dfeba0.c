// Function: sub_1DFEBA0
// Address: 0x1dfeba0
//
void __fastcall sub_1DFEBA0(_QWORD *a1)
{
  __int64 v2; // rdx
  __m128i v3; // rax
  __m128i *v4; // rax
  const __m128i *v5; // rax
  __m128i *v6; // rax
  __int8 *v7; // rax
  const __m128i *v8; // rax
  __m128i *v9; // rax
  __int8 *v10; // rax
  const __m128i *v11; // rax
  __m128i *v12; // rax
  __int8 *v13; // rax
  _BYTE *v14; // rsi
  __int64 *v15; // rdi
  const __m128i *v16; // rcx
  const __m128i *v17; // rdx
  unsigned __int64 v18; // r15
  __m128i *v19; // rax
  __m128i *v20; // rcx
  const __m128i *v21; // rax
  const __m128i *v22; // rcx
  unsigned __int64 v23; // r14
  __int64 v24; // rax
  __m128i *v25; // rdi
  __m128i *v26; // rdx
  __m128i *v27; // rax
  __int64 *v28; // rdx
  __int64 *v29; // rcx
  __int64 *v30; // rax
  size_t *v31; // r12
  unsigned __int64 *v32; // r13
  size_t v33; // rsi
  __int64 v34; // r12
  __int64 i; // r13
  _QWORD v36[16]; // [rsp+10h] [rbp-330h] BYREF
  __int64 v37; // [rsp+90h] [rbp-2B0h] BYREF
  _QWORD *v38; // [rsp+98h] [rbp-2A8h]
  _QWORD *v39; // [rsp+A0h] [rbp-2A0h]
  __int64 v40; // [rsp+A8h] [rbp-298h]
  int v41; // [rsp+B0h] [rbp-290h]
  _QWORD v42[8]; // [rsp+B8h] [rbp-288h] BYREF
  const __m128i *v43; // [rsp+F8h] [rbp-248h] BYREF
  __m128i *v44; // [rsp+100h] [rbp-240h]
  __int8 *v45; // [rsp+108h] [rbp-238h]
  __int64 v46; // [rsp+110h] [rbp-230h] BYREF
  __int64 v47; // [rsp+118h] [rbp-228h]
  unsigned __int64 v48; // [rsp+120h] [rbp-220h]
  _BYTE v49[64]; // [rsp+138h] [rbp-208h] BYREF
  const __m128i *v50; // [rsp+178h] [rbp-1C8h]
  __m128i *v51; // [rsp+180h] [rbp-1C0h]
  __int8 *v52; // [rsp+188h] [rbp-1B8h]
  __int64 v53; // [rsp+190h] [rbp-1B0h] BYREF
  __int64 v54; // [rsp+198h] [rbp-1A8h]
  unsigned __int64 v55; // [rsp+1A0h] [rbp-1A0h]
  _BYTE v56[64]; // [rsp+1B8h] [rbp-188h] BYREF
  __m128i *v57; // [rsp+1F8h] [rbp-148h]
  __m128i *v58; // [rsp+200h] [rbp-140h]
  __int8 *v59; // [rsp+208h] [rbp-138h]
  __m128i v60; // [rsp+210h] [rbp-130h] BYREF
  unsigned __int64 v61; // [rsp+220h] [rbp-120h] BYREF
  char v62[64]; // [rsp+238h] [rbp-108h] BYREF
  const __m128i *v63; // [rsp+278h] [rbp-C8h]
  const __m128i *v64; // [rsp+280h] [rbp-C0h]
  __int8 *v65; // [rsp+288h] [rbp-B8h]
  _QWORD v66[2]; // [rsp+290h] [rbp-B0h] BYREF
  unsigned __int64 v67; // [rsp+2A0h] [rbp-A0h]
  char v68[64]; // [rsp+2B8h] [rbp-88h] BYREF
  const __m128i *v69; // [rsp+2F8h] [rbp-48h]
  const __m128i *v70; // [rsp+300h] [rbp-40h]
  __int8 *v71; // [rsp+308h] [rbp-38h]

  v2 = *a1;
  v43 = 0;
  memset(v36, 0, sizeof(v36));
  v44 = 0;
  v36[1] = &v36[5];
  v36[2] = &v36[5];
  v3.m128i_i64[0] = *(_QWORD *)(v2 + 328);
  v38 = v42;
  v39 = v42;
  v45 = 0;
  v3.m128i_i64[1] = *(_QWORD *)(v3.m128i_i64[0] + 88);
  v40 = 0x100000008LL;
  v42[0] = v3.m128i_i64[0];
  v60 = v3;
  LODWORD(v36[3]) = 8;
  v41 = 0;
  v37 = 1;
  sub_1D530F0(&v43, 0, &v60);
  sub_1DFC2B0((__int64)&v37);
  sub_16CCEE0(&v53, (__int64)v56, 8, (__int64)v36);
  v4 = (__m128i *)v36[13];
  memset(&v36[13], 0, 24);
  v57 = v4;
  v58 = (__m128i *)v36[14];
  v59 = (__int8 *)v36[15];
  sub_16CCEE0(&v46, (__int64)v49, 8, (__int64)&v37);
  v5 = v43;
  v43 = 0;
  v50 = v5;
  v6 = v44;
  v44 = 0;
  v51 = v6;
  v7 = v45;
  v45 = 0;
  v52 = v7;
  sub_16CCEE0(&v60, (__int64)v62, 8, (__int64)&v46);
  v8 = v50;
  v50 = 0;
  v63 = v8;
  v9 = v51;
  v51 = 0;
  v64 = v9;
  v10 = v52;
  v52 = 0;
  v65 = v10;
  sub_16CCEE0(v66, (__int64)v68, 8, (__int64)&v53);
  v11 = v57;
  v57 = 0;
  v69 = v11;
  v12 = v58;
  v58 = 0;
  v70 = v12;
  v13 = v59;
  v59 = 0;
  v71 = v13;
  if ( v50 )
    j_j___libc_free_0(v50, v52 - (__int8 *)v50);
  if ( v48 != v47 )
    _libc_free(v48);
  if ( v57 )
    j_j___libc_free_0(v57, v59 - (__int8 *)v57);
  if ( v55 != v54 )
    _libc_free(v55);
  if ( v43 )
    j_j___libc_free_0(v43, v45 - (__int8 *)v43);
  if ( v39 != v38 )
    _libc_free((unsigned __int64)v39);
  if ( v36[13] )
    j_j___libc_free_0(v36[13], v36[15] - v36[13]);
  if ( v36[2] != v36[1] )
    _libc_free(v36[2]);
  v14 = v49;
  v15 = &v46;
  sub_16CCCB0(&v46, (__int64)v49, (__int64)&v60);
  v16 = v64;
  v17 = v63;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v18 = (char *)v64 - (char *)v63;
  if ( v64 == v63 )
  {
    v18 = 0;
    v19 = 0;
  }
  else
  {
    if ( v18 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_71;
    v19 = (__m128i *)sub_22077B0((char *)v64 - (char *)v63);
    v16 = v64;
    v17 = v63;
  }
  v50 = v19;
  v51 = v19;
  v52 = &v19->m128i_i8[v18];
  if ( v16 == v17 )
  {
    v20 = v19;
  }
  else
  {
    v20 = (__m128i *)((char *)v19 + (char *)v16 - (char *)v17);
    do
    {
      if ( v19 )
        *v19 = _mm_loadu_si128(v17);
      ++v19;
      ++v17;
    }
    while ( v19 != v20 );
  }
  v14 = v56;
  v15 = &v53;
  v51 = v20;
  sub_16CCCB0(&v53, (__int64)v56, (__int64)v66);
  v21 = v70;
  v22 = v69;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v23 = (char *)v70 - (char *)v69;
  if ( v70 != v69 )
  {
    if ( v23 <= 0x7FFFFFFFFFFFFFF0LL )
    {
      v24 = sub_22077B0((char *)v70 - (char *)v69);
      v22 = v69;
      v25 = (__m128i *)v24;
      v21 = v70;
      goto LABEL_28;
    }
LABEL_71:
    sub_4261EA(v15, v14, v17);
  }
  v25 = 0;
LABEL_28:
  v57 = v25;
  v58 = v25;
  v59 = &v25->m128i_i8[v23];
  if ( v22 == v21 )
  {
    v27 = v25;
  }
  else
  {
    v26 = v25;
    v27 = (__m128i *)((char *)v25 + (char *)v21 - (char *)v22);
    do
    {
      if ( v26 )
        *v26 = _mm_loadu_si128(v22);
      ++v26;
      ++v22;
    }
    while ( v26 != v27 );
  }
  v58 = v27;
LABEL_35:
  v28 = (__int64 *)v51;
  v29 = (__int64 *)v50;
  if ( (char *)v51 - (char *)v50 == (char *)v27 - (char *)v25 )
    goto LABEL_38;
  while ( 1 )
  {
    do
    {
      sub_1DFD480((__int64)a1, *(v28 - 2));
      --v51;
      v29 = (__int64 *)v50;
      v28 = (__int64 *)v51;
      if ( v51 != v50 )
      {
        sub_1DFC2B0((__int64)&v46);
        v25 = v57;
        v27 = v58;
        goto LABEL_35;
      }
      v25 = v57;
    }
    while ( (char *)v51 - (char *)v50 != (char *)v58 - (char *)v57 );
LABEL_38:
    if ( v29 == v28 )
      break;
    v30 = (__int64 *)v25;
    while ( *v29 == *v30 && v29[1] == v30[1] )
    {
      v29 += 2;
      v30 += 2;
      if ( v29 == v28 )
        goto LABEL_43;
    }
  }
LABEL_43:
  if ( v25 )
    j_j___libc_free_0(v25, v59 - (__int8 *)v25);
  if ( v55 != v54 )
    _libc_free(v55);
  if ( v50 )
    j_j___libc_free_0(v50, v52 - (__int8 *)v50);
  if ( v48 != v47 )
    _libc_free(v48);
  if ( v69 )
    j_j___libc_free_0(v69, v71 - (__int8 *)v69);
  if ( v67 != v66[1] )
    _libc_free(v67);
  if ( v63 )
    j_j___libc_free_0(v63, v65 - (__int8 *)v63);
  if ( v61 != v60.m128i_i64[1] )
    _libc_free(v61);
  sub_1E29340(&v60, a1[1] + 232LL);
  v31 = (size_t *)v60.m128i_i64[0];
  v32 = (unsigned __int64 *)(v60.m128i_i64[0] + 8LL * v60.m128i_u32[2]);
  if ( (unsigned __int64 *)v60.m128i_i64[0] != v32 )
  {
    do
    {
      v33 = *v31++;
      sub_1DFDA90((__int64)a1, v33);
    }
    while ( v32 != v31 );
    v32 = (unsigned __int64 *)v60.m128i_i64[0];
  }
  if ( v32 != &v61 )
    _libc_free((unsigned __int64)v32);
  v34 = *(_QWORD *)(*a1 + 328LL);
  for ( i = *a1 + 320LL; i != v34; v34 = *(_QWORD *)(v34 + 8) )
    sub_1DFE8A0((__int64)a1, v34);
}
