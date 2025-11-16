// Function: sub_1A45460
// Address: 0x1a45460
//
void __fastcall sub_1A45460(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rax
  _BYTE *v4; // rsi
  _QWORD *v5; // rdi
  __int64 v6; // rdx
  const __m128i *v7; // rcx
  const __m128i *v8; // r8
  unsigned __int64 v9; // rbx
  __int64 v10; // rax
  __m128i *v11; // rdi
  __m128i *v12; // rdx
  const __m128i *v13; // rax
  __m128i *v14; // rax
  __m128i *v15; // rax
  __int8 *v16; // rax
  const __m128i *v17; // rcx
  const __m128i *v18; // r8
  unsigned __int64 v19; // rbx
  __int64 v20; // rax
  __m128i *v21; // rdi
  __m128i *v22; // rdx
  const __m128i *v23; // rax
  __m128i *v24; // rax
  __m128i *v25; // rax
  __int8 *v26; // rax
  __int64 v27; // [rsp+0h] [rbp-330h] BYREF
  _QWORD *v28; // [rsp+8h] [rbp-328h]
  _QWORD *v29; // [rsp+10h] [rbp-320h]
  __int64 v30; // [rsp+18h] [rbp-318h]
  int v31; // [rsp+20h] [rbp-310h]
  _QWORD v32[8]; // [rsp+28h] [rbp-308h] BYREF
  const __m128i *v33; // [rsp+68h] [rbp-2C8h] BYREF
  const __m128i *v34; // [rsp+70h] [rbp-2C0h]
  __int64 v35; // [rsp+78h] [rbp-2B8h]
  _QWORD v36[16]; // [rsp+80h] [rbp-2B0h] BYREF
  _QWORD v37[2]; // [rsp+100h] [rbp-230h] BYREF
  unsigned __int64 v38; // [rsp+110h] [rbp-220h]
  _BYTE v39[64]; // [rsp+128h] [rbp-208h] BYREF
  __m128i *v40; // [rsp+168h] [rbp-1C8h]
  __m128i *v41; // [rsp+170h] [rbp-1C0h]
  __int8 *v42; // [rsp+178h] [rbp-1B8h]
  _QWORD v43[2]; // [rsp+180h] [rbp-1B0h] BYREF
  unsigned __int64 v44; // [rsp+190h] [rbp-1A0h]
  char v45[64]; // [rsp+1A8h] [rbp-188h] BYREF
  __m128i *v46; // [rsp+1E8h] [rbp-148h]
  __m128i *v47; // [rsp+1F0h] [rbp-140h]
  __int8 *v48; // [rsp+1F8h] [rbp-138h]
  _QWORD v49[2]; // [rsp+200h] [rbp-130h] BYREF
  unsigned __int64 v50; // [rsp+210h] [rbp-120h]
  _BYTE v51[64]; // [rsp+228h] [rbp-108h] BYREF
  __m128i *v52; // [rsp+268h] [rbp-C8h]
  __m128i *v53; // [rsp+270h] [rbp-C0h]
  __int8 *v54; // [rsp+278h] [rbp-B8h]
  __m128i v55; // [rsp+280h] [rbp-B0h] BYREF
  unsigned __int64 v56; // [rsp+290h] [rbp-A0h]
  char v57[64]; // [rsp+2A8h] [rbp-88h] BYREF
  __m128i *v58; // [rsp+2E8h] [rbp-48h]
  __m128i *v59; // [rsp+2F0h] [rbp-40h]
  __int8 *v60; // [rsp+2F8h] [rbp-38h]

  v32[0] = a2;
  memset(v36, 0, sizeof(v36));
  LODWORD(v36[3]) = 8;
  v36[1] = &v36[5];
  v36[2] = &v36[5];
  v28 = v32;
  v29 = v32;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v30 = 0x100000008LL;
  v31 = 0;
  v27 = 1;
  v3 = sub_157EBA0(a2);
  v55.m128i_i64[0] = a2;
  v55.m128i_i64[1] = v3;
  LODWORD(v56) = 0;
  sub_13FDF40(&v33, 0, &v55);
  sub_13FE0F0((__int64)&v27);
  v4 = v51;
  v5 = v49;
  sub_16CCCB0(v49, (__int64)v51, (__int64)v36);
  v7 = (const __m128i *)v36[14];
  v8 = (const __m128i *)v36[13];
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v9 = v36[14] - v36[13];
  if ( v36[14] == v36[13] )
  {
    v11 = 0;
  }
  else
  {
    if ( v9 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_46;
    v10 = sub_22077B0(v36[14] - v36[13]);
    v7 = (const __m128i *)v36[14];
    v8 = (const __m128i *)v36[13];
    v11 = (__m128i *)v10;
  }
  v52 = v11;
  v53 = v11;
  v54 = &v11->m128i_i8[v9];
  if ( v8 != v7 )
  {
    v12 = v11;
    v13 = v8;
    do
    {
      if ( v12 )
      {
        *v12 = _mm_loadu_si128(v13);
        v12[1].m128i_i64[0] = v13[1].m128i_i64[0];
      }
      v13 = (const __m128i *)((char *)v13 + 24);
      v12 = (__m128i *)((char *)v12 + 24);
    }
    while ( v13 != v7 );
    v11 = (__m128i *)((char *)v11 + 8 * ((unsigned __int64)((char *)&v13[-2].m128i_u64[1] - (char *)v8) >> 3) + 24);
  }
  v53 = v11;
  sub_16CCEE0(&v55, (__int64)v57, 8, (__int64)v49);
  v14 = v52;
  v5 = v37;
  v4 = v39;
  v52 = 0;
  v58 = v14;
  v15 = v53;
  v53 = 0;
  v59 = v15;
  v16 = v54;
  v54 = 0;
  v60 = v16;
  sub_16CCCB0(v37, (__int64)v39, (__int64)&v27);
  v17 = v34;
  v18 = v33;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v19 = (char *)v34 - (char *)v33;
  if ( v34 != v33 )
  {
    if ( v19 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v20 = sub_22077B0((char *)v34 - (char *)v33);
      v17 = v34;
      v18 = v33;
      v21 = (__m128i *)v20;
      goto LABEL_13;
    }
LABEL_46:
    sub_4261EA(v5, v4, v6);
  }
  v21 = 0;
LABEL_13:
  v40 = v21;
  v41 = v21;
  v42 = &v21->m128i_i8[v19];
  if ( v17 != v18 )
  {
    v22 = v21;
    v23 = v18;
    do
    {
      if ( v22 )
      {
        *v22 = _mm_loadu_si128(v23);
        v22[1].m128i_i64[0] = v23[1].m128i_i64[0];
      }
      v23 = (const __m128i *)((char *)v23 + 24);
      v22 = (__m128i *)((char *)v22 + 24);
    }
    while ( v17 != v23 );
    v21 = (__m128i *)((char *)v21 + 8 * ((unsigned __int64)((char *)&v17[-2].m128i_u64[1] - (char *)v18) >> 3) + 24);
  }
  v41 = v21;
  sub_16CCEE0(v43, (__int64)v45, 8, (__int64)v37);
  v24 = v40;
  v40 = 0;
  v46 = v24;
  v25 = v41;
  v41 = 0;
  v47 = v25;
  v26 = v42;
  v42 = 0;
  v48 = v26;
  sub_19380F0((__int64)v43, (__int64)&v55, a1);
  if ( v46 )
    j_j___libc_free_0(v46, v48 - (__int8 *)v46);
  if ( v44 != v43[1] )
    _libc_free(v44);
  if ( v40 )
    j_j___libc_free_0(v40, v42 - (__int8 *)v40);
  if ( v38 != v37[1] )
    _libc_free(v38);
  if ( v58 )
    j_j___libc_free_0(v58, v60 - (__int8 *)v58);
  if ( v56 != v55.m128i_i64[1] )
    _libc_free(v56);
  if ( v52 )
    j_j___libc_free_0(v52, v54 - (__int8 *)v52);
  if ( v50 != v49[1] )
    _libc_free(v50);
  if ( v33 )
    j_j___libc_free_0(v33, v35 - (_QWORD)v33);
  if ( v29 != v28 )
    _libc_free((unsigned __int64)v29);
  if ( v36[13] )
    j_j___libc_free_0(v36[13], v36[15] - v36[13]);
  if ( v36[2] != v36[1] )
    _libc_free(v36[2]);
}
