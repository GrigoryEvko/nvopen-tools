// Function: sub_22DEF40
// Address: 0x22def40
//
void __fastcall sub_22DEF40(__int64 a1, _QWORD *a2)
{
  _BYTE *v4; // rsi
  char *v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdx
  __int64 v10; // r9
  unsigned __int64 v11; // rcx
  __int64 v12; // r8
  unsigned __int64 v13; // r14
  __int64 v14; // rax
  unsigned __int64 v15; // rdi
  __m128i *v16; // rdx
  const __m128i *v17; // rax
  const __m128i *v18; // rcx
  unsigned __int64 v19; // r8
  unsigned __int64 v20; // r14
  __int64 v21; // rax
  unsigned __int64 v22; // rdi
  __m128i *v23; // rdx
  const __m128i *v24; // rax
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rcx
  _QWORD *v27; // r8
  unsigned __int64 v28; // rax
  char v29; // si
  char v30[8]; // [rsp+0h] [rbp-210h] BYREF
  unsigned __int64 v31; // [rsp+8h] [rbp-208h]
  char v32; // [rsp+1Ch] [rbp-1F4h]
  _BYTE v33[64]; // [rsp+20h] [rbp-1F0h] BYREF
  unsigned __int64 v34; // [rsp+60h] [rbp-1B0h]
  unsigned __int64 v35; // [rsp+68h] [rbp-1A8h]
  unsigned __int64 v36; // [rsp+70h] [rbp-1A0h]
  char v37[8]; // [rsp+80h] [rbp-190h] BYREF
  unsigned __int64 v38; // [rsp+88h] [rbp-188h]
  char v39; // [rsp+9Ch] [rbp-174h]
  _BYTE v40[64]; // [rsp+A0h] [rbp-170h] BYREF
  unsigned __int64 v41; // [rsp+E0h] [rbp-130h]
  unsigned __int64 i; // [rsp+E8h] [rbp-128h]
  unsigned __int64 v43; // [rsp+F0h] [rbp-120h]
  _QWORD v44[3]; // [rsp+100h] [rbp-110h] BYREF
  char v45; // [rsp+11Ch] [rbp-F4h]
  __int64 v46; // [rsp+160h] [rbp-B0h]
  unsigned __int64 v47; // [rsp+168h] [rbp-A8h]
  char v48[8]; // [rsp+178h] [rbp-98h] BYREF
  unsigned __int64 v49; // [rsp+180h] [rbp-90h]
  char v50; // [rsp+194h] [rbp-7Ch]
  const __m128i *v51; // [rsp+1D8h] [rbp-38h]
  const __m128i *v52; // [rsp+1E0h] [rbp-30h]

  sub_22DEC60(v44, a2);
  v4 = v33;
  v5 = v30;
  sub_C8CD80((__int64)v30, (__int64)v33, (__int64)v44, v6, v7, v8);
  v11 = v47;
  v12 = v46;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v13 = v47 - v46;
  if ( v47 == v46 )
  {
    v15 = 0;
  }
  else
  {
    if ( v13 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_56;
    v14 = sub_22077B0(v47 - v46);
    v11 = v47;
    v12 = v46;
    v15 = v14;
  }
  v34 = v15;
  v35 = v15;
  v36 = v15 + v13;
  if ( v12 != v11 )
  {
    v16 = (__m128i *)v15;
    v17 = (const __m128i *)v12;
    do
    {
      if ( v16 )
      {
        *v16 = _mm_loadu_si128(v17);
        v16[1] = _mm_loadu_si128(v17 + 1);
        v16[2].m128i_i64[0] = v17[2].m128i_i64[0];
      }
      v17 = (const __m128i *)((char *)v17 + 40);
      v16 = (__m128i *)((char *)v16 + 40);
    }
    while ( (const __m128i *)v11 != v17 );
    v11 = (v11 - 40 - v12) >> 3;
    v15 += 8 * v11 + 40;
  }
  v35 = v15;
  v5 = v37;
  v4 = v40;
  sub_C8CD80((__int64)v37, (__int64)v40, (__int64)v48, v11, v12, v10);
  v18 = v52;
  v19 = (unsigned __int64)v51;
  v41 = 0;
  i = 0;
  v43 = 0;
  v20 = (char *)v52 - (char *)v51;
  if ( v52 == v51 )
  {
    v22 = 0;
    goto LABEL_13;
  }
  if ( v20 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_56:
    sub_4261EA(v5, v4, v9);
  v21 = sub_22077B0((char *)v52 - (char *)v51);
  v18 = v52;
  v19 = (unsigned __int64)v51;
  v22 = v21;
LABEL_13:
  v41 = v22;
  i = v22;
  v43 = v22 + v20;
  if ( v18 == (const __m128i *)v19 )
  {
    v25 = v22;
  }
  else
  {
    v23 = (__m128i *)v22;
    v24 = (const __m128i *)v19;
    do
    {
      if ( v23 )
      {
        *v23 = _mm_loadu_si128(v24);
        v23[1] = _mm_loadu_si128(v24 + 1);
        v23[2].m128i_i64[0] = v24[2].m128i_i64[0];
      }
      v24 = (const __m128i *)((char *)v24 + 40);
      v23 = (__m128i *)((char *)v23 + 40);
    }
    while ( v18 != v24 );
    v25 = v22 + 8 * (((unsigned __int64)&v18[-3].m128i_u64[1] - v19) >> 3) + 40;
  }
  for ( i = v25; ; v25 = i )
  {
    v26 = v34;
    if ( v35 - v34 != v25 - v22 )
      goto LABEL_23;
    if ( v34 == v35 )
      break;
    v28 = v22;
    while ( *(_QWORD *)v26 == *(_QWORD *)v28 )
    {
      v29 = *(_BYTE *)(v26 + 32);
      if ( v29 != *(_BYTE *)(v28 + 32) )
        break;
      if ( v29 )
      {
        if ( ((*(__int64 *)(v26 + 8) >> 1) & 3) != 0 )
        {
          if ( ((*(__int64 *)(v26 + 8) >> 1) & 3) != ((*(__int64 *)(v28 + 8) >> 1) & 3) )
            break;
        }
        else if ( *(_DWORD *)(v26 + 24) != *(_DWORD *)(v28 + 24) )
        {
          break;
        }
      }
      v26 += 40LL;
      v28 += 40LL;
      if ( v35 == v26 )
        goto LABEL_34;
    }
LABEL_23:
    v27 = *(_QWORD **)(v35 - 40);
    if ( (*v27 & 4) != 0 )
    {
      sub_22DEF40(a1, *(_QWORD *)(v35 - 40));
    }
    else if ( a2 != (_QWORD *)sub_22DBE80(a1, *v27 & 0xFFFFFFFFFFFFFFF8LL) )
    {
      sub_C64ED0("BB map does not match region nesting", 1u);
    }
    sub_22DE060((__int64)v30);
    v22 = v41;
  }
LABEL_34:
  if ( v22 )
    j_j___libc_free_0(v22);
  if ( !v39 )
    _libc_free(v38);
  if ( v34 )
    j_j___libc_free_0(v34);
  if ( !v32 )
    _libc_free(v31);
  if ( v51 )
    j_j___libc_free_0((unsigned __int64)v51);
  if ( !v50 )
    _libc_free(v49);
  if ( v46 )
    j_j___libc_free_0(v46);
  if ( !v45 )
    _libc_free(v44[1]);
}
