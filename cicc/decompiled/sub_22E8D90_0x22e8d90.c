// Function: sub_22E8D90
// Address: 0x22e8d90
//
void __fastcall sub_22E8D90(__int64 a1)
{
  _BYTE *v2; // rsi
  char *v3; // rdi
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rdx
  __int64 v8; // r9
  unsigned __int64 v9; // rcx
  __int64 v10; // r8
  unsigned __int64 v11; // r13
  __int64 v12; // rax
  unsigned __int64 v13; // rdi
  __m128i *v14; // rdx
  const __m128i *v15; // rax
  const __m128i *v16; // rcx
  unsigned __int64 v17; // r8
  unsigned __int64 v18; // r13
  __int64 v19; // rax
  unsigned __int64 v20; // rdi
  __m128i *v21; // rdx
  const __m128i *v22; // rax
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rcx
  unsigned __int64 v25; // rax
  char v26; // si
  char v27[8]; // [rsp+0h] [rbp-210h] BYREF
  unsigned __int64 v28; // [rsp+8h] [rbp-208h]
  char v29; // [rsp+1Ch] [rbp-1F4h]
  _BYTE v30[64]; // [rsp+20h] [rbp-1F0h] BYREF
  unsigned __int64 v31; // [rsp+60h] [rbp-1B0h]
  unsigned __int64 v32; // [rsp+68h] [rbp-1A8h]
  unsigned __int64 v33; // [rsp+70h] [rbp-1A0h]
  char v34[8]; // [rsp+80h] [rbp-190h] BYREF
  unsigned __int64 v35; // [rsp+88h] [rbp-188h]
  char v36; // [rsp+9Ch] [rbp-174h]
  _BYTE v37[64]; // [rsp+A0h] [rbp-170h] BYREF
  unsigned __int64 v38; // [rsp+E0h] [rbp-130h]
  unsigned __int64 i; // [rsp+E8h] [rbp-128h]
  unsigned __int64 v40; // [rsp+F0h] [rbp-120h]
  _QWORD v41[3]; // [rsp+100h] [rbp-110h] BYREF
  char v42; // [rsp+11Ch] [rbp-F4h]
  __int64 v43; // [rsp+160h] [rbp-B0h]
  unsigned __int64 v44; // [rsp+168h] [rbp-A8h]
  char v45[8]; // [rsp+178h] [rbp-98h] BYREF
  unsigned __int64 v46; // [rsp+180h] [rbp-90h]
  char v47; // [rsp+194h] [rbp-7Ch]
  const __m128i *v48; // [rsp+1D8h] [rbp-38h]
  const __m128i *v49; // [rsp+1E0h] [rbp-30h]

  sub_22E71B0(v41, *(__int64 **)(a1 + 8));
  v2 = v30;
  v3 = v27;
  sub_C8CD80((__int64)v27, (__int64)v30, (__int64)v41, v4, v5, v6);
  v9 = v44;
  v10 = v43;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v11 = v44 - v43;
  if ( v44 == v43 )
  {
    v13 = 0;
  }
  else
  {
    if ( v11 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_50;
    v12 = sub_22077B0(v44 - v43);
    v9 = v44;
    v10 = v43;
    v13 = v12;
  }
  v31 = v13;
  v32 = v13;
  v33 = v13 + v11;
  if ( v10 != v9 )
  {
    v14 = (__m128i *)v13;
    v15 = (const __m128i *)v10;
    do
    {
      if ( v14 )
      {
        *v14 = _mm_loadu_si128(v15);
        v14[1] = _mm_loadu_si128(v15 + 1);
        v14[2].m128i_i64[0] = v15[2].m128i_i64[0];
      }
      v15 = (const __m128i *)((char *)v15 + 40);
      v14 = (__m128i *)((char *)v14 + 40);
    }
    while ( (const __m128i *)v9 != v15 );
    v9 = (v9 - 40 - v10) >> 3;
    v13 += 8 * v9 + 40;
  }
  v32 = v13;
  v3 = v34;
  v2 = v37;
  sub_C8CD80((__int64)v34, (__int64)v37, (__int64)v45, v9, v10, v8);
  v16 = v49;
  v17 = (unsigned __int64)v48;
  v38 = 0;
  i = 0;
  v40 = 0;
  v18 = (char *)v49 - (char *)v48;
  if ( v49 == v48 )
  {
    v20 = 0;
    goto LABEL_13;
  }
  if ( v18 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_50:
    sub_4261EA(v3, v2, v7);
  v19 = sub_22077B0((char *)v49 - (char *)v48);
  v16 = v49;
  v17 = (unsigned __int64)v48;
  v20 = v19;
LABEL_13:
  v38 = v20;
  i = v20;
  v40 = v20 + v18;
  if ( (const __m128i *)v17 == v16 )
  {
    v23 = v20;
  }
  else
  {
    v21 = (__m128i *)v20;
    v22 = (const __m128i *)v17;
    do
    {
      if ( v21 )
      {
        *v21 = _mm_loadu_si128(v22);
        v21[1] = _mm_loadu_si128(v22 + 1);
        v21[2].m128i_i64[0] = v22[2].m128i_i64[0];
      }
      v22 = (const __m128i *)((char *)v22 + 40);
      v21 = (__m128i *)((char *)v21 + 40);
    }
    while ( v22 != v16 );
    v23 = v20 + 8 * (((unsigned __int64)&v22[-3].m128i_u64[1] - v17) >> 3) + 40;
  }
  for ( i = v23; ; v23 = i )
  {
    v24 = v31;
    if ( v32 - v31 != v23 - v20 )
      goto LABEL_20;
    if ( v31 == v32 )
      break;
    v25 = v20;
    while ( *(_QWORD *)v24 == *(_QWORD *)v25 )
    {
      v26 = *(_BYTE *)(v24 + 32);
      if ( v26 != *(_BYTE *)(v25 + 32)
        || v26 && (*(_DWORD *)(v24 + 24) != *(_DWORD *)(v25 + 24) || *(_QWORD *)(v24 + 8) != *(_QWORD *)(v25 + 8)) )
      {
        break;
      }
      v24 += 40LL;
      v25 += 40LL;
      if ( v32 == v24 )
        goto LABEL_30;
    }
LABEL_20:
    sub_22E8150(a1, *(__int64 **)(v32 - 40));
    sub_22E7560((__int64)v27);
    v20 = v38;
  }
LABEL_30:
  if ( v20 )
    j_j___libc_free_0(v20);
  if ( !v36 )
    _libc_free(v35);
  if ( v31 )
    j_j___libc_free_0(v31);
  if ( !v29 )
    _libc_free(v28);
  if ( v48 )
    j_j___libc_free_0((unsigned __int64)v48);
  if ( !v47 )
    _libc_free(v46);
  if ( v43 )
    j_j___libc_free_0(v43);
  if ( !v42 )
    _libc_free(v41[1]);
}
