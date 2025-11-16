// Function: sub_E9E5B0
// Address: 0xe9e5b0
//
__int64 __fastcall sub_E9E5B0(
        __int64 a1,
        _BYTE *a2,
        __int64 a3,
        _BYTE *a4,
        __int64 a5,
        unsigned __int32 a6,
        __int128 a7,
        char a8,
        __int128 a9,
        __int64 a10)
{
  _BYTE *v10; // r10
  __int64 v13; // rdi
  char v14; // r13
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rdx
  __m128i v19; // xmm2
  __m128i v20; // xmm3
  __m128i *v21; // rdi
  __int64 v22; // rsi
  size_t v23; // r9
  __int64 v24; // rdx
  __m128i *v25; // rdi
  __int64 v26; // rsi
  size_t v27; // r8
  __int64 v28; // rdx
  char v29; // al
  __m128i v30; // xmm4
  __int64 v31; // rax
  __int64 v33; // rax
  size_t v34; // rdx
  size_t v35; // rdx
  __int64 v36; // [rsp+0h] [rbp-F0h]
  unsigned __int8 v38; // [rsp+1Fh] [rbp-D1h]
  __m128i v39; // [rsp+20h] [rbp-D0h] BYREF
  char v40; // [rsp+30h] [rbp-C0h]
  __m128i v41; // [rsp+40h] [rbp-B0h] BYREF
  char v42; // [rsp+50h] [rbp-A0h]
  __m128i v43; // [rsp+60h] [rbp-90h] BYREF
  __int64 v44; // [rsp+70h] [rbp-80h]
  __m128i v45; // [rsp+80h] [rbp-70h] BYREF
  __int64 v46; // [rsp+90h] [rbp-60h]
  __m128i *v47; // [rsp+A0h] [rbp-50h] BYREF
  size_t n; // [rsp+A8h] [rbp-48h]
  _QWORD src[8]; // [rsp+B0h] [rbp-40h] BYREF

  v10 = a2;
  v13 = *(_QWORD *)(a1 + 8);
  v45.m128i_i32[0] = a6;
  v14 = a8;
  v44 = a10;
  v40 = a8;
  v15 = v13 + 1736;
  v38 = a10;
  v16 = *(_QWORD *)(v13 + 1744);
  v39 = _mm_loadu_si128((const __m128i *)&a7);
  v43 = _mm_loadu_si128((const __m128i *)&a9);
  if ( !v16 )
    goto LABEL_21;
  do
  {
    while ( 1 )
    {
      v17 = *(_QWORD *)(v16 + 16);
      v18 = *(_QWORD *)(v16 + 24);
      if ( a6 <= *(_DWORD *)(v16 + 32) )
        break;
      v16 = *(_QWORD *)(v16 + 24);
      if ( !v18 )
        goto LABEL_6;
    }
    v15 = v16;
    v16 = *(_QWORD *)(v16 + 16);
  }
  while ( v17 );
LABEL_6:
  if ( v13 + 1736 == v15 || a6 < *(_DWORD *)(v15 + 32) )
  {
LABEL_21:
    v36 = a3;
    v47 = &v45;
    v33 = sub_E9E2A0((_QWORD *)(v13 + 1728), v15, (unsigned int **)&v47);
    a3 = v36;
    v10 = a2;
    v15 = v33;
  }
  v19 = _mm_loadu_si128(&v39);
  v47 = (__m128i *)src;
  v20 = _mm_loadu_si128(&v43);
  v42 = v40;
  v41 = v19;
  v46 = v44;
  v45 = v20;
  sub_E97AA0((__int64 *)&v47, v10, (__int64)&v10[a3]);
  v21 = *(__m128i **)(v15 + 440);
  if ( v47 == (__m128i *)src )
  {
    v34 = n;
    if ( n )
    {
      if ( n == 1 )
        v21->m128i_i8[0] = src[0];
      else
        memcpy(v21, src, n);
      v34 = n;
      v21 = *(__m128i **)(v15 + 440);
    }
    *(_QWORD *)(v15 + 448) = v34;
    v21->m128i_i8[v34] = 0;
    v21 = v47;
  }
  else
  {
    v22 = src[0];
    v23 = n;
    if ( v21 == (__m128i *)(v15 + 456) )
    {
      *(_QWORD *)(v15 + 440) = v47;
      *(_QWORD *)(v15 + 448) = v23;
      *(_QWORD *)(v15 + 456) = v22;
    }
    else
    {
      v24 = *(_QWORD *)(v15 + 456);
      *(_QWORD *)(v15 + 440) = v47;
      *(_QWORD *)(v15 + 448) = v23;
      *(_QWORD *)(v15 + 456) = v22;
      if ( v21 )
      {
        v47 = v21;
        src[0] = v24;
        goto LABEL_12;
      }
    }
    v47 = (__m128i *)src;
    v21 = (__m128i *)src;
  }
LABEL_12:
  n = 0;
  v21->m128i_i8[0] = 0;
  if ( v47 != (__m128i *)src )
    j_j___libc_free_0(v47, src[0] + 1LL);
  v47 = (__m128i *)src;
  sub_E97AA0((__int64 *)&v47, a4, (__int64)&a4[a5]);
  v25 = *(__m128i **)(v15 + 472);
  if ( v47 == (__m128i *)src )
  {
    v35 = n;
    if ( n )
    {
      if ( n == 1 )
        v25->m128i_i8[0] = src[0];
      else
        memcpy(v25, src, n);
      v35 = n;
      v25 = *(__m128i **)(v15 + 472);
    }
    *(_QWORD *)(v15 + 480) = v35;
    v25->m128i_i8[v35] = 0;
    v25 = v47;
  }
  else
  {
    v26 = src[0];
    v27 = n;
    if ( v25 == (__m128i *)(v15 + 488) )
    {
      *(_QWORD *)(v15 + 472) = v47;
      *(_QWORD *)(v15 + 480) = v27;
      *(_QWORD *)(v15 + 488) = v26;
    }
    else
    {
      v28 = *(_QWORD *)(v15 + 488);
      *(_QWORD *)(v15 + 472) = v47;
      *(_QWORD *)(v15 + 480) = v27;
      *(_QWORD *)(v15 + 488) = v26;
      if ( v25 )
      {
        v47 = v25;
        src[0] = v28;
        goto LABEL_18;
      }
    }
    v47 = (__m128i *)src;
    v25 = (__m128i *)src;
  }
LABEL_18:
  n = 0;
  v25->m128i_i8[0] = 0;
  if ( v47 != (__m128i *)src )
    j_j___libc_free_0(v47, src[0] + 1LL);
  v29 = v42;
  v30 = _mm_loadu_si128(&v41);
  *(_DWORD *)(v15 + 504) = 0;
  *(_BYTE *)(v15 + 524) = v29;
  *(__m128i *)(v15 + 508) = v30;
  *(__m128i *)(v15 + 528) = _mm_loadu_si128(&v45);
  v31 = v46;
  *(_BYTE *)(v15 + 553) &= v14;
  *(_QWORD *)(v15 + 544) = v31;
  *(_BYTE *)(v15 + 554) |= v14;
  *(_BYTE *)(v15 + 552) |= v38;
  return v38;
}
