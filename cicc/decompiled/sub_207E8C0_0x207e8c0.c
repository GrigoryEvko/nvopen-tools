// Function: sub_207E8C0
// Address: 0x207e8c0
//
__int64 __fastcall sub_207E8C0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        unsigned int a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        __m128i a9,
        __int128 a10,
        unsigned __int8 a11)
{
  __int64 v14; // rax
  __int64 *v15; // r15
  __int32 v16; // edx
  __m128i *v17; // rsi
  unsigned __int64 v18; // rax
  __int32 v19; // eax
  __m128i *v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // rsi
  __int64 v23; // rsi
  __int64 *v24; // rax
  __m128i v25; // xmm2
  const __m128i *v26; // rsi
  int v27; // edx
  unsigned __int64 v28; // rax
  unsigned int v29; // edx
  __m128i *v30; // rcx
  __m128i *v31; // rdx
  const __m128i *v32; // rcx
  __int64 v33; // rdi
  __int64 v34; // rdx
  __int64 v35; // rsi
  __int64 result; // rax
  unsigned __int8 v38; // [rsp+18h] [rbp-C8h]
  unsigned int i; // [rsp+1Ch] [rbp-C4h]
  __int64 v40; // [rsp+58h] [rbp-88h] BYREF
  const __m128i *v41; // [rsp+60h] [rbp-80h] BYREF
  __m128i *v42; // [rsp+68h] [rbp-78h]
  const __m128i *v43; // [rsp+70h] [rbp-70h]
  __m128i v44; // [rsp+80h] [rbp-60h] BYREF
  __m128i v45; // [rsp+90h] [rbp-50h] BYREF
  __int64 v46; // [rsp+A0h] [rbp-40h]

  v40 = a3;
  v38 = a11;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  sub_1FD3FA0(&v41, a5);
  for ( i = a5 + a4; i != a4; v42 = (__m128i *)((char *)v17 + 40) )
  {
    while ( 1 )
    {
      v18 = v40 & 0xFFFFFFFFFFFFFFF8LL;
      v14 = (*(_BYTE *)((v40 & 0xFFFFFFFFFFFFFFF8LL) + 23) & 0x40) != 0
          ? *(_QWORD *)(v18 - 8)
          : v18 - 24LL * (*(_DWORD *)(v18 + 20) & 0xFFFFFFF);
      v15 = *(__int64 **)(v14 + 24LL * a4);
      v44 = 0u;
      v45 = 0u;
      LODWORD(v46) = 0;
      v44.m128i_i64[1] = (__int64)sub_20685E0(a1, v15, a7, a8, a9);
      v45.m128i_i32[0] = v16;
      v45.m128i_i64[1] = *v15;
      sub_20A1C00(&v44, &v40, a4);
      v17 = v42;
      if ( v42 != v43 )
        break;
      ++a4;
      sub_1D27190(&v41, v42, &v44);
      if ( i == a4 )
        goto LABEL_11;
    }
    if ( v42 )
    {
      a7 = _mm_loadu_si128(&v44);
      *v42 = a7;
      a8 = _mm_loadu_si128(&v45);
      v17[1] = a8;
      v17[2].m128i_i64[0] = v46;
      v17 = v42;
    }
    ++a4;
  }
LABEL_11:
  v19 = *(_DWORD *)(a1 + 536);
  v20 = *(__m128i **)a1;
  v44.m128i_i64[0] = 0;
  v44.m128i_i32[2] = v19;
  if ( !v20 || &v44 == &v20[3] || (v21 = v20[3].m128i_i64[0], (v44.m128i_i64[0] = v21) == 0) )
  {
    v22 = *(_QWORD *)(a2 + 88);
    if ( !v22 )
    {
      *(_QWORD *)(a2 + 88) = v44.m128i_i64[0];
      goto LABEL_19;
    }
    goto LABEL_15;
  }
  sub_1623A60((__int64)&v44, v21, 2);
  v22 = *(_QWORD *)(a2 + 88);
  if ( v22 )
LABEL_15:
    sub_161E7C0(a2 + 88, v22);
  v23 = v44.m128i_i64[0];
  *(_QWORD *)(a2 + 88) = v44.m128i_i64[0];
  if ( v23 )
    sub_1623A60(a2 + 88, v23, 2);
  v19 = v44.m128i_i32[2];
LABEL_19:
  *(_DWORD *)(a2 + 96) = v19;
  v24 = sub_2051C20((__int64 *)a1, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
  v25 = _mm_loadu_si128((const __m128i *)&a10);
  v26 = v41;
  *(_QWORD *)a2 = v24;
  v41 = 0;
  *(_DWORD *)(a2 + 8) = v27;
  v28 = v40 & 0xFFFFFFFFFFFFFFF8LL;
  v29 = *(unsigned __int16 *)((v40 & 0xFFFFFFFFFFFFFFF8LL) + 18);
  *(_QWORD *)(a2 + 16) = a6;
  *(_QWORD *)(a2 + 40) = a10;
  *(_DWORD *)(a2 + 32) = (v29 >> 2) & 0x3FFFDFFF;
  *(_DWORD *)(a2 + 48) = v25.m128i_i32[2];
  v30 = v42;
  v42 = 0;
  v31 = v30;
  *(_QWORD *)(a2 + 64) = v30;
  v32 = v43;
  v43 = 0;
  v33 = *(_QWORD *)(a2 + 56);
  *(_QWORD *)(a2 + 56) = v26;
  *(_DWORD *)(a2 + 28) = -858993459 * (((char *)v31 - (char *)v26) >> 3);
  v34 = *(_QWORD *)(a2 + 72);
  *(_QWORD *)(a2 + 72) = v32;
  if ( v33 )
  {
    j_j___libc_free_0(v33, v34 - v33);
    v28 = v40 & 0xFFFFFFFFFFFFFFF8LL;
  }
  LOBYTE(v28) = *(_QWORD *)(v28 + 8) != 0;
  v35 = v44.m128i_i64[0];
  result = *(_BYTE *)(a2 + 24) & 0x5F | (v38 << 7) | (unsigned int)(32 * v28);
  *(_BYTE *)(a2 + 24) = result;
  if ( v35 )
    result = sub_161E7C0((__int64)&v44, v35);
  if ( v41 )
    return j_j___libc_free_0(v41, (char *)v43 - (char *)v41);
  return result;
}
