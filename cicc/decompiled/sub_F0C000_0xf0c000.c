// Function: sub_F0C000
// Address: 0xf0c000
//
__int64 __fastcall sub_F0C000(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  unsigned int v5; // r12d
  __m128i v6; // xmm1
  unsigned __int64 v7; // xmm2_8
  __m128i v8; // xmm3
  __int64 v9; // rax
  __m128i v11; // xmm5
  unsigned __int64 v12; // xmm6_8
  __m128i v13; // xmm7
  __int64 v14; // rax
  unsigned __int64 v15; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v16; // [rsp+8h] [rbp-B8h]
  unsigned int v17; // [rsp+10h] [rbp-B0h]
  __int64 v18; // [rsp+18h] [rbp-A8h]
  unsigned int v19; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v20; // [rsp+30h] [rbp-90h] BYREF
  __int64 v21; // [rsp+38h] [rbp-88h]
  unsigned int v22; // [rsp+40h] [rbp-80h]
  __int64 v23; // [rsp+48h] [rbp-78h]
  unsigned int v24; // [rsp+50h] [rbp-70h]
  __m128i v25; // [rsp+60h] [rbp-60h] BYREF
  __m128i v26; // [rsp+70h] [rbp-50h]
  unsigned __int64 v27; // [rsp+80h] [rbp-40h]
  __int64 v28; // [rsp+88h] [rbp-38h]
  __m128i v29; // [rsp+90h] [rbp-30h]
  __int64 v30; // [rsp+A0h] [rbp-20h]

  v20 = a3 & 0xFFFFFFFFFFFFFFFBLL;
  v22 = 1;
  v21 = 0;
  v24 = 1;
  v23 = 0;
  v15 = a2 & 0xFFFFFFFFFFFFFFFBLL;
  v17 = 1;
  v16 = 0;
  v19 = 1;
  v18 = 0;
  if ( a5 )
  {
    v6 = _mm_loadu_si128(a1 + 7);
    v7 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
    v8 = _mm_loadu_si128(a1 + 9);
    v9 = a1[10].m128i_i64[0];
    v25 = _mm_loadu_si128(a1 + 6);
    v27 = v7;
    v30 = v9;
    v28 = a4;
    v26 = v6;
    v29 = v8;
    v5 = sub_9B0100((__int64 *)&v15, (__int64 *)&v20, &v25);
    if ( v19 > 0x40 && v18 )
      j_j___libc_free_0_0(v18);
    if ( v17 > 0x40 && v16 )
      j_j___libc_free_0_0(v16);
    if ( v24 > 0x40 && v23 )
      j_j___libc_free_0_0(v23);
    if ( v22 > 0x40 && v21 )
      j_j___libc_free_0_0(v21);
    LOBYTE(v5) = v5 == 3;
  }
  else
  {
    v11 = _mm_loadu_si128(a1 + 7);
    v12 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
    v13 = _mm_loadu_si128(a1 + 9);
    v14 = a1[10].m128i_i64[0];
    v25 = _mm_loadu_si128(a1 + 6);
    v27 = v12;
    v30 = v14;
    v28 = a4;
    v26 = v11;
    v29 = v13;
    LOBYTE(v5) = (unsigned int)sub_9AC900((__int64 *)&v15, (__int64 *)&v20, &v25) == 3;
    if ( v19 > 0x40 && v18 )
      j_j___libc_free_0_0(v18);
    if ( v17 > 0x40 && v16 )
      j_j___libc_free_0_0(v16);
    if ( v24 > 0x40 && v23 )
      j_j___libc_free_0_0(v23);
    if ( v22 > 0x40 && v21 )
      j_j___libc_free_0_0(v21);
  }
  return v5;
}
