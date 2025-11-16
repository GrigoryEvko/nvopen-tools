// Function: sub_1010290
// Address: 0x1010290
//
__int64 __fastcall sub_1010290(unsigned int a1, _BYTE *a2, unsigned __int8 *a3, const __m128i *a4, int a5)
{
  __int64 result; // rax
  int v8; // r13d
  __m128i v9; // xmm0
  __m128i v10; // xmm1
  __m128i v11; // xmm2
  __m128i v12; // xmm3
  unsigned __int8 *v13; // rax
  __m128i v14; // xmm4
  __m128i v15; // xmm5
  __m128i v16; // xmm6
  __m128i v17; // xmm7
  unsigned __int8 *v18; // r15
  __int64 v19; // [rsp+8h] [rbp-B8h]
  __int64 v20; // [rsp+10h] [rbp-B0h]
  unsigned __int8 *v22; // [rsp+28h] [rbp-98h]
  __int64 v23; // [rsp+30h] [rbp-90h] BYREF
  __int64 v24; // [rsp+38h] [rbp-88h]
  __m128i v25; // [rsp+40h] [rbp-80h] BYREF
  __m128i v26; // [rsp+50h] [rbp-70h]
  __m128i v27; // [rsp+60h] [rbp-60h]
  __m128i v28; // [rsp+70h] [rbp-50h]
  __int64 v29; // [rsp+80h] [rbp-40h]

  if ( *a2 != 82 )
    return 0;
  v20 = *((_QWORD *)a2 - 8);
  if ( !v20 )
    return 0;
  v19 = *((_QWORD *)a2 - 4);
  if ( !v19 )
    return 0;
  v8 = sub_B53900((__int64)a2);
  if ( (unsigned int)(v8 - 32) > 1 )
    return 0;
  v9 = _mm_loadu_si128(a4);
  v10 = _mm_loadu_si128(a4 + 1);
  v11 = _mm_loadu_si128(a4 + 2);
  v29 = a4[4].m128i_i64[0];
  v12 = _mm_loadu_si128(a4 + 3);
  BYTE1(v29) = 0;
  v23 = v20;
  v24 = v19;
  v25 = v9;
  v26 = v10;
  v27 = v11;
  v28 = v12;
  v13 = sub_100F630(a3, (__int64)&v23, 1, &v25, 1u, 0, a5);
  if ( v13 )
  {
    v22 = v13;
    result = sub_AD6840(a1, *((_QWORD *)v13 + 1), 0);
    if ( (a1 != 28) + 32 != v8 )
    {
      if ( v22 == (unsigned __int8 *)result )
        return (__int64)a3;
      return 0;
    }
    if ( v22 != (unsigned __int8 *)result )
    {
      if ( v22 == sub_AD93D0(a1, *((_QWORD *)v22 + 1), 0, 0) )
        return (__int64)a2;
      return 0;
    }
  }
  else
  {
    v14 = _mm_loadu_si128(a4);
    v15 = _mm_loadu_si128(a4 + 1);
    v16 = _mm_loadu_si128(a4 + 2);
    v29 = a4[4].m128i_i64[0];
    v17 = _mm_loadu_si128(a4 + 3);
    BYTE1(v29) = 0;
    v23 = v19;
    v25 = v14;
    v24 = v20;
    v26 = v15;
    v27 = v16;
    v28 = v17;
    v18 = sub_100F630(a3, (__int64)&v23, 1, &v25, 1u, 0, a5);
    if ( !v18 )
      return 0;
    result = sub_AD6840(a1, *((_QWORD *)v18 + 1), 0);
    if ( (a1 != 28) + 32 != v8 )
    {
      if ( v18 == (unsigned __int8 *)result )
        return (__int64)a3;
      return 0;
    }
    if ( v18 != (unsigned __int8 *)result )
    {
      if ( v18 == sub_AD93D0(a1, *((_QWORD *)v18 + 1), 0, 0) )
        return (__int64)a2;
      return 0;
    }
  }
  return result;
}
