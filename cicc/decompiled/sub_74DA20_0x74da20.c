// Function: sub_74DA20
// Address: 0x74da20
//
__int64 __fastcall sub_74DA20(
        _QWORD *a1,
        const __m128i *a2,
        void (__fastcall **a3)(char *, _QWORD, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, _QWORD, __int64, __int64, __int64))
{
  __m128i v4; // xmm1
  __int64 v5; // rax
  __m128i v6; // xmm0
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 result; // rax
  __m128i v12; // [rsp+0h] [rbp-E0h] BYREF
  __m128i v13; // [rsp+20h] [rbp-C0h]
  __m128i v14; // [rsp+30h] [rbp-B0h]
  __m128i v15; // [rsp+40h] [rbp-A0h]
  __m128i v16; // [rsp+50h] [rbp-90h]
  __m128i v17; // [rsp+60h] [rbp-80h]
  __m128i v18; // [rsp+70h] [rbp-70h]
  __m128i v19; // [rsp+90h] [rbp-50h]
  __m128i v20; // [rsp+B0h] [rbp-30h]

  if ( *a1 )
  {
    sub_74DA20();
    v4 = _mm_loadu_si128(a2 + 1);
    v12 = _mm_loadu_si128(a2);
    v5 = a1[2];
    v6 = _mm_loadu_si128(a2 + 8);
    v13 = _mm_loadu_si128(a2 + 2);
    v14 = _mm_loadu_si128(a2 + 3);
    v15 = _mm_loadu_si128(a2 + 4);
    v16 = _mm_loadu_si128(a2 + 5);
    v17 = _mm_loadu_si128(a2 + 6);
    v18 = _mm_loadu_si128(a2 + 7);
    v19 = _mm_loadu_si128(a2 + 9);
    v20 = _mm_loadu_si128(a2 + 11);
    (*a3)(
      "(",
      a3,
      v7,
      v8,
      v9,
      v10,
      v12.m128i_i64[0],
      v12.m128i_i64[1],
      v4.m128i_i64[0],
      v4.m128i_i64[1],
      v13.m128i_i64[0],
      v13.m128i_i64[1],
      v14.m128i_i64[0],
      v14.m128i_i64[1],
      v15.m128i_i64[0],
      v15.m128i_i64[1],
      v16.m128i_i64[0],
      v16.m128i_i64[1],
      v17.m128i_i64[0],
      v17.m128i_i64[1],
      v18.m128i_i64[0],
      v18.m128i_i64[1],
      v6.m128i_i64[0],
      v6.m128i_i64[1],
      v19.m128i_i64[0],
      v19.m128i_i64[1],
      *(_QWORD *)(v5 + 40),
      _mm_loadu_si128(a2 + 10).m128i_i64[1],
      v20.m128i_i64[0],
      v20.m128i_i64[1]);
    sub_74B930((__int64)&v12, (__int64)a3);
    return ((__int64 (__fastcall *)(char *, void (__fastcall **)(char *, _QWORD, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, _QWORD, __int64, __int64, __int64)))*a3)(
             ")",
             a3);
  }
  return result;
}
