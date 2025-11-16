// Function: sub_39CAEF0
// Address: 0x39caef0
//
__int64 __fastcall sub_39CAEF0(__int64 *a1, __int64 a2, __int16 a3, __int64 a4)
{
  __int64 v7; // rdi
  __m128i *v8; // rsi
  unsigned int v9; // eax
  __int64 v11; // [rsp+8h] [rbp-38h]
  __m128i v12; // [rsp+10h] [rbp-30h] BYREF

  v7 = a1[25];
  if ( !*(_BYTE *)(v7 + 4513) || !a1[77] )
    return sub_39CADF0((__int64)a1, a2, a3, a4);
  if ( a4 )
  {
    v12.m128i_i64[0] = a4;
    v12.m128i_i64[1] = (__int64)a1;
    v8 = *(__m128i **)(v7 + 616);
    if ( v8 == *(__m128i **)(v7 + 624) )
    {
      v11 = a4;
      sub_39CAC70((unsigned __int64 *)(v7 + 608), v8, &v12);
      a4 = v11;
    }
    else
    {
      if ( v8 )
      {
        *v8 = _mm_loadu_si128(&v12);
        v8 = *(__m128i **)(v7 + 616);
      }
      *(_QWORD *)(v7 + 616) = v8 + 1;
    }
    v7 = a1[25];
  }
  v9 = sub_39BFF80(v7 + 5512, a4, 0);
  v12.m128i_i16[3] = 7937;
  v12.m128i_i16[2] = a3;
  v12.m128i_i32[0] = 1;
  v12.m128i_i64[1] = v9;
  return sub_39A31C0((__int64 *)(a2 + 8), a1 + 11, v12.m128i_i64);
}
