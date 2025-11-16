// Function: sub_7F8570
// Address: 0x7f8570
//
__m128i *__fastcall sub_7F8570(__m128i *a1, __int64 a2)
{
  __int64 i; // r12
  __m128i *result; // rax
  __int64 v4; // rsi
  __int64 v5; // r12
  __m128i v6; // xmm4
  __m128i v7; // xmm3
  __m128i v8; // xmm2
  __m128i v9; // xmm6
  __m128i v10; // xmm1
  __m128i v11; // xmm0
  __int64 v12; // rdx
  __m128i v13; // xmm7
  __m128i v14; // xmm5
  __m128i v15; // xmm6
  __int64 v16; // rcx
  __m128i *v17; // [rsp+8h] [rbp-78h] BYREF
  __m128i v18; // [rsp+10h] [rbp-70h]
  __m128i v19; // [rsp+20h] [rbp-60h]
  __m128i v20; // [rsp+30h] [rbp-50h]
  __m128i v21; // [rsp+40h] [rbp-40h]
  __m128i v22; // [rsp+50h] [rbp-30h]
  __int64 v23; // [rsp+60h] [rbp-20h]

  for ( i = a1->m128i_i64[0]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( (unsigned int)sub_8D2D50(i) )
  {
    result = (__m128i *)sub_8D6740(i);
    v4 = (__int64)result;
    if ( result != (__m128i *)i )
    {
      if ( !result
        || !dword_4F07588
        || (result = (__m128i *)result[2].m128i_i64[0], *(__m128i **)(i + 32) != result)
        || !result )
      {
        v5 = a1[1].m128i_i64[0];
        v17 = a1;
        sub_6E8160((__int64 *)&v17, v4, 0, 0, 1, 0, 0, 0, dword_4F07508);
        result = v17;
        v17[1].m128i_i64[0] = v5;
        if ( result != a1 )
        {
          v6 = _mm_loadu_si128(a1);
          v7 = _mm_loadu_si128(a1 + 1);
          v8 = _mm_loadu_si128(a1 + 2);
          *a1 = _mm_loadu_si128(result);
          v9 = _mm_loadu_si128(result + 1);
          v10 = _mm_loadu_si128(a1 + 3);
          v11 = _mm_loadu_si128(a1 + 4);
          v12 = a1[5].m128i_i64[0];
          v18 = v6;
          a1[1] = v9;
          v13 = _mm_loadu_si128(result + 2);
          v23 = v12;
          a1[2] = v13;
          v14 = _mm_loadu_si128(result + 3);
          v19 = v7;
          a1[3] = v14;
          v15 = _mm_loadu_si128(result + 4);
          v20 = v8;
          a1[4] = v15;
          v16 = result[5].m128i_i64[0];
          v21 = v10;
          a1[5].m128i_i64[0] = v16;
          result[5].m128i_i64[0] = v12;
          *result = v6;
          result[1] = v7;
          result[2] = v8;
          result[3] = v10;
          result[4] = v11;
          result = v17;
          v22 = v11;
          a1[4].m128i_i64[1] = (__int64)v17;
        }
      }
    }
  }
  else
  {
    result = (__m128i *)sub_7E1F40(i);
    if ( (_DWORD)result )
      return (__m128i *)sub_7F8400(a1, a2);
  }
  return result;
}
