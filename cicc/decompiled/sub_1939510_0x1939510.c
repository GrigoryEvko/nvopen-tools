// Function: sub_1939510
// Address: 0x1939510
//
__int64 __fastcall sub_1939510(const __m128i *a1)
{
  const __m128i *v1; // r14
  __int64 v2; // r13
  __int64 v3; // r15
  __m128i v4; // xmm1
  __int64 v5; // rax
  __int64 *v6; // rbx
  __int64 v8; // [rsp+0h] [rbp-40h]
  __int64 v9; // [rsp+8h] [rbp-38h]

  v1 = a1;
  v2 = a1->m128i_i64[1];
  v3 = a1->m128i_i64[0];
  v9 = a1[1].m128i_i64[0];
  v8 = a1[1].m128i_i64[1];
  while ( 1 )
  {
    v5 = v1[-2].m128i_i64[1];
    v6 = (__int64 *)v1;
    v1 -= 2;
    if ( (int)sub_16AEA10(v2 + 24, v5 + 24) >= 0 )
      break;
    v4 = _mm_loadu_si128(v1 + 1);
    v1[2] = _mm_loadu_si128(v1);
    v1[3] = v4;
  }
  *v6 = v3;
  v6[1] = v2;
  v6[2] = v9;
  v6[3] = v8;
  return v8;
}
