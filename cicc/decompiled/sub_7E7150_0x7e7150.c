// Function: sub_7E7150
// Address: 0x7e7150
//
__int64 __fastcall sub_7E7150(__m128i *a1, __int64 a2, __m128i **a3)
{
  __m128i *v4; // rax
  __int32 v5; // ecx
  __int16 v6; // dx
  __int64 result; // rax

  sub_7E7090(a1, a2, a3);
  v4 = *a3;
  a1->m128i_i64[0] = (*a3)->m128i_i64[0];
  v5 = v4->m128i_i32[2];
  v6 = v4->m128i_i16[6];
  result = a1[5].m128i_i64[0];
  a1->m128i_i32[2] = v5;
  a1->m128i_i16[6] = v6;
  *(_DWORD *)result = v5;
  *(_WORD *)(result + 4) = v6;
  return result;
}
