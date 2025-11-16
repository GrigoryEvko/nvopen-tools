// Function: sub_AF2F00
// Address: 0xaf2f00
//
__int64 __fastcall sub_AF2F00(__m128i *a1, int a2, int a3, __int64 a4, int a5, int a6, __int128 a7, __int64 a8)
{
  __m128i v9; // xmm0
  __int64 result; // rax

  sub_B971C0((_DWORD)a1, a2, 16, a3, a5, a6, 0, 0);
  v9 = _mm_loadu_si128((const __m128i *)&a7);
  a1->m128i_i16[1] = 41;
  result = a8;
  a1[2].m128i_i64[1] = a4;
  a1[2].m128i_i64[0] = result;
  a1[1] = v9;
  return result;
}
