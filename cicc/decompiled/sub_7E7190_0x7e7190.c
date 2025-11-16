// Function: sub_7E7190
// Address: 0x7e7190
//
__int64 __fastcall sub_7E7190(const __m128i *a1, __int64 a2, __m128i **a3)
{
  __int64 result; // rax

  sub_7E7090(a1, a2, a3);
  result = a1[5].m128i_i64[0];
  *(_BYTE *)(result + 24) &= ~1u;
  return result;
}
