// Function: sub_7F80B0
// Address: 0x7f80b0
//
__m128i *__fastcall sub_7F80B0(char *a1, __m128i **a2, __int64 a3)
{
  __m128i *result; // rax
  __m128i *v4; // r12

  result = *a2;
  if ( !*a2 )
  {
    v4 = sub_7F7840(a1, 1, a3, 0);
    sub_7362F0((__int64)v4, 0);
    *a2 = v4;
    *(_BYTE *)(*(_QWORD *)(v4[9].m128i_i64[1] + 168) + 16LL) &= ~2u;
    return *a2;
  }
  return result;
}
