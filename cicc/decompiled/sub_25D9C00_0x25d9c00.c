// Function: sub_25D9C00
// Address: 0x25d9c00
//
__m128i *__fastcall sub_25D9C00(__int64 a1, __m128i *a2)
{
  __m128i *result; // rax
  _QWORD *v3; // rdx

  result = (__m128i *)sub_25D9B40(a1, (unsigned __int64 *)a2);
  if ( v3 )
    return sub_25D6B40(a1, (__int64)result, v3, a2);
  return result;
}
