// Function: sub_2851C60
// Address: 0x2851c60
//
__m128i *__fastcall sub_2851C60(__int64 a1, __m128i *a2)
{
  __m128i *result; // rax
  __int64 v3; // rdx

  result = (__m128i *)sub_28513D0(a1, (unsigned __int64 *)a2);
  if ( v3 )
    return sub_2851B90(a1, (__int64)result, v3, a2);
  return result;
}
