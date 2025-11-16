// Function: sub_253B210
// Address: 0x253b210
//
__m128i *__fastcall sub_253B210(__int64 a1, __m128i *a2)
{
  __m128i *result; // rax
  __int64 v3; // rdx

  result = (__m128i *)sub_253A680(a1, (unsigned __int64 *)a2);
  if ( v3 )
    return sub_253B140(a1, (__int64)result, v3, a2);
  return result;
}
