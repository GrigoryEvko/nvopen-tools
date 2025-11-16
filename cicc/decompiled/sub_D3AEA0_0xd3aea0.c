// Function: sub_D3AEA0
// Address: 0xd3aea0
//
__m128i *__fastcall sub_D3AEA0(__int64 a1, __m128i *a2)
{
  __m128i *result; // rax
  __int64 v3; // rdx

  result = (__m128i *)sub_D3ADE0(a1, (unsigned __int64 *)a2);
  if ( v3 )
    return sub_D32350(a1, (__int64)result, v3, a2);
  return result;
}
