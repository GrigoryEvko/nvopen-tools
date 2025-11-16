// Function: sub_2ED6530
// Address: 0x2ed6530
//
__m128i *__fastcall sub_2ED6530(__int64 a1, __m128i *a2)
{
  __m128i *result; // rax
  _QWORD *v3; // rdx

  result = (__m128i *)sub_2E64B40(a1, (unsigned __int64 *)a2);
  if ( v3 )
    return sub_2ED16E0(a1, (__int64)result, v3, a2);
  return result;
}
