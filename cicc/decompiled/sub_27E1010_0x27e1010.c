// Function: sub_27E1010
// Address: 0x27e1010
//
__m128i *__fastcall sub_27E1010(__int64 a1, __m128i *a2)
{
  __m128i *result; // rax
  _QWORD *v3; // rdx

  result = (__m128i *)sub_27E0F50(a1, (unsigned __int64 *)a2);
  if ( v3 )
    return sub_27DB8C0(a1, (__int64)result, v3, a2);
  return result;
}
