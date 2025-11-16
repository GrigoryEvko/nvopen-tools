// Function: sub_1E1EE50
// Address: 0x1e1ee50
//
__m128i *__fastcall sub_1E1EE50(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rdx
  unsigned __int64 v3; // r12
  __m128i *result; // rax
  const __m128i *v5; // rcx
  const __m128i *v6; // rdx
  __m128i *v7; // rcx

  sub_16CCCB0(a1, (__int64)(a1 + 5), a2);
  v3 = *(_QWORD *)(a2 + 112) - *(_QWORD *)(a2 + 104);
  a1[13] = 0;
  a1[14] = 0;
  a1[15] = 0;
  if ( v3 )
  {
    if ( v3 > 0x7FFFFFFFFFFFFFF0LL )
      sub_4261EA(a1, a1 + 5, v2);
    result = (__m128i *)sub_22077B0(v3);
  }
  else
  {
    v3 = 0;
    result = 0;
  }
  a1[13] = result;
  a1[14] = result;
  a1[15] = (char *)result + v3;
  v5 = *(const __m128i **)(a2 + 112);
  v6 = *(const __m128i **)(a2 + 104);
  if ( v5 == v6 )
  {
    a1[14] = result;
  }
  else
  {
    v7 = (__m128i *)((char *)result + (char *)v5 - (char *)v6);
    do
    {
      if ( result )
        *result = _mm_loadu_si128(v6);
      ++result;
      ++v6;
    }
    while ( result != v7 );
    a1[14] = v7;
  }
  return result;
}
