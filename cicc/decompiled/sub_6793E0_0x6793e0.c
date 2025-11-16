// Function: sub_6793E0
// Address: 0x6793e0
//
__int64 *__fastcall sub_6793E0(__int64 a1, __int64 *a2, int a3, int a4, __int64 a5)
{
  __int64 v7; // rbx
  __m128i v8; // xmm0
  __int64 *result; // rax
  __int64 *v10; // rdx
  __m128i v11; // [rsp+0h] [rbp-50h] BYREF
  __m128i v12[4]; // [rsp+10h] [rbp-40h] BYREF

  sub_6790F0((__int64)&v11, 0, a3, a4, 1);
  if ( !a2 || !a1 )
    return (__int64 *)sub_7AEA70(&v11);
  v7 = qword_4CFDE88;
  if ( qword_4CFDE88 )
    qword_4CFDE88 = *(_QWORD *)qword_4CFDE88;
  else
    v7 = sub_823970(64);
  *(_QWORD *)v7 = 0;
  *(_QWORD *)(v7 + 48) = 0;
  sub_879020(v7 + 8, 0);
  v8 = _mm_loadu_si128(&v11);
  *(_QWORD *)(v7 + 48) = a1;
  *(_QWORD *)(v7 + 56) = a5;
  *(__m128i *)(v7 + 8) = v8;
  *(__m128i *)(v7 + 24) = _mm_loadu_si128(v12);
  result = (__int64 *)*a2;
  if ( *a2 )
  {
    do
    {
      v10 = result;
      result = (__int64 *)*result;
    }
    while ( result );
    *v10 = v7;
  }
  else
  {
    *a2 = v7;
  }
  if ( a3 )
  {
    *(_BYTE *)(a1 + 32) |= 0x10u;
    *(_QWORD *)(a1 + 48) = a1;
  }
  return result;
}
