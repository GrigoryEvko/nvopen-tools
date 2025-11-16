// Function: sub_25D6B40
// Address: 0x25d6b40
//
__m128i *__fastcall sub_25D6B40(__int64 a1, __int64 a2, _QWORD *a3, const __m128i *a4)
{
  bool v4; // r8
  _QWORD *v5; // r15
  __m128i *v8; // r12
  unsigned __int64 v10; // rax
  char v11; // [rsp+Ch] [rbp-34h]

  v4 = 1;
  v5 = (_QWORD *)(a1 + 8);
  if ( !a2 && a3 != v5 )
  {
    v10 = a3[4];
    if ( a4->m128i_i64[0] >= v10 )
    {
      v4 = 0;
      if ( a4->m128i_i64[0] == v10 )
        v4 = a4->m128i_i64[1] < a3[5];
    }
  }
  v11 = v4;
  v8 = (__m128i *)sub_22077B0(0x30u);
  v8[2] = _mm_loadu_si128(a4);
  sub_220F040(v11, (__int64)v8, a3, v5);
  ++*(_QWORD *)(a1 + 40);
  return v8;
}
