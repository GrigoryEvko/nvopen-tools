// Function: sub_26D8D40
// Address: 0x26d8d40
//
__m128i *__fastcall sub_26D8D40(__int64 a1, __m128i *a2)
{
  __m128i *result; // rax
  _QWORD *v3; // rdx
  _QWORD *v4; // r13
  bool v5; // r8
  __m128i *v6; // rbx
  unsigned __int64 v7; // rax
  char v8; // [rsp+Ch] [rbp-34h]

  result = (__m128i *)sub_26D8C80(a1, (unsigned __int64 *)a2);
  if ( v3 )
  {
    v4 = v3;
    v5 = 1;
    if ( !result && v3 != (_QWORD *)(a1 + 8) )
    {
      v7 = v3[4];
      if ( a2->m128i_i64[0] >= v7 )
      {
        v5 = 0;
        if ( a2->m128i_i64[0] == v7 )
          v5 = a2->m128i_i64[1] < v3[5];
      }
    }
    v8 = v5;
    v6 = (__m128i *)sub_22077B0(0x30u);
    v6[2] = _mm_loadu_si128(a2);
    sub_220F040(v8, (__int64)v6, v4, (_QWORD *)(a1 + 8));
    ++*(_QWORD *)(a1 + 40);
    return v6;
  }
  return result;
}
