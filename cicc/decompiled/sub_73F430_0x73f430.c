// Function: sub_73F430
// Address: 0x73f430
//
__m128i *__fastcall sub_73F430(const __m128i *a1, int a2)
{
  __m128i *v2; // r13
  __int64 v3; // rax

  v2 = (__m128i *)a1;
  if ( *(_QWORD *)(a1[10].m128i_i64[1] + 40) )
  {
    v2 = (__m128i *)sub_7259C0(7);
    sub_73BCD0(a1, v2, a2);
    v3 = v2[10].m128i_i64[1];
    *(_BYTE *)(v3 + 21) &= ~1u;
    *(_QWORD *)(v3 + 40) = 0;
  }
  return v2;
}
