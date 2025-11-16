// Function: sub_735FB0
// Address: 0x735fb0
//
__m128i *__fastcall sub_735FB0(__int64 a1, __int8 a2, int a3)
{
  __m128i *v4; // rax
  __m128i *v5; // r12

  v4 = sub_725D10(a2);
  v4[7].m128i_i64[1] = a1;
  v5 = v4;
  if ( a3 != -1 )
    sub_735E40((__int64)v4, a3);
  return v5;
}
