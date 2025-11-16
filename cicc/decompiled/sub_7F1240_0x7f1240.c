// Function: sub_7F1240
// Address: 0x7f1240
//
__m128i *__fastcall sub_7F1240(__m128i *a1, __int64 a2)
{
  __m128i *v2; // r12
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // rdi

  v2 = a1;
  if ( (unsigned int)sub_8D29A0(a2) )
  {
    v2 = (__m128i *)sub_73DBF0(0x14u, a2, (__int64)a1);
    sub_7F1170(v2);
    if ( dword_4F077C4 == 2 )
      sub_7E2350(v2);
  }
  else
  {
    v5 = a1->m128i_i64[0];
    if ( a2 != v2->m128i_i64[0] && !(unsigned int)sub_8D97D0(v5, a2, 1, v3, v4) )
    {
      v2 = (__m128i *)sub_73E110((__int64)v2, a2);
      if ( dword_4F077C4 != 2 )
        sub_7D8AA0(v2->m128i_i64);
    }
  }
  return v2;
}
