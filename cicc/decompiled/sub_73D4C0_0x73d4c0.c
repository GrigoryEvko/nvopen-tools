// Function: sub_73D4C0
// Address: 0x73d4c0
//
__m128i *__fastcall sub_73D4C0(const __m128i *a1, int a2)
{
  const __m128i *v2; // r12
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // rax

  v2 = a1;
  if ( (unsigned int)sub_8D3410(a1) )
  {
    if ( a2
      && (v4 = sub_8D40F0(a1), (v5 = v4) != 0)
      && (*(_BYTE *)(v4 + 140) & 0xFB) == 8
      && (unsigned int)sub_8D4C10(v4, dword_4F077C4 != 2) )
    {
      v6 = sub_73D4C0(v5, dword_4F077C4 == 2);
      return sub_73C420(a1, v6);
    }
    else
    {
      return (__m128i *)v2;
    }
  }
  else
  {
    while ( v2[8].m128i_i8[12] == 12 )
    {
      if ( !(unsigned int)sub_8D4C10(v2, 1) )
        return (__m128i *)v2;
      v2 = (const __m128i *)v2[10].m128i_i64[0];
    }
    return (__m128i *)v2;
  }
}
