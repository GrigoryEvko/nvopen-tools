// Function: sub_827240
// Address: 0x827240
//
void __fastcall sub_827240(__m128i *a1, const __m128i *a2, __m128i *a3, int a4, int a5)
{
  __m128i *i; // r12
  __m128i *v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r12
  __m128i *v11; // rdi

  for ( i = a3; i[8].m128i_i8[12] == 12; i = (__m128i *)i[10].m128i_i64[0] )
    ;
  a1->m128i_i32[2] = 4;
  a1[3] = _mm_loadu_si128(a2);
  a1[4] = _mm_loadu_si128(a2 + 1);
  a1[5] = _mm_loadu_si128(a2 + 2);
  if ( a4 )
  {
    if ( !a5 )
      a1[5].m128i_i8[4] = a1[5].m128i_i8[4] & 0xF9 | (2 * a1[5].m128i_i8[4]) & 4;
    if ( (a2[1].m128i_i16[0] & 0x108) == 0 )
    {
      v10 = sub_8D46C0(i);
      v11 = sub_73D790(*(_QWORD *)(a1[3].m128i_i64[0] + 152));
      if ( (a1[4].m128i_i8[0] & 4) == 0 )
        v11 = sub_73D720(v11);
      if ( (*(_BYTE *)(v10 + 140) & 0xFB) == 8 && (unsigned int)sub_8D5780(v11, v10) )
        a1[5].m128i_i8[4] |= 2u;
    }
  }
  else if ( unk_4D04460 )
  {
    if ( (unsigned __int8)(i[8].m128i_i8[12] - 9) <= 2u )
    {
      if ( a2->m128i_i64[0] )
      {
        if ( i == sub_73D790(*(_QWORD *)(a2->m128i_i64[0] + 152))
          || (v7 = sub_73D790(*(_QWORD *)(a2->m128i_i64[0] + 152)), (unsigned int)sub_8D97D0(i, v7, 32, v8, v9)) )
        {
          a1[4].m128i_i8[0] |= 2u;
        }
      }
    }
  }
}
