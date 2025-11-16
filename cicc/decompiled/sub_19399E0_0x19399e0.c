// Function: sub_19399E0
// Address: 0x19399e0
//
void __fastcall sub_19399E0(__m128i *src, const __m128i *a2)
{
  const __m128i *i; // rbx
  const __m128i *v4; // rdi
  __int64 v5; // r13
  __int64 v6; // r9
  __int64 v7; // r8
  __int64 v8; // r15
  __int64 v9; // [rsp-48h] [rbp-48h]
  __int64 v10; // [rsp-40h] [rbp-40h]

  if ( src != a2 )
  {
    for ( i = src + 2; a2 != i; src[1].m128i_i64[1] = v8 )
    {
      while ( 1 )
      {
        v5 = i->m128i_i64[1];
        if ( (int)sub_16AEA10(v5 + 24, src->m128i_i64[1] + 24) < 0 )
          break;
        v4 = i;
        i += 2;
        sub_1939510(v4);
        if ( a2 == i )
          return;
      }
      v6 = i->m128i_i64[0];
      v7 = i[1].m128i_i64[0];
      v8 = i[1].m128i_i64[1];
      if ( src != i )
      {
        v9 = i[1].m128i_i64[0];
        v10 = i->m128i_i64[0];
        memmove(&src[2], src, (char *)i - (char *)src);
        v7 = v9;
        v6 = v10;
      }
      i += 2;
      src->m128i_i64[0] = v6;
      src->m128i_i64[1] = v5;
      src[1].m128i_i64[0] = v7;
    }
  }
}
