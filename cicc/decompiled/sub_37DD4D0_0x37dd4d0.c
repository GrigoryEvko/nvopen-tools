// Function: sub_37DD4D0
// Address: 0x37dd4d0
//
void __fastcall sub_37DD4D0(
        __m128i *a1,
        const __m128i *a2,
        unsigned __int8 (__fastcall *a3)(__m128i *, const __m128i *))
{
  const __m128i *v4; // r13
  unsigned __int8 v7; // al
  const __m128i *v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rdx

  if ( a1 != a2 )
  {
    v4 = a1 + 1;
    while ( a2 != v4 )
    {
      while ( 1 )
      {
        v7 = a3((__m128i *)v4, a1);
        v8 = v4++;
        if ( !v7 )
          break;
        v9 = v4[-1].m128i_i64[0];
        v10 = v4[-1].m128i_i64[1];
        v11 = v8 - a1;
        if ( (char *)v8 - (char *)a1 > 0 )
        {
          do
          {
            v12 = v8[-1].m128i_i64[0];
            --v8;
            v8[1].m128i_i64[0] = v12;
            v8[1].m128i_i32[2] = v8->m128i_i32[2];
            --v11;
          }
          while ( v11 );
        }
        a1->m128i_i64[0] = v9;
        a1->m128i_i32[2] = v10;
        if ( a2 == v4 )
          return;
      }
      sub_37DD460(v8, a3);
    }
  }
}
