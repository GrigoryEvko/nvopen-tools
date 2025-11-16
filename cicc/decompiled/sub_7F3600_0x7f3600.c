// Function: sub_7F3600
// Address: 0x7f3600
//
void __fastcall sub_7F3600(__m128i *a1, unsigned int a2, unsigned int a3, int a4)
{
  __m128i *v7; // r12
  __m128i *v8; // rdx
  __m128i *v9; // rax
  __m128i *v10; // rbx
  __m128i *v11; // rsi
  __m128i *v12; // rax

  v7 = a1;
  if ( a4 )
  {
    v8 = 0;
    if ( !a1 )
      return;
    while ( 1 )
    {
      v9 = (__m128i *)v7[1].m128i_i64[0];
      v7[1].m128i_i64[0] = (__int64)v8;
      v8 = v7;
      if ( !v9 )
        break;
      v7 = v9;
    }
  }
  else if ( !a1 )
  {
    return;
  }
  v10 = v7;
  do
  {
    while ( (a2 & 1) != 0 )
    {
      a2 >>= 1;
      a3 >>= 1;
      sub_7F2A70(v10, 0);
      v10 = (__m128i *)v10[1].m128i_i64[0];
      if ( !v10 )
        goto LABEL_10;
    }
    a2 >>= 1;
    v11 = (__m128i *)(a3 & 1);
    a3 >>= 1;
    sub_7EE560(v10, v11);
    v10 = (__m128i *)v10[1].m128i_i64[0];
  }
  while ( v10 );
LABEL_10:
  if ( a4 )
  {
    while ( 1 )
    {
      v12 = (__m128i *)v7[1].m128i_i64[0];
      v7[1].m128i_i64[0] = (__int64)v10;
      v10 = v7;
      if ( !v12 )
        break;
      v7 = v12;
    }
  }
}
