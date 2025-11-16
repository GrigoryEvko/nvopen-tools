// Function: sub_1E44A30
// Address: 0x1e44a30
//
void __fastcall sub_1E44A30(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rsi
  __int64 v7; // r15
  __int64 v8; // r14
  __int64 v9; // rdi
  __int64 v10; // r14
  __int64 v11; // rcx

  v4 = a2 - a1;
  v7 = a3 + v4;
  if ( v4 <= 576 )
  {
    sub_1E44500(a1, a2);
  }
  else
  {
    v8 = a1;
    do
    {
      v9 = v8;
      v8 += 672;
      sub_1E44500(v9, v8);
    }
    while ( a2 - v8 > 576 );
    sub_1E44500(v8, a2);
    if ( v4 > 672 )
    {
      v10 = 7;
      do
      {
        sub_1E44390(a1, a2, a3, v10);
        v11 = 2 * v10;
        v10 *= 4;
        sub_1E44390(a3, v7, a1, v11);
      }
      while ( (__int64)(0xAAAAAAAAAAAAAAABLL * (v4 >> 5)) > v10 );
    }
  }
}
