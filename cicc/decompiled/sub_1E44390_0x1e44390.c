// Function: sub_1E44390
// Address: 0x1e44390
//
__int64 __fastcall sub_1E44390(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 v8; // r14
  __int64 v9; // r15
  __int64 v10; // rdi

  v5 = 0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 5);
  v6 = a1;
  v7 = 2 * a4;
  if ( 2 * a4 <= v5 )
  {
    v8 = 96 * a4;
    v9 = 192 * a4;
    do
    {
      v10 = v6;
      v6 += v9;
      a3 = sub_1E43E20(v10, v6 + v8 - v9, v6 + v8 - v9, v6, a3);
      v5 = 0xAAAAAAAAAAAAAAABLL * ((a2 - v6) >> 5);
    }
    while ( v7 <= v5 );
  }
  if ( a4 <= v5 )
    v5 = a4;
  return sub_1E43E20(v6, v6 + 96 * v5, v6 + 96 * v5, a2, a3);
}
