// Function: sub_35E5C90
// Address: 0x35e5c90
//
__int64 __fastcall sub_35E5C90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r11
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // r12
  __int64 v10; // r13
  __int64 v11; // rdi

  v4 = a2;
  v6 = a1;
  v7 = 0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 3);
  v8 = 2 * a4;
  if ( 2 * a4 <= v7 )
  {
    v9 = 48 * a4;
    v10 = 24 * a4;
    do
    {
      v11 = v6;
      v6 += v9;
      a3 = sub_35E5B40(v11, v6 + v10 - v9, v6 + v10 - v9, v6, a3);
      v7 = 0xAAAAAAAAAAAAAAABLL * ((v4 - v6) >> 3);
    }
    while ( v8 <= v7 );
  }
  if ( a4 <= v7 )
    v7 = a4;
  return sub_35E5B40(v6, v6 + 24 * v7, v6 + 24 * v7, v4, a3);
}
