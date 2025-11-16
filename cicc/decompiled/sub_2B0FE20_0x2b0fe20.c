// Function: sub_2B0FE20
// Address: 0x2b0fe20
//
__int64 __fastcall sub_2B0FE20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v10; // rbx
  __int64 v11; // r15
  __int64 v12; // r13
  __int64 v13; // rdi
  __int64 v14; // rsi

  v7 = (a2 - a1) >> 6;
  v8 = 2 * a4;
  v10 = a1;
  if ( 2 * a4 <= v7 )
  {
    v11 = a4 << 7;
    v12 = -64 * a4;
    do
    {
      v13 = v10;
      v10 += v11;
      a3 = sub_2B0FD10(v13, v10 + v12, v10 + v12, v10, a3, a6);
      v7 = (a2 - v10) >> 6;
    }
    while ( v8 <= v7 );
  }
  v14 = a4;
  if ( a4 > v7 )
    v14 = v7;
  return sub_2B0FD10(v10, v10 + (v14 << 6), v10 + (v14 << 6), a2, a3, a6);
}
