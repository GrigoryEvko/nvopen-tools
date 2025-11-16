// Function: sub_22A7450
// Address: 0x22a7450
//
__int64 __fastcall sub_22A7450(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // r13
  __int64 v8; // rbx
  __int64 v9; // r14
  __int64 v10; // rdi

  v5 = 0x6DB6DB6DB6DB6DB7LL * ((a2 - a1) >> 3);
  v6 = a1;
  v7 = 2 * a4;
  if ( 2 * a4 <= v5 )
  {
    v8 = 56 * a4;
    v9 = 112 * a4;
    do
    {
      v10 = v6;
      v6 += v9;
      a3 = sub_22A72A0(v10, v6 + v8 - v9, v6 + v8 - v9, v6, a3);
      v5 = 0x6DB6DB6DB6DB6DB7LL * ((a2 - v6) >> 3);
    }
    while ( v5 >= v7 );
  }
  if ( v5 > a4 )
    v5 = a4;
  return sub_22A72A0(v6, v6 + 56 * v5, v6 + 56 * v5, a2, a3);
}
