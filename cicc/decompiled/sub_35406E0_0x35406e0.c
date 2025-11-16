// Function: sub_35406E0
// Address: 0x35406e0
//
__int64 __fastcall sub_35406E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r13
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // r14
  __int64 v10; // rdi

  v5 = 2 * a4;
  v6 = a1;
  v7 = 0x2E8BA2E8BA2E8BA3LL * ((a2 - a1) >> 3);
  if ( 2 * a4 <= v7 )
  {
    v8 = 176 * a4;
    v9 = 88 * a4;
    do
    {
      v10 = v6;
      v6 += v8;
      a3 = sub_35400A0(v10, v6 + v9 - v8, v6 + v9 - v8, v6, a3);
      v7 = 0x2E8BA2E8BA2E8BA3LL * ((a2 - v6) >> 3);
    }
    while ( v5 <= v7 );
  }
  if ( a4 <= v7 )
    v7 = a4;
  return sub_35400A0(v6, v6 + 88 * v7, v6 + 88 * v7, a2, a3);
}
