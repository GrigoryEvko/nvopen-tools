// Function: sub_12774D0
// Address: 0x12774d0
//
__int64 __fastcall sub_12774D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // r14
  __int64 v10; // rsi

  sub_16BD4C0(a5, *(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL));
  v7 = *(_QWORD *)(a2 + 16);
  v8 = v7 + 40;
  v9 = v7 + 8 * (5LL * *(unsigned int *)(a2 + 8) + 5);
  if ( v7 + 40 != v9 )
  {
    do
    {
      v10 = *(_QWORD *)(v8 + 24);
      v8 += 40;
      sub_16BD4C0(a5, v10);
      sub_16BD430(a5, *(unsigned __int8 *)(v8 - 8));
      sub_16BD430(a5, *(unsigned __int8 *)(v8 - 7));
    }
    while ( v9 != v8 );
  }
  return sub_16BD750(a5, a3);
}
