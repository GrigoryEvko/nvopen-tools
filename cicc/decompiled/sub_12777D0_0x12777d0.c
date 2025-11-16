// Function: sub_12777D0
// Address: 0x12777d0
//
__int64 __fastcall sub_12777D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rsi

  sub_16BD4C0(a3, *(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL));
  result = *(_QWORD *)(a2 + 16);
  v5 = result + 40;
  v6 = result + 8 * (5LL * *(unsigned int *)(a2 + 8) + 5);
  if ( result + 40 != v6 )
  {
    do
    {
      v7 = *(_QWORD *)(v5 + 24);
      v5 += 40;
      sub_16BD4C0(a3, v7);
      sub_16BD430(a3, *(unsigned __int8 *)(v5 - 8));
      result = sub_16BD430(a3, *(unsigned __int8 *)(v5 - 7));
    }
    while ( v6 != v5 );
  }
  return result;
}
