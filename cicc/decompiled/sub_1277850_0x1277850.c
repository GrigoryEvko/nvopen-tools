// Function: sub_1277850
// Address: 0x1277850
//
__int64 __fastcall sub_1277850(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rsi

  sub_16BD4C0(a3, *(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL));
  v4 = *(_QWORD *)(a2 + 16);
  v5 = v4 + 40;
  v6 = v4 + 8 * (5LL * *(unsigned int *)(a2 + 8) + 5);
  if ( v4 + 40 != v6 )
  {
    do
    {
      v7 = *(_QWORD *)(v5 + 24);
      v5 += 40;
      sub_16BD4C0(a3, v7);
      sub_16BD430(a3, *(unsigned __int8 *)(v5 - 8));
      sub_16BD430(a3, *(unsigned __int8 *)(v5 - 7));
    }
    while ( v6 != v5 );
  }
  return sub_16BDDB0(a3);
}
