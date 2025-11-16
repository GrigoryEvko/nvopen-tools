// Function: sub_30CA2E0
// Address: 0x30ca2e0
//
__int64 __fastcall sub_30CA2E0(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  char v6; // al
  __int64 v7; // rcx

  v4 = sub_B2BE50(*(_QWORD *)a2);
  v5 = sub_B6F970(v4);
  v6 = (*(__int64 (__fastcall **)(__int64, char *, __int64))(*(_QWORD *)v5 + 32LL))(v5, "inline", 6);
  v7 = 0;
  if ( v6 )
    v7 = *(_QWORD *)(a2 + 56);
  sub_30DF350(
    a1,
    a3,
    *(_QWORD *)(a2 + 8),
    *(_QWORD *)(a2 + 16),
    (unsigned int)sub_30CA2A0,
    *(_QWORD *)(a2 + 24),
    (__int64)sub_30CA260,
    *(_QWORD *)(a2 + 32),
    (__int64)sub_30CA2C0,
    *(_QWORD *)(a2 + 40),
    **(_QWORD **)(a2 + 48),
    v7);
  return a1;
}
