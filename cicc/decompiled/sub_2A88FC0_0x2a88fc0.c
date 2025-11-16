// Function: sub_2A88FC0
// Address: 0x2a88fc0
//
__int64 __fastcall sub_2A88FC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r13
  char v6; // al
  __int64 v9; // rax
  __int64 v10; // r13
  unsigned __int64 v11; // rax

  v5 = *(_QWORD *)(a3 - 64);
  v6 = *(_BYTE *)(*(_QWORD *)(v5 + 8) + 8LL);
  if ( (unsigned __int8)(v6 - 15) <= 1u )
    return 0xFFFFFFFFLL;
  if ( v6 == 18 )
    return 0xFFFFFFFFLL;
  v9 = sub_B43CB0(a3);
  if ( !sub_2A88720(v5, a1, v9) )
    return 0xFFFFFFFFLL;
  v10 = *(_QWORD *)(a3 - 32);
  v11 = sub_9208B0(a4, *(_QWORD *)(*(_QWORD *)(a3 - 64) + 8LL));
  return sub_2A88540(a1, a2, v10, v11, a4);
}
