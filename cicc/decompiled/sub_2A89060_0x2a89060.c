// Function: sub_2A89060
// Address: 0x2a89060
//
__int64 __fastcall sub_2A89060(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v5; // al
  __int64 v8; // rax
  __int64 v9; // r15
  unsigned __int64 v10; // rax

  v5 = *(_BYTE *)(*(_QWORD *)(a3 + 8) + 8LL);
  if ( (unsigned __int8)(v5 - 15) <= 1u )
    return 0xFFFFFFFFLL;
  if ( v5 == 18 )
    return 0xFFFFFFFFLL;
  v8 = sub_B43CB0(a3);
  if ( !sub_2A88720(a3, a1, v8) )
    return 0xFFFFFFFFLL;
  v9 = *(_QWORD *)(a3 - 32);
  v10 = sub_9208B0(a4, *(_QWORD *)(a3 + 8));
  return sub_2A88540(a1, a2, v9, v10, a4);
}
