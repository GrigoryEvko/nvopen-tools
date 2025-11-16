// Function: sub_15FB6D0
// Address: 0x15fb6d0
//
char __fastcall sub_15FB6D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12

  if ( *(_BYTE *)(a1 + 16) != 38 )
    return 0;
  v4 = *(_QWORD *)(a1 - 48);
  if ( *(_BYTE *)(v4 + 16) > 0x10u )
    return 0;
  if ( (_BYTE)a2 || sub_15F24C0(a1) )
    return sub_1595F50(v4, a2, a3, a4);
  return sub_1595D90(v4, a2, a3, a4);
}
