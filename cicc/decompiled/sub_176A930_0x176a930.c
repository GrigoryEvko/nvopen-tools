// Function: sub_176A930
// Address: 0x176a930
//
char __fastcall sub_176A930(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v4; // al
  __int64 v6; // rax
  __int64 v7; // rcx

  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 == 35 )
  {
    if ( *a1 == *(_QWORD *)(a2 - 48) )
      return sub_1757CC0(*(_BYTE **)(a2 - 24), a2, a3, a4);
    return 0;
  }
  if ( v4 != 5 )
    return 0;
  if ( *(_WORD *)(a2 + 18) != 11 )
    return 0;
  v6 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v7 = *(_QWORD *)(a2 - 24 * v6);
  if ( *a1 != v7 )
    return 0;
  return sub_1757E30(*(_BYTE **)(a2 + 24 * (1 - v6)), a2, 1 - v6, v7);
}
