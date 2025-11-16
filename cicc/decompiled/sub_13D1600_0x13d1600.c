// Function: sub_13D1600
// Address: 0x13d1600
//
__int64 __fastcall sub_13D1600(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rax
  __int64 *v4; // r8
  unsigned __int8 v5; // al
  unsigned int v7; // ebx
  __int64 v8; // r13

  v3 = 0;
  if ( *((_BYTE *)a1 + 16) < 0x11u )
    v3 = a1;
  v4 = v3;
  v5 = *(_BYTE *)(a3 + 16);
  if ( *(_BYTE *)(a2 + 16) > 0x10u )
  {
    if ( v5 <= 0x10u )
      goto LABEL_8;
    return 0;
  }
  if ( v5 > 0x10u )
    return 0;
  if ( v4 )
    return sub_1584660(v4);
LABEL_8:
  if ( v5 != 13 )
  {
    if ( v5 == 9 )
      return sub_1599EF0(*a1);
    return 0;
  }
  v7 = *(_DWORD *)(a3 + 32);
  v8 = *a1;
  if ( v7 > 0x40 )
  {
    if ( v7 - (unsigned int)sub_16A57B0(a3 + 24) <= 0x40 && *(_QWORD *)(v8 + 32) > **(_QWORD **)(a3 + 24) )
      return 0;
  }
  else if ( *(_QWORD *)(v8 + 32) > *(_QWORD *)(a3 + 24) )
  {
    return 0;
  }
  return sub_1599EF0(v8);
}
