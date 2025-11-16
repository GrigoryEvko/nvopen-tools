// Function: sub_12A0120
// Address: 0x12a0120
//
__int64 __fastcall sub_12A0120(__int64 *a1, __int64 a2)
{
  char v2; // al
  char *v4; // rax
  unsigned int v5; // ebx
  __int64 v6; // r14
  char *v7; // r12
  size_t v8; // rdx
  char v9; // al
  __int64 v10; // rdx

  v2 = *(_BYTE *)(a2 + 140);
  if ( v2 != 2 )
  {
    if ( v2 != 3 )
      sub_127B550("unhandled basic type in debug info gen!", (_DWORD *)(a2 + 64), 1);
    v4 = sub_7465E0(*(_BYTE *)(a2 + 160), 1);
    if ( !strcmp(v4, "long double") || !strcmp(v4, "__float80") )
    {
      v4 = "double";
      v5 = 4;
    }
    else
    {
      v5 = 4;
      if ( dword_4D0470C )
      {
        if ( !strcmp(v4, "__float128") )
          v4 = "double";
      }
    }
LABEL_6:
    v6 = *(_QWORD *)(a2 + 128);
    v7 = v4;
    v8 = strlen(v4);
    return sub_15A5A00(a1 + 2, v7, v8, 8 * v6, v5);
  }
  v9 = *(_BYTE *)(a2 + 161);
  if ( (v9 & 8) == 0 || (**(_BYTE **)(a2 + 176) & 1) == 0 )
    goto LABEL_13;
  v10 = *(_QWORD *)(a2 + 168);
  if ( (v9 & 0x10) != 0 )
    v10 = *(_QWORD *)(v10 + 96);
  if ( v10 )
    return sub_129FD50(a1, a2);
LABEL_13:
  if ( (*(_BYTE *)(a2 + 162) & 4) == 0 )
  {
    switch ( *(_BYTE *)(a2 + 160) )
    {
      case 2:
        v5 = 8;
        break;
      case 3:
      case 5:
      case 7:
      case 9:
      case 0xB:
        v5 = 5;
        break;
      case 4:
      case 6:
      case 8:
      case 0xA:
      case 0xC:
        v5 = 7;
        break;
      default:
        v5 = 6;
        break;
    }
    v4 = sub_7465D0(a2);
    if ( !v4 )
      sub_127B550("unexpected: NULL basic type name!", (_DWORD *)(a2 + 64), 1);
    goto LABEL_6;
  }
  return sub_129EC00((__int64)a1);
}
