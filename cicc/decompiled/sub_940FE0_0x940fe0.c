// Function: sub_940FE0
// Address: 0x940fe0
//
__int64 __fastcall sub_940FE0(__int64 *a1, __int64 a2)
{
  char v2; // al
  char *v4; // rax
  int v5; // ebx
  __int64 v6; // r14
  int v7; // r12d
  int v8; // edx
  char v9; // al
  __int64 v10; // rdx

  v2 = *(_BYTE *)(a2 + 140);
  if ( v2 != 2 )
  {
    if ( v2 != 3 )
      sub_91B8A0("unhandled basic type in debug info gen!", (_DWORD *)(a2 + 64), 1);
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
    v7 = (int)v4;
    v8 = strlen(v4);
    return sub_ADC9A0((int)a1 + 16, v7, v8, 8 * (int)v6, v5, 0, 0);
  }
  v9 = *(_BYTE *)(a2 + 161);
  if ( (v9 & 8) == 0 || (**(_BYTE **)(a2 + 176) & 1) == 0 )
    goto LABEL_13;
  v10 = *(_QWORD *)(a2 + 168);
  if ( (v9 & 0x10) != 0 )
    v10 = *(_QWORD *)(v10 + 96);
  if ( v10 )
    return sub_940BE0(a1, a2);
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
      sub_91B8A0("unexpected: NULL basic type name!", (_DWORD *)(a2 + 64), 1);
    goto LABEL_6;
  }
  return sub_93F6B0((int)a1);
}
