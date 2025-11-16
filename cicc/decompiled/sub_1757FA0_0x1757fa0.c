// Function: sub_1757FA0
// Address: 0x1757fa0
//
bool __fastcall sub_1757FA0(int a1, __int64 a2, _BYTE *a3)
{
  unsigned int v3; // ecx
  bool result; // al
  unsigned int v5; // ebx
  unsigned int v6; // ecx
  unsigned int v7; // ebx

  switch ( a1 )
  {
    case '"':
      *a3 = 1;
      v3 = *(_DWORD *)(a2 + 8);
      if ( v3 > 0x40 )
      {
        if ( (*(_QWORD *)(*(_QWORD *)a2 + 8LL * ((v3 - 1) >> 6)) & (1LL << ((unsigned __int8)v3 - 1))) == 0
          && v3 - 1 == (unsigned int)sub_16A58F0(a2) )
        {
          goto LABEL_13;
        }
        goto LABEL_5;
      }
      result = (1LL << ((unsigned __int8)v3 - 1)) - 1 == *(_QWORD *)a2;
      break;
    case '#':
      *a3 = 1;
      v6 = *(_DWORD *)(a2 + 8);
      if ( v6 <= 0x40 )
      {
        result = 1LL << ((unsigned __int8)v6 - 1) == *(_QWORD *)a2;
      }
      else
      {
        if ( (*(_QWORD *)(*(_QWORD *)a2 + 8LL * ((v6 - 1) >> 6)) & (1LL << ((unsigned __int8)v6 - 1))) == 0
          || v6 - 1 != (unsigned int)sub_16A58A0(a2) )
        {
          goto LABEL_5;
        }
LABEL_13:
        result = 1;
      }
      break;
    case '$':
    case '%':
    case '\'':
LABEL_5:
      result = 0;
      break;
    case '&':
      *a3 = 0;
      v5 = *(_DWORD *)(a2 + 8);
      if ( v5 > 0x40 )
        goto LABEL_7;
      goto LABEL_9;
    case '(':
      *a3 = 1;
      v7 = *(_DWORD *)(a2 + 8);
      if ( v7 <= 0x40 )
        result = *(_QWORD *)a2 == 0;
      else
        result = v7 == (unsigned int)sub_16A57B0(a2);
      break;
    case ')':
      *a3 = 1;
      v5 = *(_DWORD *)(a2 + 8);
      if ( v5 <= 0x40 )
LABEL_9:
        result = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v5) == *(_QWORD *)a2;
      else
LABEL_7:
        result = v5 == (unsigned int)sub_16A58F0(a2);
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
