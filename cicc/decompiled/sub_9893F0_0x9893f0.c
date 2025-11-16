// Function: sub_9893F0
// Address: 0x9893f0
//
bool __fastcall sub_9893F0(int a1, __int64 a2, _BYTE *a3)
{
  unsigned int v3; // ebx
  bool result; // al
  unsigned int v5; // ebx
  unsigned int v6; // ecx
  unsigned int v7; // ecx
  unsigned int v8; // ebx

  switch ( a1 )
  {
    case '"':
      *a3 = 1;
      v6 = *(_DWORD *)(a2 + 8);
      if ( v6 > 0x40 )
        goto LABEL_9;
      result = (1LL << ((unsigned __int8)v6 - 1)) - 1 == *(_QWORD *)a2;
      break;
    case '#':
      *a3 = 1;
      v7 = *(_DWORD *)(a2 + 8);
      if ( v7 > 0x40 )
        goto LABEL_13;
      result = 1LL << ((unsigned __int8)v7 - 1) == *(_QWORD *)a2;
      break;
    case '$':
      *a3 = 0;
      v7 = *(_DWORD *)(a2 + 8);
      if ( v7 > 0x40 )
      {
LABEL_13:
        if ( (*(_QWORD *)(*(_QWORD *)a2 + 8LL * ((v7 - 1) >> 6)) & (1LL << ((unsigned __int8)v7 - 1))) == 0
          || v7 - 1 != (unsigned int)sub_C44590(a2) )
        {
          goto LABEL_11;
        }
        goto LABEL_15;
      }
      result = 1LL << ((unsigned __int8)v7 - 1) == *(_QWORD *)a2;
      break;
    case '%':
      *a3 = 0;
      v6 = *(_DWORD *)(a2 + 8);
      if ( v6 > 0x40 )
      {
LABEL_9:
        if ( (*(_QWORD *)(*(_QWORD *)a2 + 8LL * ((v6 - 1) >> 6)) & (1LL << ((unsigned __int8)v6 - 1))) == 0
          && v6 - 1 == (unsigned int)sub_C445E0(a2) )
        {
          goto LABEL_15;
        }
LABEL_11:
        result = 0;
      }
      else
      {
        result = (1LL << ((unsigned __int8)v6 - 1)) - 1 == *(_QWORD *)a2;
      }
      break;
    case '&':
      *a3 = 0;
      v5 = *(_DWORD *)(a2 + 8);
      if ( !v5 )
        goto LABEL_15;
      if ( v5 > 0x40 )
        goto LABEL_7;
      result = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v5) == *(_QWORD *)a2;
      break;
    case '\'':
      *a3 = 0;
      v8 = *(_DWORD *)(a2 + 8);
      if ( v8 <= 0x40 )
        result = *(_QWORD *)a2 == 0;
      else
        result = v8 == (unsigned int)sub_C444A0(a2);
      break;
    case '(':
      *a3 = 1;
      v3 = *(_DWORD *)(a2 + 8);
      if ( v3 <= 0x40 )
        result = *(_QWORD *)a2 == 0;
      else
        result = v3 == (unsigned int)sub_C444A0(a2);
      break;
    case ')':
      *a3 = 1;
      v5 = *(_DWORD *)(a2 + 8);
      if ( v5 )
      {
        if ( v5 <= 0x40 )
          result = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v5) == *(_QWORD *)a2;
        else
LABEL_7:
          result = v5 == (unsigned int)sub_C445E0(a2);
      }
      else
      {
LABEL_15:
        result = 1;
      }
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
