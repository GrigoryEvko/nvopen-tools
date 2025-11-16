// Function: sub_1489CC0
// Address: 0x1489cc0
//
__int64 __fastcall sub_1489CC0(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rcx
  __int64 v8; // rdx
  unsigned int v9; // esi

  switch ( a2 )
  {
    case ' ':
    case '!':
      if ( (unsigned __int8)sub_1452FA0(a3, a5) && (unsigned __int8)sub_1452FA0(a4, a6) )
        return 1;
      goto LABEL_3;
    case '"':
    case '#':
      if ( !(unsigned __int8)sub_1481140(a1, 0x23u, a3, a5) )
        goto LABEL_3;
      v7 = a6;
      v8 = a4;
      v9 = 37;
      break;
    case '$':
    case '%':
      JUMPOUT(0x1489D53);
    case '&':
    case '\'':
      if ( !(unsigned __int8)sub_1481140(a1, 0x27u, a3, a5) )
        goto LABEL_3;
      v7 = a6;
      v8 = a4;
      v9 = 41;
      break;
    case '(':
    case ')':
      JUMPOUT(0x1489DB3);
  }
  if ( !(unsigned __int8)sub_1481140(a1, v9, v8, v7) )
LABEL_3:
    JUMPOUT(0x1489D23);
  return 1;
}
