// Function: sub_B532C0
// Address: 0xb532c0
//
char __fastcall sub_B532C0(__int64 a1, _QWORD *a2, int a3)
{
  unsigned int v3; // eax
  char v4; // al

  switch ( a3 )
  {
    case ' ':
      if ( *(_DWORD *)(a1 + 8) <= 0x40u )
        LOBYTE(v3) = *(_QWORD *)a1 == *a2;
      else
        LOBYTE(v3) = sub_C43C50(a1, a2);
      break;
    case '!':
      if ( *(_DWORD *)(a1 + 8) <= 0x40u )
        v4 = *(_QWORD *)a1 == *a2;
      else
        v4 = sub_C43C50(a1, a2);
      LOBYTE(v3) = v4 ^ 1;
      break;
    case '"':
      LOBYTE(v3) = (int)sub_C49970(a1, a2) > 0;
      break;
    case '#':
      LOBYTE(v3) = (int)sub_C49970(a1, a2) >= 0;
      break;
    case '$':
      v3 = (unsigned int)sub_C49970(a1, a2) >> 31;
      break;
    case '%':
      LOBYTE(v3) = (int)sub_C49970(a1, a2) <= 0;
      break;
    case '&':
      LOBYTE(v3) = (int)sub_C4C880(a1, a2) > 0;
      break;
    case '\'':
      LOBYTE(v3) = (int)sub_C4C880(a1, a2) >= 0;
      break;
    case '(':
      v3 = (unsigned int)sub_C4C880(a1, a2) >> 31;
      break;
    case ')':
      LOBYTE(v3) = (int)sub_C4C880(a1, a2) <= 0;
      break;
    default:
      BUG();
  }
  return v3;
}
