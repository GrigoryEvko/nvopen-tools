// Function: sub_70CCB0
// Address: 0x70ccb0
//
_WORD *__fastcall sub_70CCB0(__int64 a1, char a2, __int64 a3, __int64 a4)
{
  unsigned int v5; // eax
  _BOOL4 v6; // ebx

  v5 = sub_621060(a1, a3);
  switch ( a2 )
  {
    case ':':
      v6 = v5 == 0;
      break;
    case ';':
      v6 = v5 != 0;
      break;
    case '<':
      v6 = (int)v5 > 0;
      break;
    case '=':
      goto LABEL_3;
    case '>':
      v5 = ~v5;
LABEL_3:
      v6 = v5 >> 31;
      break;
    case '?':
      v6 = (int)v5 <= 0;
      break;
    default:
      sub_721090(a1);
  }
  sub_724A80(a4, 1);
  return sub_620D80((_WORD *)(a4 + 176), v6);
}
