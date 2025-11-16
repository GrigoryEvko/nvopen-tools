// Function: sub_14E99E0
// Address: 0x14e99e0
//
_BYTE *__fastcall sub_14E99E0(_BYTE *a1, unsigned __int16 a2, unsigned int a3)
{
  _BYTE *result; // rax
  unsigned int v4; // r9d

  result = a1;
  v4 = HIWORD(a3);
  if ( a2 > 0x2Cu )
  {
    if ( a2 > 0x1F02u && (unsigned __int16)(a2 - 7968) <= 1u )
    {
LABEL_10:
      if ( (_WORD)a3 && BYTE2(a3) )
      {
        a1[1] = 1;
        *a1 = HIBYTE(a3) == 0 ? 4 : 8;
        return result;
      }
    }
LABEL_6:
    a1[1] = 0;
    return result;
  }
  if ( !a2 )
    goto LABEL_6;
  switch ( a2 )
  {
    case 1u:
      if ( !(_WORD)a3 || !BYTE2(a3) )
        goto LABEL_6;
      a1[1] = 1;
      *a1 = BYTE2(a3);
      break;
    case 5u:
    case 0x12u:
    case 0x26u:
    case 0x2Au:
      *(_WORD *)a1 = 258;
      break;
    case 6u:
    case 0x13u:
    case 0x1Cu:
    case 0x28u:
    case 0x2Cu:
      *(_WORD *)a1 = 260;
      break;
    case 7u:
    case 0x14u:
    case 0x20u:
    case 0x24u:
      *(_WORD *)a1 = 264;
      break;
    case 0xBu:
    case 0xCu:
    case 0x11u:
    case 0x25u:
    case 0x29u:
      *(_WORD *)a1 = 257;
      break;
    case 0xEu:
    case 0x17u:
    case 0x1Du:
    case 0x1Fu:
      goto LABEL_10;
    case 0x10u:
      if ( !(_WORD)a3 || !BYTE2(a3) )
        goto LABEL_6;
      if ( (_WORD)a3 != 2 )
        LOBYTE(v4) = HIBYTE(a3) == 0 ? 4 : 8;
      a1[1] = 1;
      *a1 = v4;
      break;
    case 0x19u:
    case 0x21u:
      *(_WORD *)a1 = 256;
      break;
    case 0x1Eu:
      *(_WORD *)a1 = 272;
      break;
    case 0x27u:
      *(_WORD *)a1 = 259;
      break;
    default:
      goto LABEL_6;
  }
  return result;
}
