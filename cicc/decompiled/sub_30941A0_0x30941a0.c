// Function: sub_30941A0
// Address: 0x30941a0
//
__int64 __fastcall sub_30941A0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rcx
  __int64 v3; // rsi
  __int64 v4; // rdx

  switch ( *(_WORD *)(a1 + 68) )
  {
    case 0xA63:
    case 0xA66:
    case 0xA68:
    case 0xA6A:
    case 0xA6D:
    case 0xA6F:
      result = 8;
      v2 = 4;
      goto LABEL_3;
    case 0xA64:
    case 0xA67:
    case 0xA69:
    case 0xA6B:
    case 0xA6E:
    case 0xA70:
      result = 10;
      v2 = 6;
      goto LABEL_3;
    case 0xA71:
    case 0xA72:
    case 0xA73:
    case 0xA74:
    case 0xA75:
    case 0xA76:
      result = 7;
      v2 = 3;
LABEL_3:
      if ( (unsigned int)result >= (*(_DWORD *)(a1 + 40) & 0xFFFFFFu) )
        goto LABEL_5;
      v3 = *(_QWORD *)(a1 + 32);
      v4 = v3 + 40 * v2;
      if ( *(_BYTE *)v4 != 1 || *(_QWORD *)(v4 + 24) != 101 )
        goto LABEL_5;
      if ( *(_BYTE *)(v3 + 40LL * (unsigned int)result) != 9 )
        result = 0;
      break;
    default:
LABEL_5:
      result = 0;
      break;
  }
  return result;
}
