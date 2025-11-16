// Function: sub_3031480
// Address: 0x3031480
//
bool __fastcall sub_3031480(unsigned __int16 *a1, _WORD *a2)
{
  unsigned __int16 v2; // ax
  __int64 v3; // rax
  unsigned __int64 v4; // rax
  bool result; // al

  v2 = *a1;
  if ( *a1 )
  {
    if ( v2 == 1 || (unsigned __int16)(v2 - 504) <= 7u )
LABEL_15:
      BUG();
    v3 = *(_QWORD *)&byte_444C4A0[16 * v2 - 16];
  }
  else
  {
    v3 = sub_3007260((__int64)a1);
  }
  if ( v3 <= 0 )
    goto LABEL_15;
  v4 = v3 - 1;
  if ( v4 )
  {
    _BitScanReverse64(&v4, v4);
    switch ( 1LL << (64 - ((unsigned __int8)v4 ^ 0x3Fu)) )
    {
      case 0LL:
      case 1LL:
      case 3LL:
      case 5LL:
      case 6LL:
      case 7LL:
      case 9LL:
      case 10LL:
      case 11LL:
      case 12LL:
      case 13LL:
      case 14LL:
      case 15LL:
      case 17LL:
      case 18LL:
      case 19LL:
      case 20LL:
      case 21LL:
      case 22LL:
      case 23LL:
      case 24LL:
      case 25LL:
      case 26LL:
      case 27LL:
      case 28LL:
      case 29LL:
      case 30LL:
      case 31LL:
        goto LABEL_15;
      case 2LL:
      case 4LL:
      case 8LL:
        *a2 = 5;
        return *a1 != 5;
      case 16LL:
        *a2 = 6;
        return *a1 != 6;
      case 32LL:
        *a2 = 7;
        return *a1 != 7;
      default:
        if ( ((unsigned int)v4 ^ 0x3F) != 58 )
          goto LABEL_15;
        *a2 = 8;
        result = *a1 != 8;
        break;
    }
  }
  else
  {
    *a2 = 2;
    return *a1 != 2;
  }
  return result;
}
