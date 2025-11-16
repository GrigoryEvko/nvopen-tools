// Function: sub_E0CC40
// Address: 0xe0cc40
//
__int64 __fastcall sub_E0CC40(unsigned __int16 a1, unsigned int a2)
{
  __int16 v3; // cx
  unsigned int v4; // esi
  unsigned int v5; // eax
  unsigned int v6; // edx

  v3 = a2;
  v4 = HIBYTE(a2);
  v5 = HIWORD(a2);
  if ( a1 <= 0x2Cu )
  {
    if ( a1 )
    {
      switch ( a1 )
      {
        case 1u:
          if ( v3 && (_BYTE)v5 )
            goto LABEL_23;
          return 0;
        case 5u:
        case 0x12u:
        case 0x26u:
        case 0x2Au:
          return 258;
        case 6u:
        case 0x13u:
        case 0x1Cu:
        case 0x28u:
        case 0x2Cu:
          return 260;
        case 7u:
        case 0x14u:
        case 0x20u:
        case 0x24u:
          return 264;
        case 0xBu:
        case 0xCu:
        case 0x11u:
        case 0x25u:
        case 0x29u:
          return 257;
        case 0xEu:
        case 0x17u:
        case 0x1Du:
        case 0x1Fu:
          goto LABEL_9;
        case 0x10u:
          if ( !v3 || !(_BYTE)v5 )
            return 0;
          if ( v3 != 2 )
            goto LABEL_19;
          goto LABEL_23;
        case 0x19u:
        case 0x21u:
          return 256;
        case 0x1Eu:
          return 272;
        case 0x27u:
        case 0x2Bu:
          return 259;
        default:
          return 0;
      }
    }
    return 0;
  }
  if ( (unsigned __int16)(a1 - 7968) > 1u )
    return 0;
LABEL_9:
  if ( !v3 || !(_BYTE)v5 )
    return 0;
LABEL_19:
  if ( (_BYTE)v4 )
  {
    if ( (_BYTE)v4 != 1 )
      BUG();
    LOBYTE(v5) = 8;
  }
  else
  {
    LOBYTE(v5) = 4;
  }
LABEL_23:
  v6 = (unsigned __int8)v5;
  BYTE1(v6) = 1;
  return v6;
}
