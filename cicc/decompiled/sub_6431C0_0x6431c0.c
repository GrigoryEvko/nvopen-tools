// Function: sub_6431C0
// Address: 0x6431c0
//
__int64 __fastcall sub_6431C0(__int64 a1, unsigned __int8 a2)
{
  while ( (*(_BYTE *)(sub_72A270(a1, a2) + 88) & 0x70) != 0x10 )
  {
    if ( a2 > 0x2Bu )
    {
      if ( a2 != 59 )
LABEL_20:
        sub_721090(a1);
      a1 = *(_QWORD *)(a1 + 112);
      if ( !a1 )
        return 0;
      a2 = 59;
    }
    else
    {
      if ( a2 <= 5u )
        goto LABEL_20;
      switch ( a2 )
      {
        case 6u:
          a1 = *(_QWORD *)(a1 + 112);
          if ( !a1 )
            return 0;
          a2 = 6;
          break;
        case 7u:
          a1 = *(_QWORD *)(a1 + 112);
          if ( !a1 )
            return 0;
          a2 = 7;
          break;
        case 0xBu:
          a1 = *(_QWORD *)(a1 + 112);
          if ( !a1 )
            return 0;
          a2 = 11;
          break;
        case 0x1Cu:
          a1 = *(_QWORD *)(a1 + 112);
          if ( !a1 )
            return 0;
          a2 = 28;
          break;
        case 0x2Bu:
          a1 = *(_QWORD *)(a1 + 112);
          if ( !a1 )
            return 0;
          a2 = 43;
          break;
        default:
          goto LABEL_20;
      }
    }
  }
  return 1;
}
