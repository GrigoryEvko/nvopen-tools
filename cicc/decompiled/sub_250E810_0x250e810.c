// Function: sub_250E810
// Address: 0x250e810
//
bool __fastcall sub_250E810(__int64 a1)
{
  bool result; // al

  result = sub_B2FC80(a1);
  if ( result )
    return 0;
  if ( (*(_BYTE *)(a1 + 32) & 0xFu) - 7 > 1 )
  {
    switch ( *(_BYTE *)(a1 + 32) & 0xF )
    {
      case 0:
      case 1:
      case 3:
      case 5:
      case 6:
        result = 1;
        break;
      case 2:
      case 4:
      case 9:
      case 0xA:
        return result;
      default:
        BUG();
    }
  }
  return result;
}
