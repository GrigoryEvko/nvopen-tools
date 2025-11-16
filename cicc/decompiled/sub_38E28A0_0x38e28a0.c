// Function: sub_38E28A0
// Address: 0x38e28a0
//
void __fastcall sub_38E28A0(__int64 a1, int a2)
{
  __int16 v2; // bx

  v2 = a2;
  if ( a2 == 3 )
  {
    if ( (unsigned int)sub_38E27C0(a1) )
      return;
  }
  else
  {
    switch ( a2 )
    {
      case 0:
      case 1:
      case 2:
        break;
      case 4:
      case 7:
      case 8:
      case 9:
      case 10:
        v2 = 6;
        break;
      case 5:
        v2 = 4;
        break;
      case 6:
        v2 = 5;
        break;
    }
  }
  *(_WORD *)(a1 + 12) = *(_WORD *)(a1 + 12) & 0xFFF8 | v2;
}
