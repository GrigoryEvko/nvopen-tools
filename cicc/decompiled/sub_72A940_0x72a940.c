// Function: sub_72A940
// Address: 0x72a940
//
__int64 __fastcall sub_72A940(__int64 a1)
{
  if ( *(_BYTE *)(a1 + 173) == 6 )
  {
    switch ( *(_BYTE *)(a1 + 176) )
    {
      case 0:
      case 1:
      case 2:
      case 3:
      case 4:
      case 6:
        return *(_QWORD *)(a1 + 184);
      case 5:
        return 0;
      default:
        sub_721090();
    }
  }
  return 0;
}
