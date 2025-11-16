// Function: sub_3011FC0
// Address: 0x3011fc0
//
bool __fastcall sub_3011FC0(__int64 a1)
{
  char v1; // al
  bool result; // al

  v1 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 == 39 )
  {
    result = 0;
    if ( ***(_BYTE ***)(a1 - 8) == 21 )
      return !(*(_WORD *)(a1 + 2) & 1);
  }
  else if ( v1 == 80 )
  {
    result = 0;
    if ( **(_BYTE **)(a1 - 32) == 21 )
      return sub_3011DA0(a1) == 0;
  }
  else
  {
    if ( v1 != 81 )
      BUG();
    return 0;
  }
  return result;
}
