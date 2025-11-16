// Function: sub_EA1710
// Address: 0xea1710
//
void __fastcall sub_EA1710(__int64 a1, unsigned int a2)
{
  __int16 v2; // bx

  v2 = a2;
  sub_EA1700(a1);
  if ( a2 == 2 )
  {
    v2 = 16;
  }
  else if ( a2 > 2 )
  {
    if ( a2 != 10 )
      BUG();
    v2 = 24;
  }
  else if ( a2 )
  {
    v2 = 8;
  }
  *(_WORD *)(a1 + 12) = *(_WORD *)(a1 + 12) & 0xFFE7 | v2;
}
