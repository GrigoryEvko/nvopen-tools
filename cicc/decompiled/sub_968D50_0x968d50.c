// Function: sub_968D50
// Address: 0x968d50
//
__int64 __fastcall sub_968D50(__int64 a1, int a2)
{
  unsigned __int16 v2; // r13
  __int16 v3; // ax
  unsigned int v4; // r8d

  v2 = sub_B59DB0();
  v3 = sub_B59EF0(a1);
  v4 = 1;
  if ( a2 )
  {
    if ( !HIBYTE(v2) || (v4 = 0, (_BYTE)v2 != 7) )
    {
      v4 = HIBYTE(v3);
      if ( HIBYTE(v3) )
        LOBYTE(v4) = (_BYTE)v3 != 2;
    }
  }
  return v4;
}
