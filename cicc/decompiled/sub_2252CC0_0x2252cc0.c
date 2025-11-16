// Function: sub_2252CC0
// Address: 0x2252cc0
//
__int64 __fastcall sub_2252CC0(char a1, __int64 a2)
{
  unsigned __int8 v2; // al

  if ( a1 == -1 )
    return 0;
  v2 = a1 & 0x70;
  if ( (a1 & 0x70) != 0x30 )
  {
    if ( v2 > 0x30u )
    {
      if ( v2 == 64 )
        return sub_39F7FD0(a2);
      if ( v2 == 80 )
        return 0;
    }
    else
    {
      if ( v2 == 32 )
        return sub_39F8010(a2);
      if ( v2 <= 0x20u && (a1 & 0x60) == 0 )
        return 0;
    }
    abort();
  }
  return sub_39F8000(a2);
}
