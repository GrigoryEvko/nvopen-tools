// Function: sub_31A48F0
// Address: 0x31a48f0
//
char __fastcall sub_31A48F0(__int64 a1, unsigned int a2)
{
  unsigned int v2; // eax

  v2 = *(_DWORD *)(a1 + 12);
  if ( v2 > 5 )
  {
    LOBYTE(v2) = 0;
  }
  else if ( v2 > 1 )
  {
    LOBYTE(v2) = a2 <= 1;
  }
  else if ( v2 )
  {
    LOBYTE(v2) = 0;
    if ( a2 - 1 <= 0xF )
      LOBYTE(v2) = ((a2 - 1) & a2) == 0;
  }
  else if ( a2 && (a2 & (a2 - 1)) == 0 )
  {
    LOBYTE(v2) = *(_DWORD *)"@" >= a2;
  }
  return v2;
}
