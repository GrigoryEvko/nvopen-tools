// Function: sub_1BF1850
// Address: 0x1bf1850
//
char __fastcall sub_1BF1850(__int64 a1, unsigned int a2)
{
  unsigned int v2; // eax

  v2 = *(_DWORD *)(a1 + 12);
  if ( v2 > 3 )
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
