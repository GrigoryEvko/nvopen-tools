// Function: sub_B527F0
// Address: 0xb527f0
//
char __fastcall sub_B527F0(__int64 a1)
{
  __int16 v1; // cx
  __int64 v2; // rax

  v1 = *(_WORD *)(a1 + 2);
  if ( *(_BYTE *)a1 == 82 )
  {
    LOBYTE(v2) = (v1 & 0x3Fu) - 32 <= 1;
  }
  else
  {
    LOBYTE(v2) = 0;
    if ( (v1 & 0x30) == 0 )
      return (0xC3C3uLL >> v1) & 1;
  }
  return v2;
}
