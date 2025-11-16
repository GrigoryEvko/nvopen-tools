// Function: sub_15E5030
// Address: 0x15e5030
//
__int64 __fastcall sub_15E5030(_BYTE *a1)
{
  unsigned int v1; // r8d
  char v2; // al

  v1 = 0;
  if ( (a1[32] & 0xF) == 3 )
  {
    v2 = a1[32] >> 6;
    v1 = 1;
    if ( v2 != 2 )
    {
      if ( a1[16] != 3 || (v1 = a1[80] & 1, (a1[80] & 1) != 0) )
        LOBYTE(v1) = v2 != 0;
    }
  }
  return v1;
}
