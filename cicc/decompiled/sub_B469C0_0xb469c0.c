// Function: sub_B469C0
// Address: 0xb469c0
//
char __fastcall sub_B469C0(unsigned __int8 *a1)
{
  int v1; // ecx
  int v2; // eax
  unsigned int v3; // ecx

  v1 = *a1;
  if ( (_BYTE)v1 == 85 )
  {
    return (unsigned int)sub_B46970(a1) ^ 1;
  }
  else
  {
    LOBYTE(v2) = 0;
    if ( (unsigned int)(v1 - 30) > 0xA )
    {
      v3 = v1 - 39;
      LOBYTE(v2) = 1;
      if ( v3 <= 0x38 )
        LOBYTE(v2) = ((1LL << v3) & 0x100060000000001LL) == 0;
    }
  }
  return v2;
}
