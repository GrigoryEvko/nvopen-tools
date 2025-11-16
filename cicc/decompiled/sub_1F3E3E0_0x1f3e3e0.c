// Function: sub_1F3E3E0
// Address: 0x1f3e3e0
//
__int64 __fastcall sub_1F3E3E0(__int64 a1, __int64 a2)
{
  unsigned int v2; // edx
  char v3; // al
  char v5; // [rsp+Fh] [rbp-1h] BYREF

  v2 = 8 * sub_15A9520(a2, 0);
  if ( v2 == 32 )
  {
    v3 = 5;
  }
  else if ( v2 > 0x20 )
  {
    v3 = 6;
    if ( v2 != 64 )
    {
      v3 = 0;
      if ( v2 == 128 )
        v3 = 7;
    }
  }
  else
  {
    v3 = 3;
    if ( v2 != 8 )
      v3 = 4 * (v2 == 16);
  }
  v5 = v3;
  return sub_1F3E310(&v5);
}
