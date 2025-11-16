// Function: sub_12F9520
// Address: 0x12f9520
//
__int64 __fastcall sub_12F9520(unsigned int a1)
{
  unsigned int v1; // eax
  unsigned int v2; // eax
  unsigned int v3; // ecx
  char v4; // cl
  int v5; // eax
  unsigned __int64 v6; // rdx

  if ( !a1 )
    return 0;
  _BitScanReverse(&v1, a1);
  v2 = v1 ^ 0x1F;
  v3 = v2;
  if ( (char)(v2 - 21) >= 0 )
  {
    LOBYTE(v3) = v2 - 21;
    return (unsigned int)((unsigned __int64)a1 << ((unsigned __int8)v2 - 21)) + ((24 - v3) << 10);
  }
  else
  {
    v4 = v2 - 17;
    v5 = (char)(v2 - 17);
    v6 = (unsigned __int64)a1 << v4;
    if ( v4 < 0 )
      v6 = (a1 >> -(char)v5) | (a1 << v4 != 0);
    return sub_12F9B80(0, 28 - v5, v6);
  }
}
