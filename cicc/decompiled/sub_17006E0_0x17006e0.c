// Function: sub_17006E0
// Address: 0x17006e0
//
__int64 __fastcall sub_17006E0(__int64 a1)
{
  char v1; // al
  int v2; // edx
  __int64 result; // rax
  int v4; // ecx

  v1 = *(_BYTE *)(a1 + 808);
  if ( v1 < 0 )
    return (v1 & 0x40) != 0;
  v2 = *(_DWORD *)(a1 + 520);
  result = 1;
  if ( v2 != 10 )
  {
    v4 = *(_DWORD *)(a1 + 516);
    if ( v4 != 13 )
    {
      LOBYTE(result) = v2 == 16;
      LOBYTE(v2) = v4 == 15;
      return v2 & (unsigned int)result;
    }
  }
  return result;
}
