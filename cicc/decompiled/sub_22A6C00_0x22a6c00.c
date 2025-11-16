// Function: sub_22A6C00
// Address: 0x22a6c00
//
__int64 __fastcall sub_22A6C00(__int64 a1)
{
  unsigned int v1; // eax
  int v2; // edx

  v2 = *(_DWORD *)(a1 + 12);
  LOBYTE(v1) = v2 == 3;
  LOBYTE(v2) = v2 == 8;
  return v2 | v1;
}
