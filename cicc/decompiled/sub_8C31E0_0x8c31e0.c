// Function: sub_8C31E0
// Address: 0x8c31e0
//
__int64 __fastcall sub_8C31E0(__int64 a1)
{
  char v1; // al
  unsigned int v2; // r8d

  v1 = *(_BYTE *)(a1 - 8);
  v2 = 1;
  if ( (v1 & 2) == 0 && ((v1 & 4) != 0) != dword_4D03B64 )
  {
    v2 = 0;
    *(_BYTE *)(a1 - 8) = (4 * (dword_4D03B64 & 1)) | v1 & 0xFB;
  }
  return v2;
}
