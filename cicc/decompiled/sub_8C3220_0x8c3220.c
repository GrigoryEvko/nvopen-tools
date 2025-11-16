// Function: sub_8C3220
// Address: 0x8c3220
//
__int64 __fastcall sub_8C3220(__int64 a1, char a2)
{
  unsigned int v2; // r8d

  v2 = 1;
  if ( ((*(_BYTE *)(a1 - 8) & 4) != 0) == dword_4D03B64 )
    return v2;
  *(_BYTE *)(a1 - 8) = (4 * (dword_4D03B64 & 1)) | *(_BYTE *)(a1 - 8) & 0xFB;
  if ( a2 != 40 )
  {
    v2 = 0;
    if ( a2 == 11 && (*(_BYTE *)(a1 + 194) & 0x40) == 0 )
      *(_QWORD *)(a1 + 232) = 0;
    return v2;
  }
  *(_QWORD *)(a1 + 128) = 0;
  return 0;
}
