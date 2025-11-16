// Function: sub_7CEAF0
// Address: 0x7ceaf0
//
_BOOL8 __fastcall sub_7CEAF0(__int64 a1, int a2, __int64 a3)
{
  int v3; // eax
  _BOOL8 result; // rax
  char v6; // dl

  v3 = 19;
  if ( (a2 & 0x2000) == 0 )
  {
    if ( (a2 & 0x40C03) != 0
      || (v3 = 2, dword_4F04C64 != -1)
      && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12) & 2) != 0
      && (a2 & 4) != 0 )
    {
      v3 = ((*(_BYTE *)(a3 + 16) & 0x38) == 0) + 2;
    }
  }
  if ( *(unsigned __int8 *)(a1 + 80) != v3 )
    return 0;
  v6 = *(_BYTE *)(sub_87D520(a1) + 90);
  if ( ((v6 & 0x10) != 0) != ((a2 & 0x800000) != 0) )
    return 0;
  result = 1;
  if ( (v6 & 0x10) != 0 )
    return ((v6 & 0x20) != 0) == (*(_BYTE *)(a3 + 16) & 1);
  return result;
}
