// Function: sub_7D0430
// Address: 0x7d0430
//
__int64 __fastcall sub_7D0430(_QWORD *a1, int a2, __int64 a3)
{
  unsigned __int8 v4; // si

  v4 = 19;
  if ( (a2 & 0x2000) == 0 )
  {
    if ( (a2 & 0x40C03) != 0
      || (v4 = 2, dword_4F04C64 != -1)
      && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12) & 2) != 0
      && (a2 & 4) != 0 )
    {
      v4 = ((*(_BYTE *)(a3 + 16) & 0x38) == 0) + 2;
    }
  }
  return sub_7D0130(a1, v4, a2, a3);
}
