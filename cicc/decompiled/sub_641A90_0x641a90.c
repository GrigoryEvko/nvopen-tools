// Function: sub_641A90
// Address: 0x641a90
//
__int64 __fastcall sub_641A90(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rax
  char v4; // al

  if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 9) & 0xE) == 6 && *a2 != a1 )
  {
    v3 = a2[5];
    if ( v3 )
    {
      v4 = *(_BYTE *)(v3 + 28);
      if ( !v4 || v4 == 3 )
        a2 = 0;
    }
  }
  if ( dword_4F04C58 != -1 )
    a1 = 0;
  return sub_877E90(a1, a2);
}
