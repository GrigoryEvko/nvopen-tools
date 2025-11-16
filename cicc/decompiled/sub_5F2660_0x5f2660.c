// Function: sub_5F2660
// Address: 0x5f2660
//
_BOOL8 sub_5F2660()
{
  __int64 v0; // rax
  _WORD *v2; // rax

  v0 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( *(_BYTE *)(v0 + 4) == 6 && (v2 = *(_WORD **)(v0 + 600), (v2[4] & 0x180) != 0) )
    return (*(_DWORD *)(*(_QWORD *)v2 + 176LL) & 0x44000) == 0;
  else
    return 0;
}
