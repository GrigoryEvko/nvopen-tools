// Function: sub_6879B0
// Address: 0x6879b0
//
_BOOL8 sub_6879B0()
{
  __int64 v0; // rax
  char v1; // dl
  _BOOL8 result; // rax
  __int64 v3; // rax
  __int64 v4; // rdx

  v0 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v1 = *(_BYTE *)(v0 + 4);
  if ( v1 == 1 )
  {
    v1 = *(_BYTE *)(v0 - 772);
    v0 -= 776;
  }
  if ( (unsigned __int8)(v1 - 8) <= 1u )
  {
    v1 = *(_BYTE *)(v0 - 772);
    v0 -= 776;
  }
  if ( (unsigned __int8)(v1 - 6) > 1u
    || (v3 = *(_QWORD *)(v0 + 208), *(_BYTE *)(v3 + 140) != 9)
    || (v4 = *(_QWORD *)(v3 + 168), result = 1, (*(_BYTE *)(v4 + 109) & 0x20) == 0) )
  {
    result = 0;
    if ( unk_4D03C50 )
      return *(_QWORD *)(unk_4D03C50 + 112LL) != 0;
  }
  return result;
}
