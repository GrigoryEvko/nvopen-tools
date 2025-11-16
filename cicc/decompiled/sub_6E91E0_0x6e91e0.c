// Function: sub_6E91E0
// Address: 0x6e91e0
//
__int64 __fastcall sub_6E91E0(unsigned int a1, _DWORD *a2)
{
  __int64 v2; // rax
  __int64 result; // rax
  int v4; // r8d

  if ( !word_4D04898 )
    return 0;
  v2 = qword_4D03C50;
  if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 1) == 0 )
    return 0;
  *(_BYTE *)(qword_4D03C50 + 19LL) |= 0x20u;
  if ( *(_BYTE *)(v2 + 16) > 3u || (*(_BYTE *)(v2 + 18) & 0x10) != 0 )
    return 0;
  v4 = sub_6E5430();
  result = 1;
  if ( v4 )
  {
    sub_6851C0(a1, a2);
    return 1;
  }
  return result;
}
