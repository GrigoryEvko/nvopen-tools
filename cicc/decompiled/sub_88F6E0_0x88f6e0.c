// Function: sub_88F6E0
// Address: 0x88f6e0
//
__int64 __fastcall sub_88F6E0(__int64 a1)
{
  __int64 v1; // rbx
  __int64 result; // rax

  v1 = qword_4F04C68[0] + 776LL * (int)dword_4F04C5C;
  result = sub_85ED80(a1, v1);
  if ( !(_DWORD)result )
  {
    if ( *(_BYTE *)(v1 + 4) != 6 )
      return sub_685440(7u, 0x2F7u, a1);
    if ( !unk_4D0473C )
      return sub_685440(7u, 0x2F7u, a1);
    if ( (*(_BYTE *)(a1 + 81) & 0x10) == 0 )
      return sub_685440(7u, 0x2F7u, a1);
    result = *(_QWORD *)(v1 + 208);
    if ( *(_QWORD *)(a1 + 64) != result )
      return sub_685440(7u, 0x2F7u, a1);
  }
  return result;
}
