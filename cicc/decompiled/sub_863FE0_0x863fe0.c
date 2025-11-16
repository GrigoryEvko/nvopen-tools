// Function: sub_863FE0
// Address: 0x863fe0
//
_QWORD *__fastcall sub_863FE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  _QWORD *result; // rax
  __int64 v7; // rdx
  unsigned int v8; // r13d
  int v9; // r12d
  int v10; // r15d

  result = (_QWORD *)dword_4F04C64;
  v7 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v8 = *(_DWORD *)(v7 + 568);
  if ( v8 )
  {
    *(_DWORD *)(v7 + 568) = v8 - 1;
    return result;
  }
  v9 = *(_DWORD *)(v7 + 572);
  v10 = *(_DWORD *)(v7 + 576);
  if ( (*(_BYTE *)(v7 + 9) & 1) != 0 )
  {
    sub_7B8260();
    while ( dword_4F04C64 > v9 )
LABEL_5:
      sub_863FC0(a1, a2, v7, a4, a5, a6);
  }
  else if ( v9 < dword_4F04C64 )
  {
    goto LABEL_5;
  }
  unk_4F04C2C = v10;
  if ( unk_4F04C48 != -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) != 0 )
  {
    if ( dword_4D047C8 )
      v8 = sub_7D3BE0(a1, a2, v7, a4, a5);
  }
  return sub_85FE80(dword_4F04C64, 1, v8);
}
