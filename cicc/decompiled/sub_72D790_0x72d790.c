// Function: sub_72D790
// Address: 0x72d790
//
__int64 __fastcall sub_72D790(__int64 a1, int a2, __int64 a3, int a4, _DWORD *a5)
{
  _QWORD *v7; // r15
  __int64 v8; // r12

  v7 = (_QWORD *)sub_8D46C0(a1);
  if ( a4
    && dword_4F04C64 != -1
    && (unk_4F04C48 == -1
     || (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) == 0
     || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0)
    && a5 )
  {
    if ( a4 == 4 )
      sub_684B30(0x6B3u, a5);
    else
      sub_684B30(0x6EAu, a5);
  }
  if ( !a2 && !(unsigned int)sub_8D30C0(a1) )
    return sub_72D600(v7);
  v8 = sub_8D21F0(a1);
  if ( (*(_BYTE *)(v8 + 140) & 0xFB) == 8 )
    sub_8D4C10(v8, dword_4F077C4 != 2);
  return v8;
}
