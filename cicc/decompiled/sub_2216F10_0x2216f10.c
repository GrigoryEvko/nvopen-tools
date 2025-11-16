// Function: sub_2216F10
// Address: 0x2216f10
//
__int64 __fastcall sub_2216F10(__int64 a1, __int64 a2, char a3, __int64 a4)
{
  __int64 v5; // rbx
  _QWORD *v6; // rax

  v5 = a2;
  *(_DWORD *)(a1 + 8) = a4 != 0;
  *(_QWORD *)a1 = off_4A053F0;
  v6 = (_QWORD *)sub_2208E60();
  *(_QWORD *)(a1 + 16) = v6;
  *(_BYTE *)(a1 + 24) = a3 & (a2 != 0);
  *(_QWORD *)(a1 + 32) = v6[15];
  *(_QWORD *)(a1 + 40) = v6[14];
  if ( !a2 )
    v5 = v6[13];
  *(_QWORD *)(a1 + 48) = v5;
  *(_BYTE *)(a1 + 56) = 0;
  *(_BYTE *)(a1 + 569) = 0;
  *(_QWORD *)(a1 + 57) = 0;
  *(_QWORD *)(a1 + 305) = 0;
  memset(
    (void *)((a1 + 65) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)a1 - (((_DWORD)a1 + 65) & 0xFFFFFFF8) + 313) >> 3));
  *(_QWORD *)(a1 + 313) = 0;
  *(_QWORD *)(a1 + 561) = 0;
  memset(
    (void *)((a1 + 321) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)a1 - (((_DWORD)a1 + 321) & 0xFFFFFFF8) + 569) >> 3));
  return 0;
}
