// Function: sub_24A99B0
// Address: 0x24a99b0
//
__int64 __fastcall sub_24A99B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_ED2CB0(a3, *(char **)a2, *(_QWORD *)(a2 + 8));
  v5[0] = sub_24A5670(a3, 2);
  sub_2A41DC0(a3, v5, 1);
  if ( *(_BYTE *)(a2 + 32) )
    sub_24522D0(a3);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 32) = &unk_4F82418;
  *(_QWORD *)a1 = 1;
  if ( &unk_4F82418 != (_UNKNOWN *)&qword_4F82400 && &unk_4F82418 != &unk_4F82420 )
  {
    *(_DWORD *)(a1 + 20) = 2;
    *(_QWORD *)(a1 + 40) = &unk_4F82420;
    *(_QWORD *)a1 = 2;
  }
  return a1;
}
