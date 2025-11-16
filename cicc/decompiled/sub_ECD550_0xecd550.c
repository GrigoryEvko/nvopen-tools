// Function: sub_ECD550
// Address: 0xecd550
//
__int64 __fastcall sub_ECD550(__int64 a1)
{
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_BYTE *)(a1 + 88) = 0;
  *(_QWORD *)a1 = &unk_49E49F8;
  *(_QWORD *)(a1 + 8) = a1 + 24;
  *(_QWORD *)(a1 + 72) = a1 + 88;
  *(_WORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 16777217;
  *(_BYTE *)(a1 + 120) = 0;
  *(_DWORD *)(a1 + 124) = 10;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_DWORD *)(a1 + 56) = 64;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 16) = 0x100000001LL;
  *(_DWORD *)(a1 + 24) = 11;
  return 0x100000001LL;
}
