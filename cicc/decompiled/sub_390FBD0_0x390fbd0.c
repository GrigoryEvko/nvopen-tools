// Function: sub_390FBD0
// Address: 0x390fbd0
//
__int64 __fastcall sub_390FBD0(__int64 a1)
{
  char v1; // al

  *(_QWORD *)a1 = 0;
  *(_WORD *)(a1 + 12) = 0;
  v1 = *(_BYTE *)(a1 + 14);
  *(_DWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 14) = v1 & 0xFC | 2;
  *(_QWORD *)(a1 + 40) = 0x1000000000LL;
  *(_QWORD *)(a1 + 72) = a1 + 88;
  *(_QWORD *)(a1 + 80) = 0x400000000LL;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_BYTE *)(a1 + 64) = 0;
  *(_DWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 240) = a1 + 224;
  *(_QWORD *)(a1 + 248) = a1 + 224;
  *(_QWORD *)(a1 + 256) = 0;
  *(_QWORD *)(a1 + 264) = 0;
  *(_QWORD *)(a1 + 272) = 0;
  *(_QWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 288) = 0;
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  *(_BYTE *)(a1 + 312) = 0;
  return a1 + 224;
}
