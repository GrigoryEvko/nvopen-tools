// Function: sub_E92760
// Address: 0xe92760
//
__int64 __fastcall sub_E92760(__int64 a1, int a2, __int64 a3, __int64 a4, char a5, char a6, __int64 a7)
{
  char v7; // r9
  char v10; // r8
  __int64 v12; // rdi

  v7 = ((16 * a5) | (32 * a6)) & 0x3F;
  v10 = *(_BYTE *)(a1 + 48);
  v12 = a1 + 56;
  *(_QWORD *)(v12 - 56) = &unk_49E3588;
  *(_QWORD *)(v12 - 32) = 0;
  *(_QWORD *)(v12 - 40) = a7;
  *(_BYTE *)(v12 - 24) = 0;
  *(_DWORD *)(v12 - 20) = 0;
  *(_QWORD *)(v12 - 16) = 0;
  *(_BYTE *)(v12 - 8) = v10 & 0xC0 | v7;
  sub_E81B30(v12, 14, 0);
  *(_QWORD *)(a1 + 64) = a1;
  *(_QWORD *)(a1 + 88) = a1 + 104;
  *(_QWORD *)(a1 + 96) = 0x100000001LL;
  *(_QWORD *)(a1 + 128) = a3;
  *(_QWORD *)(a1 + 136) = a4;
  *(_DWORD *)(a1 + 144) = a2;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_DWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 112;
  return a1 + 112;
}
