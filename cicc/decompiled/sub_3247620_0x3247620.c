// Function: sub_3247620
// Address: 0x3247620
//
void *__fastcall sub_3247620(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int16 v6; // ax
  __int16 v7; // dx

  v6 = sub_31DF670(a2);
  *(_QWORD *)(a1 + 16) = a3;
  *(_QWORD *)(a1 + 24) = a1 + 40;
  v7 = *(_WORD *)(a1 + 100);
  *(_QWORD *)(a1 + 104) = a2;
  *(_QWORD *)(a1 + 112) = a4;
  *(_BYTE *)(a1 + 8) = 0;
  *(_WORD *)(a1 + 100) = v7 & 0xE000 | (v6 << 9) & 0x1E00;
  *(_QWORD *)(a1 + 32) = 0x200000000LL;
  *(_QWORD *)(a1 + 88) = 0;
  *(_DWORD *)(a1 + 96) = 0;
  *(_BYTE *)(a1 + 103) = 0;
  *(_QWORD *)a1 = &unk_4A35CD8;
  *(_QWORD *)(a1 + 120) = 0;
  *(_DWORD *)(a1 + 128) = 0;
  *(_BYTE *)(a1 + 136) = 0;
  return &unk_4A35CD8;
}
