// Function: sub_2189770
// Address: 0x2189770
//
void *__fastcall sub_2189770(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  *(_QWORD *)(a1 + 8) = 0;
  *(_WORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 16) = a2;
  *(_QWORD *)(a1 + 24) = a3;
  *(_QWORD *)(a1 + 32) = a4;
  *(_DWORD *)(a1 + 44) = 0;
  *(_QWORD *)a1 = &unk_4A03298;
  return &unk_4A03298;
}
