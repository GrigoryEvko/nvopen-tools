// Function: sub_39886F0
// Address: 0x39886f0
//
void *__fastcall sub_39886F0(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  int v4; // eax

  v4 = *(_DWORD *)(*(_QWORD *)(a4 + 232) + 504LL);
  *(_QWORD *)(a1 + 8) = a1 + 24;
  *(_QWORD *)(a1 + 16) = 0x200000000LL;
  *(_QWORD *)(a1 + 56) = 0;
  *(_DWORD *)(a1 + 64) = a2;
  *(_DWORD *)(a1 + 68) = 0;
  *(_BYTE *)(a1 + 80) = (unsigned int)(v4 - 34) <= 1;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 88) = a3;
  *(_QWORD *)a1 = &unk_4A3FB68;
  *(_QWORD *)(a1 + 96) = a4;
  return &unk_4A3FB68;
}
