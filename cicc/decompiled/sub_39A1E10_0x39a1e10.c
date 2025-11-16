// Function: sub_39A1E10
// Address: 0x39a1e10
//
void *__fastcall sub_39A1E10(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int16 v6; // ax

  v6 = sub_3971A70(a2);
  *(_QWORD *)(a1 + 88) = a2;
  *(_QWORD *)(a1 + 96) = a3;
  *(_DWORD *)(a1 + 64) = v6;
  *(_QWORD *)(a1 + 104) = a4;
  *(_QWORD *)(a1 + 8) = a1 + 24;
  *(_QWORD *)(a1 + 16) = 0x200000000LL;
  *(_QWORD *)(a1 + 56) = 0;
  *(_DWORD *)(a1 + 68) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 80) = 0;
  *(_QWORD *)a1 = &unk_4A3FC88;
  return &unk_4A3FC88;
}
