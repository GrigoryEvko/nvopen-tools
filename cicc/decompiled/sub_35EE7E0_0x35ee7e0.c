// Function: sub_35EE7E0
// Address: 0x35ee7e0
//
void *__fastcall sub_35EE7E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  *(_QWORD *)(a1 + 8) = 0;
  *(_WORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = a1 + 80;
  *(_QWORD *)(a1 + 72) = 0x400000001LL;
  *(_QWORD *)(a1 + 16) = a2;
  *(_QWORD *)(a1 + 24) = a3;
  *(_QWORD *)(a1 + 32) = a4;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0x10000;
  *(_DWORD *)(a1 + 80) = 17;
  *(_QWORD *)a1 = &unk_4A3AE60;
  return &unk_4A3AE60;
}
