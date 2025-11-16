// Function: sub_30B0D20
// Address: 0x30b0d20
//
__int64 __fastcall sub_30B0D20(__int64 a1, __int64 a2)
{
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 40) = a1 + 56;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)a1 = &unk_4A32430;
  *(_QWORD *)(a1 + 64) = a1 + 80;
  *(_DWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 56) = 1;
  *(_QWORD *)(a1 + 80) = a2;
  *(_QWORD *)(a1 + 72) = 0x200000001LL;
  return 0x200000001LL;
}
