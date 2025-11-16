// Function: sub_16E4AB0
// Address: 0x16e4ab0
//
__int64 __fastcall sub_16E4AB0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  sub_16E3E10((_QWORD *)a1, a3);
  *(_QWORD *)(a1 + 16) = a2;
  *(_DWORD *)(a1 + 24) = a4;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)a1 = &unk_49EF9A8;
  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)(a1 + 40) = 0x800000000LL;
  *(_QWORD *)(a1 + 88) = 0;
  *(_BYTE *)(a1 + 96) = 0;
  return 0x800000000LL;
}
