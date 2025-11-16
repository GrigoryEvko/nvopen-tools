// Function: sub_19E5A00
// Address: 0x19e5a00
//
__int64 __fastcall sub_19E5A00(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // r15d
  __int64 v5; // r12

  v4 = *(_DWORD *)(a2 + 20);
  v5 = sub_145CBF0((__int64 *)(a1 + 64), 64, 16);
  *(_QWORD *)(v5 + 8) = 0xFFFFFFFD0000000ALL;
  *(_DWORD *)(v5 + 32) = v4 & 0xFFFFFFF;
  *(_QWORD *)(v5 + 48) = a3;
  *(_QWORD *)(v5 + 56) = a2;
  *(_QWORD *)(v5 + 16) = 0;
  *(_QWORD *)(v5 + 24) = 0;
  *(_DWORD *)(v5 + 36) = 0;
  *(_QWORD *)(v5 + 40) = 0;
  *(_QWORD *)v5 = &unk_49F4DD0;
  sub_19E5840(a1, a2, v5);
  return v5;
}
