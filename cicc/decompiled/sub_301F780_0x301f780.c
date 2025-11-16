// Function: sub_301F780
// Address: 0x301f780
//
__int64 __fastcall sub_301F780(__int64 a1, __int64 a2)
{
  sub_E989A0((_QWORD *)a1, a2);
  *(_BYTE *)(a1 + 160) = 0;
  *(_QWORD *)a1 = &unk_4A2E2B0;
  *(_QWORD *)(a1 + 16) = a1 + 32;
  *(_QWORD *)(a1 + 24) = 0x400000000LL;
  return 0x400000000LL;
}
