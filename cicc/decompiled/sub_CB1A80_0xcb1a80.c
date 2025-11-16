// Function: sub_CB1A80
// Address: 0xcb1a80
//
__int64 __fastcall sub_CB1A80(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  sub_CB09D0((_QWORD *)a1, a3);
  *(_QWORD *)(a1 + 16) = a2;
  *(_DWORD *)(a1 + 24) = a4;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)a1 = &unk_49DCF98;
  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)(a1 + 40) = 0x800000000LL;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  return 0x800000000LL;
}
