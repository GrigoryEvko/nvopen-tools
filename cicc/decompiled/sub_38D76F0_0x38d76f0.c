// Function: sub_38D76F0
// Address: 0x38d76f0
//
__int64 __fastcall sub_38D76F0(__int64 a1, int a2, int a3, __int64 a4)
{
  __int64 v6; // rdi

  *(_BYTE *)(a1 + 44) &= 0xF8u;
  v6 = a1 + 48;
  *(_QWORD *)(v6 - 40) = a4;
  *(_QWORD *)(v6 - 32) = 0;
  *(_QWORD *)(v6 - 48) = &unk_4A3E548;
  *(_QWORD *)(v6 - 24) = 1;
  *(_QWORD *)(v6 - 12) = 0;
  sub_38CF760(v6, 13, 0, a1);
  *(_DWORD *)(a1 + 144) = a2;
  *(_QWORD *)(a1 + 104) = a1 + 96;
  *(_QWORD *)(a1 + 112) = a1 + 128;
  *(_DWORD *)(a1 + 148) = a3;
  *(_QWORD *)(a1 + 96) = (a1 + 96) | 4;
  *(_QWORD *)(a1 + 120) = 0x100000000LL;
  return 0x100000000LL;
}
