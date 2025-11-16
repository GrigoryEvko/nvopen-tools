// Function: sub_16933F0
// Address: 0x16933f0
//
__int64 __fastcall sub_16933F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6, __int64 a7)
{
  *(_BYTE *)(a1 + 44) &= 0xFCu;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = a3;
  *(_QWORD *)(a1 + 24) = a4;
  *(_QWORD *)(a1 + 32) = a5;
  *(_QWORD *)(a1 + 16) = a7;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_DWORD *)(a1 + 40) = a6;
  *(_QWORD *)(a1 + 56) = 0x200000000LL;
  return 0x200000000LL;
}
