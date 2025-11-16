// Function: sub_1693480
// Address: 0x1693480
//
__int64 __fastcall sub_1693480(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  *(_BYTE *)(a1 + 44) &= 0xFCu;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = a3;
  *(_QWORD *)(a1 + 24) = a4;
  *(_QWORD *)(a1 + 32) = a5;
  *(_QWORD *)(a1 + 16) = a9;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_DWORD *)(a1 + 40) = a6;
  *(_QWORD *)(a1 + 64) = a7;
  *(_QWORD *)(a1 + 72) = a8;
  *(_QWORD *)(a1 + 56) = 0x200000002LL;
  return 0x200000002LL;
}
