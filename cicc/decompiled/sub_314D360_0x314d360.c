// Function: sub_314D360
// Address: 0x314d360
//
__int64 __fastcall sub_314D360(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __int64 a7,
        __int64 a8)
{
  *(_BYTE *)(a1 + 44) &= 0xF8u;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = a3;
  *(_QWORD *)(a1 + 24) = a4;
  *(_QWORD *)(a1 + 32) = a5;
  *(_QWORD *)(a1 + 16) = a8;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_DWORD *)(a1 + 40) = a6;
  *(_QWORD *)(a1 + 64) = a7;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 56) = 0x200000001LL;
  return 0x200000001LL;
}
