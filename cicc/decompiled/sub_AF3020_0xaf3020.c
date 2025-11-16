// Function: sub_AF3020
// Address: 0xaf3020
//
__int64 __fastcall sub_AF3020(
        __int64 a1,
        int a2,
        int a3,
        int a4,
        char a5,
        int a6,
        int a7,
        __int64 a8,
        char a9,
        unsigned int a10,
        int a11,
        char a12,
        __int64 a13,
        __int64 a14)
{
  sub_B971C0(a1, a2, 17, a3, a13, a14, 0, 0);
  *(_WORD *)(a1 + 2) = 17;
  *(_QWORD *)(a1 + 24) = a8;
  *(_DWORD *)(a1 + 16) = a4;
  *(_DWORD *)(a1 + 32) = a7;
  *(_DWORD *)(a1 + 20) = a6;
  *(_DWORD *)(a1 + 36) = a11;
  *(_BYTE *)(a1 + 40) = a5;
  *(_BYTE *)(a1 + 43) = a12;
  *(_BYTE *)(a1 + 41) = a9;
  *(_BYTE *)(a1 + 42) = a10;
  return a10;
}
