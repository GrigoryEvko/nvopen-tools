// Function: sub_AF3420
// Address: 0xaf3420
//
__int64 __fastcall sub_AF3420(
        __int64 a1,
        int a2,
        int a3,
        int a4,
        int a5,
        int a6,
        int a7,
        int a8,
        unsigned int a9,
        __int64 a10,
        __int64 a11)
{
  sub_B971C0(a1, a2, 18, a3, a10, a11, 0, 0);
  *(_DWORD *)(a1 + 16) = a4;
  *(_WORD *)(a1 + 2) = 46;
  *(_DWORD *)(a1 + 20) = a5;
  *(_DWORD *)(a1 + 28) = a7;
  *(_DWORD *)(a1 + 24) = a6;
  *(_DWORD *)(a1 + 32) = a8;
  *(_DWORD *)(a1 + 36) = a9;
  return a9;
}
