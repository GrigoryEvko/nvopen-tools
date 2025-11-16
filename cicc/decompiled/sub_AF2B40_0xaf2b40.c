// Function: sub_AF2B40
// Address: 0xaf2b40
//
__int64 __fastcall sub_AF2B40(
        __int64 a1,
        int a2,
        int a3,
        int a4,
        __int64 a5,
        int a6,
        unsigned int a7,
        __int64 a8,
        __int64 a9)
{
  sub_B971C0(a1, a2, 36, a3, a8, a9, 0, 0);
  *(_DWORD *)(a1 + 16) = a4;
  *(_WORD *)(a1 + 2) = 33;
  *(_QWORD *)(a1 + 24) = a5;
  *(_DWORD *)(a1 + 4) = a6;
  *(_DWORD *)(a1 + 20) = a7;
  *(_QWORD *)(a1 + 32) = 0;
  *(_DWORD *)(a1 + 40) = 0;
  return a7;
}
