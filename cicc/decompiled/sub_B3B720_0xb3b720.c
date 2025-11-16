// Function: sub_B3B720
// Address: 0xb3b720
//
__int64 __fastcall sub_B3B720(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        char a5,
        char a6,
        unsigned int a7,
        char a8)
{
  __int64 v11; // rax

  v11 = sub_BCE3C0(*a2, 0);
  sub_BD35F0(a1, v11, 25);
  *(_QWORD *)(a1 + 24) = a1 + 40;
  sub_B3AE60((__int64 *)(a1 + 24), *(_BYTE **)a3, *(_QWORD *)a3 + *(_QWORD *)(a3 + 8));
  *(_QWORD *)(a1 + 56) = a1 + 72;
  sub_B3AE60((__int64 *)(a1 + 56), *(_BYTE **)a4, *(_QWORD *)a4 + *(_QWORD *)(a4 + 8));
  *(_QWORD *)(a1 + 88) = a2;
  *(_BYTE *)(a1 + 97) = a6;
  *(_BYTE *)(a1 + 96) = a5;
  *(_DWORD *)(a1 + 100) = a7;
  *(_BYTE *)(a1 + 104) = a8;
  return a7;
}
