// Function: sub_25CD050
// Address: 0x25cd050
//
__int64 __fastcall sub_25CD050(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11)
{
  *(_QWORD *)a1 = a2;
  *(_BYTE *)(a1 + 48) = 1;
  *(_QWORD *)(a1 + 56) = a2 + 8 * a3;
  *(_BYTE *)(a1 + 104) = 1;
  *(_QWORD *)(a1 + 8) = a7;
  *(_QWORD *)(a1 + 16) = a8;
  *(_QWORD *)(a1 + 24) = a9;
  *(_QWORD *)(a1 + 32) = a10;
  *(_QWORD *)(a1 + 40) = a11;
  *(_QWORD *)(a1 + 64) = a7;
  *(_QWORD *)(a1 + 72) = a8;
  *(_QWORD *)(a1 + 80) = a9;
  *(_QWORD *)(a1 + 88) = a10;
  *(_QWORD *)(a1 + 96) = a11;
  return a1;
}
