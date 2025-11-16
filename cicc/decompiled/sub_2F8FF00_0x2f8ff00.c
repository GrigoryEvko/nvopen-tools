// Function: sub_2F8FF00
// Address: 0x2f8ff00
//
__int64 __fastcall sub_2F8FF00(__int64 a1, __int64 a2, __int64 a3)
{
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 24) = a1 + 40;
  *(_QWORD *)(a1 + 32) = 0x1000000000LL;
  *(_QWORD *)(a1 + 344) = a1 + 360;
  *(_QWORD *)(a1 + 8) = a3;
  *(_BYTE *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 320) = 0;
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 336) = 0;
  *(_QWORD *)(a1 + 352) = 0x600000000LL;
  *(_DWORD *)(a1 + 408) = 0;
  return 0x600000000LL;
}
