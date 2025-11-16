// Function: sub_3100780
// Address: 0x3100780
//
__int64 __fastcall sub_3100780(__int64 a1, __int64 a2)
{
  *(_QWORD *)(a1 + 400) = 0;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 160) = a1 + 176;
  *(_QWORD *)(a1 + 8) = 0x800000000LL;
  *(_QWORD *)(a1 + 88) = 0x800000000LL;
  *(_QWORD *)(a1 + 168) = 0x800000000LL;
  *(_QWORD *)(a1 + 240) = a1 + 256;
  *(_QWORD *)(a1 + 248) = 0x800000000LL;
  *(_QWORD *)(a1 + 328) = 0x800000000LL;
  *(_QWORD *)(a1 + 320) = a1 + 336;
  *(_QWORD *)(a1 + 408) = a1 + 432;
  *(_QWORD *)(a1 + 416) = 32;
  *(_DWORD *)(a1 + 424) = 0;
  *(_BYTE *)(a1 + 428) = 1;
  *(_QWORD *)(a1 + 688) = a2;
  return a1 + 432;
}
