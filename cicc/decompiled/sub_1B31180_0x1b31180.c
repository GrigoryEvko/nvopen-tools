// Function: sub_1B31180
// Address: 0x1b31180
//
void __fastcall sub_1B31180(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  *(_QWORD *)(a1 + 24) = a3;
  *(_QWORD *)(a1 + 72) = a3;
  *(_QWORD *)a1 = a1 | 4;
  *(_QWORD *)(a1 + 3264) = a1 + 3280;
  *(_QWORD *)(a1 + 3272) = 0x1400000000LL;
  *(_QWORD *)(a1 + 32) = a4;
  *(_QWORD *)(a1 + 112) = a1 + 128;
  *(_QWORD *)(a1 + 3464) = a1 + 3448;
  *(_QWORD *)(a1 + 3472) = a1 + 3448;
  *(_QWORD *)(a1 + 8) = a1;
  *(_QWORD *)(a1 + 16) = a2;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_DWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 104) = 0;
  *(_DWORD *)(a1 + 124) = 32;
  *(_QWORD *)(a1 + 3200) = 0;
  *(_QWORD *)(a1 + 3208) = 0;
  *(_QWORD *)(a1 + 3216) = 0;
  *(_DWORD *)(a1 + 3224) = 0;
  *(_QWORD *)(a1 + 3232) = 0;
  *(_QWORD *)(a1 + 3240) = 0;
  *(_QWORD *)(a1 + 3248) = 0;
  *(_DWORD *)(a1 + 3256) = 0;
  *(_DWORD *)(a1 + 3448) = 0;
  *(_QWORD *)(a1 + 3456) = 0;
  *(_QWORD *)(a1 + 3480) = 0;
  memset((void *)(a1 + 128), 0, 0x60u);
  *(_QWORD *)(a1 + 128) = a1 + 144;
  *(_QWORD *)(a1 + 136) = 0x400000000LL;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  *(_QWORD *)(a1 + 184) = 0x400000000LL;
  *(_DWORD *)(a1 + 120) = 1;
  sub_1B306C0((__int64 *)a1);
}
