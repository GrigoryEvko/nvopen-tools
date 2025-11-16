// Function: sub_131DCA0
// Address: 0x131dca0
//
__int64 __fastcall sub_131DCA0(__int64 a1)
{
  char *v1; // rax
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 result; // rax

  *(_DWORD *)(a1 + 24) = 0;
  v1 = off_4C6F2B8[0];
  *(_QWORD *)(a1 + 40) = -1;
  *(_QWORD *)(a1 + 32) = v1;
  *(_QWORD *)(a1 + 48) = -1;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  memset(*(void **)(a1 + 80), 0, 0x2880u);
  *(_QWORD *)(*(_QWORD *)(a1 + 80) + 10368LL) = 0;
  *(_QWORD *)(*(_QWORD *)(a1 + 80) + 10376LL) = 0;
  *(_QWORD *)(*(_QWORD *)(a1 + 80) + 10384LL) = 0;
  *(_QWORD *)(*(_QWORD *)(a1 + 80) + 10392LL) = 0;
  *(_QWORD *)(*(_QWORD *)(a1 + 80) + 10400LL) = 0;
  *(_QWORD *)(*(_QWORD *)(a1 + 80) + 10408LL) = 0;
  v2 = *(_QWORD *)(a1 + 80);
  *(_QWORD *)(v2 + 10416) = 0;
  *(_QWORD *)(v2 + 15592) = 0;
  memset(
    (void *)((v2 + 10424) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)v2 - (((_DWORD)v2 + 10424) & 0xFFFFFFF8) + 15600) >> 3));
  memset((void *)(*(_QWORD *)(a1 + 80) + 15600LL), 0, 0x24C0u);
  memset((void *)(*(_QWORD *)(a1 + 80) + 25008LL), 0, 0x2550u);
  v3 = *(_QWORD *)(a1 + 80);
  *(_QWORD *)(v3 + 34560) = 0;
  *(_QWORD *)(v3 + 37752) = 0;
  memset(
    (void *)((v3 + 34568) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)v3 - (((_DWORD)v3 + 34568) & 0xFFFFFFF8) + 37760) >> 3));
  result = *(_QWORD *)(a1 + 80);
  *(_QWORD *)(result + 37760) = 0;
  return result;
}
