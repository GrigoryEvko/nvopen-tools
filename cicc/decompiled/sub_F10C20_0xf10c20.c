// Function: sub_F10C20
// Address: 0xf10c20
//
__int64 __fastcall sub_F10C20(__int64 a1, __int64 a2, int a3)
{
  *(_QWORD *)(a1 + 2064) = 0;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x10000000000LL;
  *(_QWORD *)(a1 + 2128) = a1 + 2144;
  *(_QWORD *)(a1 + 2072) = 0;
  *(_QWORD *)(a1 + 2080) = 0;
  *(_DWORD *)(a1 + 2088) = 0;
  *(_QWORD *)(a1 + 2096) = 0;
  *(_QWORD *)(a1 + 2104) = 0;
  *(_QWORD *)(a1 + 2112) = 0;
  *(_DWORD *)(a1 + 2120) = 0;
  *(_QWORD *)(a1 + 2136) = 0x1000000000LL;
  *(_QWORD *)(a1 + 2272) = a2;
  *(_DWORD *)(a1 + 2280) = a3;
  return 0x1000000000LL;
}
