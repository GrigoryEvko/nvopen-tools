// Function: sub_33CA300
// Address: 0x33ca300
//
__int64 __fastcall sub_33CA300(__int64 a1, int a2, int a3, unsigned __int8 **a4, __int64 a5, int a6)
{
  unsigned __int8 *v6; // rsi

  *(_DWORD *)(a1 + 24) = a2;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 28) = 0;
  *(_WORD *)(a1 + 34) = -1;
  *(_DWORD *)(a1 + 36) = -1;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = a5;
  *(_QWORD *)(a1 + 56) = 0;
  *(_DWORD *)(a1 + 64) = 0;
  *(_DWORD *)(a1 + 68) = a6;
  *(_DWORD *)(a1 + 72) = a3;
  v6 = *a4;
  *(_QWORD *)(a1 + 80) = *a4;
  if ( v6 )
  {
    sub_B976B0((__int64)a4, v6, a1 + 80);
    *a4 = 0;
  }
  *(_QWORD *)(a1 + 88) = 0xFFFFFFFFLL;
  *(_WORD *)(a1 + 32) = 0;
  return 0;
}
