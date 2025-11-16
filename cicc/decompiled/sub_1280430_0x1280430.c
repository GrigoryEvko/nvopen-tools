// Function: sub_1280430
// Address: 0x1280430
//
__int64 __fastcall sub_1280430(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6, unsigned __int8 a7)
{
  *(_QWORD *)(a1 + 8) = a3;
  *(_DWORD *)a1 = 1;
  *(_QWORD *)(a1 + 24) = a5;
  *(_QWORD *)(a1 + 32) = a4;
  *(_DWORD *)(a1 + 16) = a6;
  *(_DWORD *)(a1 + 40) = a7;
  return a1;
}
