// Function: sub_922980
// Address: 0x922980
//
__int64 __fastcall sub_922980(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6, unsigned __int8 a7)
{
  *(_QWORD *)(a1 + 8) = a3;
  *(_DWORD *)a1 = 1;
  *(_QWORD *)(a1 + 32) = a5;
  *(_QWORD *)(a1 + 40) = a4;
  *(_DWORD *)(a1 + 24) = a6;
  *(_DWORD *)(a1 + 48) = a7;
  return a1;
}
