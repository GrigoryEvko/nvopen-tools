// Function: sub_1456320
// Address: 0x1456320
//
__int64 __fastcall sub_1456320(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 16) = a3;
  *(_DWORD *)(a1 + 24) = 2;
  *(_QWORD *)(a1 + 32) = a4;
  *(_QWORD *)(a1 + 40) = a5;
  return 0;
}
