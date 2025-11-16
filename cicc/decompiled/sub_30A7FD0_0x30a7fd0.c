// Function: sub_30A7FD0
// Address: 0x30a7fd0
//
__int64 __fastcall sub_30A7FD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // [rsp+8h] [rbp-18h] BYREF

  *(_QWORD *)(a1 + 24) = a1 + 8;
  *(_QWORD *)(a1 + 32) = a1 + 8;
  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  v7 = a1;
  sub_30A6FB0(a2, (__int64)sub_30A8A60, (__int64)&v7, a4, a5, a6);
  return a1;
}
