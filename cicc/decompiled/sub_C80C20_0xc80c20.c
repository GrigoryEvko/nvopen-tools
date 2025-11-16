// Function: sub_C80C20
// Address: 0xc80c20
//
__int64 __fastcall sub_C80C20(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  *(_OWORD *)(a1 + 32) = 0;
  *(_OWORD *)(a1 + 16) = 0;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = a3;
  *(_QWORD *)(a1 + 32) = a3;
  *(_DWORD *)(a1 + 40) = a4;
  sub_C80AD0(a1);
  return a1;
}
