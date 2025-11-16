// Function: sub_2E1DCC0
// Address: 0x2e1dcc0
//
unsigned __int64 __fastcall sub_2E1DCC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  unsigned __int64 result; // rax

  *(_QWORD *)a1 = a2;
  v6 = *(_QWORD *)(a2 + 32);
  *(_QWORD *)(a1 + 16) = a3;
  *(_QWORD *)(a1 + 8) = v6;
  *(_QWORD *)(a1 + 24) = a4;
  *(_QWORD *)(a1 + 32) = a5;
  result = sub_2E1D8A0(a1, a2, a3, a4, a5, a6);
  *(_DWORD *)(a1 + 192) = 0;
  return result;
}
