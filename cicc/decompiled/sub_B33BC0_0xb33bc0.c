// Function: sub_B33BC0
// Address: 0xb33bc0
//
__int64 __fastcall sub_B33BC0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  __int64 v8; // rax
  _QWORD v11[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = sub_AA4B30(*(_QWORD *)(a1 + 48));
  v11[0] = *(_QWORD *)(a3 + 8);
  v8 = sub_B6E160(v7, a2, v11, 1);
  v11[0] = a3;
  return sub_B33A00(a1, v8, (int)v11, 1, a5, a4, 0, 0);
}
