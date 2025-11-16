// Function: sub_F77240
// Address: 0xf77240
//
signed __int64 __fastcall sub_F77240(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v8[2]; // [rsp+0h] [rbp-10h] BYREF

  v6 = *(_QWORD *)(a1 + 8);
  v8[0] = *(_QWORD *)(a1 + 16);
  v8[1] = v6;
  return sub_F76FE0(v8, a2, v8[0], a4, a5, a6);
}
