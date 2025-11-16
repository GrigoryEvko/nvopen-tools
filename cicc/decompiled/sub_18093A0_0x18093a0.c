// Function: sub_18093A0
// Address: 0x18093a0
//
__int64 __fastcall sub_18093A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5, __int64 a6, __int64 a7)
{
  __int64 v9; // rax
  _QWORD v11[8]; // [rsp+10h] [rbp-40h] BYREF

  v11[0] = a6;
  v11[1] = a7;
  v9 = sub_1644EA0(a5, v11, 2, 0);
  return sub_1632080(a1, a2, a3, v9, a4);
}
