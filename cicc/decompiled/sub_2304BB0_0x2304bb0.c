// Function: sub_2304BB0
// Address: 0x2304bb0
//
_QWORD *__fastcall sub_2304BB0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // rax
  _QWORD *v5; // rbx
  __int64 v7[46]; // [rsp+0h] [rbp-300h] BYREF
  __int64 v8[50]; // [rsp+170h] [rbp-190h] BYREF

  sub_D21760((__int64)v7, a2 + 8, a3, a4);
  sub_D1D360((__int64)v8, v7);
  v4 = (_QWORD *)sub_22077B0(0x170u);
  v5 = v4;
  if ( v4 )
  {
    *v4 = &unk_4A15AC8;
    sub_D1D360((__int64)(v4 + 1), v8);
  }
  sub_D1D5E0((__int64)v8);
  *a1 = v5;
  sub_D1D5E0((__int64)v7);
  return a1;
}
