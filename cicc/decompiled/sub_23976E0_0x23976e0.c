// Function: sub_23976E0
// Address: 0x23976e0
//
_QWORD *__fastcall sub_23976E0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // rax
  _QWORD *v5; // rbx
  _BYTE v7[288]; // [rsp+0h] [rbp-260h] BYREF
  _BYTE v8[320]; // [rsp+120h] [rbp-140h] BYREF

  sub_FFA780((__int64)v7, a2 + 8, a3, a4);
  sub_234F330((__int64)v8, (__int64)v7);
  v4 = (_QWORD *)sub_22077B0(0x120u);
  v5 = v4;
  if ( v4 )
  {
    *v4 = &unk_4A0B128;
    sub_234F330((__int64)(v4 + 1), (__int64)v8);
  }
  sub_D77880((__int64)v8);
  *a1 = v5;
  sub_D77880((__int64)v7);
  return a1;
}
