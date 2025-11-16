// Function: sub_2304DA0
// Address: 0x2304da0
//
_QWORD *__fastcall sub_2304DA0(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // rax
  _QWORD *v4; // rbx
  _BYTE v6[80]; // [rsp+0h] [rbp-C0h] BYREF
  _QWORD v7[14]; // [rsp+50h] [rbp-70h] BYREF

  sub_D12090((__int64)v6, a3);
  sub_D0F960(v7, (__int64)v6);
  v3 = (_QWORD *)sub_22077B0(0x50u);
  v4 = v3;
  if ( v3 )
  {
    *v3 = &unk_4A0B4C0;
    sub_D0F960(v3 + 1, (__int64)v7);
  }
  sub_D0FA70((__int64)v7);
  *a1 = v4;
  sub_D0FA70((__int64)v6);
  return a1;
}
