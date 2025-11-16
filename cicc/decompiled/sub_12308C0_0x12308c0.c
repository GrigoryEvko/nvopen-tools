// Function: sub_12308C0
// Address: 0x12308c0
//
__int64 __fastcall sub_12308C0(__int64 **a1, _QWORD *a2, __int64 *a3)
{
  unsigned int v3; // r12d
  __int64 v5; // r14
  _QWORD *v6; // rax
  _QWORD *v7; // r13
  __int64 v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = sub_122FE20(a1, v8, a3);
  if ( (_BYTE)v3 )
    return v3;
  v5 = v8[0];
  v6 = sub_BD2C40(72, 1u);
  v7 = v6;
  if ( v6 )
    sub_B4BCC0((__int64)v6, v5, 0, 0);
  *a2 = v7;
  return v3;
}
