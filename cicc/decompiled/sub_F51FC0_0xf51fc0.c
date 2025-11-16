// Function: sub_F51FC0
// Address: 0xf51fc0
//
void __fastcall sub_F51FC0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r15
  __int64 v4; // rax
  __int64 v5; // r12
  _QWORD *v6; // [rsp+0h] [rbp-50h]
  __int64 v7; // [rsp+8h] [rbp-48h]
  __int64 v8[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = sub_B12000(a1 + 72);
  v3 = sub_B11F60(a1 + 80);
  if ( (unsigned __int8)sub_F50590(*(_QWORD *)(a2 + 8), a1) )
  {
    sub_AE7AF0((__int64)v8, a1);
    v6 = sub_B98A20(a2, a1);
    v7 = sub_B10CD0((__int64)v8);
    v4 = sub_22077B0(96);
    v5 = v4;
    if ( v4 )
      sub_B12150(v4, (__int64)v6, v2, v3, v7, 1);
    sub_AA8740(*(_QWORD *)(a2 + 40), v5, a2);
    if ( v8[0] )
      sub_B91220((__int64)v8, v8[0]);
  }
}
