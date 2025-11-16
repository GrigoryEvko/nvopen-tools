// Function: sub_16A0BB0
// Address: 0x16a0bb0
//
__int64 __fastcall sub_16A0BB0(
        double a1,
        double a2,
        double a3,
        __int64 a4,
        __int64 *a5,
        __int64 *a6,
        __int64 *a7,
        unsigned int a8)
{
  bool v12; // r14
  void *v13; // r14
  __int64 *v14; // rsi
  __int64 *v15; // rsi
  __int64 *v16; // rsi
  __int64 *v17; // rsi
  bool v19; // al
  unsigned int v20; // [rsp+Ch] [rbp-C4h]
  __int64 v21; // [rsp+18h] [rbp-B8h]
  _BYTE v22[8]; // [rsp+20h] [rbp-B0h] BYREF
  _QWORD v23[3]; // [rsp+28h] [rbp-A8h] BYREF
  _BYTE v24[8]; // [rsp+40h] [rbp-90h] BYREF
  _QWORD v25[3]; // [rsp+48h] [rbp-88h] BYREF
  _BYTE v26[8]; // [rsp+60h] [rbp-70h] BYREF
  _QWORD v27[3]; // [rsp+68h] [rbp-68h] BYREF
  _BYTE v28[8]; // [rsp+80h] [rbp-50h] BYREF
  _QWORD v29[9]; // [rsp+88h] [rbp-48h] BYREF

  if ( (unsigned int)sub_169C920((__int64)a5) == 1 )
    goto LABEL_20;
  if ( (unsigned int)sub_169C920((__int64)a6) == 1 || (unsigned int)sub_169C920((__int64)a5) == 3 )
    goto LABEL_19;
  if ( (unsigned int)sub_169C920((__int64)a6) == 3 )
    goto LABEL_20;
  if ( !(unsigned int)sub_169C920((__int64)a5) && !(unsigned int)sub_169C920((__int64)a6) )
  {
    v12 = sub_169C950((__int64)a5);
    if ( v12 != sub_169C950((__int64)a6) )
    {
      v19 = sub_169C950((__int64)a7);
      sub_169CAA0((__int64)a7, 0, v19, 0, *(float *)&a1);
      return 1;
    }
  }
  if ( !(unsigned int)sub_169C920((__int64)a5) )
  {
LABEL_20:
    sub_16A0170(a7, a5);
    return 0;
  }
  if ( !(unsigned int)sub_169C920((__int64)a6) )
  {
LABEL_19:
    sub_16A0170(a7, a6);
    return 0;
  }
  v21 = a5[1];
  v13 = sub_16982C0();
  v14 = (__int64 *)(v21 + 8);
  if ( *(void **)(v21 + 8) == v13 )
    sub_169C6E0(v23, (__int64)v14);
  else
    sub_16986C0(v23, v14);
  v15 = (__int64 *)(a5[1] + 40);
  if ( (void *)*v15 == v13 )
    sub_169C6E0(v25, (__int64)v15);
  else
    sub_16986C0(v25, v15);
  v16 = (__int64 *)(a6[1] + 8);
  if ( (void *)*v16 == v13 )
    sub_169C6E0(v27, (__int64)v16);
  else
    sub_16986C0(v27, v16);
  v17 = (__int64 *)(a6[1] + 40);
  if ( (void *)*v17 == v13 )
    sub_169C6E0(v29, (__int64)v17);
  else
    sub_16986C0(v29, v17);
  v20 = sub_16A0410((__int64)a7, (__int64)v22, (__int64)v24, (__int64)v26, (__int64)v28, a8, a1, a2, a3);
  sub_127D120(v29);
  sub_127D120(v27);
  sub_127D120(v25);
  sub_127D120(v23);
  return v20;
}
