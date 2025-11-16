// Function: sub_1B74740
// Address: 0x1b74740
//
__int64 __fastcall sub_1B74740(
        __int64 a1,
        unsigned int a2,
        __int64 *a3,
        __int64 a4,
        _BYTE *a5,
        double a6,
        double a7,
        double a8)
{
  __int64 v12; // rax
  unsigned __int8 *v13; // rsi
  __int64 v14; // rax
  __int64 v15; // r12
  unsigned __int8 *v17; // [rsp+18h] [rbp-88h] BYREF
  __int64 v18[5]; // [rsp+20h] [rbp-80h] BYREF
  int v19; // [rsp+48h] [rbp-58h]
  __int64 v20; // [rsp+50h] [rbp-50h]
  __int64 v21; // [rsp+58h] [rbp-48h]

  v12 = sub_16498A0(a4);
  v13 = *(unsigned __int8 **)(a4 + 48);
  v18[0] = 0;
  v18[3] = v12;
  v14 = *(_QWORD *)(a4 + 40);
  v18[4] = 0;
  v18[1] = v14;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v18[2] = a4 + 24;
  v17 = v13;
  if ( v13 )
  {
    sub_1623A60((__int64)&v17, (__int64)v13, 2);
    if ( v18[0] )
      sub_161E7C0((__int64)v18, v18[0]);
    v18[0] = (__int64)v17;
    if ( v17 )
      sub_1623210((__int64)&v17, v17, (__int64)v18);
  }
  v15 = sub_1B73C70(a1, a2, a3, v18, a5, a6, a7, a8);
  if ( v18[0] )
    sub_161E7C0((__int64)v18, v18[0]);
  return v15;
}
