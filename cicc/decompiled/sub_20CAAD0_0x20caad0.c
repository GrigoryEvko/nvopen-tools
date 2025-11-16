// Function: sub_20CAAD0
// Address: 0x20caad0
//
__int64 __fastcall sub_20CAAD0(
        __int64 *a1,
        void (__fastcall *a2)(__int64, __int64 *, __int64, __int64, __int64, _QWORD, unsigned __int8 **, __int64 *),
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v13; // rax
  unsigned __int8 *v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rax
  double v17; // xmm4_8
  double v18; // xmm5_8
  __int64 *v20; // [rsp+8h] [rbp-88h] BYREF
  unsigned __int8 *v21; // [rsp+18h] [rbp-78h] BYREF
  __int64 v22[5]; // [rsp+20h] [rbp-70h] BYREF
  int v23; // [rsp+48h] [rbp-48h]
  __int64 v24; // [rsp+50h] [rbp-40h]
  __int64 v25; // [rsp+58h] [rbp-38h]

  v20 = a1;
  v13 = sub_16498A0((__int64)a1);
  v24 = 0;
  v25 = 0;
  v14 = (unsigned __int8 *)a1[6];
  v22[3] = v13;
  v23 = 0;
  v15 = a1[5];
  v22[0] = 0;
  v22[1] = v15;
  v22[4] = 0;
  v22[2] = (__int64)(a1 + 3);
  v21 = v14;
  if ( v14 )
  {
    sub_1623A60((__int64)&v21, (__int64)v14, 2);
    if ( v22[0] )
      sub_161E7C0((__int64)v22, v22[0]);
    v22[0] = (__int64)v21;
    if ( v21 )
      sub_1623210((__int64)&v21, v21, (__int64)v22);
  }
  v21 = (unsigned __int8 *)&v20;
  v16 = sub_20C9DC0(
          v22,
          *v20,
          *(v20 - 6),
          (*((unsigned __int16 *)v20 + 9) >> 2) & 7,
          (__int64 (__fastcall *)(__int64, __int64 *, __int64))sub_20CD3A0,
          (__int64)&v21,
          a2,
          a3);
  sub_164D160((__int64)v20, v16, a4, a5, a6, a7, v17, v18, a10, a11);
  sub_15F20C0(v20);
  if ( v22[0] )
    sub_161E7C0((__int64)v22, v22[0]);
  return 1;
}
