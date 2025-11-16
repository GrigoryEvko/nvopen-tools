// Function: sub_20D3190
// Address: 0x20d3190
//
_BOOL8 __fastcall sub_20D3190(
        __int64 *a1,
        __int64 *a2,
        __int64 a3,
        __int64 **a4,
        __int64 a5,
        int a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        __int64 *a15,
        __int64 *a16)
{
  __int16 v20; // r8
  _QWORD *v21; // rax
  _QWORD *v22; // r12
  __int64 v23; // rdi
  unsigned __int64 *v24; // rbx
  __int64 v25; // rax
  unsigned __int64 v26; // rsi
  __int64 v27; // rsi
  __int64 v28; // rsi
  __int64 v29; // rdx
  unsigned __int8 *v30; // rsi
  double v31; // xmm4_8
  double v32; // xmm5_8
  __int16 v34; // [rsp+Ch] [rbp-84h]
  __int64 v35; // [rsp+10h] [rbp-80h]
  unsigned int v36; // [rsp+18h] [rbp-78h]
  unsigned __int8 *v38; // [rsp+38h] [rbp-58h] BYREF
  __int64 v39[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v40; // [rsp+50h] [rbp-40h]

  v20 = a6;
  switch ( a6 )
  {
    case 0:
    case 1:
    case 3:
    case 7:
      break;
    case 2:
    case 5:
      a6 = 2;
      break;
    case 4:
    case 6:
      a6 = 4;
      break;
  }
  v35 = a5;
  v34 = v20;
  v36 = a6;
  v40 = 257;
  v21 = sub_1648A60(64, 3u);
  v22 = v21;
  if ( v21 )
    sub_15F99E0((__int64)v21, a3, a4, v35, v34, v36, 1, 0);
  v23 = a2[1];
  if ( v23 )
  {
    v24 = (unsigned __int64 *)a2[2];
    sub_157E9D0(v23 + 40, (__int64)v22);
    v25 = v22[3];
    v26 = *v24;
    v22[4] = v24;
    v26 &= 0xFFFFFFFFFFFFFFF8LL;
    v22[3] = v26 | v25 & 7;
    *(_QWORD *)(v26 + 8) = v22 + 3;
    *v24 = *v24 & 7 | (unsigned __int64)(v22 + 3);
  }
  sub_164B780((__int64)v22, v39);
  v27 = *a2;
  if ( *a2 )
  {
    v38 = (unsigned __int8 *)*a2;
    sub_1623A60((__int64)&v38, v27, 2);
    v28 = v22[6];
    v29 = (__int64)(v22 + 6);
    if ( v28 )
    {
      sub_161E7C0((__int64)(v22 + 6), v28);
      v29 = (__int64)(v22 + 6);
    }
    v30 = v38;
    v22[6] = v38;
    if ( v30 )
      sub_1623210((__int64)&v38, v30, v29);
  }
  v39[0] = (__int64)"success";
  v40 = 259;
  LODWORD(v38) = 1;
  *a15 = sub_12A9E60(a2, (__int64)v22, (__int64)&v38, 1, (__int64)v39);
  v39[0] = (__int64)"newloaded";
  v40 = 259;
  LODWORD(v38) = 0;
  *a16 = sub_12A9E60(a2, (__int64)v22, (__int64)&v38, 1, (__int64)v39);
  return sub_20D2E80(*a1, (__int64)v22, a7, a8, a9, a10, v31, v32, a13, a14);
}
