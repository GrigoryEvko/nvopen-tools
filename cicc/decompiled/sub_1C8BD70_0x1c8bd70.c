// Function: sub_1C8BD70
// Address: 0x1c8bd70
//
unsigned __int64 __fastcall sub_1C8BD70(
        _BYTE *a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  unsigned __int64 result; // rax
  __int64 v16; // rax
  unsigned __int8 *v17; // rsi
  __int64 v18; // rax
  __int64 *v19; // rdi
  __int64 v20; // r11
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // r12
  __int64 v24; // rax
  double v25; // xmm4_8
  double v26; // xmm5_8
  __int64 v27; // rsi
  __int64 v28; // [rsp+8h] [rbp-C8h]
  _QWORD v30[2]; // [rsp+20h] [rbp-B0h] BYREF
  unsigned __int8 *v31[2]; // [rsp+30h] [rbp-A0h] BYREF
  _QWORD v32[2]; // [rsp+40h] [rbp-90h] BYREF
  __int64 v33[5]; // [rsp+50h] [rbp-80h] BYREF
  int v34; // [rsp+78h] [rbp-58h]
  __int64 v35; // [rsp+80h] [rbp-50h]
  __int64 v36; // [rsp+88h] [rbp-48h]

  result = sub_16431D0(*a2);
  if ( (_DWORD)result == 128 )
  {
    v16 = sub_16498A0((__int64)a2);
    v17 = (unsigned __int8 *)a2[6];
    v33[0] = 0;
    v33[3] = v16;
    v18 = a2[5];
    v33[4] = 0;
    v34 = 0;
    v35 = 0;
    v36 = 0;
    v33[1] = v18;
    v33[2] = (__int64)(a2 + 3);
    v31[0] = v17;
    if ( v17 )
    {
      sub_1623A60((__int64)v31, (__int64)v17, 2);
      if ( v33[0] )
        sub_161E7C0((__int64)v33, v33[0]);
      v33[0] = (__int64)v31[0];
      if ( v31[0] )
        sub_1623210((__int64)v31, v31[0], (__int64)v33);
      v18 = a2[5];
    }
    v19 = (__int64 *)*a2;
    v20 = *(_QWORD *)(*(_QWORD *)(v18 + 56) + 40LL);
    v21 = *(a2 - 6);
    v31[0] = (unsigned __int8 *)v32;
    v32[0] = v19;
    v30[0] = v21;
    v28 = v20;
    v30[1] = *(a2 - 3);
    v32[1] = v19;
    v31[1] = (unsigned __int8 *)0x200000002LL;
    v22 = sub_1644EA0(v19, v32, 2, 0);
    v23 = sub_1632080(v28, a3, a4, v22, 0);
    if ( (_QWORD *)v31[0] != v32 )
      _libc_free((unsigned __int64)v31[0]);
    LOWORD(v32[0]) = 257;
    v24 = sub_1285290(v33, *(_QWORD *)(*(_QWORD *)v23 + 24LL), v23, (int)v30, 2, (__int64)v31, 0);
    sub_164D160((__int64)a2, v24, a5, a6, a7, a8, v25, v26, a11, a12);
    sub_15F20C0(a2);
    result = (unsigned __int64)a1;
    v27 = v33[0];
    *a1 = 1;
    if ( v27 )
      return sub_161E7C0((__int64)v33, v27);
  }
  return result;
}
