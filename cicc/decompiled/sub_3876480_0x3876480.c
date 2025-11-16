// Function: sub_3876480
// Address: 0x3876480
//
_QWORD *__fastcall sub_3876480(
        __int64 *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 **v13; // r14
  __int64 v14; // rax
  __int64 **v15; // rax
  double v16; // xmm4_8
  double v17; // xmm5_8
  __int64 ***v18; // rax
  _QWORD *v19; // r12
  __int64 v20; // rax
  __int64 v22; // rax
  __int64 v23; // rdi
  unsigned __int64 *v24; // r14
  __int64 v25; // rax
  unsigned __int64 v26; // rcx
  __int64 v27; // rsi
  __int64 v28; // rsi
  unsigned __int8 *v29; // rsi
  unsigned __int8 *v30; // [rsp+8h] [rbp-68h] BYREF
  __int64 v31; // [rsp+10h] [rbp-60h] BYREF
  __int16 v32; // [rsp+20h] [rbp-50h]
  char v33[16]; // [rsp+30h] [rbp-40h] BYREF
  __int16 v34; // [rsp+40h] [rbp-30h]

  v11 = sub_1456E10(*a1, *(_QWORD *)(a2 + 40));
  v12 = *a1;
  v13 = (__int64 **)v11;
  v14 = sub_1456040(*(_QWORD *)(a2 + 32));
  v15 = (__int64 **)sub_1456E10(v12, v14);
  v18 = sub_38761C0(a1, *(_QWORD *)(a2 + 32), v15, a3, a4, a5, a6, v16, v17, a9, a10);
  v32 = 257;
  v19 = v18;
  if ( v13 != *v18 )
  {
    if ( *((_BYTE *)v18 + 16) > 0x10u )
    {
      v34 = 257;
      v22 = sub_15FDBD0(37, (__int64)v18, (__int64)v13, (__int64)v33, 0);
      v23 = a1[34];
      v19 = (_QWORD *)v22;
      if ( v23 )
      {
        v24 = (unsigned __int64 *)a1[35];
        sub_157E9D0(v23 + 40, v22);
        v25 = v19[3];
        v26 = *v24;
        v19[4] = v24;
        v26 &= 0xFFFFFFFFFFFFFFF8LL;
        v19[3] = v26 | v25 & 7;
        *(_QWORD *)(v26 + 8) = v19 + 3;
        *v24 = *v24 & 7 | (unsigned __int64)(v19 + 3);
      }
      sub_164B780((__int64)v19, &v31);
      v27 = a1[33];
      if ( v27 )
      {
        v30 = (unsigned __int8 *)a1[33];
        sub_1623A60((__int64)&v30, v27, 2);
        v28 = v19[6];
        if ( v28 )
          sub_161E7C0((__int64)(v19 + 6), v28);
        v29 = v30;
        v19[6] = v30;
        if ( v29 )
          sub_1623210((__int64)&v30, v29, (__int64)(v19 + 6));
      }
    }
    else
    {
      v19 = (_QWORD *)sub_15A46C0(37, v18, v13, 0);
      v20 = sub_14DBA30((__int64)v19, a1[41], 0);
      if ( v20 )
        v19 = (_QWORD *)v20;
    }
  }
  sub_38740E0((__int64)a1, (__int64)v19);
  return v19;
}
