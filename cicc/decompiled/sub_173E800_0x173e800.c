// Function: sub_173E800
// Address: 0x173e800
//
_QWORD *__fastcall sub_173E800(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        unsigned __int8 a5,
        double a6,
        double a7,
        double a8)
{
  __int64 v11; // rax
  _QWORD *v12; // r12
  __int64 v13; // rax
  __int64 v15; // rax
  __int64 v16; // rdi
  unsigned __int64 *v17; // r13
  __int64 v18; // rax
  unsigned __int64 v19; // rcx
  __int64 *v20; // rsi
  _QWORD *v21; // rdi
  __int64 v22; // rdx
  bool v23; // zf
  __int64 v24; // rsi
  _QWORD *v25; // r13
  __int64 v26; // r14
  __int64 v27; // rsi
  unsigned __int8 *v28; // rsi
  __int64 v29; // rdi
  unsigned __int64 *v30; // r13
  __int64 v31; // rax
  unsigned __int64 v32; // rcx
  __int64 v33; // rsi
  __int64 v34; // rsi
  _QWORD *v35; // [rsp+0h] [rbp-70h] BYREF
  _QWORD *v36; // [rsp+8h] [rbp-68h] BYREF
  _QWORD v37[2]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v38; // [rsp+20h] [rbp-50h]
  _QWORD v39[2]; // [rsp+30h] [rbp-40h] BYREF
  __int16 v40; // [rsp+40h] [rbp-30h]

  v11 = sub_15A0680(*(_QWORD *)a2, a3, 0);
  if ( *(_BYTE *)(a2 + 16) <= 0x10u && *(_BYTE *)(v11 + 16) <= 0x10u )
  {
    v12 = (_QWORD *)sub_15A2D80((__int64 *)a2, v11, a5, a6, a7, a8);
    v13 = sub_14DBA30((__int64)v12, *(_QWORD *)(a1 + 96), 0);
    if ( v13 )
      return (_QWORD *)v13;
    return v12;
  }
  if ( a5 )
  {
    v38 = 257;
    v12 = (_QWORD *)sub_15FB440(24, (__int64 *)a2, v11, (__int64)v37, 0);
    sub_15F2350((__int64)v12, 1);
    v29 = *(_QWORD *)(a1 + 8);
    if ( v29 )
    {
      v30 = *(unsigned __int64 **)(a1 + 16);
      sub_157E9D0(v29 + 40, (__int64)v12);
      v31 = v12[3];
      v32 = *v30;
      v12[4] = v30;
      v32 &= 0xFFFFFFFFFFFFFFF8LL;
      v12[3] = v32 | v31 & 7;
      *(_QWORD *)(v32 + 8) = v12 + 3;
      *v30 = *v30 & 7 | (unsigned __int64)(v12 + 3);
    }
    v20 = a4;
    v21 = v12;
    sub_164B780((__int64)v12, a4);
    v23 = *(_QWORD *)(a1 + 80) == 0;
    v36 = v12;
    if ( !v23 )
    {
      (*(void (__fastcall **)(__int64, _QWORD **))(a1 + 88))(a1 + 64, &v36);
      v33 = *(_QWORD *)a1;
      if ( !*(_QWORD *)a1 )
        return v12;
      v25 = v39;
      v39[0] = *(_QWORD *)a1;
      v26 = (__int64)(v12 + 6);
      sub_1623A60((__int64)v39, v33, 2);
      v34 = v12[6];
      if ( v34 )
        sub_161E7C0((__int64)(v12 + 6), v34);
      v28 = (unsigned __int8 *)v39[0];
      v12[6] = v39[0];
      if ( !v28 )
        return v12;
      goto LABEL_22;
    }
LABEL_23:
    sub_4263D6(v21, v20, v22);
  }
  v40 = 257;
  v15 = sub_15FB440(24, (__int64 *)a2, v11, (__int64)v39, 0);
  v16 = *(_QWORD *)(a1 + 8);
  v12 = (_QWORD *)v15;
  if ( v16 )
  {
    v17 = *(unsigned __int64 **)(a1 + 16);
    sub_157E9D0(v16 + 40, v15);
    v18 = v12[3];
    v19 = *v17;
    v12[4] = v17;
    v19 &= 0xFFFFFFFFFFFFFFF8LL;
    v12[3] = v19 | v18 & 7;
    *(_QWORD *)(v19 + 8) = v12 + 3;
    *v17 = *v17 & 7 | (unsigned __int64)(v12 + 3);
  }
  v20 = a4;
  v21 = v12;
  sub_164B780((__int64)v12, a4);
  v23 = *(_QWORD *)(a1 + 80) == 0;
  v35 = v12;
  if ( v23 )
    goto LABEL_23;
  (*(void (__fastcall **)(__int64, _QWORD **))(a1 + 88))(a1 + 64, &v35);
  v24 = *(_QWORD *)a1;
  if ( !*(_QWORD *)a1 )
    return v12;
  v25 = v37;
  v37[0] = *(_QWORD *)a1;
  v26 = (__int64)(v12 + 6);
  sub_1623A60((__int64)v37, v24, 2);
  v27 = v12[6];
  if ( v27 )
    sub_161E7C0((__int64)(v12 + 6), v27);
  v28 = (unsigned __int8 *)v37[0];
  v12[6] = v37[0];
  if ( !v28 )
    return v12;
LABEL_22:
  sub_1623210((__int64)v25, v28, v26);
  return v12;
}
