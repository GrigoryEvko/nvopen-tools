// Function: sub_173DE00
// Address: 0x173de00
//
_QWORD *__fastcall sub_173DE00(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        unsigned __int8 a5,
        double a6,
        double a7,
        double a8)
{
  _QWORD *v10; // r12
  __int64 v11; // rax
  __int64 v13; // rax
  __int64 v14; // rdi
  unsigned __int64 *v15; // r14
  __int64 v16; // rax
  unsigned __int64 v17; // rcx
  __int64 *v18; // rsi
  _QWORD *v19; // rdi
  __int64 v20; // rdx
  bool v21; // zf
  __int64 v22; // rsi
  _QWORD *v23; // r13
  __int64 v24; // r14
  __int64 v25; // rsi
  unsigned __int8 *v26; // rsi
  __int64 v27; // rdi
  unsigned __int64 *v28; // r14
  __int64 v29; // rax
  unsigned __int64 v30; // rcx
  __int64 v31; // rsi
  __int64 v32; // rsi
  _QWORD *v33; // [rsp+0h] [rbp-70h] BYREF
  _QWORD *v34; // [rsp+8h] [rbp-68h] BYREF
  _QWORD v35[2]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v36; // [rsp+20h] [rbp-50h]
  _QWORD v37[2]; // [rsp+30h] [rbp-40h] BYREF
  __int16 v38; // [rsp+40h] [rbp-30h]

  if ( *(_BYTE *)(a2 + 16) <= 0x10u && *(_BYTE *)(a3 + 16) <= 0x10u )
  {
    v10 = (_QWORD *)sub_15A2DA0((__int64 *)a2, a3, a5, a6, a7, a8);
    v11 = sub_14DBA30((__int64)v10, *(_QWORD *)(a1 + 96), 0);
    if ( v11 )
      return (_QWORD *)v11;
    return v10;
  }
  if ( a5 )
  {
    v36 = 257;
    v10 = (_QWORD *)sub_15FB440(25, (__int64 *)a2, a3, (__int64)v35, 0);
    sub_15F2350((__int64)v10, 1);
    v27 = *(_QWORD *)(a1 + 8);
    if ( v27 )
    {
      v28 = *(unsigned __int64 **)(a1 + 16);
      sub_157E9D0(v27 + 40, (__int64)v10);
      v29 = v10[3];
      v30 = *v28;
      v10[4] = v28;
      v30 &= 0xFFFFFFFFFFFFFFF8LL;
      v10[3] = v30 | v29 & 7;
      *(_QWORD *)(v30 + 8) = v10 + 3;
      *v28 = *v28 & 7 | (unsigned __int64)(v10 + 3);
    }
    v18 = a4;
    v19 = v10;
    sub_164B780((__int64)v10, a4);
    v21 = *(_QWORD *)(a1 + 80) == 0;
    v34 = v10;
    if ( !v21 )
    {
      (*(void (__fastcall **)(__int64, _QWORD **))(a1 + 88))(a1 + 64, &v34);
      v31 = *(_QWORD *)a1;
      if ( !*(_QWORD *)a1 )
        return v10;
      v23 = v37;
      v37[0] = *(_QWORD *)a1;
      v24 = (__int64)(v10 + 6);
      sub_1623A60((__int64)v37, v31, 2);
      v32 = v10[6];
      if ( v32 )
        sub_161E7C0((__int64)(v10 + 6), v32);
      v26 = (unsigned __int8 *)v37[0];
      v10[6] = v37[0];
      if ( !v26 )
        return v10;
      goto LABEL_22;
    }
LABEL_23:
    sub_4263D6(v19, v18, v20);
  }
  v38 = 257;
  v13 = sub_15FB440(25, (__int64 *)a2, a3, (__int64)v37, 0);
  v14 = *(_QWORD *)(a1 + 8);
  v10 = (_QWORD *)v13;
  if ( v14 )
  {
    v15 = *(unsigned __int64 **)(a1 + 16);
    sub_157E9D0(v14 + 40, v13);
    v16 = v10[3];
    v17 = *v15;
    v10[4] = v15;
    v17 &= 0xFFFFFFFFFFFFFFF8LL;
    v10[3] = v17 | v16 & 7;
    *(_QWORD *)(v17 + 8) = v10 + 3;
    *v15 = *v15 & 7 | (unsigned __int64)(v10 + 3);
  }
  v18 = a4;
  v19 = v10;
  sub_164B780((__int64)v10, a4);
  v21 = *(_QWORD *)(a1 + 80) == 0;
  v33 = v10;
  if ( v21 )
    goto LABEL_23;
  (*(void (__fastcall **)(__int64, _QWORD **))(a1 + 88))(a1 + 64, &v33);
  v22 = *(_QWORD *)a1;
  if ( !*(_QWORD *)a1 )
    return v10;
  v23 = v35;
  v35[0] = *(_QWORD *)a1;
  v24 = (__int64)(v10 + 6);
  sub_1623A60((__int64)v35, v22, 2);
  v25 = v10[6];
  if ( v25 )
    sub_161E7C0((__int64)(v10 + 6), v25);
  v26 = (unsigned __int8 *)v35[0];
  v10[6] = v35[0];
  if ( !v26 )
    return v10;
LABEL_22:
  sub_1623210((__int64)v23, v26, v24);
  return v10;
}
