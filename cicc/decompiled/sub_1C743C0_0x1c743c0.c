// Function: sub_1C743C0
// Address: 0x1c743c0
//
_QWORD *__fastcall sub_1C743C0(__int64 a1, unsigned __int8 a2, __int64 a3, double a4, double a5, double a6)
{
  __int64 v8; // r14
  unsigned __int8 *v9; // rsi
  __int64 v10; // rax
  bool v11; // cc
  __int64 v12; // rax
  unsigned __int8 *v13; // rsi
  _QWORD *v14; // r12
  __int64 v16; // rax
  unsigned __int64 *v17; // rbx
  __int64 v18; // rax
  unsigned __int64 v19; // rcx
  __int64 v20; // rsi
  unsigned __int8 *v21; // rsi
  unsigned __int8 *v22; // [rsp-C0h] [rbp-C0h] BYREF
  void *v23; // [rsp-B8h] [rbp-B8h] BYREF
  char v24; // [rsp-A8h] [rbp-A8h]
  char v25; // [rsp-A7h] [rbp-A7h]
  unsigned __int8 *v26[2]; // [rsp-98h] [rbp-98h] BYREF
  __int16 v27; // [rsp-88h] [rbp-88h]
  unsigned __int8 *v28; // [rsp-78h] [rbp-78h] BYREF
  __int64 v29; // [rsp-70h] [rbp-70h]
  unsigned __int64 *v30; // [rsp-68h] [rbp-68h]
  __int64 v31; // [rsp-60h] [rbp-60h]
  __int64 v32; // [rsp-58h] [rbp-58h]
  int v33; // [rsp-50h] [rbp-50h]
  __int64 v34; // [rsp-48h] [rbp-48h]
  __int64 v35; // [rsp-40h] [rbp-40h]

  if ( !a3 )
    BUG();
  v8 = *(_QWORD *)(a3 + 16);
  v29 = v8;
  v28 = 0;
  v31 = sub_157E9C0(v8);
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v30 = (unsigned __int64 *)a3;
  if ( a3 != v8 + 40 )
  {
    v9 = *(unsigned __int8 **)(a3 + 24);
    v26[0] = v9;
    if ( v9 )
    {
      sub_1623A60((__int64)v26, (__int64)v9, 2);
      if ( v28 )
        sub_161E7C0((__int64)&v28, (__int64)v28);
      v28 = v26[0];
      if ( v26[0] )
        sub_1623210((__int64)v26, v26[0], (__int64)&v28);
    }
  }
  v10 = sub_15A0680(*(_QWORD *)a1, 1, a2);
  v11 = *(_BYTE *)(a1 + 16) <= 0x10u;
  v25 = 1;
  v24 = 3;
  v23 = &unk_42D2000;
  if ( v11 && *(_BYTE *)(v10 + 16) <= 0x10u )
  {
    v12 = sub_15A2B30((__int64 *)a1, v10, 0, 0, a4, a5, a6);
    v13 = v28;
    v14 = (_QWORD *)v12;
LABEL_11:
    if ( v13 )
      sub_161E7C0((__int64)&v28, (__int64)v13);
    return v14;
  }
  v27 = 257;
  v16 = sub_15FB440(11, (__int64 *)a1, v10, (__int64)v26, 0);
  v14 = (_QWORD *)v16;
  if ( v29 )
  {
    v17 = v30;
    sub_157E9D0(v29 + 40, v16);
    v18 = v14[3];
    v19 = *v17;
    v14[4] = v17;
    v19 &= 0xFFFFFFFFFFFFFFF8LL;
    v14[3] = v19 | v18 & 7;
    *(_QWORD *)(v19 + 8) = v14 + 3;
    *v17 = *v17 & 7 | (unsigned __int64)(v14 + 3);
  }
  sub_164B780((__int64)v14, (__int64 *)&v23);
  if ( v28 )
  {
    v22 = v28;
    sub_1623A60((__int64)&v22, (__int64)v28, 2);
    v20 = v14[6];
    if ( v20 )
      sub_161E7C0((__int64)(v14 + 6), v20);
    v21 = v22;
    v14[6] = v22;
    if ( v21 )
      sub_1623210((__int64)&v22, v21, (__int64)(v14 + 6));
    v13 = v28;
    goto LABEL_11;
  }
  return v14;
}
