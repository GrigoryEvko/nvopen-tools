// Function: sub_1B47BD0
// Address: 0x1b47bd0
//
_QWORD *__fastcall sub_1B47BD0(__int64 *a1, __int64 a2, __int64 a3, __int64 *a4)
{
  _QWORD *v7; // r12
  __int64 v8; // rax
  __int64 v9; // rdi
  unsigned __int64 *v10; // r13
  __int64 v11; // rax
  unsigned __int64 v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // rsi
  unsigned __int8 *v15; // rsi
  __int64 v17; // rax
  __int64 v18; // rdi
  unsigned __int64 *v19; // r13
  __int64 v20; // rax
  unsigned __int64 v21; // rcx
  __int64 v22; // rsi
  __int64 v23; // rsi
  unsigned __int8 *v24; // rsi
  unsigned __int8 *v25; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v26[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v27; // [rsp+20h] [rbp-40h]

  if ( *(_BYTE *)(a3 + 16) > 0x10u )
    goto LABEL_12;
  v7 = (_QWORD *)a2;
  if ( sub_1593BB0(a3, a2, a3, (__int64)a4) )
    return v7;
  if ( *(_BYTE *)(a2 + 16) > 0x10u )
  {
LABEL_12:
    v27 = 257;
    v17 = sub_15FB440(27, (__int64 *)a2, a3, (__int64)v26, 0);
    v18 = a1[1];
    v7 = (_QWORD *)v17;
    if ( v18 )
    {
      v19 = (unsigned __int64 *)a1[2];
      sub_157E9D0(v18 + 40, v17);
      v20 = v7[3];
      v21 = *v19;
      v7[4] = v19;
      v21 &= 0xFFFFFFFFFFFFFFF8LL;
      v7[3] = v21 | v20 & 7;
      *(_QWORD *)(v21 + 8) = v7 + 3;
      *v19 = *v19 & 7 | (unsigned __int64)(v7 + 3);
    }
    sub_164B780((__int64)v7, a4);
    v22 = *a1;
    if ( *a1 )
    {
      v25 = (unsigned __int8 *)*a1;
      sub_1623A60((__int64)&v25, v22, 2);
      v23 = v7[6];
      if ( v23 )
        sub_161E7C0((__int64)(v7 + 6), v23);
      v24 = v25;
      v7[6] = v25;
      if ( v24 )
        sub_1623210((__int64)&v25, v24, (__int64)(v7 + 6));
    }
  }
  else
  {
    v27 = 257;
    v8 = sub_15FB440(27, (__int64 *)a2, a3, (__int64)v26, 0);
    v9 = a1[1];
    v7 = (_QWORD *)v8;
    if ( v9 )
    {
      v10 = (unsigned __int64 *)a1[2];
      sub_157E9D0(v9 + 40, v8);
      v11 = v7[3];
      v12 = *v10;
      v7[4] = v10;
      v12 &= 0xFFFFFFFFFFFFFFF8LL;
      v7[3] = v12 | v11 & 7;
      *(_QWORD *)(v12 + 8) = v7 + 3;
      *v10 = *v10 & 7 | (unsigned __int64)(v7 + 3);
    }
    sub_164B780((__int64)v7, a4);
    v13 = *a1;
    if ( *a1 )
    {
      v26[0] = *a1;
      sub_1623A60((__int64)v26, v13, 2);
      v14 = v7[6];
      if ( v14 )
        sub_161E7C0((__int64)(v7 + 6), v14);
      v15 = (unsigned __int8 *)v26[0];
      v7[6] = v26[0];
      if ( v15 )
        sub_1623210((__int64)v26, v15, (__int64)(v7 + 6));
    }
  }
  return v7;
}
