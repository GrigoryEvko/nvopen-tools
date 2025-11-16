// Function: sub_1B47DD0
// Address: 0x1b47dd0
//
_QWORD *__fastcall sub_1B47DD0(__int64 *a1, __int64 a2, __int64 *a3)
{
  bool v5; // cc
  __int64 v6; // rax
  __int64 v7; // rdi
  _QWORD *v8; // r12
  unsigned __int64 *v9; // r15
  __int64 v10; // rax
  unsigned __int64 v11; // rcx
  __int64 v12; // rsi
  __int64 v13; // rsi
  unsigned __int8 *v14; // rsi
  __int64 v16; // rax
  __int64 v17; // rdi
  unsigned __int64 *v18; // r14
  __int64 v19; // rax
  unsigned __int64 v20; // rcx
  __int64 v21; // rsi
  __int64 v22; // rsi
  unsigned __int8 *v23; // rsi
  unsigned __int8 *v24; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v25[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v26; // [rsp+20h] [rbp-40h]

  v5 = *(_BYTE *)(a2 + 16) <= 0x10u;
  v26 = 257;
  if ( v5 )
  {
    v6 = sub_15FB630((__int64 *)a2, (__int64)v25, 0);
    v7 = a1[1];
    v8 = (_QWORD *)v6;
    if ( v7 )
    {
      v9 = (unsigned __int64 *)a1[2];
      sub_157E9D0(v7 + 40, v6);
      v10 = v8[3];
      v11 = *v9;
      v8[4] = v9;
      v11 &= 0xFFFFFFFFFFFFFFF8LL;
      v8[3] = v11 | v10 & 7;
      *(_QWORD *)(v11 + 8) = v8 + 3;
      *v9 = *v9 & 7 | (unsigned __int64)(v8 + 3);
    }
    sub_164B780((__int64)v8, a3);
    v12 = *a1;
    if ( *a1 )
    {
      v25[0] = *a1;
      sub_1623A60((__int64)v25, v12, 2);
      v13 = v8[6];
      if ( v13 )
        sub_161E7C0((__int64)(v8 + 6), v13);
      v14 = (unsigned __int8 *)v25[0];
      v8[6] = v25[0];
      if ( v14 )
        sub_1623210((__int64)v25, v14, (__int64)(v8 + 6));
    }
  }
  else
  {
    v16 = sub_15FB630((__int64 *)a2, (__int64)v25, 0);
    v17 = a1[1];
    v8 = (_QWORD *)v16;
    if ( v17 )
    {
      v18 = (unsigned __int64 *)a1[2];
      sub_157E9D0(v17 + 40, v16);
      v19 = v8[3];
      v20 = *v18;
      v8[4] = v18;
      v20 &= 0xFFFFFFFFFFFFFFF8LL;
      v8[3] = v20 | v19 & 7;
      *(_QWORD *)(v20 + 8) = v8 + 3;
      *v18 = *v18 & 7 | (unsigned __int64)(v8 + 3);
    }
    sub_164B780((__int64)v8, a3);
    v21 = *a1;
    if ( *a1 )
    {
      v24 = (unsigned __int8 *)*a1;
      sub_1623A60((__int64)&v24, v21, 2);
      v22 = v8[6];
      if ( v22 )
        sub_161E7C0((__int64)(v8 + 6), v22);
      v23 = v24;
      v8[6] = v24;
      if ( v23 )
        sub_1623210((__int64)&v24, v23, (__int64)(v8 + 6));
    }
  }
  return v8;
}
