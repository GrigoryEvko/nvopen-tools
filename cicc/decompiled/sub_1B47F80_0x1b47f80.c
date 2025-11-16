// Function: sub_1B47F80
// Address: 0x1b47f80
//
_QWORD *__fastcall sub_1B47F80(__int64 *a1, __int64 a2, __int64 a3, __int64 *a4)
{
  unsigned __int8 v6; // al
  __int64 v7; // rax
  __int64 v8; // rdi
  _QWORD *v9; // r12
  unsigned __int64 *v10; // r15
  __int64 v11; // rax
  unsigned __int64 v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // rsi
  unsigned __int8 *v15; // rsi
  unsigned int v17; // r14d
  int v18; // eax
  __int64 v19; // rax
  __int64 v20; // rdi
  unsigned __int64 *v21; // r14
  __int64 v22; // rax
  unsigned __int64 v23; // rcx
  __int64 v24; // rsi
  __int64 v25; // rsi
  unsigned __int8 *v26; // rsi
  __int64 v27; // [rsp+8h] [rbp-68h]
  unsigned __int8 *v28; // [rsp+18h] [rbp-58h] BYREF
  _QWORD v29[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v30; // [rsp+30h] [rbp-40h]

  v6 = *(_BYTE *)(a3 + 16);
  if ( v6 > 0x10u )
    goto LABEL_15;
  if ( v6 == 13 )
  {
    v17 = *(_DWORD *)(a3 + 32);
    if ( v17 > 0x40 )
    {
      v27 = a3;
      v18 = sub_16A58F0(a3 + 24);
      a3 = v27;
      v9 = (_QWORD *)a2;
      if ( v17 == v18 )
        return v9;
      if ( *(_BYTE *)(a2 + 16) <= 0x10u )
        goto LABEL_4;
      goto LABEL_15;
    }
    v9 = (_QWORD *)a2;
    if ( *(_QWORD *)(a3 + 24) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v17) )
      return v9;
  }
  if ( *(_BYTE *)(a2 + 16) <= 0x10u )
  {
LABEL_4:
    v30 = 257;
    v7 = sub_15FB440(26, (__int64 *)a2, a3, (__int64)v29, 0);
    v8 = a1[1];
    v9 = (_QWORD *)v7;
    if ( v8 )
    {
      v10 = (unsigned __int64 *)a1[2];
      sub_157E9D0(v8 + 40, v7);
      v11 = v9[3];
      v12 = *v10;
      v9[4] = v10;
      v12 &= 0xFFFFFFFFFFFFFFF8LL;
      v9[3] = v12 | v11 & 7;
      *(_QWORD *)(v12 + 8) = v9 + 3;
      *v10 = *v10 & 7 | (unsigned __int64)(v9 + 3);
    }
    sub_164B780((__int64)v9, a4);
    v13 = *a1;
    if ( *a1 )
    {
      v29[0] = *a1;
      sub_1623A60((__int64)v29, v13, 2);
      v14 = v9[6];
      if ( v14 )
        sub_161E7C0((__int64)(v9 + 6), v14);
      v15 = (unsigned __int8 *)v29[0];
      v9[6] = v29[0];
      if ( v15 )
        sub_1623210((__int64)v29, v15, (__int64)(v9 + 6));
    }
    return v9;
  }
LABEL_15:
  v30 = 257;
  v19 = sub_15FB440(26, (__int64 *)a2, a3, (__int64)v29, 0);
  v20 = a1[1];
  v9 = (_QWORD *)v19;
  if ( v20 )
  {
    v21 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v20 + 40, v19);
    v22 = v9[3];
    v23 = *v21;
    v9[4] = v21;
    v23 &= 0xFFFFFFFFFFFFFFF8LL;
    v9[3] = v23 | v22 & 7;
    *(_QWORD *)(v23 + 8) = v9 + 3;
    *v21 = *v21 & 7 | (unsigned __int64)(v9 + 3);
  }
  sub_164B780((__int64)v9, a4);
  v24 = *a1;
  if ( *a1 )
  {
    v28 = (unsigned __int8 *)*a1;
    sub_1623A60((__int64)&v28, v24, 2);
    v25 = v9[6];
    if ( v25 )
      sub_161E7C0((__int64)(v9 + 6), v25);
    v26 = v28;
    v9[6] = v28;
    if ( v26 )
      sub_1623210((__int64)&v28, v26, (__int64)(v9 + 6));
  }
  return v9;
}
