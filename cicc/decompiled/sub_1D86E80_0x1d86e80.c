// Function: sub_1D86E80
// Address: 0x1d86e80
//
_QWORD *__fastcall sub_1D86E80(_QWORD *a1, _QWORD *a2, __int64 *a3, double a4, double a5, double a6)
{
  __int64 v8; // rax
  __int64 v9; // rsi
  unsigned int v10; // ebx
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // r15
  _BYTE *v14; // rsi
  __int64 v15; // rdx
  _QWORD *v16; // r14
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rsi
  __int64 v23; // rdx
  unsigned __int8 *v24; // rsi
  _QWORD *v26; // rax
  _BYTE *v27; // r8
  __int64 v29; // [rsp+18h] [rbp-88h]
  __int64 *v30; // [rsp+18h] [rbp-88h]
  unsigned __int8 *v31; // [rsp+28h] [rbp-78h] BYREF
  __int64 v32; // [rsp+30h] [rbp-70h] BYREF
  __int16 v33; // [rsp+40h] [rbp-60h]
  _QWORD v34[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v35; // [rsp+60h] [rbp-40h]

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  v8 = *a3;
  v9 = a3[1];
  if ( v9 - *a3 == 8 )
  {
    v27 = 0;
    goto LABEL_27;
  }
  v10 = 0;
  v11 = 0;
  do
  {
    v33 = 257;
    v13 = *(_QWORD *)(v8 + 8 * v11);
    v12 = *(_QWORD *)(v8 + 8LL * (v10 + 1));
    v16 = (_QWORD *)*a2;
    if ( *(_BYTE *)(v12 + 16) <= 0x10u )
    {
      v29 = *(_QWORD *)(v8 + 8LL * (v10 + 1));
      if ( sub_1593BB0(v29, v9, v12, v11) )
        goto LABEL_6;
      v12 = v29;
      if ( *(_BYTE *)(v13 + 16) <= 0x10u )
      {
        v13 = sub_15A2D10((__int64 *)v13, v29, a4, a5, a6);
LABEL_6:
        v34[0] = v13;
        v14 = (_BYTE *)a1[1];
        if ( v14 == (_BYTE *)a1[2] )
          goto LABEL_19;
        goto LABEL_7;
      }
    }
    v35 = 257;
    v17 = sub_15FB440(27, (__int64 *)v13, v12, (__int64)v34, 0);
    v18 = v16[16];
    v13 = v17;
    if ( v18 )
    {
      v30 = (__int64 *)v16[17];
      sub_157E9D0(v18 + 40, v17);
      v19 = *v30;
      v20 = *(_QWORD *)(v13 + 24) & 7LL;
      *(_QWORD *)(v13 + 32) = v30;
      v19 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v13 + 24) = v19 | v20;
      *(_QWORD *)(v19 + 8) = v13 + 24;
      *v30 = *v30 & 7 | (v13 + 24);
    }
    sub_164B780(v13, &v32);
    v21 = v16[15];
    if ( !v21 )
      goto LABEL_6;
    v31 = (unsigned __int8 *)v16[15];
    sub_1623A60((__int64)&v31, v21, 2);
    v22 = *(_QWORD *)(v13 + 48);
    v23 = v13 + 48;
    if ( v22 )
    {
      sub_161E7C0(v13 + 48, v22);
      v23 = v13 + 48;
    }
    v24 = v31;
    *(_QWORD *)(v13 + 48) = v31;
    if ( !v24 )
      goto LABEL_6;
    sub_1623210((__int64)&v31, v24, v23);
    v34[0] = v13;
    v14 = (_BYTE *)a1[1];
    if ( v14 == (_BYTE *)a1[2] )
    {
LABEL_19:
      sub_1287830((__int64)a1, v14, v34);
      goto LABEL_10;
    }
LABEL_7:
    if ( v14 )
    {
      *(_QWORD *)v14 = v13;
      v14 = (_BYTE *)a1[1];
    }
    a1[1] = v14 + 8;
LABEL_10:
    v9 = a3[1];
    v8 = *a3;
    v10 += 2;
    v11 = v10;
    v15 = (v9 - *a3) >> 3;
  }
  while ( v10 < (unsigned __int64)(v15 - 1) );
  if ( (v15 & 1) == 0 )
    return a1;
  v26 = (_QWORD *)a1[1];
  v27 = (_BYTE *)a1[2];
  if ( v26 == (_QWORD *)v27 )
  {
LABEL_27:
    sub_1287830((__int64)a1, v27, (_QWORD *)(v9 - 8));
    return a1;
  }
  if ( v26 )
  {
    *v26 = *(_QWORD *)(v9 - 8);
    v26 = (_QWORD *)a1[1];
  }
  a1[1] = v26 + 1;
  return a1;
}
