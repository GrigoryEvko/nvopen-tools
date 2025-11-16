// Function: sub_F9EE30
// Address: 0xf9ee30
//
__int64 __fastcall sub_F9EE30(_QWORD *a1, __int64 *a2)
{
  _QWORD *v3; // r12
  __int64 v4; // r9
  __int64 v5; // rdx
  _QWORD *v6; // r13
  _QWORD *v7; // rbx
  _QWORD *v8; // rax
  bool v9; // al
  _QWORD *v11; // rsi
  __int64 v12; // r8
  __int64 v13; // rdi
  __int64 v14; // rsi
  _QWORD *v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // [rsp+8h] [rbp-38h]

  v3 = a1 + 1;
  v4 = a1[2];
  if ( !v4 )
  {
    v6 = a1 + 1;
    v9 = 1;
LABEL_8:
    v18 = a1[5];
    if ( (_QWORD *)a1[3] == v6 && v9 )
    {
LABEL_10:
      sub_F8FE80(v4);
      a1[2] = 0;
      a1[3] = v3;
      a1[4] = v3;
      a1[5] = 0;
      return v18;
    }
    return 0;
  }
  v5 = *a2;
  v6 = a1 + 1;
  v7 = (_QWORD *)a1[2];
  while ( 1 )
  {
    while ( v7[4] < v5 )
    {
      v7 = (_QWORD *)v7[3];
      if ( !v7 )
        goto LABEL_7;
    }
    v8 = (_QWORD *)v7[2];
    if ( v7[4] <= v5 )
      break;
    v6 = v7;
    v7 = (_QWORD *)v7[2];
    if ( !v8 )
    {
LABEL_7:
      v9 = v3 == v6;
      goto LABEL_8;
    }
  }
  v11 = (_QWORD *)v7[3];
  if ( v11 )
  {
    do
    {
      while ( 1 )
      {
        v12 = v11[2];
        v13 = v11[3];
        if ( v5 < v11[4] )
          break;
        v11 = (_QWORD *)v11[3];
        if ( !v13 )
          goto LABEL_17;
      }
      v6 = v11;
      v11 = (_QWORD *)v11[2];
    }
    while ( v12 );
  }
LABEL_17:
  while ( v8 )
  {
    while ( 1 )
    {
      v14 = v8[3];
      if ( v5 <= v8[4] )
        break;
      v8 = (_QWORD *)v8[3];
      if ( !v14 )
        goto LABEL_20;
    }
    v7 = v8;
    v8 = (_QWORD *)v8[2];
  }
LABEL_20:
  v18 = a1[5];
  if ( (_QWORD *)a1[3] == v7 && v3 == v6 )
    goto LABEL_10;
  if ( v7 == v6 )
    return 0;
  do
  {
    v15 = v7;
    v7 = (_QWORD *)sub_220EF30(v7);
    v16 = sub_220F330(v15, v3);
    j_j___libc_free_0(v16, 40);
    v17 = a1[5] - 1LL;
    a1[5] = v17;
  }
  while ( v7 != v6 );
  v18 -= v17;
  return v18;
}
