// Function: sub_21F6AD0
// Address: 0x21f6ad0
//
_QWORD *__fastcall sub_21F6AD0(_QWORD *a1, unsigned __int64 a2)
{
  _QWORD *v3; // rbx
  _QWORD *v4; // r13
  _QWORD *v5; // rax
  _QWORD *v6; // rdx
  __int64 v7; // rdi
  __int64 v8; // rcx
  _QWORD *v9; // r8
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v13; // rcx
  __int64 v14; // rdx
  _QWORD *v15; // rax
  unsigned __int64 v16; // r8
  _QWORD *v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rax
  _QWORD v21[2]; // [rsp+8h] [rbp-38h] BYREF
  unsigned __int64 *v22[5]; // [rsp+18h] [rbp-28h] BYREF

  v3 = a1 + 1;
  v4 = a1 + 1;
  v5 = (_QWORD *)a1[2];
  v21[0] = a2;
  if ( !v5 )
    goto LABEL_21;
  v6 = v5;
  do
  {
    while ( 1 )
    {
      v7 = v6[2];
      v8 = v6[3];
      if ( v6[4] >= a2 )
        break;
      v6 = (_QWORD *)v6[3];
      if ( !v8 )
        goto LABEL_6;
    }
    v4 = v6;
    v6 = (_QWORD *)v6[2];
  }
  while ( v7 );
LABEL_6:
  if ( v3 == v4 )
    goto LABEL_17;
  if ( v4[4] <= a2 )
    goto LABEL_8;
  v4 = v3;
  do
  {
LABEL_17:
    while ( 1 )
    {
      v13 = v5[2];
      v14 = v5[3];
      if ( v5[4] >= a2 )
        break;
      v5 = (_QWORD *)v5[3];
      if ( !v14 )
        goto LABEL_19;
    }
    v4 = v5;
    v5 = (_QWORD *)v5[2];
  }
  while ( v13 );
LABEL_19:
  if ( v3 == v4 || v4[4] > a2 )
  {
LABEL_21:
    v22[0] = v21;
    v4 = (_QWORD *)sub_21F6A00(a1, v4, v22);
  }
  sub_21F22C0(v4[7]);
  v4[7] = 0;
  v4[8] = v4 + 6;
  v4[9] = v4 + 6;
  v4[10] = 0;
  v15 = (_QWORD *)a1[2];
  if ( v15 )
  {
    v16 = v21[0];
    v17 = v3;
    do
    {
      while ( 1 )
      {
        v18 = v15[2];
        v19 = v15[3];
        if ( v15[4] >= v21[0] )
          break;
        v15 = (_QWORD *)v15[3];
        if ( !v19 )
          goto LABEL_27;
      }
      v17 = v15;
      v15 = (_QWORD *)v15[2];
    }
    while ( v18 );
LABEL_27:
    if ( v3 != v17 && v17[4] <= v21[0] )
      goto LABEL_30;
  }
  else
  {
    v17 = v3;
  }
  v22[0] = v21;
  v20 = sub_21F6A00(a1, v17, v22);
  v16 = v21[0];
  v17 = (_QWORD *)v20;
LABEL_30:
  sub_21F6030((__int64)a1, v16, v17 + 5);
  v5 = (_QWORD *)a1[2];
  if ( !v5 )
  {
    v9 = v3;
LABEL_14:
    v22[0] = v21;
    v9 = (_QWORD *)sub_21F6A00(a1, v9, v22);
    return v9 + 5;
  }
  a2 = v21[0];
LABEL_8:
  v9 = v3;
  do
  {
    while ( 1 )
    {
      v10 = v5[2];
      v11 = v5[3];
      if ( v5[4] >= a2 )
        break;
      v5 = (_QWORD *)v5[3];
      if ( !v11 )
        goto LABEL_12;
    }
    v9 = v5;
    v5 = (_QWORD *)v5[2];
  }
  while ( v10 );
LABEL_12:
  if ( v3 == v9 || v9[4] > a2 )
    goto LABEL_14;
  return v9 + 5;
}
