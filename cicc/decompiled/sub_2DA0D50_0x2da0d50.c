// Function: sub_2DA0D50
// Address: 0x2da0d50
//
unsigned __int64 __fastcall sub_2DA0D50(_QWORD *a1, __int64 a2, unsigned __int64 **a3)
{
  _QWORD *v6; // rax
  _QWORD *v7; // rcx
  unsigned __int64 v8; // r12
  unsigned __int64 *v9; // r8
  unsigned __int64 v10; // r15
  unsigned __int64 v11; // rsi
  __int64 v12; // rax
  _QWORD *v13; // rdx
  _QWORD *v14; // rax
  _QWORD *v15; // r14
  bool v16; // al
  char v17; // di
  __int64 v19; // rax
  unsigned __int64 v20; // rsi
  __int64 v21; // rax
  unsigned __int64 v22; // [rsp+10h] [rbp-40h]
  unsigned __int64 *v23; // [rsp+18h] [rbp-38h]
  _QWORD *v24; // [rsp+18h] [rbp-38h]
  unsigned __int64 *v25; // [rsp+18h] [rbp-38h]

  v6 = (_QWORD *)sub_22077B0(0x38u);
  v7 = a1 + 1;
  v8 = (unsigned __int64)v6;
  v9 = v6 + 4;
  v10 = **a3;
  v6[5] = 0;
  v6[6] = 0;
  v6[4] = v10;
  if ( a1 + 1 == (_QWORD *)a2 )
  {
    if ( a1[5] )
    {
      v13 = (_QWORD *)a1[4];
      if ( v13[4] < v10 )
        goto LABEL_19;
    }
  }
  else
  {
    v11 = *(_QWORD *)(a2 + 32);
    if ( v10 < v11 )
    {
      v23 = v6 + 4;
      if ( a1[3] == a2 )
        goto LABEL_21;
      v12 = sub_220EF80(a2);
      v7 = a1 + 1;
      v9 = v23;
      v13 = (_QWORD *)v12;
      if ( *(_QWORD *)(v12 + 32) >= v10 )
        goto LABEL_5;
      if ( *(_QWORD *)(v12 + 24) )
      {
LABEL_21:
        v21 = a2;
LABEL_22:
        v13 = (_QWORD *)a2;
        v15 = (_QWORD *)v21;
        goto LABEL_6;
      }
LABEL_19:
      v16 = 0;
LABEL_7:
      if ( v7 == v13 || v16 )
        goto LABEL_9;
      v20 = v13[4];
      goto LABEL_24;
    }
    v22 = *(_QWORD *)(a2 + 32);
    v15 = (_QWORD *)a2;
    v25 = v6 + 4;
    if ( v10 <= v11 )
      goto LABEL_16;
    if ( a1[4] == a2 )
    {
      v21 = 0;
      goto LABEL_22;
    }
    v19 = sub_220EEE0(a2);
    v7 = a1 + 1;
    v9 = v25;
    v13 = (_QWORD *)v19;
    if ( *(_QWORD *)(v19 + 32) > v10 )
    {
      v20 = v22;
      if ( *(_QWORD *)(a2 + 24) )
      {
LABEL_9:
        v17 = 1;
LABEL_10:
        sub_220F040(v17, v8, v13, v7);
        ++a1[5];
        return v8;
      }
      v13 = (_QWORD *)a2;
LABEL_24:
      v17 = v20 > v10;
      goto LABEL_10;
    }
  }
LABEL_5:
  v24 = v7;
  v14 = sub_2D9F480((__int64)a1, v9);
  v7 = v24;
  v15 = v14;
  if ( v13 )
  {
LABEL_6:
    v16 = v15 != 0;
    goto LABEL_7;
  }
LABEL_16:
  j_j___libc_free_0(v8);
  return (unsigned __int64)v15;
}
