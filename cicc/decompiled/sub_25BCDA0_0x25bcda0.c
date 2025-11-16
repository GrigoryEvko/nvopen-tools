// Function: sub_25BCDA0
// Address: 0x25bcda0
//
unsigned __int64 __fastcall sub_25BCDA0(_QWORD *a1, __int64 a2, unsigned __int64 **a3)
{
  __int64 v6; // rax
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
  __int64 v20; // rax
  unsigned __int64 *v21; // [rsp+18h] [rbp-38h]
  _QWORD *v22; // [rsp+18h] [rbp-38h]
  unsigned __int64 *v23; // [rsp+18h] [rbp-38h]

  v6 = sub_22077B0(0x68u);
  v7 = a1 + 1;
  v8 = v6;
  v9 = (unsigned __int64 *)(v6 + 32);
  v10 = **a3;
  *(_OWORD *)(v6 + 40) = 0;
  *(_QWORD *)(v6 + 56) = v6 + 72;
  *(_QWORD *)(v6 + 32) = v10;
  *(_QWORD *)(v6 + 64) = 0x400000000LL;
  *(_OWORD *)(v6 + 72) = 0;
  *(_OWORD *)(v6 + 88) = 0;
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
      v21 = (unsigned __int64 *)(v6 + 32);
      if ( a1[3] == a2 )
        goto LABEL_21;
      v12 = sub_220EF80(a2);
      v7 = a1 + 1;
      v9 = v21;
      v13 = (_QWORD *)v12;
      if ( *(_QWORD *)(v12 + 32) >= v10 )
        goto LABEL_5;
      if ( *(_QWORD *)(v12 + 24) )
      {
LABEL_21:
        v20 = a2;
LABEL_22:
        v13 = (_QWORD *)a2;
        v15 = (_QWORD *)v20;
        goto LABEL_6;
      }
LABEL_19:
      v16 = 0;
LABEL_7:
      if ( v7 == v13 || v16 )
        goto LABEL_9;
      v11 = v13[4];
      goto LABEL_24;
    }
    v23 = (unsigned __int64 *)(v6 + 32);
    v15 = (_QWORD *)a2;
    if ( v10 <= v11 )
      goto LABEL_16;
    if ( a1[4] == a2 )
    {
      v20 = 0;
      goto LABEL_22;
    }
    v19 = sub_220EEE0(a2);
    v7 = a1 + 1;
    v9 = v23;
    v13 = (_QWORD *)v19;
    if ( *(_QWORD *)(v19 + 32) > v10 )
    {
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
      v17 = v11 > v10;
      goto LABEL_10;
    }
  }
LABEL_5:
  v22 = v7;
  v14 = sub_25BCAF0((__int64)a1, v9);
  v7 = v22;
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
