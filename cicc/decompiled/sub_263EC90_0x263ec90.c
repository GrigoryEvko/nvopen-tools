// Function: sub_263EC90
// Address: 0x263ec90
//
unsigned __int64 __fastcall sub_263EC90(_QWORD *a1, __int64 a2, unsigned __int64 **a3)
{
  __int64 v6; // rax
  _QWORD *v7; // rcx
  unsigned __int64 v8; // r12
  unsigned __int64 *v9; // rax
  unsigned __int64 *v10; // r8
  unsigned __int64 v11; // r15
  unsigned __int64 v12; // rsi
  __int64 v13; // rax
  _QWORD *v14; // rdx
  _QWORD *v15; // rax
  _QWORD *v16; // r14
  bool v17; // al
  char v18; // di
  __int64 v20; // rax
  unsigned __int64 v21; // rsi
  __int64 v22; // rax
  unsigned __int64 v23; // [rsp+10h] [rbp-40h]
  _QWORD *v24; // [rsp+18h] [rbp-38h]

  v6 = sub_22077B0(0x30u);
  v7 = a1 + 1;
  v8 = v6;
  v9 = *a3;
  *(_QWORD *)(v8 + 40) = 0;
  v10 = (unsigned __int64 *)(v8 + 32);
  v11 = *v9;
  *(_QWORD *)(v8 + 32) = *v9;
  if ( a1 + 1 == (_QWORD *)a2 )
  {
    if ( a1[5] )
    {
      v14 = (_QWORD *)a1[4];
      if ( v14[4] < v11 )
        goto LABEL_19;
    }
  }
  else
  {
    v12 = *(_QWORD *)(a2 + 32);
    if ( v11 < v12 )
    {
      if ( a1[3] == a2 )
        goto LABEL_21;
      v13 = sub_220EF80(a2);
      v7 = a1 + 1;
      v10 = (unsigned __int64 *)(v8 + 32);
      v14 = (_QWORD *)v13;
      if ( *(_QWORD *)(v13 + 32) >= v11 )
        goto LABEL_5;
      if ( *(_QWORD *)(v13 + 24) )
      {
LABEL_21:
        v22 = a2;
LABEL_22:
        v14 = (_QWORD *)a2;
        v16 = (_QWORD *)v22;
        goto LABEL_6;
      }
LABEL_19:
      v17 = 0;
LABEL_7:
      if ( v7 == v14 || v17 )
        goto LABEL_9;
      v21 = v14[4];
      goto LABEL_24;
    }
    v23 = *(_QWORD *)(a2 + 32);
    v16 = (_QWORD *)a2;
    if ( v11 <= v12 )
      goto LABEL_16;
    if ( a1[4] == a2 )
    {
      v22 = 0;
      goto LABEL_22;
    }
    v20 = sub_220EEE0(a2);
    v7 = a1 + 1;
    v10 = (unsigned __int64 *)(v8 + 32);
    v14 = (_QWORD *)v20;
    if ( *(_QWORD *)(v20 + 32) > v11 )
    {
      v21 = v23;
      if ( *(_QWORD *)(a2 + 24) )
      {
LABEL_9:
        v18 = 1;
LABEL_10:
        sub_220F040(v18, v8, v14, v7);
        ++a1[5];
        return v8;
      }
      v14 = (_QWORD *)a2;
LABEL_24:
      v18 = v21 > v11;
      goto LABEL_10;
    }
  }
LABEL_5:
  v24 = v7;
  v15 = sub_263E1A0((__int64)a1, v10);
  v7 = v24;
  v16 = v15;
  if ( v14 )
  {
LABEL_6:
    v17 = v16 != 0;
    goto LABEL_7;
  }
LABEL_16:
  j_j___libc_free_0(v8);
  return (unsigned __int64)v16;
}
