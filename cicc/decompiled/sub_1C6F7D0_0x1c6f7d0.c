// Function: sub_1C6F7D0
// Address: 0x1c6f7d0
//
_QWORD *__fastcall sub_1C6F7D0(_QWORD *a1, _QWORD *a2, unsigned __int64 **a3)
{
  __int64 v6; // rax
  _QWORD *v7; // rcx
  __int64 v8; // r12
  unsigned __int64 *v9; // r8
  unsigned __int64 v10; // r15
  unsigned __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rdx
  _QWORD *v14; // rax
  _QWORD *v15; // r14
  bool v16; // al
  _BOOL8 v17; // rdi
  __int64 v19; // rax
  unsigned __int64 v20; // rsi
  _QWORD *v21; // rax
  unsigned __int64 v22; // [rsp+10h] [rbp-40h]
  unsigned __int64 *v23; // [rsp+18h] [rbp-38h]
  _QWORD *v24; // [rsp+18h] [rbp-38h]
  unsigned __int64 *v25; // [rsp+18h] [rbp-38h]

  v6 = sub_22077B0(88);
  v7 = a1 + 1;
  v8 = v6;
  v9 = (unsigned __int64 *)(v6 + 32);
  v10 = **a3;
  v6 += 48;
  *(_DWORD *)(v8 + 48) = 0;
  *(_QWORD *)(v8 + 56) = 0;
  *(_QWORD *)(v8 + 32) = v10;
  *(_QWORD *)(v8 + 64) = v6;
  *(_QWORD *)(v8 + 72) = v6;
  *(_QWORD *)(v8 + 80) = 0;
  if ( a1 + 1 == a2 )
  {
    if ( a1[5] )
    {
      v13 = a1[4];
      if ( *(_QWORD *)(v13 + 32) < v10 )
        goto LABEL_19;
    }
  }
  else
  {
    v11 = a2[4];
    if ( v10 < v11 )
    {
      v23 = v9;
      if ( (_QWORD *)a1[3] == a2 )
        goto LABEL_21;
      v12 = sub_220EF80(a2);
      v7 = a1 + 1;
      v9 = v23;
      v13 = v12;
      if ( *(_QWORD *)(v12 + 32) >= v10 )
        goto LABEL_5;
      if ( *(_QWORD *)(v12 + 24) )
      {
LABEL_21:
        v21 = a2;
LABEL_22:
        v13 = (__int64)a2;
        v15 = v21;
        goto LABEL_6;
      }
LABEL_19:
      v16 = 0;
LABEL_7:
      if ( v7 == (_QWORD *)v13 || v16 )
        goto LABEL_9;
      v20 = *(_QWORD *)(v13 + 32);
      goto LABEL_24;
    }
    v22 = a2[4];
    v15 = a2;
    v25 = v9;
    if ( v10 <= v11 )
      goto LABEL_16;
    if ( (_QWORD *)a1[4] == a2 )
    {
      v21 = 0;
      goto LABEL_22;
    }
    v19 = sub_220EEE0(a2);
    v7 = a1 + 1;
    v9 = v25;
    v13 = v19;
    if ( *(_QWORD *)(v19 + 32) > v10 )
    {
      v20 = v22;
      if ( a2[3] )
      {
LABEL_9:
        v17 = 1;
LABEL_10:
        sub_220F040(v17, v8, v13, v7);
        ++a1[5];
        return (_QWORD *)v8;
      }
      v13 = (__int64)a2;
LABEL_24:
      v17 = v20 > v10;
      goto LABEL_10;
    }
  }
LABEL_5:
  v24 = v7;
  v14 = sub_1C6E8E0((__int64)a1, v9);
  v7 = v24;
  v15 = v14;
  if ( v13 )
  {
LABEL_6:
    v16 = v15 != 0;
    goto LABEL_7;
  }
LABEL_16:
  sub_1C6F600(0);
  j_j___libc_free_0(v8, 88);
  return v15;
}
