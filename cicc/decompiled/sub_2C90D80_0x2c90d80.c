// Function: sub_2C90D80
// Address: 0x2c90d80
//
_QWORD *__fastcall sub_2C90D80(__int64 a1, unsigned __int64 a2)
{
  _QWORD *result; // rax
  __int64 v4; // r13
  _QWORD *v5; // rbx
  unsigned __int64 v6; // rdi
  _QWORD *v7; // r9
  _QWORD *v8; // rsi
  _QWORD *v9; // rax
  _QWORD *v10; // r14
  __int64 v11; // rsi
  char v12; // al
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // r15
  __int64 v15; // r10
  _QWORD *v16; // rax
  void *v17; // rax
  _QWORD *v18; // rax
  _QWORD *v19; // r8
  _QWORD *v20; // r9
  _QWORD *v21; // rsi
  unsigned __int64 v22; // rdi
  _QWORD *v23; // rcx
  unsigned __int64 v24; // rdx
  _QWORD **v25; // rax
  unsigned __int64 v26; // rdi
  __int64 n; // [rsp+0h] [rbp-40h]
  size_t na; // [rsp+0h] [rbp-40h]
  __int64 v29; // [rsp+8h] [rbp-38h]

  result = *(_QWORD **)a1;
  v29 = *(_QWORD *)(*(_QWORD *)a1 + 40LL);
  if ( *(_QWORD *)(*(_QWORD *)a1 + 32LL) != v29 )
  {
    v4 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
    while ( 1 )
    {
      v5 = *(_QWORD **)v4;
      v6 = *(_QWORD *)(*(_QWORD *)v4 + 64LL);
      v7 = *(_QWORD **)(*(_QWORD *)(*(_QWORD *)v4 + 56LL) + 8 * (a2 % v6));
      if ( !v7 )
        goto LABEL_11;
      result = (_QWORD *)*v7;
      if ( a2 != *(_QWORD *)(*v7 + 8LL) )
        break;
LABEL_8:
      if ( !*v7 )
        goto LABEL_11;
LABEL_9:
      v4 += 8;
      if ( v29 == v4 )
        return result;
    }
    while ( 1 )
    {
      v8 = (_QWORD *)*result;
      if ( !*result )
        break;
      v7 = result;
      if ( a2 % v6 != v8[1] % v6 )
        break;
      result = (_QWORD *)*result;
      if ( a2 == v8[1] )
        goto LABEL_8;
    }
LABEL_11:
    v9 = (_QWORD *)sub_22077B0(0x10u);
    v10 = v9;
    if ( v9 )
      *v9 = 0;
    v9[1] = a2;
    v11 = v5[8];
    v12 = sub_222DA10((__int64)(v5 + 11), v11, v5[10], 1);
    v14 = v13;
    if ( !v12 )
    {
      v15 = a2 % v6;
      v16 = *(_QWORD **)(v5[7] + v15 * 8);
      if ( v16 )
      {
LABEL_15:
        *v10 = *v16;
        result = *(_QWORD **)(v5[7] + v15 * 8);
        *result = v10;
LABEL_16:
        ++v5[10];
        goto LABEL_9;
      }
LABEL_30:
      *v10 = v5[9];
      v5[9] = v10;
      if ( *v10 )
        *(_QWORD *)(v5[7] + 8LL * (*(_QWORD *)(*v10 + 8LL) % v5[8])) = v10;
      result = (_QWORD *)v5[7];
      result[v15] = v5 + 9;
      goto LABEL_16;
    }
    if ( v13 == 1 )
    {
      v20 = v5 + 13;
      v5[13] = 0;
      v19 = v5 + 13;
    }
    else
    {
      if ( v13 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(v5 + 11, v11, v13);
      n = 8 * v13;
      v17 = (void *)sub_22077B0(8 * v13);
      v18 = memset(v17, 0, n);
      v19 = v5 + 13;
      v20 = v18;
    }
    v21 = (_QWORD *)v5[9];
    v5[9] = 0;
    if ( !v21 )
    {
LABEL_27:
      v26 = v5[7];
      if ( v19 != (_QWORD *)v26 )
      {
        na = (size_t)v20;
        j_j___libc_free_0(v26);
        v20 = (_QWORD *)na;
      }
      v5[8] = v14;
      v5[7] = v20;
      v15 = a2 % v14;
      v16 = (_QWORD *)v20[v15];
      if ( v16 )
        goto LABEL_15;
      goto LABEL_30;
    }
    v22 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v23 = v21;
        v21 = (_QWORD *)*v21;
        v24 = v23[1] % v14;
        v25 = (_QWORD **)&v20[v24];
        if ( !*v25 )
          break;
        *v23 = **v25;
        **v25 = v23;
LABEL_23:
        if ( !v21 )
          goto LABEL_27;
      }
      *v23 = v5[9];
      v5[9] = v23;
      *v25 = v5 + 9;
      if ( !*v23 )
      {
        v22 = v24;
        goto LABEL_23;
      }
      v20[v22] = v23;
      v22 = v24;
      if ( !v21 )
        goto LABEL_27;
    }
  }
  return result;
}
