// Function: sub_B7FEB0
// Address: 0xb7feb0
//
_QWORD *__fastcall sub_B7FEB0(__int64 a1, unsigned __int64 a2)
{
  _QWORD *result; // rax
  _QWORD *v4; // r13
  _QWORD *v5; // rbx
  unsigned __int64 v6; // rdi
  _QWORD *v7; // r9
  _QWORD *v8; // rsi
  _QWORD *v9; // rax
  _QWORD *v10; // r14
  __int64 v11; // rsi
  char v12; // al
  unsigned __int64 v13; // rdx
  __int64 v14; // rcx
  unsigned __int64 v15; // r15
  __int64 v16; // r10
  _QWORD *v17; // rax
  void *v18; // rax
  _QWORD *v19; // rax
  _QWORD *v20; // r8
  _QWORD *v21; // r9
  _QWORD *v22; // rsi
  unsigned __int64 v23; // rdi
  _QWORD *v24; // rcx
  unsigned __int64 v25; // rdx
  _QWORD **v26; // rax
  _QWORD *v27; // rdi
  __int64 n; // [rsp+0h] [rbp-40h]
  size_t na; // [rsp+0h] [rbp-40h]
  _QWORD *v30; // [rsp+8h] [rbp-38h]

  result = *(_QWORD **)(a1 + 40);
  v30 = result;
  if ( *(_QWORD **)(a1 + 32) != result )
  {
    v4 = *(_QWORD **)(a1 + 32);
    while ( 1 )
    {
      v5 = (_QWORD *)*v4;
      v6 = *(_QWORD *)(*v4 + 64LL);
      v7 = *(_QWORD **)(*(_QWORD *)(*v4 + 56LL) + 8 * (a2 % v6));
      if ( !v7 )
        goto LABEL_11;
      result = (_QWORD *)*v7;
      if ( a2 != *(_QWORD *)(*v7 + 8LL) )
        break;
LABEL_8:
      if ( !*v7 )
        goto LABEL_11;
LABEL_9:
      if ( v30 == ++v4 )
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
    v9 = (_QWORD *)sub_22077B0(16);
    v10 = v9;
    if ( v9 )
      *v9 = 0;
    v9[1] = a2;
    v11 = v5[8];
    v12 = sub_222DA10(v5 + 11, v11, v5[10], 1);
    v15 = v13;
    if ( !v12 )
    {
      v16 = a2 % v6;
      v17 = *(_QWORD **)(v5[7] + v16 * 8);
      if ( v17 )
      {
LABEL_15:
        *v10 = *v17;
        result = *(_QWORD **)(v5[7] + v16 * 8);
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
      result[v16] = v5 + 9;
      goto LABEL_16;
    }
    if ( v13 == 1 )
    {
      v21 = v5 + 13;
      v5[13] = 0;
      v20 = v5 + 13;
    }
    else
    {
      if ( v13 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(v5 + 11, v11, v13, v14);
      n = 8 * v13;
      v18 = (void *)sub_22077B0(8 * v13);
      v19 = memset(v18, 0, n);
      v20 = v5 + 13;
      v21 = v19;
    }
    v22 = (_QWORD *)v5[9];
    v5[9] = 0;
    if ( !v22 )
    {
LABEL_27:
      v27 = (_QWORD *)v5[7];
      if ( v27 != v20 )
      {
        na = (size_t)v21;
        j_j___libc_free_0(v27, 8LL * v5[8]);
        v21 = (_QWORD *)na;
      }
      v5[8] = v15;
      v5[7] = v21;
      v16 = a2 % v15;
      v17 = (_QWORD *)v21[v16];
      if ( v17 )
        goto LABEL_15;
      goto LABEL_30;
    }
    v23 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v24 = v22;
        v22 = (_QWORD *)*v22;
        v25 = v24[1] % v15;
        v26 = (_QWORD **)&v21[v25];
        if ( !*v26 )
          break;
        *v24 = **v26;
        **v26 = v24;
LABEL_23:
        if ( !v22 )
          goto LABEL_27;
      }
      *v24 = v5[9];
      v5[9] = v24;
      *v26 = v5 + 9;
      if ( !*v24 )
      {
        v23 = v25;
        goto LABEL_23;
      }
      v21[v23] = v24;
      v23 = v25;
      if ( !v22 )
        goto LABEL_27;
    }
  }
  return result;
}
