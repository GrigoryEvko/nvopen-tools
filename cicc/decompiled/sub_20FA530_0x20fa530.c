// Function: sub_20FA530
// Address: 0x20fa530
//
_QWORD *__fastcall sub_20FA530(_QWORD *a1, unsigned __int64 **a2, __int64 a3)
{
  _QWORD *v5; // rax
  int v6; // r8d
  int v7; // r9d
  _QWORD *v8; // r12
  unsigned __int64 *v9; // rsi
  _BYTE *v10; // rdx
  _QWORD *v11; // r14
  _QWORD *v12; // r15
  __int64 *v13; // rcx
  size_t *v14; // rax
  unsigned __int64 v15; // r13
  __int64 v16; // rcx
  size_t v17; // rax
  __int64 v18; // rdx
  unsigned __int64 v19; // r9
  _QWORD *v20; // rdi
  _QWORD *v21; // rax
  _QWORD *v22; // rsi
  __int64 v23; // rax
  unsigned __int64 v24; // rdi
  __int64 v25; // r13
  unsigned __int64 v26; // rdi
  __int64 v28; // rsi
  char v29; // al
  unsigned __int64 v30; // rdx
  unsigned __int64 v31; // r14
  _QWORD *v32; // r15
  __int64 v33; // r8
  _QWORD **v34; // rax
  _QWORD *v35; // rdx
  _QWORD *v36; // r9
  _QWORD *v37; // rsi
  unsigned __int64 v38; // rdi
  _QWORD *v39; // rcx
  unsigned __int64 v40; // rdx
  _QWORD **v41; // rax
  __int64 v42; // rdx
  size_t n; // [rsp+8h] [rbp-38h]
  __int64 nb; // [rsp+8h] [rbp-38h]
  size_t na; // [rsp+8h] [rbp-38h]

  v5 = (_QWORD *)sub_22077B0(200);
  v8 = v5;
  if ( v5 )
    *v5 = 0;
  v9 = *a2;
  v10 = *(_BYTE **)a3;
  v11 = v5 + 8;
  v12 = v5 + 14;
  v13 = *(__int64 **)(a3 + 16);
  v14 = *(size_t **)(a3 + 24);
  v15 = *v9;
  v8[1] = *v9;
  LOBYTE(v10) = *v10;
  v16 = *v13;
  v17 = *v14;
  v8[6] = v8 + 8;
  *((_BYTE *)v8 + 40) = (_BYTE)v10;
  v8[2] = v17;
  v8[3] = v16;
  v8[4] = 0;
  v8[7] = 0x400000000LL;
  v8[12] = v8 + 14;
  v8[13] = 0x400000000LL;
  v8[22] = 0;
  v8[23] = 0;
  v8[24] = 0;
  if ( v17 )
  {
    v18 = *(unsigned int *)(v17 + 40);
    if ( (unsigned int)v18 >= *(_DWORD *)(v17 + 44) )
    {
      na = v17;
      sub_16CD150(v17 + 32, (const void *)(v17 + 48), 0, 8, v6, v7);
      v17 = na;
      v18 = *(unsigned int *)(na + 40);
    }
    *(_QWORD *)(*(_QWORD *)(v17 + 32) + 8 * v18) = v8 + 2;
    ++*(_DWORD *)(v17 + 40);
    v15 = v8[1];
  }
  v19 = a1[1];
  v20 = *(_QWORD **)(*a1 + 8 * (v15 % v19));
  if ( !v20 )
    goto LABEL_19;
  v21 = (_QWORD *)*v20;
  if ( v15 != *(_QWORD *)(*v20 + 8LL) )
  {
    do
    {
      v22 = (_QWORD *)*v21;
      if ( !*v21 )
        goto LABEL_19;
      v20 = v21;
      if ( v15 % v19 != v22[1] % v19 )
        goto LABEL_19;
      v21 = (_QWORD *)*v21;
    }
    while ( v15 != v22[1] );
  }
  v23 = *v20;
  if ( !*v20 )
  {
LABEL_19:
    v28 = a1[1];
    n = 8 * (v15 % v19);
    v29 = sub_222DA10(a1 + 4, v28, a1[3], 1);
    v31 = v30;
    if ( !v29 )
    {
      v32 = (_QWORD *)*a1;
      v33 = n;
      v34 = (_QWORD **)(*a1 + n);
      v35 = *v34;
      if ( *v34 )
      {
LABEL_21:
        *v8 = *v35;
        **v34 = v8;
LABEL_22:
        ++a1[3];
        return v8;
      }
LABEL_36:
      v42 = a1[2];
      a1[2] = v8;
      *v8 = v42;
      if ( v42 )
      {
        v32[*(_QWORD *)(v42 + 8) % a1[1]] = v8;
        v34 = (_QWORD **)(v33 + *a1);
      }
      *v34 = a1 + 2;
      goto LABEL_22;
    }
    if ( v30 == 1 )
    {
      v32 = a1 + 6;
      a1[6] = 0;
      v36 = a1 + 6;
    }
    else
    {
      if ( v30 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(a1 + 4, v28, v30);
      nb = 8 * v30;
      v32 = (_QWORD *)sub_22077B0(8 * v30);
      memset(v32, 0, nb);
      v36 = a1 + 6;
    }
    v37 = (_QWORD *)a1[2];
    a1[2] = 0;
    if ( !v37 )
    {
LABEL_33:
      if ( (_QWORD *)*a1 != v36 )
        j_j___libc_free_0(*a1, 8LL * a1[1]);
      a1[1] = v31;
      *a1 = v32;
      v33 = 8 * (v15 % v31);
      v34 = (_QWORD **)((char *)v32 + v33);
      v35 = *(_QWORD **)((char *)v32 + v33);
      if ( v35 )
        goto LABEL_21;
      goto LABEL_36;
    }
    v38 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v39 = v37;
        v37 = (_QWORD *)*v37;
        v40 = v39[1] % v31;
        v41 = (_QWORD **)&v32[v40];
        if ( !*v41 )
          break;
        *v39 = **v41;
        **v41 = v39;
LABEL_29:
        if ( !v37 )
          goto LABEL_33;
      }
      *v39 = a1[2];
      a1[2] = v39;
      *v41 = a1 + 2;
      if ( !*v39 )
      {
        v38 = v40;
        goto LABEL_29;
      }
      v32[v38] = v39;
      v38 = v40;
      if ( !v37 )
        goto LABEL_33;
    }
  }
  v24 = v8[12];
  v25 = v23;
  if ( v12 != (_QWORD *)v24 )
    _libc_free(v24);
  v26 = v8[6];
  if ( v11 != (_QWORD *)v26 )
    _libc_free(v26);
  j_j___libc_free_0(v8, 200);
  return (_QWORD *)v25;
}
