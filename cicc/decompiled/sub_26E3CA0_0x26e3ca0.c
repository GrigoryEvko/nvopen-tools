// Function: sub_26E3CA0
// Address: 0x26e3ca0
//
_QWORD *__fastcall sub_26E3CA0(unsigned __int64 *a1, unsigned __int64 *a2, _DWORD *a3)
{
  __int64 v5; // rax
  _QWORD *v6; // r12
  unsigned __int64 v7; // r13
  unsigned __int64 v8; // r15
  unsigned __int64 v9; // r14
  _DWORD *v10; // rax
  __int64 v11; // rax
  __int64 v12; // r13
  unsigned __int64 v14; // rsi
  _QWORD *v15; // rdi
  char v16; // al
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // r15
  _QWORD *v19; // r8
  __int64 v20; // r14
  _QWORD **v21; // rax
  _QWORD *v22; // rdx
  size_t v23; // r14
  void *v24; // rax
  _QWORD *v25; // rax
  _QWORD *v26; // r10
  _QWORD *v27; // rsi
  unsigned __int64 v28; // rdi
  _QWORD *v29; // rcx
  unsigned __int64 v30; // rdx
  _QWORD **v31; // rax
  unsigned __int64 v32; // rdx
  _QWORD *v33; // [rsp+8h] [rbp-38h]

  v5 = sub_22077B0(0x20u);
  v6 = (_QWORD *)v5;
  if ( v5 )
    *(_QWORD *)v5 = 0;
  v7 = *a2;
  v8 = a1[1];
  *(_DWORD *)(v5 + 16) = *a3;
  *(_QWORD *)(v5 + 8) = v7;
  v9 = v7 % v8;
  v10 = sub_26E3C30(a1, v7 % v8, (_DWORD *)(v5 + 8), v7);
  if ( v10 )
  {
    v11 = *(_QWORD *)v10;
    if ( v11 )
    {
      v12 = v11;
      j_j___libc_free_0((unsigned __int64)v6);
      return (_QWORD *)v12;
    }
  }
  v14 = v8;
  v15 = a1 + 4;
  v16 = sub_222DA10((__int64)(a1 + 4), v8, a1[3], 1);
  v18 = v17;
  if ( v16 )
  {
    if ( v17 == 1 )
    {
      v19 = a1 + 6;
      a1[6] = 0;
      v26 = a1 + 6;
    }
    else
    {
      if ( v17 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(v15, v14, v17);
      v23 = 8 * v17;
      v24 = (void *)sub_22077B0(8 * v17);
      v25 = memset(v24, 0, v23);
      v26 = a1 + 6;
      v19 = v25;
    }
    v27 = (_QWORD *)a1[2];
    a1[2] = 0;
    if ( !v27 )
    {
LABEL_22:
      if ( (_QWORD *)*a1 != v26 )
      {
        v33 = v19;
        j_j___libc_free_0(*a1);
        v19 = v33;
      }
      a1[1] = v18;
      *a1 = (unsigned __int64)v19;
      v9 = v7 % v18;
      goto LABEL_9;
    }
    v28 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v29 = v27;
        v27 = (_QWORD *)*v27;
        v30 = v29[3] % v18;
        v31 = (_QWORD **)&v19[v30];
        if ( !*v31 )
          break;
        *v29 = **v31;
        **v31 = v29;
LABEL_18:
        if ( !v27 )
          goto LABEL_22;
      }
      *v29 = a1[2];
      a1[2] = (unsigned __int64)v29;
      *v31 = a1 + 2;
      if ( !*v29 )
      {
        v28 = v30;
        goto LABEL_18;
      }
      v19[v28] = v29;
      v28 = v30;
      if ( !v27 )
        goto LABEL_22;
    }
  }
  v19 = (_QWORD *)*a1;
LABEL_9:
  v20 = v9;
  v6[3] = v7;
  v21 = (_QWORD **)&v19[v20];
  v22 = (_QWORD *)v19[v20];
  if ( v22 )
  {
    *v6 = *v22;
    **v21 = v6;
  }
  else
  {
    v32 = a1[2];
    a1[2] = (unsigned __int64)v6;
    *v6 = v32;
    if ( v32 )
    {
      v19[*(_QWORD *)(v32 + 24) % a1[1]] = v6;
      v21 = (_QWORD **)(v20 * 8 + *a1);
    }
    *v21 = a1 + 2;
  }
  ++a1[3];
  return v6;
}
