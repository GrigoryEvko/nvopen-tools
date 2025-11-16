// Function: sub_394A760
// Address: 0x394a760
//
__int64 __fastcall sub_394A760(unsigned __int64 *a1, _DWORD *a2)
{
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rdi
  _QWORD *v6; // r8
  __int64 v7; // r14
  _QWORD *v8; // rax
  unsigned __int64 v9; // r9
  _QWORD *v10; // rsi
  __int64 result; // rax
  __int64 v12; // rax
  _QWORD *v13; // r12
  __int64 v14; // rdx
  __int64 v15; // rsi
  char v16; // al
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // r15
  unsigned __int64 v19; // r8
  _QWORD *v20; // rax
  _QWORD *v21; // rdx
  size_t v22; // r14
  void *v23; // rax
  void *v24; // rax
  unsigned __int64 *v25; // r10
  _QWORD *v26; // rsi
  unsigned __int64 v27; // rdi
  _QWORD *v28; // rcx
  unsigned __int64 v29; // rdx
  _QWORD **v30; // rax
  unsigned __int64 v31; // rdx
  unsigned __int64 v32; // [rsp+8h] [rbp-38h]

  v4 = (unsigned int)*a2;
  v5 = a1[1];
  v6 = *(_QWORD **)(*a1 + 8 * (v4 % v5));
  v7 = 8 * (v4 % v5);
  if ( v6 )
  {
    v8 = (_QWORD *)*v6;
    v9 = (unsigned int)*a2 % v5;
    if ( *(_DWORD *)(*v6 + 8LL) == (_DWORD)v4 )
    {
LABEL_6:
      result = *v6 + 16LL;
      if ( *v6 )
        return result;
    }
    else
    {
      while ( 1 )
      {
        v10 = (_QWORD *)*v8;
        if ( !*v8 )
          break;
        v6 = v8;
        if ( v9 != *((unsigned int *)v10 + 2) % v5 )
          break;
        v8 = (_QWORD *)*v8;
        if ( *((_DWORD *)v10 + 2) == (_DWORD)v4 )
          goto LABEL_6;
      }
    }
  }
  v12 = sub_22077B0(0x40u);
  v13 = (_QWORD *)v12;
  if ( v12 )
    *(_QWORD *)v12 = 0;
  v14 = a1[3];
  v15 = a1[1];
  *(_DWORD *)(v12 + 8) = *a2;
  *(_QWORD *)(v12 + 16) = v12 + 32;
  *(_QWORD *)(v12 + 24) = 0x400000000LL;
  v16 = sub_222DA10((__int64)(a1 + 4), v15, v14, 1);
  v18 = v17;
  if ( v16 )
  {
    if ( v17 == 1 )
    {
      v19 = (unsigned __int64)(a1 + 6);
      a1[6] = 0;
      v25 = a1 + 6;
    }
    else
    {
      if ( v17 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(a1 + 4, v15, v17);
      v22 = 8 * v17;
      v23 = (void *)sub_22077B0(8 * v17);
      v24 = memset(v23, 0, v22);
      v25 = a1 + 6;
      v19 = (unsigned __int64)v24;
    }
    v26 = (_QWORD *)a1[2];
    a1[2] = 0;
    if ( !v26 )
    {
LABEL_23:
      if ( v25 != (unsigned __int64 *)*a1 )
      {
        v32 = v19;
        j_j___libc_free_0(*a1);
        v19 = v32;
      }
      a1[1] = v18;
      *a1 = v19;
      v7 = 8 * (v4 % v18);
      v20 = (_QWORD *)(v19 + v7);
      v21 = *(_QWORD **)(v19 + v7);
      if ( v21 )
        goto LABEL_11;
LABEL_26:
      v31 = a1[2];
      a1[2] = (unsigned __int64)v13;
      *v13 = v31;
      if ( v31 )
      {
        *(_QWORD *)(v19 + 8 * (*(unsigned int *)(v31 + 8) % a1[1])) = v13;
        v20 = (_QWORD *)(v7 + *a1);
      }
      *v20 = a1 + 2;
      goto LABEL_12;
    }
    v27 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v28 = v26;
        v26 = (_QWORD *)*v26;
        v29 = *((unsigned int *)v28 + 2) % v18;
        v30 = (_QWORD **)(v19 + 8 * v29);
        if ( !*v30 )
          break;
        *v28 = **v30;
        **v30 = v28;
LABEL_19:
        if ( !v26 )
          goto LABEL_23;
      }
      *v28 = a1[2];
      a1[2] = (unsigned __int64)v28;
      *v30 = a1 + 2;
      if ( !*v28 )
      {
        v27 = v29;
        goto LABEL_19;
      }
      *(_QWORD *)(v19 + 8 * v27) = v28;
      v27 = v29;
      if ( !v26 )
        goto LABEL_23;
    }
  }
  v19 = *a1;
  v20 = (_QWORD *)(*a1 + v7);
  v21 = (_QWORD *)*v20;
  if ( !*v20 )
    goto LABEL_26;
LABEL_11:
  *v13 = *v21;
  *(_QWORD *)*v20 = v13;
LABEL_12:
  ++a1[3];
  return (__int64)(v13 + 2);
}
