// Function: sub_E9AD20
// Address: 0xe9ad20
//
__int64 __fastcall sub_E9AD20(_QWORD *a1, unsigned __int64 *a2)
{
  unsigned __int64 v4; // r14
  unsigned __int64 v5; // rdi
  __int64 v6; // rdx
  _QWORD *v7; // r8
  __int64 v8; // r13
  unsigned __int64 v9; // r9
  _QWORD *v10; // rdx
  _QWORD *v11; // rsi
  __int64 result; // rax
  _QWORD *v13; // rax
  _QWORD *v14; // rbx
  unsigned __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rsi
  char v18; // al
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // r15
  _QWORD *v21; // r8
  _QWORD *v22; // rax
  _QWORD *v23; // rdx
  size_t v24; // r13
  void *v25; // rax
  _QWORD *v26; // rax
  _QWORD *v27; // r10
  _QWORD *v28; // rsi
  unsigned __int64 v29; // rdi
  _QWORD *v30; // rcx
  unsigned __int64 v31; // rdx
  _QWORD **v32; // rax
  __int64 v33; // rdx
  _QWORD *v34; // [rsp+8h] [rbp-38h]

  v4 = *a2;
  v5 = a1[1];
  v6 = *a2 % v5;
  v7 = *(_QWORD **)(*a1 + 8 * v6);
  v8 = v6;
  if ( v7 )
  {
    v9 = *a2 % v5;
    v10 = (_QWORD *)*v7;
    if ( *(_QWORD *)(*v7 + 8LL) == v4 )
    {
LABEL_6:
      result = *v7 + 16LL;
      if ( *v7 )
        return result;
    }
    else
    {
      while ( 1 )
      {
        v11 = (_QWORD *)*v10;
        if ( !*v10 )
          break;
        v7 = v10;
        if ( v9 != v11[1] % v5 )
          break;
        v10 = (_QWORD *)*v10;
        if ( v11[1] == v4 )
          goto LABEL_6;
      }
    }
  }
  v13 = (_QWORD *)sub_22077B0(112);
  v14 = v13;
  if ( v13 )
    *v13 = 0;
  v15 = *a2;
  v16 = a1[3];
  v17 = a1[1];
  v14[3] = 1;
  v14[1] = v15;
  v14[2] = v14 + 8;
  v14[4] = 0;
  v14[5] = 0;
  *((_DWORD *)v14 + 12) = 1065353216;
  v14[7] = 0;
  v14[8] = 0;
  v14[9] = 0;
  v14[10] = 0;
  v14[11] = 0;
  v14[12] = 0;
  v14[13] = 0;
  v18 = sub_222DA10(a1 + 4, v17, v16, 1);
  v20 = v19;
  if ( v18 )
  {
    if ( v19 == 1 )
    {
      a1[6] = 0;
      v21 = a1 + 6;
      v27 = a1 + 6;
    }
    else
    {
      if ( v19 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(a1 + 4, v17, v19);
      v24 = 8 * v19;
      v25 = (void *)sub_22077B0(8 * v19);
      v26 = memset(v25, 0, v24);
      v27 = a1 + 6;
      v21 = v26;
    }
    v28 = (_QWORD *)a1[2];
    a1[2] = 0;
    if ( !v28 )
    {
LABEL_23:
      if ( v27 != (_QWORD *)*a1 )
      {
        v34 = v21;
        j_j___libc_free_0(*a1, 8LL * a1[1]);
        v21 = v34;
      }
      a1[1] = v20;
      *a1 = v21;
      v8 = v4 % v20;
      v22 = &v21[v8];
      v23 = (_QWORD *)v21[v8];
      if ( v23 )
        goto LABEL_11;
LABEL_26:
      v33 = a1[2];
      a1[2] = v14;
      *v14 = v33;
      if ( v33 )
      {
        v21[*(_QWORD *)(v33 + 8) % a1[1]] = v14;
        v22 = (_QWORD *)(v8 * 8 + *a1);
      }
      *v22 = a1 + 2;
      goto LABEL_12;
    }
    v29 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v30 = v28;
        v28 = (_QWORD *)*v28;
        v31 = v30[1] % v20;
        v32 = (_QWORD **)&v21[v31];
        if ( !*v32 )
          break;
        *v30 = **v32;
        **v32 = v30;
LABEL_19:
        if ( !v28 )
          goto LABEL_23;
      }
      *v30 = a1[2];
      a1[2] = v30;
      *v32 = a1 + 2;
      if ( !*v30 )
      {
        v29 = v31;
        goto LABEL_19;
      }
      v21[v29] = v30;
      v29 = v31;
      if ( !v28 )
        goto LABEL_23;
    }
  }
  v21 = (_QWORD *)*a1;
  v22 = (_QWORD *)(*a1 + v8 * 8);
  v23 = (_QWORD *)*v22;
  if ( !*v22 )
    goto LABEL_26;
LABEL_11:
  *v14 = *v23;
  *(_QWORD *)*v22 = v14;
LABEL_12:
  ++a1[3];
  return (__int64)(v14 + 2);
}
