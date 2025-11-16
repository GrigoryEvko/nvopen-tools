// Function: sub_1CCA5A0
// Address: 0x1cca5a0
//
_QWORD *__fastcall sub_1CCA5A0(_QWORD *a1, _QWORD *a2, __int64 a3)
{
  unsigned __int64 v6; // r14
  unsigned __int64 v7; // r8
  __int64 v8; // rdx
  _QWORD *v9; // r11
  __int64 v10; // r13
  _QWORD *v11; // rsi
  _QWORD *v12; // rdi
  _QWORD *result; // rax
  _QWORD *v14; // rax
  _QWORD *v15; // r12
  __int64 v16; // rdx
  __int64 v17; // rsi
  char v18; // al
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // r8
  _QWORD *v21; // r9
  _QWORD *v22; // rax
  _QWORD *v23; // rdx
  size_t v24; // r13
  void *v25; // rax
  _QWORD *v26; // rax
  _QWORD *v27; // r11
  _QWORD *v28; // rsi
  unsigned __int64 v29; // rdi
  _QWORD *v30; // rcx
  unsigned __int64 v31; // rdx
  _QWORD **v32; // rax
  __int64 v33; // rdx
  __int64 v34; // [rsp+0h] [rbp-40h]
  unsigned __int64 v35; // [rsp+0h] [rbp-40h]
  _QWORD *v36; // [rsp+8h] [rbp-38h]
  unsigned __int64 v37; // [rsp+8h] [rbp-38h]
  _QWORD *v38; // [rsp+8h] [rbp-38h]

  v6 = *a2;
  v7 = a1[1];
  v8 = *a2 % v7;
  v9 = *(_QWORD **)(*a1 + 8 * v8);
  v10 = v8;
  if ( v9 )
  {
    v11 = (_QWORD *)*v9;
    if ( v6 == *(_QWORD *)(*v9 + 8LL) )
    {
LABEL_6:
      result = (_QWORD *)*v9;
      if ( *v9 )
        return result;
    }
    else
    {
      while ( 1 )
      {
        v12 = (_QWORD *)*v11;
        if ( !*v11 )
          break;
        v9 = v11;
        if ( v8 != v12[1] % v7 )
          break;
        v11 = (_QWORD *)*v11;
        if ( v6 == v12[1] )
          goto LABEL_6;
      }
    }
  }
  v34 = a3;
  v36 = a2;
  v14 = (_QWORD *)sub_22077B0(16);
  v15 = v14;
  if ( v14 )
    *v14 = 0;
  v16 = a1[3];
  v17 = a1[1];
  v14[1] = *v36;
  v18 = sub_222DA10(a1 + 4, v17, v16, v34);
  v20 = v19;
  if ( v18 )
  {
    if ( v19 == 1 )
    {
      v21 = a1 + 6;
      a1[6] = 0;
      v27 = a1 + 6;
    }
    else
    {
      if ( v19 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(a1 + 4, v17, v19);
      v24 = 8 * v19;
      v37 = v19;
      v25 = (void *)sub_22077B0(8 * v19);
      v26 = memset(v25, 0, v24);
      v20 = v37;
      v27 = a1 + 6;
      v21 = v26;
    }
    v28 = (_QWORD *)a1[2];
    a1[2] = 0;
    if ( !v28 )
    {
LABEL_23:
      if ( (_QWORD *)*a1 != v27 )
      {
        v35 = v20;
        v38 = v21;
        j_j___libc_free_0(*a1, 8LL * a1[1]);
        v20 = v35;
        v21 = v38;
      }
      a1[1] = v20;
      *a1 = v21;
      v10 = v6 % v20;
      v22 = &v21[v10];
      v23 = (_QWORD *)v21[v10];
      if ( v23 )
        goto LABEL_11;
LABEL_26:
      v33 = a1[2];
      a1[2] = v15;
      *v15 = v33;
      if ( v33 )
      {
        v21[*(_QWORD *)(v33 + 8) % a1[1]] = v15;
        v22 = (_QWORD *)(v10 * 8 + *a1);
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
  v22 = (_QWORD *)(*a1 + v10 * 8);
  v23 = (_QWORD *)*v22;
  if ( !*v22 )
    goto LABEL_26;
LABEL_11:
  *v15 = *v23;
  *(_QWORD *)*v22 = v15;
LABEL_12:
  ++a1[3];
  return v15;
}
