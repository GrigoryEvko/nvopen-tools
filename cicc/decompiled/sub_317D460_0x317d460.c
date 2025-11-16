// Function: sub_317D460
// Address: 0x317d460
//
unsigned __int64 *__fastcall sub_317D460(unsigned __int64 *a1, unsigned __int64 **a2, __int64 **a3)
{
  unsigned __int64 *v5; // rax
  unsigned __int64 *v6; // r12
  __int64 *v7; // rax
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // r13
  unsigned __int64 v12; // rsi
  _QWORD *v13; // r9
  __int64 v14; // r14
  _QWORD *v15; // rdi
  _QWORD *v16; // r8
  unsigned __int64 v17; // rdi
  __int64 v18; // r13
  char v20; // al
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // r15
  _QWORD *v23; // r8
  unsigned __int64 ***v24; // rax
  unsigned __int64 *v25; // rdx
  size_t v26; // r14
  void *v27; // rax
  _QWORD *v28; // rax
  _QWORD *v29; // r10
  _QWORD *v30; // rsi
  unsigned __int64 v31; // rdi
  _QWORD *v32; // rcx
  unsigned __int64 v33; // rdx
  _QWORD **v34; // rax
  unsigned __int64 v35; // rdx
  _QWORD *v36; // [rsp+8h] [rbp-38h]

  v5 = (unsigned __int64 *)sub_22077B0(0x28u);
  v6 = v5;
  if ( v5 )
    *v5 = 0;
  v7 = *a3;
  v6[1] = **a2;
  v8 = *v7;
  *v7 = 0;
  v6[2] = v8;
  v9 = v7[1];
  v7[1] = 0;
  v6[3] = v9;
  v10 = v7[2];
  v7[2] = 0;
  v11 = v6[1];
  v12 = a1[1];
  v6[4] = v10;
  v13 = *(_QWORD **)(*a1 + 8 * (v11 % v12));
  v14 = v11 % v12;
  if ( !v13 )
    goto LABEL_13;
  v15 = (_QWORD *)*v13;
  if ( v11 != *(_QWORD *)(*v13 + 8LL) )
  {
    do
    {
      v16 = (_QWORD *)*v15;
      if ( !*v15 )
        goto LABEL_13;
      v13 = v15;
      if ( v11 % v12 != v16[1] % v12 )
        goto LABEL_13;
      v15 = (_QWORD *)*v15;
    }
    while ( v11 != v16[1] );
  }
  if ( !*v13 )
  {
LABEL_13:
    v20 = sub_222DA10((__int64)(a1 + 4), v12, a1[3], 1);
    v22 = v21;
    if ( !v20 )
    {
      v23 = (_QWORD *)*a1;
      v24 = (unsigned __int64 ***)(*a1 + v14 * 8);
      v25 = (unsigned __int64 *)*v24;
      if ( *v24 )
      {
LABEL_15:
        *v6 = *v25;
        **v24 = v6;
LABEL_16:
        ++a1[3];
        return v6;
      }
LABEL_30:
      v35 = a1[2];
      a1[2] = (unsigned __int64)v6;
      *v6 = v35;
      if ( v35 )
      {
        v23[*(_QWORD *)(v35 + 8) % a1[1]] = v6;
        v24 = (unsigned __int64 ***)(v14 * 8 + *a1);
      }
      *v24 = (unsigned __int64 **)(a1 + 2);
      goto LABEL_16;
    }
    if ( v21 == 1 )
    {
      v23 = a1 + 6;
      a1[6] = 0;
      v29 = a1 + 6;
    }
    else
    {
      if ( v21 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(a1 + 4, v12, v21);
      v26 = 8 * v21;
      v27 = (void *)sub_22077B0(8 * v21);
      v28 = memset(v27, 0, v26);
      v29 = a1 + 6;
      v23 = v28;
    }
    v30 = (_QWORD *)a1[2];
    a1[2] = 0;
    if ( !v30 )
    {
LABEL_27:
      if ( (_QWORD *)*a1 != v29 )
      {
        v36 = v23;
        j_j___libc_free_0(*a1);
        v23 = v36;
      }
      a1[1] = v22;
      *a1 = (unsigned __int64)v23;
      v14 = v11 % v22;
      v24 = (unsigned __int64 ***)&v23[v14];
      v25 = (unsigned __int64 *)v23[v14];
      if ( v25 )
        goto LABEL_15;
      goto LABEL_30;
    }
    v31 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v32 = v30;
        v30 = (_QWORD *)*v30;
        v33 = v32[1] % v22;
        v34 = (_QWORD **)&v23[v33];
        if ( !*v34 )
          break;
        *v32 = **v34;
        **v34 = v32;
LABEL_23:
        if ( !v30 )
          goto LABEL_27;
      }
      *v32 = a1[2];
      a1[2] = (unsigned __int64)v32;
      *v34 = a1 + 2;
      if ( !*v32 )
      {
        v31 = v33;
        goto LABEL_23;
      }
      v23[v31] = v32;
      v31 = v33;
      if ( !v30 )
        goto LABEL_27;
    }
  }
  v17 = v6[2];
  v18 = *v13;
  if ( v17 )
    j_j___libc_free_0(v17);
  j_j___libc_free_0((unsigned __int64)v6);
  return (unsigned __int64 *)v18;
}
