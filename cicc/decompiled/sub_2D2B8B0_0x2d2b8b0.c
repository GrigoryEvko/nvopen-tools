// Function: sub_2D2B8B0
// Address: 0x2d2b8b0
//
unsigned __int64 *__fastcall sub_2D2B8B0(unsigned __int64 *a1, __int64 *a2)
{
  unsigned __int64 v4; // r13
  __int64 v5; // r14
  __int64 *v6; // rax
  __int64 v7; // rax
  unsigned __int64 *v9; // rax
  unsigned __int64 *v10; // r12
  __int64 v11; // rdx
  _QWORD *v12; // rdi
  __int64 v13; // rsi
  char v14; // al
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // r15
  _QWORD *v17; // r8
  __int64 v18; // r14
  unsigned __int64 ***v19; // rax
  unsigned __int64 *v20; // rdx
  size_t v21; // r14
  void *v22; // rax
  _QWORD *v23; // rax
  _QWORD *v24; // r10
  _QWORD *v25; // rsi
  unsigned __int64 v26; // rdi
  _QWORD *v27; // rcx
  unsigned __int64 v28; // rdx
  _QWORD **v29; // rax
  unsigned __int64 v30; // rdx
  _QWORD *v31; // [rsp+8h] [rbp-38h]

  v4 = *a2;
  v5 = *a2 % a1[1];
  v6 = sub_2D2B810(a1, v5, a2, *a2);
  if ( v6 )
  {
    v7 = *v6;
    if ( v7 )
      return (unsigned __int64 *)(v7 + 16);
  }
  v9 = (unsigned __int64 *)sub_22077B0(0x48u);
  v10 = v9;
  if ( v9 )
    *v9 = 0;
  v11 = a1[3];
  v12 = a1 + 4;
  v13 = a1[1];
  v9[1] = *a2;
  v9[2] = (unsigned __int64)(v9 + 4);
  v9[3] = 0x100000000LL;
  v14 = sub_222DA10((__int64)(a1 + 4), v13, v11, 1);
  v16 = v15;
  if ( v14 )
  {
    if ( v15 == 1 )
    {
      v17 = a1 + 6;
      a1[6] = 0;
      v24 = a1 + 6;
    }
    else
    {
      if ( v15 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(v12, v13, v15);
      v21 = 8 * v15;
      v22 = (void *)sub_22077B0(8 * v15);
      v23 = memset(v22, 0, v21);
      v24 = a1 + 6;
      v17 = v23;
    }
    v25 = (_QWORD *)a1[2];
    a1[2] = 0;
    if ( !v25 )
    {
LABEL_21:
      if ( v24 != (_QWORD *)*a1 )
      {
        v31 = v17;
        j_j___libc_free_0(*a1);
        v17 = v31;
      }
      a1[1] = v16;
      *a1 = (unsigned __int64)v17;
      v5 = v4 % v16;
      goto LABEL_8;
    }
    v26 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v27 = v25;
        v25 = (_QWORD *)*v25;
        v28 = v27[8] % v16;
        v29 = (_QWORD **)&v17[v28];
        if ( !*v29 )
          break;
        *v27 = **v29;
        **v29 = v27;
LABEL_17:
        if ( !v25 )
          goto LABEL_21;
      }
      *v27 = a1[2];
      a1[2] = (unsigned __int64)v27;
      *v29 = a1 + 2;
      if ( !*v27 )
      {
        v26 = v28;
        goto LABEL_17;
      }
      v17[v26] = v27;
      v26 = v28;
      if ( !v25 )
        goto LABEL_21;
    }
  }
  v17 = (_QWORD *)*a1;
LABEL_8:
  v18 = v5;
  v10[8] = v4;
  v19 = (unsigned __int64 ***)&v17[v18];
  v20 = (unsigned __int64 *)v17[v18];
  if ( v20 )
  {
    *v10 = *v20;
    **v19 = v10;
  }
  else
  {
    v30 = a1[2];
    a1[2] = (unsigned __int64)v10;
    *v10 = v30;
    if ( v30 )
    {
      v17[*(_QWORD *)(v30 + 64) % a1[1]] = v10;
      v19 = (unsigned __int64 ***)(v18 * 8 + *a1);
    }
    *v19 = (unsigned __int64 **)(a1 + 2);
  }
  ++a1[3];
  return v10 + 2;
}
