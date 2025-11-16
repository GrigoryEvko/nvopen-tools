// Function: sub_2CBF2A0
// Address: 0x2cbf2a0
//
__int64 __fastcall sub_2CBF2A0(const void ***a1, __int64 a2, const void **a3)
{
  __int64 v3; // r13
  const void **v4; // r15
  _BYTE *v6; // rax
  _BYTE *v7; // rsi
  __int64 v8; // rbx
  unsigned __int64 v9; // rdi
  __int64 v10; // rax
  void *v11; // r12
  _BYTE *v12; // rax
  void *v13; // r15
  size_t v14; // rdx
  _BYTE *v15; // rax
  void *v16; // rdi
  void *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  unsigned int v20; // ebx
  const void **v22; // [rsp+10h] [rbp-40h]
  const void **v23; // [rsp+10h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 24);
  if ( *(_BYTE *)v3 <= 0x1Cu || !*(_QWORD *)(v3 + 40) )
    return 0;
  v4 = a1[2];
  v6 = v4[1];
  v7 = *v4;
  v8 = ((_BYTE *)(*a1)[1] - (_BYTE *)**a1) >> 3;
  v9 = v6 - (_BYTE *)*v4;
  if ( v6 == *v4 )
  {
    v11 = 0;
  }
  else
  {
    if ( v9 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_38;
    v10 = sub_22077B0(v9);
    v7 = *v4;
    v11 = (void *)v10;
    v6 = v4[1];
    v9 = v6 - (_BYTE *)*v4;
  }
  if ( v7 != v6 )
    memmove(v11, v7, v9);
  a3 = a1[1];
  v12 = a3[1];
  v7 = *a3;
  v22 = a3;
  v9 = v12 - (_BYTE *)*a3;
  if ( v12 == *a3 )
  {
    v13 = 0;
  }
  else
  {
    if ( v9 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_38;
    v13 = (void *)sub_22077B0(v9);
    v12 = v22[1];
    v7 = *v22;
    v9 = v12 - (_BYTE *)*v22;
  }
  if ( v7 != v12 )
  {
    v14 = v9;
    v9 = (unsigned __int64)v13;
    memmove(v13, v7, v14);
  }
  v15 = (*a1)[1];
  v7 = **a1;
  v23 = *a1;
  a3 = (const void **)(v15 - v7);
  if ( v15 != v7 )
  {
    if ( (unsigned __int64)a3 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v16 = (void *)sub_22077B0((unsigned __int64)a3);
      v15 = v23[1];
      v7 = *v23;
      a3 = (const void **)(v15 - (_BYTE *)*v23);
      goto LABEL_16;
    }
LABEL_38:
    sub_4261EA(v9, v7, a3);
  }
  v16 = 0;
LABEL_16:
  if ( v7 == v15 )
  {
    v18 = *(_QWORD *)(v3 + 40);
    if ( (int)v8 <= 0 )
    {
      v20 = 1;
      if ( !v16 )
        goto LABEL_25;
      goto LABEL_24;
    }
    goto LABEL_18;
  }
  v17 = memmove(v16, v7, (size_t)a3);
  v18 = *(_QWORD *)(v3 + 40);
  v16 = v17;
  if ( (int)v8 > 0 )
  {
LABEL_18:
    v19 = 0;
    while ( *((_QWORD *)v16 + v19) != v18 && *((_QWORD *)v13 + v19) != v18 && *((_QWORD *)v11 + v19) != v18 )
    {
      if ( (int)v8 <= (int)++v19 )
        goto LABEL_31;
    }
    v20 = 0;
    goto LABEL_24;
  }
LABEL_31:
  v20 = 1;
LABEL_24:
  j_j___libc_free_0((unsigned __int64)v16);
LABEL_25:
  if ( v13 )
    j_j___libc_free_0((unsigned __int64)v13);
  if ( v11 )
    j_j___libc_free_0((unsigned __int64)v11);
  return v20;
}
