// Function: sub_1C745E0
// Address: 0x1c745e0
//
__int64 __fastcall sub_1C745E0(const void ***a1, __int64 a2)
{
  _QWORD *v3; // rax
  const void **v4; // rdx
  _QWORD *v5; // r13
  const void **v6; // r15
  _BYTE *v7; // rax
  _BYTE *v8; // rsi
  __int64 v9; // rbx
  size_t v10; // rdi
  __int64 v11; // rax
  void *v12; // r12
  _BYTE *v13; // rax
  void *v14; // r15
  size_t v15; // rdx
  _BYTE *v16; // rax
  signed __int64 v17; // r14
  void *v18; // rdi
  void *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  unsigned int v22; // ebx
  size_t v24; // [rsp+0h] [rbp-50h]
  size_t v25; // [rsp+8h] [rbp-48h]
  const void **v26; // [rsp+10h] [rbp-40h]
  const void **v27; // [rsp+10h] [rbp-40h]

  v3 = sub_1648700(a2);
  if ( *((_BYTE *)v3 + 16) <= 0x17u )
    return 0;
  v5 = v3;
  if ( !v3[5] )
    return 0;
  v6 = a1[2];
  v7 = v6[1];
  v8 = *v6;
  v9 = ((_BYTE *)(*a1)[1] - (_BYTE *)**a1) >> 3;
  v10 = v7 - (_BYTE *)*v6;
  v24 = v10;
  if ( v7 == *v6 )
  {
    v12 = 0;
  }
  else
  {
    if ( v10 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_38;
    v11 = sub_22077B0(v10);
    v8 = *v6;
    v12 = (void *)v11;
    v7 = v6[1];
    v10 = v7 - (_BYTE *)*v6;
  }
  if ( v8 != v7 )
    memmove(v12, v8, v10);
  v4 = a1[1];
  v13 = v4[1];
  v8 = *v4;
  v26 = v4;
  v10 = v13 - (_BYTE *)*v4;
  v25 = v10;
  if ( v13 == *v4 )
  {
    v14 = 0;
  }
  else
  {
    if ( v10 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_38;
    v14 = (void *)sub_22077B0(v10);
    v13 = v26[1];
    v8 = *v26;
    v10 = v13 - (_BYTE *)*v26;
  }
  if ( v8 != v13 )
  {
    v15 = v10;
    v10 = (size_t)v14;
    memmove(v14, v8, v15);
  }
  v16 = (*a1)[1];
  v8 = **a1;
  v27 = *a1;
  v4 = (const void **)(v16 - v8);
  v17 = v16 - v8;
  if ( v16 != v8 )
  {
    if ( (unsigned __int64)v4 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v18 = (void *)sub_22077B0(v4);
      v16 = v27[1];
      v8 = *v27;
      v4 = (const void **)(v16 - (_BYTE *)*v27);
      goto LABEL_16;
    }
LABEL_38:
    sub_4261EA(v10, v8, v4);
  }
  v18 = 0;
LABEL_16:
  if ( v8 == v16 )
  {
    v20 = v5[5];
    if ( (int)v9 <= 0 )
    {
      v22 = 1;
      if ( !v18 )
        goto LABEL_25;
      goto LABEL_24;
    }
    goto LABEL_18;
  }
  v19 = memmove(v18, v8, (size_t)v4);
  v20 = v5[5];
  v18 = v19;
  if ( (int)v9 > 0 )
  {
LABEL_18:
    v21 = 0;
    while ( *((_QWORD *)v18 + v21) != v20 && *((_QWORD *)v14 + v21) != v20 && *((_QWORD *)v12 + v21) != v20 )
    {
      if ( (int)v9 <= (int)++v21 )
        goto LABEL_31;
    }
    v22 = 0;
    goto LABEL_24;
  }
LABEL_31:
  v22 = 1;
LABEL_24:
  j_j___libc_free_0(v18, v17);
LABEL_25:
  if ( v14 )
    j_j___libc_free_0(v14, v25);
  if ( v12 )
    j_j___libc_free_0(v12, v24);
  return v22;
}
