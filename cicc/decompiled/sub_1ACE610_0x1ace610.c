// Function: sub_1ACE610
// Address: 0x1ace610
//
char __fastcall sub_1ACE610(__int64 a1, __int64 a2)
{
  char result; // al
  __int64 v3; // rax
  __int64 v4; // r15
  const void *v5; // r14
  size_t v6; // r12
  size_t v7; // rbx
  unsigned __int64 v8; // rbx
  _QWORD *v9; // rax
  _QWORD *v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // rdx
  unsigned __int64 v13; // rsi
  char *v14; // rbx
  __int64 v15; // r13
  __int64 v16; // rax
  char *v17; // r13
  __int64 v18; // rax
  const void *v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rax
  const void *v22; // rdi
  char *v23; // r15
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  int *v27; // [rsp-110h] [rbp-110h]
  char *v28; // [rsp-110h] [rbp-110h]
  unsigned __int64 v29; // [rsp-108h] [rbp-108h] BYREF
  __int64 v30[2]; // [rsp-F8h] [rbp-F8h] BYREF
  __int64 v31; // [rsp-E8h] [rbp-E8h] BYREF
  _QWORD v32[27]; // [rsp-D8h] [rbp-D8h] BYREF

  result = 1;
  if ( *(_QWORD *)(a1 + 16) )
    return result;
  result = *(_BYTE *)(a1 + 24);
  if ( !result )
    return result;
  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(const void **)(v3 + 176);
  v6 = *(_QWORD *)(v3 + 184);
  sub_15E4EB0(v30, a2);
  v7 = v30[1];
  v27 = (int *)v30[0];
  sub_16C1840(v32);
  sub_16C1A90((int *)v32, v27, v7);
  sub_16C1AA0(v32, &v29);
  v8 = v29;
  if ( (__int64 *)v30[0] != &v31 )
    j_j___libc_free_0(v30[0], v31 + 1);
  v9 = *(_QWORD **)(v4 + 16);
  if ( !v9 )
    goto LABEL_57;
  v10 = (_QWORD *)(v4 + 8);
  do
  {
    while ( 1 )
    {
      v11 = v9[2];
      v12 = v9[3];
      if ( v8 <= v9[4] )
        break;
      v9 = (_QWORD *)v9[3];
      if ( !v12 )
        goto LABEL_11;
    }
    v10 = v9;
    v9 = (_QWORD *)v9[2];
  }
  while ( v11 );
LABEL_11:
  if ( (_QWORD *)(v4 + 8) == v10 )
    goto LABEL_57;
  if ( v8 < v10[4] )
    goto LABEL_57;
  v13 = (unsigned __int16)(4 * *(unsigned __int8 *)(v4 + 178)) & 0xFFF8
      | (unsigned __int64)(v10 + 4) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v13 )
    goto LABEL_57;
  v14 = *(char **)(v13 + 24);
  v28 = *(char **)(v13 + 32);
  v15 = (v28 - v14) >> 5;
  v16 = (v28 - v14) >> 3;
  if ( v15 <= 0 )
    goto LABEL_40;
  v17 = &v14[32 * v15];
  do
  {
    if ( v6 == *(_QWORD *)(*(_QWORD *)v14 + 32LL) )
    {
      if ( !v6 || !memcmp(*(const void **)(*(_QWORD *)v14 + 24LL), v5, v6) )
        goto LABEL_23;
      v26 = *((_QWORD *)v14 + 1);
      v23 = v14 + 8;
      v19 = *(const void **)(v26 + 24);
      if ( v6 != *(_QWORD *)(v26 + 32) )
      {
LABEL_27:
        v24 = *((_QWORD *)v14 + 2);
        if ( v6 == *(_QWORD *)(v24 + 32) )
        {
          v23 = v14 + 16;
          if ( !memcmp(*(const void **)(v24 + 24), v5, v6) )
            goto LABEL_29;
          goto LABEL_35;
        }
        goto LABEL_18;
      }
LABEL_26:
      if ( !memcmp(v19, v5, v6) )
        goto LABEL_29;
      goto LABEL_27;
    }
    v18 = *((_QWORD *)v14 + 1);
    v19 = *(const void **)(v18 + 24);
    if ( v6 == *(_QWORD *)(v18 + 32) )
    {
      v23 = v14 + 8;
      if ( !v6 )
        goto LABEL_29;
      goto LABEL_26;
    }
    v20 = *((_QWORD *)v14 + 2);
    if ( v6 == *(_QWORD *)(v20 + 32) )
    {
      v23 = v14 + 16;
      if ( !v6 || !memcmp(*(const void **)(v20 + 24), v5, v6) )
      {
LABEL_29:
        v14 = v23;
        goto LABEL_23;
      }
LABEL_35:
      v25 = *((_QWORD *)v14 + 3);
      v23 = v14 + 24;
      v22 = *(const void **)(v25 + 24);
      if ( v6 != *(_QWORD *)(v25 + 32) )
        goto LABEL_19;
      goto LABEL_31;
    }
LABEL_18:
    v21 = *((_QWORD *)v14 + 3);
    v22 = *(const void **)(v21 + 24);
    if ( v6 != *(_QWORD *)(v21 + 32) )
      goto LABEL_19;
    v23 = v14 + 24;
    if ( !v6 )
      goto LABEL_29;
LABEL_31:
    if ( !memcmp(v22, v5, v6) )
    {
      v14 = v23;
      goto LABEL_23;
    }
LABEL_19:
    v14 += 32;
  }
  while ( v14 != v17 );
  v16 = (v28 - v14) >> 3;
LABEL_40:
  if ( v16 == 2 )
    goto LABEL_49;
  if ( v16 != 3 )
  {
    if ( v16 == 1 )
    {
LABEL_43:
      if ( v6 == *(_QWORD *)(*(_QWORD *)v14 + 32LL) && (!v6 || !memcmp(*(const void **)(*(_QWORD *)v14 + 24LL), v5, v6)) )
        goto LABEL_23;
    }
LABEL_57:
    BUG();
  }
  if ( v6 != *(_QWORD *)(*(_QWORD *)v14 + 32LL) || v6 && memcmp(*(const void **)(*(_QWORD *)v14 + 24LL), v5, v6) )
  {
    v14 += 8;
LABEL_49:
    if ( v6 != *(_QWORD *)(*(_QWORD *)v14 + 32LL) || v6 && memcmp(*(const void **)(*(_QWORD *)v14 + 24LL), v5, v6) )
    {
      v14 += 8;
      goto LABEL_43;
    }
  }
LABEL_23:
  if ( v28 == v14 )
    goto LABEL_57;
  return (*(_BYTE *)(*(_QWORD *)v14 + 12LL) & 0xFu) - 7 > 1;
}
