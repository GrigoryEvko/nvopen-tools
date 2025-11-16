// Function: sub_C80880
// Address: 0xc80880
//
void *__fastcall sub_C80880(__int64 a1, const char **a2, unsigned int a3)
{
  const char **v4; // rdi
  unsigned __int64 v5; // r12
  bool v6; // zf
  char *v7; // r13
  __int64 v8; // r8
  unsigned __int8 v9; // al
  const char **v10; // r9
  size_t v11; // r14
  void *result; // rax
  unsigned __int64 v13; // rdx
  void *v14; // rdi
  const char **v15; // rdi
  unsigned __int64 v16; // rax
  size_t v17; // rax
  unsigned int v18; // [rsp+10h] [rbp-80h]
  unsigned int v19; // [rsp+18h] [rbp-78h]
  const char **v20; // [rsp+18h] [rbp-78h]
  const char **v21; // [rsp+18h] [rbp-78h]
  const char *v22; // [rsp+18h] [rbp-78h]
  const char **v23; // [rsp+18h] [rbp-78h]
  const char **v24; // [rsp+20h] [rbp-70h] BYREF
  size_t v25; // [rsp+28h] [rbp-68h]
  __int64 v26; // [rsp+30h] [rbp-60h]
  _BYTE v27[88]; // [rsp+38h] [rbp-58h] BYREF

  v4 = a2;
  v5 = *(_QWORD *)(a1 + 8);
  v6 = *((_BYTE *)a2 + 33) == 1;
  v24 = (const char **)v27;
  v25 = 0;
  v7 = *(char **)a1;
  v26 = 32;
  v8 = v5;
  if ( !v6 )
    goto LABEL_6;
  v9 = *((_BYTE *)a2 + 32);
  if ( v9 == 1 )
  {
    v11 = 0;
    v10 = 0;
    goto LABEL_8;
  }
  if ( (unsigned __int8)(v9 - 3) > 3u )
  {
LABEL_6:
    a2 = (const char **)&v24;
    v19 = a3;
    sub_CA0EC0(v4, &v24);
    v11 = v25;
    v10 = v24;
    v8 = v5;
    a3 = v19;
    goto LABEL_8;
  }
  if ( v9 == 4 )
  {
    v10 = *(const char ***)*a2;
    v11 = *((_QWORD *)*a2 + 1);
    goto LABEL_8;
  }
  if ( v9 > 4u )
  {
    if ( (unsigned __int8)(v9 - 5) <= 1u )
    {
      v11 = (size_t)a2[1];
      v10 = (const char **)*a2;
      goto LABEL_8;
    }
LABEL_34:
    BUG();
  }
  if ( v9 != 3 )
    goto LABEL_34;
  v10 = (const char **)*a2;
  v11 = 0;
  if ( *a2 )
  {
    v18 = a3;
    v22 = *a2;
    v17 = strlen(*a2);
    v10 = (const char **)v22;
    a3 = v18;
    v8 = v5;
    v11 = v17;
  }
  do
  {
LABEL_8:
    if ( !v5 )
      goto LABEL_9;
    --v5;
  }
  while ( v7[v5] != 46 );
  a2 = (const char **)v8;
  v20 = v10;
  v16 = sub_C80770(v7, v8, a3);
  v10 = v20;
  if ( v16 > v5 )
  {
LABEL_9:
    v5 = *(_QWORD *)(a1 + 8);
    goto LABEL_10;
  }
  *(_QWORD *)(a1 + 8) = v5;
LABEL_10:
  result = *(void **)(a1 + 16);
  if ( v11 )
  {
    if ( *(_BYTE *)v10 == 46 )
    {
      v13 = v5 + v11;
      if ( v5 + v11 <= *(_QWORD *)(a1 + 16) )
      {
LABEL_13:
        v14 = (void *)(v5 + *(_QWORD *)a1);
LABEL_14:
        a2 = v10;
        result = memcpy(v14, v10, v11);
        v5 = *(_QWORD *)(a1 + 8);
        goto LABEL_15;
      }
    }
    else
    {
      if ( v5 + 1 > (unsigned __int64)result )
      {
        v23 = v10;
        sub_C8D290(a1, a1 + 24, v5 + 1, 1);
        v5 = *(_QWORD *)(a1 + 8);
        v10 = v23;
      }
      *(_BYTE *)(*(_QWORD *)a1 + v5) = 46;
      v5 = *(_QWORD *)(a1 + 8) + 1LL;
      *(_QWORD *)(a1 + 8) = v5;
      v13 = v5 + v11;
      if ( v5 + v11 <= *(_QWORD *)(a1 + 16) )
        goto LABEL_13;
    }
    v21 = v10;
    sub_C8D290(a1, a1 + 24, v13, 1);
    v10 = v21;
    v14 = (void *)(*(_QWORD *)a1 + *(_QWORD *)(a1 + 8));
    goto LABEL_14;
  }
  if ( (unsigned __int64)result < v5 )
  {
    a2 = (const char **)(a1 + 24);
    result = (void *)sub_C8D290(a1, a1 + 24, v5, 1);
    v5 = *(_QWORD *)(a1 + 8);
  }
LABEL_15:
  v15 = v24;
  *(_QWORD *)(a1 + 8) = v11 + v5;
  if ( v15 != (const char **)v27 )
    return (void *)_libc_free(v15, a2);
  return result;
}
