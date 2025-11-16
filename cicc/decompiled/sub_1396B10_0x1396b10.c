// Function: sub_1396B10
// Address: 0x1396b10
//
void __fastcall sub_1396B10(char *a1, char *a2)
{
  char *v2; // r12
  const void *v3; // r13
  size_t v4; // rdx
  size_t v5; // r15
  size_t v6; // rdx
  const void *v7; // rdi
  size_t v8; // rbx
  int v9; // eax
  __int64 *v10; // rbx
  char *v11; // r15
  __int64 v12; // r14
  __int64 *v13; // r15
  __int64 v14; // rdi
  char *i; // rbx
  __int64 v16; // rax
  size_t v17; // rdx
  size_t v18; // r13
  const void *v19; // rsi
  size_t v20; // rdx
  const void *v21; // rdi
  size_t v22; // r14
  int v23; // eax
  __int64 *v24; // rax
  __int64 v25; // rdi
  char *v26; // r15

  if ( a1 == a2 )
    return;
  v2 = a1 + 8;
  if ( a2 == a1 + 8 )
    return;
  do
  {
    while ( 1 )
    {
      v10 = *(__int64 **)v2;
      v12 = **(_QWORD **)v2;
      v13 = *(__int64 **)v2;
      v14 = **(_QWORD **)a1;
      if ( !v14 )
        break;
      if ( !v12 )
        goto LABEL_11;
      v3 = (const void *)sub_1649960(v14);
      v5 = v4;
      v7 = (const void *)sub_1649960(v12);
      v8 = v6;
      if ( v6 > v5 )
      {
        if ( !v5 )
          goto LABEL_37;
        v9 = memcmp(v7, v3, v5);
        if ( v9 )
        {
LABEL_36:
          if ( v9 >= 0 )
          {
LABEL_37:
            v13 = *(__int64 **)v2;
            v12 = **(_QWORD **)v2;
            break;
          }
          goto LABEL_10;
        }
      }
      else
      {
        if ( v6 )
        {
          v9 = memcmp(v7, v3, v6);
          if ( v9 )
            goto LABEL_36;
        }
        if ( v8 == v5 )
          goto LABEL_37;
      }
      if ( v8 >= v5 )
        goto LABEL_37;
LABEL_10:
      v10 = *(__int64 **)v2;
LABEL_11:
      v11 = v2 + 8;
      if ( a1 != v2 )
        memmove(a1 + 8, a1, v2 - a1);
      v2 += 8;
      *(_QWORD *)a1 = v10;
      if ( a2 == v11 )
        return;
    }
    for ( i = v2; ; i -= 8 )
    {
      v24 = (__int64 *)*((_QWORD *)i - 1);
      v25 = *v24;
      if ( v12 )
        break;
      if ( !v25 )
        goto LABEL_26;
LABEL_23:
      *(_QWORD *)i = v24;
      v12 = *v13;
    }
    if ( !v25 )
      goto LABEL_26;
    v16 = sub_1649960(v25);
    v18 = v17;
    v19 = (const void *)v16;
    v21 = (const void *)sub_1649960(v12);
    v22 = v20;
    if ( v20 > v18 )
    {
      if ( !v18 )
      {
LABEL_26:
        *(_QWORD *)i = v13;
        v26 = v2 + 8;
        goto LABEL_27;
      }
      v23 = memcmp(v21, v19, v18);
      if ( !v23 )
        goto LABEL_21;
    }
    else if ( !v20 || (v23 = memcmp(v21, v19, v20)) == 0 )
    {
      if ( v22 == v18 )
        goto LABEL_26;
LABEL_21:
      if ( v22 >= v18 )
        goto LABEL_26;
LABEL_22:
      v24 = (__int64 *)*((_QWORD *)i - 1);
      goto LABEL_23;
    }
    if ( v23 < 0 )
      goto LABEL_22;
    *(_QWORD *)i = v13;
    v26 = v2 + 8;
LABEL_27:
    v2 = v26;
  }
  while ( a2 != v26 );
}
