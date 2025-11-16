// Function: sub_14EC5C0
// Address: 0x14ec5c0
//
__int64 __fastcall sub_14EC5C0(__int64 a1, unsigned __int64 a2, const void *a3, size_t a4)
{
  _QWORD *v4; // rax
  _QWORD *v5; // r9
  _QWORD *v7; // rdi
  __int64 v10; // rcx
  __int64 v11; // rdx
  unsigned __int64 v12; // rdi
  char *v13; // rbx
  __int64 v14; // r13
  __int64 v15; // rax
  char *v16; // r13
  __int64 v17; // rax
  const void *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  const void *v21; // rdi
  char *v23; // r14
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  char *v27; // [rsp-40h] [rbp-40h]

  v4 = *(_QWORD **)(a1 + 16);
  if ( !v4 )
    return 0;
  v5 = (_QWORD *)(a1 + 8);
  v7 = (_QWORD *)(a1 + 8);
  do
  {
    while ( 1 )
    {
      v10 = v4[2];
      v11 = v4[3];
      if ( a2 <= v4[4] )
        break;
      v4 = (_QWORD *)v4[3];
      if ( !v11 )
        goto LABEL_6;
    }
    v7 = v4;
    v4 = (_QWORD *)v4[2];
  }
  while ( v10 );
LABEL_6:
  if ( v5 == v7 )
    return 0;
  if ( a2 < v7[4] )
    return 0;
  v12 = (unsigned __int16)(4 * *(unsigned __int8 *)(a1 + 178)) & 0xFFF8
      | (unsigned __int64)(v7 + 4) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v12 )
    return 0;
  v13 = *(char **)(v12 + 24);
  v27 = *(char **)(v12 + 32);
  v14 = (v27 - v13) >> 5;
  v15 = (v27 - v13) >> 3;
  if ( v14 > 0 )
  {
    v16 = &v13[32 * v14];
    while ( a4 != *(_QWORD *)(*(_QWORD *)v13 + 32LL) )
    {
      v17 = *((_QWORD *)v13 + 1);
      v18 = *(const void **)(v17 + 24);
      if ( a4 == *(_QWORD *)(v17 + 32) )
      {
        v23 = v13 + 8;
        if ( !a4 )
          goto LABEL_30;
        goto LABEL_27;
      }
      v19 = *((_QWORD *)v13 + 2);
      if ( a4 == *(_QWORD *)(v19 + 32) )
      {
        v23 = v13 + 16;
        if ( !a4 || !memcmp(*(const void **)(v19 + 24), a3, a4) )
        {
LABEL_30:
          v13 = v23;
          goto LABEL_18;
        }
LABEL_36:
        v25 = *((_QWORD *)v13 + 3);
        v23 = v13 + 24;
        v21 = *(const void **)(v25 + 24);
        if ( *(_QWORD *)(v25 + 32) != a4 )
          goto LABEL_14;
        goto LABEL_32;
      }
LABEL_13:
      v20 = *((_QWORD *)v13 + 3);
      v21 = *(const void **)(v20 + 24);
      if ( a4 != *(_QWORD *)(v20 + 32) )
        goto LABEL_14;
      v23 = v13 + 24;
      if ( !a4 )
        goto LABEL_30;
LABEL_32:
      if ( !memcmp(v21, a3, a4) )
      {
        v13 = v23;
        goto LABEL_18;
      }
LABEL_14:
      v13 += 32;
      if ( v13 == v16 )
      {
        v15 = (v27 - v13) >> 3;
        goto LABEL_21;
      }
    }
    if ( !a4 || !memcmp(*(const void **)(*(_QWORD *)v13 + 24LL), a3, a4) )
      goto LABEL_18;
    v26 = *((_QWORD *)v13 + 1);
    v23 = v13 + 8;
    v18 = *(const void **)(v26 + 24);
    if ( a4 == *(_QWORD *)(v26 + 32) )
    {
LABEL_27:
      if ( !memcmp(v18, a3, a4) )
        goto LABEL_30;
    }
    v24 = *((_QWORD *)v13 + 2);
    if ( a4 == *(_QWORD *)(v24 + 32) )
    {
      v23 = v13 + 16;
      if ( !memcmp(*(const void **)(v24 + 24), a3, a4) )
        goto LABEL_30;
      goto LABEL_36;
    }
    goto LABEL_13;
  }
LABEL_21:
  if ( v15 != 2 )
  {
    if ( v15 != 3 )
    {
      if ( v15 != 1 )
        return 0;
      goto LABEL_24;
    }
    if ( a4 == *(_QWORD *)(*(_QWORD *)v13 + 32LL) && (!a4 || !memcmp(*(const void **)(*(_QWORD *)v13 + 24LL), a3, a4)) )
      goto LABEL_18;
    v13 += 8;
  }
  if ( a4 == *(_QWORD *)(*(_QWORD *)v13 + 32LL) && (!a4 || !memcmp(*(const void **)(*(_QWORD *)v13 + 24LL), a3, a4)) )
    goto LABEL_18;
  v13 += 8;
LABEL_24:
  if ( a4 != *(_QWORD *)(*(_QWORD *)v13 + 32LL) )
    return 0;
  if ( !a4 || !memcmp(*(const void **)(*(_QWORD *)v13 + 24LL), a3, a4) )
  {
LABEL_18:
    if ( v27 != v13 )
      return *(_QWORD *)v13;
    return 0;
  }
  return 0;
}
