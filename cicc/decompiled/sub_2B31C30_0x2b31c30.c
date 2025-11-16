// Function: sub_2B31C30
// Address: 0x2b31c30
//
bool __fastcall sub_2B31C30(__int64 a1, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v9; // rbx
  unsigned int v10; // esi
  __int64 v11; // rax
  _QWORD *v12; // r8
  __int64 v13; // rdx
  bool result; // al
  __int64 v15; // rdi
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  _BYTE *v18; // r13
  char *v19; // rdi
  int *i; // rcx
  __int64 v21; // rdx
  const void *v22; // rsi
  __int64 v23; // rcx
  size_t v24; // rdx
  size_t v25; // rdx
  char *v26; // r8
  int *v27; // rcx
  __int64 v28; // rdx
  char *v29; // rdi
  int *v30; // rcx
  __int64 v31; // rdx
  bool v32; // [rsp+8h] [rbp-78h]
  __int64 v33; // [rsp+8h] [rbp-78h]
  _BYTE *v34; // [rsp+10h] [rbp-70h] BYREF
  __int64 v35; // [rsp+18h] [rbp-68h]
  _BYTE v36[96]; // [rsp+20h] [rbp-60h] BYREF

  v9 = a2;
  v10 = *(_DWORD *)(a1 + 152);
  if ( !v10 )
  {
    v11 = *(unsigned int *)(a1 + 120);
    v12 = *(_QWORD **)a1;
    if ( v11 != a3 )
    {
      v13 = *(unsigned int *)(a1 + 8);
      result = 0;
      if ( v13 == a3 )
      {
        v24 = 8 * v13;
        result = 1;
        if ( v24 )
          return memcmp(a2, *(const void **)a1, v24) == 0;
      }
      return result;
    }
    v19 = &a2[8 * v11];
    if ( v19 == a2 )
      return 1;
    for ( i = *(int **)(a1 + 112); ; ++i )
    {
      v21 = *i;
      if ( (unsigned int)**(unsigned __int8 **)v9 - 12 > 1 )
      {
        if ( (_DWORD)v21 == -1 )
          return 0;
      }
      else if ( (_DWORD)v21 == -1 )
      {
        goto LABEL_16;
      }
      if ( *(_QWORD *)v9 != v12[v21] )
        return 0;
LABEL_16:
      v9 += 8;
      if ( v19 == v9 )
        return 1;
    }
  }
  v15 = *(_QWORD *)(a1 + 144);
  v35 = 0xC00000000LL;
  v34 = v36;
  sub_2B0FC00(v15, v10, (__int64)&v34, a4, (__int64)&v34, a6);
  v16 = *(unsigned int *)(a1 + 8);
  if ( v16 == a3 )
  {
    v18 = v34;
    v25 = 8 * v16;
    if ( (unsigned int)v35 != v16 )
    {
      result = 1;
      if ( v25 )
        result = memcmp(a2, *(const void **)a1, v25) == 0;
      goto LABEL_8;
    }
    v29 = &a2[v25];
    if ( &a2[v25] == a2 )
      goto LABEL_36;
    v30 = (int *)v34;
    while ( 1 )
    {
      v31 = *v30;
      if ( (unsigned int)**(unsigned __int8 **)v9 - 12 <= 1 )
      {
        if ( (_DWORD)v31 == -1 )
          goto LABEL_43;
      }
      else if ( (_DWORD)v31 == -1 )
      {
        result = 0;
        goto LABEL_8;
      }
      if ( *(_QWORD *)v9 != *(_QWORD *)(*(_QWORD *)a1 + 8 * v31) )
      {
LABEL_34:
        result = 0;
        goto LABEL_8;
      }
LABEL_43:
      v9 += 8;
      ++v30;
      if ( v29 == v9 )
        goto LABEL_36;
    }
  }
  v17 = *(unsigned int *)(a1 + 120);
  if ( v17 == a3 )
  {
    v33 = *(unsigned int *)(a1 + 120);
    sub_2B319A0((__int64)&v34, *(int **)(a1 + 112), v17);
    v18 = v34;
    v22 = *(const void **)a1;
    if ( (unsigned int)v35 != v33 )
    {
      v23 = *(unsigned int *)(a1 + 8);
      result = 0;
      if ( v23 == v33 )
      {
        result = 1;
        if ( 8 * v23 )
          result = memcmp(a2, v22, 8 * v23) == 0;
      }
      goto LABEL_8;
    }
    v26 = &a2[8 * (unsigned int)v35];
    if ( v26 != a2 )
    {
      v27 = (int *)v34;
      while ( 1 )
      {
        v28 = *v27;
        if ( (unsigned int)**(unsigned __int8 **)v9 - 12 > 1 )
          break;
        if ( (_DWORD)v28 != -1 )
          goto LABEL_33;
LABEL_35:
        v9 += 8;
        ++v27;
        if ( v26 == v9 )
          goto LABEL_36;
      }
      if ( (_DWORD)v28 == -1 )
      {
        result = 0;
        goto LABEL_8;
      }
LABEL_33:
      if ( *(_QWORD *)v9 != *((_QWORD *)v22 + v28) )
        goto LABEL_34;
      goto LABEL_35;
    }
LABEL_36:
    result = 1;
    goto LABEL_8;
  }
  v18 = v34;
  result = 0;
LABEL_8:
  if ( v18 != v36 )
  {
    v32 = result;
    _libc_free((unsigned __int64)v18);
    return v32;
  }
  return result;
}
