// Function: sub_E12130
// Address: 0xe12130
//
unsigned __int64 __fastcall sub_E12130(__int64 a1, void **a2)
{
  char *v4; // rsi
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rsi
  unsigned __int64 v7; // rax
  char *v8; // rax
  char *v9; // rax
  __int64 v10; // rdi
  size_t v11; // r13
  unsigned __int64 result; // rax
  void *v13; // r8
  unsigned __int64 v14; // rax
  size_t v15; // rax
  const void *v16; // r12
  char *v17; // r8
  unsigned __int64 v18; // rsi
  unsigned __int64 v19; // rax
  __int64 v20; // rax

  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 16) + 32LL))(*(_QWORD *)(a1 + 16));
  v4 = (char *)a2[1];
  v5 = (unsigned __int64)a2[2];
  if ( (unsigned __int64)(v4 + 5) <= v5 )
  {
    v8 = (char *)*a2;
  }
  else
  {
    v6 = (unsigned __int64)(v4 + 997);
    v7 = 2 * v5;
    if ( v6 <= v7 )
      a2[2] = (void *)v7;
    else
      a2[2] = (void *)v6;
    v8 = (char *)realloc(*a2);
    *a2 = v8;
    if ( !v8 )
      goto LABEL_23;
    v4 = (char *)a2[1];
  }
  v9 = &v8[(_QWORD)v4];
  *(_DWORD *)v9 = 1768055131;
  v9[4] = 58;
  v10 = (__int64)a2[1] + 5;
  a2[1] = (void *)v10;
  v11 = *(_QWORD *)(a1 + 24);
  if ( v11 )
  {
    v15 = (size_t)a2[2];
    v16 = *(const void **)(a1 + 32);
    v17 = (char *)*a2;
    if ( v11 + v10 > v15 )
    {
      v18 = v11 + v10 + 992;
      v19 = 2 * v15;
      if ( v18 > v19 )
        a2[2] = (void *)v18;
      else
        a2[2] = (void *)v19;
      v20 = realloc(v17);
      *a2 = (void *)v20;
      v17 = (char *)v20;
      if ( !v20 )
        goto LABEL_23;
      v10 = (__int64)a2[1];
    }
    memcpy(&v17[v10], v16, v11);
    v10 = (__int64)a2[1] + v11;
    a2[1] = (void *)v10;
  }
  result = (unsigned __int64)a2[2];
  v13 = *a2;
  if ( v10 + 1 > result )
  {
    v14 = 2 * result;
    if ( v10 + 993 <= v14 )
      a2[2] = (void *)v14;
    else
      a2[2] = (void *)(v10 + 993);
    result = realloc(v13);
    *a2 = (void *)result;
    v13 = (void *)result;
    if ( result )
    {
      v10 = (__int64)a2[1];
      goto LABEL_13;
    }
LABEL_23:
    abort();
  }
LABEL_13:
  *((_BYTE *)v13 + v10) = 93;
  a2[1] = (char *)a2[1] + 1;
  return result;
}
