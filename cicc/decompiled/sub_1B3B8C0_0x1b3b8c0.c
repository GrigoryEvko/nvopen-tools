// Function: sub_1B3B8C0
// Address: 0x1b3b8c0
//
__int64 *__fastcall sub_1B3B8C0(__int64 *a1, __int64 a2, _BYTE *a3, size_t a4)
{
  __int64 v7; // r15
  int v8; // eax
  unsigned int v9; // ecx
  __int64 v10; // rdx
  _QWORD *v11; // rax
  _QWORD *i; // rdx
  size_t v13; // rax
  __int64 *v14; // rdx
  __int64 *v15; // rdi
  __int64 *result; // rax
  size_t v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // rdi
  size_t v20; // rdx
  __int64 v21; // rax
  __int64 *v22; // rdi
  _QWORD *v23; // r8
  unsigned int v24; // eax
  int v25; // eax
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rax
  __int64 v28; // r9
  _QWORD *v29; // rax
  __int64 v30; // rdx
  _QWORD *j; // rdx
  _QWORD *v32; // rax
  int v33; // [rsp+4h] [rbp-6Ch]
  __int64 v34; // [rsp+8h] [rbp-68h]
  size_t v35; // [rsp+18h] [rbp-58h] BYREF
  __int64 *v36; // [rsp+20h] [rbp-50h] BYREF
  size_t n; // [rsp+28h] [rbp-48h]
  _QWORD src[8]; // [rsp+30h] [rbp-40h] BYREF

  v7 = *a1;
  if ( *a1 )
  {
    v8 = *(_DWORD *)(v7 + 16);
    ++*(_QWORD *)v7;
    if ( !v8 )
    {
      if ( !*(_DWORD *)(v7 + 20) )
        goto LABEL_9;
      v10 = *(unsigned int *)(v7 + 24);
      if ( (unsigned int)v10 > 0x40 )
      {
        j___libc_free_0(*(_QWORD *)(v7 + 8));
        *(_QWORD *)(v7 + 8) = 0;
        *(_QWORD *)(v7 + 16) = 0;
        *(_DWORD *)(v7 + 24) = 0;
        goto LABEL_9;
      }
      goto LABEL_6;
    }
    v9 = 4 * v8;
    v10 = *(unsigned int *)(v7 + 24);
    if ( (unsigned int)(4 * v8) < 0x40 )
      v9 = 64;
    if ( (unsigned int)v10 <= v9 )
    {
LABEL_6:
      v11 = *(_QWORD **)(v7 + 8);
      for ( i = &v11[2 * v10]; i != v11; v11 += 2 )
        *v11 = -8;
      *(_QWORD *)(v7 + 16) = 0;
      goto LABEL_9;
    }
    v23 = *(_QWORD **)(v7 + 8);
    v24 = v8 - 1;
    if ( !v24 )
    {
      v33 = 128;
      v28 = 2048;
LABEL_39:
      v34 = v28;
      j___libc_free_0(v23);
      *(_DWORD *)(v7 + 24) = v33;
      v29 = (_QWORD *)sub_22077B0(v34);
      v30 = *(unsigned int *)(v7 + 24);
      *(_QWORD *)(v7 + 16) = 0;
      *(_QWORD *)(v7 + 8) = v29;
      for ( j = &v29[2 * v30]; j != v29; v29 += 2 )
      {
        if ( v29 )
          *v29 = -8;
      }
      goto LABEL_9;
    }
    _BitScanReverse(&v24, v24);
    v25 = 1 << (33 - (v24 ^ 0x1F));
    if ( v25 < 64 )
      v25 = 64;
    if ( (_DWORD)v10 != v25 )
    {
      v26 = (((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
          | (4 * v25 / 3u + 1)
          | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)
          | (((((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
            | (4 * v25 / 3u + 1)
            | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 4);
      v27 = (v26 >> 8) | v26;
      v33 = (v27 | (v27 >> 16)) + 1;
      v28 = 16 * ((v27 | (v27 >> 16)) + 1);
      goto LABEL_39;
    }
    *(_QWORD *)(v7 + 16) = 0;
    v32 = &v23[2 * (unsigned int)v10];
    do
    {
      if ( v23 )
        *v23 = -8;
      v23 += 2;
    }
    while ( v32 != v23 );
  }
  else
  {
    v21 = sub_22077B0(32);
    if ( v21 )
    {
      *(_QWORD *)v21 = 0;
      *(_QWORD *)(v21 + 8) = 0;
      *(_QWORD *)(v21 + 16) = 0;
      *(_DWORD *)(v21 + 24) = 0;
    }
    *a1 = v21;
  }
LABEL_9:
  a1[1] = a2;
  if ( !a3 )
  {
    LOBYTE(src[0]) = 0;
    v15 = (__int64 *)a1[2];
    v20 = 0;
    v36 = src;
LABEL_21:
    a1[3] = v20;
    *((_BYTE *)v15 + v20) = 0;
    result = v36;
    goto LABEL_22;
  }
  v35 = a4;
  v13 = a4;
  v36 = src;
  if ( a4 > 0xF )
  {
    v36 = (__int64 *)sub_22409D0(&v36, &v35, 0);
    v22 = v36;
    src[0] = v35;
  }
  else
  {
    if ( a4 == 1 )
    {
      LOBYTE(src[0]) = *a3;
      v14 = src;
      goto LABEL_13;
    }
    if ( !a4 )
    {
      v14 = src;
      goto LABEL_13;
    }
    v22 = src;
  }
  memcpy(v22, a3, a4);
  v13 = v35;
  v14 = v36;
LABEL_13:
  n = v13;
  *((_BYTE *)v14 + v13) = 0;
  v15 = (__int64 *)a1[2];
  result = v15;
  if ( v36 == src )
  {
    v20 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)v15 = src[0];
      else
        memcpy(v15, src, n);
      v20 = n;
      v15 = (__int64 *)a1[2];
    }
    goto LABEL_21;
  }
  v17 = n;
  v18 = src[0];
  if ( v15 == a1 + 4 )
  {
    a1[2] = (__int64)v36;
    a1[3] = v17;
    a1[4] = v18;
  }
  else
  {
    v19 = a1[4];
    a1[2] = (__int64)v36;
    a1[3] = v17;
    a1[4] = v18;
    if ( result )
    {
      v36 = result;
      src[0] = v19;
      goto LABEL_22;
    }
  }
  v36 = src;
  result = src;
LABEL_22:
  n = 0;
  *(_BYTE *)result = 0;
  if ( v36 != src )
    return (__int64 *)j_j___libc_free_0(v36, src[0] + 1LL);
  return result;
}
