// Function: sub_1EFA320
// Address: 0x1efa320
//
__int64 *__fastcall sub_1EFA320(unsigned int *a1, __int64 a2, int a3, unsigned int a4, __int64 a5)
{
  int v8; // r14d
  unsigned int v9; // eax
  __int64 v10; // rcx
  void *v11; // r15
  __int64 v12; // rdx
  unsigned int v13; // esi
  __int64 v14; // rdi
  __int64 v15; // r8
  unsigned int v16; // ecx
  __int64 *result; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  size_t v20; // rdx
  __int64 v21; // rcx
  const void **v22; // r8
  int v23; // r11d
  __int64 *v24; // r10
  unsigned int v25; // ecx
  int v26; // ecx
  unsigned int v27; // eax
  unsigned int v28; // esi
  __int64 v29; // r8
  unsigned int v30; // edx
  __int64 v31; // rdi
  int v32; // r10d
  __int64 *v33; // r9
  unsigned int v34; // eax
  unsigned int v35; // edx
  __int64 v36; // rdi
  __int64 *v37; // r8
  unsigned int v38; // r14d
  int v39; // r9d
  __int64 v40; // rsi
  __int64 v41; // rax
  const void **v42; // [rsp+0h] [rbp-50h]
  size_t n; // [rsp+8h] [rbp-48h]
  size_t na; // [rsp+8h] [rbp-48h]
  __int64 v46; // [rsp+10h] [rbp-40h]
  __int64 v47; // [rsp+10h] [rbp-40h]
  __int64 v48; // [rsp+10h] [rbp-40h]

  v8 = *(_DWORD *)(a5 + 16);
  if ( !v8 )
  {
    v9 = a1[136];
    v10 = 0;
    v11 = 0;
    if ( v9 < a1[137] )
      goto LABEL_3;
    goto LABEL_12;
  }
  n = 8LL * ((unsigned int)(v8 + 63) >> 6);
  v19 = malloc(n);
  v20 = n;
  v21 = (unsigned int)(v8 + 63) >> 6;
  v22 = (const void **)a5;
  v11 = (void *)v19;
  if ( !v19 )
  {
    if ( n || (v41 = malloc(1u), v21 = (unsigned int)(v8 + 63) >> 6, v20 = 0, v22 = (const void **)a5, !v41) )
    {
      v42 = v22;
      na = v20;
      v48 = v21;
      sub_16BD1C0("Allocation failed", 1u);
      v21 = v48;
      v20 = na;
      v22 = v42;
    }
    else
    {
      v11 = (void *)v41;
    }
  }
  v46 = v21;
  memcpy(v11, *v22, v20);
  v10 = v46;
  v9 = a1[136];
  if ( v9 >= a1[137] )
  {
LABEL_12:
    v47 = v10;
    sub_1EF9AB0((__int64)(a1 + 134), 0);
    v9 = a1[136];
    v10 = v47;
  }
LABEL_3:
  v12 = *((_QWORD *)a1 + 67) + 40LL * v9;
  if ( v12 )
  {
    *(_QWORD *)(v12 + 16) = v11;
    v11 = 0;
    *(_QWORD *)v12 = a2;
    *(_DWORD *)(v12 + 8) = a3;
    *(_DWORD *)(v12 + 12) = a4;
    *(_QWORD *)(v12 + 24) = v10;
    *(_DWORD *)(v12 + 32) = v8;
    v9 = a1[136];
  }
  a1[136] = v9 + 1;
  _libc_free((unsigned __int64)v11);
  v13 = a1[232];
  v14 = (__int64)(a1 + 226);
  if ( !v13 )
  {
    ++*((_QWORD *)a1 + 113);
    goto LABEL_23;
  }
  v15 = *((_QWORD *)a1 + 114);
  v16 = (v13 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = (__int64 *)(v15 + 16LL * v16);
  v18 = *result;
  if ( *result == a2 )
    goto LABEL_7;
  v23 = 1;
  v24 = 0;
  while ( v18 != -8 )
  {
    if ( v18 == -16 && !v24 )
      v24 = result;
    v16 = (v13 - 1) & (v23 + v16);
    result = (__int64 *)(v15 + 16LL * v16);
    v18 = *result;
    if ( *result == a2 )
      goto LABEL_7;
    ++v23;
  }
  v25 = a1[230];
  if ( v24 )
    result = v24;
  ++*((_QWORD *)a1 + 113);
  v26 = v25 + 1;
  if ( 4 * v26 >= 3 * v13 )
  {
LABEL_23:
    sub_1542080(v14, 2 * v13);
    v27 = a1[232];
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *((_QWORD *)a1 + 114);
      v30 = (v27 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v26 = a1[230] + 1;
      result = (__int64 *)(v29 + 16LL * v30);
      v31 = *result;
      if ( *result != a2 )
      {
        v32 = 1;
        v33 = 0;
        while ( v31 != -8 )
        {
          if ( v31 == -16 && !v33 )
            v33 = result;
          v30 = v28 & (v32 + v30);
          result = (__int64 *)(v29 + 16LL * v30);
          v31 = *result;
          if ( *result == a2 )
            goto LABEL_19;
          ++v32;
        }
        if ( v33 )
          result = v33;
      }
      goto LABEL_19;
    }
    goto LABEL_55;
  }
  if ( v13 - a1[231] - v26 <= v13 >> 3 )
  {
    sub_1542080(v14, v13);
    v34 = a1[232];
    if ( v34 )
    {
      v35 = v34 - 1;
      v36 = *((_QWORD *)a1 + 114);
      v37 = 0;
      v38 = (v34 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v39 = 1;
      v26 = a1[230] + 1;
      result = (__int64 *)(v36 + 16LL * v38);
      v40 = *result;
      if ( *result != a2 )
      {
        while ( v40 != -8 )
        {
          if ( v40 == -16 && !v37 )
            v37 = result;
          v38 = v35 & (v39 + v38);
          result = (__int64 *)(v36 + 16LL * v38);
          v40 = *result;
          if ( *result == a2 )
            goto LABEL_19;
          ++v39;
        }
        if ( v37 )
          result = v37;
      }
      goto LABEL_19;
    }
LABEL_55:
    ++a1[230];
    BUG();
  }
LABEL_19:
  a1[230] = v26;
  if ( *result != -8 )
    --a1[231];
  *result = a2;
  *((_DWORD *)result + 2) = 0;
LABEL_7:
  *((_DWORD *)result + 2) = a4;
  if ( *a1 >= a4 )
    a4 = *a1;
  *a1 = a4;
  return result;
}
