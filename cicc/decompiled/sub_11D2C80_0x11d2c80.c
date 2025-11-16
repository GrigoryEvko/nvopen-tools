// Function: sub_11D2C80
// Address: 0x11d2c80
//
__int64 __fastcall sub_11D2C80(__int64 *a1, __int64 a2, unsigned __int8 *a3, size_t a4)
{
  __int64 v7; // r13
  int v8; // eax
  unsigned int v9; // ecx
  __int64 v10; // rdx
  _QWORD *v11; // rax
  _QWORD *i; // rdx
  size_t v13; // rax
  __int64 *v14; // rdx
  __int64 result; // rax
  __int64 *v16; // rdi
  __int64 v17; // rdx
  size_t v18; // rcx
  __int64 v19; // rsi
  __int64 *v20; // rdi
  size_t v21; // rdx
  __int64 v22; // rax
  unsigned int v23; // eax
  _QWORD *v24; // rdi
  __int64 v25; // rax
  _QWORD *v26; // rax
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rdi
  _QWORD *v29; // rax
  __int64 v30; // rdx
  _QWORD *j; // rdx
  int v32; // [rsp+Ch] [rbp-64h]
  size_t v33; // [rsp+18h] [rbp-58h] BYREF
  __int64 *v34; // [rsp+20h] [rbp-50h] BYREF
  size_t n; // [rsp+28h] [rbp-48h]
  unsigned __int8 src[64]; // [rsp+30h] [rbp-40h] BYREF

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
        sub_C7D6A0(*(_QWORD *)(v7 + 8), 16LL * (unsigned int)v10, 8);
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
        *v11 = -4096;
      *(_QWORD *)(v7 + 16) = 0;
      goto LABEL_9;
    }
    v23 = v8 - 1;
    if ( v23 )
    {
      _BitScanReverse(&v23, v23);
      v24 = *(_QWORD **)(v7 + 8);
      v25 = (unsigned int)(1 << (33 - (v23 ^ 0x1F)));
      if ( (int)v25 < 64 )
        v25 = 64;
      if ( (_DWORD)v25 == (_DWORD)v10 )
      {
        *(_QWORD *)(v7 + 16) = 0;
        v26 = &v24[2 * v25];
        do
        {
          if ( v24 )
            *v24 = -4096;
          v24 += 2;
        }
        while ( v26 != v24 );
        goto LABEL_9;
      }
    }
    else
    {
      v24 = *(_QWORD **)(v7 + 8);
      LODWORD(v25) = 64;
    }
    v32 = v25;
    sub_C7D6A0((__int64)v24, 16LL * (unsigned int)v10, 8);
    v27 = ((((((((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
             | (4 * v32 / 3u + 1)
             | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 4)
           | (((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
           | (4 * v32 / 3u + 1)
           | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
           | (4 * v32 / 3u + 1)
           | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 4)
         | (((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
         | (4 * v32 / 3u + 1)
         | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 16;
    v28 = (v27
         | (((((((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
             | (4 * v32 / 3u + 1)
             | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 4)
           | (((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
           | (4 * v32 / 3u + 1)
           | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
           | (4 * v32 / 3u + 1)
           | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 4)
         | (((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
         | (4 * v32 / 3u + 1)
         | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(v7 + 24) = v28;
    v29 = (_QWORD *)sub_C7D670(16 * v28, 8);
    v30 = *(unsigned int *)(v7 + 24);
    *(_QWORD *)(v7 + 16) = 0;
    *(_QWORD *)(v7 + 8) = v29;
    for ( j = &v29[2 * v30]; j != v29; v29 += 2 )
    {
      if ( v29 )
        *v29 = -4096;
    }
  }
  else
  {
    v22 = sub_22077B0(32);
    if ( v22 )
    {
      *(_QWORD *)v22 = 0;
      *(_QWORD *)(v22 + 8) = 0;
      *(_QWORD *)(v22 + 16) = 0;
      *(_DWORD *)(v22 + 24) = 0;
    }
    *a1 = v22;
  }
LABEL_9:
  a1[1] = a2;
  v13 = a4;
  v34 = (__int64 *)src;
  if ( &a3[a4] && !a3 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v33 = a4;
  if ( a4 > 0xF )
  {
    v34 = (__int64 *)sub_22409D0(&v34, &v33, 0);
    v20 = v34;
    *(_QWORD *)src = v33;
  }
  else
  {
    if ( a4 == 1 )
    {
      src[0] = *a3;
      v14 = (__int64 *)src;
      goto LABEL_14;
    }
    if ( !a4 )
    {
      v14 = (__int64 *)src;
      goto LABEL_14;
    }
    v20 = (__int64 *)src;
  }
  memcpy(v20, a3, a4);
  v13 = v33;
  v14 = v34;
LABEL_14:
  n = v13;
  *((_BYTE *)v14 + v13) = 0;
  result = (__int64)v34;
  v16 = (__int64 *)a1[2];
  if ( v34 == (__int64 *)src )
  {
    v21 = n;
    if ( n )
    {
      if ( n == 1 )
      {
        result = src[0];
        *(_BYTE *)v16 = src[0];
      }
      else
      {
        result = (__int64)memcpy(v16, src, n);
      }
      v21 = n;
      v16 = (__int64 *)a1[2];
    }
    a1[3] = v21;
    *((_BYTE *)v16 + v21) = 0;
    v16 = v34;
  }
  else
  {
    v17 = *(_QWORD *)src;
    v18 = n;
    if ( v16 == a1 + 4 )
    {
      a1[2] = (__int64)v34;
      a1[3] = v18;
      a1[4] = v17;
    }
    else
    {
      v19 = a1[4];
      a1[2] = (__int64)v34;
      a1[3] = v18;
      a1[4] = v17;
      if ( v16 )
      {
        v34 = v16;
        *(_QWORD *)src = v19;
        goto LABEL_18;
      }
    }
    v34 = (__int64 *)src;
    v16 = (__int64 *)src;
  }
LABEL_18:
  n = 0;
  *(_BYTE *)v16 = 0;
  if ( v34 != (__int64 *)src )
    return j_j___libc_free_0(v34, *(_QWORD *)src + 1LL);
  return result;
}
