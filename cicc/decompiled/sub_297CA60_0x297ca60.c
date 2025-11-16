// Function: sub_297CA60
// Address: 0x297ca60
//
__int64 __fastcall sub_297CA60(__int64 a1)
{
  int v2; // ecx
  __int64 result; // rax
  __int64 v4; // r12
  __int64 v5; // r15
  __int64 v6; // r14
  __int64 v7; // rax
  _QWORD *v8; // rbx
  _QWORD *v9; // r15
  unsigned __int64 v10; // rdi
  __int64 v11; // rax
  _QWORD *v12; // rbx
  _QWORD *v13; // rdx
  unsigned __int64 v14; // rdi
  int v15; // edx
  __int64 v16; // rbx
  unsigned int v17; // eax
  _QWORD *v18; // rdi
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rax
  __int64 v21; // rcx
  __int64 i; // rdx
  _QWORD *v23; // [rsp+0h] [rbp-40h]
  int v24; // [rsp+Ch] [rbp-34h]
  int v25; // [rsp+Ch] [rbp-34h]

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v2 )
  {
    result = *(unsigned int *)(a1 + 20);
    if ( !(_DWORD)result )
      return result;
  }
  v4 = *(_QWORD *)(a1 + 8);
  result = (unsigned int)(4 * v2);
  v5 = 56LL * *(unsigned int *)(a1 + 24);
  if ( (unsigned int)result < 0x40 )
    result = 64;
  v6 = v4 + v5;
  if ( *(_DWORD *)(a1 + 24) <= (unsigned int)result )
  {
    if ( v4 != v6 )
    {
      while ( 1 )
      {
        result = *(_QWORD *)(v4 + 16);
        if ( result == -4096 )
          break;
        if ( result != -8192 || *(_QWORD *)(v4 + 8) != -8192 || *(_QWORD *)v4 != -8192 )
          goto LABEL_9;
LABEL_18:
        *(_QWORD *)(v4 + 16) = -4096;
        *(_QWORD *)(v4 + 8) = -4096;
        *(_QWORD *)v4 = -4096;
LABEL_19:
        v4 += 56;
        if ( v4 == v6 )
          goto LABEL_20;
      }
      if ( *(_QWORD *)(v4 + 8) == -4096 && *(_QWORD *)v4 == -4096 )
        goto LABEL_19;
LABEL_9:
      v7 = *(unsigned int *)(v4 + 48);
      if ( (_DWORD)v7 )
      {
        v8 = *(_QWORD **)(v4 + 32);
        v9 = &v8[11 * v7];
        do
        {
          if ( *v8 != -8192 && *v8 != -4096 )
          {
            v10 = v8[1];
            if ( (_QWORD *)v10 != v8 + 3 )
              _libc_free(v10);
          }
          v8 += 11;
        }
        while ( v9 != v8 );
        v7 = *(unsigned int *)(v4 + 48);
      }
      result = sub_C7D6A0(*(_QWORD *)(v4 + 32), 88 * v7, 8);
      goto LABEL_18;
    }
LABEL_20:
    *(_QWORD *)(a1 + 16) = 0;
    return result;
  }
  do
  {
    result = *(_QWORD *)(v4 + 16);
    if ( result == -4096 )
    {
      if ( *(_QWORD *)(v4 + 8) == -4096 && *(_QWORD *)v4 == -4096 )
        goto LABEL_38;
    }
    else if ( result == -8192 && *(_QWORD *)(v4 + 8) == -8192 && *(_QWORD *)v4 == -8192 )
    {
      goto LABEL_38;
    }
    v11 = *(unsigned int *)(v4 + 48);
    if ( (_DWORD)v11 )
    {
      v12 = *(_QWORD **)(v4 + 32);
      v13 = &v12[11 * v11];
      do
      {
        if ( *v12 != -8192 && *v12 != -4096 )
        {
          v14 = v12[1];
          if ( (_QWORD *)v14 != v12 + 3 )
          {
            v23 = v13;
            v24 = v2;
            _libc_free(v14);
            v13 = v23;
            v2 = v24;
          }
        }
        v12 += 11;
      }
      while ( v13 != v12 );
      v11 = *(unsigned int *)(v4 + 48);
    }
    v25 = v2;
    result = sub_C7D6A0(*(_QWORD *)(v4 + 32), 88 * v11, 8);
    v2 = v25;
LABEL_38:
    v4 += 56;
  }
  while ( v4 != v6 );
  v15 = *(_DWORD *)(a1 + 24);
  if ( !v2 )
  {
    if ( v15 )
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 8), v5, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return result;
    }
    goto LABEL_20;
  }
  v16 = 64;
  if ( v2 != 1 )
  {
    _BitScanReverse(&v17, v2 - 1);
    v16 = (unsigned int)(1 << (33 - (v17 ^ 0x1F)));
    if ( (int)v16 < 64 )
      v16 = 64;
  }
  v18 = *(_QWORD **)(a1 + 8);
  if ( (_DWORD)v16 == v15 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    result = (__int64)&v18[7 * v16];
    do
    {
      if ( v18 )
      {
        *v18 = -4096;
        v18[1] = -4096;
        v18[2] = -4096;
      }
      v18 += 7;
    }
    while ( (_QWORD *)result != v18 );
  }
  else
  {
    sub_C7D6A0((__int64)v18, v5, 8);
    v19 = ((((((((4 * (int)v16 / 3u + 1) | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v16 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v16 / 3u + 1) | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v16 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v16 / 3u + 1) | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v16 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v16 / 3u + 1) | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v16 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 16;
    v20 = (v19
         | (((((((4 * (int)v16 / 3u + 1) | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v16 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v16 / 3u + 1) | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v16 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v16 / 3u + 1) | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v16 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v16 / 3u + 1) | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v16 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 24) = v20;
    result = sub_C7D670(56 * v20, 8);
    v21 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = result;
    for ( i = result + 56 * v21; i != result; result += 56 )
    {
      if ( result )
      {
        *(_QWORD *)result = -4096;
        *(_QWORD *)(result + 8) = -4096;
        *(_QWORD *)(result + 16) = -4096;
      }
    }
  }
  return result;
}
