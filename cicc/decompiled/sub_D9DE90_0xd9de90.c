// Function: sub_D9DE90
// Address: 0xd9de90
//
__int64 __fastcall sub_D9DE90(__int64 a1, __int64 a2)
{
  int v3; // r15d
  __int64 result; // rax
  __int64 v5; // rbx
  __int64 v6; // r14
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 v10; // rdx
  int v11; // ebx
  unsigned int v12; // eax
  _QWORD *v13; // rdi
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 i; // rdx

  v3 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v3 )
  {
    result = *(unsigned int *)(a1 + 20);
    if ( !(_DWORD)result )
      return result;
  }
  v5 = *(_QWORD *)(a1 + 8);
  result = (unsigned int)(4 * v3);
  v6 = 88LL * *(unsigned int *)(a1 + 24);
  if ( (unsigned int)result < 0x40 )
    result = 64;
  v7 = v5 + v6;
  if ( *(_DWORD *)(a1 + 24) <= (unsigned int)result )
  {
    for ( ; v5 != v7; v5 += 88 )
    {
      result = *(_QWORD *)v5;
      if ( *(_QWORD *)v5 != -4096 )
      {
        if ( result != -8192 )
        {
          v8 = *(_QWORD *)(v5 + 40);
          if ( v8 != v5 + 56 )
            _libc_free(v8, a2);
          a2 = 8LL * *(unsigned int *)(v5 + 32);
          result = sub_C7D6A0(*(_QWORD *)(v5 + 16), a2, 8);
        }
        *(_QWORD *)v5 = -4096;
      }
    }
LABEL_14:
    *(_QWORD *)(a1 + 16) = 0;
    return result;
  }
  do
  {
    while ( 1 )
    {
      result = *(_QWORD *)v5;
      if ( *(_QWORD *)v5 != -8192 )
        break;
LABEL_19:
      v5 += 88;
      if ( v7 == v5 )
        goto LABEL_23;
    }
    if ( result != -4096 )
    {
      v9 = *(_QWORD *)(v5 + 40);
      if ( v9 != v5 + 56 )
        _libc_free(v9, a2);
      a2 = 8LL * *(unsigned int *)(v5 + 32);
      result = sub_C7D6A0(*(_QWORD *)(v5 + 16), a2, 8);
      goto LABEL_19;
    }
    v5 += 88;
  }
  while ( v7 != v5 );
LABEL_23:
  v10 = *(unsigned int *)(a1 + 24);
  if ( !v3 )
  {
    if ( (_DWORD)v10 )
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 8), v6, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return result;
    }
    goto LABEL_14;
  }
  v11 = 64;
  if ( v3 != 1 )
  {
    _BitScanReverse(&v12, v3 - 1);
    v11 = 1 << (33 - (v12 ^ 0x1F));
    if ( v11 < 64 )
      v11 = 64;
  }
  v13 = *(_QWORD **)(a1 + 8);
  if ( (_DWORD)v10 == v11 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    result = (__int64)&v13[11 * v10];
    do
    {
      if ( v13 )
        *v13 = -4096;
      v13 += 11;
    }
    while ( (_QWORD *)result != v13 );
  }
  else
  {
    sub_C7D6A0((__int64)v13, v6, 8);
    v14 = (((((((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
            | (4 * v11 / 3u + 1)
            | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 4)
          | (((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
          | (4 * v11 / 3u + 1)
          | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 8)
        | (((((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
          | (4 * v11 / 3u + 1)
          | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 4)
        | (((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
        | (4 * v11 / 3u + 1)
        | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1);
    v15 = ((v14 >> 16) | v14) + 1;
    *(_DWORD *)(a1 + 24) = v15;
    result = sub_C7D670(88 * v15, 8);
    v16 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = result;
    for ( i = result + 88 * v16; i != result; result += 88 )
    {
      if ( result )
        *(_QWORD *)result = -4096;
    }
  }
  return result;
}
