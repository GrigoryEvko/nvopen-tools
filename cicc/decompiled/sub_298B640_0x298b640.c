// Function: sub_298B640
// Address: 0x298b640
//
__int64 __fastcall sub_298B640(__int64 a1)
{
  int v2; // ecx
  __int64 result; // rax
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 v6; // r14
  __int64 v7; // r15
  __int64 v8; // r15
  unsigned __int64 v9; // r12
  unsigned __int64 v10; // rdi
  __int64 v11; // r15
  unsigned __int64 v12; // r8
  unsigned __int64 v13; // rdi
  __int64 v14; // r15
  __int64 v15; // rdx
  int v16; // ebx
  unsigned int v17; // ecx
  unsigned int v18; // eax
  _QWORD *v19; // rdi
  __int64 v20; // rcx
  __int64 i; // rdx
  unsigned __int64 v22; // [rsp+0h] [rbp-40h]
  int v23; // [rsp+Ch] [rbp-34h]
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
    if ( v6 != v4 )
    {
      do
      {
        result = *(_QWORD *)v4;
        v7 = v4 + 56;
        if ( *(_QWORD *)v4 != -4096 )
        {
          if ( result != -8192 )
          {
            v8 = *(_QWORD *)(v4 + 40);
            v9 = v8 + 56LL * *(unsigned int *)(v4 + 48);
            if ( v8 != v9 )
            {
              do
              {
                v9 -= 56LL;
                v10 = *(_QWORD *)(v9 + 8);
                if ( v10 != v9 + 24 )
                  _libc_free(v10);
              }
              while ( v8 != v9 );
              v9 = *(_QWORD *)(v4 + 40);
            }
            v7 = v4 + 56;
            if ( v9 != v4 + 56 )
              _libc_free(v9);
            result = sub_C7D6A0(*(_QWORD *)(v4 + 16), 16LL * *(unsigned int *)(v4 + 32), 8);
          }
          *(_QWORD *)v4 = -4096;
        }
        v4 = v7;
      }
      while ( v7 != v6 );
    }
    goto LABEL_19;
  }
  do
  {
    while ( 1 )
    {
      result = *(_QWORD *)v4;
      v14 = v4 + 56;
      if ( *(_QWORD *)v4 != -8192 )
        break;
LABEL_29:
      v4 = v14;
      if ( v14 == v6 )
        goto LABEL_33;
    }
    if ( result != -4096 )
    {
      v11 = *(_QWORD *)(v4 + 40);
      v12 = v11 + 56LL * *(unsigned int *)(v4 + 48);
      if ( v11 != v12 )
      {
        do
        {
          v12 -= 56LL;
          v13 = *(_QWORD *)(v12 + 8);
          if ( v13 != v12 + 24 )
          {
            v22 = v12;
            v23 = v2;
            _libc_free(v13);
            v12 = v22;
            v2 = v23;
          }
        }
        while ( v11 != v12 );
        v12 = *(_QWORD *)(v4 + 40);
      }
      v14 = v4 + 56;
      if ( v12 != v4 + 56 )
      {
        v24 = v2;
        _libc_free(v12);
        v2 = v24;
      }
      v25 = v2;
      result = sub_C7D6A0(*(_QWORD *)(v4 + 16), 16LL * *(unsigned int *)(v4 + 32), 8);
      v2 = v25;
      goto LABEL_29;
    }
    v4 += 56;
  }
  while ( v14 != v6 );
LABEL_33:
  v15 = *(unsigned int *)(a1 + 24);
  if ( !v2 )
  {
    if ( (_DWORD)v15 )
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 8), v5, 8);
      *(_DWORD *)(a1 + 24) = 0;
      goto LABEL_46;
    }
LABEL_19:
    *(_QWORD *)(a1 + 16) = 0;
    return result;
  }
  v16 = 64;
  v17 = v2 - 1;
  if ( v17 )
  {
    _BitScanReverse(&v18, v17);
    v16 = 1 << (33 - (v18 ^ 0x1F));
    if ( v16 < 64 )
      v16 = 64;
  }
  v19 = *(_QWORD **)(a1 + 8);
  if ( (_DWORD)v15 != v16 )
  {
    sub_C7D6A0((__int64)v19, v5, 8);
    result = sub_AF1560(4 * v16 / 3u + 1);
    *(_DWORD *)(a1 + 24) = result;
    if ( (_DWORD)result )
    {
      result = sub_C7D670(56LL * (unsigned int)result, 8);
      v20 = *(unsigned int *)(a1 + 24);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 8) = result;
      for ( i = result + 56 * v20; i != result; result += 56 )
      {
        if ( result )
          *(_QWORD *)result = -4096;
      }
      return result;
    }
LABEL_46:
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    return result;
  }
  *(_QWORD *)(a1 + 16) = 0;
  result = (__int64)&v19[7 * v15];
  do
  {
    if ( v19 )
      *v19 = -4096;
    v19 += 7;
  }
  while ( (_QWORD *)result != v19 );
  return result;
}
