// Function: sub_D9CC70
// Address: 0xd9cc70
//
__int64 __fastcall sub_D9CC70(__int64 a1)
{
  int v2; // r13d
  __int64 result; // rax
  __int64 v4; // r15
  __int64 v5; // rsi
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rdx
  int v15; // ebx
  unsigned int v16; // eax
  _QWORD *v17; // rdi
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 i; // rdx
  __int64 v22; // [rsp+10h] [rbp-40h]
  __int64 v23; // [rsp+18h] [rbp-38h]

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v2 )
  {
    result = *(unsigned int *)(a1 + 20);
    if ( !(_DWORD)result )
      return result;
  }
  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(unsigned int *)(a1 + 24);
  result = (unsigned int)(4 * v2);
  v6 = 168 * v5;
  if ( (unsigned int)result < 0x40 )
    result = 64;
  v23 = v4 + v6;
  if ( (unsigned int)v5 <= (unsigned int)result )
  {
    if ( v4 != v4 + v6 )
    {
      do
      {
        result = *(_QWORD *)v4;
        if ( *(_QWORD *)v4 != -4096 )
        {
          if ( result != -8192 )
          {
            v7 = *(_QWORD *)(v4 + 8);
            v8 = v7 + 112LL * *(unsigned int *)(v4 + 16);
            if ( v7 != v8 )
            {
              do
              {
                v8 -= 112;
                v9 = *(_QWORD *)(v8 + 64);
                if ( v9 != v8 + 80 )
                  _libc_free(v9, v5);
                if ( *(_BYTE *)(v8 + 32) )
                  *(_QWORD *)(v8 + 24) = 0;
                v10 = *(_QWORD *)(v8 + 24);
                *(_QWORD *)v8 = &unk_49DB368;
                LOBYTE(v5) = v10 != 0;
                if ( v10 != -4096 && v10 != 0 && v10 != -8192 )
                  sub_BD60C0((_QWORD *)(v8 + 8));
              }
              while ( v7 != v8 );
              v8 = *(_QWORD *)(v4 + 8);
            }
            result = v4 + 24;
            if ( v8 != v4 + 24 )
              result = _libc_free(v8, v5);
          }
          *(_QWORD *)v4 = -4096;
        }
        v4 += 168;
      }
      while ( v4 != v23 );
    }
LABEL_23:
    *(_QWORD *)(a1 + 16) = 0;
    return result;
  }
  do
  {
    while ( 1 )
    {
      result = *(_QWORD *)v4;
      if ( *(_QWORD *)v4 != -8192 )
        break;
LABEL_37:
      v4 += 168;
      if ( v4 == v23 )
        goto LABEL_41;
    }
    if ( result != -4096 )
    {
      v22 = *(_QWORD *)(v4 + 8);
      v11 = v22 + 112LL * *(unsigned int *)(v4 + 16);
      if ( v22 != v11 )
      {
        do
        {
          v11 -= 112;
          v12 = *(_QWORD *)(v11 + 64);
          if ( v12 != v11 + 80 )
            _libc_free(v12, v5);
          if ( *(_BYTE *)(v11 + 32) )
            *(_QWORD *)(v11 + 24) = 0;
          *(_QWORD *)v11 = &unk_49DB368;
          v13 = *(_QWORD *)(v11 + 24);
          LOBYTE(v5) = v13 != -4096;
          if ( ((unsigned __int8)v5 & (v13 != 0)) != 0 && v13 != -8192 )
            sub_BD60C0((_QWORD *)(v11 + 8));
        }
        while ( v22 != v11 );
        v11 = *(_QWORD *)(v4 + 8);
      }
      result = v4 + 24;
      if ( v11 != v4 + 24 )
        result = _libc_free(v11, v5);
      goto LABEL_37;
    }
    v4 += 168;
  }
  while ( v4 != v23 );
LABEL_41:
  v14 = *(unsigned int *)(a1 + 24);
  if ( !v2 )
  {
    if ( (_DWORD)v14 )
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 8), v6, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return result;
    }
    goto LABEL_23;
  }
  v15 = 64;
  if ( v2 != 1 )
  {
    _BitScanReverse(&v16, v2 - 1);
    v15 = 1 << (33 - (v16 ^ 0x1F));
    if ( v15 < 64 )
      v15 = 64;
  }
  v17 = *(_QWORD **)(a1 + 8);
  if ( (_DWORD)v14 == v15 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    result = (__int64)&v17[21 * v14];
    do
    {
      if ( v17 )
        *v17 = -4096;
      v17 += 21;
    }
    while ( (_QWORD *)result != v17 );
  }
  else
  {
    sub_C7D6A0((__int64)v17, v6, 8);
    v18 = (((((((4 * v15 / 3u + 1) | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 2)
            | (4 * v15 / 3u + 1)
            | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 4)
          | (((4 * v15 / 3u + 1) | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 2)
          | (4 * v15 / 3u + 1)
          | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 8)
        | (((((4 * v15 / 3u + 1) | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 2)
          | (4 * v15 / 3u + 1)
          | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 4)
        | (((4 * v15 / 3u + 1) | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 2)
        | (4 * v15 / 3u + 1)
        | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1);
    v19 = ((v18 >> 16) | v18) + 1;
    *(_DWORD *)(a1 + 24) = v19;
    result = sub_C7D670(168 * v19, 8);
    v20 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = result;
    for ( i = result + 168 * v20; i != result; result += 168 )
    {
      if ( result )
        *(_QWORD *)result = -4096;
    }
  }
  return result;
}
