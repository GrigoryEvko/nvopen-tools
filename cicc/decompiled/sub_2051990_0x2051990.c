// Function: sub_2051990
// Address: 0x2051990
//
unsigned __int64 __fastcall sub_2051990(__int64 a1)
{
  int v2; // edx
  unsigned __int64 result; // rax
  unsigned __int64 *v4; // r15
  __int64 v5; // rcx
  unsigned __int64 *v6; // r13
  unsigned __int64 v7; // rbx
  unsigned __int64 v8; // r12
  __int64 v9; // rsi
  unsigned __int64 v10; // rbx
  unsigned __int64 v11; // r12
  __int64 v12; // rsi
  int v13; // esi
  int v14; // ebx
  unsigned int v15; // edx
  unsigned int v16; // eax
  _QWORD *v17; // rdi
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rdi
  __int64 v20; // rdx
  unsigned __int64 i; // rdx
  int v22; // [rsp+Ch] [rbp-34h]
  int v23; // [rsp+Ch] [rbp-34h]

  v2 = *(_DWORD *)(a1 + 88);
  ++*(_QWORD *)(a1 + 72);
  if ( !v2 )
  {
    result = *(unsigned int *)(a1 + 92);
    if ( !(_DWORD)result )
      return result;
  }
  v4 = *(unsigned __int64 **)(a1 + 80);
  result = (unsigned int)(4 * v2);
  v5 = *(unsigned int *)(a1 + 96);
  v6 = &v4[4 * v5];
  if ( (unsigned int)result < 0x40 )
    result = 64;
  if ( (unsigned int)v5 <= (unsigned int)result )
  {
    for ( ; v4 != v6; v4 += 4 )
    {
      result = *v4;
      if ( *v4 != -8 )
      {
        if ( result != -16 )
        {
          v7 = v4[2];
          v8 = v4[1];
          if ( v7 != v8 )
          {
            do
            {
              v9 = *(_QWORD *)(v8 + 8);
              if ( v9 )
                result = sub_161E7C0(v8 + 8, v9);
              v8 += 24LL;
            }
            while ( v7 != v8 );
            v8 = v4[1];
          }
          if ( v8 )
            result = j_j___libc_free_0(v8, v4[3] - v8);
        }
        *v4 = -8;
      }
    }
LABEL_18:
    *(_QWORD *)(a1 + 88) = 0;
    return result;
  }
  do
  {
    while ( 1 )
    {
      result = *v4;
      if ( *v4 != -16 )
        break;
LABEL_27:
      v4 += 4;
      if ( v6 == v4 )
        goto LABEL_31;
    }
    if ( result != -8 )
    {
      v10 = v4[2];
      v11 = v4[1];
      if ( v10 != v11 )
      {
        do
        {
          v12 = *(_QWORD *)(v11 + 8);
          if ( v12 )
          {
            v22 = v2;
            result = sub_161E7C0(v11 + 8, v12);
            v2 = v22;
          }
          v11 += 24LL;
        }
        while ( v10 != v11 );
        v11 = v4[1];
      }
      if ( v11 )
      {
        v23 = v2;
        result = j_j___libc_free_0(v11, v4[3] - v11);
        v2 = v23;
      }
      goto LABEL_27;
    }
    v4 += 4;
  }
  while ( v6 != v4 );
LABEL_31:
  v13 = *(_DWORD *)(a1 + 96);
  if ( !v2 )
  {
    if ( v13 )
    {
      result = j___libc_free_0(*(_QWORD *)(a1 + 80));
      *(_QWORD *)(a1 + 80) = 0;
      *(_QWORD *)(a1 + 88) = 0;
      *(_DWORD *)(a1 + 96) = 0;
      return result;
    }
    goto LABEL_18;
  }
  v14 = 64;
  v15 = v2 - 1;
  if ( v15 )
  {
    _BitScanReverse(&v16, v15);
    v14 = 1 << (33 - (v16 ^ 0x1F));
    if ( v14 < 64 )
      v14 = 64;
  }
  v17 = *(_QWORD **)(a1 + 80);
  if ( v14 == v13 )
  {
    *(_QWORD *)(a1 + 88) = 0;
    result = (unsigned __int64)&v17[4 * (unsigned int)v14];
    do
    {
      if ( v17 )
        *v17 = -8;
      v17 += 4;
    }
    while ( (_QWORD *)result != v17 );
  }
  else
  {
    j___libc_free_0(v17);
    v18 = ((((((((4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 2)
             | (4 * v14 / 3u + 1)
             | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 4)
           | (((4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 2)
           | (4 * v14 / 3u + 1)
           | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 2)
           | (4 * v14 / 3u + 1)
           | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 4)
         | (((4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 2)
         | (4 * v14 / 3u + 1)
         | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 16;
    v19 = (v18
         | (((((((4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 2)
             | (4 * v14 / 3u + 1)
             | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 4)
           | (((4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 2)
           | (4 * v14 / 3u + 1)
           | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 2)
           | (4 * v14 / 3u + 1)
           | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 4)
         | (((4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 2)
         | (4 * v14 / 3u + 1)
         | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 96) = v19;
    result = sub_22077B0(32 * v19);
    v20 = *(unsigned int *)(a1 + 96);
    *(_QWORD *)(a1 + 88) = 0;
    *(_QWORD *)(a1 + 80) = result;
    for ( i = result + 32 * v20; i != result; result += 32LL )
    {
      if ( result )
        *(_QWORD *)result = -8;
    }
  }
  return result;
}
