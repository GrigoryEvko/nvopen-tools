// Function: sub_2C9CD90
// Address: 0x2c9cd90
//
__int64 __fastcall sub_2C9CD90(__int64 a1, __int64 a2)
{
  int v3; // eax
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 i; // rdx
  __int64 v7; // r12
  __int64 v8; // rcx
  int v9; // r14d
  __int64 v10; // r9
  unsigned int v11; // r8d
  __int64 *v12; // rax
  __int64 v13; // rdi
  unsigned int v14; // esi
  __int64 v15; // rbx
  int v16; // esi
  int v17; // esi
  __int64 v18; // r9
  unsigned int v19; // edi
  int v20; // eax
  __int64 *v21; // rdx
  __int64 v22; // r8
  int v23; // eax
  int v24; // esi
  int v25; // esi
  __int64 v26; // r8
  __int64 *v27; // r9
  unsigned int v28; // r15d
  int v29; // r10d
  __int64 v30; // rdi
  unsigned int v31; // ecx
  unsigned int v32; // eax
  _QWORD *v33; // rdi
  int v34; // r12d
  unsigned __int64 v35; // rax
  unsigned __int64 v36; // rdi
  __int64 v37; // rdx
  __int64 j; // rdx
  int v39; // r15d
  __int64 *v40; // r10
  __int64 v41; // [rsp+8h] [rbp-38h]
  int v42; // [rsp+8h] [rbp-38h]
  __int64 v43; // [rsp+8h] [rbp-38h]

  v3 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v3 )
  {
    result = *(unsigned int *)(a1 + 20);
    if ( !(_DWORD)result )
      goto LABEL_7;
    v5 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v5 > 0x40 )
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 8), 16LL * (unsigned int)v5, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v31 = 4 * v3;
  v5 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v3) < 0x40 )
    v31 = 64;
  if ( (unsigned int)v5 <= v31 )
  {
LABEL_4:
    result = *(_QWORD *)(a1 + 8);
    for ( i = result + 16 * v5; i != result; result += 16 )
      *(_QWORD *)result = -4096;
    *(_QWORD *)(a1 + 16) = 0;
    goto LABEL_7;
  }
  v32 = v3 - 1;
  if ( v32 )
  {
    _BitScanReverse(&v32, v32);
    v33 = *(_QWORD **)(a1 + 8);
    v34 = 1 << (33 - (v32 ^ 0x1F));
    if ( v34 < 64 )
      v34 = 64;
    if ( (_DWORD)v5 == v34 )
    {
      *(_QWORD *)(a1 + 16) = 0;
      result = (__int64)&v33[2 * (unsigned int)v5];
      do
      {
        if ( v33 )
          *v33 = -4096;
        v33 += 2;
      }
      while ( (_QWORD *)result != v33 );
      goto LABEL_7;
    }
  }
  else
  {
    v33 = *(_QWORD **)(a1 + 8);
    v34 = 64;
  }
  sub_C7D6A0((__int64)v33, 16LL * (unsigned int)v5, 8);
  v35 = ((((((((4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 2)
           | (4 * v34 / 3u + 1)
           | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 4)
         | (((4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 2)
         | (4 * v34 / 3u + 1)
         | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 2)
         | (4 * v34 / 3u + 1)
         | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 4)
       | (((4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 2)
       | (4 * v34 / 3u + 1)
       | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 16;
  v36 = (v35
       | (((((((4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 2)
           | (4 * v34 / 3u + 1)
           | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 4)
         | (((4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 2)
         | (4 * v34 / 3u + 1)
         | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 2)
         | (4 * v34 / 3u + 1)
         | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 4)
       | (((4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 2)
       | (4 * v34 / 3u + 1)
       | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 24) = v36;
  result = sub_C7D670(16 * v36, 8);
  v37 = *(unsigned int *)(a1 + 24);
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = result;
  for ( j = result + 16 * v37; j != result; result += 16 )
  {
    if ( result )
      *(_QWORD *)result = -4096;
  }
LABEL_7:
  v7 = *(_QWORD *)(a2 + 56);
  v8 = a2 + 48;
  v9 = 0;
  if ( v7 != a2 + 48 )
  {
    while ( 1 )
    {
      v14 = *(_DWORD *)(a1 + 24);
      v15 = v7 - 24;
      if ( !v7 )
        v15 = 0;
      ++v9;
      if ( !v14 )
        break;
      v10 = *(_QWORD *)(a1 + 8);
      v11 = (v14 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v12 = (__int64 *)(v10 + 16LL * v11);
      v13 = *v12;
      if ( v15 == *v12 )
      {
LABEL_10:
        *((_DWORD *)v12 + 2) = v9;
        v7 = *(_QWORD *)(v7 + 8);
        result = (__int64)(v12 + 1);
        if ( v8 == v7 )
          return result;
      }
      else
      {
        v42 = 1;
        v21 = 0;
        while ( v13 != -4096 )
        {
          if ( v13 == -8192 && !v21 )
            v21 = v12;
          v11 = (v14 - 1) & (v42 + v11);
          v12 = (__int64 *)(v10 + 16LL * v11);
          v13 = *v12;
          if ( v15 == *v12 )
            goto LABEL_10;
          ++v42;
        }
        if ( !v21 )
          v21 = v12;
        v23 = *(_DWORD *)(a1 + 16);
        ++*(_QWORD *)a1;
        v20 = v23 + 1;
        if ( 4 * v20 < 3 * v14 )
        {
          if ( v14 - *(_DWORD *)(a1 + 20) - v20 <= v14 >> 3 )
          {
            v43 = v8;
            sub_9BAAD0(a1, v14);
            v24 = *(_DWORD *)(a1 + 24);
            if ( !v24 )
            {
LABEL_72:
              ++*(_DWORD *)(a1 + 16);
              BUG();
            }
            v25 = v24 - 1;
            v26 = *(_QWORD *)(a1 + 8);
            v27 = 0;
            v28 = v25 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
            v8 = v43;
            v29 = 1;
            v20 = *(_DWORD *)(a1 + 16) + 1;
            v21 = (__int64 *)(v26 + 16LL * v28);
            v30 = *v21;
            if ( v15 != *v21 )
            {
              while ( v30 != -4096 )
              {
                if ( v30 == -8192 && !v27 )
                  v27 = v21;
                v28 = v25 & (v29 + v28);
                v21 = (__int64 *)(v26 + 16LL * v28);
                v30 = *v21;
                if ( v15 == *v21 )
                  goto LABEL_17;
                ++v29;
              }
              if ( v27 )
                v21 = v27;
            }
          }
          goto LABEL_17;
        }
LABEL_15:
        v41 = v8;
        sub_9BAAD0(a1, 2 * v14);
        v16 = *(_DWORD *)(a1 + 24);
        if ( !v16 )
          goto LABEL_72;
        v17 = v16 - 1;
        v18 = *(_QWORD *)(a1 + 8);
        v8 = v41;
        v19 = v17 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
        v20 = *(_DWORD *)(a1 + 16) + 1;
        v21 = (__int64 *)(v18 + 16LL * v19);
        v22 = *v21;
        if ( v15 != *v21 )
        {
          v39 = 1;
          v40 = 0;
          while ( v22 != -4096 )
          {
            if ( v22 == -8192 && !v40 )
              v40 = v21;
            v19 = v17 & (v39 + v19);
            v21 = (__int64 *)(v18 + 16LL * v19);
            v22 = *v21;
            if ( v15 == *v21 )
              goto LABEL_17;
            ++v39;
          }
          if ( v40 )
            v21 = v40;
        }
LABEL_17:
        *(_DWORD *)(a1 + 16) = v20;
        if ( *v21 != -4096 )
          --*(_DWORD *)(a1 + 20);
        result = (__int64)(v21 + 1);
        *v21 = v15;
        *((_DWORD *)v21 + 2) = 0;
        *((_DWORD *)v21 + 2) = v9;
        v7 = *(_QWORD *)(v7 + 8);
        if ( v8 == v7 )
          return result;
      }
    }
    ++*(_QWORD *)a1;
    goto LABEL_15;
  }
  return result;
}
