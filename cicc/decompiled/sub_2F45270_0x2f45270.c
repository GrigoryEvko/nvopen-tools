// Function: sub_2F45270
// Address: 0x2f45270
//
__int64 __fastcall sub_2F45270(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  int v4; // eax
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 i; // rdx
  __int64 v8; // rbx
  __int64 v9; // r15
  __int64 v10; // r13
  unsigned int v11; // esi
  __int64 v12; // r9
  int v13; // r11d
  _QWORD *v14; // rdx
  unsigned int v15; // edi
  _QWORD *v16; // rax
  __int64 v17; // rcx
  int v18; // eax
  int v19; // ecx
  unsigned int v20; // ecx
  unsigned int v21; // eax
  _QWORD *v22; // rdi
  int v23; // ebx
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rax
  __int64 v26; // rdx
  __int64 j; // rdx
  int v28; // eax
  int v29; // edi
  __int64 v30; // rsi
  unsigned int v31; // eax
  __int64 v32; // r9
  int v33; // r11d
  _QWORD *v34; // r10
  int v35; // eax
  int v36; // eax
  __int64 v37; // rdi
  _QWORD *v38; // r9
  unsigned int v39; // r14d
  int v40; // r10d
  __int64 v41; // rsi
  __int64 v42; // [rsp+8h] [rbp-38h]
  __int64 v43; // [rsp+8h] [rbp-38h]
  __int64 v44; // [rsp+8h] [rbp-38h]

  v2 = a1 + 16;
  v4 = *(_DWORD *)(a1 + 32);
  ++*(_QWORD *)(a1 + 16);
  *(_QWORD *)(a1 + 8) = a2;
  if ( !v4 )
  {
    result = *(unsigned int *)(a1 + 36);
    if ( !(_DWORD)result )
      goto LABEL_7;
    v6 = *(unsigned int *)(a1 + 40);
    if ( (unsigned int)v6 > 0x40 )
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 24), 16 * v6, 8);
      v2 = a1 + 16;
      *(_QWORD *)(a1 + 24) = 0;
      *(_QWORD *)(a1 + 32) = 0;
      *(_DWORD *)(a1 + 40) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v20 = 4 * v4;
  v6 = *(unsigned int *)(a1 + 40);
  if ( (unsigned int)(4 * v4) < 0x40 )
    v20 = 64;
  if ( (unsigned int)v6 <= v20 )
  {
LABEL_4:
    result = *(_QWORD *)(a1 + 24);
    for ( i = result + 16 * v6; i != result; result += 16 )
      *(_QWORD *)result = -4096;
    *(_QWORD *)(a1 + 32) = 0;
    goto LABEL_7;
  }
  v21 = v4 - 1;
  if ( !v21 )
  {
    v22 = *(_QWORD **)(a1 + 24);
    v23 = 64;
LABEL_40:
    v42 = v2;
    sub_C7D6A0((__int64)v22, 16 * v6, 8);
    v24 = ((((((((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
             | (4 * v23 / 3u + 1)
             | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 4)
           | (((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
           | (4 * v23 / 3u + 1)
           | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
           | (4 * v23 / 3u + 1)
           | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 4)
         | (((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
         | (4 * v23 / 3u + 1)
         | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 16;
    v25 = (v24
         | (((((((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
             | (4 * v23 / 3u + 1)
             | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 4)
           | (((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
           | (4 * v23 / 3u + 1)
           | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
           | (4 * v23 / 3u + 1)
           | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 4)
         | (((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
         | (4 * v23 / 3u + 1)
         | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 40) = v25;
    result = sub_C7D670(16 * v25, 8);
    v26 = *(unsigned int *)(a1 + 40);
    v2 = v42;
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 24) = result;
    for ( j = result + 16 * v26; j != result; result += 16 )
    {
      if ( result )
        *(_QWORD *)result = -4096;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v21, v21);
  v22 = *(_QWORD **)(a1 + 24);
  v23 = 1 << (33 - (v21 ^ 0x1F));
  if ( v23 < 64 )
    v23 = 64;
  if ( v23 != (_DWORD)v6 )
    goto LABEL_40;
  *(_QWORD *)(a1 + 32) = 0;
  result = (__int64)&v22[2 * (unsigned int)v23];
  do
  {
    if ( v22 )
      *v22 = -4096;
    v22 += 2;
  }
  while ( (_QWORD *)result != v22 );
LABEL_7:
  v8 = *(_QWORD *)(a2 + 56);
  v9 = a2 + 48;
  v10 = 0;
  if ( v8 != a2 + 48 )
  {
    while ( 1 )
    {
      v11 = *(_DWORD *)(a1 + 40);
      v10 += 1024;
      if ( !v11 )
        break;
      v12 = *(_QWORD *)(a1 + 24);
      v13 = 1;
      v14 = 0;
      v15 = (v11 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v16 = (_QWORD *)(v12 + 16LL * v15);
      v17 = *v16;
      if ( v8 != *v16 )
      {
        while ( v17 != -4096 )
        {
          if ( v17 == -8192 && !v14 )
            v14 = v16;
          v15 = (v11 - 1) & (v13 + v15);
          v16 = (_QWORD *)(v12 + 16LL * v15);
          v17 = *v16;
          if ( v8 == *v16 )
            goto LABEL_12;
          ++v13;
        }
        if ( !v14 )
          v14 = v16;
        v18 = *(_DWORD *)(a1 + 32);
        ++*(_QWORD *)(a1 + 16);
        v19 = v18 + 1;
        if ( 4 * (v18 + 1) < 3 * v11 )
        {
          if ( v11 - *(_DWORD *)(a1 + 36) - v19 <= v11 >> 3 )
          {
            v44 = v2;
            sub_2F45090(v2, v11);
            v35 = *(_DWORD *)(a1 + 40);
            if ( !v35 )
            {
LABEL_77:
              ++*(_DWORD *)(a1 + 32);
              BUG();
            }
            v36 = v35 - 1;
            v37 = *(_QWORD *)(a1 + 24);
            v38 = 0;
            v39 = v36 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
            v2 = v44;
            v40 = 1;
            v19 = *(_DWORD *)(a1 + 32) + 1;
            v14 = (_QWORD *)(v37 + 16LL * v39);
            v41 = *v14;
            if ( v8 != *v14 )
            {
              while ( v41 != -4096 )
              {
                if ( v41 == -8192 && !v38 )
                  v38 = v14;
                v39 = v36 & (v40 + v39);
                v14 = (_QWORD *)(v37 + 16LL * v39);
                v41 = *v14;
                if ( v8 == *v14 )
                  goto LABEL_30;
                ++v40;
              }
              if ( v38 )
                v14 = v38;
            }
          }
          goto LABEL_30;
        }
LABEL_46:
        v43 = v2;
        sub_2F45090(v2, 2 * v11);
        v28 = *(_DWORD *)(a1 + 40);
        if ( !v28 )
          goto LABEL_77;
        v29 = v28 - 1;
        v30 = *(_QWORD *)(a1 + 24);
        v2 = v43;
        v31 = (v28 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v19 = *(_DWORD *)(a1 + 32) + 1;
        v14 = (_QWORD *)(v30 + 16LL * v31);
        v32 = *v14;
        if ( v8 != *v14 )
        {
          v33 = 1;
          v34 = 0;
          while ( v32 != -4096 )
          {
            if ( !v34 && v32 == -8192 )
              v34 = v14;
            v31 = v29 & (v33 + v31);
            v14 = (_QWORD *)(v30 + 16LL * v31);
            v32 = *v14;
            if ( v8 == *v14 )
              goto LABEL_30;
            ++v33;
          }
          if ( v34 )
            v14 = v34;
        }
LABEL_30:
        *(_DWORD *)(a1 + 32) = v19;
        if ( *v14 != -4096 )
          --*(_DWORD *)(a1 + 36);
        *v14 = v8;
        result = (__int64)(v14 + 1);
        v14[1] = 0;
        goto LABEL_13;
      }
LABEL_12:
      result = (__int64)(v16 + 1);
LABEL_13:
      *(_QWORD *)result = v10;
      if ( !v8 )
        BUG();
      if ( (*(_BYTE *)v8 & 4) != 0 )
      {
        v8 = *(_QWORD *)(v8 + 8);
        if ( v9 == v8 )
          return result;
      }
      else
      {
        while ( (*(_BYTE *)(v8 + 44) & 8) != 0 )
          v8 = *(_QWORD *)(v8 + 8);
        v8 = *(_QWORD *)(v8 + 8);
        if ( v9 == v8 )
          return result;
      }
    }
    ++*(_QWORD *)(a1 + 16);
    goto LABEL_46;
  }
  return result;
}
