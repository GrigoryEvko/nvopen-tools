// Function: sub_372BA50
// Address: 0x372ba50
//
_DWORD *__fastcall sub_372BA50(__int64 a1, __int64 a2)
{
  int v3; // eax
  __int64 v4; // rdx
  _QWORD *v5; // rax
  _QWORD *i; // rdx
  __int64 v7; // r9
  _DWORD *result; // rax
  int v9; // r14d
  __int64 v10; // rbx
  __int64 v11; // r15
  unsigned int v12; // esi
  __int64 v13; // r8
  int v14; // r11d
  _QWORD *v15; // rdx
  unsigned int v16; // edi
  _QWORD *v17; // rax
  __int64 v18; // rcx
  int v19; // eax
  int v20; // ecx
  int v21; // eax
  int v22; // edi
  __int64 v23; // rsi
  unsigned int v24; // eax
  __int64 v25; // r8
  int v26; // r11d
  _QWORD *v27; // r10
  int v28; // eax
  int v29; // eax
  __int64 v30; // rdi
  _QWORD *v31; // r8
  unsigned int v32; // r13d
  int v33; // r10d
  __int64 v34; // rsi
  unsigned int v35; // ecx
  unsigned int v36; // eax
  _QWORD *v37; // rdi
  int v38; // r13d
  unsigned __int64 v39; // rdx
  unsigned __int64 v40; // rax
  _QWORD *v41; // rax
  __int64 v42; // rdx
  _QWORD *j; // rdx
  _QWORD *v44; // rax
  __int64 v45; // [rsp+0h] [rbp-40h]
  __int64 v46; // [rsp+0h] [rbp-40h]
  __int64 v47; // [rsp+8h] [rbp-38h]

  v3 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v3 )
  {
    if ( !*(_DWORD *)(a1 + 20) )
      goto LABEL_7;
    v4 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v4 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 8), 16 * v4, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v35 = 4 * v3;
  v4 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v3) < 0x40 )
    v35 = 64;
  if ( (unsigned int)v4 <= v35 )
  {
LABEL_4:
    v5 = *(_QWORD **)(a1 + 8);
    for ( i = &v5[2 * v4]; i != v5; v5 += 2 )
      *v5 = -4096;
    *(_QWORD *)(a1 + 16) = 0;
    goto LABEL_7;
  }
  v36 = v3 - 1;
  if ( !v36 )
  {
    v37 = *(_QWORD **)(a1 + 8);
    v38 = 64;
LABEL_55:
    sub_C7D6A0((__int64)v37, 16 * v4, 8);
    v39 = ((((((((4 * v38 / 3u + 1) | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 2)
             | (4 * v38 / 3u + 1)
             | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 4)
           | (((4 * v38 / 3u + 1) | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 2)
           | (4 * v38 / 3u + 1)
           | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v38 / 3u + 1) | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 2)
           | (4 * v38 / 3u + 1)
           | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 4)
         | (((4 * v38 / 3u + 1) | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 2)
         | (4 * v38 / 3u + 1)
         | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 16;
    v40 = (v39
         | (((((((4 * v38 / 3u + 1) | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 2)
             | (4 * v38 / 3u + 1)
             | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 4)
           | (((4 * v38 / 3u + 1) | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 2)
           | (4 * v38 / 3u + 1)
           | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v38 / 3u + 1) | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 2)
           | (4 * v38 / 3u + 1)
           | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 4)
         | (((4 * v38 / 3u + 1) | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1)) >> 2)
         | (4 * v38 / 3u + 1)
         | ((unsigned __int64)(4 * v38 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 24) = v40;
    v41 = (_QWORD *)sub_C7D670(16 * v40, 8);
    v42 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = v41;
    for ( j = &v41[2 * v42]; j != v41; v41 += 2 )
    {
      if ( v41 )
        *v41 = -4096;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v36, v36);
  v37 = *(_QWORD **)(a1 + 8);
  v38 = 1 << (33 - (v36 ^ 0x1F));
  if ( v38 < 64 )
    v38 = 64;
  if ( (_DWORD)v4 != v38 )
    goto LABEL_55;
  *(_QWORD *)(a1 + 16) = 0;
  v44 = &v37[2 * (unsigned int)v4];
  do
  {
    if ( v37 )
      *v37 = -4096;
    v37 += 2;
  }
  while ( v44 != v37 );
LABEL_7:
  v7 = *(_QWORD *)(a2 + 328);
  result = (_DWORD *)(a2 + 320);
  v9 = 0;
  v47 = a2 + 320;
  if ( v7 != a2 + 320 )
  {
    while ( 1 )
    {
      v10 = *(_QWORD *)(v7 + 56);
      v11 = v7 + 48;
      if ( v7 + 48 != v10 )
        break;
LABEL_16:
      v7 = *(_QWORD *)(v7 + 8);
      if ( v47 == v7 )
        return result;
    }
    while ( 1 )
    {
      v12 = *(_DWORD *)(a1 + 24);
      v9 += (*(_QWORD *)(*(_QWORD *)(v10 + 16) + 24LL) & 0x10LL) == 0;
      if ( !v12 )
        break;
      v13 = *(_QWORD *)(a1 + 8);
      v14 = 1;
      v15 = 0;
      v16 = (v12 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v17 = (_QWORD *)(v13 + 16LL * v16);
      v18 = *v17;
      if ( v10 != *v17 )
      {
        while ( v18 != -4096 )
        {
          if ( v18 == -8192 && !v15 )
            v15 = v17;
          v16 = (v12 - 1) & (v14 + v16);
          v17 = (_QWORD *)(v13 + 16LL * v16);
          v18 = *v17;
          if ( v10 == *v17 )
            goto LABEL_13;
          ++v14;
        }
        if ( !v15 )
          v15 = v17;
        v19 = *(_DWORD *)(a1 + 16);
        ++*(_QWORD *)a1;
        v20 = v19 + 1;
        if ( 4 * (v19 + 1) < 3 * v12 )
        {
          if ( v12 - *(_DWORD *)(a1 + 20) - v20 <= v12 >> 3 )
          {
            v46 = v7;
            sub_2EECB30(a1, v12);
            v28 = *(_DWORD *)(a1 + 24);
            if ( !v28 )
            {
LABEL_77:
              ++*(_DWORD *)(a1 + 16);
              BUG();
            }
            v29 = v28 - 1;
            v30 = *(_QWORD *)(a1 + 8);
            v31 = 0;
            v32 = v29 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
            v7 = v46;
            v33 = 1;
            v20 = *(_DWORD *)(a1 + 16) + 1;
            v15 = (_QWORD *)(v30 + 16LL * v32);
            v34 = *v15;
            if ( v10 != *v15 )
            {
              while ( v34 != -4096 )
              {
                if ( v34 == -8192 && !v31 )
                  v31 = v15;
                v32 = v29 & (v33 + v32);
                v15 = (_QWORD *)(v30 + 16LL * v32);
                v34 = *v15;
                if ( v10 == *v15 )
                  goto LABEL_31;
                ++v33;
              }
              if ( v31 )
                v15 = v31;
            }
          }
          goto LABEL_31;
        }
LABEL_35:
        v45 = v7;
        sub_2EECB30(a1, 2 * v12);
        v21 = *(_DWORD *)(a1 + 24);
        if ( !v21 )
          goto LABEL_77;
        v22 = v21 - 1;
        v23 = *(_QWORD *)(a1 + 8);
        v7 = v45;
        v24 = (v21 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v20 = *(_DWORD *)(a1 + 16) + 1;
        v15 = (_QWORD *)(v23 + 16LL * v24);
        v25 = *v15;
        if ( v10 != *v15 )
        {
          v26 = 1;
          v27 = 0;
          while ( v25 != -4096 )
          {
            if ( v25 == -8192 && !v27 )
              v27 = v15;
            v24 = v22 & (v26 + v24);
            v15 = (_QWORD *)(v23 + 16LL * v24);
            v25 = *v15;
            if ( v10 == *v15 )
              goto LABEL_31;
            ++v26;
          }
          if ( v27 )
            v15 = v27;
        }
LABEL_31:
        *(_DWORD *)(a1 + 16) = v20;
        if ( *v15 != -4096 )
          --*(_DWORD *)(a1 + 20);
        *v15 = v10;
        result = v15 + 1;
        *((_DWORD *)v15 + 2) = 0;
        goto LABEL_14;
      }
LABEL_13:
      result = v17 + 1;
LABEL_14:
      *result = v9;
      if ( (*(_BYTE *)v10 & 4) != 0 )
      {
        v10 = *(_QWORD *)(v10 + 8);
        if ( v11 == v10 )
          goto LABEL_16;
      }
      else
      {
        while ( (*(_BYTE *)(v10 + 44) & 8) != 0 )
          v10 = *(_QWORD *)(v10 + 8);
        v10 = *(_QWORD *)(v10 + 8);
        if ( v11 == v10 )
          goto LABEL_16;
      }
    }
    ++*(_QWORD *)a1;
    goto LABEL_35;
  }
  return result;
}
