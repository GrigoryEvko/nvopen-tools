// Function: sub_350D990
// Address: 0x350d990
//
int *__fastcall sub_350D990(__int64 a1, __int64 a2)
{
  int v4; // eax
  __int64 v5; // rdx
  _QWORD *v6; // rax
  _QWORD *i; // rdx
  int *result; // rax
  int v9; // r13d
  _DWORD *v10; // rcx
  unsigned int v11; // esi
  int v12; // r15d
  __int64 v13; // r9
  int v14; // r11d
  _QWORD *v15; // rdx
  unsigned int v16; // r8d
  _QWORD *v17; // rax
  __int64 v18; // rdi
  int v19; // eax
  int v20; // edi
  unsigned int v21; // ecx
  unsigned int v22; // eax
  _QWORD *v23; // rdi
  int v24; // r13d
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // rax
  _QWORD *v27; // rax
  __int64 v28; // rdx
  _QWORD *j; // rdx
  int v30; // eax
  int v31; // r8d
  __int64 v32; // rsi
  unsigned int v33; // eax
  __int64 v34; // r9
  int v35; // r11d
  _QWORD *v36; // r10
  int v37; // eax
  int v38; // eax
  __int64 v39; // r8
  _QWORD *v40; // r9
  unsigned int v41; // r14d
  int v42; // r10d
  __int64 v43; // rsi
  _QWORD *v44; // rax
  _DWORD *v45; // [rsp+8h] [rbp-38h]
  _DWORD *v46; // [rsp+8h] [rbp-38h]

  v4 = *(_DWORD *)(a2 + 16);
  ++*(_QWORD *)a2;
  if ( !v4 )
  {
    if ( !*(_DWORD *)(a2 + 20) )
      goto LABEL_7;
    v5 = *(unsigned int *)(a2 + 24);
    if ( (unsigned int)v5 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a2 + 8), 16 * v5, 8);
      *(_QWORD *)(a2 + 8) = 0;
      *(_QWORD *)(a2 + 16) = 0;
      *(_DWORD *)(a2 + 24) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v21 = 4 * v4;
  v5 = *(unsigned int *)(a2 + 24);
  if ( (unsigned int)(4 * v4) < 0x40 )
    v21 = 64;
  if ( (unsigned int)v5 <= v21 )
  {
LABEL_4:
    v6 = *(_QWORD **)(a2 + 8);
    for ( i = &v6[2 * v5]; i != v6; v6 += 2 )
      *v6 = -4096;
    *(_QWORD *)(a2 + 16) = 0;
    goto LABEL_7;
  }
  v22 = v4 - 1;
  if ( !v22 )
  {
    v23 = *(_QWORD **)(a2 + 8);
    v24 = 64;
LABEL_40:
    sub_C7D6A0((__int64)v23, 16 * v5, 8);
    v25 = ((((((((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
             | (4 * v24 / 3u + 1)
             | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 4)
           | (((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
           | (4 * v24 / 3u + 1)
           | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
           | (4 * v24 / 3u + 1)
           | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 4)
         | (((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
         | (4 * v24 / 3u + 1)
         | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 16;
    v26 = (v25
         | (((((((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
             | (4 * v24 / 3u + 1)
             | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 4)
           | (((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
           | (4 * v24 / 3u + 1)
           | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
           | (4 * v24 / 3u + 1)
           | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 4)
         | (((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
         | (4 * v24 / 3u + 1)
         | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a2 + 24) = v26;
    v27 = (_QWORD *)sub_C7D670(16 * v26, 8);
    v28 = *(unsigned int *)(a2 + 24);
    *(_QWORD *)(a2 + 16) = 0;
    *(_QWORD *)(a2 + 8) = v27;
    for ( j = &v27[2 * v28]; j != v27; v27 += 2 )
    {
      if ( v27 )
        *v27 = -4096;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v22, v22);
  v23 = *(_QWORD **)(a2 + 8);
  v24 = 1 << (33 - (v22 ^ 0x1F));
  if ( v24 < 64 )
    v24 = 64;
  if ( v24 != (_DWORD)v5 )
    goto LABEL_40;
  *(_QWORD *)(a2 + 16) = 0;
  v44 = &v23[2 * (unsigned int)v24];
  do
  {
    if ( v23 )
      *v23 = -4096;
    v23 += 2;
  }
  while ( v44 != v23 );
LABEL_7:
  result = *(int **)(a1 + 24);
  v9 = 0;
  v10 = result + 12;
  if ( result + 12 != (int *)a1 )
  {
    while ( 1 )
    {
      v11 = *(_DWORD *)(a2 + 24);
      v12 = v9++;
      if ( !v11 )
        break;
      v13 = *(_QWORD *)(a2 + 8);
      v14 = 1;
      v15 = 0;
      v16 = (v11 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v17 = (_QWORD *)(v13 + 16LL * v16);
      v18 = *v17;
      if ( a1 != *v17 )
      {
        while ( v18 != -4096 )
        {
          if ( v18 == -8192 && !v15 )
            v15 = v17;
          v16 = (v11 - 1) & (v14 + v16);
          v17 = (_QWORD *)(v13 + 16LL * v16);
          v18 = *v17;
          if ( a1 == *v17 )
            goto LABEL_12;
          ++v14;
        }
        if ( !v15 )
          v15 = v17;
        v19 = *(_DWORD *)(a2 + 16);
        ++*(_QWORD *)a2;
        v20 = v19 + 1;
        if ( 4 * (v19 + 1) < 3 * v11 )
        {
          if ( v11 - *(_DWORD *)(a2 + 20) - v20 <= v11 >> 3 )
          {
            v46 = v10;
            sub_2E261E0(a2, v11);
            v37 = *(_DWORD *)(a2 + 24);
            if ( !v37 )
            {
LABEL_77:
              ++*(_DWORD *)(a2 + 16);
              BUG();
            }
            v38 = v37 - 1;
            v39 = *(_QWORD *)(a2 + 8);
            v40 = 0;
            v41 = v38 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
            v42 = 1;
            v20 = *(_DWORD *)(a2 + 16) + 1;
            v10 = v46;
            v15 = (_QWORD *)(v39 + 16LL * v41);
            v43 = *v15;
            if ( a1 != *v15 )
            {
              while ( v43 != -4096 )
              {
                if ( v43 == -8192 && !v40 )
                  v40 = v15;
                v41 = v38 & (v42 + v41);
                v15 = (_QWORD *)(v39 + 16LL * v41);
                v43 = *v15;
                if ( a1 == *v15 )
                  goto LABEL_30;
                ++v42;
              }
              if ( v40 )
                v15 = v40;
            }
          }
          goto LABEL_30;
        }
LABEL_46:
        v45 = v10;
        sub_2E261E0(a2, 2 * v11);
        v30 = *(_DWORD *)(a2 + 24);
        if ( !v30 )
          goto LABEL_77;
        v31 = v30 - 1;
        v32 = *(_QWORD *)(a2 + 8);
        v33 = (v30 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v20 = *(_DWORD *)(a2 + 16) + 1;
        v10 = v45;
        v15 = (_QWORD *)(v32 + 16LL * v33);
        v34 = *v15;
        if ( a1 != *v15 )
        {
          v35 = 1;
          v36 = 0;
          while ( v34 != -4096 )
          {
            if ( !v36 && v34 == -8192 )
              v36 = v15;
            v33 = v31 & (v35 + v33);
            v15 = (_QWORD *)(v32 + 16LL * v33);
            v34 = *v15;
            if ( a1 == *v15 )
              goto LABEL_30;
            ++v35;
          }
          if ( v36 )
            v15 = v36;
        }
LABEL_30:
        *(_DWORD *)(a2 + 16) = v20;
        if ( *v15 != -4096 )
          --*(_DWORD *)(a2 + 20);
        *v15 = a1;
        result = (int *)(v15 + 1);
        *((_DWORD *)v15 + 2) = 0;
        goto LABEL_13;
      }
LABEL_12:
      result = (int *)(v17 + 1);
LABEL_13:
      *result = v12;
      if ( !a1 )
        BUG();
      if ( (*(_BYTE *)a1 & 4) != 0 )
      {
        a1 = *(_QWORD *)(a1 + 8);
        if ( v10 == (_DWORD *)a1 )
          return result;
      }
      else
      {
        while ( (*(_BYTE *)(a1 + 44) & 8) != 0 )
          a1 = *(_QWORD *)(a1 + 8);
        a1 = *(_QWORD *)(a1 + 8);
        if ( v10 == (_DWORD *)a1 )
          return result;
      }
    }
    ++*(_QWORD *)a2;
    goto LABEL_46;
  }
  return result;
}
