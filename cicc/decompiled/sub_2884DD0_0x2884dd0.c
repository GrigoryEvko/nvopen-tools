// Function: sub_2884DD0
// Address: 0x2884dd0
//
void __fastcall sub_2884DD0(__int64 a1)
{
  __int64 *v1; // rbx
  __int64 *i; // r13
  unsigned int v4; // esi
  __int64 v5; // rcx
  int v6; // r11d
  __int64 *v7; // r10
  __int64 v8; // rdi
  unsigned int j; // eax
  __int64 *v10; // r8
  __int64 v11; // r15
  int v12; // edi
  __int64 v13; // rcx
  int v14; // edi
  __int64 v15; // rsi
  int v16; // r9d
  __int64 *v17; // r8
  unsigned int m; // eax
  __int64 v19; // r11
  unsigned int v20; // eax
  unsigned int v21; // eax
  int v22; // edx
  __int64 v23; // rax
  int v24; // eax
  int v25; // edi
  __int64 v26; // rcx
  int v27; // edi
  __int64 v28; // rsi
  int v29; // r9d
  unsigned int k; // eax
  __int64 v31; // r11
  unsigned int v32; // eax

  v1 = *(__int64 **)(a1 + 32);
  for ( i = &v1[2 * *(unsigned int *)(a1 + 40)]; i != v1; v7[1] = *(v1 - 1) )
  {
LABEL_2:
    v4 = *(_DWORD *)(a1 + 24);
    if ( v4 )
    {
      v5 = v1[1];
      v6 = 1;
      v7 = 0;
      v8 = *(_QWORD *)(a1 + 8);
      for ( j = (v4 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)
                  | ((unsigned __int64)(((unsigned int)*v1 >> 9) ^ ((unsigned int)*v1 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)))); ; j = (v4 - 1) & v21 )
      {
        v10 = (__int64 *)(v8 + 16LL * j);
        v11 = *v10;
        if ( *v10 == *v1 && v10[1] == v5 )
        {
          v1 += 2;
          if ( i != v1 )
            goto LABEL_2;
          return;
        }
        if ( v11 == -4096 )
        {
          if ( v10[1] == -4096 )
          {
            v24 = *(_DWORD *)(a1 + 16);
            if ( !v7 )
              v7 = v10;
            ++*(_QWORD *)a1;
            v22 = v24 + 1;
            if ( 4 * (v24 + 1) >= 3 * v4 )
              goto LABEL_15;
            if ( v4 - *(_DWORD *)(a1 + 20) - v22 <= v4 >> 3 )
            {
              sub_2884B10(a1, v4);
              v25 = *(_DWORD *)(a1 + 24);
              if ( v25 )
              {
                v26 = v1[1];
                v27 = v25 - 1;
                v29 = 1;
                v17 = 0;
                for ( k = v27
                        & (((0xBF58476D1CE4E5B9LL
                           * (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4)
                            | ((unsigned __int64)(((unsigned int)*v1 >> 9) ^ ((unsigned int)*v1 >> 4)) << 32))) >> 31)
                         ^ (484763065 * (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4)))); ; k = v27 & v32 )
                {
                  v28 = *(_QWORD *)(a1 + 8);
                  v7 = (__int64 *)(v28 + 16LL * k);
                  v31 = *v7;
                  if ( *v7 == *v1 && v7[1] == v26 )
                    break;
                  if ( v31 == -4096 )
                  {
                    if ( v7[1] == -4096 )
                      goto LABEL_33;
                  }
                  else if ( v31 == -8192 && v7[1] == -8192 && !v17 )
                  {
                    v17 = (__int64 *)(v28 + 16LL * k);
                  }
                  v32 = v29 + k;
                  ++v29;
                }
                goto LABEL_27;
              }
LABEL_55:
              ++*(_DWORD *)(a1 + 16);
              BUG();
            }
            goto LABEL_28;
          }
        }
        else if ( v11 == -8192 && v10[1] == -8192 && !v7 )
        {
          v7 = (__int64 *)(v8 + 16LL * j);
        }
        v21 = v6 + j;
        ++v6;
      }
    }
    ++*(_QWORD *)a1;
LABEL_15:
    sub_2884B10(a1, 2 * v4);
    v12 = *(_DWORD *)(a1 + 24);
    if ( !v12 )
      goto LABEL_55;
    v13 = v1[1];
    v14 = v12 - 1;
    v16 = 1;
    v17 = 0;
    for ( m = v14
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)
                | ((unsigned __int64)(((unsigned int)*v1 >> 9) ^ ((unsigned int)*v1 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)))); ; m = v14 & v20 )
    {
      v15 = *(_QWORD *)(a1 + 8);
      v7 = (__int64 *)(v15 + 16LL * m);
      v19 = *v7;
      if ( *v7 == *v1 && v7[1] == v13 )
        break;
      if ( v19 == -4096 )
      {
        if ( v7[1] == -4096 )
        {
LABEL_33:
          if ( v17 )
            v7 = v17;
          v22 = *(_DWORD *)(a1 + 16) + 1;
          goto LABEL_28;
        }
      }
      else if ( v19 == -8192 && v7[1] == -8192 && !v17 )
      {
        v17 = (__int64 *)(v15 + 16LL * m);
      }
      v20 = v16 + m;
      ++v16;
    }
LABEL_27:
    v22 = *(_DWORD *)(a1 + 16) + 1;
LABEL_28:
    *(_DWORD *)(a1 + 16) = v22;
    if ( *v7 != -4096 || v7[1] != -4096 )
      --*(_DWORD *)(a1 + 20);
    v23 = *v1;
    v1 += 2;
    *v7 = v23;
  }
}
