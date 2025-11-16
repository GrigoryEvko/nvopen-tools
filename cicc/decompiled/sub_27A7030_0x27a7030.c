// Function: sub_27A7030
// Address: 0x27a7030
//
__int64 __fastcall sub_27A7030(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // rdi
  unsigned int v7; // edx
  __int64 *v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r15
  __int64 j; // rbx
  unsigned int v12; // esi
  int v13; // edx
  int v14; // r10d
  __int64 v15; // rdi
  int *v16; // rax
  unsigned int k; // r8d
  int *v18; // r12
  int v19; // r11d
  __int64 v20; // rdx
  __int64 v21; // r8
  int v22; // r8d
  int v23; // ecx
  int v24; // r8d
  int v25; // r10d
  __int64 v26; // rdi
  int *v27; // r9
  unsigned int n; // edx
  int v29; // r11d
  unsigned int v30; // edx
  unsigned int v31; // r8d
  int i; // ecx
  int v33; // r10d
  int v34; // ecx
  __int64 v35; // rdx
  int *v36; // rdx
  int v37; // edi
  int v38; // r8d
  int v39; // ecx
  int v40; // r8d
  int v41; // r10d
  __int64 v42; // rdi
  unsigned int m; // edx
  int v44; // r11d
  unsigned int v45; // edx
  __int64 v46; // [rsp-48h] [rbp-48h]

  result = *(unsigned int *)(a3 + 24);
  v5 = *(_QWORD *)(a3 + 8);
  if ( (_DWORD)result )
  {
    v7 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v5 + 72LL * v7);
    v9 = *v8;
    if ( a2 != *v8 )
    {
      for ( i = 1; ; i = v33 )
      {
        if ( v9 == -4096 )
          return result;
        v33 = i + 1;
        v7 = (result - 1) & (i + v7);
        v8 = (__int64 *)(v5 + 72LL * v7);
        v9 = *v8;
        if ( a2 == *v8 )
          break;
      }
    }
    result = v5 + 72 * result;
    if ( v8 != (__int64 *)result )
    {
      v10 = v8[1];
      result = 3LL * *((unsigned int *)v8 + 4);
      for ( j = v10 + 24LL * *((unsigned int *)v8 + 4); v10 != j; j -= 24 )
      {
        v12 = *(_DWORD *)(a4 + 24);
        if ( !v12 )
        {
          ++*(_QWORD *)a4;
          goto LABEL_20;
        }
        v13 = *(_DWORD *)(j - 24);
        v14 = 1;
        v15 = *(_QWORD *)(a4 + 8);
        v16 = 0;
        for ( k = (v12 - 1)
                & (((0xBF58476D1CE4E5B9LL
                   * ((unsigned int)((0xBF58476D1CE4E5B9LL * *(_QWORD *)(j - 16)) >> 31)
                    ^ (484763065 * *(_DWORD *)(j - 16))
                    | ((unsigned __int64)(unsigned int)(37 * v13) << 32))) >> 31)
                 ^ (484763065
                  * (((0xBF58476D1CE4E5B9LL * *(_QWORD *)(j - 16)) >> 31) ^ (484763065 * *(_DWORD *)(j - 16)))));
              ;
              k = (v12 - 1) & v31 )
        {
          v18 = (int *)(v15 + 48LL * k);
          v19 = *v18;
          if ( *v18 == v13 && *((_QWORD *)v18 + 1) == *(_QWORD *)(j - 16) )
          {
            v20 = (unsigned int)v18[6];
            result = (__int64)(v18 + 4);
            v21 = *(_QWORD *)(j - 8);
            if ( v20 + 1 > (unsigned __int64)(unsigned int)v18[7] )
            {
              v46 = *(_QWORD *)(j - 8);
              sub_C8D5F0((__int64)(v18 + 4), v18 + 8, v20 + 1, 8u, v21, v20 + 1);
              v20 = (unsigned int)v18[6];
              v21 = v46;
              result = (__int64)(v18 + 4);
            }
            goto LABEL_17;
          }
          if ( v19 == -1 )
            break;
          if ( v19 == -2 && *((_QWORD *)v18 + 1) == -2 && !v16 )
            v16 = (int *)(v15 + 48LL * k);
LABEL_30:
          v31 = v14 + k;
          ++v14;
        }
        if ( *((_QWORD *)v18 + 1) != -1 )
          goto LABEL_30;
        v37 = *(_DWORD *)(a4 + 16);
        if ( !v16 )
          v16 = v18;
        ++*(_QWORD *)a4;
        v34 = v37 + 1;
        if ( 4 * (v37 + 1) < 3 * v12 )
        {
          if ( v12 - *(_DWORD *)(a4 + 20) - v34 <= v12 >> 3 )
          {
            sub_27A6D40(a4, v12);
            v38 = *(_DWORD *)(a4 + 24);
            if ( v38 )
            {
              v39 = *(_DWORD *)(j - 24);
              v40 = v38 - 1;
              v41 = 1;
              v27 = 0;
              for ( m = v40
                      & (((0xBF58476D1CE4E5B9LL
                         * ((unsigned int)((0xBF58476D1CE4E5B9LL * *(_QWORD *)(j - 16)) >> 31)
                          ^ (484763065 * *(_DWORD *)(j - 16))
                          | ((unsigned __int64)(unsigned int)(37 * v39) << 32))) >> 31)
                       ^ (484763065
                        * (((0xBF58476D1CE4E5B9LL * *(_QWORD *)(j - 16)) >> 31) ^ (484763065 * *(_DWORD *)(j - 16)))));
                    ;
                    m = v40 & v45 )
              {
                v42 = *(_QWORD *)(a4 + 8);
                v16 = (int *)(v42 + 48LL * m);
                v44 = *v16;
                if ( *v16 == v39 && *((_QWORD *)v16 + 1) == *(_QWORD *)(j - 16) )
                  break;
                if ( v44 == -1 )
                {
                  if ( *((_QWORD *)v16 + 1) == -1 )
                    goto LABEL_42;
                }
                else if ( v44 == -2 && *((_QWORD *)v16 + 1) == -2 && !v27 )
                {
                  v27 = (int *)(v42 + 48LL * m);
                }
                v45 = v41 + m;
                ++v41;
              }
              goto LABEL_37;
            }
LABEL_64:
            ++*(_DWORD *)(a4 + 16);
            BUG();
          }
          goto LABEL_38;
        }
LABEL_20:
        sub_27A6D40(a4, 2 * v12);
        v22 = *(_DWORD *)(a4 + 24);
        if ( !v22 )
          goto LABEL_64;
        v23 = *(_DWORD *)(j - 24);
        v24 = v22 - 1;
        v25 = 1;
        v27 = 0;
        for ( n = v24
                & (((0xBF58476D1CE4E5B9LL
                   * ((unsigned int)((0xBF58476D1CE4E5B9LL * *(_QWORD *)(j - 16)) >> 31)
                    ^ (484763065 * *(_DWORD *)(j - 16))
                    | ((unsigned __int64)(unsigned int)(37 * v23) << 32))) >> 31)
                 ^ (484763065
                  * (((0xBF58476D1CE4E5B9LL * *(_QWORD *)(j - 16)) >> 31) ^ (484763065 * *(_DWORD *)(j - 16)))));
              ;
              n = v24 & v30 )
        {
          v26 = *(_QWORD *)(a4 + 8);
          v16 = (int *)(v26 + 48LL * n);
          v29 = *v16;
          if ( *v16 == v23 && *((_QWORD *)v16 + 1) == *(_QWORD *)(j - 16) )
            break;
          if ( v29 == -1 )
          {
            if ( *((_QWORD *)v16 + 1) == -1 )
            {
LABEL_42:
              if ( v27 )
                v16 = v27;
              v34 = *(_DWORD *)(a4 + 16) + 1;
              goto LABEL_38;
            }
          }
          else if ( v29 == -2 && *((_QWORD *)v16 + 1) == -2 && !v27 )
          {
            v27 = (int *)(v26 + 48LL * n);
          }
          v30 = v25 + n;
          ++v25;
        }
LABEL_37:
        v34 = *(_DWORD *)(a4 + 16) + 1;
LABEL_38:
        *(_DWORD *)(a4 + 16) = v34;
        if ( *v16 != -1 || *((_QWORD *)v16 + 1) != -1 )
          --*(_DWORD *)(a4 + 20);
        *v16 = *(_DWORD *)(j - 24);
        v35 = *(_QWORD *)(j - 16);
        *((_QWORD *)v16 + 3) = 0x200000000LL;
        *((_QWORD *)v16 + 1) = v35;
        v36 = v16 + 8;
        result = (__int64)(v16 + 4);
        *(_QWORD *)result = v36;
        v20 = 0;
        v21 = *(_QWORD *)(j - 8);
LABEL_17:
        *(_QWORD *)(*(_QWORD *)result + 8 * v20) = v21;
        ++*(_DWORD *)(result + 8);
      }
    }
  }
  return result;
}
