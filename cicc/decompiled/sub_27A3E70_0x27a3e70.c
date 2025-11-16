// Function: sub_27A3E70
// Address: 0x27a3e70
//
int *__fastcall sub_27A3E70(__int64 a1, int *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r8
  int v6; // r11d
  int *v7; // rdx
  unsigned int i; // eax
  int *v9; // r9
  int v10; // r13d
  unsigned int v11; // eax
  int v13; // eax
  int v14; // ecx
  int v15; // r8d
  int v16; // r8d
  __int64 v17; // rdi
  int v18; // r11d
  int *v19; // r10
  unsigned int j; // eax
  int v21; // r9d
  unsigned int v22; // eax
  int v23; // r8d
  int v24; // r8d
  __int64 v25; // rdi
  int v26; // r11d
  unsigned int k; // eax
  int v28; // r9d
  unsigned int v29; // eax

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_23;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = 1;
  v7 = 0;
  for ( i = (v4 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * ((unsigned int)((0xBF58476D1CE4E5B9LL * *((_QWORD *)a2 + 1)) >> 31) ^ (484763065 * a2[2])
              | ((unsigned __int64)(unsigned int)(37 * *a2) << 32))) >> 31)
           ^ (484763065 * (((0xBF58476D1CE4E5B9LL * *((_QWORD *)a2 + 1)) >> 31) ^ (484763065 * a2[2]))));
        ;
        i = (v4 - 1) & v11 )
  {
    v9 = (int *)(v5 + ((unsigned __int64)i << 6));
    v10 = *v9;
    if ( *v9 == *a2 && *((_QWORD *)v9 + 1) == *((_QWORD *)a2 + 1) )
      return v9 + 4;
    if ( v10 == -1 )
      break;
    if ( v10 == -2 && *((_QWORD *)v9 + 1) == -2 && !v7 )
      v7 = (int *)(v5 + ((unsigned __int64)i << 6));
LABEL_9:
    v11 = v6 + i;
    ++v6;
  }
  if ( *((_QWORD *)v9 + 1) != -1 )
    goto LABEL_9;
  v13 = *(_DWORD *)(a1 + 16);
  if ( !v7 )
    v7 = v9;
  ++*(_QWORD *)a1;
  v14 = v13 + 1;
  if ( 4 * (v13 + 1) >= 3 * v4 )
  {
LABEL_23:
    sub_27A3B90(a1, 2 * v4);
    v15 = *(_DWORD *)(a1 + 24);
    if ( v15 )
    {
      v16 = v15 - 1;
      v18 = 1;
      v19 = 0;
      for ( j = v16
              & (((0xBF58476D1CE4E5B9LL
                 * ((unsigned int)((0xBF58476D1CE4E5B9LL * *((_QWORD *)a2 + 1)) >> 31) ^ (484763065 * a2[2])
                  | ((unsigned __int64)(unsigned int)(37 * *a2) << 32))) >> 31)
               ^ (484763065 * (((0xBF58476D1CE4E5B9LL * *((_QWORD *)a2 + 1)) >> 31) ^ (484763065 * a2[2]))));
            ;
            j = v16 & v22 )
      {
        v17 = *(_QWORD *)(a1 + 8);
        v7 = (int *)(v17 + ((unsigned __int64)j << 6));
        v21 = *v7;
        if ( *v7 == *a2 && *((_QWORD *)v7 + 1) == *((_QWORD *)a2 + 1) )
          break;
        if ( v21 == -1 )
        {
          if ( *((_QWORD *)v7 + 1) == -1 )
          {
LABEL_46:
            if ( v19 )
              v7 = v19;
            v14 = *(_DWORD *)(a1 + 16) + 1;
            goto LABEL_17;
          }
        }
        else if ( v21 == -2 && *((_QWORD *)v7 + 1) == -2 && !v19 )
        {
          v19 = (int *)(v17 + ((unsigned __int64)j << 6));
        }
        v22 = v18 + j;
        ++v18;
      }
      goto LABEL_42;
    }
LABEL_51:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v4 - *(_DWORD *)(a1 + 20) - v14 <= v4 >> 3 )
  {
    sub_27A3B90(a1, v4);
    v23 = *(_DWORD *)(a1 + 24);
    if ( v23 )
    {
      v24 = v23 - 1;
      v26 = 1;
      v19 = 0;
      for ( k = v24
              & (((0xBF58476D1CE4E5B9LL
                 * ((unsigned int)((0xBF58476D1CE4E5B9LL * *((_QWORD *)a2 + 1)) >> 31) ^ (484763065 * a2[2])
                  | ((unsigned __int64)(unsigned int)(37 * *a2) << 32))) >> 31)
               ^ (484763065 * (((0xBF58476D1CE4E5B9LL * *((_QWORD *)a2 + 1)) >> 31) ^ (484763065 * a2[2]))));
            ;
            k = v24 & v29 )
      {
        v25 = *(_QWORD *)(a1 + 8);
        v7 = (int *)(v25 + ((unsigned __int64)k << 6));
        v28 = *v7;
        if ( *v7 == *a2 && *((_QWORD *)v7 + 1) == *((_QWORD *)a2 + 1) )
          break;
        if ( v28 == -1 )
        {
          if ( *((_QWORD *)v7 + 1) == -1 )
            goto LABEL_46;
        }
        else if ( v28 == -2 && *((_QWORD *)v7 + 1) == -2 && !v19 )
        {
          v19 = (int *)(v25 + ((unsigned __int64)k << 6));
        }
        v29 = v26 + k;
        ++v26;
      }
LABEL_42:
      v14 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_17;
    }
    goto LABEL_51;
  }
LABEL_17:
  *(_DWORD *)(a1 + 16) = v14;
  if ( *v7 != -1 || *((_QWORD *)v7 + 1) != -1 )
    --*(_DWORD *)(a1 + 20);
  *v7 = *a2;
  *((_QWORD *)v7 + 1) = *((_QWORD *)a2 + 1);
  *((_QWORD *)v7 + 2) = v7 + 8;
  *((_QWORD *)v7 + 3) = 0x400000000LL;
  return v7 + 4;
}
