// Function: sub_2E9C4E0
// Address: 0x2e9c4e0
//
int *__fastcall sub_2E9C4E0(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r13d
  char v4; // cl
  unsigned __int64 v5; // rax
  __int64 v6; // r12
  unsigned int v7; // r14d
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r8
  bool v11; // zf
  int *v12; // rsi
  _DWORD *v13; // rax
  __int64 v14; // rdx
  _DWORD *i; // rdx
  int *j; // rax
  int v17; // edx
  __int64 v18; // r10
  int v19; // ecx
  int v20; // r15d
  _DWORD *v21; // r14
  unsigned int v22; // edi
  _DWORD *v23; // r9
  int v24; // r11d
  int v25; // edx
  int *result; // rax
  int v27; // ecx
  int *v28; // r15
  int *v29; // rdx
  int *v30; // rax
  int *v31; // r14
  __int64 v32; // rax
  int *v33; // rax
  __int64 v34; // rdx
  int *k; // rdx
  int *v36; // r9
  int v37; // edi
  int v38; // r11d
  int *v39; // r10
  unsigned int v40; // ecx
  int *v41; // rsi
  int v42; // r8d
  int v43; // edx
  int v44; // ecx
  _BYTE v45[64]; // [rsp+10h] [rbp-40h] BYREF

  v2 = a2;
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    if ( !v4 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    v28 = (int *)(a1 + 16);
    v29 = (int *)(a1 + 32);
  }
  else
  {
    v5 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
            | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
            | (a2 - 1)
            | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
          | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 16)
        | (((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
        | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1))
       + 1;
    v2 = v5;
    if ( (unsigned int)v5 > 0x40 )
    {
      v28 = (int *)(a1 + 16);
      v29 = (int *)(a1 + 32);
      if ( !v4 )
      {
        v6 = *(_QWORD *)(a1 + 16);
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 4LL * (unsigned int)v5;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v4 )
      {
        v6 = *(_QWORD *)(a1 + 16);
        v7 = *(_DWORD *)(a1 + 24);
        v2 = 64;
        v8 = 256;
LABEL_5:
        v9 = sub_C7D670(v8, 4);
        *(_DWORD *)(a1 + 24) = v2;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
        v10 = 4LL * v7;
        v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v12 = (int *)(v6 + v10);
        if ( v11 )
        {
          v13 = *(_DWORD **)(a1 + 16);
          v14 = *(unsigned int *)(a1 + 24);
        }
        else
        {
          v13 = (_DWORD *)(a1 + 16);
          v14 = 4;
        }
        for ( i = &v13[v14]; i != v13; ++v13 )
        {
          if ( v13 )
            *v13 = -1;
        }
        for ( j = (int *)v6; v12 != j; *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2)
                                                           | *(_DWORD *)(a1 + 8) & 1 )
        {
          while ( 1 )
          {
            v17 = *j;
            if ( (unsigned int)*j <= 0xFFFFFFFD )
              break;
            if ( v12 == ++j )
              return (int *)sub_C7D6A0(v6, v10, 4);
          }
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v18 = a1 + 16;
            v19 = 3;
          }
          else
          {
            v27 = *(_DWORD *)(a1 + 24);
            v18 = *(_QWORD *)(a1 + 16);
            if ( !v27 )
              goto LABEL_74;
            v19 = v27 - 1;
          }
          v20 = 1;
          v21 = 0;
          v22 = v19 & (37 * v17);
          v23 = (_DWORD *)(v18 + 4LL * v22);
          v24 = *v23;
          if ( v17 != *v23 )
          {
            while ( v24 != -1 )
            {
              if ( v24 == -2 && !v21 )
                v21 = v23;
              v22 = v19 & (v20 + v22);
              v23 = (_DWORD *)(v18 + 4LL * v22);
              v24 = *v23;
              if ( v17 == *v23 )
                goto LABEL_21;
              ++v20;
            }
            if ( v21 )
              v23 = v21;
          }
LABEL_21:
          v25 = *j++;
          *v23 = v25;
        }
        return (int *)sub_C7D6A0(v6, v10, 4);
      }
      v28 = (int *)(a1 + 16);
      v29 = (int *)(a1 + 32);
      v2 = 64;
    }
  }
  v30 = v28;
  v31 = (int *)v45;
  do
  {
    if ( (unsigned int)*v30 <= 0xFFFFFFFD )
    {
      if ( v31 )
        *v31 = *v30;
      ++v31;
    }
    ++v30;
  }
  while ( v30 != v29 );
  if ( v2 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v32 = sub_C7D670(4LL * v2, 4);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v32;
  }
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v11 )
  {
    v33 = *(int **)(a1 + 16);
    v34 = *(unsigned int *)(a1 + 24);
  }
  else
  {
    v33 = v28;
    v34 = 4;
  }
  for ( k = &v33[v34]; k != v33; ++v33 )
  {
    if ( v33 )
      *v33 = -1;
  }
  result = (int *)v45;
  if ( v31 != (int *)v45 )
  {
    do
    {
      while ( 1 )
      {
        v17 = *result;
        if ( (unsigned int)*result <= 0xFFFFFFFD )
          break;
        if ( v31 == ++result )
          return result;
      }
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v36 = v28;
        v37 = 3;
      }
      else
      {
        v44 = *(_DWORD *)(a1 + 24);
        v36 = *(int **)(a1 + 16);
        if ( !v44 )
        {
LABEL_74:
          MEMORY[0] = v17;
          BUG();
        }
        v37 = v44 - 1;
      }
      v38 = 1;
      v39 = 0;
      v40 = v37 & (37 * v17);
      v41 = &v36[v40];
      v42 = *v41;
      if ( v17 != *v41 )
      {
        while ( v42 != -1 )
        {
          if ( v42 == -2 && !v39 )
            v39 = v41;
          v40 = v37 & (v38 + v40);
          v41 = &v36[v40];
          v42 = *v41;
          if ( v17 == *v41 )
            goto LABEL_45;
          ++v38;
        }
        if ( v39 )
          v41 = v39;
      }
LABEL_45:
      v43 = *result++;
      *v41 = v43;
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
    while ( v31 != result );
  }
  return result;
}
