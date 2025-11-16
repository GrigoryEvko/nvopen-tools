// Function: sub_29758F0
// Address: 0x29758f0
//
int *__fastcall sub_29758F0(__int64 a1, unsigned int a2)
{
  char v4; // si
  unsigned __int64 v5; // rax
  __int64 v6; // r12
  unsigned int v7; // r13d
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r9
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
  unsigned int v22; // r8d
  _DWORD *v23; // rdi
  int v24; // r11d
  int v25; // edx
  int *result; // rax
  int v27; // ecx
  int *v28; // r15
  int *v29; // rcx
  int *v30; // rax
  int *v31; // r13
  __int64 v32; // rax
  int *v33; // rax
  __int64 v34; // rdx
  int *k; // rdx
  int v36; // edx
  int *v37; // r9
  int v38; // edi
  int v39; // r11d
  _DWORD *v40; // r10
  unsigned int v41; // esi
  int *v42; // rcx
  int v43; // r8d
  int v44; // edx
  int v45; // ecx
  _BYTE v46[80]; // [rsp+10h] [rbp-50h] BYREF

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
    v29 = (int *)(a1 + 48);
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
    a2 = v5;
    if ( (unsigned int)v5 > 0x40 )
    {
      v28 = (int *)(a1 + 16);
      v29 = (int *)(a1 + 48);
      if ( !v4 )
      {
        v6 = *(_QWORD *)(a1 + 16);
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 8LL * (unsigned int)v5;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v4 )
      {
        v6 = *(_QWORD *)(a1 + 16);
        v7 = *(_DWORD *)(a1 + 24);
        a2 = 64;
        v8 = 512;
LABEL_5:
        v9 = sub_C7D670(v8, 4);
        *(_DWORD *)(a1 + 24) = a2;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
        v10 = 8LL * v7;
        v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v12 = (int *)(v6 + v10);
        if ( v11 )
        {
          v13 = *(_DWORD **)(a1 + 16);
          v14 = 2LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v13 = (_DWORD *)(a1 + 16);
          v14 = 8;
        }
        for ( i = &v13[v14]; i != v13; v13 += 2 )
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
            j += 2;
            if ( v12 == j )
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
          v23 = (_DWORD *)(v18 + 8LL * v22);
          v24 = *v23;
          if ( v17 != *v23 )
          {
            while ( v24 != -1 )
            {
              if ( v24 == -2 && !v21 )
                v21 = v23;
              v22 = v19 & (v20 + v22);
              v23 = (_DWORD *)(v18 + 8LL * v22);
              v24 = *v23;
              if ( v17 == *v23 )
                goto LABEL_21;
              ++v20;
            }
            if ( v21 )
              v23 = v21;
          }
LABEL_21:
          *v23 = v17;
          v25 = j[1];
          j += 2;
          v23[1] = v25;
        }
        return (int *)sub_C7D6A0(v6, v10, 4);
      }
      v28 = (int *)(a1 + 16);
      v29 = (int *)(a1 + 48);
      a2 = 64;
    }
  }
  v30 = v28;
  v31 = (int *)v46;
  do
  {
    if ( (unsigned int)*v30 <= 0xFFFFFFFD )
    {
      if ( v31 )
        *v31 = *v30;
      v31 += 2;
      *(v31 - 1) = v30[1];
    }
    v30 += 2;
  }
  while ( v30 != v29 );
  if ( a2 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v32 = sub_C7D670(8LL * a2, 4);
    *(_DWORD *)(a1 + 24) = a2;
    *(_QWORD *)(a1 + 16) = v32;
  }
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v11 )
  {
    v33 = *(int **)(a1 + 16);
    v34 = 2LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v33 = v28;
    v34 = 8;
  }
  for ( k = &v33[v34]; k != v33; v33 += 2 )
  {
    if ( v33 )
      *v33 = -1;
  }
  result = (int *)v46;
  if ( v31 != (int *)v46 )
  {
    do
    {
      while ( 1 )
      {
        v36 = *result;
        if ( (unsigned int)*result <= 0xFFFFFFFD )
          break;
        result += 2;
        if ( v31 == result )
          return result;
      }
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v37 = v28;
        v38 = 3;
      }
      else
      {
        v45 = *(_DWORD *)(a1 + 24);
        v37 = *(int **)(a1 + 16);
        if ( !v45 )
        {
LABEL_74:
          MEMORY[0] = 0;
          BUG();
        }
        v38 = v45 - 1;
      }
      v39 = 1;
      v40 = 0;
      v41 = v38 & (37 * v36);
      v42 = &v37[2 * v41];
      v43 = *v42;
      if ( v36 != *v42 )
      {
        while ( v43 != -1 )
        {
          if ( v43 == -2 && !v40 )
            v40 = v42;
          v41 = v38 & (v39 + v41);
          v42 = &v37[2 * v41];
          v43 = *v42;
          if ( v36 == *v42 )
            goto LABEL_45;
          ++v39;
        }
        if ( v40 )
          v42 = v40;
      }
LABEL_45:
      *v42 = v36;
      v44 = result[1];
      result += 2;
      v42[1] = v44;
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
    while ( v31 != result );
  }
  return result;
}
