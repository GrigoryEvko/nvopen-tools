// Function: sub_375BDE0
// Address: 0x375bde0
//
int *__fastcall sub_375BDE0(__int64 a1, unsigned int a2)
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
  int v32; // edx
  __int64 v33; // rax
  int *v34; // rax
  __int64 v35; // rdx
  int *k; // rdx
  int v37; // edx
  int *v38; // r9
  int v39; // edi
  int v40; // r11d
  _DWORD *v41; // r10
  unsigned int v42; // esi
  int *v43; // rcx
  int v44; // r8d
  int v45; // edx
  int v46; // ecx
  _BYTE v47[112]; // [rsp+10h] [rbp-70h] BYREF

  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 8 )
  {
    if ( !v4 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    v28 = (int *)(a1 + 16);
    v29 = (int *)(a1 + 80);
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
      v29 = (int *)(a1 + 80);
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
          v14 = 16;
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
            v19 = 7;
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
      v29 = (int *)(a1 + 80);
      a2 = 64;
    }
  }
  v30 = v28;
  v31 = (int *)v47;
  do
  {
    while ( (unsigned int)*v30 > 0xFFFFFFFD )
    {
      v30 += 2;
      if ( v30 == v29 )
        goto LABEL_33;
    }
    if ( v31 )
      *v31 = *v30;
    v32 = v30[1];
    v30 += 2;
    v31 += 2;
    *(v31 - 1) = v32;
  }
  while ( v30 != v29 );
LABEL_33:
  if ( a2 > 8 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v33 = sub_C7D670(8LL * a2, 4);
    *(_DWORD *)(a1 + 24) = a2;
    *(_QWORD *)(a1 + 16) = v33;
  }
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v11 )
  {
    v34 = *(int **)(a1 + 16);
    v35 = 2LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v34 = v28;
    v35 = 16;
  }
  for ( k = &v34[v35]; k != v34; v34 += 2 )
  {
    if ( v34 )
      *v34 = -1;
  }
  result = (int *)v47;
  if ( v31 != (int *)v47 )
  {
    do
    {
      while ( 1 )
      {
        v37 = *result;
        if ( (unsigned int)*result <= 0xFFFFFFFD )
          break;
        result += 2;
        if ( v31 == result )
          return result;
      }
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v38 = v28;
        v39 = 7;
      }
      else
      {
        v46 = *(_DWORD *)(a1 + 24);
        v38 = *(int **)(a1 + 16);
        if ( !v46 )
        {
LABEL_74:
          MEMORY[0] = 0;
          BUG();
        }
        v39 = v46 - 1;
      }
      v40 = 1;
      v41 = 0;
      v42 = v39 & (37 * v37);
      v43 = &v38[2 * v42];
      v44 = *v43;
      if ( v37 != *v43 )
      {
        while ( v44 != -1 )
        {
          if ( v44 == -2 && !v41 )
            v41 = v43;
          v42 = v39 & (v40 + v42);
          v43 = &v38[2 * v42];
          v44 = *v43;
          if ( v37 == *v43 )
            goto LABEL_48;
          ++v40;
        }
        if ( v41 )
          v43 = v41;
      }
LABEL_48:
      *v43 = v37;
      v45 = result[1];
      result += 2;
      v43[1] = v45;
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
    while ( v31 != result );
  }
  return result;
}
