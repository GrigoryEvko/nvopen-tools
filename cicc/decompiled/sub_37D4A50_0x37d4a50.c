// Function: sub_37D4A50
// Address: 0x37d4a50
//
unsigned int *__fastcall sub_37D4A50(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r14d
  char v4; // r12
  unsigned int v5; // eax
  __int64 v6; // r12
  unsigned int v7; // r13d
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r8
  bool v11; // zf
  unsigned int *v12; // rdi
  _DWORD *v13; // rax
  __int64 v14; // rdx
  _DWORD *i; // rdx
  unsigned int *j; // rax
  unsigned int v17; // edx
  __int64 v18; // r9
  int v19; // esi
  int v20; // r15d
  int *v21; // r14
  unsigned int v22; // r11d
  int *v23; // rcx
  int v24; // r10d
  unsigned int v25; // edx
  unsigned int *result; // rax
  int v27; // esi
  unsigned int *v28; // r15
  unsigned int *v29; // rdx
  unsigned int *v30; // rax
  unsigned int *v31; // r13
  __int64 v32; // rax
  unsigned int *v33; // rax
  __int64 v34; // rdx
  unsigned int *k; // rdx
  unsigned int *v36; // rdi
  int v37; // esi
  int v38; // r11d
  unsigned int *v39; // r10
  unsigned int v40; // r9d
  unsigned int *v41; // rcx
  unsigned int v42; // r8d
  unsigned int v43; // edx
  int v44; // ecx
  _BYTE v45[112]; // [rsp+10h] [rbp-70h] BYREF

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
    v28 = (unsigned int *)(a1 + 16);
    v29 = (unsigned int *)(a1 + 80);
  }
  else
  {
    v5 = sub_AF1560(a2 - 1);
    v2 = v5;
    if ( v5 > 0x40 )
    {
      v28 = (unsigned int *)(a1 + 16);
      v29 = (unsigned int *)(a1 + 80);
      if ( !v4 )
      {
        v6 = *(_QWORD *)(a1 + 16);
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 16LL * v5;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v4 )
      {
        v6 = *(_QWORD *)(a1 + 16);
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 1024;
        v2 = 64;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v2;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
        v10 = 16LL * v7;
        v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v12 = (unsigned int *)(v6 + v10);
        if ( v11 )
        {
          v13 = *(_DWORD **)(a1 + 16);
          v14 = 4LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v13 = (_DWORD *)(a1 + 16);
          v14 = 16;
        }
        for ( i = &v13[v14]; i != v13; v13 += 4 )
        {
          if ( v13 )
            *v13 = -1;
        }
        for ( j = (unsigned int *)v6;
              v12 != j;
              *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1 )
        {
          while ( 1 )
          {
            v17 = *j;
            if ( *j <= 0xFFFFFFFD )
              break;
            j += 4;
            if ( v12 == j )
              return (unsigned int *)sub_C7D6A0(v6, v10, 8);
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
          v22 = v19 & v17;
          v23 = (int *)(v18 + 16LL * (v19 & v17));
          v24 = *v23;
          if ( v17 != *v23 )
          {
            while ( v24 != -1 )
            {
              if ( v24 == -2 && !v21 )
                v21 = v23;
              v22 = v19 & (v20 + v22);
              v23 = (int *)(v18 + 16LL * v22);
              v24 = *v23;
              if ( v17 == *v23 )
                goto LABEL_21;
              ++v20;
            }
            if ( v21 )
              v23 = v21;
          }
LABEL_21:
          v25 = *j;
          j += 4;
          *v23 = v25;
          *((_QWORD *)v23 + 1) = *((_QWORD *)j - 1);
        }
        return (unsigned int *)sub_C7D6A0(v6, v10, 8);
      }
      v28 = (unsigned int *)(a1 + 16);
      v29 = (unsigned int *)(a1 + 80);
      v2 = 64;
    }
  }
  v30 = v28;
  v31 = (unsigned int *)v45;
  do
  {
    if ( *v30 <= 0xFFFFFFFD )
    {
      if ( v31 )
        *v31 = *v30;
      v31 += 4;
      *((_QWORD *)v31 - 1) = *((_QWORD *)v30 + 1);
    }
    v30 += 4;
  }
  while ( v30 != v29 );
  if ( v2 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v32 = sub_C7D670(16LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v32;
  }
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v11 )
  {
    v33 = *(unsigned int **)(a1 + 16);
    v34 = 4LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v33 = v28;
    v34 = 16;
  }
  for ( k = &v33[v34]; k != v33; v33 += 4 )
  {
    if ( v33 )
      *v33 = -1;
  }
  result = (unsigned int *)v45;
  if ( v31 != (unsigned int *)v45 )
  {
    do
    {
      while ( 1 )
      {
        v17 = *result;
        if ( *result <= 0xFFFFFFFD )
          break;
        result += 4;
        if ( v31 == result )
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
        v36 = *(unsigned int **)(a1 + 16);
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
      v40 = v37 & v17;
      v41 = &v36[4 * (v37 & v17)];
      v42 = *v41;
      if ( v17 != *v41 )
      {
        while ( v42 != -1 )
        {
          if ( v42 == -2 && !v39 )
            v39 = (int *)v41;
          v40 = v37 & (v38 + v40);
          v41 = &v36[4 * v40];
          v42 = *v41;
          if ( v17 == *v41 )
            goto LABEL_45;
          ++v38;
        }
        if ( v39 )
          v41 = (unsigned int *)v39;
      }
LABEL_45:
      v43 = *result;
      result += 4;
      *v41 = v43;
      *((_QWORD *)v41 + 1) = *((_QWORD *)result - 1);
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
    while ( v31 != result );
  }
  return result;
}
