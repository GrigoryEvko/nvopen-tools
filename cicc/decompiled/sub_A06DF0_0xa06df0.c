// Function: sub_A06DF0
// Address: 0xa06df0
//
int *__fastcall sub_A06DF0(__int64 a1, unsigned int a2)
{
  char v3; // cl
  unsigned __int64 v4; // rax
  unsigned int v5; // r14d
  int *v6; // r12
  unsigned int v7; // r13d
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rsi
  bool v11; // zf
  int *v12; // rcx
  _DWORD *v13; // rax
  __int64 v14; // rdx
  _DWORD *i; // rdx
  int *j; // rax
  int v17; // edx
  _DWORD *v18; // rdi
  unsigned int v19; // r9d
  int v20; // r8d
  __int64 v21; // r11
  int v22; // r10d
  int v23; // r14d
  _DWORD *v24; // r13
  int *result; // rax
  int v26; // edx
  _DWORD *v27; // r13
  int *v28; // r15
  int v29; // r8d
  __int64 v30; // rdi
  __int64 v31; // rax
  _DWORD *v32; // rax
  __int64 v33; // rdx
  _DWORD *k; // rdx
  int v35; // edx
  int v36; // ecx
  _DWORD *v37; // r9
  int v38; // ecx
  unsigned int v39; // edi
  _DWORD *v40; // rsi
  int v41; // r8d
  int v42; // r11d
  _DWORD *v43; // r10
  int v44; // eax
  int v45; // [rsp+Ch] [rbp-34h] BYREF
  _BYTE v46[48]; // [rsp+10h] [rbp-30h] BYREF

  v3 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 > 1 )
  {
    v4 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
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
    v5 = v4;
    if ( (unsigned int)v4 > 0x40 )
    {
      if ( !v3 )
      {
        v6 = *(int **)(a1 + 16);
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 4LL * (unsigned int)v4;
        goto LABEL_5;
      }
      v26 = *(_DWORD *)(a1 + 16);
      v27 = (_DWORD *)(a1 + 16);
      if ( v26 == -1 )
      {
        v30 = 4LL * (unsigned int)v4;
        v28 = &v45;
        goto LABEL_43;
      }
      v28 = &v45;
      if ( v26 != -2 )
      {
        v45 = *(_DWORD *)(a1 + 16);
        v28 = (int *)v46;
      }
    }
    else
    {
      if ( !v3 )
      {
        v6 = *(int **)(a1 + 16);
        v7 = *(_DWORD *)(a1 + 24);
        v5 = 64;
        v8 = 256;
LABEL_5:
        v9 = sub_C7D670(v8, 4);
        *(_DWORD *)(a1 + 24) = v5;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_6:
        v10 = 4LL * v7;
        v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v12 = &v6[(unsigned __int64)v10 / 4];
        if ( v11 )
        {
          v13 = *(_DWORD **)(a1 + 16);
          v14 = *(unsigned int *)(a1 + 24);
        }
        else
        {
          v13 = (_DWORD *)(a1 + 16);
          v14 = 1;
        }
        for ( i = &v13[v14]; i != v13; ++v13 )
        {
          if ( v13 )
            *v13 = -1;
        }
        for ( j = v6; v12 != j; *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1 )
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
            v18 = (_DWORD *)(a1 + 16);
            v19 = 0;
            v20 = 0;
            v21 = a1 + 16;
          }
          else
          {
            v29 = *(_DWORD *)(a1 + 24);
            v21 = *(_QWORD *)(a1 + 16);
            if ( !v29 )
              goto LABEL_77;
            v20 = v29 - 1;
            v19 = v20 & (37 * v17);
            v18 = (_DWORD *)(v21 + 4LL * v19);
          }
          v22 = *v18;
          v23 = 1;
          v24 = 0;
          if ( *v18 != v17 )
          {
            while ( v22 != -1 )
            {
              if ( !v24 && v22 == -2 )
                v24 = v18;
              v19 = v20 & (v23 + v19);
              v18 = (_DWORD *)(v21 + 4LL * v19);
              v22 = *v18;
              if ( v17 == *v18 )
                goto LABEL_19;
              ++v23;
            }
            if ( v24 )
              v18 = v24;
          }
LABEL_19:
          *v18 = v17;
          ++j;
        }
        return (int *)sub_C7D6A0(v6, v10, 4);
      }
      v27 = (_DWORD *)(a1 + 16);
      if ( *(_DWORD *)(a1 + 16) > 0xFFFFFFFD )
      {
        v30 = 256;
        v5 = 64;
        v28 = &v45;
        goto LABEL_43;
      }
      v45 = *(_DWORD *)(a1 + 16);
      v5 = 64;
      v28 = (int *)v46;
    }
    v30 = 4LL * v5;
LABEL_43:
    *(_BYTE *)(a1 + 8) &= ~1u;
    v31 = sub_C7D670(v30, 4);
    *(_DWORD *)(a1 + 24) = v5;
    *(_QWORD *)(a1 + 16) = v31;
    goto LABEL_44;
  }
  if ( !v3 )
  {
    v6 = *(int **)(a1 + 16);
    v7 = *(_DWORD *)(a1 + 24);
    *(_BYTE *)(a1 + 8) |= 1u;
    goto LABEL_6;
  }
  v44 = *(_DWORD *)(a1 + 16);
  v27 = (_DWORD *)(a1 + 16);
  if ( v44 == -1 || v44 == -2 )
  {
    v28 = &v45;
  }
  else
  {
    v45 = *(_DWORD *)(a1 + 16);
    v28 = (int *)v46;
  }
LABEL_44:
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v11 )
  {
    v32 = *(_DWORD **)(a1 + 16);
    v33 = *(unsigned int *)(a1 + 24);
  }
  else
  {
    v32 = v27;
    v33 = 1;
  }
  for ( k = &v32[v33]; k != v32; ++v32 )
  {
    if ( v32 )
      *v32 = -1;
  }
  for ( result = &v45; result != v28; *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2)
                                                          | *(_DWORD *)(a1 + 8) & 1 )
  {
    while ( 1 )
    {
      v35 = *result;
      if ( (unsigned int)*result <= 0xFFFFFFFD )
        break;
      if ( ++result == v28 )
        return result;
    }
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v40 = v27;
      v37 = v27;
      v39 = 0;
      v38 = 0;
    }
    else
    {
      v36 = *(_DWORD *)(a1 + 24);
      v37 = *(_DWORD **)(a1 + 16);
      if ( !v36 )
      {
LABEL_77:
        MEMORY[0] = 0;
        BUG();
      }
      v38 = v36 - 1;
      v39 = v38 & (37 * v35);
      v40 = &v37[v39];
    }
    v41 = *v40;
    v42 = 1;
    v43 = 0;
    if ( v35 != *v40 )
    {
      while ( v41 != -1 )
      {
        if ( v41 == -2 && !v43 )
          v43 = v40;
        v39 = v38 & (v42 + v39);
        v40 = &v37[v39];
        v41 = *v40;
        if ( v35 == *v40 )
          goto LABEL_58;
        ++v42;
      }
      if ( v43 )
        v40 = v43;
    }
LABEL_58:
    *v40 = v35;
    ++result;
  }
  return result;
}
