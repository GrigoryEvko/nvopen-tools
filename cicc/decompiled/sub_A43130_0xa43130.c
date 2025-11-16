// Function: sub_A43130
// Address: 0xa43130
//
int *__fastcall sub_A43130(__int64 a1, unsigned int a2)
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
  int *v12; // rdi
  _DWORD *v13; // rax
  __int64 v14; // rdx
  _DWORD *i; // rdx
  int *j; // rax
  unsigned int v17; // edx
  unsigned int *v18; // rcx
  unsigned int v19; // r10d
  int v20; // r9d
  __int64 v21; // r11
  unsigned int v22; // r8d
  int v23; // r14d
  unsigned int *v24; // r13
  __int64 v25; // rdx
  int *result; // rax
  int v27; // edx
  _DWORD *v28; // r13
  int *v29; // r15
  __int64 v30; // rax
  int v31; // ecx
  __int64 v32; // rdi
  __int64 v33; // rax
  _DWORD *v34; // rax
  __int64 v35; // rdx
  _DWORD *k; // rdx
  int v37; // ecx
  int v38; // edi
  _DWORD *v39; // r9
  int v40; // edi
  unsigned int v41; // esi
  _DWORD *v42; // rdx
  int v43; // r8d
  int v44; // r11d
  _DWORD *v45; // r10
  __int64 v46; // rcx
  int v47; // eax
  int v48; // [rsp+0h] [rbp-40h] BYREF
  __int64 v49; // [rsp+4h] [rbp-3Ch]
  int v50; // [rsp+Ch] [rbp-34h]
  _BYTE v51[48]; // [rsp+10h] [rbp-30h] BYREF

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
        v8 = 16LL * (unsigned int)v4;
        goto LABEL_5;
      }
      v27 = *(_DWORD *)(a1 + 16);
      v28 = (_DWORD *)(a1 + 16);
      if ( v27 == -1 )
      {
        v32 = 16LL * (unsigned int)v4;
        v29 = &v48;
        goto LABEL_43;
      }
      v29 = &v48;
      if ( v27 != -2 )
      {
        v30 = *(_QWORD *)(a1 + 20);
        v48 = *(_DWORD *)(a1 + 16);
        v29 = (int *)v51;
        v49 = v30;
        v50 = *(_DWORD *)(a1 + 28);
      }
    }
    else
    {
      if ( !v3 )
      {
        v6 = *(int **)(a1 + 16);
        v7 = *(_DWORD *)(a1 + 24);
        v5 = 64;
        v8 = 1024;
LABEL_5:
        v9 = sub_C7D670(v8, 4);
        *(_DWORD *)(a1 + 24) = v5;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_6:
        v10 = 16LL * v7;
        v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v12 = &v6[(unsigned __int64)v10 / 4];
        if ( v11 )
        {
          v13 = *(_DWORD **)(a1 + 16);
          v14 = 4LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v13 = (_DWORD *)(a1 + 16);
          v14 = 4;
        }
        for ( i = &v13[v14]; i != v13; v13 += 4 )
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
            j += 4;
            if ( v12 == j )
              return (int *)sub_C7D6A0(v6, v10, 4);
          }
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v18 = (unsigned int *)(a1 + 16);
            v19 = 0;
            v20 = 0;
            v21 = a1 + 16;
          }
          else
          {
            v31 = *(_DWORD *)(a1 + 24);
            v21 = *(_QWORD *)(a1 + 16);
            if ( !v31 )
              goto LABEL_77;
            v20 = v31 - 1;
            v19 = (v31 - 1) & (37 * v17);
            v18 = (unsigned int *)(v21 + 16LL * v19);
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
              v18 = (unsigned int *)(v21 + 16LL * v19);
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
          v25 = *(_QWORD *)(j + 1);
          j += 4;
          *(_QWORD *)(v18 + 1) = v25;
          v18[3] = *(j - 1);
        }
        return (int *)sub_C7D6A0(v6, v10, 4);
      }
      v28 = (_DWORD *)(a1 + 16);
      if ( *(_DWORD *)(a1 + 16) > 0xFFFFFFFD )
      {
        v32 = 1024;
        v5 = 64;
        v29 = &v48;
        goto LABEL_43;
      }
      v48 = *(_DWORD *)(a1 + 16);
      v29 = (int *)v51;
      v5 = 64;
      v49 = *(_QWORD *)(a1 + 20);
      v50 = *(_DWORD *)(a1 + 28);
    }
    v32 = 16LL * v5;
LABEL_43:
    *(_BYTE *)(a1 + 8) &= ~1u;
    v33 = sub_C7D670(v32, 4);
    *(_DWORD *)(a1 + 24) = v5;
    *(_QWORD *)(a1 + 16) = v33;
    goto LABEL_44;
  }
  if ( !v3 )
  {
    v6 = *(int **)(a1 + 16);
    v7 = *(_DWORD *)(a1 + 24);
    *(_BYTE *)(a1 + 8) |= 1u;
    goto LABEL_6;
  }
  v47 = *(_DWORD *)(a1 + 16);
  v28 = (_DWORD *)(a1 + 16);
  if ( v47 == -1 || v47 == -2 )
  {
    v29 = &v48;
  }
  else
  {
    v48 = *(_DWORD *)(a1 + 16);
    v29 = (int *)v51;
    v49 = *(_QWORD *)(a1 + 20);
    v50 = *(_DWORD *)(a1 + 28);
  }
LABEL_44:
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v11 )
  {
    v34 = *(_DWORD **)(a1 + 16);
    v35 = 4LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v34 = v28;
    v35 = 4;
  }
  for ( k = &v34[v35]; k != v34; v34 += 4 )
  {
    if ( v34 )
      *v34 = -1;
  }
  for ( result = &v48; result != v29; *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2)
                                                          | *(_DWORD *)(a1 + 8) & 1 )
  {
    while ( 1 )
    {
      v37 = *result;
      if ( (unsigned int)*result <= 0xFFFFFFFD )
        break;
      result += 4;
      if ( result == v29 )
        return result;
    }
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v42 = v28;
      v39 = v28;
      v41 = 0;
      v40 = 0;
    }
    else
    {
      v38 = *(_DWORD *)(a1 + 24);
      v39 = *(_DWORD **)(a1 + 16);
      if ( !v38 )
      {
LABEL_77:
        MEMORY[0] = 0;
        BUG();
      }
      v40 = v38 - 1;
      v41 = v40 & (37 * v37);
      v42 = &v39[4 * v41];
    }
    v43 = *v42;
    v44 = 1;
    v45 = 0;
    if ( v37 != *v42 )
    {
      while ( v43 != -1 )
      {
        if ( v43 == -2 && !v45 )
          v45 = v42;
        v41 = v40 & (v44 + v41);
        v42 = &v39[4 * v41];
        v43 = *v42;
        if ( v37 == *v42 )
          goto LABEL_58;
        ++v44;
      }
      if ( v45 )
        v42 = v45;
    }
LABEL_58:
    *v42 = v37;
    v46 = *(_QWORD *)(result + 1);
    result += 4;
    *(_QWORD *)(v42 + 1) = v46;
    v42[3] = *(result - 1);
  }
  return result;
}
