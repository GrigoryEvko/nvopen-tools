// Function: sub_9DA8C0
// Address: 0x9da8c0
//
_BYTE *__fastcall sub_9DA8C0(__int64 a1, unsigned int a2)
{
  char v4; // si
  unsigned __int64 v5; // rax
  int *v6; // r12
  unsigned int v7; // r13d
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r9
  bool v11; // zf
  int *v12; // rdi
  _DWORD *v13; // rax
  __int64 v14; // rdx
  _DWORD *i; // rdx
  int *j; // rax
  unsigned int v17; // edx
  __int64 v18; // r10
  int v19; // esi
  int v20; // r15d
  int *v21; // r14
  unsigned int v22; // r8d
  int *v23; // rcx
  int v24; // r11d
  __int64 v25; // rdx
  _BYTE *result; // rax
  int v27; // esi
  _DWORD *v28; // r15
  _DWORD *v29; // rcx
  _DWORD *v30; // rax
  _DWORD *v31; // r13
  __int64 v32; // rax
  _DWORD *v33; // rax
  __int64 v34; // rdx
  _DWORD *k; // rdx
  int v36; // edx
  _DWORD *v37; // r8
  int v38; // edi
  int v39; // r12d
  int *v40; // r10
  unsigned int v41; // esi
  int *v42; // rcx
  int v43; // r9d
  __int64 v44; // rdx
  int v45; // ecx
  _BYTE v46[112]; // [rsp+10h] [rbp-70h] BYREF

  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    if ( !v4 )
    {
      v6 = *(int **)(a1 + 16);
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    v28 = (_DWORD *)(a1 + 16);
    v29 = (_DWORD *)(a1 + 80);
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
      v28 = (_DWORD *)(a1 + 16);
      v29 = (_DWORD *)(a1 + 80);
      if ( !v4 )
      {
        v6 = *(int **)(a1 + 16);
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 16LL * (unsigned int)v5;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v4 )
      {
        v6 = *(int **)(a1 + 16);
        v7 = *(_DWORD *)(a1 + 24);
        a2 = 64;
        v8 = 1024;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = a2;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
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
          v14 = 16;
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
              return (_BYTE *)sub_C7D6A0(v6, v10, 8);
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
          v23 = (int *)(v18 + 16LL * v22);
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
          *v23 = v17;
          v25 = *((_QWORD *)j + 1);
          j += 4;
          *((_QWORD *)v23 + 1) = v25;
        }
        return (_BYTE *)sub_C7D6A0(v6, v10, 8);
      }
      v28 = (_DWORD *)(a1 + 16);
      v29 = (_DWORD *)(a1 + 80);
      a2 = 64;
    }
  }
  v30 = v28;
  v31 = v46;
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
  if ( a2 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v32 = sub_C7D670(16LL * a2, 8);
    *(_DWORD *)(a1 + 24) = a2;
    *(_QWORD *)(a1 + 16) = v32;
  }
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v11 )
  {
    v33 = *(_DWORD **)(a1 + 16);
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
  for ( result = v46;
        v31 != (_DWORD *)result;
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1 )
  {
    while ( 1 )
    {
      v36 = *(_DWORD *)result;
      if ( *(_DWORD *)result <= 0xFFFFFFFD )
        break;
      result += 16;
      if ( v31 == (_DWORD *)result )
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
      v37 = *(_DWORD **)(a1 + 16);
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
    v42 = &v37[4 * v41];
    v43 = *v42;
    if ( v36 != *v42 )
    {
      while ( v43 != -1 )
      {
        if ( v43 == -2 && !v40 )
          v40 = v42;
        v41 = v38 & (v39 + v41);
        v42 = &v37[4 * v41];
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
    v44 = *((_QWORD *)result + 1);
    result += 16;
    *((_QWORD *)v42 + 1) = v44;
  }
  return result;
}
