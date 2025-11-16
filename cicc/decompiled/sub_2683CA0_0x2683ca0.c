// Function: sub_2683CA0
// Address: 0x2683ca0
//
void __fastcall sub_2683CA0(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r14d
  __int64 v4; // r13
  char v5; // dl
  unsigned __int64 v6; // rax
  unsigned int v7; // ebx
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r14
  bool v11; // zf
  __int64 v12; // r15
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *i; // rdx
  __int64 j; // rbx
  __int64 v17; // r8
  int v18; // edi
  int v19; // r10d
  _QWORD *v20; // r9
  unsigned int v21; // edx
  _QWORD *v22; // rsi
  __int64 v23; // rcx
  __int64 v24; // rax
  int v25; // edx
  _QWORD *v26; // r15
  _QWORD *v27; // rbx
  _QWORD *v28; // r13
  __int64 v29; // rdx
  __int64 v30; // rdi
  void *v31; // rsi
  __int64 v32; // rax
  _QWORD *v33; // rax
  __int64 v34; // rdx
  _QWORD *k; // rdx
  _BYTE *m; // rbx
  _QWORD *v37; // r8
  int v38; // edi
  int v39; // r10d
  _QWORD *v40; // r9
  unsigned int v41; // edx
  _QWORD *v42; // rsi
  __int64 v43; // rcx
  int v44; // edx
  _QWORD *v45; // [rsp+8h] [rbp-158h]
  _BYTE v46[336]; // [rsp+10h] [rbp-150h] BYREF

  v2 = a2;
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    if ( !v5 )
    {
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_6;
    }
    v26 = (_QWORD *)(a1 + 304);
    v45 = (_QWORD *)(a1 + 16);
  }
  else
  {
    v6 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
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
    v2 = v6;
    if ( (unsigned int)v6 > 0x40 )
    {
      v26 = (_QWORD *)(a1 + 304);
      v45 = (_QWORD *)(a1 + 16);
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 72LL * (unsigned int)v6;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v2 = 64;
        v8 = 4608;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v2;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_6:
        v10 = 72LL * v7;
        v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v12 = v4 + v10;
        if ( v11 )
        {
          v13 = *(_QWORD **)(a1 + 16);
          v14 = 9LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v13 = (_QWORD *)(a1 + 16);
          v14 = 36;
        }
        for ( i = &v13[v14]; i != v13; v13 += 9 )
        {
          if ( v13 )
            *v13 = -4096;
        }
        for ( j = v4; v12 != j; j += 72 )
        {
          v24 = *(_QWORD *)j;
          if ( *(_QWORD *)j != -4096 && v24 != -8192 )
          {
            if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
            {
              v17 = a1 + 16;
              v18 = 3;
            }
            else
            {
              v25 = *(_DWORD *)(a1 + 24);
              v17 = *(_QWORD *)(a1 + 16);
              if ( !v25 )
                goto LABEL_80;
              v18 = v25 - 1;
            }
            v19 = 1;
            v20 = 0;
            v21 = v18 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
            v22 = (_QWORD *)(v17 + 72LL * v21);
            v23 = *v22;
            if ( *v22 != v24 )
            {
              while ( v23 != -4096 )
              {
                if ( v23 == -8192 && !v20 )
                  v20 = v22;
                v21 = v18 & (v19 + v21);
                v22 = (_QWORD *)(v17 + 72LL * v21);
                v23 = *v22;
                if ( v24 == *v22 )
                  goto LABEL_16;
                ++v19;
              }
              if ( v20 )
                v22 = v20;
            }
LABEL_16:
            *v22 = v24;
            sub_C8CF70((__int64)(v22 + 1), v22 + 5, 4, j + 40, j + 8);
            *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
            if ( !*(_BYTE *)(j + 36) )
              _libc_free(*(_QWORD *)(j + 16));
          }
        }
        sub_C7D6A0(v4, v10, 8);
        return;
      }
      v26 = (_QWORD *)(a1 + 304);
      v2 = 64;
      v45 = (_QWORD *)(a1 + 16);
    }
  }
  v27 = v45;
  v28 = v46;
  do
  {
    v29 = *v27;
    if ( *v27 != -4096 && v29 != -8192 )
    {
      if ( v28 )
        *v28 = v29;
      v30 = (__int64)(v28 + 1);
      v31 = v28 + 5;
      v28 += 9;
      sub_C8CF70(v30, v31, 4, (__int64)(v27 + 5), (__int64)(v27 + 1));
      if ( !*((_BYTE *)v27 + 36) )
        _libc_free(v27[2]);
    }
    v27 += 9;
  }
  while ( v27 != v26 );
  if ( v2 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v32 = sub_C7D670(72LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v32;
  }
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v11 )
  {
    v33 = *(_QWORD **)(a1 + 16);
    v34 = 9LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v33 = v45;
    v34 = 36;
  }
  for ( k = &v33[v34]; k != v33; v33 += 9 )
  {
    if ( v33 )
      *v33 = -4096;
  }
  for ( m = v46; v28 != (_QWORD *)m; m += 72 )
  {
    v24 = *(_QWORD *)m;
    if ( *(_QWORD *)m != -4096 && v24 != -8192 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v37 = v45;
        v38 = 3;
      }
      else
      {
        v44 = *(_DWORD *)(a1 + 24);
        v37 = *(_QWORD **)(a1 + 16);
        if ( !v44 )
        {
LABEL_80:
          MEMORY[0] = v24;
          BUG();
        }
        v38 = v44 - 1;
      }
      v39 = 1;
      v40 = 0;
      v41 = v38 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v42 = &v37[9 * v41];
      v43 = *v42;
      if ( v24 != *v42 )
      {
        while ( v43 != -4096 )
        {
          if ( v43 == -8192 && !v40 )
            v40 = v42;
          v41 = v38 & (v39 + v41);
          v42 = &v37[9 * v41];
          v43 = *v42;
          if ( v24 == *v42 )
            goto LABEL_49;
          ++v39;
        }
        if ( v40 )
          v42 = v40;
      }
LABEL_49:
      *v42 = v24;
      sub_C8CF70((__int64)(v42 + 1), v42 + 5, 4, (__int64)(m + 40), (__int64)(m + 8));
      v11 = m[36] == 0;
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
      if ( v11 )
        _libc_free(*((_QWORD *)m + 2));
    }
  }
}
