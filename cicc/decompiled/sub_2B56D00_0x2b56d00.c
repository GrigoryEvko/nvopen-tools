// Function: sub_2B56D00
// Address: 0x2b56d00
//
void __fastcall sub_2B56D00(__int64 a1, unsigned int a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r14d
  __int64 v8; // r13
  char v9; // cl
  unsigned __int64 v10; // rax
  unsigned int v11; // ebx
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // r14
  bool v15; // zf
  __int64 v16; // r15
  _QWORD *v17; // rax
  __int64 v18; // rdx
  _QWORD *i; // rdx
  __int64 j; // rbx
  __int64 v21; // rsi
  __int64 v22; // rcx
  int v23; // r11d
  __int64 *v24; // r10
  __int64 v25; // rdx
  __int64 *v26; // rdi
  __int64 v27; // r9
  unsigned __int64 v28; // rdi
  __int64 v29; // rax
  int v30; // ecx
  char **v31; // rbx
  char **v32; // r13
  char **v33; // r15
  char *v34; // rcx
  __int64 v35; // rcx
  unsigned __int64 v36; // rdi
  __int64 v37; // rax
  char **v38; // rax
  __int64 v39; // rdx
  char **k; // rdx
  char **v41; // rbx
  __int64 v42; // r8
  int v43; // esi
  int v44; // r10d
  __int64 v45; // r9
  unsigned int v46; // edx
  __int64 *v47; // rdi
  __int64 v48; // rcx
  __int64 v49; // rdx
  unsigned __int64 v50; // rdi
  int v51; // edx
  char **v52; // [rsp+8h] [rbp-158h]
  _BYTE v53[336]; // [rsp+10h] [rbp-150h] BYREF

  v6 = a2;
  v8 = *(_QWORD *)(a1 + 16);
  v9 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    if ( !v9 )
    {
      v11 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    v31 = (char **)(a1 + 304);
    v52 = (char **)(a1 + 16);
  }
  else
  {
    a3 = ((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
        | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1)) >> 16;
    v10 = (a3
         | (((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
           | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
           | (a2 - 1)
           | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
         | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
         | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
         | (a2 - 1)
         | ((unsigned __int64)(a2 - 1) >> 1))
        + 1;
    v6 = v10;
    if ( (unsigned int)v10 > 0x40 )
    {
      v31 = (char **)(a1 + 304);
      v52 = (char **)(a1 + 16);
      if ( !v9 )
      {
        v11 = *(_DWORD *)(a1 + 24);
        v12 = 72LL * (unsigned int)v10;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v9 )
      {
        v11 = *(_DWORD *)(a1 + 24);
        v6 = 64;
        v12 = 4608;
LABEL_5:
        v13 = sub_C7D670(v12, 8);
        *(_DWORD *)(a1 + 24) = v6;
        *(_QWORD *)(a1 + 16) = v13;
LABEL_8:
        v14 = 72LL * v11;
        v15 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v16 = v8 + v14;
        if ( v15 )
        {
          v17 = *(_QWORD **)(a1 + 16);
          v18 = 9LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v17 = (_QWORD *)(a1 + 16);
          v18 = 36;
        }
        for ( i = &v17[v18]; i != v17; v17 += 9 )
        {
          if ( v17 )
            *v17 = -4096;
        }
        for ( j = v8; v16 != j; j += 72 )
        {
          v29 = *(_QWORD *)j;
          if ( *(_QWORD *)j != -4096 && v29 != -8192 )
          {
            if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
            {
              v21 = a1 + 16;
              v22 = 3;
            }
            else
            {
              v30 = *(_DWORD *)(a1 + 24);
              v21 = *(_QWORD *)(a1 + 16);
              if ( !v30 )
                goto LABEL_86;
              v22 = (unsigned int)(v30 - 1);
            }
            v23 = 1;
            v24 = 0;
            v25 = (unsigned int)v22 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
            v26 = (__int64 *)(v21 + 72 * v25);
            v27 = *v26;
            if ( *v26 != v29 )
            {
              while ( v27 != -4096 )
              {
                if ( v27 == -8192 && !v24 )
                  v24 = v26;
                a5 = (unsigned int)(v23 + 1);
                v25 = (unsigned int)v22 & (v23 + (_DWORD)v25);
                v26 = (__int64 *)(v21 + 72LL * (unsigned int)v25);
                v27 = *v26;
                if ( v29 == *v26 )
                  goto LABEL_18;
                ++v23;
              }
              if ( v24 )
                v26 = v24;
            }
LABEL_18:
            *v26 = v29;
            v26[1] = (__int64)(v26 + 3);
            v26[2] = 0x600000000LL;
            if ( *(_DWORD *)(j + 16) )
              sub_2B0A420((__int64)(v26 + 1), (char **)(j + 8), v25, v22, a5, v27);
            *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
            v28 = *(_QWORD *)(j + 8);
            if ( v28 != j + 24 )
              _libc_free(v28);
          }
        }
        sub_C7D6A0(v8, v14, 8);
        return;
      }
      v31 = (char **)(a1 + 304);
      v6 = 64;
      v52 = (char **)(a1 + 16);
    }
  }
  v32 = v52;
  v33 = (char **)v53;
  do
  {
    v34 = *v32;
    if ( *v32 != (char *)-4096LL && v34 != (char *)-8192LL )
    {
      if ( v33 )
        *v33 = v34;
      v33[1] = (char *)(v33 + 3);
      v35 = *((unsigned int *)v32 + 4);
      v33[2] = (char *)0x600000000LL;
      if ( (_DWORD)v35 )
        sub_2B0A420((__int64)(v33 + 1), v32 + 1, a3, v35, a5, a6);
      v36 = (unsigned __int64)v32[1];
      v33 += 9;
      if ( (char **)v36 != v32 + 3 )
        _libc_free(v36);
    }
    v32 += 9;
  }
  while ( v32 != v31 );
  if ( v6 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v37 = sub_C7D670(72LL * v6, 8);
    *(_DWORD *)(a1 + 24) = v6;
    *(_QWORD *)(a1 + 16) = v37;
  }
  v15 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v15 )
  {
    v38 = *(char ***)(a1 + 16);
    v39 = 9LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v38 = v52;
    v39 = 36;
  }
  for ( k = &v38[v39]; k != v38; v38 += 9 )
  {
    if ( v38 )
      *v38 = (char *)-4096LL;
  }
  v41 = (char **)v53;
  if ( v33 != (char **)v53 )
  {
    do
    {
      v29 = (__int64)*v41;
      if ( *v41 != (char *)-8192LL && v29 != -4096 )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v42 = (__int64)v52;
          v43 = 3;
        }
        else
        {
          v51 = *(_DWORD *)(a1 + 24);
          v42 = *(_QWORD *)(a1 + 16);
          if ( !v51 )
          {
LABEL_86:
            MEMORY[0] = v29;
            BUG();
          }
          v43 = v51 - 1;
        }
        v44 = 1;
        v45 = 0;
        v46 = v43 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
        v47 = (__int64 *)(v42 + 72LL * v46);
        v48 = *v47;
        if ( v29 != *v47 )
        {
          while ( v48 != -4096 )
          {
            if ( v48 == -8192 && !v45 )
              v45 = (__int64)v47;
            v46 = v43 & (v44 + v46);
            v47 = (__int64 *)(v42 + 72LL * v46);
            v48 = *v47;
            if ( v29 == *v47 )
              goto LABEL_53;
            ++v44;
          }
          if ( v45 )
            v47 = (__int64 *)v45;
        }
LABEL_53:
        *v47 = v29;
        v47[1] = (__int64)(v47 + 3);
        v47[2] = 0x600000000LL;
        v49 = *((unsigned int *)v41 + 4);
        if ( (_DWORD)v49 )
          sub_2B0A420((__int64)(v47 + 1), v41 + 1, v49, v48, v42, v45);
        v50 = (unsigned __int64)v41[1];
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        if ( (char **)v50 != v41 + 3 )
          _libc_free(v50);
      }
      v41 += 9;
    }
    while ( v33 != v41 );
  }
}
