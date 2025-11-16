// Function: sub_FAB660
// Address: 0xfab660
//
_QWORD *__fastcall sub_FAB660(__int64 a1, _QWORD *a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r13d
  __int64 v8; // r14
  char v9; // cl
  unsigned __int64 v10; // rax
  unsigned int v11; // ebx
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // r13
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
  __int64 v28; // rdi
  _QWORD *result; // rax
  int v30; // ecx
  char **v31; // rbx
  char **v32; // r14
  char **v33; // r15
  char *v34; // rcx
  __int64 v35; // rcx
  char **v36; // rdi
  __int64 v37; // rax
  __int64 v38; // rdx
  _QWORD *k; // rdx
  char **v40; // rbx
  __int64 v41; // r8
  __int64 v42; // rsi
  int v43; // r10d
  __int64 v44; // r9
  unsigned int v45; // edx
  __int64 *v46; // rdi
  __int64 v47; // rcx
  __int64 v48; // rdx
  char **v49; // rdi
  int v50; // esi
  char **v51; // [rsp+8h] [rbp-118h]
  _BYTE v52[272]; // [rsp+10h] [rbp-110h] BYREF

  v6 = (unsigned int)a2;
  v8 = *(_QWORD *)(a1 + 16);
  v9 = *(_BYTE *)(a1 + 8) & 1;
  if ( (unsigned int)a2 <= 4 )
  {
    if ( !v9 )
    {
      v11 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    v31 = (char **)(a1 + 240);
    v51 = (char **)(a1 + 16);
  }
  else
  {
    a3 = ((((((((unsigned int)((_DWORD)a2 - 1) | ((unsigned __int64)(unsigned int)((_DWORD)a2 - 1) >> 1)) >> 2)
            | (unsigned int)((_DWORD)a2 - 1)
            | ((unsigned __int64)(unsigned int)((_DWORD)a2 - 1) >> 1)) >> 4)
          | (((unsigned int)((_DWORD)a2 - 1) | ((unsigned __int64)(unsigned int)((_DWORD)a2 - 1) >> 1)) >> 2)
          | (unsigned int)((_DWORD)a2 - 1)
          | ((unsigned __int64)(unsigned int)((_DWORD)a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)((_DWORD)a2 - 1) | ((unsigned __int64)(unsigned int)((_DWORD)a2 - 1) >> 1)) >> 2)
          | (unsigned int)((_DWORD)a2 - 1)
          | ((unsigned __int64)(unsigned int)((_DWORD)a2 - 1) >> 1)) >> 4)
        | (((unsigned int)((_DWORD)a2 - 1) | ((unsigned __int64)(unsigned int)((_DWORD)a2 - 1) >> 1)) >> 2)
        | (unsigned int)((_DWORD)a2 - 1)
        | ((unsigned __int64)(unsigned int)((_DWORD)a2 - 1) >> 1)) >> 16;
    v10 = (a3
         | (((((((unsigned int)((_DWORD)a2 - 1) | ((unsigned __int64)(unsigned int)((_DWORD)a2 - 1) >> 1)) >> 2)
             | (unsigned int)((_DWORD)a2 - 1)
             | ((unsigned __int64)(unsigned int)((_DWORD)a2 - 1) >> 1)) >> 4)
           | (((unsigned int)((_DWORD)a2 - 1) | ((unsigned __int64)(unsigned int)((_DWORD)a2 - 1) >> 1)) >> 2)
           | (unsigned int)((_DWORD)a2 - 1)
           | ((unsigned __int64)(unsigned int)((_DWORD)a2 - 1) >> 1)) >> 8)
         | (((((unsigned int)((_DWORD)a2 - 1) | ((unsigned __int64)(unsigned int)((_DWORD)a2 - 1) >> 1)) >> 2)
           | (unsigned int)((_DWORD)a2 - 1)
           | ((unsigned __int64)(unsigned int)((_DWORD)a2 - 1) >> 1)) >> 4)
         | (((unsigned int)((_DWORD)a2 - 1) | ((unsigned __int64)(unsigned int)((_DWORD)a2 - 1) >> 1)) >> 2)
         | (unsigned int)((_DWORD)a2 - 1)
         | ((unsigned __int64)(unsigned int)((_DWORD)a2 - 1) >> 1))
        + 1;
    v6 = v10;
    if ( (unsigned int)v10 > 0x40 )
    {
      a2 = (_QWORD *)(a1 + 16);
      v31 = (char **)(a1 + 240);
      v51 = (char **)(a1 + 16);
      if ( !v9 )
      {
        v11 = *(_DWORD *)(a1 + 24);
        v12 = 56LL * (unsigned int)v10;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v9 )
      {
        v11 = *(_DWORD *)(a1 + 24);
        v6 = 64;
        v12 = 3584;
LABEL_5:
        v13 = sub_C7D670(v12, 8);
        *(_DWORD *)(a1 + 24) = v6;
        *(_QWORD *)(a1 + 16) = v13;
LABEL_8:
        v14 = 56LL * v11;
        v15 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v16 = v8 + v14;
        if ( v15 )
        {
          v17 = *(_QWORD **)(a1 + 16);
          v18 = 7LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v17 = (_QWORD *)(a1 + 16);
          v18 = 28;
        }
        for ( i = &v17[v18]; i != v17; v17 += 7 )
        {
          if ( v17 )
            *v17 = -4096;
        }
        for ( j = v8; v16 != j; j += 56 )
        {
          result = *(_QWORD **)j;
          if ( *(_QWORD *)j != -4096 && result != (_QWORD *)-8192LL )
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
            v25 = (unsigned int)v22 & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
            v26 = (__int64 *)(v21 + 56 * v25);
            v27 = *v26;
            if ( (_QWORD *)*v26 != result )
            {
              while ( v27 != -4096 )
              {
                if ( v27 == -8192 && !v24 )
                  v24 = v26;
                a5 = (unsigned int)(v23 + 1);
                v25 = (unsigned int)v22 & (v23 + (_DWORD)v25);
                v26 = (__int64 *)(v21 + 56LL * (unsigned int)v25);
                v27 = *v26;
                if ( result == (_QWORD *)*v26 )
                  goto LABEL_18;
                ++v23;
              }
              if ( v24 )
                v26 = v24;
            }
LABEL_18:
            *v26 = (__int64)result;
            v26[1] = (__int64)(v26 + 3);
            v26[2] = 0x400000000LL;
            if ( *(_DWORD *)(j + 16) )
            {
              v21 = j + 8;
              sub_F8EFD0((__int64)(v26 + 1), (char **)(j + 8), v25, v22, a5, v27);
            }
            *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
            v28 = *(_QWORD *)(j + 8);
            if ( v28 != j + 24 )
              _libc_free(v28, v21);
          }
        }
        return (_QWORD *)sub_C7D6A0(v8, v14, 8);
      }
      v31 = (char **)(a1 + 240);
      v6 = 64;
      v51 = (char **)(a1 + 16);
    }
  }
  v32 = v51;
  v33 = (char **)v52;
  do
  {
    v34 = *v32;
    if ( *v32 != (char *)-4096LL && v34 != (char *)-8192LL )
    {
      if ( v33 )
        *v33 = v34;
      v33[1] = (char *)(v33 + 3);
      v35 = *((unsigned int *)v32 + 4);
      v33[2] = (char *)0x400000000LL;
      if ( (_DWORD)v35 )
      {
        a2 = v32 + 1;
        sub_F8EFD0((__int64)(v33 + 1), v32 + 1, a3, v35, a5, a6);
      }
      v36 = (char **)v32[1];
      v33 += 7;
      if ( v36 != v32 + 3 )
        _libc_free(v36, a2);
    }
    v32 += 7;
  }
  while ( v32 != v31 );
  if ( v6 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v37 = sub_C7D670(56LL * v6, 8);
    *(_DWORD *)(a1 + 24) = v6;
    *(_QWORD *)(a1 + 16) = v37;
  }
  v15 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v15 )
  {
    result = *(_QWORD **)(a1 + 16);
    v38 = 7LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    result = v51;
    v38 = 28;
  }
  for ( k = &result[v38]; k != result; result += 7 )
  {
    if ( result )
      *result = -4096;
  }
  v40 = (char **)v52;
  if ( v33 != (char **)v52 )
  {
    do
    {
      result = *v40;
      if ( *v40 != (char *)-8192LL && result != (_QWORD *)-4096LL )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v41 = (__int64)v51;
          v42 = 3;
        }
        else
        {
          v50 = *(_DWORD *)(a1 + 24);
          v41 = *(_QWORD *)(a1 + 16);
          if ( !v50 )
          {
LABEL_86:
            MEMORY[0] = result;
            BUG();
          }
          v42 = (unsigned int)(v50 - 1);
        }
        v43 = 1;
        v44 = 0;
        v45 = v42 & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
        v46 = (__int64 *)(v41 + 56LL * v45);
        v47 = *v46;
        if ( result != (_QWORD *)*v46 )
        {
          while ( v47 != -4096 )
          {
            if ( v47 == -8192 && !v44 )
              v44 = (__int64)v46;
            v45 = v42 & (v43 + v45);
            v46 = (__int64 *)(v41 + 56LL * v45);
            v47 = *v46;
            if ( result == (_QWORD *)*v46 )
              goto LABEL_53;
            ++v43;
          }
          if ( v44 )
            v46 = (__int64 *)v44;
        }
LABEL_53:
        *v46 = (__int64)result;
        v46[1] = (__int64)(v46 + 3);
        v46[2] = 0x400000000LL;
        v48 = *((unsigned int *)v40 + 4);
        if ( (_DWORD)v48 )
        {
          v42 = (__int64)(v40 + 1);
          sub_F8EFD0((__int64)(v46 + 1), v40 + 1, v48, v47, v41, v44);
        }
        v49 = (char **)v40[1];
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        result = v40 + 3;
        if ( v49 != v40 + 3 )
          result = (_QWORD *)_libc_free(v49, v42);
      }
      v40 += 7;
    }
    while ( v33 != v40 );
  }
  return result;
}
