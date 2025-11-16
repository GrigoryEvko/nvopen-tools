// Function: sub_11CE470
// Address: 0x11ce470
//
_QWORD *__fastcall sub_11CE470(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // ebx
  __int64 v8; // r14
  char v9; // cl
  unsigned __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // rdi
  __int64 v13; // r13
  __int64 v14; // rax
  bool v15; // zf
  __int64 v16; // r15
  _QWORD *v17; // rax
  __int64 v18; // rdx
  char **v19; // r13
  __int64 v20; // rdx
  __int64 v21; // r13
  _QWORD *i; // rdx
  __int64 j; // rbx
  __int64 v24; // rsi
  __int64 v25; // rcx
  int v26; // r11d
  __int64 *v27; // r10
  __int64 v28; // rdx
  __int64 *v29; // rdi
  __int64 v30; // r9
  __int64 v31; // rdi
  _QWORD *result; // rax
  int v33; // ecx
  char **v34; // r14
  char **v35; // r15
  char *v36; // rax
  __int64 v37; // rcx
  char **v38; // rdi
  __int64 v39; // rax
  __int64 v40; // rdx
  _QWORD *k; // rdx
  char **v42; // rbx
  __int64 v43; // rcx
  __int64 v44; // rsi
  int v45; // r10d
  __int64 v46; // r9
  unsigned int v47; // edx
  __int64 *v48; // rdi
  __int64 v49; // r8
  __int64 v50; // rdx
  char **v51; // rdi
  int v52; // edx
  __int64 v53; // [rsp+8h] [rbp-B8h]
  __int64 v54; // [rsp+8h] [rbp-B8h]
  _BYTE v55[176]; // [rsp+10h] [rbp-B0h] BYREF

  v7 = (unsigned int)a2;
  v8 = *(_QWORD *)(a1 + 16);
  v9 = *(_BYTE *)(a1 + 8) & 1;
  if ( (unsigned int)a2 <= 4 )
  {
    v19 = (char **)(a1 + 16);
    v20 = a1 + 144;
    if ( !v9 )
    {
      v21 = *(unsigned int *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      v13 = 32 * v21;
      v15 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
      *(_QWORD *)(a1 + 8) &= 1uLL;
      v16 = v8 + v13;
      if ( v15 )
        goto LABEL_6;
      goto LABEL_9;
    }
  }
  else
  {
    v10 = ((((((((((unsigned int)((_DWORD)a2 - 1) | ((unsigned __int64)(unsigned int)((_DWORD)a2 - 1) >> 1)) >> 2)
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
           | ((unsigned __int64)(unsigned int)((_DWORD)a2 - 1) >> 1)) >> 16)
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
    v7 = v10;
    if ( (unsigned int)v10 > 0x40 )
    {
      v19 = (char **)(a1 + 16);
      v20 = a1 + 144;
      if ( !v9 )
      {
        v11 = *(unsigned int *)(a1 + 24);
        v12 = 32LL * (unsigned int)v10;
LABEL_5:
        v13 = 32 * v11;
        v14 = sub_C7D670(v12, 8);
        v15 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v16 = v8 + v13;
        *(_QWORD *)(a1 + 16) = v14;
        *(_DWORD *)(a1 + 24) = v7;
        if ( v15 )
        {
LABEL_6:
          v17 = *(_QWORD **)(a1 + 16);
          v18 = 4LL * *(unsigned int *)(a1 + 24);
LABEL_10:
          for ( i = &v17[v18]; i != v17; v17 += 4 )
          {
            if ( v17 )
              *v17 = -4096;
          }
          for ( j = v8; v16 != j; j += 32 )
          {
            result = *(_QWORD **)j;
            if ( *(_QWORD *)j != -4096 && result != (_QWORD *)-8192LL )
            {
              if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
              {
                v24 = a1 + 16;
                v25 = 3;
              }
              else
              {
                v33 = *(_DWORD *)(a1 + 24);
                v24 = *(_QWORD *)(a1 + 16);
                if ( !v33 )
                  goto LABEL_84;
                v25 = (unsigned int)(v33 - 1);
              }
              v26 = 1;
              v27 = 0;
              v28 = (unsigned int)v25 & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
              v29 = (__int64 *)(v24 + 32 * v28);
              v30 = *v29;
              if ( (_QWORD *)*v29 != result )
              {
                while ( v30 != -4096 )
                {
                  if ( v30 == -8192 && !v27 )
                    v27 = v29;
                  a5 = (unsigned int)(v26 + 1);
                  v28 = (unsigned int)v25 & (v26 + (_DWORD)v28);
                  v29 = (__int64 *)(v24 + 32LL * (unsigned int)v28);
                  v30 = *v29;
                  if ( result == (_QWORD *)*v29 )
                    goto LABEL_18;
                  ++v26;
                }
                if ( v27 )
                  v29 = v27;
              }
LABEL_18:
              *v29 = (__int64)result;
              v29[1] = (__int64)(v29 + 3);
              v29[2] = 0x100000000LL;
              if ( *(_DWORD *)(j + 16) )
              {
                v24 = j + 8;
                sub_11CD810((__int64)(v29 + 1), (char **)(j + 8), v28, v25, a5, v30);
              }
              *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
              v31 = *(_QWORD *)(j + 8);
              if ( v31 != j + 24 )
                _libc_free(v31, v24);
            }
          }
          return (_QWORD *)sub_C7D6A0(v8, v13, 8);
        }
LABEL_9:
        v17 = (_QWORD *)(a1 + 16);
        v18 = 16;
        goto LABEL_10;
      }
    }
    else
    {
      if ( !v9 )
      {
        v11 = *(unsigned int *)(a1 + 24);
        v7 = 64;
        v12 = 2048;
        goto LABEL_5;
      }
      v19 = (char **)(a1 + 16);
      v20 = a1 + 144;
      v7 = 64;
    }
  }
  v34 = v19;
  v35 = (char **)v55;
  do
  {
    v36 = *v34;
    if ( *v34 != (char *)-4096LL && v36 != (char *)-8192LL )
    {
      if ( v35 )
        *v35 = v36;
      v37 = *((unsigned int *)v34 + 4);
      v35[1] = (char *)(v35 + 3);
      v35[2] = (char *)0x100000000LL;
      if ( (_DWORD)v37 )
      {
        a2 = v34 + 1;
        v54 = v20;
        sub_11CD810((__int64)(v35 + 1), v34 + 1, v20, v37, a5, a6);
        v20 = v54;
      }
      v38 = (char **)v34[1];
      v35 += 4;
      if ( v38 != v34 + 3 )
      {
        v53 = v20;
        _libc_free(v38, a2);
        v20 = v53;
      }
    }
    v34 += 4;
  }
  while ( v34 != (char **)v20 );
  if ( v7 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v39 = sub_C7D670(32LL * v7, 8);
    *(_DWORD *)(a1 + 24) = v7;
    *(_QWORD *)(a1 + 16) = v39;
  }
  v15 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v15 )
  {
    result = *(_QWORD **)(a1 + 16);
    v40 = 4LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    result = v19;
    v40 = 16;
  }
  for ( k = &result[v40]; k != result; result += 4 )
  {
    if ( result )
      *result = -4096;
  }
  v42 = (char **)v55;
  if ( v35 != (char **)v55 )
  {
    do
    {
      result = *v42;
      if ( *v42 != (char *)-8192LL && result != (_QWORD *)-4096LL )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v43 = (__int64)v19;
          v44 = 3;
        }
        else
        {
          v52 = *(_DWORD *)(a1 + 24);
          v43 = *(_QWORD *)(a1 + 16);
          if ( !v52 )
          {
LABEL_84:
            MEMORY[0] = result;
            BUG();
          }
          v44 = (unsigned int)(v52 - 1);
        }
        v45 = 1;
        v46 = 0;
        v47 = v44 & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
        v48 = (__int64 *)(v43 + 32LL * v47);
        v49 = *v48;
        if ( result != (_QWORD *)*v48 )
        {
          while ( v49 != -4096 )
          {
            if ( v49 == -8192 && !v46 )
              v46 = (__int64)v48;
            v47 = v44 & (v45 + v47);
            v48 = (__int64 *)(v43 + 32LL * v47);
            v49 = *v48;
            if ( result == (_QWORD *)*v48 )
              goto LABEL_51;
            ++v45;
          }
          if ( v46 )
            v48 = (__int64 *)v46;
        }
LABEL_51:
        *v48 = (__int64)result;
        v48[1] = (__int64)(v48 + 3);
        v48[2] = 0x100000000LL;
        v50 = *((unsigned int *)v42 + 4);
        if ( (_DWORD)v50 )
        {
          v44 = (__int64)(v42 + 1);
          sub_11CD810((__int64)(v48 + 1), v42 + 1, v50, v43, v49, v46);
        }
        v51 = (char **)v42[1];
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        result = v42 + 3;
        if ( v51 != v42 + 3 )
          result = (_QWORD *)_libc_free(v51, v44);
      }
      v42 += 4;
    }
    while ( v35 != v42 );
  }
  return result;
}
