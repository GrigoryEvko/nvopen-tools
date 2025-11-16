// Function: sub_2F29C00
// Address: 0x2f29c00
//
void __fastcall sub_2F29C00(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r14d
  char v8; // cl
  unsigned __int64 v9; // rax
  __int64 v10; // r15
  unsigned int v11; // r12d
  __int64 v12; // rdi
  __int64 v13; // rax
  bool v14; // zf
  unsigned int *v15; // r13
  _DWORD *v16; // rax
  __int64 v17; // rdx
  _DWORD *k; // rdx
  char **v19; // r12
  __int64 v20; // r15
  __int64 v21; // rdx
  __int64 v22; // r9
  int v23; // esi
  unsigned int v24; // edi
  int v25; // r11d
  unsigned int *v26; // r10
  unsigned __int64 v27; // r8
  unsigned int m; // eax
  unsigned int *v29; // rbx
  __int64 v30; // rcx
  int v31; // esi
  char *v32; // rax
  unsigned __int64 v33; // rdi
  char **v34; // rsi
  char **v35; // r15
  char **v36; // r13
  __int64 v37; // rcx
  unsigned __int64 v38; // rdi
  __int64 v39; // rax
  _DWORD *v40; // rax
  __int64 v41; // rdx
  _DWORD *i; // rdx
  unsigned int v43; // edx
  char **v44; // rdi
  __int64 v45; // rcx
  unsigned int v46; // esi
  int v47; // r10d
  __int64 v48; // r9
  unsigned int j; // eax
  char **v50; // r15
  __int64 v51; // r8
  unsigned int v52; // eax
  int v53; // ecx
  char *v54; // rax
  __int64 v55; // rdx
  unsigned __int64 v56; // rdi
  unsigned int v57; // eax
  __int64 v58; // [rsp+8h] [rbp-108h]
  __int64 v60; // [rsp+10h] [rbp-100h]
  __int64 v61; // [rsp+18h] [rbp-F8h]
  char **v62; // [rsp+18h] [rbp-F8h]
  _BYTE v63[240]; // [rsp+20h] [rbp-F0h] BYREF

  v6 = a2;
  v8 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    if ( !v8 )
    {
      v10 = *(_QWORD *)(a1 + 16);
      v11 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    v62 = (char **)(a1 + 16);
    v34 = (char **)(a1 + 208);
    goto LABEL_42;
  }
  v9 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
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
  v6 = v9;
  if ( (unsigned int)v9 > 0x40 )
  {
    v62 = (char **)(a1 + 16);
    v34 = (char **)(a1 + 208);
    if ( !v8 )
    {
      v10 = *(_QWORD *)(a1 + 16);
      v11 = *(_DWORD *)(a1 + 24);
      v12 = 48LL * (unsigned int)v9;
      goto LABEL_5;
    }
    goto LABEL_42;
  }
  if ( v8 )
  {
    v6 = 64;
    v62 = (char **)(a1 + 16);
    v34 = (char **)(a1 + 208);
LABEL_42:
    v35 = v62;
    v36 = (char **)v63;
    while ( 1 )
    {
      if ( *(_DWORD *)v35 == -1 )
      {
        if ( *((_DWORD *)v35 + 1) != -1 )
          goto LABEL_45;
      }
      else if ( *(_DWORD *)v35 != -2 || *((_DWORD *)v35 + 1) != -2 )
      {
LABEL_45:
        if ( v36 )
          *v36 = *v35;
        v37 = *((unsigned int *)v35 + 4);
        v36[2] = (char *)0x200000000LL;
        v36[1] = (char *)(v36 + 3);
        if ( (_DWORD)v37 )
          sub_2F29500((__int64)(v36 + 1), v35 + 1, (__int64)(v36 + 3), v37, a5, a6);
        v38 = (unsigned __int64)v35[1];
        v36 += 6;
        *(v36 - 1) = v35[5];
        if ( (char **)v38 != v35 + 3 )
          _libc_free(v38);
      }
      v35 += 6;
      if ( v35 == v34 )
      {
        if ( v6 > 4 )
        {
          *(_BYTE *)(a1 + 8) &= ~1u;
          v39 = sub_C7D670(48LL * v6, 8);
          *(_DWORD *)(a1 + 24) = v6;
          *(_QWORD *)(a1 + 16) = v39;
        }
        v14 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        if ( v14 )
        {
          v40 = *(_DWORD **)(a1 + 16);
          v41 = 12LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v40 = v62;
          v41 = 48;
        }
        for ( i = &v40[v41]; i != v40; v40 += 12 )
        {
          if ( v40 )
          {
            *v40 = -1;
            v40[1] = -1;
          }
        }
        v19 = (char **)v63;
        if ( v36 == (char **)v63 )
          return;
        while ( 2 )
        {
          v43 = *(_DWORD *)v19;
          if ( *(_DWORD *)v19 == -1 )
          {
            if ( *((_DWORD *)v19 + 1) == -1 )
              goto LABEL_74;
          }
          else if ( v43 == -2 && *((_DWORD *)v19 + 1) == -2 )
          {
            goto LABEL_74;
          }
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v44 = v62;
            v45 = 3;
          }
          else
          {
            v53 = *(_DWORD *)(a1 + 24);
            v44 = *(char ***)(a1 + 16);
            if ( !v53 )
            {
LABEL_102:
              MEMORY[0] = *v19;
              BUG();
            }
            v45 = (unsigned int)(v53 - 1);
          }
          v46 = *((_DWORD *)v19 + 1);
          v47 = 1;
          v48 = 0;
          for ( j = v45
                  & (((0xBF58476D1CE4E5B9LL * ((37 * v46) | ((unsigned __int64)(37 * v43) << 32))) >> 31)
                   ^ (756364221 * v46)); ; j = v45 & v52 )
          {
            v50 = &v44[6 * j];
            v51 = *(unsigned int *)v50;
            if ( (char *)__PAIR64__(v46, v43) == *v50 )
              break;
            if ( (_DWORD)v51 == -1 )
            {
              if ( *((_DWORD *)v50 + 1) == -1 )
              {
                if ( v48 )
                  v50 = (char **)v48;
                break;
              }
            }
            else if ( (_DWORD)v51 == -2 && *((_DWORD *)v50 + 1) == -2 && !v48 )
            {
              v48 = (__int64)&v44[6 * j];
            }
            v52 = v47 + j;
            ++v47;
          }
          v54 = *v19;
          v50[2] = (char *)0x200000000LL;
          *v50 = v54;
          v50[1] = (char *)(v50 + 3);
          v55 = *((unsigned int *)v19 + 4);
          if ( (_DWORD)v55 )
            sub_2F29500((__int64)(v50 + 1), v19 + 1, v55, v45, v51, v48);
          v50[5] = v19[5];
          v56 = (unsigned __int64)v19[1];
          *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
          if ( (char **)v56 != v19 + 3 )
            _libc_free(v56);
LABEL_74:
          v19 += 6;
          if ( v36 == v19 )
            return;
          continue;
        }
      }
    }
  }
  v10 = *(_QWORD *)(a1 + 16);
  v11 = *(_DWORD *)(a1 + 24);
  v6 = 64;
  v12 = 3072;
LABEL_5:
  v13 = sub_C7D670(v12, 8);
  *(_DWORD *)(a1 + 24) = v6;
  *(_QWORD *)(a1 + 16) = v13;
LABEL_8:
  v14 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v60 = 48LL * v11;
  v15 = (unsigned int *)(v10 + v60);
  if ( v14 )
  {
    v16 = *(_DWORD **)(a1 + 16);
    v17 = 12LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v16 = (_DWORD *)(a1 + 16);
    v17 = 48;
  }
  for ( k = &v16[v17]; k != v16; v16 += 12 )
  {
    if ( v16 )
    {
      *v16 = -1;
      v16[1] = -1;
    }
  }
  v19 = (char **)v10;
  v61 = a1 + 16;
  if ( v15 != (unsigned int *)v10 )
  {
    v58 = v10;
    v20 = a1;
    while ( 1 )
    {
      v21 = *(unsigned int *)v19;
      if ( (_DWORD)v21 == -1 )
      {
        if ( *((_DWORD *)v19 + 1) != -1 )
          goto LABEL_18;
      }
      else if ( (_DWORD)v21 != -2 || *((_DWORD *)v19 + 1) != -2 )
      {
LABEL_18:
        if ( (*(_BYTE *)(v20 + 8) & 1) != 0 )
        {
          v22 = v61;
          v23 = 3;
        }
        else
        {
          v31 = *(_DWORD *)(v20 + 24);
          v22 = *(_QWORD *)(v20 + 16);
          if ( !v31 )
            goto LABEL_102;
          v23 = v31 - 1;
        }
        v24 = *((_DWORD *)v19 + 1);
        v25 = 1;
        v26 = 0;
        v27 = (0xBF58476D1CE4E5B9LL * ((37 * v24) | ((unsigned __int64)(unsigned int)(37 * v21) << 32))) >> 31;
        for ( m = v23 & (v27 ^ (756364221 * v24)); ; m = v23 & v57 )
        {
          v29 = (unsigned int *)(v22 + 48LL * m);
          v30 = *v29;
          if ( __PAIR64__(v24, v21) == *(_QWORD *)v29 )
            break;
          if ( (_DWORD)v30 == -1 )
          {
            if ( v29[1] == -1 )
            {
              if ( v26 )
                v29 = v26;
              break;
            }
          }
          else if ( (_DWORD)v30 == -2 && v29[1] == -2 && !v26 )
          {
            v26 = (unsigned int *)(v22 + 48LL * m);
          }
          v57 = v25 + m;
          ++v25;
        }
        v32 = *v19;
        *((_QWORD *)v29 + 2) = 0x200000000LL;
        *(_QWORD *)v29 = v32;
        *((_QWORD *)v29 + 1) = v29 + 6;
        if ( *((_DWORD *)v19 + 4) )
          sub_2F29500((__int64)(v29 + 2), v19 + 1, v21, v30, v27, v22);
        *((_QWORD *)v29 + 5) = v19[5];
        *(_DWORD *)(v20 + 8) = (2 * (*(_DWORD *)(v20 + 8) >> 1) + 2) | *(_DWORD *)(v20 + 8) & 1;
        v33 = (unsigned __int64)v19[1];
        if ( (char **)v33 != v19 + 3 )
          _libc_free(v33);
      }
      v19 += 6;
      if ( v15 == (unsigned int *)v19 )
      {
        v10 = v58;
        break;
      }
    }
  }
  sub_C7D6A0(v10, v60, 8);
}
