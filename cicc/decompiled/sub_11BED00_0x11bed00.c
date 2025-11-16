// Function: sub_11BED00
// Address: 0x11bed00
//
unsigned int *__fastcall sub_11BED00(__int64 a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  unsigned __int64 v8; // rax
  unsigned int v9; // ebx
  __int64 v10; // rdi
  bool v11; // zf
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 k; // rdx
  __int64 m; // rbx
  __int64 v16; // r13
  __int64 v17; // rcx
  int v18; // r12d
  unsigned int v19; // r14d
  unsigned int v20; // eax
  __int64 v21; // r9
  unsigned __int64 v22; // rsi
  __int64 v23; // r8
  unsigned int n; // eax
  __int64 v25; // rdi
  __int64 v26; // rdx
  int v27; // edx
  unsigned int *result; // rax
  __int64 v29; // rdi
  unsigned int *v30; // r13
  unsigned int *v31; // rbx
  unsigned int *v32; // r12
  __int64 v33; // rcx
  unsigned int *v34; // rdi
  __int64 v35; // rax
  unsigned int *v36; // rax
  __int64 v37; // rcx
  unsigned int *i; // rcx
  unsigned int *v39; // rbx
  __int64 v40; // r14
  __int64 v41; // r15
  unsigned int *v42; // rsi
  int v43; // r13d
  unsigned int v44; // r12d
  unsigned int v45; // eax
  __int64 v46; // rcx
  __int64 v47; // r8
  int v48; // r10d
  __int64 v49; // r9
  int j; // eax
  __int64 v51; // rdi
  __int64 v52; // rdx
  int v53; // eax
  int v54; // ecx
  __int64 v55; // rdx
  unsigned int *v56; // rdi
  int v57; // eax
  __int64 v58; // [rsp+8h] [rbp-568h]
  unsigned int *v59; // [rsp+8h] [rbp-568h]
  __int64 v60; // [rsp+10h] [rbp-560h]
  __int64 v61; // [rsp+20h] [rbp-550h]
  unsigned int *v62; // [rsp+20h] [rbp-550h]
  unsigned int v63; // [rsp+28h] [rbp-548h]
  __int64 v64; // [rsp+28h] [rbp-548h]
  unsigned int v65; // [rsp+3Ch] [rbp-534h] BYREF
  unsigned int v66[332]; // [rsp+40h] [rbp-530h] BYREF

  v63 = (unsigned int)a2;
  v60 = *(_QWORD *)(a1 + 16);
  v7 = *(_BYTE *)(a1 + 8) & 1;
  if ( (unsigned int)a2 <= 0x10 )
  {
    if ( !(_BYTE)v7 )
    {
      v9 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    v30 = (unsigned int *)(a1 + 1296);
    v59 = (unsigned int *)(a1 + 16);
    goto LABEL_41;
  }
  v8 = ((((((((((unsigned int)((_DWORD)a2 - 1) | ((unsigned __int64)(unsigned int)((_DWORD)a2 - 1) >> 1)) >> 2)
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
  v63 = v8;
  if ( (unsigned int)v8 > 0x40 )
  {
    a2 = (unsigned int *)(a1 + 16);
    v30 = (unsigned int *)(a1 + 1296);
    v59 = (unsigned int *)(a1 + 16);
    if ( !(_BYTE)v7 )
    {
      v9 = *(_DWORD *)(a1 + 24);
      v10 = 80LL * (unsigned int)v8;
      goto LABEL_5;
    }
    goto LABEL_41;
  }
  if ( (_BYTE)v7 )
  {
    v30 = (unsigned int *)(a1 + 1296);
    v63 = 64;
    v59 = (unsigned int *)(a1 + 16);
LABEL_41:
    v31 = v59;
    v32 = v66;
    while ( 1 )
    {
      while ( *(_QWORD *)v31 == -4096 )
      {
        if ( v31[2] != 100 )
          goto LABEL_43;
        v31 += 20;
        if ( v31 == v30 )
        {
LABEL_53:
          if ( v63 > 0x10 )
          {
            *(_BYTE *)(a1 + 8) &= ~1u;
            v35 = sub_C7D670(80LL * v63, 8);
            *(_DWORD *)(a1 + 24) = v63;
            *(_QWORD *)(a1 + 16) = v35;
          }
          v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
          *(_QWORD *)(a1 + 8) &= 1uLL;
          if ( v11 )
          {
            v36 = *(unsigned int **)(a1 + 16);
            v37 = 20LL * *(unsigned int *)(a1 + 24);
          }
          else
          {
            v36 = v59;
            v37 = 320;
          }
          for ( i = &v36[v37]; i != v36; v36 += 20 )
          {
            if ( v36 )
            {
              *(_QWORD *)v36 = -4096;
              v36[2] = 100;
            }
          }
          v39 = v66;
          result = &v65;
          if ( v32 == v66 )
            return result;
          v62 = v32;
          v40 = a1;
          while ( 2 )
          {
            v41 = *(_QWORD *)v39;
            if ( *(_QWORD *)v39 == -4096 )
            {
              if ( v39[2] == 100 )
                goto LABEL_81;
            }
            else if ( v41 == -8192 && v39[2] == 101 )
            {
              goto LABEL_81;
            }
            if ( (*(_BYTE *)(v40 + 8) & 1) != 0 )
            {
              v42 = v59;
              v43 = 15;
            }
            else
            {
              v54 = *(_DWORD *)(v40 + 24);
              v42 = *(unsigned int **)(v40 + 16);
              if ( !v54 )
              {
                MEMORY[0] = *(_QWORD *)v39;
                BUG();
              }
              v43 = v54 - 1;
            }
            v44 = v39[2];
            v65 = v44;
            v45 = sub_CF97C0(&v65);
            v48 = 1;
            v49 = 0;
            for ( j = v43
                    & (((0xBF58476D1CE4E5B9LL
                       * (v45 | ((unsigned __int64)(((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4)) << 32))) >> 31)
                     ^ (484763065 * v45)); ; j = v43 & v53 )
            {
              v51 = (__int64)&v42[20 * j];
              v52 = *(_QWORD *)v51;
              if ( v41 == *(_QWORD *)v51 && *(_DWORD *)(v51 + 8) == v44 )
                break;
              if ( v52 == -4096 )
              {
                if ( *(_DWORD *)(v51 + 8) == 100 )
                {
                  if ( v49 )
                    v51 = v49;
                  break;
                }
              }
              else if ( v52 == -8192 && *(_DWORD *)(v51 + 8) == 101 && !v49 )
              {
                v49 = (__int64)&v42[20 * j];
              }
              v53 = v48 + j;
              ++v48;
            }
            *(_QWORD *)v51 = v41;
            *(_DWORD *)(v51 + 8) = v39[2];
            *(_QWORD *)(v51 + 16) = v51 + 32;
            *(_QWORD *)(v51 + 24) = 0x200000000LL;
            v55 = v39[6];
            if ( (_DWORD)v55 )
            {
              v42 = v39 + 4;
              sub_11BDDF0(v51 + 16, (char **)v39 + 2, v55, v46, v47, v49);
            }
            v56 = (unsigned int *)*((_QWORD *)v39 + 2);
            *(_DWORD *)(v40 + 8) = (2 * (*(_DWORD *)(v40 + 8) >> 1) + 2) | *(_DWORD *)(v40 + 8) & 1;
            result = v39 + 8;
            if ( v56 != v39 + 8 )
              result = (unsigned int *)_libc_free(v56, v42);
LABEL_81:
            v39 += 20;
            if ( v62 == v39 )
              return result;
            continue;
          }
        }
      }
      if ( *(_QWORD *)v31 != -8192 || v31[2] != 101 )
      {
LABEL_43:
        if ( v32 )
          *(__m128i *)v32 = _mm_loadu_si128((const __m128i *)v31);
        v33 = v31[6];
        *((_QWORD *)v32 + 3) = 0x200000000LL;
        *((_QWORD *)v32 + 2) = v32 + 8;
        if ( (_DWORD)v33 )
        {
          a2 = v31 + 4;
          sub_11BDDF0((__int64)(v32 + 4), (char **)v31 + 2, v7, v33, a5, a6);
        }
        v34 = (unsigned int *)*((_QWORD *)v31 + 2);
        v32 += 20;
        if ( v34 != v31 + 8 )
          _libc_free(v34, a2);
      }
      v31 += 20;
      if ( v31 == v30 )
        goto LABEL_53;
    }
  }
  v63 = 64;
  v9 = *(_DWORD *)(a1 + 24);
  v10 = 5120;
LABEL_5:
  *(_QWORD *)(a1 + 16) = sub_C7D670(v10, 8);
  *(_DWORD *)(a1 + 24) = v63;
LABEL_8:
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v58 = 80LL * v9;
  v61 = v58 + v60;
  if ( v11 )
  {
    v12 = *(_QWORD *)(a1 + 16);
    v13 = 80LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v12 = a1 + 16;
    v13 = 1280;
  }
  for ( k = v12 + v13; k != v12; v12 += 80 )
  {
    if ( v12 )
    {
      *(_QWORD *)v12 = -4096;
      *(_DWORD *)(v12 + 8) = 100;
    }
  }
  for ( m = v60; v61 != m; m += 80 )
  {
    v16 = *(_QWORD *)m;
    if ( *(_QWORD *)m == -4096 )
    {
      if ( *(_DWORD *)(m + 8) != 100 )
        goto LABEL_17;
    }
    else if ( v16 != -8192 || *(_DWORD *)(m + 8) != 101 )
    {
LABEL_17:
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v17 = a1 + 16;
        v18 = 15;
      }
      else
      {
        v27 = *(_DWORD *)(a1 + 24);
        v17 = *(_QWORD *)(a1 + 16);
        if ( !v27 )
        {
          MEMORY[0] = *(_QWORD *)m;
          BUG();
        }
        v18 = v27 - 1;
      }
      v19 = *(_DWORD *)(m + 8);
      v64 = v17;
      v66[0] = v19;
      v20 = sub_CF97C0(v66);
      v21 = 0;
      v22 = 0xBF58476D1CE4E5B9LL;
      v23 = 1;
      for ( n = v18
              & (((0xBF58476D1CE4E5B9LL
                 * (v20 | ((unsigned __int64)(((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4)) << 32))) >> 31)
               ^ (484763065 * v20)); ; n = v18 & v57 )
      {
        v25 = v64 + 80LL * n;
        v26 = *(_QWORD *)v25;
        if ( v16 == *(_QWORD *)v25 && *(_DWORD *)(v25 + 8) == v19 )
          break;
        if ( v26 == -4096 )
        {
          if ( *(_DWORD *)(v25 + 8) == 100 )
          {
            if ( v21 )
              v25 = v21;
            break;
          }
        }
        else if ( v26 == -8192 && *(_DWORD *)(v25 + 8) == 101 && !v21 )
        {
          v21 = v64 + 80LL * n;
        }
        v57 = v23 + n;
        v23 = (unsigned int)(v23 + 1);
      }
      *(_QWORD *)v25 = v16;
      *(_DWORD *)(v25 + 8) = *(_DWORD *)(m + 8);
      *(_QWORD *)(v25 + 16) = v25 + 32;
      *(_QWORD *)(v25 + 24) = 0x200000000LL;
      if ( *(_DWORD *)(m + 24) )
      {
        v22 = m + 16;
        sub_11BDDF0(v25 + 16, (char **)(m + 16), v26, v64, v23, v21);
      }
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
      v29 = *(_QWORD *)(m + 16);
      if ( v29 != m + 32 )
        _libc_free(v29, v22);
    }
  }
  return (unsigned int *)sub_C7D6A0(v60, v58, 8);
}
