// Function: sub_1E9CCA0
// Address: 0x1e9cca0
//
void __fastcall sub_1E9CCA0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, int a5, int a6)
{
  char v7; // cl
  unsigned __int64 v8; // rax
  int v9; // r14d
  __int64 v10; // r12
  __int64 v11; // r13
  _QWORD *v12; // r15
  __int64 v13; // rcx
  unsigned __int64 v14; // rdi
  _DWORD *v15; // rax
  __int64 v16; // rcx
  __int64 v17; // rsi
  __int64 v18; // rsi
  _DWORD *v19; // rsi
  _DWORD *v20; // r12
  _BYTE *k; // r13
  int v22; // esi
  _DWORD *v23; // r9
  int v24; // edi
  int v25; // r8d
  int v26; // r11d
  unsigned __int64 v27; // r10
  unsigned __int64 v28; // r10
  int v29; // eax
  int *v30; // r10
  unsigned int m; // eax
  int *v32; // r14
  int v33; // edx
  unsigned int v34; // eax
  unsigned int *v35; // rdx
  int v36; // r13d
  __int64 v37; // rdx
  bool v38; // zf
  unsigned int *v39; // r13
  _DWORD *v40; // rax
  __int64 v41; // rcx
  _DWORD *i; // rcx
  char **v43; // r14
  __int64 v44; // rcx
  __int64 v45; // r8
  int v46; // esi
  unsigned int v47; // edi
  int v48; // r10d
  unsigned __int64 v49; // r9
  unsigned __int64 v50; // r9
  int v51; // eax
  int *v52; // r9
  unsigned int j; // eax
  int *v54; // r15
  int v55; // r11d
  int v56; // esi
  char *v57; // rax
  unsigned __int64 v58; // rdi
  __int64 v59; // rax
  unsigned int v60; // eax
  int v61; // edi
  __int64 v62; // rax
  __int64 v63; // rdx
  unsigned __int64 v64; // rdi
  __int64 v65; // [rsp+10h] [rbp-100h]
  _DWORD *v66; // [rsp+18h] [rbp-F8h]
  unsigned int *v67; // [rsp+18h] [rbp-F8h]
  _BYTE v68[240]; // [rsp+20h] [rbp-F0h] BYREF

  v7 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 3 )
  {
    if ( v7 )
      return;
    v35 = *(unsigned int **)(a1 + 16);
    v36 = *(_DWORD *)(a1 + 24);
    *(_BYTE *)(a1 + 8) |= 1u;
    v67 = v35;
    goto LABEL_40;
  }
  v8 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
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
  v9 = v8;
  if ( (unsigned int)v8 > 0x40 )
  {
    v10 = 12LL * (unsigned int)v8;
    if ( v7 )
      goto LABEL_5;
    v36 = *(_DWORD *)(a1 + 24);
    v67 = *(unsigned int **)(a1 + 16);
    goto LABEL_74;
  }
  if ( !v7 )
  {
    v36 = *(_DWORD *)(a1 + 24);
    v10 = 768;
    v9 = 64;
    v67 = *(unsigned int **)(a1 + 16);
LABEL_74:
    v59 = sub_22077B0(v10 * 4);
    *(_DWORD *)(a1 + 24) = v9;
    *(_QWORD *)(a1 + 16) = v59;
LABEL_40:
    v37 = (__int64)v67;
    v38 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
    *(_QWORD *)(a1 + 8) &= 1uLL;
    v39 = &v67[12 * v36];
    if ( v38 )
    {
      v40 = *(_DWORD **)(a1 + 16);
      v41 = 12LL * *(unsigned int *)(a1 + 24);
    }
    else
    {
      v40 = (_DWORD *)(a1 + 16);
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
    v43 = (char **)v67;
    if ( v39 == v67 )
    {
LABEL_63:
      j___libc_free_0(v67);
      return;
    }
    while ( 1 )
    {
      v44 = *(unsigned int *)v43;
      if ( (_DWORD)v44 == -1 )
      {
        if ( *((_DWORD *)v43 + 1) != -1 )
          goto LABEL_49;
      }
      else if ( (_DWORD)v44 != -2 || *((_DWORD *)v43 + 1) != -2 )
      {
LABEL_49:
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v45 = a1 + 16;
          v46 = 3;
        }
        else
        {
          v56 = *(_DWORD *)(a1 + 24);
          v45 = *(_QWORD *)(a1 + 16);
          if ( !v56 )
          {
            MEMORY[0] = *v43;
            BUG();
          }
          v46 = v56 - 1;
        }
        v47 = *((_DWORD *)v43 + 1);
        v48 = 1;
        v49 = ((((37 * v47) | ((unsigned __int64)(unsigned int)(37 * v44) << 32))
              - 1
              - ((unsigned __int64)(37 * v47) << 32)) >> 22)
            ^ (((37 * v47) | ((unsigned __int64)(unsigned int)(37 * v44) << 32))
             - 1
             - ((unsigned __int64)(37 * v47) << 32));
        v50 = ((9 * (((v49 - 1 - (v49 << 13)) >> 8) ^ (v49 - 1 - (v49 << 13)))) >> 15)
            ^ (9 * (((v49 - 1 - (v49 << 13)) >> 8) ^ (v49 - 1 - (v49 << 13))));
        v51 = ((v50 - 1 - (v50 << 27)) >> 31) ^ (v50 - 1 - ((_DWORD)v50 << 27));
        v52 = 0;
        for ( j = v46 & v51; ; j = v46 & v60 )
        {
          v54 = (int *)(v45 + 48LL * j);
          v55 = *v54;
          if ( __PAIR64__(v47, v44) == *(_QWORD *)v54 )
            break;
          if ( v55 == -1 )
          {
            if ( v54[1] == -1 )
            {
              if ( v52 )
                v54 = v52;
              break;
            }
          }
          else if ( v55 == -2 && v54[1] == -2 && !v52 )
          {
            v52 = (int *)(v45 + 48LL * j);
          }
          v60 = v48 + j;
          ++v48;
        }
        v57 = *v43;
        *((_QWORD *)v54 + 2) = 0x200000000LL;
        *(_QWORD *)v54 = v57;
        *((_QWORD *)v54 + 1) = v54 + 6;
        if ( *((_DWORD *)v43 + 4) )
          sub_1E9C460((__int64)(v54 + 2), v43 + 1, v37, v44, v45, (int)v52);
        *((_QWORD *)v54 + 5) = v43[5];
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        v58 = (unsigned __int64)v43[1];
        if ( (char **)v58 != v43 + 3 )
          _libc_free(v58);
      }
      v43 += 6;
      if ( v39 == (unsigned int *)v43 )
        goto LABEL_63;
    }
  }
  v10 = 768;
  v9 = 64;
LABEL_5:
  v11 = a1 + 16;
  v65 = a1 + 208;
  v66 = (_DWORD *)(a1 + 16);
  v12 = v68;
  do
  {
    if ( *(_DWORD *)v11 == -1 )
    {
      if ( *(_DWORD *)(v11 + 4) == -1 )
        goto LABEL_14;
    }
    else if ( *(_DWORD *)v11 == -2 && *(_DWORD *)(v11 + 4) == -2 )
    {
      goto LABEL_14;
    }
    if ( v12 )
      *v12 = *(_QWORD *)v11;
    v13 = *(unsigned int *)(v11 + 16);
    v12[1] = v12 + 3;
    v12[2] = 0x200000000LL;
    if ( (_DWORD)v13 )
      sub_1E9C460((__int64)(v12 + 1), (char **)(v11 + 8), a3, v13, a5, a6);
    v14 = *(_QWORD *)(v11 + 8);
    v12 += 6;
    *(v12 - 1) = *(_QWORD *)(v11 + 40);
    if ( v14 != v11 + 24 )
      _libc_free(v14);
LABEL_14:
    v11 += 48;
  }
  while ( v11 != v65 );
  *(_BYTE *)(a1 + 8) &= ~1u;
  v15 = (_DWORD *)sub_22077B0(v10 * 4);
  v17 = *(_QWORD *)(a1 + 8);
  *(_DWORD *)(a1 + 24) = v9;
  *(_QWORD *)(a1 + 16) = v15;
  v18 = v17 & 1;
  if ( (_BYTE)v18 )
    v15 = v66;
  *(_QWORD *)(a1 + 8) = v18;
  if ( (_BYTE)v18 )
    v10 = 48;
  v19 = v15;
  v20 = &v15[v10];
  while ( 1 )
  {
    if ( v19 )
    {
      *v15 = -1;
      v15[1] = -1;
    }
    v15 += 12;
    if ( v20 == v15 )
      break;
    v19 = v15;
  }
  for ( k = v68; v12 != (_QWORD *)k; k += 48 )
  {
    v22 = *(_DWORD *)k;
    if ( *(_DWORD *)k == -1 )
    {
      if ( *((_DWORD *)k + 1) != -1 )
        goto LABEL_27;
    }
    else if ( v22 != -2 || *((_DWORD *)k + 1) != -2 )
    {
LABEL_27:
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v23 = v66;
        v24 = 3;
      }
      else
      {
        v61 = *(_DWORD *)(a1 + 24);
        v23 = *(_DWORD **)(a1 + 16);
        if ( !v61 )
        {
          MEMORY[0] = *(_QWORD *)k;
          BUG();
        }
        v24 = v61 - 1;
      }
      v25 = *((_DWORD *)k + 1);
      v26 = 1;
      v27 = ((((unsigned int)(37 * v25) | ((unsigned __int64)(unsigned int)(37 * v22) << 32))
            - 1
            - ((unsigned __int64)(unsigned int)(37 * v25) << 32)) >> 22)
          ^ (((unsigned int)(37 * v25) | ((unsigned __int64)(unsigned int)(37 * v22) << 32))
           - 1
           - ((unsigned __int64)(unsigned int)(37 * v25) << 32));
      v28 = ((9 * (((v27 - 1 - (v27 << 13)) >> 8) ^ (v27 - 1 - (v27 << 13)))) >> 15)
          ^ (9 * (((v27 - 1 - (v27 << 13)) >> 8) ^ (v27 - 1 - (v27 << 13))));
      v29 = ((v28 - 1 - (v28 << 27)) >> 31) ^ (v28 - 1 - ((_DWORD)v28 << 27));
      v30 = 0;
      for ( m = v24 & v29; ; m = v24 & v34 )
      {
        v32 = &v23[12 * m];
        v33 = *v32;
        if ( v22 == *v32 && v25 == v32[1] )
          break;
        if ( v33 == -1 )
        {
          if ( v32[1] == -1 )
          {
            if ( v30 )
              v32 = v30;
            break;
          }
        }
        else if ( v33 == -2 && v32[1] == -2 && !v30 )
        {
          v30 = &v23[12 * m];
        }
        v34 = v26 + m;
        ++v26;
      }
      v62 = *(_QWORD *)k;
      *((_QWORD *)v32 + 2) = 0x200000000LL;
      *(_QWORD *)v32 = v62;
      *((_QWORD *)v32 + 1) = v32 + 6;
      v63 = *((unsigned int *)k + 4);
      if ( (_DWORD)v63 )
        sub_1E9C460((__int64)(v32 + 2), (char **)k + 1, v63, v16, v25, (int)v23);
      *((_QWORD *)v32 + 5) = *((_QWORD *)k + 5);
      v64 = *((_QWORD *)k + 1);
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
      if ( (_BYTE *)v64 != k + 24 )
        _libc_free(v64);
    }
  }
}
