// Function: sub_20EBDE0
// Address: 0x20ebde0
//
_DWORD *__fastcall sub_20EBDE0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 v4; // r15
  unsigned __int64 v5; // rax
  _DWORD *result; // rax
  _DWORD *v7; // r8
  _DWORD *i; // rdx
  char *v9; // r13
  _DWORD *v10; // rbx
  unsigned int v11; // eax
  _DWORD *v12; // r15
  int v13; // edx
  int v14; // ecx
  __int64 v15; // rdi
  int v16; // r10d
  unsigned int v17; // edx
  unsigned int *v18; // r9
  unsigned int *v19; // r12
  unsigned int v20; // esi
  _DWORD *v21; // r9
  _DWORD *v22; // rdx
  _DWORD *v23; // rcx
  __int64 v24; // rdi
  _DWORD *v25; // rax
  unsigned int v26; // eax
  unsigned int v27; // eax
  _DWORD *v28; // rsi
  __int64 v29; // r11
  __int64 v30; // rax
  int v31; // r10d
  void *v32; // rax
  unsigned int v33; // r9d
  unsigned __int64 v34; // rdi
  int *v35; // rax
  int *v36; // rdx
  int v37; // ecx
  int v38; // esi
  char *v39; // rsi
  size_t v40; // rdx
  __int64 v41; // rax
  unsigned int v42; // edx
  __int64 v43; // rdx
  _DWORD *j; // rdx
  _DWORD *v45; // [rsp+8h] [rbp-48h]
  _DWORD *v46; // [rsp+8h] [rbp-48h]
  _DWORD *v47; // [rsp+10h] [rbp-40h]
  _DWORD *v48; // [rsp+10h] [rbp-40h]
  unsigned int v49; // [rsp+10h] [rbp-40h]
  unsigned int v50; // [rsp+10h] [rbp-40h]
  __int64 v51; // [rsp+18h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v51 = v4;
  v5 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
      | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
      | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
      | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
      | (unsigned int)(a2 - 1)
      | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
     + 1;
  if ( (unsigned int)v5 < 0x40 )
    LODWORD(v5) = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_DWORD *)sub_22077B0(168LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = (_DWORD *)(v4 + 168 * v3);
    for ( i = &result[42 * *(unsigned int *)(a1 + 24)]; i != result; result += 42 )
    {
      if ( result )
        *result = -1;
    }
    v9 = (char *)(v4 + 104);
    v10 = (_DWORD *)(v4 + 24);
    if ( v7 == (_DWORD *)v4 )
      return (_DWORD *)j___libc_free_0(v51);
    while ( 1 )
    {
      v11 = *(v10 - 6);
      v12 = v10 - 6;
      if ( v11 > 0xFFFFFFFD )
        goto LABEL_10;
      v13 = *(_DWORD *)(a1 + 24);
      if ( !v13 )
      {
        MEMORY[0] = 0;
        BUG();
      }
      v14 = v13 - 1;
      v15 = *(_QWORD *)(a1 + 8);
      v16 = 1;
      v17 = (v13 - 1) & (37 * v11);
      v18 = 0;
      v19 = (unsigned int *)(v15 + 168LL * v17);
      v20 = *v19;
      if ( v11 != *v19 )
      {
        while ( v20 != -1 )
        {
          if ( !v18 && v20 == -2 )
            v18 = v19;
          v17 = v14 & (v16 + v17);
          v19 = (unsigned int *)(v15 + 168LL * v17);
          v20 = *v19;
          if ( v11 == *v19 )
            goto LABEL_15;
          ++v16;
        }
        if ( v18 )
          v19 = v18;
      }
LABEL_15:
      *v19 = v11;
      v21 = v19 + 2;
      v22 = v10 - 4;
      v23 = v19 + 6;
      v24 = (__int64)(v19 + 22);
      v25 = v19 + 6;
      *((_QWORD *)v19 + 1) = 0;
      *((_QWORD *)v19 + 2) = 1;
      do
      {
        if ( v25 )
          *v25 = -1;
        ++v25;
      }
      while ( v25 != (_DWORD *)v24 );
      v26 = v12[4] & 0xFFFFFFFE;
      v12[4] = v19[4] & 0xFFFFFFFE | v12[4] & 1;
      v19[4] = v26 | v19[4] & 1;
      v27 = v19[5];
      v19[5] = *(v10 - 1);
      *(v10 - 1) = v27;
      if ( (v19[4] & 1) != 0 )
        break;
      if ( (v12[4] & 1) != 0 )
      {
        v28 = v19 + 6;
        v22 = v19 + 2;
        v23 = v10;
        v21 = v10 - 4;
LABEL_22:
        *((_BYTE *)v22 + 8) |= 1u;
        v29 = *((_QWORD *)v22 + 2);
        v30 = 0;
        v31 = v22[6];
        do
        {
          v28[v30] = v23[v30];
          ++v30;
        }
        while ( v30 != 16 );
        *((_BYTE *)v21 + 8) &= ~1u;
        *((_QWORD *)v21 + 2) = v29;
        v21[6] = v31;
        goto LABEL_25;
      }
      v41 = *((_QWORD *)v19 + 3);
      *((_QWORD *)v19 + 3) = *(_QWORD *)v10;
      v42 = v10[2];
      *(_QWORD *)v10 = v41;
      LODWORD(v41) = v19[8];
      v19[8] = v42;
      v10[2] = v41;
LABEL_25:
      v32 = v19 + 26;
      *((_QWORD *)v19 + 11) = v19 + 26;
      *((_QWORD *)v19 + 12) = 0x1000000000LL;
      v33 = v10[18];
      if ( v33 && (_DWORD *)v24 != v10 + 16 )
      {
        v39 = (char *)*((_QWORD *)v10 + 8);
        if ( v9 == v39 )
        {
          v40 = 4LL * v33;
          if ( v33 <= 0x10 )
            goto LABEL_39;
          v46 = v7;
          v50 = v10[18];
          sub_16CD150(v24, v19 + 26, v33, 4, (int)v7, v33);
          v32 = (void *)*((_QWORD *)v19 + 11);
          v39 = (char *)*((_QWORD *)v10 + 8);
          v33 = v50;
          v40 = 4LL * (unsigned int)v10[18];
          v7 = v46;
          if ( v40 )
          {
LABEL_39:
            v45 = v7;
            v49 = v33;
            memcpy(v32, v39, v40);
            v7 = v45;
            v33 = v49;
          }
          v19[24] = v33;
          v10[18] = 0;
        }
        else
        {
          *((_QWORD *)v19 + 11) = v39;
          v19[24] = v10[18];
          v19[25] = v10[19];
          *((_QWORD *)v10 + 8) = v9;
          v10[19] = 0;
          v10[18] = 0;
        }
      }
      ++*(_DWORD *)(a1 + 16);
      v34 = *((_QWORD *)v10 + 8);
      if ( (char *)v34 != v9 )
      {
        v47 = v7;
        _libc_free(v34);
        v7 = v47;
      }
      if ( (v12[4] & 1) == 0 )
      {
        v48 = v7;
        j___libc_free_0(*(_QWORD *)v10);
        v7 = v48;
      }
LABEL_10:
      v9 += 168;
      if ( v7 == v10 + 36 )
        return (_DWORD *)j___libc_free_0(v51);
      v10 += 42;
    }
    v28 = v10;
    if ( (v12[4] & 1) != 0 )
    {
      v35 = (int *)(v19 + 6);
      v36 = v10;
      do
      {
        v37 = *v36;
        v38 = *v35++;
        ++v36;
        *(v35 - 1) = v37;
        *(v36 - 1) = v38;
      }
      while ( (int *)v24 != v35 );
      goto LABEL_25;
    }
    goto LABEL_22;
  }
  v43 = *(unsigned int *)(a1 + 24);
  *(_QWORD *)(a1 + 16) = 0;
  for ( j = &result[42 * v43]; j != result; result += 42 )
  {
    if ( result )
      *result = -1;
  }
  return result;
}
