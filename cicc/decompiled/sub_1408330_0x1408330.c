// Function: sub_1408330
// Address: 0x1408330
//
_QWORD *__fastcall sub_1408330(__int64 a1, int a2)
{
  __int64 v3; // rbx
  _QWORD *v4; // r13
  unsigned __int64 v5; // rax
  _QWORD *result; // rax
  _QWORD *v7; // r8
  _QWORD *i; // rdx
  _QWORD *v9; // rbx
  _QWORD *v10; // r13
  char *v11; // rax
  __int64 v12; // rdx
  char *v13; // r15
  bool v14; // al
  int v15; // ecx
  int v16; // ecx
  __int64 v17; // r9
  int v18; // r8d
  unsigned int v19; // esi
  __int64 *v20; // r10
  __int64 *v21; // r12
  __int64 v22; // rdi
  char *v23; // rcx
  char *v24; // rsi
  char *v25; // r9
  _QWORD *v26; // rdi
  _QWORD *v27; // rdx
  unsigned int v28; // edx
  int v29; // edx
  char *v30; // rdx
  __int64 v31; // r11
  __int64 v32; // rax
  int v33; // r10d
  void *v34; // rax
  unsigned int v35; // r9d
  unsigned __int64 v36; // rdi
  __int64 v37; // rsi
  __int64 *v38; // rcx
  __int64 *v39; // rdx
  bool v40; // r9
  __int64 v41; // r8
  __int64 v42; // r8
  __int64 v43; // rsi
  __int64 v44; // r8
  __int64 v45; // rsi
  _DWORD *v46; // r10
  const void *v47; // rsi
  size_t v48; // rdx
  __int64 v49; // rax
  int v50; // edx
  __int64 v51; // rdx
  _QWORD *j; // rdx
  _DWORD *v53; // [rsp+8h] [rbp-48h]
  _DWORD *v54; // [rsp+8h] [rbp-48h]
  unsigned int v55; // [rsp+14h] [rbp-3Ch]
  unsigned int v56; // [rsp+14h] [rbp-3Ch]
  _QWORD *v57; // [rsp+18h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD **)(a1 + 8);
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
  result = (_QWORD *)sub_22077B0(168LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v4[21 * v3];
    for ( i = &result[21 * *(unsigned int *)(a1 + 24)]; i != result; result += 21 )
    {
      if ( result )
        *result = -8;
    }
    v9 = v4 + 13;
    if ( v7 != v4 )
    {
      v57 = v4;
      v10 = v7;
      while ( 1 )
      {
        v12 = *(v9 - 13);
        v13 = (char *)(v9 - 13);
        v14 = v12 != -16 && v12 != -8;
        if ( !v14 )
        {
LABEL_10:
          v11 = (char *)(v9 + 21);
          if ( v10 == v9 + 8 )
            goto LABEL_30;
          goto LABEL_11;
        }
        v15 = *(_DWORD *)(a1 + 24);
        if ( !v15 )
        {
          MEMORY[0] = *(v9 - 13);
          BUG();
        }
        v16 = v15 - 1;
        v17 = *(_QWORD *)(a1 + 8);
        v18 = 1;
        v19 = v16 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v20 = 0;
        v21 = (__int64 *)(v17 + 168LL * v19);
        v22 = *v21;
        if ( v12 != *v21 )
        {
          while ( v22 != -8 )
          {
            if ( !v20 && v22 == -16 )
              v20 = v21;
            v19 = v16 & (v18 + v19);
            v21 = (__int64 *)(v17 + 168LL * v19);
            v22 = *v21;
            if ( v12 == *v21 )
              goto LABEL_15;
            ++v18;
          }
          if ( v20 )
            v21 = v20;
        }
LABEL_15:
        v23 = (char *)(v21 + 3);
        *v21 = v12;
        v24 = (char *)(v21 + 1);
        v21[1] = 0;
        v25 = (char *)(v9 - 12);
        v26 = v21 + 11;
        v27 = v21 + 3;
        v21[2] = 1;
        do
        {
          if ( v27 )
          {
            *v27 = -2;
            v27[1] = -8;
          }
          v27 += 2;
        }
        while ( v27 != v26 );
        v28 = *((_DWORD *)v13 + 4) & 0xFFFFFFFE;
        *((_DWORD *)v13 + 4) = v21[2] & 0xFFFFFFFE | *((_DWORD *)v13 + 4) & 1;
        *((_DWORD *)v21 + 4) = v28 | v21[2] & 1;
        v29 = *((_DWORD *)v21 + 5);
        *((_DWORD *)v21 + 5) = *((_DWORD *)v9 - 21);
        *((_DWORD *)v9 - 21) = v29;
        if ( (v21[2] & 1) != 0 )
        {
          v30 = (char *)(v9 - 10);
          if ( (v13[16] & 1) != 0 )
          {
            v37 = v21[3];
            v38 = v21 + 4;
            v39 = v9 - 9;
            if ( v37 == -2 )
            {
              while ( 1 )
              {
                v41 = *(v39 - 1);
                v40 = *v38 != -8;
                if ( v41 != -2 )
                  break;
LABEL_43:
                if ( *v39 == -8 || !v40 )
                  goto LABEL_39;
LABEL_45:
                v44 = *(v38 - 1);
                v45 = *v38;
                *(v38 - 1) = *(v39 - 1);
                *v38 = *v39;
                *(v39 - 1) = v44;
                *v39 = v45;
LABEL_40:
                v39 += 2;
                v38 += 2;
                if ( v9 - 1 == v39 )
                  goto LABEL_25;
                v37 = *(v38 - 1);
                if ( v37 != -2 )
                  goto LABEL_34;
              }
            }
            else
            {
LABEL_34:
              v40 = v14;
              if ( v37 == -16 )
                v40 = *v38 != -16;
              v41 = *(v39 - 1);
              if ( v41 == -2 )
                goto LABEL_43;
            }
            if ( (v41 != -16 || *v39 != -16) && v40 )
              goto LABEL_45;
LABEL_39:
            *(v38 - 1) = *(v39 - 1);
            v42 = *v39;
            *(v39 - 1) = v37;
            v43 = *v38;
            *v38 = v42;
            *v39 = v43;
            goto LABEL_40;
          }
        }
        else
        {
          if ( (v13[16] & 1) == 0 )
          {
            v49 = v21[3];
            v21[3] = *(v9 - 10);
            v50 = *((_DWORD *)v9 - 18);
            *(v9 - 10) = v49;
            LODWORD(v49) = *((_DWORD *)v21 + 8);
            *((_DWORD *)v21 + 8) = v50;
            *((_DWORD *)v9 - 18) = v49;
            goto LABEL_25;
          }
          v30 = (char *)(v21 + 3);
          v23 = (char *)(v9 - 10);
          v25 = (char *)(v21 + 1);
          v24 = (char *)(v9 - 12);
        }
        v25[8] |= 1u;
        v31 = *((_QWORD *)v25 + 2);
        v32 = 0;
        v33 = *((_DWORD *)v25 + 6);
        do
        {
          *(__m128i *)&v30[v32] = _mm_loadu_si128((const __m128i *)&v23[v32]);
          v32 += 16;
        }
        while ( v32 != 64 );
        v24[8] &= ~1u;
        *((_QWORD *)v24 + 2) = v31;
        *((_DWORD *)v24 + 6) = v33;
LABEL_25:
        v34 = v21 + 13;
        v21[11] = (__int64)(v21 + 13);
        v21[12] = 0x400000000LL;
        v35 = *((_DWORD *)v9 - 2);
        if ( v35 && v26 != v9 - 2 )
        {
          v46 = (_DWORD *)*(v9 - 2);
          if ( v9 == (_QWORD *)v46 )
          {
            v47 = v9;
            v48 = 16LL * v35;
            if ( v35 <= 4 )
              goto LABEL_50;
            v54 = (_DWORD *)*(v9 - 2);
            v56 = *((_DWORD *)v9 - 2);
            sub_16CD150(v26, v21 + 13, v35, 16);
            v34 = (void *)v21[11];
            v47 = (const void *)*(v9 - 2);
            v35 = v56;
            v48 = 16LL * *((unsigned int *)v9 - 2);
            v46 = v54;
            if ( v48 )
            {
LABEL_50:
              v53 = v46;
              v55 = v35;
              memcpy(v34, v47, v48);
              v46 = v53;
              v35 = v55;
            }
            *((_DWORD *)v21 + 24) = v35;
            *(v46 - 2) = 0;
          }
          else
          {
            v21[11] = (__int64)v46;
            *((_DWORD *)v21 + 24) = *((_DWORD *)v9 - 2);
            *((_DWORD *)v21 + 25) = *((_DWORD *)v9 - 1);
            *(v9 - 2) = v9;
            *((_DWORD *)v9 - 1) = 0;
            *((_DWORD *)v9 - 2) = 0;
          }
        }
        ++*(_DWORD *)(a1 + 16);
        v36 = *(v9 - 2);
        if ( (_QWORD *)v36 != v9 )
          _libc_free(v36);
        if ( (v13[16] & 1) != 0 )
          goto LABEL_10;
        j___libc_free_0(*(v9 - 10));
        v11 = (char *)(v9 + 21);
        if ( v10 == v9 + 8 )
        {
LABEL_30:
          v4 = v57;
          return (_QWORD *)j___libc_free_0(v4);
        }
LABEL_11:
        v9 = v11;
      }
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v51 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[21 * v51]; j != result; result += 21 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
