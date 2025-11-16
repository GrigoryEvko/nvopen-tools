// Function: sub_1388C50
// Address: 0x1388c50
//
_QWORD *__fastcall sub_1388C50(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r14
  unsigned __int64 v5; // rax
  _QWORD *result; // rax
  __int64 *v7; // r13
  _QWORD *i; // rcx
  __int64 *v9; // rbx
  __int64 v10; // rcx
  int v11; // eax
  int v12; // eax
  __int64 v13; // rdi
  int v14; // r10d
  __int64 *v15; // r9
  unsigned int v16; // esi
  __int64 *v17; // r12
  __int64 v18; // r8
  char v19; // al
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  __int64 v22; // r12
  _QWORD *v23; // rcx
  _QWORD *v24; // r12
  __int64 v25; // rdi
  __int64 v26; // rdi
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rdx
  _QWORD *j; // rdx
  _QWORD *v33; // [rsp+8h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(__int64 **)(a1 + 8);
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
  result = (_QWORD *)sub_22077B0(432LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v4[54 * v3];
    for ( i = &result[54 * *(unsigned int *)(a1 + 24)]; i != result; result += 54 )
    {
      if ( result )
        *result = -8;
    }
    if ( v7 != v4 )
    {
      v9 = v4;
      do
      {
        while ( 1 )
        {
          v10 = *v9;
          if ( *v9 != -16 && v10 != -8 )
          {
            v11 = *(_DWORD *)(a1 + 24);
            if ( !v11 )
            {
              MEMORY[0] = *v9;
              BUG();
            }
            v12 = v11 - 1;
            v13 = *(_QWORD *)(a1 + 8);
            v14 = 1;
            v15 = 0;
            v16 = v12 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
            v17 = (__int64 *)(v13 + 432LL * v16);
            v18 = *v17;
            if ( v10 != *v17 )
            {
              while ( v18 != -8 )
              {
                if ( !v15 && v18 == -16 )
                  v15 = v17;
                v16 = v12 & (v14 + v16);
                v17 = (__int64 *)(v13 + 432LL * v16);
                v18 = *v17;
                if ( v10 == *v17 )
                  goto LABEL_15;
                ++v14;
              }
              if ( v15 )
                v17 = v15;
            }
LABEL_15:
            *v17 = v10;
            v19 = *((_BYTE *)v9 + 424);
            *((_BYTE *)v17 + 424) = v19;
            if ( v19 )
            {
              v17[3] = 0;
              v17[2] = 0;
              *((_DWORD *)v17 + 8) = 0;
              v17[1] = 1;
              v27 = v9[2];
              ++v9[1];
              v28 = v17[2];
              v17[2] = v27;
              LODWORD(v27) = *((_DWORD *)v9 + 6);
              v9[2] = v28;
              LODWORD(v28) = *((_DWORD *)v17 + 6);
              *((_DWORD *)v17 + 6) = v27;
              LODWORD(v27) = *((_DWORD *)v9 + 7);
              *((_DWORD *)v9 + 6) = v28;
              LODWORD(v28) = *((_DWORD *)v17 + 7);
              *((_DWORD *)v17 + 7) = v27;
              LODWORD(v27) = *((_DWORD *)v9 + 8);
              *((_DWORD *)v9 + 7) = v28;
              LODWORD(v28) = *((_DWORD *)v17 + 8);
              *((_DWORD *)v17 + 8) = v27;
              *((_DWORD *)v9 + 8) = v28;
              v17[6] = 0;
              v17[5] = 1;
              v17[7] = 0;
              *((_DWORD *)v17 + 16) = 0;
              v29 = v9[6];
              ++v9[5];
              v30 = v17[6];
              v17[6] = v29;
              v9[6] = v30;
              LODWORD(v30) = *((_DWORD *)v17 + 14);
              *((_DWORD *)v17 + 14) = *((_DWORD *)v9 + 14);
              LODWORD(v29) = *((_DWORD *)v9 + 15);
              *((_DWORD *)v9 + 14) = v30;
              LODWORD(v30) = *((_DWORD *)v17 + 15);
              *((_DWORD *)v17 + 15) = v29;
              LODWORD(v29) = *((_DWORD *)v9 + 16);
              *((_DWORD *)v9 + 15) = v30;
              LODWORD(v30) = *((_DWORD *)v17 + 16);
              *((_DWORD *)v17 + 16) = v29;
              *((_DWORD *)v9 + 16) = v30;
              v17[9] = (__int64)(v17 + 11);
              v17[10] = 0x800000000LL;
              if ( *((_DWORD *)v9 + 20) )
                sub_1381800((__int64)(v17 + 9), (char **)v9 + 9);
              v17[35] = (__int64)(v17 + 37);
              v17[36] = 0x800000000LL;
              if ( *((_DWORD *)v9 + 72) )
                sub_13816C0((__int64)(v17 + 35), (char **)v9 + 35);
            }
            ++*(_DWORD *)(a1 + 16);
            if ( *((_BYTE *)v9 + 424) )
              break;
          }
          v9 += 54;
          if ( v7 == v9 )
            return (_QWORD *)j___libc_free_0(v4);
        }
        v20 = v9[35];
        if ( (__int64 *)v20 != v9 + 37 )
          _libc_free(v20);
        v21 = v9[9];
        if ( (__int64 *)v21 != v9 + 11 )
          _libc_free(v21);
        j___libc_free_0(v9[6]);
        v22 = *((unsigned int *)v9 + 8);
        if ( (_DWORD)v22 )
        {
          v23 = (_QWORD *)v9[2];
          v24 = &v23[4 * v22];
          do
          {
            if ( *v23 != -8 && *v23 != -16 )
            {
              v25 = v23[1];
              if ( v25 )
              {
                v33 = v23;
                j_j___libc_free_0(v25, v23[3] - v25);
                v23 = v33;
              }
            }
            v23 += 4;
          }
          while ( v24 != v23 );
        }
        v26 = v9[2];
        v9 += 54;
        j___libc_free_0(v26);
      }
      while ( v7 != v9 );
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v31 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[54 * v31]; j != result; result += 54 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
