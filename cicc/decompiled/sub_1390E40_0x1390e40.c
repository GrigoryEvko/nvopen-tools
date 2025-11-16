// Function: sub_1390E40
// Address: 0x1390e40
//
_QWORD *__fastcall sub_1390E40(__int64 a1, int a2)
{
  unsigned int v3; // ebx
  __int64 *v4; // r13
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rcx
  __int64 *v8; // r15
  _QWORD *i; // rcx
  __int64 *v10; // rbx
  __int64 v11; // rcx
  int v12; // eax
  int v13; // esi
  __int64 v14; // rdi
  int v15; // r10d
  __int64 *v16; // r9
  unsigned int v17; // eax
  __int64 *v18; // r12
  __int64 v19; // r8
  char v20; // al
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  __int64 v23; // rdi
  __int64 v24; // rdi
  __int64 v25; // rcx
  __int64 v26; // rax
  __int64 v27; // rdx
  _QWORD *j; // rdx

  v3 = *(_DWORD *)(a1 + 24);
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
  result = (_QWORD *)sub_22077B0(424LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[53 * v3];
    for ( i = &result[53 * v7]; i != result; result += 53 )
    {
      if ( result )
        *result = -8;
    }
    if ( v8 != v4 )
    {
      v10 = v4;
      do
      {
        while ( 1 )
        {
          v11 = *v10;
          if ( *v10 != -16 && v11 != -8 )
          {
            v12 = *(_DWORD *)(a1 + 24);
            if ( !v12 )
            {
              MEMORY[0] = *v10;
              BUG();
            }
            v13 = v12 - 1;
            v14 = *(_QWORD *)(a1 + 8);
            v15 = 1;
            v16 = 0;
            v17 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
            v18 = (__int64 *)(v14 + 424LL * v17);
            v19 = *v18;
            if ( v11 != *v18 )
            {
              while ( v19 != -8 )
              {
                if ( !v16 && v19 == -16 )
                  v16 = v18;
                v17 = v13 & (v15 + v17);
                v18 = (__int64 *)(v14 + 424LL * v17);
                v19 = *v18;
                if ( v11 == *v18 )
                  goto LABEL_15;
                ++v15;
              }
              if ( v16 )
                v18 = v16;
            }
LABEL_15:
            *v18 = v11;
            v20 = *((_BYTE *)v10 + 416);
            *((_BYTE *)v18 + 416) = v20;
            if ( v20 )
            {
              v18[3] = 0;
              v18[2] = 0;
              *((_DWORD *)v18 + 8) = 0;
              v18[1] = 1;
              v25 = v10[2];
              ++v10[1];
              v26 = v18[2];
              v18[2] = v25;
              LODWORD(v25) = *((_DWORD *)v10 + 6);
              v10[2] = v26;
              LODWORD(v26) = *((_DWORD *)v18 + 6);
              *((_DWORD *)v18 + 6) = v25;
              LODWORD(v25) = *((_DWORD *)v10 + 7);
              *((_DWORD *)v10 + 6) = v26;
              LODWORD(v26) = *((_DWORD *)v18 + 7);
              *((_DWORD *)v18 + 7) = v25;
              LODWORD(v25) = *((_DWORD *)v10 + 8);
              *((_DWORD *)v10 + 7) = v26;
              LODWORD(v26) = *((_DWORD *)v18 + 8);
              *((_DWORD *)v18 + 8) = v25;
              *((_DWORD *)v10 + 8) = v26;
              v18[5] = v10[5];
              v18[6] = v10[6];
              v18[7] = v10[7];
              v10[7] = 0;
              v10[6] = 0;
              v10[5] = 0;
              v18[8] = (__int64)(v18 + 10);
              v18[9] = 0x800000000LL;
              if ( *((_DWORD *)v10 + 18) )
                sub_138EE50((__int64)(v18 + 8), (char **)v10 + 8);
              v18[34] = (__int64)(v18 + 36);
              v18[35] = 0x800000000LL;
              if ( *((_DWORD *)v10 + 70) )
                sub_138ED10((__int64)(v18 + 34), (char **)v10 + 34);
            }
            ++*(_DWORD *)(a1 + 16);
            if ( *((_BYTE *)v10 + 416) )
              break;
          }
          v10 += 53;
          if ( v8 == v10 )
            return (_QWORD *)j___libc_free_0(v4);
        }
        v21 = v10[34];
        if ( (__int64 *)v21 != v10 + 36 )
          _libc_free(v21);
        v22 = v10[8];
        if ( (__int64 *)v22 != v10 + 10 )
          _libc_free(v22);
        v23 = v10[5];
        if ( v23 )
          j_j___libc_free_0(v23, v10[7] - v23);
        v24 = v10[2];
        v10 += 53;
        j___libc_free_0(v24);
      }
      while ( v8 != v10 );
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v27 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[53 * v27]; j != result; result += 53 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
