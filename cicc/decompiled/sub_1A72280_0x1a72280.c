// Function: sub_1A72280
// Address: 0x1a72280
//
_QWORD *__fastcall sub_1A72280(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r14
  unsigned __int64 v5; // rax
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // r14
  _QWORD *i; // rdx
  __int64 *j; // r15
  __int64 v11; // rdi
  int v12; // edx
  int v13; // esi
  __int64 v14; // r8
  int v15; // r11d
  __int64 *v16; // r10
  unsigned int v17; // ecx
  __int64 *v18; // rdx
  __int64 v19; // r9
  __int64 v20; // rsi
  __int64 v21; // rcx
  __int64 v22; // rbx
  __int64 v23; // r12
  unsigned __int64 v24; // rdi
  __int64 v25; // rdx
  _QWORD *k; // rdx
  __int64 *v27; // [rsp+8h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(__int64 **)(a1 + 8);
  v27 = v4;
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
  result = (_QWORD *)sub_22077B0((unsigned __int64)(unsigned int)v5 << 6);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[8 * v3];
    for ( i = &result[8 * v7]; i != result; result += 8 )
    {
      if ( result )
        *result = -8;
    }
    for ( j = v27; v8 != j; j += 8 )
    {
      v11 = *j;
      if ( *j != -16 && v11 != -8 )
      {
        v12 = *(_DWORD *)(a1 + 24);
        if ( !v12 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v13 = v12 - 1;
        v14 = *(_QWORD *)(a1 + 8);
        v15 = 1;
        v16 = 0;
        v17 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v18 = (__int64 *)(v14 + ((unsigned __int64)v17 << 6));
        v19 = *v18;
        if ( v11 != *v18 )
        {
          while ( v19 != -8 )
          {
            if ( v19 == -16 && !v16 )
              v16 = v18;
            v17 = v13 & (v15 + v17);
            v18 = (__int64 *)(v14 + ((unsigned __int64)v17 << 6));
            v19 = *v18;
            if ( v11 == *v18 )
              goto LABEL_13;
            ++v15;
          }
          if ( v16 )
            v18 = v16;
        }
LABEL_13:
        v18[3] = 0;
        v18[2] = 0;
        *((_DWORD *)v18 + 8) = 0;
        *v18 = v11;
        v18[1] = 1;
        v20 = j[2];
        ++j[1];
        v21 = v18[2];
        v18[2] = v20;
        LODWORD(v20) = *((_DWORD *)j + 6);
        j[2] = v21;
        LODWORD(v21) = *((_DWORD *)v18 + 6);
        *((_DWORD *)v18 + 6) = v20;
        LODWORD(v20) = *((_DWORD *)j + 7);
        *((_DWORD *)j + 6) = v21;
        LODWORD(v21) = *((_DWORD *)v18 + 7);
        *((_DWORD *)v18 + 7) = v20;
        LODWORD(v20) = *((_DWORD *)j + 8);
        *((_DWORD *)j + 7) = v21;
        LODWORD(v21) = *((_DWORD *)v18 + 8);
        *((_DWORD *)v18 + 8) = v20;
        *((_DWORD *)j + 8) = v21;
        v18[5] = j[5];
        v18[6] = j[6];
        v18[7] = j[7];
        j[7] = 0;
        j[6] = 0;
        j[5] = 0;
        ++*(_DWORD *)(a1 + 16);
        v22 = j[6];
        v23 = j[5];
        if ( v22 != v23 )
        {
          do
          {
            v24 = *(_QWORD *)(v23 + 8);
            if ( v24 != v23 + 24 )
              _libc_free(v24);
            v23 += 56;
          }
          while ( v22 != v23 );
          v23 = j[5];
        }
        if ( v23 )
          j_j___libc_free_0(v23, j[7] - v23);
        j___libc_free_0(j[2]);
      }
    }
    return (_QWORD *)j___libc_free_0(v27);
  }
  else
  {
    v25 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[8 * v25]; k != result; result += 8 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
