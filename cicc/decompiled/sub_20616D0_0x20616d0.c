// Function: sub_20616D0
// Address: 0x20616d0
//
_QWORD *__fastcall sub_20616D0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r14
  unsigned __int64 v5; // rax
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // r14
  _QWORD *i; // rdx
  __int64 *j; // r15
  __int64 v11; // rdx
  int v12; // ecx
  int v13; // edi
  __int64 v14; // r8
  int v15; // r11d
  _QWORD *v16; // r10
  unsigned int v17; // esi
  _QWORD *v18; // rcx
  __int64 v19; // r9
  __int64 v20; // rbx
  __int64 v21; // r12
  __int64 v22; // rsi
  __int64 v23; // rdx
  _QWORD *k; // rdx
  __int64 *v25; // [rsp+8h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(__int64 **)(a1 + 8);
  v25 = v4;
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
  result = (_QWORD *)sub_22077B0(32LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[4 * v3];
    for ( i = &result[4 * v7]; i != result; result += 4 )
    {
      if ( result )
        *result = -8;
    }
    for ( j = v25; v8 != j; j += 4 )
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
        v18 = (_QWORD *)(v14 + 32LL * v17);
        v19 = *v18;
        if ( v11 != *v18 )
        {
          while ( v19 != -8 )
          {
            if ( v19 == -16 && !v16 )
              v16 = v18;
            v17 = v13 & (v15 + v17);
            v18 = (_QWORD *)(v14 + 32LL * v17);
            v19 = *v18;
            if ( v11 == *v18 )
              goto LABEL_13;
            ++v15;
          }
          if ( v16 )
            v18 = v16;
        }
LABEL_13:
        *v18 = v11;
        v18[1] = j[1];
        v18[2] = j[2];
        v18[3] = j[3];
        j[2] = 0;
        j[1] = 0;
        j[3] = 0;
        ++*(_DWORD *)(a1 + 16);
        v20 = j[2];
        v21 = j[1];
        if ( v20 != v21 )
        {
          do
          {
            v22 = *(_QWORD *)(v21 + 8);
            if ( v22 )
              sub_161E7C0(v21 + 8, v22);
            v21 += 24;
          }
          while ( v20 != v21 );
          v21 = j[1];
        }
        if ( v21 )
          j_j___libc_free_0(v21, j[3] - v21);
      }
    }
    return (_QWORD *)j___libc_free_0(v25);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[4 * v23]; k != result; result += 4 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
