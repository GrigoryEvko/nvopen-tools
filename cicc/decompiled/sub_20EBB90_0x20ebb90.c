// Function: sub_20EBB90
// Address: 0x20ebb90
//
_DWORD *__fastcall sub_20EBB90(__int64 a1, int a2)
{
  __int64 v3; // rbx
  int *v4; // r12
  unsigned __int64 v5; // rdi
  _DWORD *result; // rax
  __int64 v7; // rdx
  int *v8; // r14
  _DWORD *i; // rdx
  int *v10; // rbx
  int v11; // eax
  int v12; // edx
  int v13; // ecx
  __int64 v14; // rdi
  int v15; // r11d
  int *v16; // r9
  unsigned int v17; // esi
  int *v18; // rdx
  int v19; // r8d
  unsigned __int64 *v20; // r15
  unsigned __int64 v21; // r8
  unsigned __int64 v22; // rdi
  __int64 v23; // rdx
  _DWORD *j; // rdx
  unsigned __int64 v25; // [rsp+8h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(int **)(a1 + 8);
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
  result = (_DWORD *)sub_22077B0(16LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[4 * v3];
    for ( i = &result[4 * v7]; i != result; result += 4 )
    {
      if ( result )
        *result = 0x7FFFFFFF;
    }
    if ( v8 != v4 )
    {
      v10 = v4;
      do
      {
        while ( 1 )
        {
          v11 = *v10;
          if ( (unsigned int)(*v10 + 0x7FFFFFFF) <= 0xFFFFFFFD )
          {
            v12 = *(_DWORD *)(a1 + 24);
            if ( !v12 )
            {
              MEMORY[0] = 0;
              BUG();
            }
            v13 = v12 - 1;
            v14 = *(_QWORD *)(a1 + 8);
            v15 = 1;
            v16 = 0;
            v17 = (v12 - 1) & (37 * v11);
            v18 = (int *)(v14 + 16LL * v17);
            v19 = *v18;
            if ( v11 != *v18 )
            {
              while ( v19 != 0x7FFFFFFF )
              {
                if ( !v16 && v19 == 0x80000000 )
                  v16 = v18;
                v17 = v13 & (v15 + v17);
                v18 = (int *)(v14 + 16LL * v17);
                v19 = *v18;
                if ( v11 == *v18 )
                  goto LABEL_14;
                ++v15;
              }
              if ( v16 )
                v18 = v16;
            }
LABEL_14:
            *v18 = v11;
            *((_QWORD *)v18 + 1) = *((_QWORD *)v10 + 1);
            *((_QWORD *)v10 + 1) = 0;
            ++*(_DWORD *)(a1 + 16);
            v20 = (unsigned __int64 *)*((_QWORD *)v10 + 1);
            if ( v20 )
              break;
          }
          v10 += 4;
          if ( v8 == v10 )
            return (_DWORD *)j___libc_free_0(v4);
        }
        sub_1DB4CE0(*((_QWORD *)v10 + 1));
        v21 = v20[12];
        if ( v21 )
        {
          v25 = v20[12];
          sub_20EA3F0(*(_QWORD *)(v21 + 16));
          j_j___libc_free_0(v25, 48);
        }
        v22 = v20[8];
        if ( (unsigned __int64 *)v22 != v20 + 10 )
          _libc_free(v22);
        if ( (unsigned __int64 *)*v20 != v20 + 2 )
          _libc_free(*v20);
        v10 += 4;
        j_j___libc_free_0(v20, 120);
      }
      while ( v8 != v10 );
    }
    return (_DWORD *)j___libc_free_0(v4);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[4 * v23]; j != result; result += 4 )
    {
      if ( result )
        *result = 0x7FFFFFFF;
    }
  }
  return result;
}
