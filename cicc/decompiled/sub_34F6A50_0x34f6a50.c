// Function: sub_34F6A50
// Address: 0x34f6a50
//
_DWORD *__fastcall sub_34F6A50(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r14
  __int64 v5; // r12
  unsigned int v6; // edi
  _DWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  _DWORD *i; // rdx
  __int64 v11; // rbx
  int v12; // eax
  int v13; // edx
  int v14; // ecx
  __int64 v15; // r8
  int v16; // r10d
  int *v17; // r9
  unsigned int v18; // esi
  int *v19; // rdx
  int v20; // edi
  unsigned __int64 *v21; // r14
  unsigned __int64 v22; // r9
  unsigned __int64 v23; // rdi
  __int64 v24; // rdx
  _DWORD *j; // rdx
  unsigned __int64 v26; // [rsp+0h] [rbp-40h]
  __int64 v27; // [rsp+8h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  result = (_DWORD *)sub_C7D670(16LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v27 = 16 * v4;
    v9 = v5 + 16 * v4;
    for ( i = &result[4 * v8]; i != result; result += 4 )
    {
      if ( result )
        *result = 0x7FFFFFFF;
    }
    if ( v9 != v5 )
    {
      v11 = v5;
      do
      {
        while ( 1 )
        {
          v12 = *(_DWORD *)v11;
          if ( (unsigned int)(*(_DWORD *)v11 + 0x7FFFFFFF) <= 0xFFFFFFFD )
          {
            v13 = *(_DWORD *)(a1 + 24);
            if ( !v13 )
            {
              MEMORY[0] = 0;
              BUG();
            }
            v14 = v13 - 1;
            v15 = *(_QWORD *)(a1 + 8);
            v16 = 1;
            v17 = 0;
            v18 = (v13 - 1) & (37 * v12);
            v19 = (int *)(v15 + 16LL * v18);
            v20 = *v19;
            if ( v12 != *v19 )
            {
              while ( v20 != 0x7FFFFFFF )
              {
                if ( !v17 && v20 == 0x80000000 )
                  v17 = v19;
                v18 = v14 & (v16 + v18);
                v19 = (int *)(v15 + 16LL * v18);
                v20 = *v19;
                if ( v12 == *v19 )
                  goto LABEL_14;
                ++v16;
              }
              if ( v17 )
                v19 = v17;
            }
LABEL_14:
            *v19 = v12;
            *((_QWORD *)v19 + 1) = *(_QWORD *)(v11 + 8);
            *(_QWORD *)(v11 + 8) = 0;
            ++*(_DWORD *)(a1 + 16);
            v21 = *(unsigned __int64 **)(v11 + 8);
            if ( v21 )
              break;
          }
          v11 += 16;
          if ( v9 == v11 )
            return (_DWORD *)sub_C7D6A0(v5, v27, 8);
        }
        sub_2E0AFD0(*(_QWORD *)(v11 + 8));
        v22 = v21[12];
        if ( v22 )
        {
          v26 = v21[12];
          sub_34F51B0(*(_QWORD *)(v22 + 16));
          j_j___libc_free_0(v26);
        }
        v23 = v21[8];
        if ( (unsigned __int64 *)v23 != v21 + 10 )
          _libc_free(v23);
        if ( (unsigned __int64 *)*v21 != v21 + 2 )
          _libc_free(*v21);
        v11 += 16;
        j_j___libc_free_0((unsigned __int64)v21);
      }
      while ( v9 != v11 );
    }
    return (_DWORD *)sub_C7D6A0(v5, v27, 8);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[4 * v24]; j != result; result += 4 )
    {
      if ( result )
        *result = 0x7FFFFFFF;
    }
  }
  return result;
}
