// Function: sub_2ED8E20
// Address: 0x2ed8e20
//
_DWORD *__fastcall sub_2ED8E20(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r12
  __int64 v5; // r15
  unsigned int v6; // eax
  _DWORD *result; // rax
  __int64 v8; // r12
  _DWORD *i; // rdx
  unsigned int v10; // edx
  int v11; // ecx
  __int64 v12; // rcx
  __int64 v13; // r9
  int v14; // r11d
  unsigned int *v15; // r10
  unsigned int v16; // esi
  unsigned int *v17; // rdi
  __int64 v18; // r8
  __int64 v19; // r13
  __int64 v20; // r8
  unsigned __int64 v21; // r14
  unsigned __int64 v22; // rdi
  __int64 v23; // rdx
  _DWORD *j; // rdx
  __int64 v25; // [rsp+0h] [rbp-40h]
  __int64 v26; // [rsp+8h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v26 = v5;
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
  result = (_DWORD *)sub_C7D670(88LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v25 = 88 * v4;
    v8 = 88 * v4 + v5;
    for ( i = &result[22 * *(unsigned int *)(a1 + 24)]; i != result; result += 22 )
    {
      if ( result )
        *result = -1;
    }
    while ( v8 != v5 )
    {
      while ( 1 )
      {
        v10 = *(_DWORD *)v5;
        if ( *(_DWORD *)v5 <= 0xFFFFFFFD )
        {
          v11 = *(_DWORD *)(a1 + 24);
          if ( !v11 )
          {
            MEMORY[0] = 0;
            BUG();
          }
          v12 = (unsigned int)(v11 - 1);
          v13 = *(_QWORD *)(a1 + 8);
          v14 = 1;
          v15 = 0;
          v16 = v12 & (37 * v10);
          v17 = (unsigned int *)(v13 + 88LL * v16);
          v18 = *v17;
          if ( v10 != (_DWORD)v18 )
          {
            while ( (_DWORD)v18 != -1 )
            {
              if ( !v15 && (_DWORD)v18 == -2 )
                v15 = v17;
              v16 = v12 & (v14 + v16);
              v17 = (unsigned int *)(v13 + 88LL * v16);
              v18 = *v17;
              if ( v10 == (_DWORD)v18 )
                goto LABEL_14;
              ++v14;
            }
            if ( v15 )
              v17 = v15;
          }
LABEL_14:
          *v17 = v10;
          *((_QWORD *)v17 + 1) = v17 + 6;
          *((_QWORD *)v17 + 2) = 0x200000000LL;
          if ( *(_DWORD *)(v5 + 16) )
            sub_2ED61C0((__int64)(v17 + 2), v5 + 8, (__int64)(v17 + 6), v12, v18, v13);
          ++*(_DWORD *)(a1 + 16);
          v19 = *(_QWORD *)(v5 + 8);
          v20 = 32LL * *(unsigned int *)(v5 + 16);
          v21 = v19 + v20;
          if ( v19 != v19 + v20 )
          {
            do
            {
              v21 -= 32LL;
              v22 = *(_QWORD *)(v21 + 8);
              if ( v22 != v21 + 24 )
                _libc_free(v22);
            }
            while ( v19 != v21 );
            v21 = *(_QWORD *)(v5 + 8);
          }
          if ( v21 != v5 + 24 )
            break;
        }
        v5 += 88;
        if ( v8 == v5 )
          return (_DWORD *)sub_C7D6A0(v26, v25, 8);
      }
      v5 += 88;
      _libc_free(v21);
    }
    return (_DWORD *)sub_C7D6A0(v26, v25, 8);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[22 * v23]; j != result; result += 22 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
