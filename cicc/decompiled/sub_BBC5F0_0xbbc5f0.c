// Function: sub_BBC5F0
// Address: 0xbbc5f0
//
_QWORD *__fastcall sub_BBC5F0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r14
  __int64 *v5; // rbx
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 *v9; // r13
  _QWORD *i; // rdx
  __int64 *v11; // rbx
  __int64 v12; // rax
  int v13; // edx
  int v14; // esi
  __int64 v15; // rdi
  int v16; // r10d
  _QWORD *v17; // r9
  unsigned int v18; // ecx
  _QWORD *v19; // rdx
  __int64 v20; // r8
  __int64 v21; // rsi
  _QWORD *v22; // rax
  _QWORD *v23; // rcx
  __int64 *v24; // r14
  __int64 *v25; // r15
  _QWORD *v26; // rdi
  _QWORD *j; // rdx
  __int64 v28; // [rsp+0h] [rbp-40h]
  __int64 *v29; // [rsp+8h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(__int64 **)(a1 + 8);
  v29 = v5;
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
  result = (_QWORD *)sub_C7D670(32LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v28 = 32 * v4;
    v9 = &v5[4 * v4];
    for ( i = &result[4 * v8]; i != result; result += 4 )
    {
      if ( result )
        *result = -4096;
    }
    v11 = v5 + 1;
    if ( v9 != v29 )
    {
      while ( 1 )
      {
        v12 = *(v11 - 1);
        if ( v12 != -8192 && v12 != -4096 )
        {
          v13 = *(_DWORD *)(a1 + 24);
          if ( !v13 )
          {
            MEMORY[0] = *(v11 - 1);
            BUG();
          }
          v14 = v13 - 1;
          v15 = *(_QWORD *)(a1 + 8);
          v16 = 1;
          v17 = 0;
          v18 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v19 = (_QWORD *)(v15 + 32LL * v18);
          v20 = *v19;
          if ( v12 != *v19 )
          {
            while ( v20 != -4096 )
            {
              if ( v20 == -8192 && !v17 )
                v17 = v19;
              v18 = v14 & (v16 + v18);
              v19 = (_QWORD *)(v15 + 32LL * v18);
              v20 = *v19;
              if ( v12 == *v19 )
                goto LABEL_13;
              ++v16;
            }
            if ( v17 )
              v19 = v17;
          }
LABEL_13:
          *v19 = v12;
          v21 = *v11;
          v22 = v19 + 1;
          v19[1] = *v11;
          v23 = (_QWORD *)v11[1];
          v19[2] = v23;
          v19[3] = v11[2];
          if ( v11 == (__int64 *)v21 )
          {
            v19[2] = v22;
            v19[1] = v22;
          }
          else
          {
            *v23 = v22;
            *(_QWORD *)(v19[1] + 8LL) = v22;
            v11[1] = (__int64)v11;
            *v11 = (__int64)v11;
            v11[2] = 0;
          }
          ++*(_DWORD *)(a1 + 16);
          v24 = (__int64 *)*v11;
          while ( v11 != v24 )
          {
            v25 = v24;
            v24 = (__int64 *)*v24;
            v26 = (_QWORD *)v25[3];
            if ( v26 )
              (*(void (__fastcall **)(_QWORD *, __int64, _QWORD, _QWORD *, __int64, _QWORD *, __int64))(*v26 + 8LL))(
                v26,
                v21,
                *v26,
                v23,
                v20,
                v17,
                v28);
            v21 = 32;
            j_j___libc_free_0(v25, 32);
          }
        }
        if ( v9 == v11 + 3 )
          break;
        v11 += 4;
      }
    }
    return (_QWORD *)sub_C7D6A0(v29, v28, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[4 * *(unsigned int *)(a1 + 24)]; j != result; result += 4 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
