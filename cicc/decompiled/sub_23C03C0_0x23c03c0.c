// Function: sub_23C03C0
// Address: 0x23c03c0
//
_QWORD *__fastcall sub_23C03C0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r14
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 *v8; // r15
  _QWORD *i; // rdx
  __int64 *v10; // rbx
  __int64 v11; // rax
  int v12; // edx
  int v13; // edx
  __int64 v14; // r8
  __int64 *v15; // r9
  int v16; // r10d
  unsigned int v17; // esi
  __int64 *v18; // r13
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  _QWORD *v23; // rdi
  _QWORD *j; // rdx
  __int64 v25; // [rsp+8h] [rbp-38h]

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
  result = (_QWORD *)sub_C7D670(40LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v25 = 40 * v4;
    v8 = (__int64 *)(v5 + 40 * v4);
    for ( i = &result[5 * *(unsigned int *)(a1 + 24)]; i != result; result += 5 )
    {
      if ( result )
        *result = 0x7FFFFFFFFFFFFFFFLL;
    }
    if ( v8 != (__int64 *)v5 )
    {
      v10 = (__int64 *)v5;
      do
      {
        while ( 1 )
        {
          v11 = *v10;
          if ( *v10 <= 0x7FFFFFFFFFFFFFFDLL )
          {
            v12 = *(_DWORD *)(a1 + 24);
            if ( !v12 )
            {
              MEMORY[0] = 0;
              BUG();
            }
            v13 = v12 - 1;
            v14 = *(_QWORD *)(a1 + 8);
            v15 = 0;
            v16 = 1;
            v17 = v13 & (37 * v11);
            v18 = (__int64 *)(v14 + 40LL * v17);
            v19 = *v18;
            if ( v11 != *v18 )
            {
              while ( v19 != 0x7FFFFFFFFFFFFFFFLL )
              {
                if ( !v15 && v19 == 0x7FFFFFFFFFFFFFFELL )
                  v15 = v18;
                v17 = v13 & (v16 + v17);
                v18 = (__int64 *)(v14 + 40LL * v17);
                v19 = *v18;
                if ( v11 == *v18 )
                  goto LABEL_14;
                ++v16;
              }
              if ( v15 )
                v18 = v15;
            }
LABEL_14:
            *v18 = v11;
            v20 = v10[2];
            v18[3] = 0;
            v18[2] = v20 & 6;
            v21 = v10[4];
            v18[4] = v21;
            if ( v21 != -4096 && v21 != 0 && v21 != -8192 )
              sub_BD6050((unsigned __int64 *)v18 + 2, v10[2] & 0xFFFFFFFFFFFFFFF8LL);
            v18[1] = (__int64)&unk_4A15D10;
            ++*(_DWORD *)(a1 + 16);
            v10[1] = (__int64)&unk_49DB368;
            v22 = v10[4];
            if ( v22 != 0 && v22 != -4096 && v22 != -8192 )
              break;
          }
          v10 += 5;
          if ( v8 == v10 )
            return (_QWORD *)sub_C7D6A0(v5, v25, 8);
        }
        v23 = v10 + 2;
        v10 += 5;
        sub_BD60C0(v23);
      }
      while ( v8 != v10 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v25, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[5 * *(unsigned int *)(a1 + 24)]; j != result; result += 5 )
    {
      if ( result )
        *result = 0x7FFFFFFFFFFFFFFFLL;
    }
  }
  return result;
}
