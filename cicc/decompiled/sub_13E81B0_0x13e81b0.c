// Function: sub_13E81B0
// Address: 0x13e81b0
//
_QWORD *__fastcall sub_13E81B0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r13
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // r15
  _QWORD *i; // rdx
  __int64 *j; // rbx
  __int64 v11; // rax
  int v12; // edx
  int v13; // esi
  __int64 v14; // rdi
  int v15; // r11d
  _QWORD *v16; // r9
  unsigned int v17; // ecx
  _QWORD *v18; // rdx
  __int64 v19; // r8
  __int64 v20; // r8
  __int64 v21; // rax
  __int64 v22; // r12
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rdi
  __int64 v27; // rdx
  _QWORD *k; // rdx
  __int64 v29; // [rsp+0h] [rbp-40h]
  __int64 v30; // [rsp+0h] [rbp-40h]
  __int64 v31; // [rsp+8h] [rbp-38h]
  __int64 v32; // [rsp+8h] [rbp-38h]
  __int64 v33; // [rsp+8h] [rbp-38h]
  __int64 v34; // [rsp+8h] [rbp-38h]

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
  result = (_QWORD *)sub_22077B0(16LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[2 * v3];
    for ( i = &result[2 * v7]; i != result; result += 2 )
    {
      if ( result )
        *result = -8;
    }
    for ( j = v4; v8 != j; j += 2 )
    {
      v11 = *j;
      if ( *j == -16 || v11 == -8 )
        continue;
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
      v18 = (_QWORD *)(v14 + 16LL * v17);
      v19 = *v18;
      if ( v11 != *v18 )
      {
        while ( v19 != -8 )
        {
          if ( v19 == -16 && !v16 )
            v16 = v18;
          v17 = v13 & (v15 + v17);
          v18 = (_QWORD *)(v14 + 16LL * v17);
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
      j[1] = 0;
      ++*(_DWORD *)(a1 + 16);
      v20 = j[1];
      if ( !v20 )
        continue;
      if ( (*(_BYTE *)(v20 + 48) & 1) != 0 )
      {
        v22 = v20 + 56;
        v23 = v20 + 248;
      }
      else
      {
        v21 = *(unsigned int *)(v20 + 64);
        v22 = *(_QWORD *)(v20 + 56);
        if ( !(_DWORD)v21 )
          goto LABEL_29;
        v23 = v22 + 48 * v21;
      }
      do
      {
        if ( *(_QWORD *)v22 != -16 && *(_QWORD *)v22 != -8 && *(_DWORD *)(v22 + 8) == 3 )
        {
          if ( *(_DWORD *)(v22 + 40) > 0x40u )
          {
            v25 = *(_QWORD *)(v22 + 32);
            if ( v25 )
            {
              v29 = v23;
              v33 = v20;
              j_j___libc_free_0_0(v25);
              v23 = v29;
              v20 = v33;
            }
          }
          if ( *(_DWORD *)(v22 + 24) > 0x40u )
          {
            v26 = *(_QWORD *)(v22 + 16);
            if ( v26 )
            {
              v30 = v23;
              v34 = v20;
              j_j___libc_free_0_0(v26);
              v23 = v30;
              v20 = v34;
            }
          }
        }
        v22 += 48;
      }
      while ( v22 != v23 );
      if ( (*(_BYTE *)(v20 + 48) & 1) != 0 )
        goto LABEL_22;
      v22 = *(_QWORD *)(v20 + 56);
LABEL_29:
      v32 = v20;
      j___libc_free_0(v22);
      v20 = v32;
LABEL_22:
      *(_QWORD *)v20 = &unk_49EE2B0;
      v24 = *(_QWORD *)(v20 + 24);
      if ( v24 != 0 && v24 != -8 && v24 != -16 )
      {
        v31 = v20;
        sub_1649B30(v20 + 8);
        v20 = v31;
      }
      j_j___libc_free_0(v20, 248);
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v27 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[2 * v27]; k != result; result += 2 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
