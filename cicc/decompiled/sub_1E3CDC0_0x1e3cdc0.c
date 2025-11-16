// Function: sub_1E3CDC0
// Address: 0x1e3cdc0
//
_QWORD *__fastcall sub_1E3CDC0(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rcx
  unsigned int k; // r9d
  __int64 v9; // rdx
  __int64 v10; // r14
  _QWORD *i; // rdx
  __int64 j; // rbx
  unsigned __int64 v13; // rdx
  int v14; // r8d
  __int64 v15; // r15
  int v16; // eax
  __int64 v17; // rdi
  __int64 v18; // r10
  int v19; // r11d
  int v20; // r8d
  char v21; // al
  __int64 v22; // rax
  _QWORD *m; // rdx
  unsigned int v24; // r9d
  int v25; // [rsp+4h] [rbp-4Ch]
  __int64 v26; // [rsp+8h] [rbp-48h]
  __int64 *v27; // [rsp+10h] [rbp-40h]
  unsigned int v28; // [rsp+18h] [rbp-38h]
  int v29; // [rsp+1Ch] [rbp-34h]
  int v30; // [rsp+1Ch] [rbp-34h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
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
    v9 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v10 = v4 + 16 * v3;
    for ( i = &result[2 * v9]; i != result; result += 2 )
    {
      if ( result )
        *result = 0;
    }
    for ( j = v4; v10 != j; j += 16 )
    {
      v13 = *(_QWORD *)j - 1LL;
      if ( v13 <= 0xFFFFFFFFFFFFFFFDLL )
      {
        v14 = *(_DWORD *)(a1 + 24);
        v29 = v14;
        if ( !v14 )
        {
          MEMORY[0] = *(_QWORD *)j;
          BUG();
        }
        v15 = *(_QWORD *)(a1 + 8);
        v16 = sub_1E1C690((__int64 *)j, a2, v13, v7, v14, k);
        v17 = *(_QWORD *)j;
        v18 = 0;
        v19 = 1;
        v20 = v29 - 1;
        for ( k = (v29 - 1) & v16; ; k = v20 & v24 )
        {
          v7 = v15 + 16LL * k;
          a2 = *(_QWORD *)v7;
          if ( (unsigned __int64)(*(_QWORD *)v7 - 1LL) > 0xFFFFFFFFFFFFFFFDLL
            || (unsigned __int64)(v17 - 1) > 0xFFFFFFFFFFFFFFFDLL )
          {
            if ( v17 == a2 )
              goto LABEL_18;
            v22 = *(_QWORD *)v7;
            a2 = v17;
          }
          else
          {
            v25 = v19;
            v26 = v18;
            v27 = (__int64 *)(v15 + 16LL * k);
            v28 = k;
            v30 = v20;
            v21 = sub_1E15D60(v17, a2, 3u);
            v20 = v30;
            k = v28;
            v7 = (__int64)v27;
            v18 = v26;
            v19 = v25;
            if ( v21 )
            {
              a2 = *(_QWORD *)j;
              goto LABEL_18;
            }
            v22 = *v27;
            a2 = *(_QWORD *)j;
          }
          if ( !v22 )
            break;
          if ( !v18 && v22 == -1 )
            v18 = v7;
          v24 = v19 + k;
          v17 = a2;
          ++v19;
        }
        if ( v18 )
          v7 = v18;
LABEL_18:
        *(_QWORD *)v7 = a2;
        *(_DWORD *)(v7 + 8) = *(_DWORD *)(j + 8);
        ++*(_DWORD *)(a1 + 16);
      }
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( m = &result[2 * *(unsigned int *)(a1 + 24)]; m != result; result += 2 )
    {
      if ( result )
        *result = 0;
    }
  }
  return result;
}
