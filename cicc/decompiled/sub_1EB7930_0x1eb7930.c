// Function: sub_1EB7930
// Address: 0x1eb7930
//
_DWORD *__fastcall sub_1EB7930(__int64 a1, int a2)
{
  __int64 v3; // rbx
  _DWORD *v4; // r13
  unsigned __int64 v5; // rax
  _DWORD *result; // rax
  __int64 v7; // rcx
  _DWORD *v8; // r15
  _DWORD *i; // rdx
  _DWORD *v10; // rbx
  char *v11; // rax
  unsigned int v12; // eax
  int v13; // edx
  int v14; // edx
  __int64 v15; // r9
  int v16; // r10d
  unsigned int v17; // ecx
  unsigned int *v18; // rdi
  unsigned int *v19; // r8
  unsigned int v20; // esi
  void *v21; // rdi
  unsigned int v22; // r9d
  unsigned __int64 v23; // rdi
  _DWORD *v24; // r10
  const void *v25; // rsi
  size_t v26; // rdx
  _DWORD *j; // rdx
  _DWORD *v28; // [rsp+8h] [rbp-48h]
  _DWORD *v29; // [rsp+8h] [rbp-48h]
  unsigned int *v30; // [rsp+10h] [rbp-40h]
  unsigned int *v31; // [rsp+10h] [rbp-40h]
  unsigned int v32; // [rsp+1Ch] [rbp-34h]
  unsigned int v33; // [rsp+1Ch] [rbp-34h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_DWORD **)(a1 + 8);
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
  result = (_DWORD *)sub_22077B0(56LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[14 * v3];
    for ( i = &result[14 * v7]; i != result; result += 14 )
    {
      if ( result )
        *result = -1;
    }
    v10 = v4 + 6;
    if ( v8 != v4 )
    {
      while ( 1 )
      {
        v12 = *(v10 - 6);
        if ( v12 > 0xFFFFFFFD )
          goto LABEL_10;
        v13 = *(_DWORD *)(a1 + 24);
        if ( !v13 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v14 = v13 - 1;
        v15 = *(_QWORD *)(a1 + 8);
        v16 = 1;
        v17 = v14 & (37 * v12);
        v18 = 0;
        v19 = (unsigned int *)(v15 + 56LL * v17);
        v20 = *v19;
        if ( v12 != *v19 )
        {
          while ( v20 != -1 )
          {
            if ( !v18 && v20 == -2 )
              v18 = v19;
            v17 = v14 & (v16 + v17);
            v19 = (unsigned int *)(v15 + 56LL * v17);
            v20 = *v19;
            if ( v12 == *v19 )
              goto LABEL_15;
            ++v16;
          }
          if ( v18 )
            v19 = v18;
        }
LABEL_15:
        v21 = v19 + 6;
        *v19 = v12;
        *((_QWORD *)v19 + 1) = v19 + 6;
        *((_QWORD *)v19 + 2) = 0x400000000LL;
        v22 = *(v10 - 2);
        if ( v22 && v19 + 2 != v10 - 4 )
        {
          v24 = (_DWORD *)*((_QWORD *)v10 - 2);
          if ( v10 == v24 )
          {
            v25 = v10;
            v26 = 8LL * v22;
            if ( v22 <= 4 )
              goto LABEL_23;
            v29 = (_DWORD *)*((_QWORD *)v10 - 2);
            v31 = v19;
            v33 = *(v10 - 2);
            sub_16CD150((__int64)(v19 + 2), v19 + 6, v22, 8, (int)v19, v22);
            v19 = v31;
            v25 = (const void *)*((_QWORD *)v10 - 2);
            v22 = v33;
            v26 = 8LL * (unsigned int)*(v10 - 2);
            v21 = (void *)*((_QWORD *)v31 + 1);
            v24 = v29;
            if ( v26 )
            {
LABEL_23:
              v28 = v24;
              v30 = v19;
              v32 = v22;
              memcpy(v21, v25, v26);
              v24 = v28;
              v19 = v30;
              v22 = v32;
            }
            v19[4] = v22;
            *(v24 - 2) = 0;
          }
          else
          {
            *((_QWORD *)v19 + 1) = v24;
            v19[4] = *(v10 - 2);
            v19[5] = *(v10 - 1);
            *((_QWORD *)v10 - 2) = v10;
            *(v10 - 1) = 0;
            *(v10 - 2) = 0;
          }
        }
        ++*(_DWORD *)(a1 + 16);
        v23 = *((_QWORD *)v10 - 2);
        if ( v10 == (_DWORD *)v23 )
        {
LABEL_10:
          v11 = (char *)(v10 + 14);
          if ( v8 == v10 + 8 )
            return (_DWORD *)j___libc_free_0(v4);
        }
        else
        {
          _libc_free(v23);
          v11 = (char *)(v10 + 14);
          if ( v8 == v10 + 8 )
            return (_DWORD *)j___libc_free_0(v4);
        }
        v10 = v11;
      }
    }
    return (_DWORD *)j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[14 * *(unsigned int *)(a1 + 24)]; j != result; result += 14 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
