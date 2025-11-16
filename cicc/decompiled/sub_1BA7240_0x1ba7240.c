// Function: sub_1BA7240
// Address: 0x1ba7240
//
_DWORD *__fastcall sub_1BA7240(__int64 a1, int a2)
{
  __int64 v3; // rbx
  _DWORD *v4; // r14
  unsigned int v5; // eax
  _DWORD *result; // rax
  _DWORD *v7; // r15
  _DWORD *i; // rdx
  _DWORD *v9; // rbx
  char *v10; // rax
  unsigned int v11; // eax
  int v12; // edx
  int v13; // edx
  __int64 v14; // rdi
  unsigned int *v15; // r9
  int v16; // r10d
  unsigned int v17; // ecx
  unsigned int *v18; // r8
  unsigned int v19; // esi
  void *v20; // rdi
  unsigned int v21; // r9d
  unsigned __int64 v22; // rdi
  _DWORD *v23; // r10
  const void *v24; // rsi
  size_t v25; // rdx
  _DWORD *j; // rdx
  _DWORD *v27; // [rsp+8h] [rbp-48h]
  _DWORD *v28; // [rsp+8h] [rbp-48h]
  unsigned int *v29; // [rsp+10h] [rbp-40h]
  unsigned int *v30; // [rsp+10h] [rbp-40h]
  unsigned int v31; // [rsp+1Ch] [rbp-34h]
  unsigned int v32; // [rsp+1Ch] [rbp-34h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_DWORD **)(a1 + 8);
  v5 = sub_1454B60((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_DWORD *)sub_22077B0(40LL * v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v4[10 * v3];
    for ( i = &result[10 * *(unsigned int *)(a1 + 24)]; i != result; result += 10 )
    {
      if ( result )
        *result = -1;
    }
    v9 = v4 + 6;
    if ( v7 != v4 )
    {
      while ( 1 )
      {
        v11 = *(v9 - 6);
        if ( v11 > 0xFFFFFFFD )
          goto LABEL_10;
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
        v18 = (unsigned int *)(v14 + 40LL * v17);
        v19 = *v18;
        if ( v11 != *v18 )
        {
          while ( v19 != -1 )
          {
            if ( !v15 && v19 == -2 )
              v15 = v18;
            v17 = v13 & (v16 + v17);
            v18 = (unsigned int *)(v14 + 40LL * v17);
            v19 = *v18;
            if ( v11 == *v18 )
              goto LABEL_15;
            ++v16;
          }
          if ( v15 )
            v18 = v15;
        }
LABEL_15:
        v20 = v18 + 6;
        *v18 = v11;
        *((_QWORD *)v18 + 1) = v18 + 6;
        *((_QWORD *)v18 + 2) = 0x200000000LL;
        v21 = *(v9 - 2);
        if ( v21 && v18 + 2 != v9 - 4 )
        {
          v23 = (_DWORD *)*((_QWORD *)v9 - 2);
          if ( v9 == v23 )
          {
            v24 = v9;
            v25 = 8LL * v21;
            if ( v21 <= 2 )
              goto LABEL_23;
            v28 = (_DWORD *)*((_QWORD *)v9 - 2);
            v30 = v18;
            v32 = *(v9 - 2);
            sub_16CD150((__int64)(v18 + 2), v18 + 6, v21, 8, (int)v18, v21);
            v18 = v30;
            v24 = (const void *)*((_QWORD *)v9 - 2);
            v21 = v32;
            v25 = 8LL * (unsigned int)*(v9 - 2);
            v20 = (void *)*((_QWORD *)v30 + 1);
            v23 = v28;
            if ( v25 )
            {
LABEL_23:
              v27 = v23;
              v29 = v18;
              v31 = v21;
              memcpy(v20, v24, v25);
              v23 = v27;
              v18 = v29;
              v21 = v31;
            }
            v18[4] = v21;
            *(v23 - 2) = 0;
          }
          else
          {
            *((_QWORD *)v18 + 1) = v23;
            v18[4] = *(v9 - 2);
            v18[5] = *(v9 - 1);
            *((_QWORD *)v9 - 2) = v9;
            *(v9 - 1) = 0;
            *(v9 - 2) = 0;
          }
        }
        ++*(_DWORD *)(a1 + 16);
        v22 = *((_QWORD *)v9 - 2);
        if ( v9 == (_DWORD *)v22 )
        {
LABEL_10:
          v10 = (char *)(v9 + 10);
          if ( v7 == v9 + 4 )
            return (_DWORD *)j___libc_free_0(v4);
        }
        else
        {
          _libc_free(v22);
          v10 = (char *)(v9 + 10);
          if ( v7 == v9 + 4 )
            return (_DWORD *)j___libc_free_0(v4);
        }
        v9 = v10;
      }
    }
    return (_DWORD *)j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[10 * *(unsigned int *)(a1 + 24)]; j != result; result += 10 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
