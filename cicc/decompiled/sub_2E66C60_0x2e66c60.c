// Function: sub_2E66C60
// Address: 0x2e66c60
//
void __fastcall sub_2E66C60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  bool v6; // zf
  _QWORD *v7; // rax
  __int64 v8; // rdx
  _QWORD *i; // rdx
  __int64 v10; // rdi
  int v11; // esi
  __int64 v12; // r9
  __int64 v13; // r8
  unsigned int v14; // edx
  __int64 *v15; // r12
  __int64 v16; // rcx
  __int64 v17; // rdx
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  __int64 v20; // rax
  int v21; // edx

  v5 = a2;
  v6 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v6 )
  {
    v7 = *(_QWORD **)(a1 + 16);
    v8 = 9LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v7 = (_QWORD *)(a1 + 16);
    v8 = 36;
  }
  for ( i = &v7[v8]; i != v7; v7 += 9 )
  {
    if ( v7 )
      *v7 = -4096;
  }
  if ( a2 != a3 )
  {
    do
    {
      v20 = *(_QWORD *)v5;
      if ( *(_QWORD *)v5 != -8192 && v20 != -4096 )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v10 = a1 + 16;
          v11 = 3;
        }
        else
        {
          v21 = *(_DWORD *)(a1 + 24);
          v10 = *(_QWORD *)(a1 + 16);
          if ( !v21 )
          {
            MEMORY[0] = *(_QWORD *)v5;
            BUG();
          }
          v11 = v21 - 1;
        }
        v12 = 1;
        v13 = 0;
        v14 = v11 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        v15 = (__int64 *)(v10 + 72LL * v14);
        v16 = *v15;
        if ( v20 != *v15 )
        {
          while ( v16 != -4096 )
          {
            if ( v16 == -8192 && !v13 )
              v13 = (__int64)v15;
            v14 = v11 & (v12 + v14);
            v15 = (__int64 *)(v10 + 72LL * v14);
            v16 = *v15;
            if ( v20 == *v15 )
              goto LABEL_11;
            v12 = (unsigned int)(v12 + 1);
          }
          if ( v13 )
            v15 = (__int64 *)v13;
        }
LABEL_11:
        *v15 = v20;
        v15[1] = (__int64)(v15 + 3);
        v15[2] = 0x200000000LL;
        v17 = *(unsigned int *)(v5 + 16);
        if ( (_DWORD)v17 )
          sub_2E64060((__int64)(v15 + 1), (char **)(v5 + 8), v17, v16, v13, v12);
        v15[6] = 0x200000000LL;
        v15[5] = (__int64)(v15 + 7);
        if ( *(_DWORD *)(v5 + 48) )
          sub_2E64060((__int64)(v15 + 5), (char **)(v5 + 40), v17, v16, v13, v12);
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        v18 = *(_QWORD *)(v5 + 40);
        if ( v18 != v5 + 56 )
          _libc_free(v18);
        v19 = *(_QWORD *)(v5 + 8);
        if ( v19 != v5 + 24 )
          _libc_free(v19);
      }
      v5 += 72;
    }
    while ( a3 != v5 );
  }
}
