// Function: sub_390E9A0
// Address: 0x390e9a0
//
void __fastcall sub_390E9A0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rax
  _QWORD *v6; // rax
  unsigned __int64 v7; // r15
  _QWORD *i; // rdx
  unsigned __int64 v9; // rbx
  __int64 v10; // rax
  int v11; // edx
  __int64 v12; // rdx
  __int64 v13; // r8
  _QWORD *v14; // r10
  unsigned int v15; // r9d
  __int64 v16; // rcx
  _QWORD *v17; // rdi
  __int64 v18; // rsi
  unsigned __int64 v19; // rdi
  unsigned int v20; // r11d
  __int64 v21; // rcx
  __int64 v22; // rdx
  _QWORD *j; // rdx

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
  v6 = (_QWORD *)sub_22077B0(88LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = v6;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = v4 + 88 * v3;
    for ( i = &v6[11 * *(unsigned int *)(a1 + 24)]; i != v6; v6 += 11 )
    {
      if ( v6 )
        *v6 = -8;
    }
    if ( v7 != v4 )
    {
      v9 = v4;
      do
      {
        v10 = *(_QWORD *)v9;
        if ( *(_QWORD *)v9 != -16 && v10 != -8 )
        {
          v11 = *(_DWORD *)(a1 + 24);
          if ( !v11 )
          {
            MEMORY[0] = *(_QWORD *)v9;
            BUG();
          }
          v12 = (unsigned int)(v11 - 1);
          v13 = *(_QWORD *)(a1 + 8);
          v14 = 0;
          v15 = v12 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
          v16 = 1;
          v17 = (_QWORD *)(v13 + 88LL * v15);
          v18 = *v17;
          if ( v10 != *v17 )
          {
            while ( v18 != -8 )
            {
              if ( !v14 && v18 == -16 )
                v14 = v17;
              v20 = v16 + 1;
              v21 = (unsigned int)v12 & (v15 + (_DWORD)v16);
              v15 = v21;
              v16 = 11 * v21;
              v17 = (_QWORD *)(v13 + 8 * v16);
              v18 = *v17;
              if ( v10 == *v17 )
                goto LABEL_14;
              v16 = v20;
            }
            if ( v14 )
              v17 = v14;
          }
LABEL_14:
          *v17 = v10;
          v17[1] = v17 + 3;
          v17[2] = 0x800000000LL;
          if ( *(_DWORD *)(v9 + 16) )
            sub_390DA00((__int64)(v17 + 1), (char **)(v9 + 8), v12, v16, v13, v15);
          ++*(_DWORD *)(a1 + 16);
          v19 = *(_QWORD *)(v9 + 8);
          if ( v19 != v9 + 24 )
            _libc_free(v19);
        }
        v9 += 88LL;
      }
      while ( v7 != v9 );
    }
    j___libc_free_0(v4);
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &v6[11 * v22]; j != v6; v6 += 11 )
    {
      if ( v6 )
        *v6 = -8;
    }
  }
}
