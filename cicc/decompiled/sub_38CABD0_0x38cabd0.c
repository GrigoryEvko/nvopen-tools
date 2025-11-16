// Function: sub_38CABD0
// Address: 0x38cabd0
//
void __fastcall sub_38CABD0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rdi
  _QWORD *v6; // rax
  __int64 v7; // rdx
  unsigned __int64 v8; // rbx
  _QWORD *i; // rdx
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  int v12; // ecx
  int v13; // edi
  __int64 v14; // r8
  int v15; // r14d
  __int64 *v16; // r10
  unsigned int v17; // esi
  __int64 *v18; // rcx
  __int64 v19; // r9
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
  v6 = (_QWORD *)sub_22077B0(16LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = v6;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = v4 + 16 * v3;
    for ( i = &v6[2 * v7]; i != v6; v6 += 2 )
    {
      if ( v6 )
        *v6 = -8;
    }
    if ( v8 != v4 )
    {
      v10 = v4;
      do
      {
        v11 = *(_QWORD *)v10;
        if ( *(_QWORD *)v10 != -16 && v11 != -8 )
        {
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = *(_QWORD *)v10;
            BUG();
          }
          v13 = v12 - 1;
          v14 = *(_QWORD *)(a1 + 8);
          v15 = 1;
          v16 = 0;
          v17 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v18 = (__int64 *)(v14 + 16LL * v17);
          v19 = *v18;
          if ( v11 != *v18 )
          {
            while ( v19 != -8 )
            {
              if ( v19 == -16 && !v16 )
                v16 = v18;
              v17 = v13 & (v15 + v17);
              v18 = (__int64 *)(v14 + 16LL * v17);
              v19 = *v18;
              if ( v11 == *v18 )
                goto LABEL_14;
              ++v15;
            }
            if ( v16 )
              v18 = v16;
          }
LABEL_14:
          *v18 = v11;
          *((_DWORD *)v18 + 2) = *(_DWORD *)(v10 + 8);
          ++*(_DWORD *)(a1 + 16);
        }
        v10 += 16LL;
      }
      while ( v8 != v10 );
    }
    j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &v6[2 * *(unsigned int *)(a1 + 24)]; j != v6; v6 += 2 )
    {
      if ( v6 )
        *v6 = -8;
    }
  }
}
