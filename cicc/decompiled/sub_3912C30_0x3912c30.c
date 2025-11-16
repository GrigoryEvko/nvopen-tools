// Function: sub_3912C30
// Address: 0x3912c30
//
void __fastcall sub_3912C30(__int64 a1, int a2)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rdi
  _DWORD *v6; // rax
  __int64 v7; // rdx
  unsigned __int64 v8; // rbx
  _DWORD *i; // rdx
  unsigned __int64 v10; // rax
  int v11; // edx
  int v12; // ecx
  int v13; // esi
  __int64 v14; // r9
  int *v15; // r10
  int v16; // r11d
  unsigned int v17; // edi
  int *v18; // rcx
  int v19; // r8d
  __int64 v20; // rdx
  _DWORD *j; // rdx

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
  v6 = (_DWORD *)sub_22077B0(16LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = v6;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = v4 + 16 * v3;
    for ( i = &v6[4 * v7]; i != v6; v6 += 4 )
    {
      if ( v6 )
        *v6 = -1;
    }
    if ( v8 != v4 )
    {
      v10 = v4;
      do
      {
        while ( 1 )
        {
          v11 = *(_DWORD *)v10;
          if ( *(_DWORD *)v10 <= 0xFFFFFFFD )
            break;
          v10 += 16LL;
          if ( v8 == v10 )
            goto LABEL_15;
        }
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
        v17 = (v12 - 1) & (37 * v11);
        v18 = (int *)(v14 + 16LL * v17);
        v19 = *v18;
        if ( v11 != *v18 )
        {
          while ( v19 != -1 )
          {
            if ( !v15 && v19 == -2 )
              v15 = v18;
            v17 = v13 & (v16 + v17);
            v18 = (int *)(v14 + 16LL * v17);
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
        v20 = *(_QWORD *)(v10 + 4);
        v10 += 16LL;
        *(_QWORD *)(v18 + 1) = v20;
        v18[3] = *(_DWORD *)(v10 - 4);
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v8 != v10 );
    }
LABEL_15:
    j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &v6[4 * *(unsigned int *)(a1 + 24)]; j != v6; v6 += 4 )
    {
      if ( v6 )
        *v6 = -1;
    }
  }
}
