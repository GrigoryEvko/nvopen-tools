// Function: sub_2D32810
// Address: 0x2d32810
//
void __fastcall sub_2D32810(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  bool v6; // zf
  _QWORD *v7; // rax
  __int64 v8; // rdx
  _QWORD *i; // rdx
  __int64 v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // r10
  __int64 v15; // r9
  unsigned int j; // eax
  __int64 v17; // r15
  __int64 v18; // r8
  int v19; // ecx
  __int64 v20; // rax
  unsigned __int64 v21; // rdi
  int v22; // eax
  __int64 v23; // [rsp+8h] [rbp-38h]

  v5 = a2;
  v6 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v6 )
  {
    v7 = *(_QWORD **)(a1 + 16);
    v8 = 11LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v7 = (_QWORD *)(a1 + 16);
    v8 = 44;
  }
  for ( i = &v7[v8]; i != v7; v7 += 11 )
  {
    if ( v7 )
    {
      *v7 = -4096;
      v7[1] = -4096;
    }
  }
  v23 = a1 + 16;
  if ( a2 != a3 )
  {
    do
    {
      while ( 1 )
      {
        v10 = *(_QWORD *)v5;
        if ( *(_QWORD *)v5 != -4096 )
          break;
        if ( *(_QWORD *)(v5 + 8) == -4096 )
        {
LABEL_23:
          v5 += 88;
          if ( a3 == v5 )
            return;
        }
        else
        {
LABEL_10:
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v11 = v23;
            v12 = 3;
          }
          else
          {
            v19 = *(_DWORD *)(a1 + 24);
            v11 = *(_QWORD *)(a1 + 16);
            if ( !v19 )
            {
              MEMORY[0] = *(_QWORD *)v5;
              BUG();
            }
            v12 = (unsigned int)(v19 - 1);
          }
          v13 = *(_QWORD *)(v5 + 8);
          v14 = 0;
          v15 = 1;
          for ( j = v12
                  & (((0xBF58476D1CE4E5B9LL
                     * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)
                      | ((unsigned __int64)(((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)) << 32))) >> 31)
                   ^ (484763065 * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)))); ; j = v12 & v22 )
          {
            v17 = v11 + 88LL * j;
            v18 = *(_QWORD *)v17;
            if ( v10 == *(_QWORD *)v17 && *(_QWORD *)(v17 + 8) == v13 )
              break;
            if ( v18 == -4096 )
            {
              if ( *(_QWORD *)(v17 + 8) == -4096 )
              {
                if ( v14 )
                  v17 = v14;
                break;
              }
            }
            else if ( v18 == -8192 && *(_QWORD *)(v17 + 8) == -8192 && !v14 )
            {
              v14 = v11 + 88LL * j;
            }
            v22 = v15 + j;
            v15 = (unsigned int)(v15 + 1);
          }
          *(_QWORD *)v17 = v10;
          v20 = *(_QWORD *)(v5 + 8);
          *(_QWORD *)(v17 + 24) = 0x600000000LL;
          *(_QWORD *)(v17 + 8) = v20;
          *(_QWORD *)(v17 + 16) = v17 + 32;
          if ( *(_DWORD *)(v5 + 24) )
            sub_2D23900(v17 + 16, (char **)(v5 + 16), v10, v12, v18, v15);
          *(_DWORD *)(v17 + 80) = *(_DWORD *)(v5 + 80);
          *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
          v21 = *(_QWORD *)(v5 + 16);
          if ( v21 == v5 + 32 )
            goto LABEL_23;
          _libc_free(v21);
          v5 += 88;
          if ( a3 == v5 )
            return;
        }
      }
      if ( v10 != -8192 || *(_QWORD *)(v5 + 8) != -8192 )
        goto LABEL_10;
      v5 += 88;
    }
    while ( a3 != v5 );
  }
}
