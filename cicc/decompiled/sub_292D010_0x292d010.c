// Function: sub_292D010
// Address: 0x292d010
//
void __fastcall sub_292D010(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 *v5; // rbx
  bool v6; // zf
  _QWORD *v7; // rax
  __int64 v8; // rdx
  _QWORD *i; // rdx
  _QWORD *v10; // rdx
  __int64 v11; // r10
  int v12; // ecx
  __int64 v13; // rdi
  __int64 v14; // rsi
  int v15; // r9d
  _QWORD *v16; // r8
  unsigned __int64 v17; // rdi
  __int64 v18; // rax
  int v19; // ecx

  v5 = a2;
  v6 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v6 )
  {
    v7 = *(_QWORD **)(a1 + 16);
    v8 = 4LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v7 = (_QWORD *)(a1 + 16);
    v8 = 4;
  }
  for ( i = &v7[v8]; i != v7; v7 += 4 )
  {
    if ( v7 )
      *v7 = -4096;
  }
  if ( a2 != a3 )
  {
    do
    {
      v18 = *v5;
      if ( *v5 != -4096 && v18 != -8192 )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v10 = (_QWORD *)(a1 + 16);
          LODWORD(v11) = 0;
          v12 = 0;
          v13 = a1 + 16;
        }
        else
        {
          v19 = *(_DWORD *)(a1 + 24);
          v13 = *(_QWORD *)(a1 + 16);
          if ( !v19 )
          {
            MEMORY[0] = *v5;
            BUG();
          }
          v12 = v19 - 1;
          v11 = v12 & (((unsigned int)v18 >> 4) ^ ((unsigned int)v18 >> 9));
          v10 = (_QWORD *)(v13 + 32 * v11);
        }
        v14 = *v10;
        v15 = 1;
        v16 = 0;
        if ( v18 != *v10 )
        {
          while ( v14 != -4096 )
          {
            if ( v14 == -8192 && !v16 )
              v16 = v10;
            v11 = v12 & (unsigned int)(v11 + v15);
            v10 = (_QWORD *)(v13 + 32 * v11);
            v14 = *v10;
            if ( v18 == *v10 )
              goto LABEL_11;
            ++v15;
          }
          if ( v16 )
            v10 = v16;
        }
LABEL_11:
        *v10 = v18;
        v10[1] = v5[1];
        v10[2] = v5[2];
        v10[3] = v5[3];
        v5[1] = 0;
        v5[3] = 0;
        v5[2] = 0;
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        v17 = v5[1];
        if ( v17 )
          j_j___libc_free_0(v17);
      }
      v5 += 4;
    }
    while ( a3 != v5 );
  }
}
