// Function: sub_23965B0
// Address: 0x23965b0
//
void __fastcall sub_23965B0(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 *v5; // rbx
  bool v6; // zf
  _QWORD *v7; // rax
  __int64 v8; // rdx
  _QWORD *i; // rdx
  __int64 v10; // rsi
  int v11; // edi
  int v12; // r10d
  _QWORD *v13; // r9
  unsigned int v14; // ecx
  _QWORD *v15; // rdx
  __int64 v16; // r8
  __int64 v17; // rax
  unsigned __int64 *v18; // rax
  unsigned __int64 v19; // r14
  __int64 v20; // rax
  int v21; // edx

  v5 = a2;
  v6 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v6 )
  {
    v7 = *(_QWORD **)(a1 + 16);
    v8 = 2LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v7 = (_QWORD *)(a1 + 16);
    v8 = 4;
  }
  for ( i = &v7[v8]; i != v7; v7 += 2 )
  {
    if ( v7 )
      *v7 = -4096;
  }
  if ( a2 != a3 )
  {
    do
    {
      v20 = *v5;
      if ( *v5 != -4096 && v20 != -8192 )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v10 = a1 + 16;
          v11 = 1;
        }
        else
        {
          v21 = *(_DWORD *)(a1 + 24);
          v10 = *(_QWORD *)(a1 + 16);
          if ( !v21 )
          {
            MEMORY[0] = *v5;
            BUG();
          }
          v11 = v21 - 1;
        }
        v12 = 1;
        v13 = 0;
        v14 = v11 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        v15 = (_QWORD *)(v10 + 16LL * v14);
        v16 = *v15;
        if ( v20 != *v15 )
        {
          while ( v16 != -4096 )
          {
            if ( v16 == -8192 && !v13 )
              v13 = v15;
            v14 = v11 & (v12 + v14);
            v15 = (_QWORD *)(v10 + 16LL * v14);
            v16 = *v15;
            if ( v20 == *v15 )
              goto LABEL_11;
            ++v12;
          }
          if ( v13 )
            v15 = v13;
        }
LABEL_11:
        *v15 = v20;
        v15[1] = v5[1];
        v5[1] = 0;
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        v17 = v5[1];
        if ( v17 )
        {
          if ( (v17 & 4) != 0 )
          {
            v18 = (unsigned __int64 *)(v17 & 0xFFFFFFFFFFFFFFF8LL);
            v19 = (unsigned __int64)v18;
            if ( v18 )
            {
              if ( (unsigned __int64 *)*v18 != v18 + 2 )
                _libc_free(*v18);
              j_j___libc_free_0(v19);
            }
          }
        }
      }
      v5 += 2;
    }
    while ( a3 != v5 );
  }
}
