// Function: sub_2ED90B0
// Address: 0x2ed90b0
//
void __fastcall sub_2ED90B0(__int64 a1, int *a2, int *a3)
{
  int *v5; // rbx
  bool v6; // zf
  _DWORD *v7; // rax
  __int64 v8; // rdx
  _DWORD *i; // rdx
  int v10; // eax
  __int64 v11; // rdi
  int v12; // esi
  int v13; // r10d
  int *v14; // r9
  unsigned int v15; // ecx
  int *v16; // rdx
  int v17; // r8d
  __int64 v18; // rax
  unsigned __int64 *v19; // rax
  unsigned __int64 v20; // r14
  int v21; // edx

  v5 = a2;
  v6 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v6 )
  {
    v7 = *(_DWORD **)(a1 + 16);
    v8 = 4LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v7 = (_DWORD *)(a1 + 16);
    v8 = 16;
  }
  for ( i = &v7[v8]; i != v7; v7 += 4 )
  {
    if ( v7 )
      *v7 = -1;
  }
  if ( a2 != a3 )
  {
    do
    {
      while ( 1 )
      {
        v10 = *v5;
        if ( (unsigned int)*v5 <= 0xFFFFFFFD )
        {
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v11 = a1 + 16;
            v12 = 3;
          }
          else
          {
            v21 = *(_DWORD *)(a1 + 24);
            v11 = *(_QWORD *)(a1 + 16);
            if ( !v21 )
            {
              MEMORY[0] = *v5;
              BUG();
            }
            v12 = v21 - 1;
          }
          v13 = 1;
          v14 = 0;
          v15 = v12 & (37 * v10);
          v16 = (int *)(v11 + 16LL * v15);
          v17 = *v16;
          if ( v10 != *v16 )
          {
            while ( v17 != -1 )
            {
              if ( v17 == -2 && !v14 )
                v14 = v16;
              v15 = v12 & (v13 + v15);
              v16 = (int *)(v11 + 16LL * v15);
              v17 = *v16;
              if ( v10 == *v16 )
                goto LABEL_14;
              ++v13;
            }
            if ( v14 )
              v16 = v14;
          }
LABEL_14:
          *v16 = *v5;
          *((_QWORD *)v16 + 1) = *((_QWORD *)v5 + 1);
          *((_QWORD *)v5 + 1) = 0;
          *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
          v18 = *((_QWORD *)v5 + 1);
          if ( v18 )
          {
            if ( (v18 & 2) != 0 )
            {
              v19 = (unsigned __int64 *)(v18 & 0xFFFFFFFFFFFFFFFCLL);
              v20 = (unsigned __int64)v19;
              if ( v19 )
                break;
            }
          }
        }
        v5 += 4;
        if ( a3 == v5 )
          return;
      }
      if ( (unsigned __int64 *)*v19 != v19 + 2 )
        _libc_free(*v19);
      v5 += 4;
      j_j___libc_free_0(v20);
    }
    while ( a3 != v5 );
  }
}
