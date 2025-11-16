// Function: sub_349EBA0
// Address: 0x349eba0
//
void __fastcall sub_349EBA0(__int64 a1, int *a2, int *a3)
{
  int *v5; // rbx
  bool v6; // zf
  _DWORD *v7; // rax
  __int64 v8; // rdx
  _DWORD *i; // rdx
  int v10; // eax
  __int64 v11; // rdi
  int v12; // ecx
  int v13; // r9d
  int *v14; // r8
  __int64 v15; // r10
  int *v16; // rdx
  int v17; // esi
  __int64 v18; // r13
  unsigned __int64 v19; // r12
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  int v22; // edx

  v5 = a2;
  v6 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v6 )
  {
    v7 = *(_DWORD **)(a1 + 16);
    v8 = 8LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v7 = (_DWORD *)(a1 + 16);
    v8 = 32;
  }
  for ( i = &v7[v8]; i != v7; v7 += 8 )
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
            v22 = *(_DWORD *)(a1 + 24);
            v11 = *(_QWORD *)(a1 + 16);
            if ( !v22 )
            {
              MEMORY[0] = 0;
              BUG();
            }
            v12 = v22 - 1;
          }
          v13 = 1;
          v14 = 0;
          LODWORD(v15) = v12 & (37 * v10);
          v16 = (int *)(v11 + 32LL * (unsigned int)v15);
          v17 = *v16;
          if ( v10 != *v16 )
          {
            while ( v17 != -1 )
            {
              if ( v17 == -2 && !v14 )
                v14 = v16;
              v15 = v12 & (unsigned int)(v15 + v13);
              v16 = (int *)(v11 + 32 * v15);
              v17 = *v16;
              if ( v10 == *v16 )
                goto LABEL_14;
              ++v13;
            }
            if ( v14 )
              v16 = v14;
          }
LABEL_14:
          *v16 = v10;
          *((_QWORD *)v16 + 1) = *((_QWORD *)v5 + 1);
          *((_QWORD *)v16 + 2) = *((_QWORD *)v5 + 2);
          *((_QWORD *)v16 + 3) = *((_QWORD *)v5 + 3);
          *((_QWORD *)v5 + 2) = 0;
          *((_QWORD *)v5 + 1) = 0;
          *((_QWORD *)v5 + 3) = 0;
          *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
          v18 = *((_QWORD *)v5 + 2);
          v19 = *((_QWORD *)v5 + 1);
          if ( v18 != v19 )
          {
            do
            {
              v20 = *(_QWORD *)(v19 + 336);
              if ( v20 != v19 + 352 )
                _libc_free(v20);
              v21 = *(_QWORD *)(v19 + 64);
              if ( v21 != v19 + 80 )
                _libc_free(v21);
              v19 += 384LL;
            }
            while ( v18 != v19 );
            v19 = *((_QWORD *)v5 + 1);
          }
          if ( v19 )
            break;
        }
        v5 += 8;
        if ( a3 == v5 )
          return;
      }
      v5 += 8;
      j_j___libc_free_0(v19);
    }
    while ( a3 != v5 );
  }
}
