// Function: sub_1B1DC30
// Address: 0x1b1dc30
//
void __fastcall sub_1B1DC30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // rdi
  __int64 *v8; // rax
  __int64 *v9; // r13
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rcx
  int v14; // r14d
  __int64 *v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rdx

  v7 = a1 + 96;
  if ( v7 != a2 )
  {
    v8 = *(__int64 **)a2;
    v9 = (__int64 *)(a2 + 16);
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v12 = *(unsigned int *)(a2 + 8);
      v13 = *(unsigned int *)(a1 + 104);
      v14 = *(_DWORD *)(a2 + 8);
      if ( v12 <= v13 )
      {
        if ( *(_DWORD *)(a2 + 8) )
        {
          v17 = *(_QWORD *)(a1 + 96);
          v18 = v17 + 16 * v12;
          do
          {
            v19 = *v9;
            v17 += 16;
            v9 += 2;
            *(_QWORD *)(v17 - 16) = v19;
            *(_QWORD *)(v17 - 8) = *(v9 - 1);
          }
          while ( v17 != v18 );
        }
      }
      else
      {
        if ( v12 > *(unsigned int *)(a1 + 108) )
        {
          *(_DWORD *)(a1 + 104) = 0;
          sub_16CD150(v7, (const void *)(a1 + 112), v12, 16, a5, a6);
          v9 = *(__int64 **)a2;
          v12 = *(unsigned int *)(a2 + 8);
          v13 = 0;
          v15 = *(__int64 **)a2;
        }
        else
        {
          v15 = (__int64 *)(a2 + 16);
          if ( *(_DWORD *)(a1 + 104) )
          {
            v20 = *(_QWORD *)(a1 + 96);
            v13 *= 16LL;
            v21 = v20 + v13;
            do
            {
              v22 = *v9;
              v20 += 16;
              v9 += 2;
              *(_QWORD *)(v20 - 16) = v22;
              *(_QWORD *)(v20 - 8) = *(v9 - 1);
            }
            while ( v20 != v21 );
            v9 = *(__int64 **)a2;
            v12 = *(unsigned int *)(a2 + 8);
            v15 = (__int64 *)(*(_QWORD *)a2 + v13);
          }
        }
        v16 = 2 * v12;
        if ( v15 != &v9[v16] )
          memcpy((void *)(v13 + *(_QWORD *)(a1 + 96)), v15, v16 * 8 - v13);
      }
      *(_DWORD *)(a1 + 104) = v14;
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v11 = *(_QWORD *)(a1 + 96);
      if ( v11 != a1 + 112 )
      {
        _libc_free(v11);
        v8 = *(__int64 **)a2;
      }
      *(_QWORD *)(a1 + 96) = v8;
      *(_DWORD *)(a1 + 104) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 108) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v9;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
