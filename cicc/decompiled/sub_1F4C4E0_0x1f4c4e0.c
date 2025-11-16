// Function: sub_1F4C4E0
// Address: 0x1f4c4e0
//
void __fastcall sub_1F4C4E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  int *v7; // r12
  int *v8; // rax
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rcx
  int v13; // r14d
  int *v14; // rsi
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  __int64 v17; // rcx
  int v18; // edx
  unsigned __int64 v19; // rax
  int v20; // edx

  if ( a1 != a2 )
  {
    v7 = (int *)(a2 + 16);
    v8 = *(int **)a2;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v11 = *(unsigned int *)(a2 + 8);
      v12 = *(unsigned int *)(a1 + 8);
      v13 = *(_DWORD *)(a2 + 8);
      if ( v11 <= v12 )
      {
        if ( *(_DWORD *)(a2 + 8) )
        {
          v16 = *(_QWORD *)a1;
          v17 = a2 + 8 * v11 + 16;
          do
          {
            v18 = *v7;
            v7 += 2;
            v16 += 8LL;
            *(_DWORD *)(v16 - 8) = v18;
            *(_DWORD *)(v16 - 4) = *(v7 - 1);
          }
          while ( v7 != (int *)v17 );
        }
      }
      else
      {
        if ( v11 > *(unsigned int *)(a1 + 12) )
        {
          *(_DWORD *)(a1 + 8) = 0;
          sub_16CD150(a1, (const void *)(a1 + 16), v11, 8, a5, a6);
          v7 = *(int **)a2;
          v11 = *(unsigned int *)(a2 + 8);
          v12 = 0;
          v14 = *(int **)a2;
        }
        else
        {
          v14 = (int *)(a2 + 16);
          if ( *(_DWORD *)(a1 + 8) )
          {
            v12 *= 8LL;
            v19 = *(_QWORD *)a1;
            do
            {
              v20 = *v7;
              v7 += 2;
              v19 += 8LL;
              *(_DWORD *)(v19 - 8) = v20;
              *(_DWORD *)(v19 - 4) = *(v7 - 1);
            }
            while ( (int *)(a2 + v12 + 16) != v7 );
            v7 = *(int **)a2;
            v11 = *(unsigned int *)(a2 + 8);
            v14 = (int *)(*(_QWORD *)a2 + v12);
          }
        }
        v15 = 2 * v11;
        if ( v14 != &v7[v15] )
          memcpy((void *)(v12 + *(_QWORD *)a1), v14, v15 * 4 - v12);
      }
      *(_DWORD *)(a1 + 8) = v13;
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v10 = *(_QWORD *)a1;
      if ( v10 != a1 + 16 )
      {
        _libc_free(v10);
        v8 = *(int **)a2;
      }
      *(_QWORD *)a1 = v8;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v7;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
