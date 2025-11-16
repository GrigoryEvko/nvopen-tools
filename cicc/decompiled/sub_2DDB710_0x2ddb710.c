// Function: sub_2DDB710
// Address: 0x2ddb710
//
void __fastcall sub_2DDB710(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int *v8; // rax
  int *v9; // rbx
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  int v12; // r14d
  int *v13; // rsi
  __int64 v14; // rdx
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rax
  __int64 v17; // rcx
  int v18; // edx
  unsigned __int64 v19; // rax
  int v20; // edx

  if ( a1 != a2 )
  {
    v8 = *(int **)a2;
    v9 = (int *)(a2 + 16);
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v10 = *(unsigned int *)(a2 + 8);
      v11 = *(unsigned int *)(a1 + 8);
      v12 = *(_DWORD *)(a2 + 8);
      if ( v10 <= v11 )
      {
        if ( *(_DWORD *)(a2 + 8) )
        {
          v16 = *(_QWORD *)a1;
          v17 = a2 + 8 * v10 + 16;
          do
          {
            v18 = *v9;
            v9 += 2;
            v16 += 8LL;
            *(_DWORD *)(v16 - 8) = v18;
            *(_DWORD *)(v16 - 4) = *(v9 - 1);
          }
          while ( v9 != (int *)v17 );
        }
        goto LABEL_8;
      }
      if ( v10 > *(unsigned int *)(a1 + 12) )
      {
        *(_DWORD *)(a1 + 8) = 0;
        sub_C8D5F0(a1, (const void *)(a1 + 16), v10, 8u, a5, a6);
        v11 = 0;
        v14 = 2LL * *(unsigned int *)(a2 + 8);
        v13 = *(int **)a2;
        if ( *(_QWORD *)a2 == v14 * 4 + *(_QWORD *)a2 )
          goto LABEL_8;
      }
      else
      {
        v13 = (int *)(a2 + 16);
        if ( *(_DWORD *)(a1 + 8) )
        {
          v11 *= 8LL;
          v19 = *(_QWORD *)a1;
          do
          {
            v20 = *v9;
            v9 += 2;
            v19 += 8LL;
            *(_DWORD *)(v19 - 8) = v20;
            *(_DWORD *)(v19 - 4) = *(v9 - 1);
          }
          while ( (int *)(a2 + v11 + 16) != v9 );
          v9 = *(int **)a2;
          v10 = *(unsigned int *)(a2 + 8);
          v13 = (int *)(*(_QWORD *)a2 + v11);
        }
        v14 = 2 * v10;
        if ( v13 == &v9[v14] )
          goto LABEL_8;
      }
      memcpy((void *)(v11 + *(_QWORD *)a1), v13, v14 * 4 - v11);
LABEL_8:
      *(_DWORD *)(a1 + 8) = v12;
      *(_DWORD *)(a2 + 8) = 0;
      return;
    }
    v15 = *(_QWORD *)a1;
    if ( v15 != a1 + 16 )
    {
      _libc_free(v15);
      v8 = *(int **)a2;
    }
    *(_QWORD *)a1 = v8;
    *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
    *(_QWORD *)a2 = v9;
    *(_QWORD *)(a2 + 8) = 0;
  }
}
