// Function: sub_9B6A20
// Address: 0x9b6a20
//
void __fastcall sub_9B6A20(__int64 *a1, __int64 a2)
{
  __int64 *v4; // rax
  __int64 *v5; // rbx
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // rcx
  int v8; // r14d
  __int64 *v9; // rsi
  __int64 v10; // rdx
  __int64 *v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rax
  unsigned __int64 v16; // rsi
  __int64 v17; // rdx

  if ( a1 != (__int64 *)a2 )
  {
    v4 = *(__int64 **)a2;
    v5 = (__int64 *)(a2 + 16);
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v6 = *(unsigned int *)(a2 + 8);
      v7 = *((unsigned int *)a1 + 2);
      v8 = *(_DWORD *)(a2 + 8);
      if ( v6 <= v7 )
      {
        if ( *(_DWORD *)(a2 + 8) )
        {
          v12 = *a1;
          v13 = *a1 + 16 * v6;
          do
          {
            v14 = *v5;
            v12 += 16;
            v5 += 2;
            *(_QWORD *)(v12 - 16) = v14;
            *(_QWORD *)(v12 - 8) = *(v5 - 1);
          }
          while ( v12 != v13 );
        }
        goto LABEL_8;
      }
      if ( v6 > *((unsigned int *)a1 + 3) )
      {
        *((_DWORD *)a1 + 2) = 0;
        sub_C8D5F0(a1, a1 + 2, v6, 16);
        v7 = 0;
        v10 = 2LL * *(unsigned int *)(a2 + 8);
        v9 = *(__int64 **)a2;
        if ( *(_QWORD *)a2 == v10 * 8 + *(_QWORD *)a2 )
          goto LABEL_8;
      }
      else
      {
        v9 = (__int64 *)(a2 + 16);
        if ( *((_DWORD *)a1 + 2) )
        {
          v15 = *a1;
          v7 *= 16LL;
          v16 = *a1 + v7;
          do
          {
            v17 = *v5;
            v15 += 16;
            v5 += 2;
            *(_QWORD *)(v15 - 16) = v17;
            *(_QWORD *)(v15 - 8) = *(v5 - 1);
          }
          while ( v15 != v16 );
          v5 = *(__int64 **)a2;
          v6 = *(unsigned int *)(a2 + 8);
          v9 = (__int64 *)(*(_QWORD *)a2 + v7);
        }
        v10 = 2 * v6;
        if ( v9 == &v5[v10] )
          goto LABEL_8;
      }
      memcpy((void *)(v7 + *a1), v9, v10 * 8 - v7);
LABEL_8:
      *((_DWORD *)a1 + 2) = v8;
      *(_DWORD *)(a2 + 8) = 0;
      return;
    }
    v11 = (__int64 *)*a1;
    if ( v11 != a1 + 2 )
    {
      _libc_free(v11, a2);
      v4 = *(__int64 **)a2;
    }
    *a1 = (__int64)v4;
    *((_DWORD *)a1 + 2) = *(_DWORD *)(a2 + 8);
    *((_DWORD *)a1 + 3) = *(_DWORD *)(a2 + 12);
    *(_QWORD *)a2 = v5;
    *(_QWORD *)(a2 + 8) = 0;
  }
}
