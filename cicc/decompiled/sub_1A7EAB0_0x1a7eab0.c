// Function: sub_1A7EAB0
// Address: 0x1a7eab0
//
void __fastcall sub_1A7EAB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  _QWORD *v6; // r13
  _QWORD *v8; // rax
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rcx
  int v13; // r14d
  _QWORD *v14; // rsi
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rsi
  __int64 v21; // rdx

  if ( a1 != a2 )
  {
    v6 = (_QWORD *)(a2 + 16);
    v8 = *(_QWORD **)a2;
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
          v17 = *(_QWORD *)a1 + 16 * v11;
          do
          {
            v18 = *v6;
            v16 += 16LL;
            v6 += 2;
            *(_QWORD *)(v16 - 16) = v18;
            *(_DWORD *)(v16 - 8) = *((_DWORD *)v6 - 2);
          }
          while ( v16 != v17 );
        }
      }
      else
      {
        if ( v11 > *(unsigned int *)(a1 + 12) )
        {
          *(_DWORD *)(a1 + 8) = 0;
          sub_16CD150(a1, (const void *)(a1 + 16), v11, 16, a5, a6);
          v6 = *(_QWORD **)a2;
          v11 = *(unsigned int *)(a2 + 8);
          v12 = 0;
          v14 = *(_QWORD **)a2;
        }
        else
        {
          v14 = (_QWORD *)(a2 + 16);
          if ( *(_DWORD *)(a1 + 8) )
          {
            v19 = *(_QWORD *)a1;
            v12 *= 16LL;
            v20 = *(_QWORD *)a1 + v12;
            do
            {
              v21 = *v6;
              v19 += 16LL;
              v6 += 2;
              *(_QWORD *)(v19 - 16) = v21;
              *(_DWORD *)(v19 - 8) = *((_DWORD *)v6 - 2);
            }
            while ( v19 != v20 );
            v6 = *(_QWORD **)a2;
            v11 = *(unsigned int *)(a2 + 8);
            v14 = (_QWORD *)(*(_QWORD *)a2 + v12);
          }
        }
        v15 = 2 * v11;
        if ( v14 != &v6[v15] )
          memcpy((void *)(v12 + *(_QWORD *)a1), v14, v15 * 8 - v12);
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
        v8 = *(_QWORD **)a2;
      }
      *(_QWORD *)a1 = v8;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v6;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
