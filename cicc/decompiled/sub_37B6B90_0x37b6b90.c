// Function: sub_37B6B90
// Address: 0x37b6b90
//
void __fastcall sub_37B6B90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _DWORD *v8; // rax
  _DWORD *v9; // rbx
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  int v12; // r14d
  _DWORD *v13; // rsi
  __int64 v14; // rdx
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  int v18; // ecx
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rsi
  int v21; // edx

  if ( a1 != a2 )
  {
    v8 = *(_DWORD **)a2;
    v9 = (_DWORD *)(a2 + 16);
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
          v17 = *(_QWORD *)a1 + 16 * v10;
          do
          {
            v18 = *v9;
            v16 += 16LL;
            v9 += 4;
            *(_DWORD *)(v16 - 16) = v18;
            *(_QWORD *)(v16 - 8) = *((_QWORD *)v9 - 1);
          }
          while ( v16 != v17 );
        }
        goto LABEL_8;
      }
      if ( v10 > *(unsigned int *)(a1 + 12) )
      {
        *(_DWORD *)(a1 + 8) = 0;
        sub_C8D5F0(a1, (const void *)(a1 + 16), v10, 0x10u, a5, a6);
        v11 = 0;
        v14 = 4LL * *(unsigned int *)(a2 + 8);
        v13 = *(_DWORD **)a2;
        if ( *(_QWORD *)a2 == v14 * 4 + *(_QWORD *)a2 )
          goto LABEL_8;
      }
      else
      {
        v13 = (_DWORD *)(a2 + 16);
        if ( *(_DWORD *)(a1 + 8) )
        {
          v19 = *(_QWORD *)a1;
          v11 *= 16LL;
          v20 = *(_QWORD *)a1 + v11;
          do
          {
            v21 = *v9;
            v19 += 16LL;
            v9 += 4;
            *(_DWORD *)(v19 - 16) = v21;
            *(_QWORD *)(v19 - 8) = *((_QWORD *)v9 - 1);
          }
          while ( v19 != v20 );
          v9 = *(_DWORD **)a2;
          v10 = *(unsigned int *)(a2 + 8);
          v13 = (_DWORD *)(*(_QWORD *)a2 + v11);
        }
        v14 = 4 * v10;
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
      v8 = *(_DWORD **)a2;
    }
    *(_QWORD *)a1 = v8;
    *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
    *(_QWORD *)a2 = v9;
    *(_QWORD *)(a2 + 8) = 0;
  }
}
