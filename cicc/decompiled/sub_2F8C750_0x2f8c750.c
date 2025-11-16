// Function: sub_2F8C750
// Address: 0x2f8c750
//
void __fastcall sub_2F8C750(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // r13
  int v9; // eax
  __int64 v10; // r12
  __int64 v11; // rbx
  unsigned __int64 v12; // rdi

  v6 = *a1;
  v7 = *a1 + 80LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v7 )
  {
    do
    {
      if ( a2 )
      {
        *(_DWORD *)a2 = *(_DWORD *)v6;
        v9 = *(_DWORD *)(v6 + 4);
        *(_DWORD *)(a2 + 16) = 0;
        *(_DWORD *)(a2 + 4) = v9;
        *(_QWORD *)(a2 + 8) = a2 + 24;
        *(_DWORD *)(a2 + 20) = 6;
        if ( *(_DWORD *)(v6 + 16) )
          sub_2F8ABB0(a2 + 8, (char **)(v6 + 8), a3, a4, a5, a6);
        *(_DWORD *)(a2 + 72) = *(_DWORD *)(v6 + 72);
      }
      v6 += 80;
      a2 += 80;
    }
    while ( v7 != v6 );
    v10 = *a1;
    v11 = *a1 + 80LL * *((unsigned int *)a1 + 2);
    if ( *a1 != v11 )
    {
      do
      {
        v11 -= 80;
        v12 = *(_QWORD *)(v11 + 8);
        if ( v12 != v11 + 24 )
          _libc_free(v12);
      }
      while ( v11 != v10 );
    }
  }
}
