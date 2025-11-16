// Function: sub_2CB5BE0
// Address: 0x2cb5be0
//
void __fastcall sub_2CB5BE0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // r14
  int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // r12
  __int64 v12; // rbx
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi

  v6 = *a1;
  v7 = *a1 + 120LL * *((unsigned int *)a1 + 2);
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
        *(_DWORD *)(a2 + 20) = 4;
        v10 = *(unsigned int *)(v6 + 16);
        if ( (_DWORD)v10 )
          sub_2CAF4B0(a2 + 8, (char **)(v6 + 8), v10, a4, a5, a6);
        *(_DWORD *)(a2 + 64) = 0;
        *(_QWORD *)(a2 + 56) = a2 + 72;
        *(_DWORD *)(a2 + 68) = 4;
        if ( *(_DWORD *)(v6 + 64) )
          sub_2CAF4B0(a2 + 56, (char **)(v6 + 56), v10, a4, a5, a6);
        *(_QWORD *)(a2 + 104) = *(_QWORD *)(v6 + 104);
        *(_DWORD *)(a2 + 112) = *(_DWORD *)(v6 + 112);
      }
      v6 += 120;
      a2 += 120;
    }
    while ( v7 != v6 );
    v11 = *a1;
    v12 = *a1 + 120LL * *((unsigned int *)a1 + 2);
    if ( *a1 != v12 )
    {
      do
      {
        v12 -= 120;
        v13 = *(_QWORD *)(v12 + 56);
        if ( v13 != v12 + 72 )
          _libc_free(v13);
        v14 = *(_QWORD *)(v12 + 8);
        if ( v14 != v12 + 24 )
          _libc_free(v14);
      }
      while ( v12 != v11 );
    }
  }
}
