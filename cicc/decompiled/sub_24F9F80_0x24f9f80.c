// Function: sub_24F9F80
// Address: 0x24f9f80
//
void __fastcall sub_24F9F80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 *v6; // r12
  __int64 v7; // r13
  int v9; // eax
  __int64 v10; // rdx
  unsigned __int64 *v11; // r12
  unsigned __int64 *v12; // rbx
  unsigned __int64 v13; // rdi

  v6 = *(unsigned __int64 **)a1;
  v7 = *(_QWORD *)a1 + 152LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v7 )
  {
    do
    {
      if ( a2 )
      {
        *(_DWORD *)(a2 + 8) = 0;
        *(_QWORD *)a2 = a2 + 16;
        *(_DWORD *)(a2 + 12) = 6;
        v10 = *((unsigned int *)v6 + 2);
        if ( (_DWORD)v10 )
          sub_24F9410(a2, (char **)v6, v10, a4, a5, a6);
        v9 = *((_DWORD *)v6 + 16);
        *(_DWORD *)(a2 + 80) = 0;
        *(_DWORD *)(a2 + 84) = 6;
        *(_DWORD *)(a2 + 64) = v9;
        *(_QWORD *)(a2 + 72) = a2 + 88;
        if ( *((_DWORD *)v6 + 20) )
          sub_24F9410(a2 + 72, (char **)v6 + 9, v10, a4, a5, a6);
        *(_DWORD *)(a2 + 136) = *((_DWORD *)v6 + 34);
        *(_BYTE *)(a2 + 144) = *((_BYTE *)v6 + 144);
        *(_BYTE *)(a2 + 145) = *((_BYTE *)v6 + 145);
        *(_BYTE *)(a2 + 146) = *((_BYTE *)v6 + 146);
        *(_BYTE *)(a2 + 147) = *((_BYTE *)v6 + 147);
      }
      v6 += 19;
      a2 += 152;
    }
    while ( (unsigned __int64 *)v7 != v6 );
    v11 = *(unsigned __int64 **)a1;
    v12 = (unsigned __int64 *)(*(_QWORD *)a1 + 152LL * *(unsigned int *)(a1 + 8));
    if ( *(unsigned __int64 **)a1 != v12 )
    {
      do
      {
        v12 -= 19;
        v13 = v12[9];
        if ( (unsigned __int64 *)v13 != v12 + 11 )
          _libc_free(v13);
        if ( (unsigned __int64 *)*v12 != v12 + 2 )
          _libc_free(*v12);
      }
      while ( v12 != v11 );
    }
  }
}
