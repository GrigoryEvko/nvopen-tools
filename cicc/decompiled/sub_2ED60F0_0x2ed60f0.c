// Function: sub_2ED60F0
// Address: 0x2ed60f0
//
void __fastcall sub_2ED60F0(__int64 **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r12
  __int64 v7; // r13
  __int64 v10; // rax
  char **v11; // rsi
  __int64 v12; // rdi
  __int64 *v13; // r12
  __int64 v14; // rbx
  unsigned __int64 v15; // rdi

  v6 = *a1;
  v7 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  if ( *a1 != (__int64 *)v7 )
  {
    do
    {
      while ( 1 )
      {
        if ( a2 )
        {
          v10 = *v6;
          *(_DWORD *)(a2 + 16) = 0;
          *(_DWORD *)(a2 + 20) = 2;
          *(_QWORD *)a2 = v10;
          *(_QWORD *)(a2 + 8) = a2 + 24;
          if ( *((_DWORD *)v6 + 4) )
            break;
        }
        v6 += 4;
        a2 += 32;
        if ( (__int64 *)v7 == v6 )
          goto LABEL_7;
      }
      v11 = (char **)(v6 + 1);
      v12 = a2 + 8;
      v6 += 4;
      a2 += 32;
      sub_2ED1840(v12, v11, a3, a4, a5, a6);
    }
    while ( (__int64 *)v7 != v6 );
LABEL_7:
    v13 = *a1;
    v14 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
    if ( *a1 != (__int64 *)v14 )
    {
      do
      {
        v14 -= 32;
        v15 = *(_QWORD *)(v14 + 8);
        if ( v15 != v14 + 24 )
          _libc_free(v15);
      }
      while ( (__int64 *)v14 != v13 );
    }
  }
}
