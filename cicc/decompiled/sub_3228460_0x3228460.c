// Function: sub_3228460
// Address: 0x3228460
//
void __fastcall sub_3228460(__int64 **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 *v7; // r12
  __int64 v8; // r13
  __int64 v11; // rax
  char **v12; // rsi
  __int64 v13; // rdi
  __int64 *v14; // r12
  __int64 v15; // rbx
  unsigned __int64 v16; // rdi

  v6 = *((unsigned int *)a1 + 2);
  v7 = *a1;
  v8 = (__int64)&(*a1)[7 * v6];
  if ( *a1 != (__int64 *)v8 )
  {
    do
    {
      while ( 1 )
      {
        if ( a2 )
        {
          v11 = *v7;
          *(_DWORD *)(a2 + 16) = 0;
          *(_DWORD *)(a2 + 20) = 2;
          *(_QWORD *)a2 = v11;
          *(_QWORD *)(a2 + 8) = a2 + 24;
          if ( *((_DWORD *)v7 + 4) )
            break;
        }
        v7 += 7;
        a2 += 56;
        if ( (__int64 *)v8 == v7 )
          goto LABEL_7;
      }
      v12 = (char **)(v7 + 1);
      v13 = a2 + 8;
      v7 += 7;
      a2 += 56;
      sub_32187E0(v13, v12, v6, a4, a5, a6);
    }
    while ( (__int64 *)v8 != v7 );
LABEL_7:
    v14 = *a1;
    v15 = (__int64)&(*a1)[7 * *((unsigned int *)a1 + 2)];
    if ( *a1 != (__int64 *)v15 )
    {
      do
      {
        v15 -= 56;
        v16 = *(_QWORD *)(v15 + 8);
        if ( v16 != v15 + 24 )
          _libc_free(v16);
      }
      while ( (__int64 *)v15 != v14 );
    }
  }
}
