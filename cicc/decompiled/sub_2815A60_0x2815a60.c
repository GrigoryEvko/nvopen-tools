// Function: sub_2815A60
// Address: 0x2815a60
//
void __fastcall sub_2815A60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 *v6; // r12
  __int64 v7; // r13
  char **v10; // rsi
  __int64 v11; // rdi
  unsigned __int64 *v12; // r12
  unsigned __int64 *v13; // rbx

  v6 = *(unsigned __int64 **)a1;
  v7 = *(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v7 )
  {
    do
    {
      while ( 1 )
      {
        if ( a2 )
        {
          *(_DWORD *)(a2 + 8) = 0;
          *(_QWORD *)a2 = a2 + 16;
          *(_DWORD *)(a2 + 12) = 4;
          if ( *((_DWORD *)v6 + 2) )
            break;
        }
        v6 += 6;
        a2 += 48;
        if ( (unsigned __int64 *)v7 == v6 )
          goto LABEL_7;
      }
      v10 = (char **)v6;
      v11 = a2;
      v6 += 6;
      a2 += 48;
      sub_2813D00(v11, v10, a3, a4, a5, a6);
    }
    while ( (unsigned __int64 *)v7 != v6 );
LABEL_7:
    v12 = *(unsigned __int64 **)a1;
    v13 = (unsigned __int64 *)(*(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8));
    if ( *(unsigned __int64 **)a1 != v13 )
    {
      do
      {
        v13 -= 6;
        if ( (unsigned __int64 *)*v13 != v13 + 2 )
          _libc_free(*v13);
      }
      while ( v13 != v12 );
    }
  }
}
