// Function: sub_DA5080
// Address: 0xda5080
//
void __fastcall sub_DA5080(unsigned __int64 *a1, unsigned __int64 *a2, _QWORD **a3)
{
  unsigned __int64 *i; // r13
  __int64 v5; // rax
  unsigned __int64 v6; // r14
  unsigned __int64 *j; // rbx
  unsigned __int64 *v8; // r12
  __int64 v9; // [rsp+18h] [rbp-38h]

  if ( a1 != a2 )
  {
    for ( i = a1 + 1; a2 != i; *a1 = v6 )
    {
      while ( 1 )
      {
        v5 = sub_DA4700(*a3, *a3[1], *i, *a1, (__int64)a3[2], 0);
        v6 = *i;
        if ( BYTE4(v5) )
        {
          if ( (int)v5 < 0 )
            break;
        }
        for ( j = i; ; j[1] = *j )
        {
          v8 = j;
          v9 = sub_DA4700(*a3, *a3[1], v6, *(j - 1), (__int64)a3[2], 0);
          if ( !BYTE4(v9) )
            break;
          --j;
          if ( (int)v9 >= 0 )
            break;
        }
        *v8 = v6;
        if ( a2 == ++i )
          return;
      }
      if ( a1 != i )
        memmove(a1 + 1, a1, (char *)i - (char *)a1);
      ++i;
    }
  }
}
