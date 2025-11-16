// Function: sub_2B82060
// Address: 0x2b82060
//
void __fastcall sub_2B82060(__int64 *a1, __int64 *a2, __int64 (__fastcall *a3)(__int64, __int64, __int64), __int64 a4)
{
  __int64 *i; // rbx
  char v7; // al
  __int64 v8; // r14
  __int64 *j; // r15
  char v10; // al
  __int64 *v11; // [rsp+18h] [rbp-38h]

  if ( a1 != a2 )
  {
    for ( i = a1 + 1; a2 != i; *a1 = v8 )
    {
      while ( 1 )
      {
        v7 = a3(a4, *i, *a1);
        v8 = *i;
        if ( v7 )
          break;
        for ( j = i; ; j[1] = *j )
        {
          v11 = j;
          v10 = a3(a4, v8, *--j);
          if ( !v10 )
            break;
        }
        ++i;
        *v11 = v8;
        if ( a2 == i )
          return;
      }
      if ( a1 != i )
        memmove(a1 + 1, a1, (char *)i - (char *)a1);
      ++i;
    }
  }
}
