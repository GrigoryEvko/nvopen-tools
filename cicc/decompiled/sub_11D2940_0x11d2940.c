// Function: sub_11D2940
// Address: 0x11d2940
//
void __fastcall sub_11D2940(__int64 *src, __int64 *a2)
{
  __int64 *i; // rbx
  bool v3; // al
  __int64 v4; // r12
  __int64 *j; // r15
  __int64 v6; // rsi
  __int64 *v7; // r13

  if ( src != a2 )
  {
    for ( i = src + 1; a2 != i; ++i )
    {
      while ( 1 )
      {
        v3 = sub_B445A0(*i, *src);
        v4 = *i;
        if ( v3 )
          break;
        for ( j = i; ; j[1] = *j )
        {
          v6 = *(j - 1);
          v7 = j--;
          if ( !sub_B445A0(v4, v6) )
            break;
        }
        *v7 = v4;
        if ( a2 == ++i )
          return;
      }
      if ( src != i )
        memmove(src + 1, src, (char *)i - (char *)src);
      *src = v4;
    }
  }
}
