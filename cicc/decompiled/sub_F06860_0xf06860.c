// Function: sub_F06860
// Address: 0xf06860
//
void __fastcall sub_F06860(__int64 *src, __int64 *a2)
{
  __int64 *i; // rbx
  bool v4; // al
  __int64 v5; // r12
  __int64 *j; // r15
  __int64 v7; // rdi
  __int64 *v8; // r13

  if ( src != a2 )
  {
    for ( i = src + 1; a2 != i; ++i )
    {
      while ( 1 )
      {
        v4 = sub_B445A0(*src, *i);
        v5 = *i;
        if ( v4 )
          break;
        for ( j = i; ; j[1] = *j )
        {
          v7 = *(j - 1);
          v8 = j--;
          if ( !sub_B445A0(v7, v5) )
            break;
        }
        *v8 = v5;
        if ( a2 == ++i )
          return;
      }
      if ( src != i )
        memmove(src + 1, src, (char *)i - (char *)src);
      *src = v5;
    }
  }
}
