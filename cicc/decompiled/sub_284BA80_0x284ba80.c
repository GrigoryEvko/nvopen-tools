// Function: sub_284BA80
// Address: 0x284ba80
//
void __fastcall sub_284BA80(__int64 *a1, __int64 *a2, __int64 *a3)
{
  __int64 *i; // r13
  __int64 v5; // rbx
  __int64 v6; // r14
  unsigned __int64 v7; // rbx
  __int64 v8; // r14
  __int64 *j; // r15
  __int64 v10; // rsi
  unsigned __int64 v11; // rbx
  __int64 *v13; // [rsp+18h] [rbp-38h]

  if ( a1 != a2 )
  {
    for ( i = a1 + 1; a2 != i; *a1 = v5 )
    {
      while ( 1 )
      {
        v6 = *i;
        v7 = sub_FDD860(a3, *a1);
        if ( v7 > sub_FDD860(a3, v6) )
          break;
        v8 = *i;
        for ( j = i; ; j[1] = *j )
        {
          v10 = *(j - 1);
          v13 = j--;
          v11 = sub_FDD860(a3, v10);
          if ( v11 <= sub_FDD860(a3, v8) )
            break;
        }
        ++i;
        *v13 = v8;
        if ( a2 == i )
          return;
      }
      v5 = *i;
      if ( a1 != i )
        memmove(a1 + 1, a1, (char *)i - (char *)a1);
      ++i;
    }
  }
}
