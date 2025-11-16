// Function: sub_2895090
// Address: 0x2895090
//
void __fastcall sub_2895090(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 *i; // rbx
  char v5; // al
  __int64 v6; // r13
  __int64 *j; // r15
  __int64 v8; // rdx
  __int64 *v9; // r14

  if ( a1 != a2 )
  {
    for ( i = a1 + 1; a2 != i; *a1 = v6 )
    {
      while ( 1 )
      {
        v5 = sub_B19DB0(*(_QWORD *)(a3 + 40), *i, *a1);
        v6 = *i;
        if ( v5 )
          break;
        for ( j = i; ; j[1] = *j )
        {
          v8 = *(j - 1);
          v9 = j--;
          if ( !(unsigned __int8)sub_B19DB0(*(_QWORD *)(a3 + 40), v6, v8) )
            break;
        }
        *v9 = v6;
        if ( a2 == ++i )
          return;
      }
      if ( a1 != i )
        memmove(a1 + 1, a1, (char *)i - (char *)a1);
      ++i;
    }
  }
}
