// Function: sub_F07380
// Address: 0xf07380
//
void __fastcall sub_F07380(__int64 *a1, __int64 *a2)
{
  __int64 *i; // rbx
  __int64 v3; // r13
  __int64 v4; // r12
  __int64 v5; // rax
  bool v6; // al
  __int64 v7; // r12
  __int64 *j; // r15
  __int64 v9; // rdi
  __int64 *v10; // r13
  __int64 v11; // r14
  __int64 v12; // rax

  if ( a1 != a2 )
  {
    for ( i = a1 + 1; a2 != i; *a1 = v7 )
    {
      while ( 1 )
      {
        v3 = *i;
        v4 = sub_B140A0(*a1);
        v5 = sub_B140A0(v3);
        v6 = sub_B445A0(v4, v5);
        v7 = *i;
        if ( v6 )
          break;
        for ( j = i; ; j[1] = *j )
        {
          v9 = *(j - 1);
          v10 = j--;
          v11 = sub_B140A0(v9);
          v12 = sub_B140A0(v7);
          if ( !sub_B445A0(v11, v12) )
            break;
        }
        *v10 = v7;
        if ( a2 == ++i )
          return;
      }
      if ( a1 != i )
        memmove(a1 + 1, a1, (char *)i - (char *)a1);
      ++i;
    }
  }
}
