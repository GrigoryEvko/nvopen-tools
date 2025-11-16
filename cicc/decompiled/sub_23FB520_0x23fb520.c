// Function: sub_23FB520
// Address: 0x23fb520
//
void __fastcall sub_23FB520(__int64 ***a1, __int64 ***a2)
{
  __int64 ***i; // r12
  __int64 **v3; // rbx
  __int64 **v4; // r13
  unsigned int v5; // ebx
  __int64 **v6; // r15
  __int64 ***j; // r14
  __int64 **v8; // r13
  unsigned int v9; // ebx
  __int64 ***v10; // [rsp+18h] [rbp-38h]

  if ( a1 != a2 )
  {
    for ( i = a1 + 1; a2 != i; *a1 = v3 )
    {
      while ( 1 )
      {
        v4 = *a1;
        v5 = sub_22DADF0(***i);
        if ( v5 < (unsigned int)sub_22DADF0(**v4) )
          break;
        v6 = *i;
        for ( j = i; ; j[1] = *j )
        {
          v8 = *(j - 1);
          v10 = j--;
          v9 = sub_22DADF0(**v6);
          if ( v9 >= (unsigned int)sub_22DADF0(**v8) )
            break;
        }
        ++i;
        *v10 = v6;
        if ( a2 == i )
          return;
      }
      v3 = *i;
      if ( a1 != i )
        memmove(a1 + 1, a1, (char *)i - (char *)a1);
      ++i;
    }
  }
}
