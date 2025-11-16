// Function: sub_1A4FC40
// Address: 0x1a4fc40
//
void __fastcall sub_1A4FC40(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 *i; // rbx
  __int64 v4; // r12
  __int64 v5; // r12
  __int64 *v6; // r15
  __int64 v7; // rdx
  __int64 *v8; // r13
  __int64 v9; // [rsp+18h] [rbp-48h] BYREF
  __int64 v10[7]; // [rsp+28h] [rbp-38h] BYREF

  v9 = a3;
  if ( a1 != a2 )
  {
    for ( i = a1 + 1; a2 != i; *a1 = v4 )
    {
      while ( !sub_1A4F560(&v9, *i, *a1) )
      {
        v5 = *i;
        v6 = i;
        v10[0] = v9;
        while ( 1 )
        {
          v7 = *(v6 - 1);
          v8 = v6--;
          if ( !sub_1A4F560(v10, v5, v7) )
            break;
          v6[1] = *v6;
        }
        *v8 = v5;
        if ( a2 == ++i )
          return;
      }
      v4 = *i;
      if ( a1 != i )
        memmove(a1 + 1, a1, (char *)i - (char *)a1);
      ++i;
    }
  }
}
