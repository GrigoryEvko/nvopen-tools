// Function: sub_284C3D0
// Address: 0x284c3d0
//
void __fastcall sub_284C3D0(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 *i; // r12
  __int64 v4; // rbx
  __int64 *v5; // r15
  __int64 v6; // r14
  __int64 v7; // [rsp+18h] [rbp-48h] BYREF
  __int64 v8[7]; // [rsp+28h] [rbp-38h] BYREF

  v7 = a3;
  if ( a1 != a2 )
  {
    for ( i = a1 + 1; i != a2; *a1 = v4 )
    {
      while ( 1 )
      {
        v4 = *i;
        if ( sub_284BC70(&v7, *i, *a1) )
          break;
        v5 = i;
        v8[0] = v7;
        while ( 1 )
        {
          v6 = *(v5 - 1);
          if ( !sub_284BC70(v8, v4, v6) )
            break;
          *v5-- = v6;
        }
        *v5 = v4;
        if ( ++i == a2 )
          return;
      }
      if ( a1 != i )
        memmove(a1 + 1, a1, (char *)i - (char *)a1);
      ++i;
    }
  }
}
