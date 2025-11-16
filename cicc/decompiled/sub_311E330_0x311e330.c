// Function: sub_311E330
// Address: 0x311e330
//
void __fastcall sub_311E330(unsigned __int64 **a1, unsigned __int64 **a2, __int64 a3)
{
  unsigned __int64 **i; // rbx
  unsigned __int64 *v4; // r12
  unsigned __int64 **v5; // r15
  unsigned __int64 *v6; // rax
  __int64 v7; // [rsp+18h] [rbp-48h] BYREF
  __int64 v8; // [rsp+20h] [rbp-40h] BYREF
  unsigned __int64 *v9; // [rsp+28h] [rbp-38h] BYREF

  v7 = a3;
  if ( a1 != a2 )
  {
    for ( i = a1 + 1; a2 != i; *a1 = v4 )
    {
      while ( !(unsigned __int8)sub_311D9B0(&v7, i, a1) )
      {
        v5 = i - 1;
        v8 = v7;
        v9 = *i;
        while ( (unsigned __int8)sub_311D9B0(&v8, &v9, v5) )
        {
          v6 = *v5--;
          v5[2] = v6;
        }
        ++i;
        v5[1] = v9;
        if ( a2 == i )
          return;
      }
      v4 = *i;
      if ( a1 != i )
        memmove(a1 + 1, a1, (char *)i - (char *)a1);
      ++i;
    }
  }
}
