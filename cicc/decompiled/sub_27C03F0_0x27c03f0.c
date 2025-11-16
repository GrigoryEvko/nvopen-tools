// Function: sub_27C03F0
// Address: 0x27c03f0
//
void __fastcall sub_27C03F0(__int64 *src, __int64 *a2, __int64 a3)
{
  __int64 *i; // rbx
  __int64 v6; // r15
  char v7; // al
  __int64 *v8; // rdi

  if ( src != a2 )
  {
    for ( i = src + 1; a2 != i; *src = v6 )
    {
      while ( 1 )
      {
        if ( *src != *i )
        {
          sub_B196A0(*(_QWORD *)(a3 + 16), *i, *src);
          if ( v7 )
            break;
        }
        v8 = i++;
        sub_27C0370(v8, a3);
        if ( a2 == i )
          return;
      }
      v6 = *i;
      if ( src != i )
        memmove(src + 1, src, (char *)i - (char *)src);
      ++i;
    }
  }
}
