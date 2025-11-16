// Function: sub_155ED50
// Address: 0x155ed50
//
void __fastcall sub_155ED50(__int64 *src, __int64 *a2)
{
  __int64 *i; // rbx
  __int64 v3; // r12
  __int64 *v4; // r15
  __int64 v5; // rax
  __int64 v6[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( src != a2 )
  {
    for ( i = src + 1; a2 != i; ++i )
    {
      while ( !sub_155E9A0(i, *src) )
      {
        v4 = i - 1;
        v6[0] = *i;
        while ( sub_155E9A0(v6, *v4) )
        {
          v5 = *v4--;
          v4[2] = v5;
        }
        ++i;
        v4[1] = v6[0];
        if ( a2 == i )
          return;
      }
      v3 = *i;
      if ( src != i )
        memmove(src + 1, src, (char *)i - (char *)src);
      *src = v3;
    }
  }
}
