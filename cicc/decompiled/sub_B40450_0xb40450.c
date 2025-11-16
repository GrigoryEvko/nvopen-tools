// Function: sub_B40450
// Address: 0xb40450
//
void __fastcall sub_B40450(__int64 *src, __int64 *a2, __int64 a3)
{
  __int64 *i; // rbx
  __int64 v6; // r15
  __int64 v7; // r14
  unsigned __int64 v8; // r15
  __int64 *v9; // rdi

  if ( src != a2 )
  {
    for ( i = src + 1; a2 != i; ++i )
    {
      while ( 1 )
      {
        v7 = *src;
        v8 = *(_QWORD *)(sub_B3FBB0(a3, *i) + 784);
        if ( v8 > *(_QWORD *)(sub_B3FBB0(a3, v7) + 784) )
          break;
        v9 = i++;
        sub_B403E0(v9, a3);
        if ( a2 == i )
          return;
      }
      v6 = *i;
      if ( src != i )
        memmove(src + 1, src, (char *)i - (char *)src);
      *src = v6;
    }
  }
}
