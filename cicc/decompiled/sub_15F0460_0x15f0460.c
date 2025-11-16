// Function: sub_15F0460
// Address: 0x15f0460
//
void __fastcall sub_15F0460(__int64 *src, __int64 *a2, __int64 a3)
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
        v8 = *(_QWORD *)(sub_15EFCB0(a3, *i) + 784);
        if ( v8 > *(_QWORD *)(sub_15EFCB0(a3, v7) + 784) )
          break;
        v9 = i++;
        sub_15F03F0(v9, a3);
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
