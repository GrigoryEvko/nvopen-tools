// Function: sub_39A6D90
// Address: 0x39a6d90
//
void __fastcall sub_39A6D90(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  char **i; // rbx
  char *v6; // rdx
  char v7; // al

  if ( a3 )
  {
    v4 = 8LL * *(unsigned int *)(a3 + 8);
    for ( i = (char **)(a3 - v4); (char **)a3 != i; ++i )
    {
      while ( 1 )
      {
        v6 = *i;
        v7 = **i;
        if ( v7 != 22 )
          break;
        sub_39A6AB0(a1, a2, (__int64)v6);
LABEL_5:
        if ( (char **)a3 == ++i )
          return;
      }
      if ( v7 != 23 )
        goto LABEL_5;
      sub_39A6B70(a1, a2, (__int64)v6);
    }
  }
}
