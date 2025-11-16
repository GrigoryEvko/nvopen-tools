// Function: sub_385B9D0
// Address: 0x385b9d0
//
void __fastcall sub_385B9D0(unsigned int *src, char *a2, _QWORD *a3)
{
  unsigned int *i; // r12
  unsigned int *v6; // r9
  __int64 v7; // r15
  __int64 v8; // rsi
  unsigned int *v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rdx

  if ( src != (unsigned int *)a2 )
  {
    for ( i = src + 1; i != (unsigned int *)a2; *src = v7 )
    {
      while ( 1 )
      {
        v6 = i;
        v7 = *i;
        v8 = *(_QWORD *)(*a3 + 16 * v7);
        if ( v8 < *(_QWORD *)(*a3 + 16LL * *src) )
          break;
        v9 = i - 1;
        v10 = *(i - 1);
        if ( v8 < *(_QWORD *)(*a3 + 16 * v10) )
        {
          do
          {
            v9[1] = v10;
            v6 = v9;
            v11 = *--v9;
            LODWORD(v10) = v11;
          }
          while ( *(_QWORD *)(*a3 + 16 * v7) < *(_QWORD *)(*a3 + 16 * v11) );
        }
        ++i;
        *v6 = v7;
        if ( i == (unsigned int *)a2 )
          return;
      }
      if ( src != i )
        memmove(src + 1, src, (char *)i - (char *)src);
      ++i;
    }
  }
}
