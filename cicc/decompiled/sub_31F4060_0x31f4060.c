// Function: sub_31F4060
// Address: 0x31f4060
//
void __fastcall sub_31F4060(char *src, char *a2)
{
  char *i; // r12
  __int64 v4; // r13
  char *v5; // rdi
  unsigned __int16 v6; // cx
  __int64 v7; // rdx
  char *v8; // rax

  if ( src != a2 )
  {
    for ( i = src + 8; i != a2; *(_QWORD *)src = v4 )
    {
      while ( 1 )
      {
        v4 = *(_QWORD *)i;
        v5 = i;
        v6 = *(_WORD *)(**(_QWORD **)i + 20LL);
        if ( v6 < *(_WORD *)(**(_QWORD **)src + 20LL) )
          break;
        v7 = *((_QWORD *)i - 1);
        v8 = i - 8;
        if ( v6 < *(_WORD *)(*(_QWORD *)v7 + 20LL) )
        {
          do
          {
            *((_QWORD *)v8 + 1) = v7;
            v5 = v8;
            v7 = *((_QWORD *)v8 - 1);
            v8 -= 8;
          }
          while ( *(_WORD *)(*(_QWORD *)v4 + 20LL) < *(_WORD *)(*(_QWORD *)v7 + 20LL) );
        }
        i += 8;
        *(_QWORD *)v5 = v4;
        if ( i == a2 )
          return;
      }
      if ( src != i )
        memmove(src + 8, src, i - src);
      i += 8;
    }
  }
}
