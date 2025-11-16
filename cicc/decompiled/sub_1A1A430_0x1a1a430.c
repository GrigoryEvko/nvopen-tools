// Function: sub_1A1A430
// Address: 0x1a1a430
//
void __fastcall sub_1A1A430(char *src, char *a2)
{
  char *i; // rbx
  __int64 v4; // r12
  char *v5; // rcx
  unsigned __int64 v6; // rsi
  __int64 v7; // rdx
  char *v8; // rax

  if ( src != a2 )
  {
    for ( i = src + 8; a2 != i; *(_QWORD *)src = v4 )
    {
      while ( 1 )
      {
        v4 = *(_QWORD *)i;
        v5 = i;
        v6 = *(_QWORD *)(*(_QWORD *)i + 32LL);
        if ( v6 < *(_QWORD *)(*(_QWORD *)src + 32LL) )
          break;
        v7 = *((_QWORD *)i - 1);
        v8 = i - 8;
        if ( v6 < *(_QWORD *)(v7 + 32) )
        {
          do
          {
            *((_QWORD *)v8 + 1) = v7;
            v5 = v8;
            v7 = *((_QWORD *)v8 - 1);
            v8 -= 8;
          }
          while ( *(_QWORD *)(v4 + 32) < *(_QWORD *)(v7 + 32) );
        }
        i += 8;
        *(_QWORD *)v5 = v4;
        if ( a2 == i )
          return;
      }
      if ( src != i )
        memmove(src + 8, src, i - src);
      i += 8;
    }
  }
}
