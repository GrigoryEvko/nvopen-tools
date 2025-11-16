// Function: sub_1E40C40
// Address: 0x1e40c40
//
void __fastcall sub_1E40C40(char *src, char *a2)
{
  char *i; // rbx
  __int64 v4; // r12
  char *v5; // rcx
  unsigned int v6; // esi
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
        v6 = *(_DWORD *)(*(_QWORD *)i + 192LL);
        if ( v6 > *(_DWORD *)(*(_QWORD *)src + 192LL) )
          break;
        v7 = *((_QWORD *)i - 1);
        v8 = i - 8;
        if ( v6 > *(_DWORD *)(v7 + 192) )
        {
          do
          {
            *((_QWORD *)v8 + 1) = v7;
            v5 = v8;
            v7 = *((_QWORD *)v8 - 1);
            v8 -= 8;
          }
          while ( *(_DWORD *)(v4 + 192) > *(_DWORD *)(v7 + 192) );
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
