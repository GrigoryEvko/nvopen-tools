// Function: sub_1DD4EB0
// Address: 0x1dd4eb0
//
void __fastcall sub_1DD4EB0(char *src, char *a2)
{
  char *i; // rbx
  unsigned __int16 v3; // r12
  int v4; // r15d
  char *v5; // rdx
  char *j; // rax

  if ( src != a2 )
  {
    for ( i = src + 8; a2 != i; *((_DWORD *)src + 1) = v4 )
    {
      while ( 1 )
      {
        v3 = *(_WORD *)i;
        v4 = *((_DWORD *)i + 1);
        v5 = i;
        if ( *(_WORD *)i < *(_WORD *)src )
          break;
        for ( j = i - 8; v3 < *(_WORD *)j; j -= 8 )
        {
          *((_QWORD *)j + 1) = *(_QWORD *)j;
          v5 = j;
        }
        i += 8;
        *(_WORD *)v5 = v3;
        *((_DWORD *)v5 + 1) = v4;
        if ( a2 == i )
          return;
      }
      if ( src != i )
        memmove(src + 8, src, i - src);
      i += 8;
      *(_WORD *)src = v3;
    }
  }
}
