// Function: sub_31AFD00
// Address: 0x31afd00
//
void __fastcall sub_31AFD00(char *src, char *a2)
{
  char *i; // rbx
  __int64 v3; // r12
  __int64 v4; // r15
  char *j; // r14
  __int64 v6; // rax
  char *v7; // r12

  if ( src != a2 )
  {
    for ( i = src + 8; a2 != i; i += 8 )
    {
      while ( !sub_B445A0(*(_QWORD *)(*(_QWORD *)i + 16LL), *(_QWORD *)(*(_QWORD *)src + 16LL)) )
      {
        v4 = *(_QWORD *)i;
        for ( j = i; ; *((_QWORD *)j + 1) = *(_QWORD *)j )
        {
          v6 = *((_QWORD *)j - 1);
          v7 = j;
          j -= 8;
          if ( !sub_B445A0(*(_QWORD *)(v4 + 16), *(_QWORD *)(v6 + 16)) )
            break;
        }
        *(_QWORD *)v7 = v4;
        i += 8;
        if ( a2 == i )
          return;
      }
      v3 = *(_QWORD *)i;
      if ( src != i )
        memmove(src + 8, src, i - src);
      *(_QWORD *)src = v3;
    }
  }
}
