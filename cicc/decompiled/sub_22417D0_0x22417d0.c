// Function: sub_22417D0
// Address: 0x22417d0
//
char *__fastcall sub_22417D0(__int64 *a1, char a2, unsigned __int64 a3)
{
  __int64 v3; // r12
  unsigned __int64 v4; // rax
  __int64 v5; // rbx
  char *v6; // rax

  v3 = -1;
  v4 = a1[1];
  if ( a3 < v4 )
  {
    v5 = *a1;
    v6 = (char *)memchr((const void *)(*a1 + a3), a2, v4 - a3);
    if ( v6 )
      return &v6[-v5];
  }
  return (char *)v3;
}
