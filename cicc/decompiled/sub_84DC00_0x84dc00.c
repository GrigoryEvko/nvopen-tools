// Function: sub_84DC00
// Address: 0x84dc00
//
_BYTE *__fastcall sub_84DC00(__int64 *a1, const char *a2)
{
  const char *v2; // rbx
  size_t v3; // r12
  __int64 v4; // r14
  char *v5; // rax
  char *v6; // rcx
  size_t v7; // r12
  _BYTE *result; // rax

  v2 = a2;
  v3 = strlen(a2);
  sub_84D980(a1, v3 + a1[2] + 1);
  v4 = a1[2];
  sub_84D980(a1, v3 + v4);
  if ( v3 )
  {
    v5 = (char *)(v4 + *a1);
    v6 = &v5[v3];
    do
    {
      if ( v5 )
        *v5 = *v2;
      ++v5;
      ++v2;
    }
    while ( v6 != v5 );
  }
  v7 = a1[2] + v3;
  a1[2] = v7;
  if ( a1[1] == v7 )
    sub_7CB020(a1);
  result = (_BYTE *)(*a1 + v7);
  if ( result )
    *result = 0;
  a1[2] = v7 + 1;
  return result;
}
