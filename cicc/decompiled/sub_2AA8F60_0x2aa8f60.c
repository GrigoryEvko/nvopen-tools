// Function: sub_2AA8F60
// Address: 0x2aa8f60
//
unsigned __int64 *__fastcall sub_2AA8F60(const char **a1, __int64 a2)
{
  size_t v2; // rax
  const char *v3; // r13
  char *v4; // r12
  size_t v5; // rax

  v2 = 0;
  v3 = *(const char **)a2;
  if ( *(_QWORD *)a2 )
    v2 = strlen(*(const char **)a2);
  *a1 = v3;
  a1[1] = (const char *)v2;
  v4 = *(char **)(a2 + 8);
  v5 = strlen(v4);
  return sub_2241130((unsigned __int64 *)a1 + 2, 0, (unsigned __int64)a1[3], v4, v5);
}
