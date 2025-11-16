// Function: sub_370C330
// Address: 0x370c330
//
char *__fastcall sub_370C330(int a1)
{
  const char *v1; // r8
  __int64 *v2; // rax

  v1 = "<no type>";
  if ( !a1 )
    return (char *)v1;
  v2 = &qword_504EAA0;
  if ( a1 == 259 )
    return "std::nullptr_t";
  do
  {
    if ( *((_DWORD *)v2 + 4) == (unsigned __int8)a1 )
      return (char *)*v2;
    v2 += 3;
  }
  while ( v2 != &qword_504EE78 );
  return "<unknown simple type>";
}
