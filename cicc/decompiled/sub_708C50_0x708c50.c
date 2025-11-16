// Function: sub_708C50
// Address: 0x708c50
//
char *sub_708C50()
{
  __int64 v0; // rax
  char *v1; // rcx
  __int64 v2; // rdx

  v0 = 0;
  memset(qword_4F064C0, 0, sizeof(qword_4F064C0));
  do
  {
    v2 = byte_4B6D300[v0];
    if ( byte_4B6D300[v0] )
    {
      v1 = (char *)*(&off_4B6DFA0 + v0);
      if ( (_DWORD)v2 == 42 )
      {
        v1 = "()";
      }
      else if ( (_DWORD)v2 == 43 )
      {
        v1 = "[]";
      }
      qword_4F064C0[v2] = v1;
    }
    ++v0;
  }
  while ( v0 != 357 );
  qword_4F064C0[3] = "new[]";
  qword_4F064C0[4] = "delete[]";
  return "delete[]";
}
