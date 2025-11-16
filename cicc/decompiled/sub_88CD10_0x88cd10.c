// Function: sub_88CD10
// Address: 0x88cd10
//
char *__fastcall sub_88CD10(int a1, __int64 a2)
{
  void (**v2)(void); // rbx
  const char *v3; // r12
  size_t v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  const char *v9; // r13

  if ( a1 == -1 )
  {
    sub_88B860();
  }
  else
  {
    v2 = (void (**)(void))((char *)&unk_49772F0 + 24 * a1);
    v2[1]();
    v3 = (const char *)*v2;
    if ( *v2 )
    {
      v4 = strlen((const char *)*v2);
      qword_4F06898 = (const char *)sub_822B10(v4 + 5, a2, v5, v6, v7, v8);
      *(_DWORD *)qword_4F06898 = 6449516;
      v9 = qword_4F06898;
      *(_WORD *)&v9[strlen(qword_4F06898)] = 95;
      return strcat((char *)qword_4F06898, v3);
    }
  }
  qword_4F06898 = "lib";
  return (char *)&qword_4F06898;
}
