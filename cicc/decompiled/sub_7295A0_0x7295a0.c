// Function: sub_7295A0
// Address: 0x7295a0
//
char *__fastcall sub_7295A0(char *src)
{
  size_t v2; // rax
  __int64 v3; // rdi
  size_t v4; // rbx
  char *result; // rax

  v2 = strlen(src);
  v3 = unk_4F06C40;
  v4 = unk_4F06C40 + v2;
  if ( unk_4F06C40 + v2 + 1 > qword_4F06C48 )
  {
    sub_729510(v4 + 1);
    v3 = unk_4F06C40;
  }
  result = strcpy((char *)qword_4F06C50 + v3, src);
  unk_4F06C40 = v4;
  return result;
}
