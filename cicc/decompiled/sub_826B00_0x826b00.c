// Function: sub_826B00
// Address: 0x826b00
//
__int64 sub_826B00()
{
  __int64 result; // rax
  const char *v1; // r14
  char *v2; // r13
  int v3; // ebx
  size_t v4; // r12

  result = qword_4F1F648;
  if ( !qword_4F1F648 )
  {
    v1 = off_4B7D3F8;
    v2 = sub_723F40(0);
    v3 = strlen(v2);
    v4 = (int)(strlen(v1) + v3 + 21);
    qword_4F1F648 = (__int64)sub_7247C0(v4);
    snprintf((char *)qword_4F1F648, v4, "%s%lu_%s_", v1, v3, v2);
    return qword_4F1F648;
  }
  return result;
}
