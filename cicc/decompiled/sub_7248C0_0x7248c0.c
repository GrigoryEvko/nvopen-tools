// Function: sub_7248C0
// Address: 0x7248c0
//
char *__fastcall sub_7248C0(int a1, const char *a2, size_t a3)
{
  bool v4; // zf
  __int64 v5; // rdi
  char *v6; // r8
  char *result; // rax

  if ( a1 )
  {
    v4 = unk_4F073B8 == a1;
    v5 = a3 + 1;
    if ( v4 )
      v6 = (char *)sub_7247C0(v5);
    else
      v6 = (char *)sub_822B10(v5);
  }
  else
  {
    v6 = (char *)sub_823970(a3 + 1);
  }
  result = strncpy(v6, a2, a3);
  result[a3] = 0;
  return result;
}
