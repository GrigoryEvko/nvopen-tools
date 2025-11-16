// Function: sub_7E1620
// Address: 0x7e1620
//
char *__fastcall sub_7E1620(char *src)
{
  size_t v1; // rax
  char *v2; // rax

  v1 = strlen(src);
  v2 = (char *)sub_7E1510(v1 + 1);
  return strcpy(v2, src);
}
