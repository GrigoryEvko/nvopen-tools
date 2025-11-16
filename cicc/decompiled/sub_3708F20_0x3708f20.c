// Function: sub_3708F20
// Address: 0x3708f20
//
void *__fastcall sub_3708F20(char **a1)
{
  char *v1; // rax
  char *v2; // r12
  void *result; // rax

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  v1 = (char *)sub_22077B0(0xFF00u);
  v2 = v1 + 65280;
  *a1 = v1;
  a1[2] = v1 + 65280;
  result = memset(v1, 0, 0xFF00u);
  a1[1] = v2;
  return result;
}
