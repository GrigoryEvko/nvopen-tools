// Function: sub_37B9A30
// Address: 0x37b9a30
//
char *__fastcall sub_37B9A30(char **a1, unsigned int a2, _QWORD *a3, char a4)
{
  char *v5; // rax
  __int64 v6; // rdx
  char *v7; // rdx
  char *result; // rax

  *a1 = 0;
  a1[1] = 0;
  v5 = sub_E922F0(a3, a2);
  v7 = &v5[2 * v6];
  *a1 = v5;
  result = v7 - 2;
  if ( a4 )
    result = v7;
  a1[1] = result;
  return result;
}
