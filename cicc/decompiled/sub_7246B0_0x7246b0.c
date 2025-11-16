// Function: sub_7246B0
// Address: 0x7246b0
//
FILE *__fastcall sub_7246B0(char *a1, int a2, int *a3)
{
  bool v3; // zf
  char *v4; // rsi

  v3 = a2 == 0;
  v4 = "rb";
  if ( v3 )
    v4 = "r";
  return sub_7244D0(a1, v4, a3);
}
