// Function: sub_2254180
// Address: 0x2254180
//
char *__fastcall sub_2254180(const char *a1, long double *a2, _DWORD *a3, __locale_t *a4)
{
  long double v5; // fst7
  char *result; // rax
  char *endptr; // [rsp+8h] [rbp-20h] BYREF

  v5 = strtold_l(a1, &endptr, *a4);
  result = endptr;
  *a2 = v5;
  if ( result == a1 || *result )
  {
    *a2 = 0.0;
    *a3 = 4;
  }
  else if ( v5 == INFINITY )
  {
    *a2 = 1.189731495357231765e4932;
    *a3 = 4;
  }
  else if ( v5 == -INFINITY )
  {
    *a2 = -1.189731495357231765e4932;
    *a3 = 4;
  }
  return result;
}
