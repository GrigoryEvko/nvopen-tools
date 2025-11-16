// Function: sub_7245B0
// Address: 0x7245b0
//
FILE *__fastcall sub_7245B0(char *a1, int *a2, _DWORD *a3)
{
  FILE *result; // rax
  FILE *v5; // [rsp+8h] [rbp-18h]

  *a3 = 0;
  result = sub_7244D0(a1, "r", a2);
  if ( result )
  {
    if ( unk_4F076F8 )
    {
      v5 = result;
      sub_720D80(result, a3, (__int64)a1);
      return v5;
    }
  }
  return result;
}
