// Function: sub_7214D0
// Address: 0x7214d0
//
FILE *__fastcall sub_7214D0(__int64 a1, _DWORD *a2)
{
  const char *v2; // rax
  FILE *result; // rax
  FILE *v4; // [rsp+8h] [rbp-18h]

  v2 = (const char *)sub_7212A0(a1);
  result = fopen(v2, "r");
  *a2 = 0;
  if ( result )
  {
    if ( unk_4F076F8 )
    {
      v4 = result;
      sub_720D80(result, a2, a1);
      return v4;
    }
  }
  return result;
}
