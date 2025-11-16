// Function: sub_721540
// Address: 0x721540
//
int __fastcall sub_721540(__int64 a1)
{
  const char *v1; // r12
  int result; // eax

  v1 = (const char *)sub_7212A0(a1);
  result = chdir(v1);
  if ( result )
    sub_685220(0x281u, (__int64)v1);
  return result;
}
