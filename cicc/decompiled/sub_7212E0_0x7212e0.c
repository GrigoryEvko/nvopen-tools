// Function: sub_7212E0
// Address: 0x7212e0
//
int __fastcall sub_7212E0(__int64 a1)
{
  int *v1; // rax
  int *v2; // rbx
  const char *v3; // r12
  int result; // eax

  v1 = __errno_location();
  *v1 = 0;
  v2 = v1;
  v3 = (const char *)sub_7212A0(a1);
  result = remove(v3);
  if ( result )
    sub_686660(0xDBu, (__int64)v3, *v2);
  return result;
}
