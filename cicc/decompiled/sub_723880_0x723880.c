// Function: sub_723880
// Address: 0x723880
//
unsigned __int64 __fastcall sub_723880(char *nptr, _DWORD *a2)
{
  int *v2; // rax
  int *v3; // r12
  unsigned __int64 result; // rax
  char *endptr; // [rsp+8h] [rbp-28h] BYREF

  *a2 = 0;
  v2 = __errno_location();
  *v2 = 0;
  v3 = v2;
  result = strtoul(nptr, &endptr, 0);
  if ( result - 1 > 0xFFFFFFFFFFFFFFFDLL && (*v3 == 34 || *v3 == 22) || *nptr && endptr && *endptr )
    *a2 = 1;
  return result;
}
