// Function: sub_74A140
// Address: 0x74a140
//
__int64 *__fastcall sub_74A140(char a1, __int64 *a2, _DWORD *a3, void (__fastcall **a4)(char *))
{
  __int64 *result; // rax
  __int64 *v7; // r12

  result = sub_736C60(a1, a2);
  if ( result )
  {
    v7 = result;
    if ( *a3 )
      ((void (__fastcall *)(char *, void (__fastcall **)(char *)))*a4)(" ", a4);
    ((void (__fastcall *)(const char *, void (__fastcall **)(char *)))*a4)("__attribute__((", a4);
    ((void (__fastcall *)(__int64, void (__fastcall **)(char *)))*a4)(v7[2], a4);
    sub_74A070((__int64 *)v7[4], a4);
    result = (__int64 *)((__int64 (__fastcall *)(const char *, void (__fastcall **)(char *)))*a4)("))", a4);
    *a3 = 1;
  }
  return result;
}
